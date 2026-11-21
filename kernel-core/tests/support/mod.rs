//! A host machine for the kernel: memory from a `Vec`, a fake architecture,
//! and helpers to drive syscalls as a thread would. Shared by the tests and
//! the fuzz target.
#![allow(dead_code)]

pub mod fuzz;

use kernel_core::Kernel;
use kernel_core::boot::{Region, RegionKind};
use kernel_core::cap::Cap;
use kernel_core::hal::Hal;
use kernel_core::mem::{Paddr, Phys};
use kernel_core::root::{INFO_VADDR, RootSpec};
use silica_abi::root::{RootInfo, SLOT_ASPACE, UntypedDesc};
use silica_abi::{Error, ObjectType, Rights, Syscall};
use std::cell::{Cell, RefCell};

/// Physical address of the first byte of the host machine's memory.
pub const RAM_BASE: u64 = 0x10_0000;

pub struct TestHal {
    pub now: Cell<u64>,
    pub timer: Cell<u64>,
    /// When set, `irq_pending` is true this many more times.
    pub pending: Cell<u32>,
    pub console: RefCell<Vec<u8>>,
    pub flushes: Cell<u64>,
    pub switches: Cell<u64>,
    pub left: Cell<u64>,
    /// The simulated processor the test is currently acting as.
    pub cpu: Cell<usize>,
    /// (processor, waited) for every kick, in order.
    pub kicks: RefCell<Vec<(usize, bool)>>,
    pub resets: Cell<u64>,
    pub forgets: Cell<u64>,
    /// What the kernel asked of the interrupt controller, in order.
    pub irq_log: RefCell<Vec<String>>,
}

const ADDR: u64 = 0x000F_FFFF_FFFF_F000;

impl Hal for TestHal {
    const IP: usize = 0;
    const SP: usize = 1;
    const FLAGS: usize = 2;
    const SYS: usize = 3;
    const ARG: [usize; 6] = [4, 5, 6, 7, 8, 9];
    const FLAGS_INIT: u64 = 0x202;
    const SYSCALL_LEN: u64 = 2;
    const USER_TOP: u64 = 0x7FFF_FFFF_F000;

    fn pte_table(table: Paddr) -> u64 {
        table | 7
    }
    fn pte_leaf(frame: Paddr, rights: Rights, level: u8, device: bool) -> u64 {
        let mut e = frame | 5;
        if rights.has(Rights::WRITE) {
            e |= 2;
        }
        if !rights.has(Rights::EXEC) {
            e |= 1 << 63;
        }
        if level == 1 {
            e |= 0x80;
        }
        if device {
            e |= 0x10;
        }
        e
    }
    fn pte_present(pte: u64) -> bool {
        pte & 1 != 0
    }
    fn pte_is_leaf(pte: u64, level: u8) -> bool {
        level == 0 || pte & 0x80 != 0
    }
    fn pte_addr(pte: u64) -> Paddr {
        pte & ADDR
    }
    fn init_root(&self, phys: &Phys, root: Paddr) {
        phys.write_val(root + 511 * 8, &0xDEAD_0000_0000_0003u64);
    }
    fn init_thread(&self, phys: &Phys, thread: Paddr) {
        phys.write_val(thread + kernel_core::thread::FPU_OFFSET as u64, &0x1F80u64);
    }
    fn switch_thread(&self, _phys: &Phys, _from: Paddr, _to: Paddr, _root: Paddr) {
        self.switches.set(self.switches.get() + 1);
    }
    fn leave_aspace(&self, root: Paddr) {
        self.left.set(root);
    }
    fn tlb_flush(&self, _root: Paddr, _vaddr: u64, _len: u64) {
        self.flushes.set(self.flushes.get() + 1);
    }
    fn cpu(&self) -> usize {
        self.cpu.get()
    }
    fn kick(&self, cpu: usize, wait: bool) {
        // Simulated processors take turns: none is in user mode while
        // another is in the kernel, so there is nothing to wait for.
        self.kicks.borrow_mut().push((cpu, wait));
    }
    fn tlb_reset(&self) {
        self.resets.set(self.resets.get() + 1);
    }
    fn tlb_forget(&self, _root: Paddr, _cpu: usize) {
        self.forgets.set(self.forgets.get() + 1);
    }
    fn now(&self) -> u64 {
        self.now.get()
    }
    fn set_timer(&self, deadline: u64) {
        self.timer.set(deadline);
    }
    fn irq_pending(&self) -> bool {
        let n = self.pending.get();
        if n > 0 {
            self.pending.set(n - 1);
        }
        n > 0
    }
    fn idle_begin(&self) {}
    fn idle(&self) {
        assert!(self.timer.get() != u64::MAX, "idle with no timer armed: every thread is blocked for ever");
        self.now.set(self.now.get().max(self.timer.get()));
    }
    fn irq_route(&self, line: usize, cpu: usize, level: bool, active_low: bool) -> bool {
        self.irq_log.borrow_mut().push(format!("route {line} cpu{cpu} level={level} low={active_low}"));
        line < 24
    }
    fn irq_mask(&self, line: usize, masked: bool) {
        self.irq_log.borrow_mut().push(format!("{} {line}", if masked { "mask" } else { "unmask" }));
    }
    fn port_io(&self, port: u16, _width: u8, write: bool, value: u32) -> u32 {
        if write { 0 } else { port as u32 ^ value }
    }
    fn console(&self, bytes: &[u8]) {
        self.console.borrow_mut().extend_from_slice(bytes);
    }
}

/// A minimal ELF64 executable: one read-execute segment at 0x40_0000 and
/// one read-write segment of `bss` zero bytes at 0x60_0000.
pub fn tiny_elf(code: &[u8], bss: u64) -> Vec<u8> {
    let mut f = vec![0u8; 64 + 2 * 56];
    f[..4].copy_from_slice(b"\x7fELF");
    f[4] = 2;
    f[5] = 1;
    f[6] = 1;
    f[16..18].copy_from_slice(&2u16.to_le_bytes());
    f[18..20].copy_from_slice(&62u16.to_le_bytes());
    f[24..32].copy_from_slice(&0x40_0000u64.to_le_bytes());
    f[32..40].copy_from_slice(&64u64.to_le_bytes());
    f[54..56].copy_from_slice(&56u16.to_le_bytes());
    f[56..58].copy_from_slice(&2u16.to_le_bytes());
    let off = f.len() as u64;
    let mut ph = |i: usize, flags: u32, off: u64, vaddr: u64, filesz: u64, memsz: u64| {
        let o = 64 + i * 56;
        f[o..o + 4].copy_from_slice(&1u32.to_le_bytes());
        f[o + 4..o + 8].copy_from_slice(&flags.to_le_bytes());
        f[o + 8..o + 16].copy_from_slice(&off.to_le_bytes());
        f[o + 16..o + 24].copy_from_slice(&vaddr.to_le_bytes());
        f[o + 32..o + 40].copy_from_slice(&filesz.to_le_bytes());
        f[o + 40..o + 48].copy_from_slice(&memsz.to_le_bytes());
    };
    ph(0, 5, off, 0x40_0000, code.len() as u64, code.len() as u64);
    ph(1, 6, 0, 0x60_0000, 0, bss);
    f.extend_from_slice(code);
    f
}

/// Page-aligned memory for the machine, freed with it.
struct Ram {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}
impl Ram {
    fn new(bytes: usize) -> Ram {
        let layout = std::alloc::Layout::from_size_align(bytes, 4096).unwrap();
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());
        // Not zero: the kernel must clear memory itself before using it.
        // SAFETY: `ptr` is valid for `bytes` bytes.
        unsafe { std::ptr::write_bytes(ptr, 0xA5, bytes) };
        Ram { ptr, layout }
    }
}
impl Drop for Ram {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout; the kernel using it
        // is dropped with the Machine before this field.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

impl Machine {
    /// As processor `cpu`: let its scheduler run (as on any kernel exit).
    pub fn settle(&self, cpu: usize) {
        self.on(cpu);
        if self.k.can_run() {
            self.k.enter();
            self.k.schedule();
            self.k.leave();
        }
    }

    /// The fault path as the architecture drives it.
    pub fn fault(&self, kind: silica_abi::Fault, addr: u64, code: u64) {
        self.k.enter();
        self.k.fault(kind, addr, code);
        if self.k.can_run() {
            self.k.schedule();
        }
        self.k.leave();
    }

    /// A device interrupt on line `line`, on the current processor.
    pub fn irq(&self, line: usize) {
        self.k.enter();
        self.k.irq(line);
        if self.k.can_run() {
            self.k.schedule();
        }
        self.k.leave();
    }

    /// A timer interrupt on the current processor.
    pub fn tick(&self) {
        self.k.enter();
        self.k.on_timer();
        if self.k.can_run() {
            self.k.schedule();
        }
        self.k.leave();
    }
}

pub struct Machine {
    pub k: Kernel<TestHal>,
    _ram: Ram,
    pub root: Paddr,
    /// The root CapSpace.
    pub cs: Cap,
    pub info: Paddr,
    pub mib: u64,
}

impl Machine {
    /// A machine with `mib` MiB of memory, one processor and a root task.
    pub fn new(mib: u64) -> Machine {
        Machine::with_cpus(mib, 1)
    }

    /// Act as processor `cpu` from now on. The simulated processors take
    /// turns: each kernel entry runs to completion before the next.
    pub fn on(&self, cpu: usize) {
        self.k.hal.cpu.set(cpu);
    }

    /// A machine with `cpus` processors; the root task runs on processor 0.
    pub fn with_cpus(mib: u64, cpus: usize) -> Machine {
        let ram = Ram::new((mib << 20) as usize);
        let hal = TestHal {
            now: Cell::new(1000),
            timer: Cell::new(u64::MAX),
            pending: Cell::new(0),
            console: RefCell::new(Vec::new()),
            flushes: Cell::new(0),
            switches: Cell::new(0),
            left: Cell::new(0),
            cpu: Cell::new(0),
            kicks: RefCell::new(Vec::new()),
            resets: Cell::new(0),
            forgets: Cell::new(0),
            irq_log: RefCell::new(Vec::new()),
        };
        let base = ram.ptr.wrapping_sub(RAM_BASE as usize);
        // SAFETY: every physical address the kernel is told about lies in
        // [RAM_BASE, RAM_BASE + len), which `base` maps onto `ram`; `ram`
        // lives as long as the kernel and is used by nothing else.
        let k = Kernel::new(hal, unsafe { Phys::new(base) });
        let regions = [
            Region { base: RAM_BASE, len: (mib << 20) - 0x4000, kind: RegionKind::Usable as u32, reserved: 0 },
            Region { base: RAM_BASE + (mib << 20) - 0x4000, len: 0x4000, kind: RegionKind::Ramdisk as u32, reserved: 0 },
        ];
        let elf = tiny_elf(&[0x0F, 0x05, 0xEB, 0xFC], 0x3000);
        let spec = RootSpec {
            regions: &regions,
            elf: &elf,
            machine: 62,
            reserved: &[(0xA000, 0xD000), (0x3000, 0x6000), (0x5000, 0x8000)],
            device_top: 0,
            ports: true,
            ramdisk: [RAM_BASE + (mib << 20) - 0x4000, 0x4000],
            rsdp: 0,
            framebuffer: [0; 5],
            clock: [1_000_000_000, 0, 0],
            wall_clock: [0; 2],
            entropy: [7; 32],
            cmdline: b"test=host",
            cpus: cpus as u64,
        };
        k.set_cpu_count(cpus);
        k.enter();
        let root = k.boot_root(&spec).expect("root task");
        let tv = k.thread_view(root);
        let cs = k.view(tv.slots[0]).expect("root cspace").cap;
        let asp = k.view(k.slot_addr(&cs, SLOT_ASPACE).unwrap()).unwrap().cap.obj;
        let info = k.translate(asp, INFO_VADDR).expect("root info mapped");
        k.schedule();
        k.leave();
        assert_eq!(k.current(), root);
        Machine { k, _ram: ram, root, cs, info, mib }
    }

    pub fn info(&self) -> &RootInfo {
        // SAFETY: `info` is the frame the kernel filled with a RootInfo,
        // inside `ram`; the kernel never writes it again.
        unsafe { &*self.k.phys.ptr(self.info).cast::<RootInfo>() }
    }

    /// Bytes of the machine's memory.
    pub fn mem(&self, p: Paddr, len: u64) -> &[u8] {
        assert!(p >= RAM_BASE && p + len <= RAM_BASE + (self.mib << 20));
        // SAFETY: the range was just checked to lie inside `ram`; the kernel
        // runs only inside syscalls made through this machine.
        unsafe { self.k.phys.bytes(p, len) }
    }

    /// Whether a range of the machine's memory is all zero. A slice
    /// comparison, which Miri runs natively; a byte loop takes minutes there.
    pub fn is_zero(&self, p: Paddr, len: u64) -> bool {
        *self.mem(p, len) == vec![0u8; len as usize][..]
    }

    pub fn untyped(&self) -> Vec<(u64, UntypedDesc)> {
        let i = self.info();
        (0..i.untyped_count).map(|n| (silica_abi::root::SLOT_FIRST_UNTYPED + n, i.untyped[n as usize])).collect()
    }

    /// Slot and descriptor of the largest ordinary-memory Untyped.
    pub fn big_untyped(&self) -> (u64, UntypedDesc) {
        self.untyped().into_iter().filter(|(_, d)| d.kind == 0).max_by_key(|(_, d)| d.bits).unwrap()
    }

    /// Run until thread `t` is current, then make syscall `n` as it. Returns
    /// the status once the call has completed or the thread has blocked.
    pub fn sys_as(&self, t: Paddr, n: Syscall, a: [u64; 6]) -> SysResult {
        let mut spins = 0;
        while self.k.current() != t {
            let c = self.k.current();
            // Everyone else yields until `t` gets its turn.
            self.k.regs(c)[TestHal::SYS].set(Syscall::Yield as u64);
            self.k.enter();
            self.k.syscall();
            self.k.schedule();
            self.k.leave();
            spins += 1;
            assert!(spins < 1000, "thread {t:#x} never became current");
        }
        self.raw(n as u64, a)
    }

    /// Make syscall number `n` as the current thread, restarting it while preempted.
    pub fn raw(&self, n: u64, a: [u64; 6]) -> SysResult {
        let t = self.k.current();
        let r = self.k.regs(t);
        loop {
            r[TestHal::IP].set(0x40_0002);
            r[TestHal::SYS].set(n);
            for i in 0..6 {
                r[TestHal::ARG[i]].set(a[i]);
            }
            self.k.enter();
            self.k.syscall();
            self.k.leave();
            // A preempted syscall has its instruction pointer moved back onto
            // the syscall instruction; anything else means it is over (the
            // call may have given this very thread a new instruction pointer).
            if r[TestHal::IP].get() != 0x40_0000 || self.k.current() != t {
                break;
            }
        }
        let out = SysResult { thread: t, status: r[TestHal::SYS].get(), blocked: self.k.thread_view(t).state != 1 };
        if self.k.can_run() {
            self.k.enter();
            self.k.schedule();
            self.k.leave();
        }
        out
    }

    pub fn sys(&self, n: Syscall, a: [u64; 6]) -> SysResult {
        self.sys_as(self.root, n, a)
    }

    /// The syscall as the root thread; panics unless it succeeds.
    pub fn ok(&self, n: Syscall, a: [u64; 6]) {
        let r = self.sys(n, a);
        assert!(!r.blocked && r.status == 0, "{n:?}{a:x?} failed: {:?}", Error::result(r.status));
    }

    /// The error the syscall returns to the root thread.
    pub fn err(&self, n: Syscall, a: [u64; 6]) -> Error {
        let r = self.sys(n, a);
        assert!(!r.blocked, "{n:?} blocked");
        Error::result(r.status).expect_err("syscall succeeded")
    }

    pub fn arg(&self, t: Paddr, i: usize) -> u64 {
        self.k.regs(t)[TestHal::ARG[i]].get()
    }
    pub fn status(&self, t: Paddr) -> u64 {
        self.k.regs(t)[TestHal::SYS].get()
    }

    /// The capability in root CapSpace slot `i`.
    pub fn cap(&self, i: u64) -> Option<Cap> {
        self.k.view(self.k.slot_addr(&self.cs, i).unwrap()).map(|v| v.cap)
    }

    pub fn retype(&self, ut: u64, ty: ObjectType, bits: u64, slot: u64, count: u64) {
        self.ok(Syscall::Retype, [ut, ty as u64, bits, silica_abi::root::SLOT_CAPSPACE, slot, count]);
    }

    /// A new thread sharing the root's CapSpace and address space, runnable.
    /// Uses slots `slot` (thread) and returns the thread object.
    pub fn spawn(&self, ut: u64, slot: u64) -> Paddr {
        self.retype(ut, ObjectType::Thread, 0, slot, 1);
        self.ok(Syscall::ThreadConfigure, [slot, silica_abi::root::SLOT_CAPSPACE, SLOT_ASPACE, 0, 0, 0]);
        self.ok(Syscall::ThreadStart, [slot, 0x40_0000, 0, 0, 1, 0]);
        self.cap(slot).unwrap().obj
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SysResult {
    pub thread: Paddr,
    pub status: u64,
    pub blocked: bool,
}

/// xorshift64*: the tests' deterministic random source.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    pub fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
}
