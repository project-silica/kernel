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
/// Processors a test machine can have.
pub const TEST_CPUS: usize = 4;

impl TestHal {
    /// The deadline the current processor's timer was last set to.
    pub fn timer(&self) -> u64 {
        self.timers[self.cpu.get()].get()
    }
}

pub struct TestHal {
    pub now: Cell<u64>,
    /// Each simulated processor's timer: its deadline, and whether it is
    /// armed (set and not yet fired).
    pub timers: [Cell<u64>; TEST_CPUS],
    pub armed: [Cell<bool>; TEST_CPUS],
    /// When set, `irq_pending` is true this many more times.
    pub pending: Cell<u32>,
    pub console: RefCell<Vec<u8>>,
    pub flushes: Cell<u64>,
    pub switches: Cell<u64>,
    pub left: Cell<u64>,
    /// The address-space root each simulated processor has loaded (0: the
    /// kernel's own), as CR3 would hold it. This HAL keeps the kernel in
    /// every root, as x86_64 does, so no root may be built in memory a
    /// processor still has loaded (`init_root` checks).
    pub roots: [Cell<Paddr>; TEST_CPUS],
    /// How many times a processor answered a request to leave a root.
    pub answers: Cell<u64>,
    /// The simulated processor the test is currently acting as.
    pub cpu: Cell<usize>,
    /// (processor, waited) for every kick, in order.
    pub kicks: RefCell<Vec<(usize, bool)>>,
    pub resets: Cell<u64>,
    pub forgets: Cell<u64>,
    /// What the kernel asked of the interrupt controller, in order.
    pub irq_log: RefCell<Vec<String>>,
    /// Idle waits a watched word ended (`IrqWatch`), and watches over words
    /// that ended with nothing.
    pub watch_wakes: Cell<u64>,
    pub watch_ends: Cell<u64>,
    /// A word of the machine's memory (its host address; 0: none) read in
    /// each watch over words: what a device would see while the processor
    /// watches.
    pub spy: Cell<*const u32>,
    pub spied: Cell<u32>,
    /// The machine's IOMMU: how far device addresses reach (0: no IOMMU),
    /// whether it maps 2 MiB with one entry, which device identity is
    /// attached to which root table, and how often its cache was flushed.
    pub io_top: Cell<u64>,
    pub io_large: Cell<bool>,
    pub attached: RefCell<std::collections::BTreeMap<u64, Paddr>>,
    pub io_flushes: Cell<u64>,
}

const ADDR: u64 = 0x000F_FFFF_FFFF_F000;

impl Hal for TestHal {
    const IP: usize = 0;
    const SP: usize = 1;
    const FLAGS: usize = 2;
    const SYS: usize = 3;
    const ARG: [usize; 6] = [4, 5, 6, 7, 8, 9];
    const BY_SYSCALL: usize = 10;
    const FLAGS_INIT: u64 = 0x202;
    const SYSCALL_LEN: u64 = 2;
    const USER_TOP: u64 = 0x7FFF_FFFF_F000;
    const KERNEL_IN_ROOT: bool = true;

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
    fn io_pte_table(table: Paddr) -> u64 {
        table | 3
    }
    fn io_pte_leaf(frame: Paddr, rights: Rights, level: u8) -> u64 {
        frame | 1 | if rights.has(Rights::WRITE) { 2 } else { 0 } | if level == 1 { 0x80 } else { 0 }
    }
    fn io_top(&self) -> u64 {
        self.io_top.get()
    }
    fn io_large(&self) -> bool {
        self.io_large.get()
    }
    fn io_attach(&self, _phys: &Phys, id: u64, root: Paddr) -> Result<Paddr, Error> {
        if self.io_top.get() == 0 || id >> 16 != 0 {
            return Err(Error::Unsupported);
        }
        // Whatever the device had cached of its previous address space goes.
        self.io_flushes.set(self.io_flushes.get() + 1);
        let mut attached = self.attached.borrow_mut();
        let old = if root == 0 { attached.remove(&id) } else { attached.insert(id, root) };
        Ok(old.unwrap_or(0))
    }
    fn io_flush(&self, _phys: &Phys, root: Paddr) {
        assert!(self.attached.borrow().values().any(|r| *r == root), "a flush for an address space no device uses");
        self.io_flushes.set(self.io_flushes.get() + 1);
    }
    fn io_release(&self, _phys: &Phys, root: Paddr) {
        self.attached.borrow_mut().retain(|_, r| *r != root);
    }
    fn msi_first(&self) -> usize {
        64
    }
    fn msi_route(&self, line: usize, cpu: usize, source: u64) -> Option<(u64, u64)> {
        self.irq_log.borrow_mut().push(format!("message {line} cpu{cpu} from {source:#x}"));
        Some((0xFEE0_0000 | (cpu as u64) << 12, 0x30 + line as u64))
    }
    fn init_root(&self, phys: &Phys, root: Paddr) {
        for (c, loaded) in self.roots.iter().enumerate() {
            assert_ne!(loaded.get(), root, "a root is built in memory processor {c} still runs on");
        }
        phys.write_val(root + 511 * 8, &0xDEAD_0000_0000_0003u64);
    }
    fn init_thread(&self, phys: &Phys, thread: Paddr) {
        phys.write_val(thread + kernel_core::thread::FPU_OFFSET as u64, &0x1F80u64);
    }
    fn switch_thread(&self, _phys: &Phys, _from: Paddr, _to: Paddr, root: Paddr) {
        self.switches.set(self.switches.get() + 1);
        self.roots[self.cpu.get()].set(root);
    }
    fn leave_aspace(&self, root: Paddr) {
        self.left.set(root);
        let loaded = &self.roots[self.cpu.get()];
        if loaded.get() == root {
            loaded.set(0);
        }
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
    fn wait_left(&self, cpu: usize, request: &kernel_core::lock::Leave) {
        // The simulated processor is not running: it answers as it would
        // while waiting for its lock (`Kernel::enter`).
        let loaded = &self.roots[cpu];
        if loaded.get() == request.asked() {
            loaded.set(0);
        }
        request.answer();
        self.answers.set(self.answers.get() + 1);
    }
    fn tlb_reset(&self) {
        self.resets.set(self.resets.get() + 1);
        self.roots[self.cpu.get()].set(0);
    }
    fn tlb_forget(&self, _root: Paddr, _cpu: usize) {
        self.forgets.set(self.forgets.get() + 1);
    }
    fn now(&self) -> u64 {
        self.now.get()
    }
    fn set_timer(&self, deadline: u64) {
        let c = self.cpu.get();
        self.timers[c].set(deadline);
        self.armed[c].set(deadline != u64::MAX);
    }
    fn irq_pending(&self) -> bool {
        let n = self.pending.get();
        if n > 0 {
            self.pending.set(n - 1);
        }
        n > 0
    }
    fn idle_begin(&self) {}
    fn idle(&self, until: u64, words: Option<&[kernel_core::hal::Watched]>) -> bool {
        // A watched word already changed: the watch sees it before any
        // timer (as the architecture's watch loop does). Otherwise a watch
        // over words ends with nothing (`watch_ends` of them, then halts as
        // the caller asks next).
        if let Some(words) = words.filter(|w| !w.is_empty()) {
            let changed = || {
                words.iter().any(|w| {
                    // SAFETY: the words are this machine's own memory, in the
                    // window the kernel was given.
                    let v = unsafe { w.at.read_volatile() };
                    v & w.mask != w.seen
                })
            };
            if changed() {
                self.watch_wakes.set(self.watch_wakes.get() + 1);
                return true;
            }
            let spy = self.spy.get();
            if !spy.is_null() {
                // SAFETY: a word in this machine's memory the test named.
                self.spied.set(unsafe { spy.read_volatile() });
            }
            self.watch_ends.set(self.watch_ends.get() + 1);
            return false;
        }
        // As hardware: a timer armed sooner is left; each processor's timer
        // fires once, at its deadline (at once if that has passed), and not
        // again until it is set anew. Idling with none armed would sleep
        // for ever (here: the kernel would spin).
        let c = self.cpu.get();
        if !self.armed[c].get() || until < self.timers[c].get() {
            self.set_timer(until);
        }
        assert!(self.armed[c].replace(false), "idle with no timer armed: the processor would sleep for ever");
        self.now.set(self.now.get().max(self.timers[c].get()));
        true
    }
    fn timer_armed(&self) -> bool {
        self.armed[self.cpu.get()].get()
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
    fn power_off(&self, _: u64) {}
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

    /// The current processor, with nothing to run, waits for work in the
    /// scheduler's idle loop until something wakes a thread (it must: a
    /// timer armed, or a watched word changed).
    pub fn idle(&self) {
        self.k.enter();
        self.k.schedule();
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
        assert!(cpus <= TEST_CPUS, "the test machine has at most {TEST_CPUS} processors");
        let ram = Ram::new((mib << 20) as usize);
        let hal = TestHal {
            now: Cell::new(1000),
            timers: [const { Cell::new(u64::MAX) }; TEST_CPUS],
            armed: [const { Cell::new(false) }; TEST_CPUS],
            pending: Cell::new(0),
            console: RefCell::new(Vec::new()),
            flushes: Cell::new(0),
            switches: Cell::new(0),
            left: Cell::new(0),
            roots: [const { Cell::new(0) }; TEST_CPUS],
            answers: Cell::new(0),
            cpu: Cell::new(0),
            kicks: RefCell::new(Vec::new()),
            resets: Cell::new(0),
            forgets: Cell::new(0),
            irq_log: RefCell::new(Vec::new()),
            watch_wakes: Cell::new(0),
            watch_ends: Cell::new(0),
            spy: Cell::new(core::ptr::null()),
            spied: Cell::new(0),
            io_top: Cell::new(1 << 48),
            io_large: Cell::new(true),
            attached: RefCell::new(std::collections::BTreeMap::new()),
            io_flushes: Cell::new(0),
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
            entropy_sources: 0,
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
            r[TestHal::BY_SYSCALL].set(1);
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
        // Back in user mode: a later look at this thread (here, as on a
        // machine, through an interrupt) finds a frame no system call saved.
        r[TestHal::BY_SYSCALL].set(0);
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
