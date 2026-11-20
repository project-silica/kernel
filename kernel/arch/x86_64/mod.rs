//! x86_64: boot, the `Hal` implementation, the clock and the panic path.
//!
//! Required of the processor: x2APIC, the TSC-deadline timer, an invariant
//! TSC, XSAVE and NX. Every x86_64 processor made since 2013 and QEMU (KVM,
//! or TCG with `-cpu max`) has them; the kernel stops with a message naming
//! the missing one otherwise. There is no PIC, PIT or xAPIC fallback.

mod cpu;
mod serial;

use core::arch::asm;
use core::cell::Cell;
use core::fmt::Write;
use cpu::{cpuid, rdmsr, rdtsc, wrmsr};
use kernel_core::boot::{BOOT_MAGIC, BOOT_VERSION, BootInfo, Cmdline, PHYS_MAP_BASE, Region};
use kernel_core::hal::Hal;
use kernel_core::mem::{Paddr, Phys};
use kernel_core::root::RootSpec;
use kernel_core::thread::FPU_OFFSET;
use kernel_core::{ExitStatus, Kernel, klog};
use silica_abi::Rights;

const REG_RAX: usize = 0;
const REG_RDX: usize = 3;
const REG_RSI: usize = 4;
const REG_RDI: usize = 5;
const REG_R8: usize = 7;
const REG_R9: usize = 8;
const REG_R10: usize = 9;
const REG_RIP: usize = 15;
const REG_RFLAGS: usize = 16;
const REG_RSP: usize = 17;
const REG_FS_BASE: usize = 18;
const REG_GS_BASE: usize = 19;

const PTE_P: u64 = 1;
const PTE_W: u64 = 2;
const PTE_U: u64 = 4;
const PTE_PCD: u64 = 0x10;
const PTE_PS: u64 = 0x80;
const PTE_NX: u64 = 1 << 63;
const PTE_ADDR: u64 = 0x000F_FFFF_FFFF_F000;

const TIMER_VECTOR: u64 = 0x20;
const SPURIOUS_VECTOR: u64 = 0xFF;
const MSR_APIC_BASE: u32 = 0x1B;
const MSR_TSC_DEADLINE: u32 = 0x6E0;
const MSR_X2APIC_EOI: u32 = 0x80B;
const MSR_X2APIC_SVR: u32 = 0x80F;
const MSR_X2APIC_IRR0: u32 = 0x820;
const MSR_X2APIC_LVT_TIMER: u32 = 0x832;
const MSR_FS_BASE: u32 = 0xC000_0100;
const MSR_KERNEL_GS_BASE: u32 = 0xC000_0102;

/// The x86_64 architecture state of the one processor running kernel code.
pub struct X86 {
    tsc_hz: Cell<u64>,
    /// Timestamp counter at monotonic time 0.
    tsc0: Cell<u64>,
    /// Nanoseconds per tick, and ticks per nanosecond, as 32.32 fixed point.
    ns_per_tick: Cell<u64>,
    ticks_per_ns: Cell<u64>,
    /// Timestamp counter value of the armed timer (`u64::MAX`: none).
    deadline: Cell<u64>,
    /// The page table with only the kernel's mappings.
    kernel_root: Cell<u64>,
    slide: Cell<u64>,
    stray_irqs: Cell<u64>,
}

struct Global(Kernel<X86>);
// SAFETY: at M1 exactly one processor runs kernel code, with interrupts off
// except inside `idle`, whose interrupt handlers only use `Cell`s between
// whole operations.
unsafe impl Sync for Global {}

static KERNEL: Global = Global(Kernel::new(
    X86 {
        tsc_hz: Cell::new(0),
        tsc0: Cell::new(0),
        ns_per_tick: Cell::new(0),
        ticks_per_ns: Cell::new(0),
        deadline: Cell::new(u64::MAX),
        kernel_root: Cell::new(0),
        slide: Cell::new(0),
        stray_irqs: Cell::new(0),
    },
    // SAFETY: the loader maps all physical memory at PHYS_MAP_BASE and the
    // kernel only uses addresses from the boot regions through it.
    unsafe { Phys::new(PHYS_MAP_BASE as *mut u8) },
));

/// The kernel.
pub fn kernel() -> &'static Kernel<X86> {
    &KERNEL.0
}

impl Hal for X86 {
    const IP: usize = REG_RIP;
    const SP: usize = REG_RSP;
    const FLAGS: usize = REG_RFLAGS;
    const SYS: usize = REG_RAX;
    const ARG: [usize; 6] = [REG_RDI, REG_RSI, REG_RDX, REG_R10, REG_R8, REG_R9];
    const FLAGS_INIT: u64 = 0x202;
    const SYSCALL_LEN: u64 = 2;
    /// The last page below the canonical boundary stays unmapped, so no
    /// `syscall` instruction can sit where its return address would be
    /// non-canonical.
    const USER_TOP: u64 = 0x0000_7FFF_FFFF_F000;

    fn pte_table(table: Paddr) -> u64 {
        table | PTE_P | PTE_W | PTE_U
    }
    fn pte_leaf(frame: Paddr, rights: Rights, level: u8, device: bool) -> u64 {
        let mut e = frame | PTE_P | PTE_U;
        if rights.has(Rights::WRITE) {
            e |= PTE_W;
        }
        if !rights.has(Rights::EXEC) {
            e |= PTE_NX;
        }
        if level == 1 {
            e |= PTE_PS;
        }
        if device {
            e |= PTE_PCD;
        }
        e
    }
    fn pte_present(pte: u64) -> bool {
        pte & PTE_P != 0
    }
    fn pte_is_leaf(pte: u64, level: u8) -> bool {
        level == 0 || pte & PTE_PS != 0
    }
    fn pte_addr(pte: u64) -> Paddr {
        pte & PTE_ADDR
    }

    fn init_root(&self, phys: &Phys, root: Paddr) {
        // SAFETY: both tables are whole pages of kernel-owned memory; the
        // new root is not in use yet.
        unsafe {
            core::ptr::copy_nonoverlapping(
                phys.ptr(self.kernel_root.get() + 2048),
                phys.ptr(root + 2048),
                2048,
            )
        }
    }

    fn init_thread(&self, phys: &Phys, thread: Paddr) {
        // The legacy area of an XSAVE image: default x87 control word and
        // MXCSR (all exceptions masked). XSTATE_BV stays 0, so XRSTOR puts
        // every other component in its initial state.
        phys.write_val(thread + FPU_OFFSET as u64, &0x037Fu16);
        phys.write_val(thread + FPU_OFFSET as u64 + 24, &0x1F80u32);
    }

    fn switch_thread(&self, phys: &Phys, from: Paddr, to: Paddr, root: Paddr) {
        // SAFETY: `from` and `to` are live Thread objects (4 KiB, so the save
        // area at FPU_OFFSET is 64-byte aligned and large enough: checked
        // against CPUID at boot). The MSRs exist on every x86_64 processor.
        unsafe {
            if from != 0 {
                let area = phys.ptr(from + FPU_OFFSET as u64);
                asm!("xsave64 [{}]", in(reg) area, in("eax") u32::MAX, in("edx") u32::MAX, options(nostack));
                let regs = kernel().regs(from);
                regs[REG_FS_BASE].set(rdmsr(MSR_FS_BASE));
                regs[REG_GS_BASE].set(rdmsr(MSR_KERNEL_GS_BASE));
            }
            let area = phys.ptr(to + FPU_OFFSET as u64);
            asm!("xrstor64 [{}]", in(reg) area, in("eax") u32::MAX, in("edx") u32::MAX, options(nostack));
            let regs = kernel().regs(to);
            wrmsr(MSR_FS_BASE, canonical(regs[REG_FS_BASE].get()));
            wrmsr(MSR_KERNEL_GS_BASE, canonical(regs[REG_GS_BASE].get()));
            cpu::set_regs(regs.as_ptr() as u64);
            let want = if root == 0 {
                self.kernel_root.get()
            } else {
                root
            };
            if cpu::read_cr3() & PTE_ADDR != want {
                cpu::write_cr3(want);
            }
        }
    }

    fn leave_aspace(&self, root: Paddr) {
        if cpu::read_cr3() & PTE_ADDR == root {
            // SAFETY: the kernel root maps the kernel exactly as `root` did.
            unsafe { cpu::write_cr3(self.kernel_root.get()) }
        }
    }

    fn tlb_flush(&self, root: Paddr, vaddr: u64, len: u64) {
        if cpu::read_cr3() & PTE_ADDR != root {
            return; // not loaded: nothing cached (no PCID yet)
        }
        if len > 32 * 4096 {
            // SAFETY: reloading the current root only drops cached translations.
            unsafe { cpu::write_cr3(root) }
            return;
        }
        for page in (vaddr..vaddr + len).step_by(4096) {
            // SAFETY: INVLPG only drops a cached translation.
            unsafe { asm!("invlpg [{}]", in(reg) page, options(nostack, preserves_flags)) }
        }
    }

    fn now(&self) -> u64 {
        let ticks = rdtsc().wrapping_sub(self.tsc0.get());
        ((ticks as u128 * self.ns_per_tick.get() as u128) >> 32) as u64
    }

    fn set_timer(&self, deadline: u64) {
        let tsc = if deadline == u64::MAX {
            u64::MAX
        } else {
            let ticks = (deadline as u128 * self.ticks_per_ns.get() as u128) >> 32;
            // 0 disarms the timer, so a deadline at tick 0 is moved to tick 1.
            (self.tsc0.get() as u128 + ticks).clamp(1, u64::MAX as u128 - 1) as u64
        };
        self.deadline.set(tsc);
        // SAFETY: the TSC-deadline MSR exists (checked at boot); 0 disarms.
        unsafe { wrmsr(MSR_TSC_DEADLINE, if tsc == u64::MAX { 0 } else { tsc }) }
    }

    fn irq_pending(&self) -> bool {
        if rdtsc() >= self.deadline.get() {
            return true;
        }
        // Any request for a vector from 32 up in the local APIC.
        (1..8).any(|i| rdmsr(MSR_X2APIC_IRR0 + i) != 0)
    }

    fn idle(&self) {
        // SAFETY: STI takes effect after the next instruction, so no
        // interrupt can arrive between the decision to wait and HLT. The
        // handler runs on this stack and returns here.
        unsafe { asm!("sti", "hlt", "cli", options(nomem, nostack)) }
    }

    fn port_io(&self, port: u16, width: u8, write: bool, value: u32) -> u32 {
        let mut v = value;
        // SAFETY: the caller holds a capability covering this port range;
        // port I/O does not touch memory.
        unsafe {
            match (width, write) {
                (1, true) => {
                    asm!("out dx, al", in("dx") port, in("al") v as u8, options(nomem, nostack, preserves_flags))
                }
                (2, true) => {
                    asm!("out dx, ax", in("dx") port, in("ax") v as u16, options(nomem, nostack, preserves_flags))
                }
                (_, true) => {
                    asm!("out dx, eax", in("dx") port, in("eax") v, options(nomem, nostack, preserves_flags))
                }
                (1, false) => {
                    let b: u8;
                    asm!("in al, dx", in("dx") port, out("al") b, options(nomem, nostack, preserves_flags));
                    v = b as u32;
                }
                (2, false) => {
                    let w: u16;
                    asm!("in ax, dx", in("dx") port, out("ax") w, options(nomem, nostack, preserves_flags));
                    v = w as u32;
                }
                (_, false) => {
                    asm!("in eax, dx", in("dx") port, out("eax") v, options(nomem, nostack, preserves_flags))
                }
            }
        }
        v
    }

    fn console(&self, bytes: &[u8]) {
        serial::write(bytes);
    }
}

/// A user FS or GS base the processor accepts (it faults on a non-canonical one).
fn canonical(v: u64) -> u64 {
    if v < 1 << 47 || v >= !0 << 47 { v } else { 0 }
}

/// An interrupt from the local APIC or a device.
pub fn irq(vector: u64) {
    let k = kernel();
    match vector {
        TIMER_VECTOR => k.on_timer(),
        SPURIOUS_VECTOR => return,
        v => {
            // Device interrupts are bound to Notifications at M2 (C12).
            // Until then nothing is unmasked, so this is a stray.
            if k.hal.stray_irqs.replace(k.hal.stray_irqs.get() + 1) == 0 {
                klog!(k, "kernel: unexpected interrupt vector {v}");
            }
        }
    }
    // SAFETY: writing 0 to the x2APIC EOI register ends the interrupt in service.
    unsafe { wrmsr(MSR_X2APIC_EOI, 0) }
}

struct Out;
impl Write for Out {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        serial::write(s.as_bytes());
        Ok(())
    }
}

fn die() -> ! {
    serial::outb(0xF4, ExitStatus::Failure.code());
    loop {
        // SAFETY: stop this processor; interrupts are off.
        unsafe { asm!("cli", "hlt", options(nomem, nostack)) }
    }
}

/// Print return addresses by following frame pointers within a kernel stack.
/// Addresses are printed without the load slide, so they match the ELF.
fn backtrace(first: u64, mut rbp: u64) {
    let slide = kernel().hal.slide.get();
    let _ = write!(Out, "#0 {:#018x}\r\n", first.wrapping_sub(slide));
    let on_stack = |a: u64| {
        a % 8 == 0
            && [&cpu::KSTACK]
                .iter()
                .any(|s| a >= s.base() + 4096 && a + 16 <= s.top())
    };
    for n in 1..32 {
        if !on_stack(rbp) {
            break;
        }
        // SAFETY: `rbp` was just checked to lie inside the mapped kernel stack.
        let (next, ret) = unsafe { (*(rbp as *const u64), *((rbp + 8) as *const u64)) };
        if ret == 0 {
            break;
        }
        let _ = write!(Out, "#{} {:#018x}\r\n", n, ret.wrapping_sub(slide));
        rbp = next;
    }
}

/// The kernel panicked: say why, show where, stop.
pub fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    let _ = write!(Out, "\r\nPANIC: {}\r\n", info.message());
    if let Some(l) = info.location() {
        let _ = write!(Out, "  at {}:{}\r\n", l.file(), l.line());
    }
    let _ = write!(
        Out,
        "  kernel slide {:#x}; backtrace (link addresses):\r\n",
        kernel().hal.slide.get()
    );
    let (rip, rbp): (u64, u64);
    // SAFETY: reads the instruction and frame pointers.
    unsafe {
        asm!("lea {}, [rip]", "mov {}, rbp", out(reg) rip, out(reg) rbp, options(nomem, nostack))
    }
    backtrace(rip, rbp);
    die()
}

/// A processor exception in kernel mode: a kernel bug.
pub fn kernel_fault(f: &cpu::KernelFrame) -> ! {
    let cr2: u64;
    // SAFETY: reading CR2 in ring 0 has no side effect.
    unsafe { asm!("mov {}, cr2", out(reg) cr2, options(nomem, nostack, preserves_flags)) }
    let _ = write!(
        Out,
        "\r\nPANIC: kernel exception {} (error {:#x}) at {:#x}, address {:#x}, rsp {:#x}\r\n  kernel slide {:#x}; backtrace (link addresses):\r\n",
        f.vector,
        f.error,
        f.rip,
        cr2,
        f.rsp,
        kernel().hal.slide.get()
    );
    backtrace(f.rip, f.gpr[6]);
    die()
}

fn require(ok: bool, what: &str) {
    if !ok {
        panic!("this processor lacks {what}, which Silica requires");
    }
}

/// Control registers, extended state, the local APIC.
fn init_processor() {
    let f1 = cpuid(1, 0);
    let f7 = cpuid(7, 0);
    require(f1[2] & 1 << 21 != 0, "x2APIC");
    require(f1[2] & 1 << 24 != 0, "the TSC-deadline timer");
    require(f1[2] & 1 << 26 != 0, "XSAVE");
    // SAFETY: boot-time processor setup with interrupts off. Each bit set
    // is one the processor reports supporting.
    unsafe {
        let mut cr0: u64;
        asm!("mov {}, cr0", out(reg) cr0, options(nomem, nostack, preserves_flags));
        cr0 = (cr0 & !(1 << 2 | 1 << 3)) | 1 << 1 | 1 << 5 | 1 << 16; // no EM/TS; MP, NE, WP
        asm!("mov cr0, {}", in(reg) cr0, options(nomem, nostack, preserves_flags));
        let mut cr4: u64;
        asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
        cr4 |= 1 << 7 | 1 << 9 | 1 << 10 | 1 << 18; // PGE, OSFXSR, OSXMMEXCPT, OSXSAVE
        if f7[1] & 1 != 0 {
            cr4 |= 1 << 16; // FSGSBASE
        }
        if f7[1] & 1 << 7 != 0 {
            cr4 |= 1 << 20; // SMEP
        }
        if f7[1] & 1 << 20 != 0 {
            cr4 |= 1 << 21; // SMAP: the kernel never touches user addresses
        }
        if f7[2] & 1 << 2 != 0 {
            cr4 |= 1 << 11; // UMIP
        }
        asm!("mov cr4, {}", in(reg) cr4, options(nomem, nostack, preserves_flags));
        // x87, SSE, AVX, and AVX-512 when its state fits the Thread object.
        let supported = cpuid(0xD, 0)[0] as u64;
        let mut xcr0 = supported & 0b111;
        require(xcr0 & 0b11 == 0b11, "SSE state in XSAVE");
        if supported & 0xE0 == 0xE0 {
            xcr0 |= 0xE0;
        }
        loop {
            asm!("xsetbv", in("ecx") 0u32, in("eax") xcr0 as u32, in("edx") 0u32, options(nomem, nostack, preserves_flags));
            if cpuid(0xD, 0)[1] as usize <= 4096 - FPU_OFFSET {
                break;
            }
            require(
                xcr0 & 0xE0 != 0,
                "an extended state small enough for a Thread object",
            );
            xcr0 &= !0xE0;
        }
        // Legacy interrupt controllers off: remap out of the exception
        // vectors, then mask every line.
        for (port, v) in [
            (0x20, 0x11),
            (0xA0, 0x11),
            (0x21, 0xF0),
            (0xA1, 0xF8),
            (0x21, 4),
            (0xA1, 2),
            (0x21, 1),
            (0xA1, 1),
            (0x21, 0xFF),
            (0xA1, 0xFF),
        ] {
            serial::outb(port, v);
        }
        wrmsr(MSR_APIC_BASE, rdmsr(MSR_APIC_BASE) | 1 << 11 | 1 << 10);
        wrmsr(MSR_X2APIC_SVR, 0x100 | SPURIOUS_VECTOR);
        wrmsr(0x808, 0); // task priority: accept everything
        wrmsr(0x835, 0x1_0000); // LINT0 masked
        wrmsr(0x836, 0x400); // LINT1: NMI
        wrmsr(0x837, 0x1_0000); // error LVT masked
        wrmsr(MSR_X2APIC_LVT_TIMER, TIMER_VECTOR | 0b10 << 17); // TSC-deadline mode
        wrmsr(MSR_TSC_DEADLINE, 0);
    }
}

/// The frequency of the timestamp counter.
fn tsc_frequency(bi: &BootInfo, phys: &Phys) -> u64 {
    // 1. The processor says (Intel: crystal frequency and ratio).
    if cpuid(0, 0)[0] >= 0x15 {
        let [den, num, hz, _] = cpuid(0x15, 0);
        if den != 0 && num != 0 && hz != 0 {
            return hz as u64 * num as u64 / den as u64;
        }
    }
    // 2. The hypervisor says (KVM paravirtual clock: exact scale factors).
    let hv = cpuid(0x4000_0000, 0);
    if cpuid(1, 0)[2] & 1 << 31 != 0
        && hv[1] == 0x4B4D_564B
        && hv[2] == 0x564B_4D56
        && cpuid(0x4000_0001, 0)[0] & 1 << 3 != 0
    {
        #[repr(C, align(64))]
        struct PvClock(core::cell::UnsafeCell<[u32; 16]>);
        // SAFETY: written by the hypervisor only between the two MSR writes below.
        unsafe impl Sync for PvClock {}
        static PV: PvClock = PvClock(core::cell::UnsafeCell::new([0; 16]));
        let pa = PV.0.get() as u64 - bi.kernel_virt + bi.kernel_phys;
        // SAFETY: MSR_KVM_SYSTEM_TIME_NEW exists (feature bit checked); the
        // structure is 32 bytes inside one page of the kernel image.
        let (mul, shift) = unsafe {
            wrmsr(0x4B56_4D01, pa | 1);
            let p = PV.0.get().cast::<u8>();
            let v = (
                core::ptr::read_volatile(p.add(24).cast::<u32>()),
                core::ptr::read_volatile(p.add(28).cast::<i8>()),
            );
            wrmsr(0x4B56_4D01, 0);
            v
        };
        if mul != 0 {
            // ns = ((ticks << shift) * mul) >> 32
            let hz = (1_000_000_000u128 << 32) / mul as u128;
            return if shift >= 0 {
                (hz >> shift) as u64
            } else {
                (hz << -shift) as u64
            };
        }
    }
    // 3. Measure against the ACPI power-management timer (3.579545 MHz).
    if let Some(port) = acpi_pm_timer(bi, phys) {
        let read = || kernel().hal.port_io(port, 4, false, 0) & 0xFF_FFFF;
        let start = read();
        let t0 = rdtsc();
        let ticks = 35_795; // 10 ms
        while read().wrapping_sub(start) & 0xFF_FFFF < ticks {}
        return (rdtsc() - t0) * 100;
    }
    // 4. The processor's nominal frequency.
    if cpuid(0, 0)[0] >= 0x16 && cpuid(0x16, 0)[0] != 0 {
        return cpuid(0x16, 0)[0] as u64 * 1_000_000;
    }
    panic!(
        "no way to measure the timestamp counter: no CPUID leaf 0x15, no KVM clock, no ACPI PM timer"
    )
}

/// The I/O port of the ACPI PM timer, from the FADT.
fn acpi_pm_timer(bi: &BootInfo, phys: &Phys) -> Option<u16> {
    let limit = bi.phys_map_len;
    let table = |p: u64, len: u64| {
        // SAFETY: the range is checked to be inside the physical map; ACPI
        // tables are firmware memory nothing writes.
        (p != 0 && p.checked_add(len)? <= limit).then(|| unsafe { phys.bytes(p, len) })
    };
    let u32_at = |b: &[u8], o: usize| Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?));
    let u64_at = |b: &[u8], o: usize| Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?));
    let rsdp = table(bi.rsdp, 36)?;
    let xsdt_addr = u64_at(rsdp, 24)?;
    let len = u32_at(table(xsdt_addr, 36)?, 4)? as u64;
    let xsdt = table(xsdt_addr, len)?;
    for i in 0..(len as usize).saturating_sub(36) / 8 {
        let t = u64_at(xsdt, 36 + i * 8)?;
        let Some(h) = table(t, 80) else { continue };
        if &h[..4] == b"FACP" {
            let port = u32_at(h, 76)?;
            return (port != 0 && port < 0xFFFC).then_some(port as u16);
        }
    }
    None
}

/// The entry of the kernel's own page table that maps `vaddr`, at the
/// 4 KiB level.
fn kernel_pte(phys: &Phys, vaddr: u64) -> Option<*mut u64> {
    let mut table = cpu::read_cr3() & PTE_ADDR;
    for level in (0..4).rev() {
        let e = phys
            .ptr(table + (vaddr >> (12 + 9 * level) & 511) * 8)
            .cast::<u64>();
        if level == 0 {
            return Some(e);
        }
        // SAFETY: `table` is a page-table page of the tables the loader
        // built, reached through the physical map.
        let v = unsafe { *e };
        if v & PTE_P == 0 || v & PTE_PS != 0 {
            return None;
        }
        table = v & PTE_ADDR;
    }
    None
}

#[inline(never)]
fn deliberate_panic() {
    panic!("deliberate panic requested with panic=test");
}

/// Kernel entry from the boot stub, on the kernel stack, interrupts off.
pub extern "C" fn kmain(info: u64) -> ! {
    serial::init();
    let k = kernel();
    // SAFETY: the loader passes the physical address of a BootInfo in
    // loader memory, which nothing overwrites before the root task runs.
    let bi: BootInfo = unsafe { core::ptr::read(k.phys.ptr(info).cast::<BootInfo>()) };
    if bi.magic != BOOT_MAGIC || bi.version != BOOT_VERSION {
        serial::write(
            b"ERROR: kernel: the loader's boot information has the wrong magic or version\r\n",
        );
        die();
    }
    k.hal.slide.set(bi.kernel_slide);
    // SAFETY: first thing after the console, once, on the boot stack.
    unsafe { cpu::init() };
    init_processor();

    // The kernel's own address space: drop the loader's low alias of
    // physical memory, make the physical map non-executable, and unmap the
    // page below the stack so an overflow faults instead of corrupting data.
    let root = cpu::read_cr3() & PTE_ADDR;
    k.hal.kernel_root.set(root);
    let guard = kernel_pte(&k.phys, cpu::KSTACK.base());
    // SAFETY: edits the live kernel page table in ways that keep every
    // address the kernel uses mapped: it runs from the image mapping and
    // reaches memory through the physical map, neither of which changes
    // except for losing execute permission on the latter.
    unsafe {
        let pml4 = k.phys.ptr(root).cast::<u64>();
        *pml4 = 0;
        *pml4.add(256) |= PTE_NX;
        if let Some(g) = guard {
            *g = 0;
        }
        // Toggling CR4.PGE drops every cached translation, global ones too.
        let cr4: u64;
        asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
        asm!("mov cr4, {}", in(reg) cr4 & !(1 << 7), options(nostack, preserves_flags));
        asm!("mov cr4, {}", in(reg) cr4, options(nostack, preserves_flags));
    }

    let hz = tsc_frequency(&bi, &k.phys);
    k.hal.tsc_hz.set(hz);
    k.hal.tsc0.set(rdtsc());
    k.hal
        .ns_per_tick
        .set(((1_000_000_000u128 << 32) / hz as u128) as u64);
    k.hal
        .ticks_per_ns
        .set((((hz as u128) << 32) / 1_000_000_000) as u64);

    // SAFETY: loader memory, read-only until the root task owns it.
    let (regions, cmdline, ramdisk) = unsafe {
        (
            core::slice::from_raw_parts(
                k.phys.ptr(bi.regions).cast::<Region>(),
                bi.region_count as usize,
            ),
            k.phys.bytes(bi.cmdline, bi.cmdline_len),
            k.phys.bytes(bi.ramdisk, bi.ramdisk_len),
        )
    };
    let usable: u64 = regions.iter().filter(|r| r.kind == 1).map(|r| r.len).sum();
    klog!(
        k,
        "silica: {} MiB free, clock {} Hz, kernel slide {:#x}",
        usable >> 20,
        hz,
        bi.kernel_slide
    );
    if Cmdline(cmdline).get("panic") == Some(b"test") {
        deliberate_panic();
    }

    let Some(image) = kernel_core::ramdisk::find(ramdisk, "root") else {
        klog!(
            k,
            "ERROR: kernel: no root task: the ramdisk is missing, malformed, or has no entry named \"root\""
        );
        die();
    };
    let top = regions.last().map_or(0, |r| r.base + r.len).max(4 << 30);
    let spec = RootSpec {
        regions,
        elf: image,
        machine: kernel_core::elf::EM_X86_64,
        // The I/O APIC and local APIC stay with the kernel.
        reserved: &[(0xFEC0_0000, 0xFEF0_0000)],
        device_top: top,
        ports: true,
        ramdisk: [bi.ramdisk, bi.ramdisk_len],
        rsdp: bi.rsdp,
        framebuffer: [
            bi.fb_base,
            bi.fb_width as u64,
            bi.fb_height as u64,
            bi.fb_stride as u64,
            bi.fb_format as u64,
        ],
        clock: [hz, bi.loader_tsc, k.hal.tsc0.get()],
        wall_clock: [
            bi.wall_secs,
            ((bi.wall_tsc.wrapping_sub(k.hal.tsc0.get()) as i64 as i128 * 1_000_000_000)
                / hz as i128) as u64,
        ],
        entropy: bi.entropy,
        cmdline,
    };
    if let Err(e) = k.boot_root(&spec) {
        klog!(k, "ERROR: kernel: cannot start the root task: {e:?}");
        die();
    }
    // SAFETY: the tables, the processor and the root thread are set up;
    // this is the first exit to user mode.
    unsafe { cpu::exit_to_user() }
}
