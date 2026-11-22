//! aarch64 architecture glue (component B05).
//!
//! The loader enters `_start` at EL1 with the MMU on (the kernel half in
//! TTBR1, all RAM at `PHYS_MAP_BASE`, see `boot-uefi/src/arch/aarch64.rs`).
//! The kernel takes its boot stack and vectors (`cpu.rs`), maps the console
//! UART the ACPI SPCR table names and the GICv3 the MADT describes into its
//! device window (`gic.rs`), sets up the EL1 virtual timer and starts the
//! root task. User address spaces are TTBR0's; the kernel's mappings are
//! TTBR1's and never change. PSCI starts the other processors. Every address-space switch
//! currently flushes the local TLB.

mod cpu;
mod gic;
mod smp;

use core::arch::{asm, global_asm};
use core::cell::Cell;
use core::cell::UnsafeCell;
use core::fmt::Write;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use cpu::{PerCpu, percpu};
use kernel_core::ExitStatus;
use kernel_core::Kernel;
use kernel_core::boot::{BOOT_MAGIC, BOOT_VERSION, BootInfo, Cmdline, PHYS_MAP_BASE, Region, RegionTable};
use kernel_core::hal::Hal;
use kernel_core::lock::Presence;
use kernel_core::mem::{Paddr, Phys};
use kernel_core::root::RootSpec;
use silica_abi::{FAULT_FETCH, FAULT_PRESENT, FAULT_WRITE, Fault, Rights};

const BLOCK: u64 = 2 << 20;
const STACK_SIZE: usize = 64 << 10;

/// The boot processor's stack.
#[repr(C, align(4096))]
struct Stack(UnsafeCell<[u8; STACK_SIZE]>);
// SAFETY: used only as the boot processor's stack, through the stack pointer.
unsafe impl Sync for Stack {}
// SAFETY: the linker puts only this page-aligned stack in the segment.
#[unsafe(link_section = ".boot_stack")]
static BOOT_STACK: Stack = Stack(UnsafeCell::new([0; STACK_SIZE]));

global_asm!(
    r#"
.section .text.boot, "ax"
.global _start
_start:
    msr daifset, #0xf
    adrp x9, {stack}
    add x9, x9, :lo12:{stack}
    add x9, x9, #{size}
    mov sp, x9
    adrp x9, vectors
    add x9, x9, :lo12:vectors
    msr vbar_el1, x9
    isb
    mov x29, #0
    mov x30, #0
    bl {kmain}
    b .

"#,
    stack = sym BOOT_STACK,
    size = const STACK_SIZE,
    kmain = sym kmain,
);

// ---- console ---------------------------------------------------------------

/// The console UART's virtual address (0: none known).
static UART: AtomicU64 = AtomicU64::new(0);
/// The kernel was told a test harness listens for the exit status.
static HARNESS: AtomicBool = AtomicBool::new(false);

fn write_bytes(s: &[u8]) {
    let uart = UART.load(Ordering::Relaxed);
    if uart == 0 {
        return;
    }
    for &b in s {
        let mut spins = 0;
        // SAFETY: `uart` is a PL011-compatible UART mapped as device memory
        // in the device window: flag register at 0x18 (bit 5: transmit FIFO
        // full), data register at 0.
        unsafe {
            while core::ptr::read_volatile((uart + 0x18) as *const u32) & 0x20 != 0 && spins < 100_000 {
                spins += 1;
            }
            core::ptr::write_volatile(uart as *mut u32, b as u32);
        }
    }
}

struct Out;
impl Write for Out {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        write_bytes(s.as_bytes());
        Ok(())
    }
}

/// Stop this processor: tell a test harness the kernel failed, then wait.
fn die() -> ! {
    stop(ExitStatus::Failure)
}

fn stop(status: ExitStatus) -> ! {
    if HARNESS.load(Ordering::Relaxed) {
        // Semihosting SYS_EXIT_EXTENDED, ADP_Stopped_ApplicationExit, with the
        // status encoded as x86's debug-exit device reports it.
        let block = [0x2_0026u64, (status.code() as u64) << 1 | 1];
        // SAFETY: the call reads the two-word block and ends the emulation.
        unsafe { asm!("hlt #0xf000", in("w0") 0x20u32, in("x1") block.as_ptr(), options(nostack)) }
    }
    loop {
        // SAFETY: masking interrupts and waiting change nothing else.
        unsafe { asm!("msr daifset, #0xf", "wfe", options(nomem, nostack)) }
    }
}

// ---- the device window -------------------------------------------------------
//
// Device registers are mapped Device-nGnRE in 2 MiB blocks at DEVICE_BASE,
// through two tables in the kernel image, so the window exists before the
// kernel owns any memory. Normal memory is never mapped over devices.

const DEVICE_BASE: u64 = 0xFFFF_FF00_0000_0000;
const ATTR_DEVICE: u64 = 1 << 2;
const AF: u64 = 1 << 10;
const PXN: u64 = 1 << 53;
const UXN: u64 = 1 << 54;
const ADDR: u64 = 0x0000_FFFF_FFFF_F000;

#[repr(C, align(4096))]
struct Table(UnsafeCell<[u64; 512]>);
// SAFETY: changed only by the boot processor during boot.
unsafe impl Sync for Table {}
static DEVICE_L1: Table = Table(UnsafeCell::new([0; 512]));
static DEVICE_L2: Table = Table(UnsafeCell::new([0; 512]));
static DEVICE_USED: AtomicU64 = AtomicU64::new(0);

/// Where the kernel image is in physical memory: (virtual start, physical start).
static IMAGE: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

fn image_phys(v: u64) -> u64 {
    v - IMAGE[0].load(Ordering::Relaxed) + IMAGE[1].load(Ordering::Relaxed)
}

/// Map `[phys, phys + len)` of device registers at consecutive virtual
/// addresses and return the address of `phys`.
fn map_device_range(phys: u64, len: u64) -> Option<u64> {
    let first = phys & !(BLOCK - 1);
    let blocks = (phys + len.max(1)).div_ceil(BLOCK) - first / BLOCK;
    let va = map_device(first)?;
    // Each further block, new or mapped before, must follow at the next
    // virtual block.
    for b in 1..blocks {
        if map_device(first + b * BLOCK)? != va + b * BLOCK {
            return None;
        }
    }
    Some(va + (phys - first))
}

/// Map the device registers at physical `phys` (with the rest of their
/// 2 MiB block) and return their virtual address. Boot processor, boot only.
fn map_device(phys: u64) -> Option<u64> {
    let block = phys & !(BLOCK - 1);
    // SAFETY: the tables are the boot processor's alone during boot, and
    // TTBR1's root is the loader's table, mapped through the physical map.
    unsafe {
        let l2 = &mut *DEVICE_L2.0.get();
        for (i, e) in l2.iter().enumerate().take(DEVICE_USED.load(Ordering::Relaxed) as usize) {
            if *e & ADDR == block {
                return Some(DEVICE_BASE + i as u64 * BLOCK + (phys - block));
            }
        }
        let i = DEVICE_USED.fetch_add(1, Ordering::Relaxed) as usize;
        if i >= 512 {
            return None;
        }
        if i == 0 {
            let root: u64;
            asm!("mrs {}, ttbr1_el1", out(reg) root, options(nomem, nostack, preserves_flags));
            let l1 = &mut *DEVICE_L1.0.get();
            l1[0] = image_phys(DEVICE_L2.0.get() as u64) | 0b11;
            let root = (PHYS_MAP_BASE + (root & ADDR)) as *mut u64;
            *root.add((DEVICE_BASE >> 39 & 511) as usize) = image_phys(DEVICE_L1.0.get() as u64) | 0b11;
        }
        l2[i] = block | 0b01 | ATTR_DEVICE | AF | PXN | UXN;
        asm!("dsb ishst", "isb", options(nostack, preserves_flags));
        Some(DEVICE_BASE + i as u64 * BLOCK + (phys - block))
    }
}

fn acpi8(p: u64) -> [u8; 8] {
    // SAFETY: ACPI tables are in RAM the physical map covers; callers stay
    // inside the lengths the tables give.
    unsafe { core::ptr::read_unaligned((PHYS_MAP_BASE + p) as *const [u8; 8]) }
}
fn acpi64(p: u64) -> u64 {
    u64::from_le_bytes(acpi8(p))
}
fn acpi32(p: u64) -> u32 {
    acpi64(p) as u32
}

/// The physical address and length of the ACPI table with signature `sig`.
fn acpi_table(rsdp: u64, sig: &[u8; 4]) -> Option<(u64, u64)> {
    if rsdp == 0 || acpi8(rsdp) != *b"RSD PTR " {
        return None;
    }
    let xsdt = acpi64(rsdp + 24);
    if xsdt == 0 || acpi8(xsdt)[..4] != *b"XSDT" {
        return None;
    }
    let len = acpi32(xsdt + 4) as u64;
    let mut at = xsdt + 36;
    while at + 8 <= xsdt + len {
        let t = acpi64(at);
        at += 8;
        if t != 0 && acpi8(t)[..4] == *sig {
            return Some((t, acpi32(t + 4) as u64));
        }
    }
    None
}

/// The base of the UART the ACPI SPCR table names, if it is a PL011 or an
/// SBSA generic UART in memory space.
fn spcr_uart(rsdp: u64) -> Option<u64> {
    let (t, len) = acpi_table(rsdp, b"SPCR")?;
    let (kind, space, base) = (acpi8(t + 36)[0], acpi8(t + 40)[0], acpi64(t + 44));
    (len >= 80 && matches!(kind, 0x03 | 0x0E) && space == 0 && base != 0).then_some(base)
}

/// What the MADT says about the GIC: the distributor, the redistributor
/// region, and the boot processor's redistributor if it is given per
/// processor instead.
struct Madt {
    gicd: u64,
    gicr: (u64, u64),
}

fn madt(rsdp: u64) -> Option<Madt> {
    let (t, len) = acpi_table(rsdp, b"APIC")?;
    let mut m = Madt { gicd: 0, gicr: (0, 0) };
    let me = gic::my_affinity();
    let mut at = t + 44;
    while at + 2 <= t + len {
        let (kind, n) = (acpi8(at)[0], acpi8(at)[1] as u64);
        if n < 2 {
            break;
        }
        match kind {
            0x0C if n >= 24 => m.gicd = acpi64(at + 8),
            0x0E if n >= 16 => m.gicr = (acpi64(at + 4), acpi32(at + 12) as u64),
            // A GICC entry with its own redistributor, used when there is
            // no redistributor region.
            0x0B if n >= 76 && m.gicr.0 == 0 && gic::affinity(acpi64(at + 68)) == me => {
                let r = acpi64(at + 60);
                if r != 0 {
                    m.gicr = (r, 0x2_0000);
                }
            }
            _ => {}
        }
        at += n;
    }
    (m.gicd != 0 && m.gicr.0 != 0).then_some(m)
}

// ---- faults and panics -------------------------------------------------------

/// Check the installed translation on the processor that will use the stack.
fn verify_stack(top: u64) {
    let base = top - STACK_SIZE as u64;
    let (guard, first): (u64, u64);
    // SAFETY: translation probes only update PAR_EL1. Interrupts are masked.
    unsafe {
        asm!("at s1e1r, {below}", "isb", "mrs {guard}, par_el1",
             "at s1e1w, {base}", "isb", "mrs {first}, par_el1",
             below = in(reg) base - 1, base = in(reg) base,
             guard = out(reg) guard, first = out(reg) first, options(nostack));
    }
    // PAR.F and FST[5:2]: a translation fault, at any table level.
    assert!(guard & 0x79 == 9 && first & 1 == 0, "invalid kernel stack mappings at {base:#x}: guard {guard:#x}, stack {first:#x}");
}

/// Walk frame records only on the stack this processor is actually using.
/// The boot stack is available even before per-processor state is installed.
fn backtrace(first: u64, mut fp: u64) {
    let slide = SLIDE.load(Ordering::Relaxed);
    let _ = write!(Out, "#0 {:#018x}\r\n", first.wrapping_sub(slide));
    let sp: u64;
    // SAFETY: reading our current stack pointer.
    unsafe { asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags)) };
    let boot_top = BOOT_STACK.0.get() as u64 + STACK_SIZE as u64;
    let tops = core::iter::once(boot_top)
        .chain((1..kernel_core::smp::MAX_CPUS).filter_map(cpu::percpu_of).map(|p| p.kstack_top.load(Ordering::Relaxed)));
    let Some(hi) = tops.into_iter().find(|&top| (top - STACK_SIZE as u64..top).contains(&sp)) else { return };
    for n in 1..32 {
        if !(fp.is_multiple_of(16) && fp >= sp && fp <= hi - 16) {
            break;
        }
        // SAFETY: `fp` was just checked to lie inside this processor's kernel stack.
        let (next, ret) = unsafe { (*(fp as *const u64), *((fp + 8) as *const u64)) };
        if ret == 0 {
            break;
        }
        let _ = write!(Out, "#{} {:#018x}\r\n", n, ret.wrapping_sub(slide));
        if next <= fp {
            break;
        }
        fp = next;
    }
}

/// `panic=null`: reach a thread object through the null address, which must
/// fault (physical page 0 is never mapped for the kernel), so the test sees
/// a kernel exception with a backtrace through this function.
#[inline(never)]
fn deliberate_null_access(k: &'static Kernel<Arm>) -> ! {
    let v = core::hint::black_box(k.regs(core::hint::black_box(0)))[0].get();
    panic!("physical page 0 is mapped: read {v:#x} through a null object address");
}

#[inline(never)]
fn deliberate_panic(cpu: usize) -> ! {
    panic!("deliberate panic on processor {cpu}");
}

static SLIDE: AtomicU64 = AtomicU64::new(0);

/// The kernel panicked: say why, show where, stop.
pub fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    let _ = write!(Out, "\r\nPANIC: {}\r\n", info.message());
    if let Some(l) = info.location() {
        let _ = write!(Out, "  at {}:{}\r\n", l.file(), l.line());
    }
    let _ = write!(Out, "  kernel slide {:#x}; backtrace (link addresses):\r\n", SLIDE.load(Ordering::Relaxed));
    let (pc, fp): (u64, u64);
    // SAFETY: reads the program counter and the frame pointer.
    unsafe { asm!("adr {}, .", "mov {}, x29", out(reg) pc, out(reg) fp, options(nomem, nostack)) }
    backtrace(pc, fp);
    die()
}

/// An exception taken in the kernel (the only kind there is so far).
extern "C" fn kernel_fault(esr: u64, elr: u64, far: u64, entry: u64, fp: u64) -> ! {
    let _ = write!(
        Out,
        "\r\nPANIC: kernel exception through vector {} (syndrome {:#x}: class {:#x}) at {:#x}, address {:#x}\r\n  kernel slide {:#x}; backtrace (link addresses):\r\n",
        entry,
        esr,
        esr >> 26 & 0x3F,
        elr.wrapping_sub(SLIDE.load(Ordering::Relaxed)),
        far,
        SLIDE.load(Ordering::Relaxed)
    );
    backtrace(elr, fp);
    die()
}

// ---- the kernel's view of the machine ---------------------------------------

/// What the aarch64 layer knows about the machine. Set by the boot
/// processor before any other starts, then only read.
pub struct Arm {
    /// Counter value at monotonic time 0.
    count0: Cell<u64>,
    /// Nanoseconds per tick and ticks per nanosecond, 32.32 fixed point.
    ns_per_tick: Cell<u64>,
    ticks_per_ns: Cell<u64>,
    /// An empty first-level table: TTBR0 when no user address space is loaded.
    empty_root: Cell<u64>,
    /// The largest ASID: 255 or 65535 (the loader enables 16-bit ASIDs
    /// where the processor has them).
    asid_max: Cell<u64>,
}

struct Global(Kernel<Arm>);
// SAFETY: every processor enters the kernel through `Kernel::enter`, and
// kernel-core's lock protocol decides what each may touch. The `Arm` fields
// are written by the boot processor before any other starts.
unsafe impl Sync for Global {}

static KERNEL: Global = Global(Kernel::new(
    Arm { count0: Cell::new(0), ns_per_tick: Cell::new(0), ticks_per_ns: Cell::new(0), empty_root: Cell::new(0), asid_max: Cell::new(255) },
    // SAFETY: the loader maps all RAM at PHYS_MAP_BASE and the kernel only
    // uses addresses from the boot regions through it.
    unsafe { Phys::new(PHYS_MAP_BASE as *mut u8) },
));

/// The kernel.
pub fn kernel() -> &'static Kernel<Arm> {
    &KERNEL.0
}

fn counter_frequency() -> u64 {
    let f: u64;
    // SAFETY: reading the counter frequency.
    unsafe { asm!("mrs {}, cntfrq_el0", out(reg) f, options(nomem, nostack, preserves_flags)) }
    f
}

impl Arm {
    /// The table TTBR0 holds when no user address space is loaded.
    pub fn empty_root(&self) -> u64 {
        self.empty_root.get()
    }

    /// The TTBR0 value that loads `root`. With ASIDs a switch keeps the
    /// translations cached for each address space: it has an ASID per
    /// processor for that processor's current generation, kept in its
    /// header page (words owned by this module from offset 2048, as on
    /// x86_64). A processor hands out its ASIDs in order; when they run out
    /// it starts a new generation, so every address space takes a fresh one
    /// there. An ASID is dropped from this processor's TLB as it is handed
    /// out, so whatever an earlier holder left cached is gone first. ASID 0
    /// is the empty root's.
    fn ttbr0_for(&self, phys: &Phys, root: Paddr) -> u64 {
        if root == self.empty_root.get() {
            return root;
        }
        let me = percpu();
        let generation = me.asid_gen.get();
        // SAFETY: `root` is the root table of a live AddressSpace object,
        // whose header page precedes it; kernel-core uses only its first
        // bytes, and the second half holds one word per processor.
        let slot = unsafe { &mut *phys.ptr(root - 4096 + 2048 + me.index as u64 * 8).cast::<u64>() };
        if *slot >> 16 == generation {
            return root | (*slot & 0xFFFF) << 48;
        }
        let id = me.asid_next.get();
        if id >= self.asid_max.get() {
            me.asid_next.set(1);
            me.asid_gen.set(generation + 1);
        } else {
            me.asid_next.set(id + 1);
        }
        *slot = generation << 16 | id;
        // SAFETY: TLB maintenance only drops cached translations.
        unsafe { asm!("tlbi aside1, {}", "dsb nsh", in(reg) id << 48, options(nostack, preserves_flags)) }
        root | id << 48
    }
}

/// This processor's translation control register.
fn tcr() -> u64 {
    let v: u64;
    // SAFETY: reading a system register.
    unsafe { asm!("mrs {}, tcr_el1", out(reg) v, options(nomem, nostack, preserves_flags)) }
    v
}

fn counter() -> u64 {
    let v: u64;
    // SAFETY: reading the virtual count; ISB keeps it in program order.
    unsafe { asm!("isb", "mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack, preserves_flags)) }
    v
}

fn ttbr0() -> u64 {
    let v: u64;
    // SAFETY: reading the translation table base.
    unsafe { asm!("mrs {}, ttbr0_el1", out(reg) v, options(nomem, nostack, preserves_flags)) }
    v & ADDR
}

/// The ASID TTBR0 holds now.
fn current_asid() -> u64 {
    let v: u64;
    // SAFETY: reading the translation table base.
    unsafe { asm!("mrs {}, ttbr0_el1", out(reg) v, options(nomem, nostack, preserves_flags)) }
    v >> 48
}

/// Load a TTBR0 value (a root and its ASID). Translations cached for other
/// ASIDs stay: user mappings are not global, so each is tagged with its
/// address space's ASID (`Arm::ttbr0_for`).
fn load_ttbr0(value: u64) {
    // SAFETY: TTBR0 holds only user mappings; the kernel runs from TTBR1,
    // which this does not touch.
    unsafe { asm!("msr ttbr0_el1, {}", "isb", in(reg) value, options(nostack, preserves_flags)) }
}

/// Load `root` (with ASID 0) and drop every cached translation on this
/// processor: once at start-up, when the loader's may still be cached.
fn reset_ttbr0(root: u64) {
    // SAFETY: as `load_ttbr0`; TLB maintenance only drops cached translations.
    unsafe { asm!("msr ttbr0_el1, {}", "isb", "tlbi vmalle1", "dsb nsh", "isb", in(reg) root, options(nostack, preserves_flags)) }
}

// Leaf attributes of user mappings.
const PAGE_L3: u64 = 0b11;
const BLOCK_L2: u64 = 0b01;
const AP_EL0: u64 = 1 << 6;
const AP_RO: u64 = 1 << 7;
const SH_INNER: u64 = 3 << 8;
const NG: u64 = 1 << 11;

impl Hal for Arm {
    const IP: usize = cpu::REG_PC;
    const SP: usize = cpu::REG_SP;
    const FLAGS: usize = cpu::REG_SPSR;
    const SYS: usize = 8;
    const BY_SYSCALL: usize = cpu::REG_BY_SVC;
    const ARG: [usize; 6] = [0, 1, 2, 3, 4, 5];
    /// EL0, interrupts unmasked.
    const FLAGS_INIT: u64 = 0;
    const SYSCALL_LEN: u64 = 4;
    const USER_TOP: u64 = 0x0000_7FFF_FFFF_F000;

    fn pte_table(table: Paddr) -> u64 {
        table | 0b11
    }
    fn pte_leaf(frame: Paddr, rights: Rights, level: u8, device: bool) -> u64 {
        // User memory: EL0 access, not global (tagged by address space),
        // never executable by the kernel.
        let mut e = frame | AF | NG | AP_EL0 | SH_INNER | PXN;
        e |= if level == 0 { PAGE_L3 } else { BLOCK_L2 };
        if !rights.has(Rights::WRITE) {
            e |= AP_RO;
        }
        if !rights.has(Rights::EXEC) {
            e |= UXN;
        }
        if device {
            e |= ATTR_DEVICE;
        }
        e
    }
    fn pte_present(pte: u64) -> bool {
        pte & 1 != 0
    }
    fn pte_is_leaf(pte: u64, level: u8) -> bool {
        level == 0 || pte & 0b10 == 0
    }
    fn pte_addr(pte: u64) -> Paddr {
        pte & ADDR
    }

    /// The kernel lives in TTBR1: a user root starts empty.
    fn init_root(&self, _phys: &Phys, _root: Paddr) {}
    /// Retyped memory is zero: all vector registers zero, FPCR default.
    fn init_thread(&self, _phys: &Phys, _thread: Paddr) {}

    fn switch_thread(&self, phys: &Phys, from: Paddr, to: Paddr, root: Paddr) {
        let k = kernel();
        let area = |t: Paddr| phys.ptr(t + kernel_core::thread::FPU_OFFSET as u64);
        // Vector state, as on x86: a thread that entered by SVC keeps only
        // its control and status words (a system call clobbers the vector
        // registers, as a call does) and finds them cleared; an interrupted
        // or faulted thread gets a full save and restore.
        //
        // SAFETY: `from` and `to` are live Thread objects of 4 KiB, whose
        // area from FPU_OFFSET has room for the 512 bytes of q0..q31.
        unsafe {
            if from != 0 {
                let regs = k.regs(from);
                if regs[cpu::REG_BY_SVC].get() == 0 {
                    cpu::silica_simd_save(area(from));
                }
                regs[cpu::REG_FPCTL].set(cpu::silica_fp_control());
            }
            let to_regs = k.regs(to);
            if to_regs[cpu::REG_BY_SVC].get() != 0 {
                cpu::silica_simd_clear();
            } else {
                cpu::silica_simd_load(area(to));
            }
            cpu::silica_fp_set_control(to_regs[cpu::REG_FPCTL].get());
            percpu().regs.set(to_regs.as_ptr() as u64);
        }
        let want = if root == 0 { self.empty_root.get() } else { root };
        if ttbr0() != want {
            load_ttbr0(self.ttbr0_for(phys, want));
        }
    }

    fn leave_aspace(&self, root: Paddr) {
        if ttbr0() == root {
            load_ttbr0(self.empty_root.get());
        }
    }

    fn tlb_flush(&self, root: Paddr, vaddr: u64, len: u64) {
        if ttbr0() != root {
            // Not loaded: forget its ASID here, so its next load takes a
            // fresh one, whose stale translations are dropped first.
            self.tlb_forget(root, percpu().index);
            return;
        }
        let asid = current_asid() << 48;
        // SAFETY: TLB maintenance only drops cached translations.
        unsafe {
            asm!("dsb ishst", options(nostack, preserves_flags));
            if len > 32 * 4096 {
                asm!("tlbi aside1, {}", in(reg) asid, options(nostack, preserves_flags));
            } else {
                for page in (vaddr..vaddr + len).step_by(4096) {
                    asm!("tlbi vae1, {}", in(reg) asid | (page >> 12 & 0xFFF_FFFF_FFFF), options(nostack, preserves_flags));
                }
            }
            asm!("dsb nsh", "isb", options(nostack, preserves_flags));
        }
    }

    fn now(&self) -> u64 {
        let ticks = counter().wrapping_sub(self.count0.get());
        ((ticks as u128 * self.ns_per_tick.get() as u128) >> 32) as u64
    }

    fn set_timer(&self, deadline: u64) {
        let me = percpu();
        if deadline == u64::MAX {
            me.deadline.set(u64::MAX);
            // SAFETY: disabling the EL1 virtual timer.
            unsafe { asm!("msr cntv_ctl_el0, xzr", "isb", options(nostack, preserves_flags)) }
            return;
        }
        let ticks = (deadline as u128 * self.ticks_per_ns.get() as u128) >> 32;
        let at = (self.count0.get() as u128 + ticks).min(u64::MAX as u128 - 1) as u64;
        me.deadline.set(at);
        // SAFETY: an absolute compare value, then enabled and unmasked.
        unsafe {
            asm!("msr cntv_cval_el0, {}", "msr cntv_ctl_el0, {one}", "isb", in(reg) at, one = in(reg) 1u64, options(nostack, preserves_flags))
        }
    }

    fn irq_pending(&self) -> bool {
        counter() >= percpu().deadline.get() || gic::pending()
    }

    fn idle_begin(&self) {
        // "Watching" is said here, under the lock, and never over "idle":
        // a processor that sees "idle" interrupts this one, and an
        // interrupt taken in the watch's window does not end a later halt.
        if kernel_core::sched::IDLE_WATCH_NS != 0 {
            percpu().mode.watch();
        } else {
            percpu().mode.idle();
        }
    }

    fn idle(&self, until: u64) {
        // Watch for work first, as on x86 (`IDLE_WATCH_NS`).
        if kernel_core::sched::IDLE_WATCH_NS != 0 {
            let me = percpu();
            let end = self.now().saturating_add(me.watch.length());
            loop {
                // SAFETY: a window for a pending interrupt, taken by
                // `kernel_irq` as in the wait below; masking is restored.
                unsafe { asm!("msr daifclr, #2", "isb", "msr daifset, #2", options(nomem, nostack)) }
                if !me.mode.watching() {
                    break;
                }
                let now = self.now();
                if now >= until {
                    me.mode.enter_kernel();
                    return;
                }
                if now >= end {
                    break;
                }
                core::hint::spin_loop();
            }
            if !me.mode.stop_watching() {
                me.mode.enter_kernel();
                return;
            }
        }
        self.set_timer(until);
        let halted = self.now();
        // WFI wakes for a pending interrupt even while interrupts are
        // masked, so none is lost between the decision to wait and the
        // wait; it is then taken, briefly unmasked, by `kernel_irq`.
        // SAFETY: masking state is restored before returning.
        unsafe { asm!("dsb sy", "wfi", "msr daifclr, #2", "isb", "msr daifset, #2", options(nomem, nostack)) }
        // Work, not the deadline, ended the halt: the watch learns from it.
        let now = self.now();
        if now < until && kernel_core::sched::IDLE_WATCH_NS != 0 {
            percpu().watch.woken_after(now - halted);
        }
        percpu().mode.enter_kernel();
    }

    fn cpu(&self) -> usize {
        percpu().index
    }

    fn kick(&self, cpu: usize, wait: bool) {
        let Some(them) = cpu::percpu_of(cpu) else { return };
        if !them.online.load(Ordering::Acquire) || them.mode.in_kernel() || them.mode.poke() {
            return;
        }
        gic::kick(them.mpidr.load(Ordering::Relaxed));
        while wait && them.mode.is_user() {
            core::hint::spin_loop();
        }
    }

    fn tlb_reset(&self) {
        // A new generation makes every address space's ASID on this
        // processor stale; each takes a fresh one, dropped from the TLB
        // first, when next loaded.
        let me = percpu();
        me.asid_gen.set(me.asid_gen.get() + 1);
        me.asid_next.set(1);
        load_ttbr0(self.empty_root.get());
    }

    fn tlb_forget(&self, root: Paddr, cpu: usize) {
        // SAFETY: that processor's word in the header page of a live
        // address space (see `ttbr0_for`); the caller holds its lock.
        unsafe { *kernel().phys.ptr(root - 4096 + 2048 + cpu as u64 * 8).cast::<u64>() = 0 }
    }

    fn irq_route(&self, line: usize, cpu: usize, level: bool, active_low: bool) -> bool {
        // GIC inputs are active-high; a platform with an active-low line
        // inverts it before the GIC, so a request for one is an error here.
        !active_low && cpu::percpu_of(cpu).is_some_and(|c| gic::route(line, c.mpidr.load(Ordering::Relaxed), level))
    }

    fn irq_mask(&self, line: usize, masked: bool) {
        gic::mask(line, masked);
    }

    /// There is no I/O port space on aarch64; no port capability exists.
    fn port_io(&self, _port: u16, _width: u8, _write: bool, _value: u32) -> u32 {
        0
    }

    fn console(&self, bytes: &[u8]) {
        write_bytes(bytes);
    }

    fn power_off(&self, status: u64) {
        let code = if status == 0 { ExitStatus::Success } else { ExitStatus::Failure };
        if HARNESS.load(Ordering::Relaxed) {
            stop(code);
        }
        // PSCI SYSTEM_OFF; it returns only if the firmware has no PSCI.
        smp::system_off();
    }

    fn reset(&self, status: u64) {
        let code = if status == 0 { ExitStatus::Success } else { ExitStatus::Failure };
        if HARNESS.load(Ordering::Relaxed) {
            stop(code);
        }
        smp::system_reset();
    }
}

// ---- entry from user mode ----------------------------------------------------

/// An exception from EL0: `kind` 0 synchronous (SVC or a fault), 1 interrupt.
extern "C" fn user_trap(kind: u64) {
    let k = kernel();
    // The entry stub has saved the frame and marked whether a system call
    // saved it (REG_BY_SVC).
    percpu().mode.enter_kernel();
    k.enter();
    if kind != 0 {
        irq(true);
        return;
    }
    let (esr, far): (u64, u64);
    // SAFETY: reading the syndrome and fault address of this exception.
    unsafe { asm!("mrs {}, esr_el1", "mrs {}, far_el1", out(reg) esr, out(reg) far, options(nomem, nostack, preserves_flags)) }
    let class = esr >> 26 & 0x3F;
    if class == 0x15 {
        k.syscall();
        return;
    }
    match class {
        // Instruction abort (0x20) or data abort (0x24) from EL0. Fault
        // status 0x0C-0x0F is a permission fault: something is mapped.
        0x20 | 0x24 => {
            let status = esr & 0x3F;
            let access = if (0x0C..=0x0F).contains(&status) { FAULT_PRESENT } else { 0 }
                | if class == 0x24 && esr >> 6 & 1 != 0 { FAULT_WRITE } else { 0 }
                | if class == 0x20 { FAULT_FETCH } else { 0 };
            k.fault(Fault::Page, far, access);
        }
        // Undefined instruction, illegal execution state, a trapped system
        // register access: an instruction EL0 may not run.
        0x00 | 0x0E | 0x18 => k.fault(Fault::Instruction, far, esr),
        _ => k.fault(Fault::Exception, far, esr),
    }
}

/// Interrupts for this processor until none is pending. `locked`: it
/// interrupted user code and this processor holds its lock; otherwise it
/// interrupted the idle wait, which looks at everything again when it has
/// the lock.
fn irq(locked: bool) {
    let k = kernel();
    loop {
        let id = gic::acknowledge();
        if id >= gic::SPECIAL {
            return;
        }
        match id {
            gic::PPI_TIMER => {
                // One-shot: silence it until the scheduler arms it again.
                // SAFETY: disabling the EL1 virtual timer.
                unsafe { asm!("msr cntv_ctl_el0, xzr", "isb", options(nostack, preserves_flags)) }
                percpu().deadline.set(u64::MAX);
                gic::end(id);
                if locked {
                    k.on_timer();
                }
            }
            gic::SGI_KICK => gic::end(id),
            id if id >= gic::SPI_BASE => {
                let line = (id - gic::SPI_BASE) as usize;
                // A level-triggered line stays asserted until its driver has
                // dealt with the device: hold it off until the driver says so.
                if gic::is_level(line) {
                    gic::mask(line, true);
                }
                gic::end(id);
                if locked {
                    k.irq(line);
                } else {
                    percpu().mode.enter_kernel();
                    k.enter();
                    k.irq(line);
                    percpu().mode.idle();
                    k.leave();
                }
            }
            other => {
                gic::end(other);
                let _ = write!(Out, "kernel: unexpected interrupt {other} on processor {}\r\n", percpu().index);
            }
        }
    }
}

extern "C" fn irq_idle() {
    irq(false);
}

extern "C" fn schedule_rust() {
    let k = kernel();
    k.schedule();
    percpu().mode.leave_to_user();
    k.leave();
}

// ---- boot ------------------------------------------------------------------

extern "C" fn kmain(info: u64) -> ! {
    // SAFETY: the loader passes the physical address of a BootInfo in loader
    // memory, which the physical map covers and nothing overwrites.
    let bi: BootInfo = unsafe { core::ptr::read((PHYS_MAP_BASE + info) as *const BootInfo) };
    if bi.magic != BOOT_MAGIC || bi.version != BOOT_VERSION {
        die();
    }
    IMAGE[0].store(bi.kernel_virt, Ordering::Relaxed);
    IMAGE[1].store(bi.kernel_phys, Ordering::Relaxed);
    SLIDE.store(bi.kernel_slide, Ordering::Relaxed);
    // SAFETY: loader memory the physical map covers, read-only from here on.
    let cmdline = unsafe { core::slice::from_raw_parts((PHYS_MAP_BASE + bi.cmdline) as *const u8, bi.cmdline_len as usize) };
    let cmdline = Cmdline(cmdline);
    HARNESS.store(cmdline.get("exit") == Some(b"semihosting"), Ordering::Relaxed);
    if let Some(uart) = spcr_uart(bi.rsdp).and_then(map_device) {
        UART.store(uart, Ordering::Relaxed);
    }

    // SAFETY: as for the command line.
    let regions = unsafe { core::slice::from_raw_parts((PHYS_MAP_BASE + bi.regions) as *const Region, bi.region_count as usize) };
    let usable: u64 = regions.iter().filter(|r| r.kind == 1).map(|r| r.len).sum();
    let (freq, el): (u64, u64);
    // SAFETY: reading the counter frequency and the exception level.
    unsafe { asm!("mrs {}, cntfrq_el0", "mrs {}, CurrentEL", out(reg) freq, out(reg) el, options(nomem, nostack)) }
    let _ = write!(
        Out,
        "silica: aarch64, {} MiB free, counter {} Hz, EL{}, kernel slide {:#x}\r\n",
        usable >> 20,
        freq,
        el >> 2 & 3,
        bi.kernel_slide
    );
    verify_stack(BOOT_STACK.0.get() as u64 + STACK_SIZE as u64);
    if cmdline.get("panic") == Some(b"test") {
        deliberate_panic(0);
    }
    let k = kernel();
    if cmdline.get("panic") == Some(b"null") {
        deliberate_null_access(k);
    }
    BOOT_CPU.kstack_top.store(BOOT_STACK.0.get() as u64 + STACK_SIZE as u64, Ordering::Relaxed);
    BOOT_CPU.mpidr.store(gic::my_affinity(), Ordering::Relaxed);
    // SAFETY: once, on the boot processor, with interrupts masked; the
    // per-processor data is a static.
    unsafe { cpu::install(&BOOT_CPU) };
    smp::LOW_ROOT.store(ttbr0(), Ordering::Relaxed);

    // The interrupt controller.
    let Some(m) = madt(bi.rsdp) else {
        let _ = write!(Out, "ERROR: kernel: the ACPI MADT describes no GICv3 distributor and redistributor\r\n");
        die();
    };
    let (Some(gicd), Some(gicr_region)) = (map_device_range(m.gicd, 0x1_0000), map_device_range(m.gicr.0, m.gicr.1)) else {
        let _ = write!(Out, "ERROR: kernel: the GIC does not fit the kernel's device window\r\n");
        die();
    };
    gic::init_distributor(gicd);
    let Some(gicr) = gic::find_redistributor(gicr_region, m.gicr.1, gic::my_affinity()) else {
        let _ = write!(Out, "ERROR: kernel: no GIC redistributor for the boot processor\r\n");
        die();
    };
    gic::init_processor(0, gicr);
    BOOT_CPU.online.store(true, Ordering::Release);

    // The other processors' memory comes from the region table before the
    // root task is given the rest. `cpus=1` on the command line keeps them asleep.
    // SAFETY: the boot processor's table, during boot.
    let table = unsafe { &mut *REGIONS.0.get() };
    table.fill(regions);
    let cpus = match cmdline.get("cpus") {
        Some(b"1") => 1,
        _ => smp::prepare(bi.rsdp, table, (gicr_region, m.gicr.1)),
    };
    k.set_cpu_count(cpus);
    let regions = table.as_slice();

    // Time: the generic timer's counter.
    let h = &k.hal;
    h.count0.set(counter());
    h.ns_per_tick.set(((1_000_000_000u128 << 32) / freq as u128) as u64);
    h.ticks_per_ns.set((((freq as u128) << 32) / 1_000_000_000) as u64);
    h.empty_root.set(image_phys(EMPTY_ROOT.0.get() as u64));
    h.asid_max.set(if tcr() & 1 << 36 != 0 { u16::MAX as u64 } else { u8::MAX as u64 });
    reset_ttbr0(h.empty_root.get());
    k.enter();

    // SAFETY: loader memory, read-only until the root task owns it.
    let ramdisk = unsafe { core::slice::from_raw_parts((PHYS_MAP_BASE + bi.ramdisk) as *const u8, bi.ramdisk_len as usize) };
    let Some(image) = kernel_core::ramdisk::find(ramdisk, "root") else {
        let _ = write!(Out, "ERROR: kernel: no root task: the ramdisk is missing, malformed, or has no entry named \"root\"\r\n");
        die();
    };
    let top = regions.last().map_or(0, |r| r.base + r.len).max(4 << 30);
    let uart = spcr_uart(bi.rsdp).unwrap_or(0) & !0xFFF;
    let reserved = [(m.gicd, m.gicd + 0x1_0000), (m.gicr.0, m.gicr.0 + m.gicr.1), (uart, uart + 0x1000)];
    let spec = RootSpec {
        regions,
        elf: image,
        machine: kernel_core::elf::EM_AARCH64,
        // The GIC and the kernel's console stay with the kernel.
        reserved: &reserved,
        device_top: top,
        ports: false,
        ramdisk: [bi.ramdisk, bi.ramdisk_len],
        rsdp: bi.rsdp,
        framebuffer: [bi.fb_base, bi.fb_width as u64, bi.fb_height as u64, bi.fb_stride as u64, bi.fb_format as u64],
        clock: [freq, bi.loader_tsc, h.count0.get()],
        wall_clock: [bi.wall_secs, ((bi.wall_tsc.wrapping_sub(h.count0.get()) as i64 as i128 * 1_000_000_000) / freq as i128) as u64],
        entropy: bi.entropy,
        entropy_sources: bi.entropy_sources,
        cmdline: cmdline.0,
        cpus: cpus as u64,
    };
    if let Err(e) = k.boot_root(&spec) {
        let _ = write!(Out, "ERROR: kernel: cannot start the root task: {e:?}\r\n");
        die();
    }
    smp::launch(cpus, cmdline.get("panic") == Some(b"secondary"));
    // SAFETY: the vectors, the processor and the root thread are set up and
    // the kernel is entered; this is the first exit to user mode.
    unsafe { cpu::exit_to_user() }
}

struct RegionCell(UnsafeCell<RegionTable>);
// SAFETY: used by the boot processor alone, during boot.
unsafe impl Sync for RegionCell {}
static REGIONS: RegionCell = RegionCell(UnsafeCell::new(RegionTable::new()));

/// The boot processor's data.
static BOOT_CPU: PerCpu = PerCpu::new(0, 0, 0);

/// The first-level table TTBR0 holds while no user address space is loaded.
static EMPTY_ROOT: Table = Table(UnsafeCell::new([0; 512]));
