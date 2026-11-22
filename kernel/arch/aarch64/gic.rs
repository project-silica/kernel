//! The GICv3 interrupt controller.
//!
//! Interrupt identifiers: SGI 0 is the kick between processors, PPI 27 the
//! EL1 virtual timer, device line `n` below `its::FIRST_LINE` (a Hw
//! interrupt-line capability's number) is SPI `32 + n`, and the lines from
//! there up are message lines, LPIs (`its.rs`). Everything is Group 1, non-secure, one priority.
//! The distributor and each processor's redistributor are found through the
//! ACPI MADT; the CPU interface is reached through system registers.
//!
//! A level-triggered device line is disabled when it fires and enabled
//! again when its driver acknowledges it (`mask(line, false)`), so a device
//! that keeps its line asserted costs its driver, not a processor.

use core::arch::asm;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use kernel_core::smp::MAX_CPUS;
use silica_abi::IRQ_LINES;

/// The kick between processors.
pub const SGI_KICK: u32 = 0;
/// The EL1 virtual timer.
pub const PPI_TIMER: u32 = 27;
/// The first SPI: device line 0.
pub const SPI_BASE: u32 = 32;
/// Identifiers 1020 to 1023 are not interrupts (spurious, special); those
/// from 8192 on are LPIs (`its.rs`).
pub const SPECIAL: u32 = 1020;

const PRIORITY: u8 = 0x80;

/// Distributor (virtual address in the device window).
static GICD: AtomicU64 = AtomicU64::new(0);
/// Number of SPIs the distributor implements.
static SPIS: AtomicU64 = AtomicU64::new(0);
/// Each processor's redistributor (virtual) and affinity (MPIDR, Aff3..Aff0).
pub static GICR: [AtomicU64; MAX_CPUS] = [const { AtomicU64::new(0) }; MAX_CPUS];
pub static MPIDR: [AtomicU64; MAX_CPUS] = [const { AtomicU64::new(u64::MAX) }; MAX_CPUS];
/// Device lines configured level-triggered.
static LEVEL: [AtomicBool; IRQ_LINES as usize] = [const { AtomicBool::new(false) }; IRQ_LINES as usize];

fn rd32(a: u64) -> u32 {
    // SAFETY: `a` is a GIC register in the device window, mapped Device-nGnRE.
    unsafe { core::ptr::read_volatile(a as *const u32) }
}
fn wr32(a: u64, v: u32) {
    // SAFETY: as for `rd32`.
    unsafe { core::ptr::write_volatile(a as *mut u32, v) }
}
fn wr64(a: u64, v: u64) {
    // SAFETY: as for `rd32`.
    unsafe { core::ptr::write_volatile(a as *mut u64, v) }
}
fn rd64(a: u64) -> u64 {
    // SAFETY: as for `rd32`.
    unsafe { core::ptr::read_volatile(a as *const u64) }
}

/// The affinity of an MPIDR value, as the GIC compares it (Aff3 in 39:32).
pub fn affinity(mpidr: u64) -> u64 {
    mpidr & 0xFF_00FF_FFFF
}

/// This processor's affinity.
pub fn my_affinity() -> u64 {
    let m: u64;
    // SAFETY: reading the processor's own identifier.
    unsafe { asm!("mrs {}, mpidr_el1", out(reg) m, options(nomem, nostack, preserves_flags)) }
    affinity(m)
}

fn wait_rwp(gicd: u64) {
    for _ in 0..1_000_000 {
        if rd32(gicd) & 1 << 31 == 0 {
            return;
        }
        core::hint::spin_loop();
    }
}

/// Set up the distributor (boot processor, once). `gicd` is its virtual
/// address. Every SPI starts disabled, Group 1, routed nowhere in particular.
pub fn init_distributor(gicd: u64) {
    GICD.store(gicd, Ordering::Relaxed);
    let lines = ((rd32(gicd + 0x4) & 0x1F) + 1) * 32; // GICD_TYPER.ITLinesNumber
    let spis = lines.min(1020) - SPI_BASE;
    SPIS.store(spis as u64, Ordering::Relaxed);
    wr32(gicd, 0);
    wait_rwp(gicd);
    for n in (SPI_BASE..SPI_BASE + spis).step_by(32) {
        let i = (n / 32) as u64 * 4;
        wr32(gicd + 0x180 + i, u32::MAX); // ICENABLER: disabled
        wr32(gicd + 0x280 + i, u32::MAX); // ICPENDR
        wr32(gicd + 0x080 + i, u32::MAX); // IGROUPR: Group 1
    }
    for n in SPI_BASE..SPI_BASE + spis {
        // SAFETY: IPRIORITYR is byte-accessible.
        unsafe { core::ptr::write_volatile((gicd + 0x400 + n as u64) as *mut u8, PRIORITY) };
    }
    // Affinity routing (ARE), Group 1 enabled (bit 1 in the non-secure view
    // of a two-state GIC, bit 0 in a one-state GIC: both are set).
    wr32(gicd, 1 << 4 | 1 << 1 | 1);
    wait_rwp(gicd);
}

/// Wake this processor's redistributor, enable its kick and timer, and turn
/// on its CPU interface. `gicr` is the redistributor's virtual address.
pub fn init_processor(cpu: usize, gicr: u64) {
    GICR[cpu].store(gicr, Ordering::Relaxed);
    // GICR_WAKER: clear ProcessorSleep, wait for ChildrenAsleep to clear.
    wr32(gicr + 0x14, rd32(gicr + 0x14) & !(1 << 1));
    for _ in 0..1_000_000 {
        if rd32(gicr + 0x14) & 1 << 2 == 0 {
            break;
        }
        core::hint::spin_loop();
    }
    let sgi = gicr + 0x1_0000;
    wr32(sgi + 0x080, u32::MAX); // IGROUPR0: Group 1
    wr32(sgi + 0x180, u32::MAX); // ICENABLER0: all off
    for n in 0..32u64 {
        // SAFETY: IPRIORITYR is byte-accessible.
        unsafe { core::ptr::write_volatile((sgi + 0x400 + n) as *mut u8, PRIORITY) };
    }
    wr32(sgi + 0x100, 1 << SGI_KICK | 1 << PPI_TIMER); // ISENABLER0
    // SAFETY: the CPU interface's system registers: system-register access
    // on, every priority let through, Group 1 interrupts on.
    unsafe {
        asm!(
            "mrs {t}, icc_sre_el1",
            "orr {t}, {t}, #1",
            "msr icc_sre_el1, {t}",
            "isb",
            "msr icc_pmr_el1, {pmr}",
            "msr icc_bpr1_el1, xzr",
            "msr icc_igrpen1_el1, {one}",
            "isb",
            t = out(reg) _,
            pmr = in(reg) 0xFFu64,
            one = in(reg) 1u64,
            options(nostack, preserves_flags),
        )
    }
}

/// The redistributor frame for `mpidr` inside `[base, base + len)`.
pub fn find_redistributor(base: u64, len: u64, mpidr: u64) -> Option<u64> {
    let mut at = base;
    while at + 0x2_0000 <= base + len {
        let typer = rd64(at + 0x8);
        if typer >> 32 == affinity_for_typer(mpidr) {
            return Some(at);
        }
        if typer & 1 << 4 != 0 {
            return None; // GICR_TYPER.Last
        }
        // GICv4 frames have two more 64 KiB pages (VLPI).
        at += if typer & 1 << 1 != 0 { 0x4_0000 } else { 0x2_0000 };
    }
    None
}

/// GICR_TYPER's affinity field: Aff3.Aff2.Aff1.Aff0 in 32 bits.
fn affinity_for_typer(mpidr: u64) -> u64 {
    (mpidr >> 32 & 0xFF) << 24 | (mpidr & 0xFF_FFFF)
}

/// Take the highest-priority pending interrupt (`SPECIAL` to `SPECIAL + 3`: none).
pub fn acknowledge() -> u32 {
    let id: u64;
    // SAFETY: reading the acknowledge register is how an interrupt is taken.
    unsafe { asm!("mrs {}, icc_iar1_el1", out(reg) id, options(nomem, nostack, preserves_flags)) }
    id as u32 & 0xFF_FFFF
}

/// Finish with interrupt `id` (priority drop and deactivation).
pub fn end(id: u32) {
    // SAFETY: writing the identifier taken from `acknowledge`.
    unsafe { asm!("msr icc_eoir1_el1, {}", in(reg) id as u64, options(nomem, nostack, preserves_flags)) }
}

/// Interrupt the processor whose affinity is `mpidr` with the kick SGI.
pub fn kick(mpidr: u64) {
    let aff1 = mpidr >> 8 & 0xFF;
    let aff2 = mpidr >> 16 & 0xFF;
    let aff3 = mpidr >> 32 & 0xFF;
    let target = 1u64 << (mpidr & 0xF);
    let rs = (mpidr & 0xFF) >> 4;
    let v = aff3 << 48 | aff2 << 32 | rs << 44 | aff1 << 16 | (SGI_KICK as u64) << 24 | target;
    // SAFETY: ICC_SGI1R_EL1 sends a Group 1 SGI; the barrier orders the
    // memory the kicked processor will look at before the interrupt.
    unsafe { asm!("dsb ishst", "msr icc_sgi1r_el1, {}", "isb", in(reg) v, options(nostack, preserves_flags)) }
}

/// Whether device line `line` is a wire that exists (the message lines,
/// from `its::FIRST_LINE`, are the ITS's).
pub fn has_line(line: usize) -> bool {
    line < super::its::FIRST_LINE.min(IRQ_LINES as usize) && (line as u64) < SPIS.load(Ordering::Relaxed)
}

/// Route device line `line` to the processor with affinity `mpidr`, with
/// the given trigger, and enable it. (Polarity is fixed active-high on the
/// GIC; an active-low line is inverted before it reaches it.)
pub fn route(line: usize, mpidr: u64, level: bool) -> bool {
    let gicd = GICD.load(Ordering::Relaxed);
    if gicd == 0 || !has_line(line) {
        return false;
    }
    let id = SPI_BASE as u64 + line as u64;
    let cfg = gicd + 0xC00 + id / 16 * 4;
    let bit = 2 * (id % 16) + 1;
    let v = rd32(cfg);
    wr32(cfg, if level { v & !(1 << bit) } else { v | 1 << bit });
    wr64(gicd + 0x6000 + id * 8, affinity(mpidr));
    if let Some(l) = LEVEL.get(line) {
        l.store(level, Ordering::Relaxed);
    }
    mask(line, false);
    true
}

/// Disable or enable device line `line`.
pub fn mask(line: usize, masked: bool) {
    if line >= super::its::FIRST_LINE {
        return super::its::mask(line, masked);
    }
    let gicd = GICD.load(Ordering::Relaxed);
    if gicd == 0 || !has_line(line) {
        return;
    }
    let id = SPI_BASE as u64 + line as u64;
    wr32(gicd + if masked { 0x180 } else { 0x100 } + id / 32 * 4, 1 << (id % 32));
}

/// Whether device line `line` is level-triggered.
pub fn is_level(line: usize) -> bool {
    LEVEL.get(line).is_some_and(|l| l.load(Ordering::Relaxed))
}

/// Whether an interrupt is waiting for this processor (polled by long
/// kernel operations).
pub fn pending() -> bool {
    let v: u64;
    // SAFETY: reading the highest pending interrupt changes nothing.
    unsafe { asm!("mrs {}, icc_hppir1_el1", out(reg) v, options(nomem, nostack, preserves_flags)) }
    !(SPECIAL..SPECIAL + 4).contains(&(v as u32 & 0xFF_FFFF))
}
