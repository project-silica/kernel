//! I/O APICs: routing device interrupt lines to processors (component C12).
//!
//! A line is a global system interrupt number as ACPI counts them. Line `n`
//! arrives at vector `0x30 + n` on the processor it was routed to, edge- or
//! level-triggered as its driver said when it bound it. A level-triggered
//! line is masked in the interrupt handler and unmasked by `IrqAck`, so a
//! device that keeps its line asserted cannot keep a processor in the kernel.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use kernel_core::boot::BootInfo;
use kernel_core::lock::Lock;
use kernel_core::mem::Phys;
use silica_abi::IRQ_LINES;

/// First vector used for device lines.
pub const FIRST_VECTOR: u64 = 0x30;

#[derive(Clone, Copy)]
struct IoApic {
    base: u64,
    first: u32,
    count: u32,
    /// Its identifier, as the MADT and the DMAR name it.
    id: u8,
}

struct Racy(UnsafeCell<[IoApic; 8]>);
// SAFETY: filled by the boot processor before any interrupt is routed, then only read.
unsafe impl Sync for Racy {}
static IOAPICS: Racy = Racy(UnsafeCell::new([IoApic { base: 0, first: 0, count: 0, id: 0 }; 8]));
static COUNT: AtomicUsize = AtomicUsize::new(0);
/// The index and data registers of an I/O APIC are used as a pair.
static REGS: Lock = Lock::new();
static LEVEL: [AtomicBool; IRQ_LINES as usize] = [const { AtomicBool::new(false) }; IRQ_LINES as usize];

fn read(phys: &Phys, base: u64, reg: u32) -> u32 {
    // SAFETY: the I/O APIC's two registers, at the address the firmware
    // gave, inside the physical map; callers hold `REGS`.
    unsafe {
        phys.ptr(base).cast::<u32>().write_volatile(reg);
        phys.ptr(base + 0x10).cast::<u32>().read_volatile()
    }
}

fn write(phys: &Phys, base: u64, reg: u32, v: u32) {
    // SAFETY: as for `read`.
    unsafe {
        phys.ptr(base).cast::<u32>().write_volatile(reg);
        phys.ptr(base + 0x10).cast::<u32>().write_volatile(v);
    }
}

/// Find the I/O APICs in the ACPI MADT and mask every line. Boot processor, once.
pub fn init(bi: &BootInfo, phys: &Phys) {
    let Some(madt) = super::acpi_table(bi, phys, b"APIC") else { return };
    let (mut n, mut o) = (0, 44);
    // SAFETY: boot processor, before any interrupt is routed.
    let list = unsafe { &mut *IOAPICS.0.get() };
    while o + 2 <= madt.len() && n < list.len() {
        let (ty, len) = (madt[o], madt[o + 1] as usize);
        if len < 2 || o + len > madt.len() {
            break;
        }
        if ty == 1 && len >= 12 {
            let u32_at = |i: usize| u32::from_le_bytes([madt[o + i], madt[o + i + 1], madt[o + i + 2], madt[o + i + 3]]);
            let (base, first) = (u32_at(4) as u64, u32_at(8));
            if base != 0 && base + 0x20 <= bi.phys_map_len {
                super::check_uncached(phys, base, "an I/O APIC");
                let count = (read(phys, base, 1) >> 16 & 0xFF) + 1;
                // Every pin masked; written only where it is not already
                // (the reset state, which firmware mostly leaves). Under a
                // hypervisor that keeps the I/O APIC outside the kernel
                // (QEMU's split irqchip, which interrupt remapping needs)
                // each write of an entry changes the machine's interrupt
                // routes, which can wait milliseconds to take effect:
                // writing all 24 took 0.28 ms, or 3.1 ms in one boot of three.
                for pin in 0..count {
                    if read(phys, base, 0x10 + 2 * pin) & 1 << 16 == 0 {
                        write(phys, base, 0x10 + 2 * pin, 1 << 16);
                    }
                }
                list[n] = IoApic { base, first, count, id: madt[o + 2] };
                n += 1;
            }
        }
        o += len;
    }
    COUNT.store(n, Ordering::Release);
}

fn find(line: usize) -> Option<(u64, u32, u8)> {
    // SAFETY: read-only after `init`.
    let list = unsafe { &*IOAPICS.0.get() };
    list[..COUNT.load(Ordering::Acquire)]
        .iter()
        .find(|a| line as u32 >= a.first && (line as u32) < a.first + a.count)
        .map(|a| (a.base, line as u32 - a.first, a.id))
}

/// One past the highest line that is a wire into an I/O APIC: the lines
/// from here up are raised by messages.
pub fn lines() -> usize {
    // SAFETY: read-only after `init`.
    let list = unsafe { &*IOAPICS.0.get() };
    list[..COUNT.load(Ordering::Acquire)].iter().map(|a| (a.first + a.count) as usize).max().unwrap_or(0).min(IRQ_LINES as usize)
}

/// Deliver `line` to the processor with APIC identifier `apic_id` and unmask it.
/// Where interrupts are remapped (`iommu::remapping`) the entry names the
/// line's remapping entry, which says where it goes.
pub fn route(phys: &Phys, line: usize, apic_id: u32, level: bool, active_low: bool) -> bool {
    let Some((base, pin, id)) = find(line) else { return false };
    if line >= IRQ_LINES as usize {
        return false;
    }
    let vector = FIRST_VECTOR as u32 + line as u32;
    let (high, low) = if super::iommu::remapping() {
        let Some(h) = super::iommu::remap(phys, line, apic_id, vector as u8, level, super::iommu::ioapic_source(id)) else { return false };
        // Remappable format: handle bits 14..0 at 63..49, bit 15 at 11,
        // the format bit at 48; the vector stays the entry's (EOI matching).
        let h = h as u32;
        ((h & 0x7FFF) << 17 | 1 << 16, vector | (h >> 15 & 1) << 11)
    } else if apic_id > 0xFE {
        return false;
    } else {
        (apic_id << 24, vector)
    };
    LEVEL[line].store(level, Ordering::Release);
    let low = low | (active_low as u32) << 13 | (level as u32) << 15;
    REGS.lock();
    write(phys, base, 0x10 + 2 * pin, 1 << 16);
    write(phys, base, 0x11 + 2 * pin, high);
    write(phys, base, 0x10 + 2 * pin, low);
    REGS.unlock();
    true
}

/// Deliver `line`, routed already and not remapped, to the processor with
/// APIC identifier `apic_id`: its destination field rewritten with the
/// entry held off meanwhile, so no interrupt goes to half of each.
pub fn retarget(phys: &Phys, line: usize, apic_id: u32) -> bool {
    let Some((base, pin, _)) = find(line) else { return false };
    if apic_id > 0xFE {
        return false;
    }
    REGS.lock();
    let low = read(phys, base, 0x10 + 2 * pin);
    write(phys, base, 0x10 + 2 * pin, low | 1 << 16);
    write(phys, base, 0x11 + 2 * pin, apic_id << 24);
    write(phys, base, 0x10 + 2 * pin, low);
    REGS.unlock();
    true
}

/// Hold `line` off, or let it interrupt again.
pub fn mask(phys: &Phys, line: usize, masked: bool) {
    let Some((base, pin, _)) = find(line) else { return };
    REGS.lock();
    let low = read(phys, base, 0x10 + 2 * pin);
    write(phys, base, 0x10 + 2 * pin, if masked { low | 1 << 16 } else { low & !(1 << 16) });
    REGS.unlock();
}

/// Whether `line` was routed as level-triggered.
pub fn is_level(line: usize) -> bool {
    LEVEL.get(line).is_some_and(|l| l.load(Ordering::Acquire))
}
