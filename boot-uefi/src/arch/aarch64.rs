//! What the loader does differently on aarch64.
//!
//! - Console: the firmware's text output until the UART the ACPI SPCR table
//!   names is known, then that UART (a PL011 or SBSA generic UART), which
//!   keeps working after boot services are gone.
//! - Counter: the generic timer's virtual count (`CNTVCT_EL0`).
//! - Tables: two roots, TTBR1 for the kernel half (all RAM at
//!   `PHYS_MAP_BASE` and the kernel image) and TTBR0 for the loader's
//!   identity alias of RAM. Only memory the firmware reports as RAM is
//!   mapped, as Normal write-back memory: device registers mapped Normal
//!   could be read speculatively. The kernel maps devices itself.
//! - Handover: from EL2 the loader first drops to EL1. With the MMU briefly
//!   off it installs its MAIR, TCR and both roots, invalidates the TLB and
//!   turns the MMU on, so no translation ever mixes the firmware's memory
//!   attributes with Silica's.
//! - Test status: a test harness (QEMU with semihosting) is told the exit
//!   code through the semihosting exit call, with the same value encoding
//!   as x86's debug-exit device; only when the command line asks for it,
//!   because on hardware without a debugger the call is an exception.

use crate::efi;
use core::arch::asm;
use core::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

/// ELF machine of the kernel image.
pub const MACHINE: u16 = kernel_core::elf::EM_AARCH64;
/// The loader's message for a kernel image of another architecture.
pub const NOT_OURS: &str = "SILICA\\KERNEL.ELF is not an aarch64 ELF image";
/// The kernel half and the low alias have roots of their own.
pub const SPLIT_ROOTS: bool = true;
/// Only RAM is mapped (see the module documentation).
pub const MAP_ALL: bool = false;

/// The memory-attribute layout the tables are written for: index 0 Normal
/// write-back, 1 Device-nGnRE, 2 Normal non-cacheable, 3 Device-nGnRnE.
pub const MAIR: u64 = 0x00_44_04_FF;

static CON_OUT: AtomicPtr<efi::TextOutput> = AtomicPtr::new(core::ptr::null_mut());
static UART: AtomicU64 = AtomicU64::new(0);

/// The free-running counter the kernel calibrates against.
pub fn counter() -> u64 {
    let v: u64;
    // SAFETY: reading the virtual count has no side effect; ISB keeps it in order.
    unsafe { asm!("isb", "mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack, preserves_flags)) }
    v
}

/// Print through the firmware's console until a UART is known.
pub fn console(st: *const efi::SystemTable) {
    // SAFETY: the firmware's system table is valid while boot services run.
    CON_OUT.store(unsafe { (*st).con_out }, Ordering::Relaxed);
}

/// Boot services are gone: only the UART can be used from now on.
pub fn console_after_exit() {
    CON_OUT.store(core::ptr::null_mut(), Ordering::Relaxed);
}

/// Use the serial port the ACPI SPCR table names, if it is one this loader
/// can drive (PL011 or SBSA generic UART, in memory space).
pub fn console_from_acpi(rsdp: u64) {
    if let Some(base) = spcr_uart(rsdp) {
        UART.store(base, Ordering::Relaxed);
    }
}

fn read<T: Copy>(p: u64) -> T {
    // SAFETY: only called on addresses inside ACPI tables the firmware
    // published, which it maps for as long as the loader runs.
    unsafe { (p as *const T).read_unaligned() }
}

fn spcr_uart(rsdp: u64) -> Option<u64> {
    if rsdp == 0 || read::<[u8; 8]>(rsdp) != *b"RSD PTR " || read::<u8>(rsdp + 15) < 2 {
        return None;
    }
    let xsdt: u64 = read(rsdp + 24);
    if xsdt == 0 || read::<[u8; 4]>(xsdt) != *b"XSDT" {
        return None;
    }
    let len = read::<u32>(xsdt + 4) as u64;
    let mut at = xsdt + 36;
    while at + 8 <= xsdt + len {
        let t: u64 = read(at);
        at += 8;
        if t == 0 || read::<[u8; 4]>(t) != *b"SPCR" || read::<u32>(t + 4) < 80 {
            continue;
        }
        // Interface type 3: PL011; 0x0E: SBSA generic UART (same registers
        // as far as this loader uses them). Base address: a generic address
        // structure in system memory (space 0).
        let kind: u8 = read(t + 36);
        let (space, base): (u8, u64) = (read(t + 40), read(t + 44));
        if matches!(kind, 0x03 | 0x0E) && space == 0 && base != 0 {
            return Some(base);
        }
    }
    None
}

pub fn print(s: &str) {
    let uart = UART.load(Ordering::Relaxed);
    if uart != 0 {
        for b in s.bytes() {
            let mut spins = 0;
            // SAFETY: `uart` is the base of a PL011-compatible UART the
            // firmware maps as device memory: flag register at 0x18 (bit 5
            // TXFF: transmit FIFO full), data register at 0.
            unsafe {
                while core::ptr::read_volatile((uart + 0x18) as *const u32) & 0x20 != 0 && spins < 100_000 {
                    spins += 1;
                }
                core::ptr::write_volatile(uart as *mut u32, b as u32);
            }
        }
        return;
    }
    let out = CON_OUT.load(Ordering::Relaxed);
    if out.is_null() {
        return;
    }
    let mut wide = [0u16; 128];
    for chunk in s.as_bytes().chunks(wide.len() - 1) {
        for (i, b) in chunk.iter().enumerate() {
            wide[i] = *b as u16;
        }
        wide[chunk.len()] = 0;
        // SAFETY: the protocol is valid while boot services run; the
        // string is NUL-terminated.
        unsafe { ((*out).output_string)(out, wide.as_ptr()) };
    }
}

/// Report `code` and stop: through semihosting when `harness` says a test
/// harness is listening, then wait for ever.
pub fn stop(code: u8, harness: bool) -> ! {
    if harness {
        // SYS_EXIT_EXTENDED with ADP_Stopped_ApplicationExit; the status is
        // encoded as x86's debug-exit device reports it, (code << 1) | 1.
        let block = [0x2_0026u64, (code as u64) << 1 | 1];
        // SAFETY: the semihosting call reads the two-word block and ends
        // the emulation.
        unsafe { asm!("hlt #0xf000", in("w0") 0x20u32, in("x1") block.as_ptr(), options(nostack)) }
    }
    loop {
        // SAFETY: waiting for an event changes nothing.
        unsafe { asm!("wfe", options(nomem, nostack)) }
    }
}

// Descriptors, 4 KiB granule.
const VALID: u64 = 1;
const TABLE: u64 = 0b11;
const BLOCK: u64 = 0b01;
const PAGE: u64 = 0b11;
const NORMAL: u64 = 0 << 2;
const READ_ONLY: u64 = 1 << 7;
const INNER_SHAREABLE: u64 = 3 << 8;
const AF: u64 = 1 << 10;
const PXN: u64 = 1 << 53;
const UXN: u64 = 1 << 54;
/// The address bits of an entry.
pub const ADDR: u64 = 0x0000_FFFF_FFFF_F000;

pub fn present(e: u64) -> bool {
    e & VALID != 0
}
/// An entry pointing to a next-level table.
pub fn table(addr: u64) -> u64 {
    addr | TABLE
}
fn attrs(write: bool, exec: bool) -> u64 {
    NORMAL | INNER_SHAREABLE | AF | UXN | if write { 0 } else { READ_ONLY } | if exec { 0 } else { PXN }
}
/// A 2 MiB mapping of normal memory, for the kernel only.
pub fn block(phys: u64, write: bool, exec: bool) -> u64 {
    phys | BLOCK | attrs(write, exec)
}
/// A 4 KiB mapping of normal memory, for the kernel only.
pub fn page(phys: u64, write: bool, exec: bool) -> u64 {
    phys | PAGE | attrs(write, exec)
}

/// Make code written through the data cache visible to instruction fetch:
/// clean each line to the point of unification, then invalidate the
/// instruction cache.
pub fn sync_code(start: u64, len: u64) {
    let ctr: u64;
    // SAFETY: reading the cache type register has no side effect.
    unsafe { asm!("mrs {}, ctr_el0", out(reg) ctr, options(nomem, nostack, preserves_flags)) }
    let line = 4u64 << (ctr >> 16 & 0xF);
    let mut a = start & !(line - 1);
    while a < start + len {
        // SAFETY: cleaning a cache line of memory this loader owns.
        unsafe { asm!("dc cvau, {}", in(reg) a, options(nostack, preserves_flags)) }
        a += line;
    }
    // SAFETY: barriers and an instruction-cache invalidation only.
    unsafe { asm!("dsb ish", "ic ialluis", "dsb ish", "isb", options(nostack, preserves_flags)) }
}

/// The translation control value for 48-bit addresses in both halves, 4 KiB
/// granules, write-back walks, and the processor's physical address size.
fn tcr() -> u64 {
    let mmfr0: u64;
    // SAFETY: reading an ID register has no side effect.
    unsafe { asm!("mrs {}, id_aa64mmfr0_el1", out(reg) mmfr0, options(nomem, nostack, preserves_flags)) }
    let ips = (mmfr0 & 0xF).min(6);
    let asid16 = if mmfr0 >> 4 & 0xF == 2 { 1u64 << 36 } else { 0 };
    let half = |shift: u32| (16u64 | 1 << 8 | 1 << 10 | 3 << 12) << shift;
    half(0) | half(16) | 2 << 30 | ips << 32 | asid16
}

/// Drop to EL1 if the firmware left us at EL2, install Silica's MAIR, TCR
/// and tables with the MMU off, turn it on and jump to the kernel with the
/// BootInfo address in x0.
///
/// # Safety
/// `low_root` must map this code at its current address and `root` the
/// kernel at `entry`; `info` must point to a valid `BootInfo`.
pub unsafe fn handover(root: u64, low_root: u64, entry: u64, info: u64) -> ! {
    // SAFETY: guaranteed by the caller. No stack is used after the switch
    // to EL1 (SP_EL1 is not set up); the kernel sets its own.
    unsafe {
        asm!(
            "msr daifset, #0xf",
            "mrs x9, CurrentEL",
            "cmp x9, #8",
            "b.ne 2f",
            // EL2. First its MMU off: the firmware may run with VHE
            // (HCR_EL2.E2H), and changing HCR_EL2 under a live EL2
            // translation would change what its tables mean. This code is
            // identity-mapped, so it keeps running.
            "mrs x10, sctlr_el2",
            "bic x10, x10, #1",
            "msr sctlr_el2, x10",
            "isb",
            // EL1 is AArch64 (HCR_EL2.RW; E2H and TGE clear), may use the
            // counter and the physical timer, is not trapped for floating
            // point or SIMD, starts with its MMU off, and is entered with
            // interrupts masked.
            "mov x10, #(1 << 31)",
            "msr hcr_el2, x10",
            "mrs x10, cnthctl_el2",
            "orr x10, x10, #3",
            "msr cnthctl_el2, x10",
            "msr cntvoff_el2, xzr",
            "mov x10, #0x33ff",
            "msr cptr_el2, x10",
            "msr hstr_el2, xzr",
            "ldr x10, =0x30d00800",
            "msr sctlr_el1, x10",
            "mov x10, #0x3c5",
            "msr spsr_el2, x10",
            "adr x10, 2f",
            "msr elr_el2, x10",
            "isb",
            "eret",
            "2:",
            // EL1: floating point and SIMD not trapped (CPACR_EL1.FPEN).
            "mov x10, #(3 << 20)",
            "msr cpacr_el1, x10",
            "mrs x9, sctlr_el1",
            "bic x10, x9, #1",
            "msr sctlr_el1, x10",
            "isb",
            "msr mair_el1, x4",
            "msr tcr_el1, x5",
            "msr ttbr0_el1, x2",
            "msr ttbr1_el1, x3",
            "isb",
            "tlbi vmalle1",
            "dsb nsh",
            "isb",
            // MMU, data and instruction caches on.
            "orr x9, x9, #1",
            "orr x9, x9, #(1 << 2)",
            "orr x9, x9, #(1 << 12)",
            "msr sctlr_el1, x9",
            "isb",
            "br x1",
            in("x0") info,
            in("x1") entry,
            in("x2") low_root,
            in("x3") root,
            in("x4") MAIR,
            in("x5") tcr(),
            options(noreturn)
        )
    }
}
