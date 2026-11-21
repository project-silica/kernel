//! What the loader does differently on x86_64: the serial port and the
//! emulator's exit device are I/O ports, the counter is the TSC, and one
//! page-table root holds the kernel half and the loader's identity alias.

use crate::efi;
use core::arch::asm;

/// ELF machine of the kernel image.
pub const MACHINE: u16 = kernel_core::elf::EM_X86_64;
/// The loader's message for a kernel image of another architecture.
pub const NOT_OURS: &str = "SILICA\\KERNEL.ELF is not an x86_64 ELF image";
/// One root for both halves: the low alias shares it with the kernel's.
pub const SPLIT_ROOTS: bool = false;
/// The physical-memory map covers everything below the top of memory: the
/// processor's memory-type ranges keep device memory uncached.
pub const MAP_ALL: bool = true;

const COM1: u16 = 0x3F8;

fn outb(port: u16, v: u8) {
    // SAFETY: writing a byte to the first serial port or the emulator's exit
    // device has no memory effect.
    unsafe { asm!("out dx, al", in("dx") port, in("al") v, options(nomem, nostack, preserves_flags)) }
}
fn inb(port: u16) -> u8 {
    let v: u8;
    // SAFETY: reading the serial line status register has no memory effect.
    unsafe { asm!("in al, dx", in("dx") port, out("al") v, options(nomem, nostack, preserves_flags)) }
    v
}

/// The free-running counter the kernel calibrates against.
pub fn counter() -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: RDTSC only reads the timestamp counter.
    unsafe { asm!("rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags)) }
    (hi as u64) << 32 | lo as u64
}

/// 64 bits from the processor's generator: RDSEED (its entropy source)
/// where there is one, else RDRAND (a generator reseeded from it). `None`
/// when it has neither, or when it keeps reporting failure (a few tries
/// cover a source momentarily drained by other processors).
pub fn cpu_random() -> Option<u64> {
    // SAFETY: CPUID only reads identification registers.
    #[allow(unused_unsafe)]
    let (l1, l7) = unsafe { (core::arch::x86_64::__cpuid(1), core::arch::x86_64::__cpuid_count(7, 0)) };
    let step = |seed: bool| -> Option<u64> {
        let (v, ok): (u64, u8);
        // SAFETY: RDSEED and RDRAND only produce a value and the carry
        // flag; their presence was checked with CPUID.
        unsafe {
            if seed {
                asm!("rdseed {}", "setc {}", out(reg) v, out(reg_byte) ok, options(nomem, nostack));
            } else {
                asm!("rdrand {}", "setc {}", out(reg) v, out(reg_byte) ok, options(nomem, nostack));
            }
        }
        (ok != 0).then_some(v)
    };
    if l7.ebx >> 18 & 1 != 0 {
        for _ in 0..64 {
            if let Some(v) = step(true) {
                return Some(v);
            }
            core::hint::spin_loop();
        }
    }
    if l1.ecx >> 30 & 1 != 0 {
        for _ in 0..16 {
            if let Some(v) = step(false) {
                return Some(v);
            }
        }
    }
    None
}

/// Nothing to set up: the serial port is always at COM1.
pub fn console(_st: *const efi::SystemTable) {}
/// Nothing to look up in ACPI.
pub fn console_from_acpi(_rsdp: u64) {}
/// Boot services are gone: nothing changes for an I/O-port console.
pub fn console_after_exit() {}

pub fn print(s: &str) {
    for b in s.bytes() {
        let mut spins = 0;
        while inb(COM1 + 5) & 0x20 == 0 && spins < 100_000 {
            spins += 1;
        }
        outb(COM1, b);
    }
}

/// Report `code` through the emulator's exit device (harmless where there
/// is none) and halt.
pub fn stop(code: u8, _harness: bool) -> ! {
    outb(0xF4, code);
    loop {
        // SAFETY: halting with interrupts as the firmware left them.
        unsafe { asm!("hlt", options(nomem, nostack)) }
    }
}

// Page-table entries.
const P: u64 = 1;
const W: u64 = 2;
const PS: u64 = 0x80;
const G: u64 = 0x100;
const NX: u64 = 1 << 63;
/// The address bits of an entry.
pub const ADDR: u64 = 0x000F_FFFF_FFFF_F000;

pub fn present(e: u64) -> bool {
    e & P != 0
}
/// An entry pointing to a next-level table.
pub fn table(addr: u64) -> u64 {
    addr | P | W
}
/// A 2 MiB mapping of normal memory.
pub fn block(phys: u64, write: bool, exec: bool) -> u64 {
    page(phys, write, exec) | PS
}
/// A 4 KiB mapping of normal memory.
pub fn page(phys: u64, write: bool, exec: bool) -> u64 {
    phys | P | G | if write { W } else { 0 } | if exec { 0 } else { NX }
}

/// Code written to memory is executed as written: nothing to do on x86.
pub fn sync_code(_start: u64, _len: u64) {}

/// Enable NX (the tables use it), switch to the new tables, which map this
/// code at the address it is running at, and jump to the kernel with the
/// BootInfo address in the first argument register.
///
/// # Safety
/// `root` must map this code at its current address and the kernel at
/// `entry`; `info` must point to a valid `BootInfo`.
pub unsafe fn handover(root: u64, _low_root: u64, entry: u64, info: u64) -> ! {
    // SAFETY: guaranteed by the caller. The kernel sets its own stack.
    unsafe {
        asm!(
            "cli",
            "mov ecx, 0xC0000080",
            "rdmsr",
            "or eax, 0x800",
            "wrmsr",
            "mov cr3, r8",
            "jmp r9",
            // Fixed registers: RDMSR and WRMSR use rax, rcx and rdx.
            in("r8") root,
            in("r9") entry,
            in("rdi") info,
            options(noreturn)
        )
    }
}
