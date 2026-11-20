//! x86_64: the `syscall` instruction, the timestamp counter and process entry.

use core::arch::asm;

/// Make a kernel call. Returns the status and the six result registers.
///
/// This is the only way into the kernel. It is safe to execute with any
/// arguments: the kernel validates every one. The typed wrappers in the
/// crate root mark the operations that can pull memory or a stack out from
/// under running code as `unsafe`.
pub fn syscall(n: u64, a: [u64; 6]) -> (u64, [u64; 6]) {
    let (status, a0, a1, a2, a3, a4, a5): (u64, u64, u64, u64, u64, u64, u64);
    // SAFETY: the kernel preserves every register except rax and the six
    // argument registers, and `syscall` itself clobbers rcx and r11.
    unsafe {
        asm!(
            "syscall",
            inout("rax") n => status,
            inout("rdi") a[0] => a0,
            inout("rsi") a[1] => a1,
            inout("rdx") a[2] => a2,
            inout("r10") a[3] => a3,
            inout("r8") a[4] => a4,
            inout("r9") a[5] => a5,
            out("rcx") _,
            out("r11") _,
            options(nostack),
        );
    }
    (status, [a0, a1, a2, a3, a4, a5])
}

/// The processor's timestamp counter.
pub fn timestamp() -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: RDTSC only reads the counter.
    unsafe { asm!("rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags)) }
    (hi as u64) << 32 | lo as u64
}

// Process entry: the kernel (for the root task) or the process manager
// starts a program here with its start-up argument in the first argument
// register and a stack. `silica_main` is the program's entry function.
#[cfg(target_os = "silica")]
core::arch::global_asm!(
    ".section .text._start, \"ax\"",
    ".global _start",
    "_start:",
    "xor ebp, ebp",
    "and rsp, -16",
    "call silica_main",
    "ud2",
);
