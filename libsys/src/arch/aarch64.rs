//! aarch64: the `svc` instruction, the generic timer's counter and process entry.

use core::arch::asm;

/// Make a kernel call. Returns the status and the six result registers.
///
/// This is the only way into the kernel. It is safe to execute with any
/// arguments: the kernel validates every one. The typed wrappers in the
/// crate root mark the operations that can pull memory or a stack out from
/// under running code as `unsafe`.
///
/// Register contract: the call number goes in x8 and the status comes back
/// there; arguments and results are x0 to x5. The kernel preserves the
/// registers a function call preserves (x19 to x29, sp, and FPCR and FPSR)
/// and, like a function call, not the vector registers.
pub fn syscall(n: u64, a: [u64; 6]) -> (u64, [u64; 6]) {
    let (status, a0, a1, a2, a3, a4, a5): (u64, u64, u64, u64, u64, u64, u64);
    // SAFETY: every register the kernel may change is declared: the result
    // registers explicitly, the rest through the calling convention's
    // caller-saved set.
    unsafe {
        asm!(
            "svc #0",
            inout("x8") n => status,
            inout("x0") a[0] => a0,
            inout("x1") a[1] => a1,
            inout("x2") a[2] => a2,
            inout("x3") a[3] => a3,
            inout("x4") a[4] => a4,
            inout("x5") a[5] => a5,
            clobber_abi("C"),
            options(nostack),
        );
    }
    (status, [a0, a1, a2, a3, a4, a5])
}

/// The generic timer's virtual count (readable from user mode).
pub fn timestamp() -> u64 {
    let v: u64;
    // SAFETY: reading the counter; ISB keeps it in program order.
    unsafe { asm!("isb", "mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack, preserves_flags)) }
    v
}

// Process entry: the kernel (for the root task) or the process manager
// starts a program here with its start-up argument in x0 and a stack.
// `silica_main` is the program's entry function.
#[cfg(target_os = "silica")]
core::arch::global_asm!(
    ".section .text._start, \"ax\"",
    ".global _start",
    "_start:",
    "mov x29, #0",
    "mov x30, #0",
    "mov x9, sp",
    "and x9, x9, #-16",
    "mov sp, x9",
    "bl silica_main",
    "udf #0",
);
