//! Silica kernel: the architecture layer around `kernel-core`.
#![no_std]
#![no_main]
// `syscall_rust` never returns, so it keeps no registers for its caller.
#![cfg_attr(target_arch = "x86_64", feature(rust_preserve_none_cc))]

#[cfg_attr(target_arch = "x86_64", path = "../arch/x86_64/mod.rs")]
#[cfg_attr(target_arch = "aarch64", path = "../arch/aarch64/mod.rs")]
#[cfg_attr(target_arch = "riscv64", path = "../arch/riscv64/mod.rs")]
mod arch;

#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    arch::panic(info)
}
