//! Silica kernel entry. At M0 this is a stub that proves the build for every
//! target; the boot path (B01 to B03) replaces it at M1.
#![no_std]
#![no_main]

#[cfg_attr(target_arch = "x86_64", path = "../arch/x86_64/mod.rs")]
#[cfg_attr(target_arch = "aarch64", path = "../arch/aarch64/mod.rs")]
#[cfg_attr(target_arch = "riscv64", path = "../arch/riscv64/mod.rs")]
mod arch;
mod hal;

use hal::Arch;

/// Kernel entry point named by the linker.
// SAFETY: `_start` is the only symbol with this name in the image; the
// linker uses it as the entry point and nothing else calls it.
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    let _banner = kernel_core::BOOT_BANNER;
    arch::Current::halt()
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
    arch::Current::halt()
}
