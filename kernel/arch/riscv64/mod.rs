//! riscv64 architecture glue. Not yet booting: the kernel builds for this target
//! so the arch-free code stays portable (component B05).

/// Kernel entry point named by the linker.
// SAFETY: `_start` is the only symbol with this name in the image; the
// linker uses it as the entry point and nothing else calls it.
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    let _banner = kernel_core::BOOT_BANNER;
    panic(&"unsupported")
}

/// Stop this processor for ever.
pub fn panic<T: ?Sized>(_info: &T) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
