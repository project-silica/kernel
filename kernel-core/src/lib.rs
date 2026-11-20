//! Arch-free kernel logic. Everything here builds for the host so it can be
//! tested without an emulator (`spec/ARCHITECTURE.md` section 12).
//!
//! `unsafe` is confined to [`mem`], which turns physical addresses into
//! object references (`spec/UNSAFE_MODULES.md`).
#![no_std]
#![deny(unsafe_code)]

/// Defines a constructor for `lock.rs`: a `const fn` here.
macro_rules! ctor {
    ($(#[$m:meta])* $v:vis fn $name:ident() -> $t:ty $body:block) => {
        $(#[$m])* $v const fn $name() -> $t $body
    };
}

pub mod boot;
pub mod cap;
pub mod elf;
pub mod hal;
pub mod ipc;
pub mod kernel;
pub mod lock;
pub mod log;
pub mod mem;
pub mod ramdisk;
pub mod root;
pub mod smp;
pub mod thread;
pub mod untyped;
pub mod vm;

pub use kernel::Kernel;

/// The atomics `lock.rs` is written against (the loom model substitutes its own).
mod sync {
    pub(crate) use core::hint::spin_loop as spin;
    pub(crate) use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
}

/// The line the kernel writes to the serial console once the root task runs.
pub const BOOT_BANNER: &str = "BOOT OK";

/// Exit status a test kernel reports through the emulator's exit device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ExitStatus {
    /// All checks passed.
    Success = 0x10,
    /// A check failed.
    Failure = 0x11,
}

impl ExitStatus {
    /// The value written to the exit device.
    pub const fn code(self) -> u8 {
        self as u8
    }
}
