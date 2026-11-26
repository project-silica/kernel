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

pub mod arm;
pub mod boot;
pub mod cap;
/// The image format readers (their own crate, shared with the loader and userspace).
pub use silica_formats::{acpi, elf, ramdisk};
#[cfg(test)]
mod exhaustive;
pub mod hal;
pub mod ipc;
pub mod kernel;
pub mod lock;
pub mod log;
pub mod mem;
mod order;
#[cfg(kani)]
mod proofs;
pub mod root;
pub mod sched;
pub mod smp;
pub mod thread;
pub mod untyped;
pub mod vm;
pub mod watch;
pub mod x86;

pub use kernel::Kernel;

/// A constant that an experiment may set when the kernel is built: the
/// value of the environment variable given by `option_env!("SILICA_TUNE_...")`
/// at the use, a decimal number, or `default` when it is not set (every
/// ordinary build). `tools/bench.sh --sweep` builds one kernel per value and
/// measures them side by side; the value that wins becomes the default, with
/// the measurement beside it. Nothing is read at run time, so a constant
/// declared this way costs what any constant costs.
pub const fn tuned(set: Option<&str>, default: u64) -> u64 {
    let Some(s) = set else { return default };
    let b = s.as_bytes();
    assert!(!b.is_empty(), "a SILICA_TUNE_ variable is set to nothing");
    let (mut v, mut i) = (0u64, 0);
    while i < b.len() {
        assert!(b[i].is_ascii_digit(), "a SILICA_TUNE_ variable is not a decimal number");
        v = v * 10 + (b[i] - b'0') as u64;
        i += 1;
    }
    v
}

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
