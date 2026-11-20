//! Arch-free kernel logic. Everything here builds for the host so it can be
//! tested without an emulator (`spec/ARCHITECTURE.md` section 12).
#![no_std]
#![forbid(unsafe_code)]

/// The line every boot path writes to the serial console once it is up.
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
