//! `libsys`: the stable system library (`spec/ARCHITECTURE.md` section 4).
//! Empty at M0; C08 adds the syscall wrappers.
#![no_std]
#![forbid(unsafe_code)]

/// Platform API level (`spec/CI_RELEASE.md` section 7). Zero until the first
/// accepted proposal in `spec/rfc/`.
pub const API_LEVEL: u32 = 0;
