//! M0 stub tests: prove the host harness runs kernel-core tests.
use kernel_core::{BOOT_BANNER, ExitStatus};

#[test]
fn banner_is_boot_ok() {
    assert_eq!(BOOT_BANNER, "BOOT OK");
}

#[test]
fn exit_codes_are_distinct() {
    assert_ne!(ExitStatus::Success.code(), ExitStatus::Failure.code());
}
