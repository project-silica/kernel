//! The syscall boundary: any sequence of syscalls, faults, timer ticks and
//! pending interrupts from four threads. See `tests/support/fuzz.rs` for the
//! grammar and the invariants checked after every input.
#![no_main]

#[path = "../../tests/support/mod.rs"]
mod support;

libfuzzer_sys::fuzz_target!(|data: &[u8]| support::fuzz::run(data));
