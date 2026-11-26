//! What only a test image's kernel holds (the `harness` feature): the words
//! a test puts on the command line to hear the outcome (`exit=port`) and to
//! make the kernel fail on purpose (`panic=test`, `panic=null`,
//! `panic=park`). A kernel built for a machine is built without this file:
//! no word on its command line reaches any of it.

use super::{Cmdline, X86, serial};
use core::sync::atomic::{AtomicBool, Ordering};
use kernel_core::{ExitStatus, Kernel};

/// `exit=port`: a test harness listens on QEMU's debug-exit device (on a
/// machine without the device, port 0xF4 is nobody's to write).
static LISTENS: AtomicBool = AtomicBool::new(false);
/// `panic=park`: processor 1 parks holding its lock (`smp::ap_main`).
pub static PARK: AtomicBool = AtomicBool::new(false);

/// Take the test's words from the command line, before anything can fail.
pub fn words(cmdline: &[u8]) {
    LISTENS.store(Cmdline(cmdline).get("exit") == Some(b"port"), Ordering::Relaxed);
    PARK.store(Cmdline(cmdline).get("panic") == Some(b"park"), Ordering::Relaxed);
}

/// Whether a harness hears the status: a panic then stops for it to read
/// instead of resetting the machine.
pub fn listens() -> bool {
    LISTENS.load(Ordering::Relaxed)
}

/// Tell the harness `status` (0: success), which ends the emulator.
pub fn exit(status: u64) {
    if listens() {
        serial::outb(0xF4, if status == 0 { ExitStatus::Success } else { ExitStatus::Failure }.code());
    }
}

/// `panic=test`: a panic with a backtrace through this function.
/// `panic=null`: reach a thread object through the null address, which must
/// fault (physical page 0 is never mapped for the kernel), so the test sees
/// a kernel exception with a backtrace through that function.
pub fn deliberate(cmdline: &[u8], k: &'static Kernel<X86>) {
    match Cmdline(cmdline).get("panic") {
        Some(b"test") => deliberate_panic(),
        Some(b"null") => deliberate_null_access(k),
        _ => {}
    }
}

#[inline(never)]
fn deliberate_null_access(k: &'static Kernel<X86>) -> ! {
    let v = core::hint::black_box(k.regs(core::hint::black_box(0)))[0].get();
    panic!("physical page 0 is mapped: read {v:#x} through a null object address");
}

#[inline(never)]
fn deliberate_panic() {
    panic!("deliberate panic requested with panic=test");
}
