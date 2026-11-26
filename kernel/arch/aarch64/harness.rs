//! What only a test image's kernel holds (the `harness` feature): the words
//! a test puts on the command line to hear the outcome (`exit=semihosting`)
//! and to make the kernel fail on purpose (`panic=test`, `panic=null`,
//! `panic=secondary`). A kernel built for a machine is built without this
//! file: no word on its command line reaches any of it.

use super::{Arm, Cmdline};
use core::arch::asm;
use core::sync::atomic::{AtomicBool, Ordering};
use kernel_core::{ExitStatus, Kernel};

/// `exit=semihosting`: a test harness (QEMU with semihosting) listens for
/// the status. On hardware without a debugger the call is an exception.
static LISTENS: AtomicBool = AtomicBool::new(false);
/// `panic=secondary`: processor 1 panics as it arrives (`smp::ap_main`).
pub static SECONDARY: AtomicBool = AtomicBool::new(false);

/// Take the test's words from the command line, before anything can fail.
pub fn words(cmdline: Cmdline<'_>) {
    LISTENS.store(cmdline.get("exit") == Some(b"semihosting"), Ordering::Relaxed);
    SECONDARY.store(cmdline.get("panic") == Some(b"secondary"), Ordering::Release);
}

/// Whether a harness hears the status: a panic then stops for it to read
/// instead of resetting the machine.
pub fn listens() -> bool {
    LISTENS.load(Ordering::Relaxed)
}

/// Tell the harness `status`, which ends the emulator (returns when none
/// listens).
pub fn exit(status: ExitStatus) {
    if listens() {
        // Semihosting SYS_EXIT_EXTENDED, ADP_Stopped_ApplicationExit, with the
        // status encoded as x86's debug-exit device reports it.
        let block = [0x2_0026u64, (status.code() as u64) << 1 | 1];
        // SAFETY: the call reads the two-word block and ends the emulation.
        unsafe { asm!("hlt #0xf000", in("w0") 0x20u32, in("x1") block.as_ptr(), options(nostack)) }
    }
}

/// `panic=test`: a panic on the boot processor, before the kernel exists.
pub fn deliberate_early(cmdline: Cmdline<'_>) {
    if cmdline.get("panic") == Some(b"test") {
        deliberate_panic(0);
    }
}

/// `panic=null`: reach a thread object through the null address, which must
/// fault (physical page 0 is never mapped for the kernel), so the test sees
/// a kernel exception with a backtrace through this function.
pub fn deliberate(cmdline: Cmdline<'_>, k: &'static Kernel<Arm>) {
    if cmdline.get("panic") == Some(b"null") {
        deliberate_null_access(k);
    }
}

#[inline(never)]
fn deliberate_null_access(k: &'static Kernel<Arm>) -> ! {
    let v = core::hint::black_box(k.regs(core::hint::black_box(0)))[0].get();
    panic!("physical page 0 is mapped: read {v:#x} through a null object address");
}

/// A panic a test asked for (`panic=test`, or `panic=secondary` on
/// processor 1).
#[inline(never)]
pub fn deliberate_panic(cpu: usize) -> ! {
    panic!("deliberate panic on processor {cpu}, asked for with panic=");
}
