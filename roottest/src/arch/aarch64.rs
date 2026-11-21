//! What the in-OS tests do differently on aarch64.

use core::arch::asm;
use libsys::Message;
use libsys::*;

/// Whether the machine has I/O ports (and the root task a port capability).
pub const HAS_PORTS: bool = false;
/// Hardware-resource capabilities the root task starts with.
pub const HW_CAPS: u64 = 1;

/// A system call with no arguments; returns the status register.
pub fn raw_syscall(n: u64) -> u64 {
    let r: u64;
    // SAFETY: a syscall with no arguments; it changes what a function call may change.
    unsafe { asm!("svc #0", inout("x8") n => r, clobber_abi("C"), options(nostack)) }
    r
}

/// Write to the test harness's console through semihosting (SYS_WRITE0),
/// which QEMU sends to the serial log. A system image has a console driver;
/// this test program talks to the harness the way it reports its result.
pub fn console_write(bytes: &[u8]) {
    let mut buf = [0u8; 256];
    for chunk in bytes.chunks(buf.len() - 1) {
        buf[..chunk.len()].copy_from_slice(chunk);
        buf[chunk.len()] = 0;
        // SAFETY: the call reads the NUL-terminated string in `buf`.
        unsafe { asm!("hlt #0xf000", in("w0") 0x04u32, in("x1") buf.as_ptr(), options(nostack, readonly)) }
    }
}

/// Tell the test harness the result through semihosting (QEMU started with
/// user-mode semihosting), with the encoding of x86's debug-exit device.
pub fn exit_harness(ok: bool) {
    let block = [0x2_0026u64, if ok { 0x10 << 1 | 1 } else { 0x11 << 1 | 1 }];
    // SAFETY: the semihosting exit call reads the two-word block and ends
    // the emulation; without a harness it is an undefined instruction.
    unsafe { asm!("hlt #0xf000", in("w0") 0x20u32, in("x1") block.as_ptr(), options(nostack)) }
}

/// Fault on purpose: 2 an undefined instruction, 3 6 7 8 privileged ones
/// (masking interrupts, reading and writing EL1 registers, a hypervisor call).
///
/// # Safety
/// Faults; only for a thread whose fault is the test.
pub unsafe fn deliberate(what: u64) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        match what {
            2 => asm!("udf #0"),
            3 => asm!("msr daifset, #2"),
            6 => asm!("mrs {}, sctlr_el1", out(reg) _),
            7 => asm!("mrs {}, ttbr1_el1", out(reg) _),
            8 => asm!("hvc #0"),
            _ => {}
        }
    }
}

/// Whether a fault message is what a privileged instruction gives here: an
/// instruction EL0 may not run.
pub fn privileged_fault(kind: u64, _m: &Message) -> bool {
    kind == Fault::Instruction as u64
}

/// A return instruction (`ret`).
pub const RET: &[u8] = &[0xC0, 0x03, 0x5F, 0xD6];

/// Make code written through the data cache visible to instruction fetch.
pub fn sync_code(at: usize, len: usize) {
    let ctr: u64;
    // SAFETY: reading the cache type register (allowed at EL0).
    unsafe { asm!("mrs {}, ctr_el0", out(reg) ctr, options(nomem, nostack, preserves_flags)) }
    let (dline, iline) = (4usize << (ctr >> 16 & 0xF), 4usize << (ctr & 0xF));
    // SAFETY: cache maintenance by address on memory this program maps.
    unsafe {
        let mut a = at & !(dline - 1);
        while a < at + len {
            asm!("dc cvau, {}", in(reg) a, options(nostack, preserves_flags));
            a += dline;
        }
        asm!("dsb ish", options(nostack, preserves_flags));
        let mut a = at & !(iline - 1);
        while a < at + len {
            asm!("ic ivau, {}", in(reg) a, options(nostack, preserves_flags));
            a += iline;
        }
        asm!("dsb ish", "isb", options(nostack, preserves_flags));
    }
}

/// The counter benchmarks time with, read after earlier instructions finish.
pub fn counter() -> u64 {
    let v: u64;
    // SAFETY: reading the virtual count after an instruction barrier.
    unsafe { asm!("isb", "mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack)) }
    v
}

/// Spin for `ticks` of the counter with `v` in vector registers, making no
/// system call, and return what the registers hold afterwards.
pub fn hold_in_vector_registers(v: u64, ticks: u64) -> [u64; 3] {
    let (a, b, c): (u64, u64, u64);
    // SAFETY: only registers declared here are touched.
    unsafe {
        asm!(
            "dup v0.2d, {v}",
            "mov v15.16b, v0.16b",
            "dup v7.2d, {v}",
            "mrs {end}, cntvct_el0", "add {end}, {end}, {ticks}",
            "2: yield", "mrs {now}, cntvct_el0", "cmp {now}, {end}", "b.lo 2b",
            "umov {a}, v0.d[0]",
            "umov {b}, v7.d[0]",
            "umov {c}, v15.d[1]",
            v = in(reg) v, ticks = in(reg) ticks, end = out(reg) _, now = out(reg) _, a = out(reg) a, b = out(reg) b, c = out(reg) c,
            out("v0") _, out("v7") _, out("v15") _,
            options(nostack),
        );
    }
    [a, b, c]
}

/// Fill vector registers with `v` and yield once; return this thread's
/// floating-point control register and thread pointer afterwards.
pub fn yield_holding_secret(v: u64) -> [u64; 2] {
    let (fpcr, tp): (u64, u64);
    // SAFETY: only registers declared here are touched; call 4 is Yield.
    unsafe {
        asm!(
            "dup v0.2d, {v}", "mov v1.16b, v0.16b", "mov v7.16b, v0.16b", "mov v15.16b, v0.16b",
            "mov x8, #4", "svc #0",
            "mrs x20, fpcr", "mrs x21, tpidr_el0",
            v = in(reg) v, out("x20") fpcr, out("x21") tp, clobber_abi("C"), options(nostack),
        );
    }
    [fpcr, tp]
}

/// What [`yield_holding_secret`] and [`fp_control_across_yield`] report for
/// a thread that never changed them.
pub const CONTROLS_DEFAULT: [u64; 2] = [0, 0];

/// Yield, then return what two vector registers hold (both should be clear
/// after a system call); there is no x87 stack here.
pub fn vectors_after_yield() -> (u64, u64, bool) {
    let (a, b): (u64, u64);
    // SAFETY: only registers declared here are touched; call 4 is Yield.
    unsafe {
        asm!(
            "mov x8, #4", "svc #0",
            "umov x20, v15.d[1]", "umov x21, v7.d[0]",
            out("x20") a, out("x21") b, clobber_abi("C"), options(nostack),
        );
    }
    (a, b, false)
}

/// Set a non-default rounding mode and thread pointer, yield, and return
/// what was set and what is there afterwards (restoring the defaults).
pub fn fp_control_across_yield() -> ([u64; 2], [u64; 2]) {
    let set = [3u64 << 22, 0x1234_5000]; // round toward zero
    let (fpcr, tp): (u64, u64);
    // SAFETY: sets FPCR and TPIDR_EL0 around a Yield; restores the defaults
    // (this program uses no thread pointer).
    unsafe {
        asm!(
            "msr fpcr, {f}", "msr tpidr_el0, {t}", "mov x8, #4", "svc #0",
            "mrs x20, fpcr", "mrs x21, tpidr_el0", "msr fpcr, xzr", "msr tpidr_el0, xzr",
            f = in(reg) set[0], t = in(reg) set[1], out("x20") fpcr, out("x21") tp, clobber_abi("C"), options(nostack),
        );
    }
    (set, [fpcr, tp])
}

/// One Yield with live vector state, timed in counter ticks.
pub fn timed_yield(v: u64) -> u64 {
    let t: u64;
    // SAFETY: loads vector registers, then times one Yield call.
    unsafe {
        asm!(
            "dup v15.2d, {v}",
            "isb", "mrs x21, cntvct_el0",
            "mov x8, #4", "svc #0",
            "isb", "mrs x22, cntvct_el0", "sub x22, x22, x21",
            v = in(reg) v, out("x21") _, out("x22") t, clobber_abi("C"),
            options(nostack),
        );
    }
    t
}

// Code for threads in other address spaces: position-independent, no stack,
// no data. x0 holds the thread's argument.
core::arch::global_asm!(
    ".section .text.stubs, \"ax\"",
    ".global stub_echo, stub_echo_end, stub_yield, stub_yield_end",
    // An echo server on the endpoint in x0: ReplyRecv (3) for ever. The
    // reply is the request, because the message registers are left as received.
    "stub_echo:",
    "mov x19, x0",
    "2: mov x8, #3",
    "mov x0, x19",
    "svc #0",
    "b 2b",
    "stub_echo_end:",
    // Yield (4) for ever with live vector state.
    "stub_yield:",
    "mov x19, x0",
    "3: dup v0.2d, x19",
    "mov v15.16b, v0.16b",
    "mov x8, #4",
    "svc #0",
    "b 3b",
    "stub_yield_end:",
);
