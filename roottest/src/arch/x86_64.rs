//! What the in-OS tests do differently on x86_64.

use core::arch::asm;
use libsys::Message;
use libsys::root::*;
use libsys::*;

/// Whether the machine has I/O ports (and the root task a port capability).
pub const HAS_PORTS: bool = true;
/// Hardware-resource capabilities the root task starts with.
pub const HW_CAPS: u64 = 2;

/// A system call with no arguments; returns the status register.
pub fn raw_syscall(n: u64) -> u64 {
    let r: u64;
    // SAFETY: a syscall with no arguments; it changes what a function call may change.
    unsafe { asm!("syscall", inout("rax") n => r, clobber_abi("sysv64"), options(nostack)) }
    r
}

/// Write to the console: the first serial port.
pub fn console_write(bytes: &[u8]) {
    let ports = Cap(SLOT_PORTS);
    for &b in bytes {
        while port_in(ports, 0x3FD, 1).unwrap_or(0x20) & 0x20 == 0 {}
        let _ = port_out(ports, 0x3F8, 1, b as u32);
    }
}

/// Tell the test harness the result through the emulator's exit device.
pub fn exit_harness(ok: bool) {
    let _ = port_out(Cap(SLOT_PORTS), 0xF4, 1, if ok { 0x10 } else { 0x11 });
}

/// Fault on purpose: 2 an undefined instruction, 3 6 7 8 privileged ones
/// (cli, hlt, mov from cr3, in).
///
/// # Safety
/// Faults; only for a thread whose fault is the test.
pub unsafe fn deliberate(what: u64) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        match what {
            2 => asm!("ud2"),
            3 => asm!("cli"),
            6 => asm!("hlt"),
            7 => asm!("mov rax, cr3", out("rax") _),
            8 => asm!("in al, dx", in("dx") 0x3F8u16, out("al") _),
            _ => {}
        }
    }
}

/// Whether a fault message is what a privileged instruction gives here: a
/// general-protection exception (vector 13).
pub fn privileged_fault(kind: u64, m: &Message) -> bool {
    kind == Fault::Exception as u64 && m.words[1] == 13
}

/// A return instruction.
pub const RET: &[u8] = &[0xC3];

/// Code written as data is executed as written: nothing to do.
pub fn sync_code(_at: usize, _len: usize) {}

/// The counter benchmarks time with, read after earlier instructions finish.
pub fn counter() -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: LFENCE and RDTSC read the timestamp counter after earlier
    // instructions have finished.
    unsafe { asm!("lfence", "rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack)) }
    (hi as u64) << 32 | lo as u64
}

/// Spin for `ticks` of the counter with `v` in vector registers, making no
/// system call, and return what the registers hold afterwards.
pub fn hold_in_vector_registers(v: u64, ticks: u64) -> [u64; 3] {
    let (a, b, c): (u64, u64, u64);
    // SAFETY: only registers declared here are touched.
    unsafe {
        asm!(
            "vmovq xmm0, {v}",
            "vpbroadcastq ymm15, xmm0",
            "vmovq xmm7, {v}",
            "rdtsc", "shl rdx, 32", "lea {end}, [rax + rdx]", "add {end}, {ticks}",
            "2: pause", "rdtsc", "shl rdx, 32", "add rax, rdx", "cmp rax, {end}", "jb 2b",
            "vmovq {a}, xmm0",
            "vmovq {b}, xmm7",
            "vextracti128 xmm1, ymm15, 1",
            "vpextrq {c}, xmm1, 1",
            "vzeroupper",
            v = in(reg) v, ticks = in(reg) ticks, end = out(reg) _, a = out(reg) a, b = out(reg) b, c = out(reg) c,
            out("rax") _, out("rdx") _, out("xmm0") _, out("xmm1") _, out("xmm7") _, out("xmm15") _,
            options(nostack),
        );
    }
    [a, b, c]
}

/// Fill the vector and x87 registers with `v` and yield once.
pub fn yield_holding_secret(v: u64) {
    // SAFETY: only registers declared here are touched; `syscall` 4 is Yield.
    unsafe {
        asm!(
            "vmovq xmm0, {v}", "vpbroadcastq ymm0, xmm0", "vmovdqa ymm1, ymm0", "vmovdqa ymm7, ymm0", "vmovdqa ymm15, ymm0",
            "fld1",
            "mov eax, 4", "syscall",
            "fstp st(0)",
            v = in(reg) v, clobber_abi("sysv64"), options(nostack),
        );
    }
}

/// Yield, then return what two vector registers hold and whether the x87
/// stack is in use: all should be clear after a system call.
pub fn vectors_after_yield() -> (u64, u64, bool) {
    let (a, b, x87): (u64, u64, u16);
    // SAFETY: only registers declared here are touched; `syscall` 4 is Yield.
    unsafe {
        asm!(
            "mov eax, 4", "syscall",
            "vextracti128 xmm2, ymm15, 1", "vpextrq r12, xmm2, 1", "vmovq r13, xmm7",
            "fnstsw ax",
            out("r12") a, out("r13") b, lateout("ax") x87, clobber_abi("sysv64"), options(nostack),
        );
    }
    (a, b, x87 & 0x3800 != 0)
}

/// Set a non-default floating-point control word, yield, and return what
/// was set and what is there afterwards (restoring the default).
pub fn fp_control_across_yield() -> (u64, u64) {
    let mxcsr_in = 0x1F80u32 | 0x6000; // round toward zero
    let mut mxcsr_out = 0u32;
    // SAFETY: loads and stores MXCSR around a Yield; restores the default.
    unsafe {
        asm!(
            // Registers the system call preserves, so the pointers survive it.
            "ldmxcsr [r12]", "mov eax, 4", "syscall", "stmxcsr [r13]", "ldmxcsr [r14]",
            in("r12") &mxcsr_in, in("r13") &mut mxcsr_out, in("r14") &0x1F80u32,
            clobber_abi("sysv64"), options(nostack),
        );
    }
    (mxcsr_in as u64, mxcsr_out as u64)
}

/// One Yield with live vector state, timed in counter ticks.
pub fn timed_yield(v: u64) -> u64 {
    let t: u64;
    // SAFETY: loads AVX registers, then times one Yield syscall.
    unsafe {
        asm!(
            "vmovq xmm0, {v}",
            "vpbroadcastq ymm15, xmm0",
            "lfence", "rdtsc", "shl rdx, 32", "lea r12, [rax + rdx]",
            "mov eax, 4",
            "syscall",
            "lfence", "rdtsc", "shl rdx, 32", "add rax, rdx", "sub rax, r12",
            v = in(reg) v, out("r12") _,
            lateout("rax") t, clobber_abi("sysv64"),
            options(nostack),
        );
    }
    t
}

// Code for threads in other address spaces: position-independent, no stack,
// no data. The first argument register holds the thread's argument.
core::arch::global_asm!(
    ".section .text.stubs, \"ax\"",
    ".global stub_echo, stub_echo_end, stub_yield, stub_yield_end",
    // An echo server on the endpoint in rdi: ReplyRecv for ever. The reply
    // is the request, because the message registers are left as received.
    "stub_echo:",
    "mov r15, rdi",
    "2: mov eax, 3",
    "mov rdi, r15",
    "syscall",
    "jmp 2b",
    "stub_echo_end:",
    // Yield for ever with live AVX state.
    "stub_yield:",
    "mov r15, rdi",
    "3: vmovq xmm0, r15",
    "vpbroadcastq ymm15, xmm0",
    "mov eax, 4",
    "syscall",
    "jmp 3b",
    "stub_yield_end:",
);
