//! What the in-OS tests do differently on x86_64.

use core::arch::asm;
use libsys::Message;
use libsys::root::*;
use libsys::*;

/// Whether the machine has I/O ports (and the root task a port capability).
pub const HAS_PORTS: bool = true;
/// Hardware-resource capabilities the root task starts with (processor
/// time, `SLOT_SCHED`, among them).
pub const HW_CAPS: u64 = 6;

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
pub fn yield_holding_secret(v: u64) -> [u64; 2] {
    let (mut ctl, mut fs) = ([0u32; 2], 0u64);
    // SAFETY: only registers declared here are touched; `syscall` 4 is Yield.
    // Reading the FS base needs FSGSBASE, which the kernel turns on where
    // the processor has it.
    unsafe {
        asm!(
            "vmovq xmm0, {v}", "vpbroadcastq ymm0, xmm0", "vmovdqa ymm1, ymm0", "vmovdqa ymm7, ymm0", "vmovdqa ymm15, ymm0",
            "fld1",
            "mov eax, 4", "syscall",
            "fstp st(0)",
            "stmxcsr [r12]", "fnstcw [r12 + 4]",
            v = in(reg) v, in("r12") ctl.as_mut_ptr(), clobber_abi("sysv64"), options(nostack),
        );
        if fsgsbase() {
            asm!("rdfsbase {}", out(reg) fs, options(nostack, nomem));
        }
    }
    [ctl[0] as u64 | ((ctl[1] & 0xFFFF) as u64) << 32, fs]
}

/// What [`yield_holding_secret`] and [`fp_control_across_yield`] report for
/// a thread that never changed them: MXCSR and the x87 control word at their
/// defaults, no FS base.
pub const CONTROLS_DEFAULT: [u64; 2] = [0x1F80 | 0x037F << 32, 0];

/// Whether user mode may read and write the FS and GS bases directly.
fn fsgsbase() -> bool {
    let b: u32;
    // SAFETY: CPUID leaf 7 exists on every processor the kernel runs on;
    // rbx is saved around it because the compiler reserves it.
    unsafe {
        asm!("mov {t:r}, rbx", "cpuid", "xchg {t:r}, rbx", t = out(reg) b, inout("eax") 7 => _, inout("ecx") 0 => _, out("edx") _, options(nomem, nostack, preserves_flags));
    }
    b & 1 != 0
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

/// Set non-default floating-point control words (round toward zero) and,
/// where user mode can, an FS base; yield; return what was set and what is
/// there afterwards, as [`CONTROLS_DEFAULT`] lays them out (restoring the
/// defaults).
pub fn fp_control_across_yield() -> ([u64; 2], [u64; 2]) {
    let ctl_in = [0x1F80u32 | 0x6000, 0x037F | 0x0C00];
    let mut ctl_out = [0u32; 2];
    let defaults = [0x1F80u32, 0x037F];
    let fs_in = if fsgsbase() { 0x1234_5000u64 } else { 0 };
    let mut fs_out = 0u64;
    // SAFETY: loads and stores the control words around a Yield and
    // restores the defaults; the FS base is only touched with FSGSBASE on
    // and is put back to 0 (this program uses no thread pointer).
    unsafe {
        if fs_in != 0 {
            asm!("wrfsbase {}", in(reg) fs_in, options(nostack, nomem));
        }
        asm!(
            // Registers the system call preserves, so the pointers survive it.
            "ldmxcsr [r12]", "fldcw [r12 + 4]", "mov eax, 4", "syscall",
            "stmxcsr [r13]", "fnstcw [r13 + 4]", "ldmxcsr [r14]", "fldcw [r14 + 4]",
            in("r12") ctl_in.as_ptr(), in("r13") ctl_out.as_mut_ptr(), in("r14") defaults.as_ptr(),
            clobber_abi("sysv64"), options(nostack),
        );
        if fs_in != 0 {
            asm!("rdfsbase {}", "wrfsbase {z}", out(reg) fs_out, z = in(reg) 0u64, options(nostack, nomem));
        }
    }
    let pack = |c: [u32; 2]| c[0] as u64 | ((c[1] & 0xFFFF) as u64) << 32;
    ([pack(ctl_in), fs_in], [pack(ctl_out), fs_out])
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
    ".global stub_echo, stub_echo_end, stub_yield, stub_yield_end, stub_spin, stub_spin_end, stub_notify, stub_notify_end",
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
    // Compute for ever, never entering the kernel: a processor's load.
    "stub_spin:",
    "4: pause",
    "jmp 4b",
    "stub_spin_end:",
    // A driver's loop for ever (`serve_events`): SignalRecv, signalling
    // its client's notification (bits 40 and up of rdi) and receiving on
    // its endpoint (bits 0 to 19) while listening on its own notification
    // (bits 20 to 39).
    "stub_notify:",
    "mov r15, rdi",
    "5: mov rdi, r15",
    "shr rdi, 20",
    "and edi, 0xFFFFF",
    "shl rdi, 32",
    "mov rcx, r15",
    "and ecx, 0xFFFFF",
    "or rdi, rcx",
    "xor esi, esi",
    "xor edx, edx",
    "mov r10, r15",
    "shr r10, 40",
    "mov eax, 23",
    "syscall",
    "jmp 5b",
    "stub_notify_end:",
);
