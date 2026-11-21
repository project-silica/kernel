//! Per-processor state, exception entry and exit, and the vector registers.
//!
//! Each processor's `PerCpu` is at TPIDR_EL1. An exception from EL0 saves
//! the thread's registers into its frame (the `regs` of its Thread object,
//! pointed to by `PerCpu::regs`) on that processor's kernel stack, calls
//! into Rust, and leaves through `exit_to_user`, which lets the scheduler
//! pick the thread, loads its frame and returns with ERET. Frame layout
//! (index * 8): x0..x30 at 0..30, SP_EL0 31, ELR 32, SPSR 33, TPIDR_EL0 34,
//! 35 not 0 when the thread entered by SVC, 36 FPCR (low half) and FPSR.

use core::arch::global_asm;
use core::cell::Cell;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use kernel_core::lock::Presence;
use kernel_core::smp::MAX_CPUS;

pub const REG_SP: usize = 31;
pub const REG_PC: usize = 32;
pub const REG_SPSR: usize = 33;
pub const REG_TLS: usize = 34;
pub const REG_BY_SVC: usize = 35;
pub const REG_FPCTL: usize = 36;

/// One processor's data. The first fields are read by the entry code at
/// fixed offsets.
#[repr(C)]
pub struct PerCpu {
    /// 0: the frame of the thread this processor runs.
    pub regs: Cell<u64>,
    /// 8: the top of this processor's kernel stack.
    pub kstack_top: AtomicU64,
    /// 16: this processor's index.
    pub index: usize,
    /// This processor's MPIDR affinity.
    pub mpidr: AtomicU64,
    pub mode: Presence,
    pub online: AtomicBool,
    /// When the timer is set to fire, in counter ticks (u64::MAX: never).
    pub deadline: Cell<u64>,
    /// ASID allocation on this processor: the current generation, and the
    /// next identifier to hand out in it (see `Arm::ttbr0_for`).
    pub asid_gen: Cell<u64>,
    pub asid_next: Cell<u64>,
}

impl PerCpu {
    /// Processor `index`'s data, before it starts.
    pub const fn new(index: usize, kstack_top: u64, mpidr: u64) -> PerCpu {
        PerCpu {
            regs: Cell::new(0),
            kstack_top: AtomicU64::new(kstack_top),
            index,
            mpidr: AtomicU64::new(mpidr),
            mode: Presence::new(),
            online: AtomicBool::new(false),
            deadline: Cell::new(u64::MAX),
            asid_gen: Cell::new(1),
            asid_next: Cell::new(1),
        }
    }
}

// SAFETY: a processor's `Cell`s are used only by that processor, or by
// another holding its lock (kernel-core's protocol); the rest is atomic or
// written once before the processor starts.
unsafe impl Sync for PerCpu {}

static CPUS: [AtomicPtr<PerCpu>; MAX_CPUS] = [const { AtomicPtr::new(core::ptr::null_mut()) }; MAX_CPUS];

/// This processor's data.
pub fn percpu() -> &'static PerCpu {
    let p: u64;
    // SAFETY: TPIDR_EL1 holds this processor's PerCpu from `install` on;
    // nothing runs on a processor before that.
    unsafe {
        core::arch::asm!("mrs {}, tpidr_el1", out(reg) p, options(nomem, nostack, preserves_flags, pure));
        &*(p as *const PerCpu)
    }
}

/// Processor `i`'s data, once it has registered.
pub fn percpu_of(i: usize) -> Option<&'static PerCpu> {
    let p = CPUS.get(i)?.load(Ordering::Acquire);
    // SAFETY: registered PerCpu values live for ever.
    (!p.is_null()).then(|| unsafe { &*p })
}

/// Make `p` this processor's data and point VBAR_EL1 at the vectors.
///
/// # Safety
/// Once per processor, early, with interrupts masked; `p` lives for ever.
pub unsafe fn install(p: &'static PerCpu) {
    CPUS[p.index].store(core::ptr::from_ref(p).cast_mut(), Ordering::Release);
    // SAFETY: guaranteed by the caller. CNTKCTL_EL1.EL0VCTEN lets user code
    // read the virtual counter (the monotonic clock without a system call);
    // SCTLR_EL1.SPAN = 0 sets PSTATE.PAN on every exception, so the kernel
    // cannot touch user memory even by accident; FP and SIMD are not trapped;
    // user code may maintain the caches for code it writes.
    unsafe {
        core::arch::asm!(
            "msr tpidr_el1, {p}",
            "adrp {t}, vectors",
            "add {t}, {t}, :lo12:vectors",
            "msr vbar_el1, {t}",
            "mov {t}, #2",
            "msr cntkctl_el1, {t}",
            "mov {t}, #(3 << 20)",
            "msr cpacr_el1, {t}",
            "mrs {t}, sctlr_el1",
            "bic {t}, {t}, #(1 << 23)",
            // UCI and UCT: user code may clean and invalidate cache lines by
            // address and read the cache type (to run code it wrote).
            "orr {t}, {t}, #(1 << 26)",
            "orr {t}, {t}, #(1 << 15)",
            "msr sctlr_el1, {t}",
            ".arch_extension pan",
            "msr pan, #1",
            "isb",
            p = in(reg) core::ptr::from_ref(p),
            t = out(reg) _,
            options(nostack),
        )
    }
}

unsafe extern "C" {
    /// Leave the kernel for the thread the scheduler picks. The caller has
    /// entered the kernel (`Kernel::enter`).
    pub fn exit_to_user() -> !;
}

// The vector table. Entries from EL0 (AArch64) save the thread's frame and
// call `user_trap` with the kind; the kernel's own synchronous exceptions
// are faults; the only interrupts taken in the kernel are in the idle wait.
// AArch32 and SError entries are fatal.
global_asm!(
    r#"
.section .text.fast, "ax"
.balign 2048
.global vectors
vectors:
.irp n, 0,1,2,3
.balign 128
    mov x3, #\n
    b kernel_bad
.endr
// Current EL with SP_EL1.
.balign 128
    b kernel_sync
.balign 128
    b kernel_irq
.balign 128
    mov x3, #6
    b kernel_bad
.balign 128
    mov x3, #7
    b kernel_bad
// Lower EL, AArch64.
.balign 128
    sub sp, sp, #16
    stp x0, x1, [sp]
    mov x1, #0
    b user_entry
.balign 128
    sub sp, sp, #16
    stp x0, x1, [sp]
    mov x1, #1
    b user_entry
.balign 128
    sub sp, sp, #16
    stp x0, x1, [sp]
    mov x1, #1
    b user_entry
.balign 128
    mov x3, #11
    b kernel_bad
.irp n, 12,13,14,15
.balign 128
    mov x3, #\n
    b kernel_bad
.endr

// Save the thread's frame. On entry x1 is the kind (0 synchronous, 1
// interrupt); the thread's x0 and x1 are on the stack.
user_entry:
    mrs x0, tpidr_el1
    ldr x0, [x0]
    stp x2, x3, [x0, #16]
    stp x4, x5, [x0, #32]
    stp x6, x7, [x0, #48]
    stp x8, x9, [x0, #64]
    stp x10, x11, [x0, #80]
    stp x12, x13, [x0, #96]
    stp x14, x15, [x0, #112]
    stp x16, x17, [x0, #128]
    stp x18, x19, [x0, #144]
    stp x20, x21, [x0, #160]
    stp x22, x23, [x0, #176]
    stp x24, x25, [x0, #192]
    stp x26, x27, [x0, #208]
    stp x28, x29, [x0, #224]
    str x30, [x0, #240]
    ldp x2, x3, [sp], #16
    stp x2, x3, [x0, #0]
    mrs x2, sp_el0
    mrs x3, elr_el1
    stp x2, x3, [x0, #248]
    mrs x2, spsr_el1
    mrs x3, tpidr_el0
    stp x2, x3, [x0, #264]
    mov x29, #0
    mov x30, #0
    mov x0, x1
    bl {user_trap}
    b exit_to_user

.global exit_to_user
exit_to_user:
    mrs x0, tpidr_el1
    ldr x1, [x0, #8]
    mov sp, x1
    bl {schedule}
    mrs x0, tpidr_el1
    ldr x0, [x0]
    ldp x2, x3, [x0, #248]
    msr sp_el0, x2
    msr elr_el1, x3
    // Only the condition flags come from the frame: the thread always
    // returns to EL0, with interrupts unmasked, whatever its saved SPSR says.
    ldp x2, x3, [x0, #264]
    and x2, x2, #0xF0000000
    msr spsr_el1, x2
    msr tpidr_el0, x3
    ldp x2, x3, [x0, #16]
    ldp x4, x5, [x0, #32]
    ldp x6, x7, [x0, #48]
    ldp x8, x9, [x0, #64]
    ldp x10, x11, [x0, #80]
    ldp x12, x13, [x0, #96]
    ldp x14, x15, [x0, #112]
    ldp x16, x17, [x0, #128]
    ldp x18, x19, [x0, #144]
    ldp x20, x21, [x0, #160]
    ldp x22, x23, [x0, #176]
    ldp x24, x25, [x0, #192]
    ldp x26, x27, [x0, #208]
    ldp x28, x29, [x0, #224]
    ldr x30, [x0, #240]
    ldp x0, x1, [x0, #0]
    // Page-table writes are visible to the walker before user code runs.
    dsb ish
    eret
    // Never reached: no speculation past ERET.
    dsb nsh
    isb

// A synchronous exception in the kernel: a kernel bug.
kernel_sync:
    mrs x0, esr_el1
    mrs x1, elr_el1
    mrs x2, far_el1
    mov x3, #4
    mov x4, x29
    b {fault}
kernel_bad:
    mrs x0, esr_el1
    mrs x1, elr_el1
    mrs x2, far_el1
    mov x4, x29
    b {fault}

// An interrupt in the kernel (the idle wait): save what a call clobbers.
kernel_irq:
    sub sp, sp, #192
    stp x0, x1, [sp, #0]
    stp x2, x3, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp x8, x9, [sp, #64]
    stp x10, x11, [sp, #80]
    stp x12, x13, [sp, #96]
    stp x14, x15, [sp, #112]
    stp x16, x17, [sp, #128]
    stp x18, x29, [sp, #144]
    mrs x0, elr_el1
    mrs x1, spsr_el1
    stp x30, x0, [sp, #160]
    str x1, [sp, #176]
    bl {irq_idle}
    ldr x1, [sp, #176]
    ldp x30, x0, [sp, #160]
    msr elr_el1, x0
    msr spsr_el1, x1
    ldp x0, x1, [sp, #0]
    ldp x2, x3, [sp, #16]
    ldp x4, x5, [sp, #32]
    ldp x6, x7, [sp, #48]
    ldp x8, x9, [sp, #64]
    ldp x10, x11, [sp, #80]
    ldp x12, x13, [sp, #96]
    ldp x14, x15, [sp, #112]
    ldp x16, x17, [sp, #128]
    ldp x18, x29, [sp, #144]
    add sp, sp, #192
    eret
    dsb nsh
    isb
"#,
    user_trap = sym super::user_trap,
    schedule = sym super::schedule_rust,
    fault = sym super::kernel_fault,
    irq_idle = sym super::irq_idle,
);

// The vector registers. The kernel never uses them (it is built without
// FP and SIMD), so they hold the running thread's values until a switch.
global_asm!(
    r#"
.section .text, "ax"
.arch_extension fp
.arch_extension simd
// x0: 512-byte save area.
.global silica_simd_save
silica_simd_save:
    stp q0, q1, [x0, #0]
    stp q2, q3, [x0, #32]
    stp q4, q5, [x0, #64]
    stp q6, q7, [x0, #96]
    stp q8, q9, [x0, #128]
    stp q10, q11, [x0, #160]
    stp q12, q13, [x0, #192]
    stp q14, q15, [x0, #224]
    stp q16, q17, [x0, #256]
    stp q18, q19, [x0, #288]
    stp q20, q21, [x0, #320]
    stp q22, q23, [x0, #352]
    stp q24, q25, [x0, #384]
    stp q26, q27, [x0, #416]
    stp q28, q29, [x0, #448]
    stp q30, q31, [x0, #480]
    ret
.global silica_simd_load
silica_simd_load:
    ldp q0, q1, [x0, #0]
    ldp q2, q3, [x0, #32]
    ldp q4, q5, [x0, #64]
    ldp q6, q7, [x0, #96]
    ldp q8, q9, [x0, #128]
    ldp q10, q11, [x0, #160]
    ldp q12, q13, [x0, #192]
    ldp q14, q15, [x0, #224]
    ldp q16, q17, [x0, #256]
    ldp q18, q19, [x0, #288]
    ldp q20, q21, [x0, #320]
    ldp q22, q23, [x0, #352]
    ldp q24, q25, [x0, #384]
    ldp q26, q27, [x0, #416]
    ldp q28, q29, [x0, #448]
    ldp q30, q31, [x0, #480]
    ret
.global silica_simd_clear
silica_simd_clear:
.irp n, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    movi v\n\().2d, #0
.endr
    ret
// x0 out: FPCR in the low half, FPSR in the high half.
.global silica_fp_control
silica_fp_control:
    mrs x0, fpcr
    mrs x1, fpsr
    orr x0, x0, x1, lsl #32
    ret
// x0: as returned by silica_fp_control.
.global silica_fp_set_control
silica_fp_set_control:
    lsr x1, x0, #32
    and x0, x0, #0xFFFFFFFF
    msr fpcr, x0
    msr fpsr, x1
    ret
"#
);

unsafe extern "C" {
    pub fn silica_simd_save(area: *mut u8);
    pub fn silica_simd_load(area: *const u8);
    pub fn silica_simd_clear();
    pub fn silica_fp_control() -> u64;
    pub fn silica_fp_set_control(v: u64);
}
