//! Processor tables and the ways into and out of the kernel: GDT, TSS, IDT,
//! the `syscall` entry, the trap stubs and the return to user mode.
//!
//! The kernel runs with interrupts off on one stack per processor and never
//! blocks: every entry saves the user registers into the current Thread
//! object, runs to completion, and leaves through `exit_to_user`, which asks
//! the scheduler whom to return to. Interrupts are taken in kernel mode only
//! inside the idle wait.

use core::arch::{asm, global_asm};
use core::cell::UnsafeCell;
use silica_abi::Fault;

/// Size of the kernel stack and of the stack for NMI, double fault and
/// machine check.
pub const STACK_SIZE: usize = 64 * 1024;

/// What `gs` points to in kernel mode. Offsets are used by the assembly below.
#[repr(C)]
pub struct PerCpu {
    /// 0: top of the kernel stack.
    pub kstack: u64,
    /// 8: the current thread's register frame.
    pub regs: u64,
    /// 16: scratch for the syscall entry.
    pub scratch: u64,
    /// 24: not 0 when the processor needs its buffers cleared (VERW) on
    /// the way back to user mode.
    pub verw: u64,
}

#[repr(C, packed)]
struct Tss {
    reserved0: u32,
    rsp0: u64,
    rsp1: u64,
    rsp2: u64,
    reserved1: u64,
    ist: [u64; 7],
    reserved2: u64,
    reserved3: u16,
    iomap: u16,
}

#[repr(C, align(4096))]
pub struct Stack(pub [u8; STACK_SIZE + 4096]);

#[repr(C, align(64))]
struct Tables {
    gdt: [u64; 7],
    tss: Tss,
    percpu: PerCpu,
    idt: [[u64; 2]; 256],
}

struct Racy<T>(UnsafeCell<T>);
// SAFETY: these are per-processor tables of the single processor that runs
// kernel code at M1, written during boot and by `set_regs` with interrupts off.
unsafe impl<T> Sync for Racy<T> {}

static TABLES: Racy<Tables> = Racy(UnsafeCell::new(Tables {
    gdt: [
        0,
        0x00AF_9A00_0000_FFFF, // 0x08 kernel code
        0x00CF_9200_0000_FFFF, // 0x10 kernel data
        0x00CF_F200_0000_FFFF, // 0x18 user data
        0x00AF_FA00_0000_FFFF, // 0x20 user code
        0,                     // 0x28 TSS (two entries)
        0,
    ],
    tss: Tss { reserved0: 0, rsp0: 0, rsp1: 0, rsp2: 0, reserved1: 0, ist: [0; 7], reserved2: 0, reserved3: 0, iomap: 104 },
    percpu: PerCpu { kstack: 0, regs: 0, scratch: 0, verw: 0 },
    idt: [[0; 2]; 256],
}));

/// The kernel stack; its lowest page is unmapped as a guard.
pub static KSTACK: Racy2 = Racy2(UnsafeCell::new(Stack([0; STACK_SIZE + 4096])));
static ISTACK: Racy2 = Racy2(UnsafeCell::new(Stack([0; STACK_SIZE + 4096])));

pub struct Racy2(UnsafeCell<Stack>);
// SAFETY: stacks are only used by the processor as stacks.
unsafe impl Sync for Racy2 {}
impl Racy2 {
    /// Lowest address of the stack's memory (the guard page).
    pub fn base(&self) -> u64 {
        self.0.get() as u64
    }
    /// One past the highest address.
    pub fn top(&self) -> u64 {
        self.base() + (STACK_SIZE + 4096) as u64
    }
}

pub fn rdmsr(msr: u32) -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: callers read MSRs that exist on the processor (checked by
    // CPUID where optional); reading has no memory effect.
    unsafe { asm!("rdmsr", in("ecx") msr, out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags)) }
    (hi as u64) << 32 | lo as u64
}

/// # Safety
/// The MSR must exist and the value must keep the kernel's execution
/// environment valid.
pub unsafe fn wrmsr(msr: u32, v: u64) {
    // SAFETY: guaranteed by the caller.
    unsafe { asm!("wrmsr", in("ecx") msr, in("eax") v as u32, in("edx") (v >> 32) as u32, options(nomem, nostack, preserves_flags)) }
}

pub fn cpuid(leaf: u32, sub: u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    // SAFETY: CPUID is available on every x86_64 processor; rbx is saved
    // around it because the compiler reserves it.
    unsafe {
        asm!("mov {t:r}, rbx", "cpuid", "xchg {t:r}, rbx", t = out(reg) b, inout("eax") leaf => a, inout("ecx") sub => c, out("edx") d, options(nomem, nostack, preserves_flags))
    }
    [a, b, c, d]
}

pub fn rdtsc() -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: RDTSC only reads the timestamp counter.
    unsafe { asm!("rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags)) }
    (hi as u64) << 32 | lo as u64
}

pub fn read_cr3() -> u64 {
    let v: u64;
    // SAFETY: reading CR3 in ring 0 has no side effect.
    unsafe { asm!("mov {}, cr3", out(reg) v, options(nomem, nostack, preserves_flags)) }
    v
}

/// # Safety
/// `root` must be a page table that maps the kernel as the current one does.
pub unsafe fn write_cr3(root: u64) {
    // SAFETY: guaranteed by the caller.
    unsafe { asm!("mov cr3, {}", in(reg) root, options(nostack, preserves_flags)) }
}

/// Clear processor buffers on every return to user mode from now on.
pub fn enable_verw() {
    // SAFETY: as for `set_regs`.
    unsafe { (*TABLES.0.get()).percpu.verw = 1 }
}

/// A data segment selector for VERW to verify; the instruction's side
/// effect of overwriting the processor's internal buffers is the point.
static VERW_SELECTOR: u16 = 0x10;

/// Point the entry stubs at the register frame of the thread to run.
pub fn set_regs(frame: u64) {
    // SAFETY: one processor, interrupts off; the assembly reads this field
    // only on kernel entry and exit.
    unsafe { (*TABLES.0.get()).percpu.regs = frame }
}

/// Load the tables and set up `syscall`. Called once on the boot processor.
///
/// # Safety
/// Must run once, early, on the kernel's initial stack with interrupts off.
pub unsafe fn init() {
    let t = TABLES.0.get();
    // SAFETY: boot-time exclusive access to the tables; the descriptors
    // built here describe memory that lives for ever (statics).
    unsafe {
        let tss = &raw const (*t).tss as u64;
        (*t).gdt[5] = 0x0000_8900_0000_0067 | (tss & 0xFF_FFFF) << 16 | (tss >> 24 & 0xFF) << 56;
        (*t).gdt[6] = tss >> 32;
        (*t).tss.rsp0 = KSTACK.top();
        (*t).tss.ist[0] = ISTACK.top();
        (*t).percpu.kstack = KSTACK.top();
        let stubs = &raw const trap_stubs as u64;
        for v in 0..256u64 {
            let h = stubs + v * 16;
            // NMI, double fault and machine check run on their own stack.
            let ist = if matches!(v, 2 | 8 | 18) { 1 } else { 0 };
            (*t).idt[v as usize] = [(h & 0xFFFF) | 0x08 << 16 | ist << 32 | 0x8E << 40 | (h >> 16 & 0xFFFF) << 48, h >> 32];
        }
        let gdtr: [u16; 5] = pseudo(&raw const (*t).gdt as u64, 7 * 8 - 1);
        let idtr: [u16; 5] = pseudo(&raw const (*t).idt as u64, 256 * 16 - 1);
        asm!(
            "lgdt [{gdtr}]",
            "lidt [{idtr}]",
            "push 0x08",
            "lea {tmp}, [rip + 2f]",
            "push {tmp}",
            "retfq",
            "2:",
            "mov ax, 0x10",
            "mov ds, ax",
            "mov es, ax",
            "mov ss, ax",
            "xor eax, eax",
            "mov fs, ax",
            "mov gs, ax",
            "mov ax, 0x28",
            "ltr ax",
            gdtr = in(reg) gdtr.as_ptr(),
            idtr = in(reg) idtr.as_ptr(),
            tmp = out(reg) _,
            out("rax") _,
        );
        wrmsr(0xC000_0101, &raw const (*t).percpu as u64); // GS base: per-CPU data
        wrmsr(0xC000_0102, 0); // the user's GS base while in the kernel
        wrmsr(0xC000_0080, rdmsr(0xC000_0080) | 1); // EFER.SCE
        wrmsr(0xC000_0081, 0x10 << 48 | 0x08 << 32); // STAR: user and kernel selectors
        wrmsr(0xC000_0082, &raw const syscall_entry as u64); // LSTAR
        wrmsr(0xC000_0084, 0x4_4700); // SFMASK: clear TF, IF, DF, NT, AC
    }
}

fn pseudo(base: u64, limit: u16) -> [u16; 5] {
    [limit, base as u16, (base >> 16) as u16, (base >> 32) as u16, (base >> 48) as u16]
}

unsafe extern "C" {
    static trap_stubs: u8;
    static syscall_entry: u8;
    /// Leave the kernel for the thread the scheduler picks.
    pub fn exit_to_user() -> !;
}

extern "C" fn syscall_rust() {
    super::kernel().syscall();
}

extern "C" fn schedule_rust() {
    super::kernel().schedule();
}

extern "C" fn trap_user(vector: u64, err: u64) {
    let k = super::kernel();
    match vector {
        14 => {
            let cr2: u64;
            // SAFETY: reading CR2 in ring 0 has no side effect.
            unsafe { asm!("mov {}, cr2", out(reg) cr2, options(nomem, nostack, preserves_flags)) }
            k.fault(Fault::Page, cr2, err);
        }
        6 => k.fault(Fault::Instruction, 0, 0),
        2 | 8 | 18 => panic!("fatal exception {vector} (error {err:#x}) in user mode"),
        32.. => super::irq(vector),
        v => k.fault(Fault::Exception, v, err),
    }
}

/// Registers pushed by the kernel-mode trap path.
#[repr(C)]
pub struct KernelFrame {
    pub gpr: [u64; 15],
    pub vector: u64,
    pub error: u64,
    pub rip: u64,
    pub cs: u64,
    pub rflags: u64,
    pub rsp: u64,
    pub ss: u64,
}

extern "C" fn trap_kernel(f: &KernelFrame) {
    if f.vector >= 32 {
        super::irq(f.vector);
        return;
    }
    super::kernel_fault(f);
}

// Register frame offsets (index * 8) match `REG_*` in mod.rs: rax rbx rcx
// rdx rsi rdi rbp r8..r15 at 0..14, rip 15, rflags 16, rsp 17; 20 is not 0
// when the frame was saved by `syscall` (rcx and r11 hold nothing to keep).
// `lfence` after each conditional `swapgs` keeps the processor from running
// ahead with the wrong GS base (Spectre v1, swapgs variant).
global_asm!(
    r#"
.section .text.boot, "ax"
.global _start
_start:
    cli
    lea rsp, [rip + {kstack} + {stack_size} + 4096]
    xor ebp, ebp
    call {kmain}
    ud2

.section .text.fast, "ax"
.global syscall_entry
.align 64
syscall_entry:
    swapgs
    mov gs:[16], rsp
    mov rsp, gs:[8]
    mov [rsp + 0], rax
    mov [rsp + 8], rbx
    mov [rsp + 16], rcx
    mov [rsp + 24], rdx
    mov [rsp + 32], rsi
    mov [rsp + 40], rdi
    mov [rsp + 48], rbp
    mov [rsp + 56], r8
    mov [rsp + 64], r9
    mov [rsp + 72], r10
    mov [rsp + 80], r11
    mov [rsp + 88], r12
    mov [rsp + 96], r13
    mov [rsp + 104], r14
    mov [rsp + 112], r15
    mov [rsp + 120], rcx
    mov [rsp + 128], r11
    mov rax, gs:[16]
    mov [rsp + 136], rax
    mov qword ptr [rsp + 160], 1
    mov rsp, gs:[0]
    xor ebp, ebp
    call {syscall}

.global exit_to_user
exit_to_user:
    mov rsp, gs:[0]
    call {schedule}
    mov rax, gs:[8]
    cmp byte ptr gs:[24], 0
    je 5f
    verw word ptr [rip + {verw}]
5:
    cmp qword ptr [rax + 160], 0
    je 6f
    // The thread entered by `syscall`: rcx and r11 are dead, so `sysret`
    // can use them. Its rip is canonical: it was saved by `syscall` below
    // USER_TOP or checked by ThreadStart.
    mov rbx, [rax + 8]
    mov rdx, [rax + 24]
    mov rsi, [rax + 32]
    mov rdi, [rax + 40]
    mov rbp, [rax + 48]
    mov r8, [rax + 56]
    mov r9, [rax + 64]
    mov r10, [rax + 72]
    mov r12, [rax + 88]
    mov r13, [rax + 96]
    mov r14, [rax + 104]
    mov r15, [rax + 112]
    mov rcx, [rax + 120]
    mov r11, [rax + 128]
    mov rsp, [rax + 136]
    mov rax, [rax]
    swapgs
    sysretq
6:
    push 0x1B
    push qword ptr [rax + 136]
    push qword ptr [rax + 128]
    push 0x23
    push qword ptr [rax + 120]
    mov rbx, [rax + 8]
    mov rcx, [rax + 16]
    mov rdx, [rax + 24]
    mov rsi, [rax + 32]
    mov rdi, [rax + 40]
    mov rbp, [rax + 48]
    mov r8, [rax + 56]
    mov r9, [rax + 64]
    mov r10, [rax + 72]
    mov r11, [rax + 80]
    mov r12, [rax + 88]
    mov r13, [rax + 96]
    mov r14, [rax + 104]
    mov r15, [rax + 112]
    mov rax, [rax]
    swapgs
    iretq

trap_common:
    cld
    test byte ptr [rsp + 24], 3
    jz 2f
    swapgs
    lfence
    push rax
    mov rax, gs:[8]
    mov qword ptr [rax + 160], 0
    mov [rax + 8], rbx
    mov [rax + 16], rcx
    mov [rax + 24], rdx
    mov [rax + 32], rsi
    mov [rax + 40], rdi
    mov [rax + 48], rbp
    mov [rax + 56], r8
    mov [rax + 64], r9
    mov [rax + 72], r10
    mov [rax + 80], r11
    mov [rax + 88], r12
    mov [rax + 96], r13
    mov [rax + 104], r14
    mov [rax + 112], r15
    pop rbx
    mov [rax], rbx
    mov rbx, [rsp + 16]
    mov [rax + 120], rbx
    mov rbx, [rsp + 32]
    mov [rax + 128], rbx
    mov rbx, [rsp + 40]
    mov [rax + 136], rbx
    mov rdi, [rsp]
    mov rsi, [rsp + 8]
    mov rsp, gs:[0]
    xor ebp, ebp
    call {trap_user}
    jmp exit_to_user
2:
    lfence
    push r15
    push r14
    push r13
    push r12
    push r11
    push r10
    push r9
    push r8
    push rbp
    push rdi
    push rsi
    push rdx
    push rcx
    push rbx
    push rax
    mov rdi, rsp
    call {trap_kernel}
    pop rax
    pop rbx
    pop rcx
    pop rdx
    pop rsi
    pop rdi
    pop rbp
    pop r8
    pop r9
    pop r10
    pop r11
    pop r12
    pop r13
    pop r14
    pop r15
    add rsp, 16
    iretq

// One 16-byte stub per vector: push a zero where the processor pushes no
// error code, push the vector, join the common path.
.macro TRAP_STUB n
    .align 16
    .if (\n == 8) || (\n >= 10 && \n <= 14) || (\n == 17) || (\n == 21) || (\n == 29) || (\n == 30)
    .else
    push 0
    .endif
    push \n
    jmp trap_common
.endm
.align 16
.global trap_stubs
trap_stubs:
.altmacro
.set vec, 0
.rept 256
    TRAP_STUB %vec
    .set vec, vec + 1
.endr
.noaltmacro
"#,
    kstack = sym KSTACK,
    stack_size = const STACK_SIZE,
    kmain = sym super::kmain,
    syscall = sym syscall_rust,
    schedule = sym schedule_rust,
    verw = sym VERW_SELECTOR,
    trap_user = sym trap_user,
    trap_kernel = sym trap_kernel,
);
