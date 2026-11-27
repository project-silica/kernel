//! Processor tables and the ways into and out of the kernel: GDT, TSS, IDT,
//! the `syscall` entry, the trap stubs and the return to user mode.
//!
//! The kernel runs with interrupts off on one stack per processor and never
//! blocks: every entry saves the user registers into the current Thread
//! object, runs to completion, and leaves through `exit_to_user`, which asks
//! the scheduler whom to return to. Interrupts are taken in kernel mode only
//! inside the idle wait.

use core::arch::{asm, global_asm};
use core::cell::{Cell, UnsafeCell};
use core::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use kernel_core::lock::Presence;
use kernel_core::smp::MAX_CPUS;
use silica_abi::Fault;

/// Size of the boot processor's kernel stack and of its stack for NMI,
/// double fault and machine check. The other processors' stacks come from
/// `smp.rs`.
pub const STACK_SIZE: usize = 64 * 1024;

/// What `gs` points to in kernel mode: one per processor. The offsets of
/// the first five fields are used by the assembly below. Everything a
/// system call reads here is in the first 64 bytes.
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
    /// 32: the address of this structure.
    pub this: u64,
    /// 40: the processor's index in the kernel.
    pub index: usize,
    /// `kernel_core::smp::cpu_slot(index)`, for `Hal::cpu_slot`.
    pub slot: usize,
    /// The address of this processor's kernel lock (`Kernel::cpu_lock`),
    /// which the system-call entry takes before it saves anything.
    pub lock: u64,
    /// The address of this processor's state in the kernel's table
    /// (`Kernel::cpu_state_of`), for `Hal::cpu_state`.
    pub state: u64,
    /// Its x2APIC identifier.
    pub apic_id: u32,
    /// Where the processor is; read by other processors.
    pub mode: Presence,
    /// Whether the processor has entered the kernel's scheduler.
    pub online: AtomicBool,
    /// Lowest and highest address of its kernel stack, for backtraces.
    pub stack: (u64, u64),
    /// Timestamp counter value of the armed timer (`u64::MAX`: none).
    pub deadline: Cell<u64>,
    pub polls: Cell<u32>,
    /// Process-context identifiers: the generation and the next free one.
    pub pcid_gen: Cell<u64>,
    pub pcid_next: Cell<u64>,
    /// How long this processor takes to come out of an idle wait once its
    /// timer fires, in nanoseconds (a running average; see `X86::idle`).
    pub wake_ns: Cell<u64>,
    /// How long this processor watches for work before it halts.
    pub watch: kernel_core::sched::Watch,
}
const _: () = assert!(core::mem::offset_of!(PerCpu, index) == 40 && core::mem::offset_of!(PerCpu, lock) < 64);

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

/// One processor's descriptor tables and data.
#[repr(C, align(64))]
pub struct Tables {
    gdt: [u64; 7],
    tss: Tss,
    pub percpu: PerCpu,
}

impl Tables {
    pub const fn new() -> Tables {
        Tables {
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
            percpu: PerCpu {
                kstack: 0,
                regs: 0,
                scratch: 0,
                verw: 0,
                this: 0,
                index: 0,
                apic_id: 0,
                mode: Presence::new(),
                online: AtomicBool::new(false),
                stack: (0, 0),
                deadline: Cell::new(u64::MAX),
                polls: Cell::new(0),
                pcid_gen: Cell::new(0),
                pcid_next: Cell::new(1),
                wake_ns: Cell::new(0),
                watch: kernel_core::sched::Watch::new(),
                slot: 0,
                lock: 0,
                state: 0,
            },
        }
    }
}

struct Racy<T>(UnsafeCell<T>);
// SAFETY: the boot processor's tables and the shared IDT are written during
// boot before any other processor runs; afterwards each processor touches
// only its own `PerCpu` except for the atomics in it.
unsafe impl<T> Sync for Racy<T> {}

static BOOT_TABLES: Racy<Tables> = Racy(UnsafeCell::new(Tables::new()));
static IDT: Racy<[[u64; 2]; 256]> = Racy(UnsafeCell::new([[0; 2]; 256]));
/// Every processor's tables, by index (null: no such processor).
static CPUS: [AtomicPtr<Tables>; MAX_CPUS] = [const { AtomicPtr::new(core::ptr::null_mut()) }; MAX_CPUS];

/// The boot processor's kernel stack; its lowest page is unmapped as a guard.
pub static KSTACK: Racy2 = Racy2(UnsafeCell::new(Stack([0; STACK_SIZE + 4096])));
static ISTACKS: [Racy2; 3] = [const { Racy2(UnsafeCell::new(Stack([0; STACK_SIZE + 4096]))) }; 3];

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

/// This processor's data.
pub fn percpu() -> &'static PerCpu {
    let p: u64;
    // SAFETY: in the kernel GS points at this processor's `PerCpu`, whose
    // field at offset 32 is its own address; it lives for ever.
    unsafe {
        asm!("mov {}, gs:[32]", out(reg) p, options(nostack, readonly, preserves_flags));
        &*(p as *const PerCpu)
    }
}

/// Processor `index`'s data, once its tables exist.
pub fn percpu_of(index: usize) -> Option<&'static PerCpu> {
    let t = CPUS.get(index)?.load(Ordering::Acquire);
    // SAFETY: a registered table is never freed.
    (!t.is_null()).then(|| unsafe { &(*t).percpu })
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

/// Clear processor buffers on every return to user mode from now on (this processor).
pub fn enable_verw() {
    // SAFETY: writes this processor's own flag, read only by its exit path.
    unsafe { asm!("mov qword ptr gs:[24], 1", options(nostack, preserves_flags)) }
}

/// A data segment selector for VERW to verify; the instruction's side
/// effect of overwriting the processor's internal buffers is the point.
static VERW_SELECTOR: u16 = 0x10;

/// Point this processor's entry stubs at the register frame of the thread to run.
pub fn set_regs(frame: u64) {
    // SAFETY: writes this processor's own field, read by its entry and
    // exit stubs; interrupts are off.
    unsafe { asm!("mov gs:[8], {}", in(reg) frame, options(nostack, preserves_flags)) }
}

/// The boot processor's tables (before any other is registered).
pub fn boot_tables() -> *mut Tables {
    BOOT_TABLES.0.get()
}

/// The boot processor's stacks: (kernel lowest, kernel top, the NMI,
/// double-fault and machine-check stacks' tops).
pub fn boot_stacks() -> (u64, u64, [u64; 3]) {
    (KSTACK.base() + 4096, KSTACK.top(), [ISTACKS[0].top(), ISTACKS[1].top(), ISTACKS[2].top()])
}

/// Fill the interrupt descriptor table every processor shares. Boot processor, once.
///
/// # Safety
/// Before any processor loads the table.
pub unsafe fn init_idt() {
    let stubs = &raw const trap_stubs as u64;
    for v in 0..256u64 {
        let h = stubs + v * 16;
        // NMI, double fault and machine check each run on a stack of its
        // own (TSS IST 1 to 3): sharing one, a machine check taken while an
        // NMI's frame was still on it (or the reverse) wrote over that frame
        // (the review of nmi-exit).
        let ist = match v {
            2 => 1,
            8 => 2,
            18 => 3,
            _ => 0,
        };
        // SAFETY: guaranteed by the caller.
        unsafe { (*IDT.0.get())[v as usize] = [(h & 0xFFFF) | 0x08 << 16 | ist << 32 | 0x8E << 40 | (h >> 16 & 0xFFFF) << 48, h >> 32] };
    }
}

/// Make `t` this processor's tables: load them and set up `syscall`.
/// `stack` is (lowest address, top) of its kernel stack, `ists` the tops of
/// its stacks for NMI, double fault and machine check, in that order.
///
/// # Safety
/// Once per processor, early, with interrupts off; `t` and the stacks must
/// live for ever and belong to this processor alone.
/// Out of line: it sets the GS base, which `Hal::cpu_slot` and `Hal::cpu`
/// read as constant (`nomem`), so no read of either may be moved into it.
#[inline(never)]
pub unsafe fn init(t: *mut Tables, index: usize, apic_id: u32, stack: (u64, u64), ists: [u64; 3]) {
    // SAFETY: exclusive access to this processor's tables during its boot;
    // the descriptors built here describe memory that lives for ever.
    unsafe {
        let tss = &raw const (*t).tss as u64;
        (*t).gdt[5] = 0x0000_8900_0000_0067 | (tss & 0xFF_FFFF) << 16 | (tss >> 24 & 0xFF) << 56;
        (*t).gdt[6] = tss >> 32;
        (*t).tss.rsp0 = stack.1;
        // Each of the NMI, double-fault and machine-check stacks starts
        // 16-byte aligned (the processor aligns it so, and the frame must be
        // where `paranoid_common` looks), and the word above it holds this
        // processor's PerCpu address: how that entry tells whose GS base
        // is loaded, wherever it interrupted.
        for (i, top) in ists.into_iter().enumerate() {
            let ist = (top - 16) & !15;
            *(ist as *mut u64).add(1) = &raw const (*t).percpu as u64;
            (*t).tss.ist[i] = ist;
        }
        (*t).percpu.kstack = stack.1;
        (*t).percpu.stack = stack;
        (*t).percpu.this = &raw const (*t).percpu as u64;
        (*t).percpu.index = index;
        (*t).percpu.slot = kernel_core::smp::cpu_slot(index);
        (*t).percpu.lock = super::kernel().cpu_lock(index) as *const kernel_core::lock::Lock as u64;
        (*t).percpu.state = core::ptr::from_ref(super::kernel().cpu_state_of(index)) as u64;
        (*t).percpu.apic_id = apic_id;
        CPUS[index].store(t, Ordering::Release);
        let gdtr: [u16; 5] = pseudo(&raw const (*t).gdt as u64, 7 * 8 - 1);
        let idtr: [u16; 5] = pseudo(IDT.0.get() as u64, 256 * 16 - 1);
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
    /// Leave the kernel for the thread the scheduler picks. The caller has
    /// entered the kernel (`Kernel::enter`).
    pub fn exit_to_user() -> !;
    /// The second half of `exit_to_user`: return to the thread already
    /// scheduled (its frame at gs:[8]); the caller has let go of the lock.
    static restore_user: u8;
}

// Each way in marks the processor as in the kernel and takes its lock; the
// one way out picks the thread, marks it as in user mode while still
// holding the lock, and lets go. Another processor holding this one's lock
// therefore sees either "in the kernel" (it will look at what changed when
// it gets the lock) or "in user mode" (it must be interrupted first).

/// `contended`: the entry found this processor's lock held by another
/// processor and did not take it.
///
/// It does not return: it schedules and goes straight to `restore_user`.
/// Everything the thread had is in its frame, so nothing this function is
/// called with needs keeping: returning to the stub and calling the
/// scheduler from there cost each system call a second call and twelve
/// saves and restores (`profile.sh --lines`: 16 instructions at each of
/// the two functions' entry and exit per round trip). Under the C
/// convention it still saved the five registers a C caller keeps (rbx,
/// r12 to r15) on every system call, for a caller it never returns to;
/// under `preserve_none` it keeps none of them (`contended` comes in r12).
extern "rust-preserve-none" fn syscall_rust(contended: u64) -> ! {
    let k = super::kernel();
    percpu().mode.enter_kernel();
    if contended == 0 {
        k.entered();
    } else {
        k.enter();
    }
    k.syscall();
    k.schedule();
    percpu().mode.leave_to_user();
    k.leave();
    // SAFETY: `restore_user` loads the scheduled thread's frame (gs:[8])
    // and returns to user mode; it takes nothing from this stack, which
    // the next entry starts afresh (gs:[0]).
    unsafe { asm!("jmp {r}", r = sym restore_user, options(noreturn)) }
}

/// The way out of the kernel for a processor that was not entered from
/// user mode (`exit_to_user`): `preserve_none`, as `syscall_rust`, since the
/// stub needs nothing kept across it.
extern "rust-preserve-none" fn schedule_rust() {
    let k = super::kernel();
    k.schedule();
    percpu().mode.leave_to_user();
    k.leave();
}

/// A trap or interrupt from user mode (`vector`, the error code `err`, in
/// r12 and r13). It does not return, as `syscall_rust` does not: it
/// schedules and goes to `restore_user` itself. Returning to the stub to
/// call `schedule_rust` cost every fault and interrupt a second call, and
/// under the C convention each saved the five registers a C caller keeps.
extern "rust-preserve-none" fn trap_user(vector: u64, err: u64) -> ! {
    let k = super::kernel();
    percpu().mode.enter_kernel();
    // The platform's NMI, or the request to stop (`super::nmi`), before
    // this processor's lock: a processor waiting for its lock, held by one
    // that panicked, must still hear the request to stop. `nmi` takes no
    // lock and reads no per-processor data.
    if vector == 2 {
        super::nmi();
        unblock_nmis();
    }
    k.enter();
    match vector {
        14 => {
            let cr2: u64;
            // SAFETY: reading CR2 in ring 0 has no side effect.
            unsafe { asm!("mov {}, cr2", out(reg) cr2, options(nomem, nostack, preserves_flags)) }
            // The x86 error code's present, write and fetch bits as the
            // ABI's architecture-neutral access bits.
            let access =
                (err & 1) * silica_abi::FAULT_PRESENT | (err >> 1 & 1) * silica_abi::FAULT_WRITE | (err >> 4 & 1) * silica_abi::FAULT_FETCH;
            k.fault(Fault::Page, cr2, access);
        }
        6 => k.fault(Fault::Instruction, 0, 0),
        // Handled above, before the lock.
        2 => {}
        // A machine check consumed by the program: its fault (`super::machine_check_user`).
        18 => super::machine_check_user(k),
        8 => panic!("double fault (error {err:#x}) in user mode"),
        32.. => super::irq(vector, true),
        v => k.fault(Fault::Exception, v, err),
    }
    k.schedule();
    percpu().mode.leave_to_user();
    k.leave();
    // SAFETY: as in `syscall_rust`.
    unsafe { asm!("jmp {r}", r = sym restore_user, options(noreturn)) }
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
    // NMIs and machine checks come in through `paranoid_common`, with the
    // kernel's GS base whatever they interrupted.
    match f.vector {
        2 => return super::nmi(),
        18 => return super::machine_check_kernel(),
        _ => {}
    }
    if f.vector >= 32 {
        // Only the idle wait takes interrupts in kernel mode, and it does
        // not hold the processor's lock.
        super::irq(f.vector, false);
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
    // Try this processor's lock before the frame is saved, while few
    // stores wait to drain: XCHG is a full barrier and waits for them all.
    // Holding the lock, nobody can look at the frame while it is written.
    // bl is 0 when the lock was taken here, 1 when another processor holds
    // it (syscall_rust then follows the usual order: frame saved, presence
    // marked, lock awaited).
    mov rax, gs:[{lock}]
    mov bl, 1
    xchg [rax], bl
    // rcx and r11 hold the return address and flags (saved below as
    // rip and rflags): a thread that entered by `syscall` resumes by
    // `sysret`, which loads them from there, so their own slots (16, 80)
    // are not written. Only the `iret` way out reads those, for a thread
    // that entered by a trap (which writes them) or was started afresh
    // (all zero).
    mov [rsp + 24], rdx
    mov [rsp + 32], rsi
    mov [rsp + 40], rdi
    mov [rsp + 48], rbp
    mov [rsp + 56], r8
    mov [rsp + 64], r9
    mov [rsp + 72], r10
    mov [rsp + 88], r12
    mov [rsp + 96], r13
    mov [rsp + 104], r14
    mov [rsp + 112], r15
    mov [rsp + 120], rcx
    mov [rsp + 128], r11
    mov rax, gs:[16]
    mov [rsp + 136], rax
    mov qword ptr [rsp + 160], 1
    movzx r12d, bl
    mov rsp, gs:[0]
    xor ebp, ebp
    call {syscall}
    ud2

.global exit_to_user
exit_to_user:
    mov rsp, gs:[0]
    call {schedule}
.global restore_user
restore_user:
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
    mov r12, [rsp]
    mov r13, [rsp + 8]
    mov rsp, gs:[0]
    xor ebp, ebp
    call {trap_user}
    ud2
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

// NMI, double fault and machine check (on their own stack): from user mode
// the ordinary entry; in the kernel they may arrive where GS is still the
// user's (between `syscall` and its `swapgs`, or after the `swapgs` on the
// way out), so the GS base is compared with this processor's own PerCpu
// address, which `init` keeps in the word above this stack: the kernel's,
// or it is swapped in now and back on the way out. (It was told by the
// sign of the GS base, which a program can make negative with WRGSBASE:
// such an NMI ran the kernel on the program's GS base.) Ordinary
// interrupts never arrive there and keep the cheaper path.
paranoid_common:
    cld
    test byte ptr [rsp + 24], 3
    jnz trap_common
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
    mov ecx, 0xC0000101
    rdmsr
    shl rdx, 32
    or rax, rdx
    xor r12d, r12d
    // Above the 15 registers, the vector and error code, and the
    // processor's 5 words: the word `init` put above the stack.
    cmp rax, [rsp + 184]
    je 3f
    swapgs
    mov r12d, 1
3:
    lfence
    mov rdi, rsp
    call {trap_kernel}
    test r12d, r12d
    jz 4f
    swapgs
4:
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
// error code, push the vector, join the common path (NMI, double fault and
// machine check: the paranoid one).
.macro TRAP_STUB n
    .align 16
    .if (\n == 8) || (\n >= 10 && \n <= 14) || (\n == 17) || (\n == 21) || (\n == 29) || (\n == 30)
    .else
    push 0
    .endif
    push \n
    .if (\n == 2) || (\n == 8) || (\n == 18)
    jmp paranoid_common
    .else
    jmp trap_common
    .endif
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
    lock = const core::mem::offset_of!(PerCpu, lock),
    trap_user = sym trap_user,
    trap_kernel = sym trap_kernel,
);

/// Let NMIs in again after one taken from user mode. The processor holds
/// further NMIs off from the NMI until its next IRET, and the way back to
/// user mode may be SYSRET (when the thread that runs next entered by a
/// system call), which does not end that: until some later IRET, every NMI
/// to this processor waited, a panic's request to stop among them (the
/// review of kernel hot-paths). An IRET to the next instruction, in kernel
/// mode on this stack with these flags (interrupts stay off), ends it.
fn unblock_nmis() {
    // SAFETY: builds an interrupt frame for the instruction after IRETQ,
    // with this stack, these flags and the kernel's selectors, and returns
    // to it; the kernel has no red zone below the stack pointer to clobber.
    unsafe {
        asm!(
            "mov {t:e}, ss",
            "push {t}",
            "lea {t}, [rsp + 8]",
            "push {t}",
            "pushfq",
            "mov {t:e}, cs",
            "push {t}",
            "lea {t}, [rip + 2f]",
            "push {t}",
            "iretq",
            "2:",
            t = out(reg) _,
        );
    }
}
