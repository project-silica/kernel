//! x86_64: boot, the `Hal` implementation, the clock and the panic path.
//!
//! Required of the processor: x2APIC, an invariant TSC, XSAVE and NX. Every
//! x86_64 processor made since 2013 and QEMU (KVM, or TCG with `-cpu max`)
//! has them; the kernel stops with a message naming the missing one
//! otherwise. There is no PIC, PIT or xAPIC fallback. The timer is the
//! TSC-deadline timer; where the platform does not offer it (QEMU's own
//! APIC emulation) the local APIC's one-shot counter is used instead.

mod cpu;
mod fbcon;
#[cfg(feature = "harness")]
mod harness;
mod ioapic;
mod iommu;
mod serial;
mod smp;

use core::arch::asm;
use core::cell::Cell;
use core::fmt::Write;
use core::sync::atomic::Ordering;
use cpu::percpu;
use cpu::{cpuid, rdmsr, rdtsc, wrmsr};
use kernel_core::boot::{BOOT_MAGIC, BOOT_VERSION, BootInfo, Cmdline, PHYS_MAP_BASE, Region, RegionTable};
use kernel_core::hal::Hal;
use kernel_core::mem::{Paddr, Phys};
use kernel_core::root::RootSpec;
use kernel_core::thread::FPU_OFFSET;
use kernel_core::{Kernel, klog};
use silica_abi::Rights;

const REG_RAX: usize = 0;
const REG_RDX: usize = 3;
const REG_RSI: usize = 4;
const REG_RDI: usize = 5;
const REG_R8: usize = 7;
const REG_R9: usize = 8;
const REG_R10: usize = 9;
const REG_RIP: usize = 15;
const REG_RFLAGS: usize = 16;
const REG_RSP: usize = 17;
const REG_FS_BASE: usize = 18;
const REG_GS_BASE: usize = 19;
/// Not 0 when the frame was saved by `syscall` (set by the entry stubs).
const REG_BY_SYSCALL: usize = 20;
/// MXCSR (low half) and the x87 control word of a thread switched out
/// inside a system call.
const REG_FPCTL: usize = 21;

pub(super) const PTE_P: u64 = 1;
pub(super) const PTE_W: u64 = 2;
const PTE_U: u64 = 4;
const PTE_PCD: u64 = 0x10;
/// Accessed and dirty. The kernel sets them in every entry it writes: the
/// processor sets a clear one itself on first use, with a microcode assist
/// that costs more than the rest of a page fault (measured in a virtual
/// machine: the first write to a mapped page 620 timestamp ticks with them
/// clear, against 6 for the next cache line). Nothing in the kernel reads
/// them; which pages were used is a pager's question, to be asked
/// explicitly when a pager needs it.
const PTE_A: u64 = 0x20;
const PTE_D: u64 = 0x40;
const PTE_PS: u64 = 0x80;
const PTE_NX: u64 = 1 << 63;
pub(super) const PTE_ADDR: u64 = 0x000F_FFFF_FFFF_F000;

const TIMER_VECTOR: u64 = 0x20;
/// Another processor wants this one in the kernel.
const KICK_VECTOR: u64 = 0xF1;
const SPURIOUS_VECTOR: u64 = 0xFF;
const MSR_APIC_BASE: u32 = 0x1B;
const MSR_TSC_DEADLINE: u32 = 0x6E0;
const MSR_X2APIC_EOI: u32 = 0x80B;
const MSR_X2APIC_SVR: u32 = 0x80F;
const MSR_X2APIC_IRR0: u32 = 0x820;
const MSR_X2APIC_LVT_TIMER: u32 = 0x832;
const MSR_X2APIC_TIMER_COUNT: u32 = 0x838;
const MSR_X2APIC_TIMER_NOW: u32 = 0x839;
const MSR_X2APIC_TIMER_DIVIDE: u32 = 0x83E;
const MSR_FS_BASE: u32 = 0xC000_0100;
const MSR_KERNEL_GS_BASE: u32 = 0xC000_0102;

/// What the x86_64 layer knows about the machine. Set by the boot processor
/// before any other starts, then only read; per-processor state is in
/// `cpu::PerCpu`.
pub struct X86 {
    tsc_hz: Cell<u64>,
    /// Timestamp counter at monotonic time 0.
    tsc0: Cell<u64>,
    /// Nanoseconds per tick, and ticks per nanosecond, as 32.32 fixed point.
    ns_per_tick: Cell<u64>,
    ticks_per_ns: Cell<u64>,
    /// The page table with only the kernel's mappings.
    kernel_root: Cell<u64>,
    /// Local APIC timer counts per nanosecond as 32.32 fixed point; 0 when
    /// the TSC-deadline timer is in use.
    apic_per_ns: Cell<u64>,
    slide: Cell<u64>,
    /// How extended state is saved: 0 XSAVE, 1 XSAVEOPT, 2 XSAVES (compacted).
    xsave_kind: Cell<u8>,
    xcr0: Cell<u64>,
    /// XGETBV with ECX = 1 reports which state components are in use.
    xgetbv1: Cell<bool>,
    fsgsbase: Cell<bool>,
    /// The processors have process-context identifiers.
    pcid: Cell<bool>,
    /// Bits: 1 enhanced IBRS on, 2 BHI_DIS_S on, 4 VERW on exit, 8 no
    /// enhanced IBRS (indirect branches unprotected), 16 Meltdown-affected.
    mitigations: Cell<u8>,
    /// Isolation against speculation (`Hal::isolate`): whether the
    /// processor has IA32_SPEC_CTRL and IBPB, the SPEC_CTRL bits always on
    /// (enhanced IBRS, BHI), and those added while an isolated address
    /// space runs (STIBP, SSBD).
    spec_ctrl: Cell<bool>,
    ibpb: Cell<bool>,
    spec_base: Cell<u64>,
    spec_isolated: Cell<u64>,
}

struct Global(Kernel<X86>);
// SAFETY: every processor enters the kernel through `Kernel::enter`, and
// kernel-core's lock protocol (`smp.rs` there) decides what each may touch.
// The `X86` fields are written by the boot processor before any other starts.
unsafe impl Sync for Global {}

static KERNEL: Global = Global(Kernel::new(
    X86 {
        tsc_hz: Cell::new(0),
        tsc0: Cell::new(0),
        ns_per_tick: Cell::new(0),
        ticks_per_ns: Cell::new(0),
        kernel_root: Cell::new(0),
        apic_per_ns: Cell::new(0),
        slide: Cell::new(0),
        xsave_kind: Cell::new(0),
        xcr0: Cell::new(0),
        xgetbv1: Cell::new(false),
        fsgsbase: Cell::new(false),
        pcid: Cell::new(false),
        mitigations: Cell::new(0),
        spec_ctrl: Cell::new(false),
        ibpb: Cell::new(false),
        spec_base: Cell::new(0),
        spec_isolated: Cell::new(0),
    },
    // SAFETY: the loader maps all physical memory at PHYS_MAP_BASE and the
    // kernel only uses addresses from the boot regions through it.
    unsafe { Phys::new(PHYS_MAP_BASE as *mut u8) },
));

/// The kernel.
pub fn kernel() -> &'static Kernel<X86> {
    &KERNEL.0
}

impl Hal for X86 {
    const IP: usize = REG_RIP;
    const SP: usize = REG_RSP;
    const FLAGS: usize = REG_RFLAGS;
    const SYS: usize = REG_RAX;
    const BY_SYSCALL: usize = REG_BY_SYSCALL;
    const ARG: [usize; 6] = [REG_RDI, REG_RSI, REG_RDX, REG_R10, REG_R8, REG_R9];
    const FLAGS_INIT: u64 = 0x202;
    const SYSCALL_LEN: u64 = 2;
    /// The last page below the canonical boundary stays unmapped, so no
    /// `syscall` instruction can sit where its return address would be
    /// non-canonical.
    const USER_TOP: u64 = 0x0000_7FFF_FFFF_F000;
    /// CR3 is the only root: every program's carries the kernel's upper half.
    const KERNEL_IN_ROOT: bool = true;

    fn pte_table(table: Paddr) -> u64 {
        table | PTE_P | PTE_W | PTE_U | PTE_A
    }
    fn pte_leaf(frame: Paddr, rights: Rights, level: u8, device: bool) -> u64 {
        let mut e = frame | PTE_P | PTE_U | PTE_A;
        if rights.has(Rights::WRITE) {
            // Dirty only with write: a read-only entry marked dirty means a
            // shadow stack to the processor.
            e |= PTE_W | PTE_D;
        }
        if !rights.has(Rights::EXEC) {
            e |= PTE_NX;
        }
        if level == 1 {
            e |= PTE_PS;
        }
        if device {
            e |= PTE_PCD;
        }
        e
    }
    fn pte_present(pte: u64) -> bool {
        pte & PTE_P != 0
    }
    fn pte_is_leaf(pte: u64, level: u8) -> bool {
        level == 0 || pte & PTE_PS != 0
    }
    fn pte_addr(pte: u64) -> Paddr {
        pte & PTE_ADDR
    }

    fn io_pte_table(table: Paddr) -> u64 {
        table | PTE_P | PTE_W
    }
    fn io_pte_leaf(frame: Paddr, rights: Rights, level: u8) -> u64 {
        // Read and write as a processor's present and writable bits; the
        // IOMMU ignores or defines nothing else that is set here.
        frame | PTE_P | if rights.has(Rights::WRITE) { PTE_W } else { 0 } | if level == 1 { PTE_PS } else { 0 }
    }
    fn io_top(&self) -> u64 {
        iommu::top()
    }
    fn io_large(&self) -> bool {
        iommu::large()
    }
    fn io_attach(&self, phys: &Phys, id: u64, root: Paddr) -> Result<Paddr, silica_abi::Error> {
        iommu::attach(phys, id, root)
    }
    fn io_flush(&self, phys: &Phys, root: Paddr) {
        iommu::flush(phys, root);
    }
    fn io_mapped(&self, phys: &Phys, root: Paddr) {
        iommu::mapped(phys, root);
    }
    fn io_release(&self, phys: &Phys, root: Paddr) {
        iommu::release(phys, root);
    }
    fn io_taken(&self, phys: &Phys, root: Paddr) {
        iommu::taken(phys, root);
    }
    fn msi_first(&self) -> usize {
        ioapic::lines()
    }
    fn isolate(&self, entering: bool) {
        // SAFETY: IA32_SPEC_CTRL and IA32_PRED_CMD exist where the flags say
        // (CPUID, `init_mitigations`); the bits written are enumerated. The
        // return-stack fill calls and returns within this asm only, on the
        // kernel stack (the kernel has no red zone), and leaves it as it was.
        unsafe {
            if self.spec_ctrl.get() {
                wrmsr(0x48, self.spec_base.get() | if entering { self.spec_isolated.get() } else { 0 });
            }
            if self.ibpb.get() {
                wrmsr(0x49, 1); // IBPB: forget what earlier code taught the indirect predictors
            }
            // The return stack: 32 entries of this code's own, so a return in
            // the next address space never predicts to an address the
            // previous one left there.
            asm!(
                "mov {n:e}, 16",
                "2:",
                "call 3f",
                "int3",
                "3:",
                "call 4f",
                "int3",
                "4:",
                "dec {n:e}",
                "jnz 2b",
                "add rsp, 256",
                n = out(reg) _,
            );
        }
    }
    fn msi_release(&self, line: usize) {
        iommu::unremap(&kernel().phys, line);
    }
    fn msi_route(&self, line: usize, cpu: usize, source: u64) -> Option<(u64, u64)> {
        let apic = cpu::percpu_of(cpu)?.apic_id;
        if line >= silica_abi::IRQ_LINES as usize {
            return None;
        }
        let vector = ioapic::FIRST_VECTOR + line as u64;
        if iommu::remapping() {
            // The message names the line's remapping entry, which delivers
            // it only from `source` (segment 0: the units remap no other).
            let sid = u16::try_from(source).ok()?;
            let handle = iommu::remap(&kernel().phys, line, apic, vector as u8, false, Some(sid))? as u64;
            // Remappable format: handle bits 14..0 at 19..5, bit 15 at 2.
            return Some((0xFEE0_0000 | (handle & 0x7FFF) << 5 | 1 << 4 | (handle >> 15 & 1) << 2, 0));
        }
        // The message a device writes to raise the line: to one processor,
        // edge-triggered, at the line's vector. Without interrupt remapping
        // the processor is named by eight bits, and nothing checks who sent it.
        (apic < 0xFF).then_some((0xFEE0_0000 | (apic as u64) << 12, vector))
    }

    fn init_root(&self, phys: &Phys, root: Paddr) {
        // SAFETY: both tables are whole pages of kernel-owned memory; the
        // new root is not in use yet.
        unsafe { core::ptr::copy_nonoverlapping(phys.ptr(self.kernel_root.get() + 2048), phys.ptr(root + 2048), 2048) }
    }

    fn init_thread(&self, phys: &Phys, thread: Paddr) {
        // The legacy area of an XSAVE image: default x87 control word and
        // MXCSR (all exceptions masked). XSTATE_BV stays 0, so a restore puts
        // every component in its initial state. XRSTORS wants the compacted
        // format declared in XCOMP_BV.
        let area = thread + FPU_OFFSET as u64;
        phys.write_val(area, &0x037Fu16);
        phys.write_val(area + 24, &MXCSR_DEFAULT);
        if self.xsave_kind.get() == 2 {
            phys.write_val(area + 520, &(1u64 << 63 | self.xcr0.get()));
        }
    }

    // In line in each way out of the kernel (`syscall_rust`, which never
    // returns, saves no registers; a call here saved six again: profile.sh
    // --kvm --bench ctx put 47% of the switch in this function's own frame).
    #[inline(always)]
    fn switch_thread(&self, phys: &Phys, from: Paddr, to: Paddr, root: Paddr) {
        let kind = self.xsave_kind.get();
        let to_regs = kernel().regs(to);
        let (fs, gs) = (canonical(to_regs[REG_FS_BASE].get()), canonical(to_regs[REG_GS_BASE].get()));
        // Extended state. A thread that entered the kernel by a system call
        // has no vector or x87 data to keep: the ABI says a system call
        // clobbers them, as a function call does. Only its control words are
        // saved, and when it resumes the registers are cleared so it cannot
        // see another thread's data. A thread that was interrupted or took a
        // fault gets a full save and restore.
        //
        // SAFETY: `from` and `to` are live Thread objects (4 KiB, so the save
        // area at FPU_OFFSET is 64-byte aligned and large enough: checked
        // against CPUID at boot, as were the instructions used here). While
        // in the kernel the user's GS base is the inactive one, reached by
        // SWAPGS; interrupts are off, so nothing runs between the two swaps.
        unsafe {
            // The control words the processor holds now (the outgoing
            // thread's): a thread that resumes from a system call gets its
            // own loaded only where they differ, and they rarely do.
            let mut hw = [0u32; 2];
            asm!("stmxcsr [{p}]", "fnstcw [{p} + 4]", p = in(reg) hw.as_mut_ptr(), options(nostack));
            if from != 0 {
                let regs = kernel().regs(from);
                if regs[REG_BY_SYSCALL].get() != 0 {
                    regs[REG_FPCTL].set(hw[0] as u64 | (hw[1] as u64) << 32);
                } else {
                    let area = phys.ptr(from + FPU_OFFSET as u64);
                    match kind {
                        2 => asm!("xsaves64 [{}]", in(reg) area, in("eax") u32::MAX, in("edx") u32::MAX, options(nostack)),
                        1 => asm!("xsaveopt64 [{}]", in(reg) area, in("eax") u32::MAX, in("edx") u32::MAX, options(nostack)),
                        _ => asm!("xsave64 [{}]", in(reg) area, in("eax") u32::MAX, in("edx") u32::MAX, options(nostack)),
                    }
                }
            }
            if to_regs[REG_BY_SYSCALL].get() != 0 {
                let ctl = to_regs[REG_FPCTL].get();
                let ctl = [ctl as u32, (ctl >> 32) as u32];
                if self.xcr0.get() != 0b111 || !self.xgetbv1.get() {
                    // Other than exactly x87, SSE and AVX is enabled (more,
                    // or no AVX, without which VZEROALL is undefined), or
                    // the processor cannot say what is in use: restore the
                    // initial image.
                    self.restore(INIT_STATE.0.get().cast());
                    hw = [MXCSR_DEFAULT, FCW_DEFAULT];
                } else {
                    // VZEROALL clears every SSE and AVX register (MXCSR is
                    // kept). x87 is reset only when XINUSE says it holds
                    // anything, and FNINIT sets its control word to the default.
                    let x87: u32;
                    asm!(
                        "vzeroall", "mov ecx, 1", "xgetbv", "and eax, 1", "jz 2f", "fninit", "2:",
                        out("eax") x87, out("ecx") _, out("edx") _, options(nostack, nomem),
                    );
                    if x87 != 0 {
                        hw[1] = FCW_DEFAULT;
                    }
                }
                if ctl[0] != hw[0] {
                    asm!("ldmxcsr [{p}]", p = in(reg) ctl.as_ptr(), options(nostack));
                }
                if ctl[1] & 0xFFFF != hw[1] & 0xFFFF {
                    asm!("fldcw [{p} + 4]", p = in(reg) ctl.as_ptr(), options(nostack));
                }
            } else {
                self.restore(phys.ptr(to + FPU_OFFSET as u64));
            }
            let (old_fs, old_gs): (u64, u64);
            if self.fsgsbase.get() {
                // Written only when they change: most threads have neither.
                asm!(
                    "rdfsbase {of}", "cmp {of}, {nf}", "je 2f", "wrfsbase {nf}", "2:",
                    "swapgs", "rdgsbase {og}", "cmp {og}, {ng}", "je 3f", "wrgsbase {ng}", "3:", "swapgs",
                    of = out(reg) old_fs, nf = in(reg) fs, og = out(reg) old_gs, ng = in(reg) gs,
                    options(nostack, nomem),
                );
            } else {
                old_fs = rdmsr(MSR_FS_BASE);
                old_gs = rdmsr(MSR_KERNEL_GS_BASE);
                wrmsr(MSR_FS_BASE, fs);
                wrmsr(MSR_KERNEL_GS_BASE, gs);
            }
            if from != 0 {
                let regs = kernel().regs(from);
                regs[REG_FS_BASE].set(old_fs);
                regs[REG_GS_BASE].set(old_gs);
            }
            cpu::set_regs(to_regs.as_ptr() as u64);
            let want = if root == 0 { self.kernel_root.get() } else { root };
            if cpu::read_cr3() & PTE_ADDR != want {
                cpu::write_cr3(self.cr3_for(phys, want));
            }
        }
    }

    fn leave_aspace(&self, root: Paddr) {
        if cpu::read_cr3() & PTE_ADDR == root {
            // SAFETY: the kernel root maps the kernel exactly as `root` did.
            unsafe { cpu::write_cr3(self.cr3_for(&kernel().phys, self.kernel_root.get())) }
        }
    }

    fn tlb_flush(&self, root: Paddr, vaddr: u64, len: u64) {
        if cpu::read_cr3() & PTE_ADDR != root {
            // Not loaded. Without PCIDs nothing of it is cached; with them,
            // forget its identifier so its next load starts clean.
            self.tlb_forget(root, percpu().index);
            return;
        }
        if len > 32 * 4096 {
            // SAFETY: reloading the current root without the keep bit drops
            // its cached translations.
            unsafe { cpu::write_cr3(cpu::read_cr3() & !(1 << 63)) }
            return;
        }
        for page in (vaddr..vaddr + len).step_by(4096) {
            // SAFETY: INVLPG only drops a cached translation.
            unsafe { asm!("invlpg [{}]", in(reg) page, options(nostack, preserves_flags)) }
        }
    }

    fn now(&self) -> u64 {
        let ticks = rdtsc().wrapping_sub(self.tsc0.get());
        ((ticks as u128 * self.ns_per_tick.get() as u128) >> 32) as u64
    }

    fn set_timer(&self, deadline: u64) {
        // No later than the console must be fed again (`serial::due`).
        let deadline = deadline.min(serial::due());
        let tsc = self.tsc_at(deadline);
        let per_ns = self.apic_per_ns.get();
        // Writing the timer leaves a virtual machine; writing what it holds
        // already (above all: disarming a timer that has fired, on every
        // way into idle) is skipped. The deadline is forgotten when the
        // timer fires (`irq`), as the hardware disarms itself then.
        if per_ns == 0 && percpu().deadline.replace(tsc) == tsc {
            return;
        }
        percpu().deadline.set(tsc);
        // SAFETY: the timer MSRs exist (x2APIC and, for the first branch,
        // TSC-deadline were checked at boot). A zero count or deadline disarms.
        unsafe {
            if per_ns == 0 {
                wrmsr(MSR_TSC_DEADLINE, if tsc == u64::MAX { 0 } else { tsc });
            } else if tsc == u64::MAX {
                wrmsr(MSR_X2APIC_TIMER_COUNT, 0);
            } else {
                // One-shot: a count from now. Too long a wait fires early
                // and the scheduler arms the timer again.
                let ns = deadline.saturating_sub(self.now());
                let count = ((ns as u128 * per_ns as u128) >> 32).clamp(1, u32::MAX as u128) as u64;
                wrmsr(MSR_X2APIC_TIMER_COUNT, count);
            }
        }
    }

    fn irq_pending(&self) -> bool {
        let me = percpu();
        if rdtsc() >= me.deadline.get() {
            return true;
        }
        // Any request for a vector from 32 up in the local APIC. Reading
        // the APIC can cost a trip to the hypervisor, so look only every
        // sixteenth time: callers poll once per small step of a long loop.
        let n = me.polls.get().wrapping_add(1);
        me.polls.set(n);
        n % 16 == 0 && (1..8).any(|i| rdmsr(MSR_X2APIC_IRR0 + i) != 0)
    }

    fn idle_begin(&self) {
        // "Watching" is said here, under the lock, and never over "idle":
        // a processor that sees "idle" interrupts this one, and an
        // interrupt taken in the watch's window does not end a later halt.
        if kernel_core::sched::IDLE_WATCH_NS != 0 {
            percpu().mode.watch();
        } else {
            percpu().mode.idle();
        }
    }

    fn timer_armed(&self) -> bool {
        percpu().deadline.get() != u64::MAX
    }

    fn idle(&self, until: u64, words: Option<&[kernel_core::hal::Watched]>) -> bool {
        // An idle processor feeds the console first, and wakes no later
        // than when it must be fed again.
        report_nmis(kernel());
        serial::drain(self.now());
        let until = until.min(serial::due());
        // The timer is armed ahead of `until` by how long this processor
        // has been taking to come out of the wait (on a virtual machine,
        // mostly the host waking the virtual processor's thread), and the
        // rest is spent spinning: the deadline is met on time instead of
        // late by that latency. The estimate is a running average of how
        // late past the armed time the processor resumed, each sample
        // capped so that a rare long stall (the host core leaving a deep
        // sleep state) neither drags it up nor makes every wake spin long.
        let me = percpu();
        // First watch for work for a while (`IDLE_WATCH_NS`): a processor
        // that has some pokes this one's presence word instead of sending
        // an interrupt, which in a virtual machine saves leaving it twice
        // (the sender's interrupt and this processor's halt), and on any
        // machine the wake from a sleep state. Each turn opens a window
        // for a pending interrupt, handled on this stack as in the halt
        // below. A poke ends the watch, as does the deadline, and a device
        // interrupt (its handler leaves the word at "idle"). Any other
        // interrupt taken here carries no news: nobody sends this processor
        // one for work while it says "watching" (`idle_begin`).
        // Without `words` the watch already ran (`Hal::idle`): halt now.
        if let Some(words) = words
            && kernel_core::sched::IDLE_WATCH_NS != 0
        {
            let end = self.now().saturating_add(me.watch.length());
            loop {
                // SAFETY: STI takes effect after the next instruction, so
                // the window is the NOP; the handler returns here.
                unsafe { asm!("sti", "nop", "cli", options(nomem, nostack)) }
                if !me.mode.watching() {
                    break;
                }
                // A device wrote a watched line's news (`IrqWatch`): the
                // kernel delivers it as the interrupt, which has not come yet.
                if changed(words) {
                    me.mode.enter_kernel();
                    return true;
                }
                let now = self.now();
                if now >= until {
                    // The deadline came while watching: no timer was needed.
                    me.mode.enter_kernel();
                    return true;
                }
                if now >= end {
                    if !words.is_empty() {
                        // The kernel tells the devices to interrupt before
                        // this processor halts (`IrqQuiet`).
                        me.mode.enter_kernel();
                        return false;
                    }
                    break;
                }
                core::hint::spin_loop();
            }
            if !me.mode.stop_watching() {
                me.mode.enter_kernel();
                return true;
            }
        } else if kernel_core::sched::IDLE_WATCH_NS != 0 && !me.mode.stop_watching() {
            // Poked since `idle_begin`: there is work to look for.
            me.mode.enter_kernel();
            return true;
        }
        let early = me.wake_ns.get();
        let armed = if until == u64::MAX { u64::MAX } else { until.saturating_sub(early).max(1) };
        // Only ever earlier (`Hal::idle`): a timer armed sooner fires early
        // once and the scheduler comes back here.
        if self.tsc_at(armed) < me.deadline.get() {
            self.set_timer(armed);
        }
        let halted = self.now();
        // SAFETY: STI takes effect after the next instruction, so no
        // interrupt can arrive between the decision to wait and HLT (one
        // that is already pending ends the HLT at once). The handler runs
        // on this stack and returns here.
        unsafe { asm!("sti", "hlt", "cli", options(nomem, nostack)) }
        // Work, not the deadline, ended the halt: the watch learns how long
        // it would have had to be (`Watch`).
        // Woken before the armed time: work ended the halt (or a timer left
        // from before, armed sooner, fired), and the watch learns from it.
        // At or after it: the timer, late by the wake-up latency, which the
        // next wait arms ahead by. (Asking whether the timer "fired" instead
        // of comparing times got timer wakes wrong and cost 9 us a deadline.)
        let now = self.now();
        if now < armed && kernel_core::sched::IDLE_WATCH_NS != 0 {
            me.watch.woken_after(now - halted);
        }
        if until != u64::MAX && now >= armed {
            let late = (now - armed).min(WAKE_ADVANCE_MAX_NS);
            me.wake_ns.set((early * 7 + late) / 8);
            while self.now() < until {
                core::hint::spin_loop();
            }
        }
        me.mode.enter_kernel();
        true
    }

    fn cpu_slot(&self) -> usize {
        let s: usize;
        // SAFETY: reads this processor's slot from its `PerCpu`; it never
        // changes, so the read is pure.
        unsafe {
            asm!("mov {}, gs:[{off}]", out(reg) s, off = const core::mem::offset_of!(cpu::PerCpu, slot), options(nostack, pure, readonly, preserves_flags))
        }
        s
    }

    fn cpu(&self) -> usize {
        let i: usize;
        // SAFETY: reads this processor's index from its `PerCpu` (offset 40
        // in `cpu::PerCpu`); it never changes, so the read is pure.
        unsafe { asm!("mov {}, gs:[40]", out(reg) i, options(nostack, pure, readonly, preserves_flags)) }
        i
    }

    fn kick(&self, cpu: usize, wait: bool) {
        let Some(them) = cpu::percpu_of(cpu) else { return };
        // A processor in the kernel looks at what changed before it leaves
        // (the caller holds its lock, so it has not passed that point).
        // One that is idle and still watching is told with a store.
        if !them.online.load(Ordering::Acquire) || them.mode.in_kernel() || them.mode.poke() {
            return;
        }
        // SAFETY: the x2APIC interrupt command register: a fixed interrupt
        // to one processor by its identifier.
        unsafe { wrmsr(0x830, (them.apic_id as u64) << 32 | KICK_VECTOR) };
        // Bounded: a processor that never enters the kernel (its interrupt
        // lost, or a firmware SMI holding it) is sent the kick once more
        // after a second, then named (`kernel_core::smp::stalled`).
        let mut stall = kernel_core::smp::Stall::new();
        let mut resent = false;
        while wait && them.mode.is_user() {
            let now = self.now();
            stall.turn(now, cpu, "entered the kernel when asked");
            if !resent && stall.longer_than(now, 1_000_000_000) {
                resent = true;
                // SAFETY: as above.
                unsafe { wrmsr(0x830, (them.apic_id as u64) << 32 | KICK_VECTOR) };
            }
            core::hint::spin_loop();
        }
    }

    fn wait_left(&self, cpu: usize, request: &kernel_core::lock::Leave) {
        // It answers while it waits for its lock, which the caller holds.
        let mut stall = kernel_core::smp::Stall::new();
        while !request.answered() {
            stall.turn(self.now(), cpu, "left an address space being destroyed");
            core::hint::spin_loop();
        }
    }

    fn tlb_reset(&self) {
        // A new generation makes every address space's identifier on this
        // processor stale, so each starts clean when next loaded.
        let me = percpu();
        if self.pcid.get() {
            me.pcid_gen.set(me.pcid_gen.get() + 1);
            me.pcid_next.set(1);
        }
        // SAFETY: the kernel root maps the kernel as every root does.
        unsafe { cpu::write_cr3(self.kernel_root.get()) }
    }

    fn tlb_forget(&self, root: Paddr, cpu: usize) {
        if self.pcid.get() {
            // SAFETY: that processor's word in the header page of a live
            // address space (see `cr3_for`); the caller holds its lock.
            unsafe { *kernel().phys.ptr(root - 4096 + 2048 + cpu as u64 * 8).cast::<u64>() = 0 }
        }
    }

    fn irq_route(&self, line: usize, cpu: usize, level: bool, active_low: bool) -> bool {
        cpu::percpu_of(cpu).is_some_and(|c| ioapic::route(&kernel().phys, line, c.apic_id, level, active_low))
    }

    fn irq_mask(&self, line: usize, masked: bool) {
        ioapic::mask(&kernel().phys, line, masked);
    }

    fn irq_hold(&self, line: usize, held: bool) {
        if line < ioapic::lines() {
            // A level line is unmasked by its driver's acknowledgement only.
            if held || !ioapic::is_level(line) {
                ioapic::mask(&kernel().phys, line, held);
            }
        } else {
            // A message line: its remapping entry delivers to the discard
            // vector while held. Without remapping the device holds the
            // message and the kernel cannot hold it off.
            iommu::hold(&kernel().phys, line, held);
        }
    }

    fn irq_move(&self, line: usize, cpu: usize) -> bool {
        let Some(apic) = cpu::percpu_of(cpu).map(|c| c.apic_id) else { return false };
        // Remapped, every line (wire or message) is its remapping entry; a
        // message line without remapping names its processor in the device,
        // which only its driver writes.
        if iommu::remapping() {
            iommu::retarget(&kernel().phys, line, apic)
        } else {
            line < ioapic::lines() && ioapic::retarget(&kernel().phys, line, apic)
        }
    }

    fn port_io(&self, port: u16, width: u8, write: bool, value: u32) -> u32 {
        let mut v = value;
        // SAFETY: the caller holds a capability covering this port range;
        // port I/O does not touch memory.
        unsafe {
            match (width, write) {
                (1, true) => {
                    asm!("out dx, al", in("dx") port, in("al") v as u8, options(nomem, nostack, preserves_flags))
                }
                (2, true) => {
                    asm!("out dx, ax", in("dx") port, in("ax") v as u16, options(nomem, nostack, preserves_flags))
                }
                (_, true) => {
                    asm!("out dx, eax", in("dx") port, in("eax") v, options(nomem, nostack, preserves_flags))
                }
                (1, false) => {
                    let b: u8;
                    asm!("in al, dx", in("dx") port, out("al") b, options(nomem, nostack, preserves_flags));
                    v = b as u32;
                }
                (2, false) => {
                    let w: u16;
                    asm!("in ax, dx", in("dx") port, out("ax") w, options(nomem, nostack, preserves_flags));
                    v = w as u32;
                }
                (_, false) => {
                    asm!("in eax, dx", in("dx") port, out("eax") v, options(nomem, nostack, preserves_flags))
                }
            }
        }
        v
    }

    fn console(&self, bytes: &[u8]) {
        serial::write_whole(bytes, self.now());
        fbcon::write(bytes);
    }

    fn power_off(&self, status: u64) {
        #[cfg(feature = "harness")]
        harness::exit(status);
        let _ = status;
        let Some(p) = power() else { return };
        let ([a, b], (ta, tb)) = (p.registers.pm1_control, p.s5);
        // The ports the firmware's FADT names for this; nothing the kernel
        // holds depends on the machine staying on.
        if a == 0 {
            return;
        }
        acpi_mode(&p.registers);
        // SLP_TYP in bits 10 to 12, SLP_EN bit 13.
        let off = |port: u16, typ: u8| {
            let v = serial::inw(port);
            serial::outw(port, (v & !(7 << 10)) | (typ as u16 & 7) << 10 | 1 << 13);
        };
        if b != 0 {
            off(b, tb);
        }
        off(a, ta);
        // The machine takes a moment to go.
        let t = rdtsc();
        while rdtsc() - t < self.tsc_hz.get() {
            core::hint::spin_loop();
        }
    }

    fn reset(&self, status: u64) {
        #[cfg(feature = "harness")]
        harness::exit(status);
        let _ = status;
        if reset_register() {
            let t = rdtsc();
            while rdtsc() - t < self.tsc_hz.get() {
                core::hint::spin_loop();
            }
        }
    }
}

/// How to turn this machine off and reset it, read from its ACPI tables once
/// at boot.
#[derive(Clone, Copy)]
struct Power {
    registers: kernel_core::acpi::Power,
    /// `\_S5`'s values for PM1a and PM1b.
    s5: (u8, u8),
}

struct PowerCell(core::cell::UnsafeCell<Option<Power>>);
// SAFETY: written by the boot processor before any other runs, then read.
unsafe impl Sync for PowerCell {}
static POWER: PowerCell = PowerCell(core::cell::UnsafeCell::new(None));

fn power() -> Option<Power> {
    // SAFETY: written once at boot, before any other processor starts.
    unsafe { *POWER.0.get() }
}

/// Read the FADT and the DSDT's `\_S5` (boot processor, once).
fn init_power(bi: &BootInfo, phys: &Phys) {
    let Some(fadt) = acpi_table(bi, phys, b"FACP") else { return };
    let Some(registers) = kernel_core::acpi::power(fadt) else { return };
    let dsdt = {
        let x = if fadt.len() >= 148 { u64::from_le_bytes(fadt[140..148].try_into().unwrap_or([0; 8])) } else { 0 };
        let at = if x != 0 { x } else { u32::from_le_bytes(fadt[40..44].try_into().unwrap_or([0; 4])) as u64 };
        let limit = bi.phys_map_len;
        // SAFETY: the range is checked to lie in the physical map; ACPI
        // tables are firmware memory nothing writes.
        let header = (at != 0 && at + 36 <= limit).then(|| unsafe { phys.bytes(at, 36) });
        header.filter(|h| &h[..4] == b"DSDT").and_then(|h| {
            let len = u32::from_le_bytes(h[4..8].try_into().ok()?) as u64;
            // SAFETY: as above.
            (at + len <= limit).then(|| unsafe { phys.bytes(at, len) })
        })
    };
    let s5 = dsdt.and_then(|d| kernel_core::acpi::sleep_type(d, b"_S5_"));
    // Without \_S5's values the control registers are no use (reset may
    // still be possible).
    let registers = if s5.is_some() { registers } else { kernel_core::acpi::Power { pm1_control: [0, 0], ..registers } };
    if s5.is_none() {
        klog!(kernel(), "kernel: the firmware describes no soft-off state (\\_S5): the machine cannot be turned off");
    }
    // SAFETY: boot processor, alone.
    unsafe { *POWER.0.get() = Some(Power { registers, s5: s5.unwrap_or((0, 0)) }) };
}

/// Put the machine in ACPI mode if the firmware left it in legacy mode
/// (port I/O to the registers the FADT names).
fn acpi_mode(r: &kernel_core::acpi::Power) {
    // SCI_EN is bit 0 of PM1a's control register.
    if serial::inw(r.pm1_control[0]) & 1 == 0 && r.smi_command != 0 && r.acpi_enable != 0 {
        serial::outb(r.smi_command, r.acpi_enable);
        for _ in 0..1_000_000 {
            if serial::inw(r.pm1_control[0]) & 1 != 0 {
                break;
            }
            core::hint::spin_loop();
        }
    }
}

/// The extended state every thread starts with, for clearing the registers.
#[repr(C, align(64))]
struct InitState(core::cell::UnsafeCell<[u8; 4096 - FPU_OFFSET]>);
// SAFETY: written once during boot, then only read by XRSTOR.
unsafe impl Sync for InitState {}
static INIT_STATE: InitState = InitState(core::cell::UnsafeCell::new([0; 4096 - FPU_OFFSET]));

impl X86 {
    /// The TSC value of monotonic time `deadline` (`u64::MAX`: none), as
    /// the deadline timer takes it: 0 disarms, so tick 0 becomes tick 1.
    fn tsc_at(&self, deadline: u64) -> u64 {
        if deadline == u64::MAX {
            return u64::MAX;
        }
        let ticks = (deadline as u128 * self.ticks_per_ns.get() as u128) >> 32;
        (self.tsc0.get() as u128 + ticks).clamp(1, u64::MAX as u128 - 1) as u64
    }

    /// Load extended state from a save area.
    ///
    /// # Safety
    /// `area` must be a 64-byte-aligned XSAVE image in the format chosen at boot.
    unsafe fn restore(&self, area: *const u8) {
        // SAFETY: guaranteed by the caller.
        unsafe {
            if self.xsave_kind.get() == 2 {
                asm!("xrstors64 [{}]", in(reg) area, in("eax") u32::MAX, in("edx") u32::MAX, options(nostack));
            } else {
                asm!("xrstor64 [{}]", in(reg) area, in("eax") u32::MAX, in("edx") u32::MAX, options(nostack));
            }
        }
    }

    /// The CR3 value that loads `root`. With process-context identifiers a
    /// switch keeps the translations cached for the address space: each has
    /// an identifier per processor for that processor's current generation,
    /// kept in its header page (words owned by this module from offset
    /// 2048); when a processor's 4095 identifiers run out it starts a new
    /// generation and every address space gets a fresh one there, flushed
    /// on first use. Identifier 0 is the kernel-only root.
    fn cr3_for(&self, phys: &Phys, root: Paddr) -> u64 {
        if !self.pcid.get() {
            return root;
        }
        if root == self.kernel_root.get() {
            return root | 1 << 63;
        }
        let me = percpu();
        let generation = me.pcid_gen.get();
        // SAFETY: `root` is the root table of a live AddressSpace object,
        // whose header page precedes it; kernel-core uses only its first
        // bytes, and the second half holds one word per processor.
        let slot = unsafe { &mut *phys.ptr(root - 4096 + 2048 + me.index as u64 * 8).cast::<u64>() };
        if *slot >> 12 == generation {
            return root | (*slot & 0xFFF) | 1 << 63;
        }
        let id = me.pcid_next.get();
        if id == 4095 {
            me.pcid_next.set(1);
            me.pcid_gen.set(generation + 1);
        } else {
            me.pcid_next.set(id + 1);
        }
        *slot = generation << 12 | id;
        root | id
    }
}

/// The most an idle processor arms its timer ahead of a deadline, and so
/// the most it spins waiting for it (`X86::idle`).
const WAKE_ADVANCE_MAX_NS: u64 = 50_000;

/// MXCSR and the x87 control word as the processor's initial state has them.
const MXCSR_DEFAULT: u32 = 0x1F80;
const FCW_DEFAULT: u32 = 0x037F;

/// A user FS or GS base the processor accepts (it faults on a non-canonical one).
fn canonical(v: u64) -> u64 {
    if v < 1 << 47 || v >= !0 << 47 { v } else { 0 }
}

/// An interrupt from the local APIC or a device. `locked`: it interrupted
/// user code and this processor holds its lock; otherwise it interrupted
/// the idle wait, which looks at everything again when it has the lock.
pub fn irq(vector: u64, locked: bool) {
    let k = kernel();
    match vector {
        TIMER_VECTOR if locked => {
            percpu().deadline.set(u64::MAX);
            console_tick(k);
            k.on_timer()
        }
        TIMER_VECTOR => {
            percpu().deadline.set(u64::MAX);
            console_tick(k);
        }
        KICK_VECTOR => {}
        iommu::FAULT_VECTOR => {
            let refused = iommu::fault(&k.phys, k.hal.now());
            if refused.n != 0 {
                // Whoever attached each device hears of it, through the
                // kernel (which the idle wait has not entered).
                let space_of = |sid| iommu::space_of(&k.phys, sid);
                if locked {
                    k.device_faults(&refused.sids[..refused.n], space_of);
                } else {
                    percpu().mode.enter_kernel();
                    k.enter();
                    k.device_faults(&refused.sids[..refused.n], space_of);
                    percpu().mode.idle();
                    k.leave();
                }
            }
        }
        iommu::DISCARD_VECTOR => iommu::discarded(),
        SPURIOUS_VECTOR => return,
        v if (ioapic::FIRST_VECTOR..ioapic::FIRST_VECTOR + silica_abi::IRQ_LINES).contains(&v) => {
            let line = (v - ioapic::FIRST_VECTOR) as usize;
            // A level-triggered line stays asserted until its driver has
            // dealt with the device: hold it off until the driver says so.
            if ioapic::is_level(line) {
                ioapic::mask(&k.phys, line, true);
            }
            // SAFETY: ends the interrupt in service at the local APIC.
            unsafe { wrmsr(MSR_X2APIC_EOI, 0) };
            if locked {
                k.irq(line);
            } else {
                // Out of the idle wait, which does not hold the lock.
                percpu().mode.enter_kernel();
                k.enter();
                k.irq(line);
                percpu().mode.idle();
                k.leave();
            }
            return;
        }
        v => {
            // Device interrupts are bound to Notifications later in M2
            // (C12). Until then nothing is unmasked, so this is a stray.
            let _ = write!(Out, "kernel: unexpected interrupt vector {v} on processor {}\r\n", percpu().index);
        }
    }
    // SAFETY: writing 0 to the x2APIC EOI register ends the interrupt in service.
    unsafe { wrmsr(MSR_X2APIC_EOI, 0) }
}

struct RegionCell(core::cell::UnsafeCell<RegionTable>);
// SAFETY: used by the boot processor alone, during boot.
unsafe impl Sync for RegionCell {}
static REGIONS: RegionCell = RegionCell(core::cell::UnsafeCell::new(RegionTable::new()));

/// Drop every cached translation on this processor, global ones too.
fn flush_all() {
    // SAFETY: toggling CR4.PGE only empties the TLB.
    unsafe {
        let cr4: u64;
        asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
        asm!("mov cr4, {}", in(reg) cr4 & !(1 << 7), options(nostack, preserves_flags));
        asm!("mov cr4, {}", in(reg) cr4, options(nostack, preserves_flags));
    }
}

/// The boot processor has its per-processor data (GS is valid).
static STARTED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

struct Out;
impl Write for Out {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        serial::write(s.as_bytes());
        fbcon::write(s.as_bytes());
        record(s.as_bytes());
        Ok(())
    }
}

/// The crash record's region (`kernel_core::crash`, physical; 0: the loader
/// could not keep one), and this kernel's part in writing it.
static CRASH: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static RECORD: kernel_core::crash::Recorder = kernel_core::crash::Recorder::new();

/// The crash record's region, mapped, when there is one.
fn crash_region() -> Option<&'static mut [u8]> {
    let at = CRASH.load(Ordering::Relaxed);
    // SAFETY: the region the loader kept for the record, `REGION_BYTES`
    // long, in the physical map; nothing else maps or uses it, and only the
    // boot processor (at boot) or the one panicking processor (the others
    // stopped) touches it.
    (at != 0).then(|| unsafe { core::slice::from_raw_parts_mut(kernel().phys.ptr(at), kernel_core::crash::REGION_BYTES as usize) })
}

/// A panic begins its record: the console's last lines first.
fn record_panic() {
    RECORD.begin(crash_region(), serial::recent);
}

/// What the panic says goes into its record as well.
fn record(bytes: &[u8]) {
    RECORD.say(crash_region, bytes);
}

/// The end of a panic: its record sealed, then the machine reset, so that
/// one nobody watches comes back and its next boot says why it went. The
/// reset is the one the power capability makes (the register the firmware
/// names); on a machine whose firmware names none the kernel stops here
/// with the record kept, and the hardware watchdog the service manager
/// armed is what resets it.
fn after_panic() -> ! {
    RECORD.seal(crash_region());
    // A test image told that a harness listens stops for it to read.
    #[cfg(feature = "harness")]
    if harness::listens() {
        die();
    }
    if power().is_some_and(|p| p.registers.reset.is_some()) {
        let _ = write!(Out, "kernel: resetting the machine\r\n");
        reset_register();
    } else {
        let _ = write!(Out, "kernel: the firmware names no reset register: stopped, the crash record kept for the boot after a reset\r\n");
    }
    die()
}

/// Write the reset register the firmware's FADT names (ACPI), if it names
/// one: the machine resets a moment later.
fn reset_register() -> bool {
    let Some((port, value)) = power().and_then(|p| p.registers.reset) else { return false };
    serial::outb(port, value);
    true
}

/// At boot: the last boot's crash record, if one survived the reset, said
/// on the console whole and kept for the root task (returned), and cleared;
/// this boot's number taken.
fn last_panic(at: u64) -> &'static [u8] {
    struct Kept(core::cell::UnsafeCell<[u8; silica_abi::root::LAST_PANIC]>);
    // SAFETY: written once by the boot processor before any other starts.
    unsafe impl Sync for Kept {}
    static KEPT: Kept = Kept(core::cell::UnsafeCell::new([0; silica_abi::root::LAST_PANIC]));
    CRASH.store(at, Ordering::Relaxed);
    let Some(region) = crash_region() else {
        klog!(kernel(), "kernel: no memory kept for a crash record: a panic halts this machine");
        return &[];
    };
    // SAFETY: written once, by the boot processor, before the root task
    // reads the copy the kernel makes of it.
    let kept = unsafe { &mut *KEPT.0.get() };
    RECORD.start(region, kept, |was, text| {
        let _ = write!(Out, "kernel: the last boot (boot {was}) ended in a panic; its record:\r\n");
        serial::write(text);
        let _ = write!(Out, "\r\nkernel: end of the last boot's record\r\n");
    })
}

fn die() -> ! {
    #[cfg(feature = "harness")]
    harness::exit(kernel_core::ExitStatus::Failure.code() as u64);
    loop {
        // SAFETY: stop this processor; interrupts are off.
        unsafe { asm!("cli", "hlt", options(nomem, nostack)) }
    }
}

/// Print return addresses by following frame pointers within a kernel stack.
/// Addresses are printed without the load slide, so they match the ELF.
fn backtrace(first: u64, mut rbp: u64) {
    let slide = kernel().hal.slide.get();
    let _ = write!(Out, "#0 {:#018x}\r\n", first.wrapping_sub(slide));
    // The boot processor's stack until per-processor data exists.
    let (lo, hi) = if STARTED.load(Ordering::Relaxed) { percpu().stack } else { (cpu::KSTACK.base() + 4096, cpu::KSTACK.top()) };
    let on_stack = |a: u64| a.is_multiple_of(8) && a >= lo && a + 16 <= hi;
    for n in 1..32 {
        if !on_stack(rbp) {
            break;
        }
        // SAFETY: `rbp` was just checked to lie inside the mapped kernel stack.
        let (next, ret) = unsafe { (*(rbp as *const u64), *((rbp + 8) as *const u64)) };
        if ret == 0 {
            break;
        }
        let _ = write!(Out, "#{} {:#018x}\r\n", n, ret.wrapping_sub(slide));
        rbp = next;
    }
}

/// The kernel panicked: say why, show where, stop.
pub fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    stop_others();
    record_panic();
    fbcon::panic();
    let _ = write!(Out, "\r\nPANIC: {}\r\n", info.message());
    if let Some(l) = info.location() {
        let _ = write!(Out, "  at {}:{}\r\n", l.file(), l.line());
    }
    let _ = write!(Out, "  kernel slide {:#x}; backtrace (link addresses):\r\n", kernel().hal.slide.get());
    let (rip, rbp): (u64, u64);
    // SAFETY: reads the instruction and frame pointers.
    unsafe { asm!("lea {}, [rip]", "mov {}, rbp", out(reg) rip, out(reg) rbp, options(nomem, nostack)) }
    backtrace(rip, rbp);
    after_panic()
}

/// A processor exception in kernel mode: a kernel bug.
pub fn kernel_fault(f: &cpu::KernelFrame) -> ! {
    stop_others();
    record_panic();
    let cr2: u64;
    // SAFETY: reading CR2 in ring 0 has no side effect.
    unsafe { asm!("mov {}, cr2", out(reg) cr2, options(nomem, nostack, preserves_flags)) }
    fbcon::panic();
    let _ = write!(
        Out,
        "\r\nPANIC: kernel exception {} (error {:#x}) at {:#x}, address {:#x}, rsp {:#x}\r\n  kernel slide {:#x}; backtrace (link addresses):\r\n",
        f.vector,
        f.error,
        f.rip,
        cr2,
        f.rsp,
        kernel().hal.slide.get()
    );
    backtrace(f.rip, f.gpr[6]);
    after_panic()
}

fn require(ok: bool, what: &str) {
    if !ok {
        panic!("this processor lacks {what}, which Silica requires");
    }
}

/// Control registers, extended state, the local APIC, mitigations: on
/// every processor. The boot processor (`first`) also records what the
/// machine has, for all of them.
pub(super) fn init_processor(first: bool) {
    let k = kernel();
    let f1 = cpuid(1, 0);
    let f7 = cpuid(7, 0);
    // x86-64-v3, NX, x2APIC, XSAVE and a constant TSC (`kernel_core::x86`),
    // on every processor: the loader checked the boot processor already.
    let v = kernel_core::x86::check(cpuid);
    if let Some(m) = v.missing {
        require(false, m);
    }
    if first && v.tsc_trusted {
        klog!(k, "kernel: the hypervisor reports no invariant TSC; the clock trusts it to count at a constant rate");
    }
    // SAFETY: boot-time processor setup with interrupts off. Each bit set
    // is one the processor reports supporting.
    unsafe {
        let mut cr0: u64;
        asm!("mov {}, cr0", out(reg) cr0, options(nomem, nostack, preserves_flags));
        cr0 = (cr0 & !(1 << 2 | 1 << 3)) | 1 << 1 | 1 << 5 | 1 << 16; // no EM/TS; MP, NE, WP
        asm!("mov cr0, {}", in(reg) cr0, options(nomem, nostack, preserves_flags));
        // The PAT as reset, but for entry 1 (PWT alone): write-combining,
        // which the loader gives the framebuffer (`page_framebuffer`). No
        // other mapping uses that entry, so nothing else changes type;
        // global translations of the framebuffer cached as write-through
        // go with the PGE toggle below.
        wrmsr(0x277, 0x0007_0406_0007_0106);
        let mut cr4: u64;
        asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
        asm!("mov cr4, {}", in(reg) cr4 & !(1 << 7), options(nomem, nostack, preserves_flags));
        cr4 |= 1 << 7 | 1 << 9 | 1 << 10 | 1 << 18; // PGE, OSFXSR, OSXMMEXCPT, OSXSAVE
        if f7[1] & 1 != 0 {
            cr4 |= 1 << 16; // FSGSBASE
        }
        if f1[2] & 1 << 17 != 0 {
            cr4 |= 1 << 17; // PCIDE (CR3 holds identifier 0 here, as required)
            percpu().pcid_gen.set(1);
        }
        if first {
            k.hal.fsgsbase.set(f7[1] & 1 != 0);
            k.hal.pcid.set(f1[2] & 1 << 17 != 0);
        }
        if f7[1] & 1 << 7 != 0 {
            cr4 |= 1 << 20; // SMEP
        }
        if f7[1] & 1 << 20 != 0 {
            cr4 |= 1 << 21; // SMAP: the kernel never touches user addresses
        }
        if f7[2] & 1 << 2 != 0 {
            cr4 |= 1 << 11; // UMIP
        }
        // Machine checks are taken (CR4.MCE) and every bank reports
        // (`init_mca`): without this a memory error shuts the processor
        // down with nothing said.
        let mca = f1[3] & (1 << 7 | 1 << 14) == 1 << 7 | 1 << 14;
        if mca {
            cr4 |= 1 << 6;
        }
        asm!("mov cr4, {}", in(reg) cr4, options(nomem, nostack, preserves_flags));
        if mca {
            init_mca(first);
        }
        // x87, SSE, AVX, and AVX-512 when its state fits the Thread object.
        let supported = cpuid(0xD, 0)[0] as u64;
        let mut xcr0 = supported & 0b111;
        require(xcr0 & 0b111 == 0b111, "SSE and AVX state in XSAVE");
        if supported & 0xE0 == 0xE0 {
            xcr0 |= 0xE0;
        }
        loop {
            asm!("xsetbv", in("ecx") 0u32, in("eax") xcr0 as u32, in("edx") 0u32, options(nomem, nostack, preserves_flags));
            if cpuid(0xD, 0)[1] as usize <= 4096 - FPU_OFFSET {
                break;
            }
            require(xcr0 & 0xE0 != 0, "an extended state small enough for a Thread object");
            xcr0 &= !0xE0;
        }
        // The cheapest save the processor has: XSAVES skips components that
        // are in their initial state or unchanged since they were restored.
        let xs = cpuid(0xD, 1)[0];
        let kind = if xs & 1 << 3 != 0 {
            wrmsr(0xDA0, 0); // IA32_XSS: no supervisor components
            2
        } else {
            (xs & 1) as u8
        };
        if first {
            k.hal.xcr0.set(xcr0);
            k.hal.xsave_kind.set(kind);
            k.hal.xgetbv1.set(xs & 1 << 2 != 0);
            // The initial image: default control words, nothing in use.
            let init = INIT_STATE.0.get().cast::<u8>();
            init.cast::<u16>().write(0x037F);
            init.add(24).cast::<u32>().write(MXCSR_DEFAULT);
            if kind == 2 {
                init.add(520).cast::<u64>().write(1 << 63 | xcr0);
            }
        }
        init_mitigations(first);
        // Legacy interrupt controllers off: remap out of the exception
        // vectors, then mask every line.
        for (port, v) in
            [(0x20, 0x11), (0xA0, 0x11), (0x21, 0xF0), (0xA1, 0xF8), (0x21, 4), (0xA1, 2), (0x21, 1), (0xA1, 1), (0x21, 0xFF), (0xA1, 0xFF)]
        {
            serial::outb(port, v);
        }
        wrmsr(MSR_APIC_BASE, rdmsr(MSR_APIC_BASE) | 1 << 11 | 1 << 10);
        wrmsr(MSR_X2APIC_SVR, 0x100 | SPURIOUS_VECTOR);
        wrmsr(0x808, 0); // task priority: accept everything
        wrmsr(0x835, 0x1_0000); // LINT0 masked
        wrmsr(0x836, 0x400); // LINT1: NMI
        wrmsr(0x837, 0x1_0000); // error LVT masked
        if f1[2] & 1 << 24 != 0 {
            wrmsr(MSR_X2APIC_LVT_TIMER, TIMER_VECTOR | 0b10 << 17); // TSC-deadline mode
            wrmsr(MSR_TSC_DEADLINE, 0);
        } else {
            wrmsr(MSR_X2APIC_TIMER_DIVIDE, 0b1011); // divide by 1
            // One-shot; masked on the boot processor until it has calibrated.
            wrmsr(MSR_X2APIC_LVT_TIMER, if first { 1 << 16 } else { TIMER_VECTOR });
        }
    }
}

/// Speculative-execution mitigations, chosen from what this processor
/// reports (component C18). What Silica's structure makes unnecessary: the
/// kernel never reads through a user-supplied pointer, so there is nothing
/// to sanitize for Spectre v1 beyond the `lfence` after `swapgs`.
///
/// # Safety
/// Called once during processor setup.
unsafe fn init_mitigations(first: bool) {
    let k = kernel();
    let f7 = cpuid(7, 0);
    let caps = if f7[3] & 1 << 29 != 0 { rdmsr(0x10A) } else { 0 };
    let intel = cpuid(0, 0)[1] == 0x756E_6547;
    let mut spec = 0u64;
    let mut done = 0u8;
    // Enhanced IBRS: indirect branches in the kernel are never steered by
    // user-mode training. Set once and left on.
    if f7[3] & 1 << 26 != 0 && caps & 1 << 1 != 0 {
        spec |= 1;
        done |= 1;
    }
    // Branch history injection: stop user-mode history from steering kernel branches.
    if f7[0] >= 2 && cpuid(7, 2)[3] & 1 << 4 != 0 {
        spec |= 1 << 10;
        done |= 2;
    }
    if spec != 0 {
        // SAFETY: IA32_SPEC_CTRL exists (CPUID.7:EDX[26]); the bits set are enumerated.
        unsafe { wrmsr(0x48, spec) };
    }
    // What isolating an address space can use (`X86::isolate`): IA32_SPEC_CTRL
    // and IBPB (Intel: CPUID.7:EDX 26; AMD: 0x80000008 EBX 14 and 12), STIBP
    // (EDX 27; EBX 15) and SSBD (EDX 31; EBX 24).
    let amd = if cpuid(0x8000_0000, 0)[0] >= 0x8000_0008 { cpuid(0x8000_0008, 0)[1] } else { 0 };
    let has = |intel_bit: u32, amd_bit: u32| f7[3] & 1 << intel_bit != 0 || amd & 1 << amd_bit != 0;
    if first {
        k.hal.spec_ctrl.set(has(26, 14) || has(27, 15) || has(31, 24));
        k.hal.ibpb.set(has(26, 12));
        k.hal.spec_base.set(spec);
        k.hal.spec_isolated.set(if has(27, 15) { 1 << 1 } else { 0 } | if has(31, 24) { 1 << 2 } else { 0 });
    }
    // Buffers that can leak across the user/kernel boundary (MDS, RFDS):
    // overwrite them on the way back to user mode.
    if caps & 1 << 28 != 0 || (intel && f7[3] & 1 << 10 != 0 && caps & 1 << 5 == 0) {
        cpu::enable_verw();
        done |= 4;
    }
    // What cannot be mitigated by this kernel as built is said out loud.
    if done & 1 == 0 {
        done |= 8;
    }
    if intel && caps & 1 == 0 {
        done |= 16;
    }
    if first {
        k.hal.mitigations.set(done);
    }
}

/// Without a TSC-deadline timer: measure the local APIC timer against the
/// timestamp counter and switch it to one-shot mode.
fn init_oneshot_timer(tsc_hz: u64) {
    if cpuid(1, 0)[2] & 1 << 24 != 0 {
        return;
    }
    // The median of three samples of 2 ms: one disturbed sample does not
    // become every timer's error.
    let mut samples = [0u64; 3];
    for s in &mut samples {
        // SAFETY: x2APIC timer registers, present on every x2APIC.
        *s = unsafe {
            wrmsr(MSR_X2APIC_TIMER_COUNT, u32::MAX as u64);
            let t0 = rdtsc();
            while rdtsc() - t0 < tsc_hz / 500 {}
            let left = rdmsr(MSR_X2APIC_TIMER_NOW);
            wrmsr(MSR_X2APIC_TIMER_COUNT, 0);
            u32::MAX as u64 - left
        };
    }
    samples.sort_unstable();
    let counted = samples[1];
    // SAFETY: as above: one-shot, unmasked.
    unsafe { wrmsr(MSR_X2APIC_LVT_TIMER, TIMER_VECTOR) };
    // `counted` ticks in 2 ms.
    let per_ns = ((counted as u128 * 500) << 32) / 1_000_000_000;
    kernel().hal.apic_per_ns.set((per_ns as u64).max(1));
}

/// The frequency of the timestamp counter.
fn tsc_frequency(bi: &BootInfo, phys: &Phys) -> u64 {
    // 1. The processor says (Intel: crystal frequency and ratio).
    if cpuid(0, 0)[0] >= 0x15 {
        let [den, num, hz, _] = cpuid(0x15, 0);
        if den != 0 && num != 0 && hz != 0 {
            return hz as u64 * num as u64 / den as u64;
        }
    }
    // 2. The hypervisor says (KVM paravirtual clock: exact scale factors).
    let hv = cpuid(0x4000_0000, 0);
    if cpuid(1, 0)[2] & 1 << 31 != 0 && hv[1] == 0x4B4D_564B && hv[2] == 0x564B_4D56 && cpuid(0x4000_0001, 0)[0] & 1 << 3 != 0 {
        #[repr(C, align(64))]
        struct PvClock(core::cell::UnsafeCell<[u32; 16]>);
        // SAFETY: written by the hypervisor only between the two MSR writes below.
        unsafe impl Sync for PvClock {}
        static PV: PvClock = PvClock(core::cell::UnsafeCell::new([0; 16]));
        let pa = PV.0.get() as u64 - bi.kernel_virt + bi.kernel_phys;
        // SAFETY: MSR_KVM_SYSTEM_TIME_NEW exists (feature bit checked); the
        // structure is 32 bytes inside one page of the kernel image.
        let (mul, shift) = unsafe {
            wrmsr(0x4B56_4D01, pa | 1);
            let p = PV.0.get().cast::<u8>();
            let v = (core::ptr::read_volatile(p.add(24).cast::<u32>()), core::ptr::read_volatile(p.add(28).cast::<i8>()));
            wrmsr(0x4B56_4D01, 0);
            v
        };
        if mul != 0 {
            // ns = ((ticks << shift) * mul) >> 32
            let hz = (1_000_000_000u128 << 32) / mul as u128;
            return if shift >= 0 { (hz >> shift) as u64 } else { (hz << -shift) as u64 };
        }
    }
    // 3. Measure against the ACPI power-management timer (3.579545 MHz):
    // the median of three 10 ms samples, each bounded (a timer that does
    // not count is not waited for for ever: the next way is tried).
    if let Some(port) = acpi_pm_timer(bi, phys)
        && let Some(hz) = pm_timer_hz(port)
    {
        return hz;
    }
    // 4. The processor's nominal frequency.
    if cpuid(0, 0)[0] >= 0x16 && cpuid(0x16, 0)[0] != 0 {
        return cpuid(0x16, 0)[0] as u64 * 1_000_000;
    }
    panic!("no way to measure the timestamp counter: no CPUID leaf 0x15, no KVM clock, no ACPI PM timer")
}

/// The ACPI table with signature `sig`, whole. `None` when the firmware has
/// none or its tables lie outside the memory the kernel can reach.
pub(super) fn acpi_table<'a>(bi: &BootInfo, phys: &'a Phys, sig: &[u8; 4]) -> Option<&'a [u8]> {
    let limit = bi.phys_map_len;
    let inside = |p: u64, len: u64| p != 0 && p.checked_add(len).is_some_and(|e| e <= limit);
    // The shared walker (`kernel_core::acpi::find_table`, the one the
    // tests check), reading only inside the physical map.
    let mut read = |p: u64, buf: &mut [u8]| {
        if !inside(p, buf.len() as u64) {
            return false;
        }
        // SAFETY: the range is checked to be inside the physical map; ACPI
        // tables are firmware memory nothing writes.
        buf.copy_from_slice(unsafe { phys.bytes(p, buf.len() as u64) });
        true
    };
    let (t, len) = kernel_core::acpi::find_table(&mut read, bi.rsdp, sig)?;
    // SAFETY: as above, the whole table checked.
    inside(t, len as u64).then(|| unsafe { phys.bytes(t, len as u64) })
}

/// The I/O port of the ACPI PM timer, from the FADT.
fn acpi_pm_timer(bi: &BootInfo, phys: &Phys) -> Option<u16> {
    let fadt = acpi_table(bi, phys, b"FACP")?;
    let port = u32::from_le_bytes(fadt.get(76..80)?.try_into().ok()?);
    (port != 0 && port < 0xFFFC).then_some(port as u16)
}

/// The entry of the kernel's own page table that maps `vaddr` at `level`
/// (0: a 4 KiB page, 1: a 2 MiB page or the table for it).
pub(super) fn kernel_pte(phys: &Phys, vaddr: u64, level: u64) -> Option<*mut u64> {
    let mut table = cpu::read_cr3() & PTE_ADDR;
    for l in (0..4).rev() {
        let e = phys.ptr(table + (vaddr >> (12 + 9 * l) & 511) * 8).cast::<u64>();
        if l == level {
            return Some(e);
        }
        // SAFETY: `table` is a page-table page of the tables the loader
        // built, reached through the physical map.
        let v = unsafe { *e };
        if v & PTE_P == 0 || v & PTE_PS != 0 {
            return None;
        }
        table = v & PTE_ADDR;
    }
    None
}

/// Kernel entry from the boot stub, on the kernel stack, interrupts off.
pub extern "C" fn kmain(info: u64) -> ! {
    let entered = rdtsc();
    serial::init();
    let k = kernel();
    // SAFETY: the loader passes the physical address of a BootInfo in
    // loader memory, which nothing overwrites before the root task runs.
    let bi: BootInfo = unsafe { core::ptr::read(k.phys.ptr(info).cast::<BootInfo>()) };
    if bi.magic != BOOT_MAGIC || bi.version != BOOT_VERSION {
        serial::write(b"ERROR: kernel: the loader's boot information has the wrong magic or version\r\n");
        die();
    }
    k.hal.slide.set(bi.kernel_slide);
    // SAFETY: first thing after the console, once, on the boot stack, which
    // with the boot tables lives for ever.
    unsafe {
        cpu::init_idt();
        let (base, top, ist) = cpu::boot_stacks();
        cpu::init(cpu::boot_tables(), 0, 0, (base, top), ist);
    }
    STARTED.store(true, Ordering::Release);
    init_processor(true);
    let apic_id = rdmsr(0x802) as u32;
    // SAFETY: the boot processor's own tables; no other processor runs yet.
    unsafe { (*cpu::boot_tables()).percpu.apic_id = apic_id };
    percpu().online.store(true, Ordering::Release);
    k.enter();

    // The kernel's own address space: drop the loader's low alias of
    // physical memory, make the physical map non-executable, and unmap the
    // page below the stack so an overflow faults instead of corrupting data.
    let root = cpu::read_cr3() & PTE_ADDR;
    k.hal.kernel_root.set(root);
    let guard = kernel_pte(&k.phys, cpu::KSTACK.base(), 0);
    // SAFETY: edits the live kernel page table in ways that keep every
    // address the kernel uses mapped: it runs from the image mapping and
    // reaches memory through the physical map, neither of which changes
    // except for losing execute permission on the latter.
    unsafe {
        let pml4 = k.phys.ptr(root).cast::<u64>();
        *pml4 = 0;
        *pml4.add(256) |= PTE_NX;
        if let Some(g) = guard {
            *g = 0;
        }
    }
    flush_all();

    // SAFETY: loader memory, read-only during boot (as read below).
    let firmware_regions = unsafe { core::slice::from_raw_parts(k.phys.ptr(bi.regions).cast::<Region>(), bi.region_count as usize) };
    fbcon::init(&bi, &k.phys, firmware_regions);
    if !serial::present() {
        klog!(
            k,
            "kernel: no serial port: the kernel's lines are on the screen{}",
            if fbcon::present() { " until the system starts, and a panic always" } else { ", and there is no screen either" }
        );
    }
    ioapic::init(&bi, &k.phys);
    init_power(&bi, &k.phys);
    let hz = tsc_frequency(&bi, &k.phys);
    k.hal.tsc_hz.set(hz);
    k.hal.tsc0.set(rdtsc());
    k.hal.ns_per_tick.set(((1_000_000_000u128 << 32) / hz as u128) as u64);
    k.hal.ticks_per_ns.set((((hz as u128) << 32) / 1_000_000_000) as u64);
    init_oneshot_timer(hz);

    // SAFETY: loader memory, read-only until the root task owns it; the
    // region table is the boot processor's alone during boot.
    let (table, cmdline, ramdisk) = unsafe {
        let table = &mut *REGIONS.0.get();
        let from = core::slice::from_raw_parts(k.phys.ptr(bi.regions).cast::<Region>(), bi.region_count as usize);
        table.fill(from);
        (table, k.phys.bytes(bi.cmdline, bi.cmdline_len), k.phys.bytes(bi.ramdisk, bi.ramdisk_len))
    };
    #[cfg(feature = "harness")]
    harness::words(cmdline);
    let last_panic = last_panic(bi.crash);
    // Physical page 0 is never an object (address 0 means "none" in the
    // kernel), so it is unmapped in the physical map: a kernel bug that
    // reaches an object through a null address faults at once, as on
    // aarch64, instead of writing into page 0. The map's first 2 MiB block
    // becomes a table of pages with the block's attributes (bits 7 and 12
    // mean different things in the two kinds of entry, so only the common
    // ones are copied).
    // Where the loader mapped that 2 MiB in pages already (it mixes memory
    // and device ranges), page 0's entry is cleared in place.
    if let Some(pde) = kernel_pte(&k.phys, PHYS_MAP_BASE, 1) {
        // SAFETY: `pde` is the live entry of the physical map's first 2 MiB;
        // a table it names is the loader's, which the kernel now owns; no
        // other processor runs yet, and nothing touches physical page 0.
        unsafe {
            let v = *pde;
            if v & PTE_P != 0 && v & PTE_PS == 0 {
                *k.phys.ptr(v & PTE_ADDR).cast::<u64>() = 0;
                flush_all();
            }
        }
    }
    if let (Some(pt), Some(pde)) = (table.carve(4096, 4096, 1 << 20, 1 << 32), kernel_pte(&k.phys, PHYS_MAP_BASE, 1)) {
        // SAFETY: `pde` is the live entry of the physical map's first 2 MiB
        // and `pt` a page nobody else has; no other processor runs yet, and
        // the kernel touches nothing in physical page 0.
        unsafe {
            let v = *pde;
            if v & PTE_P != 0 && v & PTE_PS != 0 {
                let attrs = v & (0x17F | PTE_NX) & !PTE_PS;
                let base = v & PTE_ADDR & !0x1F_FFFF;
                let t = k.phys.ptr(pt).cast::<u64>();
                *t = 0;
                for i in 1..512u64 {
                    *t.add(i as usize) = (base + i * 4096) | attrs;
                }
                *pde = pt | PTE_P | PTE_W;
                flush_all();
            }
        }
    }

    // The other processors: memory for each, taken before the root task is
    // given the rest. `cpus=1` on the command line keeps them asleep.
    let cpus = match Cmdline(cmdline).get("cpus") {
        Some(b"1") => 1,
        _ => smp::prepare(&bi, &k.phys, table, root, apic_id),
    };
    k.set_cpu_count(cpus);
    // They come up while the IOMMU is set up, and are counted (those that
    // arrived) before the root task is told how many there are.
    let smp_from = rdtsc();
    if cpus > 1 {
        smp::start(hz);
    }
    // The I/O APIC and local APIC stay with the kernel, and so do the
    // IOMMUs' registers: from here on a device reaches only what a driver
    // maps for it.
    let mut reserved = [(0u64, 0u64); 17];
    reserved[0] = (0xFEC0_0000, 0xFEF0_0000);
    let t_iommu = rdtsc();
    if Cmdline(cmdline).get("iommu") != Some(b"off") {
        iommu::init(&bi, &k.phys, table, &mut reserved);
    }
    let t_iommu_done = rdtsc();
    let described = cpus;
    let cpus = if cpus > 1 { smp::wait(cpus, hz) } else { 1 };
    let smp_done = rdtsc();
    if cpus < described {
        klog!(k, "kernel: {cpus} of the {described} processors the firmware describes started; the others are not used");
    }
    k.set_cpu_count(cpus);
    let regions = table.as_slice();
    let usable: u64 = regions.iter().filter(|r| r.kind == 1).map(|r| r.len).sum();
    klog!(k, "silica: {} MiB free, {} processors, clock {} Hz, kernel slide {:#x}", usable >> 20, cpus, hz, bi.kernel_slide);
    if iommu::top() == 0 {
        klog!(k, "kernel: no IOMMU in use: a device that does DMA reaches all of memory, whatever its driver holds");
    }
    let m = k.hal.mitigations.get();
    if m & 8 != 0 {
        klog!(
            k,
            "WARNING: kernel: this processor has no enhanced IBRS; kernel indirect branches are not protected from user-mode training"
        );
    }
    if m & 16 != 0 {
        // Every program could read all of memory: the kernel and the physical
        // map stay mapped in every address space (no kernel page-table
        // isolation yet). Not run unless the owner says so, knowing it.
        if Cmdline(cmdline).get("meltdown") != Some(b"accept") {
            klog!(
                k,
                "ERROR: kernel: this processor is affected by Meltdown and Silica does not isolate kernel page tables: any program could read all of memory. Not started; `meltdown=accept` on the kernel command line runs it anyway"
            );
            die();
        }
        klog!(
            k,
            "WARNING: kernel: this processor is affected by Meltdown and Silica does not isolate kernel page tables; the owner accepted that any program can read all of memory (meltdown=accept)"
        );
    }
    match kernel_core::boot::entropy_verdict(bi.entropy_sources, Cmdline(cmdline).get("entropy")) {
        Err(why) => {
            klog!(k, "{why}");
            die();
        }
        Ok(Some(warning)) => klog!(k, "{warning}"),
        Ok(None) => {}
    }
    // A test image's deliberate panic, once: a boot that follows a panic
    // (its record is here) goes on, so its root task can be seen to be
    // given the record.
    #[cfg(feature = "harness")]
    if last_panic.is_empty() {
        harness::deliberate(cmdline, k);
    }

    // Nothing in the archive runs, the root task included, unless the whole
    // of it is as the image builder wrote it (`ramdisk::check`).
    if let Err(why) = kernel_core::ramdisk::check(ramdisk, crc32c) {
        klog!(k, "ERROR: kernel: no root task: the boot archive is refused: {why}");
        die();
    }
    let Some(image) = kernel_core::ramdisk::find(ramdisk, "root") else {
        klog!(k, "ERROR: kernel: no root task: the boot archive has no entry named \"root\"");
        die();
    };
    // Device memory is handed out as far as the processor's addresses go:
    // firmware puts 64-bit device registers above all memory.
    let width = if cpuid(0x8000_0000, 0)[0] >= 0x8000_0008 { (cpuid(0x8000_0008, 0)[0] & 0xFF) as u64 } else { 36 };
    let top = regions.last().map_or(0, |r| r.base + r.len).max(1 << width.clamp(32, 46));
    let spec = RootSpec {
        regions,
        elf: image,
        machine: kernel_core::elf::EM_X86_64,
        reserved: &reserved,
        device_top: top,
        ports: true,
        ramdisk: [bi.ramdisk, bi.ramdisk_len],
        rsdp: bi.rsdp,
        framebuffer: [bi.fb_base, bi.fb_width as u64, bi.fb_height as u64, bi.fb_stride as u64, bi.fb_format as u64],
        clock: [hz, bi.loader_tsc, k.hal.tsc0.get()],
        wall_clock: [bi.wall_secs, ((bi.wall_tsc.wrapping_sub(k.hal.tsc0.get()) as i64 as i128 * 1_000_000_000) / hz as i128) as u64],
        entropy: bi.entropy,
        entropy_sources: bi.entropy_sources,
        cmdline,
        cpus: cpus as u64,
        last_panic,
    };
    if let Err(e) = k.boot_root(&spec) {
        klog!(k, "ERROR: kernel: cannot start the root task: {e:?}");
        die();
    }
    // Where the boot's time went, by this processor's counter: the loader
    // (from its start to the kernel's: mostly the firmware's own exit, which
    // the loader's line breaks down), the kernel (of which the IOMMU's
    // set-up), and the other processors, from waking them to the last one
    // counted (they come up while the IOMMU is set up).
    let us = |a: u64, b: u64| b.wrapping_sub(a) / (hz / 1_000_000).max(1);
    let now = rdtsc();
    klog!(
        k,
        "kernel: boot: loader {} us, kernel {} us (the IOMMU {} us), {} processors started in {} us",
        us(bi.loader_tsc, entered),
        us(entered, now),
        us(t_iommu, t_iommu_done),
        cpus,
        us(smp_from, smp_done)
    );
    // The screen is the display driver's from here.
    fbcon::release();
    // SAFETY: the tables, the processor and the root thread are set up and
    // the kernel is entered; this is the first exit to user mode.
    unsafe { cpu::exit_to_user() }
}

/// Whether a watched word (`Hal::idle`) no longer holds what it held at the
/// line's last delivery.
#[inline(always)]
fn changed(words: &[kernel_core::hal::Watched]) -> bool {
    words.iter().any(|w| {
        // SAFETY: `at` is in the kernel's window on physical memory, in a
        // frame of RAM (not device memory: `IrqWatch` refuses those) that was
        // live when the scheduler gathered it under this processor's lock.
        // If it is destroyed meanwhile the processor is kicked out of this
        // wait; a read before that sees reused memory, which is only a reason
        // for the kernel to look again under the lock (`poll_watches`).
        let v = unsafe { w.at.read_volatile() };
        v & w.mask != w.seen
    })
}

/// Continue a CRC32C over `data` with the processor's instruction (SSE4.2,
/// which `kernel_core::x86` requires): eight bytes an instruction, a boot
/// archive of a few megabytes in a fraction of a millisecond.
fn crc32c(state: u32, data: &[u8]) -> u32 {
    let (words, tail) = data.as_chunks::<8>();
    let mut c = state as u64;
    for w in words {
        // SAFETY: CRC32 only computes on registers.
        unsafe { asm!("crc32 {c}, {v}", c = inout(reg) c, v = in(reg) u64::from_le_bytes(*w), options(pure, nomem, nostack)) }
    }
    let mut c = c as u32;
    for &b in tail {
        // SAFETY: as above.
        unsafe { asm!("crc32 {c:e}, {b}", c = inout(reg) c, b = in(reg_byte) b, options(pure, nomem, nostack)) }
    }
    c
}

/// The timer fired: feed the console what waits, and while something
/// still does, keep the timer armed for it (the scheduler arms it only for
/// its own deadlines, and `set_timer` takes the console's into account).
fn console_tick(k: &kernel_core::Kernel<X86>) {
    report_nmis(k);
    let now = k.hal.now();
    serial::drain(now);
    if serial::due() != u64::MAX {
        k.hal.set_timer(u64::MAX);
    }
}

// ---- Machine checks, NMIs, and stopping the machine -----------------------

const MSR_MCG_CAP: u32 = 0x179;
const MSR_MCG_STATUS: u32 = 0x17A;
const MSR_MCG_CTL: u32 = 0x17B;
/// Bank `i`'s control, status and address registers.
const fn mc_ctl(i: u32) -> u32 {
    0x400 + 4 * i
}
const fn mc_status(i: u32) -> u32 {
    0x401 + 4 * i
}
const fn mc_addr(i: u32) -> u32 {
    0x402 + 4 * i
}
/// MCi_STATUS: valid, uncorrected, processor context corrupt, address valid.
const MCI_VAL: u64 = 1 << 63;
const MCI_UC: u64 = 1 << 61;
const MCI_PCC: u64 = 1 << 57;
const MCI_ADDRV: u64 = 1 << 58;
/// MCG_STATUS: the interrupted instruction can be restarted.
const MCG_RIPV: u64 = 1;

/// Every machine-check bank reports. On the boot processor, errors left
/// from before (a previous boot's, the firmware's) are said and cleared.
fn init_mca(first: bool) {
    let cap = rdmsr(MSR_MCG_CAP);
    let banks = (cap & 0xFF) as u32;
    if cap & 1 << 8 != 0 {
        mc_write(MSR_MCG_CTL, u64::MAX);
    }
    for i in 0..banks {
        let status = rdmsr(mc_status(i));
        if first && status & MCI_VAL != 0 {
            klog!(kernel(), "kernel: machine-check bank {i} holds an error from before this boot: status {status:#x}");
        }
        mc_write(mc_ctl(i), u64::MAX);
        mc_write(mc_status(i), 0);
    }
}

/// The worst error the banks hold now: (any valid, any uncorrected, any
/// that corrupted the processor's context), with the first uncorrected
/// one's status and address. The banks are cleared.
fn take_machine_check() -> (bool, bool, bool, u64, u64) {
    let banks = (rdmsr(MSR_MCG_CAP) & 0xFF) as u32;
    let (mut any, mut uc, mut pcc, mut status, mut addr) = (false, false, false, 0, 0);
    for i in 0..banks {
        let s = rdmsr(mc_status(i));
        if s & MCI_VAL == 0 {
            continue;
        }
        any = true;
        pcc |= s & MCI_PCC != 0;
        if s & MCI_UC != 0 && !uc {
            uc = true;
            status = s;
            addr = if s & MCI_ADDRV != 0 { rdmsr(mc_addr(i)) } else { 0 };
        }
        mc_write(mc_status(i), 0);
    }
    (any, uc, pcc, status, addr)
}

/// A machine check while a program ran. An error the processor corrected
/// is said; one that corrupted the processor's context stops the machine;
/// any other uncorrected error was consumed by the program (its memory, or
/// what it was computing with), and is its fault: reported to its fault
/// endpoint like any exception (vector 18, the bank's status), as a page
/// fault would be, and the machine goes on.
pub fn machine_check_user(k: &kernel_core::Kernel<X86>) {
    let mcg = rdmsr(MSR_MCG_STATUS);
    let (any, uc, pcc, status, addr) = take_machine_check();
    mc_write(MSR_MCG_STATUS, 0);
    if pcc || (uc && mcg & MCG_RIPV == 0) {
        panic!("machine check: the processor's state is lost (bank status {status:#x}, address {addr:#x})");
    }
    if uc {
        k.fault(silica_abi::Fault::Exception, 18, status);
    } else if any {
        klog!(k, "kernel: the processor corrected a machine-check error");
    }
}

/// A machine check in the kernel: an error the processor corrected is
/// said; any other stops the machine (the kernel's own state may be what
/// was lost).
pub fn machine_check_kernel() {
    let (any, uc, _, status, addr) = take_machine_check();
    mc_write(MSR_MCG_STATUS, 0);
    if uc {
        panic!("machine check in the kernel (bank status {status:#x}, address {addr:#x})");
    }
    if any {
        NMI_PENDING.fetch_or(2, Ordering::Relaxed);
    }
}

/// The machine is stopping (a panic): every other processor halts on the
/// NMI it is sent.
static PANICKING: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
/// NMIs taken and not yet said, and corrected machine checks in the kernel
/// (bit 1), said from a safe place (`report_nmis`): an NMI may arrive while
/// its processor holds any lock, so it takes none.
static NMI_COUNT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static NMI_REASON: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static NMI_PENDING: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// An NMI, from user mode or the kernel. While the machine stops it is
/// the request to halt; otherwise it is a platform's report (a parity or
/// channel error, a watchdog), counted with its reason and said later.
/// Takes no lock and needs no per-processor data.
pub fn nmi() {
    if PANICKING.load(Ordering::Acquire) {
        loop {
            // SAFETY: stopping this processor for good.
            unsafe { asm!("cli", "hlt", options(nomem, nostack)) }
        }
    }
    // The legacy NMI status: bit 7 memory parity (SERR#), bit 6 I/O channel check.
    let reason = serial::inb(0x61) & 0xC0;
    NMI_REASON.fetch_or(reason as u64, Ordering::Relaxed);
    NMI_COUNT.fetch_add(1, Ordering::Relaxed);
    NMI_PENDING.fetch_or(1, Ordering::Relaxed);
}

/// Say the NMIs and corrected machine checks taken since the last time.
fn report_nmis(k: &kernel_core::Kernel<X86>) {
    if NMI_PENDING.load(Ordering::Relaxed) == 0 {
        return;
    }
    let pending = NMI_PENDING.swap(0, Ordering::Relaxed);
    if pending & 1 != 0 {
        let (n, r) = (NMI_COUNT.swap(0, Ordering::Relaxed), NMI_REASON.swap(0, Ordering::Relaxed));
        klog!(
            k,
            "kernel: {n} NMI(s) from the platform{}{}",
            if r & 0x80 != 0 { "; a memory parity error was reported (SERR#)" } else { "" },
            if r & 0x40 != 0 { "; an I/O channel check was reported" } else { "" }
        );
    }
    if pending & 2 != 0 {
        klog!(k, "kernel: the processor corrected a machine-check error in the kernel");
    }
}

/// Stop every other processor before this one says why the machine stops:
/// none goes on in a kernel whose state is no longer known. An NMI reaches
/// them whatever they are doing; each halts (`nmi`).
fn stop_others() {
    if PANICKING.swap(true, Ordering::AcqRel) || !STARTED.load(Ordering::Acquire) {
        return;
    }
    // SAFETY: the x2APIC interrupt command register: an NMI to every
    // processor but this one (destination shorthand 0b11, delivery 0b100).
    unsafe { wrmsr(0x830, 0xC_0400) };
}

/// Write a machine-check register.
fn mc_write(msr: u32, v: u64) {
    // SAFETY: only the machine-check MSRs, which exist (CPUID's MCE and MCA
    // were checked before any is used): writing them configures reporting
    // or clears a recorded error, and touches no memory.
    unsafe { wrmsr(msr, v) }
}

/// Whether physical page `pa` is reached through the physical map uncached
/// (PCD and PWT on its entry, at whatever level the walk ends): the loader
/// maps device registers so (`map_physical`), and the kernel relies on it
/// for every register it reaches through the map.
pub(super) fn mapped_uncached(phys: &Phys, pa: u64) -> bool {
    let va = PHYS_MAP_BASE + pa;
    let mut table = cpu::read_cr3() & PTE_ADDR;
    for l in (0..4).rev() {
        // SAFETY: page-table pages of the kernel's own tables, reached
        // through the physical map.
        let e = unsafe { *phys.ptr(table + (va >> (12 + 9 * l) & 511) * 8).cast::<u64>() };
        if e & PTE_P == 0 {
            return false;
        }
        if l == 0 || e & PTE_PS != 0 {
            return e & (1 << 4 | 1 << 3) == 1 << 4 | 1 << 3;
        }
        table = e & PTE_ADDR;
    }
    false
}

/// Say so, as an error, if the device registers at `pa` (`what`) would be
/// reached cached.
pub(super) fn check_uncached(phys: &Phys, pa: u64, what: &str) {
    if !mapped_uncached(phys, pa) {
        klog!(kernel(), "ERROR: kernel: {what} at {pa:#x} is mapped cacheable: its registers would be reached through the cache");
    }
}

/// The timestamp counter's frequency measured against the ACPI PM timer at
/// `port`: the median of three samples of 10 ms (35,795 of its ticks), so
/// one disturbed sample (a firmware interrupt, a host preempting a virtual
/// processor) does not become the clock's error for the machine's life.
/// `None` when the timer does not count: a sample that has taken 2^33 TSC
/// ticks (seconds at any frequency) is given up.
fn pm_timer_hz(port: u16) -> Option<u64> {
    let read = || kernel().hal.port_io(port, 4, false, 0) & 0xFF_FFFF;
    let mut samples = [0u64; 3];
    for s in &mut samples {
        let start = read();
        let t0 = rdtsc();
        while read().wrapping_sub(start) & 0xFF_FFFF < 35_795 {
            if rdtsc().wrapping_sub(t0) > 1 << 33 {
                klog!(kernel(), "kernel: the ACPI PM timer does not count; the timestamp counter is not measured against it");
                return None;
            }
        }
        *s = (rdtsc() - t0) * 100;
    }
    samples.sort_unstable();
    Some(samples[1])
}
