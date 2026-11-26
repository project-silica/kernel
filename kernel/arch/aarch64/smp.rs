//! The other processors: found in the ACPI MADT, started with PSCI CPU_ON.
//!
//! A processor starts at `ap_entry` (its physical address) with its MMU and
//! caches off, at EL2 or EL1, with x0 the physical address of its context.
//! It drops to EL1 if need be, installs the boot processor's MAIR, TCR and
//! TTBR1 and the loader's identity root in TTBR0 (so turning the MMU on
//! does not move the code under it), and jumps to `ap_main` in the kernel
//! half on its own stack. Everything it reads before its MMU is on is
//! cleaned to the point of coherency first.
//!
//! Each processor uses a kernel stack alias with an unmapped page below it.

use super::{PHYS_MAP_BASE, acpi_table, cpu, gic, image_phys, kernel, madt_entries, t32, t64};
use core::arch::{asm, global_asm};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use kernel_core::boot::RegionTable;
use kernel_core::smp::MAX_CPUS;

const PAGE: u64 = 4096;
const STACK: u64 = super::STACK_SIZE as u64;
/// Per processor: its stack, then a page for its PerCpu and start context.
const EACH: u64 = STACK + PAGE;

/// What `ap_entry` reads, at the start of the processor's own page.
#[repr(C)]
struct Context {
    mair: u64,
    tcr: u64,
    ttbr0: u64,
    ttbr1: u64,
    stack_top: u64,
    entry: u64,
    index: u64,
}

/// Each started processor's page (physical), by index.
static PAGES: [AtomicU64; MAX_CPUS] = [const { AtomicU64::new(0) }; MAX_CPUS];
/// PSCI through HVC (else SMC).
static HVC: AtomicBool = AtomicBool::new(true);
/// The firmware has PSCI.
static PSCI: AtomicBool = AtomicBool::new(false);
/// The loader's identity root, used once by each processor to turn its MMU on.
pub static LOW_ROOT: AtomicU64 = AtomicU64::new(0);

global_asm!(
    r#"
.section .text, "ax"
.balign 4096
.global ap_entry
ap_entry:
    msr daifset, #0xf
    mov x19, x0
    mrs x9, CurrentEL
    cmp x9, #8
    b.ne 2f
    mov x10, #(1 << 31)
    msr hcr_el2, x10
    mrs x10, cnthctl_el2
    orr x10, x10, #3
    msr cnthctl_el2, x10
    msr cntvoff_el2, xzr
    mov x10, #0x33ff
    msr cptr_el2, x10
    msr hstr_el2, xzr
    ldr x10, =0x30d00800
    msr sctlr_el1, x10
    mov x10, #0x3c5
    msr spsr_el2, x10
    adr x10, 2f
    msr elr_el2, x10
    isb
    eret
2:
    mov x10, #(3 << 20)
    msr cpacr_el1, x10
    ldp x1, x2, [x19, #0]
    ldp x3, x4, [x19, #16]
    msr mair_el1, x1
    msr tcr_el1, x2
    msr ttbr0_el1, x3
    msr ttbr1_el1, x4
    isb
    tlbi vmalle1
    dsb nsh
    isb
    ldr x10, =0x30d01805
    msr sctlr_el1, x10
    isb
    ldp x5, x6, [x19, #32]
    ldr x0, [x19, #48]
    mov sp, x5
    mov x29, #0
    mov x30, #0
    br x6
.ltorg
.global ap_entry_end
ap_entry_end:
"#
);

unsafe extern "C" {
    static ap_entry: u8;
    static ap_entry_end: u8;
}

/// Clean `[va, va + len)` to the point of coherency, for a processor that
/// reads it with its caches off.
fn clean(va: u64, len: u64) {
    let mut a = va & !63;
    while a < va + len {
        // SAFETY: cleaning cache lines of memory the kernel owns.
        unsafe { asm!("dc cvac, {}", in(reg) a, options(nostack, preserves_flags)) }
        a += 64;
    }
    // SAFETY: a barrier.
    unsafe { asm!("dsb sy", options(nostack, preserves_flags)) }
}

/// Turn the machine off through PSCI SYSTEM_OFF. Returns only when the
/// firmware has no PSCI (or refuses).
pub fn system_off() {
    if PSCI.load(Ordering::Relaxed) {
        psci(0x8400_0008, 0, 0, 0);
    }
}

/// PSCI SYSTEM_RESET; returns only if the firmware has no PSCI.
pub fn system_reset() {
    if PSCI.load(Ordering::Relaxed) {
        psci(0x8400_0009, 0, 0, 0);
    }
}

pub(super) fn psci(f: u64, a: u64, b: u64, c: u64) -> i64 {
    let r: u64;
    // SAFETY: a PSCI call to the firmware; CPU_ON does not touch this
    // processor's state beyond the result.
    unsafe {
        if HVC.load(Ordering::Relaxed) {
            asm!("hvc #0", inout("x0") f => r, inout("x1") a => _, inout("x2") b => _, inout("x3") c => _, options(nostack))
        } else {
            asm!("smc #0", inout("x0") f => r, inout("x1") a => _, inout("x2") b => _, inout("x3") c => _, options(nostack))
        }
    }
    r as i64
}

/// The other enabled processors' MPIDR affinities, from the MADT.
fn others(rsdp: u64, out: &mut [u64; MAX_CPUS]) -> usize {
    let Some(t) = acpi_table(rsdp, b"APIC") else { return 0 };
    let me = gic::my_affinity();
    let mut n = 0;
    for (kind, e) in madt_entries(t) {
        if n == MAX_CPUS - 1 {
            break;
        }
        // A GICC entry, enabled (flags bit 0): one online-capable but not
        // enabled (a hot-plug slot) is not there.
        if kind == 0x0B
            && e.len() >= 76
            && t32(e, 12).is_some_and(|f| f & 1 != 0)
            && let Some(m) = t64(e, 68).map(gic::affinity)
            && m != me
        {
            out[n] = m;
            n += 1;
        }
    }
    n
}

/// Whether the firmware has PSCI (and SMCCC with it), from the FADT's Arm
/// boot flags, which also say its conduit (HVC or SMC). Once, at boot.
pub fn detect_psci(rsdp: u64) -> bool {
    let Some(fadt) = acpi_table(rsdp, b"FACP") else { return false };
    let flags = fadt.get(129).copied().unwrap_or(0);
    if flags & 1 == 0 {
        return false;
    }
    HVC.store(flags & 2 != 0, Ordering::Relaxed);
    PSCI.store(true, Ordering::Relaxed);
    true
}

/// Whether `detect_psci` found PSCI.
pub fn psci_present() -> bool {
    PSCI.load(Ordering::Relaxed)
}

/// Find the other processors and take memory for each, before the root task
/// is given the rest. Returns how many processors there will be, this one
/// included; `launch` starts them.
pub fn prepare(rsdp: u64, regions: &mut RegionTable, gicr: (u64, u64)) -> usize {
    if !detect_psci(rsdp) {
        let _ = core::fmt::Write::write_str(&mut super::Out, "kernel: the firmware has no PSCI: one processor\r\n");
        return 1;
    }
    let mut ids = [0u64; MAX_CPUS];
    let found = others(rsdp, &mut ids);
    if found == 0 {
        return 1;
    }
    // Use the second GiB of the device window for normal-memory stack
    // aliases. Each 17-page virtual extent starts with an absent PTE.
    const WINDOW: u64 = super::DEVICE_BASE + (1 << 30);
    const _: () = assert!(EACH * MAX_CPUS as u64 <= 1 << 30);
    let bytes = EACH * found as u64;
    let leaf_tables = bytes.div_ceil(super::BLOCK);
    let Some(base) = regions.carve(bytes + PAGE * (1 + leaf_tables), PAGE, 0, u64::MAX) else {
        panic!("no contiguous memory for {found} secondary kernel stacks");
    };
    let l2 = base + bytes;
    let phys = &kernel().phys;
    phys.zero(l2, PAGE * (1 + leaf_tables));
    for i in 0..leaf_tables {
        phys.write_val(l2 + i * 8, &((l2 + PAGE * (1 + i)) | 3));
    }
    for i in 0..found as u64 {
        for offset in (0..STACK).step_by(PAGE as usize) {
            let index = (i * EACH + PAGE + offset) / PAGE;
            // Normal write-back, inner-shareable, EL1 read/write, no execute.
            let entry = (base + i * EACH + offset) | 3 | (3 << 8) | super::AF | super::PXN | super::UXN;
            phys.write_val(l2 + PAGE + index * 8, &entry);
        }
    }
    // SAFETY: boot processor only. Slot zero maps the GIC, slot one was
    // absent. Publish the completed subtree before any secondary starts.
    unsafe {
        (*super::DEVICE_L1.0.get())[1] = l2 | 3;
        asm!("dsb ishst", "isb", options(nostack));
    }
    GICR_REGION[0].store(gicr.0, Ordering::Relaxed);
    GICR_REGION[1].store(gicr.1, Ordering::Relaxed);
    let (mair, tcr, ttbr1): (u64, u64, u64);
    // SAFETY: reading this processor's translation registers.
    unsafe {
        asm!("mrs {}, mair_el1", "mrs {}, tcr_el1", "mrs {}, ttbr1_el1", out(reg) mair, out(reg) tcr, out(reg) ttbr1, options(nomem, nostack));
    }
    // SAFETY: both symbols label the trampoline in the kernel image.
    let entry_len = unsafe { (&raw const ap_entry_end).offset_from(&raw const ap_entry) } as u64;
    clean(&raw const ap_entry as u64, entry_len);
    for (i, &mpidr) in ids[..found].iter().enumerate() {
        let index = i + 1;
        let stack = base + i as u64 * EACH;
        let page = stack + STACK;
        let stack_top = WINDOW + (i as u64 + 1) * EACH;
        let page_va = PHYS_MAP_BASE + page;
        // SAFETY: carved memory, the kernel's alone; the PerCpu sits after
        // the context in the same page and lives for ever.
        unsafe {
            core::ptr::write_bytes(page_va as *mut u8, 0, PAGE as usize);
            (page_va as *mut Context).write(Context {
                mair,
                tcr,
                ttbr0: LOW_ROOT.load(Ordering::Relaxed),
                ttbr1,
                stack_top,
                entry: ap_main as usize as u64,
                index: index as u64,
            });
            ((page_va + 256) as *mut cpu::PerCpu).write(cpu::PerCpu::new(index, stack_top, mpidr));
        }
        clean(page_va, 256);
        PAGES[index].store(page_va + 256, Ordering::Release);
    }
    found + 1
}

/// The secondaries that arrived (`kernel_core::smp::Admission`).
static ADMISSION: kernel_core::smp::Admission = kernel_core::smp::Admission::new();

/// Start the processors `prepare` set up in slots `1..slots`, all at once,
/// and count those that arrive: the number of processors the machine has,
/// the boot one included, which is what the root task is told. Waits while
/// they keep arriving (a second with none arriving ends it); a processor
/// the firmware refused to start, or one that never arrived, is said and
/// not counted.
pub fn launch(slots: usize) -> usize {
    let entry_pa = image_phys(&raw const ap_entry as u64);
    let mut woken = 0;
    for slot in 1..slots {
        let p = PAGES[slot].load(Ordering::Acquire);
        // SAFETY: `prepare` wrote this processor's PerCpu there.
        let mpidr = unsafe { &*(p as *const cpu::PerCpu) }.mpidr.load(Ordering::Relaxed);
        let r = psci(0xC400_0003, mpidr, entry_pa, p - 256 - PHYS_MAP_BASE);
        if r != 0 {
            let _ = core::fmt::Write::write_fmt(&mut super::Out, format_args!("kernel: PSCI CPU_ON {mpidr:#x} failed: {r}\r\n"));
            continue;
        }
        woken += 1;
    }
    let freq = super::counter_frequency();
    let mut last = (ADMISSION.arrived(), super::counter());
    while ADMISSION.arrived() < woken {
        let now = (ADMISSION.arrived(), super::counter());
        if now.0 != last.0 {
            last = now;
        } else if now.1.wrapping_sub(last.1) > freq {
            break;
        }
        core::hint::spin_loop();
    }
    let cpus = ADMISSION.close();
    if cpus < slots {
        let _ = core::fmt::Write::write_fmt(
            &mut super::Out,
            format_args!("kernel: {cpus} of the {slots} processors the firmware describes started; the others are not used\r\n"),
        );
    }
    cpus
}

static GICR_REGION: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
/// The redistributor region's physical address.
pub static GICR_PHYS_BASE: AtomicU64 = AtomicU64::new(0);

/// A secondary, in slot `slot` (its page, stack and data), arrives: it
/// takes the next index, or stops for good if the boot processor has
/// already closed the count (said by `launch`).
extern "C" fn ap_main(slot: usize) -> ! {
    let stop = |why: &str| -> ! {
        let _ = core::fmt::Write::write_fmt(
            &mut super::Out,
            format_args!("kernel: a processor ({:#x}) {why}; it stops and is not used\r\n", gic::my_affinity()),
        );
        loop {
            // SAFETY: stopping this processor, interrupts masked, touching nothing.
            unsafe { asm!("wfe", options(nomem, nostack)) }
        }
    };
    // Its interrupt controller first: a processor that cannot take
    // interrupts is not admitted, so it is never counted.
    let region = (GICR_REGION[0].load(Ordering::Relaxed), GICR_REGION[1].load(Ordering::Relaxed));
    let Some(r) = gic::find_redistributor(region.0, region.1, gic::my_affinity()) else { stop("has no GIC redistributor") };
    if !gic::wake(r) {
        stop("has a GIC redistributor that did not wake within 100 ms");
    }
    let Some(index) = ADMISSION.admit() else {
        loop {
            // SAFETY: stopping this processor, interrupts masked, touching nothing.
            unsafe { asm!("wfe", options(nomem, nostack)) }
        }
    };
    let page = PAGES[slot].load(Ordering::Acquire) as *mut cpu::PerCpu;
    // SAFETY: the page was filled for this processor before it was started
    // and is this processor's alone; it takes the index it was admitted
    // with before anything reads its data.
    let p = unsafe {
        let (stack_top, mpidr) = ((*page).kstack_top.load(Ordering::Relaxed), (*page).mpidr.load(Ordering::Relaxed));
        page.write(cpu::PerCpu::new(index, stack_top, mpidr));
        &*page
    };
    // SAFETY: this processor's own data, on its first and only pass here,
    // interrupts masked.
    unsafe { cpu::install(p) };
    super::verify_stack(p.kstack_top.load(Ordering::Relaxed));
    #[cfg(feature = "harness")]
    if index == 1 && super::harness::SECONDARY.load(Ordering::Acquire) {
        super::harness::deliberate_panic(index);
    }
    super::reset_ttbr0(kernel().hal.empty_root());
    gic::init_processor(index, r);
    super::its::enable_processor(index, r, GICR_PHYS_BASE.load(Ordering::Relaxed) + (r - region.0));
    p.online.store(true, Ordering::Release);
    let k = kernel();
    k.enter();
    // SAFETY: vectors, per-processor data and the kernel lock are set up.
    unsafe { cpu::exit_to_user() }
}
