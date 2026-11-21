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
//! Known gap (spec/COMPONENTS.md B08): these stacks are in the physical
//! map, with no guard page below them.

use super::{PHYS_MAP_BASE, acpi_table, acpi8, acpi32, acpi64, cpu, gic, image_phys, kernel};
use core::arch::{asm, global_asm};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use kernel_core::boot::RegionTable;
use kernel_core::smp::MAX_CPUS;

const PAGE: u64 = 4096;
const STACK: u64 = 64 << 10;
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

fn psci(f: u64, a: u64, b: u64, c: u64) -> i64 {
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
    let Some((t, len)) = acpi_table(rsdp, b"APIC") else { return 0 };
    let me = gic::my_affinity();
    let (mut n, mut at) = (0, t + 44);
    while at + 2 <= t + len && n < MAX_CPUS - 1 {
        let (kind, l) = (acpi8(at)[0], acpi8(at)[1] as u64);
        if l < 2 {
            break;
        }
        if kind == 0x0B && l >= 76 && acpi32(at + 12) & 1 != 0 {
            let m = gic::affinity(acpi64(at + 68));
            if m != me {
                out[n] = m;
                n += 1;
            }
        }
        at += l;
    }
    n
}

/// Find the other processors and take memory for each, before the root task
/// is given the rest. Returns how many processors there will be, this one
/// included; `launch` starts them.
pub fn prepare(rsdp: u64, regions: &mut RegionTable, gicr: (u64, u64)) -> usize {
    // PSCI: the FADT's Arm boot flags say whether it exists and its conduit.
    let Some((fadt, flen)) = acpi_table(rsdp, b"FACP") else { return 1 };
    let flags = if flen >= 131 { acpi8(fadt + 129)[0] } else { 0 };
    if flags & 1 == 0 {
        let _ = core::fmt::Write::write_str(&mut super::Out, "kernel: the firmware has no PSCI: one processor\r\n");
        return 1;
    }
    HVC.store(flags & 2 != 0, Ordering::Relaxed);
    let mut ids = [0u64; MAX_CPUS];
    let found = others(rsdp, &mut ids);
    if found == 0 {
        return 1;
    }
    let Some(base) = regions.carve(EACH * found as u64, PAGE, 0, u64::MAX) else { return 1 };
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
                stack_top: PHYS_MAP_BASE + stack + STACK,
                entry: ap_main as usize as u64,
                index: index as u64,
            });
            ((page_va + 256) as *mut cpu::PerCpu).write(cpu::PerCpu::new(index, PHYS_MAP_BASE + stack + STACK, mpidr));
        }
        clean(page_va, 256);
        PAGES[index].store(page_va + 256, Ordering::Release);
    }
    found + 1
}

/// Start the processors `prepare` set up, one at a time, each within a
/// second; say which did not start.
pub fn launch(cpus: usize) {
    let entry_pa = image_phys(&raw const ap_entry as u64);
    for index in 1..cpus {
        let p = PAGES[index].load(Ordering::Acquire);
        // SAFETY: `prepare` wrote this processor's PerCpu there.
        let mpidr = unsafe { &*(p as *const cpu::PerCpu) }.mpidr.load(Ordering::Relaxed);
        let r = psci(0xC400_0003, mpidr, entry_pa, p - 256 - PHYS_MAP_BASE);
        if r != 0 {
            let _ = core::fmt::Write::write_fmt(&mut super::Out, format_args!("kernel: PSCI CPU_ON {mpidr:#x} failed: {r}\r\n"));
            continue;
        }
        let (t0, freq) = (super::counter(), super::counter_frequency());
        while cpu::percpu_of(index).is_none_or(|c| !c.online.load(Ordering::Acquire)) {
            if super::counter().wrapping_sub(t0) > freq {
                let _ = core::fmt::Write::write_fmt(&mut super::Out, format_args!("kernel: processor {mpidr:#x} did not start\r\n"));
                break;
            }
            core::hint::spin_loop();
        }
    }
}

static GICR_REGION: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

extern "C" fn ap_main(index: usize) -> ! {
    // SAFETY: the page was filled for this processor before it was started.
    let p = unsafe { &*(PAGES[index].load(Ordering::Acquire) as *const cpu::PerCpu) };
    // SAFETY: this processor's own data, on its first and only pass here,
    // interrupts masked.
    unsafe { cpu::install(p) };
    super::load_ttbr0(kernel().hal.empty_root());
    let region = (GICR_REGION[0].load(Ordering::Relaxed), GICR_REGION[1].load(Ordering::Relaxed));
    match gic::find_redistributor(region.0, region.1, gic::my_affinity()) {
        Some(r) => gic::init_processor(index, r),
        None => {
            let _ = core::fmt::Write::write_str(&mut super::Out, "kernel: no GIC redistributor for a processor; it stops\r\n");
            loop {
                // SAFETY: waiting with interrupts masked.
                unsafe { asm!("wfe", options(nomem, nostack)) }
            }
        }
    }
    p.online.store(true, Ordering::Release);
    let k = kernel();
    k.enter();
    // SAFETY: vectors, per-processor data and the kernel lock are set up.
    unsafe { cpu::exit_to_user() }
}
