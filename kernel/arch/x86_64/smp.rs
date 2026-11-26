//! Finding and starting the other processors.
//!
//! The boot processor reads the processor list from the ACPI MADT, carves
//! each one a kernel stack, an interrupt stack and a table page out of free
//! memory (with unmapped guard pages below both stacks), and starts them all
//! at once with a broadcast INIT and STARTUP. An application processor
//! begins in real mode in a page below 1 MiB (`TRAMPOLINE`), gets to long
//! mode on a throw-away page table, jumps to `ap_entry` in the kernel,
//! finds its stack by its APIC identifier and joins the scheduler. The boot
//! processor closes admission before giving userspace its processor count.

use super::cpu::{self, Tables, cpuid, rdtsc, wrmsr};
use super::{PTE_ADDR, PTE_P, PTE_W};
use core::arch::global_asm;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use kernel_core::boot::BootInfo;
use kernel_core::boot::RegionTable;
use kernel_core::mem::Phys;
use kernel_core::smp::MAX_CPUS;

const PAGE: u64 = 4096;
const CHUNK: u64 = 2 << 20;
/// Pages per processor: guard, kernel stack, guard, interrupt stack, tables.
const KSTACK_PAGES: u64 = 7;
const ISTACK_PAGES: u64 = 2;
const SLOT_PAGES: u64 = 1 + KSTACK_PAGES + 1 + ISTACK_PAGES + 1;

/// What `ap_entry` looks up by APIC identifier. Field offsets are used by the assembly.
#[repr(C)]
#[derive(Clone, Copy)]
struct ApSlot {
    apic_id: u64,
    stack_top: u64,
    index: u64,
    stack_base: u64,
    ist_top: u64,
    tables: u64,
}

struct Slots(UnsafeCell<[ApSlot; MAX_CPUS]>);
// SAFETY: written by the boot processor before it starts the others, which only read.
unsafe impl Sync for Slots {}
static SLOTS: Slots =
    Slots(UnsafeCell::new([ApSlot { apic_id: 0, stack_top: 0, index: 0, stack_base: 0, ist_top: 0, tables: 0 }; MAX_CPUS]));
static SLOT_COUNT: AtomicU64 = AtomicU64::new(0);
static KERNEL_CR3: AtomicU64 = AtomicU64::new(0);

// Real mode to long mode, position-independent: it finds its own address
// from CS. Written out as machine code because it mixes 16-, 32- and 64-bit
// code and must be copied below 1 MiB.
//
//   00  cli; cld; mov ax, cs; mov ds, ax
//   06  xor ebx, ebx; mov bx, cs; shl ebx, 4        ebx = address of this page
//   0f  lgdt [0xa0]
//   15  mov eax, cr0; or eax, 1; mov cr0, eax         protected mode
//   1f  jmp far [0xa8]                                to 0x24, 32-bit code
//   24  mov ax, 0x10; mov ds, ax; mov ss, ax; mov es, ax
//   2e  mov eax, cr4; or eax, 0x20; mov cr4, eax      PAE
//   37  mov eax, [ebx + 0xb8]; mov cr3, eax           the temporary page table
//   40  mov ecx, 0xc0000080; rdmsr; or eax, 0x900; wrmsr   long mode, NX
//   4e  mov eax, cr0; or eax, 0x80010000; mov cr0, eax     paging
//   59  jmp far [ebx + 0xb0]                          to 0x5f, 64-bit code
//   5f  mov ebx, ebx; mov rax, [rbx + 0xc0]; jmp rax  into the kernel
//   80  GDT: null, 32-bit code, data, 64-bit code
//   a0  GDT pointer   a8  far pointer to 0x24   b0  far pointer to 0x5f
//   b8  page table (32 bits)   c0  kernel entry (64 bits)
const TRAMPOLINE: [u8; 0x6A] = [
    0xFA, 0xFC, 0x8C, 0xC8, 0x8E, 0xD8, 0x66, 0x31, 0xDB, 0x8C, 0xCB, 0x66, 0xC1, 0xE3, 0x04, 0x66, 0x0F, 0x01, 0x16, 0xA0, 0x00, 0x0F,
    0x20, 0xC0, 0x66, 0x83, 0xC8, 0x01, 0x0F, 0x22, 0xC0, 0x66, 0xFF, 0x2E, 0xA8, 0x00, 0x66, 0xB8, 0x10, 0x00, 0x8E, 0xD8, 0x8E, 0xD0,
    0x8E, 0xC0, 0x0F, 0x20, 0xE0, 0x83, 0xC8, 0x20, 0x0F, 0x22, 0xE0, 0x8B, 0x83, 0xB8, 0x00, 0x00, 0x00, 0x0F, 0x22, 0xD8, 0xB9, 0x80,
    0x00, 0x00, 0xC0, 0x0F, 0x32, 0x0D, 0x00, 0x09, 0x00, 0x00, 0x0F, 0x30, 0x0F, 0x20, 0xC0, 0x0D, 0x00, 0x00, 0x01, 0x80, 0x0F, 0x22,
    0xC0, 0xFF, 0xAB, 0xB0, 0x00, 0x00, 0x00, 0x89, 0xDB, 0x48, 0x8B, 0x83, 0xC0, 0x00, 0x00, 0x00, 0xFF, 0xE0,
];

global_asm!(
    r#"
.section .text, "ax"
.global ap_entry
ap_entry:
    mov eax, 0xB
    xor ecx, ecx
    cpuid
    mov edx, edx
    lea rsi, [rip + {slots}]
    mov rcx, [rip + {count}]
2:
    test rcx, rcx
    jz 4f
    cmp [rsi], rdx
    je 3f
    add rsi, 48
    dec rcx
    jmp 2b
3:
    mov rsp, [rsi + 8]
    mov rdi, [rsi + 16]
    mov rax, [rip + {cr3}]
    mov cr3, rax
    xor ebp, ebp
    call {main}
4:
    cli
    hlt
    jmp 4b
"#,
    slots = sym SLOTS,
    count = sym SLOT_COUNT,
    cr3 = sym KERNEL_CR3,
    main = sym ap_main,
);

unsafe extern "C" {
    static ap_entry: u8;
}

/// The secondaries that arrived (`kernel_core::smp::Admission`).
static ADMISSION: kernel_core::smp::Admission = kernel_core::smp::Admission::new();

/// `panic=park` (tests): processor 1 parks holding its lock (`ap_main`).
pub static PARK: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// A secondary, in its slot `slot` (its stacks and tables), arrives: it
/// takes the next index, or stops for good if the boot processor has
/// already closed the count (said by `wait`).
extern "C" fn ap_main(slot: usize) -> ! {
    // A processor whose counter disagrees is not admitted (said by `wait`).
    if !tsc_agrees() {
        TSC_REJECTED.fetch_add(1, Ordering::Relaxed);
        loop {
            // SAFETY: stopping this processor, touching nothing.
            unsafe { core::arch::asm!("cli", "hlt", options(nomem, nostack)) }
        }
    }
    let Some(index) = ADMISSION.admit() else {
        loop {
            // SAFETY: stopping this processor, touching nothing.
            unsafe { core::arch::asm!("cli", "hlt", options(nomem, nostack)) }
        }
    };
    // SAFETY: the slot was filled for this processor before it was started.
    let slot = unsafe { (*SLOTS.0.get())[slot - 1] };
    // SAFETY: this processor's own tables and stacks, on its first and only pass here.
    unsafe { cpu::init(slot.tables as *mut Tables, index, slot.apic_id as u32, (slot.stack_base, slot.stack_top), slot.ist_top) };
    super::init_processor(false);
    cpu::percpu().online.store(true, Ordering::Release);
    let k = super::kernel();
    k.enter();
    // `panic=park` (tests): processor 1 keeps its lock with interrupts off
    // for good, as a processor hung in hardware or held by firmware would;
    // the next entry that takes every lock must stop the machine naming it
    // (`kernel_core::smp::stalled`), not wait for ever.
    if index == 1 && PARK.load(Ordering::Relaxed) {
        loop {
            core::hint::spin_loop();
        }
    }
    // SAFETY: tables, processor state and the kernel lock are set up.
    unsafe { cpu::exit_to_user() }
}

/// The APIC identifiers of the enabled processors, from the ACPI MADT.
fn madt_processors(bi: &BootInfo, phys: &Phys, out: &mut [u32; MAX_CPUS]) -> usize {
    let Some(madt) = super::acpi_table(bi, phys, b"APIC") else { return 0 };
    let (mut n, mut o) = (0, 44);
    while o + 2 <= madt.len() && n < MAX_CPUS {
        let (ty, len) = (madt[o], madt[o + 1] as usize);
        if len < 2 || o + len > madt.len() {
            break;
        }
        let e = &madt[o..o + len];
        let u32_at = |i: usize| u32::from_le_bytes([e[i], e[i + 1], e[i + 2], e[i + 3]]);
        // Type 0: local APIC (8-bit identifier). Type 9: local x2APIC.
        // Flags bit 0: enabled; bit 1: can be enabled later.
        match ty {
            // Enabled processors only: one the firmware lists as online
            // capable but not enabled (a hot-plug slot) is not there.
            0 if len >= 8 && u32_at(4) & 1 != 0 => {
                out[n] = e[3] as u32;
                n += 1;
            }
            9 if len >= 16 && u32_at(8) & 1 != 0 => {
                out[n] = u32_at(4);
                n += 1;
            }
            _ => {}
        }
        o += len;
    }
    n
}

/// Find the processors and give each its memory. Returns how many there
/// are, the boot processor included. Boot processor only, before `start`.
pub fn prepare(bi: &BootInfo, phys: &Phys, regions: &mut RegionTable, kernel_root: u64, boot_apic_id: u32) -> usize {
    let mut ids = [0u32; MAX_CPUS];
    let found = madt_processors(bi, phys, &mut ids);
    let others = ids[..found].iter().filter(|&&id| id != boot_apic_id).count().min(MAX_CPUS - 1);
    if others == 0 {
        return 1;
    }
    // The start-up page below 1 MiB and three pages of temporary page table
    // below 4 GiB (a processor leaving real mode can load only 32 bits into CR3).
    let (Some(low), Some(tables)) = (regions.carve(PAGE, PAGE, 0x1000, 1 << 20), regions.carve(3 * PAGE, PAGE, 1 << 20, 1 << 32)) else {
        return 1;
    };
    let per_chunk = (CHUNK / PAGE - 1) / SLOT_PAGES;
    let mut made = 0usize;
    let mut chunk = 0u64;
    // SAFETY: boot processor, before any other runs; every address written
    // is memory just carved out of the free regions, reached through the
    // physical map; the page-table edits keep everything else mapped.
    unsafe {
        let slots = &mut *SLOTS.0.get();
        for &id in ids[..found].iter().filter(|&&id| id != boot_apic_id).take(others) {
            if (made as u64).is_multiple_of(per_chunk) {
                let Some(c) = regions.carve(CHUNK, CHUNK, 1 << 20, u64::MAX) else { break };
                chunk = c;
                // Map this 2 MiB with 4 KiB pages so guard pages can be left out.
                let pt = phys.ptr(chunk).cast::<u64>();
                for i in 0..512u64 {
                    *pt.add(i as usize) = (chunk + i * PAGE) | PTE_P | PTE_W | 0x100 | 1 << 63;
                }
                let Some(pde) = super::kernel_pte(phys, super::PHYS_MAP_BASE + chunk, 1) else { break };
                *pde = chunk | PTE_P | PTE_W;
            }
            let base = chunk + PAGE + (made as u64 % per_chunk) * SLOT_PAGES * PAGE;
            let pt = phys.ptr(chunk).cast::<u64>();
            let page = |n: u64| ((base - chunk) / PAGE + n) as usize;
            *pt.add(page(0)) = 0;
            *pt.add(page(1 + KSTACK_PAGES)) = 0;
            let virt = |p: u64| super::PHYS_MAP_BASE + p;
            let stack_base = base + PAGE;
            let stack_top = stack_base + KSTACK_PAGES * PAGE;
            let ist_top = stack_top + PAGE + ISTACK_PAGES * PAGE;
            let t = phys.ptr(ist_top).cast::<Tables>();
            t.write(Tables::new());
            slots[made] = ApSlot {
                apic_id: id as u64,
                stack_top: virt(stack_top),
                index: made as u64 + 1,
                stack_base: virt(stack_base),
                ist_top: virt(ist_top),
                tables: t as u64,
            };
            made += 1;
        }
        super::flush_all();
        SLOT_COUNT.store(made as u64, Ordering::Release);
        KERNEL_CR3.store(kernel_root, Ordering::Release);

        // The temporary page table: the first 2 MiB mapped at address 0 for
        // the start-up page, and the kernel's half as the kernel has it.
        let (pml4, pdpt, pd) = (tables, tables + PAGE, tables + 2 * PAGE);
        phys.zero(tables, 3 * PAGE);
        core::ptr::copy_nonoverlapping(phys.ptr(kernel_root + 2048), phys.ptr(pml4 + 2048), 2048);
        *phys.ptr(pml4).cast::<u64>() = pdpt | PTE_P | PTE_W;
        *phys.ptr(pdpt).cast::<u64>() = pd | PTE_P | PTE_W;
        *phys.ptr(pd).cast::<u64>() = PTE_P | PTE_W | 0x80;
        debug_assert!(pml4 & !PTE_ADDR == 0);

        let p = phys.ptr(low);
        core::ptr::write_bytes(p, 0, PAGE as usize);
        core::ptr::copy_nonoverlapping(TRAMPOLINE.as_ptr(), p, TRAMPOLINE.len());
        let gdt: [u64; 4] = [0, 0x00CF_9A00_0000_FFFF, 0x00CF_9200_0000_FFFF, 0x00AF_9A00_0000_FFFF];
        p.add(0x80).cast::<[u64; 4]>().write_unaligned(gdt);
        p.add(0xA0).cast::<u16>().write_unaligned(31);
        p.add(0xA2).cast::<u32>().write_unaligned(low as u32 + 0x80);
        p.add(0xA8).cast::<u32>().write_unaligned(low as u32 + 0x24);
        p.add(0xAC).cast::<u16>().write_unaligned(0x08);
        p.add(0xB0).cast::<u32>().write_unaligned(low as u32 + 0x5F);
        p.add(0xB4).cast::<u16>().write_unaligned(0x18);
        p.add(0xB8).cast::<u32>().write_unaligned(pml4 as u32);
        p.add(0xC0).cast::<u64>().write_unaligned(&raw const ap_entry as u64);
        START_PAGE.store(low, Ordering::Release);
    }
    made + 1
}

static START_PAGE: AtomicU64 = AtomicU64::new(0);

/// Start every other processor. They come up on their own time.
pub fn start(tsc_hz: u64) {
    let page = START_PAGE.load(Ordering::Acquire);
    if page == 0 {
        return;
    }
    TSC_HZ.store(tsc_hz, Ordering::Relaxed);
    if cpuid(7, 0)[1] & 1 << 1 != 0 {
        BSP_ADJUST.store(cpu::rdmsr(MSR_TSC_ADJUST), Ordering::Relaxed);
        HAS_ADJUST.store(true, Ordering::Release);
    }
    // On hardware a processor needs 10 ms between INIT and STARTUP and
    // 200 us between the two STARTUPs; a hypervisor's virtual processors need none.
    let wait = |us: u64| {
        if cpuid(1, 0)[2] & 1 << 31 == 0 {
            let end = rdtsc() + tsc_hz / 1_000_000 * us;
            while rdtsc() < end {
                core::hint::spin_loop();
            }
        }
    };
    // SAFETY: the x2APIC interrupt command register; the start-up page and
    // everything the new processors read are in place.
    unsafe {
        wrmsr(0x830, 0xC_4500); // INIT to all but this one
        wait(10_000);
        wrmsr(0x830, 0xC_4600 | page >> 12); // STARTUP at the start-up page
        wait(200);
        wrmsr(0x830, 0xC_4600 | page >> 12);
    }
}

/// Wait for the `expected - 1` secondaries `start` woke, then close the
/// count: the number of processors the machine has, the boot one included,
/// which is what the root task is told. Waits while they keep arriving: a
/// second with none arriving ends it (on hardware each takes well under a
/// millisecond; an emulated machine of many processors takes longer, but
/// steadily).
pub fn wait(expected: usize, tsc_hz: u64) -> usize {
    let mut last = (ADMISSION.arrived(), rdtsc());
    while ADMISSION.arrived() + TSC_REJECTED.load(Ordering::Relaxed) + 1 < expected {
        // What an arriving processor compares its counter with (`ap_main`).
        BSP_TSC.store(ordered_tsc(), Ordering::Release);
        let now = (ADMISSION.arrived(), rdtsc());
        if now.0 != last.0 {
            last = now;
        } else if now.1.wrapping_sub(last.1) > tsc_hz {
            break;
        }
        core::hint::spin_loop();
    }
    let n = ADMISSION.close();
    let rejected = TSC_REJECTED.load(Ordering::Relaxed);
    if rejected != 0 {
        kernel_core::klog!(
            super::kernel(),
            "ERROR: kernel: could not verify timestamp synchronization on {rejected} processor(s); they are not used"
        );
    }
    n
}

const MSR_TSC_ADJUST: u32 = 0x3B;
/// The counter's frequency, the boot processor's IA32_TSC_ADJUST (where
/// processors have it), and the counter it publishes while it waits for
/// the others (0: not yet waiting).
static TSC_HZ: AtomicU64 = AtomicU64::new(0);
static HAS_ADJUST: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
static BSP_ADJUST: AtomicU64 = AtomicU64::new(0);
static BSP_TSC: AtomicU64 = AtomicU64::new(0);

/// The timestamp counter, read after everything before it (LFENCE).
fn ordered_tsc() -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: the fences keep the counter read between the shared-memory
    // operations that bracket it. The compiler must preserve that order too.
    unsafe { core::arch::asm!("lfence", "rdtsc", "lfence", out("eax") lo, out("edx") hi, options(nostack, preserves_flags)) };
    (hi as u64) << 32 | lo as u64
}

/// Whether this processor's timestamp counter agrees with the boot
/// processor's: the kernel's clock is one counter read on whichever
/// processor asks, so counters that differ make time run backwards for a
/// thread that moves. First this processor's IA32_TSC_ADJUST is made the
/// boot processor's (firmware may leave them different; Linux does the
/// same), then its counter is compared with the one the boot processor
/// publishes as it waits. A fresh remote sample must fit between two reads
/// of the local counter at most 50 us apart.
fn tsc_agrees() -> bool {
    if HAS_ADJUST.load(Ordering::Acquire) {
        // SAFETY: IA32_TSC_ADJUST exists on this processor as on the boot
        // processor (the same model); writing it moves this counter only.
        unsafe { wrmsr(MSR_TSC_ADJUST, BSP_ADJUST.load(Ordering::Relaxed)) };
    }
    kernel_core::x86::counters_agree(ordered_tsc, || BSP_TSC.load(Ordering::Acquire), TSC_HZ.load(Ordering::Relaxed))
}
/// Processors whose counters could not be verified before the deadline.
static TSC_REJECTED: AtomicUsize = AtomicUsize::new(0);
