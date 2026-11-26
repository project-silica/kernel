//! aarch64 architecture glue (component B05).
//!
//! The loader enters `_start` at EL1 with the MMU on (the kernel half in
//! TTBR1, all RAM at `PHYS_MAP_BASE`, see `boot-uefi/src/arch/aarch64.rs`).
//! The kernel takes its boot stack and vectors (`cpu.rs`), maps the console
//! UART the ACPI SPCR table names and the GICv3 the MADT describes into its
//! device window (`gic.rs`), sets up the EL1 virtual timer and starts the
//! root task. User address spaces are TTBR0's; the kernel's mappings are
//! TTBR1's and never change. PSCI starts the other processors. Every address-space switch
//! currently flushes the local TLB.

mod cpu;
mod gic;
mod its;
mod smp;

use core::arch::{asm, global_asm};
use core::cell::Cell;
use core::cell::UnsafeCell;
use core::fmt::Write;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use cpu::{PerCpu, percpu};
use kernel_core::ExitStatus;
use kernel_core::Kernel;
use kernel_core::boot::{BOOT_MAGIC, BOOT_VERSION, BootInfo, Cmdline, PHYS_MAP_BASE, Region, RegionKind, RegionTable};
use kernel_core::hal::Hal;
use kernel_core::lock::Presence;
use kernel_core::mem::{Paddr, Phys};
use kernel_core::root::RootSpec;
use silica_abi::{FAULT_FETCH, FAULT_PRESENT, FAULT_WRITE, Fault, Rights};

const BLOCK: u64 = 2 << 20;
const STACK_SIZE: usize = 64 << 10;

/// The boot processor's stack.
#[repr(C, align(4096))]
struct Stack(UnsafeCell<[u8; STACK_SIZE]>);
// SAFETY: used only as the boot processor's stack, through the stack pointer.
unsafe impl Sync for Stack {}
// SAFETY: the linker puts only this page-aligned stack in the segment.
#[unsafe(link_section = ".boot_stack")]
static BOOT_STACK: Stack = Stack(UnsafeCell::new([0; STACK_SIZE]));

global_asm!(
    r#"
.section .text.boot, "ax"
.global _start
_start:
    msr daifset, #0xf
    adrp x9, {stack}
    add x9, x9, :lo12:{stack}
    add x9, x9, #{size}
    mov sp, x9
    adrp x9, vectors
    add x9, x9, :lo12:vectors
    msr vbar_el1, x9
    isb
    mov x29, #0
    mov x30, #0
    bl {kmain}
    b .

"#,
    stack = sym BOOT_STACK,
    size = const STACK_SIZE,
    kmain = sym kmain,
);

// ---- console ---------------------------------------------------------------

/// The console UART's virtual address (0: none known).
static UART: AtomicU64 = AtomicU64::new(0);
/// The kernel was told a test harness listens for the exit status.
static HARNESS: AtomicBool = AtomicBool::new(false);

/// The console's lock, its bytes not yet sent (16 KiB), and when it must be
/// fed again (`u64::MAX`: nothing waits): as on x86_64 (`serial.rs`), the
/// UART is given what its FIFO takes and never waited for; every processor
/// arms its timer no later than `CONSOLE_DUE`.
static CONSOLE_LOCK: kernel_core::lock::Lock = kernel_core::lock::Lock::new();
static CONSOLE: kernel_core::log::TxRing<16384> = kernel_core::log::TxRing::new();
static CONSOLE_DUE: AtomicU64 = AtomicU64::new(u64::MAX);

/// The PL011's transmit FIFO (32 bytes at 115,200 baud: 2.8 ms) takes at
/// least half its depth again within this long.
const CONSOLE_DRAIN_NS: u64 = 1_400_000;

/// Give the UART what it takes now (while its transmit FIFO is not full)
/// and say when to look again. Holding `CONSOLE_LOCK`.
fn console_feed(now: u64) {
    let uart = UART.load(Ordering::Relaxed);
    // SAFETY: `uart` is a PL011-compatible UART mapped as device memory in
    // the device window: flag register at 0x18 (bit 5: transmit FIFO
    // full), data register at 0.
    while CONSOLE.pending() && unsafe { core::ptr::read_volatile((uart + 0x18) as *const u32) } & 0x20 == 0 {
        if let Some(b) = CONSOLE.take() {
            // SAFETY: as above.
            unsafe { core::ptr::write_volatile(uart as *mut u32, b as u32) }
        }
    }
    CONSOLE_DUE.store(if CONSOLE.pending() { now.saturating_add(CONSOLE_DRAIN_NS) } else { u64::MAX }, Ordering::Relaxed);
}

/// Feed the console if it is due at `now` (a timer interrupt, the idle path).
fn console_drain(now: u64) {
    if CONSOLE_DUE.load(Ordering::Relaxed) > now || UART.load(Ordering::Relaxed) == 0 {
        return;
    }
    CONSOLE_LOCK.lock();
    console_feed(now);
    CONSOLE_LOCK.unlock();
}

/// The panic and boot path: what the console still holds, then `s`,
/// waiting for the UART (a bounded number of times per byte). Takes no
/// lock: the processor may hold it already.
fn write_bytes(s: &[u8]) {
    let uart = UART.load(Ordering::Relaxed);
    if uart == 0 {
        return;
    }
    for b in core::iter::from_fn(|| CONSOLE.take()).chain(s.iter().copied()) {
        let mut spins = 0;
        // SAFETY: `uart` is a PL011-compatible UART mapped as device memory
        // in the device window: flag register at 0x18 (bit 5: transmit FIFO
        // full), data register at 0.
        unsafe {
            while core::ptr::read_volatile((uart + 0x18) as *const u32) & 0x20 != 0 && spins < 100_000 {
                spins += 1;
            }
            core::ptr::write_volatile(uart as *mut u32, b as u32);
        }
    }
}

struct Out;
impl Write for Out {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        write_bytes(s.as_bytes());
        record(s.as_bytes());
        Ok(())
    }
}

/// The crash record's region (`kernel_core::crash`, physical; 0: none), this
/// boot's number, and while a panic writes its record, how long it is
/// (`usize::MAX`: no panic is writing). As on x86_64.
static CRASH: AtomicU64 = AtomicU64::new(0);
static BOOT: AtomicU64 = AtomicU64::new(0);
static RECORD_LEN: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(usize::MAX);

/// The crash record's region, mapped, when there is one.
fn crash_region() -> Option<&'static mut [u8]> {
    let at = CRASH.load(Ordering::Relaxed);
    // SAFETY: the region the loader kept for the record, `REGION_BYTES`
    // long, in the physical map; nothing else maps or uses it, and only the
    // boot processor (at boot) or the first processor to panic touches it.
    (at != 0)
        .then(|| unsafe { core::slice::from_raw_parts_mut((PHYS_MAP_BASE + at) as *mut u8, kernel_core::crash::REGION_BYTES as usize) })
}

/// A panic begins its record: the console's last lines first.
fn record_panic() {
    let Some(region) = crash_region() else { return };
    if RECORD_LEN.compare_exchange(usize::MAX, 0, Ordering::AcqRel, Ordering::Relaxed).is_err() {
        return;
    }
    kernel_core::crash::begin(region);
    let mut tail = [0u8; 8192];
    let n = CONSOLE.recent(&mut tail);
    RECORD_LEN.store(kernel_core::crash::append(region, 0, &tail[..n]), Ordering::Release);
}

/// What the panic says goes into its record as well.
fn record(bytes: &[u8]) {
    let len = RECORD_LEN.load(Ordering::Acquire);
    if len == usize::MAX {
        return;
    }
    if let Some(region) = crash_region() {
        RECORD_LEN.store(kernel_core::crash::append(region, len, bytes), Ordering::Release);
    }
}

/// The end of a panic: its record sealed, then the machine reset (PSCI)
/// unless a harness listens for the status.
fn after_panic() -> ! {
    let len = RECORD_LEN.swap(usize::MAX, Ordering::AcqRel);
    if let Some(region) = crash_region()
        && len != usize::MAX
    {
        kernel_core::crash::seal(region, BOOT.load(Ordering::Relaxed), len);
    }
    if HARNESS.load(Ordering::Relaxed) {
        die();
    }
    let _ = write!(Out, "kernel: resetting the machine\r\n");
    smp::system_reset();
    die()
}

/// At boot: the last boot's crash record, if one survived, said on the
/// console whole, kept for the root task (returned), and cleared.
fn last_panic(at: u64) -> &'static [u8] {
    struct Kept(core::cell::UnsafeCell<[u8; silica_abi::root::LAST_PANIC]>);
    // SAFETY: written once by the boot processor before any other starts.
    unsafe impl Sync for Kept {}
    static KEPT: Kept = Kept(core::cell::UnsafeCell::new([0; silica_abi::root::LAST_PANIC]));
    CRASH.store(at, Ordering::Relaxed);
    let Some(region) = crash_region() else {
        let _ = write!(Out, "kernel: no memory kept for a crash record: a panic halts this machine\r\n");
        return &[];
    };
    BOOT.store(kernel_core::crash::next_boot(region), Ordering::Relaxed);
    let Some((was, text)) = kernel_core::crash::read(region) else { return &[] };
    let _ = write!(Out, "kernel: the last boot (boot {was}) ended in a panic; its record:\r\n");
    write_bytes(text);
    let _ = write!(Out, "\r\nkernel: end of the last boot's record\r\n");
    // SAFETY: as above.
    let kept = unsafe { &mut *KEPT.0.get() };
    let n = text.len().min(kept.len());
    kept[..n].copy_from_slice(&text[..n]);
    kernel_core::crash::clear(region);
    &kept[..n]
}

/// Stop this processor: tell a test harness the kernel failed, then wait.
fn die() -> ! {
    stop(ExitStatus::Failure)
}

fn stop(status: ExitStatus) -> ! {
    if HARNESS.load(Ordering::Relaxed) {
        // Semihosting SYS_EXIT_EXTENDED, ADP_Stopped_ApplicationExit, with the
        // status encoded as x86's debug-exit device reports it.
        let block = [0x2_0026u64, (status.code() as u64) << 1 | 1];
        // SAFETY: the call reads the two-word block and ends the emulation.
        unsafe { asm!("hlt #0xf000", in("w0") 0x20u32, in("x1") block.as_ptr(), options(nostack)) }
    }
    loop {
        // SAFETY: masking interrupts and waiting change nothing else.
        unsafe { asm!("msr daifset, #0xf", "wfe", options(nomem, nostack)) }
    }
}

// ---- the device window -------------------------------------------------------
//
// Device registers are mapped Device-nGnRE in 2 MiB blocks at DEVICE_BASE,
// through two tables in the kernel image, so the window exists before the
// kernel owns any memory. Normal memory is never mapped over devices.

const DEVICE_BASE: u64 = 0xFFFF_FF00_0000_0000;
const ATTR_DEVICE: u64 = 1 << 2;
const AF: u64 = 1 << 10;
const PXN: u64 = 1 << 53;
const UXN: u64 = 1 << 54;
const ADDR: u64 = 0x0000_FFFF_FFFF_F000;

#[repr(C, align(4096))]
struct Table(UnsafeCell<[u64; 512]>);
// SAFETY: changed only by the boot processor during boot.
unsafe impl Sync for Table {}
static DEVICE_L1: Table = Table(UnsafeCell::new([0; 512]));
static DEVICE_L2: Table = Table(UnsafeCell::new([0; 512]));
static DEVICE_USED: AtomicU64 = AtomicU64::new(0);

/// Where the kernel image is in physical memory: (virtual start, physical start).
static IMAGE: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

fn image_phys(v: u64) -> u64 {
    v - IMAGE[0].load(Ordering::Relaxed) + IMAGE[1].load(Ordering::Relaxed)
}

/// Map `[phys, phys + len)` of device registers at consecutive virtual
/// addresses and return the address of `phys`.
fn map_device_range(phys: u64, len: u64) -> Option<u64> {
    let first = phys & !(BLOCK - 1);
    let blocks = (phys + len.max(1)).div_ceil(BLOCK) - first / BLOCK;
    let va = map_device(first)?;
    // Each further block, new or mapped before, must follow at the next
    // virtual block.
    for b in 1..blocks {
        if map_device(first + b * BLOCK)? != va + b * BLOCK {
            return None;
        }
    }
    Some(va + (phys - first))
}

/// Map the device registers at physical `phys` (with the rest of their
/// 2 MiB block) and return their virtual address. Boot processor, boot only.
fn map_device(phys: u64) -> Option<u64> {
    let block = phys & !(BLOCK - 1);
    // SAFETY: the tables are the boot processor's alone during boot, and
    // TTBR1's root is the loader's table, mapped through the physical map.
    unsafe {
        let l2 = &mut *DEVICE_L2.0.get();
        for (i, e) in l2.iter().enumerate().take(DEVICE_USED.load(Ordering::Relaxed) as usize) {
            if *e & ADDR == block {
                return Some(DEVICE_BASE + i as u64 * BLOCK + (phys - block));
            }
        }
        let i = DEVICE_USED.fetch_add(1, Ordering::Relaxed) as usize;
        if i >= 512 {
            return None;
        }
        if i == 0 {
            let root: u64;
            asm!("mrs {}, ttbr1_el1", out(reg) root, options(nomem, nostack, preserves_flags));
            let l1 = &mut *DEVICE_L1.0.get();
            l1[0] = image_phys(DEVICE_L2.0.get() as u64) | 0b11;
            let root = (PHYS_MAP_BASE + (root & ADDR)) as *mut u64;
            *root.add((DEVICE_BASE >> 39 & 511) as usize) = image_phys(DEVICE_L1.0.get() as u64) | 0b11;
        }
        l2[i] = block | 0b01 | ATTR_DEVICE | AF | PXN | UXN;
        asm!("dsb ishst", "isb", options(nostack, preserves_flags));
        Some(DEVICE_BASE + i as u64 * BLOCK + (phys - block))
    }
}

/// The loader's table of memory regions (address, entries), recorded first
/// thing at boot: what the physical map covers, for reading firmware tables.
static BOOT_REGIONS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Whether `[p, p + len)` lies in memory the physical map covers: every page
/// of it in a region the loader mapped (anything but device and reserved
/// ranges, which are holes in the map; reading one would fault).
fn mapped(p: u64, len: u64) -> bool {
    let (at, n) = (BOOT_REGIONS[0].load(Ordering::Relaxed), BOOT_REGIONS[1].load(Ordering::Relaxed));
    if at == 0 {
        return false;
    }
    // SAFETY: the loader's region table, in loader memory the physical map
    // covers, read-only for the kernel's life.
    let regions = unsafe { core::slice::from_raw_parts((PHYS_MAP_BASE + at) as *const Region, n as usize) };
    let Some(end) = p.checked_add(len) else { return false };
    let mut page = p & !0xFFF;
    while page < end {
        let held = regions.iter().any(|r| r.kind != RegionKind::Firmware as u32 && r.base <= page && page < r.base.saturating_add(r.len));
        if !held {
            return false;
        }
        page += 0x1000;
    }
    true
}

/// Read physical memory for the shared ACPI walker: false where the map
/// does not reach.
fn phys_read(p: u64, buf: &mut [u8]) -> bool {
    if !mapped(p, buf.len() as u64) {
        return false;
    }
    // SAFETY: just checked to lie in mapped memory; firmware tables are
    // memory nothing writes.
    unsafe { core::ptr::copy_nonoverlapping((PHYS_MAP_BASE + p) as *const u8, buf.as_mut_ptr(), buf.len()) };
    true
}

/// The ACPI table with signature `sig`, whole: found by the shared walker
/// (`kernel_core::acpi::find_table`, the one the tests check), and only
/// where it lies in mapped memory. `None` otherwise.
fn acpi_table(rsdp: u64, sig: &[u8; 4]) -> Option<&'static [u8]> {
    let (t, len) = kernel_core::acpi::find_table(&mut phys_read, rsdp, sig)?;
    // SAFETY: as in `phys_read`, the whole of it checked.
    mapped(t, len as u64).then(|| unsafe { core::slice::from_raw_parts((PHYS_MAP_BASE + t) as *const u8, len as usize) })
}

/// Little-endian fields of a table, `None` past its end.
fn t32(t: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(t.get(at..at.checked_add(4)?)?.try_into().ok()?))
}
fn t64(t: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(t.get(at..at.checked_add(8)?)?.try_into().ok()?))
}

/// The entries of a MADT (`table`), as (type, bytes): ends at the first
/// malformed one.
fn madt_entries(t: &[u8]) -> impl Iterator<Item = (u8, &[u8])> {
    let mut at = 44usize;
    core::iter::from_fn(move || {
        let (kind, n) = (*t.get(at)?, *t.get(at + 1)? as usize);
        if n < 2 {
            return None;
        }
        let e = t.get(at..at.checked_add(n)?)?;
        at += n;
        Some((kind, e))
    })
}

/// The base of the UART the ACPI SPCR table names, if it is a PL011 or an
/// SBSA generic UART in memory space.
fn spcr_uart(rsdp: u64) -> Option<u64> {
    let t = acpi_table(rsdp, b"SPCR")?;
    let (kind, space, base) = (*t.get(36)?, *t.get(40)?, t64(t, 44)?);
    (t.len() >= 80 && matches!(kind, 0x03 | 0x0E) && space == 0 && base != 0).then_some(base)
}

/// What the MADT says about the GIC: the distributor, the redistributor
/// region, and the boot processor's redistributor if it is given per
/// processor instead.
struct Madt {
    gicd: u64,
    gicr: (u64, u64),
    /// The first ITS's registers (0: none).
    its: u64,
}

fn madt(rsdp: u64) -> Option<Madt> {
    let t = acpi_table(rsdp, b"APIC")?;
    let mut m = Madt { gicd: 0, gicr: (0, 0), its: 0 };
    let me = gic::my_affinity();
    for (kind, e) in madt_entries(t) {
        match kind {
            0x0C if e.len() >= 24 => m.gicd = t64(e, 8)?,
            0x0F if e.len() >= 20 && m.its == 0 => m.its = t64(e, 8)?,
            0x0E if e.len() >= 16 => m.gicr = (t64(e, 4)?, t32(e, 12)? as u64),
            // A GICC entry with its own redistributor, used when there is
            // no redistributor region.
            0x0B if e.len() >= 76 && m.gicr.0 == 0 && gic::affinity(t64(e, 68)?) == me => {
                let r = t64(e, 60)?;
                if r != 0 {
                    m.gicr = (r, 0x2_0000);
                }
            }
            _ => {}
        }
    }
    (m.gicd != 0 && m.gicr.0 != 0).then_some(m)
}

// ---- faults and panics -------------------------------------------------------

/// Check the installed translation on the processor that will use the stack.
fn verify_stack(top: u64) {
    let base = top - STACK_SIZE as u64;
    let (guard, first): (u64, u64);
    // SAFETY: translation probes only update PAR_EL1. Interrupts are masked.
    unsafe {
        asm!("at s1e1r, {below}", "isb", "mrs {guard}, par_el1",
             "at s1e1w, {base}", "isb", "mrs {first}, par_el1",
             below = in(reg) base - 1, base = in(reg) base,
             guard = out(reg) guard, first = out(reg) first, options(nostack));
    }
    // PAR.F and FST[5:2]: a translation fault, at any table level.
    assert!(guard & 0x79 == 9 && first & 1 == 0, "invalid kernel stack mappings at {base:#x}: guard {guard:#x}, stack {first:#x}");
}

/// Walk frame records only on the stack this processor is actually using.
/// The boot stack is available even before per-processor state is installed.
fn backtrace(first: u64, mut fp: u64) {
    let slide = SLIDE.load(Ordering::Relaxed);
    let _ = write!(Out, "#0 {:#018x}\r\n", first.wrapping_sub(slide));
    let sp: u64;
    // SAFETY: reading our current stack pointer.
    unsafe { asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags)) };
    let boot_top = BOOT_STACK.0.get() as u64 + STACK_SIZE as u64;
    let tops = core::iter::once(boot_top)
        .chain((1..kernel_core::smp::MAX_CPUS).filter_map(cpu::percpu_of).map(|p| p.kstack_top.load(Ordering::Relaxed)));
    let Some(hi) = tops.into_iter().find(|&top| (top - STACK_SIZE as u64..top).contains(&sp)) else { return };
    for n in 1..32 {
        if !(fp.is_multiple_of(16) && fp >= sp && fp <= hi - 16) {
            break;
        }
        // SAFETY: `fp` was just checked to lie inside this processor's kernel stack.
        let (next, ret) = unsafe { (*(fp as *const u64), *((fp + 8) as *const u64)) };
        if ret == 0 {
            break;
        }
        let _ = write!(Out, "#{} {:#018x}\r\n", n, ret.wrapping_sub(slide));
        if next <= fp {
            break;
        }
        fp = next;
    }
}

/// `panic=null`: reach a thread object through the null address, which must
/// fault (physical page 0 is never mapped for the kernel), so the test sees
/// a kernel exception with a backtrace through this function.
#[inline(never)]
fn deliberate_null_access(k: &'static Kernel<Arm>) -> ! {
    let v = core::hint::black_box(k.regs(core::hint::black_box(0)))[0].get();
    panic!("physical page 0 is mapped: read {v:#x} through a null object address");
}

#[inline(never)]
fn deliberate_panic(cpu: usize) -> ! {
    panic!("deliberate panic on processor {cpu}");
}

static SLIDE: AtomicU64 = AtomicU64::new(0);

/// The kernel panicked: say why, show where, stop.
pub fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    record_panic();
    let _ = write!(Out, "\r\nPANIC: {}\r\n", info.message());
    if let Some(l) = info.location() {
        let _ = write!(Out, "  at {}:{}\r\n", l.file(), l.line());
    }
    let _ = write!(Out, "  kernel slide {:#x}; backtrace (link addresses):\r\n", SLIDE.load(Ordering::Relaxed));
    let (pc, fp): (u64, u64);
    // SAFETY: reads the program counter and the frame pointer.
    unsafe { asm!("adr {}, .", "mov {}, x29", out(reg) pc, out(reg) fp, options(nomem, nostack)) }
    backtrace(pc, fp);
    after_panic()
}

/// An exception taken in the kernel (the only kind there is so far).
extern "C" fn kernel_fault(esr: u64, elr: u64, far: u64, entry: u64, fp: u64) -> ! {
    record_panic();
    let _ = write!(
        Out,
        "\r\nPANIC: kernel exception through vector {} (syndrome {:#x}: class {:#x}) at {:#x}, address {:#x}\r\n  kernel slide {:#x}; backtrace (link addresses):\r\n",
        entry,
        esr,
        esr >> 26 & 0x3F,
        elr.wrapping_sub(SLIDE.load(Ordering::Relaxed)),
        far,
        SLIDE.load(Ordering::Relaxed)
    );
    backtrace(elr, fp);
    after_panic()
}

// ---- the kernel's view of the machine ---------------------------------------

/// What the aarch64 layer knows about the machine. Set by the boot
/// processor before any other starts, then only read.
pub struct Arm {
    /// Counter value at monotonic time 0.
    count0: Cell<u64>,
    /// Nanoseconds per tick and ticks per nanosecond, 32.32 fixed point.
    ns_per_tick: Cell<u64>,
    ticks_per_ns: Cell<u64>,
    /// An empty first-level table: TTBR0 when no user address space is loaded.
    empty_root: Cell<u64>,
    /// The largest ASID: 255 or 65535 (the loader enables 16-bit ASIDs
    /// where the processor has them).
    asid_max: Cell<u64>,
}

struct Global(Kernel<Arm>);
// SAFETY: every processor enters the kernel through `Kernel::enter`, and
// kernel-core's lock protocol decides what each may touch. The `Arm` fields
// are written by the boot processor before any other starts.
unsafe impl Sync for Global {}

static KERNEL: Global = Global(Kernel::new(
    Arm { count0: Cell::new(0), ns_per_tick: Cell::new(0), ticks_per_ns: Cell::new(0), empty_root: Cell::new(0), asid_max: Cell::new(255) },
    // SAFETY: the loader maps all RAM at PHYS_MAP_BASE and the kernel only
    // uses addresses from the boot regions through it.
    unsafe { Phys::new(PHYS_MAP_BASE as *mut u8) },
));

/// The kernel.
pub fn kernel() -> &'static Kernel<Arm> {
    &KERNEL.0
}

pub(super) fn counter_frequency() -> u64 {
    let f: u64;
    // SAFETY: reading the counter frequency.
    unsafe { asm!("mrs {}, cntfrq_el0", out(reg) f, options(nomem, nostack, preserves_flags)) }
    f
}

impl Arm {
    /// The counter value of monotonic time `deadline`.
    fn count_at(&self, deadline: u64) -> u64 {
        let ticks = (deadline as u128 * self.ticks_per_ns.get() as u128) >> 32;
        (self.count0.get() as u128 + ticks).min(u64::MAX as u128 - 1) as u64
    }

    /// The table TTBR0 holds when no user address space is loaded.
    pub fn empty_root(&self) -> u64 {
        self.empty_root.get()
    }

    /// The TTBR0 value that loads `root`. With ASIDs a switch keeps the
    /// translations cached for each address space: it has an ASID per
    /// processor for that processor's current generation, kept in its
    /// header page (words owned by this module from offset 2048, as on
    /// x86_64). A processor hands out its ASIDs in order; when they run out
    /// it starts a new generation, so every address space takes a fresh one
    /// there. An ASID is dropped from this processor's TLB as it is handed
    /// out, so whatever an earlier holder left cached is gone first. ASID 0
    /// is the empty root's.
    fn ttbr0_for(&self, phys: &Phys, root: Paddr) -> u64 {
        if root == self.empty_root.get() {
            return root;
        }
        let me = percpu();
        let generation = me.asid_gen.get();
        // SAFETY: `root` is the root table of a live AddressSpace object,
        // whose header page precedes it; kernel-core uses only its first
        // bytes, and the second half holds one word per processor.
        let slot = unsafe { &mut *phys.ptr(root - 4096 + 2048 + me.index as u64 * 8).cast::<u64>() };
        if *slot >> 16 == generation {
            return root | (*slot & 0xFFFF) << 48;
        }
        let id = me.asid_next.get();
        if id >= self.asid_max.get() {
            me.asid_next.set(1);
            me.asid_gen.set(generation + 1);
        } else {
            me.asid_next.set(id + 1);
        }
        *slot = generation << 16 | id;
        // SAFETY: TLB maintenance only drops cached translations.
        unsafe { asm!("tlbi aside1, {}", "dsb nsh", in(reg) id << 48, options(nostack, preserves_flags)) }
        root | id << 48
    }
}

/// This processor's translation control register.
fn tcr() -> u64 {
    let v: u64;
    // SAFETY: reading a system register.
    unsafe { asm!("mrs {}, tcr_el1", out(reg) v, options(nomem, nostack, preserves_flags)) }
    v
}

pub(super) fn counter() -> u64 {
    let v: u64;
    // SAFETY: reading the virtual count; ISB keeps it in program order.
    unsafe { asm!("isb", "mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack, preserves_flags)) }
    v
}

fn ttbr0() -> u64 {
    let v: u64;
    // SAFETY: reading the translation table base.
    unsafe { asm!("mrs {}, ttbr0_el1", out(reg) v, options(nomem, nostack, preserves_flags)) }
    v & ADDR
}

/// The ASID TTBR0 holds now.
fn current_asid() -> u64 {
    let v: u64;
    // SAFETY: reading the translation table base.
    unsafe { asm!("mrs {}, ttbr0_el1", out(reg) v, options(nomem, nostack, preserves_flags)) }
    v >> 48
}

/// Load a TTBR0 value (a root and its ASID). Translations cached for other
/// ASIDs stay: user mappings are not global, so each is tagged with its
/// address space's ASID (`Arm::ttbr0_for`).
fn load_ttbr0(value: u64) {
    // SAFETY: TTBR0 holds only user mappings; the kernel runs from TTBR1,
    // which this does not touch.
    unsafe { asm!("msr ttbr0_el1, {}", "isb", in(reg) value, options(nostack, preserves_flags)) }
}

/// Load `root` (with ASID 0) and drop every cached translation on this
/// processor: once at start-up, when the loader's may still be cached.
fn reset_ttbr0(root: u64) {
    // SAFETY: as `load_ttbr0`; TLB maintenance only drops cached translations.
    unsafe { asm!("msr ttbr0_el1, {}", "isb", "tlbi vmalle1", "dsb nsh", "isb", in(reg) root, options(nostack, preserves_flags)) }
}

// Leaf attributes of user mappings.
const PAGE_L3: u64 = 0b11;
const BLOCK_L2: u64 = 0b01;
const AP_EL0: u64 = 1 << 6;
const AP_RO: u64 = 1 << 7;
const SH_INNER: u64 = 3 << 8;
const NG: u64 = 1 << 11;

impl Hal for Arm {
    const IP: usize = cpu::REG_PC;
    const SP: usize = cpu::REG_SP;
    const FLAGS: usize = cpu::REG_SPSR;
    const SYS: usize = 8;
    const BY_SYSCALL: usize = cpu::REG_BY_SVC;
    const ARG: [usize; 6] = [0, 1, 2, 3, 4, 5];
    /// EL0, interrupts unmasked.
    const FLAGS_INIT: u64 = 0;
    const SYSCALL_LEN: u64 = 4;
    const USER_TOP: u64 = 0x0000_7FFF_FFFF_F000;
    /// The kernel runs on TTBR1; TTBR0 holds only the program's half.
    const KERNEL_IN_ROOT: bool = false;

    fn pte_table(table: Paddr) -> u64 {
        table | 0b11
    }
    fn pte_leaf(frame: Paddr, rights: Rights, level: u8, device: bool) -> u64 {
        // User memory: EL0 access, not global (tagged by address space),
        // never executable by the kernel.
        let mut e = frame | AF | NG | AP_EL0 | SH_INNER | PXN;
        e |= if level == 0 { PAGE_L3 } else { BLOCK_L2 };
        if !rights.has(Rights::WRITE) {
            e |= AP_RO;
        }
        if !rights.has(Rights::EXEC) {
            e |= UXN;
        }
        if device {
            e |= ATTR_DEVICE;
        }
        e
    }
    fn pte_present(pte: u64) -> bool {
        pte & 1 != 0
    }
    fn pte_is_leaf(pte: u64, level: u8) -> bool {
        level == 0 || pte & 0b10 == 0
    }
    fn pte_addr(pte: u64) -> Paddr {
        pte & ADDR
    }

    /// The kernel lives in TTBR1: a user root starts empty.
    fn init_root(&self, _phys: &Phys, _root: Paddr) {}
    /// Retyped memory is zero: all vector registers zero, FPCR default.
    fn init_thread(&self, _phys: &Phys, _thread: Paddr) {}

    fn switch_thread(&self, phys: &Phys, from: Paddr, to: Paddr, root: Paddr) {
        let k = kernel();
        let area = |t: Paddr| phys.ptr(t + kernel_core::thread::FPU_OFFSET as u64);
        // Vector state, as on x86: a thread that entered by SVC keeps only
        // its control and status words (a system call clobbers the vector
        // registers, as a call does) and finds them cleared; an interrupted
        // or faulted thread gets a full save and restore.
        //
        // SAFETY: `from` and `to` are live Thread objects of 4 KiB, whose
        // area from FPU_OFFSET has room for the 512 bytes of q0..q31.
        unsafe {
            if from != 0 {
                let regs = k.regs(from);
                if regs[cpu::REG_BY_SVC].get() == 0 {
                    cpu::silica_simd_save(area(from));
                }
                regs[cpu::REG_FPCTL].set(cpu::silica_fp_control());
            }
            let to_regs = k.regs(to);
            if to_regs[cpu::REG_BY_SVC].get() != 0 {
                cpu::silica_simd_clear();
            } else {
                cpu::silica_simd_load(area(to));
            }
            cpu::silica_fp_set_control(to_regs[cpu::REG_FPCTL].get());
            percpu().regs.set(to_regs.as_ptr() as u64);
        }
        let want = if root == 0 { self.empty_root.get() } else { root };
        if ttbr0() != want {
            load_ttbr0(self.ttbr0_for(phys, want));
        }
    }

    fn leave_aspace(&self, root: Paddr) {
        if ttbr0() == root {
            load_ttbr0(self.empty_root.get());
        }
    }

    fn tlb_flush(&self, root: Paddr, vaddr: u64, len: u64) {
        if ttbr0() != root {
            // Not loaded: forget its ASID here, so its next load takes a
            // fresh one, whose stale translations are dropped first.
            self.tlb_forget(root, percpu().index);
            return;
        }
        let asid = current_asid() << 48;
        // SAFETY: TLB maintenance only drops cached translations.
        unsafe {
            asm!("dsb ishst", options(nostack, preserves_flags));
            if len > 32 * 4096 {
                asm!("tlbi aside1, {}", in(reg) asid, options(nostack, preserves_flags));
            } else {
                for page in (vaddr..vaddr + len).step_by(4096) {
                    asm!("tlbi vae1, {}", in(reg) asid | (page >> 12 & 0xFFF_FFFF_FFFF), options(nostack, preserves_flags));
                }
            }
            asm!("dsb nsh", "isb", options(nostack, preserves_flags));
        }
    }

    fn now(&self) -> u64 {
        let ticks = counter().wrapping_sub(self.count0.get());
        ((ticks as u128 * self.ns_per_tick.get() as u128) >> 32) as u64
    }

    fn set_timer(&self, deadline: u64) {
        // No later than the console must be fed again (`CONSOLE_DUE`).
        let deadline = deadline.min(CONSOLE_DUE.load(Ordering::Relaxed));
        let me = percpu();
        if deadline == u64::MAX {
            me.deadline.set(u64::MAX);
            // SAFETY: disabling the EL1 virtual timer.
            unsafe { asm!("msr cntv_ctl_el0, xzr", "isb", options(nostack, preserves_flags)) }
            return;
        }
        let at = self.count_at(deadline);
        me.deadline.set(at);
        // SAFETY: an absolute compare value, then enabled and unmasked.
        unsafe {
            asm!("msr cntv_cval_el0, {}", "msr cntv_ctl_el0, {one}", "isb", in(reg) at, one = in(reg) 1u64, options(nostack, preserves_flags))
        }
    }

    fn irq_pending(&self) -> bool {
        counter() >= percpu().deadline.get() || gic::pending()
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
        console_drain(self.now());
        let until = until.min(CONSOLE_DUE.load(Ordering::Relaxed));
        // Watch for work first, as on x86 (`IDLE_WATCH_NS`); without `words`
        // the watch already ran (`Hal::idle`).
        if let Some(words) = words
            && kernel_core::sched::IDLE_WATCH_NS != 0
        {
            let me = percpu();
            let end = self.now().saturating_add(me.watch.length());
            loop {
                // SAFETY: a window for a pending interrupt, taken by
                // `kernel_irq` as in the wait below; masking is restored.
                unsafe { asm!("msr daifclr, #2", "isb", "msr daifset, #2", options(nomem, nostack)) }
                if !me.mode.watching() {
                    break;
                }
                // A watched line's news (`IrqWatch`), before its interrupt.
                if changed(words) {
                    me.mode.enter_kernel();
                    return true;
                }
                let now = self.now();
                if now >= until {
                    me.mode.enter_kernel();
                    return true;
                }
                if now >= end {
                    if !words.is_empty() {
                        // Devices told to be quiet are told to interrupt first.
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
        } else if kernel_core::sched::IDLE_WATCH_NS != 0 && !percpu().mode.stop_watching() {
            percpu().mode.enter_kernel();
            return true;
        }
        // Only ever earlier (`Hal::idle`): a timer armed sooner fires early
        // once and the scheduler comes back here.
        if until != u64::MAX && self.count_at(until) < percpu().deadline.get() {
            self.set_timer(until);
        }
        let halted = self.now();
        // WFI wakes for a pending interrupt even while interrupts are
        // masked, so none is lost between the decision to wait and the
        // wait; it is then taken, briefly unmasked, by `kernel_irq`.
        // SAFETY: masking state is restored before returning.
        unsafe { asm!("dsb sy", "wfi", "msr daifclr, #2", "isb", "msr daifset, #2", options(nomem, nostack)) }
        // Woken before the deadline: work (or a timer armed sooner) ended the
        // halt, and the watch learns from it.
        let now = self.now();
        if now < until && kernel_core::sched::IDLE_WATCH_NS != 0 {
            percpu().watch.woken_after(now - halted);
        }
        percpu().mode.enter_kernel();
        true
    }

    fn cpu(&self) -> usize {
        percpu().index
    }

    fn kick(&self, cpu: usize, wait: bool) {
        let Some(them) = cpu::percpu_of(cpu) else { return };
        if !them.online.load(Ordering::Acquire) || them.mode.in_kernel() || them.mode.poke() {
            return;
        }
        gic::kick(them.mpidr.load(Ordering::Relaxed));
        while wait && them.mode.is_user() {
            core::hint::spin_loop();
        }
    }

    fn wait_left(&self, _cpu: usize, request: &kernel_core::lock::Leave) {
        // Never asked here (`KERNEL_IN_ROOT` is false), but as on x86: it
        // answers while it waits for its lock.
        while !request.answered() {
            core::hint::spin_loop();
        }
    }

    fn tlb_reset(&self) {
        // A new generation makes every address space's ASID on this
        // processor stale; each takes a fresh one, dropped from the TLB
        // first, when next loaded.
        let me = percpu();
        me.asid_gen.set(me.asid_gen.get() + 1);
        me.asid_next.set(1);
        load_ttbr0(self.empty_root.get());
    }

    fn tlb_forget(&self, root: Paddr, cpu: usize) {
        // SAFETY: that processor's word in the header page of a live
        // address space (see `ttbr0_for`); the caller holds its lock.
        unsafe { *kernel().phys.ptr(root - 4096 + 2048 + cpu as u64 * 8).cast::<u64>() = 0 }
    }

    fn irq_route(&self, line: usize, cpu: usize, level: bool, active_low: bool) -> bool {
        // GIC inputs are active-high; a platform with an active-low line
        // inverts it before the GIC, so a request for one is an error here.
        !active_low && cpu::percpu_of(cpu).is_some_and(|c| gic::route(line, c.mpidr.load(Ordering::Relaxed), level))
    }

    fn irq_mask(&self, line: usize, masked: bool) {
        gic::mask(line, masked);
    }

    fn irq_move(&self, line: usize, cpu: usize) -> bool {
        let Some(mpidr) = cpu::percpu_of(cpu).map(|c| c.mpidr.load(Ordering::Relaxed)) else { return false };
        if line >= its::FIRST_LINE { its::retarget(line, cpu) } else { gic::retarget(line, mpidr) }
    }

    /// There is no I/O port space on aarch64; no port capability exists.
    fn port_io(&self, _port: u16, _width: u8, _write: bool, _value: u32) -> u32 {
        0
    }

    fn console(&self, bytes: &[u8]) {
        if UART.load(Ordering::Relaxed) == 0 {
            return;
        }
        CONSOLE_LOCK.lock();
        CONSOLE.put(bytes);
        console_feed(self.now());
        CONSOLE_LOCK.unlock();
    }

    fn msi_first(&self) -> usize {
        if its::ready() { its::FIRST_LINE } else { usize::MAX }
    }

    fn isolate(&self, _entering: bool) {
        // Cores with FEAT_CSV2 keep branch predictions apart by context
        // already; without it the firmware's workaround flushes them.
        if SPECTRE_WORKAROUND.load(Ordering::Relaxed) {
            smp::psci(0x8000_8000, 0, 0, 0); // SMCCC_ARCH_WORKAROUND_1
        }
    }
    fn msi_release(&self, line: usize) {
        its::release(line);
    }
    fn msi_route(&self, line: usize, cpu: usize, source: u64) -> Option<(u64, u64)> {
        its::route(line, cpu, source)
    }

    fn power_off(&self, status: u64) {
        let code = if status == 0 { ExitStatus::Success } else { ExitStatus::Failure };
        if HARNESS.load(Ordering::Relaxed) {
            stop(code);
        }
        // PSCI SYSTEM_OFF; it returns only if the firmware has no PSCI.
        smp::system_off();
    }

    fn reset(&self, status: u64) {
        let code = if status == 0 { ExitStatus::Success } else { ExitStatus::Failure };
        if HARNESS.load(Ordering::Relaxed) {
            stop(code);
        }
        smp::system_reset();
    }
}

// ---- entry from user mode ----------------------------------------------------

/// An exception from EL0: `kind` 0 synchronous (SVC or a fault), 1 interrupt.
extern "C" fn user_trap(kind: u64) {
    let k = kernel();
    // The entry stub has saved the frame and marked whether a system call
    // saved it (REG_BY_SVC).
    percpu().mode.enter_kernel();
    k.enter();
    if kind != 0 {
        irq(true);
        return;
    }
    let (esr, far): (u64, u64);
    // SAFETY: reading the syndrome and fault address of this exception.
    unsafe { asm!("mrs {}, esr_el1", "mrs {}, far_el1", out(reg) esr, out(reg) far, options(nomem, nostack, preserves_flags)) }
    let class = esr >> 26 & 0x3F;
    if class == 0x15 {
        k.syscall();
        return;
    }
    match class {
        // Instruction abort (0x20) or data abort (0x24) from EL0. Fault
        // status 0x0C-0x0F is a permission fault: something is mapped.
        0x20 | 0x24 => {
            let status = esr & 0x3F;
            let access = if (0x0C..=0x0F).contains(&status) { FAULT_PRESENT } else { 0 }
                | if class == 0x24 && esr >> 6 & 1 != 0 { FAULT_WRITE } else { 0 }
                | if class == 0x20 { FAULT_FETCH } else { 0 };
            k.fault(Fault::Page, far, access);
        }
        // Undefined instruction, illegal execution state, a trapped system
        // register access: an instruction EL0 may not run.
        0x00 | 0x0E | 0x18 => k.fault(Fault::Instruction, far, esr),
        _ => k.fault(Fault::Exception, far, esr),
    }
}

/// Interrupts for this processor until none is pending. `locked`: it
/// interrupted user code and this processor holds its lock; otherwise it
/// interrupted the idle wait, which looks at everything again when it has
/// the lock.
fn irq(locked: bool) {
    let k = kernel();
    loop {
        let id = gic::acknowledge();
        // 1020 to 1023 say there is nothing (more) to take; LPIs, from
        // 8192, are interrupts like any other and must be ended.
        if (gic::SPECIAL..gic::SPECIAL + 4).contains(&id) {
            return;
        }
        match id {
            gic::PPI_TIMER => {
                // One-shot: silence it until the scheduler arms it again.
                // SAFETY: disabling the EL1 virtual timer.
                unsafe { asm!("msr cntv_ctl_el0, xzr", "isb", options(nostack, preserves_flags)) }
                percpu().deadline.set(u64::MAX);
                gic::end(id);
                // Feed the console, and keep the timer armed while
                // anything waits there (the scheduler arms it only for its
                // own deadlines).
                console_drain(k.hal.now());
                if CONSOLE_DUE.load(Ordering::Relaxed) != u64::MAX {
                    k.hal.set_timer(u64::MAX);
                }
                if locked {
                    k.on_timer();
                }
            }
            gic::SGI_KICK => gic::end(id),
            id if id >= gic::SPI_BASE => {
                // An LPI is a message line; anything else a wire.
                let line = its::line_of(id).unwrap_or((id - gic::SPI_BASE) as usize);
                // A level-triggered line stays asserted until its driver has
                // dealt with the device: hold it off until the driver says so.
                if gic::is_level(line) {
                    gic::mask(line, true);
                }
                gic::end(id);
                if locked {
                    k.irq(line);
                } else {
                    percpu().mode.enter_kernel();
                    k.enter();
                    k.irq(line);
                    percpu().mode.idle();
                    k.leave();
                }
            }
            other => {
                gic::end(other);
                let _ = write!(Out, "kernel: unexpected interrupt {other} on processor {}\r\n", percpu().index);
            }
        }
    }
}

extern "C" fn irq_idle() {
    irq(false);
}

extern "C" fn schedule_rust() {
    let k = kernel();
    k.schedule();
    percpu().mode.leave_to_user();
    k.leave();
}

// ---- boot ------------------------------------------------------------------

extern "C" fn kmain(info: u64) -> ! {
    // SAFETY: the loader passes the physical address of a BootInfo in loader
    // memory, which the physical map covers and nothing overwrites.
    let bi: BootInfo = unsafe { core::ptr::read((PHYS_MAP_BASE + info) as *const BootInfo) };
    if bi.magic != BOOT_MAGIC || bi.version != BOOT_VERSION {
        die();
    }
    BOOT_REGIONS[0].store(bi.regions, Ordering::Relaxed);
    BOOT_REGIONS[1].store(bi.region_count, Ordering::Relaxed);
    IMAGE[0].store(bi.kernel_virt, Ordering::Relaxed);
    IMAGE[1].store(bi.kernel_phys, Ordering::Relaxed);
    SLIDE.store(bi.kernel_slide, Ordering::Relaxed);
    // SAFETY: loader memory the physical map covers, read-only from here on.
    let cmdline = unsafe { core::slice::from_raw_parts((PHYS_MAP_BASE + bi.cmdline) as *const u8, bi.cmdline_len as usize) };
    let cmdline = Cmdline(cmdline);
    HARNESS.store(cmdline.get("exit") == Some(b"semihosting"), Ordering::Relaxed);
    if let Some(uart) = spcr_uart(bi.rsdp).and_then(map_device) {
        UART.store(uart, Ordering::Relaxed);
    }
    let last_panic = last_panic(bi.crash);

    // SAFETY: as for the command line.
    let regions = unsafe { core::slice::from_raw_parts((PHYS_MAP_BASE + bi.regions) as *const Region, bi.region_count as usize) };
    let usable: u64 = regions.iter().filter(|r| r.kind == 1).map(|r| r.len).sum();
    let (freq, el): (u64, u64);
    // SAFETY: reading the counter frequency and the exception level.
    unsafe { asm!("mrs {}, cntfrq_el0", "mrs {}, CurrentEL", out(reg) freq, out(reg) el, options(nomem, nostack)) }
    let _ = write!(
        Out,
        "silica: aarch64, {} MiB free, counter {} Hz, EL{}, kernel slide {:#x}\r\n",
        usable >> 20,
        freq,
        el >> 2 & 3,
        bi.kernel_slide
    );
    verify_stack(BOOT_STACK.0.get() as u64 + STACK_SIZE as u64);
    if cmdline.get("panic") == Some(b"test") {
        deliberate_panic(0);
    }
    let k = kernel();
    if cmdline.get("panic") == Some(b"null") {
        deliberate_null_access(k);
    }
    BOOT_CPU.kstack_top.store(BOOT_STACK.0.get() as u64 + STACK_SIZE as u64, Ordering::Relaxed);
    BOOT_CPU.mpidr.store(gic::my_affinity(), Ordering::Relaxed);
    // SAFETY: once, on the boot processor, with interrupts masked; the
    // per-processor data is a static.
    unsafe { cpu::install(&BOOT_CPU) };
    if !cpu::has_pan() {
        let _ = write!(
            Out,
            "kernel: this processor has no PAN (Armv8.0): the kernel runs without the guard against touching user memory by accident\r\n"
        );
    }
    smp::LOW_ROOT.store(ttbr0(), Ordering::Relaxed);

    // The interrupt controller.
    let Some(m) = madt(bi.rsdp) else {
        let _ = write!(Out, "ERROR: kernel: the ACPI MADT describes no GICv3 distributor and redistributor\r\n");
        die();
    };
    let (Some(gicd), Some(gicr_region)) = (map_device_range(m.gicd, 0x1_0000), map_device_range(m.gicr.0, m.gicr.1)) else {
        let _ = write!(Out, "ERROR: kernel: the GIC does not fit the kernel's device window\r\n");
        die();
    };
    gic::init_distributor(gicd);
    let Some(gicr) = gic::find_redistributor(gicr_region, m.gicr.1, gic::my_affinity()) else {
        let _ = write!(Out, "ERROR: kernel: no GIC redistributor for the boot processor\r\n");
        die();
    };
    if !gic::wake(gicr) {
        panic!("the boot processor's GIC redistributor did not wake within 100 ms (GICR_WAKER.ChildrenAsleep stays set)");
    }
    gic::init_processor(0, gicr);
    BOOT_CPU.online.store(true, Ordering::Release);

    // The other processors' memory comes from the region table before the
    // root task is given the rest. `cpus=1` on the command line keeps them asleep.
    // SAFETY: the boot processor's table, during boot.
    let table = unsafe { &mut *REGIONS.0.get() };
    table.fill(regions);
    let cpus = match cmdline.get("cpus") {
        Some(b"1") => 1,
        _ => smp::prepare(bi.rsdp, table, (gicr_region, m.gicr.1)),
    };
    k.set_cpu_count(cpus);
    if !smp::psci_present() {
        smp::detect_psci(bi.rsdp);
    }
    decide_isolation();
    // Message-signalled interrupts, through the ITS when there is one.
    smp::GICR_PHYS_BASE.store(m.gicr.0, Ordering::Relaxed);
    let its_ok = m.its != 0
        && map_device_range(m.its, 0x2_0000)
            .is_some_and(|its| its::init(its, m.its, cpus, table, acpi_table(bi.rsdp, b"IORT"), |p| PHYS_MAP_BASE + p));
    if m.its != 0 && !its_ok {
        let _ = write!(Out, "kernel: the GIC's ITS cannot be used; devices have no message interrupts\r\n");
    }
    its::enable_processor(0, gicr, m.gicr.0 + (gicr - gicr_region));
    let regions = table.as_slice();

    // Time: the generic timer's counter.
    let h = &k.hal;
    h.count0.set(counter());
    h.ns_per_tick.set(((1_000_000_000u128 << 32) / freq as u128) as u64);
    h.ticks_per_ns.set((((freq as u128) << 32) / 1_000_000_000) as u64);
    h.empty_root.set(image_phys(EMPTY_ROOT.0.get() as u64));
    h.asid_max.set(if tcr() & 1 << 36 != 0 { u16::MAX as u64 } else { u8::MAX as u64 });
    reset_ttbr0(h.empty_root.get());
    k.enter();

    // SAFETY: loader memory, read-only until the root task owns it.
    let ramdisk = unsafe { core::slice::from_raw_parts((PHYS_MAP_BASE + bi.ramdisk) as *const u8, bi.ramdisk_len as usize) };
    // Nothing in the archive runs, the root task included, unless the whole
    // of it is as the image builder wrote it (`ramdisk::check`).
    if let Err(why) = kernel_core::ramdisk::check(ramdisk, crc32c) {
        let _ = write!(Out, "ERROR: kernel: no root task: the boot archive is refused: {why}\r\n");
        die();
    }
    let Some(image) = kernel_core::ramdisk::find(ramdisk, "root") else {
        let _ = write!(Out, "ERROR: kernel: no root task: the boot archive has no entry named \"root\"\r\n");
        die();
    };
    // Device memory is handed out as far as the processor's physical
    // addresses go: firmware puts 64-bit device registers above all memory
    // (QEMU's virt machine at 512 GiB).
    let parange: u64;
    // SAFETY: reading an ID register.
    unsafe { asm!("mrs {}, id_aa64mmfr0_el1", out(reg) parange, options(nomem, nostack, preserves_flags)) }
    let width = [32u32, 36, 40, 42, 44, 48, 52].get((parange & 0xF) as usize).copied().unwrap_or(48).min(48);
    let top = regions.last().map_or(0, |r| r.base + r.len).max(1 << width);
    let uart = spcr_uart(bi.rsdp).unwrap_or(0) & !0xFFF;
    let reserved = [(m.gicd, m.gicd + 0x1_0000), (m.gicr.0, m.gicr.0 + m.gicr.1), (uart, uart + 0x1000), (m.its, m.its + 0x2_0000)];
    match kernel_core::boot::entropy_verdict(bi.entropy_sources, cmdline.get("entropy")) {
        Err(why) => {
            let _ = write!(Out, "{why}\r\n");
            die();
        }
        Ok(Some(warning)) => {
            let _ = write!(Out, "{warning}\r\n");
        }
        Ok(None) => {}
    }
    // The other processors, started and counted (those that arrived)
    // before the root task is told how many there are.
    let cpus = if cpus > 1 { smp::launch(cpus, cmdline.get("panic") == Some(b"secondary")) } else { 1 };
    k.set_cpu_count(cpus);
    let spec = RootSpec {
        regions,
        elf: image,
        machine: kernel_core::elf::EM_AARCH64,
        // The GIC and the kernel's console stay with the kernel.
        reserved: &reserved,
        device_top: top,
        ports: false,
        ramdisk: [bi.ramdisk, bi.ramdisk_len],
        rsdp: bi.rsdp,
        framebuffer: [bi.fb_base, bi.fb_width as u64, bi.fb_height as u64, bi.fb_stride as u64, bi.fb_format as u64],
        clock: [freq, bi.loader_tsc, h.count0.get()],
        wall_clock: [bi.wall_secs, ((bi.wall_tsc.wrapping_sub(h.count0.get()) as i64 as i128 * 1_000_000_000) / freq as i128) as u64],
        entropy: bi.entropy,
        entropy_sources: bi.entropy_sources,
        cmdline: cmdline.0,
        cpus: cpus as u64,
        last_panic,
    };
    if let Err(e) = k.boot_root(&spec) {
        let _ = write!(Out, "ERROR: kernel: cannot start the root task: {e:?}\r\n");
        die();
    }
    // SAFETY: the vectors, the processor and the root thread are set up and
    // the kernel is entered; this is the first exit to user mode.
    unsafe { cpu::exit_to_user() }
}

struct RegionCell(UnsafeCell<RegionTable>);
// SAFETY: used by the boot processor alone, during boot.
unsafe impl Sync for RegionCell {}
static REGIONS: RegionCell = RegionCell(UnsafeCell::new(RegionTable::new()));

/// The boot processor's data.
static BOOT_CPU: PerCpu = PerCpu::new(0, 0, 0);

/// The first-level table TTBR0 holds while no user address space is loaded.
static EMPTY_ROOT: Table = Table(UnsafeCell::new([0; 512]));

/// Whether a watched word (`Hal::idle`) no longer holds what it held at the
/// line's last delivery.
#[inline(always)]
fn changed(words: &[kernel_core::hal::Watched]) -> bool {
    words.iter().any(|w| {
        // SAFETY: as on x86_64: `at` is in the kernel's window on physical
        // memory, in a frame of RAM live when the scheduler gathered it; a
        // read of reused memory only makes the kernel look again under the
        // lock. The device's write is seen through coherent DMA.
        let v = unsafe { w.at.read_volatile() };
        v & w.mask != w.seen
    })
}

/// Continue a CRC32C over `data`: with the processor's CRC32 instructions
/// where it has them (ID_AA64ISAR0_EL1.CRC32: every Armv8.1 core, most
/// Armv8.0 ones), eight bytes an instruction; else the portable code.
fn crc32c(state: u32, data: &[u8]) -> u32 {
    let isar0: u64;
    // SAFETY: reading an identification register has no effect.
    unsafe { asm!("mrs {}, id_aa64isar0_el1", out(reg) isar0, options(nomem, nostack, preserves_flags)) }
    if isar0 >> 16 & 0xF == 0 {
        return kernel_core::ramdisk::crc32c_update(state, data);
    }
    let (words, tail) = data.as_chunks::<8>();
    let mut c = state;
    for w in words {
        // SAFETY: CRC32CX only computes on registers; the processor has it.
        unsafe {
            asm!(".arch_extension crc", "crc32cx {c:w}, {c:w}, {v}", c = inout(reg) c, v = in(reg) u64::from_le_bytes(*w), options(pure, nomem, nostack))
        }
    }
    for &b in tail {
        // SAFETY: as above.
        unsafe {
            asm!(".arch_extension crc", "crc32cb {c:w}, {c:w}, {b:w}", c = inout(reg) c, b = in(reg) b as u32, options(pure, nomem, nostack))
        }
    }
    c
}

/// Isolating an address space (`Hal::isolate`) calls the firmware's
/// SMCCC_ARCH_WORKAROUND_1: set at boot on a core without FEAT_CSV2 whose
/// firmware offers it.
static SPECTRE_WORKAROUND: AtomicBool = AtomicBool::new(false);

/// What isolating an address space needs on this core, decided once at boot
/// (after PSCI is known): nothing where the core keeps branch predictions
/// apart by context (FEAT_CSV2, ID_AA64PFR0_EL1 bits 59:56); the firmware's
/// workaround where it offers one (SMCCC 1.1, ARCH_FEATURES says 0); else
/// the boot says isolation against branch-target injection is not there.
fn decide_isolation() {
    let pfr0: u64;
    // SAFETY: reading an identification register.
    unsafe { asm!("mrs {}, id_aa64pfr0_el1", out(reg) pfr0, options(nomem, nostack, preserves_flags)) }
    if pfr0 >> 56 & 0xF != 0 || !smp::psci_present() {
        if pfr0 >> 56 & 0xF == 0 {
            let _ = write!(
                Out,
                "WARNING: kernel: this core lacks FEAT_CSV2 and the firmware has no SMCCC: isolated address spaces are not protected from branch-target injection\r\n"
            );
        }
        return;
    }
    let version = smp::psci(0x8000_0000, 0, 0, 0); // SMCCC_VERSION
    if version >= 0x1_0001 && smp::psci(0x8000_0001, 0x8000_8000, 0, 0) == 0 {
        SPECTRE_WORKAROUND.store(true, Ordering::Relaxed);
    } else {
        let _ = write!(
            Out,
            "WARNING: kernel: this core lacks FEAT_CSV2 and the firmware has no workaround: isolated address spaces are not protected from branch-target injection\r\n"
        );
    }
}
