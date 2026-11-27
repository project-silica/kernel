//! Intel VT-d: what a device reaches by DMA (component C16).
//!
//! Every remapping unit the firmware describes (ACPI DMAR) translates from
//! boot on, through one root table the units of segment 0 share, so the
//! kernel need not know which unit a device sits behind. A device reaches
//! nothing until `DeviceAttach` points its context entry at the tables of
//! an address space made for devices, and nothing again once that address
//! space is gone. Units of other segments get an empty root table: their
//! devices reach nothing at all (attaching them is `Unsupported`).
//!
//! A device of segment 0 is attached only where a driven unit translates
//! it: a unit that covers every device no other unit names
//! (INCLUDE_PCI_ALL), or a device scope that names the device or the bridge
//! it is below (read from configuration space when it is attached). Any
//! other device reaches memory whatever it is attached to, so attaching it
//! is `Unsupported` too, and the device manager, which asks before it lets
//! a device master the bus, starts no driver that does DMA for it.
//!
//! Tables are in the legacy second-level format, which is the processor's
//! own radix with read and write bits, so `kernel-core` walks them with the
//! same code. A unit that translates 48 bits is given the address space's
//! root table; one that translates only 39 is given the table under the
//! root's entry 0, which `kernel-core` puts there when the address space is
//! made. Invalidations go through the queued interface and are waited for:
//! when `Unmap` returns, no unit holds the translation.
//!
//! The kernel's own memory for this is carved at boot: the root table, one
//! context table per bus that has an attached device (from a pool sized
//! by the MCFG's buses), and a queue page per unit.
//!
//! A refused access is recorded by the unit and raises its fault interrupt,
//! which says who was refused what and tells whoever attached the device;
//! the device is then held quiet until that report is taken (`fault`).
//!
//! Interrupts are remapped too, where every unit can (ECAP.IR, with
//! extended interrupt mode for x2APIC destinations): a device's message is
//! only an index into the kernel's table of [`IRT_ENTRIES`] entries, one per
//! interrupt line, and the unit delivers it only if the entry names that
//! device as its source ([`remap`]). Messages in the old format, which name
//! a processor and vector directly, are blocked. So a device can raise no
//! interrupt but the ones its driver bound, and none of another device's. The
//! I/O APICs' lines are entries as well, their source the I/O APIC the DMAR
//! table places. Units of other segments get a table of their own, empty.

use super::cpu::percpu;
use core::cell::UnsafeCell;
use core::sync::atomic::Ordering;
use kernel_core::boot::{BootInfo, RegionTable};
use kernel_core::hal::Hal as _;
use kernel_core::klog;
use kernel_core::lock::Lock;
use kernel_core::mem::{Paddr, Phys};
use silica_abi::Error;

const PAGE: u64 = 4096;
const MAX_UNITS: usize = 32;
/// Buses that can have an attached device when the firmware gives no MCFG
/// (legacy configuration access reaches 256). With one, the pool holds a
/// context table for every bus of segment 0 it names (`contexts`).
const CONTEXT_TABLES: u64 = 256;
/// Device address spaces that can have devices attached at one time.
const DOMAINS: usize = 256;
/// The vector the units' fault interrupts arrive at.
pub const FAULT_VECTOR: u64 = 0xF2;

const CAP: u64 = 0x08;
const ECAP: u64 = 0x10;
const GCMD: u64 = 0x18;
const GSTS: u64 = 0x1C;
const RTADDR: u64 = 0x20;
const FSTS: u64 = 0x34;
const FECTL: u64 = 0x38;
const FEDATA: u64 = 0x3C;
const FEADDR: u64 = 0x40;
const FEUADDR: u64 = 0x44;
const IQT: u64 = 0x88;
const IQA: u64 = 0x90;
const IRTA: u64 = 0xB8;
const ECAP_IR: u64 = 1 << 3;
const ECAP_EIM: u64 = 1 << 4;
const GCMD_IRE: u32 = 1 << 25;
const GCMD_SIRTP: u32 = 1 << 24;
const GCMD_CFI: u32 = 1 << 23;
/// Entries of the interrupt remapping table: one per line, one page.
pub const IRT_ENTRIES: usize = 256;
/// The table's size as IRTA says it: 2^(S + 1) entries.
const IRT_SIZE: u64 = 7;
/// I/O APICs whose source identifier the DMAR gives.
const MAX_IOAPICS: usize = 8;
const GCMD_TE: u32 = 1 << 31;
const GCMD_SRTP: u32 = 1 << 30;
const GCMD_QIE: u32 = 1 << 26;

#[derive(Clone, Copy)]
struct Unit {
    /// Physical address of the registers.
    base: u64,
    /// Its invalidation queue: 256 descriptors of 16 bytes.
    queue: u64,
    /// Offset of the fault recording registers, and how many there are.
    fault_regs: u64,
    fault_count: u64,
    /// The GCMD bits that stay set.
    gcmd: u32,
}

struct State {
    units: [Unit; MAX_UNITS],
    n: usize,
    /// The root table of segment 0, and the pool of context tables after it.
    root: u64,
    pool: u64,
    pool_used: u64,
    /// Context tables in the pool: one per bus of segment 0 (the MCFG).
    contexts: u64,
    /// A word the units write when a queued wait completes.
    status: u64,
    /// Whether the units walk four levels (48 bits) or three (39).
    four_levels: bool,
    large: bool,
    /// A unit caches missing translations: tell it about new ones too.
    caching: bool,
    top: u64,
    /// Domain identifiers in use, one bit each (identifier 0 is never used).
    domains: [u64; DOMAINS / 64],
    domain_limit: usize,
    /// Devices attached per bus, so releasing an address space looks only
    /// at buses that have any.
    per_bus: [u16; 256],
    /// The interrupt remapping table (0: interrupts are not remapped), and
    /// whether its destinations are 32-bit x2APIC identifiers.
    irt: u64,
    eime: bool,
    /// Each I/O APIC's identifier and source identifier, from the DMAR.
    ioapics: [(u8, u16); MAX_IOAPICS],
    ioapic_count: usize,
    /// Which devices of segment 0 the driven units translate
    /// (`kernel_core::x86::covered`): all of them (a unit with
    /// INCLUDE_PCI_ALL), else those their device scopes name.
    all_covered: bool,
    scopes: [kernel_core::x86::Scope; kernel_core::x86::MAX_SCOPES],
    nscopes: usize,
    /// Segment 0's configuration windows (MCFG: base, first and last bus),
    /// for the bridges a scope names: those inside the physical map.
    ecam: [(u64, u8, u8); kernel_core::x86::MAX_WINDOWS],
    windows: usize,
}

struct Racy(UnsafeCell<State>);
// SAFETY: filled by the boot processor before any other runs; afterwards
// only touched holding `LOCK`.
unsafe impl Sync for Racy {}
static STATE: Racy = Racy(UnsafeCell::new(State {
    units: [Unit { base: 0, queue: 0, fault_regs: 0, fault_count: 0, gcmd: 0 }; MAX_UNITS],
    n: 0,
    root: 0,
    pool: 0,
    pool_used: 0,
    contexts: 0,
    status: 0,
    four_levels: true,
    large: true,
    caching: false,
    top: 0,
    domains: [1, 0, 0, 0],
    domain_limit: DOMAINS,
    per_bus: [0; 256],
    irt: 0,
    eime: false,
    ioapics: [(0, 0); MAX_IOAPICS],
    ioapic_count: 0,
    all_covered: false,
    scopes: [kernel_core::x86::Scope { sub: false, start_bus: 0, path: [(0, 0); kernel_core::x86::MAX_PATH], len: 0 };
        kernel_core::x86::MAX_SCOPES],
    nscopes: 0,
    ecam: [(0, 0, 0); kernel_core::x86::MAX_WINDOWS],
    windows: 0,
}));
static LOCK: Lock = Lock::new();
/// `top` again, readable without the lock: 0 until the units translate.
static TOP: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static LARGE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
/// Interrupts are remapped: readable without the lock.
static REMAPPING: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

fn r32(phys: &Phys, base: u64, reg: u64) -> u32 {
    // SAFETY: a register of a remapping unit at the address the firmware
    // gave, inside the physical map (checked in `init`).
    unsafe { phys.ptr(base + reg).cast::<u32>().read_volatile() }
}
fn w32(phys: &Phys, base: u64, reg: u64, v: u32) {
    // SAFETY: as for `r32`.
    unsafe { phys.ptr(base + reg).cast::<u32>().write_volatile(v) }
}
fn r64(phys: &Phys, base: u64, reg: u64) -> u64 {
    // SAFETY: as for `r32`.
    unsafe { phys.ptr(base + reg).cast::<u64>().read_volatile() }
}
fn w64(phys: &Phys, base: u64, reg: u64, v: u64) {
    // SAFETY: as for `r32`.
    unsafe { phys.ptr(base + reg).cast::<u64>().write_volatile(v) }
}

/// One 64-bit word of a table the kernel carved for the units.
fn word(phys: &Phys, at: u64) -> *mut u64 {
    phys.ptr(at).cast::<u64>()
}

/// FSTS: the invalidation queue stopped at a descriptor it could not take
/// (IQE), a completion error (ICE), a device-TLB invalidation timed out
/// (ITE). Left set by `fault` for the wait that is held up by them.
const FSTS_QUEUE_ERRORS: u32 = 1 << 4 | 1 << 5 | 1 << 6;
/// The longest a unit may take to do what it is told: it takes
/// microseconds; this is a thousand times that and more.
const ANSWER_NS: u64 = 10_000_000;

/// Spin until `done`, which a working unit `u` makes true within
/// microseconds. A unit that reports a queue error, or does not answer
/// within `ANSWER_NS`, has failed: the kernel can no longer say what the
/// devices behind it reach, so the machine stops, saying which unit and
/// why (never a wait of minutes, every processor that needs the IOMMU
/// stuck behind its lock).
fn wait(phys: &Phys, u: &Unit, what: &str, done: impl Fn() -> bool) {
    let start = super::kernel().hal.now();
    while !done() {
        let status = r32(phys, u.base, FSTS);
        if status & FSTS_QUEUE_ERRORS != 0 {
            panic!("the IOMMU at {:#x} reports an invalidation queue error (fault status {status:#x}) during {what}", u.base);
        }
        if super::kernel().hal.now().saturating_sub(start) > ANSWER_NS {
            panic!("the IOMMU at {:#x} did not answer within 10 ms ({what})", u.base);
        }
        core::hint::spin_loop();
    }
}

/// Give unit `u` a one-shot command, or change a bit that stays, and wait
/// until its status shows it.
fn command(phys: &Phys, u: &Unit, bit: u32, what: &str) {
    w32(phys, u.base, GCMD, u.gcmd | bit);
    wait(phys, u, what, || r32(phys, u.base, GSTS) & bit != 0);
}

/// Queue descriptors on every unit, then wait until each has processed them.
fn invalidate(phys: &Phys, s: &State, descriptors: &[[u64; 2]]) {
    for u in &s.units[..s.n] {
        // SAFETY: the status word is the kernel's, written by the units
        // only when a wait descriptor completes.
        unsafe { word(phys, s.status).write_volatile(0) };
        let mut tail = r64(phys, u.base, IQT) >> 4 & 255;
        let wait_desc = [5 | 1 << 5 | 1u64 << 32, s.status];
        for d in descriptors.iter().chain(core::iter::once(&wait_desc)) {
            // SAFETY: descriptor `tail` of the unit's queue page; the unit
            // reads it only after the tail register moves past it.
            unsafe {
                word(phys, u.queue + tail * 16).write_volatile(d[0]);
                word(phys, u.queue + tail * 16 + 8).write_volatile(d[1]);
            }
            tail = (tail + 1) & 255;
        }
        w64(phys, u.base, IQT, tail << 4);
        // SAFETY: as above.
        wait(phys, u, "a queued invalidation", || unsafe { word(phys, s.status).read_volatile() as u32 } == 1);
    }
}

/// Forget what is cached of the context entry of `sid`.
fn context_of(did: u64, sid: u64) -> [u64; 2] {
    [1 | 3 << 4 | did << 16 | sid << 32, 0]
}
/// Forget every cached translation of domain `did`.
fn domain(did: u64) -> [u64; 2] {
    [2 | 2 << 4 | 1 << 6 | 1 << 7 | did << 16, 0]
}

/// The two words kept per address space in the half of its header page the
/// architecture owns: its domain identifier (0: none) and how many devices
/// are attached to it.
fn header(phys: &Phys, root: Paddr) -> *mut u64 {
    word(phys, root - PAGE + 2048)
}

/// Find the units and make every one translate, with nothing attached:
/// from here on no device reaches memory until its driver says what.
/// Boot processor, once, before the root task gets the rest of memory.
pub fn init(bi: &BootInfo, phys: &Phys, table: &mut RegionTable, reserved: &mut [(u64, u64)]) {
    let Some(dmar) = super::acpi_table(bi, phys, b"DMAR") else { return };
    // SAFETY: boot processor, alone.
    let s = unsafe { &mut *STATE.0.get() };
    let (mut o, mut other_segments) = (48, [0u64; MAX_UNITS]);
    let mut others = 0;
    // A unit of segment 0 that cannot be driven: its devices would reach
    // memory whatever their drivers attach them to.
    let mut undriven = None;
    while o + 4 <= dmar.len() {
        let field = |at: usize, n: usize| dmar.get(o + at..o + at + n).map(|b| b.iter().rev().fold(0u64, |v, x| v << 8 | *x as u64));
        let (Some(ty), Some(len)) = (field(0, 2), field(2, 2)) else { break };
        if len < 4 || o + len as usize > dmar.len() {
            break;
        }
        if let (0, Some(segment), Some(base)) = (ty, field(6, 2), field(8, 8)) {
            use kernel_core::x86::{UnitUse, unit_use};
            let usage = unit_use(base, segment, bi.phys_map_len, s.n, MAX_UNITS);
            if usage == UnitUse::Drive {
                // Which devices it translates: every one no other unit
                // names (INCLUDE_PCI_ALL), else those its scopes name.
                if field(4, 1).unwrap_or(0) as u8 & kernel_core::x86::INCLUDE_PCI_ALL != 0 {
                    s.all_covered = true;
                }
                s.units[s.n].base = base;
                s.n += 1;
                // Its device scopes: an I/O APIC's (type 3) gives the source
                // identifier its interrupts carry.
                let mut at = 16;
                while let (Some(kind), Some(n)) = (field(at, 1), field(at + 1, 1)) {
                    if n < 6 || at + n as usize > len as usize {
                        break;
                    }
                    // A PCI endpoint (1) or sub-hierarchy (2): its start bus
                    // and path of (device, function) pairs. One past what is
                    // kept, or with a longer path, is not kept: the devices
                    // only it names are not confined (refused).
                    if let (1 | 2, Some(bus)) = (kind, field(at + 5, 1)) {
                        let elems = (n as usize - 6) / 2;
                        if s.nscopes < kernel_core::x86::MAX_SCOPES && (1..=kernel_core::x86::MAX_PATH).contains(&elems) {
                            let mut scope =
                                kernel_core::x86::Scope { sub: kind == 2, start_bus: bus as u8, len: elems as u8, ..Default::default() };
                            for (i, p) in scope.path[..elems].iter_mut().enumerate() {
                                *p = (field(at + 6 + 2 * i, 1).unwrap_or(0) as u8, field(at + 7 + 2 * i, 1).unwrap_or(0) as u8);
                            }
                            s.scopes[s.nscopes] = scope;
                            s.nscopes += 1;
                        } else {
                            klog!(
                                super::kernel(),
                                "kernel: a device scope of the IOMMU at {base:#x} has an empty path or one longer than {} elements, or is past the {} kept; the devices only it names are not confined",
                                kernel_core::x86::MAX_PATH,
                                kernel_core::x86::MAX_SCOPES
                            );
                        }
                    }
                    if let (3, Some(id), Some(bus), Some(dev), Some(func)) =
                        (kind, field(at + 4, 1), field(at + 5, 1), field(at + 6, 1), field(at + 7, 1))
                        && n >= 8
                        && s.ioapic_count < MAX_IOAPICS
                    {
                        s.ioapics[s.ioapic_count] = (id as u8, (bus << 8 | dev << 3 | func) as u16);
                        s.ioapic_count += 1;
                    }
                    at += n as usize;
                }
            } else if usage == UnitUse::Undriven {
                undriven = Some(base);
            } else if usage == UnitUse::Block && others < MAX_UNITS {
                other_segments[others] = base;
                others += 1;
            } else {
                klog!(
                    super::kernel(),
                    "kernel: an IOMMU of segment {segment} at {base:#x} is out of reach; that segment's devices are not confined"
                );
            }
        }
        o += len as usize;
    }
    // Confinement is promised for every device or for none: with a unit of
    // segment 0 left undriven, or none at all, the machine says it cannot
    // confine DMA (`dma_top` 0, every attach `Unsupported`) rather than
    // attach devices that reach memory anyway.
    if let Some(base) = undriven {
        klog!(
            super::kernel(),
            "kernel: the IOMMU at {base:#x} cannot be driven (out of reach, or more than {MAX_UNITS} units): no device is confined"
        );
        s.n = 0;
        return;
    }
    if s.n == 0 {
        if others != 0 {
            klog!(super::kernel(), "kernel: no IOMMU covers segment 0: no device is confined");
        }
        return;
    }
    // What every unit of segment 0 can do.
    let (mut widths, mut mgaw, mut remap) = (0b110u64, 64, true);
    for u in &mut s.units[..s.n] {
        let (cap, ecap) = (r64(phys, u.base, CAP), r64(phys, u.base, ECAP));
        widths &= cap >> 8 & 0x1F;
        mgaw = mgaw.min((cap >> 16 & 0x3F) + 1);
        s.large &= cap >> 34 & 1 != 0;
        s.caching |= cap >> 7 & 1 != 0;
        s.domain_limit = s.domain_limit.min(1 << (4 + 2 * (cap & 7)));
        (u.fault_regs, u.fault_count) = ((cap >> 24 & 0x3FF) * 16, (cap >> 40 & 0xFF) + 1);
        if ecap & 2 == 0 || u.fault_regs + u.fault_count * 16 > PAGE {
            widths = 0;
        }
        remap &= ecap & ECAP_IR != 0 && ecap & ECAP_EIM != 0;
    }
    if widths == 0 {
        // Too old a unit (no queued invalidation, or neither 39 nor 48
        // bits): the machine runs as one without an IOMMU, and says so.
        klog!(super::kernel(), "kernel: this machine's IOMMU is of a kind Silica does not drive; devices are not confined");
        s.n = 0;
        return;
    }
    s.four_levels = widths & 0b100 != 0;
    // Root table, an empty root for other segments, the wait status word's
    // page, the interrupt remapping table and an empty one for other
    // segments, the context tables, and a queue page per unit.
    // A context table for every bus segment 0 has (the MCFG), not a fixed
    // number: a server with devices past bus 64 had them refused.
    s.contexts =
        super::acpi_table(bi, phys, b"MCFG").and_then(|t| silica_formats::acpi::mcfg_buses(t, 0)).unwrap_or(CONTEXT_TABLES).clamp(1, 256);
    // Where segment 0's bridges are read (a scope's path crosses them, and a
    // sub-hierarchy is the buses below one): the MCFG's windows, each used
    // only where it lies inside the physical map. Without one, no device
    // below a bridge a scope names is confined.
    match super::acpi_table(bi, phys, b"MCFG") {
        None => klog!(super::kernel(), "kernel: no MCFG: no device below a bridge an IOMMU names is confined"),
        Some(t) => {
            for w in silica_formats::acpi::mcfg_windows(t, 0) {
                if !kernel_core::x86::ecam_window_fits(w, bi.phys_map_len) {
                    klog!(
                        super::kernel(),
                        "kernel: the MCFG window at {:#x} (buses {:#x} to {:#x}) is outside the physical map; devices below its bridges are not confined",
                        w.0,
                        w.1,
                        w.2
                    );
                } else if s.windows == kernel_core::x86::MAX_WINDOWS {
                    klog!(
                        super::kernel(),
                        "kernel: segment 0 has more than {} MCFG windows; devices below the bridges of the rest are not confined",
                        s.windows
                    );
                } else {
                    s.ecam[s.windows] = w;
                    s.windows += 1;
                }
            }
        }
    }
    if !s.all_covered && s.nscopes == 0 {
        klog!(super::kernel(), "kernel: the IOMMUs name no device and cover none of the rest: no device is confined");
    }
    let pages = 5 + s.contexts + (s.n + others) as u64;
    let Some(mem) = table.carve(pages * PAGE, PAGE, 1 << 20, bi.phys_map_len) else {
        klog!(super::kernel(), "kernel: no memory for the IOMMU's tables; devices are not confined");
        s.n = 0;
        return;
    };
    phys.zero(mem, pages * PAGE);
    (s.root, s.status, s.pool) = (mem, mem + 2 * PAGE, mem + 5 * PAGE);
    let queues = s.pool + s.contexts * PAGE;
    // A unit of another segment remaps nothing: its devices' identifiers
    // could be those of segment 0's.
    let (irt, empty_irt) = if remap { (Some(mem + 3 * PAGE), Some(mem + 4 * PAGE)) } else { (None, None) };
    for (i, base) in other_segments[..others].iter().enumerate() {
        let mut u = Unit { base: *base, queue: queues + (s.n + i) as u64 * PAGE, fault_regs: 0, fault_count: 0, gcmd: 0 };
        start(phys, &mut u, mem + PAGE, s.status, empty_irt);
    }
    let boot_apic = percpu().apic_id as u64;
    for i in 0..s.n {
        super::check_uncached(phys, s.units[i].base, "an IOMMU");
        let mut u = s.units[i];
        u.queue = queues + i as u64 * PAGE;
        start(phys, &mut u, s.root, s.status, irt);
        // Its fault interrupt, to the boot processor.
        if boot_apic < 0xFF {
            w32(phys, u.base, FEDATA, FAULT_VECTOR as u32);
            w32(phys, u.base, FEADDR, 0xFEE0_0000 | (boot_apic as u32) << 12);
            w32(phys, u.base, FEUADDR, 0);
            w32(phys, u.base, FECTL, 0);
        }
        s.units[i] = u;
    }
    for (slot, u) in
        reserved.iter_mut().filter(|r| r.1 == 0).zip(s.units[..s.n].iter().map(|u| u.base).chain(other_segments[..others].iter().copied()))
    {
        *slot = (u, u + PAGE);
    }
    s.top = 1 << mgaw.min(if s.four_levels { 48 } else { 39 });
    TOP.store(s.top, Ordering::Release);
    LARGE.store(s.large, Ordering::Release);
    if let Some(table) = irt {
        (s.irt, s.eime) = (table, true);
        REMAPPING.store(true, Ordering::Release);
        klog!(super::kernel(), "kernel: interrupts are remapped: a device raises only the interrupts its driver bound");
        if s.ioapic_count == 0 {
            klog!(super::kernel(), "kernel: the DMAR names no I/O APIC: its lines are remapped without checking who raised them");
        }
    } else {
        klog!(super::kernel(), "kernel: this machine's IOMMU does not remap interrupts: a device can raise any interrupt, as anyone");
    }
}

/// Point unit `u` at root table `root`, switch its queue on, drop whatever
/// the firmware left cached, and make it translate; with `irt`, remap
/// interrupts through that table and block the old format.
fn start(phys: &Phys, u: &mut Unit, root: u64, status: u64, irt: Option<u64>) {
    // Whatever the firmware left on: translation off first, so the root
    // table can change without a window of stale entries. Interrupt
    // remapping, if the firmware uses it and this kernel does not, stays.
    u.gcmd = if irt.is_some() { 0 } else { r32(phys, u.base, GSTS) & (GCMD_IRE | GCMD_CFI) };
    w32(phys, u.base, GCMD, u.gcmd);
    let off = GCMD_TE | GCMD_QIE | if irt.is_some() { GCMD_IRE } else { 0 };
    wait(phys, u, "switching translation off", || r32(phys, u.base, GSTS) & off == 0);
    w64(phys, u.base, IQT, 0);
    w64(phys, u.base, IQA, u.queue);
    u.gcmd |= GCMD_QIE;
    command(phys, u, GCMD_QIE, "switching its queue on");
    if let Some(table) = irt {
        // Extended interrupt mode: destinations are x2APIC identifiers.
        w64(phys, u.base, IRTA, table | 1 << 11 | IRT_SIZE);
        command(phys, u, GCMD_SIRTP, "taking the interrupt remapping table");
        let one = State { units: [*u; MAX_UNITS], n: 1, status, ..empty() };
        invalidate(phys, &one, &[[4, 0]]);
        // On, with the old format blocked (CFI stays clear).
        u.gcmd |= GCMD_IRE;
        command(phys, u, GCMD_IRE, "switching interrupt remapping on");
    }
    w64(phys, u.base, RTADDR, root);
    command(phys, u, GCMD_SRTP, "taking the root table");
    // Global invalidation of the context cache and the IOTLB.
    let one = State { units: [*u; MAX_UNITS], n: 1, status, ..empty() };
    invalidate(phys, &one, &[[1 | 1 << 4, 0], [2 | 1 << 4 | 1 << 6 | 1 << 7, 0]]);
    u.gcmd |= GCMD_TE;
    command(phys, u, GCMD_TE, "switching translation on");
}

const fn empty() -> State {
    State {
        units: [Unit { base: 0, queue: 0, fault_regs: 0, fault_count: 0, gcmd: 0 }; MAX_UNITS],
        n: 0,
        root: 0,
        pool: 0,
        pool_used: 0,
        contexts: 0,
        status: 0,
        four_levels: true,
        large: true,
        caching: false,
        top: 0,
        domains: [1, 0, 0, 0],
        domain_limit: DOMAINS,
        per_bus: [0; 256],
        irt: 0,
        eime: false,
        ioapics: [(0, 0); MAX_IOAPICS],
        ioapic_count: 0,
        all_covered: false,
        scopes: [kernel_core::x86::Scope { sub: false, start_bus: 0, path: [(0, 0); kernel_core::x86::MAX_PATH], len: 0 };
            kernel_core::x86::MAX_SCOPES],
        nscopes: 0,
        ecam: [(0, 0, 0); kernel_core::x86::MAX_WINDOWS],
        windows: 0,
    }
}

/// One past the highest device address; 0 when DMA is not confined.
pub fn top() -> u64 {
    TOP.load(Ordering::Acquire)
}

/// Whether 2 MiB is mapped with one entry.
pub fn large() -> bool {
    LARGE.load(Ordering::Acquire)
}

/// Whether interrupts are remapped: a message line is then raised through
/// [`remap`], and an I/O APIC line too.
pub fn remapping() -> bool {
    REMAPPING.load(Ordering::Acquire)
}

/// The source identifier of the I/O APIC with identifier `id`, as the
/// DMAR gives it.
pub fn ioapic_source(id: u8) -> Option<u16> {
    LOCK.lock();
    // SAFETY: under `LOCK`.
    let s = unsafe { &*STATE.0.get() };
    let found = s.ioapics[..s.ioapic_count].iter().find(|(i, _)| *i == id).map(|(_, sid)| *sid);
    LOCK.unlock();
    found
}

/// Make remapping entry `line` deliver `vector` to the processor with x2APIC
/// identifier `apic_id`, level- or edge-triggered, and only when the
/// device with source identifier `source` raises it (`None`: whoever does;
/// for an I/O APIC the DMAR does not place). Returns the entry's handle
/// for the message or the I/O APIC's entry, once no unit holds the old
/// entry; `None` when interrupts are not remapped.
pub fn remap(phys: &Phys, line: usize, apic_id: u32, vector: u8, level: bool, source: Option<u16>) -> Option<u16> {
    if !remapping() || line >= IRT_ENTRIES {
        return None;
    }
    LOCK.lock();
    // SAFETY: under `LOCK`.
    let s = unsafe { &*STATE.0.get() };
    let at = s.irt + line as u64 * 16;
    // Present, physical destination, fixed delivery; the source checked
    // in full (SVT 1, SQ 0).
    let low = 1 | (level as u64) << 4 | (vector as u64) << 16 | (apic_id as u64) << 32;
    let high = source.map_or(0, |sid| sid as u64 | 1 << 18);
    // SAFETY: entry `line` of the kernel's table. Not present while it
    // changes, so no unit reads half of the old and half of the new.
    unsafe {
        word(phys, at).write_volatile(0);
        word(phys, at + 8).write_volatile(high);
        word(phys, at).write_volatile(low);
    }
    // The entry, by index, out of every unit's cache.
    invalidate(phys, s, &[[4 | 1 << 4 | (line as u64) << 32, 0]]);
    LOCK.unlock();
    Some(line as u16)
}

/// Make remapping entry `line`, present already, deliver to the processor
/// with x2APIC identifier `apic_id`, everything else about it unchanged:
/// how a line follows the thread that waits on it without its device (or
/// its I/O APIC entry, which names the remapping entry) being written.
/// `false` when interrupts are not remapped or the entry is not in use.
pub fn retarget(phys: &Phys, line: usize, apic_id: u32) -> bool {
    if !remapping() || line >= IRT_ENTRIES {
        return false;
    }
    LOCK.lock();
    // SAFETY: under `LOCK`.
    let s = unsafe { &*STATE.0.get() };
    let at = s.irt + line as u64 * 16;
    // SAFETY: entry `line` of the kernel's table. The destination is the
    // upper half of its first word, which one aligned store replaces whole:
    // a unit reads the old destination or the new, never a mixture.
    let moved = unsafe {
        let low = word(phys, at).read_volatile();
        if low & 1 != 0 {
            word(phys, at).write_volatile(low & 0xFFFF_FFFF | (apic_id as u64) << 32);
        }
        low & 1 != 0
    };
    if moved {
        invalidate(phys, s, &[[4 | 1 << 4 | (line as u64) << 32, 0]]);
    }
    LOCK.unlock();
    moved
}

/// `Hal::io_attach`.
pub fn attach(phys: &Phys, id: u64, root: Paddr) -> Result<Paddr, Error> {
    if top() == 0 || id >> 16 != 0 {
        return Err(Error::Unsupported);
    }
    LOCK.lock();
    // SAFETY: under `LOCK`.
    let s = unsafe { &mut *STATE.0.get() };
    // A device no driven unit translates reaches memory whatever it is
    // attached to: it is attached to nothing (`kernel_core::x86::covered`).
    let translated = {
        let st: &State = s;
        kernel_core::x86::covered(id as u32, st.all_covered, &st.scopes[..st.nscopes], |bus, devfn| bridge(phys, st, bus, devfn))
    };
    let r = if translated { attach_locked(phys, s, id, root) } else { Err(Error::Unsupported) };
    LOCK.unlock();
    r
}

/// A bridge's secondary and subordinate bus numbers, read from segment 0's
/// configuration space where `init` found it (`State::ecam`,
/// `kernel_core::x86::ecam_address`): `None` for a function that is not a
/// bridge, absent, or on a bus outside every window.
fn bridge(phys: &Phys, s: &State, bus: u8, devfn: u8) -> Option<(u8, u8)> {
    let at = kernel_core::x86::ecam_address(&s.ecam[..s.windows], bus, devfn)?;
    // SAFETY: a dword of segment 0's configuration space, in a window `init`
    // checked lies inside the physical map; reading it has no effect.
    let read = |reg: u64| unsafe { phys.ptr(at + reg).cast::<u32>().read_volatile() };
    if read(0) & 0xFFFF == 0xFFFF || (read(0x0C) >> 16) & 0x7F != 1 {
        return None;
    }
    let buses = read(0x18);
    Some(((buses >> 8) as u8, (buses >> 16) as u8))
}

fn attach_locked(phys: &Phys, s: &mut State, id: u64, root: Paddr) -> Result<Paddr, Error> {
    let (bus, devfn) = (id >> 8, id & 0xFF);
    // The table a context entry points at for an address space, and back.
    let skip = if s.four_levels { 0 } else { PAGE };
    // SAFETY (all table accesses below): the root table, the context tables
    // and the header words are kernel memory nobody else writes; the units
    // read them, and are told to forget what they read where it changes.
    unsafe {
        let root_entry = word(phys, s.root + bus * 16);
        if *root_entry & 1 == 0 {
            if root == 0 {
                return Ok(0);
            }
            if s.pool_used == s.contexts {
                return Err(Error::NoMemory);
            }
            let page = s.pool + s.pool_used * PAGE;
            s.pool_used += 1;
            root_entry.write_volatile(page | 1);
            if s.caching {
                invalidate(phys, s, &[[1 | 1 << 4, 0]]);
            }
        }
        let entry = word(phys, (*root_entry & !0xFFF) + devfn * 16);
        let (lo, hi) = (*entry, *entry.add(1));
        let old = if lo & 1 != 0 { (lo & !0xFFF) - skip } else { 0 };
        if old == root {
            return Ok(old);
        }
        // What can fail comes before anything changes.
        let mut did = 0;
        if root != 0 {
            did = *header(phys, root);
            if did == 0 {
                let limit = s.domain_limit.min(DOMAINS);
                did = (1..limit).find(|d| s.domains[d / 64] & 1 << (d % 64) == 0).ok_or(Error::NoMemory)? as u64;
                s.domains[did as usize / 64] |= 1 << (did % 64);
                *header(phys, root) = did;
            }
        }
        if old != 0 {
            let old_did = hi >> 8 & 0xFFFF;
            entry.write_volatile(0);
            entry.add(1).write_volatile(0);
            invalidate(phys, s, &[context_of(old_did, id), domain(old_did)]);
            leave(phys, s, old, old_did);
            s.per_bus[bus as usize] -= 1;
        }
        if root != 0 {
            entry.add(1).write_volatile(if s.four_levels { 2 } else { 1 } | did << 8);
            entry.write_volatile((root + skip) | 1);
            if s.caching {
                invalidate(phys, s, &[context_of(0, id), domain(did)]);
            }
            *header(phys, root).add(1) += 1;
            s.per_bus[bus as usize] += 1;
        }
        listen(phys, s);
        Ok(old)
    }
}

/// One device fewer is attached to `root`; with the last, its domain
/// identifier is free for another address space.
///
/// # Safety
/// As for the table accesses of `attach_locked`.
unsafe fn leave(phys: &Phys, s: &mut State, root: Paddr, did: u64) {
    // SAFETY: the caller's.
    unsafe {
        let h = header(phys, root);
        *h.add(1) -= 1;
        if *h.add(1) == 0 {
            *h = 0;
            s.domains[did as usize / 64] &= !(1 << (did % 64));
        }
    }
}

/// `Hal::io_flush`: mappings were removed from `root`.
pub fn flush(phys: &Phys, root: Paddr) {
    LOCK.lock();
    // SAFETY: under `LOCK`; the header word is the kernel's.
    let (s, did) = unsafe { (&*STATE.0.get(), *header(phys, root)) };
    if did != 0 {
        invalidate(phys, s, &[domain(did)]);
    }
    LOCK.unlock();
}

/// `Hal::io_mapped`: a unit that caches missing translations is told.
pub fn mapped(phys: &Phys, root: Paddr) {
    // SAFETY: `caching` is set at boot and only read afterwards.
    if unsafe { (*STATE.0.get()).caching } {
        flush(phys, root);
    }
}

/// `Hal::io_release`: every device attached to `root` reaches nothing.
pub fn release(phys: &Phys, root: Paddr) {
    LOCK.lock();
    // SAFETY: under `LOCK`; table accesses as in `attach_locked`.
    unsafe {
        let s = &mut *STATE.0.get();
        let did = *header(phys, root);
        let target = root + if s.four_levels { 0 } else { PAGE };
        for bus in 0..256u64 {
            if s.per_bus[bus as usize] == 0 || did == 0 {
                continue;
            }
            let table = *word(phys, s.root + bus * 16) & !0xFFF;
            for devfn in 0..256u64 {
                let entry = word(phys, table + devfn * 16);
                if *entry & 1 != 0 && *entry & !0xFFF == target {
                    entry.write_volatile(0);
                    entry.add(1).write_volatile(0);
                    invalidate(phys, s, &[context_of(did, bus << 8 | devfn)]);
                    s.per_bus[bus as usize] -= 1;
                    leave(phys, s, root, did);
                }
            }
        }
        if did != 0 {
            invalidate(phys, s, &[domain(did)]);
        }
        listen(phys, s);
    }
    LOCK.unlock();
}

/// The devices a fault interrupt found refused, by source identifier (at
/// most eight distinct a pass; more are counted in the log's line only).
pub struct Refused {
    pub sids: [u16; 8],
    pub n: usize,
}

/// The console's ration of lines about refused accesses (the kernel log's
/// rate: a serial port takes so many a second, whoever causes them).
static FAULT_RATE: kernel_core::log::Rate = kernel_core::log::Rate::new();

/// A unit's fault interrupt: say which device was refused what, and return
/// the devices, whose attachers are told (`deliver`). Runs with or without
/// the processor's lock; it touches only the units' registers and tables.
///
/// What is reported is then held quiet until someone takes the report,
/// which the kernel is told of ([`taken`]), never for a time: a device
/// refused a memory access has its fault processing disabled in its own
/// context entry (it is still refused, and nobody else's reports stop);
/// what a unit cannot hold per device (a refused interrupt message, a
/// device on a bus with no context table) masks that unit's fault
/// interrupt. Before, ten lines in a second held every unit's fault
/// interrupt off for the rest of it, by the timer.
pub fn fault(phys: &Phys, now: u64) -> Refused {
    let mut out = Refused { sids: [0; 8], n: 0 };
    LOCK.lock();
    // SAFETY: under `LOCK`.
    let s = unsafe { &*STATE.0.get() };
    for u in &s.units[..s.n] {
        // Until a pass finds no fault recorded: one recorded while an
        // earlier record was still pending raised no interrupt of its own
        // (the unit interrupts when its first pending fault appears).
        let mut found = true;
        while found {
            found = false;
            for i in 0..u.fault_count {
                let at = u.fault_regs + i * 16;
                let hi = r64(phys, u.base, at + 8);
                if hi >> 63 == 0 {
                    continue;
                }
                found = true;
                let lo = r64(phys, u.base, at);
                let (address, sid, reason, read) = (lo & !0xFFF, hi & 0xFFFF, hi >> 32 & 0xFF, hi >> 62 & 1 != 0);
                w64(phys, u.base, at + 8, 1 << 63);
                if !out.sids[..out.n].contains(&(sid as u16)) && out.n < out.sids.len() {
                    out.sids[out.n] = sid as u16;
                    out.n += 1;
                }
                // SAFETY: under `LOCK`; table accesses as in `attach_locked`.
                if reason >= 0x20 || !unsafe { quiet(phys, s, sid) } {
                    w32(phys, u.base, FECTL, 1 << 31);
                }
                let Some(missed) = FAULT_RATE.allow(now) else { continue };
                let (b, d, f) = (sid >> 8, sid >> 3 & 31, sid & 7);
                let k = super::kernel();
                if missed != 0 {
                    klog!(k, "kernel: {missed} refused device accesses were not reported (more than ten a second)");
                }
                if reason == 0x25 {
                    klog!(
                        k,
                        "kernel: device {b:02x}:{d:02x}.{f} was refused an interrupt in the old format (IOMMU fault reason {reason:#x})"
                    );
                } else if reason >= 0x20 {
                    // An interrupt request: the entry it named is in the
                    // top bits of the record (VT-d 7.2.2).
                    klog!(
                        k,
                        "kernel: device {b:02x}:{d:02x}.{f} was refused interrupt entry {} (IOMMU fault reason {reason:#x})",
                        lo >> 48
                    );
                } else {
                    let what = if read { "read" } else { "write" };
                    klog!(k, "kernel: device {b:02x}:{d:02x}.{f} was refused a {what} at {address:#x} (IOMMU fault reason {reason})");
                }
            }
        }
        // Overflow and pending bits are cleared by writing them back. A
        // unit whose records were all full dropped the faults after them
        // (QEMU's has one record): said, so a missing line is explained.
        let status = r32(phys, u.base, FSTS);
        if status & 1 != 0 && FAULT_RATE.allow(now).is_some() {
            klog!(super::kernel(), "kernel: the IOMMU's fault records overflowed: refused accesses went unreported");
        }
        // Only the fault records' overflow is cleared here: a queue error
        // is left for the wait it holds up, which says what failed.
        w32(phys, u.base, FSTS, status & !FSTS_QUEUE_ERRORS);
    }
    LOCK.unlock();
    out
}

/// Disable fault processing for device `sid`, in its context entry (the
/// bit counts whether or not the entry is present, and goes with the entry
/// when the device is attached anew or detached): the unit still refuses
/// the device and records nothing. `false` when the device's bus has no
/// context table to say it in.
///
/// # Safety
/// As for the table accesses of `attach_locked`.
unsafe fn quiet(phys: &Phys, s: &State, sid: u64) -> bool {
    // SAFETY: the caller's.
    unsafe {
        let root_entry = *word(phys, s.root + (sid >> 8) * 16);
        if root_entry & 1 == 0 {
            return false;
        }
        let entry = word(phys, (root_entry & !0xFFF) + (sid & 0xFF) * 16);
        let lo = *entry;
        entry.write_volatile(lo | 2);
        if lo & 1 != 0 || s.caching {
            invalidate(phys, s, &[context_of(*entry.add(1) >> 8 & 0xFFFF, sid)]);
        }
        true
    }
}

/// The units' fault interrupts are taken again: something changed that
/// makes their reports news (a report was taken, a device was attached or
/// its space went). One masked meanwhile interrupts at once if it holds a
/// record.
fn listen(phys: &Phys, s: &State) {
    for u in &s.units[..s.n] {
        w32(phys, u.base, FECTL, 0);
    }
}

/// `Hal::io_taken`: the holder of `root` has read its count of refusals;
/// its devices' next refusals are reported again.
pub fn taken(phys: &Phys, root: Paddr) {
    LOCK.lock();
    // SAFETY: under `LOCK`; table accesses as in `attach_locked`.
    unsafe {
        let s = &*STATE.0.get();
        let target = root + if s.four_levels { 0 } else { PAGE };
        for bus in (0..256u64).filter(|b| s.per_bus[*b as usize] != 0) {
            let table = *word(phys, s.root + bus * 16) & !0xFFF;
            for devfn in 0..256u64 {
                let entry = word(phys, table + devfn * 16);
                if *entry & 3 == 3 && *entry & !0xFFF == target {
                    entry.write_volatile(*entry & !2);
                    invalidate(phys, s, &[context_of(*entry.add(1) >> 8 & 0xFFFF, bus << 8 | devfn)]);
                }
            }
        }
        listen(phys, s);
    }
    LOCK.unlock();
}

/// The device address space device `sid` is attached to now, by the root
/// `Hal::io_attach` named it with (0: none).
pub fn space_of(phys: &Phys, sid: u16) -> Paddr {
    if top() == 0 {
        return 0;
    }
    LOCK.lock();
    // SAFETY: under `LOCK`.
    let s = unsafe { &*STATE.0.get() };
    let skip = if s.four_levels { 0 } else { PAGE };
    let (bus, devfn) = (sid as u64 >> 8, sid as u64 & 0xFF);
    // SAFETY: the root and context tables are the kernel's (see `attach_locked`).
    let root = unsafe {
        let root_entry = *word(phys, s.root + bus * 16);
        if root_entry & 1 == 0 {
            0
        } else {
            let lo = *word(phys, (root_entry & !0xFFF) + devfn * 16);
            if lo & 1 != 0 { (lo & !0xFFF) - skip } else { 0 }
        }
    };
    LOCK.unlock();
    root
}

/// Remapping entry `line` delivers nothing from now on (its line was
/// unbound): a device that writes its message is refused, and the entry
/// is out of every unit's cache when this returns.
pub fn unremap(phys: &Phys, line: usize) {
    if !remapping() || line >= IRT_ENTRIES {
        return;
    }
    LOCK.lock();
    // SAFETY: under `LOCK`.
    let s = unsafe { &*STATE.0.get() };
    let at = s.irt + line as u64 * 16;
    // The entry stays present, its source check and destination kept, and
    // delivers to the discard vector: the line is gone, but its device may
    // still have the vector unmasked (its driver was ended, and nothing has
    // reached the device yet). A message through a not-present entry is a
    // fault the unit records for every message, and QEMU's VT-d model,
    // updating KVM's route for a vector still in use, aborted on it (about
    // one `recovery` run in five).
    // SAFETY: entry `line` of the kernel's table; one aligned store changes
    // the vector, so a unit reads the old entry or the parked one.
    unsafe {
        let low = word(phys, at).read_volatile();
        if low & 1 != 0 {
            word(phys, at).write_volatile(low & !(0xFF << 16) & !(1 << 4) | DISCARD_VECTOR << 16);
        }
    }
    invalidate(phys, s, &[[4 | 1 << 4 | (line as u64) << 32, 0]]);
    LOCK.unlock();
}

/// Hold the message line `line` off (its entry delivers to the discard
/// vector) or let it interrupt again at its own vector: how a line that fires
/// again before its waiter took its news stops costing a processor. One
/// aligned store changes the vector; the entry stays present.
pub fn hold(phys: &Phys, line: usize, held: bool) {
    if !remapping() || line >= IRT_ENTRIES {
        return;
    }
    LOCK.lock();
    // SAFETY: under `LOCK`.
    let s = unsafe { &*STATE.0.get() };
    let at = s.irt + line as u64 * 16;
    let vector = if held { DISCARD_VECTOR } else { super::ioapic::FIRST_VECTOR + line as u64 };
    // SAFETY: entry `line` of the kernel's table; only its vector changes.
    let present = unsafe {
        let low = word(phys, at).read_volatile();
        if low & 1 != 0 {
            word(phys, at).write_volatile(low & !(0xFF << 16) | vector << 16);
        }
        low & 1 != 0
    };
    if present {
        invalidate(phys, s, &[[4 | 1 << 4 | (line as u64) << 32, 0]]);
    }
    LOCK.unlock();
}

/// The vector a released line's remapping entry delivers to ([`unremap`]):
/// counted and acknowledged, never anyone's line.
pub const DISCARD_VECTOR: u64 = 0xF3;
static DISCARDED: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// A device raised a line that was released: counted.
pub fn discarded() {
    DISCARDED.fetch_add(1, Ordering::Relaxed);
}

/// How many messages released lines have delivered since boot.
pub fn discarded_count() -> u64 {
    DISCARDED.load(Ordering::Relaxed)
}
