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
//! Tables are in the legacy second-level format, which is the processor's
//! own radix with read and write bits, so `kernel-core` walks them with the
//! same code. A unit that translates 48 bits is given the address space's
//! root table; one that translates only 39 is given the table under the
//! root's entry 0, which `kernel-core` puts there when the address space is
//! made. Invalidations go through the queued interface and are waited for:
//! when `Unmap` returns, no unit holds the translation.
//!
//! The kernel's own memory for this is carved at boot: the root table, one
//! context table per bus that has an attached device (from a pool of
//! [`CONTEXT_TABLES`]), and a queue page per unit.
//!
//! A refused access is recorded by the unit and raises its fault interrupt,
//! which logs who was refused what (the first [`FAULT_LINES`] times).

use super::cpu::percpu;
use core::cell::UnsafeCell;
use core::fmt::Write;
use core::sync::atomic::{AtomicU32, Ordering};
use kernel_core::boot::{BootInfo, RegionTable};
use kernel_core::lock::Lock;
use kernel_core::mem::{Paddr, Phys};
use silica_abi::Error;

const PAGE: u64 = 4096;
const MAX_UNITS: usize = 8;
/// Buses that can have an attached device at one time.
const CONTEXT_TABLES: u64 = 64;
/// Device address spaces that can have devices attached at one time.
const DOMAINS: usize = 256;
/// Refused accesses logged before the fault interrupt is switched off.
const FAULT_LINES: u32 = 32;
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
    status: 0,
    four_levels: true,
    large: true,
    caching: false,
    top: 0,
    domains: [1, 0, 0, 0],
    domain_limit: DOMAINS,
    per_bus: [0; 256],
}));
static LOCK: Lock = Lock::new();
static REPORTED: AtomicU32 = AtomicU32::new(0);
/// `top` again, readable without the lock: 0 until the units translate.
static TOP: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static LARGE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

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

/// Spin until `done`, which a working unit makes true within microseconds.
fn wait(what: &str, done: impl Fn() -> bool) {
    let mut spins = 0u64;
    while !done() {
        spins += 1;
        if spins > 1 << 32 {
            panic!("an IOMMU does not answer ({what})");
        }
        core::hint::spin_loop();
    }
}

/// Give unit `u` a one-shot command, or change a bit that stays, and wait
/// until its status shows it.
fn command(phys: &Phys, u: &Unit, bit: u32, what: &str) {
    w32(phys, u.base, GCMD, u.gcmd | bit);
    wait(what, || r32(phys, u.base, GSTS) & bit != 0);
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
        wait("a queued invalidation", || unsafe { word(phys, s.status).read_volatile() as u32 } == 1);
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
    while o + 4 <= dmar.len() {
        let field = |at: usize, n: usize| dmar.get(o + at..o + at + n).map(|b| b.iter().rev().fold(0u64, |v, x| v << 8 | *x as u64));
        let (Some(ty), Some(len)) = (field(0, 2), field(2, 2)) else { break };
        if len < 4 || o + len as usize > dmar.len() {
            break;
        }
        if let (0, Some(segment), Some(base)) = (ty, field(6, 2), field(8, 8)) {
            let fits = base != 0 && base.is_multiple_of(PAGE) && base + PAGE <= bi.phys_map_len;
            if fits && segment == 0 && s.n < MAX_UNITS {
                s.units[s.n].base = base;
                s.n += 1;
            } else if fits && others < MAX_UNITS {
                other_segments[others] = base;
                others += 1;
            } else {
                let _ = write!(super::Out, "kernel: an IOMMU at {base:#x} is out of reach; its devices are not confined\r\n");
            }
        }
        o += len as usize;
    }
    if s.n + others == 0 {
        return;
    }
    // What every unit of segment 0 can do.
    let (mut widths, mut mgaw) = (0b110u64, 64);
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
    }
    if widths == 0 {
        // Too old a unit (no queued invalidation, or neither 39 nor 48
        // bits): the machine runs as one without an IOMMU, and says so.
        let _ = write!(super::Out, "kernel: this machine's IOMMU is of a kind Silica does not drive; devices are not confined\r\n");
        s.n = 0;
        return;
    }
    s.four_levels = widths & 0b100 != 0;
    // Root table, an empty root for other segments, the wait status word's
    // page, the context tables, and a queue page per unit.
    let pages = 3 + CONTEXT_TABLES + (s.n + others) as u64;
    let Some(mem) = table.carve(pages * PAGE, PAGE, 1 << 20, bi.phys_map_len) else {
        let _ = write!(super::Out, "kernel: no memory for the IOMMU's tables; devices are not confined\r\n");
        s.n = 0;
        return;
    };
    phys.zero(mem, pages * PAGE);
    (s.root, s.status, s.pool) = (mem, mem + 2 * PAGE, mem + 3 * PAGE);
    let queues = s.pool + CONTEXT_TABLES * PAGE;
    for (i, base) in other_segments[..others].iter().enumerate() {
        let mut u = Unit { base: *base, queue: queues + (s.n + i) as u64 * PAGE, fault_regs: 0, fault_count: 0, gcmd: 0 };
        start(phys, &mut u, mem + PAGE, s.status);
    }
    let boot_apic = percpu().apic_id as u64;
    for i in 0..s.n {
        let mut u = s.units[i];
        u.queue = queues + i as u64 * PAGE;
        start(phys, &mut u, s.root, s.status);
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
}

/// Point unit `u` at root table `root`, switch its queue on, drop whatever
/// the firmware left cached, and make it translate.
fn start(phys: &Phys, u: &mut Unit, root: u64, status: u64) {
    // Whatever the firmware left on: translation off first, so the root
    // table can change without a window of stale entries.
    u.gcmd = r32(phys, u.base, GSTS) & (1 << 25 | 1 << 23); // interrupt remapping, if the firmware uses it, stays
    w32(phys, u.base, GCMD, u.gcmd);
    wait("switching translation off", || r32(phys, u.base, GSTS) & (GCMD_TE | GCMD_QIE) == 0);
    w64(phys, u.base, IQT, 0);
    w64(phys, u.base, IQA, u.queue);
    u.gcmd |= GCMD_QIE;
    command(phys, u, GCMD_QIE, "switching its queue on");
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
        status: 0,
        four_levels: true,
        large: true,
        caching: false,
        top: 0,
        domains: [1, 0, 0, 0],
        domain_limit: DOMAINS,
        per_bus: [0; 256],
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

/// `Hal::io_attach`.
pub fn attach(phys: &Phys, id: u64, root: Paddr) -> Result<Paddr, Error> {
    if top() == 0 || id >> 16 != 0 {
        return Err(Error::Unsupported);
    }
    LOCK.lock();
    // SAFETY: under `LOCK`.
    let s = unsafe { &mut *STATE.0.get() };
    let r = attach_locked(phys, s, id, root);
    LOCK.unlock();
    r
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
            if s.pool_used == CONTEXT_TABLES {
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
    }
    LOCK.unlock();
}

/// A unit's fault interrupt: say which device was refused what. Runs with
/// or without the processor's lock; it touches only the units' registers.
pub fn fault(phys: &Phys) {
    LOCK.lock();
    // SAFETY: under `LOCK`.
    let s = unsafe { &*STATE.0.get() };
    for u in &s.units[..s.n] {
        for i in 0..u.fault_count {
            let at = u.fault_regs + i * 16;
            let hi = r64(phys, u.base, at + 8);
            if hi >> 63 == 0 {
                continue;
            }
            let (address, sid, reason, read) = (r64(phys, u.base, at) & !0xFFF, hi & 0xFFFF, hi >> 32 & 0xFF, hi >> 62 & 1 != 0);
            w64(phys, u.base, at + 8, 1 << 63);
            let n = REPORTED.fetch_add(1, Ordering::Relaxed);
            if n < FAULT_LINES {
                let _ = write!(
                    super::Out,
                    "kernel: device {:02x}:{:02x}.{} was refused a {} at {:#x} (IOMMU fault reason {})\r\n",
                    sid >> 8,
                    sid >> 3 & 31,
                    sid & 7,
                    if read { "read" } else { "write" },
                    address,
                    reason
                );
            } else {
                if n == FAULT_LINES {
                    let _ = write!(super::Out, "kernel: further refused device accesses are not reported\r\n");
                }
                w32(phys, u.base, FECTL, 1 << 31);
            }
        }
        // Overflow and pending bits are cleared by writing them back.
        w32(phys, u.base, FSTS, r32(phys, u.base, FSTS));
    }
    LOCK.unlock();
}
