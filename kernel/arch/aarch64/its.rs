//! The GICv3 ITS: message-signalled interrupts on aarch64 (C33).
//!
//! A device raises an MSI by writing an event number to the ITS's
//! translation register; the ITS looks the (device, event) pair up in the
//! device's interrupt translation table and delivers a locality-specific
//! interrupt (LPI) to a processor's redistributor. Message line `n` (from
//! [`FIRST_LINE`] up) is event `n - FIRST_LINE` and LPI `8192 + n -
//! FIRST_LINE`, whichever device raises it; the device's identity is its
//! PCI requester ID (segment 0, the identity mapping QEMU's `virt` machine
//! and the IORT of simple machines give).
//!
//! The kernel carves at boot: the LPI property table (one byte per LPI,
//! shared by every redistributor), a pending table per processor (64 KiB
//! aligned, as GICR_PENDBASER requires), the ITS's device and collection
//! tables (flat), its command queue, and a pool of translation tables for
//! devices. Commands are issued under a lock and waited for (SYNC, then the
//! read pointer catching up), so a mapping is in force when the call that
//! made it returns.
//!
//! Not yet: two-level device tables (a machine with more than 2^16 device
//! identities), the IORT's identity mapping read rather than assumed, and
//! more than one ITS.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU64, Ordering};
use kernel_core::boot::RegionTable;
use kernel_core::lock::Lock;
use kernel_core::smp::MAX_CPUS;

/// The first message line: the lines below are wires (SPIs).
pub const FIRST_LINE: usize = 128;
/// Message lines there are, and so events and LPIs.
pub const LINES: usize = silica_abi::IRQ_LINES as usize - FIRST_LINE;
/// The first LPI.
pub const LPI_BASE: u32 = 8192;
/// Interrupt identifier bits the redistributors are told about: enough for
/// every LPI used.
const ID_BITS: u32 = 14;
/// Event bits of a device's translation table.
const EVENT_BITS: u32 = 6;
const _: () = assert!(LINES <= 1 << EVENT_BITS && LPI_BASE as usize + LINES <= 1 << ID_BITS);
const PRIORITY: u8 = 0x80;
const PAGE: u64 = 4096;
/// Devices with a translation table at one time.
const DEVICES: usize = 128;

struct State {
    /// The ITS's registers (virtual) and its translation register (physical).
    base: u64,
    translater: u64,
    /// The LPI property table (virtual) and its physical address.
    props: u64,
    props_phys: u64,
    /// Each processor's pending table (physical), 64 KiB each.
    pending: u64,
    /// The command queue (virtual, physical) and where the next command goes.
    queue: u64,
    queue_phys: u64,
    write: u64,
    /// Translation tables: their pool (virtual, physical), each slot's size,
    /// and the device each slot belongs to (u64::MAX: free).
    itt: u64,
    itt_phys: u64,
    itt_size: u64,
    itt_owner: [u64; DEVICES],
    /// Whether processor `i`'s collection was mapped, and the redistributor
    /// target the ITS wants for it (a physical address or a processor number).
    collections: [bool; MAX_CPUS],
    pta: bool,
    /// What each message line is mapped as: the device (u64::MAX: nothing).
    line_device: [u64; LINES],
    /// The DeviceIDs the device table covers (from 0), and the firmware's
    /// I/O Remapping Table, which maps a PCI requester ID to its DeviceID
    /// (none: the identity, for segment 0 only).
    devices: u64,
    iort: Option<&'static [u8]>,
}

struct Racy(UnsafeCell<State>);
// SAFETY: filled by the boot processor before any other runs; afterwards
// only touched holding `LOCK`.
unsafe impl Sync for Racy {}
static STATE: Racy = Racy(UnsafeCell::new(State {
    base: 0,
    translater: 0,
    props: 0,
    props_phys: 0,
    pending: 0,
    queue: 0,
    queue_phys: 0,
    write: 0,
    itt: 0,
    itt_phys: 0,
    itt_size: 0,
    itt_owner: [u64::MAX; DEVICES],
    collections: [false; MAX_CPUS],
    pta: false,
    line_device: [u64::MAX; LINES],
    devices: 0,
    iort: None,
}));
static LOCK: Lock = Lock::new();
/// Physical address of each processor's redistributor frame (0: unknown).
pub static GICR_PHYS: [AtomicU64; MAX_CPUS] = [const { AtomicU64::new(0) }; MAX_CPUS];
/// Set once the ITS is in use.
static READY: AtomicU64 = AtomicU64::new(0);

fn rd64(a: u64) -> u64 {
    // SAFETY: an ITS or redistributor register in the device window.
    unsafe { core::ptr::read_volatile(a as *const u64) }
}
fn wr64(a: u64, v: u64) {
    // SAFETY: as for `rd64`.
    unsafe { core::ptr::write_volatile(a as *mut u64, v) }
}
fn rd32(a: u64) -> u32 {
    // SAFETY: as for `rd64`.
    unsafe { core::ptr::read_volatile(a as *const u32) }
}
fn wr32(a: u64, v: u32) {
    // SAFETY: as for `rd64`.
    unsafe { core::ptr::write_volatile(a as *mut u32, v) }
}
fn barrier() {
    // SAFETY: a barrier: tables and commands written before reach the ITS
    // before the register write that tells it to look.
    unsafe { core::arch::asm!("dsb ish", options(nostack, preserves_flags)) }
}

/// Inner shareable, write-back cacheable: the ITS and the redistributors
/// read the tables coherently with the processors. The ITS's registers
/// keep the inner cacheability in bits 61:59, the redistributors' in 9:7.
const CACHE: u64 = 0b111 << 59 | 0b01 << 10;
const CACHE_R: u64 = 0b111 << 7 | 0b01 << 10;

/// Whether the ITS is in use: message lines exist.
pub fn ready() -> bool {
    READY.load(Ordering::Acquire) != 0
}

/// Find the ITS's tables a home, switch it on (boot processor, once,
/// before the root task is given the rest of memory). `base` is its
/// registers' virtual address and `phys` their physical one. `virt` turns
/// a physical address of carved memory into the kernel's virtual one.
pub fn init(base: u64, phys: u64, cpus: usize, table: &mut RegionTable, iort: Option<&'static [u8]>, virt: impl Fn(u64) -> u64) -> bool {
    // SAFETY: boot processor, alone.
    let s = unsafe { &mut *STATE.0.get() };
    s.iort = iort;
    let typer = rd64(base + 0x8);
    let itt_entry = (typer >> 4 & 0xF) + 1;
    let dev_bits = (typer >> 13 & 0x1F) + 1;
    s.pta = typer & 1 << 19 != 0;
    if typer & 1 == 0 || dev_bits < 16 {
        return false; // no physical LPIs, or too few device identities for PCI
    }
    // Everything carved in one go, each piece aligned for what reads it.
    let props_len = (1u64 << ID_BITS) - LPI_BASE as u64;
    let (Some(props), Some(pending), Some(queue)) = (
        table.carve(props_len.next_multiple_of(PAGE), 1 << 16, 1 << 20, u64::MAX),
        table.carve(cpus as u64 * 0x1_0000, 0x1_0000, 1 << 20, u64::MAX),
        table.carve(PAGE, 1 << 16, 1 << 20, u64::MAX),
    ) else {
        return false;
    };
    s.itt_size = (itt_entry << EVENT_BITS).next_multiple_of(256);
    let Some(itt) = table.carve((s.itt_size * DEVICES as u64).next_multiple_of(PAGE), 1 << 16, 1 << 20, u64::MAX) else { return false };
    for (p, len) in [(props, props_len), (pending, cpus as u64 * 0x1_0000), (queue, PAGE), (itt, s.itt_size * DEVICES as u64)] {
        // SAFETY: memory just carved for the kernel, mapped by the physical map.
        unsafe { core::ptr::write_bytes(virt(p) as *mut u8, 0, len as usize) };
    }
    (s.base, s.translater, s.props, s.props_phys, s.pending) = (base, phys + 0x1_0040, virt(props), props, pending);
    (s.queue, s.queue_phys, s.itt, s.itt_phys) = (virt(queue), queue, virt(itt), itt);
    // The device and collection tables, flat, in 4 KiB pages.
    for n in 0..8u64 {
        let reg = base + 0x100 + n * 8;
        let v = rd64(reg);
        let (kind, entry) = (v >> 56 & 7, (v >> 48 & 0x1F) + 1);
        let entries = match kind {
            1 => {
                s.devices = 1u64 << dev_bits.min(16);
                s.devices
            }
            4 => cpus.next_power_of_two() as u64,
            _ => continue,
        };
        let pages = (entries * entry).div_ceil(PAGE).max(1);
        if pages > 256 {
            return false;
        }
        let Some(t) = table.carve(pages * PAGE, 1 << 16, 1 << 20, u64::MAX) else { return false };
        // SAFETY: as above.
        unsafe { core::ptr::write_bytes(virt(t) as *mut u8, 0, (pages * PAGE) as usize) };
        // Valid, the type and entry size as they were, 4 KiB pages, the size.
        wr64(reg, 1 << 63 | CACHE | (v & (7 << 56 | 0x1F << 48)) | t | (pages - 1));
        if rd64(reg) >> 8 & 3 != 0 {
            return false; // the ITS does not take 4 KiB pages
        }
    }
    wr64(base + 0x80, 1 << 63 | CACHE | queue); // GITS_CBASER: one page
    wr64(base + 0x88, 0); // GITS_CWRITER
    wr32(base, rd32(base) | 1); // GITS_CTLR.Enabled
    READY.store(1, Ordering::Release);
    true
}

/// Point this processor's redistributor at the LPI tables and let it take
/// LPIs. Every processor, once, after `init` (whether or not it succeeded).
/// `gicr` is the redistributor's virtual address, `gicr_phys` its physical.
pub fn enable_processor(cpu: usize, gicr: u64, gicr_phys: u64) {
    if !ready() {
        return;
    }
    if rd64(gicr + 0x8) & 1 == 0 {
        return; // GICR_TYPER.PLPIS: this redistributor takes no LPIs
    }
    GICR_PHYS[cpu].store(gicr_phys, Ordering::Relaxed);
    // SAFETY: written by the boot processor before any other runs.
    let s = unsafe { &*STATE.0.get() };
    wr64(gicr + 0x70, s.props_phys | CACHE_R | (ID_BITS as u64 - 1)); // GICR_PROPBASER
    wr64(gicr + 0x78, s.pending + cpu as u64 * 0x1_0000 | CACHE_R | 1 << 62); // GICR_PENDBASER, zeroed
    barrier();
    wr32(gicr, rd32(gicr) | 1); // GICR_CTLR.EnableLPIs
}

/// Queue one command and wait until the ITS has done it.
fn command(s: &mut State, words: [u64; 4]) {
    let at = s.write;
    for (i, w) in words.iter().enumerate() {
        // SAFETY: a slot of the command queue, the kernel's own memory.
        unsafe { core::ptr::write_volatile((s.queue + at + i as u64 * 8) as *mut u64, *w) };
    }
    s.write = (at + 32) % PAGE;
    barrier();
    wr64(s.base + 0x88, s.write);
    for _ in 0..10_000_000 {
        if rd64(s.base + 0x90) & 0xF_FFE0 == s.write {
            return;
        }
        core::hint::spin_loop();
    }
    panic!("the GIC ITS does not take commands");
}

/// The redistributor target of processor `cpu` in a command.
fn target(s: &State, cpu: usize) -> Option<u64> {
    let phys = GICR_PHYS.get(cpu)?.load(Ordering::Relaxed);
    if phys == 0 {
        return None;
    }
    Some(if s.pta { phys } else { (rd64(super::gic::GICR[cpu].load(Ordering::Relaxed) + 0x8) >> 8 & 0xFFFF) << 16 })
}

fn sync(s: &mut State, cpu: usize) {
    if let Some(t) = target(s, cpu) {
        command(s, [0x05, 0, t, 0]);
    }
}

/// Deliver message line `line` to processor `cpu`, raised by the device
/// with requester ID `device`. Returns the message: the translation
/// register's physical address and the event number.
pub fn route(line: usize, cpu: usize, device: u64) -> Option<(u64, u64)> {
    let n = line.checked_sub(FIRST_LINE).filter(|n| *n < LINES && ready())?;
    LOCK.lock();
    // SAFETY: under `LOCK`.
    let s = unsafe { &mut *STATE.0.get() };
    let r = device_id(s, device).and_then(|id| route_locked(s, n, cpu, id));
    LOCK.unlock();
    r
}

/// The ITS DeviceID of the device with identity `device` (`segment << 16 |
/// requester ID`): as the firmware's IORT maps it, or the requester ID
/// itself on a machine without one (segment 0 only). `None` when the table
/// maps it to no ITS, or to a DeviceID beyond the device table: never cut
/// to fit, which would make it another device.
fn device_id(s: &State, device: u64) -> Option<u64> {
    let (segment, rid) = (u32::try_from(device >> 16).ok()?, (device & 0xFFFF) as u32);
    let id = match s.iort {
        Some(t) => kernel_core::acpi::iort_its_device(t, segment, rid)?,
        None if segment == 0 => rid,
        None => return None,
    };
    (u64::from(id) < s.devices).then_some(u64::from(id))
}

/// Message line `line` is no longer bound: its mapping goes, and its
/// device's translation table with its last line.
pub fn release(line: usize) {
    let Some(n) = line.checked_sub(FIRST_LINE).filter(|n| *n < LINES && ready()) else { return };
    LOCK.lock();
    // SAFETY: under `LOCK`.
    let s = unsafe { &mut *STATE.0.get() };
    unmap_line(s, n);
    LOCK.unlock();
}

/// Take line `n`'s mapping away (DISCARD), and its device's translation
/// table when no other line uses it (MAPD, not valid), freeing the slot.
fn unmap_line(s: &mut State, n: usize) {
    let old = s.line_device[n];
    if old == u64::MAX {
        return;
    }
    command(s, [0x0F | old << 32, n as u64, 0, 0]);
    s.line_device[n] = u64::MAX;
    // SAFETY: the property table's byte for this LPI, the kernel's memory.
    unsafe { core::ptr::write_volatile((s.props + n as u64) as *mut u8, PRIORITY) };
    if !s.line_device.contains(&old)
        && let Some(slot) = s.itt_owner.iter().position(|o| *o == old)
    {
        command(s, [0x08 | old << 32, 0, 0, 0]);
        s.itt_owner[slot] = u64::MAX;
    }
}

fn route_locked(s: &mut State, n: usize, cpu: usize, device: u64) -> Option<(u64, u64)> {
    let rd = target(s, cpu)?;
    if !s.collections[cpu] {
        // MAPC: collection `cpu` is that processor's redistributor.
        command(s, [0x09, 0, 1 << 63 | rd | cpu as u64, 0]);
        s.collections[cpu] = true;
    }
    // The line's earlier mapping goes first, and its device's table with
    // its last line.
    unmap_line(s, n);
    // The device's translation table, mapped the first time (MAPD).
    if !s.itt_owner.contains(&device) {
        let slot = s.itt_owner.iter().position(|o| *o == u64::MAX)?;
        s.itt_owner[slot] = device;
        let itt = s.itt_phys + slot as u64 * s.itt_size;
        command(s, [0x08 | device << 32, EVENT_BITS as u64 - 1, 1 << 63 | itt, 0]);
    }
    // MAPTI: event n of the device is LPI 8192 + n, on collection `cpu`.
    command(s, [0x0A | device << 32, n as u64 | (LPI_BASE as u64 + n as u64) << 32, cpu as u64, 0]);
    s.line_device[n] = device;
    // SAFETY: the property table's byte for this LPI, the kernel's memory.
    unsafe { core::ptr::write_volatile((s.props + n as u64) as *mut u8, PRIORITY | 1) };
    barrier();
    command(s, [0x0C | device << 32, n as u64, 0, 0]); // INV
    sync(s, cpu);
    Some((s.translater, n as u64))
}

/// Hold message line `line` off, or let it interrupt again.
pub fn mask(line: usize, masked: bool) {
    let Some(n) = line.checked_sub(FIRST_LINE).filter(|n| *n < LINES && ready()) else { return };
    LOCK.lock();
    // SAFETY: under `LOCK`.
    let s = unsafe { &mut *STATE.0.get() };
    let device = s.line_device[n];
    if device != u64::MAX {
        // SAFETY: as in `route_locked`.
        unsafe { core::ptr::write_volatile((s.props + n as u64) as *mut u8, PRIORITY | !masked as u8) };
        barrier();
        command(s, [0x0C | device << 32, n as u64, 0, 0]);
    }
    LOCK.unlock();
}

/// The message line of LPI `id`, if it is one of the kernel's.
pub fn line_of(id: u32) -> Option<usize> {
    let n = id.checked_sub(LPI_BASE)? as usize;
    (n < LINES).then_some(FIRST_LINE + n)
}
