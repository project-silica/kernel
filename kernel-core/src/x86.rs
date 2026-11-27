//! What Silica requires of an x86_64 processor, decided from CPUID alone,
//! so the loader (before it sets EFER.NXE and builds tables with the NX
//! bit) and the kernel (on every processor) refuse the same machines with
//! the same message, and the decision is tested on the host.
//!
//! The baseline is x86-64-v3 (Haswell, Excavator and later: every x86_64
//! processor made since 2015): the user programs are built for it, so a
//! processor without it would boot and then fault on the first AVX2
//! instruction of the root task, explaining nothing. Above it the kernel
//! needs NX (W^X is a promise of `Map`), x2APIC, XSAVE, and a TSC that
//! counts at a constant rate in every state (the clock, the timers and the
//! idle accounting are the TSC): on bare metal that is the invariant TSC
//! (CPUID 0x80000007 EDX bit 8). A hypervisor often does not report it, and
//! keeps the guest's TSC constant itself (its paravirtual clocks rest on
//! that); there the kernel trusts it, and says so ([`Verdict::tsc_trusted`]).

/// One CPUID result: EAX, EBX, ECX, EDX.
pub type Leaf = [u32; 4];

/// The verdict on a processor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Verdict {
    /// The first required feature it lacks, by name (`None`: it has them).
    pub missing: Option<&'static str>,
    /// Running under a hypervisor that does not report an invariant TSC:
    /// the TSC is trusted to be constant, and the boot says so.
    pub tsc_trusted: bool,
}

/// Decide from `cpuid(leaf, subleaf)`.
pub fn check(cpuid: impl Fn(u32, u32) -> Leaf) -> Verdict {
    let max = cpuid(0, 0)[0];
    let ext_max = cpuid(0x8000_0000, 0)[0];
    let f1 = cpuid(1, 0);
    let f7 = if max >= 7 { cpuid(7, 0) } else { [0; 4] };
    let e1 = if ext_max >= 0x8000_0001 { cpuid(0x8000_0001, 0) } else { [0; 4] };
    let e7 = if ext_max >= 0x8000_0007 { cpuid(0x8000_0007, 0) } else { [0; 4] };
    let ecx1 = |bit: u32| f1[2] & 1 << bit != 0;
    let required: [(bool, &'static str); 21] = [
        (e1[3] & 1 << 20 != 0, "NX (no-execute pages)"),
        (e1[3] & 1 << 29 != 0, "long mode"),
        (ecx1(21), "x2APIC"),
        (ecx1(26), "XSAVE"),
        (f1[3] & 1 << 4 != 0, "a TSC"),
        // x86-64-v2
        (f1[3] & 1 << 8 != 0, "CMPXCHG8B"),
        (ecx1(0), "SSE3 (x86-64-v2)"),
        (ecx1(9), "SSSE3 (x86-64-v2)"),
        (ecx1(13), "CMPXCHG16B (x86-64-v2)"),
        (ecx1(19), "SSE4.1 (x86-64-v2)"),
        (ecx1(20), "SSE4.2 (x86-64-v2)"),
        (ecx1(23), "POPCNT (x86-64-v2)"),
        (e1[2] & 1 != 0, "LAHF in long mode (x86-64-v2)"),
        // x86-64-v3
        (ecx1(28), "AVX (x86-64-v3)"),
        (f7[1] & 1 << 5 != 0, "AVX2 (x86-64-v3)"),
        (f7[1] & 1 << 3 != 0, "BMI1 (x86-64-v3)"),
        (f7[1] & 1 << 8 != 0, "BMI2 (x86-64-v3)"),
        (ecx1(12), "FMA (x86-64-v3)"),
        (ecx1(29), "F16C (x86-64-v3)"),
        (ecx1(22), "MOVBE (x86-64-v3)"),
        (e1[2] & 1 << 5 != 0, "LZCNT (x86-64-v3)"),
    ];
    let mut missing = required.iter().find(|(has, _)| !has).map(|(_, name)| *name);
    let invariant = e7[3] & 1 << 8 != 0;
    let hypervisor = ecx1(31);
    if missing.is_none() && !invariant && !hypervisor {
        missing = Some("an invariant TSC");
    }
    Verdict { missing, tsc_trusted: missing.is_none() && !invariant && hypervisor }
}

/// What to do with an IOMMU unit (a DMAR hardware unit definition) at
/// register base `base` of PCI segment `segment`, `driven` units of
/// segment 0 being driven already, at most `max`, with the physical map
/// reaching `limit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitUse {
    /// Drive it: its devices are confined.
    Drive,
    /// A unit of another segment: started with an empty root table, so its
    /// devices reach nothing (the kernel does not drive other segments).
    Block,
    /// A unit of segment 0 that cannot be driven (registers out of reach or
    /// misaligned, or one too many): its devices would reach memory
    /// whatever their drivers attach them to, so the machine must confine
    /// none and say so.
    Undriven,
    /// A unit of another segment out of reach: that segment's devices are
    /// not confined (and the kernel attaches none of them).
    Unreachable,
}

/// Decide [`UnitUse`] for one unit.
pub fn unit_use(base: u64, segment: u64, limit: u64, driven: usize, max: usize) -> UnitUse {
    let fits = base != 0 && base.is_multiple_of(4096) && base.checked_add(4096).is_some_and(|e| e <= limit);
    match (segment, fits) {
        (0, true) if driven < max => UnitUse::Drive,
        (0, _) => UnitUse::Undriven,
        (_, true) => UnitUse::Block,
        (_, false) => UnitUse::Unreachable,
    }
}

/// The DMAR hardware unit flag that a unit covers every device of its
/// segment that no other unit's device scope names.
pub const INCLUDE_PCI_ALL: u8 = 1;

/// The most device scopes the kernel keeps of the units it drives; a
/// device named only by a scope past them is not confined (it is refused).
pub const MAX_SCOPES: usize = 64;
/// The longest path of a device scope the kernel follows (bridges to cross
/// and the device): a longer one names nothing it confines.
pub const MAX_PATH: usize = 4;

/// One DMAR device scope of a unit the kernel drives (VT-d section 8.3.1):
/// a PCI endpoint (`sub` false) or a PCI sub-hierarchy (`sub`: a bridge and
/// every bus below it), found from `start_bus` through `path`, each element
/// a (device, function) whose secondary bus the next element is on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Scope {
    pub sub: bool,
    pub start_bus: u8,
    pub path: [(u8, u8); MAX_PATH],
    pub len: u8,
}

/// Whether device `id` of segment 0 (bus << 8 | device << 3 | function) is
/// translated by a unit the kernel drives: by one that covers every device
/// no other unit names (`all`, `INCLUDE_PCI_ALL`), or by a scope that names
/// the device or, a sub-hierarchy, the bridge it is below. `bridge(bus,
/// devfn)` reads a bridge's secondary and subordinate bus numbers from its
/// configuration space (`None`: no bridge there, or none readable). A
/// device no scope reaches is translated by no unit and would reach memory
/// whatever it is attached to, so the kernel attaches it to nothing
/// (`Unsupported`), and the device manager starts no driver that does DMA
/// for it (the review of 2026-11-27: every device of segment 0 was taken
/// to be translated, whatever the units named).
pub fn covered(id: u32, all: bool, scopes: &[Scope], bridge: impl Fn(u8, u8) -> Option<(u8, u8)>) -> bool {
    // Segment 0's units translate segment 0's devices alone.
    if id >> 16 != 0 {
        return false;
    }
    if all {
        return true;
    }
    let (bus, devfn) = ((id >> 8) as u8, id as u8);
    scopes.iter().any(|s| {
        let n = s.len as usize;
        if n == 0 || n > MAX_PATH {
            return false;
        }
        // Down the bridges to the last element's bus. A bridge leads to
        // buses below its own only once they are numbered (`at < secondary
        // <= subordinate`, as the device manager's scan requires): an
        // unnumbered one (secondary 0) leads nowhere, not back to bus 0.
        let below = |at: u8, (secondary, subordinate): (u8, u8)| at < secondary && secondary <= subordinate;
        let mut at = s.start_bus;
        for &(d, f) in &s.path[..n - 1] {
            match bridge(at, d << 3 | f) {
                Some(b) if below(at, b) => at = b.0,
                _ => return false,
            }
        }
        let (d, f) = s.path[n - 1];
        let target = d << 3 | f;
        if (bus, devfn) == (at, target) {
            return true;
        }
        s.sub && bridge(at, target).is_some_and(|b| below(at, b) && (b.0..=b.1).contains(&bus))
    })
}

/// The configuration windows of segment 0 the kernel keeps (MCFG entries).
pub const MAX_WINDOWS: usize = 4;

/// Where bus `bus`, function `devfn` has its configuration space, in one of
/// `windows` (MCFG entries of segment 0: base, first bus, last bus). A
/// window's base is bus 0's address whatever bus the window starts at
/// (PCI Firmware Specification 3.2, table 4-3), so the function is at
/// `base + (bus << 20 | devfn << 12)`. `None` on a bus outside every window.
pub fn ecam_address(windows: &[(u64, u8, u8)], bus: u8, devfn: u8) -> Option<u64> {
    windows
        .iter()
        .find(|&&(base, first, last)| base != 0 && (first..=last).contains(&bus))
        .map(|&(base, _, _)| base + ((bus as u64) << 20 | (devfn as u64) << 12))
}

/// Whether window `(base, first, last)` can be read through the first
/// `limit` bytes of physical memory (the kernel's map): its base on a 1 MiB
/// boundary and its last bus's space, up to `base + (last + 1) << 20`,
/// inside the map.
pub fn ecam_window_fits((base, first, last): (u64, u8, u8), limit: u64) -> bool {
    first <= last && base.is_multiple_of(1 << 20) && base.checked_add(((last as u64) + 1) << 20).is_some_and(|end| end <= limit)
}

/// Compare a remote timestamp sampled between two local counter reads.
/// Waiting for two distinct publications excludes a timestamp that was
/// already sampled, but not yet published, at the first local read.
/// A long scheduling delay makes the sample inconclusive.
pub fn counters_agree(mut local: impl FnMut() -> u64, mut published: impl FnMut() -> u64, hz: u64) -> bool {
    let start = local();
    loop {
        let before = local();
        let mut sample = published();
        for _ in 0..2 {
            let previous = sample;
            while sample == previous {
                if local().wrapping_sub(start) >= hz {
                    return false;
                }
                core::hint::spin_loop();
                sample = published();
            }
        }
        let after = local();
        let elapsed = after.wrapping_sub(before);
        if elapsed <= hz / 20_000 && sample.wrapping_sub(before) <= elapsed {
            return true;
        }
        if after.wrapping_sub(start) >= hz {
            return false;
        }
    }
}
