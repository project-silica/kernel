//! The processor features Silica requires (`kernel_core::x86::check`),
//! decided from CPUID tables of processors it must refuse and accept.

use kernel_core::x86::{Leaf, check};
use std::collections::HashMap;

/// A processor as CPUID describes it: the x86-64-v3 baseline with NX, an
/// invariant TSC, x2APIC and XSAVE (a Haswell, say), less what `off` names
/// as (leaf, register, bit), plus what `on` adds.
fn cpu(off: &[(u32, usize, u32)], on: &[(u32, usize, u32)]) -> impl Fn(u32, u32) -> Leaf {
    let mut t: HashMap<u32, Leaf> = HashMap::new();
    t.insert(0, [0xD, 0, 0, 0]);
    t.insert(0x8000_0000, [0x8000_0008, 0, 0, 0]);
    let ecx1 = [0, 9, 12, 13, 19, 20, 21, 22, 23, 26, 27, 28, 29].iter().fold(0u32, |a, b| a | 1 << b);
    t.insert(1, [0, 0, ecx1, 1 << 4 | 1 << 8]);
    t.insert(7, [0, 1 << 3 | 1 << 5 | 1 << 8, 0, 0]);
    t.insert(0x8000_0001, [0, 0, 1 | 1 << 5, 1 << 20 | 1 << 29]);
    t.insert(0x8000_0007, [0, 0, 0, 1 << 8]);
    for &(leaf, reg, bit) in off {
        t.get_mut(&leaf).unwrap()[reg] &= !(1 << bit);
    }
    for &(leaf, reg, bit) in on {
        t.get_mut(&leaf).unwrap()[reg] |= 1 << bit;
    }
    move |leaf, _| t.get(&leaf).copied().unwrap_or([0; 4])
}

#[test]
fn the_baseline_processor_is_accepted() {
    let v = check(cpu(&[], &[]));
    assert_eq!((v.missing, v.tsc_trusted), (None, false));
}

#[test]
fn comparing_counters_requires_a_fresh_sample_and_a_bounded_interval() {
    use core::cell::Cell;
    use kernel_core::x86::counters_agree;
    let check = |offset: i64, stalled: u64| {
        let ticks = Cell::new(0u64);
        counters_agree(
            || {
                ticks.set(ticks.get() + 1);
                ticks.get()
            },
            || if ticks.get() < stalled { 0 } else { ticks.get().wrapping_add_signed(offset) },
            100_000,
        )
    };
    assert!(check(0, 200), "a descheduled publisher can resume");
    assert!(!check(40, 0));
    assert!(!check(-40, 0));
    assert!(!check(0, u64::MAX), "a stopped publisher cannot hold boot forever");
}

#[test]
fn a_processor_without_avx_avx2_or_nx_is_refused_by_name() {
    assert_eq!(check(cpu(&[(1, 2, 28)], &[])).missing, Some("AVX (x86-64-v3)"));
    assert_eq!(check(cpu(&[(7, 1, 5)], &[])).missing, Some("AVX2 (x86-64-v3)"));
    assert_eq!(check(cpu(&[(0x8000_0001, 3, 20)], &[])).missing, Some("NX (no-execute pages)"));
    assert_eq!(check(cpu(&[(1, 2, 21)], &[])).missing, Some("x2APIC"));
    assert_eq!(check(cpu(&[(0x8000_0001, 2, 5)], &[])).missing, Some("LZCNT (x86-64-v3)"));
    // A processor whose CPUID stops before leaf 7 has no AVX2.
    let old = cpu(&[], &[]);
    let short = move |l: u32, s: u32| if l == 0 { [1, 0, 0, 0] } else { old(l, s) };
    assert_eq!(check(short).missing, Some("AVX2 (x86-64-v3)"));
}

#[test]
fn a_tsc_that_is_not_invariant_is_refused_on_bare_metal_and_trusted_under_a_hypervisor() {
    let v = check(cpu(&[(0x8000_0007, 3, 8)], &[]));
    assert_eq!(v.missing, Some("an invariant TSC"));
    let v = check(cpu(&[(0x8000_0007, 3, 8)], &[(1, 2, 31)]));
    assert_eq!((v.missing, v.tsc_trusted), (None, true));
}

#[test]
fn an_iommu_unit_that_cannot_be_driven_confines_nothing() {
    use kernel_core::x86::{UnitUse as U, unit_use};
    let limit = 1 << 40;
    assert_eq!(unit_use(0xFED9_0000, 0, limit, 0, 32), U::Drive);
    assert_eq!(unit_use(0xFED9_0800, 0, limit, 0, 32), U::Undriven, "registers not page-aligned");
    assert_eq!(unit_use(limit, 0, limit, 0, 32), U::Undriven, "registers past the physical map");
    assert_eq!(unit_use(!0xFFF, 0, u64::MAX, 0, 32), U::Undriven, "a base whose end wraps");
    assert_eq!(unit_use(0xFED9_0000, 0, limit, 32, 32), U::Undriven, "one unit too many");
    assert_eq!(unit_use(0, 0, limit, 0, 32), U::Undriven);
    assert_eq!(unit_use(0xFED9_0000, 1, limit, 0, 32), U::Block);
    assert_eq!(unit_use(0xFED9_0800, 1, limit, 0, 32), U::Unreachable);
}

/// A device of segment 0 is translated only where a driven unit covers it:
/// a unit that covers every device no other names (INCLUDE_PCI_ALL), or a
/// device scope naming it, or naming the bridge it is below. QEMU's units
/// name every device on bus 0 and each root port as a sub-hierarchy; a
/// device none of them reaches is translated by no unit (the review of
/// 2026-11-27: every device of segment 0 was taken to be translated).
#[test]
fn only_a_device_a_driven_unit_covers_is_confined() {
    use kernel_core::x86::{MAX_PATH, Scope, covered};
    let scope = |sub: bool, bus: u8, path: &[(u8, u8)]| {
        let mut s = Scope { sub, start_bus: bus, len: path.len() as u8, ..Default::default() };
        s.path[..path.len()].copy_from_slice(path);
        s
    };
    let id = |bus: u32, dev: u32, func: u32| bus << 8 | dev << 3 | func;
    // Bus 0: an endpoint at 00:01.0, a root port at 00:02.0 with buses 1 to
    // 3 below it, and on bus 1 a switch's port 01:00.0 leading to bus 2.
    let bridge = |bus: u8, devfn: u8| match (bus, devfn) {
        (0, 0x10) => Some((1, 3)),
        (1, 0x00) => Some((2, 2)),
        _ => None,
    };
    let scopes = [scope(false, 0, &[(1, 0)]), scope(true, 0, &[(2, 0)])];
    assert!(covered(id(0, 1, 0), true, &[], bridge), "a unit covering the rest covers everything");
    assert!(covered(id(0, 1, 0), false, &scopes, bridge), "the endpoint a scope names");
    assert!(!covered(id(0, 1, 1), false, &scopes, bridge), "another function of that device");
    assert!(!covered(id(0, 4, 0), false, &scopes, bridge), "a device on bus 0 no scope names");
    assert!(covered(id(0, 2, 0), false, &scopes, bridge), "the bridge of a sub-hierarchy");
    assert!(covered(id(1, 0, 0), false, &scopes, bridge) && covered(id(3, 31, 7), false, &scopes, bridge), "the buses below it");
    assert!(!covered(id(4, 0, 0), false, &scopes, bridge), "a bus past the bridge's subordinate");
    assert!(!covered(id(1, 0, 0), false, &[scope(false, 0, &[(2, 0)])], bridge), "below a bridge named only as an endpoint");
    // A path that crosses a bridge: 00:02.0, then 01:00.0, then 02:05.0.
    let deep = [scope(false, 0, &[(2, 0), (0, 0), (5, 0)])];
    assert!(covered(id(2, 5, 0), false, &deep, bridge));
    assert!(!covered(id(1, 5, 0), false, &deep, bridge), "the last element is on the bridge's secondary bus, not the first's");
    assert!(!covered(id(2, 5, 0), false, &deep, |_, _| None), "a path through a bridge that cannot be read reaches nothing");
    assert!(!covered(id(0, 1, 0), false, &[scope(false, 0, &[])], bridge), "an empty path names nothing");
    assert!(!covered(id(0, 1, 0), false, &[Scope { len: MAX_PATH as u8 + 1, ..scopes[0] }], bridge), "nor one past the longest kept");
    assert!(!covered(1 << 16 | id(0, 1, 0), true, &scopes, bridge), "another segment's device");
    assert!(!covered(id(0, 1, 0), false, &[], bridge), "no scope and no unit covering the rest: nothing");
}

/// A bridge whose buses are not numbered (secondary 0, as an empty root
/// port's may be) leads nowhere: a path crossing it names nothing, and a
/// sub-hierarchy below it holds no bus. Crossing one fell back to bus 0,
/// so a scope through an unnumbered 00:02.0 to (5, 0) covered 00:05.0
/// (the review of 2026-11-27). Each crossing needs the bridge's buses
/// below its own (`at < secondary <= subordinate`), as the device
/// manager's scan requires.
#[test]
fn a_path_through_an_unnumbered_bridge_reaches_nothing() {
    use kernel_core::x86::{Scope, covered};
    let scope = |sub: bool, path: &[(u8, u8)]| {
        let mut s = Scope { sub, start_bus: 0, len: path.len() as u8, ..Default::default() };
        s.path[..path.len()].copy_from_slice(path);
        s
    };
    let id = |bus: u32, dev: u32, func: u32| bus << 8 | dev << 3 | func;
    let unnumbered = |bus: u8, devfn: u8| ((bus, devfn) == (0, 0x10)).then_some((0, 0));
    assert!(!covered(id(0, 5, 0), false, &[scope(false, &[(2, 0), (5, 0)])], unnumbered), "a path through an unnumbered bridge");
    assert!(!covered(id(0, 1, 0), false, &[scope(true, &[(2, 0)])], unnumbered), "below an unnumbered bridge");
    // Buses that go backwards or upside down are no more numbered.
    let backwards = |bus: u8, devfn: u8| ((bus, devfn) == (3, 0x10)).then_some((2, 5));
    let mut s = scope(false, &[(2, 0), (5, 0)]);
    s.start_bus = 3;
    assert!(!covered(id(2, 5, 0), false, &[s], backwards), "a secondary bus at or before the bridge's own");
    let inverted = |bus: u8, devfn: u8| ((bus, devfn) == (0, 0x10)).then_some((4, 2));
    assert!(!covered(id(3, 0, 0), false, &[scope(true, &[(2, 0)])], inverted), "a subordinate bus before the secondary");
    // Numbered, the same scopes reach what is below.
    let numbered = |bus: u8, devfn: u8| ((bus, devfn) == (0, 0x10)).then_some((1, 1));
    assert!(covered(id(1, 5, 0), false, &[scope(false, &[(2, 0), (5, 0)])], numbered));
    assert!(covered(id(1, 0, 0), false, &[scope(true, &[(2, 0)])], numbered));
}

/// Configuration space is at the MCFG window's base plus the bus's own
/// offset: the base is bus 0's address even when the window starts at a
/// later bus (PCI Firmware Specification 3.2, table 4-3). Counting from the
/// window's first bus read another bus's devices (the review of
/// 2026-11-27). A bus outside every window has none; a window is used only
/// where its buses lie inside the kernel's physical map.
#[test]
fn configuration_space_is_found_from_bus_zero_s_address() {
    use kernel_core::x86::{ecam_address, ecam_window_fits};
    let windows = [(0xE000_0000u64, 0x10u8, 0x1Fu8), (0xF000_0000, 0x80, 0x80)];
    assert_eq!(ecam_address(&windows, 0x10, 0), Some(0xE000_0000 + (0x10 << 20)));
    assert_eq!(ecam_address(&windows, 0x1F, 0xFF), Some(0xE000_0000 + (0x1F << 20 | 0xFF << 12)));
    assert_eq!(ecam_address(&windows, 0x80, 0x08), Some(0xF000_0000 + (0x80 << 20 | 0x08 << 12)));
    assert_eq!(ecam_address(&windows, 0x0F, 0), None, "before the first window");
    assert_eq!(ecam_address(&windows, 0x20, 0), None, "between windows");
    assert_eq!(ecam_address(&[], 0, 0), None);
    let limit = 0x1_0000_0000;
    assert!(ecam_window_fits((0xE000_0000, 0x10, 0x1F), limit));
    assert!(ecam_window_fits((0xF000_0000, 0x00, 0xFF), limit), "its last bus ending at the map's end");
    assert!(!ecam_window_fits((0xF010_0000, 0x00, 0xFF), limit), "its last bus past the map");
    assert!(ecam_window_fits((0xF000_0000, 0x00, 0x0F), limit));
    assert!(!ecam_window_fits((0xE000_8000, 0x00, 0x0F), limit), "a base not on a 1 MiB boundary");
    assert!(!ecam_window_fits((!0xF_FFFF, 0, 0xFF), u64::MAX), "a window whose end wraps");
}
