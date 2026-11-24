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
