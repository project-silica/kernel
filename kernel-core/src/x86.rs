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
