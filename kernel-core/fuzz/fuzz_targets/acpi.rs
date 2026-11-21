//! The ACPI reader: firmware tables are read by the root task (and, in
//! time, the kernel) before anything has verified them. Whatever the bytes
//! say, it finds a table or does not; it never panics and never reads
//! outside what it was given.
#![no_main]

use silica_formats::acpi;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    // The input as physical memory starting at 0x1000; its first two bytes
    // say where in it the RSDP is.
    const BASE: u64 = 0x1000;
    let mut read = |at: u64, buf: &mut [u8]| {
        let got = at.checked_sub(BASE).and_then(|o| data.get(o as usize..)).and_then(|s| s.get(..buf.len()));
        match got {
            Some(s) => {
                buf.copy_from_slice(s);
                true
            }
            None => false,
        }
    };
    let rsdp = BASE + data.first().map_or(0, |&b| b as u64) + (data.get(1).map_or(0, |&b| b as u64) << 8);
    for sig in [b"SPCR", b"APIC", b"FACP", b"MCFG"] {
        if let Some((at, len)) = acpi::find_table(&mut read, rsdp, sig) {
            assert!(at >= BASE && len >= 36);
        }
    }
    if let Some((at, _)) = acpi::dsdt(&mut read, rsdp) {
        assert!(at >= BASE);
    }
    // The input as an AML table (a DSDT or SSDT).
    for hid in [&b"ARMH0011"[..], b"PNP0A08", b"A"] {
        for r in acpi::devices(data, hid).take(256) {
            let _ = r;
        }
    }
});
