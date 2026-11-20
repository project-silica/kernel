//! The ELF reader and the ramdisk lookup: both parse bytes from the boot
//! volume before anything has verified them, and must reject, never panic.
#![no_main]

use kernel_core::elf::Elf;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    for machine in [62, 183, 243] {
        if let Ok(e) = Elf::parse(data, machine) {
            let _ = e.span();
            for s in e.segments() {
                assert!(s.data.len() as u64 <= s.memsz);
            }
            for r in e.relative_relocs().take(4096) {
                let _ = r;
            }
        }
    }
    if let Some(d) = kernel_core::ramdisk::find(data, "root") {
        assert!(d.len() <= data.len());
    }
});
