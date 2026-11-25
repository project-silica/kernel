//! The boot archive's validator and readers (`silica_formats::ramdisk`): the
//! kernel checks the archive from the boot medium before it starts the root
//! task from it, so whatever the bytes are, `check` decides without
//! panicking, and an archive it accepts is read only inside itself.
#![no_main]

use silica_formats::ramdisk;

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    // The readers on the bytes as they are: never a panic.
    let _ = ramdisk::check(data, ramdisk::crc32c_update);
    let _ = ramdisk::entries(data).count();
    // The same bytes with a correct checksum and length field, so the rules
    // past the checksum are reached too.
    let mut a = data.to_vec();
    if a.len() >= ramdisk::HEADER {
        a[..8].copy_from_slice(&ramdisk::MAGIC);
        a[8..12].copy_from_slice(&ramdisk::VERSION.to_le_bytes());
        let len = a.len() as u64;
        a[16..24].copy_from_slice(&len.to_le_bytes());
        a[24..28].fill(0);
        let sum = ramdisk::crc32c(&a);
        a[24..28].copy_from_slice(&sum.to_le_bytes());
    }
    if ramdisk::check(&a, ramdisk::crc32c_update).is_ok() {
        let base = a.as_ptr() as usize;
        let mut names = Vec::new();
        for (name, body) in ramdisk::entries(&a) {
            let at = body.as_ptr() as usize;
            assert!(at >= base && at + body.len() <= base + a.len(), "an entry outside its archive");
            assert!(!names.contains(&name), "two entries named {name}");
            assert_eq!(ramdisk::find(&a, name).map(|b| b.as_ptr()), Some(body.as_ptr()));
            names.push(name);
        }
    }
});
