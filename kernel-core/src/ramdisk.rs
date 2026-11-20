//! Reader for the initial ramdisk archive (format version 1, described in
//! `os/image-builder/README.md`). The kernel only looks up the root task;
//! it checks the structure it walks and leaves the checksum to the root
//! task, which reads the whole archive anyway.

const HEADER: usize = 32;
const ENTRY: usize = 48;

/// The data of the entry called `name`, or `None` when the archive is
/// malformed or has no such entry.
pub fn find<'a>(archive: &'a [u8], name: &str) -> Option<&'a [u8]> {
    let u64_at = |o: usize| Some(u64::from_le_bytes(archive.get(o..o + 8)?.try_into().ok()?));
    if archive.get(..8)? != b"SILRAMDK"
        || u64_at(8)? & 0xFFFF_FFFF != 1
        || u64_at(16)? != archive.len() as u64
    {
        return None;
    }
    let count = (u64_at(8)? >> 32) as usize;
    for i in 0..count {
        let e = HEADER.checked_add(i.checked_mul(ENTRY)?)?;
        let raw = archive.get(e..e + 32)?;
        let len = raw.iter().position(|&b| b == 0).unwrap_or(32);
        if &raw[..len] == name.as_bytes() {
            let off = usize::try_from(u64_at(e + 32)?).ok()?;
            let size = usize::try_from(u64_at(e + 40)?).ok()?;
            return archive.get(off..off.checked_add(size)?);
        }
    }
    None
}
