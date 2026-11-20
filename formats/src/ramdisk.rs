//! Reader for the initial ramdisk archive (format version 1, described in
//! `os/image-builder/README.md`). The readers check the structure they walk;
//! the checksum is the business of whoever reads the whole archive.

const HEADER: usize = 32;
const ENTRY: usize = 48;

fn u64_at(archive: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(archive.get(o..o.checked_add(8)?)?.try_into().ok()?))
}

/// The entries of the archive as (name, data), in order. Empty when the
/// header is malformed; ends early at the first malformed entry.
pub fn entries(archive: &[u8]) -> impl Iterator<Item = (&str, &[u8])> {
    let ok = archive.get(..8) == Some(b"SILRAMDK")
        && u64_at(archive, 8).is_some_and(|v| v & 0xFFFF_FFFF == 1)
        && u64_at(archive, 16) == Some(archive.len() as u64);
    let count = if ok { (u64_at(archive, 8).unwrap_or(0) >> 32) as usize } else { 0 };
    (0..count).map_while(move |i| {
        let e = HEADER.checked_add(i.checked_mul(ENTRY)?)?;
        let raw = archive.get(e..e.checked_add(32)?)?;
        let len = raw.iter().position(|&b| b == 0).unwrap_or(32);
        let name = core::str::from_utf8(&raw[..len]).ok()?;
        let off = usize::try_from(u64_at(archive, e + 32)?).ok()?;
        let size = usize::try_from(u64_at(archive, e + 40)?).ok()?;
        Some((name, archive.get(off..off.checked_add(size)?)?))
    })
}

/// The data of the entry called `name`, or `None` when the archive is
/// malformed or has no such entry.
pub fn find<'a>(archive: &'a [u8], name: &str) -> Option<&'a [u8]> {
    entries(archive).find(|(n, _)| *n == name).map(|(_, d)| d)
}
