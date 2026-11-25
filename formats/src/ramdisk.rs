//! The initial ramdisk archive (format version 2, described in
//! `os/image-builder/README.md`): the one reader and validator the image
//! builder, the kernel and userspace all use, so the parser that is tested
//! is the parser that decides what boots.
//!
//! [`check`] validates a whole archive against every rule of the format
//! (lengths, magic, version, checksum, zero padding and reserved bytes,
//! names, uniqueness, exact placement); [`entries`] and [`find`] read one
//! that passed it. The kernel checks the archive before it starts the root
//! task from it, so a damaged or truncated archive is refused, said why,
//! and nothing in it runs.
//!
//! The checksum is CRC32C (Castagnoli), which the processors Silica runs on
//! compute in hardware (SSE4.2 on x86-64-v3; Arm's CRC32 instructions): a
//! boot archive of a few megabytes is checked in well under a millisecond.
//! It finds damage, not tampering: an archive's authenticity is the loader's
//! business with the rest of the boot chain (signed boot, not yet built).

use core::fmt;

/// First eight bytes of every archive.
pub const MAGIC: [u8; 8] = *b"SILRAMDK";
/// The format version: 2 (1 had a CRC32 checksum, which no processor here
/// computes in hardware).
pub const VERSION: u32 = 2;
/// Header size in bytes.
pub const HEADER: usize = 32;
/// Size of one table entry in bytes.
pub const ENTRY: usize = 48;
/// Longest entry name in bytes (the field holds 32, the last is always NUL).
pub const NAME_MAX: usize = 31;
/// Entry data starts at multiples of this.
pub const ALIGN: usize = 16;
/// Most entries an archive may have: a boot archive holds a few dozen
/// programs, and the uniqueness check compares every pair.
pub const MAX_ENTRIES: usize = 4096;

/// Why an archive was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Malformed {
    /// The archive is shorter than its header or table claims, or its
    /// length differs from its length field.
    Truncated,
    /// The magic bytes are wrong.
    BadMagic,
    /// The version is not [`VERSION`].
    BadVersion(u32),
    /// The checksum does not match: the archive is damaged.
    BadChecksum,
    /// A reserved field or padding byte is not zero.
    NotZero,
    /// More than [`MAX_ENTRIES`] entries.
    TooMany,
    /// Entry `n`'s name is empty, too long, has bytes after its NUL, or a
    /// character outside `A-Z a-z 0-9 . _ -`.
    BadName(usize),
    /// Entry `n` has the name of an earlier one.
    Duplicate(usize),
    /// Entry `n`'s offset or size is not its canonical placement.
    BadEntry(usize),
}

impl fmt::Display for Malformed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Malformed::Truncated => write!(f, "its length does not match its header"),
            Malformed::BadMagic => write!(f, "it is not a ramdisk archive"),
            Malformed::BadVersion(v) => write!(f, "format version {v}, not {VERSION}"),
            Malformed::BadChecksum => write!(f, "its checksum does not match: it is damaged"),
            Malformed::NotZero => write!(f, "a reserved or padding byte is not zero"),
            Malformed::TooMany => write!(f, "more than {MAX_ENTRIES} entries"),
            Malformed::BadName(i) => write!(f, "entry {i} has a malformed name"),
            Malformed::Duplicate(i) => write!(f, "entry {i} repeats an earlier entry's name"),
            Malformed::BadEntry(i) => write!(f, "entry {i} is not where the format places it"),
        }
    }
}

fn u32_at(archive: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(archive.get(o..o.checked_add(4)?)?.try_into().ok()?))
}

fn u64_at(archive: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(archive.get(o..o.checked_add(8)?)?.try_into().ok()?))
}

const fn align(n: usize) -> Option<usize> {
    match n.checked_add(ALIGN - 1) {
        Some(m) => Some(m / ALIGN * ALIGN),
        None => None,
    }
}

/// Whether `name` is a valid entry name.
pub fn name_ok(name: &[u8]) -> bool {
    !name.is_empty() && name.len() <= NAME_MAX && name.iter().all(|&c| c.is_ascii_alphanumeric() || c == b'.' || c == b'_' || c == b'-')
}

/// The name field of entry `i` (in a table already known to fit).
fn name_field(archive: &[u8], i: usize) -> &[u8] {
    let e = HEADER + i * ENTRY;
    &archive[e..e + 32]
}

/// The name of entry `i`: the field up to its first NUL.
fn name_of(field: &[u8]) -> &[u8] {
    &field[..field.iter().position(|&b| b == 0).unwrap_or(field.len())]
}

/// Validate `archive` against every rule of the format. `crc` continues a
/// CRC32C: given the state (all ones to start) and bytes, the state after
/// them ([`crc32c_update`], or a hardware one).
pub fn check(archive: &[u8], crc: impl Fn(u32, &[u8]) -> u32) -> Result<(), Malformed> {
    if archive.len() < HEADER {
        return Err(Malformed::Truncated);
    }
    if archive[..8] != MAGIC {
        return Err(Malformed::BadMagic);
    }
    let version = u32_at(archive, 8).ok_or(Malformed::Truncated)?;
    if version != VERSION {
        return Err(Malformed::BadVersion(version));
    }
    if u64_at(archive, 16) != Some(archive.len() as u64) {
        return Err(Malformed::Truncated);
    }
    let count = u32_at(archive, 12).ok_or(Malformed::Truncated)? as usize;
    if count > MAX_ENTRIES {
        return Err(Malformed::TooMany);
    }
    let table_end = HEADER + count * ENTRY;
    if table_end > archive.len() {
        return Err(Malformed::Truncated);
    }
    // The checksum, over the archive with its own field as zeros.
    let state = crc(crc(crc(!0, &archive[..24]), &[0; 4]), &archive[28..]);
    if !state != u32_at(archive, 24).ok_or(Malformed::Truncated)? {
        return Err(Malformed::BadChecksum);
    }
    if u32_at(archive, 28) != Some(0) {
        return Err(Malformed::NotZero);
    }
    let mut expect = align(table_end).ok_or(Malformed::Truncated)?;
    if expect > archive.len() || archive[table_end..expect].iter().any(|&b| b != 0) {
        return Err(if expect > archive.len() { Malformed::Truncated } else { Malformed::NotZero });
    }
    for i in 0..count {
        let field = name_field(archive, i);
        let name = name_of(field);
        if field[name.len()..].iter().any(|&b| b != 0) || !name_ok(name) {
            return Err(Malformed::BadName(i));
        }
        if (0..i).any(|j| name_of(name_field(archive, j)) == name) {
            return Err(Malformed::Duplicate(i));
        }
        let e = HEADER + i * ENTRY;
        let (offset, size) = (u64_at(archive, e + 32), u64_at(archive, e + 40));
        if offset != Some(expect as u64) {
            return Err(Malformed::BadEntry(i));
        }
        let end = size
            .and_then(|s| usize::try_from(s).ok())
            .and_then(|s| expect.checked_add(s))
            .filter(|&end| end <= archive.len())
            .ok_or(Malformed::BadEntry(i))?;
        let next = align(end).filter(|&n| n <= archive.len()).ok_or(Malformed::BadEntry(i))?;
        if archive[end..next].iter().any(|&b| b != 0) {
            return Err(Malformed::NotZero);
        }
        expect = next;
    }
    if expect != archive.len() {
        return Err(Malformed::Truncated);
    }
    Ok(())
}

/// The entries of an archive [`check`] accepted, as (name, data), in order.
/// On any other bytes: empty when the header is malformed, ending early at
/// the first malformed entry (never a panic).
pub fn entries(archive: &[u8]) -> impl Iterator<Item = (&str, &[u8])> {
    let ok =
        archive.get(..8) == Some(&MAGIC[..]) && u32_at(archive, 8) == Some(VERSION) && u64_at(archive, 16) == Some(archive.len() as u64);
    let count = if ok { u32_at(archive, 12).unwrap_or(0) as usize } else { 0 };
    (0..count.min(MAX_ENTRIES)).map_while(move |i| {
        let e = HEADER.checked_add(i.checked_mul(ENTRY)?)?;
        let raw = archive.get(e..e.checked_add(32)?)?;
        let name = core::str::from_utf8(name_of(raw)).ok()?;
        let off = usize::try_from(u64_at(archive, e + 32)?).ok()?;
        let size = usize::try_from(u64_at(archive, e + 40)?).ok()?;
        Some((name, archive.get(off..off.checked_add(size)?)?))
    })
}

/// The data of the entry called `name` in an archive [`check`] accepted (in
/// which names are unique), or `None`.
pub fn find<'a>(archive: &'a [u8], name: &str) -> Option<&'a [u8]> {
    entries(archive).find(|(n, _)| *n == name).map(|(_, d)| d)
}

/// CRC32C (Castagnoli, reflected polynomial 0x82F63B78), eight bytes at a
/// time with eight tables: what a processor without the instructions uses.
const TABLES: [[u32; 256]; 8] = {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0x82F6_3B78 } else { c >> 1 };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut n = 1;
    while n < 8 {
        let mut i = 0;
        while i < 256 {
            let prev = t[n - 1][i];
            t[n][i] = (prev >> 8) ^ t[0][(prev & 0xFF) as usize];
            i += 1;
        }
        n += 1;
    }
    t
};

/// Continue a CRC32C from `state` over `data` (start with `!0`, finish
/// with `!`).
pub fn crc32c_update(mut state: u32, data: &[u8]) -> u32 {
    let (words, tail) = data.as_chunks::<8>();
    for w in words {
        let v = u64::from_le_bytes(*w) ^ state as u64;
        state = TABLES[7][(v & 0xFF) as usize]
            ^ TABLES[6][(v >> 8 & 0xFF) as usize]
            ^ TABLES[5][(v >> 16 & 0xFF) as usize]
            ^ TABLES[4][(v >> 24 & 0xFF) as usize]
            ^ TABLES[3][(v >> 32 & 0xFF) as usize]
            ^ TABLES[2][(v >> 40 & 0xFF) as usize]
            ^ TABLES[1][(v >> 48 & 0xFF) as usize]
            ^ TABLES[0][(v >> 56) as usize];
    }
    for &b in tail {
        state = (state >> 8) ^ TABLES[0][((state ^ b as u32) & 0xFF) as usize];
    }
    state
}

/// The CRC32C of `data`.
pub fn crc32c(data: &[u8]) -> u32 {
    !crc32c_update(!0, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_matches_its_check_values() {
        // RFC 3720 appendix B.4 and the standard check value.
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(&[0u8; 32]), 0x8A91_36AA);
        assert_eq!(crc32c(&[0xFFu8; 32]), 0x62A8_AB43);
        let up: std::vec::Vec<u8> = (0..32).collect();
        assert_eq!(crc32c(&up), 0x46DD_794E);
        // In pieces, any split, the same.
        let data: std::vec::Vec<u8> = (0..1000u32).map(|i| (i * 37 % 251) as u8).collect();
        for split in [0, 1, 7, 8, 9, 500, 999, 1000] {
            assert_eq!(!crc32c_update(crc32c_update(!0, &data[..split]), &data[split..]), crc32c(&data));
        }
    }
}
