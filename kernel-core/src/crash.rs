//! The crash record: what a kernel panic leaves for the next boot.
//!
//! A panic on a machine nobody watches used to halt every processor for
//! ever, and its message and backtrace went with the machine. Now it writes
//! them, after the console's last lines, into a region of memory the loader
//! keeps out of every allocation and does not clear, and resets the machine
//! (unless a test harness is listening). The next boot finds the record in
//! the same region, checks it, says it on the console whole, gives it to the
//! root task, and clears it. RAM keeps its contents across a warm reset; a
//! record that did not survive (power lost, firmware that scrubs memory) is
//! simply absent, never misread: it is taken only with its magic, its length
//! and its checksum right.
//!
//! The functions work on the region as bytes, so the format is tested on
//! the host; the architecture maps the region and calls them.

/// Bytes the loader keeps for the record.
pub const REGION_BYTES: u64 = 16 * 1024;
/// "SILICRSH", little-endian.
const MAGIC: u64 = u64::from_le_bytes(*b"SILICRSH");
/// Magic, boot number, length, checksum, reserved.
const HEADER: usize = 32;
/// Where the boot count is: the header's last word, outside the record.
const COUNT: usize = 28;

/// FNV-1a over `parts`: enough to tell a record from leftover memory.
fn sum(parts: &[&[u8]]) -> u32 {
    let mut h: u32 = 0x811C_9DC5;
    for p in parts {
        for &b in *p {
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
    }
    h
}

fn word(region: &[u8], at: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&region[at..at + 8]);
    u64::from_le_bytes(w)
}

/// Start a record in `region`: what was there stops being one.
pub fn begin(region: &mut [u8]) {
    let n = region.len().min(COUNT);
    region[..n].fill(0);
}

/// Append `bytes` after the `len` bytes already in the record in `region`,
/// as much as fits; returns the new length.
pub fn append(region: &mut [u8], len: usize, bytes: &[u8]) -> usize {
    let room = region.len().saturating_sub(HEADER + len);
    let n = bytes.len().min(room);
    if n > 0 {
        region[HEADER + len..HEADER + len + n].copy_from_slice(&bytes[..n]);
    }
    len + n
}

/// Make the `len` bytes of the record in `region`, written in boot number
/// `boot`, a record the next boot will take.
pub fn seal(region: &mut [u8], boot: u64, len: usize) {
    if region.len() < HEADER || HEADER + len > region.len() {
        return;
    }
    let (b, l) = (boot.to_le_bytes(), (len as u64).to_le_bytes());
    let check = sum(&[&b, &l, &region[HEADER..HEADER + len]]);
    region[8..16].copy_from_slice(&b);
    region[16..24].copy_from_slice(&l);
    region[24..28].copy_from_slice(&check.to_le_bytes());
    region[0..8].copy_from_slice(&MAGIC.to_le_bytes());
}

/// A record being written into `region` (at least a header's bytes long).
pub struct Writer<'a> {
    region: &'a mut [u8],
    len: usize,
    boot: u64,
}

impl<'a> Writer<'a> {
    /// Start a record of boot number `boot` in `region`: anything there
    /// before stops being a record until [`Writer::seal`].
    pub fn begin(region: &'a mut [u8], boot: u64) -> Writer<'a> {
        begin(region);
        Writer { region, len: 0, boot }
    }

    /// Append `bytes`, as much of them as fits.
    pub fn append(&mut self, bytes: &[u8]) {
        self.len = append(self.region, self.len, bytes);
    }

    /// Make it a record the next boot will take.
    pub fn seal(self) {
        seal(self.region, self.boot, self.len);
    }
}

impl core::fmt::Write for Writer<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.append(s.as_bytes());
        Ok(())
    }
}

/// The record in `region`, if there is a whole one: the boot number it was
/// written in and its text.
pub fn read(region: &[u8]) -> Option<(u64, &[u8])> {
    if region.len() < HEADER || word(region, 0) != MAGIC {
        return None;
    }
    let (boot, len) = (word(region, 8), word(region, 16));
    let len = usize::try_from(len).ok().filter(|l| *l <= region.len() - HEADER)?;
    let text = &region[HEADER..HEADER + len];
    let check = u32::from_le_bytes([region[24], region[25], region[26], region[27]]);
    (check == sum(&[&boot.to_le_bytes(), &(len as u64).to_le_bytes(), text])).then_some((boot, text))
}

/// Forget the record in `region` (it has been reported); the boot count
/// stays.
pub fn clear(region: &mut [u8]) {
    let n = region.len().min(COUNT);
    region[..n].fill(0);
}

/// This boot's number: the region's count of boots, kept across warm resets
/// outside the record, plus one (1 on a machine that lost it).
pub fn next_boot(region: &mut [u8]) -> u64 {
    if region.len() < HEADER {
        return 0;
    }
    let n = u32::from_le_bytes([region[COUNT], region[COUNT + 1], region[COUNT + 2], region[COUNT + 3]]).wrapping_add(1).max(1);
    region[COUNT..COUNT + 4].copy_from_slice(&n.to_le_bytes());
    n as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use core::fmt::Write;
    use std::vec;

    #[test]
    fn a_sealed_record_is_read_back_whole_and_nothing_else_is() {
        let mut region = vec![0xA5u8; 4096];
        // Leftover memory is no record.
        assert_eq!(read(&region), None);
        let mut w = Writer::begin(&mut region, 7);
        w.append(b"console tail\r\n");
        let _ = write!(w, "PANIC: deliberate panic\r\n#0 {:#018x}\r\n", 0x1234u64);
        // Unsealed: still no record.
        assert_eq!(read(w.region), None);
        w.seal();
        let (boot, text) = read(&region).expect("a sealed record reads back");
        assert_eq!(boot, 7);
        assert_eq!(text, b"console tail\r\nPANIC: deliberate panic\r\n#0 0x0000000000001234\r\n");
        // One changed byte anywhere in it, or a length past the region, is no record.
        for at in [HEADER, HEADER + 5, 9, 17, 25] {
            let mut bad = region.clone();
            bad[at] ^= 1;
            assert_eq!(read(&bad), None, "byte {at} changed");
        }
        let mut long = region.clone();
        long[16..24].copy_from_slice(&(5000u64).to_le_bytes());
        assert_eq!(read(&long), None);
        clear(&mut region);
        assert_eq!(read(&region), None);
    }

    #[test]
    fn the_boot_count_outlives_records() {
        let mut region = vec![0u8; 256];
        assert_eq!(next_boot(&mut region), 1);
        assert_eq!(next_boot(&mut region), 2);
        let mut w = Writer::begin(&mut region, 2);
        w.append(b"x");
        w.seal();
        assert_eq!(read(&region).map(|(b, _)| b), Some(2));
        clear(&mut region);
        assert_eq!(next_boot(&mut region), 3);
    }

    #[test]
    fn a_record_longer_than_the_region_keeps_its_beginning() {
        let mut region = vec![0u8; 64];
        let mut w = Writer::begin(&mut region, 1);
        w.append(&[b'x'; 100]);
        w.seal();
        assert_eq!(read(&region).map(|(_, t)| t.len()), Some(64 - HEADER));
    }
}
