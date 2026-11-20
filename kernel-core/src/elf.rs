//! A reader for the ELF64 images the system loads: the kernel (by the loader)
//! and the root task (by the kernel). It trusts nothing in the file: every
//! offset is checked before use and a malformed file is an error.

/// Why an image was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElfError {
    /// Too short, wrong magic, not 64-bit little-endian, or not an executable.
    NotElf,
    /// Built for another machine.
    WrongMachine,
    /// A header or segment lies outside the file or overflows.
    Truncated,
    /// A loadable segment has a size, alignment or address the loader cannot honour.
    BadSegment,
}

/// `e_machine` for x86_64.
pub const EM_X86_64: u16 = 62;
/// `e_machine` for aarch64.
pub const EM_AARCH64: u16 = 183;
/// `e_machine` for RISC-V.
pub const EM_RISCV: u16 = 243;

/// Segment may be executed.
pub const PF_X: u32 = 1;
/// Segment may be written.
pub const PF_W: u32 = 2;

/// One loadable segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment<'a> {
    /// Virtual address of the first byte.
    pub vaddr: u64,
    /// Size in memory; bytes past `data` are zero.
    pub memsz: u64,
    /// `PF_` flags.
    pub flags: u32,
    /// Bytes present in the file.
    pub data: &'a [u8],
}

/// A validated image.
#[derive(Clone, Copy, Debug)]
pub struct Elf<'a> {
    file: &'a [u8],
    /// Entry point.
    pub entry: u64,
    phoff: usize,
    phnum: usize,
    /// Whether the image is position-independent (`ET_DYN`).
    pub pie: bool,
}

fn u16_at(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o.checked_add(2)?)?.try_into().ok()?))
}
fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o.checked_add(4)?)?.try_into().ok()?))
}
fn u64_at(b: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(o..o.checked_add(8)?)?.try_into().ok()?))
}

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PHENT: usize = 56;

impl<'a> Elf<'a> {
    /// Validate the headers of `file` for `machine`.
    pub fn parse(file: &'a [u8], machine: u16) -> Result<Elf<'a>, ElfError> {
        if file.len() < 64 || file[..4] != *b"\x7fELF" || file[4] != 2 || file[5] != 1 {
            return Err(ElfError::NotElf);
        }
        let ty = u16_at(file, 16).ok_or(ElfError::NotElf)?;
        if ty != 2 && ty != 3 {
            return Err(ElfError::NotElf);
        }
        if u16_at(file, 18) != Some(machine) {
            return Err(ElfError::WrongMachine);
        }
        let entry = u64_at(file, 24).ok_or(ElfError::Truncated)?;
        let phoff = usize::try_from(u64_at(file, 32).ok_or(ElfError::Truncated)?).map_err(|_| ElfError::Truncated)?;
        let phentsize = u16_at(file, 54).ok_or(ElfError::Truncated)? as usize;
        let phnum = u16_at(file, 56).ok_or(ElfError::Truncated)? as usize;
        if phentsize != PHENT {
            return Err(ElfError::NotElf);
        }
        let end = phnum.checked_mul(PHENT).and_then(|n| n.checked_add(phoff)).ok_or(ElfError::Truncated)?;
        if end > file.len() {
            return Err(ElfError::Truncated);
        }
        let elf = Elf { file, entry, phoff, phnum, pie: ty == 3 };
        for s in elf.raw_segments() {
            s?;
        }
        Ok(elf)
    }

    fn raw_segments(&self) -> impl Iterator<Item = Result<Segment<'a>, ElfError>> + '_ {
        (0..self.phnum).filter_map(move |i| {
            let o = self.phoff + i * PHENT;
            let f = self.file;
            if u32_at(f, o)? != PT_LOAD {
                return None;
            }
            Some((|| {
                let flags = u32_at(f, o + 4).ok_or(ElfError::Truncated)?;
                let off = u64_at(f, o + 8).ok_or(ElfError::Truncated)?;
                let vaddr = u64_at(f, o + 16).ok_or(ElfError::Truncated)?;
                let filesz = u64_at(f, o + 32).ok_or(ElfError::Truncated)?;
                let memsz = u64_at(f, o + 40).ok_or(ElfError::Truncated)?;
                if filesz > memsz || vaddr.checked_add(memsz).is_none() {
                    return Err(ElfError::BadSegment);
                }
                let off = usize::try_from(off).map_err(|_| ElfError::Truncated)?;
                let len = usize::try_from(filesz).map_err(|_| ElfError::Truncated)?;
                let data = off.checked_add(len).and_then(|e| f.get(off..e)).ok_or(ElfError::Truncated)?;
                Ok(Segment { vaddr, memsz, flags, data })
            })())
        })
    }

    /// The loadable segments, in file order.
    pub fn segments(&self) -> impl Iterator<Item = Segment<'a>> + '_ {
        self.raw_segments().filter_map(Result::ok)
    }

    /// Lowest and one-past-highest virtual address of the loadable segments,
    /// rounded out to 4 KiB. `None` for an image with no segments.
    pub fn span(&self) -> Option<(u64, u64)> {
        let mut lo = u64::MAX;
        let mut hi = 0u64;
        for s in self.segments() {
            lo = lo.min(s.vaddr & !0xFFF);
            hi = hi.max(s.vaddr.checked_add(s.memsz)?.checked_add(0xFFF)? & !0xFFF);
        }
        (lo < hi).then_some((lo, hi))
    }

    /// The `R_*_RELATIVE` relocations of a position-independent image, as
    /// (virtual address to patch, addend) pairs. Other relocation types are
    /// reported as `Err`: the kernel is linked so that none exist.
    pub fn relative_relocs(&self) -> impl Iterator<Item = Result<(u64, u64), ElfError>> + '_ {
        let f = self.file;
        let mut rela = 0u64;
        let mut relasz = 0u64;
        for i in 0..self.phnum {
            let o = self.phoff + i * PHENT;
            if u32_at(f, o) != Some(PT_DYNAMIC) {
                continue;
            }
            // The dynamic section: (tag, value) pairs inside the file.
            let off = usize::try_from(u64_at(f, o + 8).unwrap_or(0)).unwrap_or(usize::MAX);
            let len = usize::try_from(u64_at(f, o + 32).unwrap_or(0)).unwrap_or(usize::MAX);
            let Some(dynamic) = off.checked_add(len).and_then(|end| f.get(off..end)) else {
                continue;
            };
            for pair in dynamic.as_chunks::<16>().0 {
                let (Some(tag), Some(val)) = (u64_at(pair, 0), u64_at(pair, 8)) else {
                    break;
                };
                match tag {
                    0 => break,
                    7 => rela = val,
                    8 => relasz = val,
                    _ => {}
                }
            }
        }
        // DT_RELA is a virtual address; find its file offset through the segments.
        let table = self.segments().find_map(|s| {
            let o = usize::try_from(rela.checked_sub(s.vaddr)?).ok()?;
            s.data.get(o..o.checked_add(usize::try_from(relasz).ok()?)?)
        });
        let table = if relasz == 0 { Some(&f[..0]) } else { table };
        let count = table.map_or(1, |t| t.len() / 24);
        (0..count).map(move |i| {
            let t = table.ok_or(ElfError::Truncated)?;
            let at = u64_at(t, i * 24).ok_or(ElfError::Truncated)?;
            let info = u64_at(t, i * 24 + 8).ok_or(ElfError::Truncated)?;
            let addend = u64_at(t, i * 24 + 16).ok_or(ElfError::Truncated)?;
            // R_X86_64_RELATIVE = 8, R_AARCH64_RELATIVE = 1027, R_RISCV_RELATIVE = 3.
            match info & 0xFFFF_FFFF {
                8 | 1027 | 3 => Ok((at, addend)),
                _ => Err(ElfError::BadSegment),
            }
        })
    }
}
