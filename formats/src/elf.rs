//! A reader for the ELF64 images the system loads: the kernel (by the
//! loader), the root task (by the kernel) and every other program (by the
//! process spawner). It trusts nothing in the file: every offset is checked
//! before use and a malformed file is an error.

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
/// `e_machine` of the machine this code is built for: what a program loader
/// running here accepts.
pub const EM_NATIVE: u16 = if cfg!(target_arch = "aarch64") {
    EM_AARCH64
} else if cfg!(target_arch = "riscv64") {
    EM_RISCV
} else {
    EM_X86_64
};

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
    /// The `R_*_RELATIVE` relocation number of the image's architecture.
    relative: u32,
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
        let relative = match machine {
            EM_X86_64 => 8,
            EM_AARCH64 => 1027,
            EM_RISCV => 3,
            _ => 0, // none: a position-independent image cannot be relocated
        };
        let elf = Elf { file, entry, phoff, phnum, pie: ty == 3, relative };
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
            // Only the image's own architecture's number: the same number
            // means another relocation elsewhere (3 is R_X86_64_PC32).
            if info & 0xFFFF_FFFF == self.relative as u64 { Ok((at, addend)) } else { Err(ElfError::BadSegment) }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A position-independent image for `machine` with one relocation of
    /// type `ty`: one PT_LOAD over the whole file at 0x1000, a PT_DYNAMIC
    /// naming the relocation table.
    fn image(machine: u16, ty: u64) -> [u8; 256] {
        let mut f = [0u8; 256];
        let put = |f: &mut [u8; 256], o: usize, v: &[u8]| f[o..o + v.len()].copy_from_slice(v);
        put(&mut f, 0, b"\x7fELF\x02\x01\x01");
        put(&mut f, 16, &3u16.to_le_bytes()); // ET_DYN
        put(&mut f, 18, &machine.to_le_bytes());
        put(&mut f, 32, &64u64.to_le_bytes()); // program headers
        put(&mut f, 54, &(PHENT as u16).to_le_bytes());
        put(&mut f, 56, &2u16.to_le_bytes());
        // PT_LOAD: offset 0, vaddr 0x1000, the whole file, readable.
        put(&mut f, 64, &PT_LOAD.to_le_bytes());
        put(&mut f, 68, &4u32.to_le_bytes());
        put(&mut f, 64 + 16, &0x1000u64.to_le_bytes());
        put(&mut f, 64 + 32, &256u64.to_le_bytes());
        put(&mut f, 64 + 40, &256u64.to_le_bytes());
        // PT_DYNAMIC at file offset 176: DT_RELA (vaddr of offset 224), DT_RELASZ, DT_NULL.
        put(&mut f, 120, &PT_DYNAMIC.to_le_bytes());
        put(&mut f, 120 + 8, &176u64.to_le_bytes());
        put(&mut f, 120 + 32, &48u64.to_le_bytes());
        put(&mut f, 176, &7u64.to_le_bytes());
        put(&mut f, 184, &(0x1000u64 + 224).to_le_bytes());
        put(&mut f, 192, &8u64.to_le_bytes());
        put(&mut f, 200, &24u64.to_le_bytes());
        // The relocation: patch 0x1008 with 0x42.
        put(&mut f, 224, &0x1008u64.to_le_bytes());
        put(&mut f, 232, &ty.to_le_bytes());
        put(&mut f, 240, &0x42u64.to_le_bytes());
        f
    }

    fn relocs(machine: u16, ty: u64) -> Vec<Result<(u64, u64), ElfError>> {
        let f = image(machine, ty);
        Elf::parse(&f, machine).unwrap().relative_relocs().collect()
    }

    #[test]
    fn only_the_images_own_relative_relocation_is_accepted() {
        assert_eq!(relocs(EM_X86_64, 8), [Ok((0x1008, 0x42))]);
        assert_eq!(relocs(EM_AARCH64, 1027), [Ok((0x1008, 0x42))]);
        assert_eq!(relocs(EM_RISCV, 3), [Ok((0x1008, 0x42))]);
        // 3 is R_RISCV_RELATIVE but R_X86_64_PC32; 8 is R_X86_64_RELATIVE
        // but something else on aarch64.
        assert_eq!(relocs(EM_X86_64, 3), [Err(ElfError::BadSegment)]);
        assert_eq!(relocs(EM_AARCH64, 8), [Err(ElfError::BadSegment)]);
    }
}
