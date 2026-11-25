//! The boot protocol: what the loader hands to the kernel (component B02),
//! and the kernel command line (B06).

/// `BootInfo::magic`.
pub const BOOT_MAGIC: u64 = 0x544F_4F42_4C49_5321; // "!SILBOOT"
/// Version of this structure. The kernel refuses any other.
pub const BOOT_VERSION: u32 = 1;

/// Virtual address at which the loader maps all physical memory.
pub const PHYS_MAP_BASE: u64 = 0xFFFF_8000_0000_0000;
/// Lowest virtual address the kernel image may be loaded at.
pub const KERNEL_BASE: u64 = 0xFFFF_FFFF_8000_0000;

/// Kind of a physical memory region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum RegionKind {
    /// Free memory.
    Usable = 1,
    /// The kernel image and the page tables the loader built. Kept by the kernel.
    Kernel = 2,
    /// The ramdisk archive.
    Ramdisk = 3,
    /// Loader data that is free once the kernel has read it (this structure,
    /// the region table, the command line).
    Loader = 4,
    /// Firmware ranges that must stay untouched and may not be memory at
    /// all (memory-mapped I/O, reserved ranges). Handed to userspace as
    /// device memory.
    Firmware = 5,
    /// Firmware data in RAM that must stay untouched (ACPI tables, runtime
    /// services). Handed to userspace as kept memory that maps as ordinary
    /// memory: tables are read with ordinary loads, unaligned ones too
    /// (on Arm a device mapping faults on those).
    FirmwareMemory = 6,
}

/// One physical memory region. Regions do not overlap and are sorted by base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Region {
    /// First byte.
    pub base: u64,
    /// Length in bytes, a multiple of 4096.
    pub len: u64,
    /// A [`RegionKind`] value.
    pub kind: u32,
    /// Zero.
    pub reserved: u32,
}

/// Handed to the kernel entry point by physical address in the first
/// argument register. Every address in it is physical.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct BootInfo {
    /// [`BOOT_MAGIC`].
    pub magic: u64,
    /// [`BOOT_VERSION`].
    pub version: u32,
    /// Zero.
    pub reserved: u32,
    /// Bytes of physical memory mapped at [`PHYS_MAP_BASE`].
    pub phys_map_len: u64,
    /// Where the kernel image is in physical memory.
    pub kernel_phys: u64,
    /// Where the kernel image is mapped (link address plus slide).
    pub kernel_virt: u64,
    /// Length of the mapped kernel image.
    pub kernel_len: u64,
    /// Difference between the address the kernel runs at and its link address.
    pub kernel_slide: u64,
    /// Region table and its entry count.
    pub regions: u64,
    /// Number of entries in `regions`.
    pub region_count: u64,
    /// Ramdisk archive (length 0 when absent).
    pub ramdisk: u64,
    /// Length of the ramdisk archive.
    pub ramdisk_len: u64,
    /// Kernel command line bytes (length 0 when absent).
    pub cmdline: u64,
    /// Length of the command line.
    pub cmdline_len: u64,
    /// ACPI RSDP, or 0.
    pub rsdp: u64,
    /// Framebuffer base, or 0.
    pub fb_base: u64,
    /// Framebuffer width in pixels.
    pub fb_width: u32,
    /// Framebuffer height in pixels.
    pub fb_height: u32,
    /// Pixels per scan line.
    pub fb_stride: u32,
    /// 0 none, 1 RGBX, 2 BGRX (8 bits each, byte order in memory).
    pub fb_format: u32,
    /// The root task's secret, given to it in its start registers
    /// (`silica_abi::root::secret`).
    pub entropy: [u8; 32],
    /// `RootInfo::entropy_sources`.
    pub entropy_sources: u64,
    /// Wall-clock time read from the firmware, seconds since 1970 (0 unknown).
    pub wall_secs: u64,
    /// Timestamp counter when `wall_secs` was read.
    pub wall_tsc: u64,
    /// Timestamp counter when the loader started.
    pub loader_tsc: u64,
}

/// The kernel command line: space-separated `key=value` words and bare flags.
#[derive(Clone, Copy)]
pub struct Cmdline<'a>(pub &'a [u8]);

impl<'a> Cmdline<'a> {
    /// The value of `key`, the empty slice for a bare flag, `None` when absent.
    pub fn get(&self, key: &str) -> Option<&'a [u8]> {
        for word in self.0.split(|&b| b == b' ' || b == b'\n' || b == b'\r' || b == b'\t') {
            let (k, v) = match word.iter().position(|&b| b == b'=') {
                Some(i) => (&word[..i], &word[i + 1..]),
                None => (word, &word[word.len()..]),
            };
            if !k.is_empty() && k == key.as_bytes() {
                return Some(v);
            }
        }
        None
    }
}

/// The copy of the loader's memory map the kernel takes its own memory from
/// before handing the rest to the root task.
pub struct RegionTable {
    r: [Region; 1100],
    n: usize,
}

impl RegionTable {
    /// An empty table.
    pub const fn new() -> RegionTable {
        RegionTable { r: [Region { base: 0, len: 0, kind: 0, reserved: 0 }; 1100], n: 0 }
    }

    /// Copy the loader's regions in, leaving room for the splits `carve` makes.
    pub fn fill(&mut self, from: &[Region]) {
        self.n = from.len().min(self.r.len() - 64);
        self.r[..self.n].copy_from_slice(&from[..self.n]);
    }

    /// The regions.
    pub fn as_slice(&self) -> &[Region] {
        &self.r[..self.n]
    }

    /// Take `len` bytes aligned to `align` from free memory inside
    /// `[min, max)`, as high as possible, and mark them the kernel's.
    pub fn carve(&mut self, len: u64, align: u64, min: u64, max: u64) -> Option<u64> {
        for i in (0..self.n).rev() {
            let r = self.r[i];
            if r.kind != RegionKind::Usable as u32 {
                continue;
            }
            let (lo, hi) = (r.base.max(min), (r.base + r.len).min(max));
            if hi < lo + len {
                continue;
            }
            let at = (hi - len) & !(align - 1);
            if at < lo || self.n + 2 > self.r.len() {
                continue;
            }
            // Replace region i by up to three: before, the carved part, after.
            let parts =
                [(r.base, at - r.base, r.kind), (at, len, RegionKind::Kernel as u32), (at + len, r.base + r.len - at - len, r.kind)];
            let keep = parts.iter().filter(|p| p.1 != 0).count();
            self.r.copy_within(i + 1..self.n, i + keep);
            let mut j = i;
            for (base, len, kind) in parts {
                if len != 0 {
                    self.r[j] = Region { base, len, kind, reserved: 0 };
                    j += 1;
                }
            }
            self.n = self.n + keep - 1;
            return Some(at);
        }
        None
    }
}

impl Default for RegionTable {
    fn default() -> RegionTable {
        RegionTable::new()
    }
}

/// Whether the machine starts with a boot secret gathered from `sources`
/// (`RootInfo::entropy_sources`: the loader's trusted sources, the
/// processor's generator and the firmware's; 0 is timing jitter alone),
/// when the command line's `entropy=` says `setting`. Every program's seed
/// and every service's key descends from the boot secret, so without a
/// trusted source the machine is not started, unless its owner accepts
/// that (`entropy=accept`): `Err` with the line that says why, or `Ok`
/// with a warning line to say (or none).
pub fn entropy_verdict(sources: u64, setting: Option<&[u8]>) -> Result<Option<&'static str>, &'static str> {
    match (sources, setting) {
        (0, Some(b"accept")) => Ok(Some(
            "WARNING: kernel: no source of randomness the loader trusts; the owner accepted that every key and seed on this machine may be guessable (entropy=accept)",
        )),
        (0, _) => Err(
            "ERROR: kernel: the loader found no source of randomness it trusts (the processor's RDSEED or RDRAND, Arm's RNDR, the firmware's generator): every key and seed on this machine would be guessable. Not started; `entropy=accept` on the kernel command line runs it anyway",
        ),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod entropy {
    #[test]
    fn a_boot_with_no_trusted_source_is_refused_unless_the_owner_accepts() {
        use super::entropy_verdict as v;
        assert!(v(0, None).is_err());
        assert!(v(0, Some(b"yes")).is_err());
        assert!(matches!(v(0, Some(b"accept")), Ok(Some(_))));
        assert_eq!(v(1, None), Ok(None));
        assert_eq!(v(3, Some(b"accept")), Ok(None));
    }
}
