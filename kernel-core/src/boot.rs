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
    /// Firmware memory that must stay untouched (ACPI tables, runtime
    /// services, reserved ranges). Handed to userspace as device memory.
    Firmware = 5,
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
    /// Random bytes from the firmware; all zero when it has no source.
    pub entropy: [u8; 32],
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
        for word in self
            .0
            .split(|&b| b == b' ' || b == b'\n' || b == b'\r' || b == b'\t')
        {
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
