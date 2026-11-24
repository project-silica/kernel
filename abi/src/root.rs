//! What the root task finds when it starts.
//!
//! The kernel creates one thread from the ramdisk entry named `root` and
//! gives it every capability there is. Its CapSpace is laid out as below, and
//! its first argument register holds the address of a read-only [`RootInfo`].
//! The next four hold the boot secret ([`secret`]): every source of
//! randomness the loader found, hashed together, so it is unpredictable if
//! any one of them is. It is in registers, not in a frame, so that no frame
//! the root task hands on carries it: the root task keeps it in its own
//! memory and derives what others need from it.

/// Slot of the root thread's own Thread capability.
pub const SLOT_THREAD: u64 = 1;
/// Slot of the root CapSpace capability.
pub const SLOT_CAPSPACE: u64 = 2;
/// Slot of the root AddressSpace capability.
pub const SLOT_ASPACE: u64 = 3;
/// Slot of the Hw capability for every I/O port (x86_64; empty elsewhere).
pub const SLOT_PORTS: u64 = 4;
/// Slot of the Hw capability for every interrupt line.
pub const SLOT_IRQS: u64 = 5;
/// Slot of the root thread's IPC buffer frame.
pub const SLOT_IPCBUF: u64 = 6;
/// Slot of the console capability: writes to the kernel's console
/// (`Syscall::Console`).
pub const SLOT_CONSOLE: u64 = 8;
/// Slot of the power capability: turns the machine off (`Syscall::PowerOff`).
pub const SLOT_POWER: u64 = 9;
/// Slot of the device-identity capability: every device of the machine, by
/// PCI segment, bus, device and function (`Syscall::DeviceAttach`,
/// `Syscall::IrqBind` with `IRQ_MSI`).
pub const SLOT_DEVICES: u64 = 10;
/// Slot of the processor-time capability: every processor of the machine,
/// by index (`Syscall::SchedConfigure`: a weight above the default, or a
/// deadline reservation, on the processors it covers). Narrowed by
/// `CapCopy` as any range is.
pub const SLOT_SCHED: u64 = 11;
/// Slot of the Untyped the root task's own objects were made from; what is
/// left of it is the root task's to use.
pub const SLOT_BOOT_UNTYPED: u64 = 7;
/// Slot of the frame holding the [`RootInfo`] itself, so the root task can
/// hand the description of the machine on (read-only) to whoever manages
/// its devices.
pub const SLOT_INFO: u64 = 15;
/// First slot of the Untyped capabilities listed in [`RootInfo::untyped`].
pub const SLOT_FIRST_UNTYPED: u64 = 16;

/// `RootInfo::magic`.
pub const ROOT_MAGIC: u64 = 0x4F46_4E49_544F_4F52; // "ROOTINFO"
/// `RootInfo::entropy_sources`: the firmware's random number protocol.
pub const ENTROPY_FIRMWARE: u64 = 1;
/// `RootInfo::entropy_sources`: the processor's own generator (RDSEED or
/// RDRAND on x86, RNDRRS or RNDR on Arm).
pub const ENTROPY_CPU: u64 = 2;
/// Most Untyped descriptors in a [`RootInfo`].
pub const MAX_UNTYPED: usize = 1000;

/// One Untyped capability handed to the root task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct UntypedDesc {
    /// Physical address.
    pub base: u64,
    /// Size in bits.
    pub bits: u8,
    /// One of the `KIND_` constants.
    pub kind: u8,
    /// Zero.
    pub reserved: [u8; 6],
}

/// Ordinary memory.
pub const KIND_RAM: u8 = 0;
/// Device or firmware memory: not zeroed, only retyped into frames or
/// smaller untyped.
pub const KIND_DEVICE: u8 = 1;
/// Ordinary memory the firmware keeps data in (ACPI tables, runtime
/// services): never zeroed, only frames and smaller untyped, mapped as
/// ordinary memory. Its contents are the firmware's; reading the tables
/// is what it is for.
pub const KIND_FIRMWARE: u8 = 3;
/// Ordinary memory holding the ramdisk archive: like device memory it is
/// never zeroed and becomes only frames and smaller untyped, but those map
/// as ordinary memory.
pub const KIND_RAMDISK: u8 = 2;

/// The root task's start-up record. Read-only, mapped by the kernel.
#[repr(C)]
pub struct RootInfo {
    /// [`ROOT_MAGIC`].
    pub magic: u64,
    /// Number of slots in the root CapSpace.
    pub capspace_slots: u64,
    /// First slot the kernel left empty.
    pub first_free_slot: u64,
    /// Number of entries used in `untyped`.
    pub untyped_count: u64,
    /// Virtual address of the IPC buffer.
    pub ipc_buffer: u64,
    /// Physical address and length of the ramdisk archive (0 when absent).
    pub ramdisk: [u64; 2],
    /// Physical address of the ACPI RSDP (0 when absent).
    pub rsdp: u64,
    /// Framebuffer: physical base, width, height, pixels per line, format
    /// (0 none, 1 RGBX, 2 BGRX).
    pub framebuffer: [u64; 5],
    /// The processor's timestamp counter: ticks per second, its value when
    /// the boot loader started, and its value at monotonic time 0.
    pub clock: [u64; 3],
    /// Wall-clock time from the firmware, in seconds since 1970, and the
    /// monotonic nanoseconds at which it was read (both 0 when unknown).
    pub wall_clock: [u64; 2],
    /// Which sources went into the boot secret ([`ENTROPY_FIRMWARE`],
    /// [`ENTROPY_CPU`]); timing jitter always does, and is never counted.
    /// 0: no source the loader trusts was present, and the secret must not
    /// be used for keys until something better (a device) is mixed in.
    ///
    /// The secret itself is not here: this record describes the machine
    /// and is lent on (to whoever manages its devices), so it holds nothing
    /// secret. The root thread starts with it in its second to fifth
    /// argument registers ([`secret`]).
    pub entropy_sources: u64,
    /// Number of processors. Threads start on the processor that created
    /// them and can be moved to any index below this.
    pub cpus: u64,
    /// One past the highest address a device can be given in an address
    /// space made for devices, when the machine confines DMA (it has an
    /// IOMMU the kernel drives): a device then reaches only what its driver
    /// mapped for it. 0: it cannot, and a device that does DMA reaches all
    /// of memory whatever its driver holds, so starting such a driver is
    /// trusting it with the machine.
    pub dma_top: u64,
    /// Interrupt lines from this one up are message-signalled
    /// (`IRQ_MSI`); below it they are wires, numbered as ACPI numbers them.
    pub msi_first: u64,
    /// Length of the kernel command line in `cmdline`.
    pub cmdline_len: u64,
    /// The kernel command line.
    pub cmdline: [u8; 256],
    /// The Untyped capabilities, in slot order from `SLOT_FIRST_UNTYPED`.
    pub untyped: [UntypedDesc; MAX_UNTYPED],
}

/// The boot secret from the root thread's start registers: the second to
/// fifth argument registers (`a1` to `a4`), little-endian, in order. 32
/// bytes, independent of the kernel's own randomness (its address slide):
/// knowing one says nothing about the other.
pub fn secret(words: [u64; 4]) -> [u8; 32] {
    let mut s = [0u8; 32];
    for (i, w) in words.iter().enumerate() {
        s[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
    }
    s
}
