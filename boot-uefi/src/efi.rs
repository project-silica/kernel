//! The few UEFI definitions the M0 stub needs (UEFI 2.10). No external crate:
//! `uefi` is not on the allowlist (`spec/ARCHITECTURE.md` section 12).

use core::ffi::c_void;

/// EFI_STATUS.
pub type Status = usize;
/// EFI_SUCCESS.
pub const SUCCESS: Status = 0;
/// EFI_HANDLE.
pub type Handle = *mut c_void;

/// EFI_GUID.
#[repr(C)]
pub struct Guid(pub u32, pub u16, pub u16, pub [u8; 8]);

/// EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL_GUID.
pub const PCI_ROOT_BRIDGE_IO: Guid = Guid(
    0x2F70_7EBB,
    0x4A1A,
    0x11D4,
    [0x9A, 0x38, 0x00, 0x90, 0x27, 0x3F, 0xC1, 0x4D],
);

/// EFI_SYSTEM_TABLE, up to the fields the stub reads.
#[repr(C)]
pub struct SystemTable {
    hdr: [u8; 24],
    firmware_vendor: *const u16,
    firmware_revision: u32,
    con_in_handle: Handle,
    con_in: *mut c_void,
    con_out_handle: Handle,
    con_out: *mut c_void,
    std_err_handle: Handle,
    std_err: *mut c_void,
    runtime_services: *mut c_void,
    /// EFI_BOOT_SERVICES.
    pub boot_services: *const BootServices,
}

/// EFI_BOOT_SERVICES, up to LocateProtocol.
#[repr(C)]
pub struct BootServices {
    hdr: [u8; 24],
    /// RaiseTPL through OpenProtocolInformation, ProtocolsPerHandle and
    /// LocateHandleBuffer: 37 entries the stub does not call.
    unused: [usize; 37],
    /// LocateProtocol.
    pub locate_protocol:
        unsafe extern "efiapi" fn(*const Guid, *mut c_void, *mut *mut c_void) -> Status,
}

/// EFI_PCI_ROOT_BRIDGE_IO_PROTOCOL, up to Io.Write.
#[repr(C)]
pub struct PciRootBridgeIo {
    parent_handle: Handle,
    poll_mem: usize,
    poll_io: usize,
    mem_read: usize,
    mem_write: usize,
    io_read: usize,
    /// Io.Write(This, Width, Address, Count, Buffer).
    pub io_write:
        unsafe extern "efiapi" fn(*mut PciRootBridgeIo, u32, u64, usize, *mut c_void) -> Status,
}

/// EfiPciWidthUint8.
pub const WIDTH_U8: u32 = 0;
