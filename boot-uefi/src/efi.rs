//! The UEFI definitions the loader needs (UEFI 2.10). No external crate:
//! `uefi` is not on the allowlist (`spec/ARCHITECTURE.md` section 12).

use core::ffi::c_void;

/// EFI_STATUS.
pub type Status = usize;
/// EFI_SUCCESS.
pub const SUCCESS: Status = 0;
/// EFI_BUFFER_TOO_SMALL.
pub const BUFFER_TOO_SMALL: Status = (1 << (usize::BITS - 1)) | 5;
/// EFI_NOT_FOUND.
pub const NOT_FOUND: Status = (1 << (usize::BITS - 1)) | 14;
/// EFI_HANDLE.
pub type Handle = *mut c_void;

/// EFI_GUID.
#[repr(C)]
#[derive(PartialEq, Eq)]
pub struct Guid(pub u32, pub u16, pub u16, pub [u8; 8]);

/// EFI_LOADED_IMAGE_PROTOCOL_GUID.
pub const LOADED_IMAGE: Guid = Guid(0x5B1B_31A1, 0x9562, 0x11D2, [0x8E, 0x3F, 0x00, 0xA0, 0xC9, 0x69, 0x72, 0x3B]);
/// EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID.
pub const SIMPLE_FS: Guid = Guid(0x964E_5B22, 0x6459, 0x11D2, [0x8E, 0x39, 0x00, 0xA0, 0xC9, 0x69, 0x72, 0x3B]);
/// EFI_GRAPHICS_OUTPUT_PROTOCOL_GUID.
pub const GOP: Guid = Guid(0x9042_A9DE, 0x23DC, 0x4A38, [0x96, 0xFB, 0x7A, 0xDE, 0xD0, 0x80, 0x51, 0x6A]);
/// EFI_RNG_PROTOCOL_GUID.
pub const RNG: Guid = Guid(0x3152_BCA5, 0xEADE, 0x433D, [0x86, 0x2E, 0xC0, 0x1C, 0xDC, 0x29, 0x1F, 0x44]);
/// EFI_ACPI_20_TABLE_GUID.
pub const ACPI2: Guid = Guid(0x8868_E871, 0xE4F1, 0x11D3, [0xBC, 0x22, 0x00, 0x80, 0xC7, 0x3C, 0x88, 0x81]);

/// EFI_SYSTEM_TABLE.
#[repr(C)]
pub struct SystemTable {
    hdr: [u8; 24],
    firmware_vendor: *const u16,
    firmware_revision: u32,
    con_in_handle: Handle,
    con_in: *mut c_void,
    con_out_handle: Handle,
    /// EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL of the console.
    pub con_out: *mut TextOutput,
    std_err_handle: Handle,
    std_err: *mut c_void,
    /// EFI_RUNTIME_SERVICES.
    pub runtime_services: *const RuntimeServices,
    /// EFI_BOOT_SERVICES.
    pub boot_services: *const BootServices,
    /// Number of entries in `config_table`.
    pub config_count: usize,
    /// EFI_CONFIGURATION_TABLE array.
    pub config_table: *const ConfigTable,
}

/// EFI_CONFIGURATION_TABLE.
#[repr(C)]
pub struct ConfigTable {
    /// Which table.
    pub guid: Guid,
    /// The table.
    pub table: *const c_void,
}

/// EFI_RUNTIME_SERVICES, up to SetVariable (UEFI 2.10, section 4.5).
#[repr(C)]
pub struct RuntimeServices {
    hdr: [u8; 24],
    /// GetTime(Time, Capabilities).
    pub get_time: unsafe extern "efiapi" fn(*mut Time, *mut c_void) -> Status,
    set_time: usize,
    get_wakeup_time: usize,
    set_wakeup_time: usize,
    set_virtual_address_map: usize,
    convert_pointer: usize,
    /// GetVariable(Name, VendorGuid, Attributes (optional), DataSize, Data).
    pub get_variable: unsafe extern "efiapi" fn(*const u16, *const Guid, *mut u32, *mut usize, *mut c_void) -> Status,
    get_next_variable_name: usize,
    /// SetVariable(Name, VendorGuid, Attributes, DataSize, Data).
    pub set_variable: unsafe extern "efiapi" fn(*const u16, *const Guid, u32, usize, *const c_void) -> Status,
}

/// EFI_VARIABLE_NON_VOLATILE: kept across resets.
pub const VARIABLE_NON_VOLATILE: u32 = 1;
/// EFI_VARIABLE_BOOTSERVICE_ACCESS: readable while boot services run.
/// Without EFI_VARIABLE_RUNTIME_ACCESS (4), not after ExitBootServices:
/// no operating system can ask the firmware for it.
pub const VARIABLE_BOOTSERVICE_ACCESS: u32 = 2;
/// Silica's vendor GUID for its UEFI variables (generated for it).
pub const SILICA_VARIABLES: Guid = Guid(0x5C1A_1CA0, 0x2026, 0x4B27, [0x9E, 0x5A, 0x71, 0xCA, 0x5E, 0xC2, 0xE7, 0x01]);

/// EFI_TIME.
#[repr(C)]
#[derive(Default)]
pub struct Time {
    /// 1900..9999.
    pub year: u16,
    /// 1..12.
    pub month: u8,
    /// 1..31.
    pub day: u8,
    /// 0..23.
    pub hour: u8,
    /// 0..59.
    pub minute: u8,
    /// 0..59.
    pub second: u8,
    pad1: u8,
    nanosecond: u32,
    /// Minutes from UTC, or 0x7FF for "local time, unspecified".
    pub timezone: i16,
    daylight: u8,
    pad2: u8,
}

/// EFI_BOOT_SERVICES, as a table of function slots after the header.
#[repr(C)]
pub struct BootServices {
    hdr: [u8; 24],
    raise_tpl: usize,
    restore_tpl: usize,
    /// AllocatePages(Type, MemoryType, Pages, Memory).
    pub allocate_pages: unsafe extern "efiapi" fn(u32, u32, usize, *mut u64) -> Status,
    free_pages: usize,
    /// GetMemoryMap(MapSize, Map, MapKey, DescriptorSize, DescriptorVersion).
    pub get_memory_map: unsafe extern "efiapi" fn(*mut usize, *mut u8, *mut usize, *mut usize, *mut u32) -> Status,
    unused1: [usize; 11],
    /// HandleProtocol(Handle, Protocol, Interface).
    pub handle_protocol: unsafe extern "efiapi" fn(Handle, *const Guid, *mut *mut c_void) -> Status,
    unused2: [usize; 9],
    /// ExitBootServices(ImageHandle, MapKey).
    pub exit_boot_services: unsafe extern "efiapi" fn(Handle, usize) -> Status,
    unused3: [usize; 10],
    /// LocateProtocol(Protocol, Registration, Interface).
    pub locate_protocol: unsafe extern "efiapi" fn(*const Guid, *mut c_void, *mut *mut c_void) -> Status,
}

/// EFI_MEMORY_DESCRIPTOR.
#[repr(C)]
pub struct MemoryDescriptor {
    /// EFI_MEMORY_TYPE.
    pub ty: u32,
    /// First byte.
    pub phys: u64,
    virt: u64,
    /// Length in 4 KiB pages.
    pub pages: u64,
    /// EFI_MEMORY_* attributes ([`MEMORY_RUNTIME`]).
    pub attr: u64,
}

/// EFI_LOADED_IMAGE_PROTOCOL, up to DeviceHandle.
#[repr(C)]
pub struct LoadedImage {
    revision: u32,
    parent: Handle,
    system_table: *const c_void,
    /// The device the image was loaded from.
    pub device: Handle,
}

/// EFI_SIMPLE_FILE_SYSTEM_PROTOCOL.
#[repr(C)]
pub struct SimpleFs {
    revision: u64,
    /// OpenVolume(This, Root).
    pub open_volume: unsafe extern "efiapi" fn(*mut SimpleFs, *mut *mut File) -> Status,
}

/// EFI_FILE_PROTOCOL, up to SetPosition.
#[repr(C)]
pub struct File {
    revision: u64,
    /// Open(This, NewHandle, FileName, OpenMode, Attributes).
    pub open: unsafe extern "efiapi" fn(*mut File, *mut *mut File, *const u16, u64, u64) -> Status,
    /// Close(This).
    pub close: unsafe extern "efiapi" fn(*mut File) -> Status,
    delete: usize,
    /// Read(This, BufferSize, Buffer).
    pub read: unsafe extern "efiapi" fn(*mut File, *mut usize, *mut u8) -> Status,
    write: usize,
    /// GetPosition(This, Position).
    pub get_position: unsafe extern "efiapi" fn(*mut File, *mut u64) -> Status,
    /// SetPosition(This, Position).
    pub set_position: unsafe extern "efiapi" fn(*mut File, u64) -> Status,
}

/// EFI_GRAPHICS_OUTPUT_PROTOCOL.
#[repr(C)]
pub struct Gop {
    query_mode: usize,
    set_mode: usize,
    blt: usize,
    /// Current mode.
    pub mode: *const GopMode,
}

/// EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE.
#[repr(C)]
pub struct GopMode {
    max_mode: u32,
    mode: u32,
    /// Mode information.
    pub info: *const GopInfo,
    info_size: usize,
    /// Framebuffer physical address.
    pub fb_base: u64,
    fb_size: usize,
}

/// EFI_GRAPHICS_OUTPUT_MODE_INFORMATION.
#[repr(C)]
pub struct GopInfo {
    version: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// 0 RGBX, 1 BGRX, 2 bit mask, 3 no framebuffer.
    pub format: u32,
    masks: [u32; 4],
    /// Pixels per scan line.
    pub stride: u32,
}

/// EFI_RNG_PROTOCOL.
#[repr(C)]
pub struct Rng {
    get_info: usize,
    /// GetRNG(This, Algorithm, ValueLength, Value).
    pub get_rng: unsafe extern "efiapi" fn(*mut Rng, *const Guid, usize, *mut u8) -> Status,
}

/// EfiConventionalMemory and the types free after ExitBootServices.
pub const MEM_LOADER_CODE: u32 = 1;
/// EfiLoaderData.
pub const MEM_LOADER_DATA: u32 = 2;
/// EfiBootServicesCode.
pub const MEM_BS_CODE: u32 = 3;
/// EfiBootServicesData.
pub const MEM_BS_DATA: u32 = 4;
/// EfiConventionalMemory.
pub const MEM_CONVENTIONAL: u32 = 7;
/// EfiRuntimeServicesCode and EfiRuntimeServicesData.
pub const MEM_RUNTIME_CODE: u32 = 5;
pub const MEM_RUNTIME_DATA: u32 = 6;
/// OS-defined memory type: the kernel image and boot page tables.
pub const MEM_SILICA_KERNEL: u32 = 0x8000_0001;
/// OS-defined memory type: the ramdisk.
pub const MEM_SILICA_RAMDISK: u32 = 0x8000_0002;
/// OS-defined memory type: the crash record (`kernel_core::crash`).
pub const MEM_SILICA_CRASH: u32 = 0x8000_0003;

/// EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL (the part this loader uses).
#[repr(C)]
pub struct TextOutput {
    reset: *const c_void,
    /// Write a NUL-terminated UCS-2 string.
    pub output_string: unsafe extern "efiapi" fn(*mut TextOutput, *const u16) -> Status,
}

/// EfiReservedMemoryType: not to be used (may not even be memory).
pub const MEM_RESERVED: u32 = 0;
/// EfiMemoryMappedIO: device registers.
pub const MEM_MMIO: u32 = 11;
/// EfiMemoryMappedIOPortSpace: device registers.
pub const MEM_MMIO_PORT: u32 = 12;
/// EFI_MEMORY_RUNTIME: the firmware keeps the range for its runtime services.
pub const MEMORY_RUNTIME: u64 = 1 << 63;
