//! Silica UEFI loader. At M0 it is the boot stub that proves the harness end
//! to end: it writes `BOOT OK` to the first serial port and leaves through
//! the emulator's debug-exit device. B01 turns it into the real loader.
//!
//! Port I/O goes through the firmware's PCI Root Bridge I/O protocol, so the
//! stub needs no assembly.
#![no_std]
#![no_main]

mod efi;

use core::ffi::c_void;
use core::ptr;
use kernel_core::{BOOT_BANNER, ExitStatus};

/// First serial port (COM1) data register.
const COM1: u64 = 0x3F8;
/// QEMU `isa-debug-exit` device (iobase=0xf4, iosize=0x04).
const DEBUG_EXIT: u64 = 0xF4;

/// Write one byte to an I/O port through the firmware.
fn outb(io: *mut efi::PciRootBridgeIo, port: u64, mut value: u8) -> efi::Status {
    // SAFETY: `io` came from LocateProtocol and stays valid while boot
    // services run; `value` lives for the whole call; count 1 at width 8
    // reads exactly one byte from it.
    unsafe { ((*io).io_write)(io, efi::WIDTH_U8, port, 1, ptr::from_mut(&mut value).cast()) }
}

/// UEFI image entry point.
// SAFETY: `efi_main` is the entry symbol the UEFI target's linker expects;
// only the firmware calls it.
#[unsafe(no_mangle)]
pub extern "efiapi" fn efi_main(_image: efi::Handle, st: *const efi::SystemTable) -> efi::Status {
    let mut io: *mut c_void = ptr::null_mut();
    // SAFETY: the firmware passes a valid system table whose boot services
    // are usable until ExitBootServices, which the stub never calls.
    let status = unsafe {
        let bs = (*st).boot_services;
        ((*bs).locate_protocol)(&efi::PCI_ROOT_BRIDGE_IO, ptr::null_mut(), &mut io)
    };
    if status != efi::SUCCESS || io.is_null() {
        return status;
    }
    let io = io.cast::<efi::PciRootBridgeIo>();
    for b in BOOT_BANNER.bytes().chain(*b"\r\n") {
        outb(io, COM1, b);
    }
    outb(io, DEBUG_EXIT, ExitStatus::Success.code())
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
