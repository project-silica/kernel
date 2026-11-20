//! The first serial port (16550 at 0x3F8): the kernel's own console.

use core::arch::asm;

const COM1: u16 = 0x3F8;

/// Write a byte to an I/O port.
pub fn outb(port: u16, v: u8) {
    // SAFETY: port output has no effect on memory the compiler knows about;
    // callers pass ports the kernel owns or a port range a capability covers.
    unsafe {
        asm!("out dx, al", in("dx") port, in("al") v, options(nomem, nostack, preserves_flags))
    }
}

/// Read a byte from an I/O port.
pub fn inb(port: u16) -> u8 {
    let v: u8;
    // SAFETY: as for `outb`.
    unsafe {
        asm!("in al, dx", in("dx") port, out("al") v, options(nomem, nostack, preserves_flags))
    }
    v
}

/// 115200 baud, 8 data bits, no parity, one stop bit, FIFOs on, no interrupts.
pub fn init() {
    outb(COM1 + 1, 0x00);
    outb(COM1 + 3, 0x80);
    outb(COM1, 0x01);
    outb(COM1 + 1, 0x00);
    outb(COM1 + 3, 0x03);
    outb(COM1 + 2, 0xC7);
    outb(COM1 + 4, 0x03);
}

/// Send bytes, waiting for the transmitter. Gives up on a port that never
/// becomes ready (no serial port present) so a missing console cannot hang
/// the kernel.
pub fn write(bytes: &[u8]) {
    for &b in bytes {
        let mut spins = 0u32;
        while inb(COM1 + 5) & 0x20 == 0 && spins < 1_000_000 {
            spins += 1;
            core::hint::spin_loop();
        }
        outb(COM1, b);
    }
}
