//! The first serial port (16550 at 0x3F8): the kernel's own console.

use core::arch::asm;

const COM1: u16 = 0x3F8;

/// Write a byte to an I/O port.
pub fn outb(port: u16, v: u8) {
    // SAFETY: port output has no effect on memory the compiler knows about;
    // callers pass ports the kernel owns or a port range a capability covers.
    unsafe { asm!("out dx, al", in("dx") port, in("al") v, options(nomem, nostack, preserves_flags)) }
}

/// Read a byte from an I/O port.
pub fn inb(port: u16) -> u8 {
    let v: u8;
    // SAFETY: as for `outb`.
    unsafe { asm!("in al, dx", in("dx") port, out("al") v, options(nomem, nostack, preserves_flags)) }
    v
}

/// Read a 16-bit word from an I/O port.
pub fn inw(port: u16) -> u16 {
    let v: u16;
    // SAFETY: as for `outb`.
    unsafe { asm!("in ax, dx", in("dx") port, out("ax") v, options(nomem, nostack, preserves_flags)) }
    v
}

/// Write a 16-bit word to an I/O port.
pub fn outw(port: u16, v: u16) {
    // SAFETY: as for `outb`.
    unsafe { asm!("out dx, ax", in("dx") port, in("ax") v, options(nomem, nostack, preserves_flags)) }
}

/// Whether the machine has COM1 (its scratch register keeps what is
/// written; on a machine without one the port reads as all ones).
static PRESENT: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);

/// Whether COM1 is there to write to.
pub fn present() -> bool {
    PRESENT.load(core::sync::atomic::Ordering::Relaxed)
}

/// 115200 baud, 8 data bits, no parity, one stop bit, FIFOs on, no interrupts.
pub fn init() {
    outb(COM1 + 7, 0x5A);
    let kept = inb(COM1 + 7) == 0x5A;
    outb(COM1 + 7, 0xA5);
    PRESENT.store(kept && inb(COM1 + 7) == 0xA5, core::sync::atomic::Ordering::Relaxed);
    outb(COM1 + 1, 0x00);
    outb(COM1 + 3, 0x80);
    outb(COM1, 0x01);
    outb(COM1 + 1, 0x00);
    outb(COM1 + 3, 0x03);
    outb(COM1 + 2, 0xC7);
    outb(COM1 + 4, 0x03);
}

/// One writer at a time for whole calls of [`write_whole`].
static WRITER: kernel_core::lock::Lock = kernel_core::lock::Lock::new();

/// Send `bytes` with no other processor's bytes among them: a kernel line,
/// or one piece a program writes through its console capability. Not for
/// the panic path, which must not wait for a lock its own processor may hold.
pub fn write_whole(bytes: &[u8]) {
    WRITER.lock();
    write(bytes);
    WRITER.unlock();
}

/// Send bytes, waiting for the transmitter. Gives up on a port that never
/// becomes ready (no serial port present) so a missing console cannot hang
/// the kernel.
pub fn write(bytes: &[u8]) {
    if !present() {
        return;
    }
    for &b in bytes {
        let mut spins = 0u32;
        while inb(COM1 + 5) & 0x20 == 0 && spins < 1_000_000 {
            spins += 1;
            core::hint::spin_loop();
        }
        outb(COM1, b);
    }
}
