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

/// 115200 baud, 8 data bits, no parity, one stop bit, 16-byte FIFOs on
/// (every 16550A; `feed` writes 16 bytes when the transmitter is empty),
/// no interrupts.
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
    // A 16550A says its FIFOs are on (IIR bits 7 and 6); an older UART has
    // a one-byte holding register, and is fed a byte at a time.
    DEPTH.store(if inb(COM1 + 2) & 0xC0 == 0xC0 { 16 } else { 1 }, core::sync::atomic::Ordering::Relaxed);
}

/// Bytes the transmitter takes when it says it is empty.
static DEPTH: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(1);

/// The console's lock: one writer at a time, and one feeder of the port.
static WRITER: kernel_core::lock::Lock = kernel_core::lock::Lock::new();

/// What the console has been given and not yet sent (16 KiB: 1.4 s of
/// output at 115,200 baud).
static RING: kernel_core::log::TxRing<16384> = kernel_core::log::TxRing::new();

/// When the bytes still waiting must be looked at again (monotonic
/// nanoseconds; `u64::MAX`: nothing waits). Every processor arms its timer
/// no later than this (`X86::set_timer`), so the console drains on a busy
/// machine as on an idle one, and costs nothing when it is empty.
static DUE: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(u64::MAX);

/// How long the transmitter takes to send what it holds when it says it is
/// empty: 16 bytes (its FIFO) of 10 bits at 115,200 baud is 1.39 ms.
const DRAIN_NS: u64 = 1_400_000;

/// Queue `bytes` with no other processor's bytes among them (a kernel line,
/// or one piece a program writes through its console capability), and send
/// what the port takes now. Never waits for the port. Not for the panic
/// path, which must not wait for a lock its own processor may hold.
pub fn write_whole(bytes: &[u8], now: u64) {
    if !present() {
        return;
    }
    WRITER.lock();
    RING.put(bytes);
    feed(now);
    WRITER.unlock();
}

/// Send what waits, as far as the port takes it now, if it is due at `now`
/// (a timer interrupt, the idle path).
pub fn drain(now: u64) {
    if DUE.load(core::sync::atomic::Ordering::Relaxed) > now {
        return;
    }
    WRITER.lock();
    feed(now);
    WRITER.unlock();
}

/// When the console must be looked at next (`u64::MAX`: nothing waits).
pub fn due() -> u64 {
    DUE.load(core::sync::atomic::Ordering::Relaxed)
}

/// Fill the transmitter while it is empty (it says so for its whole FIFO:
/// 16 bytes), and say when to look again. Holding `WRITER`.
fn feed(now: u64) {
    while RING.pending() && inb(COM1 + 5) & 0x20 != 0 {
        for _ in 0..DEPTH.load(core::sync::atomic::Ordering::Relaxed) {
            match RING.take() {
                Some(b) => outb(COM1, b),
                None => break,
            }
        }
    }
    let due = if RING.pending() { now.saturating_add(DRAIN_NS) } else { u64::MAX };
    DUE.store(due, core::sync::atomic::Ordering::Relaxed);
}

/// The panic path: send what waits and then `bytes`, waiting for the port
/// (a bounded number of times per byte, so a missing or stuck port cannot
/// hang it). Takes no lock: the processor may hold it already.
pub fn write(bytes: &[u8]) {
    if !present() {
        return;
    }
    let send = |b: u8| {
        let mut spins = 0u32;
        while inb(COM1 + 5) & 0x20 == 0 && spins < 1_000_000 {
            spins += 1;
            core::hint::spin_loop();
        }
        outb(COM1, b);
    };
    while let Some(b) = RING.take() {
        send(b);
    }
    for &b in bytes {
        send(b);
    }
}
