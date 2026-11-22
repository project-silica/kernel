//! The kernel's console output.

use crate::hal::Hal;
use crate::lock::Lock;
use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

/// The writer whose line the console is in the middle of (0: none; a line
/// ends with a newline). Kept with [`LINE`] held.
static OPEN: AtomicU64 = AtomicU64::new(0);
static LINE: Lock = Lock::new();

/// The kernel's own lines, as a writer.
pub const KERNEL: u64 = u64::MAX;

/// Write `bytes` for `writer` (the kernel, or a thread through its console
/// capability), whole. A writer that finds the line left unfinished by
/// another starts a fresh one, so a program's prompt and a kernel line never
/// share a line, and every line a reader anchors on starts where it should.
pub fn write<H: Hal>(hal: &H, writer: u64, bytes: &[u8]) {
    write_parts(hal, writer, &[bytes]);
}

/// [`write()`] of `parts`, one after another, with nobody else's text among them.
pub fn write_parts<H: Hal>(hal: &H, writer: u64, parts: &[&[u8]]) {
    let Some(last) = parts.iter().rev().find_map(|p| p.last()) else { return };
    LINE.lock();
    let open = OPEN.load(Ordering::Relaxed);
    if open != 0 && open != writer {
        hal.console(b"\r\n");
    }
    for p in parts.iter().filter(|p| !p.is_empty()) {
        hal.console(p);
    }
    OPEN.store(if *last == b'\n' { 0 } else { writer }, Ordering::Relaxed);
    LINE.unlock();
}

/// One line for the architecture's console, gathered and written with one
/// call, so that lines from different processors (and the pieces programs
/// write through their console capability) do not tear one another. A line
/// longer than the buffer is written in several calls.
pub struct Console<'a, H: Hal> {
    hal: &'a H,
    buf: [u8; 200],
    len: usize,
}

impl<'a, H: Hal> Console<'a, H> {
    /// An empty line for `hal`'s console.
    pub fn new(hal: &'a H) -> Console<'a, H> {
        Console { hal, buf: [0; 200], len: 0 }
    }

    /// Write out what has been gathered.
    pub fn flush(&mut self) {
        if self.len != 0 {
            write(self.hal, KERNEL, &self.buf[..self.len]);
            self.len = 0;
        }
    }
}

impl<H: Hal> fmt::Write for Console<'_, H> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for chunk in s.as_bytes().chunks(self.buf.len()) {
            if self.len + chunk.len() > self.buf.len() {
                self.flush();
            }
            self.buf[self.len..self.len + chunk.len()].copy_from_slice(chunk);
            self.len += chunk.len();
        }
        Ok(())
    }
}

/// Write one line to the kernel console.
#[macro_export]
macro_rules! klog {
    ($k:expr, $($arg:tt)*) => {{
        use core::fmt::Write as _;
        let mut c = $crate::log::Console::new(&$k.hal);
        let _ = write!(c, $($arg)*);
        let _ = c.write_str("\r\n");
        c.flush();
    }};
}
