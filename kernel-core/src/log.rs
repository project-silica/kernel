//! The kernel's console output.

use crate::hal::Hal;
use core::fmt;

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
            self.hal.console(&self.buf[..self.len]);
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
