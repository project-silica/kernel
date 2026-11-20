//! The kernel's console output.

use crate::hal::Hal;
use core::fmt;

/// Adapter from `core::fmt` to the architecture's console.
pub struct Console<'a, H: Hal>(pub &'a H);

impl<H: Hal> fmt::Write for Console<'_, H> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0.console(s.as_bytes());
        Ok(())
    }
}

/// Write one line to the kernel console.
#[macro_export]
macro_rules! klog {
    ($k:expr, $($arg:tt)*) => {{
        use core::fmt::Write as _;
        let mut c = $crate::log::Console(&$k.hal);
        let _ = write!(c, $($arg)*);
        let _ = c.write_str("\r\n");
    }};
}
