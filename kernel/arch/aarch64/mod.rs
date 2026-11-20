//! aarch64 architecture glue (stub).

use crate::hal::Arch;

/// The architecture this kernel is built for.
pub struct Current;

impl Arch for Current {
    fn halt() -> ! {
        loop {
            core::hint::spin_loop();
        }
    }
}
