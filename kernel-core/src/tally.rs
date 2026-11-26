//! A count only a test image's kernel keeps (the `harness` feature), read
//! by the tests that judge where interrupts arrive. A kernel built for a
//! machine neither stores nor counts it: the path it sits on pays nothing.

/// See the module.
pub(crate) struct Tally(#[cfg(feature = "harness")] core::sync::atomic::AtomicU64);

impl Tally {
    pub(crate) const fn new() -> Tally {
        Tally(
            #[cfg(feature = "harness")]
            core::sync::atomic::AtomicU64::new(0),
        )
    }

    #[inline(always)]
    pub(crate) fn add(&self) {
        #[cfg(feature = "harness")]
        self.0.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn clear(&self) {
        #[cfg(feature = "harness")]
        self.0.store(0, core::sync::atomic::Ordering::Relaxed);
    }

    /// The count; 0 in a kernel that keeps none.
    pub(crate) fn get(&self) -> u64 {
        #[cfg(feature = "harness")]
        return self.0.load(core::sync::atomic::Ordering::Relaxed);
        #[cfg(not(feature = "harness"))]
        0
    }
}
