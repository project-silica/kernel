//! The one trait every architecture implements (`spec/ARCHITECTURE.md` section 6, HAL).

/// Architecture operations the arch-free kernel needs.
pub trait Arch {
    /// Stop this CPU forever.
    fn halt() -> !;
}
