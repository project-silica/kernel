//! What the aarch64 kernel decides from a processor's syndrome, as pure
//! functions the host can test.

/// What an SError taken from a program means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SError {
    /// Corrected by the hardware: said, and the program goes on.
    Corrected,
    /// An uncorrected error the processor says was contained (the RAS
    /// extension's unrecoverable, restartable or recoverable kinds): the
    /// program consumed it, so it is the program's fault, as an x86 machine
    /// check in user mode is.
    Contained,
    /// Nothing says it was contained (a processor without the RAS
    /// extension reports every SError so, or the error was uncontainable):
    /// the machine's state may be lost, and it stops.
    Uncontainable,
}

/// The verdict on an SError taken from EL0 with syndrome `esr` (ESR_EL1,
/// exception class 0x2F). Only an architected syndrome (IDS 0) of an
/// asynchronous SError (DFSC 0x11) carries the error's kind (AET).
pub fn serror(esr: u64) -> SError {
    let (ids, aet, dfsc) = (esr >> 24 & 1, esr >> 10 & 7, esr & 0x3F);
    if ids != 0 || dfsc != 0x11 {
        return SError::Uncontainable;
    }
    match aet {
        0b110 => SError::Corrected,
        0b001..=0b011 => SError::Contained,
        _ => SError::Uncontainable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLASS: u64 = 0x2F << 26;

    #[test]
    fn an_serror_is_the_programs_only_when_the_processor_says_it_was_contained() {
        let with = |aet: u64| CLASS | aet << 10 | 0x11;
        assert_eq!(serror(with(0b110)), SError::Corrected);
        for aet in [0b001, 0b010, 0b011] {
            assert_eq!(serror(with(aet)), SError::Contained, "AET {aet:#b}");
        }
        assert_eq!(serror(with(0b000)), SError::Uncontainable, "uncontainable");
        // No RAS extension: an implementation-defined syndrome, or one that
        // is not an asynchronous SError's, says nothing about containment.
        assert_eq!(serror(CLASS | 1 << 24 | 0b011 << 10 | 0x11), SError::Uncontainable);
        assert_eq!(serror(CLASS | 0b011 << 10), SError::Uncontainable);
        assert_eq!(serror(CLASS), SError::Uncontainable);
    }
}
