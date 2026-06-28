//! The crate's single error type. One enum, returned by every fallible public
//! API, hand-rolled `Display` to stay `libm`-only and WASM-clean.
use core::fmt;

/// Crate-wide error. Every public API returns `Result<_, StatError>`; no panics
/// cross the API boundary. `#[non_exhaustive]` so adding a variant in a later
/// phase is not a breaking change.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum StatError {
    /// Input slice was empty.
    EmptyInput,
    /// Fewer usable observations than required.
    TooFewObservations {
        /// Minimum number of finite observations the computation needs.
        needed: usize,
        /// How many finite observations were actually available.
        got: usize,
    },
    /// Paired inputs had differing lengths.
    MismatchedLengths {
        /// Length of the first input.
        a: usize,
        /// Length of the second input.
        b: usize,
    },
    /// No usable (non-NaN) observations remained after applying the NaN policy.
    AllNaN,
    /// Input violated a structural precondition; the message names it.
    DomainError(&'static str),
    /// A probability or confidence-level argument fell outside its valid range.
    ProbabilityOutOfRange(f64),
}

impl fmt::Display for StatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatError::EmptyInput => write!(f, "input slice is empty"),
            StatError::TooFewObservations { needed, got } =>
                write!(f, "need at least {needed} observations, got {got}"),
            StatError::MismatchedLengths { a, b } =>
                write!(f, "mismatched lengths: {a} vs {b}"),
            StatError::AllNaN => write!(f, "no usable (non-NaN) observations"),
            StatError::DomainError(m) => write!(f, "domain error: {m}"),
            StatError::ProbabilityOutOfRange(p) =>
                write!(f, "probability {p} outside [0, 1]"),
        }
    }
}

impl std::error::Error for StatError {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn display_is_human_readable() {
        assert_eq!(StatError::EmptyInput.to_string(), "input slice is empty");
        assert_eq!(
            StatError::TooFewObservations { needed: 2, got: 1 }.to_string(),
            "need at least 2 observations, got 1",
        );
    }
    #[test]
    fn errors_compare_by_value() {
        assert_eq!(StatError::AllNaN, StatError::AllNaN);
        assert_ne!(StatError::EmptyInput, StatError::AllNaN);
    }
}
