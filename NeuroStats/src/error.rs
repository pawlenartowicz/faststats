//! Error type shared by every fallible entry point.

use core::fmt;

/// Precondition failures. Every variant names the exact trigger; no variant is
/// raised for a numerically degenerate but well-formed input (a flat map, an
/// empty threshold grid) — those return a well-defined result instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuroError {
    /// `mask.len()` (or `stat.len()`) does not match the expected node count.
    MismatchedLengths {
        /// Length implied by `dims` (for masks) or by the `Domain` (for maps).
        expected: usize,
        /// Length actually passed.
        got: usize,
    },
    /// More than `u32::MAX` in-mask nodes — the CSR indices are `u32`.
    TooManyNodes,
    /// `stat` contains a NaN or ±∞. The operator has no NaN policy: filter or
    /// zero such nodes before calling.
    NonFiniteStat,
    /// A `TfceParams` field is invalid: `step <= 0`, any of `e`, `h`, `start`,
    /// `step` non-finite, or under `Weighting::Exact` `h <= -1` or `start < 0`;
    /// or `b_requested == 0` in `tfce_one_sample`; or an empty null (no draws)
    /// in `finalize`; or `e` non-finite in `tfce_bands`.
    InvalidParams,
    /// Fewer than 2 subjects: the one-sample t needs `n ≥ 2` for `ddof = 1`.
    TooFewSubjects,
    /// CSR adjacency rejected by `Domain::from_csr`: empty `offsets`,
    /// non-monotone `offsets`, `offsets[n] != neighbours.len()`, a neighbour
    /// index `>= n`, or a self-loop.
    InvalidAdjacency,
    /// `tfce_bands` band list rejected: `thresholds` and `weights` differ in
    /// length, any value non-finite, or `thresholds` not strictly increasing.
    InvalidBands,
}

impl fmt::Display for NeuroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MismatchedLengths { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            Self::TooManyNodes => write!(f, "more than u32::MAX in-mask nodes"),
            Self::NonFiniteStat => write!(f, "statistic map contains NaN or infinity"),
            Self::InvalidParams => write!(
                f,
                "invalid TFCE parameters (step <= 0, non-finite, or Exact with h <= -1 / start < 0)"
            ),
            Self::TooFewSubjects => write!(f, "one-sample t needs at least 2 subjects"),
            Self::InvalidAdjacency => write!(
                f,
                "invalid CSR adjacency (offsets shape, index out of range, or self-loop)"
            ),
            Self::InvalidBands => write!(
                f,
                "invalid TFCE bands (length mismatch, non-finite, or thresholds not strictly increasing)"
            ),
        }
    }
}

impl core::error::Error for NeuroError {}
