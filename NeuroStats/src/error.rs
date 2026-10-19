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
    /// `stat` contains a NaN or ±∞; or the brain data, the tested regressor or
    /// a confound column given to the permutation engine does, or is so large
    /// that its sum of squares overflows. There is no NaN policy: filter or
    /// zero such nodes before calling.
    NonFiniteStat,
    /// A `TfceParams` field is invalid: `step <= 0`, any of `e`, `h`, `start`,
    /// `step` non-finite, or under `Weighting::Exact` `h <= -1` or `start < 0`;
    /// or in `PermOpts` a draw budget `n_perm` of 0 or above `u32::MAX`, a
    /// batch size of 0, or a `NullState` to continue that the run cannot
    /// continue (the cases are listed on `NullState`); or a null in `finalize` whose
    /// draw count differs from the draws the problem still needs; or `e` non-finite in `tfce_bands`;
    /// or a row index past the end of the source in `Store`.
    InvalidParams,
    /// Too few subjects for the model: the t needs `n − q − 1 ≥ 1` residual
    /// degrees of freedom, `q` being the number of nuisance columns kept
    /// (`n ≥ 2` for the one-sample test). Also no subject at all in
    /// `nuisance_basis` or `Store`.
    TooFewSubjects,
    /// The tested column is in the span of the nuisance columns: its norm
    /// after residualising is at most `1e-10` of its norm before. A tested
    /// regressor that is constant or a combination of the confounds; or, when
    /// the intercept is tested, a confound that is constant.
    DegenerateDesign,
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
            Self::NonFiniteStat => write!(f, "input contains NaN or infinity"),
            Self::InvalidParams => write!(
                f,
                "invalid parameters (TFCE settings: step <= 0, non-finite, or Exact with h <= -1 / start < 0; or a draw budget, batch size, row index or null state out of range)"
            ),
            Self::TooFewSubjects => write!(
                f,
                "too few subjects: the t statistic needs n - q - 1 >= 1 (q nuisance columns)"
            ),
            Self::DegenerateDesign => write!(
                f,
                "degenerate design: the tested column is in the span of the confounds and the intercept"
            ),
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
