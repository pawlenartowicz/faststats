//! Probability-integral transform (PIT) family.
//!
//! Requires the `dist` feature — these functions are thin shims over
//! [`crate::dist::ContinuousCdf`] and carry no math of their own.
//! Built last (Task 5): depends on Area 1's `ContinuousCdf` trait in `crate::dist`.

use crate::dist::ContinuousCdf;
use crate::error::StatError;

/// Probability-integral transform: maps `x` to its CDF probability under `dist`.
///
/// **Convention:** `pit(x, d) = d.cdf(x)`. Equivalent to converting a raw
/// observation to a uniform(0,1) quantile under the assumed model.
/// No error — `cdf` is total for continuous distributions.
///
/// **Matches** `dist.cdf(x)` from `crate::dist::ContinuousCdf`.
///
/// # Examples
/// ```
/// use commonstats::dist::continuous::Normal;
/// use commonstats::transform::pit;
/// let n = Normal::new(0.0, 1.0).unwrap();
/// assert!((pit(0.0, &n) - 0.5).abs() < 1e-12);
/// ```
pub fn pit<D: ContinuousCdf>(x: f64, dist: &D) -> f64 {
    dist.cdf(x)
}

/// Inverse PIT: maps a probability `u ∈ [0, 1]` back to a quantile under `dist`.
///
/// **Convention:** `inv_pit(u, d) = d.quantile(u)`.
///
/// # Errors
/// - [`StatError::ProbabilityOutOfRange`] if `u ∉ [0, 1]`.
///
/// # Examples
/// ```
/// use commonstats::dist::continuous::Normal;
/// use commonstats::transform::inv_pit;
/// let n = Normal::new(0.0, 1.0).unwrap();
/// assert!((inv_pit(0.5, &n).unwrap() - 0.0).abs() < 1e-10);
/// ```
pub fn inv_pit<D: ContinuousCdf>(u: f64, dist: &D) -> Result<f64, StatError> {
    dist.quantile(u)
}

/// Quantile mapping: transport `x` from one distribution's scale to another's.
///
/// **Convention:** `quantile_map(x, from, to) = to.quantile(from.cdf(x))`.
/// Converts `x` to a probability under `from`, then to the corresponding
/// quantile under `to`. Useful for data-space rescaling when the source and
/// target distributions are known.
///
/// # Errors
/// - [`StatError::ProbabilityOutOfRange`] if `from.cdf(x)` falls outside [0, 1]
///   (only possible if `cdf` is miscalibrated; should not occur for correct impls).
///
/// # Examples
/// ```
/// use commonstats::dist::continuous::Normal;
/// use commonstats::transform::quantile_map;
/// let from = Normal::new(0.0, 1.0).unwrap();
/// let to   = Normal::new(1.0, 1.0).unwrap();
/// // N(0,1) median (x=0, p=0.5) maps to N(1,1) median (x=1)
/// assert!((quantile_map(0.0, &from, &to).unwrap() - 1.0).abs() < 1e-8);
/// ```
pub fn quantile_map<D1: ContinuousCdf, D2: ContinuousCdf>(
    x: f64,
    from: &D1,
    to: &D2,
) -> Result<f64, StatError> {
    to.quantile(from.cdf(x))
}
