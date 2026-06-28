//! Blom (1958) normal scores using average ranks.

use crate::error::StatError;
use crate::nan::{clean, NanPolicy};
use crate::special::erfc_inv;
use crate::transform::rank::{rank, Ties};
use core::f64::consts::SQRT_2;

/// Φ⁻¹(p) — the standard normal quantile function.
///
/// Derivation: `erfc_inv(p) = −norm_ppf(p/2)/√2` (from `special/inverse.rs`)
/// ⟹ `norm_ppf(p) = −√2·erfc_inv(2p)`. We route through the **public**
/// `special::erfc_inv` because `norm_ppf` is private in that module.
#[inline]
pub(crate) fn norm_quantile(p: f64) -> f64 {
    -SQRT_2 * erfc_inv(2.0 * p)
}

/// Blom (1958) normal scores: Φ⁻¹((rᵢ − 3/8) / (n + 1/4)).
///
/// **Convention:** `rᵢ` is the average rank of the i-th finite observation;
/// `a = 3/8` (Blom only — van der Waerden `a=0` and Rankit `a=1/2` are not
/// provided). Denominator `n + 1/4 = n − 2·(3/8) + 1`. Φ⁻¹ via public
/// `special::erfc_inv`: `Φ⁻¹(p) = −√2·erfc_inv(2p)`.
/// NaN values are omitted (Omit policy). Output length = finite count.
///
/// **Matches** `scipy.stats.norm.ppf((rankdata(v,'average') − 3/8) / (n + 1/4))`.
///
/// # Parameters
/// - `v`: input slice; may contain NaN.
///
/// # Returns
/// Normal scores in the order of the finite values of `v`.
///
/// # Errors
/// - [`StatError::EmptyInput`] if `v` is empty.
/// - [`StatError::AllNaN`] if all values are NaN.
///
/// # Examples
/// ```
/// use commonstats::transform::normal_scores;
/// let zs = normal_scores(&[1.0, 2.0, 3.0]).unwrap();
/// // Symmetric: first score is negative, last is positive
/// assert!(zs[0] < 0.0 && zs[2] > 0.0);
/// ```
pub fn normal_scores(v: &[f64]) -> Result<Vec<f64>, StatError> {
    // clean NaN-filters and errors on empty/all-NaN
    let data = clean(v, NanPolicy::Omit)?;
    let n = data.len() as f64;
    // Average ranks of the cleaned data
    let ranks = rank(&data, Ties::Average)?;
    Ok(ranks
        .iter()
        .map(|&r| {
            let p = (r - 0.375) / (n + 0.25);
            norm_quantile(p)
        })
        .collect())
}
