//! Box–Cox and Yeo–Johnson power transforms.
//!
//! Both accept a caller-supplied λ (no MLE optimizer — a `*_mle` variant is
//! forward-compatible if added later). NaN inputs are omitted via `nan::clean`.

use crate::error::StatError;
use crate::nan::{clean, NanPolicy};

/// Box–Cox (1964) power transform for positive data.
///
/// **Convention:** `(x^λ − 1)/λ` for λ≠0; `ln(x)` for λ=0. Requires `x > 0`
/// for every finite input element; the first `x ≤ 0` encountered aborts with
/// `DomainError` — no partial output is returned.
/// NaN values are omitted (Omit policy).
///
/// **Matches** `scipy.stats.boxcox(v, lmbda=lambda)`.
///
/// # Parameters
/// - `v`: input slice; finite values must all be strictly positive.
/// - `lambda`: transform parameter (any finite `f64`).
///
/// # Returns
/// Transformed values in the order of the finite inputs.
///
/// # Errors
/// - [`StatError::EmptyInput`] if `v` is empty.
/// - [`StatError::AllNaN`] if all values are NaN.
/// - [`StatError::DomainError`] if any finite value is `≤ 0`.
///
/// # Examples
/// ```
/// use commonstats::transform::box_cox;
/// // lambda=1.0: (x^1 - 1)/1 = x - 1
/// let r = box_cox(&[1.0, 2.0, 3.0], 1.0).unwrap();
/// assert!((r[0] - 0.0).abs() < 1e-14);
/// assert!((r[1] - 1.0).abs() < 1e-14);
/// ```
pub fn box_cox(v: &[f64], lambda: f64) -> Result<Vec<f64>, StatError> {
    let data = clean(v, NanPolicy::Omit)?;
    // Validate domain before producing any output — fail-fast on first violation.
    if data.iter().any(|&x| x <= 0.0) {
        return Err(StatError::DomainError(
            "box_cox requires x > 0 for all inputs",
        ));
    }
    Ok(data
        .iter()
        .map(|&x| {
            if lambda == 0.0 {
                // λ=0 branch: ln(x)
                libm::log(x)
            } else {
                // general branch: (x^λ - 1) / λ
                (libm::pow(x, lambda) - 1.0) / lambda
            }
        })
        .collect())
}

/// Yeo–Johnson (2000) power transform for all real inputs.
///
/// **Convention:** four branches by sign of `x` and value of `λ`:
/// - `x ≥ 0, λ ≠ 0`: `((x+1)^λ − 1) / λ`
/// - `x ≥ 0, λ = 0`: `ln(x+1)`
/// - `x < 0, λ ≠ 2`: `−((−x+1)^(2−λ) − 1) / (2−λ)`
/// - `x < 0, λ = 2`: `−ln(−x+1)`
///
/// NaN values are omitted (Omit policy). Accepts all real `x`.
///
/// **Matches** `scipy.stats.yeojohnson(v, lmbda=lambda)`.
///
/// # Parameters
/// - `v`: input slice; may contain any finite `f64` and NaN.
/// - `lambda`: transform parameter.
///
/// # Returns
/// Transformed values in the order of the finite inputs.
///
/// # Errors
/// - [`StatError::EmptyInput`] if `v` is empty.
/// - [`StatError::AllNaN`] if all values are NaN.
///
/// # Examples
/// ```
/// use commonstats::transform::yeo_johnson;
/// // lambda=2.0, x<0 branch: -ln(-x+1)
/// let r = yeo_johnson(&[-1.0], 2.0).unwrap();
/// assert!((r[0] - (-f64::ln(2.0))).abs() < 1e-14);
/// ```
pub fn yeo_johnson(v: &[f64], lambda: f64) -> Result<Vec<f64>, StatError> {
    let data = clean(v, NanPolicy::Omit)?;
    Ok(data
        .iter()
        .map(|&x| yj_scalar(x, lambda))
        .collect())
}

/// Scalar Yeo–Johnson transform — four branches.
#[inline]
fn yj_scalar(x: f64, lambda: f64) -> f64 {
    if x >= 0.0 {
        if lambda == 0.0 {
            libm::log(x + 1.0)
        } else {
            (libm::pow(x + 1.0, lambda) - 1.0) / lambda
        }
    } else {
        let two_minus_lam = 2.0 - lambda;
        if lambda == 2.0 {
            -libm::log(-x + 1.0)
        } else {
            -(libm::pow(-x + 1.0, two_minus_lam) - 1.0) / two_minus_lam
        }
    }
}
