//! Batch descriptives — thin finalize() wrappers over the accum accumulators.
//! One source of truth: each fn folds a (NaN-omitted) slice and reads out.
use crate::accum::moments::{CoMoment, Moments, Variance, comoment_pairs};
use crate::accum::simple::{Count, MinMax, Sum};
use crate::accum::{Accumulator, from_slice, quantile_sorted};
use crate::error::StatError;
use crate::nan::{NanPolicy, clean};
use alloc::vec::Vec;

fn omit(xs: &[f64]) -> Result<Vec<f64>, StatError> {
    clean(xs, NanPolicy::Omit)
}

/// Count of finite observations in `xs` (NaN dropped under the Omit default).
pub fn count(xs: &[f64]) -> usize {
    let c: Count = from_slice(xs);
    c.finalize() as usize
}
/// Neumaier-compensated sum of the finite values in `xs`.
pub fn sum(xs: &[f64]) -> f64 {
    let s: Sum = from_slice(xs);
    s.finalize()
}
/// Minimum finite value in `xs`. [`StatError::AllNaN`] if no value is finite.
pub fn min(xs: &[f64]) -> Result<f64, StatError> {
    let mm: MinMax = from_slice(xs);
    mm.finalize().map(|(lo, _)| lo).ok_or(StatError::AllNaN)
}
/// Maximum finite value in `xs`. [`StatError::AllNaN`] if no value is finite.
pub fn max(xs: &[f64]) -> Result<f64, StatError> {
    let mm: MinMax = from_slice(xs);
    mm.finalize().map(|(_, hi)| hi).ok_or(StatError::AllNaN)
}
/// Sample median (type-7 linear interpolation of order statistics).
///
/// Convention: type-7 / linear-interpolation rule — position h = (n−1)·q,
/// interpolate between order statistics at ⌊h⌋ and ⌈h⌉. Matches
/// `numpy.percentile(xs, 50, method='linear')` and R's `median()`.
///
/// `xs`: finite observations; NaN dropped under the Omit default.
/// Returns [`StatError::EmptyInput`] when `xs` is empty; [`StatError::AllNaN`]
/// when every value is NaN. Single observation: returns it as-is. Two
/// observations: returns their mean.
///
/// ```
/// assert!((commonstats::median(&[3., 1., 4., 1., 5., 9.]).unwrap() - 3.5).abs() < 1e-12);
/// ```
pub fn median(xs: &[f64]) -> Result<f64, StatError> {
    let mut v = omit(xs)?; // Err(EmptyInput) or Err(AllNaN) propagated by ?
    v.sort_by(f64::total_cmp); // total order; NaN already dropped by omit
    Ok(quantile_sorted(&v, 0.50)) // type-7: linear interpolation of order statistics
}
/// max − min over the finite values of `xs`. [`StatError::AllNaN`] if none finite.
pub fn range(xs: &[f64]) -> Result<f64, StatError> {
    let mm: MinMax = from_slice(xs);
    mm.finalize()
        .map(|(lo, hi)| hi - lo)
        .ok_or(StatError::AllNaN)
}
/// Arithmetic mean of the finite values in `xs` (NaN dropped under the Omit
/// default). [`StatError::EmptyInput`] if `xs` is empty; [`StatError::AllNaN`]
/// if every value is NaN.
///
/// ```
/// let m = commonstats::mean(&[2.0, 4.0, 6.0]).unwrap();
/// assert!((m - 4.0).abs() < 1e-12);
/// ```
pub fn mean(xs: &[f64]) -> Result<f64, StatError> {
    let v = omit(xs)?;
    Ok(v.iter().sum::<f64>() / v.len() as f64)
}

fn variance_of(xs: &[f64]) -> Result<Variance, StatError> {
    let v = omit(xs)?;
    if v.len() < 2 {
        return Err(StatError::TooFewObservations {
            needed: 2,
            got: v.len(),
        });
    }
    Ok(Variance::from_slice_two_pass(&v)) // two-pass override: whole vector in hand
}
/// Degrees-of-freedom convention for [`var`] / [`sd`]: the divisor is `n − ddof`.
///
/// There is deliberately no default. The same data gives a different number under
/// each, and silently picking one is the most common variance/SD mix-up — so the
/// caller states which estimator they mean at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ddof {
    /// Sample estimator: divisor `n − 1` (Bessel-corrected, ddof = 1). Matches `numpy.var(ddof=1)`.
    Sample,
    /// Population estimator: divisor `n` (ddof = 0). Matches `numpy.var(ddof=0)`.
    Population,
}

/// Variance of the finite values in `xs` under the chosen degrees-of-freedom
/// convention `ddof` ([`Ddof::Sample`] = ddof 1, [`Ddof::Population`] = ddof 0).
/// Matches `numpy.var(xs, ddof=…)`.
///
/// `xs`: observations; needs ≥ 2 finite values (NaN dropped under the Omit
/// default). [`StatError::TooFewObservations`] otherwise.
///
/// ```
/// use commonstats::{var, Ddof};
/// let xs = [2., 4., 4., 4., 5., 5., 7., 9.];
/// assert!((var(&xs, Ddof::Sample).unwrap()     - 32.0 / 7.0).abs() < 1e-13);
/// assert!((var(&xs, Ddof::Population).unwrap() - 4.0).abs() < 1e-13);
/// ```
pub fn var(xs: &[f64], ddof: Ddof) -> Result<f64, StatError> {
    let v = variance_of(xs)?;
    Ok(match ddof {
        Ddof::Sample => v.var_sample(),
        Ddof::Population => v.var_pop(),
    })
}
/// Standard deviation of the finite values in `xs` under the chosen
/// degrees-of-freedom convention `ddof` — the square root of [`var`]. Matches
/// `numpy.std(xs, ddof=…)`. Needs ≥ 2 finite values; [`StatError::TooFewObservations`] otherwise.
///
/// ```
/// use commonstats::{sd, Ddof};
/// let s = sd(&[2., 4., 4., 4., 5., 5., 7., 9.], Ddof::Sample).unwrap();
/// assert!((s - (32.0_f64 / 7.0).sqrt()).abs() < 1e-13);
/// ```
pub fn sd(xs: &[f64], ddof: Ddof) -> Result<f64, StatError> {
    Ok(libm::sqrt(var(xs, ddof)?))
}

/// Sample skewness (Fisher–Pearson g1, no bias correction) of the finite values
/// in `xs`. Needs ≥ 2. Matches `scipy.stats.skew(xs, bias=True)`.
///
/// ```
/// let g = commonstats::skewness(&[2., 4., 4., 4., 5., 5., 7., 9.]).unwrap();
/// // scipy.stats.skew([2,4,4,4,5,5,7,9], bias=True) = 0.65625
/// assert!((g - 0.65625).abs() < 1e-12);
/// ```
pub fn skewness(xs: &[f64]) -> Result<f64, StatError> {
    let v = omit(xs)?;
    if v.len() < 2 {
        return Err(StatError::TooFewObservations {
            needed: 2,
            got: v.len(),
        });
    }
    let m: Moments = from_slice(&v);
    Ok(m.skewness())
}
/// Excess kurtosis (Fisher, no bias correction) of the finite values in `xs`.
/// Needs ≥ 2. Matches `scipy.stats.kurtosis(xs, bias=True, fisher=True)`.
///
/// ```
/// let k = commonstats::kurtosis(&[1., 2., 3., 4., 5.]).unwrap();
/// // scipy.stats.kurtosis([1,2,3,4,5], bias=True, fisher=True) = -1.3
/// assert!((k - (-1.3)).abs() < 1e-12);
/// ```
pub fn kurtosis(xs: &[f64]) -> Result<f64, StatError> {
    let v = omit(xs)?;
    if v.len() < 2 {
        return Err(StatError::TooFewObservations {
            needed: 2,
            got: v.len(),
        });
    }
    let m: Moments = from_slice(&v);
    Ok(m.kurtosis_excess())
}

/// Pairwise; rows with a NaN in either coordinate are dropped together.
fn comoment_of(a: &[f64], b: &[f64]) -> Result<CoMoment, StatError> {
    let c = comoment_pairs(a, b)?;
    if c.count() < 2 {
        return Err(StatError::TooFewObservations {
            needed: 2,
            got: c.count() as usize,
        });
    }
    Ok(c)
}
/// Sample covariance (ddof = 1) of paired `a`, `b`. Pairs with a NaN in either
/// coordinate are dropped together. [`StatError::MismatchedLengths`] if lengths
/// differ; needs ≥ 2 complete pairs. Matches `numpy.cov(a, b, ddof=1)[0,1]`.
///
/// ```
/// let c = commonstats::cov(&[1., 2., 3.], &[2., 4., 6.]).unwrap();
/// assert!((c - 2.0).abs() < 1e-13);
/// ```
pub fn cov(a: &[f64], b: &[f64]) -> Result<f64, StatError> {
    Ok(comoment_of(a, b)?.covariance_sample())
}
/// Pearson correlation r of paired `a`, `b`. Pairs with a NaN in either coordinate
/// are dropped together. [`StatError::MismatchedLengths`] if lengths differ; needs
/// ≥ 2 complete pairs. Matches `scipy.stats.pearsonr(a, b).statistic`.
///
/// ```
/// let r = commonstats::pearson(&[1., 2., 3., 4., 5.], &[2., 1., 4., 3., 6.]).unwrap();
/// // scipy.stats.pearsonr([1..5], [2,1,4,3,6]).statistic = 0.8219949365267865
/// assert!((r - 0.8219949365267865).abs() < 1e-13);
/// ```
pub fn pearson(a: &[f64], b: &[f64]) -> Result<f64, StatError> {
    Ok(comoment_of(a, b)?.pearson())
}

/// Five-number summary plus moments for a sample. Quantiles use the type-7
/// (R/NumPy default) rule; `sd` is ddof = 1; `kurtosis` is excess (Fisher).
#[derive(Debug, Clone, PartialEq)]
pub struct Describe {
    /// Number of finite observations summarized.
    pub n: usize,
    /// Minimum.
    pub min: f64,
    /// First quartile (type-7).
    pub q1: f64,
    /// Median (type-7).
    pub median: f64,
    /// Third quartile (type-7).
    pub q3: f64,
    /// Maximum.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample standard deviation (ddof = 1).
    pub sd: f64,
    /// Sample skewness (Fisher–Pearson g1).
    pub skewness: f64,
    /// Excess kurtosis (Fisher).
    pub kurtosis: f64,
}

/// Five-number summary plus mean/sd/skewness/kurtosis of the finite values in
/// `xs` (type-7 quantiles; sd ddof = 1; excess kurtosis). Needs ≥ 2 finite
/// values; [`StatError::TooFewObservations`] otherwise.
///
/// ```
/// let d = commonstats::describe(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap();
/// assert_eq!(d.n, 9);
/// assert_eq!(d.median, 5.0);
/// ```
pub fn describe(xs: &[f64]) -> Result<Describe, StatError> {
    let mut v = omit(xs)?;
    if v.len() < 2 {
        return Err(StatError::TooFewObservations {
            needed: 2,
            got: v.len(),
        });
    }
    v.sort_by(f64::total_cmp); // total order; NaN already dropped by omit
    let m: Moments = from_slice(&v);
    Ok(Describe {
        n: v.len(),
        min: v[0],
        q1: quantile_sorted(&v, 0.25),
        median: quantile_sorted(&v, 0.50),
        q3: quantile_sorted(&v, 0.75),
        max: v[v.len() - 1],
        mean: m.mean(),
        sd: libm::sqrt(m.var_sample()),
        skewness: m.skewness(),
        kurtosis: m.kurtosis_excess(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mean_omits_nan() {
        assert!((mean(&[1.0, f64::NAN, 3.0]).unwrap() - 2.0).abs() < 1e-15);
    }
    #[test]
    fn mean_empty_errors() {
        assert_eq!(mean(&[]), Err(crate::StatError::EmptyInput));
    }
    #[test]
    fn sd_known() {
        assert!(
            (sd(&[2., 4., 4., 4., 5., 5., 7., 9.], Ddof::Sample).unwrap()
                - (32.0_f64 / 7.0).sqrt())
            .abs()
                < 1e-13
        );
    }
    #[test]
    fn var_needs_two() {
        assert_eq!(
            var(&[1.0], Ddof::Sample),
            Err(crate::StatError::TooFewObservations { needed: 2, got: 1 })
        );
    }
    #[test]
    fn pearson_mismatch_errors() {
        assert_eq!(
            pearson(&[1., 2.], &[1., 2., 3.]),
            Err(crate::StatError::MismatchedLengths { a: 2, b: 3 })
        );
    }
    #[test]
    fn pearson_known() {
        // scipy.stats.pearsonr([1..5], [2,1,4,3,6]).statistic = 0.8219949365267865
        let r = pearson(&[1., 2., 3., 4., 5.], &[2., 1., 4., 3., 6.]).unwrap();
        assert!((r - 0.8219949365267865).abs() < 1e-13, "r {r}");
    }
    #[test]
    fn covariance_known() {
        // sample covariance (ddof=1) of ([1,2,3],[2,4,6]) = 2.0
        assert!((cov(&[1., 2., 3.], &[2., 4., 6.]).unwrap() - 2.0).abs() < 1e-13);
    }
    #[test]
    fn range_known() {
        assert!((range(&[1., 3., 2.]).unwrap() - 2.0).abs() < 1e-13);
    }
    #[test]
    fn median_even_n_interpolates() {
        // [3,1,4,1,5,9] sorted → [1,1,3,4,5,9]; type-7 median = (3+4)/2 = 3.5.
        // Matches numpy.percentile([3,1,4,1,5,9], 50, method='linear') = 3.5.
        assert!((median(&[3., 1., 4., 1., 5., 9.]).unwrap() - 3.5).abs() < 1e-12);
    }
    #[test]
    fn median_odd_n_is_middle() {
        // [1,2,3] sorted → middle element = 2.0.
        assert!((median(&[1., 2., 3.]).unwrap() - 2.0).abs() < 1e-12);
    }
    #[test]
    fn median_omits_nan() {
        // [1, NaN, 3] → effective [1,3] → median = 2.0.
        assert!((median(&[1., f64::NAN, 3.]).unwrap() - 2.0).abs() < 1e-12);
    }
    #[test]
    fn median_empty_errors() {
        assert_eq!(median(&[]), Err(crate::StatError::EmptyInput));
    }
    #[test]
    fn median_all_nan_errors() {
        assert_eq!(median(&[f64::NAN]), Err(crate::StatError::AllNaN));
    }
}

#[cfg(test)]
mod describe_tests {
    use super::*;
    #[test]
    fn describe_five_number() {
        // 1..=9, type-7 quartiles: q1=3, median=5, q3=7
        let d = describe(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap();
        assert_eq!(d.n, 9);
        assert_eq!(d.min, 1.0);
        assert_eq!(d.max, 9.0);
        assert!((d.q1 - 3.0).abs() < 1e-12);
        assert!((d.median - 5.0).abs() < 1e-12);
        assert!((d.q3 - 7.0).abs() < 1e-12);
        assert!((d.mean - 5.0).abs() < 1e-12);
        assert!((d.sd - 2.7386127875258306).abs() < 1e-12, "sd {}", d.sd); // √(60/8), ddof=1
        assert!(d.skewness.abs() < 1e-12, "skew {}", d.skewness); // symmetric → 0
        assert!((d.kurtosis - (-1.23)).abs() < 1e-9, "kurt {}", d.kurtosis); // scipy excess kurtosis
    }
    #[test]
    fn describe_empty_errors() {
        assert_eq!(describe(&[]), Err(crate::StatError::EmptyInput));
    }
}
