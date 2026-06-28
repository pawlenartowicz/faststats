//! Kernel density estimation and auto-binned histograms.
//!
//! Two independent entry points:
//! - [`kde`] — direct-sum KDE with Gaussian or Epanechnikov kernel.
//! - [`histogram_auto`] — bin-rule front-end over [`crate::accum::Histogram`].
//!
//! Both are base-feature (no gate) and WASM-clean (libm-only math).

use crate::error::StatError;
use crate::nan::{self, NanPolicy};
use crate::accum::{Histogram, HistResult, Accumulator};
use crate::accum;
use crate::descriptive;
use core::f64::consts::PI;

// ---------------------------------------------------------------------------
// KDE
// ---------------------------------------------------------------------------

/// Kernel function for KDE.
///
/// Convention: each kernel integrates to 1 over ℝ.
/// Matches scipy.stats.gaussian_kde (Gaussian) and scipy.stats.gaussian_kde
/// with `kernel='epanechnikov'` (Epanechnikov).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Gaussian kernel: K(u) = exp(−u²/2) / √(2π).
    Gaussian,
    /// Epanechnikov kernel: K(u) = 0.75·(1−u²) for |u| ≤ 1, else 0.
    Epanechnikov,
}

/// Bandwidth selection method for KDE.
///
/// Convention: bandwidth formulas match scipy.stats.gaussian_kde.
/// `sd` is always sample standard deviation (ddof=1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bandwidth {
    /// Silverman's rule: h = sd · (3n/4)^(−1/5).
    ///
    /// Matches scipy `'silverman'` covariance factor. Note: this is the
    /// sd-only variant; the robust min(sd, IQR/1.349) variant is not exposed.
    Silverman,
    /// Scott's rule: h = sd · n^(−1/5).
    ///
    /// Matches scipy `'scott'` covariance factor.
    Scott,
    /// Fixed bandwidth. Must be > 0 and finite.
    Fixed(f64),
}

/// A fitted kernel density estimator.
///
/// Created by [`kde`]. Stores a sorted copy of the finite input values
/// and the resolved bandwidth. Query via [`Kde::density`] or [`Kde::evaluate`].
///
/// Matches scipy.stats.gaussian_kde evaluated on the same bandwidth.
#[derive(Debug, Clone)]
pub struct Kde {
    /// Sorted finite observations (NaN already stripped by `kde()`).
    data: Vec<f64>,
    /// Resolved bandwidth h > 0.
    h: f64,
    /// The kernel used.
    kernel: Kernel,
}

/// Fit a kernel density estimator to `xs`.
///
/// Convention: density estimate is (1/(n·h)) · Σᵢ K((x − xᵢ)/h) where
/// K is the chosen kernel and h is the resolved bandwidth. NaN policy: Omit.
///
/// Matches scipy.stats.gaussian_kde with `bw_method='silverman'` / `'scott'`
/// / a fixed scalar (scipy's scalar bw_method sets the covariance factor,
/// which equals h because scipy uses unit-variance kernels).
///
/// # Parameters
/// - `xs`: input data; NaN values are silently omitted.
/// - `kernel`: kernel function (Gaussian or Epanechnikov).
/// - `bandwidth`: bandwidth selection rule or fixed value.
///
/// # Errors
/// - `EmptyInput` — `xs` is empty.
/// - `TooFewObservations{needed:2, got:n}` — fewer than 2 finite values.
/// - `DomainError` — zero-variance data with Silverman/Scott (h would be 0).
/// - `DomainError` — `Fixed(h)` where h ≤ 0 or h is NaN.
///
/// # Example
/// ```
/// use commonstats::density::{kde, Kernel, Bandwidth};
/// let xs = &[1.0, 2.0, 3.0, 4.0, 5.0];
/// let k = kde(xs, Kernel::Gaussian, Bandwidth::Silverman).unwrap();
/// assert!(k.bandwidth() > 0.0);
/// assert_eq!(k.n(), 5);
/// let vals = k.evaluate(&[2.5, 3.0]);
/// assert_eq!(vals.len(), 2);
/// ```
pub fn kde(xs: &[f64], kernel: Kernel, bandwidth: Bandwidth) -> Result<Kde, StatError> {
    let finite = nan::clean(xs, NanPolicy::Omit)?; // EmptyInput / AllNaN
    let n = finite.len();
    if n < 2 {
        return Err(StatError::TooFewObservations { needed: 2, got: n });
    }

    // Resolve bandwidth h.
    let h = match bandwidth {
        Bandwidth::Fixed(h) => {
            if !h.is_finite() || h <= 0.0 {
                return Err(StatError::DomainError(
                    "Fixed bandwidth must be finite and > 0",
                ));
            }
            h
        }
        Bandwidth::Silverman => {
            // h = sd * (3n/4)^(-1/5); sd ddof=1 via descriptive::sd
            let sd = descriptive::sd(&finite, descriptive::Ddof::Sample)
                .map_err(|_| StatError::DomainError("zero-variance data: bandwidth would be 0"))?;
            if sd == 0.0 {
                return Err(StatError::DomainError(
                    "zero-variance data: Silverman bandwidth would be 0",
                ));
            }
            sd * libm::pow(3.0 * n as f64 / 4.0, -0.2)
        }
        Bandwidth::Scott => {
            let sd = descriptive::sd(&finite, descriptive::Ddof::Sample)
                .map_err(|_| StatError::DomainError("zero-variance data: bandwidth would be 0"))?;
            if sd == 0.0 {
                return Err(StatError::DomainError(
                    "zero-variance data: Scott bandwidth would be 0",
                ));
            }
            sd * libm::pow(n as f64, -0.2)
        }
    };

    let mut data = finite;
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    Ok(Kde { data, h, kernel })
}

impl Kde {
    /// Evaluate the KDE density at a single point `x`.
    ///
    /// Formula: (1/(n·h)) · Σᵢ K((x − xᵢ)/h).
    /// NaN `x` → returns 0.0 (not meaningful; use `evaluate` for grids).
    ///
    /// Convention: identical to scipy.stats.gaussian_kde.evaluate([x])[0]
    /// at the same bandwidth.
    pub fn density(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        let n = self.data.len() as f64;
        let h = self.h;
        let sum: f64 = self.data.iter().map(|&xi| kernel_eval(self.kernel, (x - xi) / h)).sum();
        sum / (n * h)
    }

    /// Evaluate the KDE density at every point in `grid`. O(n·m) direct sum.
    ///
    /// Returns a `Vec<f64>` of length `grid.len()`. Non-finite grid points
    /// produce 0.0. Matches scipy.stats.gaussian_kde.evaluate(grid).
    pub fn evaluate(&self, grid: &[f64]) -> Vec<f64> {
        grid.iter().map(|&x| self.density(x)).collect()
    }

    /// The resolved bandwidth h.
    pub fn bandwidth(&self) -> f64 {
        self.h
    }

    /// Number of finite observations used to fit the KDE.
    pub fn n(&self) -> usize {
        self.data.len()
    }
}

/// Evaluate kernel K at argument `u`.
///
/// Gaussian: K(u) = exp(−u²/2) / √(2π).
/// Epanechnikov: K(u) = 0.75·(1−u²) for |u| ≤ 1, else 0.
#[inline]
fn kernel_eval(k: Kernel, u: f64) -> f64 {
    match k {
        Kernel::Gaussian => {
            // K_G(u) = exp(-u²/2) / sqrt(2π)
            libm::exp(-0.5 * u * u) / libm::sqrt(2.0 * PI)
        }
        Kernel::Epanechnikov => {
            // K_E(u) = 0.75(1-u²) for |u|≤1, else 0
            if u.abs() <= 1.0 {
                0.75 * (1.0 - u * u)
            } else {
                0.0
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Auto-bin histograms
// ---------------------------------------------------------------------------

/// Bin specification for [`histogram_auto`].
#[derive(Debug, Clone, PartialEq)]
pub enum Bins {
    /// Exactly `k` equal-width bins; must be ≥ 1.
    Fixed(usize),
    /// Bins of fixed width `w`; must be > 0 and finite.
    Width(f64),
    /// Explicit bin edges (length ≥ 2, strictly increasing).
    ///
    /// This is the variable-width path — used by `quantile_edges` from
    /// the t-digest (area 2). Does NOT go through `Histogram::new`.
    Edges(Vec<f64>),
    /// Automatic rule-based bin count.
    Rule(Rule),
}

/// Automatic bin-count rule for [`histogram_auto`].
///
/// All formulas match `numpy.histogram_bin_edges` with the same rule name.
/// n = finite observation count; sd = sample std dev (ddof=1);
/// IQR = type-7 interquartile range via `accum::quantile_sorted`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// Sturges (1926): k = ⌈log₂ n⌉ + 1. Good for near-Normal, small n.
    Sturges,
    /// Scott (1979): w = 3.49·sd·n^(−1/3), k = ⌈range/w⌉.
    Scott,
    /// Freedman–Diaconis (1981): w = 2·IQR·n^(−1/3), k = ⌈range/w⌉.
    /// Falls back to Scott when IQR = 0.
    FreedmanDiaconis,
    /// Rice rule: k = ⌈2·n^(1/3)⌉.
    Rice,
    /// Square-root rule: k = ⌈√n⌉.
    Sqrt,
}

/// Normalization applied to bin counts by [`histogram_auto`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    /// Raw counts (integers stored as f64).
    Count,
    /// Probability density: count / (n_total · bin_width). Matches numpy `density=True`.
    Density,
    /// Relative frequency: count / n_total. Matches numpy `density=False` + manual division.
    Probability,
}

/// Compute a histogram with automatic or explicit binning.
///
/// Convention:
/// - Returns `(edges, values)` where `edges.len() = n_bins + 1` and
///   `values.len() = n_bins`.
/// - `Fixed`/`Width`/`Rule` paths delegate to `accum::Histogram` (which uses
///   `[lo, hi)` half-open bins except the last bin which is closed on the right).
/// - `Edges` path: variable-width bins via `partition_point` binary search;
///   each observation falls into bin `i` where `edges[i] ≤ x < edges[i+1]`
///   (last bin closed on right).
/// - `Norm::Density`: `count[i] / (n_total · (edges[i+1] − edges[i]))` per bin.
/// - `Norm::Probability`: `count[i] / n_total`.
/// - NaN policy: Omit.
///
/// Matches `numpy.histogram(xs, bins=<rule/edges>, density=<bool>)`.
///
/// # Parameters
/// - `xs`: input data; NaN values omitted.
/// - `bins`: binning strategy.
/// - `norm`: output normalization.
///
/// # Errors
/// - `EmptyInput` — `xs` is empty.
/// - `DomainError` — min == max (single-point cloud, can't form bins).
/// - `DomainError` — `Fixed(0)`.
/// - `DomainError` — `Width(w)` where w ≤ 0 or not finite.
/// - `DomainError` — `Edges` with fewer than 2 edges or not strictly increasing.
/// - `DomainError` — `Rule::FreedmanDiaconis` when IQR = 0 and sd = 0.
///
/// # Example
/// ```
/// use commonstats::density::{histogram_auto, Bins, Rule, Norm};
/// let xs = &[1.0, 2.0, 2.5, 3.0, 4.0, 5.0];
/// let (edges, counts) = histogram_auto(xs, Bins::Rule(Rule::Sturges), Norm::Count).unwrap();
/// assert_eq!(edges.len(), counts.len() + 1);
/// assert!(edges[0] <= 1.0 && *edges.last().unwrap() >= 5.0);
/// ```
pub fn histogram_auto(
    xs: &[f64],
    bins: Bins,
    norm: Norm,
) -> Result<(Vec<f64>, Vec<f64>), StatError> {
    let finite = nan::clean(xs, NanPolicy::Omit)?;
    let n = finite.len();
    let n_f = n as f64;

    let lo = descriptive::min(&finite)?;
    let hi = descriptive::max(&finite)?;

    if lo == hi {
        return Err(StatError::DomainError("min == max: cannot form histogram bins"));
    }

    // Handle the Edges path (variable-width): does not go through Histogram::new.
    if let Bins::Edges(ref edges) = bins {
        return histogram_edges(&finite, edges, norm);
    }

    // All other paths go through Histogram::new.
    let n_bins = resolve_bin_count(&finite, &bins, lo, hi, n_f)?;

    let mut hist = Histogram::new(lo, hi, n_bins)?;
    for &x in &finite {
        hist.update(x);
    }
    let result: HistResult = hist.finalize();
    let counts_u64 = &result.counts;
    let n_bins_actual = counts_u64.len();

    // Build uniform edges from lo..hi.
    let bin_width = (hi - lo) / n_bins_actual as f64;
    let edges: Vec<f64> = (0..=n_bins_actual)
        .map(|i| lo + i as f64 * bin_width)
        .collect();

    let values = apply_norm_uniform(counts_u64, n_f, bin_width, norm);
    Ok((edges, values))
}

/// Resolve bin count (and validate) for Fixed/Width/Rule variants.
fn resolve_bin_count(
    finite: &[f64],
    bins: &Bins,
    lo: f64,
    hi: f64,
    n_f: f64,
) -> Result<usize, StatError> {
    let range = hi - lo;
    match bins {
        Bins::Fixed(k) => {
            if *k == 0 {
                return Err(StatError::DomainError("Fixed(0): bin count must be ≥ 1"));
            }
            Ok(*k)
        }
        Bins::Width(w) => {
            if !w.is_finite() || *w <= 0.0 {
                return Err(StatError::DomainError(
                    "Width must be finite and > 0",
                ));
            }
            Ok(libm::ceil(range / w) as usize)
        }
        Bins::Rule(rule) => resolve_rule_bin_count(finite, *rule, lo, hi, n_f),
        Bins::Edges(_) => unreachable!("Edges handled before resolve_bin_count"),
    }
}

/// Compute number of bins for automatic rules.
fn resolve_rule_bin_count(
    finite: &[f64],
    rule: Rule,
    lo: f64,
    hi: f64,
    n_f: f64,
) -> Result<usize, StatError> {
    let range = hi - lo;
    match rule {
        Rule::Sturges => {
            // k = ⌈log₂ n⌉ + 1
            Ok(libm::ceil(libm::log2(n_f)) as usize + 1)
        }
        Rule::Scott => {
            // w = 3.49 * sd * n^(-1/3), k = ⌈range/w⌉
            let sd = descriptive::sd(finite, descriptive::Ddof::Sample)?;
            let w = 3.49 * sd * libm::pow(n_f, -1.0 / 3.0);
            Ok(libm::ceil(range / w) as usize)
        }
        Rule::FreedmanDiaconis => {
            // w = 2 * IQR * n^(-1/3); IQR=0 → fall back to Scott
            let mut sorted = finite.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q1 = accum::quantile_sorted(&sorted, 0.25);
            let q3 = accum::quantile_sorted(&sorted, 0.75);
            let iqr = q3 - q1;
            if iqr == 0.0 {
                // Fall back to Scott; but if sd=0 too, that's DomainError.
                let sd = descriptive::sd(finite, descriptive::Ddof::Sample)?;
                if sd == 0.0 {
                    return Err(StatError::DomainError(
                        "FreedmanDiaconis: IQR=0 and sd=0 — cannot determine bin width",
                    ));
                }
                let w = 3.49 * sd * libm::pow(n_f, -1.0 / 3.0);
                return Ok(libm::ceil(range / w) as usize);
            }
            let w = 2.0 * iqr * libm::pow(n_f, -1.0 / 3.0);
            Ok(libm::ceil(range / w) as usize)
        }
        Rule::Rice => {
            // k = ⌈2 * n^(1/3)⌉
            Ok(libm::ceil(2.0 * libm::cbrt(n_f)) as usize)
        }
        Rule::Sqrt => {
            // k = ⌈√n⌉
            Ok(libm::ceil(libm::sqrt(n_f)) as usize)
        }
    }
}

/// Handle the variable-width `Bins::Edges` path via partition_point.
///
/// Does NOT use Histogram::new. Each observation falls into bin i where
/// edges[i] ≤ x < edges[i+1]; the last bin is closed on the right.
/// Norm::Density uses per-bin widths.
fn histogram_edges(
    finite: &[f64],
    edges: &[f64],
    norm: Norm,
) -> Result<(Vec<f64>, Vec<f64>), StatError> {
    if edges.len() < 2 {
        return Err(StatError::DomainError(
            "Edges must have at least 2 elements",
        ));
    }
    // Verify strictly increasing.
    for w in edges.windows(2) {
        if w[0] >= w[1] {
            return Err(StatError::DomainError(
                "Edges must be strictly increasing",
            ));
        }
    }

    let n_bins = edges.len() - 1;
    let mut counts = vec![0u64; n_bins];
    let n_total = finite.len() as f64;
    let last_edge = *edges.last().unwrap();

    for &x in finite {
        // partition_point finds first index i where edges[i] > x.
        // Bin index = i - 1. Values exactly at last edge go in last bin.
        let idx = if x == last_edge {
            n_bins
        } else {
            edges.partition_point(|&e| e <= x)
        };
        // idx is in [1, n_bins+1]: bin = idx-1; clamp out-of-range silently.
        if idx >= 1 && idx <= n_bins {
            counts[idx - 1] += 1;
        }
    }

    let values = match norm {
        Norm::Count => counts.iter().map(|&c| c as f64).collect(),
        Norm::Probability => counts.iter().map(|&c| c as f64 / n_total).collect(),
        Norm::Density => counts
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let bin_width = edges[i + 1] - edges[i];
                c as f64 / (n_total * bin_width)
            })
            .collect(),
    };

    Ok((edges.to_vec(), values))
}

/// Apply normalization for uniform-width bins (Fixed/Width/Rule paths).
fn apply_norm_uniform(counts: &[u64], n_total: f64, bin_width: f64, norm: Norm) -> Vec<f64> {
    match norm {
        Norm::Count => counts.iter().map(|&c| c as f64).collect(),
        Norm::Probability => counts.iter().map(|&c| c as f64 / n_total).collect(),
        Norm::Density => counts
            .iter()
            .map(|&c| c as f64 / (n_total * bin_width))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::StatError;

    // ---- KDE constructor error cases ----

    #[test]
    fn kde_empty_input() {
        let xs: &[f64] = &[];
        assert_eq!(kde(xs, Kernel::Gaussian, Bandwidth::Silverman).unwrap_err(), StatError::EmptyInput);
    }

    #[test]
    fn kde_too_few_observations() {
        let xs = &[1.0_f64];
        assert_eq!(
            kde(xs, Kernel::Gaussian, Bandwidth::Silverman).unwrap_err(),
            StatError::TooFewObservations { needed: 2, got: 1 }
        );
    }

    #[test]
    fn kde_zero_variance_silverman() {
        let xs = &[3.0_f64, 3.0, 3.0, 3.0];
        assert!(matches!(
            kde(xs, Kernel::Gaussian, Bandwidth::Silverman).unwrap_err(),
            StatError::DomainError(_)
        ));
    }

    #[test]
    fn kde_zero_variance_scott() {
        let xs = &[5.0_f64, 5.0, 5.0];
        assert!(matches!(
            kde(xs, Kernel::Gaussian, Bandwidth::Scott).unwrap_err(),
            StatError::DomainError(_)
        ));
    }

    #[test]
    fn kde_fixed_nonpositive() {
        let xs = &[1.0_f64, 2.0, 3.0];
        assert!(matches!(
            kde(xs, Kernel::Gaussian, Bandwidth::Fixed(0.0)).unwrap_err(),
            StatError::DomainError(_)
        ));
        assert!(matches!(
            kde(xs, Kernel::Gaussian, Bandwidth::Fixed(-1.0)).unwrap_err(),
            StatError::DomainError(_)
        ));
        assert!(matches!(
            kde(xs, Kernel::Gaussian, Bandwidth::Fixed(f64::NAN)).unwrap_err(),
            StatError::DomainError(_)
        ));
    }

    #[test]
    fn kde_fixed_resolves_bandwidth() {
        let xs = &[1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let k = kde(xs, Kernel::Gaussian, Bandwidth::Fixed(0.5)).unwrap();
        assert_eq!(k.bandwidth(), 0.5);
        assert_eq!(k.n(), 5);
    }

    #[test]
    fn kde_silverman_bandwidth_formula() {
        let xs = &[1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let k = kde(xs, Kernel::Gaussian, Bandwidth::Silverman).unwrap();
        let sd = crate::descriptive::sd(xs, crate::descriptive::Ddof::Sample).unwrap();
        let n = 5_f64;
        let expected_h = sd * libm::pow(3.0 * n / 4.0, -0.2);
        let diff = (k.bandwidth() - expected_h).abs();
        assert!(diff < 1e-14, "Silverman h: got {}, want {}", k.bandwidth(), expected_h);
    }

    #[test]
    fn kde_scott_bandwidth_formula() {
        let xs = &[1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let k = kde(xs, Kernel::Gaussian, Bandwidth::Scott).unwrap();
        let sd = crate::descriptive::sd(xs, crate::descriptive::Ddof::Sample).unwrap();
        let expected_h = sd * libm::pow(5.0_f64, -0.2);
        let diff = (k.bandwidth() - expected_h).abs();
        assert!(diff < 1e-14, "Scott h: got {}, want {}", k.bandwidth(), expected_h);
    }

    // ---- KDE density formula ----

    #[test]
    fn kde_gaussian_density_formula() {
        // Single point: xs=[0.0, 1.0], Fixed(1.0)
        // density(0.5) = (1/2·1) · [K_G((0.5-0)/1) + K_G((0.5-1)/1)]
        //              = 0.5 · [exp(-0.125)/sqrt(2π) + exp(-0.125)/sqrt(2π)]
        //              = exp(-0.125)/sqrt(2π)
        let xs = &[0.0_f64, 1.0];
        let k = kde(xs, Kernel::Gaussian, Bandwidth::Fixed(1.0)).unwrap();
        let expected = libm::exp(-0.125) / libm::sqrt(2.0 * PI);
        let got = k.density(0.5);
        let diff = (got - expected).abs();
        assert!(diff < 1e-14, "density at 0.5: got {got}, want {expected}");
    }

    #[test]
    fn kde_epanechnikov_density_formula() {
        // xs=[0.0, 1.0], Fixed(2.0), x=0.5
        // density(0.5) = (1/2·2) · [K_E((0.5-0)/2) + K_E((0.5-1)/2)]
        //              = 0.25 · [0.75(1-0.0625) + 0.75(1-0.0625)]
        //              = 0.25 · 2 · 0.75 · 0.9375 = 0.3515625
        let xs = &[0.0_f64, 1.0];
        let k = kde(xs, Kernel::Epanechnikov, Bandwidth::Fixed(2.0)).unwrap();
        let u1 = (0.5 - 0.0) / 2.0; // 0.25
        let u2 = (0.5 - 1.0) / 2.0; // -0.25
        let expected = 0.25 * (0.75 * (1.0 - u1 * u1) + 0.75 * (1.0 - u2 * u2));
        let got = k.density(0.5);
        let diff = (got - expected).abs();
        assert!(diff < 1e-15, "epanechnikov density: got {got}, want {expected}");
    }

    #[test]
    fn kde_evaluate_length() {
        let xs = &[1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let k = kde(xs, Kernel::Gaussian, Bandwidth::Fixed(1.0)).unwrap();
        let grid = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(k.evaluate(grid).len(), grid.len());
    }

    #[test]
    fn kde_nan_in_input_omitted() {
        let xs = &[1.0_f64, f64::NAN, 3.0, f64::NAN, 5.0];
        let k = kde(xs, Kernel::Gaussian, Bandwidth::Silverman).unwrap();
        assert_eq!(k.n(), 3);
    }

    // ---- histogram_auto error cases ----

    #[test]
    fn hist_empty_input() {
        let xs: &[f64] = &[];
        assert_eq!(
            histogram_auto(xs, Bins::Fixed(5), Norm::Count).unwrap_err(),
            StatError::EmptyInput
        );
    }

    #[test]
    fn hist_min_eq_max() {
        let xs = &[3.0_f64, 3.0, 3.0];
        assert!(matches!(
            histogram_auto(xs, Bins::Fixed(5), Norm::Count).unwrap_err(),
            StatError::DomainError(_)
        ));
    }

    #[test]
    fn hist_fixed_zero() {
        let xs = &[1.0_f64, 2.0, 3.0];
        assert!(matches!(
            histogram_auto(xs, Bins::Fixed(0), Norm::Count).unwrap_err(),
            StatError::DomainError(_)
        ));
    }

    #[test]
    fn hist_width_nonpositive() {
        let xs = &[1.0_f64, 2.0, 3.0];
        assert!(matches!(
            histogram_auto(xs, Bins::Width(0.0), Norm::Count).unwrap_err(),
            StatError::DomainError(_)
        ));
        assert!(matches!(
            histogram_auto(xs, Bins::Width(-1.0), Norm::Count).unwrap_err(),
            StatError::DomainError(_)
        ));
    }

    #[test]
    fn hist_edges_too_few() {
        let xs = &[1.0_f64, 2.0, 3.0];
        assert!(matches!(
            histogram_auto(xs, Bins::Edges(vec![1.0]), Norm::Count).unwrap_err(),
            StatError::DomainError(_)
        ));
    }

    #[test]
    fn hist_edges_not_strictly_increasing() {
        let xs = &[1.0_f64, 2.0, 3.0];
        assert!(matches!(
            histogram_auto(xs, Bins::Edges(vec![1.0, 2.0, 2.0, 3.0]), Norm::Count).unwrap_err(),
            StatError::DomainError(_)
        ));
    }

    // ---- bin rules ----

    #[test]
    fn hist_sturges_bin_count() {
        // n=16 → k = ceil(log2(16)) + 1 = 4+1 = 5
        let xs: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let (edges, counts) = histogram_auto(&xs, Bins::Rule(Rule::Sturges), Norm::Count).unwrap();
        assert_eq!(counts.len(), 5, "Sturges n=16: expected 5 bins");
        assert_eq!(edges.len(), 6);
        let total: f64 = counts.iter().sum();
        assert_eq!(total, 16.0);
    }

    #[test]
    fn hist_rice_bin_count() {
        // n=8 → k = ceil(2 * 8^(1/3)) = ceil(2*2) = 4
        let xs: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let (_, counts) = histogram_auto(&xs, Bins::Rule(Rule::Rice), Norm::Count).unwrap();
        assert_eq!(counts.len(), 4, "Rice n=8: expected 4 bins");
    }

    #[test]
    fn hist_sqrt_bin_count() {
        // n=9 → k = ceil(sqrt(9)) = 3
        let xs: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let (_, counts) = histogram_auto(&xs, Bins::Rule(Rule::Sqrt), Norm::Count).unwrap();
        assert_eq!(counts.len(), 3, "Sqrt n=9: expected 3 bins");
    }

    #[test]
    fn hist_edges_partition_point_assignment() {
        // edges = [0,1,2,3]; xs = [0.0,0.5,1.0,1.5,2.0,2.5,2.9,3.0]
        // bins: [0,1)→2, [1,2)→2, [2,3]→4
        let xs = &[0.0_f64, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9, 3.0];
        let edges = vec![0.0, 1.0, 2.0, 3.0];
        let (ret_edges, counts) =
            histogram_auto(xs, Bins::Edges(edges.clone()), Norm::Count).unwrap();
        assert_eq!(ret_edges, edges);
        assert_eq!(counts, vec![2.0, 2.0, 4.0]);
    }

    #[test]
    fn hist_norm_probability() {
        let xs = &[1.0_f64, 2.0, 3.0, 4.0];
        let (_, vals) = histogram_auto(xs, Bins::Fixed(2), Norm::Probability).unwrap();
        let total: f64 = vals.iter().sum();
        assert!((total - 1.0).abs() < 1e-14, "Probability sum != 1: {total}");
    }

    #[test]
    fn hist_norm_density_integrates_to_one() {
        let xs = &[1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let (edges, vals) = histogram_auto(xs, Bins::Fixed(4), Norm::Density).unwrap();
        let integral: f64 = vals
            .iter()
            .enumerate()
            .map(|(i, &v)| v * (edges[i + 1] - edges[i]))
            .sum();
        assert!((integral - 1.0).abs() < 1e-13, "Density integral != 1: {integral}");
    }

    #[test]
    fn hist_norm_density_edges_variable_width() {
        // Edges: [0,1,3]; xs=[0.0,0.5,1.5,2.5]; counts=[2,2]
        // Density: [2/(4*1), 2/(4*2)] = [0.5, 0.25]
        let xs = &[0.0_f64, 0.5, 1.5, 2.5];
        let edges = vec![0.0, 1.0, 3.0];
        let (_, vals) = histogram_auto(xs, Bins::Edges(edges), Norm::Density).unwrap();
        assert!((vals[0] - 0.5).abs() < 1e-14, "density bin0: {}", vals[0]);
        assert!((vals[1] - 0.25).abs() < 1e-14, "density bin1: {}", vals[1]);
    }
}
