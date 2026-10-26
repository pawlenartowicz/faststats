//! Kernel density estimation and auto-binned histograms.
//!
//! Two independent entry points:
//! - [`kde`] — direct-sum KDE with Gaussian or Epanechnikov kernel.
//! - [`histogram_auto`] — bin-rule front-end over [`crate::accum::Histogram`].
//!
//! Both are base-feature (no gate) and WASM-clean (libm-only math).

use crate::accum;
use crate::accum::{Accumulator, HistResult, Histogram};
use crate::descriptive;
use crate::error::StatError;
use crate::nan;
use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::PI;

// ---------------------------------------------------------------------------
// KDE
// ---------------------------------------------------------------------------

/// Kernel function for KDE.
///
/// Convention: each kernel integrates to 1 over ℝ; the bandwidth `h` scales
/// the kernel as written below (for Epanechnikov, the support half-width, not
/// its standard deviation h/√5).
/// Gaussian matches `scipy.stats.gaussian_kde`; Epanechnikov matches
/// `sklearn.neighbors.KernelDensity(kernel='epanechnikov', bandwidth=h)`
/// (scipy's `gaussian_kde` has no other kernel). Both are validated against
/// mpmath truth in `tests/fixtures/kde.json`.
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
/// Matches scipy.stats.gaussian_kde evaluated on the same bandwidth (Gaussian
/// kernel; Epanechnikov: see [`Kernel`]).
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
/// Matches scipy.stats.gaussian_kde with `bw_method='silverman'` / `'scott'`;
/// `Fixed(h)` matches a scalar `bw_method = h/sd` (a scalar sets scipy's
/// covariance factor, and its kernel sd is that factor times sd, ddof=1).
///
/// # Parameters
/// - `xs`: input data; NaN values are silently omitted.
/// - `kernel`: kernel function (Gaussian or Epanechnikov).
/// - `bandwidth`: bandwidth selection rule or fixed value.
///
/// # Errors
/// - `EmptyInput` — `xs` is empty.
/// - `TooFewObservations{needed:2, got:n}` — fewer than 2 finite values.
/// - `DomainError` — Silverman/Scott on zero-variance data (h would be 0) or on
///   data whose sd is not finite (±∞ values, or an sd that overflows).
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
    let finite = nan::clean(xs)?; // EmptyInput / AllNaN
    let n = finite.len();
    if n < 2 {
        return Err(StatError::TooFewObservations { needed: 2, got: n });
    }

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
            let sd = descriptive::sd(&finite, descriptive::Ddof::Sample)?;
            if sd == 0.0 || !sd.is_finite() {
                return Err(StatError::DomainError(
                    "zero-variance or non-finite data: Silverman bandwidth undefined",
                ));
            }
            sd * libm::pow(3.0 * n as f64 / 4.0, -0.2)
        }
        Bandwidth::Scott => {
            let sd = descriptive::sd(&finite, descriptive::Ddof::Sample)?;
            if sd == 0.0 || !sd.is_finite() {
                return Err(StatError::DomainError(
                    "zero-variance or non-finite data: Scott bandwidth undefined",
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
    /// Convention: identical to `scipy.stats.gaussian_kde.evaluate([x])[0]`
    /// at the same bandwidth (Gaussian kernel; Epanechnikov: see [`Kernel`]).
    pub fn density(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        let n = self.data.len() as f64;
        let h = self.h;
        let sum: f64 = self
            .data
            .iter()
            .map(|&xi| kernel_eval(self.kernel, (x - xi) / h))
            .sum();
        sum / (n * h)
    }

    /// Evaluate the KDE density at every point in `grid`. O(n·m) direct sum.
    ///
    /// Returns a `Vec<f64>` of length `grid.len()`. Non-finite grid points
    /// produce 0.0. Matches scipy.stats.gaussian_kde.evaluate(grid) (Gaussian
    /// kernel; Epanechnikov: see [`Kernel`]).
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
    /// Bins of fixed width `w`; must be > 0 and finite. The first edge is the
    /// minimum and the last is the first edge at or past the maximum, so the
    /// last bin can extend beyond the data.
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
    /// Scott (1979): w = 3.49·sd·n^(−1/3), k = ⌈range/w⌉; 1 bin when sd = 0
    /// (possible with min ≠ max when the sd underflows).
    Scott,
    /// Freedman–Diaconis (1981): w = 2·IQR·n^(−1/3), k = ⌈range/w⌉;
    /// 1 bin when IQR = 0.
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
    /// Probability density: count / (n_in · bin_width), `n_in` the count inside
    /// the bins. Matches numpy `density=True`.
    Density,
    /// Relative frequency: count / n_in. Matches numpy `density=False` divided
    /// by the in-range count.
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
/// - `Norm::Density`: `count[i] / (n_in · (edges[i+1] − edges[i]))` per bin,
///   where `n_in` is the count inside the bins (all finite values except on
///   the `Edges` path, which drops values outside its edges).
/// - `Norm::Probability`: `count[i] / n_in`.
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
/// - `DomainError` — the bin count reaches `isize::MAX / 8`, past which its `Vec`s
///   cannot be built (e.g. `Fixed(usize::MAX)`, a tiny `Width`, or a rule whose bin
///   width is vanishingly small).
/// - `DomainError` — `Width(w)` where w ≤ 0 or not finite.
/// - `DomainError` — `hi − lo` is not finite (`±∞` in `xs`, or a span past
///   `f64::MAX`), except on the `Edges` path.
/// - `DomainError` — `Width(w)` whose last edge `lo + n_bins·w` overflows f64.
/// - `DomainError` — `Edges` with fewer than 2 edges or not strictly increasing.
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
    let finite = nan::clean(xs)?;
    let n = finite.len();
    let n_f = n as f64;

    let lo = descriptive::min(&finite)?;
    let hi = descriptive::max(&finite)?;

    if lo == hi {
        return Err(StatError::DomainError(
            "min == max: cannot form histogram bins",
        ));
    }

    // Handle the Edges path (variable-width): does not go through Histogram::new.
    if let Bins::Edges(ref edges) = bins {
        return histogram_edges(&finite, edges, norm);
    }

    // ±∞ in the data (numpy raises too: "autodetected range ... is not
    // finite") or a span past f64::MAX leaves no finite bin width.
    if !(hi - lo).is_finite() {
        return Err(StatError::DomainError(
            "histogram: data range is not finite",
        ));
    }

    let n_bins = resolve_bin_count(&finite, &bins, lo, hi, n_f)?;

    // `.max(hi)` keeps the maximum inside the bins when `lo + k·w` rounds just
    // below it.
    let top = match bins {
        Bins::Width(w) => (lo + n_bins as f64 * w).max(hi),
        _ => hi,
    };
    if top == f64::INFINITY {
        return Err(StatError::DomainError("histogram: Width bins overflow f64"));
    }
    let mut hist = Histogram::new(lo, top, n_bins)?;
    for &x in &finite {
        hist.update(x);
    }
    let HistResult { counts, edges, .. } = hist.finalize();
    let bin_width = (top - lo) / n_bins as f64;

    let values = apply_norm_uniform(&counts, n_f, bin_width, norm);
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
                return Err(StatError::DomainError("Width must be finite and > 0"));
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
            // A zero width gives 1 bin, as numpy does.
            let sd = descriptive::sd(finite, descriptive::Ddof::Sample)?;
            if sd == 0.0 {
                return Ok(1);
            }
            let w = 3.49 * sd * libm::pow(n_f, -1.0 / 3.0);
            Ok(libm::ceil(range / w) as usize)
        }
        Rule::FreedmanDiaconis => {
            // w = 2 * IQR * n^(-1/3); a zero width gives 1 bin, as numpy does.
            let mut sorted = finite.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q1 = accum::quantile_sorted(&sorted, 0.25);
            let q3 = accum::quantile_sorted(&sorted, 0.75);
            let iqr = q3 - q1;
            if iqr == 0.0 {
                return Ok(1);
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
/// Norm::Density uses per-bin widths. Values outside the edges are dropped and
/// do not count towards the Density/Probability total; with none inside, both
/// are NaN (0/0), as in numpy.
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
    for w in edges.windows(2) {
        if w[0] >= w[1] {
            return Err(StatError::DomainError("Edges must be strictly increasing"));
        }
    }

    let n_bins = edges.len() - 1;
    let mut counts = vec![0u64; n_bins];
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

    // Normalised by the in-range count, as numpy.histogram does: values outside
    // the edges are dropped before normalising.
    let n_in = counts.iter().sum::<u64>() as f64;
    let values = match norm {
        Norm::Count => counts.iter().map(|&c| c as f64).collect(),
        Norm::Probability => counts.iter().map(|&c| c as f64 / n_in).collect(),
        Norm::Density => counts
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let bin_width = edges[i + 1] - edges[i];
                c as f64 / (n_in * bin_width)
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

    // ---- histogram_auto error cases ----

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
    fn hist_zero_width_rule_one_bin() {
        // numpy 2.3.5: `histogram_bin_edges([0]*10 + [1], 'fd')` → [0, 1] and
        // `histogram_bin_edges([0, 5e-324], 'scott')` → [0, 5e-324].
        let mut xs = [0.0_f64; 11];
        xs[10] = 1.0;
        let (edges, counts) =
            histogram_auto(&xs, Bins::Rule(Rule::FreedmanDiaconis), Norm::Count).unwrap();
        assert_eq!(edges, vec![0.0, 1.0]);
        assert_eq!(counts, vec![11.0]);
        let (edges, _) =
            histogram_auto(&[0.0, 5e-324], Bins::Rule(Rule::Scott), Norm::Count).unwrap();
        assert_eq!(edges, vec![0.0, 5e-324]);
    }

    #[test]
    fn hist_bin_count_too_large() {
        assert!(matches!(
            histogram_auto(&[0.0, 1.0], Bins::Width(1e-300), Norm::Count).unwrap_err(),
            StatError::DomainError(_)
        ));
        assert!(matches!(
            histogram_auto(&[0.0, 1.0], Bins::Fixed(usize::MAX), Norm::Count).unwrap_err(),
            StatError::DomainError(_)
        ));
        let xs = &[0.0, 1.0, 1.0, 1.0 + 2.2e-16, 1.0, 1e300];
        assert!(matches!(
            histogram_auto(xs, Bins::Rule(Rule::FreedmanDiaconis), Norm::Count).unwrap_err(),
            StatError::DomainError(_)
        ));
    }

    #[test]
    fn hist_non_finite_range() {
        for xs in [
            &[0.0, 1.0, f64::INFINITY][..],
            &[f64::NEG_INFINITY, 0.0, 1.0],
            &[-1.7e308, 1.7e308],
        ] {
            for bins in [
                Bins::Fixed(2),
                Bins::Width(1.0),
                Bins::Rule(Rule::Sturges),
                Bins::Rule(Rule::Scott),
            ] {
                assert!(matches!(
                    histogram_auto(xs, bins, Norm::Count).unwrap_err(),
                    StatError::DomainError(_)
                ));
            }
        }
        assert!(matches!(
            histogram_auto(&[0.0, 1.7e308], Bins::Width(1.5e308), Norm::Count).unwrap_err(),
            StatError::DomainError(_)
        ));
        // Explicit finite edges drop ±∞, as numpy does.
        let xs = &[f64::NEG_INFINITY, 0.5, f64::INFINITY];
        let (_, counts) = histogram_auto(xs, Bins::Edges(vec![0.0, 1.0]), Norm::Count).unwrap();
        assert_eq!(counts, vec![1.0]);
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
        // Asymmetric counts, so a mis-assigned or reordered bin shows.
        let xs = &[1.0_f64, 1.0, 1.0, 4.0];
        let (edges, vals) = histogram_auto(xs, Bins::Fixed(2), Norm::Probability).unwrap();
        assert_eq!(edges, vec![1.0, 2.5, 4.0]);
        assert_eq!(vals, vec![0.75, 0.25]);
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
