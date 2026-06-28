//! Dunning merging t-digest: associative, mergeable streaming quantile/CDF
//! estimation (arXiv:1902.04023). The sole *approximation* in CommonStats —
//! rank error ≤ 0.75% at the default δ = 100. Exact `min`/`max` are tracked
//! separately so the endpoints and extreme tails are exact, not sketch-based.

use crate::accum::{Accumulator, Mergeable};
use crate::error::StatError;

/// A weighted summary point: the mean of a cluster of observations and the
/// number of observations it stands for. Sorted by `mean`; `weight ≥ 1` after
/// `compress`. Private — an implementation detail of the sketch.
#[derive(Debug, Clone, Copy)]
struct Centroid {
    /// Cluster mean (the abscissa the sketch interpolates over).
    mean: f64,
    /// Number of observations this centroid summarizes.
    weight: f64,
}

/// A Dunning merging t-digest over `f64` observations.
///
/// Estimates quantiles and the CDF of a (possibly streaming) sample with bounded
/// rank error. Implements [`Mergeable`] + [`Accumulator`] so chunks can be summed
/// in any order; the result is order-independent within the approximation bound.
///
/// **Approximation:** rank error ≤ `3/(4·delta)` (≤ 0.75% at `delta = 100`),
/// worst near the median and tighter in the tails. Errors are *rank* errors, not
/// value errors. `min`/`max` and `quantile(0)`/`quantile(1)` are exact.
///
/// Approximates `numpy.quantile(method="linear")` on the underlying sample to
/// within the rank-error bound above (no exact oracle — validated by statistical
/// tolerance, see `tests/tdigest_test.rs`).
#[derive(Debug, Clone)]
pub struct TDigest {
    /// Compressed summary, sorted by `mean`.
    centroids: Vec<Centroid>,
    /// Unsorted staging buffer; flushed into `centroids` when it reaches `buf_cap`.
    ingest_buf: Vec<f64>,
    /// Σ centroid weights + `ingest_buf.len()`.
    total_weight: f64,
    /// Count of non-NaN observations seen. `0` is the `empty()` identity sentinel.
    n: u64,
    /// Exact minimum observed (`+∞` when empty).
    min: f64,
    /// Exact maximum observed (`−∞` when empty).
    max: f64,
    /// Compression parameter δ; immutable after construction. Larger = more
    /// centroids = higher accuracy. Must be ≥ 1.
    delta: f64,
    /// Ingest-buffer capacity, `10 · delta`; flush threshold.
    buf_cap: usize,
}

/// A lightweight, flushed snapshot of a [`TDigest`].
///
/// Holds the compressed centroids, total weight, count, and exact `min`/`max`,
/// exposing the same query methods (`quantile`, `cdf`, `min`, `max`, `count`) so
/// callers can query without holding the mutable digest. Produced by
/// [`TDigest::finalize`].
#[derive(Debug, Clone)]
pub struct TDigestResult {
    /// Compressed summary, sorted by `mean`.
    centroids: Vec<Centroid>,
    /// Σ centroid weights.
    total_weight: f64,
    /// Count of non-NaN observations summarized.
    n: u64,
    /// Exact minimum (`+∞` when empty).
    min: f64,
    /// Exact maximum (`−∞` when empty).
    max: f64,
}

/// Dunning's k1 scale function (arXiv:1902.04023 §3.1).
///
/// Maps a cumulative-probability `q ∈ [0,1]` to a k-index in `[−δ/4, +δ/4]`.
/// `asin` is steep near 0 and 1, so equal k-spacing packs more centroids
/// (higher resolution) into the tails than near the median. Two adjacent
/// centroids spanning `q_lo < q_hi` may merge iff `k1(q_hi) − k1(q_lo) ≤ 1`.
/// `libm::asin` is WASM-clean.
fn k1(q: f64, delta: f64) -> f64 {
    (delta / (2.0 * core::f64::consts::PI)) * libm::asin(2.0 * q - 1.0)
}

/// Merge two centroid lists, each already sorted by `mean`, into one sorted list
/// (stable two-pointer merge). Pure — the second half of merge's associativity
/// guarantee.
fn merge_sorted_centroids(a: &[Centroid], b: &[Centroid]) -> Vec<Centroid> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].mean <= b[j].mean {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// Compress a sorted centroid list in place, merging adjacent centroids while the
/// k1 span stays within one k-unit (Dunning §3.1). Pure function of the input
/// list + `total`: this is what makes [`TDigest::merge`] associative.
///
/// `total` = Σ weights of `centroids` (passed in to avoid recomputation).
fn compress(centroids: &mut Vec<Centroid>, total: f64, delta: f64) {
    if centroids.len() <= 1 {
        return;
    }
    let mut out: Vec<Centroid> = Vec::with_capacity(centroids.len());
    let mut q_lo = 0.0_f64;
    let mut current = centroids[0];
    for &c in &centroids[1..] {
        // q_lo is a cumulative probability; the candidate upper probability after
        // absorbing c is q_lo + (current.weight + c.weight)/total, written here as
        // one expression (algebraically identical).
        let q_hi = (q_lo * total + current.weight + c.weight) / total;
        if k1(q_hi, delta) - k1(q_lo, delta) <= 1.0 {
            let new_w = current.weight + c.weight;
            current.mean = (current.mean * current.weight + c.mean * c.weight) / new_w;
            current.weight = new_w;
        } else {
            q_lo += current.weight / total;
            out.push(current);
            current = c;
        }
    }
    out.push(current);
    *centroids = out;
}

/// Interior-quantile interpolation (Dunning §4: linear interp on centroid
/// midpoints). `centroids` sorted, `total` = Σ weights, `min`/`max` exact, `n` > 0,
/// `0 < q < 1` (endpoints handled by the callers). Centroid `i` sits at cumulative
/// position `pos[i] = Σ_{j<i} w[j] + w[i]/2`; we interpolate the mean at the
/// target rank `r = q·total` between the two bracketing midpoints.
fn quantile_interior(centroids: &[Centroid], total: f64, min: f64, max: f64, q: f64) -> f64 {
    if centroids.len() == 1 {
        return centroids[0].mean;
    }
    let r = q * total;
    let mut cum = 0.0_f64; // Σ_{j<i} w[j] at the start of centroid i
    let mut prev_pos = 0.0_f64;
    let mut prev_mean = min; // left anchor: exact min at rank 0
    for c in centroids {
        let pos = cum + c.weight / 2.0;
        if r < pos {
            // interpolate between (prev_pos, prev_mean) and (pos, c.mean)
            let span = pos - prev_pos;
            let t = if span > 0.0 { (r - prev_pos) / span } else { 0.0 };
            return prev_mean + t * (c.mean - prev_mean);
        }
        prev_pos = pos;
        prev_mean = c.mean;
        cum += c.weight;
    }
    // r is at or beyond the last midpoint: interpolate toward exact max at rank total.
    let span = total - prev_pos;
    let t = if span > 0.0 { (r - prev_pos) / span } else { 0.0 };
    prev_mean + t * (max - prev_mean)
}

/// CDF core (inverse of `quantile_interior`). 0 below `min`, 1 above `max`, NaN
/// when empty (`n == 0`). Interior: locate `x` among centroid means and map back
/// to cumulative probability via the same midpoint interpolation.
fn cdf_core(centroids: &[Centroid], total: f64, n: u64, min: f64, max: f64, x: f64) -> f64 {
    if n == 0 {
        return f64::NAN;
    }
    if x < min {
        return 0.0;
    }
    if x > max {
        return 1.0;
    }
    if centroids.is_empty() {
        // n>0 but unflushed/degenerate guard; treat as a point mass at min==max.
        return if x >= min { 1.0 } else { 0.0 };
    }
    let mut cum = 0.0_f64;
    let mut prev_pos = 0.0_f64;
    let mut prev_mean = min;
    for c in centroids {
        let pos = cum + c.weight / 2.0;
        if x < c.mean {
            let span = c.mean - prev_mean;
            let t = if span > 0.0 { (x - prev_mean) / span } else { 0.0 };
            return (prev_pos + t * (pos - prev_pos)) / total;
        }
        prev_pos = pos;
        prev_mean = c.mean;
        cum += c.weight;
    }
    let span = max - prev_mean;
    let t = if span > 0.0 { (x - prev_mean) / span } else { 1.0 };
    (prev_pos + t * (total - prev_pos)) / total
}

/// `k+1` quantile-spaced (approximately equal-frequency) bin edges spanning
/// `[min, max]`, for variable-width histogramming.
///
/// **Convention:** `edges[i] = digest.quantile(i/k)` for `i ∈ 0..=k`, so
/// `edges[0]` is the exact `min` and `edges[k]` the exact `max`. Feed the result
/// to `histogram_auto(xs, Bins::Edges(edges), norm)` (§3) — there is **no**
/// `Histogram::from_edges` constructor; the variable-width binning lives in
/// `histogram_auto`'s `Bins::Edges` path. Edges are approximate (rank-spaced),
/// so equal frequency holds only within the digest's rank-error bound.
///
/// # Errors
/// `DomainError("quantile_edges requires k >= 1")` if `k == 0`;
/// `DomainError("quantile_edges on empty digest")` if the digest has no data.
///
/// ```
/// use commonstats::accum::{TDigest, quantile_edges};
/// let mut d = TDigest::empty();
/// for i in 0..1000 { d.update(i as f64); }
/// let edges = quantile_edges(&d, 4).unwrap();
/// assert_eq!(edges.len(), 5);
/// assert_eq!(edges[0], d.min());
/// assert_eq!(edges[4], d.max());
/// ```
pub fn quantile_edges(digest: &TDigest, k: usize) -> Result<Vec<f64>, StatError> {
    if k < 1 {
        return Err(StatError::DomainError("quantile_edges requires k >= 1"));
    }
    if digest.count() == 0 {
        return Err(StatError::DomainError("quantile_edges on empty digest"));
    }
    let mut edges = Vec::with_capacity(k + 1);
    for i in 0..=k {
        // quantile(0) == min and quantile(1) == max are returned exactly, so the
        // endpoints need no special-casing here.
        let q = i as f64 / k as f64;
        edges.push(digest.quantile(q)?);
    }
    Ok(edges)
}

impl TDigest {
    /// Construct an empty t-digest with compression `delta`.
    ///
    /// **Convention:** `delta` ≥ 1 (larger = more accurate, more centroids). The
    /// ingest buffer holds `10 · delta` values before each flush/compress.
    ///
    /// # Errors
    /// `DomainError` if `delta < 1` or `delta` is NaN.
    ///
    /// ```
    /// use commonstats::accum::TDigest;
    /// assert!(TDigest::new(100.0).is_ok());
    /// assert!(TDigest::new(0.5).is_err());
    /// ```
    pub fn new(delta: f64) -> Result<Self, StatError> {
        if delta.is_nan() || delta < 1.0 {
            return Err(StatError::DomainError("t-digest delta must be >= 1"));
        }
        Ok(TDigest {
            centroids: Vec::new(),
            ingest_buf: Vec::new(),
            total_weight: 0.0,
            n: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            delta,
            buf_cap: (10.0 * delta) as usize,
        })
    }

    /// Default compression δ = 100 (rank error ≤ 0.75%).
    pub fn default_delta() -> f64 {
        100.0
    }

    /// Empty identity digest at the default δ. `n = 0`, `min = +∞`, `max = −∞`.
    /// This is the body reused by [`Accumulator::empty`].
    pub fn empty() -> Self {
        // default_delta() is always ≥ 1, so new cannot fail.
        TDigest::new(TDigest::default_delta()).expect("default_delta() >= 1")
    }

    /// Number of non-NaN observations seen.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Exact minimum observed (`+∞` when empty).
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Exact maximum observed (`−∞` when empty).
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Add one observation. NaN is silently omitted (no error, no count). The
    /// ingest buffer flushes and compresses when it reaches `buf_cap`. This is
    /// the body reused by [`Accumulator::update`].
    pub fn update(&mut self, x: f64) {
        if x.is_nan() {
            return;
        }
        self.ingest_buf.push(x);
        self.total_weight += 1.0;
        self.n += 1;
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
        if self.ingest_buf.len() >= self.buf_cap {
            self.flush();
        }
    }

    /// Drain `ingest_buf` into `centroids`: sort the buffer, turn each value into
    /// a unit-weight centroid, sorted-merge into the existing list, then compress.
    /// `total_weight` already includes the buffered values (added in `update`).
    fn flush(&mut self) {
        if self.ingest_buf.is_empty() {
            return;
        }
        self.ingest_buf
            .sort_unstable_by(|a, b| a.partial_cmp(b).expect("buffer is NaN-free"));
        let fresh: Vec<Centroid> = self
            .ingest_buf
            .iter()
            .map(|&v| Centroid { mean: v, weight: 1.0 })
            .collect();
        let mut all = merge_sorted_centroids(&self.centroids, &fresh);
        compress(&mut all, self.total_weight, self.delta);
        self.centroids = all;
        self.ingest_buf.clear();
    }

    /// Estimate the value at cumulative probability `q`.
    ///
    /// **Convention:** `q ∈ [0,1]`; `quantile(0)` returns the exact `min`,
    /// `quantile(1)` the exact `max`. Interior values use Dunning's linear
    /// interpolation on centroid midpoints (`pos[i] = Σ_{j<i} w[j] + w[i]/2`,
    /// interpolated at rank `r = q·total`). Approximate within the rank-error bound.
    ///
    /// # Errors
    /// `ProbabilityOutOfRange(q)` if `q ∉ [0,1]`; `EmptyInput` if no observations.
    pub fn quantile(&self, q: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(StatError::ProbabilityOutOfRange(q));
        }
        if self.n == 0 {
            return Err(StatError::EmptyInput);
        }
        if q == 0.0 {
            return Ok(self.min);
        }
        if q == 1.0 {
            return Ok(self.max);
        }
        let mut flushed = self.clone();
        flushed.flush();
        Ok(quantile_interior(
            &flushed.centroids,
            flushed.total_weight,
            flushed.min,
            flushed.max,
            q,
        ))
    }

    /// Estimate the cumulative probability `P(X ≤ x)`.
    ///
    /// **Convention:** 0 below the exact `min`, 1 above the exact `max`; interior
    /// is the inverse of [`TDigest::quantile`]. Returns NaN for an empty digest.
    pub fn cdf(&self, x: f64) -> f64 {
        let mut flushed = self.clone();
        flushed.flush();
        cdf_core(
            &flushed.centroids,
            flushed.total_weight,
            flushed.n,
            flushed.min,
            flushed.max,
            x,
        )
    }

    #[cfg(test)]
    pub(crate) fn default_delta_used(&self) -> f64 {
        self.delta
    }
    #[cfg(test)]
    pub(crate) fn flush_for_test(&mut self) {
        self.flush();
    }
    #[cfg(test)]
    pub(crate) fn total_weight_for_test(&self) -> f64 {
        self.total_weight
    }
    #[cfg(test)]
    pub(crate) fn centroid_len_for_test(&self) -> usize {
        self.centroids.len()
    }
}

impl Mergeable for TDigest {
    /// Associative, order-independent merge using the empty-absorb idiom. Both
    /// sides flush, their centroids are sorted-merged and recompressed (a pure
    /// fn of the merged list + total), and exact `min`/`max`/`n` combine.
    fn merge(&mut self, other: &Self) {
        debug_assert_eq!(self.delta, other.delta, "t-digest merge mixes deltas");
        if self.n == 0 {
            *self = other.clone();
            return;
        }
        if other.n == 0 {
            return;
        }
        self.flush();
        let mut rhs = other.clone();
        rhs.flush();
        let total = self.total_weight + rhs.total_weight;
        let mut all = merge_sorted_centroids(&self.centroids, &rhs.centroids);
        compress(&mut all, total, self.delta);
        self.centroids = all;
        self.total_weight = total;
        self.n += rhs.n;
        self.min = self.min.min(rhs.min);
        self.max = self.max.max(rhs.max);
    }
}

impl Accumulator for TDigest {
    type Item = f64;
    type Output = TDigestResult;

    fn empty() -> Self {
        TDigest::empty()
    }

    fn update(&mut self, x: f64) {
        TDigest::update(self, x);
    }

    /// Flush and snapshot into a queryable [`TDigestResult`].
    fn finalize(&self) -> TDigestResult {
        let mut flushed = self.clone();
        flushed.flush();
        TDigestResult {
            centroids: flushed.centroids,
            total_weight: flushed.total_weight,
            n: flushed.n,
            min: flushed.min,
            max: flushed.max,
        }
    }
}

impl TDigestResult {
    /// Estimate the value at cumulative probability `q`. Same convention/errors as
    /// [`TDigest::quantile`].
    ///
    /// # Errors
    /// `ProbabilityOutOfRange(q)` if `q ∉ [0,1]`; `EmptyInput` if empty.
    pub fn quantile(&self, q: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(StatError::ProbabilityOutOfRange(q));
        }
        if self.n == 0 {
            return Err(StatError::EmptyInput);
        }
        if q == 0.0 {
            return Ok(self.min);
        }
        if q == 1.0 {
            return Ok(self.max);
        }
        Ok(quantile_interior(
            &self.centroids,
            self.total_weight,
            self.min,
            self.max,
            q,
        ))
    }

    /// Estimate `P(X ≤ x)`. Same convention as [`TDigest::cdf`]; NaN when empty.
    pub fn cdf(&self, x: f64) -> f64 {
        cdf_core(&self.centroids, self.total_weight, self.n, self.min, self.max, x)
    }

    /// Number of non-NaN observations summarized.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Exact minimum (`+∞` when empty).
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Exact maximum (`−∞` when empty).
    pub fn max(&self) -> f64 {
        self.max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Task 1: skeleton ----

    #[test]
    fn new_rejects_bad_delta() {
        assert_eq!(TDigest::new(0.5).unwrap_err(), StatError::DomainError("t-digest delta must be >= 1"));
        assert_eq!(TDigest::new(0.0).unwrap_err(), StatError::DomainError("t-digest delta must be >= 1"));
        assert_eq!(TDigest::new(-3.0).unwrap_err(), StatError::DomainError("t-digest delta must be >= 1"));
        assert_eq!(TDigest::new(f64::NAN).unwrap_err(), StatError::DomainError("t-digest delta must be >= 1"));
        assert!(TDigest::new(1.0).is_ok());
        assert!(TDigest::new(100.0).is_ok());
    }

    #[test]
    fn empty_identity_fields() {
        let d = TDigest::empty();
        assert_eq!(d.count(), 0);
        assert_eq!(d.min(), f64::INFINITY);
        assert_eq!(d.max(), f64::NEG_INFINITY);
        assert_eq!(d.default_delta_used(), TDigest::default_delta());
    }

    #[test]
    fn default_delta_is_100() {
        assert_eq!(TDigest::default_delta(), 100.0);
    }

    // ---- Task 2: k1 ----

    #[test]
    fn k1_monotonic_and_bounded() {
        let delta = 100.0;
        // Maps [0,1] -> [-delta/4, +delta/4].
        assert!((k1(0.0, delta) - (-delta / 4.0)).abs() < 1e-9);
        assert!((k1(1.0, delta) - (delta / 4.0)).abs() < 1e-9);
        assert!((k1(0.5, delta)).abs() < 1e-9); // asin(0) = 0 at the median
        // Strictly increasing across a grid.
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=100 {
            let q = i as f64 / 100.0;
            let k = k1(q, delta);
            assert!(k > prev, "k1 not increasing at q={q}");
            assert!(k >= -delta / 4.0 - 1e-9 && k <= delta / 4.0 + 1e-9);
            prev = k;
        }
    }

    // ---- Task 3: ingestion ----

    #[test]
    fn ingest_single_value() {
        let mut d = TDigest::new(100.0).unwrap();
        d.update(7.0);
        d.flush_for_test();
        assert_eq!(d.count(), 1);
        assert_eq!(d.min(), 7.0);
        assert_eq!(d.max(), 7.0);
        assert_eq!(d.total_weight_for_test(), 1.0);
    }

    #[test]
    fn ingest_counts_and_extremes() {
        let mut d = TDigest::new(100.0).unwrap();
        for x in [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0] {
            d.update(x);
        }
        d.flush_for_test();
        assert_eq!(d.count(), 8);
        assert_eq!(d.min(), 1.0);
        assert_eq!(d.max(), 9.0);
        assert_eq!(d.total_weight_for_test(), 8.0);
    }

    #[test]
    fn ingest_omits_nan_without_error() {
        let mut d = TDigest::new(100.0).unwrap();
        d.update(2.0);
        d.update(f64::NAN);
        d.update(4.0);
        d.flush_for_test();
        assert_eq!(d.count(), 2);
        assert_eq!(d.min(), 2.0);
        assert_eq!(d.max(), 4.0);
    }

    #[test]
    fn ingest_duplicates() {
        let mut d = TDigest::new(100.0).unwrap();
        for _ in 0..1000 {
            d.update(5.0);
        }
        d.flush_for_test();
        assert_eq!(d.count(), 1000);
        assert_eq!(d.total_weight_for_test(), 1000.0);
        assert_eq!(d.min(), 5.0);
        assert_eq!(d.max(), 5.0);
    }

    #[test]
    fn compress_reduces_centroid_count() {
        // δ small + many distinct values => compression must shrink the list.
        let mut d = TDigest::new(10.0).unwrap();
        for i in 0..10_000 {
            d.update(i as f64);
        }
        d.flush_for_test();
        assert!(
            d.centroid_len_for_test() < 10_000,
            "compress did not merge centroids: {}",
            d.centroid_len_for_test()
        );
        // total weight is conserved exactly.
        assert_eq!(d.total_weight_for_test(), 10_000.0);
    }

    // ---- Task 4: query ----

    #[test]
    fn quantile_endpoints_exact() {
        let mut d = TDigest::new(100.0).unwrap();
        for x in [10.0, 20.0, 30.0, 40.0, 50.0] {
            d.update(x);
        }
        assert_eq!(d.quantile(0.0).unwrap(), 10.0); // exact min
        assert_eq!(d.quantile(1.0).unwrap(), 50.0); // exact max
        let median = d.quantile(0.5).unwrap();
        assert!((median - 30.0).abs() < 1.0, "median {median} off");
    }

    #[test]
    fn quantile_rejects_bad_q() {
        let mut d = TDigest::new(100.0).unwrap();
        d.update(1.0);
        assert_eq!(d.quantile(-0.1), Err(StatError::ProbabilityOutOfRange(-0.1)));
        assert_eq!(d.quantile(1.5), Err(StatError::ProbabilityOutOfRange(1.5)));
    }

    #[test]
    fn quantile_empty_errors() {
        let d = TDigest::empty();
        assert_eq!(d.quantile(0.5), Err(StatError::EmptyInput));
    }

    #[test]
    fn cdf_endpoints_and_empty() {
        let mut d = TDigest::new(100.0).unwrap();
        for x in [10.0, 20.0, 30.0, 40.0, 50.0] {
            d.update(x);
        }
        assert_eq!(d.cdf(5.0), 0.0);  // below min
        assert_eq!(d.cdf(99.0), 1.0); // above max
        assert!(TDigest::empty().cdf(0.0).is_nan());
    }

    #[test]
    fn cdf_monotone() {
        let mut d = TDigest::new(100.0).unwrap();
        for i in 0..500 {
            d.update((i as f64 * 0.123).sin().abs());
        }
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=200 {
            let x = -0.5 + 2.0 * (i as f64 / 200.0);
            let c = d.cdf(x);
            assert!(c >= prev - 1e-12, "cdf decreased at x={x}: {c} < {prev}");
            assert!((0.0..=1.0).contains(&c));
            prev = c;
        }
    }

    #[test]
    fn finalize_snapshot_matches_digest_queries() {
        let mut d = TDigest::new(100.0).unwrap();
        for i in 0..1000 {
            d.update((i as f64 * 0.317).sin().abs());
        }
        let snap = d.finalize();
        assert_eq!(snap.count(), d.count());
        assert_eq!(snap.min(), d.min());
        assert_eq!(snap.max(), d.max());
        for q in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let a = d.quantile(q).unwrap();
            let b = snap.quantile(q).unwrap();
            assert!((a - b).abs() < 1e-12, "q={q}: digest {a} vs snapshot {b}");
        }
    }

    // ---- Task 5: merge ----

    // Deterministic, rng-free data generator (sin-based; no `rng` feature).
    fn sample(n: usize) -> Vec<f64> {
        (0..n).map(|i| ((i as f64 + 1.0) * 0.6180339887).sin().abs() * 100.0).collect()
    }

    fn digest_of(xs: &[f64]) -> TDigest {
        let mut d = TDigest::new(100.0).unwrap();
        for &x in xs {
            d.update(x);
        }
        d
    }

    #[test]
    fn merge_empty_absorb() {
        let xs = sample(2000);
        let full = digest_of(&xs);
        // empty.merge(full) == full ; full.merge(empty) == full
        let mut left = TDigest::empty();
        left.merge(&full);
        let mut right = full.clone();
        right.merge(&TDigest::empty());
        for q in [0.1, 0.5, 0.9] {
            assert_eq!(left.quantile(q).unwrap(), full.quantile(q).unwrap());
            assert_eq!(right.quantile(q).unwrap(), full.quantile(q).unwrap());
        }
        assert_eq!(left.count(), full.count());
        assert_eq!(right.count(), full.count());
    }

    #[test]
    fn merge_associative_left_vs_right_fold() {
        let xs = sample(9000);
        let (a, b, c) = (&xs[0..3000], &xs[3000..6000], &xs[6000..9000]);
        let (da, db, dc) = (digest_of(a), digest_of(b), digest_of(c));

        // (A ⊕ B) ⊕ C
        let mut left = da.clone();
        left.merge(&db);
        left.merge(&dc);
        // A ⊕ (B ⊕ C)
        let mut bc = db.clone();
        bc.merge(&dc);
        let mut right = da.clone();
        right.merge(&bc);

        for q in [0.05, 0.25, 0.5, 0.75, 0.95] {
            let l = left.quantile(q).unwrap();
            let r = right.quantile(q).unwrap();
            // Associative within the approximation: ranks of the two estimates agree
            // to well under the 0.75% rank bound.
            let lr = left.cdf(l);
            let rr = right.cdf(r);
            assert!((lr - rr).abs() <= 0.0075, "q={q}: rank {lr} vs {rr}");
        }
        assert_eq!(left.count(), 9000);
        assert_eq!(right.count(), 9000);
        assert_eq!(left.min(), right.min());
        assert_eq!(left.max(), right.max());
    }

    #[test]
    fn merge_chunked_equals_streaming() {
        let xs = sample(8000);
        // streaming: one digest over all
        let streamed = digest_of(&xs);
        // chunked: 8 sub-digests merged
        let mut chunked = TDigest::empty();
        for chunk in xs.chunks(1000) {
            chunked.merge(&digest_of(chunk));
        }
        assert_eq!(chunked.count(), streamed.count());
        assert_eq!(chunked.min(), streamed.min());
        assert_eq!(chunked.max(), streamed.max());
        for q in [0.05, 0.25, 0.5, 0.75, 0.95] {
            let target = streamed.quantile(q).unwrap();
            // both estimate the same rank to within the bound
            assert!((chunked.cdf(target) - q).abs() <= 0.0075, "q={q}");
        }
    }

    // ---- Task 6: quantile_edges ----

    #[test]
    fn quantile_edges_shape_and_endpoints() {
        let xs = sample(5000);
        let d = digest_of(&xs);
        let edges = quantile_edges(&d, 8).unwrap();
        assert_eq!(edges.len(), 9); // k+1
        assert_eq!(edges[0], d.min());
        assert_eq!(edges[8], d.max());
        // monotonically non-decreasing
        for w in edges.windows(2) {
            assert!(w[1] >= w[0], "edges not sorted: {:?}", w);
        }
    }

    #[test]
    fn quantile_edges_k1_is_just_min_max() {
        let d = digest_of(&sample(100));
        let edges = quantile_edges(&d, 1).unwrap();
        assert_eq!(edges, vec![d.min(), d.max()]);
    }

    #[test]
    fn quantile_edges_rejects_k0_and_empty() {
        let d = digest_of(&sample(100));
        assert_eq!(
            quantile_edges(&d, 0),
            Err(StatError::DomainError("quantile_edges requires k >= 1"))
        );
        assert_eq!(
            quantile_edges(&TDigest::empty(), 4),
            Err(StatError::DomainError("quantile_edges on empty digest"))
        );
    }
}
