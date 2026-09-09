//! Fixed-bin histogram + fixed-grid ECDF. Plain mergeable accumulators with no
//! RNG dependency, so they live in the always-compiled base.
use super::{Accumulator, Mergeable};
use crate::error::StatError;
use alloc::vec;
use alloc::vec::Vec;

/// Finalized histogram: per-bin counts, the `n_bins + 1` bin edges, and the
/// out-of-range tallies.
#[derive(Debug, Clone, PartialEq)]
pub struct HistResult {
    /// In-range count per bin; `counts.len() == n_bins`.
    pub counts: Vec<u64>,
    /// Bin edges, ascending, length `n_bins + 1` (`edges[0] == lo`, last `== hi`).
    pub edges: Vec<f64>,
    /// Number of observations below `lo` (numpy.histogram drops these).
    pub underflow: u64,
    /// Number of observations above `hi` (numpy.histogram drops these).
    pub overflow: u64,
}

/// Fixed-bin histogram accumulator over `n_bins` equal-width bins on `[lo, hi]`.
///
/// Convention: bins are half-open `[eᵢ, eᵢ₊₁)` except the last, `[e_{n-1}, hi]`,
/// which is closed — so `x == hi` lands in the final bin. `Matches
/// numpy.histogram(x, bins=n_bins, range=(lo, hi))` for in-range values. Values
/// `< lo` and `> hi` are tallied as under/overflow (numpy silently drops them);
/// `NaN` is omitted (crate default policy).
///
/// Mergeable: `merge` adds per-bin counts (the accumulator monoid). `finalize`
/// returns a [`HistResult`]; [`Histogram::ecdf`] reads off the fixed-grid ECDF
/// from the same counts.
///
/// ```
/// use commonstats::Histogram;
/// use commonstats::accum::Accumulator; // brings `update` into scope
/// let mut h = Histogram::new(0.0, 2.0, 4).unwrap(); // edges 0,0.5,1,1.5,2
/// for x in [0.0, 0.5, 1.0, 1.0, 1.5, 2.0] { h.update(x); }
/// assert_eq!(h.counts(), &[1, 1, 2, 2]);   // x==hi lands in the closed last bin
/// assert_eq!(h.ecdf().last(), Some(&1.0)); // cumulative ECDF ends at 1.0
/// ```
#[derive(Debug, Clone)]
pub struct Histogram {
    lo: f64,
    hi: f64,
    n_bins: usize, // 0 marks the unconfigured monoid identity (empty())
    counts: Vec<u64>,
    underflow: u64,
    overflow: u64,
}

impl Histogram {
    /// Build a histogram of `n_bins` equal-width bins spanning `[lo, hi]`.
    ///
    /// `lo`/`hi`: the (inclusive) range; require `lo < hi`. `n_bins`: number of
    /// bins; require `≥ 1`. Returns [`StatError::DomainError`] otherwise.
    pub fn new(lo: f64, hi: f64, n_bins: usize) -> Result<Self, StatError> {
        // NaN bounds are rejected explicitly: `lo >= hi` alone is false for NaN.
        if lo.is_nan() || hi.is_nan() || lo >= hi {
            return Err(StatError::DomainError("histogram: need lo < hi"));
        }
        if n_bins == 0 {
            return Err(StatError::DomainError("histogram: need n_bins >= 1"));
        }
        Ok(Self {
            lo,
            hi,
            n_bins,
            counts: vec![0; n_bins],
            underflow: 0,
            overflow: 0,
        })
    }

    #[inline]
    fn bin_width(&self) -> f64 {
        (self.hi - self.lo) / self.n_bins as f64
    }

    /// The `n_bins + 1` bin edges, ascending (`edges[0] == lo`, last `== hi`).
    pub fn edges(&self) -> Vec<f64> {
        let w = self.bin_width();
        (0..=self.n_bins).map(|i| self.lo + i as f64 * w).collect()
    }

    /// In-range per-bin counts (`len == n_bins`).
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Observations seen below `lo`.
    pub fn underflow(&self) -> u64 {
        self.underflow
    }

    /// Observations seen above `hi`.
    pub fn overflow(&self) -> u64 {
        self.overflow
    }

    /// Fixed-grid empirical CDF: the cumulative-normalized histogram. `ecdf()[i]`
    /// is the fraction of in-range observations `≤ edges[i+1]` (so the last entry
    /// is `1.0`); `len == n_bins`. Normalized by the in-range total — out-of-range
    /// observations are excluded. This is the *fixed-grid* ECDF; an exact
    /// jump-at-every-datum ECDF would need a sort/rank path (not yet implemented).
    /// Empty in-range total → all `NaN`.
    pub fn ecdf(&self) -> Vec<f64> {
        let total: u64 = self.counts.iter().sum();
        if total == 0 {
            return vec![f64::NAN; self.n_bins];
        }
        let total = total as f64;
        let mut acc = 0u64;
        self.counts
            .iter()
            .map(|&c| {
                acc += c;
                acc as f64 / total
            })
            .collect()
    }
}

impl Mergeable for Histogram {
    fn merge(&mut self, o: &Self) {
        // Absorb the configured operand when self is the empty identity —
        // mirrors `Variance::merge`'s n==0 absorb.
        if self.n_bins == 0 {
            *self = o.clone();
            return;
        }
        debug_assert_eq!(
            (self.lo, self.hi, self.n_bins),
            (o.lo, o.hi, o.n_bins),
            "merging histograms with mismatched bins"
        );
        for (c, &oc) in self.counts.iter_mut().zip(o.counts.iter()) {
            *c += oc;
        }
        self.underflow += o.underflow;
        self.overflow += o.overflow;
    }
}

impl Accumulator for Histogram {
    type Item = f64;
    type Output = HistResult;
    /// The monoid identity (no bins). The real constructor is [`Histogram::new`];
    /// `merge` absorbs the configured operand's binning.
    fn empty() -> Self {
        Self {
            lo: 0.0,
            hi: 0.0,
            n_bins: 0,
            counts: Vec::new(),
            underflow: 0,
            overflow: 0,
        }
    }
    fn update(&mut self, x: f64) {
        if x.is_nan() {
            return; // omit policy
        }
        if x < self.lo {
            self.underflow += 1;
        } else if x > self.hi {
            self.overflow += 1;
        } else {
            // x in [lo, hi]; x == hi maps past the last edge, clamp into the
            // closed final bin (numpy.histogram's right-edge convention).
            let b = ((x - self.lo) / self.bin_width()) as usize;
            self.counts[b.min(self.n_bins - 1)] += 1;
        }
    }
    fn finalize(&self) -> HistResult {
        HistResult {
            counts: self.counts.clone(),
            edges: self.edges(),
            underflow: self.underflow,
            overflow: self.overflow,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_bad_args() {
        assert!(Histogram::new(1.0, 1.0, 4).is_err());
        assert!(Histogram::new(2.0, 1.0, 4).is_err());
        assert!(Histogram::new(0.0, 1.0, 0).is_err());
    }

    // Edges [0,0.5,1,1.5,2]. In-range binning matches
    // numpy.histogram([0,0.5,1,1,1.5,2], bins=4, range=(0,2)) == [1,1,2,2];
    // x==hi (2.0) falls in the closed last bin; -1 and 3 are out of range.
    #[test]
    fn matches_numpy_histogram() {
        let mut h = Histogram::new(0.0, 2.0, 4).unwrap();
        for x in [-1.0, 0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 3.0] {
            h.update(x);
        }
        assert_eq!(h.counts(), &[1, 1, 2, 2]);
        assert_eq!(h.underflow(), 1);
        assert_eq!(h.overflow(), 1);
        let r = h.finalize();
        assert_eq!(r.edges, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
        assert_eq!(r.counts, vec![1, 1, 2, 2]);
    }

    #[test]
    fn nan_is_omitted() {
        let mut h = Histogram::new(0.0, 1.0, 2).unwrap();
        h.update(f64::NAN);
        h.update(0.5);
        assert_eq!(h.counts(), &[0, 1]);
        assert_eq!(h.underflow() + h.overflow(), 0);
    }

    // ECDF = cumsum(counts)/total over the in-range grid; last entry is 1.0.
    #[test]
    fn ecdf_is_cumulative_normalized() {
        let mut h = Histogram::new(0.0, 2.0, 4).unwrap();
        for x in [0.0, 0.5, 1.0, 1.0, 1.5, 2.0] {
            h.update(x); // counts [1,1,2,2], total 6
        }
        let e = h.ecdf();
        assert!((e[0] - 1.0 / 6.0).abs() < 1e-15);
        assert!((e[1] - 2.0 / 6.0).abs() < 1e-15);
        assert!((e[2] - 4.0 / 6.0).abs() < 1e-15);
        assert!((e[3] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn ecdf_empty_is_nan() {
        let h = Histogram::new(0.0, 1.0, 3).unwrap();
        assert!(h.ecdf().iter().all(|x| x.is_nan()));
    }

    #[test]
    fn merge_adds_counts() {
        let mk = |xs: &[f64]| {
            let mut h = Histogram::new(0.0, 2.0, 4).unwrap();
            for &x in xs {
                h.update(x);
            }
            h
        };
        let whole = mk(&[0.0, 0.5, 1.0, 1.0, 1.5, 2.0, -1.0, 3.0]);
        let mut a = mk(&[0.0, 0.5, 1.0, -1.0]);
        let b = mk(&[1.0, 1.5, 2.0, 3.0]);
        a.merge(&b);
        assert_eq!(a.counts(), whole.counts());
        assert_eq!(a.underflow(), whole.underflow());
        assert_eq!(a.overflow(), whole.overflow());
    }

    #[test]
    fn empty_merge_is_identity() {
        let mut acc = Histogram::empty();
        let mut h = Histogram::new(0.0, 1.0, 2).unwrap();
        h.update(0.25);
        h.update(0.75);
        acc.merge(&h);
        assert_eq!(acc.counts(), h.counts());
    }

    #[test]
    fn bins_partition_the_range() {
        let mut h = Histogram::new(0.0, 10.0, 5).unwrap();
        for x in [1.0, 3.0, 3.0, 7.0, 9.5] {
            h.update(x);
        }
        assert_eq!(h.counts(), &[1, 2, 0, 1, 1]);
    }
}
