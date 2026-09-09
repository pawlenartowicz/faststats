//! Mergeable distribution accumulators over per-draw scalar statistics: the
//! p-value counter ([`NullDist`]) and the percentile-CI store ([`BootDist`]).
//! Both are `Accumulator<Item = f64>`, both monoidal with no downdate.
use crate::accum::{Accumulator, Mergeable, quantile_sorted};
use crate::htest::Ci;
use alloc::vec::Vec;

/// Which tail counts as "more extreme than the observed statistic" for a
/// permutation p-value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sidedness {
    /// Lower tail: a draw is extreme when `draw ≤ observed`.
    Less,
    /// Upper tail: a draw is extreme when `draw ≥ observed`.
    Greater,
    /// Both tails: a draw is extreme when `|draw| ≥ |observed|`. Assumes the
    /// statistic is centred at zero under the null (true for difference /
    /// contrast statistics, the usual permutation-test inputs).
    TwoSided,
}

/// Monte-Carlo permutation p-value accumulator.
///
/// Folds in per-draw statistic values and counts those as-or-more-extreme than
/// the observed statistic (per [`Sidedness`]). Convention: the **add-one**
/// estimator `p = (#extreme + 1) / (B + 1)` — never returns 0, the correct
/// unbiased Monte-Carlo permutation p (Davison & Hinkley 1997; Phipson & Smyth
/// 2010). The observed statistic is the same injected `statistic` evaluated on
/// the identity (unpermuted) index, computed once by the consumer before the
/// loop. Validated by exact full-enumeration p-values on tiny samples.
///
/// `finalize()` returns the p-value; an empty accumulator (`B = 0`) yields
/// `1/1 = 1.0`.
#[derive(Debug, Clone)]
pub struct NullDist {
    observed: f64, // NaN marks the unconfigured monoid identity (empty())
    side: Sidedness,
    count: u64, // draws as-or-more-extreme than `observed`
    b: u64,     // draws folded in
}

impl NullDist {
    /// Build a p-value accumulator for `observed` (the statistic on the
    /// unpermuted data) against the alternative `side`.
    pub fn new(observed: f64, side: Sidedness) -> Self {
        Self {
            observed,
            side,
            count: 0,
            b: 0,
        }
    }
}

impl Mergeable for NullDist {
    fn merge(&mut self, o: &Self) {
        // Adopt the configured operand's setup when self is the empty identity —
        // mirrors `Variance::merge`'s n==0 absorb.
        if self.observed.is_nan() {
            self.observed = o.observed;
            self.side = o.side;
        }
        self.count += o.count;
        self.b += o.b;
    }
}

impl Accumulator for NullDist {
    type Item = f64;
    type Output = f64;
    /// The monoid identity. Carries no test configuration; the real constructor
    /// is [`NullDist::new`]. `merge` adopts the configured operand's `observed`/
    /// `side`, so `empty().merge(&configured)` is the configured accumulator.
    fn empty() -> Self {
        Self {
            observed: f64::NAN,
            side: Sidedness::TwoSided,
            count: 0,
            b: 0,
        }
    }
    fn update(&mut self, x: f64) {
        self.b += 1;
        let extreme = match self.side {
            Sidedness::Less => x <= self.observed,
            Sidedness::Greater => x >= self.observed,
            Sidedness::TwoSided => x.abs() >= self.observed.abs(),
        };
        if extreme {
            self.count += 1;
        }
    }
    fn finalize(&self) -> f64 {
        (self.count as f64 + 1.0) / (self.b as f64 + 1.0)
    }
}

/// Percentile-bootstrap CI accumulator.
///
/// Store-and-sort (decision 2026-06-21): holds every per-draw statistic; `merge`
/// concatenates; `finalize` sorts and reads off percentiles. Costs `O(B)`
/// **scalars**, not `O(B·n)` data copies, and gives an *exact* percentile CI; a
/// future t-digest backend is a drop-in `O(1)` swap behind the same `finalize` — see [`crate::accum::TDigest`].
///
/// Convention: percentile-bootstrap CI at confidence `level` (e.g. `0.95` →
/// the 2.5th/97.5th percentiles). Percentiles use linear interpolation between
/// order statistics — `Matches numpy.percentile(..., method="linear")` (its
/// default; Hyndman–Fan type 7). Validated by feeding known statistic-value sets
/// directly.
///
/// `finalize()` returns a [`Ci`]; an empty accumulator yields a NaN-filled `Ci`.
#[derive(Debug, Clone)]
pub struct BootDist {
    stats: Vec<f64>,
    level: f64, // NaN marks the unconfigured monoid identity (empty())
}

impl BootDist {
    /// Build a bootstrap-CI accumulator at confidence `level` (in `(0, 1)`, e.g.
    /// `0.95`).
    pub fn new(level: f64) -> Self {
        Self {
            stats: Vec::new(),
            level,
        }
    }
}

impl Mergeable for BootDist {
    fn merge(&mut self, o: &Self) {
        if self.level.is_nan() {
            self.level = o.level;
        }
        // Concatenate in draw-block order — keeps finalize order-independent
        // (a sort follows) while respecting the ordered-merge preference.
        self.stats.extend_from_slice(&o.stats);
    }
}

impl Accumulator for BootDist {
    type Item = f64;
    type Output = Ci;
    /// The monoid identity (empty store, unconfigured level). Real constructor is
    /// [`BootDist::new`]; `merge` adopts the configured operand's `level`.
    fn empty() -> Self {
        Self {
            stats: Vec::new(),
            level: f64::NAN,
        }
    }
    fn update(&mut self, x: f64) {
        self.stats.push(x);
    }
    fn finalize(&self) -> Ci {
        let mut sorted = self.stats.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let alpha = 1.0 - self.level;
        Ci {
            lower: quantile_sorted(&sorted, alpha / 2.0),
            upper: quantile_sorted(&sorted, 1.0 - alpha / 2.0),
            level: self.level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(side: Sidedness, observed: f64, xs: &[f64]) -> f64 {
        let mut d = NullDist::new(observed, side);
        for &x in xs {
            d.update(x);
        }
        d.finalize()
    }

    // Add-one convention, all three sidednesses, hand-counted.
    #[test]
    fn nulldist_add_one_convention() {
        let xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]; // B = 6
        // observed = 2.0
        // Greater: {2,3} extreme → (2+1)/7
        assert!((feed(Sidedness::Greater, 2.0, &xs) - 3.0 / 7.0).abs() < 1e-15);
        // Less: {-2,-1,0,1,2} extreme → (5+1)/7
        assert!((feed(Sidedness::Less, 2.0, &xs) - 6.0 / 7.0).abs() < 1e-15);
        // TwoSided |x|>=2: {-2,2,3} extreme → (3+1)/7
        assert!((feed(Sidedness::TwoSided, 2.0, &xs) - 4.0 / 7.0).abs() < 1e-15);
    }

    #[test]
    fn nulldist_never_returns_zero() {
        // No draw beats observed yet p = 1/(B+1) > 0, never 0.
        let p = feed(Sidedness::Greater, 100.0, &[1.0, 2.0, 3.0]);
        assert!((p - 1.0 / 4.0).abs() < 1e-15);
    }

    // Exact two-sided permutation p by full enumeration of a tiny two-sample
    // mean-difference test, fed value-by-value into NullDist. Groups A={1,2},
    // B={3,4}: the 6 partitions of {1,2,3,4} into two pairs give mean differences
    // (meanA - meanB): {-2,-1,-1,1,1,2}. Observed = -2. |x|>=2 → {-2,2} extreme.
    // Add-one p = (2+1)/(6+1) = 3/7.
    #[test]
    fn nulldist_matches_full_enumeration() {
        let diffs = [-2.0, -1.0, -1.0, 1.0, 1.0, 2.0];
        assert!((feed(Sidedness::TwoSided, -2.0, &diffs) - 3.0 / 7.0).abs() < 1e-15);
    }

    #[test]
    fn nulldist_merge_equals_single() {
        let xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let whole = feed(Sidedness::Greater, 1.5, &xs);
        let mut a = NullDist::new(1.5, Sidedness::Greater);
        for &x in &xs[..3] {
            a.update(x);
        }
        let mut b = NullDist::new(1.5, Sidedness::Greater);
        for &x in &xs[3..] {
            b.update(x);
        }
        a.merge(&b);
        assert!((a.finalize() - whole).abs() < 1e-15);
    }

    #[test]
    fn nulldist_empty_merge_is_identity() {
        let mut acc = NullDist::empty();
        let mut configured = NullDist::new(1.0, Sidedness::Greater);
        for x in [0.0, 1.0, 2.0] {
            configured.update(x);
        }
        acc.merge(&configured);
        assert!((acc.finalize() - configured.finalize()).abs() < 1e-15);
    }

    // numpy.percentile(arange(101), [2.5, 97.5]) == [2.5, 97.5] exactly: pos =
    // 0.025*100 = 2.5 interpolates 2↔3 → 2.5; symmetric at the top.
    #[test]
    fn bootdist_matches_numpy_percentile_clean() {
        let mut d = BootDist::new(0.95);
        for i in 0..=100 {
            d.update(i as f64);
        }
        let ci = d.finalize();
        assert!((ci.lower - 2.5).abs() < 1e-12, "lower {}", ci.lower);
        assert!((ci.upper - 97.5).abs() < 1e-12, "upper {}", ci.upper);
        assert_eq!(ci.level, 0.95);
    }

    // Hand-computed linear interpolation on a small unsorted set.
    // sorted = [1,2,4,8,16], m=5. 90% CI → alpha=0.10, q_lo=0.05, q_hi=0.95.
    // lo: pos=0.05*4=0.2 → 1 + (2-1)*0.2 = 1.2. hi: pos=0.95*4=3.8 → 8 + (16-8)*0.8 = 14.4.
    #[test]
    fn bootdist_linear_interpolation_small() {
        let mut d = BootDist::new(0.90);
        for &x in &[8.0, 1.0, 16.0, 2.0, 4.0] {
            d.update(x);
        }
        let ci = d.finalize();
        assert!((ci.lower - 1.2).abs() < 1e-12, "lower {}", ci.lower);
        assert!((ci.upper - 14.4).abs() < 1e-12, "upper {}", ci.upper);
    }

    #[test]
    fn bootdist_merge_concatenates() {
        let mut whole = BootDist::new(0.95);
        for i in 0..=100 {
            whole.update(i as f64);
        }
        let mut a = BootDist::new(0.95);
        for i in 0..=50 {
            a.update(i as f64);
        }
        let mut b = BootDist::new(0.95);
        for i in 51..=100 {
            b.update(i as f64);
        }
        a.merge(&b);
        let (ca, cw) = (a.finalize(), whole.finalize());
        assert!((ca.lower - cw.lower).abs() < 1e-12);
        assert!((ca.upper - cw.upper).abs() < 1e-12);
    }
}
