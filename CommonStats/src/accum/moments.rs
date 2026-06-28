//! Moment accumulators. Welford (update), Chan (Variance merge), Terriberry
//! (Moments M3/M4 merge), parallel co-moment (CoMoment merge).
use super::simple::Sum;
use super::{Accumulator, Mergeable};
use crate::error::StatError;
use crate::nan::{clean, NanPolicy};

/// Welford running mean. An accumulator-family primitive: batch `mean` currently
/// reads off Variance's two-pass path, so nothing consumes `Mean` directly yet.
#[derive(Clone, Default)]
pub struct Mean { pub(crate) n: u64, pub(crate) mean: f64 }
impl Mergeable for Mean {
    fn merge(&mut self, o: &Self) {
        if o.n == 0 { return; }
        let n = self.n + o.n;
        let delta = o.mean - self.mean;
        // Chan's parallel mean update; see Variance::merge for the moment algebra.
        self.mean += delta * (o.n as f64) / (n as f64);
        self.n = n;
    }
}
impl Accumulator for Mean {
    type Item = f64; type Output = f64;
    fn empty() -> Self { Self::default() }
    fn update(&mut self, x: f64) {
        self.n += 1;
        self.mean += (x - self.mean) / (self.n as f64);
    }
    fn finalize(&self) -> f64 { self.mean }
}

/// Carries (n, mean, M2). M2 = Σ(xᵢ - mean)². Chan's parallel formula for merge.
#[derive(Clone, Default)]
pub struct Variance { pub(crate) n: u64, pub(crate) mean: f64, pub(crate) m2: f64 }
impl Variance {
    /// Number of observations folded in.
    pub fn count(&self) -> u64 { self.n }
    /// Running arithmetic mean.
    pub fn mean(&self) -> f64 { self.mean }
    /// Population variance (ddof = 0). NaN when n = 0.
    pub fn var_pop(&self) -> f64 { if self.n == 0 { f64::NAN } else { self.m2 / self.n as f64 } }
    /// Sample variance (ddof = 1, Bessel-corrected). NaN when n < 2.
    pub fn var_sample(&self) -> f64 { if self.n < 2 { f64::NAN } else { self.m2 / (self.n - 1) as f64 } }
    /// Sample standard deviation (√ of the ddof = 1 variance).
    pub fn sd_sample(&self) -> f64 { self.var_sample().sqrt() }
    /// Two-pass batch override: more accurate than single-pass Welford when the
    /// whole vector is in hand. Skips NaN.
    pub fn from_slice_two_pass(xs: &[f64]) -> Self {
        let kept: Vec<f64> = xs.iter().copied().filter(|x| !x.is_nan()).collect();
        let n = kept.len() as u64;
        if n == 0 { return Self::default(); }
        let mut sx = Sum::empty();
        for &x in &kept { sx.update(x); }
        let mean = sx.finalize() / n as f64;
        let mut s2 = Sum::empty();
        for &x in &kept { s2.update((x - mean) * (x - mean)); }
        Self { n, mean, m2: s2.finalize() }
    }
}
impl Mergeable for Variance {
    fn merge(&mut self, o: &Self) {
        if o.n == 0 { return; }
        if self.n == 0 { *self = o.clone(); return; }
        let (na, nb) = (self.n as f64, o.n as f64);
        let n = na + nb;
        let delta = o.mean - self.mean;
        // Chan et al. 1979: M2_AB = M2_A + M2_B + δ²·nA·nB/(nA+nB), δ = meanB − meanA.
        self.m2 += o.m2 + delta * delta * na * nb / n;
        self.mean += delta * nb / n;
        self.n += o.n;
    }
}
impl Accumulator for Variance {
    type Item = f64; type Output = f64;
    fn empty() -> Self { Self::default() }
    fn update(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
    }
    fn finalize(&self) -> f64 { self.var_sample() }
}

/// Validated sample-variance accumulator: omit NaN, require n ≥ 2, two-pass. One
/// source of truth for the test/CI callers (ttest, effect, ci, anova).
pub(crate) fn checked_variance(xs: &[f64]) -> Result<Variance, StatError> {
    let v = clean(xs, NanPolicy::Omit)?;
    if v.len() < 2 { return Err(StatError::TooFewObservations { needed: 2, got: v.len() }); }
    Ok(Variance::from_slice_two_pass(&v))
}

/// Pooled sample variance sp² = [(nA−1)·vA + (nB−1)·vB] / (nA+nB−2) of two groups.
/// One source of truth for the equal-variance t-test, its CI, and Cohen's d.
pub(crate) fn pooled_var(sa: &Variance, sb: &Variance) -> f64 {
    let (na, nb) = (sa.count() as f64, sb.count() as f64);
    ((na - 1.0) * sa.var_sample() + (nb - 1.0) * sb.var_sample()) / (na + nb - 2.0)
}

/// Central moments to 4th order. Terriberry's parallel merge formulas.
#[derive(Clone, Default)]
pub struct Moments { pub(crate) n: u64, mean: f64, m2: f64, m3: f64, m4: f64 }
impl Moments {
    /// Number of observations folded in.
    pub fn count(&self) -> u64 { self.n }
    /// Running arithmetic mean.
    pub fn mean(&self) -> f64 { self.mean }
    /// Sample variance (ddof = 1). NaN when n < 2.
    pub fn var_sample(&self) -> f64 { if self.n < 2 { f64::NAN } else { self.m2 / (self.n - 1) as f64 } }
    /// Sample skewness, Fisher–Pearson population form g1 = √n·M3 / M2^{3/2} (no
    /// bias correction). Matches `scipy.stats.skew` (default `bias=True`).
    pub fn skewness(&self) -> f64 {
        let n = self.n as f64;
        (n).sqrt() * self.m3 / self.m2.powf(1.5)
    }
    /// Excess kurtosis, population form g2 = n·M4/M2² − 3 (no bias correction).
    /// Matches `scipy.stats.kurtosis` (default `bias=True`, Fisher).
    pub fn kurtosis_excess(&self) -> f64 {
        let n = self.n as f64;
        n * self.m4 / (self.m2 * self.m2) - 3.0
    }
}
impl Mergeable for Moments {
    fn merge(&mut self, o: &Self) {
        if o.n == 0 { return; }
        if self.n == 0 { *self = o.clone(); return; }
        let (na, nb) = (self.n as f64, o.n as f64);
        let n = na + nb;
        let d = o.mean - self.mean;
        // Terriberry's parallel extension of Chan to the 3rd/4th central moments;
        // δ = meanB − meanA, combining (n, mean, M2, M3, M4) pairwise.
        let d2 = d * d; let d3 = d2 * d; let d4 = d2 * d2;
        let m2 = self.m2 + o.m2 + d2 * na * nb / n;
        let m3 = self.m3 + o.m3
            + d3 * na * nb * (na - nb) / (n * n)
            + 3.0 * d * (na * o.m2 - nb * self.m2) / n;
        let m4 = self.m4 + o.m4
            + d4 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n)
            + 6.0 * d2 * (na * na * o.m2 + nb * nb * self.m2) / (n * n)
            + 4.0 * d * (na * o.m3 - nb * self.m3) / n;
        self.mean += d * nb / n;
        self.m2 = m2; self.m3 = m3; self.m4 = m4; self.n += o.n;
    }
}
impl Accumulator for Moments {
    type Item = f64; type Output = f64;
    fn empty() -> Self { Self::default() }
    fn update(&mut self, x: f64) {
        let n1 = self.n as f64;
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;
        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0)
            + 6.0 * delta_n2 * self.m2 - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }
    fn finalize(&self) -> f64 { self.kurtosis_excess() }
}

/// Co-moment for covariance / Pearson. Carries per-axis M2 plus the cross term.
#[derive(Clone, Default)]
pub struct CoMoment { n: u64, mean_x: f64, mean_y: f64, c2: f64, m2x: f64, m2y: f64 }
impl CoMoment {
    /// Number of paired observations folded in.
    pub fn count(&self) -> u64 { self.n }
    /// Sample covariance (ddof = 1). NaN when n < 2.
    pub fn covariance_sample(&self) -> f64 { if self.n < 2 { f64::NAN } else { self.c2 / (self.n - 1) as f64 } }
    /// Pearson product-moment correlation r = C2 / √(M2x·M2y).
    pub fn pearson(&self) -> f64 { self.c2 / (self.m2x * self.m2y).sqrt() }
}
impl Mergeable for CoMoment {
    fn merge(&mut self, o: &Self) {
        if o.n == 0 { return; }
        if self.n == 0 { *self = o.clone(); return; }
        let (na, nb) = (self.n as f64, o.n as f64);
        let n = na + nb;
        let dx = o.mean_x - self.mean_x;
        let dy = o.mean_y - self.mean_y;
        // Chan's parallel formula on the cross term: C2_AB = C2_A + C2_B +
        // δx·δy·nA·nB/(nA+nB); the per-axis M2 follow the scalar Variance rule.
        self.c2 += o.c2 + dx * dy * na * nb / n;
        self.m2x += o.m2x + dx * dx * na * nb / n;
        self.m2y += o.m2y + dy * dy * na * nb / n;
        self.mean_x += dx * nb / n;
        self.mean_y += dy * nb / n;
        self.n += o.n;
    }
}
impl Accumulator for CoMoment {
    type Item = (f64, f64); type Output = f64;
    fn empty() -> Self { Self::default() }
    fn update(&mut self, (x, y): (f64, f64)) {
        self.n += 1;
        let n = self.n as f64;
        let dx = x - self.mean_x;
        let dy = y - self.mean_y;
        self.mean_x += dx / n;
        self.mean_y += dy / n;
        self.c2 += dx * (y - self.mean_y);
        self.m2x += dx * (x - self.mean_x);
        self.m2y += dy * (y - self.mean_y);
    }
    fn finalize(&self) -> f64 { self.pearson() }
}

/// Fold paired `a`, `b` into a [`CoMoment`], dropping any pair with a NaN in
/// either coordinate (the Omit policy on pairs). Errors on length mismatch.
/// Callers apply their own minimum-count requirement to the result.
pub(crate) fn comoment_pairs(a: &[f64], b: &[f64]) -> Result<CoMoment, StatError> {
    if a.len() != b.len() { return Err(StatError::MismatchedLengths { a: a.len(), b: b.len() }); }
    let mut c = CoMoment::empty();
    for (&x, &y) in a.iter().zip(b) {
        if !x.is_nan() && !y.is_nan() { c.update((x, y)); }
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accum::from_slice;

    #[test]
    fn mean_basic() {
        let m: Mean = from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert!((m.finalize() - 2.5).abs() < 1e-15);
    }
    #[test]
    fn variance_sample_matches_known() {
        // var([2,4,4,4,5,5,7,9]) sample = 32/7, pop = 4.0
        let v: Variance = from_slice(&[2.,4.,4.,4.,5.,5.,7.,9.]);
        assert!((v.var_pop() - 4.0).abs() < 1e-13);
        assert!((v.var_sample() - 32.0/7.0).abs() < 1e-13);
    }
    #[test]
    fn variance_merge_equals_single() {
        let whole: Variance = from_slice(&[2.,4.,4.,4.,5.,5.,7.,9.]);
        let mut a: Variance = from_slice(&[2.,4.,4.,4.]);
        let b: Variance = from_slice(&[5.,5.,7.,9.]);
        a.merge(&b);
        assert!((a.var_sample() - whole.var_sample()).abs() < 1e-12);
        assert_eq!(a.count(), 8);
    }
    #[test]
    fn two_pass_agrees_with_streaming() {
        let xs = [1e8, 1e8 + 1.0, 1e8 + 2.0];
        let tp = Variance::from_slice_two_pass(&xs);
        let st: Variance = from_slice(&xs);
        assert!((tp.var_sample() - st.var_sample()).abs() < 1e-6);
        assert!((tp.var_sample() - 1.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod more_tests {
    use super::*;
    use crate::accum::from_slice;

    #[test]
    fn skew_kurtosis_known() {
        let m: Moments = from_slice(&[1.,2.,3.,4.,5.]);
        assert!(m.skewness().abs() < 1e-12);
        assert!((m.kurtosis_excess() - (-1.3)).abs() < 1e-9); // SciPy kurtosis([1..5])=-1.3
        // Non-symmetric g1 distinguishes the Fisher–Pearson population form (g1) from
        // the bias-adjusted G1: scipy.stats.skew([2,4,4,4,5,5,7,9], bias=True) = 0.65625.
        let asy: Moments = from_slice(&[2., 4., 4., 4., 5., 5., 7., 9.]);
        assert!((asy.skewness() - 0.65625).abs() < 1e-12, "skew {}", asy.skewness());
    }
    #[test]
    fn moments_merge_equals_single() {
        let whole: Moments = from_slice(&[1.,2.,3.,4.,5.,6.,7.,8.]);
        let mut a: Moments = from_slice(&[1.,2.,3.,4.]);
        let b: Moments = from_slice(&[5.,6.,7.,8.]);
        a.merge(&b);
        assert!((a.kurtosis_excess() - whole.kurtosis_excess()).abs() < 1e-10);
        assert!((a.skewness() - whole.skewness()).abs() < 1e-10);
    }
    #[test]
    fn pearson_perfect_correlation() {
        let mut c = CoMoment::empty();
        for (x, y) in [(1.,2.), (2.,4.), (3.,6.), (4.,8.)] { c.update((x, y)); }
        assert!((c.pearson() - 1.0).abs() < 1e-13);
        assert!((c.covariance_sample() - 10.0/3.0).abs() < 1e-12);
    }
    #[test]
    fn comoment_merge_equals_single() {
        let pairs = [(1.,2.),(2.,1.),(3.,5.),(4.,4.),(5.,9.),(6.,7.)];
        let mut whole = CoMoment::empty();
        for &p in &pairs { whole.update(p); }
        let mut a = CoMoment::empty();
        for &p in &pairs[..3] { a.update(p); }
        let mut b = CoMoment::empty();
        for &p in &pairs[3..] { b.update(p); }
        a.merge(&b);
        assert!((a.pearson() - whole.pearson()).abs() < 1e-12);
    }
}
