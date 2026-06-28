//! The 6 discrete distributions of the `dist` suite.

use crate::error::StatError;
use crate::dist::{Bound, DiscreteCdf, DiscreteMass, Distribution};

/// `ln C(n, k) = lnΓ(n+1) − lnΓ(k+1) − lnΓ(n−k+1)`.
pub(crate) fn ln_choose(n: f64, k: f64) -> f64 {
    crate::special::lgamma(n + 1.0)
        - crate::special::lgamma(k + 1.0)
        - crate::special::lgamma(n - k + 1.0)
}

/// Binary search for the smallest integer `k ∈ [lo, hi]` with `cdf(k) ≥ p`.
/// `p ∉ [0,1]` → `ProbabilityOutOfRange`. Used by every discrete `quantile`.
pub(crate) fn discrete_bsearch_quantile<D: DiscreteCdf + ?Sized>(
    d: &D,
    p: f64,
    lo: i64,
    hi: i64,
) -> Result<i64, StatError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(StatError::ProbabilityOutOfRange(p));
    }
    if p == 0.0 { return Ok(lo); }
    let (mut a, mut b) = (lo, hi);
    while a < b {
        let mid = a + (b - a) / 2;
        if d.cdf(mid) < p { a = mid + 1; } else { b = mid; }
    }
    Ok(a)
}

/// Bernoulli distribution with success probability `p`.
///
/// Convention: support `{0, 1}`, `mass(1) = p`. `0·ln0 := 0` so `p ∈ {0, 1}`
/// give finite `log_mass` (no NaN). Matches `scipy.stats.bernoulli(p)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bernoulli { p: f64 }

impl Bernoulli {
    /// Construct with `p ∈ [0, 1]`.
    ///
    /// # Errors
    /// `ProbabilityOutOfRange(p)` if `p ∉ [0, 1]` or non-finite.
    ///
    /// ```
    /// use commonstats::dist::discrete::Bernoulli;
    /// use commonstats::dist::DiscreteCdf;
    /// let b = Bernoulli::new(0.3).unwrap();
    /// assert_eq!(b.quantile(0.8).unwrap(), 1);
    /// assert!(Bernoulli::new(-0.1).is_err());
    /// ```
    pub fn new(p: f64) -> Result<Self, StatError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        Ok(Bernoulli { p })
    }
}

impl Distribution for Bernoulli {
    fn support_min(&self) -> Bound { Bound::Finite(0.0) }
    fn support_max(&self) -> Bound { Bound::Finite(1.0) }
    fn mean(&self) -> Option<f64> { Some(self.p) }
    fn variance(&self) -> Option<f64> { Some(self.p * (1.0 - self.p)) }
}

impl DiscreteMass for Bernoulli {
    fn mass(&self, k: i64) -> f64 {
        match k { 0 => 1.0 - self.p, 1 => self.p, _ => 0.0 }
    }
    fn log_mass(&self, k: i64) -> f64 {
        // 0·ln0 := 0 handled by mapping the zero-prob branch to ln of the value.
        match k {
            0 => if self.p >= 1.0 { f64::NEG_INFINITY } else { (1.0 - self.p).ln() },
            1 => if self.p <= 0.0 { f64::NEG_INFINITY } else { self.p.ln() },
            _ => f64::NEG_INFINITY,
        }
    }
}

impl DiscreteCdf for Bernoulli {
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 { 0.0 } else if k == 0 { 1.0 - self.p } else { 1.0 }
    }
    fn quantile(&self, p: f64) -> Result<i64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        Ok(if p <= 1.0 - self.p { 0 } else { 1 })
    }
}

/// Binomial distribution `Binom(n, p)`.
///
/// Convention: `n ≥ 1` trials, success prob `p ∈ [0, 1]`, support `{0..n}`.
/// CDF `P(X ≤ k) = I_{1−p}(n−k, k+1)` (regularized incomplete beta). Matches
/// `scipy.stats.binom(n, p)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Binomial { n: i64, p: f64 }

impl Binomial {
    /// Construct with `n ≥ 1`, `p ∈ [0, 1]`.
    ///
    /// # Errors
    /// `DomainError` if `n < 1`; `ProbabilityOutOfRange(p)` if `p ∉ [0, 1]`.
    ///
    /// ```
    /// use commonstats::dist::discrete::Binomial;
    /// use commonstats::dist::DiscreteMass;
    /// let b = Binomial::new(10, 0.3).unwrap();
    /// assert!((b.mass(3) - 0.26682793200000005).abs() < 1e-12);
    /// assert!(Binomial::new(0, 0.3).is_err());
    /// ```
    pub fn new(n: i64, p: f64) -> Result<Self, StatError> {
        if n < 1 {
            return Err(StatError::DomainError("Binomial: n must be ≥ 1"));
        }
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        Ok(Binomial { n, p })
    }
}

impl Distribution for Binomial {
    fn support_min(&self) -> Bound { Bound::Finite(0.0) }
    fn support_max(&self) -> Bound { Bound::Finite(self.n as f64) }
    fn mean(&self) -> Option<f64> { Some(self.n as f64 * self.p) }
    fn variance(&self) -> Option<f64> { Some(self.n as f64 * self.p * (1.0 - self.p)) }
}

impl DiscreteMass for Binomial {
    fn mass(&self, k: i64) -> f64 {
        if k < 0 || k > self.n { return 0.0; }
        self.log_mass(k).exp()
    }
    fn log_mass(&self, k: i64) -> f64 {
        if k < 0 || k > self.n { return f64::NEG_INFINITY; }
        let (kf, nf) = (k as f64, self.n as f64);
        // p=0 / p=1 edges: only k=0 / k=n have mass; ln of the rest is −∞.
        let lp = if self.p <= 0.0 { if k == 0 { return 0.0; } f64::NEG_INFINITY } else { self.p.ln() };
        let lq = if self.p >= 1.0 { if k == self.n { return 0.0; } f64::NEG_INFINITY } else { (1.0 - self.p).ln() };
        ln_choose(nf, kf) + kf * lp + (nf - kf) * lq
    }
}

impl DiscreteCdf for Binomial {
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 { return 0.0; }
        if k >= self.n { return 1.0; }
        // I_{1-p}(n-k, k+1) = P(X ≤ k).
        crate::special::betai((self.n - k) as f64, (k + 1) as f64, 1.0 - self.p)
    }
    fn quantile(&self, p: f64) -> Result<i64, StatError> {
        discrete_bsearch_quantile(self, p, 0, self.n)
    }
}

/// Poisson distribution with rate `λ`.
///
/// Convention: support `{0, 1, …}`, `mass(k) = e^{−λ}λ^k/k!`. CDF
/// `P(X ≤ k) = Q(k+1, λ)` (upper regularized incomplete gamma). Matches
/// `scipy.stats.poisson(λ)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poisson { lambda: f64 }

impl Poisson {
    /// Construct with `λ > 0`.
    ///
    /// # Errors
    /// `DomainError` if `λ` is non-finite or `≤ 0`.
    ///
    /// ```
    /// use commonstats::dist::discrete::Poisson;
    /// use commonstats::dist::DiscreteMass;
    /// let p = Poisson::new(3.0).unwrap();
    /// assert!((p.mass(2) - 0.22404180765538775).abs() < 1e-12);
    /// assert!(Poisson::new(0.0).is_err());
    /// ```
    pub fn new(lambda: f64) -> Result<Self, StatError> {
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(StatError::DomainError("Poisson: lambda must be finite and > 0"));
        }
        Ok(Poisson { lambda })
    }
    /// Generous upper search bound: `λ + 10·√λ + 20`, well past the tail.
    fn search_hi(&self) -> i64 {
        (self.lambda + 10.0 * self.lambda.sqrt() + 20.0).ceil() as i64
    }
}

impl Distribution for Poisson {
    fn support_min(&self) -> Bound { Bound::Finite(0.0) }
    fn support_max(&self) -> Bound { Bound::PosInfinity }
    fn mean(&self) -> Option<f64> { Some(self.lambda) }
    fn variance(&self) -> Option<f64> { Some(self.lambda) }
    fn skewness(&self) -> Option<f64> { Some(1.0 / self.lambda.sqrt()) }
    fn kurtosis(&self) -> Option<f64> { Some(1.0 / self.lambda) }
}

impl DiscreteMass for Poisson {
    fn mass(&self, k: i64) -> f64 {
        if k < 0 { return 0.0; }
        self.log_mass(k).exp()
    }
    fn log_mass(&self, k: i64) -> f64 {
        if k < 0 { return f64::NEG_INFINITY; }
        let kf = k as f64;
        -self.lambda + kf * self.lambda.ln() - crate::special::lgamma(kf + 1.0)
    }
}

impl DiscreteCdf for Poisson {
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 { return 0.0; }
        crate::special::gammq((k + 1) as f64, self.lambda)
    }
    fn quantile(&self, p: f64) -> Result<i64, StatError> {
        discrete_bsearch_quantile(self, p, 0, self.search_hi())
    }
}

/// Geometric distribution (number of trials until first success).
///
/// Convention: **1-indexed** (support `{1, 2, …}`, like scipy `geom`), NOT the
/// number-of-failures form. `mass(k) = (1−p)^{k−1} p`. CDF `1 − (1−p)^k`.
/// Matches `scipy.stats.geom(p)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometric { p: f64 }

impl Geometric {
    /// Construct with `p ∈ (0, 1]`.
    ///
    /// # Errors
    /// `ProbabilityOutOfRange(p)` if `p ∉ (0, 1]` or non-finite.
    ///
    /// ```
    /// use commonstats::dist::discrete::Geometric;
    /// use commonstats::dist::DiscreteMass;
    /// let g = Geometric::new(0.25).unwrap();
    /// assert_eq!(g.mass(0), 0.0);
    /// assert!((g.mass(1) - 0.25).abs() < 1e-15);
    /// assert!(Geometric::new(0.0).is_err());
    /// ```
    pub fn new(p: f64) -> Result<Self, StatError> {
        if !p.is_finite() || p <= 0.0 || p > 1.0 {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        Ok(Geometric { p })
    }
}

impl Distribution for Geometric {
    fn support_min(&self) -> Bound { Bound::Finite(1.0) }
    fn support_max(&self) -> Bound { Bound::PosInfinity }
    fn mean(&self) -> Option<f64> { Some(1.0 / self.p) }
    fn variance(&self) -> Option<f64> { Some((1.0 - self.p) / (self.p * self.p)) }
}

impl DiscreteMass for Geometric {
    fn mass(&self, k: i64) -> f64 {
        if k < 1 { return 0.0; }
        self.log_mass(k).exp()
    }
    fn log_mass(&self, k: i64) -> f64 {
        if k < 1 { return f64::NEG_INFINITY; }
        // p=1: only k=1 has mass; (1-p)=0 ⇒ ln 0 = −∞ for k>1, but k=1 term is p.
        if self.p >= 1.0 { return if k == 1 { 0.0 } else { f64::NEG_INFINITY }; }
        (k as f64 - 1.0) * (1.0 - self.p).ln() + self.p.ln()
    }
}

impl DiscreteCdf for Geometric {
    fn cdf(&self, k: i64) -> f64 {
        if k < 1 { return 0.0; }
        -((k as f64) * (1.0 - self.p).ln()).exp_m1() // 1 - (1-p)^k
    }
    fn quantile(&self, p: f64) -> Result<i64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        if p == 0.0 { return Ok(1); }
        if self.p >= 1.0 { return Ok(1); }
        // smallest k with 1-(1-p)^k ≥ q ⟺ k ≥ ln(1-q)/ln(1-p).
        let k = ((1.0 - p).ln() / (1.0 - self.p).ln()).ceil();
        Ok((k.max(1.0)) as i64)
    }
}

/// Negative-binomial distribution: count of successes before the `r`-th
/// failure, success prob `p`.
///
/// Convention: `r > 0` (real-valued), `p ∈ (0, 1)` is the **success**
/// probability; support `{0, 1, …}`. Oracle is `scipy.stats.nbinom(n=r,
/// p=1−p)` (scipy's `p` is the *failure*-stop probability). `mass(k) =
/// C(k+r−1, k) (1−p)^r p^k`. CDF `P(X ≤ k) = I_{1−p}(r, k+1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegBinomial { r: f64, p: f64 }

impl NegBinomial {
    /// Construct with `r > 0`, `p ∈ (0, 1)`.
    ///
    /// # Errors
    /// `DomainError` if `r` non-finite or `≤ 0`; `ProbabilityOutOfRange(p)` if
    /// `p ∉ (0, 1)`.
    ///
    /// ```
    /// use commonstats::dist::discrete::NegBinomial;
    /// use commonstats::dist::DiscreteMass;
    /// let nb = NegBinomial::new(5.0, 0.4).unwrap();
    /// assert!((nb.mass(2) - 0.186624).abs() < 1e-10);
    /// assert!(NegBinomial::new(0.0, 0.4).is_err());
    /// ```
    pub fn new(r: f64, p: f64) -> Result<Self, StatError> {
        if !r.is_finite() || r <= 0.0 {
            return Err(StatError::DomainError("NegBinomial: r must be finite and > 0"));
        }
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        Ok(NegBinomial { r, p })
    }
    /// Generous upper search bound for the quantile binary search.
    fn search_hi(&self) -> i64 {
        let mean = self.r * self.p / (1.0 - self.p);
        let var = self.r * self.p / ((1.0 - self.p) * (1.0 - self.p));
        (mean + 12.0 * var.sqrt() + 30.0).ceil() as i64
    }
}

impl Distribution for NegBinomial {
    fn support_min(&self) -> Bound { Bound::Finite(0.0) }
    fn support_max(&self) -> Bound { Bound::PosInfinity }
    fn mean(&self) -> Option<f64> { Some(self.r * self.p / (1.0 - self.p)) }
    fn variance(&self) -> Option<f64> {
        Some(self.r * self.p / ((1.0 - self.p) * (1.0 - self.p)))
    }
}

impl DiscreteMass for NegBinomial {
    fn mass(&self, k: i64) -> f64 {
        if k < 0 { return 0.0; }
        self.log_mass(k).exp()
    }
    fn log_mass(&self, k: i64) -> f64 {
        if k < 0 { return f64::NEG_INFINITY; }
        let kf = k as f64;
        crate::special::lgamma(kf + self.r)
            - crate::special::lgamma(self.r)
            - crate::special::lgamma(kf + 1.0)
            + self.r * (1.0 - self.p).ln()
            + kf * self.p.ln()
    }
}

impl DiscreteCdf for NegBinomial {
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 { return 0.0; }
        // P(X ≤ k) = I_{1-p}(r, k+1).
        crate::special::betai(self.r, (k + 1) as f64, 1.0 - self.p)
    }
    fn quantile(&self, p: f64) -> Result<i64, StatError> {
        discrete_bsearch_quantile(self, p, 0, self.search_hi())
    }
}

/// Hypergeometric distribution: `n` draws without replacement from a population
/// of `N` with `K` successes.
///
/// Convention: parameters `(N, K, n)` with `K ≤ N`, `n ≤ N`; support
/// `[max(0, n+K−N), min(n, K)]`. `mass(k) = C(K,k)·C(N−K,n−k)/C(N,n)`. Oracle is
/// `scipy.stats.hypergeom(M=N, n=K, N=n)`. CDF is a direct sum over the support
/// (no closed form).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hypergeometric { big_n: i64, k: i64, n: i64 }

impl Hypergeometric {
    /// Construct with `K ≤ N`, `n ≤ N`, all `≥ 0`.
    ///
    /// # Errors
    /// `DomainError` if any value is negative, `K > N`, or `n > N`.
    ///
    /// ```
    /// use commonstats::dist::discrete::Hypergeometric;
    /// use commonstats::dist::Distribution;
    /// let h = Hypergeometric::new(20, 7, 12).unwrap();
    /// assert_eq!(h.mean(), Some(12.0 * 7.0 / 20.0));
    /// assert!(Hypergeometric::new(20, 25, 12).is_err());
    /// ```
    pub fn new(big_n: i64, k: i64, n: i64) -> Result<Self, StatError> {
        if big_n < 0 || k < 0 || n < 0 || k > big_n || n > big_n {
            return Err(StatError::DomainError("Hypergeometric: require 0 ≤ K ≤ N and 0 ≤ n ≤ N"));
        }
        Ok(Hypergeometric { big_n, k, n })
    }
    fn k_lo(&self) -> i64 { (self.n + self.k - self.big_n).max(0) }
    fn k_hi(&self) -> i64 { self.n.min(self.k) }
}

impl Distribution for Hypergeometric {
    fn support_min(&self) -> Bound { Bound::Finite(self.k_lo() as f64) }
    fn support_max(&self) -> Bound { Bound::Finite(self.k_hi() as f64) }
    fn mean(&self) -> Option<f64> {
        Some(self.n as f64 * self.k as f64 / self.big_n as f64)
    }
    fn variance(&self) -> Option<f64> {
        let (nn, kk, n) = (self.big_n as f64, self.k as f64, self.n as f64);
        if nn <= 1.0 { return Some(0.0); }
        Some(n * (kk / nn) * ((nn - kk) / nn) * ((nn - n) / (nn - 1.0)))
    }
}

impl DiscreteMass for Hypergeometric {
    fn mass(&self, k: i64) -> f64 {
        if k < self.k_lo() || k > self.k_hi() { return 0.0; }
        self.log_mass(k).exp()
    }
    fn log_mass(&self, k: i64) -> f64 {
        if k < self.k_lo() || k > self.k_hi() { return f64::NEG_INFINITY; }
        let (nn, kk, n, kf) = (self.big_n as f64, self.k as f64, self.n as f64, k as f64);
        ln_choose(kk, kf) + ln_choose(nn - kk, n - kf) - ln_choose(nn, n)
    }
}

impl DiscreteCdf for Hypergeometric {
    fn cdf(&self, k: i64) -> f64 {
        if k < self.k_lo() { return 0.0; }
        if k >= self.k_hi() { return 1.0; }
        // Direct sum over the support up to k (no closed form).
        let mut acc = 0.0;
        for j in self.k_lo()..=k {
            acc += self.mass(j);
        }
        acc.min(1.0)
    }
    fn quantile(&self, p: f64) -> Result<i64, StatError> {
        discrete_bsearch_quantile(self, p, self.k_lo(), self.k_hi())
    }
}
