//! The 6 discrete distributions of the `dist` suite.
//!
//! Quantile strategy: `Binomial`, `Poisson`, `NegBinomial`, and `Hypergeometric`
//! delegate to `discrete_bsearch_quantile`, which binary-searches for the
//! smallest `k` with `cdf(k) ≥ p` — the single home for that algorithm, so
//! don't add per-distribution loops for them. `Bernoulli` and `Geometric`
//! invert their CDF in closed form instead.
//!
//! <!-- accuracy:begin -->
//! ## Accuracy
//!
//! Worst error ratio `r` measured by `scripts/accuracy_sweep.py` against
//! mpmath (24800 points): `r` = relative error / (max(κ, 1)·ε), κ the
//! condition number over all real inputs, ε = 2⁻⁵²; for a log output the
//! absolute error over the sensitivity of the log. `r ≤ 10`: at the accuracy
//! the problem allows; `(review)` up to 1000; `(bug)` above it, a structural error;
//! `(hard)` a probability outside [0, 1], a NaN, a non-monotone cdf or a
//! quantile outside the support. Generated from
//! `scripts/accuracy_baseline.json` by `--update-baseline`; not edited by hand.
//!
//! | Distribution | mass | log_mass | cdf | sf | quantile |
//! |---|---|---|---|---|---|
//! | Bernoulli | 0.36 | 0.45 | 0.36 | 0 | 0 |
//! | Binomial | 0.83 | 1 | 1.5 | 3.7 | 0.5 |
//! | Poisson | 1 | 2.4 | 2.3 | 4 | 0.5 |
//! | Geometric | 1.5 | 0.9 | 0.95 | 0.86 | 2.8e-14 |
//! | NegBinomial | 6.3 | 0.73 | 0.86 | 1.9 | 0.5 |
//! | NegBinomial (`from_mean_size`) | 2.5 | 2 | 2.9 | 11 (review) | 0.5 |
//! | Hypergeometric | 1.9 | 6.7 | 1.7 | 1.3 | 1 |
//! <!-- accuracy:end -->

use crate::dist::{Bound, DiscreteCdf, DiscreteMass, Distribution};
#[cfg(feature = "rng")]
use crate::dist::{DiscreteSampler, continuous::gamma_draw};
use crate::error::StatError;
#[cfg(feature = "rng")]
use crate::rng::CommonStatsRng;
#[cfg(feature = "rng")]
use crate::special::elementary::stirling_del;
use crate::special::incomplete::value_of_parts;
use crate::special::saddle::{
    Dd, LN_2PI, bd0, bd0_f64, binom_saddle, dd_add, dd_neg, ln_binom_saddle, stirlerr, two_sum,
};

/// Binary search for the smallest integer `k ≥ lo` with `cdf(k) ≥ p`.
/// `p ∉ [0,1]` → `ProbabilityOutOfRange`. Used by every discrete `quantile`.
///
/// `hi` is a starting guess for the upper bracket, not a cap: while
/// `cdf(hi) < p` the bracket moves to `[hi + 1, 2·hi]`, so a heavy tail that
/// outruns the caller's moment-based guess still gets the right answer. Growth
/// saturates at `i64::MAX`; if `cdf(i64::MAX) < p` the quantile is not an
/// `i64` and the result is `DomainError`, never a clamped `k`. On bounded
/// support `cdf(hi) = 1` and the bracket never moves (the callers answer
/// `p = 1` themselves there, with the support maximum). On unbounded support
/// `p = 1` would return the first `k` whose `cdf` rounds to 1 in f64; the
/// unbounded callers answer `p = 1` themselves with `DomainError`.
pub(crate) fn discrete_bsearch_quantile<D: DiscreteCdf + ?Sized>(
    d: &D,
    p: f64,
    lo: i64,
    hi: i64,
) -> Result<i64, StatError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(StatError::ProbabilityOutOfRange(p));
    }
    if p == 0.0 {
        return Ok(lo);
    }
    let (mut a, mut b) = (lo, hi.max(lo));
    while d.cdf(b) < p {
        if b == i64::MAX {
            return Err(StatError::DomainError("quantile exceeds i64::MAX"));
        }
        a = b + 1;
        b = b.max(1).saturating_mul(2);
    }
    while a < b {
        let mid = a + (b - a) / 2;
        if d.cdf(mid) < p {
            a = mid + 1;
        } else {
            b = mid;
        }
    }
    Ok(a)
}

/// Bernoulli distribution with success probability `p`.
///
/// Convention: support `{0, 1}`, `mass(1) = p`. `0·ln0 := 0` so `p ∈ {0, 1}`
/// give finite `log_mass` (no NaN). Matches `scipy.stats.bernoulli(p)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bernoulli {
    p: f64,
}

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
    fn support_min(&self) -> Bound {
        Bound::Finite(0.0)
    }
    fn support_max(&self) -> Bound {
        Bound::Finite(1.0)
    }
    fn mean(&self) -> Option<f64> {
        Some(self.p)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.p * (1.0 - self.p))
    }
}

impl DiscreteMass for Bernoulli {
    fn mass(&self, k: i64) -> f64 {
        match k {
            0 => 1.0 - self.p,
            1 => self.p,
            _ => 0.0,
        }
    }
    fn log_mass(&self, k: i64) -> f64 {
        // 0·ln0 := 0 handled by mapping the zero-prob branch to ln of the value.
        // `log1p(−p)`: `ln(1 − p)` reads 0 once `1 − p` rounds to 1 (p < 1.1e-16).
        match k {
            0 => {
                if self.p >= 1.0 {
                    f64::NEG_INFINITY
                } else {
                    libm::log1p(-self.p)
                }
            }
            1 => {
                if self.p <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    libm::log(self.p)
                }
            }
            _ => f64::NEG_INFINITY,
        }
    }
}

impl DiscreteCdf for Bernoulli {
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 {
            0.0
        } else if k == 0 {
            1.0 - self.p
        } else {
            1.0
        }
    }
    /// `p` itself at `k = 0`, not `1 − cdf(0)`.
    fn sf(&self, k: i64) -> f64 {
        if k < 0 {
            1.0
        } else if k == 0 {
            self.p
        } else {
            0.0
        }
    }
    /// `0` iff `p + self.p ≤ 1`, tested exactly: `1 − x` is exact for `x ≥ ½`
    /// (Sterbenz), and where the rounded `1 − p` enters (`p < ½`, `self.p ≤
    /// ½`) the test holds either way. So `quantile(1)` is 1, as
    /// `scipy.stats.bernoulli.ppf(1)`, also where `1 − self.p` rounds to 1;
    /// at `self.p = 0` it is 0, the smallest `k` with `cdf(k) ≥ 1` since
    /// `cdf(0) = 1` exactly (scipy returns 1, the top of the support, there).
    fn quantile(&self, p: f64) -> Result<i64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        let at_zero = if self.p > 0.5 {
            p <= 1.0 - self.p
        } else {
            self.p <= 1.0 - p
        };
        Ok(if at_zero { 0 } else { 1 })
    }
}

/// `(u < p) as i64` for one `u = rng.uniform()` when the rarer outcome has
/// probability `m = min(p, 1 − p) ≥ 10⁻³` (`FINE_U_BELOW_P`) or `p ∈ {0, 1}`.
/// For `0 < m < 10⁻³`, where the 2³² grid would put ≥ 1.2e-7 relative error on
/// `m` (and never fire at all for `m < 2⁻³³`), the rarer outcome is drawn
/// exactly: with `t = m·2⁵²` and `x ∈ [0, 2⁵²)` the integer behind one
/// `uniform52()`, it occurs if `x < ⌊t⌋`, not if `x > ⌊t⌋`, and on `x = ⌊t⌋`
/// (probability 2⁻⁵²) the test repeats with `m ← t − ⌊t⌋`, so `P = ⌊t⌋/2⁵² +
/// 2⁻⁵²·(t − ⌊t⌋) = m` exactly (`t`, `⌊t⌋`, `t − ⌊t⌋` and `1 − p` for `p ≥ ½`
/// are all exact in f64).
#[cfg(all(feature = "dist", feature = "rng"))]
impl DiscreteSampler for Bernoulli {
    fn sample(&self, rng: &mut CommonStatsRng) -> i64 {
        const TWO52: f64 = 4_503_599_627_370_496.0;
        let (rare, mut m) = if self.p <= 0.5 {
            (1, self.p)
        } else {
            (0, 1.0 - self.p)
        };
        if m >= FINE_U_BELOW_P || m == 0.0 {
            return (rng.uniform() < self.p) as i64;
        }
        loop {
            let t = m * TWO52;
            let floor_t = libm::floor(t);
            let x = rng.uniform52() * TWO52 - 0.5;
            if x != floor_t {
                return if x < floor_t { rare } else { 1 - rare };
            }
            m = t - floor_t;
        }
    }
}

/// Binomial distribution `Binom(n, p)`.
///
/// Convention: `n ≥ 1` trials, success prob `p ∈ [0, 1]`, support `{0..n}`.
/// CDF `P(X ≤ k) = I_{1−p}(n−k, k+1)` (regularized incomplete beta), `sf` its
/// complement `I_p(k+1, n−k)`, each evaluated directly. Matches
/// `scipy.stats.binom(n, p)`.
///
/// `log_mass` is Loader's saddle-point form (R's `dbinom`), never a sum of
/// log-gammas, whose rounding grows as ε·n·ln n. Measured against mpmath
/// (`scripts/accuracy_sweep.py`, `n` up to 10¹⁵ and past 2⁵³, `p` from 1e-20
/// to 1 − 1e-15, `k` out to the support ends): within 3.1e-16·max(1,
/// |log_mass|) relative; `tests/dist_oracle.rs::log_mass_large_parameters`
/// adds `p` down to 1e-300. `n` and `k` above 2⁵³ are rounded to f64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Binomial {
    n: i64,
    p: f64,
}

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
    fn support_min(&self) -> Bound {
        Bound::Finite(0.0)
    }
    fn support_max(&self) -> Bound {
        Bound::Finite(self.n as f64)
    }
    fn mean(&self) -> Option<f64> {
        Some(self.n as f64 * self.p)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.n as f64 * self.p * (1.0 - self.p))
    }
}

impl DiscreteMass for Binomial {
    /// `e^{E}/√f` from `binom_saddle`'s `(E, f)`, the exponent kept as a
    /// double-double and applied as `value_of_parts((1 + E_lo)/√f, E_hi)`, so
    /// the mass keeps a few ε relative however large `|log_mass|` is:
    /// `exp(log_mass)` would carry `|log_mass|·ε`, 58ε at `Binomial::new(3526,
    /// 3.4e-60).mass(1)`. `value_of_parts` rescales where `e^{E_hi}` alone
    /// would be subnormal.
    fn mass(&self, k: i64) -> f64 {
        if k < 0 || k > self.n {
            return 0.0;
        }
        if self.p <= 0.0 || self.p >= 1.0 {
            let at = if self.p <= 0.0 { 0 } else { self.n };
            return if k == at { 1.0 } else { 0.0 };
        }
        // `s` and argument order mirror `log_mass` — change together.
        let (kf, rest) = (k as f64, (self.n - k) as f64);
        let (e, f) = if self.p <= 0.5 {
            binom_saddle(kf, rest, self.p)
        } else {
            binom_saddle(rest, kf, 1.0 - self.p)
        };
        // `1 + E_lo` stands for `e^{E_lo}`. `E_lo ≤ −1` only where `|E_hi| ≥
        // 2⁵³`, the mass 0, which the clamp keeps from turning into −0. The
        // comparison form, unlike `max`, leaves a NaN `E_lo` as NaN rather
        // than reading it as a 0 mass. `NegBinomial::mass` mirrors this —
        // change together.
        let lo = 1.0 + e.1;
        let lo = if lo < 0.0 { 0.0 } else { lo };
        value_of_parts(lo / libm::sqrt(f), e.0)
    }
    fn log_mass(&self, k: i64) -> f64 {
        if k < 0 || k > self.n {
            return f64::NEG_INFINITY;
        }
        // p=0 / p=1 edges: only k=0 / k=n have mass; ln of the rest is −∞.
        if self.p <= 0.0 || self.p >= 1.0 {
            let at = if self.p <= 0.0 { 0 } else { self.n };
            return if k == at { 0.0 } else { f64::NEG_INFINITY };
        }
        // Saddle-point form with the smaller of p, 1 − p as `s` (`1 − p` is
        // exact for p ≥ ½, Sterbenz); k and n − k exact in i64. Mirrors
        // `mass` — change together.
        let (kf, rest) = (k as f64, (self.n - k) as f64);
        if self.p <= 0.5 {
            ln_binom_saddle(kf, rest, self.p)
        } else {
            ln_binom_saddle(rest, kf, 1.0 - self.p)
        }
    }
}

impl Binomial {
    /// `(P(X ≤ k), P(X > k))` for `0 ≤ k < n`: `I_{1−p}(n−k, k+1)` and its
    /// complement `I_p(k+1, n−k)`, from one kernel call given the exact `p`
    /// beside `1 − p` (`betai_parts`), which evaluates the smaller tail
    /// directly. `p` is never rebuilt from a rounded `1 − p`, which would put
    /// ε/(2p) relative on it (all of it once `p < 2⁻⁵⁴`, where `1 − p` rounds
    /// to 1). The guards are on `p` itself for the same reason.
    fn tails(&self, k: i64) -> (f64, f64) {
        if self.p == 0.0 {
            return (1.0, 0.0);
        }
        if self.p == 1.0 {
            return (0.0, 1.0);
        }
        let (m, e, complement) = crate::special::incomplete::betai_parts(
            (self.n - k) as f64,
            (k + 1) as f64,
            1.0 - self.p,
            self.p,
        );
        crate::special::incomplete::pq_from_parts(m, e, complement)
    }
}

impl DiscreteCdf for Binomial {
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        if k >= self.n {
            return 1.0;
        }
        self.tails(k).0
    }
    fn sf(&self, k: i64) -> f64 {
        if k < 0 {
            return 1.0;
        }
        if k >= self.n {
            return 0.0;
        }
        self.tails(k).1
    }
    /// `quantile(1)` is `n`, as `scipy.stats.binom.ppf(1)`; at `p = 0` it is
    /// 0, the smallest `k` with `cdf(k) ≥ 1` since `cdf(0) = 1` exactly
    /// (scipy returns `n`, the top of the support, there).
    fn quantile(&self, p: f64) -> Result<i64, StatError> {
        if p == 1.0 && self.p > 0.0 {
            return Ok(self.n);
        }
        discrete_bsearch_quantile(self, p, 0, self.n)
    }
}

/// With `q = min(p, 1 − p)`: inversion by sequential search from `k = 0` when
/// `n·q < 10` (one uniform), otherwise BTRS transformed rejection (Hörmann
/// 1993, "The generation of binomial random variates", J. Statist. Comput.
/// Simul. 46:101–110; two uniforms per attempt). `p > 0.5` draws `n − X` with
/// `X ~ Binom(n, 1 − p)`.
///
/// BTRS accepts on `ln f(k)/f(m)`, `m` the mode. For `n < 10⁶` it is the
/// log-gamma sum `lnΓ(m+1) + lnΓ(n−m+1) − lnΓ(k+1) − lnΓ(n−k+1) + (k−m)·ln(p/q)`,
/// whose rounding, ≈ ε·n·ln n, is ≤ 1e-9 there but O(1) by `n ≈ 10¹²` (in 4·10⁵
/// draws it gave `var/(n·p·q)` = 0.92 at `Binom(10¹⁵, 0.3)`, 1.35 at
/// `Binom(10¹⁵, 10⁻¹⁴)`: the cancellation is in `n`, not `n·p`). From
/// `n = 10⁶` the ratio is taken in Stirling-difference form (see
/// `ln_fact_ratio_stirling`), accurate to ≈ ε·|k − m| absolute, and `k` is
/// formed as an `i64` offset from `m`, so draws are exact integers above 2⁵³
/// too.
///
/// The proposal's position uniform `u` is `uniform()` (2³² grid) below
/// `n·p·q = 10⁶` and `uniform52()` (2⁵²) from there (`FINE_U_FROM`): each `k`
/// near the mode is hit by ≈ (grid size)/(2.8·√(n·p·q)) values of `u`, and a
/// single probability carries a lattice error up to that count's reciprocal,
/// ≤ 6.5e-7 below the switch and ≤ 1.9e-6 above it up to `n·p·q = 9.2·10¹⁸`.
///
/// Validated by `tests/dist_oracle.rs`: χ² fit and moments in
/// `sampler_moments_and_fit`, moments at `n` up to 10¹⁷ in
/// `sampler_huge_parameter_moments`, pinned draws in `sampler_known_answers`,
/// the `u` switch in `sampler_fine_uniform_above_threshold`.
#[cfg(all(feature = "dist", feature = "rng"))]
impl DiscreteSampler for Binomial {
    fn sample(&self, rng: &mut CommonStatsRng) -> i64 {
        // `1 − p` is exact for `p ∈ [0.5, 1]` (Sterbenz).
        if self.p <= 0.5 {
            binomial_draw(self.n, self.p, rng)
        } else {
            self.n - binomial_draw(self.n, 1.0 - self.p, rng)
        }
    }
}

/// Parameter (`n` for BTRS, `λ` for PTRS) from which the rejection samplers
/// take the tail test's log pmf in Stirling-difference form. Below it the
/// log-gamma form is kept: its rounding, measured against mpmath, is ≤ 1.2e-9
/// absolute at 10⁶ (so acceptance moves by < 1e-8 relative), and keeping it
/// leaves every draw stream below 10⁶ unchanged.
#[cfg(feature = "rng")]
const STIRLING_FROM: f64 = 1e6;

/// Variance from which BTRS, PTRS and the hypergeometric inversion take their
/// position uniform `u` from `uniform52` (2⁵² grid) instead of `uniform`
/// (2³²). A grid of `G` values hits each `k` near the mode with only ≈
/// `G/(2.8·σ)` of them (BTRS/PTRS: the proposal's slope in `u` at the centre
/// is `4a + b ≈ 1.1·b ≈ 2.8·σ`; inversion: `G·f(mode) ≈ G/(2.5·σ)`), and each
/// single probability is off by up to one grid point, a relative ≈ 2.8·σ/G.
/// With `G = 2³²` that is 6.5e-7 at this threshold (1.54·10⁶ points per `k`,
/// counted at `λ = 10⁶`), 6.5e-4 at `σ² = 10¹²`, 20% at 10¹⁷; with `G = 2⁵²`
/// it stays ≤ 1.9e-6 up to `σ² = 9.2·10¹⁸` (`i64::MAX`). Below the threshold
/// the 32-bit `u` is kept, so those draw streams are unchanged. The acceptance
/// uniform `v` stays 32-bit: it is only compared against the acceptance ratio,
/// which its grid moves by ≤ 2⁻³² absolute whatever the variance. The
/// threshold is ≥ `STIRLING_FROM` (a Poisson variance is `λ`; a binomial `n·p·q
/// ≥ 10⁶` needs `n ≥ 4·10⁶`), so only the Stirling-form loops test it.
#[cfg(feature = "rng")]
const FINE_U_FROM: f64 = 1e6;

/// Probability below which the single-comparison samplers take their uniform
/// from `uniform52` (2⁵² grid) instead of `uniform` (2³²): the rarer Bernoulli
/// outcome (drawn exactly then, see its sampler), `P(X ≥ 1)` ≈ `λ` / `n·p` in
/// the Poisson / binomial inversion branches, and the geometric `p` (every `k
/// ≤ 1/p` has probability between `p/e` and `p`). A probability `P` read off a
/// grid of `G` points is off by up to `1/G` absolute, so by `1/(G·P)`
/// relative; counted exactly for the geometric inversion over `k ≤ 1/p`, the
/// worst is `e/(G·p)`: 5.6e-7 at `p = 10⁻³` with `G = 2³²` (4.7e-8 at 10⁻²,
/// 6e-4 at 10⁻⁶), and with `G = 2⁵²` 5.7e-10 at `p = 10⁻⁶`, ≤ 1e-6 down to
/// `p ≈ 6·10⁻¹⁰` (2.6e-6 at 2.2e-10). At or above the threshold the 32-bit
/// `uniform` is kept, so those draw streams are unchanged. For the geometric
/// it is the `FINE_U_FROM` rule in another guise: variance `(1 − p)/p² ≈ 10⁶`
/// at `p = 10⁻³`.
#[cfg(feature = "rng")]
const FINE_U_BELOW_P: f64 = 1e-3;

/// `lnΓ(m+1) − lnΓ(k+1) + d·ln m` for `m ≥ 8`, `k ≥ 0` an integer (as an
/// f64, so a caller's saturated `i64` draw never enters), and `d = k − m`
/// supplied without cancellation; `del_m = δ(m)`, hoisted by the caller. With
/// Stirling's `lnΓ(x+1) = (x+½)·ln x − x + ½·ln 2π + δ(x)` (`δ` =
/// `stirling_del`) it equals `d − (k+½)·ln(k/m) + δ(m) − δ(k)`, which never
/// forms the two log-gammas of size `m·ln m`; absolute error ≈ ε·|d|. `ln(k/m)`
/// is `log1p(d/m)` near `k = m`; below `m/2` the quotient `k/m` is formed
/// directly, as it has no cancellation there and `1 + d/m` would round to 0
/// once `k/m < ε`. `k < 8` is outside `δ`'s range and takes the log-gammas
/// directly: there the value is ≈ −m, so their ε·m·ln m rounding is small
/// relative to it.
#[cfg(feature = "rng")]
fn ln_fact_ratio_stirling(m: f64, del_m: f64, k: f64, d: f64) -> f64 {
    if k < 8.0 {
        crate::special::lgamma(m + 1.0) - crate::special::lgamma(k + 1.0) + d * libm::log(m)
    } else {
        let ln_km = if d < -0.5 * m {
            libm::log(k / m)
        } else {
            libm::log1p(d / m)
        };
        d - (k + 0.5) * ln_km + del_m - stirling_del(k)
    }
}

/// One `Binom(n, p)` draw for `p ∈ [0, 0.5]`; see the `Binomial` sampler doc.
#[cfg(feature = "rng")]
fn binomial_draw(n: i64, p: f64, rng: &mut CommonStatsRng) -> i64 {
    let nf = n as f64;
    let q = 1.0 - p;
    if nf * p < 10.0 {
        // pmf recurrence f(k) = f(k−1)·(p/q)·(n−k+1)/k from f(0) = qⁿ, which is
        // > e^{−14} here: `n·ln q ≥ −2 ln 2·n·p` for `p ≤ ½`, and `n·p < 10`.
        // Below `n·p = FINE_U_BELOW_P`, `P(X ≥ 1) ≈ n·p` needs the 2⁵² grid,
        // and `f(0)` is `exp(n·log1p(−p))`: `pow(1 − p, n)` carries the
        // rounding of `1 − p`, ≈ ε/p relative in `1 − f(0)`.
        let fine = p > 0.0 && nf * p < FINE_U_BELOW_P;
        let u = if fine { rng.uniform52() } else { rng.uniform() };
        let r = p / q;
        let mut f = if fine {
            libm::exp(nf * libm::log1p(-p))
        } else {
            libm::pow(q, nf)
        };
        let (mut s, mut k) = (f, 0);
        while u > s && k < n {
            k += 1;
            f *= r * (nf - k as f64 + 1.0) / k as f64;
            s += f;
        }
        return k;
    }
    // BTRS constants (Hörmann 1993, Algorithm BTRS).
    let spq = libm::sqrt(nf * p * q);
    let b = 1.15 + 2.53 * spq;
    let a = -0.0873 + 0.0248 * b + 0.01 * p;
    let c = nf * p + 0.5;
    let v_r = 0.92 - 4.2 / b;
    let alpha = (2.83 + 5.1 / b) * spq;
    if nf >= STIRLING_FROM {
        // Mode `m ≥ 10` and `n − m ≥ 10` (from `n·p ≥ 10`, `p ≤ ½`), both in
        // `δ`'s range. `k = m + d` with integer `d`, so `k` is exact in `i64`.
        let m = libm::floor((nf + 1.0) * p);
        let mi = m as i64;
        let nm = (n - mi) as f64;
        let c_off = c - m;
        // ln f(k)/f(m) = d·ln((n−m)·p/(m·q)) + R(m; k) + R(n−m; n−k), R =
        // `ln_fact_ratio_stirling`: the `d·ln m`, `−d·ln(n−m)` it adds cancel
        // against `d·ln(p/q)` into one log of a number ≈ 1.
        let ln_r = libm::log(nm * p / (m * q));
        let (del_m, del_nm) = (stirling_del(m), stirling_del(nm));
        let fine = spq * spq >= FINE_U_FROM;
        loop {
            let u = if fine { rng.uniform52() } else { rng.uniform() } - 0.5;
            let v = rng.uniform();
            let us = 0.5 - u.abs();
            let d = libm::floor((2.0 * a / us + b) * u + c_off);
            let k = mi.saturating_add(d as i64);
            if k < 0 || k > n {
                continue;
            }
            if us >= 0.07 && v <= v_r {
                return k;
            }
            let lhs = libm::log(v * alpha / (a / (us * us) + b));
            let rhs = d * ln_r
                + ln_fact_ratio_stirling(m, del_m, k as f64, d)
                + ln_fact_ratio_stirling(nm, del_nm, (n - k) as f64, -d);
            if lhs <= rhs {
                return k;
            }
        }
    }
    // Tail-test constants, formed only once the squeeze first fails.
    let mut tail: Option<(f64, f64, f64)> = None;
    loop {
        let u = rng.uniform() - 0.5;
        let v = rng.uniform();
        // `uniform() ∈ (0, 1)` keeps `us > 0`.
        let us = 0.5 - u.abs();
        let k = libm::floor((2.0 * a / us + b) * u + c);
        if k < 0.0 || k > nf {
            continue;
        }
        if us >= 0.07 && v <= v_r {
            return k as i64;
        }
        let (lpq, m, h) = *tail.get_or_insert_with(|| {
            let m = libm::floor((nf + 1.0) * p);
            let h = crate::special::lgamma(m + 1.0) + crate::special::lgamma(nf - m + 1.0);
            (libm::log(p / q), m, h)
        });
        // Accept on ln f(k)/f(m), the exact log pmf ratio to the mode `m`.
        let lhs = libm::log(v * alpha / (a / (us * us) + b));
        let rhs = h - crate::special::lgamma(k + 1.0) - crate::special::lgamma(nf - k + 1.0)
            + (k - m) * lpq;
        if lhs <= rhs {
            return k as i64;
        }
    }
}

/// Poisson distribution with rate `λ`.
///
/// Convention: support `{0, 1, …}`, `mass(k) = e^{−λ}λ^k/k!`. CDF
/// `P(X ≤ k) = Q(k+1, λ)` (upper regularized incomplete gamma), `sf` the lower
/// `P(k+1, λ)`, each evaluated directly. Matches `scipy.stats.poisson(λ)`.
///
/// `log_mass` is Loader's saddle-point form (R's `dpois`), `−δ(k) − bd0(k, λ)
/// − ½·ln(2πk)`, where `−λ + k·ln λ − lnΓ(k+1)` loses ≈ ε·λ·ln λ. Measured
/// against mpmath (`scripts/accuracy_sweep.py`, `λ` from 1e-10 to 10¹⁵):
/// within 4.9e-16·max(1, |log_mass|) relative where the log mass is well
/// conditioned in `λ` (its sensitivity `|k − λ|` at most `max(1,
/// |log_mass|)`); elsewhere the error follows that sensitivity, at most
/// 2.4·ε times it: 4.3e-15 relative at `λ = 64409.7`, `k = 82175`, where
/// `bd0`'s `k·ln(k/λ) − (k − λ)` cancels ~9×.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poisson {
    lambda: f64,
}

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
            return Err(StatError::DomainError(
                "Poisson: lambda must be finite and > 0",
            ));
        }
        Ok(Poisson { lambda })
    }
    /// Starting upper bracket for `discrete_bsearch_quantile`: `λ + 10·√λ + 20`.
    fn search_hi(&self) -> i64 {
        libm::ceil(self.lambda + 10.0 * libm::sqrt(self.lambda) + 20.0) as i64
    }
}

impl Distribution for Poisson {
    fn support_min(&self) -> Bound {
        Bound::Finite(0.0)
    }
    fn support_max(&self) -> Bound {
        Bound::PosInfinity
    }
    fn mean(&self) -> Option<f64> {
        Some(self.lambda)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.lambda)
    }
    fn skewness(&self) -> Option<f64> {
        Some(1.0 / libm::sqrt(self.lambda))
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(1.0 / self.lambda)
    }
}

impl DiscreteMass for Poisson {
    /// `e^{E}/√(2πk)`, `E = −δ(k) − bd0(k, λ)` (`log_mass`'s form) with `bd0`
    /// as a double-double from the exact `d = k − λ`, applied as
    /// `value_of_parts((1 + E_lo)/√(2πk), E_hi)`, so the mass keeps a few ε
    /// relative however large `|log_mass|` is: `exp(log_mass)` would
    /// carry `|log_mass|·ε`, 110ε at `λ = 1e-300`, `k = 1`. Outside `bd0`'s
    /// double-double window `E` is f64, where its error is under ε/8 absolute,
    /// or the mass is subnormal (`λ` subnormal) or below the f64 range.
    /// `e^{−λ}` at `k = 0`.
    fn mass(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        if k == 0 {
            return libm::exp(-self.lambda);
        }
        let kf = k as f64;
        let bd = bd0(kf, (self.lambda, 0.0), two_sum(kf, -self.lambda));
        let e = dd_add((-stirlerr(kf), 0.0), dd_neg(bd));
        // `1 + E_lo` stands for `e^{E_lo}`: `bd0`'s own `lo` is ≤ 1e3·ε, and the
        // sum's rounding is at most `δ(k) < 0.09`.
        value_of_parts((1.0 + e.1) / libm::sqrt(core::f64::consts::TAU * kf), e.0)
    }
    fn log_mass(&self, k: i64) -> f64 {
        if k < 0 {
            return f64::NEG_INFINITY;
        }
        if k == 0 {
            return -self.lambda;
        }
        // Loader's `dpois_raw`: `−δ(k) − bd0(k, λ) − ½·ln(2πk)`, no term of
        // size λ·ln λ (see `ln_binom_saddle`).
        let kf = k as f64;
        -stirlerr(kf) - bd0_f64(kf, self.lambda, kf - self.lambda) - 0.5 * (LN_2PI + libm::log(kf))
    }
}

impl DiscreteCdf for Poisson {
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        // `k as f64 + 1`: `k + 1` overflows at `k = i64::MAX`, which the
        // quantile search probes.
        crate::special::gammq(k as f64 + 1.0, self.lambda)
    }
    /// `P(k+1, λ)`, the lower incomplete gamma, the complement of `cdf`'s
    /// `Q(k+1, λ)` evaluated directly.
    fn sf(&self, k: i64) -> f64 {
        if k < 0 {
            return 1.0;
        }
        crate::special::gammp(k as f64 + 1.0, self.lambda)
    }
    /// `DomainError` at `p = 1` (the support is unbounded, so the quantile is
    /// +∞), and where `λ` is large enough that `search_hi()` cannot bracket a
    /// finite `i64` quantile, e.g. `Poisson::new(1e19).quantile(0.5)`.
    fn quantile(&self, p: f64) -> Result<i64, StatError> {
        if p == 1.0 {
            return Err(StatError::DomainError("quantile exceeds i64::MAX"));
        }
        discrete_bsearch_quantile(self, p, 0, self.search_hi())
    }
}

/// Inversion by sequential search from `k = 0` for `λ < 10` (one uniform);
/// PTRS transformed rejection for `λ ≥ 10` (Hörmann 1993, "The transformed
/// rejection method for generating Poisson random variables", Insurance:
/// Mathematics and Economics 12(1):39–45; two uniforms per attempt). A draw
/// above `i64::MAX`, possible only for `λ ≳ 9.2e18`, saturates to `i64::MAX`.
///
/// PTRS accepts on `ln f(k)`. For `λ < 10⁶` it is `−λ + k·ln λ − lnΓ(k+1)`,
/// whose rounding, ≈ ε·λ·ln λ, is ≤ 1.2e-9 there but O(1) by `λ ≈ 10¹²` (in
/// 4·10⁵ draws it gave `var/λ` = 1.04 at `λ = 10¹⁵`, 1.63 at 10¹⁷). From
/// `λ = 10⁶` it is taken in Stirling-difference form, `ln f(k) = −[k·ln(1 +
/// d/λ) − d] − ½·ln(2πk) − δ(k)` with `d = k − λ` (`δ` the Stirling
/// correction; exact log-gammas for `k < 8`), accurate to ≈ ε·|d| absolute,
/// and `k` is formed as an `i64` offset from `⌊λ⌋`, so draws are exact
/// integers above 2⁵³ too (`λ` itself is an f64, so above 2⁵³ it is the
/// nearest representable value to the caller's rate).
///
/// The proposal's position uniform `u` is `uniform()` (2³² grid) below `λ =
/// 10⁶` and `uniform52()` (2⁵²) from there (`FINE_U_FROM`): each `k` near `λ`
/// is hit by ≈ (grid size)/(2.8·√λ) values of `u`, and a single probability
/// carries a lattice error up to that count's reciprocal, ≤ 6.5e-7 below the
/// switch and ≤ 1.9e-6 above it up to `λ = 9.2·10¹⁸`.
///
/// Validated by `tests/dist_oracle.rs`: χ² fit and moments in
/// `sampler_moments_and_fit`, moments at `λ` up to 10¹⁷ in
/// `sampler_huge_parameter_moments`, pinned draws in `sampler_known_answers`,
/// the `u` switch in `sampler_fine_uniform_above_threshold`.
#[cfg(all(feature = "dist", feature = "rng"))]
impl DiscreteSampler for Poisson {
    fn sample(&self, rng: &mut CommonStatsRng) -> i64 {
        poisson_draw(self.lambda, rng)
    }
}

/// One `Poisson(λ)` draw, `λ > 0`; see the `Poisson` sampler doc. Takes `λ`
/// rather than a `Poisson` so the `NegBinomial` mixture can call it per draw.
#[cfg(feature = "rng")]
fn poisson_draw(lambda: f64, rng: &mut CommonStatsRng) -> i64 {
    if lambda < 10.0 {
        // pmf recurrence f(k) = f(k−1)·λ/k from f(0) = e^{−λ} > e^{−10}. The
        // walk also stops once the pmf underflows to 0, in case rounding keeps
        // the partial sum below `u`. `P(X ≥ 1) ≈ λ` needs the 2⁵² grid below
        // `λ = FINE_U_BELOW_P`.
        let u = if lambda < FINE_U_BELOW_P {
            rng.uniform52()
        } else {
            rng.uniform()
        };
        let mut f = libm::exp(-lambda);
        let (mut s, mut k) = (f, 0);
        while u > s && f > 0.0 {
            k += 1;
            f *= lambda / k as f64;
            s += f;
        }
        return k;
    }
    // PTRS constants (Hörmann 1993, Algorithm PTRS).
    let b = 0.931 + 2.53 * libm::sqrt(lambda);
    let a = -0.059 + 0.02483 * b;
    let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let v_r = 0.9277 - 3.6224 / (b - 2.0);
    if lambda >= STIRLING_FROM {
        // `k = ⌊λ⌋ + j` with integer `j`, so `d = k − λ = j − {λ}` never
        // subtracts two numbers of size λ. `ln f(k) = R(λ; k) − ½·ln(2πλ) −
        // δ(λ)`, R = `ln_fact_ratio_stirling`: the Stirling form above with the
        // `λ`-only terms split off.
        let lam_int = libm::floor(lambda);
        let lam_frac = lambda - lam_int;
        let base = lam_int as i64;
        let del_lam = stirling_del(lambda);
        let ln_f_ref = -0.5 * libm::log(2.0 * core::f64::consts::PI * lambda) - del_lam;
        let fine = lambda >= FINE_U_FROM;
        loop {
            let u = if fine { rng.uniform52() } else { rng.uniform() } - 0.5;
            let v = rng.uniform();
            let us = 0.5 - u.abs();
            let j = libm::floor((2.0 * a / us + b) * u + (lam_frac + 0.43));
            // The tail test reads `kf`, consistent with `d = j − {λ}`. The draw
            // is the exact offset from `⌊λ⌋` while `⌊λ⌋` fits an `i64`; above
            // that the saturating cast (`base + j` would wrap below it).
            let kf = lam_int + j;
            let k = if base == i64::MAX {
                kf as i64
            } else {
                base.saturating_add(j as i64)
            };
            if us >= 0.07 && v <= v_r {
                return k;
            }
            if kf < 0.0 || (us < 0.013 && v > us) {
                continue;
            }
            let lhs = libm::log(v * inv_alpha / (a / (us * us) + b));
            if lhs <= ln_f_ref + ln_fact_ratio_stirling(lambda, del_lam, kf, j - lam_frac) {
                return k;
            }
        }
    }
    // `ln λ`, formed only once the squeeze first fails.
    let mut log_lambda: Option<f64> = None;
    loop {
        let u = rng.uniform() - 0.5;
        let v = rng.uniform();
        // `uniform() ∈ (0, 1)` keeps `us > 0`.
        let us = 0.5 - u.abs();
        let k = libm::floor((2.0 * a / us + b) * u + lambda + 0.43);
        if us >= 0.07 && v <= v_r {
            return k as i64;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        let log_lambda = *log_lambda.get_or_insert_with(|| libm::log(lambda));
        let lhs = libm::log(v * inv_alpha / (a / (us * us) + b));
        if lhs <= -lambda + k * log_lambda - crate::special::lgamma(k + 1.0) {
            return k as i64;
        }
    }
}

/// Geometric distribution (number of trials until first success).
///
/// Convention: **1-indexed** (support `{1, 2, …}`, like scipy `geom`), NOT the
/// number-of-failures form. `mass(k) = (1−p)^{k−1} p`. CDF `1 − (1−p)^k`.
/// Matches `scipy.stats.geom(p)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometric {
    p: f64,
}

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
    fn support_min(&self) -> Bound {
        Bound::Finite(1.0)
    }
    fn support_max(&self) -> Bound {
        Bound::PosInfinity
    }
    fn mean(&self) -> Option<f64> {
        Some(1.0 / self.p)
    }
    fn variance(&self) -> Option<f64> {
        Some((1.0 - self.p) / (self.p * self.p))
    }
}

impl DiscreteMass for Geometric {
    /// `p·exp((k − 1)·log1p(−p))` by `value_of_parts`, so a subnormal mass is
    /// rounded once: `ln p` stays out of the exponent, whose
    /// rounding, ≈ ε·|(k − 1)·ln(1 − p)| absolute, is then at most its own
    /// sensitivity to `p`; `exp(log_mass)` would carry `|ln p|·ε` more,
    /// 110ε at `p = 1e-300`, where `mass(1)` is `p` exactly.
    fn mass(&self, k: i64) -> f64 {
        if k < 1 {
            return 0.0;
        }
        if self.p >= 1.0 {
            return if k == 1 { 1.0 } else { 0.0 };
        }
        value_of_parts(self.p, (k as f64 - 1.0) * libm::log1p(-self.p))
    }
    fn log_mass(&self, k: i64) -> f64 {
        if k < 1 {
            return f64::NEG_INFINITY;
        }
        // p=1: only k=1 has mass; (1-p)=0 ⇒ ln 0 = −∞ for k>1, but k=1 term is p.
        if self.p >= 1.0 {
            return if k == 1 { 0.0 } else { f64::NEG_INFINITY };
        }
        (k as f64 - 1.0) * libm::log1p(-self.p) + libm::log(self.p)
    }
}

impl DiscreteCdf for Geometric {
    fn cdf(&self, k: i64) -> f64 {
        if k < 1 {
            return 0.0;
        }
        -libm::expm1((k as f64) * libm::log1p(-self.p)) // 1 - (1-p)^k
    }
    /// `(1 − p)^k` as `exp(k·log1p(−p))`.
    fn sf(&self, k: i64) -> f64 {
        if k < 1 {
            return 1.0;
        }
        libm::exp((k as f64) * libm::log1p(-self.p))
    }
    /// `DomainError` where the root exceeds `i64::MAX`: `p = 1` for any
    /// `self.p < 1` (the support is unbounded, so the quantile is +∞), or a
    /// tiny `self.p`. At `self.p = 1` all mass is at `k = 1`, so `quantile(1)`
    /// is 1.
    fn quantile(&self, p: f64) -> Result<i64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        let k = self.root(p);
        // 2⁶³ = `i64::MAX + 1`, exact in f64.
        if k >= 9_223_372_036_854_775_808.0 {
            return Err(StatError::DomainError("quantile exceeds i64::MAX"));
        }
        Ok(k as i64)
    }
}

impl Geometric {
    /// The quantile as an f64, `≥ 1`: the smallest `k` with `1 − (1−p)^k ≥ q`,
    /// `⌈ln(1−q)/ln(1−p)⌉`; `+∞` at `q = 1`. `log1p`: `ln` of a rounded `1 −
    /// p` is off by ≈ ε/p relative for small `p`.
    fn root(&self, q: f64) -> f64 {
        if q == 0.0 || self.p >= 1.0 {
            return 1.0;
        }
        libm::ceil(libm::log1p(-q) / libm::log1p(-self.p)).max(1.0)
    }
}

/// Inversion, the `quantile` root at `u`, with `u` from `uniform52()` (2⁵²
/// grid) for `p < 10⁻³` (`FINE_U_BELOW_P`) and `uniform()` (2³²) from
/// there. A draw above `i64::MAX`, possible only for `p < 4·10⁻¹⁸`, saturates
/// to `i64::MAX`, as the `Poisson` sampler's does (`quantile` returns an error
/// there). Each `k ≤ 1/p` has
/// probability ≥ `p/e`, so its lattice error is ≤ `e/(G·p)` relative for a
/// grid of `G` points: ≤ 5.6e-7 on the 32-bit side of the switch, 5.7e-10 at
/// `p = 10⁻⁶`, and ≤ 1e-6 down to `p ≈ 6·10⁻¹⁰`; below that the error grows
/// as `1/p` (the ratio `ln(1 − u)/ln(1 − p)` is itself rounded to ≈ ε·k).
/// Validated by `tests/dist_oracle.rs`: χ² fit in `sampler_moments_and_fit`,
/// the `u` switch in `sampler_fine_uniform_small_probability`.
#[cfg(all(feature = "dist", feature = "rng"))]
impl DiscreteSampler for Geometric {
    fn sample(&self, rng: &mut CommonStatsRng) -> i64 {
        let u = if self.p < FINE_U_BELOW_P {
            rng.uniform52()
        } else {
            rng.uniform()
        };
        self.root(u) as i64
    }
}

/// Negative-binomial distribution: count of successes before the `r`-th
/// failure, success prob `p`.
///
/// Convention: `r > 0` (real-valued), `p ∈ (0, 1)` is the **success**
/// probability; support `{0, 1, …}`. Oracle is `scipy.stats.nbinom(n=r,
/// p=1−p)` (scipy's `p` is the *failure*-stop probability). `mass(k) =
/// C(k+r−1, k) (1−p)^r p^k`. CDF `P(X ≤ k) = I_{1−p}(r, k+1)`, `sf` its
/// complement `I_p(k+1, r)`, each evaluated directly.
///
/// `log_mass` is `ln(r/(r+k))` plus a binomial log pmf in Loader's
/// saddle-point form (R's `dnbinom`). Measured against mpmath
/// (`scripts/accuracy_sweep.py`, `r` from 1e-10 to 10¹⁵, `p` up to 1 −
/// 1e-12, `from_mean_size` with `θ` up to 10¹⁵): within 3.0e-16·max(1,
/// |log_mass|) relative where the log mass is well conditioned in `r` and `p`
/// (its sensitivity to them at most `max(1, |log_mass|)`); elsewhere the error
/// follows that sensitivity, at most 2·ε times it: 1.0e-11 relative at `r =
/// 7.5·10¹³`, `p = 0.99937`, `k = 1.2·10¹⁷`, 7.3e-14 at
/// `from_mean_size(1.4·10⁷, 8.3·10¹¹)`, `k = 1.4·10⁷`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegBinomial {
    r: f64,
    p: f64,
    /// Failure probability `1 − p`, stored rather than recomputed: from
    /// `from_mean_size` it is `θ/(μ + θ)` directly, which keeps its digits when
    /// `μ ≫ θ` puts `p` next to 1.
    q: f64,
}

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
            return Err(StatError::DomainError(
                "NegBinomial: r must be finite and > 0",
            ));
        }
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        Ok(NegBinomial { r, p, q: 1.0 - p })
    }
    /// Construct from mean `μ` and size `θ`, the parameterization of R's
    /// `dnbinom(k, size = θ, mu = μ)` / `rnbinom(n, size = θ, mu = μ)`:
    /// variance `μ + μ²/θ`. Maps to `r = θ`, `p = μ/(μ + θ)` (success
    /// probability in this type's convention), so `r·p/(1−p) = μ`.
    ///
    /// # Errors
    /// `DomainError` if `mu` or `size` is non-finite or `≤ 0`;
    /// `ProbabilityOutOfRange` if `μ/(μ + θ)` rounds to 0 or 1 in f64.
    ///
    /// ```
    /// use commonstats::dist::discrete::NegBinomial;
    /// use commonstats::dist::{DiscreteMass, Distribution};
    /// let nb = NegBinomial::from_mean_size(3.0, 0.5).unwrap();
    /// assert!((nb.mean().unwrap() - 3.0).abs() < 1e-12);
    /// assert!((nb.variance().unwrap() - (3.0 + 9.0 / 0.5)).abs() < 1e-10);
    /// // R: dnbinom(2, size = 0.5, mu = 3) = 0.104133069094378916
    /// assert!((nb.mass(2) - 0.104_133_069_094_378_92).abs() < 1e-14);
    /// assert!(NegBinomial::from_mean_size(3.0, 0.0).is_err());
    /// ```
    pub fn from_mean_size(mu: f64, size: f64) -> Result<Self, StatError> {
        if !mu.is_finite() || !size.is_finite() || mu <= 0.0 || size <= 0.0 {
            return Err(StatError::DomainError(
                "NegBinomial: mu and size must be finite and > 0",
            ));
        }
        // Where `μ + θ` overflows both are ≥ 2^970, so halving them for the
        // ratios is exact and leaves `p` and `q` unchanged.
        let (m, s) = if (mu + size).is_finite() {
            (mu, size)
        } else {
            (0.5 * mu, 0.5 * size)
        };
        let nb = Self::new(size, m / (m + s))?;
        Ok(NegBinomial {
            q: s / (m + s),
            ..nb
        })
    }
    /// Starting upper bracket for `discrete_bsearch_quantile`: mean + 12 sd +
    /// 30, saturated to `i64::MAX` by the cast (`r·p/q` reaches 10²⁷ for `p`
    /// near 1), where the search then reports a quantile past `i64` as an error.
    fn search_hi(&self) -> i64 {
        let mean = self.r * self.p / self.q;
        let var = self.r * self.p / (self.q * self.q);
        libm::ceil(mean + 12.0 * libm::sqrt(var) + 30.0) as i64
    }
}

impl Distribution for NegBinomial {
    fn support_min(&self) -> Bound {
        Bound::Finite(0.0)
    }
    fn support_max(&self) -> Bound {
        Bound::PosInfinity
    }
    fn mean(&self) -> Option<f64> {
        Some(self.r * self.p / self.q)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.r * self.p / (self.q * self.q))
    }
}

impl DiscreteMass for NegBinomial {
    /// `r/(r+k)·e^{E}/√f`, `(E, f)` from `binom_saddle` for `b(r; r+k, q)`
    /// (`log_mass`'s form), the exponent kept as a double-double and applied
    /// as `value_of_parts(r/(r + k)·(1 + E_lo)/√f, E_hi)`, so its size adds no
    /// error: `exp(log_mass)` would carry `|log_mass|·ε`, 1.2e-14
    /// relative at `NegBinomial::new(1, 2.2e-56).mass(1)`. What remains is the
    /// error of `E`'s terms: a few ε for `r ≥ 8` and down to `r = 1e-50`
    /// (≤ 7ε measured against mpmath); below that `stirlerr(r)`'s lgamma
    /// difference adds up to ≈ ε·|ln r|/4 absolute, 51ε relative at `r = 1e-100`,
    /// 143ε at `r = 1e-307`. The ratio `r/(r + k) ≤ 1` goes first: at `k = 0`
    /// and huge `r`, `E_lo` reaches ½ulp(`E_hi`) and `r·(1 + E_lo)` would
    /// overflow. `r + k` is finite for every finite `r`, so the ratio needs no
    /// `k/r` overflow guard.
    fn mass(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        // The `p`/`q` choice mirrors `log_mass` — change together.
        let kf = k as f64;
        let (e, f) = if self.q <= self.p {
            binom_saddle(self.r, kf, self.q)
        } else {
            binom_saddle(kf, self.r, self.p)
        };
        // The clamp of `Binomial::mass` — change together.
        let lo = 1.0 + e.1;
        let lo = if lo < 0.0 { 0.0 } else { lo };
        value_of_parts(self.r / (self.r + kf) * (lo / libm::sqrt(f)), e.0)
    }
    fn log_mass(&self, k: i64) -> f64 {
        if k < 0 {
            return f64::NEG_INFINITY;
        }
        // R's `dnbinom`: `ln(r/(r+k)) + ln b(r; r+k, q)`, `b` the binomial pmf
        // in `ln_binom_saddle` form. The smaller of the stored `p`, `q` is
        // taken as exact and the other as its complement: next to 1 the
        // larger one's own rounding would swamp `ln` of it. Mirrors `mass` —
        // change together.
        let kf = k as f64;
        let core = if self.q <= self.p {
            ln_binom_saddle(self.r, kf, self.q)
        } else {
            ln_binom_saddle(kf, self.r, self.p)
        };
        let t = kf / self.r;
        let ln_frac = if t.is_finite() {
            -libm::log1p(t)
        } else {
            libm::log(self.r) - libm::log(self.r + kf)
        };
        ln_frac + core
    }
}

impl NegBinomial {
    /// `(P(X ≤ k), P(X > k))` for `k ≥ 0`: `I_q(r, k+1)` and its complement
    /// `I_p(k+1, r)`, from one kernel call given both stored `q` and `p`
    /// (`betai_parts`), which evaluates the smaller tail directly. Neither is
    /// rebuilt as `1 −` the other, which would put ε/(2p) relative on a small
    /// `p` (all of it once `1 − p` rounds to 1, as at `from_mean_size` with
    /// `θ ≫ μ`). `k as f64 + 1`: `k + 1` overflows at `k = i64::MAX`, which
    /// the quantile search probes.
    fn tails(&self, k: i64) -> (f64, f64) {
        let (m, e, complement) =
            crate::special::incomplete::betai_parts(self.r, k as f64 + 1.0, self.q, self.p);
        crate::special::incomplete::pq_from_parts(m, e, complement)
    }
}

impl DiscreteCdf for NegBinomial {
    fn cdf(&self, k: i64) -> f64 {
        if k < 0 {
            return 0.0;
        }
        self.tails(k).0
    }
    fn sf(&self, k: i64) -> f64 {
        if k < 0 {
            return 1.0;
        }
        self.tails(k).1
    }
    /// `DomainError` at `p = 1`: the support is unbounded, so the quantile is
    /// +∞.
    fn quantile(&self, p: f64) -> Result<i64, StatError> {
        if p == 1.0 {
            return Err(StatError::DomainError("quantile exceeds i64::MAX"));
        }
        discrete_bsearch_quantile(self, p, 0, self.search_hi())
    }
}

/// Gamma–Poisson mixture: `λ ~ Gamma(shape r, rate (1−p)/p)`, then
/// `X ~ Poisson(λ)`, so `E[X] = E[λ] = r·p/(1−p)`. `λ` is drawn as
/// `G·p/(1−p)` with `G ~ Gamma(r, 1)` from the `Gamma` sampler
/// (Marsaglia–Tsang), which keeps the rate finite for `p` near 0; the Poisson
/// draw is the `Poisson` sampler's, including its saturation at `i64::MAX`. A
/// `λ` that underflows to 0 returns 0.
#[cfg(all(feature = "dist", feature = "rng"))]
impl DiscreteSampler for NegBinomial {
    fn sample(&self, rng: &mut CommonStatsRng) -> i64 {
        let lambda = gamma_draw(self.r, rng) * (self.p / self.q);
        if lambda > 0.0 {
            poisson_draw(lambda, rng)
        } else {
            0
        }
    }
}

/// Hypergeometric distribution: `n` draws without replacement from a population
/// of `N` with `K` successes.
///
/// Convention: parameters `(N, K, n)` with `K ≤ N`, `n ≤ N`; support
/// `[max(0, n+K−N), min(n, K)]`. `mass(k) = C(K,k)·C(N−K,n−k)/C(N,n)`. Oracle is
/// `scipy.stats.hypergeom(M=N, n=K, N=n)`. `cdf` and `sf` are sums over the
/// support (no closed form), each tail over its own terms (see its
/// `DiscreteCdf` impl).
///
/// `mass` and `log_mass` are three binomial pmfs in Loader's saddle-point form
/// (R's `dhyper`), where a sum of log-gammas loses ≈ ε·N·ln N, with the
/// exponent kept as a double-double. Measured against mpmath
/// (`scripts/accuracy_sweep.py`, `N` from 2 to 10¹⁵, normal-range values):
/// `mass` within 4.3e-16 relative, `log_mass` within 2.7e-16, `cdf` within
/// 3.8e-16 and `sf` within 3.0e-16, also in the far tails where `ln mass`
/// nears −745. Beyond them `log_mass` is within 1.5e-15 relative (at
/// `log_mass` = −2.1·10¹¹, `N` = 1.1·10¹⁴).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hypergeometric {
    big_n: i64,
    k: i64,
    n: i64,
}

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
            return Err(StatError::DomainError(
                "Hypergeometric: require 0 ≤ K ≤ N and 0 ≤ n ≤ N",
            ));
        }
        Ok(Hypergeometric { big_n, k, n })
    }
    fn k_lo(&self) -> i64 {
        (self.n + self.k - self.big_n).max(0)
    }
    fn k_hi(&self) -> i64 {
        self.n.min(self.k)
    }
}

impl Distribution for Hypergeometric {
    fn support_min(&self) -> Bound {
        Bound::Finite(self.k_lo() as f64)
    }
    fn support_max(&self) -> Bound {
        Bound::Finite(self.k_hi() as f64)
    }
    fn mean(&self) -> Option<f64> {
        Some(self.n as f64 * self.k as f64 / self.big_n as f64)
    }
    fn variance(&self) -> Option<f64> {
        let (nn, kk, n) = (self.big_n as f64, self.k as f64, self.n as f64);
        if nn <= 1.0 {
            return Some(0.0);
        }
        Some(n * (kk / nn) * ((nn - kk) / nn) * ((nn - n) / (nn - 1.0)))
    }
}

impl Hypergeometric {
    /// The `k`-free part of `saddle`: `(s, flip, E₃, f₃)`, with `s` the
    /// smaller of `n/N`, `(N−n)/N`, `flip` set when it is `(N−n)/N`, and
    /// `(E₃, f₃)` the third binomial `b(n; N, p)`, formed once per `cdf` or `sf` call.
    fn saddle_base(&self) -> (f64, bool, Dd, f64) {
        let (big_n, n) = (self.big_n, self.n);
        let flip = n > big_n - n;
        let (a3, b3, s) = if flip {
            (big_n - n, n, (big_n - n) as f64 / big_n as f64)
        } else {
            (n, big_n - n, n as f64 / big_n as f64)
        };
        let (e3, f3) = binom_saddle(a3 as f64, b3 as f64, s);
        (s, flip, e3, f3)
    }
    /// `(E, F)` with `mass(k) = √F·e^{E}`, `E` a double-double, for `k` in
    /// the support and `n` strictly between 0 and `N`, given `saddle_base()`.
    /// R's `dhyper`: with any `p ∈ (0, 1)`, `f(k) = b(k; K, p)·b(n−k; N−K,
    /// p) / b(n; N, p)` (`b` the binomial pmf; the powers of p and 1 − p
    /// cancel). `p = n/N` puts all three near their modes; each `b` in
    /// `binom_saddle` form with `s` attached to the drawn or undrawn counts.
    /// The three `½·ln(2π·a·b/n)` terms combine into `F`, one product.
    fn saddle(&self, (s, flip, e3, f3): (f64, bool, Dd, f64), k: i64) -> (Dd, f64) {
        let (big_n, kk, n) = (self.big_n, self.k, self.n);
        let (a1, b1) = (k, kk - k);
        let (a2, b2) = (n - k, big_n - kk - n + k);
        let f = |a: i64, b: i64| {
            let (a, b) = if flip { (b, a) } else { (a, b) };
            binom_saddle(a as f64, b as f64, s)
        };
        let ((e1, f1), (e2, f2)) = (f(a1, b1), f(a2, b2));
        (dd_add(dd_add(e1, e2), dd_neg(e3)), f3 / (f1 * f2))
    }
    /// `mass(k)` for `k` in the support, `k_lo < k_hi`, given `saddle_base()`:
    /// `√F·e^{E}` from `saddle`, the exponent kept as a double-double, so the
    /// mass keeps ≈ ε relative even where `ln f(k)` nears −745 (as
    /// `exp(log_mass)` it would carry `|log_mass|·ε`). `value_of_parts`
    /// rescales where `e^{E_hi}` alone would be subnormal.
    fn mass_with(&self, base: (f64, bool, Dd, f64), k: i64) -> f64 {
        let (e, f) = self.saddle(base, k);
        value_of_parts(libm::sqrt(f) * (1.0 + e.1), e.0)
    }
}

impl DiscreteMass for Hypergeometric {
    /// See `mass_with`.
    fn mass(&self, k: i64) -> f64 {
        if k < self.k_lo() || k > self.k_hi() {
            return 0.0;
        }
        if self.k_lo() == self.k_hi() {
            return 1.0;
        }
        self.mass_with(self.saddle_base(), k)
    }
    fn log_mass(&self, k: i64) -> f64 {
        if k < self.k_lo() || k > self.k_hi() {
            return f64::NEG_INFINITY;
        }
        if self.k_lo() == self.k_hi() {
            return 0.0;
        }
        let (e, f) = self.saddle(self.saddle_base(), k);
        e.0 + (e.1 + 0.5 * libm::log(f))
    }
}

impl Hypergeometric {
    /// `Σ mass(j)` over `j ∈ [a, b]`, given `saddle_base()`, compensated
    /// (Neumaier: each addition's rounding error, exact from `two_sum`, is
    /// summed apart and added back). A plain running sum over the up to 10⁵
    /// terms of a wide support gathered 17ε relative in the harness sweep.
    fn sum_mass(&self, base: (f64, bool, Dd, f64), a: i64, b: i64) -> f64 {
        let (s, c) = (a..=b).fold((0.0, 0.0), |(s, c), j| {
            let (t, e) = two_sum(s, self.mass_with(base, j));
            (t, c + e)
        });
        s + c
    }
}

/// Both tails are sums over the support (no closed form), each tail summed
/// over its own terms: `cdf` from `k_lo` up, `sf` from `k_hi` down, so a small
/// tail keeps ~ε relative. Where the other tail is the shorter sum and lies
/// past the mean (so is likely ≤ ½), `1 −` it is taken instead, which is
/// cheaper and keeps ~ε relative while that sum is ≤ ½; a sum above ½ (an
/// atom, as at `N = 100, K = 1, n = 1`, `k = 0`) falls back to the direct one.
impl DiscreteCdf for Hypergeometric {
    fn cdf(&self, k: i64) -> f64 {
        let (lo, hi) = (self.k_lo(), self.k_hi());
        if k < lo {
            return 0.0;
        }
        if k >= hi {
            return 1.0;
        }
        let base = self.saddle_base();
        if self.mean().is_some_and(|m| k as f64 >= m) && hi - k < k - lo + 1 {
            let s = self.sum_mass(base, k + 1, hi);
            if s <= 0.5 {
                return 1.0 - s;
            }
        }
        self.sum_mass(base, lo, k).min(1.0)
    }
    fn sf(&self, k: i64) -> f64 {
        let (lo, hi) = (self.k_lo(), self.k_hi());
        if k < lo {
            return 1.0;
        }
        if k >= hi {
            return 0.0;
        }
        let base = self.saddle_base();
        if self.mean().is_some_and(|m| (k as f64) < m) && k - lo + 1 < hi - k {
            let c = self.sum_mass(base, lo, k);
            if c <= 0.5 {
                return 1.0 - c;
            }
        }
        self.sum_mass(base, k + 1, hi).min(1.0)
    }
    /// `quantile(1)` is `min(n, K)`, the support maximum, as
    /// `scipy.stats.hypergeom.ppf(1)`; every point of the support has
    /// positive mass.
    fn quantile(&self, p: f64) -> Result<i64, StatError> {
        if p == 1.0 {
            return Ok(self.k_hi());
        }
        discrete_bsearch_quantile(self, p, self.k_lo(), self.k_hi())
    }
}

/// Inversion in the order mode, mode+1, mode−1, mode+2, … (sides whose
/// support is used up drop out), one uniform per draw: `u` is reduced by
/// `f(mode) = exp(log_mass(mode))`, `mode = ⌊(n+1)(K+1)/(N+2)⌋`, and then by
/// each term reached through the pmf ratio `f(j+1)/f(j) =
/// (K−j)(n−j)/((j+1)(N−K−n+j+1))` until it is ≤ 0. Expected cost O(sd) steps,
/// against O(support) per bisection probe for inversion through `quantile`, and
/// no term is formed from `f(k_lo)`, which underflows at large `N`. The
/// visiting order is not monotone in `u`, so a draw is not `quantile(u)`. If
/// rounding leaves `u > 0` once both sides are exhausted (support end reached
/// or term underflowed to 0), the draw is the mode.
///
/// Every term inherits the relative error of `f(mode)`, the absolute error of
/// `log_mass(mode)` (at most 2.7e-16·|log_mass(mode)|, see the type doc), so
/// the draw is off by about that much in total variation. `u` is `uniform()` (2³² grid) below
/// variance 10⁶ and `uniform52()` (2⁵²) from there (`FINE_U_FROM`), so a
/// single probability near the mode, ≈ 1/(2.5·σ), is quantized to ≤ 5.9e-7
/// relative below the switch and ≤ 1.7e-6 above it (at most one grid point).
///
/// Validated by `tests/dist_oracle.rs::hypergeometric_sampler_fit` (χ² fit
/// to `mass` and moments, `N` from 20 to 10⁶, plus degenerate supports) and
/// `sampler_moments_and_fit`; pinned draws in `sampler_known_answers`, the `u`
/// switch in `sampler_fine_uniform_above_threshold`.
#[cfg(all(feature = "dist", feature = "rng"))]
impl DiscreteSampler for Hypergeometric {
    fn sample(&self, rng: &mut CommonStatsRng) -> i64 {
        let (lo, hi) = (self.k_lo(), self.k_hi());
        // i128: `(n+1)(K+1)` overflows i64 once n, K ≳ 3e9.
        let mode = ((self.n as i128 + 1) * (self.k as i128 + 1) / (self.big_n as i128 + 2)) as i64;
        let mode = mode.clamp(lo, hi);
        let f_mode = libm::exp(self.log_mass(mode));
        let fine = self.variance().is_some_and(|v| v >= FINE_U_FROM);
        let mut u = if fine { rng.uniform52() } else { rng.uniform() } - f_mode;
        if u <= 0.0 {
            return mode;
        }
        // `rest = N − K − n` makes the ratio denominators `(j+1)(rest+j+1)`.
        let (kk, n) = (self.k as f64, self.n as f64);
        let rest = (self.big_n - self.k - self.n) as f64;
        let (mut up, mut down) = (mode, mode);
        let (mut f_up, mut f_down) = (f_mode, f_mode);
        loop {
            let mut stepped = false;
            if up < hi && f_up > 0.0 {
                let j = up as f64;
                f_up *= (kk - j) * (n - j) / ((j + 1.0) * (rest + j + 1.0));
                up += 1;
                u -= f_up;
                if u <= 0.0 {
                    return up;
                }
                stepped = true;
            }
            if down > lo && f_down > 0.0 {
                // f(j−1)/f(j) = j·(rest+j) / ((K−j+1)(n−j+1)).
                let j = down as f64;
                f_down *= j * (rest + j) / ((kk - j + 1.0) * (n - j + 1.0));
                down -= 1;
                u -= f_down;
                if u <= 0.0 {
                    return down;
                }
                stepped = true;
            }
            if !stepped {
                return mode;
            }
        }
    }
}
