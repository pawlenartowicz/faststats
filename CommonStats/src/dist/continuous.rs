//! The 12 continuous distributions of the `dist` suite.
//!
//! Quantile strategy for t, χ², F, Gamma, and the inverse Gaussian:
//! Cornish–Fisher, Wilson–Hilferty, or (inverse Gaussian) normal-term seed
//! (Abramowitz & Stegun §26.7; Wilson & Hilferty 1931), then one shared
//! bracketed Newton solver, `bracketed_newton`, on the log of the smaller tail
//! mass: `ln cdf` below the median and `ln sf` above it. t, by symmetry, always
//! solves on its upper half: `ln sf`, or near the median the central mass
//! `P(0 < X < x)`. Each residual is formed from its own small mass, never as
//! `1 −` a near-1 one.
//!
//! <!-- accuracy:begin -->
//! ## Accuracy
//!
//! Worst error ratio `r` measured by `scripts/accuracy_sweep.py` against
//! mpmath (81453 points): `r` = relative error / (max(κ, 1)·ε), κ the
//! condition number over all real inputs, ε = 2⁻⁵²; for a log output the
//! absolute error over the sensitivity of the log. `r ≤ 10`: at the accuracy
//! the problem allows; `(review)` up to 1000; `(bug)` above it, a structural error;
//! `(hard)` a probability outside [0, 1], a NaN, a non-monotone cdf or a
//! quantile outside the support. Generated from
//! `scripts/accuracy_baseline.json` by `--update-baseline`; not edited by hand.
//!
//! | Distribution | cdf | sf | pdf | log_density | quantile | isf |
//! |---|---|---|---|---|---|---|
//! | Normal | 0.54 | 0.58 | 0.96 | 1.1 | 1.1 | 0.97 |
//! | StudentT | 2.2 | 2.2 | 2.3e2 (review) | 6.2 | 4.2 | 4.2 |
//! | ChiSquared | 1.8 | 3.1 | 5.8e2 (review) | 2.0e2 (review) | 1.1 | 2.7 |
//! | FisherF | 1.8e298 (hard) | 1.8e298 (hard) | 2.0e3 (bug) | 1.7e2 (review) | inf (bug) | inf (bug) |
//! | Uniform | 0.46 | 0.63 | 0.44 | 0.32 | 0.47 | 0.71 |
//! | Exponential | 0.43 | 0.46 | 0.63 | 0.77 | 0.5 | 0.57 |
//! | Cauchy | 0.57 | 0.84 | 1.1 | 1 | 0.51 | 0.51 |
//! | Weibull | 0.55 | 0.36 | 1.3e2 (review) | 2.8 | 0.61 | 0.69 |
//! | LogNormal | 0.77 | 0.55 | 1.4 | 0.95 | 0.76 | 0.76 |
//! | Gamma | 1.3 | 3.8 | 5.4e2 (review) | 6.8 | 1.8 | 0.98 |
//! | Beta | 3.3 | 14 (review) | 2.0e2 (review) | 13 (review) | 5.5 | 2.7 |
//! | InverseGaussian | 1.1 | 2.5 | 1.5e2 (review) | 3.3 | 22 (review) | 32 (review) |
//! <!-- accuracy:end -->

#[cfg(feature = "rng")]
use crate::dist::Sampler;
use crate::dist::{
    Bound, ContinuousCdf, ContinuousDensity, Distribution, gamma_log_density,
    gamma_log_density_consts, gamma_log_density_with, norm_quantile,
};
use crate::error::StatError;
#[cfg(feature = "rng")]
use crate::rng::CommonStatsRng;
use crate::special::elementary::erfcx;
use crate::special::saddle::{beta_lambda, beta_saddle_dev, ln_beta_saddle_const, log1pmx};

/// Normal (Gaussian) distribution `N(μ, σ)`.
///
/// Convention: `σ` is the **standard deviation** (not variance). Density
/// `φ((x−μ)/σ)/σ`. `kurtosis` is excess (0). Matches `scipy.stats.norm(μ, σ)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal {
    pub(crate) mean: f64,
    pub(crate) sd: f64,
}

impl Normal {
    /// Construct `N(mean, sd)`.
    ///
    /// # Parameters
    /// - `mean`: location `μ` (any finite real).
    /// - `sd`: standard deviation `σ > 0`.
    ///
    /// # Errors
    /// `DomainError` if `sd ≤ 0` or either argument is non-finite.
    ///
    /// ```
    /// use commonstats::dist::continuous::Normal;
    /// use commonstats::dist::{ContinuousCdf, Distribution};
    /// let n = Normal::new(0.0, 1.0).unwrap();
    /// assert!((n.cdf(0.0) - 0.5).abs() < 1e-15);
    /// assert_eq!(n.quantile(0.0).unwrap(), f64::NEG_INFINITY);
    /// assert!(Normal::new(0.0, -1.0).is_err());
    /// ```
    pub fn new(mean: f64, sd: f64) -> Result<Self, StatError> {
        if !mean.is_finite() || !sd.is_finite() || sd <= 0.0 {
            return Err(StatError::DomainError("Normal: sd must be finite and > 0"));
        }
        Ok(Normal { mean, sd })
    }
}

impl Distribution for Normal {
    fn support_min(&self) -> Bound {
        Bound::NegInfinity
    }
    fn support_max(&self) -> Bound {
        Bound::PosInfinity
    }
    fn mean(&self) -> Option<f64> {
        Some(self.mean)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.sd * self.sd)
    }
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(0.0)
    }
    fn entropy(&self) -> Option<f64> {
        Some(
            0.5 * libm::log(2.0 * core::f64::consts::PI * core::f64::consts::E)
                + libm::log(self.sd),
        )
    }
}

impl ContinuousDensity for Normal {
    fn density(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.sd;
        libm::exp(-0.5 * z * z) / (self.sd * libm::sqrt(2.0 * core::f64::consts::PI))
    }
    fn log_density(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.sd;
        -0.5 * z * z - libm::log(self.sd) - 0.5 * libm::log(2.0 * core::f64::consts::PI)
    }
}

impl ContinuousCdf for Normal {
    fn cdf(&self, x: f64) -> f64 {
        // 0.5·erfc(−(x−μ)/(σ√2)) — no 1−cdf cancellation in either tail.
        0.5 * crate::special::erfc(-(x - self.mean) / (self.sd * core::f64::consts::SQRT_2))
    }
    fn sf(&self, x: f64) -> f64 {
        0.5 * crate::special::erfc((x - self.mean) / (self.sd * core::f64::consts::SQRT_2))
    }
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        Ok(self.mean + self.sd * norm_quantile(p))
    }
    /// Symmetry: `isf(q) = μ − σ·Φ⁻¹(q)`.
    fn isf(&self, q: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(StatError::ProbabilityOutOfRange(q));
        }
        Ok(self.mean - self.sd * norm_quantile(q))
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for Normal {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        self.mean + self.sd * norm_quantile(rng.uniform())
    }
}

/// Student's t distribution with `df` degrees of freedom.
///
/// Convention: standard (location 0, scale 1). Mean defined for `df>1`,
/// variance `df>2`, skewness `df>3`, excess kurtosis `df>4`. CDF via the
/// regularized incomplete beta; quantile via Cornish–Fisher seed + Newton,
/// which is skipped at large df where the seed's truncation error is below f64
/// resolution.
/// Matches `scipy.stats.t(df)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentT {
    df: f64,
}

impl StudentT {
    /// Construct `t(df)` with `df > 0`.
    ///
    /// # Errors
    /// `DomainError` if `df` is non-finite or `≤ 0`.
    ///
    /// ```
    /// use commonstats::dist::continuous::StudentT;
    /// use commonstats::dist::ContinuousCdf;
    /// let t = StudentT::new(5.0).unwrap();
    /// assert_eq!(t.quantile(0.5).unwrap(), 0.0);
    /// assert!(StudentT::new(0.0).is_err());
    /// ```
    pub fn new(df: f64) -> Result<Self, StatError> {
        if !df.is_finite() || df <= 0.0 {
            return Err(StatError::DomainError(
                "StudentT: df must be finite and > 0",
            ));
        }
        Ok(StudentT { df })
    }
}

impl Distribution for StudentT {
    fn support_min(&self) -> Bound {
        Bound::NegInfinity
    }
    fn support_max(&self) -> Bound {
        Bound::PosInfinity
    }
    fn mean(&self) -> Option<f64> {
        if self.df > 1.0 { Some(0.0) } else { None }
    }
    fn variance(&self) -> Option<f64> {
        if self.df > 2.0 {
            Some(self.df / (self.df - 2.0))
        } else {
            None
        }
    }
    fn skewness(&self) -> Option<f64> {
        if self.df > 3.0 { Some(0.0) } else { None }
    }
    fn kurtosis(&self) -> Option<f64> {
        if self.df > 4.0 {
            Some(6.0 / (self.df - 4.0))
        } else {
            None
        }
    }
}

impl ContinuousDensity for StudentT {
    fn density(&self, x: f64) -> f64 {
        libm::exp(self.log_density(x))
    }
    fn log_density(&self, x: f64) -> f64 {
        self.log_density_with(self.ln_norm_const(), x)
    }
}

impl StudentT {
    /// `−½ ln df − ln B(½, df/2)`, the log normalizing constant of
    /// pdf = (1+x²/df)^{−(df+1)/2} / (√df·B(½, df/2)) (Abramowitz & Stegun
    /// 26.7.1, rewritten through the Beta function). `lgamma((df+1)/2) −
    /// lgamma(df/2)` cancels near the ε of each term at huge df (about 1e-5
    /// absolute error at `df = 1e10`); `lbeta` keeps the difference to double
    /// precision instead of forming the two large `lgamma`s. Split out so the
    /// quantile solver forms it once, not per Newton step.
    fn ln_norm_const(&self) -> f64 {
        -0.5 * libm::log(self.df) - crate::special::lbeta(0.5, 0.5 * self.df)
    }
    /// `log_density` given `c = ln_norm_const()`.
    fn log_density_with(&self, c: f64, x: f64) -> f64 {
        let df = self.df;
        // ln(1 + x²/df). Where `x²/df` overflows it is `2 ln|x| − ln df +
        // log1p(df/x²)`, and the last term is below 1e-308.
        let u = x / df * x;
        let ln_kernel = if u.is_finite() {
            libm::log1p(u)
        } else {
            2.0 * libm::log(libm::fabs(x)) - libm::log(df)
        };
        c - 0.5 * (df + 1.0) * ln_kernel
    }
    /// `(z, y) = (df/(df + x²), x²/(df + x²))`, each by its own division; `z`
    /// is 0 and `y` NaN when `x²` overflows.
    fn beta_args(&self, x: f64) -> (f64, f64) {
        let (df, x2) = (self.df, x * x);
        (df / (df + x2), x2 / (df + x2))
    }
}

impl ContinuousCdf for StudentT {
    fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return x;
        }
        if x.is_infinite() {
            return if x > 0.0 { 1.0 } else { 0.0 };
        }
        let (central, tail) = self.halves(x);
        if x >= 0.0 { 0.5 + central } else { tail }
    }
    /// Mirror of `cdf`: the small tail is returned directly, never as `1 − cdf`.
    fn sf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return x;
        }
        if x.is_infinite() {
            return if x > 0.0 { 0.0 } else { 1.0 };
        }
        let (central, tail) = self.halves(x);
        if x >= 0.0 { tail } else { 0.5 + central }
    }
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        // `p < 0.5`, not `<=`: at the median `-solve_tail(0.5)` would be `-0.0`.
        Ok(if p < 0.5 {
            -self.solve_tail(p)
        } else {
            self.solve_tail(1.0 - p)
        })
    }
    /// Symmetry: `isf(q) = −quantile(q)`, solved on the tail mass `q` directly.
    fn isf(&self, q: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(StatError::ProbabilityOutOfRange(q));
        }
        if q == 0.0 {
            return Ok(f64::INFINITY);
        }
        if q == 1.0 {
            return Ok(f64::NEG_INFINITY);
        }
        Ok(if q <= 0.5 {
            self.solve_tail(q)
        } else {
            -self.solve_tail(1.0 - q)
        })
    }
}

impl StudentT {
    /// The mass on one side of 0, split at `|x|`: `(central, tail)` with
    /// `central = P(0 < X < |x|) = ½·I_y(½, df/2)`, `tail = P(X > |x|) =
    /// ½·I_z(df/2, ½)`, `y = x²/(df + x²)`, `z = 1 − y`, summing to ½. `y` and
    /// `z` are each formed by their own division and passed together to the
    /// incomplete-beta kernel, which evaluates whichever of the two masses lies
    /// below its reflection point and gives the other as ½ minus it. Neither
    /// is ever recovered from `1 −` a rounded near-1 argument: near `x = 0`
    /// that would lose every digit of `central` (and of `cdf(x) − ½`), and at
    /// large df, where `z` sits within `x²/df` of 1, digits of the tail. Where
    /// `z` is subnormal or 0 (`x²` overflowed) the tail is the closed form of
    /// `far_tail_ln`, which takes `ln z` from `x` and `df`.
    fn halves(&self, x: f64) -> (f64, f64) {
        let a = 0.5 * self.df;
        let (z, y) = self.beta_args(x);
        if z < f64::MIN_POSITIVE {
            if let Some(ln_tail) = self.far_tail_ln(x, z) {
                let tail = libm::exp(ln_tail);
                return (0.5 - tail, tail);
            }
            if z == 0.0 {
                // `x²` overflowed and the far-tail form does not apply, so
                // df ≳ 1e145 and the tail `≈ (1 + x²/df)^{−df/2}` is below the
                // f64 range.
                return (0.5, 0.0);
            }
        }
        let (m, e, central_side) = crate::special::incomplete::betai_parts(a, 0.5, z, y);
        let v = crate::special::incomplete::value_of_parts(0.5 * m, e);
        if central_side {
            (v, 0.5 - v)
        } else {
            (0.5 - v, v)
        }
    }

    /// `ln P(X > |x|) = ln(½·I_z(df/2, ½))`, `z = df/(df + x²)` (from
    /// `beta_args`), in the far tail; see `ln_betai_small_z`. `ln z ≈ ln df −
    /// 2 ln|x|` drops `ln(1 + df/x²) ≈ z`, below resolution there, and never
    /// forms `x²`.
    fn far_tail_ln(&self, x: f64, z: f64) -> Option<f64> {
        ln_betai_small_z(0.5 * self.df, 0.5, z, || {
            libm::log(self.df) - 2.0 * libm::log(libm::fabs(x))
        })
        .map(|l| l - core::f64::consts::LN_2)
    }

    /// `x ≥ 0` with upper-tail mass `t ∈ (0, 0.5]`: `P(X > x) = t`. Both
    /// `quantile` and `isf` map their argument onto this by symmetry, so the
    /// Newton residual is formed from the small mass on its side — the tail
    /// `t`, or near `t = ½` the central mass `½ − t` — and never from `1 − cdf`.
    fn solve_tail(&self, t: f64) -> f64 {
        if t == 0.5 {
            return 0.0;
        }
        let df = self.df;
        // Initial guess (Cornish–Fisher expansion, Abramowitz & Stegun §26.7), then Newton.
        let x = if df > 200.0 {
            let z = -norm_quantile(t);
            let z2 = z * z;
            let x = z
                + (z * (z2 + 1.0)) / (4.0 * df)
                + (z * (5.0 * z2 * z2 + 16.0 * z2 + 3.0)) / (96.0 * df * df);
            // The first omitted term, g₃/df³ with g₃ = (3z⁷ + 19z⁵ + 17z³ −
            // 15z)/384 (A&S 26.7.5), bounds the seed's truncation error. Below
            // f64 resolution the seed is the root to within its own rounding;
            // Newton (2 to 5 `betai` evaluations, ~10× the cost at df ≥ 1e7)
            // leaves it no closer: both are within 1.4e-15 of mpmath at
            // df = 1e5…1e12.
            let g3 = z * (((3.0 * z2 + 19.0) * z2 + 17.0) * z2 - 15.0) / 384.0;
            if libm::fabs(g3) / (df * df * df) <= f64::EPSILON * libm::fabs(x) {
                return x;
            }
            x
        } else if (df - 1.0).abs() < 1e-9 {
            1.0 / libm::tan(core::f64::consts::PI * t)
        } else if (df - 2.0).abs() < 1e-9 {
            let alpha = 4.0 * t * (1.0 - t);
            libm::sqrt(2.0 / alpha - 2.0)
        } else {
            -norm_quantile(t)
        };
        let c = self.ln_norm_const();
        let ln_pdf = |x: f64| self.log_density_with(c, x);
        if t > 0.25 {
            // Central mass `½ − t`, exact for `t ∈ [¼, ½]` (Sterbenz). Near t = ½
            // the root is near 0 and `sf(x) − t` carries the absolute ~5e-17
            // rounding of a number near ½, a relative error of ~5e-17/(½ − t) in
            // x; the residual on the central mass keeps full relative precision.
            // It rises with x like a cdf, `∝ x` near 0.
            bracketed_newton(
                x,
                0.5 - t,
                false,
                true,
                |x| libm::log(self.halves(x).0),
                ln_pdf,
            )
        } else {
            // `ln_betai` keeps the log of a tail below the normal range, where
            // `ln` of the tail value would see only its subnormal bits.
            let ln_tail = |x: f64| {
                let (z, y) = self.beta_args(x);
                self.far_tail_ln(x, z).unwrap_or_else(|| {
                    crate::special::incomplete::ln_betai(0.5 * df, 0.5, z, y)
                        - core::f64::consts::LN_2
                })
            };
            bracketed_newton(x, t, true, true, ln_tail, ln_pdf)
        }
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for StudentT {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        self.quantile(rng.uniform())
            .expect("uniform() ∈ (0,1) is a valid quantile arg")
    }
}

/// Chi-squared distribution `χ²(k)`.
///
/// Convention: `k` degrees of freedom (`> 0`, real-valued). `χ²(k) = Gamma(k/2,
/// rate 1/2)`. CDF `P(k/2, x/2)`; sf via `Q`. Quantile = `Gamma`'s:
/// Wilson–Hilferty seed + Newton on `ln P` or `ln Q`, whichever tail is smaller.
/// `density(0)` is `+∞` for `k < 2` (singular at 0), `½` at `k = 2` and `0`
/// for `k > 2`. Matches
/// `scipy.stats.chi2(k)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquared {
    k: f64,
}

impl ChiSquared {
    /// Construct `χ²(k)` with `k > 0`.
    ///
    /// # Errors
    /// `DomainError` if `k` is non-finite or `≤ 0`.
    ///
    /// ```
    /// use commonstats::dist::continuous::ChiSquared;
    /// use commonstats::dist::Distribution;
    /// let c = ChiSquared::new(4.0).unwrap();
    /// assert_eq!(c.mean(), Some(4.0));
    /// assert!(ChiSquared::new(0.0).is_err());
    /// ```
    pub fn new(k: f64) -> Result<Self, StatError> {
        if !k.is_finite() || k <= 0.0 {
            return Err(StatError::DomainError(
                "ChiSquared: k must be finite and > 0",
            ));
        }
        Ok(ChiSquared { k })
    }
}

impl Distribution for ChiSquared {
    fn support_min(&self) -> Bound {
        Bound::Finite(0.0)
    }
    fn support_max(&self) -> Bound {
        Bound::PosInfinity
    }
    fn mean(&self) -> Option<f64> {
        Some(self.k)
    }
    fn variance(&self) -> Option<f64> {
        Some(2.0 * self.k)
    }
    fn skewness(&self) -> Option<f64> {
        Some(libm::sqrt(8.0 / self.k))
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(12.0 / self.k)
    }
}

impl ContinuousDensity for ChiSquared {
    fn density(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        libm::exp(self.log_density(x))
    }
    fn log_density(&self, x: f64) -> f64 {
        // χ²(k) = Gamma(shape k/2, rate 1/2).
        gamma_log_density(0.5 * self.k, 0.5, x)
    }
}

/// All four are `Gamma(k/2, rate ½)`'s (`as_gamma`): `P(k/2, x/2)`, `Q(k/2,
/// x/2)`, and its quantile solver.
impl ContinuousCdf for ChiSquared {
    fn cdf(&self, x: f64) -> f64 {
        self.as_gamma().cdf(x)
    }
    fn sf(&self, x: f64) -> f64 {
        self.as_gamma().sf(x)
    }
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        self.as_gamma().quantile(p)
    }
    /// Newton on `ln sf(x) − ln q` (`ln_gammq`) for `q ≤ 0.5`; see `Gamma::isf`.
    fn isf(&self, q: f64) -> Result<f64, StatError> {
        self.as_gamma().isf(q)
    }
}

impl ChiSquared {
    /// The same distribution as `Gamma(k/2, rate ½)`, whose cdf, sf, quantile
    /// solver (Wilson–Hilferty seed on `χ²(2α)`, then `bracketed_newton`) and
    /// sampler are used.
    fn as_gamma(&self) -> Gamma {
        Gamma {
            shape: 0.5 * self.k,
            rate: 0.5,
        }
    }
}

/// `Gamma(k/2, rate ½)`'s Marsaglia–Tsang sampler (a variable number of
/// uniforms per draw), not inversion: the same draws as `2·Gamma(k/2, 1)`.
#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for ChiSquared {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        self.as_gamma().sample(rng)
    }
}

/// Root `x > 0` of `mass(x) = target`, the quantile solver of t, χ², F, Gamma,
/// and the inverse Gaussian. `mass` is the cdf (`upper = false`) or the sf
/// (`upper = true`); callers pass the smaller tail, `target ≤ ½`, so the
/// residual is formed from a small mass and never from `1 −` a near-1 one.
///
/// Newton runs on `ln mass(x) − ln target`, `d ln mass/dx = ±pdf/mass`, in `x`
/// or (`in_ln_x`) in `ln x`. `ln mass` is asymptotically linear in `x` in an
/// exponential tail (χ², Gamma, and inverse Gaussian upper tails) and in `ln x`
/// in a power-law one (t and F upper tails, the χ², Gamma, and F lower tails
/// `∝ x^{shape}`), where a step on the raw mass moves `x` by at most about
/// `mass/pdf`, or grows it by a factor `~(1 + 1/exponent)`, and exhausts the
/// iterations short of a deep root. `mass/pdf` is formed in logs, as either can
/// underflow.
///
/// Newton alone diverges from a poor start: a step can land where the mass or
/// the pdf underflows, and where the pdf is tiny and `mass ≈ 1` the ratio
/// multiplies a step by `e^{100}` and more. So every evaluated `x` tightens a
/// bracket `lo < root < hi` (`lo = 0`, `hi = ∞` until found). A step that
/// leaves the bracket or is not finite is replaced by the geometric midpoint
/// `√lo·√hi`, bisection in `ln x`: ~57 halvings narrow the whole f64 range to
/// the stopping tolerance. Once both ends are found, so is a step longer than
/// half the step before the last one (Numerical Recipes §9.4, `rtsafe`), which
/// stops Newton cycling where the mass carries rounding noise, as `betai` does
/// at large shapes.
/// While one side is open, `x` jumps to 1 or squares away from it, doubling
/// `|ln x|`, which reaches either end of the range in ~12 steps.
///
/// Stops on a Newton step, or a bisection step, below 1e-14 relative to `x`
/// (not to `1 + x`: roots sit many decades below 1); the result never leaves
/// the bracket. Returns ∞ when `f64::MAX` still lies below the root and 0 when
/// the smallest subnormal still lies above it, and NaN when, with a side still
/// open, the mass is NaN at the point the next jump would go to (so at every
/// point it would try). That test, like every residual,
/// reads `ln_mass`, so callers form it in logs where the mass or its argument
/// leaves the normal range: the t, χ², F and Gamma tails through `ln_betai`,
/// `ln_gammp` and `ln_gammq`, and `InverseGaussian::ln_cdf` and `ln_sf`.
/// `ln` of a mass rounded to the subnormal grid keeps only its remaining bits
/// (~12 at 1e-320) and moves in steps, and a mass that underflows to 0 puts
/// its point below the root.
fn bracketed_newton(
    seed: f64,
    target: f64,
    upper: bool,
    in_ln_x: bool,
    ln_mass: impl Fn(f64) -> f64,
    ln_pdf: impl Fn(f64) -> f64,
) -> f64 {
    const MIN_SUBNORMAL: f64 = 5e-324;
    const TOL: f64 = 1e-14;
    let ln_target = libm::log(target);
    let (mut lo, mut hi) = (0.0, f64::INFINITY);
    // Relative sizes of the last two steps, for the `rtsafe` progress test.
    let (mut step_last, mut step_before) = (f64::INFINITY, f64::INFINITY);
    let mut x = if seed > 0.0 && seed < f64::INFINITY {
        seed
    } else {
        1.0
    };
    for _ in 0..200 {
        let ln_m = ln_mass(x);
        let resid = ln_m - ln_target;
        if resid == 0.0 {
            return x;
        }
        // The cdf rises with x and the sf falls. A NaN mass (`betai` out of
        // range at huge shapes) says nothing about the side, so it leaves the
        // bracket alone.
        if !resid.is_nan() {
            if (resid < 0.0) != upper {
                lo = x;
            } else {
                hi = x;
            }
        }
        let ln_scale = ln_m - ln_pdf(x) - if in_ln_x { libm::log(x) } else { 0.0 };
        let step = resid * libm::exp(ln_scale);
        let step = if upper { step } else { -step };
        let newton = if in_ln_x {
            x * libm::exp(step)
        } else {
            x + step
        };
        let newton_step = libm::fabs(newton - x) / x;
        if newton_step <= TOL {
            return if newton >= lo && newton <= hi {
                newton
            } else {
                x
            };
        }
        let closed = lo > 0.0 && hi < f64::INFINITY;
        // False for a NaN step.
        let next = if newton > lo && newton < hi && !(closed && newton_step > 0.5 * step_before) {
            newton
        } else if hi == f64::INFINITY {
            if lo == f64::MAX {
                return f64::INFINITY;
            }
            if lo < 1.0 {
                1.0
            } else {
                (lo * lo.max(2.0)).min(f64::MAX)
            }
        } else if lo == 0.0 {
            if hi == MIN_SUBNORMAL {
                return 0.0;
            }
            if hi > 1.0 {
                1.0
            } else {
                (hi * hi.min(0.5)).max(MIN_SUBNORMAL)
            }
        } else {
            libm::sqrt(lo) * libm::sqrt(hi)
        };
        // A NaN residual leaves the bracket as it was, so with a side still
        // open the jump lands on this same `x` again: the mass is NaN wherever
        // the solver can go, and there is no root to report.
        if resid.is_nan() && next == x && !closed {
            return f64::NAN;
        }
        (step_before, step_last) = (step_last, libm::fabs(next - x) / x);
        // Only a closed bracket makes a small step mean convergence: an
        // open-side jump to 1 from an `x` within 1e-14 of 1 is also small.
        if step_last <= TOL && closed {
            return next;
        }
        x = next;
    }
    x
}

/// `ln I_z(a, b)` from the leading term of
/// `I_z(a, b) = z^a (1 − z)^b / (a·B(a, b)) · ₂F₁(a + b, 1; a + 1; z)`
/// (DLMF 8.17.7), when `z·(1 + a + b) < 1e-17`: the dropped factor is
/// `1 + O(z·(a + b))`, below f64 resolution. `None` otherwise. Takes `ln z`, so
/// it holds where `z` is subnormal or underflows, or its denominator
/// overflows, none of which `ln_betai` (which takes `z`) can represent.
/// Used by the far upper tails of `StudentT` and `FisherF` and the far lower
/// tail of `FisherF`.
///
/// `z` is the caller's rounded `z` (or 0 where it cannot be trusted) and
/// `ln_z` its log, formed only when needed: a normal `z` with `z·(1 + a + b)`
/// clearly above the cutoff returns `None` without the logs. The margin
/// `1 + 1e-9` covers the rounding of `z` and of the product, far inside the
/// 6e-7 gap between the cutoff and 1e-17.
///
/// For `a < 1`, `ln(a·B(a, b))` is `O(a)` (≈ `−a·(γ + ψ(b))` as a → 0) and
/// `ln a + ln B(a, b)` would leave it `|ln a|·ε` absolute: at `a = 5e-21`
/// it rounds to 0 and the cdf to 1. There it is `ln Γ(1 + a) + ln(Γ(b)/Γ(a +
/// b))`, each term to relative precision (`ln_gamma_1p`, `ln_gamma_ratio`).
fn ln_betai_small_z(a: f64, b: f64, z: f64, ln_z: impl FnOnce() -> f64) -> Option<f64> {
    if z >= f64::MIN_POSITIVE && z * (1.0 + a + b) > 1.000_000_001e-17 {
        return None;
    }
    let ln_z = ln_z();
    if ln_z + libm::log(1.0 + a + b) > LN_SERIES_CUTOFF {
        return None;
    }
    let ln_a_beta = if a < 1.0 {
        crate::special::elementary::ln_gamma_1p(a) + ln_gamma_ratio(a, b)
    } else {
        libm::log(a) + crate::special::lbeta(a, b)
    };
    Some(a * ln_z - ln_a_beta)
}

/// `ln(Γ(b)/Γ(a + b))` for `0 < a < 1`, `b > 0`, to relative precision also
/// where it is `O(a)`: below `b = 8` shifted up by `Γ(c + 1) = c·Γ(c)`,
/// `Σ_{k<n} ln(1 + a/(b + k)) + algdiv(a, b + n)`, `n` the least integer with
/// `b + n ≥ 8` (the recurrence DiDonato & Morris 1992 apply in `betaln` to
/// reach `algdiv`'s range). Each term keeps its own relative precision, and
/// the positive sum and the negative `algdiv` are each `O(a)` where the value
/// is. The ratio of `1/Γ` values (`rgamma_sum`/`rgamma_small`) is within `~a`
/// of 1 and known only to ε absolute, which its log would keep. Within 3e-16
/// relative against mpmath (a ∈ [5e-151, 0.9], b ∈ [1e-3, 8)), and ~3e-21
/// absolute at `a = 1e-5` near ψ's zero `b ≈ 1.4616`, where the value is
/// `O(a²)`.
fn ln_gamma_ratio(a: f64, b: f64) -> f64 {
    let mut sum = 0.0;
    let mut c = b;
    while c < 8.0 {
        sum += libm::log1p(a / c);
        c += 1.0;
    }
    sum + crate::special::elementary::algdiv(a, c)
}

/// Log of the argument below which an incomplete beta or gamma series is its
/// leading term to f64 resolution (`ln_betai_small_z`, `Gamma::ln_cdf`):
/// ≈ ln(1e-17), 6.3e-7 below it. The exact value only places the switch, and
/// is kept so the branch taken at the boundary does not move.
const LN_SERIES_CUTOFF: f64 = -39.143_947_206_188_2;

/// `quantile(p)` (`upper = false`) or `isf(q)` (`upper = true`) of a
/// distribution on `(0, ∞)`: range check, the endpoints (`0` and `∞`, swapped
/// for `isf`), and otherwise `solve(target, side)` on whichever tail mass is
/// ≤ 0.5. `1 − t` is exact in f64 for `t ∈ [0.5, 1]` (Sterbenz), so flipping
/// the tail costs nothing, while a Newton residual formed on the near-1 mass
/// would keep only ~8 digits. Serves `FisherF`, `Gamma` (and so `ChiSquared`)
/// and `InverseGaussian`.
fn positive_quantile(
    t: f64,
    upper: bool,
    solve: impl Fn(f64, bool) -> f64,
) -> Result<f64, StatError> {
    if !(0.0..=1.0).contains(&t) {
        return Err(StatError::ProbabilityOutOfRange(t));
    }
    if t == 0.0 {
        return Ok(if upper { f64::INFINITY } else { 0.0 });
    }
    if t == 1.0 {
        return Ok(if upper { 0.0 } else { f64::INFINITY });
    }
    Ok(if t > 0.5 {
        solve(1.0 - t, !upper)
    } else {
        solve(t, upper)
    })
}

/// Fisher–Snedecor F distribution `F(dfn, dfd)`.
///
/// Convention: `dfn` numerator df, `dfd` denominator df (both `> 0`). Mean
/// defined for `dfd>2`, variance `dfd>4`. CDF
/// `I_{dfn·x/(dfd+dfn·x)}(dfn/2, dfd/2)`; sf `I_{dfd/(dfd+dfn·x)}(dfd/2, dfn/2)`,
/// with both arguments formed directly and passed together to the
/// incomplete-beta kernel, which runs on the side whose argument is below its
/// reflection point; the other side is 1 minus it.
/// Quantile = Wilson–Hilferty seed + Newton on `ln cdf` or `ln sf`, whichever
/// tail is smaller.
/// Matches `scipy.stats.f(dfn, dfd)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FisherF {
    dfn: f64,
    dfd: f64,
}

impl FisherF {
    /// Construct `F(dfn, dfd)` with both `> 0`.
    ///
    /// # Errors
    /// `DomainError` if either df is non-finite or `≤ 0`.
    ///
    /// ```
    /// use commonstats::dist::continuous::FisherF;
    /// use commonstats::dist::ContinuousCdf;
    /// let f = FisherF::new(5.0, 10.0).unwrap();
    /// assert!(f.cdf(1.0) > 0.0 && f.cdf(1.0) < 1.0);
    /// assert!(FisherF::new(0.0, 10.0).is_err());
    /// ```
    pub fn new(dfn: f64, dfd: f64) -> Result<Self, StatError> {
        if !dfn.is_finite() || !dfd.is_finite() || dfn <= 0.0 || dfd <= 0.0 {
            return Err(StatError::DomainError(
                "FisherF: dfn and dfd must be finite and > 0",
            ));
        }
        Ok(FisherF { dfn, dfd })
    }
}

impl Distribution for FisherF {
    fn support_min(&self) -> Bound {
        Bound::Finite(0.0)
    }
    fn support_max(&self) -> Bound {
        Bound::PosInfinity
    }
    fn mean(&self) -> Option<f64> {
        if self.dfd > 2.0 {
            Some(self.dfd / (self.dfd - 2.0))
        } else {
            None
        }
    }
    fn variance(&self) -> Option<f64> {
        let (m, n) = (self.dfn, self.dfd);
        if n > 4.0 {
            Some(2.0 * n * n * (m + n - 2.0) / (m * (n - 2.0) * (n - 2.0) * (n - 4.0)))
        } else {
            None
        }
    }
}

impl ContinuousDensity for FisherF {
    fn density(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        libm::exp(self.log_density(x))
    }
    fn log_density(&self, x: f64) -> f64 {
        self.log_density_with(self.ln_density_consts(), x)
    }
}

impl FisherF {
    /// The `x`-free terms of `log_density`, formed once per quantile solve:
    /// `(k, c₁, c₂)` for the branch `log_density_with` takes, with `a = dfn/2`,
    /// `b = dfd/2`. Both `< 8`: `(−ln B(a, b), a·ln(dfn/dfd), 0)`. Both `≥ 8`:
    /// `(½·ln(ab/((a + b)·2π)) − bcorr(a, b), ln x₀, ln y₀)`, `x₀ = a/(a + b)`,
    /// `y₀ = b/(a + b)`. One `≥ 8`: `(fisherf_mixed_const(small, large), 0, 0)`.
    fn ln_density_consts(&self) -> (f64, f64, f64) {
        let (a, b) = (0.5 * self.dfn, 0.5 * self.dfd);
        if a < 8.0 && b < 8.0 {
            (
                -crate::special::lbeta(a, b),
                a * libm::log(self.dfn / self.dfd),
                0.0,
            )
        } else {
            self.ln_density_consts_large_df()
        }
    }
    /// `ln_density_consts` where `dfn` or `dfd` is `≥ 16`, out of line as
    /// `log_density_large_df`.
    #[inline(never)]
    fn ln_density_consts_large_df(&self) -> (f64, f64, f64) {
        let (a, b) = (0.5 * self.dfn, 0.5 * self.dfd);
        if a >= 8.0 && b >= 8.0 {
            (
                ln_beta_saddle_const(a, b),
                -libm::log1p(b / a),
                -libm::log1p(a / b),
            )
        } else if a >= 8.0 {
            (fisherf_mixed_const(b, a), 0.0, 0.0)
        } else {
            (fisherf_mixed_const(a, b), 0.0, 0.0)
        }
    }
    /// `log_density` given `consts = ln_density_consts()`. The density is
    /// `y^a·z^b/(x·B(a, b))` with `u = dfn·x/dfd`, `y = u/(1 + u)`,
    /// `z = 1/(1 + u)`, `a = dfn/2`, `b = dfd/2`. Its log is the sum of terms
    /// of size `a·ln` and `b·ln` whose total is `O(ln a + ln b)`, so a direct
    /// sum carries `ε·a·|ln|` (2e5 absolute at `dfn = 1e20`). Hence, by the
    /// sizes of `a` and `b`:
    ///
    /// - both `< 8`: `a·ln(dfn/dfd) + (a − 1)·ln x − (a + b)·ln(1 + u) − ln B`,
    ///   whose cancellation is bounded by `8·ε·|ln|`;
    /// - both `≥ 8`: the saddle-point form of `brcomp` (DiDonato & Morris 1992):
    ///   `k + a·(ln(1 + e₁) − e₁) + b·(ln(1 + e₂) − e₂) − ln x` with
    ///   `1 + e₁ = y/x₀`, `1 + e₂ = z/y₀`. `a·e₁ + b·e₂ = 0`, and both are
    ///   formed from `λ = a − (a + b)·y = a·(1 − x)·z` without rounding `y`:
    ///   `e₁ = (x − 1)·z`, `e₂ = (dfn/dfd)·(1 − x)·z`. `ln(1 + e) − e` stays
    ///   `O(1)` in the bulk (`e ~ 1/√a`);
    /// - one `≥ 8` (say `a`): `ln Γ(a + b)/Γ(a) − b·ln a` is `O(b²/a)`
    ///   (`fisherf_mixed_const`), so the density is written with `a·z` in place
    ///   of `z`, which is `dfn/(2(1 + u))`, the scale-free `dfd/(2x)` as
    ///   `dfn → ∞` (the `dfd/χ²_dfd` limit). `a·ln y` is `−a·ln(1 + 1/u)`, and
    ///   `−dfd/(2x)` where `u ≥ 1e17`. The mirror for large `b`.
    #[inline(always)]
    fn log_density_with(&self, (k, c1, c2): (f64, f64, f64), x: f64) -> f64 {
        if x <= 0.0 || x == f64::INFINITY {
            return f64::NEG_INFINITY;
        }
        let (m, n) = (self.dfn, self.dfd);
        let (a, b) = (0.5 * m, 0.5 * n);
        let r = m / n;
        let u = r * x;
        if a < 8.0 && b < 8.0 {
            // ln(1 + u). Where `u` overflows it is `ln u + log1p(1/u)`, and the
            // last term is below 1e-308.
            let ln_kernel = if u.is_finite() {
                libm::log1p(u)
            } else {
                libm::log(m) - libm::log(n) + libm::log(x)
            };
            return c1 + (a - 1.0) * libm::log(x) - (a + b) * ln_kernel + k;
        }
        self.log_density_large_df((k, c1, c2), x, r, u)
    }
    /// `log_density_with` where `dfn` or `dfd` is `≥ 16`, given `r = dfn/dfd`
    /// and `u = r·x`; kept out of line so the small-df branch stays small.
    #[inline(never)]
    fn log_density_large_df(&self, (k, c1, c2): (f64, f64, f64), x: f64, r: f64, u: f64) -> f64 {
        let (m, n) = (self.dfn, self.dfd);
        let (a, b) = (0.5 * m, 0.5 * n);
        // ln u where `u` is subnormal, underflows or overflows.
        let ln_u = || {
            if (f64::MIN_POSITIVE..f64::INFINITY).contains(&u) {
                libm::log(u)
            } else {
                libm::log(m) - libm::log(n) + libm::log(x)
            }
        };
        // ln(1 + u), as above.
        let ln_1pu = || {
            if u < f64::INFINITY {
                libm::log1p(u)
            } else {
                ln_u()
            }
        };
        // ln y.
        let ln_y = || {
            if u >= 1.0 {
                -libm::log1p(1.0 / u)
            } else {
                ln_u() - libm::log1p(u)
            }
        };
        let ln_x = libm::log(x);
        if a >= 8.0 && b >= 8.0 {
            // `ρ = e/(2 + e)` and `1/(1 − ρ) = 1 + e/2` for each `e`, from
            // `2 + e₁ = (1 + x + 2u)/(1 + u)` and `2 + e₂ = (2 + u + r)/(1 + u)`
            // (`r·x = u`), so the divisions do not chain.
            let d1 = 1.0 + x + 2.0 * u;
            let (t1, t2) = if d1 < f64::INFINITY {
                let d2 = 2.0 + u + r;
                let w = 0.5 / (1.0 + u);
                let (n1, n2) = (x - 1.0, r * (1.0 - x));
                (
                    log1pmx(n1 / d1, d1 * w, 2.0 * w * n1, c1, || ln_y() - c1),
                    log1pmx(n2 / d2, d2 * w, 2.0 * w * n2, c2, || -ln_1pu() - c2),
                )
            } else {
                // `x > 1` here, `1/u` is below 1e-308, and both `e` are far
                // outside the series range.
                let t = 1.0 / x;
                let (e1, e2) = ((1.0 - t) / (r + t), t - 1.0);
                (
                    log1pmx(1.0, 0.0, e1, c1, || ln_y() - c1),
                    log1pmx(1.0, 0.0, e2, c2, || -ln_1pu() - c2),
                )
            };
            return k + a * t1 + b * t2 - ln_x;
        }
        if a >= 8.0 {
            // a·ln y = −a·log1p(1/u); a/u = dfd/(2x) exactly.
            let a_ln_y = if u >= 1e17 { -0.5 * n / x } else { a * ln_y() };
            // ln(a·z) = ln(dfn/(2(1 + u))), or ln(dfd/(2x)) where `1 + u`
            // overflows.
            let d = 1.0 + u;
            let ln_az = if d < f64::INFINITY {
                libm::log(a / d)
            } else {
                let q = 0.5 * n / x;
                if q >= f64::MIN_POSITIVE {
                    libm::log(q)
                } else {
                    libm::log(0.5 * n) - ln_x
                }
            };
            return k + a_ln_y + b * ln_az - ln_x;
        }
        // b·ln z = −b·log1p(u); b·u = dfn·x/2 exactly.
        let b_ln_z = if u <= 1e-17 {
            -0.5 * m * x
        } else {
            -b * ln_1pu()
        };
        // ln(b·y) = ln(dfn·x/(2(1 + u))), from logs where that is not normal;
        // `ln b + ln y` for `u ≥ 1`, where `y ≥ ½`.
        let ln_by = if u < 1.0 {
            let by = 0.5 * m * x / (1.0 + u);
            if by >= f64::MIN_POSITIVE {
                libm::log(by)
            } else {
                libm::log(0.5 * m) + ln_x - libm::log1p(u)
            }
        } else {
            libm::log(b) + ln_y()
        };
        k + a * ln_by + b_ln_z - ln_x
    }
}

/// `−ln B(s, l) − s·ln l` for `l ≥ 8`, `s > 0`:
/// `−ln Γ(s) + (s + l − ½)·ln(1 + s/l) − s − (δ(l) − δ(s + l))` with the
/// Stirling correction `δ` (from `algdiv`, DiDonato & Morris 1992). `O(s²/l)` apart from `ln Γ(s)`, where `ln B` and
/// `s·ln l` are each `~s·ln l`. The constant of `FisherF::log_density_with`
/// when only one half-df is `≥ 8`.
fn fisherf_mixed_const(s: f64, l: f64) -> f64 {
    use crate::special::elementary::stirling_del;
    -crate::special::lgamma(s) + (s + l - 0.5) * libm::log1p(s / l)
        - s
        - (stirling_del(l) - stirling_del(s + l))
}

/// Below this, `FisherF` hands the incomplete-beta kernel the small one of
/// `y = dfn·x/(dfd + dfn·x)` and `z = 1 − y` as its argument: `I_z(dfd/2,
/// dfn/2)` in place of `I_y(dfn/2, dfd/2)` where `z` is below it, the mirror
/// where `y` is, and the wanted tail is that value or its complement. The
/// kernel's `x` is then the small value, not one within 1e-8 of 1. Measured
/// against mpmath, the unswapped call errs by up to 46 ulp in `ln sf` and
/// `ln cdf` (`F(7, 1e20).ln_sf(2)`, `F(1e20, 7).ln_cdf(0.5)`).
const FISHERF_SMALL_ARG: f64 = 1e-8;

/// `ln(1 − e^{l})` for `l ≤ 0`: `ln(−expm1(l))` for `l > −ln 2`, where `1 −
/// e^{l}` is small, else `log1p(−e^{l})`, where `e^{l}` is (Mächler 2012,
/// "Accurately computing log(1 − exp(−|a|))", Rmpfr vignette).
fn ln_1m_exp(l: f64) -> f64 {
    if l > -core::f64::consts::LN_2 {
        libm::log(-libm::expm1(l))
    } else {
        libm::log1p(-libm::exp(l))
    }
}

/// `ln(1 − I_x(a, b))` from the kernel's `(m, e, complement)`, `y = 1 − x`:
/// `ln m + e` where it evaluated the complement directly, else
/// `log1p(−m·e^{e})`; the mirror of `ln_betai`.
fn ln_betai_complement(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let (m, e, complement) = crate::special::incomplete::betai_parts(a, b, x, y);
    if complement {
        libm::log(m) + e
    } else {
        libm::log1p(-crate::special::incomplete::value_of_parts(m, e))
    }
}

impl ContinuousCdf for FisherF {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x == f64::INFINITY {
            return 1.0;
        }
        self.cdf_sf(x).0
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        if x == f64::INFINITY {
            return 0.0;
        }
        self.cdf_sf(x).1
    }
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        positive_quantile(p, false, |t, upper| self.solve(t, upper))
    }
    /// Newton on `ln sf(x) − ln q` for `q ≤ 0.5`; see `positive_quantile`.
    fn isf(&self, q: f64) -> Result<f64, StatError> {
        positive_quantile(q, true, |t, upper| self.solve(t, upper))
    }
}

impl FisherF {
    /// `(cdf, sf)` at `x > 0`: `cdf = I_y(dfn/2, dfd/2)`, `sf = I_z(dfd/2, dfn/2)`
    /// with `(y, z)` from `beta_args`. As in `StudentT::halves`, both go to the
    /// incomplete-beta kernel, which evaluates the side below its reflection
    /// point `(a + 1)/(a + b + 2)` to full relative precision and gives the
    /// other as `1 −` it. Evaluating the sf from a `z` that rounds to 1 would
    /// lose all of a small cdf, and the cdf from a `y` that rounds to 1 all of
    /// a small sf. A subnormal `y` keeps only its remaining bits (11 at
    /// 1e-320), and the kernel's `y^{dfn/2}` carries their error times
    /// `dfn/2`: with a small `dfn` the cdf there is not small
    /// (`F(0.01, 10).cdf(1e-320)` ≈ 0.0245) and would be off at 6e-5, and a
    /// `y` that underflows would give 0 for `F(1, 10).cdf(5e-324)` ≈ 1.7e-162.
    /// So a subnormal `y` takes the cdf as `exp` of `ln_cdf`, which forms
    /// `ln y` from `ln x`, and the sf as `−expm1` of it; a subnormal `z` the
    /// mirror, through `ln_sf`. NaN for a NaN `x`.
    fn cdf_sf(&self, x: f64) -> (f64, f64) {
        let (y, z) = self.beta_args(x);
        if y < f64::MIN_POSITIVE {
            let ln_cdf = self.ln_cdf(x);
            return (libm::exp(ln_cdf), -libm::expm1(ln_cdf));
        }
        if z < f64::MIN_POSITIVE {
            let ln_sf = self.ln_sf(x);
            return (-libm::expm1(ln_sf), libm::exp(ln_sf));
        }
        let (a, b) = (0.5 * self.dfn, 0.5 * self.dfd);
        if z < FISHERF_SMALL_ARG {
            let (m, e, complement) = crate::special::incomplete::betai_parts(b, a, z, y);
            let (sf, cdf) = crate::special::incomplete::pq_from_parts(m, e, complement);
            return (cdf, sf);
        }
        let (m, e, complement) = crate::special::incomplete::betai_parts(a, b, y, z);
        crate::special::incomplete::pq_from_parts(m, e, complement)
    }
    /// `(y, z) = (dfn·x/(dfd + dfn·x), dfd/(dfd + dfn·x))` for finite `x > 0`,
    /// each by its own division, so `z = 1 − y` keeps full relative precision
    /// where it is small (and `y` where it is). Where `dfd + dfn·x` overflows,
    /// both ratios are taken with numerator and denominator divided by `x`,
    /// which keeps them finite: for small `dfd` the sf there is not negligible
    /// (`F(10, 0.01).sf(1e308)` ≈ 0.028). Where `dfn·x` is subnormal or
    /// underflows, `y` is formed from `x·2^256` (exact) and scaled back, which
    /// rounds once, so a `y` in the normal range (`dfd` small) keeps full
    /// relative precision and a subnormal one is correctly rounded.
    fn beta_args(&self, x: f64) -> (f64, f64) {
        // 2^256 and 2^-256.
        const UP: f64 = 1.157_920_892_373_162e77;
        const DOWN: f64 = 8.636_168_555_094_445e-78;
        let (m, n) = (self.dfn, self.dfd);
        let nx = m * x;
        let d = n + nx;
        if d < f64::INFINITY {
            // `x < 2^52` wherever `dfn·x < 2^-1022` (`dfn` ≥ 2^-1074), so
            // `x·2^256` stays finite.
            let y = if nx >= f64::MIN_POSITIVE {
                nx / d
            } else {
                m * (x * UP) / d * DOWN
            };
            (y, n / d)
        } else {
            let u = n / x;
            (m / (m + u), u / (m + u))
        }
    }
    /// `ln sf(x)` for finite `x > 0`: the leading term of `ln_betai_small_z`
    /// when `z = dfd/(dfd + dfn·x)` is small enough for it (which covers a
    /// subnormal or underflowed `z`), else `ln_betai`, which keeps the log of an
    /// sf below the normal range. `ln z = ln dfd − ln dfn − ln x` drops
    /// `ln(1 + dfd/(dfn·x)) ≈ z`. Where `y` is subnormal or underflows, the
    /// sf is `1 − cdf` with the cdf from `ln_cdf`, which takes `ln y` from
    /// `ln x` (`ln_1m_exp`): at a small `dfn` the cdf there is not
    /// small (`F(0.0015, 2058).cdf(1e-320)` ≈ 0.57), and the kernel, given the
    /// rounded or zero `y`, would read it as 0 and the sf as 1.
    #[doc(hidden)]
    pub fn ln_sf(&self, x: f64) -> f64 {
        let (y, z) = self.beta_args(x);
        if y < f64::MIN_POSITIVE {
            return ln_1m_exp(self.ln_cdf(x));
        }
        ln_betai_small_z(0.5 * self.dfd, 0.5 * self.dfn, z, || {
            libm::log(self.dfd) - libm::log(self.dfn) - libm::log(x)
        })
        .unwrap_or_else(|| {
            if y < FISHERF_SMALL_ARG {
                ln_betai_complement(0.5 * self.dfn, 0.5 * self.dfd, y, z)
            } else {
                crate::special::incomplete::ln_betai(0.5 * self.dfd, 0.5 * self.dfn, z, y)
            }
        })
    }
    /// `ln cdf(x)` for finite `x > 0`, the mirror of `ln_sf`: the leading term
    /// of `ln_betai_small_z` when `y = dfn·x/(dfd + dfn·x)` is small enough for
    /// it (which covers a `y` that underflows), else `ln_betai`. Where `y` is
    /// subnormal, `ln y = ln dfn + ln x − ln dfd`, dropping
    /// `ln(1 + dfn·x/dfd) ≈ y`; elsewhere `ln y` directly (`beta_args` keeps
    /// a normal `y` to full precision also from a subnormal `dfn·x`), which
    /// avoids the rounding of three large logs that nearly cancel. Where `z`
    /// is subnormal or underflows, `ln_1m_exp(ln sf)`, the mirror of `ln_sf`.
    #[doc(hidden)]
    pub fn ln_cdf(&self, x: f64) -> f64 {
        let (y, z) = self.beta_args(x);
        if z < f64::MIN_POSITIVE {
            return ln_1m_exp(self.ln_sf(x));
        }
        ln_betai_small_z(0.5 * self.dfn, 0.5 * self.dfd, y, || {
            if y >= f64::MIN_POSITIVE {
                libm::log(y)
            } else {
                libm::log(self.dfn) + libm::log(x) - libm::log(self.dfd)
            }
        })
        .unwrap_or_else(|| {
            if z < FISHERF_SMALL_ARG {
                ln_betai_complement(0.5 * self.dfd, 0.5 * self.dfn, z, y)
            } else {
                crate::special::incomplete::ln_betai(0.5 * self.dfn, 0.5 * self.dfd, y, z)
            }
        })
    }
    /// Root of `cdf(x) = target` (`upper = false`) or `sf(x) = target`
    /// (`upper = true`), `target ∈ (0, ½]`: Wilson–Hilferty seed on `χ²(dfn)/dfn`,
    /// then `bracketed_newton` in `ln x`.
    fn solve(&self, target: f64, upper: bool) -> f64 {
        let dfn = self.dfn;
        let z = if upper {
            -norm_quantile(target)
        } else {
            norm_quantile(target)
        };
        let h = 2.0 / (9.0 * dfn);
        let wh_base = 1.0 - h + z * libm::sqrt(h);
        let chi_seed = dfn * (wh_base * wh_base * wh_base);
        let mut x = chi_seed / dfn;
        if !x.is_finite() || x <= 0.0 {
            // WH fails for very small p, as in `Gamma::solve`. Fall back to the
            // leading series term I_y(a, b) ≈ y^a/(a·B(a, b)) → p, with
            // `y = dfn·x/(dfd + dfn·x) ≈ dfn·x/dfd`.
            let (a, b) = (0.5 * dfn, 0.5 * self.dfd);
            let p = if upper { 1.0 - target } else { target };
            let ln_beta = crate::special::lbeta(a, b);
            let log_y = (libm::log(p) + libm::log(a) + ln_beta) / a;
            x = (self.dfd / dfn * libm::exp(log_y)).max(1e-300);
            if !x.is_finite() {
                x = 1e-6;
            }
        }
        // Both tails are power laws in x (`cdf ∝ x^{dfn/2}`, `sf ∝ x^{−dfd/2}`).
        // Below half the smallest subnormal the mass reads as 0, as in
        // `InverseGaussian::ln_cdf`: at large df its log reaches −1e17 and
        // more, where `ln mass − ln pdf` (the Newton scale) keeps no digits
        // and a step rounded to nothing would pass as convergence.
        let ln_mass = |x: f64| {
            let ln_m = if upper { self.ln_sf(x) } else { self.ln_cdf(x) };
            if ln_m < -1075.0 * core::f64::consts::LN_2 {
                f64::NEG_INFINITY
            } else {
                ln_m
            }
        };
        let consts = self.ln_density_consts();
        bracketed_newton(x, target, upper, true, ln_mass, |x| {
            self.log_density_with(consts, x)
        })
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for FisherF {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        self.quantile(rng.uniform())
            .expect("uniform() ∈ (0,1) is a valid quantile arg")
    }
}

/// Continuous uniform on `[a, b]`.
///
/// Convention: `a < b`, inclusive support, constant density `1/(b−a)`. Matches
/// `scipy.stats.uniform(loc=a, scale=b−a)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uniform {
    a: f64,
    b: f64,
}

impl Uniform {
    /// Construct `Uniform[a, b]` with `a < b`.
    ///
    /// # Errors
    /// `DomainError` if non-finite or `a ≥ b`.
    ///
    /// ```
    /// use commonstats::dist::continuous::Uniform;
    /// use commonstats::dist::ContinuousCdf;
    /// let u = Uniform::new(0.0, 1.0).unwrap();
    /// assert!((u.cdf(0.5) - 0.5).abs() < 1e-15);
    /// assert!(Uniform::new(1.0, 0.0).is_err());
    /// ```
    pub fn new(a: f64, b: f64) -> Result<Self, StatError> {
        if !a.is_finite() || !b.is_finite() || a >= b {
            return Err(StatError::DomainError("Uniform: require finite a < b"));
        }
        Ok(Uniform { a, b })
    }
}

impl Distribution for Uniform {
    fn support_min(&self) -> Bound {
        Bound::Finite(self.a)
    }
    fn support_max(&self) -> Bound {
        Bound::Finite(self.b)
    }
    fn mean(&self) -> Option<f64> {
        Some(0.5 * (self.a + self.b))
    }
    fn variance(&self) -> Option<f64> {
        let w = self.b - self.a;
        Some(w * w / 12.0)
    }
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(-6.0 / 5.0)
    }
    fn entropy(&self) -> Option<f64> {
        Some(libm::log(self.b - self.a))
    }
}

impl ContinuousDensity for Uniform {
    fn density(&self, x: f64) -> f64 {
        // `x < a || x > b` is false for a NaN `x`, and neither branch below
        // reads `x`, so the range check alone would return the constant
        // 1/(b−a) instead of propagating the NaN.
        if x.is_nan() {
            x
        } else if x < self.a || x > self.b {
            0.0
        } else {
            1.0 / (self.b - self.a)
        }
    }
    fn log_density(&self, x: f64) -> f64 {
        if x.is_nan() {
            x
        } else if x < self.a || x > self.b {
            f64::NEG_INFINITY
        } else {
            -libm::log(self.b - self.a)
        }
    }
}

impl ContinuousCdf for Uniform {
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.a {
            0.0
        } else if x >= self.b {
            1.0
        } else {
            (x - self.a) / (self.b - self.a)
        }
    }
    /// `(b − x)/(b − a)`: `b − x` is exact near `b` (Sterbenz), where
    /// `1 − cdf` would keep only the absolute error of a value near 1.
    fn sf(&self, x: f64) -> f64 {
        if x <= self.a {
            1.0
        } else if x >= self.b {
            0.0
        } else {
            (self.b - x) / (self.b - self.a)
        }
    }
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        Ok(self.a + p * (self.b - self.a))
    }
    /// Closed form `b − q·(b − a)`.
    fn isf(&self, q: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(StatError::ProbabilityOutOfRange(q));
        }
        Ok(self.b - q * (self.b - self.a))
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for Uniform {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        self.a + (self.b - self.a) * rng.uniform()
    }
}

/// Exponential distribution with rate `λ`.
///
/// Convention: **rate** parameterization (`scale = 1/λ`). Density `λ·e^{−λx}`
/// for `x ≥ 0`. sf is the exact `e^{−λx}` (no `1−cdf`). Matches
/// `scipy.stats.expon(scale=1/λ)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential {
    rate: f64,
}

impl Exponential {
    /// Construct with `rate λ > 0`.
    ///
    /// # Errors
    /// `DomainError` if `rate` is non-finite or `≤ 0`.
    ///
    /// ```
    /// use commonstats::dist::continuous::Exponential;
    /// use commonstats::dist::Distribution;
    /// let e = Exponential::new(2.0).unwrap();
    /// assert_eq!(e.mean(), Some(0.5));
    /// assert!(Exponential::new(0.0).is_err());
    /// ```
    pub fn new(rate: f64) -> Result<Self, StatError> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err(StatError::DomainError(
                "Exponential: rate must be finite and > 0",
            ));
        }
        Ok(Exponential { rate })
    }
}

impl Distribution for Exponential {
    fn support_min(&self) -> Bound {
        Bound::Finite(0.0)
    }
    fn support_max(&self) -> Bound {
        Bound::PosInfinity
    }
    fn mean(&self) -> Option<f64> {
        Some(1.0 / self.rate)
    }
    fn variance(&self) -> Option<f64> {
        Some(1.0 / (self.rate * self.rate))
    }
    fn skewness(&self) -> Option<f64> {
        Some(2.0)
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(6.0)
    }
    fn entropy(&self) -> Option<f64> {
        Some(1.0 - libm::log(self.rate))
    }
}

impl ContinuousDensity for Exponential {
    fn density(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.rate * libm::exp(-self.rate * x)
        }
    }
    fn log_density(&self, x: f64) -> f64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else {
            libm::log(self.rate) - self.rate * x
        }
    }
}

impl ContinuousCdf for Exponential {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            -libm::expm1(-self.rate * x)
        } // 1 - e^{-λx}
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            1.0
        } else {
            libm::exp(-self.rate * x)
        }
    }
    /// Closed form `−log1p(−p)/rate`: `ln` of the rounded `1 − p` would keep
    /// only its ~1.1e-16 absolute error, 100% relative below p = 1.1e-16.
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        Ok(-libm::log1p(-p) / self.rate)
    }
    /// Closed form `−ln q / rate`.
    fn isf(&self, q: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(StatError::ProbabilityOutOfRange(q));
        }
        if q == 0.0 {
            return Ok(f64::INFINITY);
        }
        Ok(-libm::log(q) / self.rate)
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for Exponential {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        // u ~ U(0,1) ⇒ −ln(u)/λ ~ Exp(λ); u∈(0,1) keeps ln finite.
        -libm::log(rng.uniform()) / self.rate
    }
}

/// Cauchy (Lorentz) distribution with location `loc` and scale `scale`.
///
/// Convention: no mean/variance (heavy tails); entropy `ln(4π·scale)`. CDF
/// `½ + atan(z)/π`, `z = (x−loc)/scale`, with the tail beyond `|z| = 1` as
/// `atan(1/|z|)/π` (sf the mirror); quantile `loc + scale·tan(π(p − ½))`,
/// exactly `loc` at `p = ½`. Matches `scipy.stats.cauchy(loc, scale)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cauchy {
    loc: f64,
    scale: f64,
}

impl Cauchy {
    /// Construct with `scale > 0`.
    ///
    /// # Errors
    /// `DomainError` if non-finite or `scale ≤ 0`.
    ///
    /// ```
    /// use commonstats::dist::continuous::Cauchy;
    /// use commonstats::dist::ContinuousCdf;
    /// let c = Cauchy::new(0.0, 1.0).unwrap();
    /// assert!((c.cdf(0.0) - 0.5).abs() < 1e-15);
    /// assert!(Cauchy::new(0.0, 0.0).is_err());
    /// ```
    pub fn new(loc: f64, scale: f64) -> Result<Self, StatError> {
        if !loc.is_finite() || !scale.is_finite() || scale <= 0.0 {
            return Err(StatError::DomainError(
                "Cauchy: scale must be finite and > 0",
            ));
        }
        Ok(Cauchy { loc, scale })
    }
}

impl Distribution for Cauchy {
    fn support_min(&self) -> Bound {
        Bound::NegInfinity
    }
    fn support_max(&self) -> Bound {
        Bound::PosInfinity
    }
    fn entropy(&self) -> Option<f64> {
        Some(libm::log(4.0 * core::f64::consts::PI * self.scale))
    }
}

/// Density and tails are split at `|x − loc| = scale` (`|z| = 1`). Inside, `z =
/// (x − loc)/scale` is at most 1. Outside, `w = scale/|x − loc| = 1/|z| < 1`
/// is formed by one division and takes the place of `z`: `pdf =
/// (w/|x − loc|)/(π(1 + w²))`, its log likewise, tail mass `atan(w)/π`. So no `z²` is formed (it overflows at
/// `|z| ~ 1e154`), nor `z` itself (overflowing at a tiny scale), and the far
/// tail is not `½ − atan(|z|)/π`, which keeps only the ~1e-16 absolute error
/// of a value near ½.
impl ContinuousDensity for Cauchy {
    fn density(&self, x: f64) -> f64 {
        let d = x - self.loc;
        let inv_pi = core::f64::consts::FRAC_1_PI;
        if libm::fabs(d) <= self.scale {
            let z = d / self.scale;
            // `(1/π)/scale` first: `π·scale` overflows for scale > 5.7e307.
            return inv_pi / self.scale / (1.0 + z * z);
        }
        let (w, abs_d) = self.tail_ratio(x, d);
        // `w/|x − loc| = scale/(x − loc)²`, formed without the square; where
        // `x − loc` overflowed, `abs_d` is half the distance.
        let q = if d.is_finite() { w } else { 0.5 * w } / abs_d;
        inv_pi * q / (1.0 + w * w)
    }
    fn log_density(&self, x: f64) -> f64 {
        let d = x - self.loc;
        if libm::fabs(d) <= self.scale {
            let z = d / self.scale;
            return -libm::log(core::f64::consts::PI) - libm::log(self.scale) - libm::log1p(z * z);
        }
        let (w, abs_d) = self.tail_ratio(x, d);
        let tail = -libm::log1p(w * w) - libm::log(core::f64::consts::PI);
        // `ln(scale/(x − loc)²)` as `ln(w/|x − loc|)` while that is normal:
        // `ln scale − 2·ln|z|` would cancel two logs of size `|ln scale|`.
        let q = w / abs_d;
        if (f64::MIN_POSITIVE..f64::INFINITY).contains(&q) && d.is_finite() {
            return libm::log(q) + tail;
        }
        // Here the density is not normal, so `|ln pdf| > 708` bounds the
        // cancellation. `abs_d` is half an overflowed `|x − loc|`.
        let ln_d = libm::log(abs_d)
            + if d.is_finite() || d.is_nan() {
                0.0
            } else {
                core::f64::consts::LN_2
            };
        libm::log(self.scale) - 2.0 * ln_d + tail
    }
}

impl Cauchy {
    /// `(w, |d|)` for `|d| > scale`, `d = x − loc`: `w = scale/|d| < 1`. Where
    /// `x − loc` overflows (both near the f64 limit, opposite signs), both
    /// are taken halved, so `|d|` is then half the distance and `w` still the
    /// ratio. NaN for a NaN `x`.
    fn tail_ratio(&self, x: f64, d: f64) -> (f64, f64) {
        if d.is_finite() || d.is_nan() {
            let abs_d = libm::fabs(d);
            (self.scale / abs_d, abs_d)
        } else {
            let half = libm::fabs(0.5 * x - 0.5 * self.loc);
            (0.5 * self.scale / half, half)
        }
    }
    /// `(cdf, sf)` at `x`: `½ ± atan(z)/π` for `|z| ≤ 1`, where neither is
    /// below ¼; outside, the tail beyond `x` is `atan(w)/π` (`atan(1/|z|) =
    /// π/2 − atan(|z|)`), formed directly, and the other side is `1 −` it.
    fn cdf_sf(&self, x: f64) -> (f64, f64) {
        let d = x - self.loc;
        if libm::fabs(d) <= self.scale {
            let c = libm::atan(d / self.scale) / core::f64::consts::PI;
            return (0.5 + c, 0.5 - c);
        }
        let tail = libm::atan(self.tail_ratio(x, d).0) / core::f64::consts::PI;
        if d < 0.0 {
            (tail, 1.0 - tail)
        } else {
            (1.0 - tail, tail)
        }
    }
    /// `scale·tan(π(t − ½))` for `t ∈ (0, 1)`, the quantile minus `loc`,
    /// exactly 0 at `t = ½` (`cos(π/2)` in f64 is 6.1e-17, not 0). Taken as
    /// `∓scale·cot(π·s)` on the smaller side `s = min(t, 1 − t)` (`1 − t` is
    /// exact there), whose `sin(π·s) ≈ π·s` keeps full relative precision
    /// where `tan` near `±π/2` would not. Below `s = 1e-9`, `cot(π·s) =
    /// 1/(π·s)·(1 − (π·s)²/3 − …)` is its leading term to 3e-18, formed as
    /// `(scale/π)/s`: `π·s` is subnormal for a subnormal `s`, and `cot` alone
    /// overflows below `s ≈ 1.8e-309` before the scale can bring it back.
    fn offset(&self, t: f64) -> f64 {
        if t == 0.5 {
            return 0.0;
        }
        let (s, sign) = if t < 0.5 { (t, -1.0) } else { (1.0 - t, 1.0) };
        if s < 1e-9 {
            return sign * (self.scale * core::f64::consts::FRAC_1_PI) / s;
        }
        let y = core::f64::consts::PI * s;
        sign * self.scale * (libm::cos(y) / libm::sin(y))
    }
}

impl ContinuousCdf for Cauchy {
    fn cdf(&self, x: f64) -> f64 {
        self.cdf_sf(x).0
    }
    fn sf(&self, x: f64) -> f64 {
        self.cdf_sf(x).1
    }
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(self.loc + self.offset(p))
    }
    /// Symmetry about `loc`: `isf(q) = loc − offset(q)`.
    fn isf(&self, q: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(StatError::ProbabilityOutOfRange(q));
        }
        if q == 0.0 {
            return Ok(f64::INFINITY);
        }
        if q == 1.0 {
            return Ok(f64::NEG_INFINITY);
        }
        Ok(self.loc - self.offset(q))
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for Cauchy {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        self.loc + self.offset(rng.uniform())
    }
}

/// Weibull distribution with shape `k` and scale `λ`.
///
/// Convention: `scipy.stats.weibull_min(k, scale=λ)` (loc 0). Density
/// `(k/λ)(x/λ)^{k−1} e^{−(x/λ)^k}` for `x ≥ 0`; sf exact `e^{−(x/λ)^k}`.
/// Moments use `Γ`. Matches `scipy.stats.weibull_min(k, scale=λ)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weibull {
    shape: f64,
    scale: f64,
}

impl Weibull {
    /// Construct with `shape k > 0`, `scale λ > 0`.
    ///
    /// # Errors
    /// `DomainError` if non-finite or either `≤ 0`.
    ///
    /// ```
    /// use commonstats::dist::continuous::Weibull;
    /// use commonstats::dist::ContinuousCdf;
    /// let w = Weibull::new(1.5, 2.0).unwrap();
    /// assert!(w.cdf(2.0) > 0.0);
    /// assert!(Weibull::new(0.0, 2.0).is_err());
    /// ```
    pub fn new(shape: f64, scale: f64) -> Result<Self, StatError> {
        if !shape.is_finite() || !scale.is_finite() || shape <= 0.0 || scale <= 0.0 {
            return Err(StatError::DomainError(
                "Weibull: shape and scale must be finite and > 0",
            ));
        }
        Ok(Weibull { shape, scale })
    }
}

impl Distribution for Weibull {
    fn support_min(&self) -> Bound {
        Bound::Finite(0.0)
    }
    fn support_max(&self) -> Bound {
        Bound::PosInfinity
    }
    fn mean(&self) -> Option<f64> {
        Some(self.scale * crate::special::gamma(1.0 + 1.0 / self.shape))
    }
    fn variance(&self) -> Option<f64> {
        let g1 = crate::special::gamma(1.0 + 1.0 / self.shape);
        let g2 = crate::special::gamma(1.0 + 2.0 / self.shape);
        Some(self.scale * self.scale * (g2 - g1 * g1))
    }
}

impl ContinuousDensity for Weibull {
    fn density(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x == 0.0 {
            // k<1 → +∞, k=1 → k/λ, k>1 → 0.
            return if self.shape < 1.0 {
                f64::INFINITY
            } else if self.shape == 1.0 {
                self.shape / self.scale
            } else {
                0.0
            };
        }
        libm::exp(self.log_density(x))
    }
    fn log_density(&self, x: f64) -> f64 {
        if x < 0.0 || x == f64::INFINITY {
            return f64::NEG_INFINITY;
        }
        if x == 0.0 {
            // `(k − 1)·ln z` is 0·(−∞) at k = 1; `density` handles x = 0 by case.
            return libm::log(self.density(0.0));
        }
        let (k, lam) = (self.shape, self.scale);
        let (ln_z, zk) = self.ln_z_pow(x);
        libm::log(k) - libm::log(lam) + (k - 1.0) * ln_z - zk
    }
}

impl Weibull {
    /// `(ln z, z^k)`, `z = x/λ`, for `x ≥ 0`. Where `z` is subnormal, underflows
    /// or overflows, both come from `ln z = ln x − ln λ`: at a small `k`, `z^k`
    /// is still in range there (`Weibull(0.01, 1e10).cdf(1e-312)` ≈ 6.0e-4,
    /// where `z` = 1e-322 keeps 5 bits), and the rounding of `ln z`,
    /// `|ln z|·ε`, is what `z^k`'s own sensitivity to `k` already costs.
    fn ln_z_pow(&self, x: f64) -> (f64, f64) {
        let z = x / self.scale;
        if (f64::MIN_POSITIVE..f64::INFINITY).contains(&z) {
            (libm::log(z), libm::pow(z, self.shape))
        } else {
            let ln_z = libm::log(x) - libm::log(self.scale);
            (ln_z, libm::exp(self.shape * ln_z))
        }
    }
    /// `λ·t^{1/k}` for `t = −ln(tail mass) ≥ 0`, the quantile's closed form.
    /// Where `t^{1/k}` alone leaves the normal range (overflow for `k < 1`
    /// while a small `λ` brings the product back: `Weibull(0.01, 1e-100)
    /// .isf(1e-300)` ≈ 8.6e183), `exp(ln λ + ln t/k)`, whose rounding,
    /// `(|ln λ| + |ln t|/k)·ε`, is at most about the root's sensitivity to `k`
    /// there, `|ln t|/k > 708`.
    fn root(&self, t: f64) -> f64 {
        let r = libm::pow(t, 1.0 / self.shape);
        if (f64::MIN_POSITIVE..f64::INFINITY).contains(&r) {
            self.scale * r
        } else {
            libm::exp(libm::log(self.scale) + libm::log(t) / self.shape)
        }
    }
}

impl ContinuousCdf for Weibull {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        -libm::expm1(-self.ln_z_pow(x).1) // 1 - e^{-(x/λ)^k}
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        libm::exp(-self.ln_z_pow(x).1)
    }
    /// Closed form `scale·(−ln(1 − p))^{1/shape}`, with `ln(1 − p)` as
    /// `log1p(−p)`: `ln` of the rounded `1 − p` would keep only its ~1.1e-16
    /// absolute error, 100% relative below p = 1.1e-16.
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(self.root(-libm::log1p(-p)))
    }
    /// Closed form `scale·(−ln q)^{1/shape}`.
    fn isf(&self, q: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(StatError::ProbabilityOutOfRange(q));
        }
        if q == 0.0 {
            return Ok(f64::INFINITY);
        }
        Ok(self.root(-libm::log(q)))
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for Weibull {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        self.root(-libm::log(rng.uniform()))
    }
}

/// Log-normal distribution: `ln X ~ N(μ, σ)`.
///
/// Convention: `μ`/`σ` are the mean/sd of the **log**. Matches
/// `scipy.stats.lognorm(s=σ, scale=e^μ)`. CDF/quantile delegate to a `Normal`
/// on `ln x`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormal {
    mu: f64,
    sigma: f64,
}

impl LogNormal {
    /// Construct with `σ > 0`.
    ///
    /// # Errors
    /// `DomainError` if non-finite or `σ ≤ 0`.
    ///
    /// ```
    /// use commonstats::dist::continuous::LogNormal;
    /// use commonstats::dist::ContinuousCdf;
    /// let l = LogNormal::new(0.0, 1.0).unwrap();
    /// assert!((l.quantile(0.5).unwrap() - 1.0).abs() < 1e-9);
    /// assert!(LogNormal::new(0.0, -1.0).is_err());
    /// ```
    pub fn new(mu: f64, sigma: f64) -> Result<Self, StatError> {
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(StatError::DomainError(
                "LogNormal: sigma must be finite and > 0",
            ));
        }
        Ok(LogNormal { mu, sigma })
    }
    fn z(&self) -> Normal {
        Normal {
            mean: self.mu,
            sd: self.sigma,
        }
    }
}

impl Distribution for LogNormal {
    fn support_min(&self) -> Bound {
        Bound::Finite(0.0)
    }
    fn support_max(&self) -> Bound {
        Bound::PosInfinity
    }
    fn mean(&self) -> Option<f64> {
        Some(libm::exp(self.mu + 0.5 * self.sigma * self.sigma))
    }
    fn variance(&self) -> Option<f64> {
        let s2 = self.sigma * self.sigma;
        Some((libm::exp(s2) - 1.0) * libm::exp(2.0 * self.mu + s2))
    }
}

impl ContinuousDensity for LogNormal {
    fn density(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let z = (libm::log(x) - self.mu) / self.sigma;
        let c = self.sigma * libm::sqrt(2.0 * core::f64::consts::PI);
        let e = libm::exp(-0.5 * z * z);
        // A subnormal `x·σ·√(2π)` keeps only its remaining bits (5 at
        // 2e-320), so there the exact `x` divides last. Where `e^{−z²/2}` or
        // `e^{−z²/2}/(σ√(2π))` is not normal while the density can be (a
        // small `x`), `z²/2 > 670` (a subnormal `x·σ` bounds σ below 2e15)
        // and the log form's rounding, `(z²/2 + |ln x|)·ε`, is at most about
        // the density's own sensitivity to σ, `z²`.
        if e >= f64::MIN_POSITIVE {
            let d = x * c;
            if d >= f64::MIN_POSITIVE {
                return e / d;
            }
            let n = e / c;
            if n >= f64::MIN_POSITIVE {
                return n / x;
            }
        }
        libm::exp(self.log_density(x))
    }
    fn log_density(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let z = (libm::log(x) - self.mu) / self.sigma;
        -0.5 * z * z
            - libm::log(x)
            - libm::log(self.sigma)
            - 0.5 * libm::log(2.0 * core::f64::consts::PI)
    }
}

impl ContinuousCdf for LogNormal {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        self.z().cdf(libm::log(x))
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        self.z().sf(libm::log(x))
    }
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(libm::exp(self.z().quantile(p)?))
    }
    /// `exp` of the underlying normal's `isf`.
    fn isf(&self, q: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(StatError::ProbabilityOutOfRange(q));
        }
        if q == 0.0 {
            return Ok(f64::INFINITY);
        }
        if q == 1.0 {
            return Ok(0.0);
        }
        Ok(libm::exp(self.z().isf(q)?))
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for LogNormal {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        libm::exp(self.mu + self.sigma * norm_quantile(rng.uniform()))
    }
}

/// Gamma distribution with shape `α` and **rate** `β`.
///
/// Convention: rate parameterization (`scale = 1/β`). Density
/// `β^α x^{α−1} e^{−βx} / Γ(α)` for `x > 0`. CDF `P(α, βx)`; sf via `Q`.
/// Quantile = Wilson–Hilferty seed (`χ²(2α)/(2β)`) + Newton on `ln P` or `ln Q`,
/// whichever tail is smaller.
/// Matches `scipy.stats.gamma(α, scale=1/β)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma {
    shape: f64,
    rate: f64,
}

impl Gamma {
    /// Construct with `shape α > 0`, `rate β > 0`.
    ///
    /// # Errors
    /// `DomainError` if non-finite or either `≤ 0`.
    ///
    /// ```
    /// use commonstats::dist::continuous::Gamma;
    /// use commonstats::dist::Distribution;
    /// let g = Gamma::new(2.0, 1.0).unwrap();
    /// assert_eq!(g.mean(), Some(2.0));
    /// assert!(Gamma::new(0.0, 1.0).is_err());
    /// ```
    pub fn new(shape: f64, rate: f64) -> Result<Self, StatError> {
        if !shape.is_finite() || !rate.is_finite() || shape <= 0.0 || rate <= 0.0 {
            return Err(StatError::DomainError(
                "Gamma: shape and rate must be finite and > 0",
            ));
        }
        Ok(Gamma { shape, rate })
    }
}

impl Distribution for Gamma {
    fn support_min(&self) -> Bound {
        Bound::Finite(0.0)
    }
    fn support_max(&self) -> Bound {
        Bound::PosInfinity
    }
    fn mean(&self) -> Option<f64> {
        Some(self.shape / self.rate)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.shape / (self.rate * self.rate))
    }
    fn skewness(&self) -> Option<f64> {
        Some(2.0 / libm::sqrt(self.shape))
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(6.0 / self.shape)
    }
}

impl ContinuousDensity for Gamma {
    fn density(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        libm::exp(gamma_log_density(self.shape, self.rate, x))
    }
    fn log_density(&self, x: f64) -> f64 {
        gamma_log_density(self.shape, self.rate, x)
    }
}

/// `cdf` and `sf` are `P(α, β·x)` and `Q(α, β·x)`, except where `β·x` is
/// subnormal or underflows: there, as in `ln_cdf`, the cdf is `exp` of the
/// series' leading term with `ln(β·x) = ln β + ln x`, and the sf `−expm1` of
/// it. At a small shape the cdf there is not small
/// (`Gamma(0.0019, 1.4e-88).cdf(1.9e-279)` ≈ 0.20), and a subnormal `β·x`
/// keeps only its remaining bits.
impl ContinuousCdf for Gamma {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let y = self.rate * x;
        if y < f64::MIN_POSITIVE {
            return libm::exp(self.ln_cdf(x));
        }
        crate::special::gammp(self.shape, y)
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        let y = self.rate * x;
        if y < f64::MIN_POSITIVE {
            return -libm::expm1(self.ln_cdf(x));
        }
        crate::special::gammq(self.shape, y)
    }
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        positive_quantile(p, false, |t, upper| self.solve(t, upper))
    }
    /// Newton on `ln sf(x) − ln q` (`ln_gammq`) for `q ≤ 0.5`; see `positive_quantile`.
    fn isf(&self, q: f64) -> Result<f64, StatError> {
        positive_quantile(q, true, |t, upper| self.solve(t, upper))
    }
}

impl Gamma {
    /// `ln cdf(x)` for finite `x > 0`. Where `y = β·x < 1e-17`, the series
    /// `P(α, y) = y^α e^{−y}/Γ(α + 1)·(1 + y/(α + 1) + …)` (DLMF 8.7.1) is its
    /// leading term to f64 resolution (the dropped factor is `1 − O(y)`), taken
    /// in logs. That covers a subnormal `β·x`, which has lost bits, and one that
    /// underflows: there `ln y = ln β + ln x`. `ln Γ(α + 1)` comes from
    /// `ln_gamma_1p` up to α = 1.25: `lgamma(α + 1)` would round `1 + α`
    /// first, `ε/(2α)` relative in `ln Γ(1 + α) ≈ −γα` at a tiny α. Else
    /// `ln_gammp`, which keeps the log of a cdf below the normal range.
    #[doc(hidden)]
    pub fn ln_cdf(&self, x: f64) -> f64 {
        let y = self.rate * x;
        let ln_y = if y >= f64::MIN_POSITIVE {
            libm::log(y)
        } else {
            libm::log(self.rate) + libm::log(x)
        };
        if ln_y < LN_SERIES_CUTOFF {
            let a = self.shape;
            let ln_gamma_a1 = if a <= 1.25 {
                crate::special::elementary::ln_gamma_1p(a)
            } else {
                crate::special::lgamma(a + 1.0)
            };
            a * ln_y - ln_gamma_a1
        } else {
            crate::special::incomplete::ln_gammp(self.shape, y)
        }
    }
    /// Root of `cdf(x) = target` (`upper = false`) or `sf(x) = target`
    /// (`upper = true`), `target ∈ (0, ½]`: Wilson–Hilferty seed, then
    /// `bracketed_newton`, in `ln x` on the power-law lower tail
    /// (`cdf ∝ x^α`) and in `x` on the exponential upper tail (`sf/pdf → 1/β`).
    fn solve(&self, target: f64, upper: bool) -> f64 {
        let (a, rate) = (self.shape, self.rate);
        // `bracketed_newton` anchors its open-bracket jumps at `x = 1`, so in
        // `x`-space Newton a root many decades from 1 costs extra steps: the
        // upper tail is solved on `β·x ~ Gamma(α, 1)` and scaled back (one
        // rounding). The lower tail stays at scale β, as its root can lie where
        // `x` is representable and `β·x` is below the smallest subnormal.
        if upper && rate != 1.0 {
            return Gamma {
                shape: a,
                rate: 1.0,
            }
            .solve(target, true)
                / rate;
        }
        // Seed: χ²(2α) Wilson–Hilferty (Wilson & Hilferty 1931, PNAS 17:684), then x = chi2/(2β). (chi2 has df k=2α.)
        let k = 2.0 * a;
        let z = if upper {
            -norm_quantile(target)
        } else {
            norm_quantile(target)
        };
        let h = 2.0 / (9.0 * k);
        let wh_base = 1.0 - h + z * libm::sqrt(h);
        let chi = k * (wh_base * wh_base * wh_base);
        let mut x = chi / (2.0 * rate);
        if !x.is_finite() || x <= 0.0 {
            // WH fails for very small p (chi becomes negative). Fall back to
            // the leading series term P(α,βx) ≈ (βx)^α/Γ(α+1) → p. Γ(α+1), not
            // Γ(α): with Γ(α) the seed is off by α^{-1/α} (4× at α = ½).
            let p = if upper { 1.0 - target } else { target };
            let log_x = (libm::log(p) + crate::special::lgamma(a + 1.0) - a * libm::log(rate)) / a;
            x = libm::exp(log_x).max(1e-300);
            if !x.is_finite() || x <= 0.0 {
                x = (a / rate).max(1e-6);
            }
        }
        let ln_mass = |x: f64| {
            if upper {
                crate::special::incomplete::ln_gammq(a, rate * x)
            } else {
                self.ln_cdf(x)
            }
        };
        let consts = gamma_log_density_consts(a, rate);
        bracketed_newton(x, target, upper, !upper, ln_mass, |x| {
            gamma_log_density_with(a, rate, consts, x)
        })
    }
}

/// Marsaglia & Tsang (2000), "A simple method for generating gamma variables",
/// ACM Transactions on Mathematical Software 26(3):363–372: a rejection
/// sampler costing one normal and one uniform per attempt (the normal from the
/// normal quantile of a uniform), against the inverse CDF's Newton solve on
/// the incomplete gamma function. Draws `Gamma(α, 1)` and divides by the rate.
#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for Gamma {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        gamma_draw(self.shape, rng) / self.rate
    }
}

/// One `Gamma(α, 1)` draw, `α > 0`; see the `Gamma` sampler doc. Takes `α`
/// rather than a `Gamma` so the `NegBinomial` mixture can call it per draw.
/// For `α ≥ 1`, `d·v` with `d = α − ⅓`, `v = (1 + c·z)³`, `c = 1/√(9d)`,
/// `z ~ N(0, 1)`, accepted by the squeeze `u < 1 − 0.0331·z⁴` or else by
/// `ln u < z²/2 + d·(1 − v + ln v)`. For `α < 1`, `Gamma(α + 1)·u^{1/α}` (same
/// paper); the power underflows to 0 for very small `α`.
#[cfg(feature = "rng")]
pub(crate) fn gamma_draw(shape: f64, rng: &mut CommonStatsRng) -> f64 {
    if shape < 1.0 {
        let boost = libm::exp(libm::log(rng.uniform()) / shape);
        return gamma_draw(shape + 1.0, rng) * boost;
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / libm::sqrt(9.0 * d);
    loop {
        let z = norm_quantile(rng.uniform());
        let t = 1.0 + c * z;
        if t <= 0.0 {
            continue;
        }
        let v = t * t * t;
        let u = rng.uniform();
        let z2 = z * z;
        if u < 1.0 - 0.0331 * z2 * z2 || libm::log(u) < 0.5 * z2 + d * (1.0 - v + libm::log(v)) {
            return d * v;
        }
    }
}

/// Beta distribution `Beta(α, β)` on `[0, 1]`.
///
/// Convention: standard two-parameter Beta. Density
/// `x^{α−1}(1−x)^{β−1}/B(α,β)`. CDF `I_x(α,β)`, sf `I_{1−x}(β,α)`; quantile
/// `inv_beta_reg`, isf the root of the sf itself. Matches
/// `scipy.stats.beta(α, β)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beta {
    alpha: f64,
    beta: f64,
}

impl Beta {
    /// Construct with `α > 0`, `β > 0`.
    ///
    /// # Errors
    /// `DomainError` if non-finite or either `≤ 0`.
    ///
    /// ```
    /// use commonstats::dist::continuous::Beta;
    /// use commonstats::dist::ContinuousCdf;
    /// let b = Beta::new(2.0, 3.0).unwrap();
    /// assert!((b.cdf(1.0) - 1.0).abs() < 1e-15);
    /// assert!(Beta::new(0.0, 3.0).is_err());
    /// ```
    pub fn new(alpha: f64, beta: f64) -> Result<Self, StatError> {
        if !alpha.is_finite() || !beta.is_finite() || alpha <= 0.0 || beta <= 0.0 {
            return Err(StatError::DomainError(
                "Beta: alpha and beta must be finite and > 0",
            ));
        }
        Ok(Beta { alpha, beta })
    }
}

impl Distribution for Beta {
    fn support_min(&self) -> Bound {
        Bound::Finite(0.0)
    }
    fn support_max(&self) -> Bound {
        Bound::Finite(1.0)
    }
    fn mean(&self) -> Option<f64> {
        Some(self.alpha / (self.alpha + self.beta))
    }
    fn variance(&self) -> Option<f64> {
        let s = self.alpha + self.beta;
        Some(self.alpha * self.beta / (s * s * (s + 1.0)))
    }
}

impl ContinuousDensity for Beta {
    fn density(&self, x: f64) -> f64 {
        // `(0.0..=1.0).contains(&NaN)` is false, so the range check alone
        // would read a NaN `x` as out of support and return the constant 0.
        if x.is_nan() {
            return x;
        }
        if !(0.0..=1.0).contains(&x) {
            return 0.0;
        }
        libm::exp(self.log_density(x))
    }
    /// For `min(α, β) ≥ 8` and `0 < x < 1`, the saddle-point form of `brcomp`
    /// (DiDonato & Morris 1992): `ln(x^α·y^β/B(α, β)) − ln x − ln y`, `y = 1 −
    /// x`, with the first term `ln √(αβ/((α+β)·2π)) − bcorr(α, β) − α·(e₁ −
    /// ln(1 + e₁)) − β·(e₂ − ln(1 + e₂))`, `1 + e₁ = x/x₀`, `1 + e₂ = y/y₀`
    /// about the mean `x₀ = α/(α+β)`; the sum `(α−1)·ln x + (β−1)·ln y −
    /// ln B(α, β)` would keep the rounding of terms of size `α·|ln x|`.
    /// Otherwise that sum, with `ln y = log1p(−x)` (`ln` of the rounded `1 − x`
    /// errs by up to ε/2 absolute against `|ln y| ≈ x` for small `x`) and
    /// `0·ln 0 = 0`:
    /// `x = 0` at `α = 1` and `x = 1` at `β = 1` are the finite `ln β` and
    /// `ln α`.
    fn log_density(&self, x: f64) -> f64 {
        if x.is_nan() {
            return x;
        }
        if !(0.0..=1.0).contains(&x) {
            return f64::NEG_INFINITY;
        }
        let (a, b) = (self.alpha, self.beta);
        if a.min(b) >= 8.0 && 0.0 < x && x < 1.0 {
            let y = 1.0 - x;
            let lambda = beta_lambda(a, b, x, y);
            return ln_beta_saddle_const(a, b) + beta_saddle_dev(a, b, x, y, lambda)
                - libm::log(x)
                - libm::log1p(-x);
        }
        let xlogy = |c: f64, l: f64| if c == 0.0 { 0.0 } else { c * l };
        xlogy(a - 1.0, libm::log(x)) + xlogy(b - 1.0, libm::log1p(-x)) - crate::special::lbeta(a, b)
    }
}

impl ContinuousCdf for Beta {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            crate::special::betai(self.alpha, self.beta, x)
        }
    }
    /// `I_{1−x}(β, α)`, the kernel given the exact `x` beside `1 − x`, which
    /// evaluates the smaller of the two tails directly; `x` is never rebuilt
    /// as `1 − (1 − x)` (`Beta(2, 3).sf(0.99999999)` ≈ 4.0e-24).
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            1.0
        } else if x >= 1.0 {
            0.0
        } else {
            let (m, e, complement) =
                crate::special::incomplete::betai_parts(self.beta, self.alpha, 1.0 - x, x);
            crate::special::incomplete::pq_from_parts(m, e, complement).0
        }
    }
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(1.0);
        }
        Ok(crate::special::inv_beta_reg(self.alpha, self.beta, p))
    }
    /// For `q ≤ 0.5` the root of `sf(x) = q`, bisected on `x` itself with the
    /// complement `1 − I_x(α, β)` evaluated directly, so a root near 0 keeps
    /// its relative precision (`1 − I⁻¹_q(β, α)` would keep only the absolute
    /// error of a value near 1). For `q > 0.5`, `quantile(1 − q)`, exact in
    /// f64 for `q ∈ [0.5, 1]`.
    fn isf(&self, q: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&q) {
            return Err(StatError::ProbabilityOutOfRange(q));
        }
        if q == 0.0 {
            return Ok(1.0);
        }
        if q == 1.0 {
            return Ok(0.0);
        }
        Ok(if q > 0.5 {
            crate::special::inv_beta_reg(self.alpha, self.beta, 1.0 - q)
        } else {
            crate::special::inverse::inv_beta_reg_upper(self.alpha, self.beta, q)
        })
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for Beta {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        self.quantile(rng.uniform())
            .expect("uniform() ∈ (0,1) is a valid quantile arg")
    }
}

/// Inverse Gaussian (Wald) distribution with mean `μ` and shape `λ`.
///
/// Convention: `(μ, λ)` parameterization (Chhikara & Folks 1989). Density
/// `√(λ/(2πx³))·exp(−λ(x−μ)²/(2μ²x))` for `x > 0`; variance `μ³/λ`, so a GLM
/// dispersion `φ` corresponds to `λ = 1/φ`. With `a = √(λ/x)(x/μ − 1)` and
/// `b = √(λ/x)(x/μ + 1)`: CDF `Φ(a) + e^{2λ/μ}Φ(−b)`, sf `Φ(−a) − e^{2λ/μ}Φ(−b)`
/// (computed directly, not `1 − cdf`; where the two sf terms nearly cancel —
/// far above the mean, or `λ/x` tiny — as `Φ(−a)` times a series for their
/// relative difference). Quantile = normal-term seed + Newton.
/// Matches `scipy.stats.invgauss(μ/λ, scale=λ)` and R
/// `statmod::pinvgauss(x, mean=μ, shape=λ)` (`tests/fixtures/dist_inversegaussian_*.json`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussian {
    mean: f64,
    shape: f64,
}

impl InverseGaussian {
    /// Construct with `mean μ > 0`, `shape λ > 0`.
    ///
    /// # Errors
    /// `DomainError` if non-finite or either `≤ 0`.
    ///
    /// ```
    /// use commonstats::dist::continuous::InverseGaussian;
    /// use commonstats::dist::{ContinuousCdf, Distribution};
    /// let ig = InverseGaussian::new(1.5, 2.0).unwrap();
    /// assert_eq!(ig.mean(), Some(1.5));
    /// assert!((ig.cdf(ig.quantile(0.9).unwrap()) - 0.9).abs() < 1e-14);
    /// assert!(InverseGaussian::new(1.5, 0.0).is_err());
    /// ```
    pub fn new(mean: f64, shape: f64) -> Result<Self, StatError> {
        if !mean.is_finite() || !shape.is_finite() || mean <= 0.0 || shape <= 0.0 {
            return Err(StatError::DomainError(
                "InverseGaussian: mean and shape must be finite and > 0",
            ));
        }
        Ok(InverseGaussian { mean, shape })
    }
}

impl Distribution for InverseGaussian {
    fn support_min(&self) -> Bound {
        Bound::Finite(0.0)
    }
    fn support_max(&self) -> Bound {
        Bound::PosInfinity
    }
    fn mean(&self) -> Option<f64> {
        Some(self.mean)
    }
    fn variance(&self) -> Option<f64> {
        Some(self.mean * self.mean * self.mean / self.shape)
    }
    fn skewness(&self) -> Option<f64> {
        Some(3.0 * libm::sqrt(self.mean / self.shape))
    }
    fn kurtosis(&self) -> Option<f64> {
        Some(15.0 * self.mean / self.shape)
    }
}

impl ContinuousDensity for InverseGaussian {
    fn density(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        libm::exp(self.log_density(x))
    }
    fn log_density(&self, x: f64) -> f64 {
        self.log_density_with(self.ln_norm_const(), x)
    }
}

impl InverseGaussian {
    /// `½·ln(λ/2π)`, the `x`-free term of `log_density`, formed once per
    /// quantile solve.
    fn ln_norm_const(&self) -> f64 {
        0.5 * libm::log(self.shape / (2.0 * core::f64::consts::PI))
    }
    /// `log_density` given `c = ln_norm_const()`.
    fn log_density_with(&self, c: f64, x: f64) -> f64 {
        if x <= 0.0 || x == f64::INFINITY {
            return f64::NEG_INFINITY;
        }
        let (mu, lam) = (self.mean, self.shape);
        let d = x - mu;
        // λ(x−μ)²/(2μ²x) as `½·(λ/μ)·(d/μ)·(d/x)`: each factor is a ratio of
        // like-scaled quantities, so neither `d²`, `μ²` nor `λ·d` is formed,
        // any of which over- or underflows at extreme scales (`μ, λ ~ 1e±160`)
        // while the term itself is O(1). Unless `λ/μ` itself leaves the f64
        // range, it overflows only where the term does (x ≪ μ, where −∞ is the
        // right log density).
        c - 1.5 * libm::log(x) - 0.5 * (lam / mu) * (d / mu) * (d / x)
    }
}

impl ContinuousCdf for InverseGaussian {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if x == f64::INFINITY {
            return 1.0;
        }
        let (a, b) = self.shuster_args(x);
        // `clamp` keeps a NaN `x` NaN; `f64::min`/`max` would return the bound.
        (0.5 * crate::special::erfc(-a / core::f64::consts::SQRT_2) + reflected_term(a, b))
            .clamp(0.0, 1.0)
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        if x == f64::INFINITY {
            return 0.0;
        }
        let (a, b) = self.shuster_args(x);
        // sf = ½e^{−u²}·[erfcx(u) − erfcx(u + h)] with u = a/√2 and
        // h = (b − a)/√2 = √(2λ/x). The bracket cancels when h is small against
        // max(u, 1): far above the mean (h/u = 2μ/(x − μ)), and anywhere when
        // λ/x is tiny. Its value is `erfcx(u)·erfcx_drop(u, h)`, so
        // sf = Φ(−a)·erfcx_drop(u, h), formed without the subtraction. Below the
        // mean `|u| < h/2`, so the switch there also bounds `u > −1/8`. Past the
        // switch the drop is ≥ ~1/8, and the direct difference below loses at
        // most about one digit.
        let u = a / core::f64::consts::SQRT_2;
        let h = libm::sqrt(2.0 * self.shape / x);
        if h <= 0.25 * u.max(1.0) {
            return 0.5 * crate::special::erfc(u) * erfcx_drop(u, h);
        }
        // `clamp`, not `max`: see `cdf`.
        (0.5 * crate::special::erfc(u) - reflected_term(a, b)).clamp(0.0, 1.0)
    }
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        positive_quantile(p, false, |t, upper| self.solve(t, upper))
    }
    /// Newton on `ln sf(x) − ln q` for `q ≤ 0.5`; see `positive_quantile` and `solve`.
    fn isf(&self, q: f64) -> Result<f64, StatError> {
        positive_quantile(q, true, |t, upper| self.solve(t, upper))
    }
}

/// `e^{2λ/μ}·Φ(−b)`, the second CDF term, formed as `½·e^{−a²/2}·erfcx(b/√2)`
/// through the exact identity `2λ/μ − b²/2 = −a²/2`. Neither `e^{2λ/μ}` (which
/// overflows for `λ/μ ≳ 355`) nor `Φ(−b)` (which underflows for `b ≳ 38`,
/// while the product is still `O(1)` near `x = μ`) is ever formed on its own.
fn reflected_term(a: f64, b: f64) -> f64 {
    0.5 * libm::exp(-0.5 * a * a) * erfcx(b / core::f64::consts::SQRT_2)
}

/// Relative drop `1 − erfcx(u + h)/erfcx(u)` for `u > −1/8`, `0 < h ≤ ¼·max(u, 1)`,
/// without forming the difference.
///
/// Taylor's series in `h` with `dⁿ/duⁿ erfcx(u) = (−2)ⁿ n!·Eₙ(u)`,
/// `Eₙ = e^{u²}·iⁿerfc(u)` (Abramowitz & Stegun 7.2.9), gives
/// `Σ_{n≥1} (−1)^{n+1} (2h)ⁿ Eₙ/E₀`. The recurrence `E_{n−2} = 2n·Eₙ + 2u·E_{n−1}`
/// (A&S 7.2.5) makes `E_{n−1}/(2Eₙ) = t_{n+1}`, the tails of the Laplace
/// continued fraction `t_k = u + (k/2)/t_{k+1}` (A&S 7.1.14; `t₁ = 1/(√π·erfcx(u))`),
/// so the series is `(h/t₂)(1 − (h/t₃)(1 − (h/t₄)(1 − …)))`: an alternating
/// sum whose ratio `h/t_k` is ≤ ~0.3 here (`t_k > u`, and `t₂ ≥ 0.8` for
/// `u > −1/8`), with no cancellation.
///
/// The tails are computed two ways, by stability. For `u ≥ 1.5`, backward from
/// depth `n` (`t_{n+1} ≈ u`), fused with a nested evaluation of the first `m`
/// terms: `m` is the least with `(h/u)^m ≤ 5e-18` (every ratio `h/t_k < h/u`), at
/// most 29. The depth `n = ⌊6 + 34/u + 200/u² + m/3⌋ + 1`, kept in `[m + 1, 120]`,
/// is a fit to the least depth that holds the truncation to 5e-18 against mpmath
/// (for small `h` that least depth is 115 at `u = 1.5`, 26 at 4, 12 at 10; the
/// fit gives 118, 28, 12); measured truncation ≤ 6.5e-18 over the branch.
/// Below 1.5 the continued fraction converges too slowly (depth 120 leaves
/// 2e-12 at `u = 1`), and the forward recurrence `t_{k+1} = (k/2)/(t_k − u)`
/// from `t₁` is used instead (for `u < 0` it has no subtraction at all),
/// summing until the terms drop below 1e-17 of the sum. Measured against
/// mpmath: ≤ 1.1e-15 relative for the forward branch (the `t₁ − u`
/// cancellation amplifies `erfcx`'s own error ~4×) and ≤ 3.7e-16 (3 ulp) for
/// the backward branch.
fn erfcx_drop(u: f64, h: f64) -> f64 {
    const TERMS: usize = 28;
    if u >= 1.5 {
        // ln(5e-18).
        const LN_TOL: f64 = -39.837_093_761_458_725;
        let m = ((LN_TOL / libm::log(h / u)) as usize + 1).min(TERMS + 1);
        let n = ((6.0 + (34.0 + 200.0 / u) / u + m as f64 / 3.0) as usize + 1).clamp(m + 1, 120);
        let mut t = u;
        let mut nested = 1.0;
        for k in (2..=n).rev() {
            t = u + 0.5 * k as f64 / t;
            if k <= m + 1 {
                nested = if k == 2 {
                    h / t * nested
                } else {
                    1.0 - h / t * nested
                };
            }
        }
        return nested;
    }
    let mut t = 1.0 / (libm::sqrt(core::f64::consts::PI) * erfcx(u));
    let mut term = 1.0;
    let mut sum = 0.0;
    for k in 1..=4 * TERMS {
        t = 0.5 * k as f64 / (t - u);
        term *= -h / t;
        sum -= term;
        if libm::fabs(term) < 1e-17 * sum {
            break;
        }
    }
    sum
}

impl InverseGaussian {
    /// Shuster's arguments `a = √(λ/x)(x − μ)/μ`, `b = √(λ/x)(x + μ)/μ` for a
    /// finite `x > 0` (Shuster 1968, JASA 63:1514). `x − μ` is exact near the
    /// median (Sterbenz), where `x/μ − 1` would round first.
    fn shuster_args(&self, x: f64) -> (f64, f64) {
        let r = libm::sqrt(self.shape / x) / self.mean;
        ((x - self.mean) * r, (x + self.mean) * r)
    }
    /// `ln cdf(x)` for finite `x > 0`: `log1p(−sf)` where the cdf is above ½,
    /// as `ln` of a cdf within 1e-16 of 1 would keep none of the sf's digits;
    /// `ln cdf` below it; or where the cdf is subnormal or underflows (far
    /// below the mean), its log form. There both CDF terms are `½e^{−a²/2}`
    /// times `erfcx` of a positive argument, `Φ(a) = ½e^{−a²/2}·erfcx(−a/√2)`
    /// and `reflected_term`, so
    /// `ln cdf = −a²/2 + ln(½(erfcx(−a/√2) + erfcx(b/√2)))`.
    #[doc(hidden)]
    pub fn ln_cdf(&self, x: f64) -> f64 {
        let cdf = self.cdf(x);
        if cdf > 0.5 {
            return libm::log1p(-self.sf(x));
        }
        if cdf >= f64::MIN_POSITIVE || cdf.is_nan() {
            return libm::log(cdf);
        }
        let (a, b) = self.shuster_args(x);
        let s = core::f64::consts::SQRT_2;
        let ln_cdf = -0.5 * a * a + libm::log(0.5 * (erfcx(-a / s) + erfcx(b / s)));
        // Below half the smallest subnormal the cdf is under every target and
        // reads as 0, as `cdf` itself does. Its log grows like `−λ/(2x)`
        // without bound as x → 0, and `bracketed_newton` would form its step
        // from `ln cdf − ln pdf`, two such numbers whose difference is lost.
        if ln_cdf < -1075.0 * core::f64::consts::LN_2 {
            f64::NEG_INFINITY
        } else {
            ln_cdf
        }
    }
    /// `ln sf(x)` for finite `x > 0`, the mirror of `ln_cdf`: `log1p(−cdf)`
    /// where the sf is above ½, `ln sf` below it, or where the sf is subnormal
    /// or underflows (far above the mean), its log form.
    /// Each branch of `sf` is `½e^{−a²/2}` times a factor in range, with
    /// `u = a/√2` and `h = √(2λ/x)`: `erfcx(u)·erfcx_drop(u, h)` past the same
    /// switch, else `erfcx(u) − erfcx(b/√2)`, so `ln sf = −a²/2 + ln(½·factor)`.
    /// Below half the smallest subnormal the sf reads as 0, as in `ln_cdf`:
    /// `ln sf` falls like `−λx/(2μ²)` without bound, and so does `ln pdf`,
    /// whose difference `bracketed_newton` would form its step from.
    #[doc(hidden)]
    pub fn ln_sf(&self, x: f64) -> f64 {
        let sf = self.sf(x);
        if sf > 0.5 {
            return libm::log1p(-self.cdf(x));
        }
        if sf >= f64::MIN_POSITIVE || sf.is_nan() {
            return libm::log(sf);
        }
        let (a, b) = self.shuster_args(x);
        let s = core::f64::consts::SQRT_2;
        let u = a / s;
        let h = libm::sqrt(2.0 * self.shape / x);
        let factor = if h <= 0.25 * u.max(1.0) {
            erfcx(u) * erfcx_drop(u, h)
        } else {
            erfcx(u) - erfcx(b / s)
        };
        let ln_sf = -0.5 * a * a + libm::log(0.5 * factor);
        if ln_sf < -1075.0 * core::f64::consts::LN_2 {
            f64::NEG_INFINITY
        } else {
            ln_sf
        }
    }
    /// Root of `cdf(x) = target` (`upper = false`) or `sf(x) = target`
    /// (`upper = true`), `target ∈ (0, ½]`: normal-term seed, then
    /// `bracketed_newton` in `x` (`ln sf` is asymptotically linear in `x`, slope
    /// `−λ/(2μ²)`).
    fn solve(&self, target: f64, upper: bool) -> f64 {
        let (mu, lam) = (self.mean, self.shape);
        // Seed: root of the leading normal term alone, `a(x) = z` with
        // `Φ(z) = target` (lower) or `Φ(−z) = target` (upper). In `s = √x` this is
        // `√λ·s²/μ − z·s − √λ = 0`, so `x = μ²(z + √(z² + 4λ/μ))²/(4λ)`. The
        // reflected term is added to cdf and subtracted from sf, so at the seed
        // `cdf ≥ target` (lower) or `sf ≤ target` (upper): either way the seed
        // sits at or above the root. A moment-matched Gamma seed is not used:
        // the IG lower tail (`≈ e^{−λ/(2x)}`) is far thinner than Gamma's
        // `x^k`, so that seed lands where the pdf underflows.
        let z = if upper {
            -norm_quantile(target)
        } else {
            norm_quantile(target)
        };
        let c = 4.0 * lam / mu;
        let r = libm::sqrt(z * z + c);
        // z + √(z² + c), rewritten for z < 0 to avoid cancellation.
        let w = if z >= 0.0 { z + r } else { c / (r - z) };
        let x = (mu * w) * (mu * w) / (4.0 * lam);
        let ln_mass = |x: f64| {
            if upper { self.ln_sf(x) } else { self.ln_cdf(x) }
        };
        let c = self.ln_norm_const();
        bracketed_newton(x, target, upper, false, ln_mass, |x| {
            self.log_density_with(c, x)
        })
    }
}

/// Michael, Schucany & Haas (1976), "Generating random variates using
/// transformations with multiple roots", The American Statistician
/// 30(2):88–90. With `ν ~ N(0, 1)` and `y = ν²`, the two roots
/// `x₁ ≤ μ ≤ x₂` of `λ(x − μ)²/(μ²x) = y` satisfy `x₁·x₂ = μ²`; return `x₁`
/// with probability `μ/(μ + x₁)`, else `x₂`. Two uniforms per draw: `ν` from
/// the normal quantile of the first, the root choice from the second.
#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for InverseGaussian {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        let mu = self.mean;
        let nu = norm_quantile(rng.uniform());
        // With t = μy/(2λ), x₂ = μ·w and x₁ = μ/w for w = 1 + t + √(t(2 + t)).
        // The textbook x₁ = μ + μ²y/(2λ) − (μ/(2λ))√(4μλy + μ²y²) cancels when
        // μy/λ is large; w has no subtraction. `√t·√(2 + t)` keeps `t²` from
        // overflowing. x₁ is accepted with probability μ/(μ + x₁) = w/(1 + w).
        let t = mu * nu * nu / (2.0 * self.shape);
        let w = 1.0 + t + libm::sqrt(t) * libm::sqrt(2.0 + t);
        if rng.uniform() * (1.0 + w) <= w {
            mu / w
        } else {
            mu * w
        }
    }
}
