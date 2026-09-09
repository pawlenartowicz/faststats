//! The 11 continuous distributions of the `dist` suite.
//!
//! Quantile strategy for t, χ², F, and Gamma: Cornish–Fisher or Wilson–Hilferty
//! seed (Abramowitz & Stegun §26.7; Wilson & Hilferty 1931) followed by Newton
//! iteration on the respective CDF. Any change to the Newton tolerance or seed
//! formula touches all four distributions — keep them consistent.

#[cfg(feature = "rng")]
use crate::dist::Sampler;
use crate::dist::{
    Bound, ContinuousCdf, ContinuousDensity, Distribution, gamma_log_density, norm_quantile,
};
use crate::error::StatError;
#[cfg(feature = "rng")]
use crate::rng::CommonStatsRng;

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
/// regularized incomplete beta; quantile via Cornish–Fisher seed + Newton.
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
        let df = self.df;
        let c = crate::special::lgamma(0.5 * (df + 1.0))
            - crate::special::lgamma(0.5 * df)
            - 0.5 * libm::log(df * core::f64::consts::PI);
        c - 0.5 * (df + 1.0) * libm::log(1.0 + x * x / df)
    }
}

impl ContinuousCdf for StudentT {
    fn cdf(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return if x > 0.0 { 1.0 } else { 0.0 };
        }
        let df = self.df;
        let z = df / (df + x * x);
        let half = 0.5 * crate::special::betai(0.5 * df, 0.5, z);
        if x >= 0.0 { 1.0 - half } else { half }
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
        if p == 0.5 {
            return Ok(0.0);
        }
        let df = self.df;
        let upper = p > 0.5;
        let q = if upper { p } else { 1.0 - p };
        // Initial guess (Cornish–Fisher expansion, Abramowitz & Stegun §26.7), then Newton.
        let mut x = if df > 200.0 {
            let z = norm_quantile(q);
            let z2 = z * z;
            z + (z * (z2 + 1.0)) / (4.0 * df)
                + (z * (5.0 * z2 * z2 + 16.0 * z2 + 3.0)) / (96.0 * df * df)
        } else if (df - 1.0).abs() < 1e-9 {
            (libm::tan(core::f64::consts::PI * (q - 0.5))).abs()
        } else if (df - 2.0).abs() < 1e-9 {
            let alpha = 4.0 * q * (1.0 - q);
            libm::sqrt(2.0 / alpha - 2.0) * if q > 0.5 { 1.0 } else { -1.0 }
        } else {
            norm_quantile(q)
        };
        if df > 1.0e5 {
            return Ok(if upper { x } else { -x });
        }
        let log_norm_const = crate::special::lgamma(0.5 * (df + 1.0))
            - crate::special::lgamma(0.5 * df)
            - 0.5 * libm::log(df * core::f64::consts::PI);
        for _ in 0..80 {
            // cdf via the same betai form as above, inlined for the (possibly
            // negative) seed x.
            let cdf = {
                let z = df / (df + x * x);
                let half = 0.5 * crate::special::betai(0.5 * df, 0.5, z);
                if x >= 0.0 { 1.0 - half } else { half }
            };
            let pdf_log = log_norm_const - 0.5 * (df + 1.0) * libm::log(1.0 + x * x / df);
            let pdf = libm::exp(pdf_log);
            if pdf <= 0.0 || !pdf.is_finite() {
                break;
            }
            let new_x = x - (cdf - q) / pdf;
            if !new_x.is_finite() {
                break;
            }
            let rel_change = (new_x - x).abs() / (1.0 + x.abs());
            x = new_x;
            if rel_change < 1e-14 {
                break;
            }
        }
        Ok(if upper { x } else { -x })
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
/// rate 1/2)`. CDF `P(k/2, x/2)`; sf via `Q`. Quantile = Wilson–Hilferty seed +
/// Newton on the gamma CDF.
/// `density(0)=0`/`log_density(0)=−∞` for `k=1` (singular). Matches
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
        if x <= 0.0 {
            return 0.0;
        }
        libm::exp(self.log_density(x))
    }
    fn log_density(&self, x: f64) -> f64 {
        // χ²(k) = Gamma(shape k/2, rate 1/2).
        gamma_log_density(0.5 * self.k, 0.5, x)
    }
}

impl ContinuousCdf for ChiSquared {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        crate::special::gammp(0.5 * self.k, 0.5 * x)
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        crate::special::gammq(0.5 * self.k, 0.5 * x)
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
        let k = self.k;
        // Wilson–Hilferty seed (Wilson & Hilferty 1931, PNAS 17:684).
        let z = norm_quantile(p);
        let h = 2.0 / (9.0 * k);
        let wh_base = 1.0 - h + z * libm::sqrt(h);
        let mut x = k * (wh_base * wh_base * wh_base);
        if !x.is_finite() || x <= 0.0 {
            // WH fails for very small p. Fall back to the Taylor approximation:
            // χ²(k) = Gamma(k/2, 1/2); P(α,βx) ≈ (βx)^α/(α·Γ(α)) → p.
            let a = 0.5 * k;
            let log_x = (libm::log(p) + crate::special::lgamma(a) - a * libm::log(0.5_f64)) / a;
            x = libm::exp(log_x).max(1e-300);
            if !x.is_finite() || x <= 0.0 {
                x = k.max(1e-6);
            }
        }
        let a = 0.5 * k;
        for _ in 0..80 {
            let cdf = crate::special::gammp(a, 0.5 * x);
            let ln_pdf = (a - 1.0) * libm::log(0.5 * x)
                - 0.5 * x
                - core::f64::consts::LN_2
                - crate::special::lgamma(a);
            let pdf = libm::exp(ln_pdf);
            if !pdf.is_finite() || pdf <= 0.0 {
                break;
            }
            let mut new_x = x - (cdf - p) / pdf;
            if !new_x.is_finite() || new_x <= 0.0 {
                new_x = 0.5 * x;
            }
            let rel_change = (new_x - x).abs() / (1.0 + x.abs());
            x = new_x;
            if rel_change < 1e-13 {
                break;
            }
        }
        Ok(x)
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for ChiSquared {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        self.quantile(rng.uniform())
            .expect("uniform() ∈ (0,1) is a valid quantile arg")
    }
}

/// Fisher–Snedecor F distribution `F(dfn, dfd)`.
///
/// Convention: `dfn` numerator df, `dfd` denominator df (both `> 0`). Mean
/// defined for `dfd>2`, variance `dfd>4`. CDF
/// `1 − I_{dfd/(dfd+dfn·x)}(dfd/2, dfn/2)`; sf is the un-complemented `betai`.
/// Quantile = Wilson–Hilferty seed + Newton.
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
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let (a, b) = (0.5 * self.dfn, 0.5 * self.dfd);
        a * libm::log(self.dfn / self.dfd) + (a - 1.0) * libm::log(x)
            - (a + b) * libm::log(1.0 + self.dfn / self.dfd * x)
            - crate::special::lbeta(a, b)
    }
}

impl ContinuousCdf for FisherF {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if !x.is_finite() {
            return 1.0;
        }
        let z = self.dfd / (self.dfd + self.dfn * x);
        1.0 - crate::special::betai(0.5 * self.dfd, 0.5 * self.dfn, z)
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        if !x.is_finite() {
            return 0.0;
        }
        let z = self.dfd / (self.dfd + self.dfn * x);
        crate::special::betai(0.5 * self.dfd, 0.5 * self.dfn, z)
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
        let dfn = self.dfn;
        let z = norm_quantile(p);
        let h = 2.0 / (9.0 * dfn);
        let wh_base = 1.0 - h + z * libm::sqrt(h);
        let chi_seed = dfn * (wh_base * wh_base * wh_base);
        let mut x = (chi_seed / dfn).max(1e-6);
        for _ in 0..80 {
            let cdf = self.cdf(x);
            let pdf = self.density(x);
            if !pdf.is_finite() || pdf <= 0.0 {
                break;
            }
            let mut new_x = x - (cdf - p) / pdf;
            if !new_x.is_finite() || new_x <= 0.0 {
                new_x = 0.5 * x;
            }
            let rel_change = (new_x - x).abs() / (1.0 + x.abs());
            x = new_x;
            if rel_change < 1e-13 {
                break;
            }
        }
        Ok(x)
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
        if x < self.a || x > self.b {
            0.0
        } else {
            1.0 / (self.b - self.a)
        }
    }
    fn log_density(&self, x: f64) -> f64 {
        if x < self.a || x > self.b {
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
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        Ok(self.a + p * (self.b - self.a))
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
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(-libm::log(1.0 - p) / self.rate)
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
/// `0.5 + atan((x−loc)/scale)/π`. Matches `scipy.stats.cauchy(loc, scale)`.
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

impl ContinuousDensity for Cauchy {
    fn density(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        1.0 / (core::f64::consts::PI * self.scale * (1.0 + z * z))
    }
    fn log_density(&self, x: f64) -> f64 {
        let z = (x - self.loc) / self.scale;
        -libm::log(core::f64::consts::PI * self.scale) - libm::log(1.0 + z * z)
    }
}

impl ContinuousCdf for Cauchy {
    fn cdf(&self, x: f64) -> f64 {
        0.5 + libm::atan((x - self.loc) / self.scale) / core::f64::consts::PI
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
        // tan(π(p−½)) cancels near p≈0 (→ −π/2) and p≈1 (→ +π/2). Use the
        // cotangent form instead: tan(π(p−½)) = −cot(πp).
        // Near p≈0: −cot(πp) = −cos(πp)/sin(πp) — accurate since sin(πp)≈πp.
        // Near p≈1: use symmetry q=1−p; cot(πp) = −cot(πq), so
        //   −cot(πp) = cot(πq) = cos(πq)/sin(πq) — accurate since sin(πq)≈πq.
        let cot = if p <= 0.5 {
            let pi_p = core::f64::consts::PI * p;
            -libm::cos(pi_p) / libm::sin(pi_p)
        } else {
            let pi_q = core::f64::consts::PI * (1.0 - p);
            libm::cos(pi_q) / libm::sin(pi_q)
        };
        Ok(self.loc + self.scale * cot)
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for Cauchy {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        let u = rng.uniform();
        // u ∈ (0,1) so u ≤ 0.5 is possible; both branches are accurate.
        let cot = if u <= 0.5 {
            let pi_u = core::f64::consts::PI * u;
            -libm::cos(pi_u) / libm::sin(pi_u)
        } else {
            let pi_q = core::f64::consts::PI * (1.0 - u);
            libm::cos(pi_q) / libm::sin(pi_q)
        };
        self.loc + self.scale * cot
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
        if x < 0.0 {
            return f64::NEG_INFINITY;
        }
        let (k, lam) = (self.shape, self.scale);
        let z = x / lam;
        libm::log(k) - libm::log(lam) + (k - 1.0) * libm::log(z) - libm::pow(z, k)
    }
}

impl ContinuousCdf for Weibull {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        -libm::expm1(-libm::pow(x / self.scale, self.shape)) // 1 - e^{-(x/λ)^k}
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        libm::exp(-libm::pow(x / self.scale, self.shape))
    }
    fn quantile(&self, p: f64) -> Result<f64, StatError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatError::ProbabilityOutOfRange(p));
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(self.scale * libm::pow(-libm::log(1.0 - p), 1.0 / self.shape))
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for Weibull {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        self.scale * libm::pow(-libm::log(rng.uniform()), 1.0 / self.shape)
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
        libm::exp(-0.5 * z * z) / (x * self.sigma * libm::sqrt(2.0 * core::f64::consts::PI))
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
/// Quantile = Wilson–Hilferty seed (`χ²(2α)/(2β)`) + Newton on the gamma CDF.
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
        if x <= 0.0 {
            return 0.0;
        }
        libm::exp(gamma_log_density(self.shape, self.rate, x))
    }
    fn log_density(&self, x: f64) -> f64 {
        gamma_log_density(self.shape, self.rate, x)
    }
}

impl ContinuousCdf for Gamma {
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        crate::special::gammp(self.shape, self.rate * x)
    }
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 1.0;
        }
        crate::special::gammq(self.shape, self.rate * x)
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
        let a = self.shape;
        // Seed: χ²(2α) Wilson–Hilferty (Wilson & Hilferty 1931, PNAS 17:684), then x = chi2/(2β). (chi2 has df k=2α.)
        let k = 2.0 * a;
        let z = norm_quantile(p);
        let h = 2.0 / (9.0 * k);
        let wh_base = 1.0 - h + z * libm::sqrt(h);
        let chi = k * (wh_base * wh_base * wh_base);
        let mut x = chi / (2.0 * self.rate);
        if !x.is_finite() || x <= 0.0 {
            // WH fails for very small p (chi becomes negative). Fall back to
            // the first-order approximation P(α,βx) ≈ (βx)^α/(α·Γ(α)) → p.
            let log_x = (libm::log(p) + crate::special::lgamma(a) - a * libm::log(self.rate)) / a;
            x = libm::exp(log_x).max(1e-300);
            if !x.is_finite() || x <= 0.0 {
                x = (a / self.rate).max(1e-6);
            }
        }
        for _ in 0..80 {
            let cdf = crate::special::gammp(a, self.rate * x);
            let ln_pdf = gamma_log_density(a, self.rate, x);
            let pdf = libm::exp(ln_pdf);
            if !pdf.is_finite() || pdf <= 0.0 {
                break;
            }
            let mut new_x = x - (cdf - p) / pdf;
            if !new_x.is_finite() || new_x <= 0.0 {
                new_x = 0.5 * x;
            }
            let rel_change = (new_x - x).abs() / (1.0 + x.abs());
            x = new_x;
            if rel_change < 1e-13 {
                break;
            }
        }
        Ok(x)
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for Gamma {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        self.quantile(rng.uniform())
            .expect("uniform() ∈ (0,1) is a valid quantile arg")
    }
}

/// Beta distribution `Beta(α, β)` on `[0, 1]`.
///
/// Convention: standard two-parameter Beta. Density
/// `x^{α−1}(1−x)^{β−1}/B(α,β)`. CDF `I_x(α,β)`; quantile `inv_beta_reg`.
/// Matches `scipy.stats.beta(α, β)`.
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
        if !(0.0..=1.0).contains(&x) {
            return 0.0;
        }
        libm::exp(self.log_density(x))
    }
    fn log_density(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            return f64::NEG_INFINITY;
        }
        (self.alpha - 1.0) * libm::log(x) + (self.beta - 1.0) * libm::log(1.0 - x)
            - crate::special::lbeta(self.alpha, self.beta)
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
}

#[cfg(all(feature = "dist", feature = "rng"))]
impl Sampler for Beta {
    fn sample(&self, rng: &mut CommonStatsRng) -> f64 {
        self.quantile(rng.uniform())
            .expect("uniform() ∈ (0,1) is a valid quantile arg")
    }
}
