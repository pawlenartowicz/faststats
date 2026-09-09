//! Distribution object suite (feature `dist`).
//!
//! A decomposed trait set: [`Distribution`] (support + moments),
//! [`ContinuousDensity`]/[`DiscreteMass`] (pdf/pmf), [`ContinuousCdf`]/
//! [`DiscreteCdf`] (cdf + quantile), and [`Sampler`] (inverse-CDF sampling,
//! also gated on `rng`). Every constructor returns `Result<_, StatError>`;
//! quantiles return `Err(ProbabilityOutOfRange)` for `p ∉ [0,1]`.
//!
//! Moments are **`None`** where undefined for the parameters (never a NaN
//! sentinel); `kurtosis` is **excess** (Kurt − 3), following `scipy.stats`.

use crate::error::StatError;
#[cfg(feature = "rng")]
use crate::rng::CommonStatsRng;

pub mod continuous;
pub mod discrete;

/// Typed support boundary — no `±∞` float sentinel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bound {
    /// Lower-unbounded support (`−∞`).
    NegInfinity,
    /// A finite boundary value.
    Finite(f64),
    /// Upper-unbounded support (`+∞`).
    PosInfinity,
}

impl Bound {
    /// Numeric value of the bound (`±INFINITY` for the infinite variants).
    pub fn as_f64(self) -> f64 {
        match self {
            Bound::NegInfinity => f64::NEG_INFINITY,
            Bound::Finite(x) => x,
            Bound::PosInfinity => f64::INFINITY,
        }
    }
}

/// Support boundaries + optional moments.
///
/// Moments are `None` when undefined for the parameters. `kurtosis` is
/// **excess** (Kurt − 3). The `std_dev` default (`variance().map(sqrt)`) is
/// correct for all 17 distributions in this suite.
pub trait Distribution {
    /// Lower edge of the support.
    fn support_min(&self) -> Bound;
    /// Upper edge of the support.
    fn support_max(&self) -> Bound;
    /// Mean, or `None` if undefined.
    fn mean(&self) -> Option<f64> {
        None
    }
    /// Variance, or `None` if undefined.
    fn variance(&self) -> Option<f64> {
        None
    }
    /// Standard deviation; defaults to `variance().map(sqrt)`.
    fn std_dev(&self) -> Option<f64> {
        self.variance().map(libm::sqrt)
    }
    /// Skewness, or `None` if undefined.
    fn skewness(&self) -> Option<f64> {
        None
    }
    /// Excess kurtosis (Kurt − 3), or `None` if undefined.
    fn kurtosis(&self) -> Option<f64> {
        None
    }
    /// Differential/Shannon entropy, or `None` if not provided.
    fn entropy(&self) -> Option<f64> {
        None
    }
}

/// Continuous density.
///
/// `log_density` is **required** and computed directly (not `density(x).ln()`)
/// so it stays finite in tails where `density` underflows to `0`. Outside the
/// support, `density` → `0.0` and `log_density` → `NEG_INFINITY`.
pub trait ContinuousDensity: Distribution {
    /// Probability density at `x`.
    fn density(&self, x: f64) -> f64;
    /// Natural log of the density at `x`.
    fn log_density(&self, x: f64) -> f64;
}

/// Discrete mass.
///
/// `log_mass` is **required** (same tail-underflow reason as `log_density`).
pub trait DiscreteMass: Distribution {
    /// Probability mass at integer `k`.
    fn mass(&self, k: i64) -> f64;
    /// Natural log of the mass at `k`.
    fn log_mass(&self, k: i64) -> f64;
}

/// Continuous CDF + quantile.
///
/// `sf` defaults to `1 − cdf` but MUST be overridden where cancellation bites
/// near `p ≈ 1`. `quantile` is required and returns `Err(ProbabilityOutOfRange)`
/// for `p ∉ [0,1]`; every impl supplies a closed form or a ported special-function
/// inverse.
pub trait ContinuousCdf: Distribution {
    /// Cumulative probability `P(X ≤ x)`.
    fn cdf(&self, x: f64) -> f64;
    /// Survival function `P(X > x)`; override to avoid `1 − cdf` cancellation.
    fn sf(&self, x: f64) -> f64 {
        1.0_f64 - self.cdf(x)
    }
    /// Inverse CDF: smallest `x` with `cdf(x) ≥ p`.
    ///
    /// # Errors
    /// `ProbabilityOutOfRange(p)` when `p ∉ [0, 1]`.
    fn quantile(&self, p: f64) -> Result<f64, StatError>;
}

/// Discrete CDF + quantile.
///
/// `cdf(k) = P(X ≤ k)`; `quantile(p)` = smallest `k` with `cdf(k) ≥ p`.
pub trait DiscreteCdf: Distribution {
    /// Cumulative probability `P(X ≤ k)`.
    fn cdf(&self, k: i64) -> f64;
    /// Survival function `P(X > k)`.
    fn sf(&self, k: i64) -> f64 {
        1.0_f64 - self.cdf(k)
    }
    /// Inverse CDF: smallest integer `k` with `cdf(k) ≥ p`.
    ///
    /// # Errors
    /// `ProbabilityOutOfRange(p)` when `p ∉ [0, 1]`.
    fn quantile(&self, p: f64) -> Result<i64, StatError>;
}

/// Inverse-CDF sampling.
///
/// `CommonStatsRng` is the crate's only RNG (a concrete struct), so `sample`
/// takes it by `&mut`. The default body is
/// `self.quantile(rng.uniform()).expect(...)` (`uniform() ∈ (0,1)` is always a
/// valid quantile arg); override with closed forms where they exist.
#[cfg(all(feature = "dist", feature = "rng"))]
pub trait Sampler: ContinuousCdf {
    /// Draw one sample via the inverse CDF.
    fn sample(&self, rng: &mut CommonStatsRng) -> f64;
}

/// Standard-normal inverse CDF via the public `erfc_inv`.
///
/// `Φ⁻¹(p) = −√2 · erfc_inv(2p)` (derived from `erfc_inv(p) = −norm_ppf(p/2)/√2`
/// in `special/inverse.rs`). Used instead of the private `special::norm_ppf`
/// for Normal's quantile and as the seed in the ported `t_ppf`/`chi2_ppf`/
/// `f_ppf`.
pub(crate) fn norm_quantile(p: f64) -> f64 {
    -core::f64::consts::SQRT_2 * crate::special::erfc_inv(2.0 * p)
}

/// Log density of `Gamma(shape α, rate β)` at `x`:
/// `α·ln β + (α−1)·ln x − β·x − lnΓ(α)`. `NEG_INFINITY` for `x ≤ 0`.
/// Shared by `Gamma` and `ChiSquared` (`χ²(k) = Gamma(k/2, 1/2)`).
pub(crate) fn gamma_log_density(shape: f64, rate: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    shape * libm::log(rate) + (shape - 1.0) * libm::log(x)
        - rate * x
        - crate::special::lgamma(shape)
}

pub use continuous::Beta;
pub use continuous::Cauchy;
pub use continuous::ChiSquared;
pub use continuous::Exponential;
pub use continuous::FisherF;
pub use continuous::Gamma;
pub use continuous::LogNormal;
pub use continuous::Normal;
pub use continuous::StudentT;
pub use continuous::Uniform;
pub use continuous::Weibull;
pub use discrete::Bernoulli;
pub use discrete::Binomial;
pub use discrete::Geometric;
pub use discrete::Hypergeometric;
pub use discrete::NegBinomial;
pub use discrete::Poisson;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_quantile_matches_known_points() {
        // Φ⁻¹(0.5) = 0, Φ⁻¹(0.975) ≈ 1.959963984540054 (scipy norm.ppf).
        assert!(norm_quantile(0.5).abs() < 1e-12);
        assert!((norm_quantile(0.975) - 1.959_963_984_540_054).abs() < 1e-9);
        assert!((norm_quantile(0.025) + 1.959_963_984_540_054).abs() < 1e-9);
    }

    #[test]
    fn bound_is_copy_and_eq() {
        let b = Bound::Finite(1.0);
        let c = b; // Copy
        assert_eq!(b, c);
        assert_eq!(Bound::NegInfinity, Bound::NegInfinity);
    }

    #[test]
    fn gamma_log_density_normalizes() {
        // Gamma(shape=1, rate=1) is Exponential(1): pdf(x)=e^{-x}; log pdf(2)=-2.
        assert!((gamma_log_density(1.0, 1.0, 2.0) + 2.0).abs() < 1e-12);
    }
}
