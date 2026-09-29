//! Distribution object suite (feature `dist`).
//!
//! A decomposed trait set: [`Distribution`] (support + moments),
//! [`ContinuousDensity`]/[`DiscreteMass`] (pdf/pmf), [`ContinuousCdf`]/
//! [`DiscreteCdf`] (cdf + quantile), and [`Sampler`]/[`DiscreteSampler`]
//! (sampling, by the algorithm each type names on its impl; also gated on
//! `rng`). Every constructor returns `Result<_, StatError>`; quantiles
//! return `Err(ProbabilityOutOfRange)` for `p ∉ [0,1]`.
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
/// correct for all 18 distributions in this suite.
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

/// Continuous CDF + quantile, both tails.
///
/// All four methods are required: each type evaluates its upper tail
/// (`sf`, `isf`) from its own complementary form, never `1 −` a value that is
/// itself near 1, as `1 − cdf` or `quantile(1 − q)` would be. That keeps only
/// the absolute accuracy of the value being subtracted: `1 − cdf` returns 0
/// for any tail below ~1e-16, and `1 − q` in f64 keeps only ~7 digits of a
/// `q = 1e-9` request. Taking `1 −` a value that is already a tail ≤ ½ is
/// fine (Cauchy's `cdf_sf`, FisherF's `ln_sf`, Hypergeometric's `sf` do this)
/// since there is no cancellation. `quantile` and `isf` return
/// `Err(ProbabilityOutOfRange)` for an argument outside `[0, 1]`.
pub trait ContinuousCdf: Distribution {
    /// Cumulative probability `P(X ≤ x)`.
    fn cdf(&self, x: f64) -> f64;
    /// Survival function `P(X > x)`, evaluated directly on the upper tail, so
    /// a small tail keeps its relative precision. Matches
    /// `scipy.stats.<dist>.sf(x)` (`tests/fixtures/dist_*_sf.json`).
    fn sf(&self, x: f64) -> f64;
    /// Inverse CDF: smallest `x` with `cdf(x) ≥ p`.
    ///
    /// # Errors
    /// `ProbabilityOutOfRange(p)` when `p ∉ [0, 1]`.
    fn quantile(&self, p: f64) -> Result<f64, StatError>;
    /// Inverse survival function: smallest `x` with `sf(x) ≤ q`, i.e. the
    /// upper-tail critical value for tail mass `q`, solved on `q` itself
    /// (symmetry, `−ln q`, or a root of `sf`), never as `quantile(1 − q)`.
    /// Matches `scipy.stats.<dist>.isf(q)` (`tests/fixtures/dist_*_isf.json`).
    ///
    /// # Errors
    /// `ProbabilityOutOfRange(q)` when `q ∉ [0, 1]`.
    fn isf(&self, q: f64) -> Result<f64, StatError>;
}

/// Discrete CDF + quantile, both tails.
///
/// All three methods are required: each type evaluates its upper tail `sf`
/// from its own complementary form (the complementary incomplete beta or
/// gamma, a closed form, or a sum from the top of the support), never `1 −`
/// a value that is itself near 1: `1 − cdf` returns 0 for any tail below
/// ~1e-16. Taking `1 −` a tail that is already ≤ ½ is fine (Hypergeometric's
/// `cdf`/`sf` do this) since there is no cancellation.
pub trait DiscreteCdf: Distribution {
    /// Cumulative probability `P(X ≤ k)`.
    fn cdf(&self, k: i64) -> f64;
    /// Survival function `P(X > k)`, evaluated directly on the upper tail, so
    /// a small tail keeps its relative precision. Same convention as
    /// `scipy.stats.<dist>.sf(k)`; validated against mpmath
    /// (`tests/fixtures/accuracy_discrete.json`).
    fn sf(&self, k: i64) -> f64;
    /// Inverse CDF: smallest integer `k` with `cdf(k) ≥ p`. On a bounded
    /// support `quantile(1)` is the largest `k` with positive mass, as in
    /// `scipy.stats.<dist>.ppf(1)`, not the first `k` whose f64 `cdf` rounds
    /// to 1. `Binomial(n, 0)` and `Bernoulli(0)` have their entire mass at
    /// `k = 0`, where `cdf(0) = 1` exactly, so the definition gives
    /// `quantile(1) = 0`; scipy's `ppf(1)` returns `n` and `1` there, the top
    /// of the support.
    ///
    /// # Errors
    /// `ProbabilityOutOfRange(p)` when `p ∉ [0, 1]`; `DomainError` when the
    /// quantile exceeds `i64::MAX`: `p = 1` on an unbounded support
    /// (`Poisson`, `NegBinomial`, `Geometric` with success probability < 1),
    /// where the true quantile is +∞, or `Poisson` with `λ` large enough that
    /// its search bracket cannot reach a finite `i64` quantile.
    fn quantile(&self, p: f64) -> Result<i64, StatError>;
}

/// Sampling for continuous distributions.
///
/// `CommonStatsRng` is the crate's only RNG (a concrete struct), so `sample`
/// takes it by `&mut`. Impls use the inverse CDF,
/// `self.quantile(rng.uniform()).expect(...)` (`uniform() ∈ (0,1)` is always a
/// valid quantile arg), or a closed form of it where one exists; one uniform
/// per draw. `InverseGaussian` instead uses the Michael–Schucany–Haas
/// transform (two uniforms per draw) and `Gamma` and `ChiSquared` (through
/// `Gamma`) the Marsaglia–Tsang rejection sampler (a variable number of
/// uniforms per draw); see their impls.
#[cfg(all(feature = "dist", feature = "rng"))]
pub trait Sampler: ContinuousCdf {
    /// Draw one sample.
    fn sample(&self, rng: &mut CommonStatsRng) -> f64;
}

/// Sampling for integer-valued distributions.
///
/// The discrete counterpart of [`Sampler`], with no default body: plain
/// inversion, `quantile(rng.uniform())`, resolves a probability `P` only to ≈
/// 2⁻³²/`P` relative on the 32-bit uniform grid, so each type names its own
/// algorithm on its impl (`Geometric` inverts with a 52-bit uniform for `p <
/// 10⁻³`; the others use rejection or search methods, the rejection samplers
/// consuming a variable number of uniforms per draw). A draw is a
/// deterministic function of the RNG stream, so the same `(seed, draw_id)`
/// reproduces the same draws.
#[cfg(all(feature = "dist", feature = "rng"))]
pub trait DiscreteSampler: DiscreteCdf {
    /// Draw one sample, by the algorithm named on the type's impl.
    fn sample(&self, rng: &mut CommonStatsRng) -> i64;
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

/// Log density of `Gamma(shape α, rate β)` at `x`, `ln(β^α·x^{α−1}·e^{−βx}/Γ(α))`.
/// At `x = 0`, `+∞` for `α < 1`, `ln β` for `α = 1` and `NEG_INFINITY` for
/// `α > 1`; `NEG_INFINITY` for `x < 0` and `x = +∞`. Shared by `Gamma` and `ChiSquared` (`χ²(k) = Gamma(k/2, 1/2)`).
///
/// For `α ≥ 8`, with `y = β·x`, the saddle-point form `½·ln(α/2π) − δ(α) −
/// α·(e − ln(1 + e)) − ln x`, `1 + e = y/α` (`gamma_saddle_dev`, DiDonato &
/// Morris 1986 `rcomp`): the sum `α·ln β + (α−1)·ln x − y − lnΓ(α)` would
/// keep the rounding of terms of size `α·|ln x|` (`0` for `−346.65` at
/// `χ²(1e300)`, `x = 1e300`). Where `y` leaves the normal range the sum, whose
/// terms no longer cancel. For `α < 8` the sum, `lnΓ(α)` from `lgamma`.
pub(crate) fn gamma_log_density(shape: f64, rate: f64, x: f64) -> f64 {
    gamma_log_density_with(shape, rate, gamma_log_density_consts(shape, rate), x)
}

/// The `x`-free terms of [`gamma_log_density`], for a caller (the Gamma
/// quantile solver) that evaluates it at many `x`: `(½·ln(α/2π) − δ(α), 0)`
/// for `α ≥ 8`, else `(α·ln β, lnΓ(α))`.
pub(crate) fn gamma_log_density_consts(shape: f64, rate: f64) -> (f64, f64) {
    if shape >= 8.0 {
        (
            0.5 * libm::log(shape / (2.0 * core::f64::consts::PI))
                - crate::special::elementary::stirling_del(shape),
            0.0,
        )
    } else {
        (shape * libm::log(rate), crate::special::lgamma(shape))
    }
}

/// [`gamma_log_density`] given `(c, ln_gamma) = gamma_log_density_consts(..)`.
pub(crate) fn gamma_log_density_with(
    shape: f64,
    rate: f64,
    (c, ln_gamma): (f64, f64),
    x: f64,
) -> f64 {
    if x == 0.0 {
        // `(α − 1)·ln x` is 0·(−∞) at α = 1, where the density at 0 is β.
        return if shape < 1.0 {
            f64::INFINITY
        } else if shape == 1.0 {
            libm::log(rate)
        } else {
            f64::NEG_INFINITY
        };
    }
    if x < 0.0 || x == f64::INFINITY {
        return f64::NEG_INFINITY;
    }
    let y = rate * x;
    if shape < 8.0 {
        return c + (shape - 1.0) * libm::log(x) - y - ln_gamma;
    }
    if (f64::MIN_POSITIVE..f64::INFINITY).contains(&y) {
        return c + crate::special::saddle::gamma_saddle_dev(shape, y) - libm::log(x);
    }
    shape * libm::log(rate) + (shape - 1.0) * libm::log(x) - y - crate::special::lgamma(shape)
}

pub use continuous::Beta;
pub use continuous::Cauchy;
pub use continuous::ChiSquared;
pub use continuous::Exponential;
pub use continuous::FisherF;
pub use continuous::Gamma;
pub use continuous::InverseGaussian;
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
