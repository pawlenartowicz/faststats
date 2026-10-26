#![cfg(feature = "dist")]
//! Generic properties, checked by proptest, over every distribution in
//! `dist`, complementing the fixed oracle grids in `dist_oracle.rs`: they
//! hold for any correct implementation of a distribution's trait methods,
//! independent of the specific parameters an oracle fixture happened to
//! pick. Each property runs once per distribution (a distribution whose
//! mechanism fails it stays un-ignored; only the ones reproducing a known
//! open defect are `#[ignore]`d, with the defect named).
//!
//! Determinism: [`config`] pins both the case count and the RNG seed through
//! `ProptestConfig` (not only `PROPTEST_CASES`/`PROPTEST_RNG_SEED`), so
//! `cargo test --all-features` draws the same inputs every run in CI.

use commonstats::dist::continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FisherF, Gamma, InverseGaussian, LogNormal, Normal,
    StudentT, StudentizedRange, Uniform, Weibull,
};
use commonstats::dist::discrete::{
    Bernoulli, Binomial, Geometric, Hypergeometric, NegBinomial, Poisson,
};
use commonstats::dist::{
    Bound, ContinuousCdf, ContinuousDensity, DiscreteCdf, DiscreteMass, Distribution,
};
use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestRunner};

const EPS: f64 = f64::EPSILON;
/// A representative "smallest meaningful" positive `f64`, used to probe
/// whether a `quantile`/`isf` result of exactly `0` is a genuine underflow or
/// a bug (see `backward_error`'s comments). The smallest positive
/// *subnormal* (`f64::from_bits(1)`, ~5e-324) is the wrong choice here: a
/// distribution that scales `x` by a parameter before evaluating (`Gamma`'s
/// `rate * x`) can round that internal product to exactly 0 even when `x`
/// itself is meaningfully nonzero, which would misreport a bug as genuine.
/// `f64::MIN_POSITIVE` (the smallest *normal*, full-mantissa-precision) does
/// not have that failure mode.
const SMALLEST_POSITIVE: f64 = f64::MIN_POSITIVE;

/// Backward-error slack for `quantile`/`isf`: `c·ε·(p + |x|·pdf(x)·cond)`.
/// `p` is the argument's own rounding budget, `|x|·pdf(x)` the sensitivity of
/// `cdf` to its root (an error `dx` in the root shows up in `cdf` scaled by
/// `pdf(x)`, and `x` itself carries a relative, not absolute, rounding
/// budget), and `cond` (computed at each call site) an extra factor for a
/// root built as `exp(large)`, whose relative error is `~ln|x|` times bigger
/// than `|x|·pdf(x)` alone accounts for. `c = 300` is well above the few
/// ulps a correct implementation costs (checked directly: `StudentT` and
/// `ChiSquared` need most of that headroom for their deep-tail Newton
/// solves) and far below the many-orders-of-magnitude miss a real bug
/// produces — e.g. a quantile of `1e-20` returned as `-0.0` (the form
/// `−ln(1 − p)` gives that) is wrong by 100%, not by a `c`-sized factor.
const BACKWARD_SLACK: f64 = 300.0;
/// Same shape of slack for the discrete upper-tail identity.
const UPPER_TAIL_SLACK: f64 = 50.0;
/// Relative slack of the monotone checks: two independently drawn points are
/// almost never within a few ulps of each other, so this only absorbs a
/// correctly-rounded tie. Relative, not absolute: a deep-tail cdf (or sf) is
/// compared against its own magnitude, where an absolute 1e-12 would pass any
/// drop below 1e-12.
const MONOTONE_SLACK: f64 = 1e-12;
/// `log_mass` vs `ln(mass)` slack, scaled by the log's own magnitude.
const LOGMASS_SLACK: f64 = 20.0;
/// `log_mass(k+1) - log_mass(k)` vs a closed-form successive-ratio slack,
/// scaled by the magnitude of the two `log_mass` values compared (each
/// contributes its own rounding budget).
const RATIO_SLACK: f64 = 20.0;
/// Slack for a brute-force pmf sum, scaled by the number of terms added (each
/// contributes at most one rounding).
const MASS_SUM_SLACK: f64 = 20.0;
/// Slack for the Gamma rate-scaling identity.
const SCALING_SLACK: f64 = 100.0;

/// 48 cases per property at a fixed seed: enough for the tail branches of
/// [`tail_prob`] to fire repeatedly. The seed is fixed (not left to
/// `PROPTEST_RNG_SEED`, which defaults to random) so `cargo test
/// --all-features` draws the same inputs on every run, in CI or locally.
/// That also makes a failure persistence file redundant: a failing case is
/// redrawn by every run until the strategy changes, and after a strategy
/// change a persisted seed draws a different input.
fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 48,
        rng_seed: proptest::test_runner::RngSeed::Fixed(0x636F_6D6D_6F6E_7374),
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

// ---- shared strategies --------------------------------------------------

/// Log-uniform magnitude over `[2^lo, 2^hi]`: shape/rate/scale parameters
/// drawn across decades (mirrors the accuracy harness's log-uniform
/// parameter sampling), shrinking toward the exponent nearest 0 — magnitude
/// 1, not 0 or `f64::MAX`.
fn log_uniform(lo: i32, hi: i32) -> impl Strategy<Value = f64> {
    (lo..=hi).prop_flat_map(|e| (1.0f64..2.0).prop_map(move |m| m * 2f64.powi(e)))
}

/// A location parameter: any finite real over a moderate range, shrinking
/// toward 0.
fn loc_param() -> impl Strategy<Value = f64> {
    -1.0e6f64..1.0e6
}

/// `LogNormal`'s `mu`: narrow enough that `mu + sigma * z` (`z` out to
/// [`tail_prob`]'s deepest points, `|z| ~ 40`) rarely exceeds `+-709`
/// (`exp` overflow) — paired with [`lognormal_sigma`], not the wider
/// [`pos_param`]/[`loc_param`], so `quantile`'s true root is representable
/// on the large majority of draws instead of degenerating into "any p maps
/// to 0 or +inf" (where no backward-error check is meaningful at all).
fn lognormal_mu() -> impl Strategy<Value = f64> {
    -300.0f64..300.0
}

/// `LogNormal`'s `sigma`, paired with [`lognormal_mu`]: see its doc comment.
fn lognormal_sigma() -> impl Strategy<Value = f64> {
    log_uniform(-6, 2)
}

/// A positive shape/rate/scale parameter across 60 decades (`2^-30..2^30`),
/// reaching into the huge/tiny-parameter regime the known tail defects were
/// found at without every draw landing there.
fn pos_param() -> impl Strategy<Value = f64> {
    log_uniform(-30, 30)
}

/// `StudentizedRange`'s `k`: `1 + m`, `m` log-uniform over `[2^−10, 2^8)`, from
/// just above 1 (where `F_R(w) ~ w^{k−1}` spreads the mass over many decades)
/// to ~257.
fn sr_k() -> impl Strategy<Value = f64> {
    log_uniform(-10, 7).prop_map(|m| 1.0 + m)
}

/// `StudentizedRange`'s `df`: log-uniform over `[2^−6, 2^30]`, or ∞ one draw
/// in ten. The floor sits above ν ≈ 0.01, where one quantile costs 1–2 s
/// (the type's Cost note); tiny ν is checked at fixed points (k = 2) in
/// `dist_oracle.rs`.
fn sr_df() -> impl Strategy<Value = f64> {
    prop_oneof![9 => log_uniform(-6, 30), 1 => Just(f64::INFINITY)]
}

/// Probability drawn across all magnitudes: bulk `(0, 1)` (weight 3), a
/// lower-tail value down to `1e-300` (weight 2), and an upper-tail value up
/// to `1 − 1e-15` (weight 2) — deep tails matter here because that is where
/// cancellation bugs (`1 − p`, `1 − cdf`) live and a bulk-only draw would
/// never reach them. The upper branch stops at `e = 15`, not 300: `1 − x` for
/// `x < 2^-53 ≈ 1.1e-16` rounds to exactly `1.0` in f64 (no double is closer
/// to 1 than that), so a `p` meant to sit just below 1 would silently become
/// `1.0` and violate a strict `p < 1` domain (`NegBinomial`, `Geometric`).
/// The genuinely deep upper tail is exercised instead through `isf`/`sf`,
/// whose own argument is a small `q`, never `1 − q`.
fn tail_prob() -> BoxedStrategy<f64> {
    prop_oneof![
        3 => (1u32..999).prop_map(|n| f64::from(n) / 1000.0),
        2 => (1i32..=300).prop_flat_map(|e| (1.0f64..10.0).prop_map(move |m| m * 10f64.powi(-e))),
        2 => (1i32..=15)
            .prop_flat_map(|e| (1.0f64..10.0).prop_map(move |m| 1.0 - m * 10f64.powi(-e))),
    ]
    .boxed()
}

/// Bulk-only probability (no tail branch): keeps a distribution's mean/size
/// bounded for tests that brute-force-sum its pmf.
fn bulk_prob() -> impl Strategy<Value = f64> {
    (1u32..999).prop_map(|n| f64::from(n) / 1000.0)
}

/// Any real `x`, in or out of a distribution's support: zero, or a signed
/// log-uniform magnitude. Generic across distributions since `cdf`/`density`
/// must be defined (0/1 at the edges) for every real argument.
fn any_x() -> BoxedStrategy<f64> {
    prop_oneof![
        1 => Just(0.0),
        4 => log_uniform(-20, 20),
        4 => log_uniform(-20, 20).prop_map(|v| -v),
    ]
    .boxed()
}

/// `Binomial`'s `n`: mostly modest, occasionally a huge count (1e6, 1e12) —
/// the magnitudes the known `Binomial`/`NegBinomial` cdf defects were
/// found at. Safe for the general properties because `Binomial::cdf`/`sf`
/// (`betai`) are O(1) regardless of `n`.
fn count_param() -> BoxedStrategy<i64> {
    prop_oneof![
        5 => 1i64..=10_000,
        2 => Just(1_000_000i64),
        1 => Just(1_000_000_000_000i64),
    ]
    .boxed()
}

/// `Hypergeometric`'s `(N, K, n)`: `N` capped at 500. Its `cdf` is a direct
/// sum over `k_lo..=k` (no closed form), so this also bounds every property
/// test's cost, not only the pmf-sum one.
fn hypergeometric_params() -> impl Strategy<Value = (i64, i64, i64)> {
    (1i64..=500).prop_flat_map(|big_n| (0..=big_n, 0..=big_n).prop_map(move |(k, n)| (big_n, k, n)))
}

/// `Uniform`'s `(a, b)`: two distinct location draws, sorted. `prop_filter`
/// (rather than computing `b = a + width`) sidesteps the rare case where a
/// tiny width rounds away against a large `a`.
fn uniform_ab() -> impl Strategy<Value = (f64, f64)> {
    (loc_param(), loc_param())
        .prop_filter("distinct bounds", |(a, b)| a != b)
        .prop_map(|(a, b)| if a < b { (a, b) } else { (b, a) })
}

// ---- shared assertions ----------------------------------------------------

/// `quantile`/`isf` at 0 and 1 vs a support bound: exactly, as
/// `scipy.stats.<dist>.ppf`/`isf` return it. A finite slack would pass a
/// result just outside the support (`a + 1·(b − a)` rounding past `b`).
fn assert_bound(got: f64, want: f64, tag: &str) -> Result<(), TestCaseError> {
    prop_assert!(got == want, "{tag}: got {got:e}, want {want:e}");
    Ok(())
}

/// `|x|·pdf(x)`, the sensitivity of the cdf to a relative change of its
/// root, formed in logs: at a deep tail the pdf alone can underflow while
/// the product is representable (`LogNormal(0, 3.59)` at its
/// `isf(3.18e-272)` = 8.4e54, pdf 3.7e-326).
fn x_pdf(d: &impl ContinuousDensity, x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        libm::exp(libm::log(libm::fabs(x)) + d.log_density(x))
    }
}

/// Whether the increasing `f` crosses `target` within one float of `x`
/// (`f(x⁻) ≤ target ≤ f(x⁺)`): then `x` is as close to the root as f64
/// allows, whatever `f` jumps by between neighbours, as it does next to a
/// support end (`Beta(1, 2.44e-4).quantile(0.155)` = 1 − 1.1e-16, cdf 0.0089
/// there and 1 at 1) or at a subnormal root.
fn straddles(f: impl Fn(f64) -> f64, x: f64, target: f64) -> bool {
    f(libm::nextafter(x, f64::NEG_INFINITY)) <= target
        && target <= f(libm::nextafter(x, f64::INFINITY))
}

/// `lo ≤ hi` ⇒ `cdf(lo) ≤ cdf(hi)` and `sf(lo) ≥ sf(hi)`, all four in
/// `[0, 1]` and not NaN. The sf half covers the upper tail, where the cdf
/// saturates at 1.
fn assert_monotone(
    (clo, chi): (f64, f64),
    (slo, shi): (f64, f64),
    at: &str,
) -> Result<(), TestCaseError> {
    for v in [clo, chi, slo, shi] {
        prop_assert!(
            (0.0..=1.0).contains(&v),
            "{at}: cdf/sf {v:e} outside [0, 1] or NaN"
        );
    }
    prop_assert!(
        clo <= chi * (1.0 + MONOTONE_SLACK),
        "{at}: cdf not monotone: {clo:e} > {chi:e}"
    );
    prop_assert!(
        shi <= slo * (1.0 + MONOTONE_SLACK),
        "{at}: sf not monotone: {shi:e} > {slo:e}"
    );
    Ok(())
}

/// One half of `backward_error`: `x`, the root returned for tail mass `prob`
/// by `quantile` (`upper = false`, residual on `cdf`) or `isf` (`upper =
/// true`, on `sf`). A NaN root fails. A ±∞ root passes only when the true root
/// lies past `±f64::MAX` (the mass at `±f64::MAX` has not yet reached `prob`).
/// A zero root passes when the true root lies below `SMALLEST_POSITIVE`.
/// Any other root must meet the backward-error bound or straddle the root.
fn check_root(
    d: &(impl ContinuousCdf + ContinuousDensity),
    prob: f64,
    x: f64,
    upper: bool,
) -> Result<(), TestCaseError> {
    let tag = if upper { "isf" } else { "quantile" };
    let mass = |t: f64| if upper { d.sf(t) } else { d.cdf(t) };
    prop_assert!(!x.is_nan(), "{tag}({prob:e}) = NaN");
    if x.is_infinite() {
        let edge = mass(x.signum() * f64::MAX);
        // The mass at the edge is still short of `prob` on the side the
        // root lies: above `prob` for an increasing cdf at −∞ or a
        // decreasing sf at +∞, below it otherwise.
        let beyond = if upper == (x > 0.0) {
            edge >= prob
        } else {
            edge <= prob
        };
        prop_assert!(
            beyond,
            "{tag}({prob:e}) = {x:e}, but the mass at ±f64::MAX is {edge:e}"
        );
        return Ok(());
    }
    // `x == 0` is ambiguous: genuine (the true root is below the smallest
    // representable positive float) or an underflow bug. `cdf` increases and
    // `sf` decreases, so which one shows at `SMALLEST_POSITIVE`.
    if x == 0.0 {
        let m = mass(SMALLEST_POSITIVE);
        if (!upper && m >= prob) || (upper && m < prob) {
            return Ok(());
        }
    }
    let resid = (mass(x) - prob).abs();
    // A quantile built as `exp(location + scale * z)` carries the seed's
    // rounding scaled by `|ln x|`, not by `x` itself; `cond` extends the
    // `|x|*pdf(x)` sensitivity by that factor (1 for any other root).
    let cond = libm::fabs(x).max(1.0).ln().max(1.0);
    let slack = BACKWARD_SLACK * EPS * (prob + x_pdf(d, x) * cond);
    let at_root = if upper {
        straddles(|t| -d.sf(t), x, -prob)
    } else {
        straddles(|t| d.cdf(t), x, prob)
    };
    prop_assert!(
        resid <= slack || at_root,
        "{tag}: p={prob:e} x={x:e} mass={:e} resid={resid:e} slack={slack:e}",
        mass(x)
    );
    Ok(())
}

/// `log_mass(k+1) - log_mass(k)` against `ln` of a closed-form successive
/// mass ratio, derived directly from a pmf's own recurrence (e.g. Binomial's
/// `mass(k+1)/mass(k) = (n-k)/(k+1) · p/q`), never by exponentiating and
/// dividing `mass` values. `log_mass_matches_ln_mass` compares `log_mass`
/// against `mass`, two mirrored code paths, so it misses a drift both share;
/// this closed form is independent of both.
fn assert_log_mass_ratio(
    lm_k: f64,
    lm_k1: f64,
    ln_ratio: f64,
    tag: &str,
) -> Result<(), TestCaseError> {
    let diff = (lm_k1 - lm_k - ln_ratio).abs();
    let slack = RATIO_SLACK * EPS * lm_k.abs().max(lm_k1.abs()).max(1.0);
    prop_assert!(
        diff <= slack,
        "{tag}: log_mass(k+1)-log_mass(k)={:e} vs ln(ratio)={ln_ratio:e}, diff={diff:e}, slack={slack:e}",
        lm_k1 - lm_k
    );
    Ok(())
}

// ---- generic continuous properties ---------------------------------------

/// The three generic continuous properties (backward error of
/// `quantile`/`isf`, `quantile(0)`/`quantile(1)` at the support bounds,
/// range + monotone `cdf` + no NaN) for one distribution. `$p` is the
/// parameter pattern bound from `$params`; `$ctor` builds the distribution
/// from it. Each property's ignore reason is a `meta` fragment: `cfg(all())`
/// (always compiled — not ignored) or `ignore = "<defect>"`.
macro_rules! continuous_dist_suite {
    (
        $mod_name:ident, $ty:ty, $params:expr, $p:pat, $ctor:expr,
        backward: $backward_attr:meta,
        bounds: $bounds_attr:meta,
        monotone: $monotone_attr:meta $(,)?
    ) => {
        mod $mod_name {
            use super::*;

            #[test]
            #[$backward_attr]
            fn backward_error() {
                let mut runner = TestRunner::new(config());
                runner
                    .run(&($params, tail_prob()), |($p, prob)| {
                        let d: $ty = $ctor;
                        check_root(&d, prob, d.quantile(prob).unwrap(), false)?;
                        check_root(&d, prob, d.isf(prob).unwrap(), true)
                    })
                    .unwrap();
            }

            #[test]
            #[$bounds_attr]
            fn quantile_extremes_match_support() {
                let mut runner = TestRunner::new(config());
                runner
                    .run(&$params, |$p| {
                        let d: $ty = $ctor;
                        assert_bound(
                            d.quantile(0.0).unwrap(),
                            d.support_min().as_f64(),
                            "quantile(0)",
                        )?;
                        assert_bound(
                            d.quantile(1.0).unwrap(),
                            d.support_max().as_f64(),
                            "quantile(1)",
                        )?;
                        assert_bound(d.isf(0.0).unwrap(), d.support_max().as_f64(), "isf(0)")?;
                        assert_bound(d.isf(1.0).unwrap(), d.support_min().as_f64(), "isf(1)")?;
                        Ok(())
                    })
                    .unwrap();
            }

            #[test]
            #[$monotone_attr]
            fn range_monotone_no_nan() {
                let mut runner = TestRunner::new(config());
                runner
                    .run(
                        &($params, any_x(), any_x(), tail_prob()),
                        |($p, x1, x2, p1)| {
                            let d: $ty = $ctor;
                            // Any real pair; and each of `x1` and a tail point
                            // (`quantile` of a `tail_prob` draw) against its
                            // neighbour 2^-10 above, where a tail or seam drop
                            // shows: two independent draws are decades apart,
                            // and the true increase between them hides any
                            // local drop.
                            let q = d.quantile(p1).unwrap();
                            let up = |t: f64| {
                                if t.is_finite() {
                                    t + libm::fabs(t) * (1.0 / 1024.0)
                                } else {
                                    t
                                }
                            };
                            for (a, b) in [(x1, x2), (x1, up(x1)), (q, up(q))] {
                                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                                assert_monotone(
                                    (d.cdf(lo), d.cdf(hi)),
                                    (d.sf(lo), d.sf(hi)),
                                    &format!("x = {lo:e}, {hi:e}"),
                                )?;
                            }
                            let lo = x1.min(x2);
                            let dens = d.density(lo);
                            prop_assert!(
                                !dens.is_nan() && dens >= 0.0,
                                "bad density at {lo:e}: {dens:e}"
                            );
                            Ok(())
                        },
                    )
                    .unwrap();
            }
        }
    };
}

continuous_dist_suite!(
    normal, Normal, (loc_param(), pos_param()), (mean, sd), Normal::new(mean, sd).unwrap(),
    backward: cfg(all()), bounds: cfg(all()), monotone: cfg(all()),
);
continuous_dist_suite!(
    studentt, StudentT, pos_param(), df, StudentT::new(df).unwrap(),
    backward: cfg(all()), bounds: cfg(all()), monotone: cfg(all()),
);
continuous_dist_suite!(
    chisquared, ChiSquared, pos_param(), k, ChiSquared::new(k).unwrap(),
    backward: cfg(all()), bounds: cfg(all()), monotone: cfg(all()),
);
continuous_dist_suite!(
    fisherf, FisherF, (pos_param(), pos_param()), (dfn, dfd), FisherF::new(dfn, dfd).unwrap(),
    backward: cfg(all()),
    bounds: cfg(all()), monotone: cfg(all()),
);
continuous_dist_suite!(
    uniform, Uniform, uniform_ab(), (a, b), Uniform::new(a, b).unwrap(),
    backward: cfg(all()), bounds: cfg(all()), monotone: cfg(all()),
);
continuous_dist_suite!(
    exponential, Exponential, pos_param(), rate, Exponential::new(rate).unwrap(),
    backward: cfg(all()),
    bounds: cfg(all()), monotone: cfg(all()),
);
continuous_dist_suite!(
    cauchy, Cauchy, (loc_param(), pos_param()), (loc, scale), Cauchy::new(loc, scale).unwrap(),
    backward: cfg(all()), bounds: cfg(all()), monotone: cfg(all()),
);
continuous_dist_suite!(
    weibull, Weibull, (log_uniform(-10, 20), pos_param()), (shape, scale), Weibull::new(shape, scale).unwrap(),
    backward: cfg(all()),
    bounds: cfg(all()), monotone: cfg(all()),
);
continuous_dist_suite!(
    lognormal, LogNormal, (lognormal_mu(), lognormal_sigma()), (mu, sigma),
    LogNormal::new(mu, sigma).unwrap(),
    backward: cfg(all()),
    bounds: cfg(all()), monotone: cfg(all()),
);
continuous_dist_suite!(
    gamma, Gamma, (pos_param(), pos_param()), (shape, rate), Gamma::new(shape, rate).unwrap(),
    backward: cfg(all()), bounds: cfg(all()), monotone: cfg(all()),
);
continuous_dist_suite!(
    beta, Beta, (pos_param(), pos_param()), (alpha, beta), Beta::new(alpha, beta).unwrap(),
    backward: cfg(all()),
    bounds: cfg(all()), monotone: cfg(all()),
);
continuous_dist_suite!(
    inversegaussian, InverseGaussian, (pos_param(), pos_param()), (mean, shape),
    InverseGaussian::new(mean, shape).unwrap(),
    backward: cfg(all()), bounds: cfg(all()), monotone: cfg(all()),
);
continuous_dist_suite!(
    studentizedrange, StudentizedRange, (sr_k(), sr_df()), (k, df),
    StudentizedRange::new(k, df).unwrap(),
    backward: cfg(all()), bounds: cfg(all()), monotone: cfg(all()),
);

// ---- generic discrete properties ------------------------------------------

/// `d.quantile(prob)`, or a rejected case where it is `Err` because the
/// quantile is past `i64::MAX` (the `DiscreteCdf` contract), once checked that
/// it truly is: `cdf(i64::MAX) < prob`. Any other `Err` fails the case.
fn discrete_quantile<D: DiscreteCdf>(d: &D, prob: f64) -> Result<i64, TestCaseError> {
    d.quantile(prob).map_err(|e| {
        let c = d.cdf(i64::MAX);
        if c < prob {
            TestCaseError::reject("quantile past i64::MAX")
        } else {
            TestCaseError::fail(format!(
                "quantile({prob:e}): {e}, but cdf(i64::MAX) = {c:e}"
            ))
        }
    })
}

/// The six generic discrete properties for one distribution: backward error
/// of `quantile` (p lands in `(cdf(k-1), cdf(k)]`), the upper-tail identity
/// `sf(k) - sf(k+1) = mass(k+1)`, `quantile(0)`/`quantile(1)` at the support
/// bounds, range + monotone `cdf` + no NaN, `log_mass == ln(mass)`, and the
/// pmf summing to 1. `$sum_params` is a separate, size-capped strategy for
/// the last one, since a distribution's own general params may pick a huge
/// mean/size unsuited to brute-force summation.
macro_rules! discrete_dist_suite {
    (
        $mod_name:ident, $ty:ty, $params:expr, $p:pat, $ctor:expr,
        backward: $backward_attr:meta,
        upper_tail: $ut_attr:meta,
        bounds: $bounds_attr:meta,
        monotone: $monotone_attr:meta,
        log_mass: $lm_attr:meta,
        mass_sum: $sum_params:expr, $ms_attr:meta $(,)?
    ) => {
        mod $mod_name {
            use super::*;

            #[test]
            #[$backward_attr]
            fn backward_error() {
                let mut runner = TestRunner::new(config());
                runner
                    .run(&($params, tail_prob()), |($p, prob)| {
                        let d: $ty = $ctor;
                        let k = discrete_quantile(&d, prob)?;
                        let mk = d.mass(k);
                        let slack = BACKWARD_SLACK * EPS * (prob + mk);
                        let cdf_k = d.cdf(k);
                        prop_assert!(
                            cdf_k + slack >= prob,
                            "cdf(k) < p: k={k} cdf={cdf_k:e} p={prob:e} slack={slack:e}"
                        );
                        let cdf_km1 = d.cdf(k - 1);
                        prop_assert!(
                            cdf_km1 <= prob + slack,
                            "cdf(k-1) >= p: k={k} cdf(k-1)={cdf_km1:e} p={prob:e} slack={slack:e}"
                        );
                        Ok(())
                    })
                    .unwrap();
            }

            #[test]
            #[$ut_attr]
            fn upper_tail_identity() {
                let mut runner = TestRunner::new(config());
                runner
                    .run(&($params, tail_prob()), |($p, prob)| {
                        let d: $ty = $ctor;
                        let k = discrete_quantile(&d, prob)?;
                        let (sfk, sfk1, mk1) = (d.sf(k), d.sf(k + 1), d.mass(k + 1));
                        let resid = (sfk - sfk1 - mk1).abs();
                        // The three values carry their conditioning in the
                        // real parameters, not a flat few ε: each is the
                        // exponential of a log-scale value (a kernel's
                        // prefactor, `exp(log_mass)`), ≈ |ln sf|·ε relative
                        // (`Poisson(3.98)` at `k = 25`: `sf` accurate to 34ε,
                        // `|ln sf| = 29`), and in the bulk a relative change
                        // of the location moves `sf(k)` by ≈ `(k+1)·mass(k+1)`
                        // (the discrete `x·pdf(x)` of `backward_error`,
                        // `NegBinomial(12531, 0.44)` at `k = 9812`: 52ε·sf).
                        let ln_cond = if sfk > 0.0 { libm::log(sfk).abs().max(1.0) } else { 1.0 };
                        let loc = (k as f64 + 1.0).abs() * mk1;
                        let slack = UPPER_TAIL_SLACK * EPS * (sfk * ln_cond + loc);
                        prop_assert!(
                            resid <= slack,
                            "sf(k)-sf(k+1)-mass(k+1): k={k} sf(k)={sfk:e} resid={resid:e} slack={slack:e}"
                        );
                        Ok(())
                    })
                    .unwrap();
            }

            #[test]
            #[$bounds_attr]
            fn quantile_extremes_match_support() {
                let mut runner = TestRunner::new(config());
                runner
                    .run(&$params, |$p| {
                        let d: $ty = $ctor;
                        let lo = d.support_min().as_f64();
                        let got_lo = d.quantile(0.0).unwrap() as f64;
                        prop_assert_eq!(got_lo, lo, "quantile(0)");
                        // support_max is PosInfinity for an unbounded discrete
                        // type; quantile(1.0) is then a DomainError (+∞ does
                        // not fit an i64), so the upper check only applies
                        // where the bound is finite.
                        if let Bound::Finite(hi) = d.support_max() {
                            let got_hi = d.quantile(1.0).unwrap() as f64;
                            prop_assert_eq!(got_hi, hi, "quantile(1)");
                        }
                        Ok(())
                    })
                    .unwrap();
            }

            #[test]
            #[$monotone_attr]
            fn range_monotone_no_nan() {
                let mut runner = TestRunner::new(config());
                runner
                    .run(&($params, tail_prob(), tail_prob()), |($p, p1, p2)| {
                        let d: $ty = $ctor;
                        let (k1, k2) = (discrete_quantile(&d, p1)?, discrete_quantile(&d, p2)?);
                        // The drawn pair, and `k1` against its neighbour
                        // (see the continuous suite's neighbour pairs).
                        for (a, b) in [(k1, k2), (k1, k1.saturating_add(1))] {
                            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                            assert_monotone(
                                (d.cdf(lo), d.cdf(hi)),
                                (d.sf(lo), d.sf(hi)),
                                &format!("k = {lo}, {hi}"),
                            )?;
                        }
                        let lo = k1.min(k2);
                        let m = d.mass(lo);
                        prop_assert!(!m.is_nan() && m >= 0.0, "bad mass at {lo}: {m:e}");
                        Ok(())
                    })
                    .unwrap();
            }

            #[test]
            #[$lm_attr]
            fn log_mass_matches_ln_mass() {
                let mut runner = TestRunner::new(config());
                runner
                    .run(&($params, tail_prob()), |($p, prob)| {
                        let d: $ty = $ctor;
                        let k = discrete_quantile(&d, prob)?;
                        let m = d.mass(k);
                        // Skipped where `mass` itself underflows to 0: `ln(0)`
                        // is `-inf`, not a finite value `log_mass` can be
                        // compared against.
                        if m > 0.0 && m.is_finite() {
                            let want = libm::log(m);
                            let lm = d.log_mass(k);
                            let diff = (lm - want).abs();
                            let slack = LOGMASS_SLACK * EPS * want.abs().max(1.0);
                            prop_assert!(
                                diff <= slack,
                                "log_mass != ln(mass): k={k} log_mass={lm:e} ln(mass)={want:e} slack={slack:e}"
                            );
                        }
                        Ok(())
                    })
                    .unwrap();
            }

            #[test]
            #[$ms_attr]
            fn mass_sums_to_one() {
                let mut runner = TestRunner::new(config());
                runner
                    .run(&$sum_params, |$p| {
                        let d: $ty = $ctor;
                        let lo = match d.support_min() {
                            Bound::Finite(v) => v as i64,
                            _ => unreachable!("discrete lower support bound is always finite"),
                        };
                        let hi = match d.support_max() {
                            Bound::Finite(v) => v as i64,
                            // Unbounded above: sum out to where the remaining
                            // tail is below f64 resolution, capped as a safety
                            // valve against a pathological parameter draw.
                            _ => d.quantile(1.0 - 1e-15).unwrap_or(lo),
                        };
                        let hi = hi.min(lo + 200_000);
                        let mut acc = 0.0;
                        for k in lo..=hi {
                            acc += d.mass(k);
                        }
                        let slack = MASS_SUM_SLACK * EPS * f64::from((hi - lo + 1) as u32);
                        prop_assert!(
                            (acc - 1.0).abs() <= slack,
                            "sum mass = {acc:e}, expected 1 (lo={lo}, hi={hi}, slack={slack:e})"
                        );
                        Ok(())
                    })
                    .unwrap();
            }
        }
    };
}

discrete_dist_suite!(
    bernoulli, Bernoulli, tail_prob(), pp, Bernoulli::new(pp).unwrap(),
    backward: cfg(all()),
    upper_tail: cfg(all()),
    bounds: cfg(all()),
    monotone: cfg(all()),
    log_mass: cfg(all()), mass_sum: tail_prob(), cfg(all()),
);
discrete_dist_suite!(
    binomial, Binomial, (count_param(), tail_prob()), (n, pp), Binomial::new(n, pp).unwrap(),
    backward: cfg(all()),
    upper_tail: cfg(all()),
    bounds: cfg(all()),
    monotone: cfg(all()),
    log_mass: cfg(all()),
    mass_sum: (1i64..=300, bulk_prob()), cfg(all()),
);
discrete_dist_suite!(
    poisson, Poisson, pos_param(), lambda, Poisson::new(lambda).unwrap(),
    backward: cfg(all()),
    upper_tail: cfg(all()),
    bounds: cfg(all()), monotone: cfg(all()), log_mass: cfg(all()),
    mass_sum: log_uniform(-6, 10), cfg(all()),
);
discrete_dist_suite!(
    geometric, Geometric, tail_prob(), pp, Geometric::new(pp).unwrap(),
    backward: cfg(all()),
    upper_tail: cfg(all()),
    bounds: cfg(all()), monotone: cfg(all()), log_mass: cfg(all()),
    mass_sum: bulk_prob(), cfg(all()),
);
discrete_dist_suite!(
    negbinomial, NegBinomial, (pos_param(), tail_prob()), (r, pp), NegBinomial::new(r, pp).unwrap(),
    backward: cfg(all()),
    upper_tail: cfg(all()),
    bounds: cfg(all()),
    monotone: cfg(all()),
    log_mass: cfg(all()),
    mass_sum: (log_uniform(-6, 8), bulk_prob()), cfg(all()),
);
discrete_dist_suite!(
    hypergeometric, Hypergeometric, hypergeometric_params(), (big_n, k, n),
    Hypergeometric::new(big_n, k, n).unwrap(),
    backward: cfg(all()),
    upper_tail: cfg(all()),
    bounds: cfg(all()),
    monotone: cfg(all()),
    log_mass: cfg(all()),
    mass_sum: hypergeometric_params(), cfg(all()),
);

// ---- log_mass successive-ratio recurrence ---------------------------------
//
// Five of the six discrete types (all but Bernoulli, whose two-point support
// has only one ratio to check and is already fully covered by
// `log_mass_matches_ln_mass`): `log_mass(k+1) - log_mass(k)` against a closed
// form derived from each pmf's own definition, verified independently
// (against `mass`, not `log_mass`) before wiring it in here.

#[test]
fn binomial_log_mass_ratio() {
    let mut runner = TestRunner::new(config());
    runner
        .run(
            &(count_param(), tail_prob(), tail_prob()),
            |(n, pp, prob)| {
                let b = Binomial::new(n, pp).unwrap();
                let k = b.quantile(prob).unwrap();
                prop_assume!(k < n);
                let (lm_k, lm_k1) = (b.log_mass(k), b.log_mass(k + 1));
                prop_assume!(lm_k.is_finite() && lm_k1.is_finite());
                let kf = k as f64;
                // mass(k+1)/mass(k) = (n-k)/(k+1) * p/q.
                let ln_ratio =
                    libm::log((n as f64 - kf) / (kf + 1.0)) + libm::log(pp) - libm::log(1.0 - pp);
                assert_log_mass_ratio(lm_k, lm_k1, ln_ratio, "Binomial")
            },
        )
        .unwrap();
}

#[test]
fn poisson_log_mass_ratio() {
    let mut runner = TestRunner::new(config());
    runner
        .run(&(pos_param(), tail_prob()), |(lambda, prob)| {
            let p = Poisson::new(lambda).unwrap();
            let k = p.quantile(prob).unwrap();
            let (lm_k, lm_k1) = (p.log_mass(k), p.log_mass(k + 1));
            prop_assume!(lm_k.is_finite() && lm_k1.is_finite());
            // mass(k+1)/mass(k) = lambda/(k+1).
            let ln_ratio = libm::log(lambda / (k as f64 + 1.0));
            assert_log_mass_ratio(lm_k, lm_k1, ln_ratio, "Poisson")
        })
        .unwrap();
}

#[test]
fn geometric_log_mass_ratio() {
    let mut runner = TestRunner::new(config());
    runner
        .run(&(tail_prob(), tail_prob()), |(pp, prob)| {
            let g = Geometric::new(pp).unwrap();
            let k = discrete_quantile(&g, prob)?;
            let (lm_k, lm_k1) = (g.log_mass(k), g.log_mass(k + 1));
            prop_assume!(lm_k.is_finite() && lm_k1.is_finite());
            // mass(k+1)/mass(k) = 1 - p (1-indexed support).
            let ln_ratio = libm::log1p(-pp);
            assert_log_mass_ratio(lm_k, lm_k1, ln_ratio, "Geometric")
        })
        .unwrap();
}

#[test]
fn negbinomial_log_mass_ratio() {
    let mut runner = TestRunner::new(config());
    runner
        .run(&(pos_param(), tail_prob(), tail_prob()), |(r, pp, prob)| {
            let nb = NegBinomial::new(r, pp).unwrap();
            let k = discrete_quantile(&nb, prob)?;
            // `k + 1` below must not overflow.
            prop_assume!(k < i64::MAX);
            let (lm_k, lm_k1) = (nb.log_mass(k), nb.log_mass(k + 1));
            prop_assume!(lm_k.is_finite() && lm_k1.is_finite());
            let kf = k as f64;
            // Convention here: `k` = successes before the `r`-th failure,
            // `p` = success prob, `mass(k) = C(k+r-1,k) (1-p)^r p^k`, so
            // mass(k+1)/mass(k) = (k+r)/(k+1) * p.
            let ln_ratio = libm::log((kf + r) / (kf + 1.0)) + libm::log(pp);
            assert_log_mass_ratio(lm_k, lm_k1, ln_ratio, "NegBinomial")
        })
        .unwrap();
}

#[test]
fn hypergeometric_log_mass_ratio() {
    let mut runner = TestRunner::new(config());
    runner
        .run(
            &(hypergeometric_params(), tail_prob()),
            |((big_n, big_k, n), prob)| {
                let h = Hypergeometric::new(big_n, big_k, n).unwrap();
                let k = h.quantile(prob).unwrap();
                // Support max (mirrors the crate's own `k_hi`): k+1 must stay
                // representable in the population/draw counts.
                prop_assume!(k < n.min(big_k));
                let (lm_k, lm_k1) = (h.log_mass(k), h.log_mass(k + 1));
                prop_assume!(lm_k.is_finite() && lm_k1.is_finite());
                let (bn, bk, nn, kf) = (big_n as f64, big_k as f64, n as f64, k as f64);
                // mass(k+1)/mass(k) = (K-k)(n-k) / ((k+1)(N-K-n+k+1)).
                let ln_ratio =
                    libm::log(((bk - kf) * (nn - kf)) / ((kf + 1.0) * (bn - bk - nn + kf + 1.0)));
                assert_log_mass_ratio(lm_k, lm_k1, ln_ratio, "Hypergeometric")
            },
        )
        .unwrap();
}

// ---- scaling identity ------------------------------------------------------

/// `Gamma(shape, rate)` scaled by `k`: `X/k ~ Gamma(shape, k·rate)`, so
/// `quantile_{k·rate}(p) = quantile_{rate}(p)/k` — checked on the quantiles
/// directly (not by transporting `x` through `cdf`) so the identity does not
/// itself depend on `cdf`'s accuracy.
#[test]
fn gamma_rate_scaling() {
    let mut runner = TestRunner::new(config());
    runner
        .run(
            &(pos_param(), pos_param(), pos_param(), tail_prob()),
            |(shape, rate, k, p)| {
                let g1 = Gamma::new(shape, rate).unwrap();
                let x1 = g1.quantile(p).unwrap();
                let g2 = Gamma::new(shape, rate * k).unwrap();
                let x2 = g2.quantile(p).unwrap();
                prop_assume!(x1.is_finite() && x2.is_finite());
                let diff = (x1 - k * x2).abs();
                // A tiny shape concentrates the whole distribution near 0
                // (density ~ x^{shape-1}), so `d(ln x)/d(ln p) ~ 1/shape`:
                // the quantile itself is that many times more sensitive to
                // rounding than a well-conditioned (shape ~ 1) case.
                let cond = (1.0 / shape).max(1.0);
                let slack = SCALING_SLACK * EPS * cond * x1.abs().max(libm::fabs(k * x2));
                prop_assert!(
                    diff <= slack,
                    "Gamma rate scaling: quantile(rate)={x1:e} vs k*quantile(k*rate)={:e}, diff={diff:e}",
                    k * x2
                );
                Ok(())
            },
        )
        .unwrap();
}
