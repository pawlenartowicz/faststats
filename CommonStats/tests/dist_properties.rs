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

mod common;

use commonstats::dist::continuous::{
    Beta, Cauchy, ChiSquared, Exponential, FisherF, Gamma, InverseGaussian, LogNormal, Normal,
    StudentT, Uniform, Weibull,
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
/// Slack for `quantile(0)`/`quantile(1)` vs the finite support bound, scaled
/// by the bound's own magnitude. An infinite bound instead requires exact
/// equality (see the bounds tests): any finite slack there would pass
/// trivially against `f64::INFINITY`, hiding a finite wrong answer.
const BOUNDS_SLACK: f64 = 8.0;
/// Two independently drawn points are almost never within a few ulps of each
/// other, so a fixed absolute slack (not relative) is simplest for the
/// monotone-cdf check; it only needs to absorb a correctly-rounded tie.
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
/// Slack for the `sf(t) ≈ cdf(-t)` symmetric-family identity.
const SYMMETRIC_SLACK: f64 = 50.0;
/// Slack for the cross-distribution scaling identities.
const SCALING_SLACK: f64 = 100.0;

/// 48 cases per property at a fixed seed: enough for the tail branches of
/// [`tail_prob`] to fire repeatedly while keeping the ~80 generated tests
/// well under a minute in debug. The seed is fixed (not left to
/// `PROPTEST_RNG_SEED`, which defaults to random) so `cargo test
/// --all-features` draws the same inputs on every run, in CI or locally.
fn config() -> ProptestConfig {
    ProptestConfig {
        cases: 48,
        rng_seed: proptest::test_runner::RngSeed::Fixed(0x636F_6D6D_6F6E_7374),
        // `SourceParallel` (the default) only ever finds a crate's `lib.rs`
        // when the failing test itself lives under `src/`, by walking up
        // from the source file looking for a `lib.rs`/`main.rs` *sibling* at
        // each level; every test here is an integration test under `tests/`,
        // which is never inside `src/`, so that walk never succeeds and
        // regressions silently go unrecorded. `Direct` names the resulting
        // path explicitly instead, mirroring the layout `SourceParallel`
        // would have produced had the walk succeeded. The path is relative
        // to the process's current directory, which `cargo test` sets to the
        // crate root, matching where `tests/` itself lives.
        failure_persistence: Some(Box::new(
            proptest::test_runner::FileFailurePersistence::Direct(
                "tests/proptest-regressions/dist_properties.txt",
            ),
        )),
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

/// `quantile(0)`/`quantile(1)` vs a support bound. An infinite bound must be
/// matched exactly (any finite slack there would pass trivially); a finite
/// one is checked to `BOUNDS_SLACK` ulps.
fn assert_bound(got: f64, want: f64, tag: &str) -> Result<(), TestCaseError> {
    if want.is_infinite() {
        prop_assert_eq!(got, want, "{}: got {:e}, want {:e}", tag, got, want);
    } else {
        let slack = BOUNDS_SLACK * EPS * want.abs().max(1.0);
        prop_assert!(
            (got - want).abs() <= slack,
            "{tag}: got {got:e}, want {want:e}, slack {slack:e}"
        );
    }
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

/// `sf(t) ≈ cdf(-t)` for a distribution symmetric about 0.
fn assert_symmetric(sf_t: f64, cdf_neg_t: f64) -> Result<(), TestCaseError> {
    let scale = sf_t.max(cdf_neg_t).max(1e-300);
    let diff = (sf_t - cdf_neg_t).abs();
    let slack = SYMMETRIC_SLACK * EPS * scale;
    prop_assert!(
        diff <= slack,
        "sf(t)={sf_t:e} vs cdf(-t)={cdf_neg_t:e}, diff={diff:e}, slack={slack:e}"
    );
    Ok(())
}

/// `log_mass(k+1) - log_mass(k)` against `ln` of a closed-form successive
/// mass ratio, derived directly from a pmf's own recurrence (e.g. Binomial's
/// `mass(k+1)/mass(k) = (n-k)/(k+1) · p/q`), never by exponentiating and
/// dividing `mass` values. `log_mass_matches_ln_mass` recovers `mass` from
/// `log_mass` via `exp` and compares against `ln` of that same `mass`, so it
/// can only ever measure the `exp`/`ln` round-trip, not drift in `log_mass`
/// itself; this closed form has no such dependency on `log_mass`, so it can.
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
                        let x = d.quantile(prob).unwrap();
                        // A true root outside the f64 range correctly
                        // saturates to +-infinity (`bracketed_newton`'s own
                        // documented contract); `cdf` there is trivially 0/1
                        // and does not exercise the backward-error identity,
                        // so such a draw is uninteresting, not a failure.
                        prop_assume!(x.is_finite());
                        // `x == 0` is ambiguous the same way: genuine
                        // (the true root itself is below the smallest
                        // representable positive float, so no nonzero
                        // answer exists) or a `-ln(1-p)`/`exp`-underflow bug
                        // that collapsed a representable nonzero root onto 0.
                        // `cdf` is increasing, so which one it is shows at
                        // the smallest representable step off 0: if `cdf`
                        // there already reaches `prob`, the true root is
                        // smaller still (genuine, skip); if it does not, a
                        // larger representable root was missed (a bug —
                        // let the assertion below fail on it).
                        prop_assume!(x != 0.0 || d.cdf(SMALLEST_POSITIVE) < prob);
                        let cdf_x = d.cdf(x);
                        let resid = (cdf_x - prob).abs();
                        // A quantile built as `exp(location + scale * z)`
                        // (`LogNormal`, and any other log-space root) carries
                        // the seed's rounding scaled by `|ln x|`, not by `x`
                        // itself: one ulp of a mu ~ 500 argument to `exp`
                        // becomes a ~500-ulp relative error in `x`. `cond`
                        // extends the `|x|*pdf(x)` sensitivity term by that
                        // factor; it is 1 for any root not built this way.
                        let cond = libm::fabs(x).max(1.0).ln().max(1.0);
                        let slack = BACKWARD_SLACK * EPS * (prob + x_pdf(&d, x) * cond);
                        prop_assert!(
                            resid <= slack || straddles(|t| d.cdf(t), x, prob),
                            "quantile: p={prob:e} x={x:e} cdf={cdf_x:e} resid={resid:e} slack={slack:e}"
                        );
                        let xi = d.isf(prob).unwrap();
                        prop_assume!(xi.is_finite());
                        // Mirror of the quantile-side check above, for `sf`
                        // (decreasing): genuine iff `sf` at the smallest
                        // representable step off 0 has already fallen below
                        // `prob`.
                        prop_assume!(xi != 0.0 || d.sf(SMALLEST_POSITIVE) >= prob);
                        let sf_xi = d.sf(xi);
                        let residi = (sf_xi - prob).abs();
                        let condi = libm::fabs(xi).max(1.0).ln().max(1.0);
                        let slacki = BACKWARD_SLACK * EPS * (prob + x_pdf(&d, xi) * condi);
                        prop_assert!(
                            residi <= slacki || straddles(|t| -d.sf(t), xi, -prob),
                            "isf: q={prob:e} x={xi:e} sf={sf_xi:e} resid={residi:e} slack={slacki:e}"
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
                        assert_bound(d.quantile(0.0).unwrap(), d.support_min().as_f64(), "quantile(0)")?;
                        assert_bound(d.quantile(1.0).unwrap(), d.support_max().as_f64(), "quantile(1)")?;
                        Ok(())
                    })
                    .unwrap();
            }

            #[test]
            #[$monotone_attr]
            fn range_monotone_no_nan() {
                let mut runner = TestRunner::new(config());
                runner
                    .run(&($params, any_x(), any_x()), |($p, x1, x2)| {
                        let d: $ty = $ctor;
                        let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
                        let (clo, chi) = (d.cdf(lo), d.cdf(hi));
                        prop_assert!(!clo.is_nan() && !chi.is_nan(), "NaN cdf at {lo:e}/{hi:e}");
                        prop_assert!(
                            (0.0..=1.0).contains(&clo) && (0.0..=1.0).contains(&chi),
                            "cdf out of [0,1]: {clo:e}, {chi:e}"
                        );
                        prop_assert!(
                            clo <= chi + MONOTONE_SLACK,
                            "cdf not monotone: cdf({lo:e})={clo:e} > cdf({hi:e})={chi:e}"
                        );
                        let dens = d.density(lo);
                        prop_assert!(!dens.is_nan() && dens >= 0.0, "bad density at {lo:e}: {dens:e}");
                        Ok(())
                    })
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
                        let (lo, hi) = if k1 <= k2 { (k1, k2) } else { (k2, k1) };
                        let (clo, chi) = (d.cdf(lo), d.cdf(hi));
                        prop_assert!(!clo.is_nan() && !chi.is_nan(), "NaN cdf");
                        prop_assert!(
                            (0.0..=1.0).contains(&clo) && (0.0..=1.0).contains(&chi),
                            "cdf out of [0,1]: {clo:e}, {chi:e}"
                        );
                        prop_assert!(
                            clo <= chi + MONOTONE_SLACK,
                            "cdf not monotone: cdf({lo})={clo:e} > cdf({hi})={chi:e}"
                        );
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

// ---- symmetric-family identity: sf(t) == cdf(-t) --------------------------

#[test]
fn normal_symmetric_at_zero() {
    let mut runner = TestRunner::new(config());
    runner
        .run(&(pos_param(), log_uniform(-20, 20)), |(sd, t)| {
            let n = Normal::new(0.0, sd).unwrap();
            assert_symmetric(n.sf(t), n.cdf(-t))
        })
        .unwrap();
}

#[test]
fn studentt_symmetric_at_zero() {
    let mut runner = TestRunner::new(config());
    runner
        .run(&(pos_param(), log_uniform(-20, 20)), |(df, t)| {
            let s = StudentT::new(df).unwrap();
            assert_symmetric(s.sf(t), s.cdf(-t))
        })
        .unwrap();
}

#[test]
fn cauchy_symmetric_at_zero() {
    let mut runner = TestRunner::new(config());
    runner
        .run(&(pos_param(), log_uniform(-20, 20)), |(scale, t)| {
            let c = Cauchy::new(0.0, scale).unwrap();
            assert_symmetric(c.sf(t), c.cdf(-t))
        })
        .unwrap();
}

// ---- scaling identities ----------------------------------------------------

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

/// `Weibull(shape, scale)` is `scale · Exponential(1)^{1/shape}`, so
/// `Weibull::cdf(x) == Exponential(1).cdf((x/scale)^shape)`.
#[test]
fn weibull_via_exponential() {
    let mut runner = TestRunner::new(config());
    runner
        .run(
            &(log_uniform(-5, 10), pos_param(), tail_prob()),
            |(shape, scale, p)| {
                let w = Weibull::new(shape, scale).unwrap();
                let x = w.quantile(p).unwrap();
                let e = Exponential::new(1.0).unwrap();
                let y = libm::pow(x / scale, shape);
                let (c1, c2) = (w.cdf(x), e.cdf(y));
                let diff = (c1 - c2).abs();
                let slack = SCALING_SLACK * EPS * c1.max(c2).max(p);
                prop_assert!(
                    diff <= slack,
                    "Weibull via Exponential: cdf(x)={c1:e} vs Exp.cdf((x/scale)^shape)={c2:e}, diff={diff:e}"
                );
                Ok(())
            },
        )
        .unwrap();
}

/// `LogNormal(mu, sigma)` is `exp(Normal(mu, sigma))`, so
/// `LogNormal::cdf(x) == Normal::cdf(ln x)` for `x > 0`. This is a wiring
/// guard, not an independent numerical cross-check: `LogNormal::cdf` is
/// itself implemented as `self.z().cdf(ln(x))` on an internal `Normal`, so a
/// passing run only confirms that delegation still holds, not that either
/// side's `cdf` is numerically accurate (the oracle grids and the
/// backward-error property cover that).
#[test]
fn lognormal_via_normal() {
    let mut runner = TestRunner::new(config());
    runner
        .run(
            &(loc_param(), pos_param(), tail_prob()),
            |(mu, sigma, p)| {
                let ln_d = LogNormal::new(mu, sigma).unwrap();
                let x = ln_d.quantile(p).unwrap();
                let n = Normal::new(mu, sigma).unwrap();
                let (c1, c2) = (ln_d.cdf(x), n.cdf(libm::log(x)));
                let diff = (c1 - c2).abs();
                let slack = SCALING_SLACK * EPS * c1.max(c2).max(p);
                prop_assert!(
                    diff <= slack,
                    "LogNormal via Normal: cdf(x)={c1:e} vs Normal.cdf(ln x)={c2:e}, diff={diff:e}"
                );
                Ok(())
            },
        )
        .unwrap();
}

/// `ChiSquared(k)` is defined as `Gamma(k/2, rate 1/2)` (`ChiSquared::as_gamma`),
/// so its `cdf` must exactly match a freshly built `Gamma(k/2, 0.5)`'s. Like
/// `lognormal_via_normal`, this is a wiring guard: `ChiSquared::cdf` calls
/// `self.as_gamma().cdf(x)` directly, so it is the same computation on both
/// sides by construction, and a passing run only confirms `as_gamma`'s field
/// mapping stays `(0.5 * k, 0.5)`, not that `Gamma::cdf` itself is accurate.
#[test]
fn chisquared_via_gamma() {
    let mut runner = TestRunner::new(config());
    runner
        .run(&(pos_param(), tail_prob()), |(k, p)| {
            let c = ChiSquared::new(k).unwrap();
            let x = c.quantile(p).unwrap();
            let g = Gamma::new(0.5 * k, 0.5).unwrap();
            let (c1, c2) = (c.cdf(x), g.cdf(x));
            prop_assert_eq!(
                c1,
                c2,
                "ChiSquared vs Gamma(k/2, 1/2): {:e} vs {:e}",
                c1,
                c2
            );
            Ok(())
        })
        .unwrap();
}
