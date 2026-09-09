#![cfg(feature = "dist")]
//! Class-A distribution-suite oracle grids, validated against committed mpmath
//! truth via the shared harness (accuracy-validation spec). Continuous pdf/cdf/sf
//! use the value ladder (1e-12), ppf the inverse ladder (1e-10) with tail rows
//! asserting a curated band. Discrete pmf/cdf use the value ladder; the discrete
//! ppf is an exact integer, so it asserts within ±0.5 (no float band).
mod common;
use common::{Tol, check_grid, measure_band};

use commonstats::dist::continuous::Beta;
use commonstats::dist::continuous::Cauchy;
use commonstats::dist::continuous::ChiSquared;
use commonstats::dist::continuous::Exponential;
use commonstats::dist::continuous::FisherF;
use commonstats::dist::continuous::Gamma;
use commonstats::dist::continuous::LogNormal;
use commonstats::dist::continuous::Normal;
use commonstats::dist::continuous::StudentT;
use commonstats::dist::continuous::Uniform;
use commonstats::dist::continuous::Weibull;
use commonstats::dist::discrete::Bernoulli;
use commonstats::dist::discrete::Binomial;
use commonstats::dist::discrete::Geometric;
use commonstats::dist::discrete::Hypergeometric;
use commonstats::dist::discrete::NegBinomial;
use commonstats::dist::discrete::Poisson;
use commonstats::dist::{
    ContinuousCdf, ContinuousDensity, DiscreteCdf, DiscreteMass, Distribution,
};

// Bulk ladder (spec §3): pdf/cdf/sf/pmf value-kind vs ppf inverse-kind.
const VAL: Tol = Tol {
    rel: 1e-12,
    abs: 1e-14,
};
const INV: Tol = Tol {
    rel: 1e-10,
    abs: 1e-12,
};
// Discrete ppf returns an exact integer (i64→f64); the only error is the i64→f64
// cast, so any result within ½ of the truth integer is an exact match.
const EXACT: Tol = Tol {
    rel: 1e-15,
    abs: 0.5,
};

// ---- continuous oracle grids -------------------------------------------------
#[test]
fn normal_oracle() {
    let n = Normal::new(0.5, 2.0).unwrap();
    check_grid("dist_normal_pdf", VAL, |a| n.density(a[0]));
    check_grid("dist_normal_cdf", VAL, |a| n.cdf(a[0]));
    check_grid("dist_normal_sf", VAL, |a| n.sf(a[0]));
    check_grid("dist_normal_ppf", INV, |a| n.quantile(a[0]).unwrap());
    check_grid("dist_normal_isf", INV, |a| n.isf(a[0]).unwrap());
}

#[test]
fn studentt_oracle() {
    let t = StudentT::new(7.0).unwrap();
    check_grid("dist_studentt_pdf", VAL, |a| t.density(a[0]));
    check_grid("dist_studentt_cdf", VAL, |a| t.cdf(a[0]));
    check_grid("dist_studentt_sf", VAL, |a| t.sf(a[0]));
    check_grid("dist_studentt_ppf", INV, |a| t.quantile(a[0]).unwrap());
    check_grid("dist_studentt_isf", INV, |a| t.isf(a[0]).unwrap());
}

#[test]
fn chisquared_oracle() {
    let c = ChiSquared::new(5.0).unwrap();
    check_grid("dist_chisquared_pdf", VAL, |a| c.density(a[0]));
    check_grid("dist_chisquared_cdf", VAL, |a| c.cdf(a[0]));
    check_grid("dist_chisquared_sf", VAL, |a| c.sf(a[0]));
    check_grid("dist_chisquared_ppf", INV, |a| c.quantile(a[0]).unwrap());
    check_grid("dist_chisquared_isf", INV, |a| c.isf(a[0]).unwrap());
}

#[test]
fn fisherf_oracle() {
    let f = FisherF::new(6.0, 12.0).unwrap();
    check_grid("dist_fisherf_pdf", VAL, |a| f.density(a[0]));
    check_grid("dist_fisherf_cdf", VAL, |a| f.cdf(a[0]));
    check_grid("dist_fisherf_sf", VAL, |a| f.sf(a[0]));
    check_grid("dist_fisherf_ppf", INV, |a| f.quantile(a[0]).unwrap());
    check_grid("dist_fisherf_isf", INV, |a| f.isf(a[0]).unwrap());
}

#[test]
fn uniform_dist_oracle() {
    let u = Uniform::new(-1.0, 3.0).unwrap();
    check_grid("dist_uniform_pdf", VAL, |a| u.density(a[0]));
    check_grid("dist_uniform_cdf", VAL, |a| u.cdf(a[0]));
    check_grid("dist_uniform_sf", VAL, |a| u.sf(a[0]));
    check_grid("dist_uniform_ppf", INV, |a| u.quantile(a[0]).unwrap());
    check_grid("dist_uniform_isf", INV, |a| u.isf(a[0]).unwrap());
}

#[test]
fn exponential_oracle() {
    let e = Exponential::new(1.5).unwrap();
    check_grid("dist_exponential_pdf", VAL, |a| e.density(a[0]));
    check_grid("dist_exponential_cdf", VAL, |a| e.cdf(a[0]));
    check_grid("dist_exponential_sf", VAL, |a| e.sf(a[0]));
    check_grid("dist_exponential_ppf", INV, |a| e.quantile(a[0]).unwrap());
    check_grid("dist_exponential_isf", INV, |a| e.isf(a[0]).unwrap());
}

#[test]
fn cauchy_oracle() {
    let c = Cauchy::new(1.0, 2.0).unwrap();
    check_grid("dist_cauchy_pdf", VAL, |a| c.density(a[0]));
    check_grid("dist_cauchy_cdf", VAL, |a| c.cdf(a[0]));
    check_grid("dist_cauchy_sf", VAL, |a| c.sf(a[0]));
    check_grid("dist_cauchy_ppf", INV, |a| c.quantile(a[0]).unwrap());
    check_grid("dist_cauchy_isf", INV, |a| c.isf(a[0]).unwrap());
}

#[test]
fn weibull_oracle() {
    let w = Weibull::new(2.0, 1.5).unwrap();
    check_grid("dist_weibull_pdf", VAL, |a| w.density(a[0]));
    check_grid("dist_weibull_cdf", VAL, |a| w.cdf(a[0]));
    check_grid("dist_weibull_sf", VAL, |a| w.sf(a[0]));
    check_grid("dist_weibull_ppf", INV, |a| w.quantile(a[0]).unwrap());
    check_grid("dist_weibull_isf", INV, |a| w.isf(a[0]).unwrap());
}

#[test]
fn lognormal_oracle() {
    let l = LogNormal::new(0.5, 0.75).unwrap();
    check_grid("dist_lognormal_pdf", VAL, |a| l.density(a[0]));
    check_grid("dist_lognormal_cdf", VAL, |a| l.cdf(a[0]));
    check_grid("dist_lognormal_sf", VAL, |a| l.sf(a[0]));
    check_grid("dist_lognormal_ppf", INV, |a| l.quantile(a[0]).unwrap());
    check_grid("dist_lognormal_isf", INV, |a| l.isf(a[0]).unwrap());
}

#[test]
fn gamma_oracle() {
    let g = Gamma::new(3.5, 2.0).unwrap();
    check_grid("dist_gamma_pdf", VAL, |a| g.density(a[0]));
    check_grid("dist_gamma_cdf", VAL, |a| g.cdf(a[0]));
    check_grid("dist_gamma_sf", VAL, |a| g.sf(a[0]));
    check_grid("dist_gamma_ppf", INV, |a| g.quantile(a[0]).unwrap());
    check_grid("dist_gamma_isf", INV, |a| g.isf(a[0]).unwrap());
}

#[test]
fn beta_oracle() {
    let b = Beta::new(2.5, 4.0).unwrap();
    check_grid("dist_beta_pdf", VAL, |a| b.density(a[0]));
    check_grid("dist_beta_cdf", VAL, |a| b.cdf(a[0]));
    check_grid("dist_beta_sf", VAL, |a| b.sf(a[0]));
    check_grid("dist_beta_ppf", INV, |a| b.quantile(a[0]).unwrap());
    check_grid("dist_beta_isf", INV, |a| b.isf(a[0]).unwrap());
}

#[test]
fn small_shape_deep_lower_tail_quantile() {
    // Closed form for shape ½: P(½, y) = erf(√y), so χ²(1).quantile(p) =
    // 2·erf⁻¹(p)² = π p²/2 · (1 + O(p²)); the O(p²) term is below f64 resolution
    // for p ≤ 1e-8. Gamma(½, rate ½) is the same distribution. Exercises the
    // small-p seed of the Newton solvers, which the fixtures (k=5, shape=3.5)
    // never reach.
    let c = ChiSquared::new(1.0).unwrap();
    let g = Gamma::new(0.5, 0.5).unwrap();
    for p in [1e-8, 1e-12] {
        let truth = core::f64::consts::FRAC_PI_2 * p * p;
        for (name, got) in [
            ("chi2", c.quantile(p).unwrap()),
            ("gamma", g.quantile(p).unwrap()),
        ] {
            let rel = ((got - truth) / truth).abs();
            assert!(
                rel < 1e-13,
                "{name}.quantile({p:e}) = {got:e}, want {truth:e} (rel {rel:e})"
            );
        }
    }
    // Shape 0.1: no closed form; require cdf(quantile(p)) to round-trip.
    let c = ChiSquared::new(0.2).unwrap();
    for p in [1e-6, 1e-12] {
        let x = c.quantile(p).unwrap();
        let rel = ((c.cdf(x) - p) / p).abs();
        assert!(
            rel < 1e-13,
            "chi2(0.2): cdf(quantile({p:e})) = {:e} (rel {rel:e})",
            c.cdf(x)
        );
    }
}

// ---- discrete oracle grids ---------------------------------------------------
#[test]
fn bernoulli_oracle() {
    let b = Bernoulli::new(0.4).unwrap();
    check_grid("dist_bernoulli_pmf", VAL, |a| b.mass(a[0] as i64));
    check_grid("dist_bernoulli_cdf", VAL, |a| b.cdf(a[0] as i64));
    check_grid("dist_bernoulli_ppf", EXACT, |a| {
        b.quantile(a[0]).unwrap() as f64
    });
}

#[test]
fn binomial_oracle() {
    let b = Binomial::new(20, 0.35).unwrap();
    check_grid("dist_binomial_pmf", VAL, |a| b.mass(a[0] as i64));
    check_grid("dist_binomial_cdf", VAL, |a| b.cdf(a[0] as i64));
    check_grid("dist_binomial_ppf", EXACT, |a| {
        b.quantile(a[0]).unwrap() as f64
    });
}

#[test]
fn poisson_oracle() {
    let p = Poisson::new(4.5).unwrap();
    check_grid("dist_poisson_pmf", VAL, |a| p.mass(a[0] as i64));
    check_grid("dist_poisson_cdf", VAL, |a| p.cdf(a[0] as i64));
    check_grid("dist_poisson_ppf", EXACT, |a| {
        p.quantile(a[0]).unwrap() as f64
    });
}

#[test]
fn geometric_oracle() {
    let g = Geometric::new(0.4).unwrap();
    check_grid("dist_geometric_pmf", VAL, |a| g.mass(a[0] as i64));
    check_grid("dist_geometric_cdf", VAL, |a| g.cdf(a[0] as i64));
    check_grid("dist_geometric_ppf", EXACT, |a| {
        g.quantile(a[0]).unwrap() as f64
    });
}

#[test]
fn negbinomial_oracle() {
    let nb = NegBinomial::new(4.0, 0.3).unwrap();
    check_grid("dist_negbinomial_pmf", VAL, |a| nb.mass(a[0] as i64));
    check_grid("dist_negbinomial_cdf", VAL, |a| nb.cdf(a[0] as i64));
    check_grid("dist_negbinomial_ppf", EXACT, |a| {
        nb.quantile(a[0]).unwrap() as f64
    });
}

#[test]
fn hypergeometric_oracle() {
    let h = Hypergeometric::new(30, 12, 10).unwrap();
    check_grid("dist_hypergeometric_pmf", VAL, |a| h.mass(a[0] as i64));
    check_grid("dist_hypergeometric_cdf", VAL, |a| h.cdf(a[0] as i64));
    check_grid("dist_hypergeometric_ppf", EXACT, |a| {
        h.quantile(a[0]).unwrap() as f64
    });
}

/// Band calibration (spec §5). Run with `cargo test -- --ignored --nocapture`;
/// the printed values (lightly padded) are curated by a human into the generator's
/// TAIL_BANDS ledger. Tests never write fixtures. Only continuous ppf/isf
/// fixtures carry tail rows.
#[test]
#[ignore]
fn measure_tail_bands() {
    let n = Normal::new(0.5, 2.0).unwrap();
    let t = StudentT::new(7.0).unwrap();
    let c = ChiSquared::new(5.0).unwrap();
    let f = FisherF::new(6.0, 12.0).unwrap();
    let u = Uniform::new(-1.0, 3.0).unwrap();
    let e = Exponential::new(1.5).unwrap();
    let ca = Cauchy::new(1.0, 2.0).unwrap();
    let w = Weibull::new(2.0, 1.5).unwrap();
    let l = LogNormal::new(0.5, 0.75).unwrap();
    let g = Gamma::new(3.5, 2.0).unwrap();
    let b = Beta::new(2.5, 4.0).unwrap();
    let bands = [
        (
            "normal",
            measure_band("dist_normal_ppf", |a| n.quantile(a[0]).unwrap()),
            measure_band("dist_normal_isf", |a| n.isf(a[0]).unwrap()),
        ),
        (
            "studentt",
            measure_band("dist_studentt_ppf", |a| t.quantile(a[0]).unwrap()),
            measure_band("dist_studentt_isf", |a| t.isf(a[0]).unwrap()),
        ),
        (
            "chisquared",
            measure_band("dist_chisquared_ppf", |a| c.quantile(a[0]).unwrap()),
            measure_band("dist_chisquared_isf", |a| c.isf(a[0]).unwrap()),
        ),
        (
            "fisherf",
            measure_band("dist_fisherf_ppf", |a| f.quantile(a[0]).unwrap()),
            measure_band("dist_fisherf_isf", |a| f.isf(a[0]).unwrap()),
        ),
        (
            "uniform",
            measure_band("dist_uniform_ppf", |a| u.quantile(a[0]).unwrap()),
            measure_band("dist_uniform_isf", |a| u.isf(a[0]).unwrap()),
        ),
        (
            "exponential",
            measure_band("dist_exponential_ppf", |a| e.quantile(a[0]).unwrap()),
            measure_band("dist_exponential_isf", |a| e.isf(a[0]).unwrap()),
        ),
        (
            "cauchy",
            measure_band("dist_cauchy_ppf", |a| ca.quantile(a[0]).unwrap()),
            measure_band("dist_cauchy_isf", |a| ca.isf(a[0]).unwrap()),
        ),
        (
            "weibull",
            measure_band("dist_weibull_ppf", |a| w.quantile(a[0]).unwrap()),
            measure_band("dist_weibull_isf", |a| w.isf(a[0]).unwrap()),
        ),
        (
            "lognormal",
            measure_band("dist_lognormal_ppf", |a| l.quantile(a[0]).unwrap()),
            measure_band("dist_lognormal_isf", |a| l.isf(a[0]).unwrap()),
        ),
        (
            "gamma",
            measure_band("dist_gamma_ppf", |a| g.quantile(a[0]).unwrap()),
            measure_band("dist_gamma_isf", |a| g.isf(a[0]).unwrap()),
        ),
        (
            "beta",
            measure_band("dist_beta_ppf", |a| b.quantile(a[0]).unwrap()),
            measure_band("dist_beta_isf", |a| b.isf(a[0]).unwrap()),
        ),
    ];
    for (dist, ppf, isf) in bands {
        for (kind, val) in [("ppf", ppf), ("isf", isf)] {
            match val {
                Some(b) => println!("TAIL BAND  dist_{dist}_{kind:<3} → {b:e}"),
                None => println!("TAIL BAND  dist_{dist}_{kind:<3} → (no tail rows)"),
            }
        }
    }
}

// ---- non-grid sanity tests (kept verbatim) -----------------------------------
#[test]
fn normal_basic() {
    let n = Normal::new(0.0, 1.0).unwrap();
    assert!((n.cdf(0.0) - 0.5).abs() < 1e-15);
    assert!((n.density(0.0) - 0.398_942_280_401_432_7).abs() < 1e-15);
    assert_eq!(n.quantile(0.0).unwrap(), f64::NEG_INFINITY);
    assert_eq!(n.quantile(1.0).unwrap(), f64::INFINITY);
    assert!((n.quantile(0.975).unwrap() - 1.959_963_984_540_054).abs() < 1e-9);
    assert!(Normal::new(0.0, -1.0).is_err());
    assert_eq!(n.mean(), Some(0.0));
    assert_eq!(n.variance(), Some(1.0));
    // sf override avoids 1-cdf cancellation in the far upper tail.
    assert!(n.sf(10.0) > 0.0 && n.sf(10.0) < 1e-20);
}

#[test]
fn studentt_basic() {
    let t = StudentT::new(5.0).unwrap();
    assert!((t.cdf(0.0) - 0.5).abs() < 1e-12);
    assert_eq!(t.quantile(0.5).unwrap(), 0.0);
    // scipy: t.ppf(0.975, 5) = 2.5705818366147395
    assert!((t.quantile(0.975).unwrap() - 2.570_581_836_614_74).abs() < 1e-9);
    // Deep upper tail via the direct betai form (mpmath, df=7, x=50); the
    // default `1 − cdf` would carry ~1e-6 relative error here.
    let t7 = StudentT::new(7.0).unwrap();
    let sf = t7.sf(50.0);
    assert!(((sf - 1.675_626_297_775_02e-10) / 1.675_626_297_775_02e-10).abs() < 1e-12);
    assert!((t7.sf(-50.0) - (1.0 - 1.675_626_297_775_02e-10)).abs() < 1e-15);
    assert_eq!(t7.sf(f64::INFINITY), 0.0);
    assert_eq!(t.mean(), Some(0.0)); // df>1
    assert_eq!(StudentT::new(1.0).unwrap().mean(), None); // df=1 undefined
    assert_eq!(t.variance(), Some(5.0 / 3.0)); // df/(df-2)
    assert!(StudentT::new(0.0).is_err());
}

#[test]
fn chisquared_basic() {
    let c = ChiSquared::new(4.0).unwrap();
    assert_eq!(c.mean(), Some(4.0));
    assert_eq!(c.variance(), Some(8.0)); // 2k
    // scipy: chi2.ppf(0.95, 4) = 9.487729036781154
    assert!((c.quantile(0.95).unwrap() - 9.487_729_036_781_154).abs() < 1e-7);
    // k=1 density singular at 0.
    let c1 = ChiSquared::new(1.0).unwrap();
    assert_eq!(c1.density(0.0), 0.0);
    assert_eq!(c1.log_density(0.0), f64::NEG_INFINITY);
    assert!(ChiSquared::new(0.0).is_err());
}

#[test]
fn fisherf_basic() {
    let f = FisherF::new(5.0, 10.0).unwrap();
    // scipy: f.ppf(0.95, 5, 10) = 3.3258345179316175
    assert!((f.quantile(0.95).unwrap() - 3.325_834_517_931_617).abs() < 1e-7);
    assert_eq!(f.mean(), Some(10.0 / 8.0)); // dfd/(dfd-2), dfd>2
    assert_eq!(FisherF::new(5.0, 2.0).unwrap().mean(), None); // dfd=2 undefined
    assert_eq!(f.density(-1.0), 0.0);
    assert!(FisherF::new(0.0, 10.0).is_err());
}

#[test]
fn uniform_dist_basic() {
    let u = Uniform::new(2.0, 6.0).unwrap();
    assert_eq!(u.mean(), Some(4.0));
    assert_eq!(u.variance(), Some(16.0 / 12.0)); // (b-a)^2/12
    assert!((u.density(3.0) - 0.25).abs() < 1e-15);
    assert_eq!(u.density(1.0), 0.0);
    assert!((u.cdf(4.0) - 0.5).abs() < 1e-15);
    assert!((u.quantile(0.25).unwrap() - 3.0).abs() < 1e-15);
    assert!(Uniform::new(6.0, 2.0).is_err());
}

#[test]
fn exponential_basic() {
    let e = Exponential::new(2.0).unwrap();
    assert_eq!(e.mean(), Some(0.5)); // 1/λ
    assert_eq!(e.variance(), Some(0.25)); // 1/λ²
    assert_eq!(e.density(-1.0), 0.0);
    // cdf(x)=1-e^{-2x}; quantile(0.5)=ln2/2
    assert!((e.quantile(0.5).unwrap() - core::f64::consts::LN_2 / 2.0).abs() < 1e-15);
    assert!(e.sf(10.0) > 0.0 && e.sf(10.0) < 1e-8);
    assert!(Exponential::new(0.0).is_err());
}

#[test]
fn cauchy_basic() {
    let c = Cauchy::new(0.0, 1.0).unwrap();
    assert_eq!(c.mean(), None);
    assert_eq!(c.variance(), None);
    assert!((c.cdf(0.0) - 0.5).abs() < 1e-15);
    assert!((c.quantile(0.5).unwrap()).abs() < 1e-12);
    // scipy: cauchy.ppf(0.75) = 1.0
    assert!((c.quantile(0.75).unwrap() - 1.0).abs() < 1e-12);
    assert!(Cauchy::new(0.0, 0.0).is_err());
}

#[test]
fn weibull_basic() {
    let w = Weibull::new(1.5, 2.0).unwrap();
    assert_eq!(w.density(-1.0), 0.0);
    // scipy: weibull_min(1.5, scale=2).ppf(0.5) = 1.5664395375493028
    assert!((w.quantile(0.5).unwrap() - 1.566_439_537_5).abs() < 1e-6);
    assert!(w.sf(10.0) > 0.0 && w.sf(10.0) < 1e-4);
    assert!(Weibull::new(0.0, 2.0).is_err());
    // k=1 reduces to Exponential(scale=λ): mean=λ.
    let w1 = Weibull::new(1.0, 2.0).unwrap();
    assert!((w1.mean().unwrap() - 2.0).abs() < 1e-12);
}

#[test]
fn lognormal_basic() {
    let l = LogNormal::new(0.0, 1.0).unwrap();
    assert_eq!(l.density(-1.0), 0.0);
    assert_eq!(l.density(0.0), 0.0);
    // median = e^μ = 1
    assert!((l.quantile(0.5).unwrap() - 1.0).abs() < 1e-9);
    // mean = e^{μ+σ²/2} = e^{0.5}
    assert!((l.mean().unwrap() - 0.5_f64.exp()).abs() < 1e-12);
    assert!(LogNormal::new(0.0, -1.0).is_err());
}

#[test]
fn gamma_basic() {
    let g = Gamma::new(2.0, 1.0).unwrap(); // shape 2, rate 1
    assert_eq!(g.mean(), Some(2.0)); // α/β
    assert_eq!(g.variance(), Some(2.0)); // α/β²
    assert_eq!(g.density(-1.0), 0.0);
    // scipy: gamma(a=2, scale=1).ppf(0.5) = 1.6783469900166612
    assert!((g.quantile(0.5).unwrap() - 1.678_346_990_016_661).abs() < 1e-8);
    assert!(Gamma::new(0.0, 1.0).is_err());
    // Gamma(1, λ) == Exponential(λ): cdf(x)=1-e^{-x} at rate 1.
    let g1 = Gamma::new(1.0, 1.0).unwrap();
    assert!((g1.cdf(1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
}

#[test]
fn beta_basic() {
    let b = Beta::new(2.0, 3.0).unwrap();
    assert_eq!(b.mean(), Some(2.0 / 5.0)); // α/(α+β)
    assert_eq!(b.density(-0.1), 0.0);
    assert_eq!(b.density(1.1), 0.0);
    // scipy: beta(2,3).ppf(0.5) = 0.3857275681323895
    assert!((b.quantile(0.5).unwrap() - 0.385_727_568_1).abs() < 1e-8);
    assert!((b.cdf(1.0) - 1.0).abs() < 1e-15);
    assert!(Beta::new(0.0, 3.0).is_err());
}

#[test]
fn bernoulli_basic() {
    let b = Bernoulli::new(0.3).unwrap();
    assert!((b.mass(0) - 0.7).abs() < 1e-15);
    assert!((b.mass(1) - 0.3).abs() < 1e-15);
    assert_eq!(b.mass(2), 0.0);
    assert!((b.cdf(0) - 0.7).abs() < 1e-15);
    assert_eq!(b.cdf(1), 1.0);
    assert_eq!(b.quantile(0.5).unwrap(), 0); // cdf(0)=0.7 ≥ 0.5
    assert_eq!(b.quantile(0.8).unwrap(), 1);
    assert!(b.quantile(1.5).is_err());
    assert_eq!(b.mean(), Some(0.3));
    // p ∈ {0,1}: 0·ln0 := 0, mass finite, no NaN.
    let b0 = Bernoulli::new(0.0).unwrap();
    assert_eq!(b0.mass(0), 1.0);
    assert_eq!(b0.log_mass(0), 0.0);
    assert!(Bernoulli::new(-0.1).is_err());
}

#[test]
fn binomial_basic() {
    let b = Binomial::new(10, 0.3).unwrap();
    assert_eq!(b.mean(), Some(3.0)); // np
    assert!((b.variance().unwrap() - 2.1).abs() < 1e-14); // np(1-p)
    // scipy: binom.pmf(3, 10, 0.3) = 0.26682793200000005
    assert!((b.mass(3) - 0.266_827_932_0).abs() < 1e-12);
    assert_eq!(b.mass(-1), 0.0);
    assert_eq!(b.mass(11), 0.0);
    // scipy: binom.cdf(3,10,0.3) = 0.6496107184
    assert!((b.cdf(3) - 0.649_610_718_4).abs() < 1e-10);
    assert_eq!(b.quantile(0.0).unwrap(), 0);
    assert_eq!(b.quantile(1.0).unwrap(), 10);
    assert!(Binomial::new(0, 0.3).is_err());
    assert!(Binomial::new(10, 1.5).is_err());
}

#[test]
fn poisson_basic() {
    let p = Poisson::new(3.0).unwrap();
    assert_eq!(p.mean(), Some(3.0));
    assert_eq!(p.variance(), Some(3.0));
    // scipy: poisson.pmf(2, 3) = 0.22404180765538775
    assert!((p.mass(2) - 0.224_041_807_655_387_7).abs() < 1e-12);
    assert_eq!(p.mass(-1), 0.0);
    // scipy: poisson.cdf(2,3) = 0.42319008112684353
    assert!((p.cdf(2) - 0.423_190_081_126_843_5).abs() < 1e-10);
    assert_eq!(p.quantile(0.0).unwrap(), 0);
    assert!(Poisson::new(0.0).is_err());
}

#[test]
fn geometric_basic() {
    let g = Geometric::new(0.25).unwrap();
    assert_eq!(g.mean(), Some(4.0)); // 1/p
    assert_eq!(g.mass(0), 0.0); // 1-indexed: support starts at 1
    // scipy: geom.pmf(1, 0.25) = 0.25; geom.pmf(3,0.25)=0.140625
    assert!((g.mass(1) - 0.25).abs() < 1e-15);
    assert!((g.mass(3) - 0.140_625).abs() < 1e-15);
    // scipy: geom.cdf(2, 0.25) = 0.4375
    assert!((g.cdf(2) - 0.4375).abs() < 1e-15);
    assert_eq!(g.quantile(0.25).unwrap(), 1);
    assert!(Geometric::new(0.0).is_err()); // p ∈ (0,1]
    assert!(Geometric::new(1.0).is_ok());
}

#[test]
fn negbinomial_basic() {
    // r failures-target, p = success prob. Oracle: nbinom(n=r, p=1-p).
    let nb = NegBinomial::new(5.0, 0.4).unwrap(); // p=success prob 0.4
    // mean = r·p/(1-p) = 5·0.4/0.6 = 3.333...
    assert!((nb.mean().unwrap() - 5.0 * 0.4 / 0.6).abs() < 1e-12);
    assert_eq!(nb.mass(-1), 0.0);
    // scipy: nbinom(5, 1-0.4=0.6).pmf(2) = 0.18662400000000004
    assert!((nb.mass(2) - 0.186_624).abs() < 1e-10);
    assert_eq!(nb.quantile(0.0).unwrap(), 0);
    assert!(NegBinomial::new(0.0, 0.4).is_err());
    assert!(NegBinomial::new(5.0, 0.0).is_err()); // p ∈ (0,1)
}

#[test]
fn hypergeometric_basic() {
    // N=20 population, K=7 successes, n=12 draws (scipy hypergeom(M=20,n=7,N=12)).
    let h = Hypergeometric::new(20, 7, 12).unwrap();
    assert_eq!(h.mean(), Some(12.0 * 7.0 / 20.0)); // n·K/N
    // support is max(0, n+K-N)..min(n,K) = max(0,-1)..min(12,7) = 0..7
    assert_eq!(h.mass(8), 0.0); // above min(n,K)
    // scipy: hypergeom(M=20,n=7,N=12).pmf(4) = 0.3575851393188855
    assert!((h.mass(4) - 0.357_585_139).abs() < 1e-6);
    assert_eq!(h.quantile(0.0).unwrap(), 0);
    assert!(Hypergeometric::new(20, 25, 12).is_err()); // K > N
    assert!(Hypergeometric::new(20, 7, 25).is_err()); // n > N
}

#[cfg(all(feature = "dist", feature = "rng"))]
#[test]
fn sampler_draws_in_support() {
    use commonstats::dist::continuous::{Exponential, Normal, Uniform};
    use commonstats::dist::{Distribution, Sampler};
    use commonstats::rng::CommonStatsRng;
    let mut rng = CommonStatsRng::new(99, 0);
    let n = Normal::new(0.0, 1.0).unwrap();
    let e = Exponential::new(1.0).unwrap();
    let u = Uniform::new(-2.0, 5.0).unwrap();
    for _ in 0..10_000 {
        let xn = n.sample(&mut rng);
        assert!(xn.is_finite());
        let xe = e.sample(&mut rng);
        assert!(xe >= 0.0, "exp sample negative: {xe}");
        let xu = u.sample(&mut rng);
        assert!(xu >= u.support_min().as_f64() && xu <= u.support_max().as_f64());
    }
}
