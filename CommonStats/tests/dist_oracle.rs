#![cfg(feature = "dist")]
//! Class-A distribution-suite oracle grids, validated against committed mpmath
//! truth via the shared harness (accuracy-validation spec). Continuous pdf/cdf/sf
//! use the value ladder (1e-12), ppf the inverse ladder (1e-10) with tail rows
//! asserting a curated band. Discrete pmf/cdf use the value ladder; the discrete
//! ppf is an exact integer, so it asserts within ±0.5 (no float band).
mod common;
use common::{Tol, check_grid, measure_band, rel_err};

use commonstats::dist::continuous::Beta;
use commonstats::dist::continuous::Cauchy;
use commonstats::dist::continuous::ChiSquared;
use commonstats::dist::continuous::Exponential;
use commonstats::dist::continuous::FisherF;
use commonstats::dist::continuous::Gamma;
use commonstats::dist::continuous::InverseGaussian;
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
fn inversegaussian_oracle() {
    let ig = InverseGaussian::new(1.5, 2.0).unwrap();
    check_grid("dist_inversegaussian_pdf", VAL, |a| ig.density(a[0]));
    check_grid("dist_inversegaussian_cdf", VAL, |a| ig.cdf(a[0]));
    check_grid("dist_inversegaussian_sf", VAL, |a| ig.sf(a[0]));
    check_grid("dist_inversegaussian_ppf", INV, |a| {
        ig.quantile(a[0]).unwrap()
    });
    check_grid("dist_inversegaussian_isf", INV, |a| ig.isf(a[0]).unwrap());
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
    // F(1, ·): the WH seed goes negative, and a fixed floor would leave Newton
    // halving toward a root ~1e-200 and running out of iterations.
    let f = FisherF::new(1.0, 10.0).unwrap();
    for p in [1e-12, 1e-100] {
        let x = f.quantile(p).unwrap();
        let rel = ((f.cdf(x) - p) / p).abs();
        assert!(
            rel < 1e-13,
            "F(1, 10): cdf(quantile({p:e})) = {:e} (rel {rel:e})",
            f.cdf(x)
        );
    }
}

/// Quantiles scale with the distribution: a stopping rule relative to `1 + x`
/// would stop early once `x ≪ 1`.
#[test]
fn quantile_scale_invariance() {
    let (g, g_small) = (
        Gamma::new(3.5, 2.0).unwrap(),
        Gamma::new(3.5, 2e12).unwrap(),
    );
    let ig = InverseGaussian::new(1.5, 2.0).unwrap();
    for p in [1e-6, 0.5, 0.9] {
        let (want, got) = (g.quantile(p).unwrap() / 1e12, g_small.quantile(p).unwrap());
        let rel = rel_err(got, want);
        assert!(
            rel < 1e-13,
            "gamma at {p:e}: {got:e} vs {want:e} (rel {rel:e})"
        );
        // Scales out to 1e±250, where `λ·(x − μ)` and `μ²` over- and
        // underflow in a naive log density.
        for s in [1e-12, 1e250, 1e-250] {
            let ig_s = InverseGaussian::new(1.5 * s, 2.0 * s).unwrap();
            for (name, want, got) in [
                ("ig", ig.quantile(p).unwrap() * s, ig_s.quantile(p).unwrap()),
                ("ig isf", ig.isf(p).unwrap() * s, ig_s.isf(p).unwrap()),
            ] {
                let rel = rel_err(got, want);
                assert!(
                    rel < 1e-13,
                    "{name} at scale {s:e}, {p:e}: {got:e} vs {want:e} (rel {rel:e})"
                );
            }
            // f_s(s·x) = f(x)/s.
            let (x, want) = (0.3 * s, ig.log_density(0.3) - s.ln());
            let got = ig_s.log_density(x);
            assert!(
                (got - want).abs() < 1e-12 * want.abs().max(1.0),
                "IG log_density at scale {s:e}: {got:e} vs {want:e}"
            );
        }
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

/// Upper tails and quantile edges at the points where `sf = 1 − cdf`, a
/// rebuilt `p = 1 − (1 − p)`, or a saturating quantile loses precision. Tail
/// truth from mpmath (60 digits, `scripts/accuracy_sweep.py`), `quantile(1)`
/// from `scipy.stats.<dist>.ppf(1)`.
#[test]
fn discrete_tails_and_quantile_edges() {
    let rel = |got: f64, want: f64| ((got - want) / want).abs();
    let h = Hypergeometric::new(10_000_000, 5_000_000, 100_000).unwrap();
    assert!(rel(h.sf(55_000), 2.239_882_270_665_583_6e-222) < 1e-14);
    let h = Hypergeometric::new(52_053, 11_609, 43_033).unwrap();
    assert!(rel(h.sf(9_880), 6.841_082_558_375_914e-16) < 1e-14);
    assert_eq!(h.quantile(1.0).unwrap(), 11_609);
    let b = Binomial::new(1_000_000_000_000_000, 1e-20).unwrap();
    assert!(rel(b.cdf(0), 0.999_990_000_049_999_8) < 1e-15);
    let nb = NegBinomial::from_mean_size(0.37, 2.5e13).unwrap();
    assert!(rel(nb.cdf(0), 0.690_734_330_637_356_6) < 1e-15);
    let be = Bernoulli::new(4.261_453_073_179_319_4e-62).unwrap();
    assert_eq!(be.log_mass(0), -4.261_453_073_179_319_4e-62);
    assert_eq!(be.sf(0), 4.261_453_073_179_319_4e-62);
    assert_eq!(Bernoulli::new(6.1e-52).unwrap().quantile(1.0).unwrap(), 1);
    let b = Binomial::new(35, 0.126_363_570_472_715_68).unwrap();
    assert_eq!(b.quantile(1.0).unwrap(), 35);
    let h = Hypergeometric::new(3_543, 2_298, 2_770).unwrap();
    assert_eq!(h.quantile(1.0).unwrap(), 2_298);
    // Quantiles past `i64::MAX` are errors, not a clamped `k`.
    assert!(Geometric::new(1e-18).unwrap().quantile(0.999_999).is_err());
    let nb = NegBinomial::new(7_565_433.3, 0.999_999_999_999_372_3).unwrap();
    assert!(nb.quantile(0.5).is_err());
    assert!(Poisson::new(1e19).unwrap().quantile(0.5).is_err());
    // `quantile(1)` on an unbounded support is +∞, so an error.
    assert!(Poisson::new(3.0).unwrap().quantile(1.0).is_err());
    assert!(NegBinomial::new(3.0, 0.4).unwrap().quantile(1.0).is_err());
    assert!(Geometric::new(0.3).unwrap().quantile(1.0).is_err());
    assert_eq!(Geometric::new(1.0).unwrap().quantile(1.0).unwrap(), 1);
    // All mass at 0: `cdf(0) = 1`, so the smallest `k` with `cdf(k) ≥ 1` is 0.
    assert_eq!(Binomial::new(10, 0.0).unwrap().quantile(1.0).unwrap(), 0);
}

/// `log_mass` at large parameters, where a log-gamma sum loses ≈ ε·n·ln n
/// (≈ 4e-9 at `Hypergeometric(10⁶, 5·10⁵, 1000)`, 5e-3 at `N = 10¹²`, 2 at
/// `λ = 10¹⁵`), plus tails, `p` near 0 and 1, and small cases. Truth from
/// mpmath (60 digits); asserts 1e-13·max(1, |truth|), measured ≤ 3e-15.
#[test]
fn log_mass_large_parameters() {
    let check = |name: &str, got: f64, want: f64| {
        let err = (got - want).abs() / want.abs().max(1.0);
        assert!(
            err < 1e-13,
            "{name}: got {got:e}, want {want:e}, err {err:e}"
        );
    };
    let bi = |n: i64, p: f64, k: i64| Binomial::new(n, p).unwrap().log_mass(k);
    check(
        "binom(1e12, .5, 5e11)",
        bi(1_000_000_000_000, 0.5, 500_000_000_000),
        -14.041301910609251,
    );
    check(
        "binom(1e15, .3, mode+460)",
        bi(1_000_000_000_000_000, 0.3, 300_000_000_000_460),
        -17.408002857031953,
    );
    check(
        "binom(1e12, .3, 2.9e11)",
        bi(1_000_000_000_000, 0.3, 290_000_000_000),
        -239640872.2857421,
    );
    check(
        "binom(1e6, .999, 999000)",
        bi(1_000_000, 0.999, 999_000),
        -4.372399255942922,
    );
    check(
        "binom(20, .999, 17)",
        bi(20, 0.999, 17),
        -13.701490801228788,
    );
    check("binom(60, 1e-10, 3)", bi(60, 1e-10, 3), -58.63698724807519);
    check(
        "binom(1e12, 1-1e-10, n-100)",
        bi(1_000_000_000_000, 0.9999999999, 999_999_999_900),
        -3.2223569567046955,
    );
    let po = |lam: f64, k: i64| Poisson::new(lam).unwrap().log_mass(k);
    check(
        "pois(1e9, 1e9)",
        po(1e9, 1_000_000_000),
        -11.280571451761212,
    );
    check(
        "pois(1e12, 1e12)",
        po(1e12, 1_000_000_000_000),
        -14.73444909116903,
    );
    check(
        "pois(1e15, λ+√λ)",
        po(1e15, 1_000_000_031_622_777),
        -18.688326753796805,
    );
    check("pois(1e-3, 7)", po(1e-3, 7), -56.88044831394037);
    check("pois(4.5, 0)", po(4.5, 0), -4.5);
    check("pois(1e6, 3e4)", po(1e6, 30_000), -864809.3364980419);
    let hy = |nn: i64, kk: i64, n: i64, k: i64| Hypergeometric::new(nn, kk, n).unwrap().log_mass(k);
    check(
        "hyper(1e6, 5e5, 1000, 500)",
        hy(1_000_000, 500_000, 1000, 500),
        -3.679418742177588,
    );
    check(
        "hyper(1e12, 5e11, 1e6, 5e5)",
        hy(1_000_000_000_000, 500_000_000_000, 1_000_000, 500_000),
        -7.133546381626615,
    );
    check(
        "hyper(1e15, 1e12, 1e9, 1002998)",
        hy(
            1_000_000_000_000_000,
            1_000_000_000_000,
            1_000_000_000,
            1_002_998,
        ),
        -12.321709118023213,
    );
    check(
        "hyper(1e12, 1e6, 5e11, 500010)",
        hy(1_000_000_000_000, 1_000_000, 500_000_000_000, 500_010),
        -7.133746381626628,
    );
    check(
        "hyper(30, 12, 10, 7)",
        hy(30, 12, 10, 7),
        -3.839231568222631,
    );
    let nb = |r: f64, p: f64, k: i64| NegBinomial::new(r, p).unwrap().log_mass(k);
    check(
        "nbinom(1e12, .5, 1e12)",
        nb(1e12, 0.5, 1_000_000_000_000),
        -15.081022681449044,
    );
    check(
        "nbinom(0.1, .99, 5000)",
        nb(0.1, 0.99, 5000),
        -60.63039181035487,
    );
    check("nbinom(4, .3, 2)", nb(4.0, 0.3, 2), -1.532060291412756);
    let nbm =
        |mu: f64, size: f64, k: i64| NegBinomial::from_mean_size(mu, size).unwrap().log_mass(k);
    check(
        "nbinom(mu 10, size 1e15, 11)",
        nbm(10.0, 1e15, 11),
        -2.1738718229393883,
    );
    check(
        "nbinom(mu 1e12, size 1e3, ·)",
        nbm(1e12, 1e3, 968_377_223_382),
        -25.574829190867625,
    );
    let g = Geometric::new(1e-12).unwrap();
    check(
        "geom(1e-12, 1e14)",
        g.log_mass(100_000_000_000_000),
        -127.63102111597755,
    );
}

/// `mass` where `|log_mass|` is large but the mass is well conditioned in the
/// parameters, so `exp(log_mass)` would carry `|log_mass|·ε` (58–110ε here).
/// Truth from mpmath (60 digits) at the exact f64 parameters: the pmfs
/// `C(n,k)·pᵏ(1−p)ⁿ⁻ᵏ`, `Γ(r+k)/(Γ(r)·k!)·(1−p)ʳpᵏ`, `e^{−λ}λᵏ/k!`,
/// `(1−p)ᵏ⁻¹p`.
#[test]
fn mass_large_log_mass() {
    for (name, got, want) in [
        (
            "Binomial(3526, 3.4e-60).mass(1)",
            Binomial::new(3526, 3.403_052_273_500_439_5e-60)
                .unwrap()
                .mass(1),
            1.199_916_231_636_255e-56,
        ),
        (
            "NegBinomial(1, 2.2e-56).mass(1)",
            NegBinomial::new(1.0, 2.231_283_980_989_190_8e-56)
                .unwrap()
                .mass(1),
            2.231_283_980_989_190_8e-56,
        ),
        (
            "Geometric(1e-300).mass(1)",
            Geometric::new(1e-300).unwrap().mass(1),
            1e-300,
        ),
        (
            "Poisson(1e-300).mass(1)",
            Poisson::new(1e-300).unwrap().mass(1),
            1e-300,
        ),
    ] {
        let rel = rel_err(got, want);
        assert!(
            rel <= 4.0 * f64::EPSILON,
            "{name} = {got:e}, want {want:e} (rel {rel:e})"
        );
    }
    // Masses far below the f64 range are +0: at `|E| ≈ 7e17` the exponent's
    // low part can pass −1 (−0 unclamped), and at `r = 1e300`, `k = 0` it is
    // ≈ 1e283, which `r·(1 + E_lo)` would overflow into ∞·0 = NaN.
    for (name, got) in [
        (
            "NegBinomial(1e300, 0.5).mass(0)",
            NegBinomial::new(1e300, 0.5).unwrap().mass(0),
        ),
        (
            "NegBinomial(3e200, 0.3).mass(0)",
            NegBinomial::new(3e200, 0.3).unwrap().mass(0),
        ),
        (
            "NegBinomial(1e-10, 0.5).mass(1e18)",
            NegBinomial::new(1e-10, 0.5)
                .unwrap()
                .mass(1_000_000_000_000_000_000),
        ),
    ] {
        assert!(
            got == 0.0 && got.is_sign_positive(),
            "{name} = {got:?}, want +0"
        );
    }
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
    let ig = InverseGaussian::new(1.5, 2.0).unwrap();
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
        (
            "inversegaussian",
            measure_band("dist_inversegaussian_ppf", |a| ig.quantile(a[0]).unwrap()),
            measure_band("dist_inversegaussian_isf", |a| ig.isf(a[0]).unwrap()),
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
    // sf evaluates the tail directly, avoiding 1-cdf cancellation in the far upper tail.
    assert!(n.sf(10.0) > 0.0 && n.sf(10.0) < 1e-20);
}

#[test]
fn studentt_basic() {
    let t = StudentT::new(5.0).unwrap();
    assert!((t.cdf(0.0) - 0.5).abs() < 1e-12);
    assert_eq!(t.quantile(0.5).unwrap(), 0.0);
    // scipy: t.ppf(0.975, 5) = 2.5705818366147395
    assert!((t.quantile(0.975).unwrap() - 2.570_581_836_614_74).abs() < 1e-9);
    // Deep upper tail via the direct betai form (mpmath, df=7, x=50); naively
    // taking `1 − cdf` would carry ~1e-6 relative error here.
    let t7 = StudentT::new(7.0).unwrap();
    let sf = t7.sf(50.0);
    assert!(((sf - 1.675_626_297_775_02e-10) / 1.675_626_297_775_02e-10).abs() < 1e-12);
    assert!((t7.sf(-50.0) - (1.0 - 1.675_626_297_775_02e-10)).abs() < 1e-15);
    assert_eq!(t7.sf(f64::INFINITY), 0.0);
    assert!(t7.cdf(f64::NAN).is_nan());
    assert!(t7.sf(f64::NAN).is_nan());
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
    // scipy: chi2(k).pdf(0) = inf for k < 2 (singular at 0), 0.5 at k = 2
    // (Exponential(rate ½)), 0 for k > 2.
    let c1 = ChiSquared::new(1.0).unwrap();
    assert_eq!(c1.density(0.0), f64::INFINITY);
    assert_eq!(c1.log_density(0.0), f64::INFINITY);
    assert_eq!(ChiSquared::new(2.0).unwrap().density(0.0), 0.5);
    assert_eq!(c.density(0.0), 0.0);
    assert_eq!(c.log_density(0.0), f64::NEG_INFINITY);
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
    assert!(f.cdf(f64::NAN).is_nan());
    assert!(f.sf(f64::NAN).is_nan());
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
    // `x − loc` overflows (2e308): mpmath `scale/(π(scale² + (x − loc)²))`.
    let wide = Cauchy::new(-1e308, 1e308).unwrap();
    let pdf = wide.density(1e308);
    assert!(((pdf - 6.366_197_723_675_83e-310) / 6.366_197_723_675_83e-310).abs() < 1e-12);
    assert!((wide.log_density(1e308) + 711.950_376_440_449_5).abs() < 1e-12);
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
    // scipy: weibull_min(c, scale=λ).logpdf(0) = −ln 2 at (1, 2), +∞ for c < 1, −∞ for c > 1.
    assert_eq!(w1.log_density(0.0), -core::f64::consts::LN_2);
    assert_eq!(
        Weibull::new(0.5, 1.0).unwrap().log_density(0.0),
        f64::INFINITY
    );
    assert_eq!(
        Weibull::new(2.0, 1.0).unwrap().log_density(0.0),
        f64::NEG_INFINITY
    );
    // Range edges outside the accuracy sweep, truth from mpmath at the exact
    // f64 inputs: `(−ln q)^{1/k}` overflows for k < 1 while `scale·` it does
    // not, and `x/scale` leaves the normal range while `(x/scale)^k` does not.
    let rel = |got: f64, want: f64| ((got - want) / want).abs();
    let w = Weibull::new(0.01, 1e-100).unwrap();
    assert!(rel(w.isf(1e-300).unwrap(), 8.584_091_572_646_376e183) < 1e-12);
    let w = Weibull::new(0.001, 1e-300).unwrap();
    assert!(rel(w.quantile(0.9).unwrap(), 1.643_193_466_516_863e62) < 1e-12);
    let w = Weibull::new(0.01, 1e10).unwrap();
    assert!(rel(w.cdf(1e-312), 6.023_780_835_041_652e-4) < 1e-13);
    let w = Weibull::new(0.005, 1e-10).unwrap();
    assert!(rel(w.sf(1e300), 3.896_281_218_624_703e-16) < 1e-13);
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
    // scipy: gamma(a=1, scale=½).pdf(0) = 2, logpdf(0) = ln 2.
    let g = Gamma::new(1.0, 2.0).unwrap();
    assert_eq!(g.density(0.0), 2.0);
    assert_eq!(g.log_density(0.0), core::f64::consts::LN_2);
    // scipy: gamma(a=0.5).pdf(0) = inf.
    assert_eq!(Gamma::new(0.5, 2.0).unwrap().density(0.0), f64::INFINITY);
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
    // One ulp below ½ at huge equal shapes: x = ½ − 2⁻⁵⁴ lies ~50 standard
    // deviations (1/(2√(2a + 1))) below the mean at a = 1e35, so the sf is 1
    // to f64 resolution and the cdf below the f64 range (normal limit).
    for a in [1e35, 1e90] {
        let d = Beta::new(a, a).unwrap();
        assert_eq!(d.sf(0.499_999_999_999_999_94), 1.0);
        assert_eq!(d.cdf(0.499_999_999_999_999_94), 0.0);
    }
    // Support ends next to a shape of exactly 1, where the density is the
    // finite limit (scipy: beta(1, 1).pdf(0) = 1, beta(1, 3).pdf(0) = 3).
    let near = |got: f64, want: f64| (got - want).abs() <= 4.0 * f64::EPSILON * want;
    let flat = Beta::new(1.0, 1.0).unwrap();
    assert!(near(flat.density(0.0), 1.0) && near(flat.density(1.0), 1.0));
    assert!(near(Beta::new(1.0, 3.0).unwrap().density(0.0), 3.0));
    assert!(near(Beta::new(3.0, 1.0).unwrap().density(1.0), 3.0));
    assert!(Beta::new(3.0, 1.0).unwrap().log_density(0.0) == f64::NEG_INFINITY);
}

#[test]
fn inversegaussian_basic() {
    let ig = InverseGaussian::new(1.5, 2.0).unwrap();
    assert_eq!(ig.mean(), Some(1.5));
    assert_eq!(ig.variance(), Some(1.5 * 1.5 * 1.5 / 2.0)); // μ³/λ
    assert_eq!(ig.density(0.0), 0.0);
    assert_eq!(ig.log_density(-1.0), f64::NEG_INFINITY);
    assert_eq!(ig.cdf(f64::INFINITY), 1.0);
    assert_eq!(ig.sf(f64::INFINITY), 0.0);
    assert_eq!(ig.quantile(0.0).unwrap(), 0.0);
    assert_eq!(ig.isf(0.0).unwrap(), f64::INFINITY);
    assert!(InverseGaussian::new(0.0, 2.0).is_err());
    assert!(InverseGaussian::new(1.5, f64::NAN).is_err());
    assert!(ig.cdf(f64::NAN).is_nan());
    assert!(ig.sf(f64::NAN).is_nan());
    // Far upper tail, where Newton on raw `sf` exhausts its iterations far
    // from the root. mpmath (60 dps): root of ln sf(x) = ln q.
    for (q, want) in [
        (1e-50, 243.985_082_797_044_84),
        (1e-100, 500.620_404_686_735_1),
        (1e-200, 1_016.322_095_252_721_8),
    ] {
        let x = ig.isf(q).unwrap();
        assert!(
            ((x - want) / want).abs() < 1e-13,
            "IG(1.5,2).isf({q:e}) = {x}, want {want}"
        );
    }
    // λ/μ ≥ 355: e^{2λ/μ} overflows and Φ(−b) underflows on their own, while
    // their product is O(1) near x = μ. mpmath (60 dps):
    // ncdf(a) + exp(2λ/μ)·ncdf(−b), a,b = √(λ/x)(x/μ ∓ 1).
    let cases = [
        (
            1.0,
            1e3,
            0.95,
            0.054_069_920_566_546_78,
            0.945_930_079_433_453_2,
        ),
        (
            1.0,
            1e3,
            1.0,
            0.506_306_255_528_466_6,
            0.493_693_744_471_533_3,
        ),
        (
            1.0,
            1e3,
            1.05,
            0.940_505_689_455_142_1,
            0.059_494_310_544_857_83,
        ),
        (
            1.0,
            1e5,
            1.0,
            0.500_630_781_553_559,
            0.499_369_218_446_440_94,
        ),
        (
            1.0,
            1e5,
            1.005,
            0.942_807_177_916_371_7,
            0.057_192_822_083_628_33,
        ),
    ];
    for (mu, lam, x, cdf, sf) in cases {
        let d = InverseGaussian::new(mu, lam).unwrap();
        let (c, s) = (d.cdf(x), d.sf(x));
        assert!(
            ((c - cdf) / cdf).abs() < 1e-13,
            "IG({mu},{lam}).cdf({x}) = {c:e}, want {cdf:e}"
        );
        assert!(
            ((s - sf) / sf).abs() < 1e-13,
            "IG({mu},{lam}).sf({x}) = {s:e}, want {sf:e}"
        );
        let q = d.quantile(cdf).unwrap();
        assert!(
            ((q - x) / x).abs() < 1e-13,
            "IG({mu},{lam}).quantile({cdf:e}) = {q:e}, want {x:e}"
        );
    }
}

#[test]
fn studentt_near_median() {
    // Quantiles within 1e-8..1e-15 of p = ½: solved on the central mass
    // `P(0 < X < x) = |p − ½|`, which a residual on the near-½ tail cannot
    // resolve. mpmath (80 dps): root x of
    // betainc(½, df/2, 0, x²/(df + x²), regularized=True)/2 = |p − ½|,
    // with p the f64 argument.
    for (df, p, want) in [
        (3.0, 0.5 + 1e-8, 2.720_699_060_022_185e-8),
        (3.0, 0.5 + 1e-12, 2.720_638_859_808_488_5e-12),
        (3.0, 0.5 - 1e-12, -2.720_638_859_808_488_5e-12),
        (1.0, 0.5 + 1e-12, 3.141_523_156_156_375e-12),
        (2.0, 0.5 - 1e-8, -2.828_427_123_257_431_5e-8),
        (7.0, 0.5 + 1e-15, 2.595_384_183_213_906e-15),
    ] {
        let t = StudentT::new(df).unwrap();
        let x = t.quantile(p).unwrap();
        let rel = rel_err(x, want);
        assert!(
            rel < 1e-14,
            "t({df}).quantile({p:e}) = {x:e}, want {want:e} (rel {rel:e})"
        );
        // isf(q) = −quantile(q).
        let x = t.isf(p).unwrap();
        let rel = rel_err(x, -want);
        assert!(
            rel < 1e-14,
            "t({df}).isf({p:e}) = {x:e}, want {:e} (rel {rel:e})",
            -want
        );
    }
    // cdf(x) − ½ for tiny x keeps full relative precision: the central form
    // `½·I_{x²/(df+x²)}(½, df/2)` rather than `1 − ½·I_{df/(df+x²)}(df/2, ½)`.
    // mpmath (80 dps): betainc(½, df/2, 0, x²/(df + x²), regularized=True)/2.
    for (df, x, central) in [
        (7.0, 1e-5, 3.849_914_508_249_342e-6),
        (7.0, 1e-10, 3.849_914_508_322_673e-11),
        (3.0, 1e-10, 3.675_525_969_478_613_8e-11),
    ] {
        let t = StudentT::new(df).unwrap();
        // `cdf − ½` is exact in f64 (Sterbenz), so this is cdf's own error.
        let (c, s) = (t.cdf(x) - 0.5, t.sf(-x) - 0.5);
        for (name, got) in [("cdf(x) − ½", c), ("sf(−x) − ½", s)] {
            assert!(
                (got - central).abs() < 1.2e-16,
                "t({df}): {name} = {got:e}, want {central:e} at x = {x:e}"
            );
        }
    }
}

#[test]
fn studentt_deep_tail_isf() {
    // The tail is a power law, `sf ∝ x^{−df}`; Newton on the raw sf stalls
    // decades below these roots. mpmath (80 dps): root x of
    // betainc(df/2, ½, 0, df/(df + x²), regularized=True)/2 = q.
    for (df, q, want) in [
        (3.0, 1e-50, 4.795_275_720_469_223e16),
        (7.0, 1e-50, 2.791_391_448_336_578_6e7),
        (0.5, 1e-50, 1.028_491_156_316_34e99),
        // Subnormal q: the far-tail leading term keeps full precision.
        (2.0, 1e-320, 7.071_107_172_647_215e159),
        // True root 3.183e319 exceeds f64::MAX.
        (1.0, 1e-320, f64::INFINITY),
    ] {
        let t = StudentT::new(df).unwrap();
        let x = t.isf(q).unwrap();
        let rel = rel_err(x, want);
        assert!(
            rel < 1e-13,
            "t({df}).isf({q:e}) = {x:e}, want {want:e} (rel {rel:e})"
        );
        assert_eq!(t.quantile(q).unwrap(), -x, "t({df}).quantile({q:e})");
    }
}

#[test]
fn fisherf_deep_upper_tail_isf() {
    // The upper tail is a power law, `sf ∝ x^{−dfd/2}`; Newton on the raw sf
    // grows x by ~(1 + 2/dfd) per step and stalls decades below these roots.
    // mpmath (80 dps): root x of
    // betainc(dfd/2, dfn/2, 0, dfd/(dfd + dfn·x), regularized=True) = q.
    for (dfn, dfd, q, want, tol) in [
        (1.0, 10.0, 1e-100, 7.554_750_338_964_087e20, 1e-13),
        (3.0, 10.0, 1e-100, 4.067_966_793_383_849e20, 1e-13),
        (1.0, 0.5, 1e-30, 1.692_470_493_793_475e119, 1e-13),
        (3.0, 1e6, 1e-200, 309.282_243_628_885_1, 1e-13),
        // Subnormal q: the far-tail leading term keeps full precision.
        (3.0, 7.0, 1e-320, 7.969_169_013_835_32e91, 1e-13),
        // True root 1.617e999 exceeds f64::MAX.
        (50.0, 0.2, 1e-100, f64::INFINITY, 0.0),
    ] {
        let f = FisherF::new(dfn, dfd).unwrap();
        let x = f.isf(q).unwrap();
        let rel = rel_err(x, want);
        assert!(
            rel <= tol,
            "F({dfn}, {dfd}).isf({q:e}) = {x:e}, want {want:e} (rel {rel:e})"
        );
    }
}

#[test]
fn gamma_family_deep_upper_tail_isf() {
    // In the exponential tail `sf/pdf → 1/β`, so Newton on the raw sf moves x
    // by ~1/β per step and exhausts its iterations short of these roots.
    // mpmath (80 dps): root x of gammainc(α, β·x, inf, regularized=True) = q,
    // with χ²(k) as α = k/2, β = ½.
    for (k, q, want, tol) in [
        (1.0, 1e-300, 1_373.872_631_222_394_1, 1e-13),
        (3.0, 1e-300, 1_388.336_773_854_685_8, 1e-13),
        (100.0, 1e-100, 752.877_564_727_371_8, 1e-13),
        (0.2, 1e-100, 446.269_581_339_238_4, 1e-13),
        (3.0, 1e-320, 1_480.504_386_712_137_2, 1e-13),
    ] {
        let c = ChiSquared::new(k).unwrap();
        let x = c.isf(q).unwrap();
        let rel = rel_err(x, want);
        assert!(
            rel < tol,
            "chi2({k}).isf({q:e}) = {x:e}, want {want:e} (rel {rel:e})"
        );
    }
    for (shape, rate, q, want, tol) in [
        (2.5, 1.5, 1e-300, 466.880_195_217_675_6, 1e-13),
        (0.1, 1.0, 1e-100, 223.134_790_669_619_2, 1e-13),
        (50.0, 2.0, 1e-100, 188.219_391_181_842_94, 1e-13),
        (2.5, 1.5, 1e-320, 497.645_063_414_246_1, 1e-13),
    ] {
        let g = Gamma::new(shape, rate).unwrap();
        let x = g.isf(q).unwrap();
        let rel = rel_err(x, want);
        assert!(
            rel < tol,
            "Gamma({shape},{rate}).isf({q:e}) = {x:e}, want {want:e} (rel {rel:e})"
        );
    }
}

#[test]
fn inversegaussian_sf_far_tail() {
    // Far above the mean (and anywhere when λ/x is tiny) the two terms of
    // `Φ(−a) − e^{2λ/μ}Φ(−b)` agree to many digits. mpmath (80 dps):
    // ncdf(−a) − exp(2λ/μ)·ncdf(−b), a,b = √(λ/x)(x/μ ∓ 1).
    for (mu, lam, x, want, tol) in [
        (3.0, 3e-8, 3e11, 1.792_258_963_843_412e-230, 1e-12),
        (3.0, 1e-8, 9.28e11, 1.001_540_245_188_323_8e-237, 1e-12),
        (1.0, 1e-8, 0.5, 1.128_279_177_437_988_4e-4, 1e-13),
        (1.0, 1e-4, 1e6, 1.495_061_549_524_409_7e-29, 1e-13),
        (1.0, 0.01, 5.623e4, 4.754_824_052_258_475e-129, 1e-12),
        (1.0, 1.333, 1000.0, 2.883_390_506_582_948e-294, 1e-12),
        (1.5, 2.0, 60.0, 2.536_655_534_835_458_4e-14, 1e-13),
    ] {
        let d = InverseGaussian::new(mu, lam).unwrap();
        let s = d.sf(x);
        let rel = rel_err(s, want);
        assert!(
            rel < tol,
            "IG({mu},{lam}).sf({x:e}) = {s:e}, want {want:e} (rel {rel:e})"
        );
    }
    // isf in the same regime, which needs the series form of the sf.
    // mpmath (80 dps): root of ln sf(x) = ln q.
    for (mu, lam, q, want) in [
        (3.0, 1e-8, 1e-237, 9.280_027_622_870_783e11),
        (1.5, 2.0, 1e-296, 1_512.342_174_990_797_7),
        (1.0, 1e-4, 1e-100, 4_234_853.455_948_873),
    ] {
        let d = InverseGaussian::new(mu, lam).unwrap();
        let x = d.isf(q).unwrap();
        let rel = rel_err(x, want);
        assert!(
            rel < 1e-13,
            "IG({mu},{lam}).isf({q:e}) = {x:e}, want {want:e} (rel {rel:e})"
        );
    }
}

#[test]
fn inversegaussian_isf_subnormal_target() {
    // Targets in the subnormal range, where `ln` of the subnormal sf keeps
    // only its remaining bits (11 at 1e-320, one at 5e-324) and moved the
    // root by up to 2e-4; the residual is `ln sf` in log form instead. Both
    // branches of the sf: the `erfcx_drop` series (far above the mean) and
    // the direct difference (IG(1, 1000), root at 3.1μ). mpmath (80 dps):
    // root x of ln(ncdf(−a) − exp(2λ/μ)·ncdf(−b)) = ln q,
    // a,b = √(λ/x)(x/μ ∓ 1), at the f64 target.
    for (mu, lam, q, want) in [
        (1.0, 1.0, 1e-320, 1_453.353_207_301_504_9),
        (1.0, 1.0, 5e-324, 1_468.547_717_408_359_4),
        (0.1, 10.0, 5e-324, 1.669_391_624_436_161_5),
        (0.1, 10.0, 1e-320, 1.654_138_361_698_073_4),
        (10.0, 0.5, 1e-320, 289_095.740_889_310_4),
        (1.0, 1000.0, 1e-320, 3.145_114_903_191_708_5),
        (1.0, 1e-4, 1e-322, 14_421_907.098_713_502),
        (1.5, 2.0, 1e-315, 1_610.565_707_871_03),
    ] {
        let x = InverseGaussian::new(mu, lam).unwrap().isf(q).unwrap();
        let rel = rel_err(x, want);
        assert!(
            rel < 1e-14,
            "IG({mu},{lam}).isf({q:e}) = {x:e}, want {want:e} (rel {rel:e})"
        );
    }
}

#[test]
fn studentt_far_tail_cdf() {
    // Far tail through `betai`'s `pow` prefactor; the exponential of the
    // tail's log carries `|ln tail|·ε` ≈ 1e-13 here. mpmath (80 dps; 420 dps
    // for the last four): betainc(df/2, ½, 0, df/(df + x²), regularized=True)/2
    // at the exact f64 inputs.
    for (df, x, want) in [
        (
            6.364_579_744_547_293,
            -8.603_397_272_219_453e36,
            4.636_713_642_169_218e-234,
        ),
        (
            10.0,
            -1.409_189_240_017_328_3e19,
            3.984_532_781_319_698e-188,
        ),
        (0.5, -1e20, 3.207_009_754_142_229e-11),
        (3.0, -1e10, 1.102_657_790_843_584e-30),
        (20.0, -40.0, 7.287_348_277_155_382e-21),
    ] {
        let t = StudentT::new(df).unwrap();
        for (name, got) in [("cdf(x)", t.cdf(x)), ("sf(−x)", t.sf(-x))] {
            let rel = rel_err(got, want);
            assert!(
                rel < 2e-15,
                "t({df}): {name} = {got:e} at x = {x:e}, want {want:e} (rel {rel:e})"
            );
        }
    }
}

#[test]
fn quantile_from_a_far_seed() {
    // Seeds far from the root, where an unbracketed Newton step overshoots:
    // left to where `sf ≈ 1` and the pdf is tiny, so the next step is scaled
    // by e^{100} and more, or right from a Wilson–Hilferty seed on a convex
    // cdf, to where the pdf underflows. mpmath (80 dps), root of
    // ln cdf(x) = ln p (quantile) or ln sf(x) = ln q (isf), with
    // `ncdf(a) + exp(2λ/μ)·ncdf(−b)` for the inverse Gaussian,
    // `gammainc(α, 0, β·x, regularized=True)` for χ²(k) = Gamma(k/2, ½), and
    // `betainc(dfn/2, dfd/2, 0, dfn·x/(dfd + dfn·x), regularized=True)` for F.
    let ig = |mu, lam| InverseGaussian::new(mu, lam).unwrap();
    let chi2 = |k| ChiSquared::new(k).unwrap();
    let gamma = |shape, rate| Gamma::new(shape, rate).unwrap();
    for (name, got, want, tol) in [
        (
            "IG(1, 0.2141).quantile(0.5229)",
            ig(1.0, 0.214_116_385_443_299_56)
                .quantile(0.522_941_277_923_908_8)
                .unwrap(),
            0.342_529_463_394_331_7,
            1e-13,
        ),
        (
            "IG(1, 0.01).isf(0.4661)",
            ig(1.0, 0.01).isf(0.4661).unwrap(),
            0.025_185_302_605_733_32,
            1e-13,
        ),
        (
            "IG(5.05e9, 2.22e6).quantile(0.3306)",
            ig(5_047_896_608.750_66, 2_217_993.207_577_607_6)
                .quantile(0.330_639_472_100_096_5)
                .unwrap(),
            2_342_224.130_774_897,
            1e-13,
        ),
        (
            "chi2(10).quantile(1e-10)",
            chi2(10.0).quantile(1e-10).unwrap(),
            0.052_331_065_631_905_41,
            1e-13,
        ),
        (
            "chi2(8).quantile(1e-8)",
            chi2(8.0).quantile(1e-8).unwrap(),
            0.044_464_473_804_344_58,
            1e-13,
        ),
        (
            "Gamma(4, 1).quantile(1e-8)",
            gamma(4.0, 1.0).quantile(1e-8).unwrap(),
            0.022_232_236_902_172_29,
            1e-13,
        ),
        // Subnormal p: the residual is formed by `ln_gammp`, not `ln` of the
        // subnormal cdf.
        (
            "Gamma(167.13, 0.001698).quantile(8.48e-319)",
            gamma(167.131_937_810_957_23, 0.001_698_018_214_829_785)
                .quantile(8.47827e-319)
                .unwrap(),
            464.281_507_770_697_6,
            1e-13,
        ),
        (
            "F(4.658, 270.35).isf(0.99999321)",
            FisherF::new(4.657_910_808_899_677, 270.351_318_892_045_2)
                .unwrap()
                .isf(0.999_993_208_375_889_8)
                .unwrap(),
            0.004_005_016_983_706_388,
            1e-13,
        ),
        // Seed μ = 1 + ulp: the open-bracket jump to 1 is a 2e-16 step there,
        // which must not count as convergence.
        (
            "IG(1, 0.01).quantile(0.5)",
            ig(1.0, 0.01).quantile(0.5).unwrap(),
            0.021_480_463_915_113_43,
            1e-13,
        ),
        // Shape ~8500 near the median, where a fixed 200-term cap on the
        // `gammp` series makes the cdf jump at `x = a + 1`.
        (
            "Gamma(8547.98, 0.4366).quantile(0.4944)",
            gamma(8_547.975_048_175_393, 0.436_554_704_228_919_4)
                .quantile(0.494_375_134_658_852_86)
                .unwrap(),
            19_576.786_412_114_575,
            1e-14,
        ),
    ] {
        let rel = rel_err(got, want);
        assert!(rel < tol, "{name} = {got:e}, want {want:e} (rel {rel:e})");
    }
}

#[test]
fn fisherf_small_df_cdf_sf() {
    // `dfd/(dfd + dfn·x)` rounds to 1 when `dfn·x ≪ dfd` (and `dfn·x/(dfd +
    // dfn·x)` when `dfn·x ≫ dfd`), so only the side with the small `betai`
    // argument keeps the mass. mpmath (80 dps): betainc(dfn/2, dfd/2, 0, y) with
    // y = dfn·x/(dfd + dfn·x), its complement from betainc(dfd/2, dfn/2, 0, 1 − y).
    let f = FisherF::new(0.01, 10.0).unwrap();
    let s = f.sf(4.780_051_509_317_061e-36);
    let rel = rel_err(s, 0.35);
    assert!(
        rel < 1e-13,
        "F(0.01, 10).sf(4.78e-36) = {s:e} (rel {rel:e})"
    );
    let c = FisherF::new(10.0, 0.01).unwrap().cdf(2.09e35);
    let rel = rel_err(c, 0.349_996_848_452_525_94);
    assert!(
        rel < 1e-13,
        "F(10, 0.01).cdf(2.09e35) = {c:e} (rel {rel:e})"
    );
    // Quantiles in that regime. The lower tail `cdf ∝ x^{dfn/2}` scales the
    // relative error of cdf by 2/dfn in x (~3e-14 measured at dfn = 0.028).
    for (dfn, dfd, p, want, tol) in [
        (0.01, 10.0, 0.65, 4.780_051_509_317_061e-36, 1e-11),
        (0.02, 10.0, 0.65, 1.226_308_950_480_960_7e-17, 1e-11),
        (
            0.027_707_343_446_009_405,
            1_457.136_944_921_265_5,
            0.568_001_459_034_916_2,
            7.606_098_723_684_63e-17,
            1e-13,
        ),
    ] {
        let x = FisherF::new(dfn, dfd).unwrap().quantile(p).unwrap();
        let rel = rel_err(x, want);
        assert!(
            rel < tol,
            "F({dfn}, {dfd}).quantile({p}) = {x:e}, want {want:e} (rel {rel:e})"
        );
    }
}

#[test]
fn fisherf_isf_small_numerator_df() {
    // At the root `dfn·x ≪ dfd`, where `dfd/(dfd + dfn·x)` sits within 1e-4 of
    // 1; the cdf side keeps the mass. mpmath (80 dps): root x of
    // betainc(dfd/2, dfn/2, 0, dfd/(dfd + dfn·x), regularized=True) = q.
    let f = FisherF::new(0.1, 10.0).unwrap();
    let want = 0.010_290_585_698_260_792;
    for (name, got) in [
        ("isf(0.3)", f.isf(0.3).unwrap()),
        ("quantile(0.7)", f.quantile(0.7).unwrap()),
    ] {
        let rel = rel_err(got, want);
        assert!(
            rel < 1e-13,
            "F(0.1, 10).{name} = {got:e}, want {want:e} (rel {rel:e})"
        );
    }
}

#[test]
fn fisherf_tiny_df_far_tail() {
    // The leading-term tail `ln I = a·ln y − ln(a·B(a, b))` at `a = df/2 ≪ 1`,
    // where `ln I` is `O(a)` and the mass on the other side is `−expm1(ln I)`.
    // `ln(a·B)` formed as `ln a + ln B` rounds to 0 there (sf −0, ln sf −∞).
    // b = 2.5 takes the shift to 8 before `algdiv`, b = 50 `algdiv` directly;
    // F(5, 1e-20) is the mirror through `ln_sf`; at b = 5e299 `algdiv`'s
    // `a/b` is subnormal. The tolerance covers the rounding of the three logs
    // forming `ln y` (~700·ε absolute each, a few ε relative in `a·ln y`).
    // mpmath (800 dps, unchanged at 1400): betainc(dfn/2, dfd/2, 0, y) at the
    // exact f64 inputs, y = dfn·x/(dfd + dfn·x), and its complement.
    let f = |dfn, dfd| FisherF::new(dfn, dfd).unwrap();
    for (name, got, want) in [
        (
            "F(1e-20, 5).sf(1e-300)",
            f(1e-20, 5.0).sf(1e-300),
            3.685_781_476_824_91e-18,
        ),
        (
            "F(1e-20, 5).ln_sf(1e-300)",
            f(1e-20, 5.0).ln_sf(1e-300),
            -40.142_049_101_040_435,
        ),
        (
            "F(1e-300, 5).sf(1e-300)",
            f(1e-300, 5.0).sf(1e-300),
            6.909_400_607_016_574e-298,
        ),
        (
            "F(1e-20, 100).sf(1e-300)",
            f(1e-20, 100.0).sf(1e-300),
            3.684_765_973_028_766_6e-18,
        ),
        (
            "F(5, 1e-20).cdf(1e300)",
            f(5.0, 1e-20).cdf(1e300),
            3.685_781_476_824_91e-18,
        ),
        (
            "F(1e-20, 1e300).sf(1)",
            f(1e-20, 1e300).sf(1.0),
            2.308_381_668_776_966_3e-19,
        ),
    ] {
        let rel = rel_err(got, want);
        assert!(rel < 1e-14, "{name} = {got:e}, want {want:e} (rel {rel:e})");
    }
}

#[test]
fn fisherf_subnormal_beta_argument() {
    // `y = dfn·x/(dfd + dfn·x)` (or `z = dfd/(dfd + dfn·x)`) subnormal or
    // underflowed, where the kernel's `y^{dfn/2}` would carry the lost bits
    // (6e-5 at F(0.01, 10).cdf(1e-320), 0 for a cdf of 1.7e-162 at
    // F(1, 10).cdf(5e-324)). F(0.01, 1e-20) at 1e-320 has a subnormal `dfn·x`
    // but a normal `y`; F(2, 1e20) at 1e-300 a normal `dfn·x` but a subnormal
    // `y`. mpmath (80 dps): betainc(dfn/2, dfd/2, 0, y) and
    // betainc(dfd/2, dfn/2, 0, z), regularized, with y and z formed exactly
    // from the f64 inputs; the other side as 1 minus it.
    let f = |dfn, dfd| FisherF::new(dfn, dfd).unwrap();
    for (name, got, want, tol) in [
        (
            "F(0.01, 10).cdf(1e-320)",
            f(0.01, 10.0).cdf(1e-320),
            0.024_519_757_436_939_554,
            1e-14,
        ),
        (
            "F(0.01, 10).cdf(5e-324)",
            f(0.01, 10.0).cdf(5e-324),
            0.023_603_973_426_974_418,
            1e-14,
        ),
        (
            "F(1, 10).cdf(5e-324)",
            f(1.0, 10.0).cdf(5e-324),
            1.729_788_129_916_989e-162,
            1e-13,
        ),
        (
            "F(0.5, 3).cdf(1e-320)",
            f(0.5, 3.0).cdf(1e-320),
            7.310_381_019_864_344e-81,
            1e-13,
        ),
        (
            "F(0.01, 1e-20).cdf(1e-320)",
            f(0.01, 1e-20).cdf(1e-320),
            3.090_295_260_494_165e-20,
            1e-14,
        ),
        (
            "F(2, 1e20).cdf(1e-300)",
            f(2.0, 1e20).cdf(1e-300),
            1e-300,
            1e-13,
        ),
        (
            "F(0.001, 0.001).sf(1e-320)",
            f(0.001, 0.001).sf(1e-320),
            0.654_084_374_317_850_2,
            1e-14,
        ),
        (
            "F(1e10, 0.02).sf(1.7e308)",
            f(1e10, 0.02).sf(1.7e308),
            7.946_333_355_492_124e-4,
            1e-13,
        ),
        (
            "F(1e20, 1).sf(1e303)",
            f(1e20, 1.0).sf(1e303),
            2.523_132_522_020_16e-152,
            1e-13,
        ),
        (
            "F(10, 0.001).sf(1e308)",
            f(10.0, 0.001).sf(1e308),
            0.698_959_984_098_350_5,
            1e-14,
        ),
    ] {
        let rel = rel_err(got, want);
        assert!(rel < tol, "{name} = {got:e}, want {want:e} (rel {rel:e})");
    }
    // The quantile inverts the same log form: at the f64 cdf of a subnormal
    // `x` it returns `x` to within one subnormal step (the root moves by
    // 2/dfn times the relative error of p, far below that step).
    for (dfn, dfd, x) in [
        (0.01, 10.0, 1e-320),
        (0.01, 10.0, 5e-324),
        (0.5, 3.0, 1e-320),
        (1.0, 10.0, 1e-322),
    ] {
        let d = f(dfn, dfd);
        let got = d.quantile(d.cdf(x)).unwrap();
        assert!(
            (got - x).abs() <= 5e-324,
            "F({dfn}, {dfd}).quantile(cdf({x:e})) = {got:e}"
        );
    }
}

#[test]
fn fisherf_large_df() {
    // log density at large df, where `a·ln(dfn/dfd) + (a − 1)·ln x −
    // (a + b)·ln(1 + dfn·x/dfd) − ln B` cancels terms of size `dfn·ln` (2e5
    // absolute at dfn = 1e20, 1e-5 at 1e10). mpmath (120 dps): that sum with
    // betaln, from the f64 inputs.
    for (dfn, dfd, x, want) in [
        (1e20, 1.0, 0.5, -0.879_217_762_364_754_8),
        (1e20, 15.0, 1.0, 0.077_408_417_308_252_72),
        (1e20, 1e20, 1.0, 21.413_765_216_175_84),
        (1e10, 100.0, 0.99999, 1.035_416_317_612_383_8),
        (1e4, 1e4, 1.1, -8.452_993_380_276_56),
        (15.0, 1e20, 1.1, -0.053_075_413_963_635_83),
        (17.0, 1e20, 0.1, -9.478_093_065_038_038),
        (1e10, 1e20, 0.1, -7_012_925_452.217_73),
        (1e4, 3.0, 1.00001, -0.771_180_082_730_484_6),
        (1e20, 0.001, 1.1, -7.700_226_888_854_865),
    ] {
        let got = FisherF::new(dfn, dfd).unwrap().log_density(x);
        let err = (got - want).abs() / want.abs().max(1.0);
        assert!(
            err < 2e-14,
            "F({dfn}, {dfd}).log_density({x}) = {got}, want {want} (err {err:e})"
        );
    }
    // Tails at large df. `F(1e4, 1e20).sf(1.1)`: the kernel reads its side
    // from the argument near 1 unless handed the small one. mpmath (120
    // dps): Σ_{j<a} C(a+b−1, j)·y^j·z^{a+b−1−j} (integer a, b). The root of
    // `F(1e20, 1).sf(x) = 8e-151` from betainc(½, 5e19, 0, z) by bisection.
    let f = |dfn, dfd| FisherF::new(dfn, dfd).unwrap();
    for (name, got, want, tol) in [
        (
            "F(1e4, 1e20).sf(1.1)",
            f(1e4, 1e20).sf(1.1),
            3.618_329_558_096_339_4e-12,
            1e-12,
        ),
        (
            "F(1e20, 1).isf(8e-151)",
            f(1e20, 1.0).isf(8e-151).unwrap(),
            9.947_183_943_243_46e299,
            // `ln z = ln dfd − ln dfn − ln x` carries ε·|ln x| (~8e-14),
            // doubled in x by the tail exponent dfd/2 = ½.
            5e-13,
        ),
    ] {
        let rel = rel_err(got, want);
        assert!(rel < tol, "{name} = {got:e}, want {want:e} (rel {rel:e})");
    }
    // Both masses stay in [0, 1] where the near-1 side is evaluated (the sf
    // was −2e-202 at F(1e20, 1).sf(0.001), true value 1 − ~1e-217).
    let s = f(1e20, 1.0).sf(0.001);
    assert!(
        (0.0..=1.0).contains(&s) && s > 0.999,
        "F(1e20, 1).sf(0.001) = {s:e}"
    );
    // Round trips through the Newton solver, which needs the density and
    // both log tails: quantile(cdf(x)) and isf(sf(x)) return x.
    for (dfn, dfd, x) in [
        (1e20, 1.0, 1e300),
        (1e20, 1.0, 6.75),
        (1e20, 1e20, 1.000_000_000_1),
        (1e4, 1e20, 1.2),
        (1e20, 15.0, 3.0),
        (15.0, 1e20, 2.0),
        (1e10, 3.0, 30.0),
    ] {
        let d = f(dfn, dfd);
        let (c, s) = (d.cdf(x), d.sf(x));
        let got = if s <= c {
            d.isf(s).unwrap()
        } else {
            d.quantile(c).unwrap()
        };
        let rel = rel_err(got, x);
        assert!(
            rel < 1e-12,
            "F({dfn}, {dfd}) round trip at {x:e}: {got:e} (rel {rel:e})"
        );
    }
}

#[test]
fn lower_tail_quantile_below_representable() {
    // True roots below the smallest subnormal are 0 in f64. F(0.1, 10) at
    // 1e-30: `cdf ≈ y^{dfn/2}/((dfn/2)·B)` gives 1.29e-599. Gamma(0.1, ·) at
    // 1e-100: `P(α, y) ≈ y^α/Γ(α + 1)` gives `β·x` ≈ 1e-1000. χ²(0.01) at
    // 1e-50: `x/2` ≈ 1e-10000.
    for (name, got) in [
        (
            "F(0.1, 10).quantile(1e-30)",
            FisherF::new(0.1, 10.0).unwrap().quantile(1e-30).unwrap(),
        ),
        (
            "F(1, 10).quantile(1e-320)",
            FisherF::new(1.0, 10.0).unwrap().quantile(1e-320).unwrap(),
        ),
        (
            "Gamma(0.1, 5e-226).quantile(1e-100)",
            Gamma::new(0.1, 5e-226).unwrap().quantile(1e-100).unwrap(),
        ),
        (
            "chi2(0.01).quantile(1e-50)",
            ChiSquared::new(0.01).unwrap().quantile(1e-50).unwrap(),
        ),
    ] {
        assert_eq!(got, 0.0, "{name}");
    }
    // Roots that are representable although the cdf (inverse Gaussian) or
    // `β·x` (Gamma) is subnormal there. mpmath (80 dps): root x of
    // ln(ncdf(a) + exp(2λ/μ)·ncdf(−b)) = ln p, and for shape ½ the closed form
    // `P(½, y) = erf(√y)`, so `x = erfinv(p)²/β`.
    let ig = InverseGaussian::new(1.5, 2.0).unwrap();
    for (name, got, want) in [
        (
            "IG(1.5, 2).quantile(1e-320)",
            ig.quantile(1e-320).unwrap(),
            0.001_361_864_439_184_727_3,
        ),
        (
            "IG(1.5, 2).quantile(1e-310)",
            ig.quantile(1e-310).unwrap(),
            0.001_405_921_055_829_320_2,
        ),
        (
            "Gamma(0.5, 5e-226).quantile(1e-160)",
            Gamma::new(0.5, 5e-226).unwrap().quantile(1e-160).unwrap(),
            1.570_796_326_794_896_6e-95,
        ),
    ] {
        let rel = rel_err(got, want);
        assert!(rel < 1e-13, "{name} = {got:e}, want {want:e} (rel {rel:e})");
    }
}

#[test]
fn subnormal_target_quantile() {
    // Target masses at and below the smallest normal f64, down to the smallest
    // subnormal (`5e-324`). The Newton residual is `ln mass(x) − ln target`,
    // with `ln mass` from `ln_betai`/`ln_gammp`/`ln_gammq`, never `ln` of a
    // subnormal mass, which keeps ~12 significant bits at 1e-320. The rest is
    // `|ln target|·ε` (~8e-14 absolute) divided by the tail's exponent in
    // `ln x` (25 for the F(50, 50) lower tail). mpmath (60 dps): root x of
    // `betainc(df/2, ½, 0, df/(df + x²), regularized=True)/2 = q` (t),
    // `betainc(dfd/2, dfn/2, 0, dfd/(dfd + dfn·x))` = q or
    // `betainc(dfn/2, dfd/2, 0, dfn·x/(dfd + dfn·x))` = p (F), and
    // `gammainc(α, β·x, inf)` = q or `gammainc(α, 0, β·x)` = p (Gamma, χ²(k) as
    // α = k/2, β = ½), all regularized, at the f64 target.
    let t = |df| StudentT::new(df).unwrap();
    let f = |dfn, dfd| FisherF::new(dfn, dfd).unwrap();
    let chi2 = ChiSquared::new(300.0).unwrap();
    let gamma = Gamma::new(167.0, 2.0).unwrap();
    #[rustfmt::skip]
    let cases: &[(&str, f64, f64, f64)] = &[
        // (name, target, got, want)
        ("t(1000).isf", 1e-300, t(1000.0).isf(1e-300).unwrap(), 54.291_388_553_051_746),
        ("t(1000).isf", 1e-310, t(1000.0).isf(1e-310).unwrap(), 55.977_986_263_897_1),
        ("t(1000).isf", 1e-320, t(1000.0).isf(1e-320).unwrap(), 57.691_255_805_989_65),
        ("t(1000).isf", 5e-324, t(1000.0).isf(5e-324).unwrap(), 58.263_765_237_171_185),
        ("t(1e5).isf", 1e-300, t(1e5).isf(1e-300).unwrap(), 37.174_670_665_466_216),
        ("t(1e5).isf", 1e-310, t(1e5).isf(1e-310).unwrap(), 37.797_114_137_188_636),
        ("t(1e5).isf", 1e-320, t(1e5).isf(1e-320).unwrap(), 38.409_765_718_346_24),
        ("t(1e5).isf", 5e-324, t(1e5).isf(5e-324).unwrap(), 38.610_246_931_049_6),
        ("F(3, 1e6).isf", 1e-300, f(3.0, 1e6).isf(1e-300).unwrap(), 463.100_088_151_646_7),
        ("F(3, 1e6).isf", 1e-310, f(3.0, 1e6).isf(1e-310).unwrap(), 478.483_213_418_893_6),
        ("F(3, 1e6).isf", 1e-320, f(3.0, 1e6).isf(1e-320).unwrap(), 493.866_711_042_122_7),
        ("F(3, 1e6).isf", 5e-324, f(3.0, 1e6).isf(5e-324).unwrap(), 498.952_900_658_804_5),
        ("F(50, 50).quantile", 1e-300, f(50.0, 50.0).quantile(1e-300).unwrap(), 2.805_238_942_909_166e-13),
        ("F(50, 50).quantile", 1e-310, f(50.0, 50.0).quantile(1e-310).unwrap(), 1.116_785_738_287_689e-13),
        ("F(50, 50).quantile", 1e-320, f(50.0, 50.0).quantile(1e-320).unwrap(), 4.446_002_123_968_89e-14),
        ("F(50, 50).quantile", 5e-324, f(50.0, 50.0).quantile(5e-324).unwrap(), 3.278_847_707_721_965e-14),
        ("F(50, 50).isf", 1e-300, f(50.0, 50.0).isf(1e-300).unwrap(), 3_564_758_725_910.715),
        ("F(50, 50).isf", 1e-310, f(50.0, 50.0).isf(1e-310).unwrap(), 8_954_269_075_223.412),
        ("F(50, 50).isf", 1e-320, f(50.0, 50.0).isf(1e-320).unwrap(), 22_492_117_010_221.14),
        ("F(50, 50).isf", 5e-324, f(50.0, 50.0).isf(5e-324).unwrap(), 30_498_519_270_807.086),
        ("chi2(300).quantile", 1e-300, chi2.quantile(1e-300).unwrap(), 1.133_371_940_679_761_5),
        ("chi2(300).quantile", 1e-310, chi2.quantile(1e-310).unwrap(), 0.971_567_795_567_615_9),
        ("chi2(300).quantile", 1e-320, chi2.quantile(1e-320).unwrap(), 0.832_927_195_457_165_6),
        ("chi2(300).quantile", 5e-324, chi2.quantile(5e-324).unwrap(), 0.791_600_773_638_386_2),
        ("chi2(300).isf", 1e-300, chi2.isf(1e-300).unwrap(), 2_279.274_731_893_038_3),
        ("chi2(300).isf", 1e-310, chi2.isf(1e-310).unwrap(), 2_332.154_313_710_249_4),
        ("chi2(300).isf", 1e-320, chi2.isf(1e-320).unwrap(), 2_384.859_184_773_119_5),
        ("chi2(300).isf", 5e-324, chi2.isf(5e-324).unwrap(), 2_402.247_681_333_656),
        ("Gamma(167, 2).quantile", 1e-300, gamma.quantile(1e-300).unwrap(), 0.504_240_435_209_895_7),
        ("Gamma(167, 2).quantile", 1e-310, gamma.quantile(1e-310).unwrap(), 0.438_954_901_904_506_37),
        ("Gamma(167, 2).quantile", 1e-320, gamma.quantile(1e-320).unwrap(), 0.382_160_685_612_86),
        ("Gamma(167, 2).quantile", 5e-324, gamma.quantile(5e-324).unwrap(), 0.365_056_352_084_206_95),
        ("Gamma(167, 2).isf", 1e-300, gamma.isf(1e-300).unwrap(), 589.432_357_504_195_4),
        ("Gamma(167, 2).isf", 1e-310, gamma.isf(1e-310).unwrap(), 602.805_553_465_139_3),
        ("Gamma(167, 2).isf", 1e-320, gamma.isf(1e-320).unwrap(), 616.131_639_260_629_5),
        ("Gamma(167, 2).isf", 5e-324, gamma.isf(5e-324).unwrap(), 620.527_586_833_255_4),
    ];
    for &(name, target, got, want) in cases {
        let rel = rel_err(got, want);
        assert!(
            rel < 1e-14,
            "{name}({target:e}) = {got:e}, want {want:e} (rel {rel:e})"
        );
    }
    // t's quantile is its isf mirrored.
    for q in [1e-300, 1e-320, 5e-324] {
        let t = t(1000.0);
        assert_eq!(
            t.quantile(q).unwrap(),
            -t.isf(q).unwrap(),
            "t(1000).quantile({q:e})"
        );
    }
}

#[test]
fn gamma_quantile_tiny_rate() {
    // `β·x` far from 1 on both sides of the rate: the upper tail is solved on
    // `Gamma(α, 1)` and scaled back. mpmath (80 dps): root x of
    // gammainc(α, β·x, inf, regularized=True) = q.
    let g = Gamma::new(0.031_480_414_179_677_82, 3.261_475_293_744_904_7e-279).unwrap();
    let (got, want) = (
        g.isf(1.909_807_671_449_669_9e-115).unwrap(),
        7.828_892_278_890_288e280,
    );
    let rel = rel_err(got, want);
    assert!(rel < 1e-13, "isf = {got:e}, want {want:e} (rel {rel:e})");
}

#[test]
fn log_density_far_tail() {
    // `x²/df` (t) and `dfn·x/dfd` (F) overflow here, while the log density is
    // finite. mpmath (80 dps): the log density with log1p of the kernel.
    for (name, got, want) in [
        (
            "t(10).log_density(1e160)",
            StudentT::new(10.0).unwrap().log_density(1e160),
            -4_040.829_443_010_204,
        ),
        (
            "t(10).log_density(−1e200)",
            StudentT::new(10.0).unwrap().log_density(-1e200),
            -5_053.966_883_927_584,
        ),
        (
            "t(3).log_density(1e300)",
            StudentT::new(3.0).unwrap().log_density(1e300),
            -2_761.905_775_865_142,
        ),
        (
            "F(0.5, 1e-3).log_density(1e307)",
            FisherF::new(0.5, 1e-3).unwrap().log_density(1e307),
            -714.852_903_302_063,
        ),
    ] {
        let rel = rel_err(got, want);
        assert!(rel < 1e-14, "{name} = {got:e}, want {want:e} (rel {rel:e})");
    }
}

#[test]
fn density_at_infinity() {
    // Every density is 0 at ±∞ and its log −∞, including where the log
    // density's terms would meet as ∞ − ∞ (χ², Gamma, F, Weibull).
    let dists: [(&str, Box<dyn ContinuousDensity>); 14] = [
        ("Normal", Box::new(Normal::new(0.5, 2.0).unwrap())),
        ("StudentT", Box::new(StudentT::new(7.0).unwrap())),
        ("ChiSquared", Box::new(ChiSquared::new(5.0).unwrap())),
        ("ChiSquared(2)", Box::new(ChiSquared::new(2.0).unwrap())),
        ("FisherF", Box::new(FisherF::new(5.0, 10.0).unwrap())),
        ("Uniform", Box::new(Uniform::new(-1.0, 2.0).unwrap())),
        ("Exponential", Box::new(Exponential::new(2.0).unwrap())),
        ("Cauchy", Box::new(Cauchy::new(0.0, 1.0).unwrap())),
        ("Weibull", Box::new(Weibull::new(1.5, 2.0).unwrap())),
        ("LogNormal", Box::new(LogNormal::new(0.0, 1.0).unwrap())),
        ("Gamma", Box::new(Gamma::new(3.5, 2.0).unwrap())),
        ("Gamma(1)", Box::new(Gamma::new(1.0, 2.0).unwrap())),
        ("Beta", Box::new(Beta::new(2.0, 3.0).unwrap())),
        (
            "InverseGaussian",
            Box::new(InverseGaussian::new(1.5, 2.0).unwrap()),
        ),
    ];
    for (name, d) in &dists {
        for x in [f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(d.density(x), 0.0, "{name}.density({x})");
            assert_eq!(
                d.log_density(x),
                f64::NEG_INFINITY,
                "{name}.log_density({x})"
            );
        }
    }
}

#[test]
fn density_nan() {
    // A NaN `x` must come out NaN, not get read by a range check as "out of
    // support" (`(0..=1).contains(&NaN)` and `x < a || x > b` are both false
    // for NaN) and returned as a fixed in-support or out-of-support constant.
    let dists: [(&str, Box<dyn ContinuousDensity>); 12] = [
        ("Normal", Box::new(Normal::new(0.5, 2.0).unwrap())),
        ("StudentT", Box::new(StudentT::new(7.0).unwrap())),
        ("ChiSquared", Box::new(ChiSquared::new(5.0).unwrap())),
        ("FisherF", Box::new(FisherF::new(5.0, 10.0).unwrap())),
        ("Uniform", Box::new(Uniform::new(-1.0, 2.0).unwrap())),
        ("Exponential", Box::new(Exponential::new(2.0).unwrap())),
        ("Cauchy", Box::new(Cauchy::new(0.0, 1.0).unwrap())),
        ("Weibull", Box::new(Weibull::new(1.5, 2.0).unwrap())),
        ("LogNormal", Box::new(LogNormal::new(0.0, 1.0).unwrap())),
        ("Gamma", Box::new(Gamma::new(3.5, 2.0).unwrap())),
        ("Beta", Box::new(Beta::new(2.0, 3.0).unwrap())),
        (
            "InverseGaussian",
            Box::new(InverseGaussian::new(1.5, 2.0).unwrap()),
        ),
    ];
    for (name, d) in &dists {
        assert!(d.density(f64::NAN).is_nan(), "{name}.density(NaN)");
        assert!(d.log_density(f64::NAN).is_nan(), "{name}.log_density(NaN)");
    }
}

#[test]
fn studentt_large_df_quantile() {
    // Where the first omitted Cornish–Fisher term is above f64 resolution
    // (df = 1.5e5, q = 1e-300: the seed alone is 6e-9 off) Newton refines the
    // seed; where it is below (df = 1e10) the seed is returned. mpmath
    // (80 dps): root x of betainc(df/2, ½, 0, df/(df + x²), regularized=True)/2 = q.
    for (df, q, want, tol) in [
        (1.5e5, 1e-300, 37.132_064_638_150_547, 1e-14),
        (1e10, 0.3, 0.524_400_512_724_756, 1e-15),
        (1e10, 1e-5, 4.264_890_795_968_831, 1e-15),
    ] {
        let t = StudentT::new(df).unwrap();
        let x = t.isf(q).unwrap();
        let rel = rel_err(x, want);
        assert!(
            rel < tol,
            "t({df:e}).isf({q:e}) = {x:e}, want {want:e} (rel {rel:e})"
        );
        assert_eq!(t.quantile(q).unwrap(), -x, "t({df:e}).quantile({q:e})");
    }
}

#[test]
fn studentt_cdf_sf_large_df_grid() {
    // `halves` passes both `y = x²/(df + x²)` and `z = df/(df + x²)`, each
    // formed by its own division, to the incomplete-beta kernel, so at large
    // df, where `z` is within `x²/df` of 1, the tail is never recovered from
    // `1 − z`. At |x| = 30 and df ≥ 1e3 the tail's condition number in `y` is
    // `≈ (df/2)·y`, 240–450, so the rounding of `y` alone accounts for up to
    // ~5e-14. mpmath (80 dps) at the f64 `x`: `sf(x) = ½·I_z(df/2, ½)` for
    // x ≥ 0, `cdf(x) = 1 − sf(x)`, and `sf(x) = cdf(−x)`.
    #[rustfmt::skip]
    let cases: &[(f64, f64, f64, f64, f64)] = &[
        (1.0, 0.01, 0.503_182_992_764_908_3, 0.496_817_007_235_091_76, 1e-14),
        (1.0, -0.01, 0.496_817_007_235_091_76, 0.503_182_992_764_908_3, 1e-14),
        (1.0, 0.5, 0.647_583_617_650_433_3, 0.352_416_382_349_566_74, 1e-14),
        (1.0, -0.5, 0.352_416_382_349_566_74, 0.647_583_617_650_433_3, 1e-14),
        (1.0, 2.0, 0.852_416_382_349_566_7, 0.147_583_617_650_433_26, 1e-14),
        (1.0, -2.0, 0.147_583_617_650_433_26, 0.852_416_382_349_566_7, 1e-14),
        (1.0, 3.0, 0.897_583_617_650_433_3, 0.102_416_382_349_566_72, 1e-14),
        (1.0, -3.0, 0.102_416_382_349_566_72, 0.897_583_617_650_433_3, 1e-14),
        (1.0, 30.0, 0.989_393_597_594_464_5, 0.010_606_402_405_535_424, 1e-14),
        (1.0, -30.0, 0.010_606_402_405_535_424, 0.989_393_597_594_464_5, 1e-14),
        (2.5, 0.01, 0.503_618_002_821_045_9, 0.496_381_997_178_954_15, 1e-14),
        (2.5, -0.01, 0.496_381_997_178_954_15, 0.503_618_002_821_045_9, 1e-14),
        (2.5, 0.5, 0.671_151_040_065_142_7, 0.328_848_959_934_857_35, 1e-14),
        (2.5, -0.5, 0.328_848_959_934_857_35, 0.671_151_040_065_142_7, 1e-14),
        (2.5, 2.0, 0.921_304_252_121_017, 0.078_695_747_878_983, 1e-14),
        (2.5, -2.0, 0.078_695_747_878_983, 0.921_304_252_121_017, 1e-14),
        (2.5, 3.0, 0.963_711_952_225_484_1, 0.036_288_047_774_515_92, 1e-14),
        (2.5, -3.0, 0.036_288_047_774_515_92, 0.963_711_952_225_484_1, 1e-14),
        (2.5, 30.0, 0.999_854_467_645_421_3, 0.000_145_532_354_578_697_7, 1e-14),
        (2.5, -30.0, 0.000_145_532_354_578_697_7, 0.999_854_467_645_421_3, 1e-14),
        (30.0, 0.01, 0.503_956_253_713_411_1, 0.496_043_746_286_588_87, 1e-14),
        (30.0, -0.01, 0.496_043_746_286_588_87, 0.503_956_253_713_411_1, 1e-14),
        (30.0, 0.5, 0.689_638_497_557_436_3, 0.310_361_502_442_563_66, 1e-14),
        (30.0, -0.5, 0.310_361_502_442_563_66, 0.689_638_497_557_436_3, 1e-14),
        (30.0, 2.0, 0.972_687_477_518_508_5, 0.027_312_522_481_491_55, 1e-14),
        (30.0, -2.0, 0.027_312_522_481_491_55, 0.972_687_477_518_508_5, 1e-14),
        (30.0, 3.0, 0.997_305_017_967_174_1, 0.002_694_982_032_825_973, 1e-14),
        (30.0, -3.0, 0.002_694_982_032_825_973, 0.997_305_017_967_174_1, 1e-14),
        (30.0, 30.0, 1.0, 3.125_895_815_304_444e-24, 1e-14),
        (30.0, -30.0, 3.125_895_815_304_444e-24, 1.0, 1e-14),
        (1e3, 0.01, 0.503_988_359_033_906_4, 0.496_011_640_966_093_6, 1e-14),
        (1e3, -0.01, 0.496_011_640_966_093_6, 0.503_988_359_033_906_4, 1e-14),
        (1e3, 0.5, 0.691_407_459_583_062_6, 0.308_592_540_416_937_4, 1e-14),
        (1e3, -0.5, 0.308_592_540_416_937_4, 0.691_407_459_583_062_6, 1e-14),
        (1e3, 2.0, 0.977_114_826_753_374_1, 0.022_885_173_246_625_82, 1e-14),
        (1e3, -2.0, 0.022_885_173_246_625_82, 0.977_114_826_753_374_1, 1e-14),
        (1e3, 3.0, 0.998_616_645_477_880_9, 0.001_383_354_522_119_096_3, 1e-14),
        (1e3, -3.0, 0.001_383_354_522_119_096_3, 0.998_616_645_477_880_9, 1e-14),
        (1e3, 30.0, 1.0, 7.687_343_722_021_741e-142, 1e-13),
        (1e3, -30.0, 7.687_343_722_021_741e-142, 1.0, 1e-13),
        (1e5, 0.01, 0.503_989_346_340_588_5, 0.496_010_653_659_411_6, 1e-14),
        (1e5, -0.01, 0.496_010_653_659_411_6, 0.503_989_346_340_588_5, 1e-14),
        (1e5, 0.5, 0.691_461_911_172_791, 0.308_538_088_827_209, 1e-14),
        (1e5, -0.5, 0.308_538_088_827_209, 0.691_461_911_172_791, 1e-14),
        (1e5, 2.0, 0.977_248_518_271_246_8, 0.022_751_481_728_753_232, 1e-14),
        (1e5, -2.0, 0.022_751_481_728_753_232, 0.977_248_518_271_246_8, 1e-14),
        (1e5, 3.0, 0.998_649_769_557_967_7, 0.001_350_230_442_032_359_5, 1e-14),
        (1e5, -3.0, 0.001_350_230_442_032_359_5, 0.998_649_769_557_967_7, 1e-14),
        (1e5, 30.0, 1.0, 3.689_268_436_111_116_6e-197, 1e-13),
        (1e5, -30.0, 3.689_268_436_111_116_6e-197, 1.0, 1e-13),
        (1e8, 0.01, 0.503_989_356_304_657_6, 0.496_010_643_695_342_5, 1e-14),
        (1e8, -0.01, 0.496_010_643_695_342_5, 0.503_989_356_304_657_6, 1e-14),
        (1e8, 0.5, 0.691_462_460_723_911, 0.308_537_539_276_088_96, 1e-14),
        (1e8, -0.5, 0.308_537_539_276_088_96, 0.691_462_460_723_911, 1e-14),
        (1e8, 2.0, 0.977_249_866_702_046_6, 0.022_750_133_297_953_376, 1e-14),
        (1e8, -2.0, 0.022_750_133_297_953_376, 0.977_249_866_702_046_6, 1e-14),
        (1e8, 3.0, 0.998_650_101_635_981_2, 0.001_349_898_364_018_747_2, 1e-14),
        (1e8, -3.0, 0.001_349_898_364_018_747_2, 0.998_650_101_635_981_2, 1e-14),
        (1e8, 30.0, 1.0, 4.916_682_142_941_633e-198, 1e-13),
        (1e8, -30.0, 4.916_682_142_941_633e-198, 1.0, 1e-13),
    ];
    for &(df, x, cdf_want, sf_want, tol) in cases {
        let t = StudentT::new(df).unwrap();
        let (cdf, sf) = (t.cdf(x), t.sf(x));
        let cdf_rel = rel_err(cdf, cdf_want);
        let sf_rel = rel_err(sf, sf_want);
        assert!(
            cdf_rel < tol,
            "t({df:e}).cdf({x}) = {cdf:e}, want {cdf_want:e} (rel {cdf_rel:e})"
        );
        assert!(
            sf_rel < tol,
            "t({df:e}).sf({x}) = {sf:e}, want {sf_want:e} (rel {sf_rel:e})"
        );
    }
}

#[test]
fn studentt_log_density_huge_df() {
    // `lgamma((df+1)/2) − lgamma(df/2)` cancels near the ε of each term at
    // huge df (about 1e-5 absolute error at df = 1e10); `lbeta` keeps the
    // difference to double precision instead of forming the two large
    // `lgamma`s. mpmath (50 dps): `−ln√df − ln B(½, df/2) −
    // (df+1)/2·ln(1+x²/df)`.
    let cases: &[(f64, f64, f64)] = &[
        (1e10, 0.5, -1.043_938_533_240_610_1),
        (1e10, 2.0, -2.918_938_533_029_672_7),
        (1e10, -3.0, -5.418_938_531_654_673),
        (1e15, 0.5, -1.043_938_533_204_673_1),
        (1e15, 2.0, -2.918_938_533_204_671),
        (1e15, -3.0, -5.418_938_533_204_657),
    ];
    for &(df, x, want) in cases {
        let got = StudentT::new(df).unwrap().log_density(x);
        let rel = rel_err(got, want);
        assert!(
            rel < 1e-13,
            "t({df:e}).log_density({x}) = {got:e}, want {want:e} (rel {rel:e})"
        );
    }
    // `density` still agrees (mpmath, 50 dps: exp of the log density above).
    let got = StudentT::new(1e10).unwrap().density(0.5);
    let want = 0.352_065_326_751_647_1;
    let rel = rel_err(got, want);
    assert!(
        rel < 1e-13,
        "t(1e10).density(0.5) = {got:e}, want {want:e} (rel {rel:e})"
    );
}

#[test]
fn fisherf_quantile_seed_large_shapes() {
    // The Wilson–Hilferty seed goes negative at deep enough `p`, falling back
    // to the leading series term `y^a/(a·B(a,b)) → p`, whose `ln B(a, b)` is
    // `lbeta(a, b)` rather than `lgamma(a) + lgamma(b) − lgamma(a + b)` (the
    // three-`lgamma` form need not cancel exactly at the shapes where this
    // fallback is reached). Round-trip through `cdf`, as in
    // `lower_tail_quantile_below_representable`.
    for (dfn, dfd, p) in [(10_000.0, 10_000.0, 1e-300), (300.0, 300.0, 1e-200)] {
        let f = FisherF::new(dfn, dfd).unwrap();
        let x = f.quantile(p).unwrap();
        let back = f.cdf(x);
        let rel = ((back - p) / p).abs();
        assert!(
            rel < 1e-12,
            "F({dfn}, {dfd}).quantile({p:e}) = {x:e}, cdf back = {back:e} (rel {rel:e})"
        );
    }
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
    // μ ≫ θ puts p next to 1; the mean must not go through a rounded 1 − p.
    let nb = NegBinomial::from_mean_size(1e9, 0.01).unwrap();
    assert!((nb.mean().unwrap() / 1e9 - 1.0).abs() < 1e-15);
    assert!((nb.variance().unwrap() / (1e9 + 1e20) - 1.0).abs() < 1e-15);
    // μ + θ overflows while p = μ/(μ + θ) = ½ does not.
    let nb = NegBinomial::from_mean_size(1e308, 1e308).unwrap();
    assert!((nb.mean().unwrap() / 1e308 - 1.0).abs() < 1e-15);
}

#[test]
fn negbinomial_heavy_tail_quantile() {
    // The moment-based starting bracket is 418 here, but P(X > 418) ≈ 3.6e-4, so
    // the search must grow the bracket for any p above ~0.9996.
    // scipy: nbinom(0.1, 0.01).ppf(1 - 1e-6) = 941.0 (scipy's p is 1 − our p).
    let nb = NegBinomial::new(0.1, 0.99).unwrap();
    assert_eq!(nb.quantile(1.0 - 1e-6).unwrap(), 941);
    // p = 1 on an unbounded support is +∞, which no i64 holds.
    assert!(nb.quantile(1.0).is_err());
    assert!(Poisson::new(4.5).unwrap().quantile(1.0).is_err());
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

/// Draws lie in the support, and the sample mean and variance sit within 5
/// standard errors of the distribution's. SE of the mean is `σ/√n`; SE of the
/// sample variance is `√((m₄ − s⁴)/n)` with `m₄` the sample 4th central moment.
#[cfg(all(feature = "dist", feature = "rng"))]
fn check_draws(name: &str, d: &impl Distribution, draws: &[f64]) {
    let n = draws.len() as f64;
    let (lo, hi) = (d.support_min().as_f64(), d.support_max().as_f64());
    for &x in draws {
        assert!(x >= lo && x <= hi, "{name}: draw {x} outside [{lo}, {hi}]");
    }
    let m = draws.iter().sum::<f64>() / n;
    let (mut m2, mut m4) = (0.0, 0.0);
    for &x in draws {
        let c = (x - m) * (x - m);
        m2 += c;
        m4 += c * c;
    }
    let (s2, m4) = (m2 / (n - 1.0), m4 / n);
    let (mu, var) = (d.mean().unwrap(), d.variance().unwrap());
    let se_mean = (var / n).sqrt();
    assert!(
        (m - mu).abs() <= 5.0 * se_mean,
        "{name}: sample mean {m} vs {mu} (5·SE = {:e})",
        5.0 * se_mean
    );
    let se_var = ((m4 - s2 * s2) / n).sqrt();
    assert!(
        (s2 - var).abs() <= 5.0 * se_var,
        "{name}: sample variance {s2} vs {var} (5·SE = {:e})",
        5.0 * se_var
    );
}

/// Pearson χ² goodness of fit of integer draws against `mass`. Cells run left
/// to right from the support minimum and close once their expected count is
/// ≥ 5; the upper tail past the last closed cell is one cell with expected
/// `n·sf`, merged into its neighbour if still < 5. p-value from
/// `ChiSquared::sf` on `cells − 1` df; asserts p > 0.001.
#[cfg(all(feature = "dist", feature = "rng"))]
fn check_chi2_fit(name: &str, d: &(impl DiscreteMass + DiscreteCdf), draws: &[i64]) {
    let n = draws.len() as f64;
    let lo = d.support_min().as_f64() as i64;
    let hi = *draws.iter().max().unwrap();
    let mut counts = vec![0.0; (hi - lo + 1) as usize];
    for &k in draws {
        counts[(k - lo) as usize] += 1.0;
    }
    let mut cells: Vec<(f64, f64)> = Vec::new();
    let (mut o, mut e, mut k) = (0.0, 0.0, lo);
    loop {
        o += counts.get((k - lo) as usize).copied().unwrap_or(0.0);
        e += n * d.mass(k);
        let rest = n * d.sf(k);
        if rest < 5.0 {
            o += counts.iter().skip((k + 1 - lo) as usize).sum::<f64>();
            e += rest;
            match cells.last_mut() {
                Some(last) if e < 5.0 => *last = (last.0 + o, last.1 + e),
                _ => cells.push((o, e)),
            }
            break;
        }
        if e >= 5.0 {
            cells.push((o, e));
            (o, e) = (0.0, 0.0);
        }
        k += 1;
    }
    let stat: f64 = cells.iter().map(|&(o, e)| (o - e) * (o - e) / e).sum();
    let df = (cells.len() - 1) as f64;
    let p = ChiSquared::new(df).unwrap().sf(stat);
    println!("{name}: chi2 = {stat:.3}, df = {df}, p = {p:.4}");
    assert!(p > 0.001, "{name}: chi2 = {stat}, df = {df}, p = {p:e}");
}

/// One-sample Kolmogorov–Smirnov test of draws against `cdf`. p-value from the
/// asymptotic Kolmogorov distribution `Q(z) = 2 Σ_{k≥1} (−1)^{k−1} e^{−2k²z²}`
/// at `z = √n·D` (no finite-n correction; the series is 1 to double precision
/// below `z = 0.2`, where 100 terms would not converge). Asserts p > 0.001.
#[cfg(all(feature = "dist", feature = "rng"))]
fn check_ks_fit(name: &str, d: &impl ContinuousCdf, draws: &[f64]) {
    let mut xs = draws.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len() as f64;
    let mut dmax: f64 = 0.0;
    for (i, &x) in xs.iter().enumerate() {
        let f = d.cdf(x);
        dmax = dmax.max(f - i as f64 / n).max((i + 1) as f64 / n - f);
    }
    let z = n.sqrt() * dmax;
    let p = if z < 0.2 {
        1.0
    } else {
        let s: f64 = (1..=100)
            .map(|k: i32| {
                let t = (-2.0 * f64::from(k * k) * z * z).exp();
                if k % 2 == 1 { t } else { -t }
            })
            .sum();
        (2.0 * s).clamp(0.0, 1.0)
    };
    println!("{name}: KS D = {dmax:.5}, p = {p:.4}");
    assert!(p > 0.001, "{name}: KS D = {dmax}, p = {p:e}");
}

/// Every sampler at `n = 10⁵` from one fixed stream: draws in support, sample
/// moments within 5 SE, and goodness of fit (χ² for discrete, KS for the
/// inverse Gaussian and Gamma). Overridden samplers run on both sides of their
/// switch and at it: Poisson λ = 10, Binomial `n·min(p, 1−p)` = 10, Gamma
/// shape 1.
#[cfg(all(feature = "dist", feature = "rng"))]
#[test]
fn sampler_moments_and_fit() {
    use commonstats::dist::{DiscreteSampler, Sampler};
    use commonstats::rng::CommonStatsRng;
    const N: usize = 100_000;
    let mut rng = CommonStatsRng::new(2024, 0);
    fn disc(name: &str, d: &(impl DiscreteSampler + DiscreteMass), rng: &mut CommonStatsRng) {
        let draws: Vec<i64> = (0..N).map(|_| d.sample(rng)).collect();
        let xs: Vec<f64> = draws.iter().map(|&k| k as f64).collect();
        check_draws(name, d, &xs);
        check_chi2_fit(name, d, &draws);
    }
    disc("bernoulli(0.3)", &Bernoulli::new(0.3).unwrap(), &mut rng);
    for (n, p) in [
        (20, 0.35),
        (999, 0.01),
        (100, 0.1),
        (1000, 0.3),
        (40, 0.8),
        (200, 0.7),
    ] {
        let bi = Binomial::new(n, p).unwrap();
        disc(&format!("binomial({n}, {p})"), &bi, &mut rng);
    }
    for lam in [0.5, 4.5, 9.99, 10.0, 50.0] {
        disc(
            &format!("poisson({lam})"),
            &Poisson::new(lam).unwrap(),
            &mut rng,
        );
    }
    disc("geometric(0.4)", &Geometric::new(0.4).unwrap(), &mut rng);
    for (r, p) in [(4.0, 0.3), (0.3, 0.8)] {
        let nb = NegBinomial::new(r, p).unwrap();
        disc(&format!("negbinomial({r}, {p})"), &nb, &mut rng);
    }
    for (mu, size) in [(3.0, 0.5), (40.0, 2.0)] {
        let nb = NegBinomial::from_mean_size(mu, size).unwrap();
        disc(&format!("negbinomial(mu {mu}, size {size})"), &nb, &mut rng);
    }
    let hy = Hypergeometric::new(30, 12, 10).unwrap();
    disc("hypergeometric(30, 12, 10)", &hy, &mut rng);
    for (mu, lam) in [(1.5, 2.0), (2.0, 0.2), (1.0, 100.0)] {
        let ig = InverseGaussian::new(mu, lam).unwrap();
        let draws: Vec<f64> = (0..N).map(|_| ig.sample(&mut rng)).collect();
        let name = format!("inversegaussian({mu}, {lam})");
        check_draws(&name, &ig, &draws);
        check_ks_fit(&name, &ig, &draws);
    }
    for (shape, rate) in [(0.3, 2.0), (1.0, 0.5), (3.5, 2.0)] {
        let g = Gamma::new(shape, rate).unwrap();
        let draws: Vec<f64> = (0..N).map(|_| g.sample(&mut rng)).collect();
        let name = format!("gamma({shape}, {rate})");
        check_draws(&name, &g, &draws);
        check_ks_fit(&name, &g, &draws);
    }
}

/// Known-answer draws: the first eight draws of each overridden sampler from a
/// fresh `CommonStatsRng::new(1, 0)`. The inverse-Gaussian and Gamma floats
/// compare bit-exact: both are IEEE `+ − × ÷` plus `libm` and the crate's own
/// `erfc_inv` (pure Rust, no platform intrinsics), so the bits are the same on
/// every target.
#[cfg(all(feature = "dist", feature = "rng"))]
#[test]
fn sampler_known_answers() {
    use commonstats::dist::{DiscreteSampler, Sampler};
    use commonstats::rng::CommonStatsRng;
    fn first8(d: &impl DiscreteSampler) -> [i64; 8] {
        let mut rng = CommonStatsRng::new(1, 0);
        core::array::from_fn(|_| d.sample(&mut rng))
    }
    assert_eq!(
        first8(&Bernoulli::new(0.3).unwrap()),
        [0, 1, 1, 0, 0, 1, 0, 0]
    );
    // λ = 4.5 by inversion, λ = 50 by PTRS.
    assert_eq!(
        first8(&Poisson::new(4.5).unwrap()),
        [5, 3, 3, 4, 7, 2, 3, 5]
    );
    assert_eq!(
        first8(&Poisson::new(50.0).unwrap()),
        [52, 45, 58, 46, 44, 53, 49, 56]
    );
    // n·p = 7 by inversion, n·p = 300 by BTRS.
    assert_eq!(
        first8(&Binomial::new(20, 0.35).unwrap()),
        [8, 5, 6, 7, 9, 4, 6, 8]
    );
    assert_eq!(
        first8(&Binomial::new(1000, 0.3).unwrap()),
        [305, 290, 316, 292, 288, 306, 298, 313]
    );
    assert_eq!(
        first8(&NegBinomial::new(4.0, 0.3).unwrap()),
        [1, 0, 0, 2, 2, 2, 0, 0]
    );
    // Variance ≥ 10⁶: the position uniform is `uniform52()`. The NegBinomial
    // mixture's `λ ≈ 10⁹` reaches PTRS through that path too.
    assert_eq!(
        first8(&Poisson::new(1e12).unwrap()),
        [
            1000000313490,
            999999961249,
            999999437009,
            999999727966,
            999999858652,
            1000000837469,
            999999187894,
            999999207949
        ]
    );
    assert_eq!(
        first8(&Binomial::new(1_000_000_000_000, 0.3).unwrap()),
        [
            300000143640,
            299999982244,
            299999742048,
            299999875355,
            299999935234,
            300000383694,
            299999627925,
            299999637112
        ]
    );
    assert_eq!(
        first8(&NegBinomial::from_mean_size(1e9, 100.0).unwrap()),
        [
            1024769813, 879581016, 1035508258, 1072886950, 906317552, 1013802622, 900481771,
            932532968
        ]
    );
    assert_eq!(
        first8(&Hypergeometric::new(100_000_000, 50_000_000, 50_000_000).unwrap()),
        [
            25002149, 25000901, 24996517, 25000992, 24999264, 24997666, 24998507, 25003124
        ]
    );
    // p < 10⁻³: `uniform52()` (Geometric inversion, exact rare Bernoulli
    // outcome, Poisson inversion). Single Bernoulli and Poisson draws are
    // almost all 0, so their pins are `Σ i·xᵢ` over draws i = 1..10⁵, which
    // fixes where the nonzero draws fall.
    assert_eq!(
        first8(&Geometric::new(1e-6).unwrap()),
        [
            941394, 330437, 1810936, 368664, 263503, 1048596, 597269, 1553593
        ]
    );
    let weighted = |d: &dyn Fn(&mut CommonStatsRng) -> i64| {
        let mut rng = CommonStatsRng::new(1, 0);
        (1..=100_000).map(|i| i * d(&mut rng)).sum::<i64>()
    };
    let (be, po) = (Bernoulli::new(1e-4).unwrap(), Poisson::new(1e-4).unwrap());
    assert_eq!(weighted(&|r| be.sample(r)), 270408);
    assert_eq!(weighted(&|r| po.sample(r)), 611368);
    let ig = InverseGaussian::new(1.5, 2.0).unwrap();
    let mut rng = CommonStatsRng::new(1, 0);
    let got: [f64; 8] = core::array::from_fn(|_| ig.sample(&mut rng));
    let want = [
        1.178616598016224,
        0.9133692004793699,
        0.6571974115996995,
        2.306278508394099,
        0.8029248337199364,
        2.0889122205660042,
        1.6735048887283095,
        2.962903941839156,
    ];
    assert_eq!(got.map(f64::to_bits), want.map(f64::to_bits), "{got:?}");
    // Shape 0.3 < 1 runs both Marsaglia–Tsang branches (the boost wraps the
    // shape-1.3 rejection loop).
    let g = Gamma::new(0.3, 2.0).unwrap();
    let mut rng = CommonStatsRng::new(1, 0);
    let got: [f64; 8] = core::array::from_fn(|_| g.sample(&mut rng));
    let want = [
        0.031921716420553005,
        0.10323412028053225,
        0.014071671077969385,
        0.03408486423612188,
        0.13607599647399984,
        0.15055120226328897,
        0.0020810761851904848,
        0.003870204487374349,
    ];
    assert_eq!(got.map(f64::to_bits), want.map(f64::to_bits), "{got:?}");
}

/// RNG words one draw consumes, for 200 seeds: the RNG's next word located
/// in a fresh copy of the stream.
#[cfg(all(feature = "dist", feature = "rng"))]
fn words_per_draw(d: &impl commonstats::dist::DiscreteSampler) -> Vec<usize> {
    use commonstats::rng::CommonStatsRng;
    (0..200)
        .map(|seed| {
            let mut rng = CommonStatsRng::new(seed, 0);
            d.sample(&mut rng);
            let next = rng.next_u32();
            let mut fresh = CommonStatsRng::new(seed, 0);
            (0..).position(|_| fresh.next_u32() == next).unwrap()
        })
        .collect()
}

/// The position uniform switches from `uniform()` (one word) to
/// `uniform52()` (two words) at variance 10⁶ (`FINE_U_FROM`): a PTRS/BTRS
/// attempt then takes 3 words (`u` + `v`) instead of 2, and a hypergeometric
/// draw 2 instead of 1. Counted by locating the RNG's next word in a fresh
/// copy of the stream, over 200 seeds (most PTRS/BTRS draws accept on the
/// first attempt, so a 2- or 3-word draw must occur).
#[cfg(all(feature = "dist", feature = "rng"))]
#[test]
fn sampler_fine_uniform_above_threshold() {
    use commonstats::dist::DiscreteSampler;
    fn check(name: &str, d: &impl DiscreteSampler, per_attempt: usize) {
        let w = words_per_draw(d);
        assert!(w.iter().all(|&c| c % per_attempt == 0), "{name}: {w:?}");
        assert!(w.contains(&per_attempt), "{name}: {w:?}");
    }
    // λ and n·p·q = 10⁶ exactly at the switch, 999 999 just below it.
    check("poisson(1e6)", &Poisson::new(1e6).unwrap(), 3);
    check("poisson(999999)", &Poisson::new(999_999.0).unwrap(), 2);
    check("poisson(1e15)", &Poisson::new(1e15).unwrap(), 3);
    check(
        "binomial(4e6, .5)",
        &Binomial::new(4_000_000, 0.5).unwrap(),
        3,
    );
    check(
        "binomial(3999996, .5)",
        &Binomial::new(3_999_996, 0.5).unwrap(),
        2,
    );
    check(
        "binomial(1e15, .7)",
        &Binomial::new(1_000_000_000_000_000, 0.7).unwrap(),
        3,
    );
    // Variance 6.25·10⁶ and 6.25·10⁵.
    let big = Hypergeometric::new(100_000_000, 50_000_000, 50_000_000).unwrap();
    let small = Hypergeometric::new(10_000_000, 5_000_000, 5_000_000).unwrap();
    assert!(words_per_draw(&big).iter().all(|&c| c == 2));
    assert!(words_per_draw(&small).iter().all(|&c| c == 1));
}

/// Single-comparison samplers take `uniform52()` (two words) instead of
/// `uniform()` (one) when the probability they resolve is below 10⁻³
/// (`FINE_U_BELOW_P`): the rarer Bernoulli outcome, `n·p` / `λ` in the
/// binomial / Poisson inversion branches, the geometric `p`. `p ∈ {0, 1}`
/// keeps one word. Then 10⁶ `Bernoulli(5·10⁻⁴)` draws hit `p` within 5 SE,
/// and `Geometric(10⁻⁴)` matches its mean and variance.
#[cfg(all(feature = "dist", feature = "rng"))]
#[test]
fn sampler_fine_uniform_small_probability() {
    use commonstats::dist::DiscreteSampler;
    use commonstats::rng::CommonStatsRng;
    fn words(name: &str, d: &impl DiscreteSampler, want: usize) {
        let w = words_per_draw(d);
        assert!(w.iter().all(|&c| c == want), "{name}: {w:?}");
    }
    words("bernoulli(9.99e-4)", &Bernoulli::new(9.99e-4).unwrap(), 2);
    words("bernoulli(1e-3)", &Bernoulli::new(1e-3).unwrap(), 1);
    words("bernoulli(1-1e-4)", &Bernoulli::new(1.0 - 1e-4).unwrap(), 2);
    words("bernoulli(0)", &Bernoulli::new(0.0).unwrap(), 1);
    words("bernoulli(1)", &Bernoulli::new(1.0).unwrap(), 1);
    words("poisson(9.99e-4)", &Poisson::new(9.99e-4).unwrap(), 2);
    words("poisson(1e-3)", &Poisson::new(1e-3).unwrap(), 1);
    words(
        "binomial(10, 9.99e-5)",
        &Binomial::new(10, 9.99e-5).unwrap(),
        2,
    );
    words("binomial(10, 1e-4)", &Binomial::new(10, 1e-4).unwrap(), 1);
    words(
        "binomial(10, 1-1e-5)",
        &Binomial::new(10, 1.0 - 1e-5).unwrap(),
        2,
    );
    words("binomial(10, 0)", &Binomial::new(10, 0.0).unwrap(), 1);
    words("geometric(9.99e-4)", &Geometric::new(9.99e-4).unwrap(), 2);
    words("geometric(1e-3)", &Geometric::new(1e-3).unwrap(), 1);
    let mut rng = CommonStatsRng::new(12, 0);
    let (p, n) = (5e-4, 1_000_000);
    let b = Bernoulli::new(p).unwrap();
    let hits = (0..n).map(|_| b.sample(&mut rng)).sum::<i64>() as f64;
    let se = (n as f64 * p * (1.0 - p)).sqrt();
    assert!(
        (hits - n as f64 * p).abs() < 5.0 * se,
        "bernoulli({p}): {hits} hits"
    );
    let g = Geometric::new(1e-4).unwrap();
    let xs: Vec<f64> = (0..100_000).map(|_| g.sample(&mut rng) as f64).collect();
    check_draws("geometric(1e-4)", &g, &xs);
}

/// PTRS/BTRS past the point where a log-gamma-sum log pmf loses O(1) to
/// cancellation (≈ 10¹² in `λ` or `n`): 4·10⁵ draws, mean and variance within
/// 5 SE. With the old acceptance test `var/λ` was 1.039 at `λ = 10¹⁵` (17 SE)
/// and `var/(n·p·q)` 0.917 at `Binom(10¹⁵, 0.3)` (37 SE). `Binom(10¹⁵, 10⁻¹⁴)`
/// has `n·p = 10` but the same `n·ln n` cancellation in its `n − k` term.
#[cfg(all(feature = "dist", feature = "rng"))]
#[test]
fn sampler_huge_parameter_moments() {
    use commonstats::dist::DiscreteSampler;
    use commonstats::rng::CommonStatsRng;
    const N: usize = 400_000;
    fn run(name: &str, d: &(impl DiscreteSampler + Distribution), rng: &mut CommonStatsRng) {
        let xs: Vec<f64> = (0..N).map(|_| d.sample(rng) as f64).collect();
        check_draws(name, d, &xs);
    }
    let mut rng = CommonStatsRng::new(2025, 0);
    for lam in [1e12, 1e15, 1e17] {
        run(
            &format!("poisson({lam:e})"),
            &Poisson::new(lam).unwrap(),
            &mut rng,
        );
    }
    for (n, p) in [
        (1_000_000_000_000_000, 0.3),
        (100_000_000_000_000_000, 0.4),
        (1_000_000_000_000_000, 1e-14),
    ] {
        let b = Binomial::new(n, p).unwrap();
        run(&format!("binomial({n:e}, {p})"), &b, &mut rng);
    }
}

#[cfg(all(feature = "dist", feature = "rng"))]
#[test]
fn poisson_sampler_saturates_above_i64() {
    // Every draw at λ ≫ 2⁶³ lies above i64::MAX and saturates to it; an
    // offset added to a saturated ⌊λ⌋ would wrap below it (or to −1).
    use commonstats::dist::DiscreteSampler;
    use commonstats::rng::CommonStatsRng;
    let mut rng = CommonStatsRng::new(7, 0);
    for lam in [1e20, 1e40, 1e300] {
        let p = Poisson::new(lam).unwrap();
        assert!(
            (0..20_000).all(|_| p.sample(&mut rng) == i64::MAX),
            "poisson({lam:e}) draw below i64::MAX"
        );
    }
}

/// Pearson χ² goodness of fit like `check_chi2_fit`, for supports too wide to
/// walk with `cdf` (the hypergeometric `cdf` is an O(support) sum): expected
/// counts from `mass` over the observed range, the two tails beyond it summed
/// from `mass` outward until a term no longer changes the sum (the pmf is
/// unimodal, so the terms only shrink). Asserts p > 0.001.
#[cfg(all(feature = "dist", feature = "rng"))]
fn check_chi2_fit_by_mass(name: &str, d: &impl DiscreteMass, draws: &[i64]) {
    let n = draws.len() as f64;
    let (s_lo, s_hi) = (
        d.support_min().as_f64() as i64,
        d.support_max().as_f64() as i64,
    );
    let lo = *draws.iter().min().unwrap();
    let hi = *draws.iter().max().unwrap();
    let mut counts = vec![0.0; (hi - lo + 1) as usize];
    for &k in draws {
        counts[(k - lo) as usize] += 1.0;
    }
    let tail = |mut k: i64, step: i64| {
        let mut acc = 0.0;
        while (s_lo..=s_hi).contains(&k) {
            let f = d.mass(k);
            if acc + f == acc {
                break;
            }
            acc += f;
            k += step;
        }
        acc
    };
    let mut cells: Vec<(f64, f64)> = Vec::new();
    let (mut o, mut e) = (0.0, n * tail(lo - 1, -1));
    for k in lo..=hi {
        o += counts[(k - lo) as usize];
        e += n * d.mass(k);
        if e >= 5.0 {
            cells.push((o, e));
            (o, e) = (0.0, 0.0);
        }
    }
    e += n * tail(hi + 1, 1);
    match cells.last_mut() {
        Some(last) if e < 5.0 => *last = (last.0 + o, last.1 + e),
        _ => cells.push((o, e)),
    }
    let stat: f64 = cells.iter().map(|&(o, e)| (o - e) * (o - e) / e).sum();
    let df = (cells.len() - 1) as f64;
    let p = ChiSquared::new(df).unwrap().sf(stat);
    println!("{name}: chi2 = {stat:.3}, df = {df}, p = {p:.4}");
    assert!(p > 0.001, "{name}: chi2 = {stat}, df = {df}, p = {p:e}");
}

/// Hypergeometric mode-outward inversion: χ² fit to `mass` from small support
/// up to `N = 10⁶` (the default bisection inversion took ~1 s per draw there),
/// plus moments within 5 SE, 10⁵ draws per parameter set. Degenerate supports
/// return their single point.
#[cfg(all(feature = "dist", feature = "rng"))]
#[test]
fn hypergeometric_sampler_fit() {
    use commonstats::dist::DiscreteSampler;
    use commonstats::rng::CommonStatsRng;
    const N: usize = 100_000;
    let mut rng = CommonStatsRng::new(31, 0);
    for (big_n, k, n) in [
        (20, 7, 12),
        (50, 3, 45),
        (1000, 900, 50),
        (10_000, 5000, 5000),
        (1_000_000, 500_000, 500_000),
        (1_000_000, 10_000, 300_000),
    ] {
        let h = Hypergeometric::new(big_n, k, n).unwrap();
        let draws: Vec<i64> = (0..N).map(|_| h.sample(&mut rng)).collect();
        let name = format!("hypergeometric({big_n}, {k}, {n})");
        let xs: Vec<f64> = draws.iter().map(|&x| x as f64).collect();
        check_draws(&name, &h, &xs);
        check_chi2_fit_by_mass(&name, &h, &draws);
    }
    for (big_n, k, n, want) in [(5, 5, 3, 3), (10, 0, 4, 0), (7, 3, 7, 3), (0, 0, 0, 0)] {
        let h = Hypergeometric::new(big_n, k, n).unwrap();
        assert!((0..100).all(|_| h.sample(&mut rng) == want));
    }
}
