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
use commonstats::dist::continuous::StudentizedRange;
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

// Bulk ladder (spec §3): pdf/cdf/sf/pmf value-kind vs ppf inverse-kind. Values
// are relative only: an absolute floor would pass any row whose truth is below it
// (an sf formed as `1 − cdf`, a density underflowed to 0). Quantiles keep a floor
// for the rows whose true quantile is 0 (a symmetric median, Uniform's p = ¼).
const VAL: Tol = Tol {
    rel: 1e-12,
    abs: 0.0,
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

/// A fixture row's `[k, df, x]` as the type; df `0.0` stands for ∞ (JSON has
/// no ∞, and df = 0 is invalid). Mirrors `_sr_df_json` in
/// `scripts/gen_oracle.py` — change together.
fn sr(a: &[f64]) -> StudentizedRange {
    let df = if a[1] == 0.0 { f64::INFINITY } else { a[1] };
    StudentizedRange::new(a[0], df).unwrap()
}

#[test]
fn studentizedrange_oracle() {
    check_grid("dist_studentizedrange_pdf", VAL, |a| sr(a).density(a[2]));
    check_grid("dist_studentizedrange_cdf", VAL, |a| sr(a).cdf(a[2]));
    check_grid("dist_studentizedrange_sf", VAL, |a| sr(a).sf(a[2]));
    check_grid("dist_studentizedrange_ppf", INV, |a| {
        sr(a).quantile(a[2]).unwrap()
    });
    check_grid("dist_studentizedrange_isf", INV, |a| {
        sr(a).isf(a[2]).unwrap()
    });
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
    // for p ≤ 1e-8. χ²(1) runs Gamma(½, rate ½)'s solver (`as_gamma`), so this
    // also covers Gamma. Exercises the small-p seed of the Newton solvers, which
    // the fixtures (k=5, shape=3.5) never reach.
    let c = ChiSquared::new(1.0).unwrap();
    for p in [1e-8, 1e-12] {
        let truth = core::f64::consts::FRAC_PI_2 * p * p;
        let got = c.quantile(p).unwrap();
        let rel = ((got - truth) / truth).abs();
        assert!(
            rel < 1e-13,
            "chi2.quantile({p:e}) = {got:e}, want {truth:e} (rel {rel:e})"
        );
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
fn beta_sf_small_shape_huge_other() {
    // sf = I_y(1e12, 0.9) with `y = 1 − x` exact, near the reflection point:
    // `bgrat`'s `ln M` cancelled `b·ln a` (18.7ε). And with a subnormal
    // shape at a subnormal `x`, where the kernel at the stand-in shape gave
    // 0; and with a small normal shape at a subnormal `x`, where `bgrat`'s
    // `z` is subnormal and the sf was `1 −` a `bfrac` value near 1 (ε, or
    // 1e-5 relative at 1e-10). Truth: mpmath (80 and 120 dps) incomplete
    // beta at the exact inputs, the last five (420 dps) as `₂F₁` forms.
    for (a, b, x, want) in [
        (
            0.9,
            1e12,
            2.899_999_999_997_389_8e-12,
            4.506_971_639_117_824_3e-2,
        ),
        (1e-310, 0.5, 5e-324, 7.458_263_662_824_989e-308),
        (1e-300, 1e-20, 5e-324, 1.000_000_000_000_000_1e-280),
        (
            1e-300,
            8.452_712_498_170_644e-40,
            5e-324,
            1.183_052_186_166_774_7e-261,
        ),
        (3e-308, 1e-300, 5e-324, 2.999_999_910_000_003e-8),
        (1e-10, 0.5, 1e-320, 7.382_135_080_039_688e-8),
        (1e-12, 3.0, 4e-310, 7.109_150_842_143_347e-10),
    ] {
        let got = Beta::new(a, b).unwrap().sf(x);
        let rel = rel_err(got, want);
        assert!(
            rel < 2e-15,
            "Beta({a:e}, {b:e}).sf({x:e}) = {got:e} (rel {rel:e})"
        );
    }
}

#[test]
fn negbinomial_tiny_size_upper_tail() {
    // `sf = I_{1−p}(k + 1, r)` with `r ≪ 1 ≪ k`, taken by TOMS 708 `bgrat`;
    // the continued fraction there kept ~30ε. Truth: mpmath (420 dps)
    // betainc(k + 1, r, 0, mu/(r + mu), regularized=True) at the exact inputs.
    let nb =
        NegBinomial::from_mean_size(2_392.685_576_414_455_7, 0.005_794_104_492_098_789).unwrap();
    let got = nb.sf(475_024);
    let rel = rel_err(got, 1.000_003_105_188_228e-3);
    assert!(rel < 2e-15, "sf = {got:e} (rel {rel:e})");
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
    // A heavy tail past the moment-based starting bracket (418, where
    // P(X > 418) ≈ 3.6e-4): the search must grow it for p above ~0.9996.
    // scipy: nbinom(0.1, 0.01).ppf(1 − 1e-6) = 941 (scipy's p is 1 − ours).
    let heavy = NegBinomial::new(0.1, 0.99).unwrap();
    assert_eq!(heavy.quantile(1.0 - 1e-6).unwrap(), 941);
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
        (
            "Hypergeometric(9.2e18, 4.6e18, 9.2e17).mass(9.2e16)",
            Hypergeometric::new(
                9_200_000_000_000_000_000,
                4_600_000_000_000_000_000,
                920_000_000_000_000_000,
            )
            .unwrap()
            .mass(92_000_000_000_000_000),
        ),
    ] {
        assert!(
            got == 0.0 && got.is_sign_positive(),
            "{name} = {got:?}, want +0"
        );
    }
}

/// `NegBinomial` at a size far from 1. Below 1 the Stirling error `δ(r) ≈
/// −½·ln r` (354 at `r = 1e-307`) cancels against the `½·ln r` of the
/// saddle form's `f`; kept as an f64 its rounding gave 51ε at `r = 1e-100`.
/// At `k = 0` and `r` beyond ≈ 1.34e300, `r·ln q` overflowed the split of
/// the exact product into NaN. Truth from mpmath (80 digits) at the exact
/// f64 parameters, `Γ(r+k)/(Γ(r)·k!)·(1−p)ʳpᵏ`, `(1−p)ʳ` as `r·log1p(−p)`.
#[test]
fn negbinomial_tiny_and_huge_size() {
    let nb = |r: f64, p: f64| NegBinomial::new(r, p).unwrap();
    for (name, got, want) in [
        (
            "NegBinomial(1e-300, 0.5).mass(3)",
            nb(1e-300, 0.5).mass(3),
            4.166_666_666_666_667e-302,
        ),
        (
            "NegBinomial(1e-300, 0.9).mass(7)",
            nb(1e-300, 0.9).mass(7),
            6.832_812_857_142_858e-302,
        ),
        (
            "NegBinomial(1e-100, 0.3).mass(1)",
            nb(1e-100, 0.3).mass(1),
            2.999_999_999_999_999_7e-101,
        ),
        (
            "NegBinomial(1e305, 1e-306).mass(0)",
            nb(1e305, 1e-306).mass(0),
            0.904_837_418_035_959_6,
        ),
    ] {
        let rel = rel_err(got, want);
        assert!(
            rel <= 4.0 * f64::EPSILON,
            "{name} = {got:e}, want {want:e} (rel {rel:e})"
        );
    }
    for (name, got, want) in [
        (
            "NegBinomial(1e305, 1e-306).log_mass(0)",
            nb(1e305, 1e-306).log_mass(0),
            -0.099_999_999_999_999_99,
        ),
        (
            "NegBinomial(1e305, 0.5).log_mass(0)",
            nb(1e305, 0.5).log_mass(0),
            -6.931_471_805_599_452e304,
        ),
        (
            "NegBinomial(1.7e308, 0.5).log_mass(0)",
            nb(1.7e308, 0.5).log_mass(0),
            -1.178_350_206_951_907e308,
        ),
    ] {
        let rel = rel_err(got, want);
        assert!(
            rel <= 2.0 * f64::EPSILON,
            "{name} = {got:e}, want {want:e} (rel {rel:e})"
        );
    }
    assert_eq!(nb(1.7e308, 0.9).log_mass(0), f64::NEG_INFINITY);
    for r in [1e305, 1.7e308] {
        let got = nb(r, 0.5).mass(0);
        assert!(
            got == 0.0 && got.is_sign_positive(),
            "mass(0) at r = {r:e}: {got:?}"
        );
    }
}

#[test]
fn gamma_pdf_at_subnormal_x() {
    // `β·x` subnormal, `α` near 1: the prefactor is formed at `2⁶⁴·β·x` and
    // scaled by `2^{−64α}`, which must not pass through the subnormal range
    // before the division by `x`. Truth: mpmath (50 digits).
    for (name, got, want) in [
        (
            "Gamma(0.99, 1).pdf(5e-324)",
            Gamma::new(0.99, 1.0).unwrap().density(5e-324),
            1_700.276_072_502_215_2,
        ),
        (
            "Gamma(1.013186788315267, 13.302949435284603).density(2.82334e-319)",
            Gamma::new(1.013_186_788_315_267, 13.302_949_435_284_603)
                .unwrap()
                .density(2.82334e-319),
            8.737_190_679_962_089e-4,
        ),
        (
            "ChiSquared(1.98).pdf(1e-320)",
            ChiSquared::new(1.98).unwrap().density(1e-320),
            793.300_343_564_606_3,
        ),
    ] {
        let rel = rel_err(got, want);
        assert!(
            rel <= 8.0 * f64::EPSILON,
            "{name} = {got:e}, want {want:e} (rel {rel:e})"
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
        (
            "studentizedrange",
            measure_band("dist_studentizedrange_ppf", |a| {
                sr(a).quantile(a[2]).unwrap()
            }),
            measure_band("dist_studentizedrange_isf", |a| sr(a).isf(a[2]).unwrap()),
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

// ---- non-grid regressions ---------------------------------------------------
/// Closed-form `mean`/`variance`, `None` where undefined (never NaN; `dist`
/// module docs), at parameters where each convention shows: `sd ≠ 1` (Normal
/// takes the standard deviation), `rate ≠ 1` (Gamma takes a rate), `σ ≠ 1`
/// (LogNormal), Weibull `k ≠ 1`. Truth: the closed forms; the Weibull (Γ)
/// and LogNormal (exp) values from mpmath (40 dps).
#[test]
fn continuous_moments() {
    type Row = (
        &'static str,
        Box<dyn Distribution>,
        Option<f64>,
        Option<f64>,
    );
    #[rustfmt::skip]
    let rows: [Row; 16] = [
        ("Normal(0.5, 2)", Box::new(Normal::new(0.5, 2.0).unwrap()), Some(0.5), Some(4.0)),
        ("StudentT(5)", Box::new(StudentT::new(5.0).unwrap()), Some(0.0), Some(5.0 / 3.0)),
        ("StudentT(2)", Box::new(StudentT::new(2.0).unwrap()), Some(0.0), None),
        ("StudentT(1)", Box::new(StudentT::new(1.0).unwrap()), None, None),
        ("ChiSquared(4)", Box::new(ChiSquared::new(4.0).unwrap()), Some(4.0), Some(8.0)),
        ("FisherF(5, 10)", Box::new(FisherF::new(5.0, 10.0).unwrap()), Some(1.25), Some(1.354_166_666_666_666_7)),
        ("FisherF(5, 4)", Box::new(FisherF::new(5.0, 4.0).unwrap()), Some(2.0), None),
        ("FisherF(5, 2)", Box::new(FisherF::new(5.0, 2.0).unwrap()), None, None),
        ("Uniform(2, 6)", Box::new(Uniform::new(2.0, 6.0).unwrap()), Some(4.0), Some(16.0 / 12.0)),
        ("Exponential(2)", Box::new(Exponential::new(2.0).unwrap()), Some(0.5), Some(0.25)),
        ("Cauchy(0, 1)", Box::new(Cauchy::new(0.0, 1.0).unwrap()), None, None),
        ("Weibull(1.5, 2)", Box::new(Weibull::new(1.5, 2.0).unwrap()), Some(1.805_490_585_901_867_3), Some(1.502_761_139_255_728)),
        ("LogNormal(0.5, 0.75)", Box::new(LogNormal::new(0.5, 0.75).unwrap()), Some(2.184_200_810_815_618), Some(3.602_164_306_159_661_8)),
        ("Gamma(3.5, 2)", Box::new(Gamma::new(3.5, 2.0).unwrap()), Some(1.75), Some(0.875)),
        ("Beta(2, 3)", Box::new(Beta::new(2.0, 3.0).unwrap()), Some(0.4), Some(0.04)),
        ("InverseGaussian(1.5, 2)", Box::new(InverseGaussian::new(1.5, 2.0).unwrap()), Some(1.5), Some(1.6875)),
    ];
    let mut fails = Vec::new();
    for (name, d, mean, var) in &rows {
        for (what, got, want) in [("mean", d.mean(), *mean), ("variance", d.variance(), *var)] {
            let ok = match (got, want) {
                (Some(g), Some(w)) if w == 0.0 => g == w,
                (Some(g), Some(w)) => rel_err(g, w) <= 4.0 * f64::EPSILON,
                _ => got == want,
            };
            if !ok {
                fails.push(format!("{name}.{what}() = {got:?}, want {want:?}"));
            }
        }
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

#[test]
fn cauchy_density_offset_overflows() {
    // `x − loc` overflows (2e308): mpmath `scale/(π(scale² + (x − loc)²))`.
    let wide = Cauchy::new(-1e308, 1e308).unwrap();
    let pdf = wide.density(1e308);
    assert!(((pdf - 6.366_197_723_675_83e-310) / 6.366_197_723_675_83e-310).abs() < 1e-12);
    assert!((wide.log_density(1e308) + 711.950_376_440_449_5).abs() < 1e-12);
}

#[test]
fn uniform_width_overflows() {
    // `b − a` overflows (2e308): mpmath (50 dps) `1/(b − a)` and its log at
    // the exact f64 inputs; cdf and sf at the midpoint 0 are ½.
    let wide = Uniform::new(-1e308, 1e308).unwrap();
    assert_eq!(wide.cdf(0.0), 0.5);
    assert_eq!(wide.sf(0.0), 0.5);
    assert!(rel_err(wide.density(0.0), 5e-309) <= 4.0 * f64::EPSILON);
    let ln_w = 709.889_355_822_726;
    assert!((wide.log_density(0.0) + ln_w).abs() <= 4.0 * f64::EPSILON * ln_w);
    assert!((wide.entropy().unwrap() - ln_w).abs() <= 4.0 * f64::EPSILON * ln_w);
    // `a + b` and `(b − a)²` overflow while the mean and variance do not:
    // mpmath `(a + b)/2` and `(b − a)²/12` at the exact f64 inputs.
    let far = Uniform::new(1e308, 1.5e308).unwrap();
    assert!(rel_err(far.mean().unwrap(), 1.25e308) <= 4.0 * f64::EPSILON);
    let broad = Uniform::new(0.0, 2e154).unwrap();
    assert!(rel_err(broad.variance().unwrap(), 3.333_333_333_333_333_6e307) <= 4.0 * f64::EPSILON);
    #[cfg(feature = "rng")]
    {
        use commonstats::dist::Sampler;
        let mut rng = commonstats::rng::CommonStatsRng::new(7, 0);
        for _ in 0..100 {
            let x = wide.sample(&mut rng);
            assert!((-1e308..=1e308).contains(&x), "draw {x:e}");
        }
    }
}

#[test]
fn weibull_range_edges() {
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
fn beta_huge_equal_shapes_half_ulp() {
    // One ulp below ½ at huge equal shapes: x = ½ − 2⁻⁵⁴ lies ~50 standard
    // deviations (1/(2√(2a + 1))) below the mean at a = 1e35, so the sf is 1
    // to f64 resolution and the cdf below the f64 range (normal limit).
    for a in [1e35, 1e90] {
        let d = Beta::new(a, a).unwrap();
        assert_eq!(d.sf(0.499_999_999_999_999_94), 1.0);
        assert_eq!(d.cdf(0.499_999_999_999_999_94), 0.0);
    }
}

#[test]
fn beta_and_fisherf_tiny_shape_log_density() {
    // At a tiny first shape `−ln B ≈ ln a` and `(a − 1)·ln x` are each ~700
    // and cancel to an O(1) log density; Beta's density falls back to `exp`
    // of it there. mpmath (50 dps) at the exact f64 inputs:
    // `(a−1)·ln x + (b−1)·ln(1−x) − ln B(a, b)` for Beta, and
    // `a·ln(m/n) + (a−1)·ln x − (a+b)·ln(1 + m·x/n) − ln B(a, b)`,
    // `a, b = m/2, n/2`, for F(m, n).
    let tol = 4.0 * f64::EPSILON;
    for (a, b, x, pdf, ln_pdf) in [
        (
            1e-310,
            0.5,
            1.979_999_999_999_98e-310,
            0.505_050_505_050_509_9,
            -0.683_096_844_706_434_4,
        ),
        (
            1e-310,
            0.5,
            1e-315,
            100_000.000_151_831_31,
            11.512_925_466_488_541,
        ),
        (
            1e-310,
            0.5,
            5e-324,
            20_240_225_330_731.0,
            30.638_693_093_227_097,
        ),
        (1e-300, 0.5, 3e-300, 1.0 / 3.0, -1.098_612_288_668_109_8),
        (1e-310, 20.0, 2e-310, 0.5, -core::f64::consts::LN_2),
    ] {
        let d = Beta::new(a, b).unwrap();
        let (got, ln_got) = (d.density(x), d.log_density(x));
        assert!(
            rel_err(got, pdf) <= tol && (ln_got - ln_pdf).abs() <= tol * ln_pdf.abs().max(1.0),
            "Beta({a:e}, {b}) at {x:e}: density {got}, log {ln_got}; want {pdf}, {ln_pdf}"
        );
    }
    for (m, n, x, ln_pdf) in [
        (2e-310, 1.0, 3e-310, -1.098_612_288_668_109_8),
        (2e-310, 1.0, 3e-300, -24.124_463_218_608_568),
        (2e-300, 1.0, 3e-300, -1.098_612_288_668_109_8),
    ] {
        let got = FisherF::new(m, n).unwrap().log_density(x);
        assert!(
            (got - ln_pdf).abs() <= tol * ln_pdf.abs().max(1.0),
            "F({m:e}, {n}).log_density({x:e}) = {got}, want {ln_pdf}"
        );
    }
}

#[test]
fn inversegaussian_large_shape_ratio() {
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
        let got = t.cdf(x) - 0.5;
        assert!(
            (got - central).abs() < 1.2e-16,
            "t({df}): cdf(x) − ½ = {got:e}, want {central:e} at x = {x:e}"
        );
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
        // Newton on the raw sf exhausts its iterations far from the root.
        (1.5, 2.0, 1e-50, 243.985_082_797_044_84),
        (1.5, 2.0, 1e-100, 500.620_404_686_735_1),
        (1.5, 2.0, 1e-200, 1_016.322_095_252_721_8),
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
        let (got, rel) = (t.cdf(x), rel_err(t.cdf(x), want));
        assert!(
            rel < 2e-15,
            "t({df}).cdf({x:e}) = {got:e}, want {want:e} (rel {rel:e})"
        );
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
        // `dfn·x/dfd` just above the smallest normal, where the kernel's
        // `bgrat` once gave up on an underflowing `shape·z` and the sf came
        // out as `1 − I` (2.2e-16, 1.1e-16).
        (
            "F(1e-20, 5).sf(1.12e-287)",
            f(1e-20, 5.0).sf(1.12e-287),
            3.535_546_802_353_762e-18,
        ),
        (
            "F(1e-20, 5).ln_sf(1.12e-287)",
            f(1e-20, 5.0).ln_sf(1.12e-287),
            -40.183_663_704_086_64,
        ),
        (
            "F(1e-300, 15.99).sf(3.56e-7)",
            f(1e-300, 15.99).sf(3.56e-7),
            3.529_018_176_473_564_3e-298,
        ),
        (
            "F(1e-100, 16).ln_sf(3.56e-207)",
            f(1e-100, 16.0).ln_sf(3.56e-207),
            -224.392_319_475_815_24,
        ),
        // `y = dfn·x/(dfd + dfn·x)` subnormal (2.2e-308): from `exp(log_density)`
        // the density kept only `|ln f|·ε` (r 130 to 230). mpmath
        // (`scripts/accuracy_sweep.py`).
        (
            "F(1e-20, 5).pdf(1.1014115599610647e-287)",
            f(1e-20, 5.0).density(1.101_411_559_961_064_7e-287),
            4.539_629_128_440_191_4e266,
        ),
    ] {
        let rel = rel_err(got, want);
        assert!(rel < 1e-14, "{name} = {got:e}, want {want:e} (rel {rel:e})");
    }
}

#[test]
fn quantile_polish_only_where_it_lowers_the_residual() {
    // Relative error of the root, in ε (the sweep's r stays below 1 either
    // way, the parameters' own conditioning being far larger). The final
    // Newton step on the linear mass is kept only where it lowers
    // `|mass − p|`: taken regardless, it moved these roots by the mass's own
    // error over the slope. Truth from mpmath (`scripts/accuracy_sweep.py`);
    // the error the unchecked step left in each comment.
    let f = |m, n| FisherF::new(m, n).unwrap();
    let ig = |m, l| InverseGaussian::new(m, l).unwrap();
    let beta = |a, b| Beta::new(a, b).unwrap();
    let p = 1.593_498_434e-314;
    for (name, got, want, tol) in [
        // Bulk, slope 0.04: 161ε.
        (
            "F(0.0725, 0.0196).isf(0.0646)",
            f(0.072_462_934_757_742_13, 0.019_634_664_094_859_61)
                .isf(0.064_649_105_052_420_48)
                .unwrap(),
            1.018_247_778_116_97e110,
            80.0,
        ),
        // Spread √(μ/λ) = 9e-16, where the mass jumps between floats: 4e4 ulp.
        (
            "IG(3.2e-22, 3.9e10).isf(0.539)",
            ig(3.247_463_541_655_184_6e-22, 39_275_152_805.719_24)
                .isf(0.539_035_530_318_648_9)
                .unwrap(),
            3.247_463_541_655_184_6e-22,
            4.0,
        ),
        // Beta for a subnormal p, after `tail_newton`, whose residual is the
        // more precise here (the kernel's step: 184ε, 199ε, 19ε, 61ε).
        (
            "Beta(2, 0.5).quantile(1.59e-314)",
            beta(2.0, 0.5).quantile(p).unwrap(),
            2.061_390_103_195_836_6e-157,
            4.0,
        ),
        (
            "Beta(1.517, 0.5).quantile(1.59e-314)",
            beta(1.517, 0.5).quantile(p).unwrap(),
            2.469_632_046_748_456_8e-207,
            4.0,
        ),
        (
            "Beta(2, 0.0087).quantile(1.59e-314)",
            beta(2.0, 0.0087).quantile(p).unwrap(),
            1.905_680_901_832_207_4e-156,
            4.0,
        ),
        (
            "Beta(5.24, 6.6e-18).quantile(1.59e-314)",
            beta(5.24, 6.6e-18).quantile(p).unwrap(),
            3.395_803_532_848_337e-57,
            4.0,
        ),
    ] {
        let rel = ((got - want) / want).abs() / f64::EPSILON;
        assert!(rel <= tol, "{name} = {got:e}, want {want:e} ({rel:.1}ε)");
    }
    // Where the log density at the root is NaN or infinite the step is not
    // taken; unchecked it made these NaN, ±∞ or negative.
    for (name, got) in [
        (
            "IG(1e300, 1e-300).quantile(1e-300)",
            ig(1e300, 1e-300).quantile(1e-300),
        ),
        (
            "IG(1e300, 1e-300).quantile(1e-100)",
            ig(1e300, 1e-300).quantile(1e-100),
        ),
        (
            "IG(1e300, 1e-300).isf(1e-100)",
            ig(1e300, 1e-300).isf(1e-100),
        ),
        (
            "IG(f64::MAX, 1).quantile(1e-300)",
            ig(f64::MAX, 1.0).quantile(1e-300),
        ),
        (
            "IG(1, 5e-324).quantile(1 − 2⁻⁵³)",
            ig(1.0, 5e-324).quantile(1.0 - f64::EPSILON / 2.0),
        ),
    ] {
        let x = got.unwrap();
        assert!(x.is_finite() && x > 0.0, "{name} = {x:e}");
    }
    // Finite, and accurate where `λ/2π` underflows: the log density's constant
    // is `½·(ln λ − ln 2π)`, not `−∞` (which sent the solver to bisection and
    // left 148ε). Truth: mpmath (400 digits), the sf residual changing sign
    // across the neighbouring floats.
    let got = ig(1.0, 5e-324).isf(1e-100).unwrap();
    let rel = rel_err(got, 3.145_319_589_900_964e-124);
    assert!(
        rel <= 2.0 * f64::EPSILON,
        "IG(1, 5e-324).isf(1e-100) = {got:e} (rel {rel:e})"
    );
    // A root near e^{-1e9}, below half the smallest subnormal: 0, not 5e-324.
    assert_eq!(beta(2.01332e-320, 0.5).isf(2.11322e-311).unwrap(), 0.0);
}

#[test]
fn steep_mass_roots() {
    // Truth and K from mpmath (`scripts/accuracy_sweep.py`).
    //
    // Inverse Gaussians of spread `√(μ/λ)` 1.7e-17 and 1.9e-16: neighbouring
    // floats at the root are 12 standard deviations apart (the cdf goes from
    // 2e-33 to 2e-127 in one ulp), so a Newton step of 1e-14 relative is ~20
    // ulp and far from converged; the solver stopped on that step size, 22ε
    // and 12ε off. The root is one of the two floats around it: within 2ε.
    let ig = |m, l| InverseGaussian::new(m, l).unwrap();
    for (name, got, want) in [
        (
            "IG(8.3e-34, 2.81).quantile(3.58e-109)",
            ig(8.298_021_952_687_533e-34, 2.807_395_033_125_542_4)
                .quantile(3.581_506_378_510_237_3e-109)
                .unwrap(),
            8.298_021_952_687_529e-34,
        ),
        (
            "IG(6.5e-34, 0.0179).quantile(2.07e-306)",
            ig(6.491_021_288_153_795e-34, 0.017_912_225_037_463_79)
                .quantile(2.066_980_244_294_14e-306)
                .unwrap(),
            6.491_021_288_153_748_5e-34,
        ),
    ] {
        let rel = ((got - want) / want).abs() / f64::EPSILON;
        assert!(rel <= 2.0, "{name} = {got:e}, want {want:e} ({rel:.1}ε)");
    }
}

#[test]
fn narrow_distribution_roots() {
    // Inverse Gaussian whose relative spread `√(μ/λ)` is 5e-26: every
    // quantile is the mean to within an ulp. The solver's
    // bisection stopped at a bracket 1e-14 wide, 26 to 32 ulp from the root.
    // Scored as the accuracy sweep scores them (error within 10·ε·K, the
    // limit class); truth and K from mpmath (`scripts/accuracy_sweep.py`).
    let ig = |m, l| InverseGaussian::new(m, l).unwrap();
    let narrow = ig(7.548_406_235_343_999e-47, 35_204.146_412_994_28);
    for (name, got, want, k) in [
        (
            "IG(7.5e-47, 35204).quantile(1e-200)",
            narrow.quantile(1e-200).unwrap(),
            7.548_406_235_343_999e-47,
            7.55e-47,
        ),
        (
            "IG(7.5e-47, 35204).isf(0.9)",
            narrow.isf(0.9).unwrap(),
            7.548_406_235_343_999e-47,
            7.55e-47,
        ),
    ] {
        let r = (got - want).abs() / (f64::EPSILON * k);
        assert!(r <= 10.0, "{name} = {got:e}, want {want:e} (r {r:.1})");
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
fn continuous_edges() {
    // Every density is 0 at ±∞ and its log −∞, including where the log
    // density's terms would meet as ∞ − ∞ (χ², Gamma, F, Weibull); likewise
    // one finite point past each finite support end (the `ContinuousDensity`
    // contract). cdf/sf are 0/1 at ±∞, `isf(0)`/`isf(1)` the support ends.
    // A NaN `x` must come out NaN, not get read by a range check as "out of
    // support" (`(0..=1).contains(&NaN)` and `x < a || x > b` are both false
    // for NaN) and returned as a fixed in-support or out-of-support constant.
    trait Cont: ContinuousDensity + ContinuousCdf {}
    impl<T: ContinuousDensity + ContinuousCdf> Cont for T {}
    let dists: [(&str, Box<dyn Cont>); 14] = [
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
    let (inf, nan) = (f64::INFINITY, f64::NAN);
    let mut fails = Vec::new();
    let mut check = |ok: bool, what: String| {
        if !ok {
            fails.push(what);
        }
    };
    for (name, d) in &dists {
        for x in [inf, -inf] {
            check(d.density(x) == 0.0, format!("{name}.density({x})"));
            check(d.log_density(x) == -inf, format!("{name}.log_density({x})"));
        }
        let (lo, hi) = (d.support_min().as_f64(), d.support_max().as_f64());
        for x in [lo - 1.0, hi + 1.0].into_iter().filter(|x| x.is_finite()) {
            check(d.density(x) == 0.0, format!("{name}.density({x})"));
            check(d.log_density(x) == -inf, format!("{name}.log_density({x})"));
        }
        check(d.density(nan).is_nan(), format!("{name}.density(NaN)"));
        check(
            d.log_density(nan).is_nan(),
            format!("{name}.log_density(NaN)"),
        );
        check(
            d.cdf(-inf) == 0.0 && d.cdf(inf) == 1.0,
            format!("{name}.cdf(±∞)"),
        );
        check(
            d.sf(-inf) == 1.0 && d.sf(inf) == 0.0,
            format!("{name}.sf(±∞)"),
        );
        check(
            d.cdf(nan).is_nan() && d.sf(nan).is_nan(),
            format!("{name}.cdf/sf(NaN)"),
        );
        check(d.isf(0.0).unwrap() == hi, format!("{name}.isf(0)"));
        check(d.isf(1.0).unwrap() == lo, format!("{name}.isf(1)"));
    }
    // Limits at a support end, as scipy (`chi2(k).pdf(0)`, `gamma(a, scale=1/β)`,
    // `weibull_min(k, scale=λ).logpdf(0)`, `beta(α, β).pdf`), and as the
    // ChiSquared rustdoc states; within 4ε where finite.
    use core::f64::consts::LN_2;
    #[rustfmt::skip]
    let ends: [(&str, f64, f64); 20] = [
        ("ChiSquared(1).density(0)", ChiSquared::new(1.0).unwrap().density(0.0), inf),
        ("ChiSquared(1).log_density(0)", ChiSquared::new(1.0).unwrap().log_density(0.0), inf),
        ("ChiSquared(2).density(0)", ChiSquared::new(2.0).unwrap().density(0.0), 0.5),
        ("ChiSquared(4).density(0)", ChiSquared::new(4.0).unwrap().density(0.0), 0.0),
        ("ChiSquared(4).log_density(0)", ChiSquared::new(4.0).unwrap().log_density(0.0), -inf),
        ("Gamma(1, 2).density(0)", Gamma::new(1.0, 2.0).unwrap().density(0.0), 2.0),
        ("Gamma(1, 2).log_density(0)", Gamma::new(1.0, 2.0).unwrap().log_density(0.0), LN_2),
        ("Gamma(0.5, 2).density(0)", Gamma::new(0.5, 2.0).unwrap().density(0.0), inf),
        ("Weibull(1, 2).log_density(0)", Weibull::new(1.0, 2.0).unwrap().log_density(0.0), -LN_2),
        ("Weibull(0.5, 1).log_density(0)", Weibull::new(0.5, 1.0).unwrap().log_density(0.0), inf),
        ("Weibull(2, 1).log_density(0)", Weibull::new(2.0, 1.0).unwrap().log_density(0.0), -inf),
        ("LogNormal(0, 1).density(0)", LogNormal::new(0.0, 1.0).unwrap().density(0.0), 0.0),
        ("InverseGaussian(1.5, 2).density(0)", InverseGaussian::new(1.5, 2.0).unwrap().density(0.0), 0.0),
        ("Beta(1, 1).density(0)", Beta::new(1.0, 1.0).unwrap().density(0.0), 1.0),
        ("Beta(1, 1).density(1)", Beta::new(1.0, 1.0).unwrap().density(1.0), 1.0),
        ("Beta(1, 3).density(0)", Beta::new(1.0, 3.0).unwrap().density(0.0), 3.0),
        ("Beta(3, 1).density(1)", Beta::new(3.0, 1.0).unwrap().density(1.0), 3.0),
        ("Beta(3, 1).log_density(0)", Beta::new(3.0, 1.0).unwrap().log_density(0.0), -inf),
        // b − a overflows: the endpoints must not come from `0 · ∞`.
        ("Uniform(-1e308, 1e308).quantile(0)", Uniform::new(-1e308, 1e308).unwrap().quantile(0.0).unwrap(), -1e308),
        ("Uniform(-1e308, 1e308).isf(0)", Uniform::new(-1e308, 1e308).unwrap().isf(0.0).unwrap(), 1e308),
    ];
    for (name, got, want) in ends {
        // Zero and infinite limits are exact: `rel_err` against 0 is `|got|`.
        let ok = if want == 0.0 || want.is_infinite() {
            got == want
        } else {
            rel_err(got, want) <= 4.0 * f64::EPSILON
        };
        check(ok, format!("{name} = {got:e}, want {want:e}"));
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
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
    }
}

#[test]
fn studentt_cdf_sf_large_df_grid() {
    // `halves` passes both `y = x²/(df + x²)` and `z = df/(df + x²)`, each
    // formed by its own division, to the incomplete-beta kernel, so at large
    // df, where `z` is within `x²/df` of 1, the tail is never recovered from
    // `1 − z`. At |x| = 30 and df ≥ 1e3 the tail's condition number in `y` is
    // `≈ (df/2)·y`, 240–450, so the rounding of `y` alone accounts for up to
    // ~5e-14. mpmath (80 dps) at the f64 `x`: `sf(x) = ½·I_z(df/2, ½)`, and
    // `cdf(x) = 1 − sf(x)`. Negative `x` takes the same `halves(|x|)` with the
    // sides swapped; the t(7) grid covers that branch.
    #[rustfmt::skip]
    let cases: &[(f64, f64, f64, f64, f64)] = &[
        (1.0, 0.01, 0.503_182_992_764_908_3, 0.496_817_007_235_091_76, 1e-14),
        (1.0, 0.5, 0.647_583_617_650_433_3, 0.352_416_382_349_566_74, 1e-14),
        (1.0, 2.0, 0.852_416_382_349_566_7, 0.147_583_617_650_433_26, 1e-14),
        (1.0, 3.0, 0.897_583_617_650_433_3, 0.102_416_382_349_566_72, 1e-14),
        (1.0, 30.0, 0.989_393_597_594_464_5, 0.010_606_402_405_535_424, 1e-14),
        (2.5, 0.01, 0.503_618_002_821_045_9, 0.496_381_997_178_954_15, 1e-14),
        (2.5, 0.5, 0.671_151_040_065_142_7, 0.328_848_959_934_857_35, 1e-14),
        (2.5, 2.0, 0.921_304_252_121_017, 0.078_695_747_878_983, 1e-14),
        (2.5, 3.0, 0.963_711_952_225_484_1, 0.036_288_047_774_515_92, 1e-14),
        (2.5, 30.0, 0.999_854_467_645_421_3, 0.000_145_532_354_578_697_7, 1e-14),
        (30.0, 0.01, 0.503_956_253_713_411_1, 0.496_043_746_286_588_87, 1e-14),
        (30.0, 0.5, 0.689_638_497_557_436_3, 0.310_361_502_442_563_66, 1e-14),
        (30.0, 2.0, 0.972_687_477_518_508_5, 0.027_312_522_481_491_55, 1e-14),
        (30.0, 3.0, 0.997_305_017_967_174_1, 0.002_694_982_032_825_973, 1e-14),
        (30.0, 30.0, 1.0, 3.125_895_815_304_444e-24, 1e-14),
        (1e3, 0.01, 0.503_988_359_033_906_4, 0.496_011_640_966_093_6, 1e-14),
        (1e3, 0.5, 0.691_407_459_583_062_6, 0.308_592_540_416_937_4, 1e-14),
        (1e3, 2.0, 0.977_114_826_753_374_1, 0.022_885_173_246_625_82, 1e-14),
        (1e3, 3.0, 0.998_616_645_477_880_9, 0.001_383_354_522_119_096_3, 1e-14),
        (1e3, 30.0, 1.0, 7.687_343_722_021_741e-142, 1e-13),
        (1e5, 0.01, 0.503_989_346_340_588_5, 0.496_010_653_659_411_6, 1e-14),
        (1e5, 0.5, 0.691_461_911_172_791, 0.308_538_088_827_209, 1e-14),
        (1e5, 2.0, 0.977_248_518_271_246_8, 0.022_751_481_728_753_232, 1e-14),
        (1e5, 3.0, 0.998_649_769_557_967_7, 0.001_350_230_442_032_359_5, 1e-14),
        (1e5, 30.0, 1.0, 3.689_268_436_111_116_6e-197, 1e-13),
        (1e8, 0.01, 0.503_989_356_304_657_6, 0.496_010_643_695_342_5, 1e-14),
        (1e8, 0.5, 0.691_462_460_723_911, 0.308_537_539_276_088_96, 1e-14),
        (1e8, 2.0, 0.977_249_866_702_046_6, 0.022_750_133_297_953_376, 1e-14),
        (1e8, 3.0, 0.998_650_101_635_981_2, 0.001_349_898_364_018_747_2, 1e-14),
        (1e8, 30.0, 1.0, 4.916_682_142_941_633e-198, 1e-13),
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
fn bernoulli_degenerate_p_and_quantile_domain() {
    // p ∈ {0,1}: 0·ln0 := 0, mass finite, no NaN.
    let b0 = Bernoulli::new(0.0).unwrap();
    assert_eq!(b0.mass(0), 1.0);
    assert_eq!(b0.log_mass(0), 0.0);
    assert!(Bernoulli::new(0.3).unwrap().quantile(1.5).is_err());
}

#[test]
fn negbinomial_mean_size_extremes() {
    // μ ≫ θ puts p next to 1; the mean must not go through a rounded 1 − p.
    let nb = NegBinomial::from_mean_size(1e9, 0.01).unwrap();
    assert!((nb.mean().unwrap() / 1e9 - 1.0).abs() < 1e-15);
    assert!((nb.variance().unwrap() / (1e9 + 1e20) - 1.0).abs() < 1e-15);
    // μ + θ overflows while p = μ/(μ + θ) = ½ does not.
    let nb = NegBinomial::from_mean_size(1e308, 1e308).unwrap();
    assert!((nb.mean().unwrap() / 1e308 - 1.0).abs() < 1e-15);
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
/// moments within 5 SE, and goodness of fit (χ² for discrete, KS for
/// continuous). Overridden samplers run on both sides of their switch and at
/// it: Poisson λ = 10, Binomial `n·min(p, 1−p)` = 10, Gamma shape 1. Normal,
/// Exponential and Uniform sample by a closed-form inverse CDF.
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
    fn cont(name: &str, d: &impl Sampler, rng: &mut CommonStatsRng) {
        let draws: Vec<f64> = (0..N).map(|_| d.sample(rng)).collect();
        check_draws(name, d, &draws);
        check_ks_fit(name, d, &draws);
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
        cont(&format!("inversegaussian({mu}, {lam})"), &ig, &mut rng);
    }
    for (shape, rate) in [(0.3, 2.0), (1.0, 0.5), (3.5, 2.0)] {
        let g = Gamma::new(shape, rate).unwrap();
        cont(&format!("gamma({shape}, {rate})"), &g, &mut rng);
    }
    cont("normal(0, 1)", &Normal::new(0.0, 1.0).unwrap(), &mut rng);
    cont("exponential(1)", &Exponential::new(1.0).unwrap(), &mut rng);
    cont(
        "uniform(-2, 5)",
        &Uniform::new(-2.0, 5.0).unwrap(),
        &mut rng,
    );
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

/// Binomial inversion at `n` so large that `pow(1 − p, n)` would carry the
/// rounding of `1 − p` (≈ n·ε in `qⁿ`): every draw terminates, `Binom(10¹⁶,
/// 3·10⁻¹⁶)` has mean `n·p`, and `P(X = 0)` of `Binom(10¹⁷, 10⁻¹⁷)` is `e⁻¹`.
#[cfg(all(feature = "dist", feature = "rng"))]
#[test]
fn binomial_inversion_huge_n_tiny_p() {
    use commonstats::dist::DiscreteSampler;
    use commonstats::rng::CommonStatsRng;
    let b = Binomial::new(10_000_000_000_000_000, 3e-16).unwrap();
    let mut sum = 0.0;
    for seed in 0..40 {
        let mut rng = CommonStatsRng::new(seed, 0);
        sum += (0..250).map(|_| b.sample(&mut rng) as f64).sum::<f64>();
    }
    let (m, draws) = (3.0, 40.0 * 250.0);
    assert!(
        (sum / draws - m).abs() < 5.0 * (m / draws).sqrt(),
        "mean {}",
        sum / draws
    );
    let b = Binomial::new(100_000_000_000_000_000, 1e-17).unwrap();
    let mut rng = CommonStatsRng::new(1, 0);
    let n = 100_000;
    let zeros = (0..n).filter(|_| b.sample(&mut rng) == 0).count() as f64;
    let p0 = (-1.0f64).exp();
    let se = (n as f64 * p0 * (1.0 - p0)).sqrt();
    assert!((zeros - n as f64 * p0).abs() < 5.0 * se, "{zeros} zeros");
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

/// Pearson χ² goodness of fit of integer draws against `mass`, without
/// `cdf` (the hypergeometric `cdf` is an O(support) sum): expected
/// counts from `mass` over the observed range, the two tails beyond it summed
/// from `mass` outward until a term no longer changes the sum (the pmf is
/// unimodal, so the terms only shrink). Asserts p > 0.001.
#[cfg(all(feature = "dist", feature = "rng"))]
fn check_chi2_fit(name: &str, d: &impl DiscreteMass, draws: &[i64]) {
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
        check_chi2_fit(&name, &h, &draws);
    }
    for (big_n, k, n, want) in [(5, 5, 3, 3), (10, 0, 4, 0), (7, 3, 7, 3), (0, 0, 0, 0)] {
        let h = Hypergeometric::new(big_n, k, n).unwrap();
        assert!((0..100).all(|_| h.sample(&mut rng) == want));
    }
}

// ---- studentized range -------------------------------------------------------

#[test]
fn studentizedrange_basic() {
    for k in [
        1.0,
        0.5,
        1e6 * (1.0 + f64::EPSILON),
        f64::NAN,
        f64::INFINITY,
    ] {
        assert!(StudentizedRange::new(k, 10.0).is_err(), "k = {k}");
    }
    for df in [0.0, -1.0, f64::NAN] {
        assert!(StudentizedRange::new(3.0, df).is_err(), "df = {df}");
    }
    let sr = StudentizedRange::new(3.0, f64::INFINITY).unwrap();
    assert_eq!(sr.mean(), None);
    assert_eq!(sr.variance(), None);
}

/// k = 2: `W = √2·|T_ν|`, so `sf(q) = 2·t_ν.sf(q/√2)` and `density(q) =
/// √2·t_ν.density(q/√2)`; at ν = ∞, `erfc(q/2)` and `e^{−q²/4}/√π`. Covers
/// the small-ν and deep-tail regions the mpmath grid samples sparsely.
#[test]
fn studentizedrange_k2_closed_form() {
    use commonstats::special::erfc;
    let r2 = core::f64::consts::SQRT_2;
    let levels = [0.5, 1e-3, 1e-10, 1e-30, 1e-100, 1e-300];
    let nus = [
        0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8,
    ];
    let check = |what: &str, nu: f64, q: f64, got: f64, want: f64| {
        let rel = rel_err(got, want);
        assert!(
            rel <= 1e-12,
            "k = 2, ν = {nu:e}, q = {q:e}: {what} = {got:e}, want {want:e} (rel {rel:e})"
        );
    };
    for nu in nus {
        let t = StudentT::new(nu).unwrap();
        let sr = StudentizedRange::new(2.0, nu).unwrap();
        let mut qs: Vec<f64> = levels
            .iter()
            .map(|&p| r2 * t.isf(p / 2.0).unwrap())
            .filter(|q| q.is_finite())
            .collect();
        qs.push(f64::MAX);
        for q in qs {
            check("sf", nu, q, sr.sf(q), 2.0 * t.sf(q / r2));
            check("density", nu, q, sr.density(q), r2 * t.density(q / r2));
        }
    }
    let n = Normal::new(0.0, 1.0).unwrap();
    let sr = StudentizedRange::new(2.0, f64::INFINITY).unwrap();
    for p in levels {
        let q = r2 * n.isf(p / 2.0).unwrap();
        check("sf", f64::INFINITY, q, sr.sf(q), erfc(q / 2.0));
        let pdf = (-q * q / 4.0).exp() / core::f64::consts::PI.sqrt();
        check("density", f64::INFINITY, q, sr.density(q), pdf);
    }
}

/// k = 2 at df far outside the fixture grid, where the mixing integrand's
/// knee lies far from the Gamma mode (about 2·ln q away at a small ν) or the
/// Gamma weight is far narrower than the range part (a large ν at a huge q):
/// there the quadrature needs more halvings than the bulk does. Truth as in
/// `studentizedrange_k2_closed_form`, and `ln sf_W(q) = ln P(F(1, ν) > q²/2)`.
#[test]
fn studentizedrange_extreme_df_converges() {
    let r2 = core::f64::consts::SQRT_2;
    let check = |what: &str, nu: f64, q: f64, got: f64, want: f64| {
        let rel = rel_err(got, want);
        assert!(
            rel <= 1e-12,
            "k = 2, ν = {nu:e}, q = {q:e}: {what} = {got:e}, want {want:e} (rel {rel:e})"
        );
    };
    for (nu, q) in [(1e-4, 1e25), (1e-4, 1e30), (0.01, 1e28)] {
        let t = StudentT::new(nu).unwrap();
        let sr = StudentizedRange::new(2.0, nu).unwrap();
        check("cdf", nu, q, sr.cdf(q), 1.0 - 2.0 * t.sf(q / r2));
        check("sf", nu, q, sr.sf(q), 2.0 * t.sf(q / r2));
    }
    for (nu, q) in [(1e5, 1778.0), (1e6, 1e5), (1e7, 1e6)] {
        let t = StudentT::new(nu).unwrap();
        let f = FisherF::new(1.0, nu).unwrap();
        let sr = StudentizedRange::new(2.0, nu).unwrap();
        let want = r2.ln() + t.log_density(q / r2);
        check("log_density", nu, q, sr.log_density(q), want);
        check("ln_sf", nu, q, sr.ln_sf(q), f.ln_sf(q * q / 2.0));
    }
}

/// Large ν: the Gamma weight's exponent `−a·(eˣ − 1 − x)` near its mode,
/// where `x ~ 1/√a`, loses `√a·ε` if `eˣ − 1` is recovered from a rounded
/// `a·eˣ` (1e-13 at ν = 3e7, found by the property suite).
#[test]
fn studentizedrange_large_df_closed_form() {
    let r2 = core::f64::consts::SQRT_2;
    for nu in [3.06e7, 1e8, 1e12] {
        let t = StudentT::new(nu).unwrap();
        let sr = StudentizedRange::new(2.0, nu).unwrap();
        for q in [0.5, 2.368, 5.0] {
            let (sf, want) = (sr.sf(q), 2.0 * t.sf(q / r2));
            let (cdf, want_c) = (sr.cdf(q), 1.0 - want);
            assert!(
                rel_err(sf, want) <= 1e-14,
                "ν = {nu:e}, q = {q}: sf {sf:e} vs {want:e}"
            );
            assert!(
                rel_err(cdf, want_c) <= 1e-14,
                "ν = {nu:e}, q = {q}: cdf {cdf:e} vs {want_c:e}"
            );
        }
    }
}

/// ν = 0.0625 at q = 1.8e31: the cdf integrand (≈ 1) has the Gamma mode near
/// `x = 0` and the knee where `F_R(q·e^{x/2})` turns ~144 away, which one
/// quadrature cannot resolve to f64 (found by the property suite at
/// `quantile(0.99)`, 1e-12 off); the cdf there is `1 − sf`.
#[test]
fn studentizedrange_small_df_far_knee() {
    let sr = StudentizedRange::new(2.0, 0.0625).unwrap();
    let t = StudentT::new(0.0625).unwrap();
    let q = 1.811_762_877_867_347_4e31;
    let sf = 2.0 * t.sf(q / core::f64::consts::SQRT_2);
    assert!(
        rel_err(sr.sf(q), sf) <= 1e-14,
        "sf {:e} vs {sf:e}",
        sr.sf(q)
    );
    assert!(
        rel_err(sr.cdf(q), 1.0 - sf) <= 1e-15,
        "cdf {:e} vs {:e}",
        sr.cdf(q),
        1.0 - sf
    );
}

#[test]
fn studentizedrange_edges() {
    let inv_sqrt_pi = 1.0 / core::f64::consts::PI.sqrt();
    for df in [5.0, f64::INFINITY] {
        for k in [1.5, 2.0, 3.0] {
            let sr = StudentizedRange::new(k, df).unwrap();
            assert_eq!(sr.cdf(0.0), 0.0);
            assert_eq!(sr.sf(0.0), 1.0);
            assert_eq!(sr.cdf(-1.0), 0.0);
            assert_eq!(sr.sf(-1.0), 1.0);
            assert_eq!(sr.cdf(f64::INFINITY), 1.0);
            assert_eq!(sr.sf(f64::INFINITY), 0.0);
            assert!(sr.cdf(f64::NAN).is_nan());
            assert!(sr.sf(f64::NAN).is_nan());
            assert!(sr.density(f64::NAN).is_nan());
            assert_eq!(sr.density(-1.0), 0.0);
        }
        assert_eq!(
            StudentizedRange::new(1.5, df).unwrap().density(0.0),
            f64::INFINITY
        );
        assert_eq!(StudentizedRange::new(3.0, df).unwrap().density(0.0), 0.0);
        let want = if df == f64::INFINITY {
            inv_sqrt_pi
        } else {
            core::f64::consts::SQRT_2 * StudentT::new(df).unwrap().density(0.0)
        };
        let got = StudentizedRange::new(2.0, df).unwrap().density(0.0);
        assert!(
            rel_err(got, want) <= 1e-12,
            "k = 2, df = {df}: {got:e} vs {want:e}"
        );
    }
}

/// `F_R(w) ~ w^{k−1}` spreads the mass over many decades of q as k → 1⁺.
#[test]
fn studentizedrange_k_near_one_is_finite_and_monotone() {
    for df in [1.0, f64::INFINITY] {
        let sr = StudentizedRange::new(1.0 + 1e-6, df).unwrap();
        let mut prev = 0.0;
        for q in [1e-300, 1e-10, 1e-3, 1.0, 10.0] {
            let (c, s) = (sr.cdf(q), sr.sf(q));
            assert!(
                c.is_finite() && (0.0..=1.0).contains(&c) && c >= prev,
                "df = {df}, q = {q:e}: cdf {c:e} (previous {prev:e})"
            );
            assert!(
                (c + s - 1.0).abs() <= 1e-15,
                "df = {df}, q = {q:e}: cdf + sf − 1 = {:e}",
                c + s - 1.0
            );
            prev = c;
        }
    }
}

/// No finite-df switch to the df = ∞ formula: the gap to it must still shrink
/// as 1/df at df = 1e8 → 1e9 (a switch makes it 0), and tiny df must stay finite.
#[test]
fn studentizedrange_large_and_tiny_df() {
    // q at sf 1e-3 for df = ∞ (scipy `studentized_range(k, inf).isf(1e-3)`).
    for (k, q) in [
        (2.0, 4.653_507_531_026_965),
        (5.0, 5.483_753_686_172_636),
        (100.0, 7.314_093_840_195_487),
    ] {
        let inf = StudentizedRange::new(k, f64::INFINITY).unwrap();
        let rel = |df: f64| rel_err(StudentizedRange::new(k, df).unwrap().sf(q), inf.sf(q));
        let (r8, r9) = (rel(1e8), rel(1e9));
        assert!(
            r9 > 0.0 && (r8 / r9 - 10.0).abs() <= 0.1,
            "k = {k}: sf vs df = ∞, rel {r8:e} at df = 1e8, {r9:e} at 1e9"
        );
    }
    let sr = StudentizedRange::new(5.0, 1e-3).unwrap();
    for q in [1.0, 1e10, 1e300] {
        let s = sr.sf(q);
        assert!(s > 0.0 && s <= 1.0, "df = 1e-3, q = {q:e}: sf {s:e}");
    }
}

/// `sf(isf(p)) = p` and `cdf(quantile(p)) = p` to 1e-12 times the slope
/// `|d ln mass/d ln q|`, which sets how far a root accurate to ε moves the
/// mass. A root past the f64 range (`p = 1e-300` with a heavy tail) must come
/// back as 0 or ∞, with the mass at the last representable q still beyond p.
/// A lower-tail target has no Bonferroni seed, so Newton starts from 1
/// (k = 100, df = 5, p = 1e-300 is the far case).
#[test]
fn studentizedrange_round_trip() {
    for k in [1.5, 2.0, 5.0, 100.0] {
        for df in [0.5, 5.0, f64::INFINITY] {
            let sr = StudentizedRange::new(k, df).unwrap();
            for p in [1e-300, 1e-10, 0.05, 0.5] {
                for upper in [false, true] {
                    let (q, mass) = if upper {
                        let q = sr.isf(p).unwrap();
                        (q, sr.sf(q))
                    } else {
                        let q = sr.quantile(p).unwrap();
                        (q, sr.cdf(q))
                    };
                    let at = format!("k = {k}, df = {df}, p = {p:e}, upper = {upper}: q = {q:e}");
                    if q == 0.0 {
                        assert!(!upper && sr.cdf(5e-324) >= p, "{at}");
                        continue;
                    }
                    if q == f64::INFINITY {
                        assert!(upper && sr.sf(f64::MAX) >= p, "{at}");
                        continue;
                    }
                    let slope = q * sr.density(q) / p;
                    let rel = rel_err(mass, p);
                    assert!(
                        rel <= 1e-12 * slope.max(1.0),
                        "{at}: mass {mass:e} (rel {rel:e}, slope {slope:e})"
                    );
                }
            }
        }
    }
}
