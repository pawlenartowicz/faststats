//! Class-A special-function oracle grids, validated against committed mpmath
//! truth via the shared harness (accuracy-validation spec). Tolerances are the
//! spec §3 bulk ladder: value-kind (erf/gamma/incomplete fns) 1e-12, inverse
//! (quantile-kind) 1e-10. Tail rows assert against a curated band.
mod common;
use common::{Tol, assert_close, assert_monotone, check_grid, measure_band};

use commonstats::special;

// Bulk ladder (spec §3): pdf/cdf/density-like vs inverse/quantile.
const VAL: Tol = Tol {
    rel: 1e-12,
    abs: 1e-14,
};
const INV: Tol = Tol {
    rel: 1e-10,
    abs: 1e-12,
};
// digamma (ψ, the lgamma derivative) is neither value- nor quantile-kind: across
// its grid it achieves ~1e-10 (worst 9.7e-11 at x≈2), mpmath-confirmed and not a
// regression — the pre-rebaseline test already used 1e-10. Tiered on its own.
const DERIV: Tol = Tol {
    rel: 1e-10,
    abs: 1e-12,
};

#[test]
fn erf_grid() {
    check_grid("erf", VAL, |a| special::erf(a[0]));
    assert_monotone("erf", |a| special::erf(a[0]));
}
#[test]
fn erfc_grid() {
    check_grid("erfc", VAL, |a| special::erfc(a[0]));
}
#[test]
fn lgamma_grid() {
    check_grid("lgamma", VAL, |a| special::lgamma(a[0]));
}
#[test]
fn gamma_grid() {
    check_grid("gamma", VAL, |a| special::gamma(a[0]));
}
#[test]
fn digamma_grid() {
    check_grid("digamma", DERIV, |a| special::digamma(a[0]));
}
#[test]
fn gammp_grid() {
    check_grid("gammp", VAL, |a| special::gammp(a[0], a[1]));
}
#[test]
fn gammq_grid() {
    check_grid("gammq", VAL, |a| special::gammq(a[0], a[1]));
}
#[test]
fn gammp_large_shape_near_mode() {
    // Near x ≈ a the series and continued fraction would need ~8·√a terms; this
    // point, just below their x = a + 1 switch, lies in Temme's region. mpmath
    // (50 dps): gammainc(a, 0, x, regularized=True) and gammainc(a, x, inf,
    // regularized=True) at a = 8547.975, x = a + 0.999.
    let (a, x) = (8_547.975, 8_548.974);
    let (p, q) = (0.505_748_610_320_787_6, 0.494_251_389_679_212_4);
    assert!((special::gammp(a, x) / p - 1.0).abs() < 1e-14);
    assert!((special::gammq(a, x) / q - 1.0).abs() < 1e-14);
}
#[test]
fn gamma_incomplete_large_shape_full_precision() {
    // The prefactor x^a·e^{−x}/Γ(a) formed as exp(−x + a·ln x − ln Γ(a)) loses
    // ~a·ε. mpmath (50 dps) gammainc(a, 0, x) / gammainc(a, x, inf), regularized.
    const TOL: Tol = Tol {
        rel: 1e-14,
        abs: 0.0,
    };
    assert_close(
        "P(1e5, 1e5)",
        special::gammp(1e5, 1e5),
        0.500_420_522_110_365_2,
        TOL,
    );
    assert_close(
        "Q(1e5, 100200)",
        special::gammq(1e5, 100_200.0),
        0.263_337_989_994_801_3,
        TOL,
    );
    assert_close(
        "P(1e5, 99800)",
        special::gammp(1e5, 99_800.0),
        0.263_751_142_923_979_1,
        TOL,
    );
    assert_close(
        "Q(8500, 8500)",
        special::gammq(8_500.0, 8_500.0),
        0.498_557_620_198_671_3,
        TOL,
    );
}
#[test]
fn gamma_incomplete_small_shape_full_precision() {
    // For a → 0, Q(a, x) ≈ a·E₁(x) while P rounds to 1, so Q must be formed
    // without `1 − P`. mpmath (60 dps) gammainc(a, 0, x) / gammainc(a, x, inf),
    // regularized, at the exact f64 inputs: (a, x, P, Q).
    const TOL: Tol = Tol {
        rel: 2e-15,
        abs: 0.0,
    };
    let cases = [
        (1e-300, 0.5, 1.0, 5.597_735_947_761_608e-301),
        (1e-100, 1e-5, 1.0, 1.093_571_980_004_369_6e-99),
        (
            1e-10,
            1.05,
            0.999_999_999_979_812_7,
            2.018_728_132_415_909_7e-11,
        ),
        (1e-5, 0.3, 0.999_990_943_217_282_8, 9.056_782_717_213_817e-6),
        (0.01, 0.2, 0.987_784_409_456_754_2, 0.012_215_590_543_245_81),
        (0.3, 0.9, 0.902_252_648_029_669_5, 0.097_747_351_970_330_45),
        (0.9, 0.1, 0.124_895_072_728_741_92, 0.875_104_927_271_258_1),
        (
            0.5,
            1e-20,
            1.128_379_167_095_512_5e-10,
            0.999_999_999_887_162,
        ),
    ];
    for (a, x, p, q) in cases {
        assert_close(&format!("P({a}, {x})"), special::gammp(a, x), p, TOL);
        assert_close(&format!("Q({a}, {x})"), special::gammq(a, x), q, TOL);
    }
}
#[test]
fn gamma_incomplete_small_shape_far_tails() {
    // a < 8. The prefactor `x^a·e^{−x}/Γ(a)` where P or Q is far below 1: one
    // rounded exponent `a·ln x − x − ln Γ(a)` kept |ln P|·ε (1e-13 at P ≈
    // 1e-241). Q near x = 1.1..2, where the continued fraction needs ~100
    // terms and the Lentz product kept 1e-14. The a ≈ 1e-10, 3e-97, 1.6e-10,
    // 0.98, 2.86, 4.59 rows are the worst points of a random sweep (a from
    // 1e-300 to 8, x from 1e-300 to 1e3) against that kernel. Truth: mpmath
    // (420 dps) power series for P and Legendre continued fraction for Q
    // (DLMF 8.7.1, 8.9.2), at the exact f64 inputs.
    const TOL: Tol = Tol {
        rel: 1e-15,
        abs: 0.0,
    };
    for (a, x, q) in [
        (1e-300, 1.1, 1.859_909_045_360_401_3e-301),
        (1e-300, 1.5, 1.000_195_824_066_326_5e-301),
        (1.06, 2.11, 0.133_653_511_292_119_25),
        (50.0, 51.0, 0.425_605_140_483_140_35),
        (4.0, 300.0, 2.340_011_961_912_955e-124),
        (
            2.898_396_167_120_894e-97,
            1.171_870_264_932_034_7,
            4.801_331_579_697_667e-98,
        ),
        (
            1.553_515_879_194_338_5e-10,
            524.098_738_953_190_7,
            7.209_182_025_076_363e-241,
        ),
        (
            4.593_663_886_255_388_5,
            475.946_124_912_283_66,
            6.340_757_162_371_153e-199,
        ),
    ] {
        assert_close(&format!("Q({a}, {x})"), special::gammq(a, x), q, TOL);
    }
    for (a, x, p) in [
        (0.95, 1e-237, 7.224_816_445_398_11e-226),
        (7.5, 1e-30, 7.125_345_439_164_574e-230),
        (
            0.984_769_977_767_129,
            1.009_204_013_398_202_2e-298,
            3.509_367_382_538_39e-294,
        ),
        (
            2.858_548_434_380_435_5,
            9.915_766_788_323_947e-85,
            1.476_366_903_908_707_6e-241,
        ),
    ] {
        assert_close(&format!("P({a}, {x})"), special::gammp(a, x), p, TOL);
    }
}
#[test]
fn gamma_incomplete_huge_shape_near_mean() {
    // Temme's uniform expansion at a up to 1e15, where the series would need
    // ~8·√a terms. Truth: mpmath (50 dps) evaluation of the same expansion
    // (DLMF 8.12.3–8.12.10) to 14 terms in 1/a and 38 in η, which agrees with
    // the hypergeometric series `x^a e^{−x}/Γ(a+1)·₁F₁(1; a+1; x)` to 1e-35 at
    // a = 1e3…1e5: (a, x, P, Q).
    const TOL: Tol = Tol {
        rel: 1e-14,
        abs: 0.0,
    };
    let cases = [
        (1e10, 1e10, 0.500_001_329_807_601_4, 0.499_998_670_192_398_7),
        (
            1e10,
            9_999_700_000.0,
            0.001_349_779_851_443_315_8,
            0.998_650_220_148_556_6,
        ),
        (
            1e10,
            10_000_100_000.0,
            0.841_344_746_072_575_8,
            0.158_655_253_927_424_22,
        ),
        (
            1e10,
            10_000_600_000.0,
            0.999_999_999_012_703_3,
            9.872_967_207_575_07e-10,
        ),
        (
            1e15,
            999_999_970_000_000.0,
            0.171_390_855_842_092_38,
            0.828_609_144_157_907_7,
        ),
        (
            1e15,
            1_000_000_050_000_000.0,
            0.943_076_849_189_447_1,
            0.056_923_150_810_552_894,
        ),
        (1e3, 1030.0, 0.828_911_903_882_394, 0.171_088_096_117_606_03),
        (
            1e5,
            99_000.0,
            7.574_199_211_747_679e-4,
            0.999_242_580_078_825_3,
        ),
    ];
    for (a, x, p, q) in cases {
        assert_close(&format!("P({a}, {x})"), special::gammp(a, x), p, TOL);
        assert_close(&format!("Q({a}, {x})"), special::gammq(a, x), q, TOL);
    }
}
#[test]
fn lbeta_no_cancellation_at_unequal_or_large_arguments() {
    // ln Γ(a) + ln Γ(b) − ln Γ(a+b) keeps the ~|ln Γ(b)|·ε rounding of the two
    // large terms. mpmath (50 dps) log(beta(a, b)).
    const TOL: Tol = Tol {
        rel: 1e-15,
        abs: 1e-15,
    };
    assert_close(
        "lbeta(0.01385, 728.57)",
        special::lbeta(0.01385, 728.57),
        4.180_355_188_400_338_7,
        TOL,
    );
    assert_close(
        "lbeta(0.5, 5e9)",
        special::lbeta(0.5, 5e9),
        -10.593_986_931_740_556,
        TOL,
    );
    assert_close(
        "lbeta(20, 3e4)",
        special::lbeta(20.0, 3e4),
        -166.845_500_987_242_97,
        TOL,
    );
    assert_close(
        "lbeta(5e9, 5e9)",
        special::lbeta(5e9, 5e9),
        -6_931_471_815.500_293,
        TOL,
    );
}
#[test]
fn betai_large_parameters() {
    // The Student-t and F cdfs at large df reach betai with one huge parameter
    // (t, F with large dfd) or two (F with both df large). mpmath (50 dps)
    // betainc(a, b, 0, x, regularized=True) at the exact f64 inputs.
    const TOL: Tol = Tol {
        rel: 1e-14,
        abs: 0.0,
    };
    let cases = [
        (0.5, 5e9, 1e-10, 0.682_689_492_137_085_9),
        (5e4, 0.5, 0.999_96, 0.045_498_644_147_683_22),
        (1.5, 5e5, 9e-7, 0.174_572_397_289_605_1),
        (0.05, 5e5, 1e-7, 0.882_243_526_031_948_2),
        (300.0, 250.0, 0.54, 0.397_732_805_885_375),
        (5e9, 5e9, 0.5, 0.5),
    ];
    for (a, b, x, want) in cases {
        assert_close(
            &format!("betai({a}, {b}, {x})"),
            special::betai(a, b, x),
            want,
            TOL,
        );
    }
    // Near the mean at a, b ~ 1e10, λ = a − (a+b)·x carries (a+b)·x·ε of
    // rounding, which moves I by ~|λ|/(ab/(a+b))·(a+b)·x·ε: ~5e-12 here.
    const LOOSE: Tol = Tol {
        rel: 1e-11,
        abs: 0.0,
    };
    assert_close(
        "betai(5e9, 5e9, 0.49999)",
        special::betai(5e9, 5e9, 0.499_99),
        0.022_750_131_939_972_57,
        LOOSE,
    );
    // x = 1 − 2⁻⁵³, within an ulp of the reflection point: the side test on
    // the rounded x sent it to the fraction that needs λ > −1 at λ = −551, and
    // `1 −` its output was 1 + 6e-14. The ~4e-14 left is λ's rounding, as above.
    assert_close(
        "betai(5e19, 5e3, 1 − 2⁻⁵³)",
        special::betai(5e19, 5e3, 0.999_999_999_999_999_9),
        2.549_303_117_404_974e-14,
        LOOSE,
    );
    assert_close(
        "betai(5e9, 5e9, 0.500004)",
        special::betai(5e9, 5e9, 0.500_004),
        0.788_144_601_414_749_3,
        LOOSE,
    );
}
#[test]
fn betai_small_shape_far_tails() {
    // min(a, b) < 8. `x^a·y^b/B(a, b)` from one rounded exponent kept
    // |ln I|·ε (1e-13 at I ≈ 1e-227), and with one parameter large its
    // `s·ln S` cancelled against the other argument's power. (2.5, 1.5) is
    // FisherF(3, 5).sf(1e40) = I_z(5/2, 3/2), z = 5/(5 + 3e40); the
    // (2.99, 7.82) and (6.3, 6.99) rows are the worst points of a random
    // sweep (a, b in [1, 8), far tails) against the previous kernel. Truth:
    // mpmath (420 dps) continued fraction DLMF 8.17.22 at the exact f64
    // inputs, `y = 1 − x` exact.
    const TOL: Tol = Tol {
        rel: 2e-15,
        abs: 0.0,
    };
    for (a, b, x, want) in [
        (2.5, 3.5, 1e-91, 3.435_807_546_332_981_7e-227),
        (2.5, 3.5, 1e-89, 3.435_807_546_332_982e-222),
        (
            2.5,
            1.5,
            1.666_666_666_666_666_5e-40,
            7.305_534_151_839_996e-100,
        ),
        (1.5, 2.5, 1e-30, 3.395_305_452_627_101e-45),
        (
            2.992_362_828_056_711_6,
            7.817_297_103_149_21,
            3.765_784_844_037_035e-80,
            2.414_331_821_526_522_4e-236,
        ),
        (
            6.295_197_460_069_474,
            6.990_920_786_807_82,
            7.712_148_992_920_3e-8,
            1.856_733_095_010_856_5e-42,
        ),
        (50.0, 0.5, 0.1, 8.380_332_558_688_315e-52),
        (5.0, 2e4, 1e-5, 2.259_336_982_320_672e-6),
        (7.75, 0.25, 0.001, 3.300_837_078_051_684e-25),
        (1e5, 3.0, 0.9995, 2.478_145_963_245_618e-19),
    ] {
        assert_close(
            &format!("betai({a}, {b}, {x})"),
            special::betai(a, b, x),
            want,
            TOL,
        );
    }
}
#[test]
fn betai_grid() {
    check_grid("betai", VAL, |a| special::betai(a[0], a[1], a[2]));
}
#[test]
fn lbeta_grid() {
    check_grid("lbeta", VAL, |a| special::lbeta(a[0], a[1]));
}
#[test]
fn erfcinv_grid() {
    check_grid("erfcinv", INV, |a| special::erfc_inv(a[0]));
}
#[test]
fn erfinv_grid() {
    check_grid("erfinv", INV, |a| special::erf_inv(a[0]));
    assert_monotone("erfinv", |a| special::erf_inv(a[0]));
}
#[test]
fn invbetareg_grid() {
    check_grid("invbetareg", INV, |a| {
        special::inv_beta_reg(a[0], a[1], a[2])
    });
}

/// Band calibration (spec §5). Run with `cargo test -- --ignored --nocapture`;
/// the printed values (lightly padded) are curated by a human into the tail rows'
/// `band` field. Tests never write fixtures.
#[test]
#[ignore]
fn measure_tail_bands() {
    let bands = [
        (
            "erfcinv",
            measure_band("erfcinv", |a| special::erfc_inv(a[0])),
        ),
        (
            "invbetareg",
            measure_band("invbetareg", |a| special::inv_beta_reg(a[0], a[1], a[2])),
        ),
    ];
    for (name, val) in bands {
        match val {
            Some(b) => println!("TAIL BAND  {name:>12} → {b:e}"),
            None => println!("TAIL BAND  {name:>12} → (no tail rows)"),
        }
    }
}

#[test]
fn logsumexp_grid() {
    // Three-source grid (mpmath truth + scipy + R stable logsumexp). Each row's
    // args is the full input slice: logsumexp(&[a0, a1, a2]).
    check_grid("logsumexp", VAL, special::logsumexp);
}
#[test]
fn logsumexp_is_stable() {
    // Naively exp() of these overflows; logsumexp must not.
    let got = special::logsumexp(&[1000.0, 1000.0]);
    assert!((got - (1000.0 + 2.0_f64.ln())).abs() < 1e-9, "got {got}");
    assert_eq!(special::logsumexp(&[]), f64::NEG_INFINITY);
}
#[test]
fn beta_known_values() {
    // Closed-form: B(½,½)=π, B(1,1)=1, B(2,3)=1/12. Guards the exp(lbeta) round-trip,
    // which costs ~2 ulp (B(1,1) lands at 0.999…982) — 1e-13 is the realistic bar.
    assert!(
        (special::beta(0.5, 0.5) - core::f64::consts::PI).abs() < 1e-13,
        "B(.5,.5) {}",
        special::beta(0.5, 0.5)
    );
    assert!(
        (special::beta(1.0, 1.0) - 1.0).abs() < 1e-13,
        "B(1,1) {}",
        special::beta(1.0, 1.0)
    );
    assert!(
        (special::beta(2.0, 3.0) - 1.0 / 12.0).abs() < 1e-13,
        "B(2,3) {}",
        special::beta(2.0, 3.0)
    );
}
