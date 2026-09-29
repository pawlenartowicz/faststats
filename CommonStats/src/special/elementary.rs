//! Elementary special functions. erf/erfc/gamma via libm (pure-Rust, WASM-clean);
//! lgamma is DiDonato & Morris's `gamln` split (below) so log-gamma is
//! identical wherever the crate uses it. beta/lbeta are gamma identities, with
//! the Stirling-series corrections of DiDonato & Morris (ACM TOMS 18(3), 1992,
//! Algorithm 708) where a plain difference of log-gammas would cancel.

/// Error function erf(x), defined on all reals, range (−1, 1). libm::erf is
/// Boost-derived, ~1e-15. Matches `scipy.special.erf` (`tests/fixtures/erf.json`).
pub fn erf(x: f64) -> f64 {
    libm::erf(x)
}

/// Complementary error function erfc(x) = 1 − erf(x), all reals → (0, 2).
/// Computed directly (not as 1 − erf) to keep accuracy in the right tail.
/// Matches `scipy.special.erfc` (`tests/fixtures/erfc.json`).
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

/// `1/√π`; halving the rounded `2/√π` is exact.
const FRAC_1_SQRT_PI: f64 = 0.5 * core::f64::consts::FRAC_2_SQRT_PI;

/// Scaled complementary error function `e^{z²}·erfc(z)` for `z ≥ 0`, which
/// stays representable where `erfc` underflows (`z ≳ 26.5`); also for small
/// negative `z` (the inverse Gaussian's `erfcx_drop` passes `z > −1/8`), where
/// the direct product below applies unchanged.
///
/// Below 4 the direct product (`e^{z²} ≤ e^{16}`, relative error ≈ `z²·ε`).
/// From 4 up, the Laplace continued fraction
/// `√π·erfcx(z) = 1/(z + (1/2)/(z + (2/2)/(z + (3/2)/(z + …))))`
/// (Abramowitz & Stegun 7.1.14), truncated at depth `⌊4 + 42/z + 130/z²⌋ + 1`
/// (23 at `z = 4`, 9 at 12, 5 from 45 up): a fit to the least depth that holds
/// the truncation to 5e-18 against mpmath. The fraction's value `t ≈ z` is
/// divided into `1/√π`: forming `t·√π` first overflows for `z > f64::MAX/√π ≈
/// 1.014e308`, where erfcx ≈ 1/(z√π) is still representable (subnormal).
/// Within 1.2 ulp (2e-16 relative) for `z ≥ 4` against mpmath, where the result
/// is normal; to the subnormal spacing above `z ≈ 2.5e307`.
pub(crate) fn erfcx(z: f64) -> f64 {
    if z < 4.0 {
        return libm::exp(z * z) * libm::erfc(z);
    }
    let n = (4.0 + (42.0 + 130.0 / z) / z) as usize + 1;
    let mut t = z;
    for k in (1..=n).rev() {
        t = z + 0.5 * k as f64 / t;
    }
    FRAC_1_SQRT_PI / t
}

/// Gamma function Γ(x) via libm::tgamma. Defined on the reals except the
/// non-positive integers (poles). Matches `scipy.special.gamma`
/// (`tests/fixtures/gamma.json`).
pub fn gamma(x: f64) -> f64 {
    libm::tgamma(x)
}

/// Natural log of the gamma function, ln Γ(x), used where Γ would overflow.
///
/// DiDonato & Morris (1992) `gamln`: for 0 < x ≤ 0.8, `ln Γ(1 + x) − ln x`;
/// up to 2.25, `ln Γ(1 + (x − 1))`; below 10, `ln Γ(1 + (t − 1)) + ln((x−1)·…·t)`
/// with `t = x − n ∈ [1.25, 2.25)` by `Γ(z + 1) = z·Γ(z)`; from 10 on,
/// Stirling's `(x − ½)·(ln x − 1) + ½·ln(2π) − ½ + δ(x)`. `ln Γ(1 + a)` is
/// `ln_gamma_1p`, at full relative precision through its zeros, so the values
/// at 1 and 2 are exactly 0 and those at the integers 3…9 are `ln((x − 1)!)`
/// correctly rounded. No reflection for x > 0, so a subnormal x stays finite
/// (`−ln x` ≈ 744). For x < 0 the reflection `ln(π / sin(πx)) − ln Γ(1−x)`,
/// NaN where Γ(x) < 0. Measured against mpmath (the `scripts/accuracy_sweep.py`
/// sweep, x from 5e-324 to 1e300): within 4.8e-16 relative, the worst next
/// to the zero at 2.
/// Matches `scipy.special.gammaln` (`tests/fixtures/lgamma.json`).
pub fn lgamma(x: f64) -> f64 {
    if x < 0.0 {
        return libm::log(core::f64::consts::PI / libm::sin(core::f64::consts::PI * x))
            - lgamma(1.0 - x);
    }
    if x <= 0.8 {
        return ln_gamma_1p(x) - libm::log(x);
    }
    if x <= 2.25 {
        // Exact: Sterbenz up to 2, and on (2, 2.25] the result's ulp divides x's.
        return ln_gamma_1p(x - 1.0);
    }
    if x < 10.0 {
        // Each t − 1 is exact (t ≥ 2.25, so t − 1 > t/2 is a multiple of t's
        // ulp), and so is the product at the integers.
        let mut t = x;
        let mut w = 1.0;
        while t >= 2.25 {
            t -= 1.0;
            w *= t;
        }
        return ln_gamma_1p(t - 1.0) + libm::log(w);
    }
    (x - 0.5) * (libm::log(x) - 1.0) + (LN_SQRT_2PI - 0.5) + stirling_del(x)
}

/// `1/Γ(1 + a) − 1` for `−0.3 ≤ a < 1`, at full relative precision as a → 0,
/// where `1/Γ(1 + a)` rounds to 1 (the role of `gam1` in DiDonato & Morris
/// 1986). From the Taylor series `1/Γ(z) = Σ_{k≥1} c_k z^k` (DLMF 5.7.1):
/// `1/Γ(1 + a) = 1/(a·Γ(a)) = Σ_{k≥1} c_k a^{k−1}`, so the value is
/// `a·Σ_{k≥2} c_k a^{k−2}`; `|c_k| < 2e-18` past k = 27.
pub(crate) fn gam1(a: f64) -> f64 {
    // c_2 … c_27 (c_2 = Euler's γ), from mpmath `taylor(rgamma, 0, 27)`.
    const C: [f64; 26] = [
        0.577_215_664_901_532_9,
        -0.655_878_071_520_253_9,
        -0.042_002_635_034_095_24,
        0.166_538_611_382_291_48,
        -0.042_197_734_555_544_33,
        -0.009_621_971_527_876_973,
        0.007_218_943_246_663_1,
        -0.001_165_167_591_859_065_2,
        -2.152_416_741_149_509_8e-4,
        1.280_502_823_881_162e-4,
        -2.013_485_478_078_824e-5,
        -1.250_493_482_142_670_6e-6,
        1.133_027_231_981_696e-6,
        -2.056_338_416_977_607e-7,
        6.116_095_104_481_416e-9,
        5.002_007_644_469_223e-9,
        -1.181_274_570_487_02e-9,
        1.043_426_711_691_100_5e-10,
        7.782_263_439_905_071e-12,
        -3.696_805_618_642_206e-12,
        5.100_370_287_454_476e-13,
        -2.058_326_053_566_506_6e-14,
        -5.348_122_539_423_018e-15,
        1.226_778_628_238_260_8e-15,
        -1.181_259_301_697_458_8e-16,
        1.186_692_254_751_600_4e-18,
    ];
    // Horner in a⁴ over the four residue classes of k mod 4 (one level of
    // Estrin's scheme): four independent chains, a quarter of the latency.
    let a2 = a * a;
    let a4 = a2 * a2;
    let mut p = [0.0; 4];
    for (k, &c) in C.iter().enumerate().rev() {
        p[k % 4] = p[k % 4] * a4 + c;
    }
    a * ((p[0] + a * p[1]) + a2 * (p[2] + a * p[3]))
}

/// `ln Γ(1 + a)` for `−0.2 ≤ a ≤ 1.25` (the role of `gamln1` in DiDonato &
/// Morris 1992), at full relative precision also next to its zeros a = 0 and
/// a = 1: below 0.7 `−ln(1 + gam1(a))`; from 0.7, with `t = a − 1` (exact),
/// `ln Γ(2 + t) = ln(1 + t) − ln(1 + gam1(t))`, whose two terms are O(t) and
/// cancel by at most a factor 4. The switch at 0.7 is where the two forms'
/// errors cross. Within 4.7ε relative against mpmath (20 000 points), where
/// the published rational fit reaches 10ε.
pub(crate) fn ln_gamma_1p(a: f64) -> f64 {
    if a < 0.7 {
        // `0 −` rather than negation, so ln Γ(1) is +0 like ln Γ(2).
        0.0 - libm::log1p(gam1(a))
    } else {
        let t = a - 1.0;
        libm::log1p(t) - libm::log1p(gam1(t))
    }
}

/// Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b), via the stable log form exp(lbeta).
/// Defined for a, b > 0. See [`lbeta`] for the validated log form.
pub fn beta(a: f64, b: f64) -> f64 {
    libm::exp(lbeta(a, b))
}

/// Log-beta ln B(a,b) = ln Γ(a) + ln Γ(b) − ln Γ(a+b). Defined for a, b > 0.
///
/// DiDonato & Morris (1992) `betaln`: when the larger argument is ≥ 8 the
/// log-gammas are never formed separately (each carries `|ln Γ|·ε` of rounding,
/// ~1e-12 absolute at 700, which the difference keeps). Both ≥ 8: the Stirling
/// form with its correction `bcorr`; only the larger ≥ 8: `ln Γ(a) +
/// ln(Γ(b)/Γ(a+b))` via `algdiv`. Both < 8: the plain log-gamma sum. ~1e-15
/// relative (absolute where ln B ≈ 0).
/// Matches `scipy.special.betaln` (`tests/fixtures/lbeta.json`).
pub fn lbeta(a: f64, b: f64) -> f64 {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    if a >= 8.0 {
        // (a−½)·ln(a/(a+b)) − b·ln(1 + a/b) − ½·ln b + ½·ln 2π + bcorr(a, b).
        let h = a / b;
        let u = -(a - 0.5) * libm::log(h / (1.0 + h));
        let v = b * libm::log1p(h);
        let w = LN_SQRT_2PI - 0.5 * libm::log(b) + bcorr(a, b);
        if u > v { w - v - u } else { w - u - v }
    } else if b >= 8.0 {
        lgamma(a) + algdiv(a, b)
    } else {
        lgamma(a) + lgamma(b) - lgamma(a + b)
    }
}

/// ½·ln(2π).
pub(crate) const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// Coefficients of the Stirling correction `δ(x) = ln Γ(x) − [(x−½)·ln x − x +
/// ½·ln 2π] ≈ Σ_{k=0}^{5} DEL[k]·x^{−(2k+1)}`: the DiDonato & Morris (1992) minimax
/// fit to the asymptotic series, ~4e-16 relative for x ≥ 8.
const DEL: [f64; 6] = [
    0.083_333_333_333_333_3,
    -0.002_777_777_777_609_91,
    7.936_506_668_253_9e-4,
    -5.952_029_313_518_7e-4,
    8.373_080_340_312_15e-4,
    -0.001_653_229_627_807_13,
];

/// Stirling correction `δ(x)` for x ≥ 8 (see `DEL`).
pub(crate) fn stirling_del(x: f64) -> f64 {
    let t = 1.0 / (x * x);
    let mut s = DEL[5];
    for &c in DEL[..5].iter().rev() {
        s = s * t + c;
    }
    s / x
}

/// `δ(b) − δ(a+b)` for b ≥ 8, `0 < a ≤ b` (both callers order their
/// arguments first), as `(c/b)·Σ DEL[k]·s_{2k+1}/b^{2k}` with
/// `x = b/(a+b)`, `c = a/(a+b) = 1 − x` and `s_n = 1 + x + … + x^{n−1}`
/// (from `b^{−n} − (a+b)^{−n} = b^{−n}(1 − x^n)`). The difference is formed
/// without subtracting the two corrections (DiDonato & Morris 1992, `algdiv`).
fn stirling_del_diff(a: f64, b: f64) -> f64 {
    let h = a / b;
    let (c, x) = (h / (1.0 + h), 1.0 / (1.0 + h));
    let x2 = x * x;
    let s3 = 1.0 + (x + x2);
    let s5 = 1.0 + (x + x2 * s3);
    let s7 = 1.0 + (x + x2 * s5);
    let s9 = 1.0 + (x + x2 * s7);
    let s11 = 1.0 + (x + x2 * s9);
    let t = 1.0 / (b * b);
    let w = ((((DEL[5] * s11 * t + DEL[4] * s9) * t + DEL[3] * s7) * t + DEL[2] * s5) * t
        + DEL[1] * s3)
        * t
        + DEL[0];
    w * c / b
}

/// `δ(a) + δ(b) − δ(a+b)` for a, b ≥ 8: the Stirling-correction part of
/// ln B(a,b) (DiDonato & Morris 1992, `bcorr`).
pub(crate) fn bcorr(a: f64, b: f64) -> f64 {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    stirling_del(a) + stirling_del_diff(a, b)
}

/// `ln(Γ(b)/Γ(a+b))` for b ≥ 8, `0 < a ≤ b` (DiDonato & Morris 1992, `algdiv`):
/// `−(a+b−½)·ln(1 + a/b) − a·(ln b − 1) + δ(b) − δ(a+b)`, which never forms the
/// two large log-gammas it is the difference of.
///
/// Where `h = a/b` is subnormal it keeps only its remaining bits (2.4e-7
/// relative in `F(1e-20, 1e300).sf(1)`), so `(b + a − ½)·ln(1 + h)` is taken
/// as `a·(1 + (a − ½)/b)`; the dropped terms are `O(a·h)`. The `δ(b) − δ(a+b)`
/// term also carries `h`, but its share of the result is `O(1/(b²·ln b))`: its
/// error there stays below ε relative for a normal `a`, and a subnormal `a`
/// leaves every term subnormal anyway.
pub(crate) fn algdiv(a: f64, b: f64) -> f64 {
    let h = a / b;
    let u = if h < f64::MIN_POSITIVE {
        a * (1.0 + (a - 0.5) / b)
    } else {
        (b + (a - 0.5)) * libm::log1p(h)
    };
    let v = a * (libm::log(b) - 1.0);
    let w = stirling_del_diff(a, b);
    if u > v { w - v - u } else { w - u - v }
}

#[cfg(test)]
mod tests {
    use super::{erfcx, lgamma};

    /// Exact zeros at 1 and 2, `ln((n − 1)!)` correctly rounded at 3…10, a
    /// subnormal argument (the reflection's `π/sin(πx)` overflows there), and
    /// a point near 1.5, next to lnΓ's minimum (≈ −0.1215 at x ≈ 1.4616),
    /// where the value is small. Truth:
    /// mpmath `loggamma` at 50 dps.
    #[test]
    fn lgamma_integers_subnormal_and_near_its_zeros() {
        assert_eq!(lgamma(1.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(lgamma(2.0).to_bits(), 0.0f64.to_bits());
        for (x, want) in [
            (3.0, core::f64::consts::LN_2),
            (4.0, 1.791_759_469_228_055),
            (5.0, 3.178_053_830_347_945_6),
            (6.0, 4.787_491_742_782_046),
            (7.0, 6.579_251_212_010_101),
            (8.0, 8.525_161_361_065_415),
            (9.0, 10.604_602_902_745_25),
            (10.0, 12.801_827_480_081_469),
            (5e-324, 744.440_071_921_381_3),
            (1.494_170_297_043_844_4, -0.120_979_051_078_413_7),
        ] {
            let got = lgamma(x);
            assert!(
                ((got - want) / want).abs() <= f64::EPSILON,
                "lgamma({x:e}) = {got:e}, want {want:e}"
            );
        }
    }

    /// Large arguments, where `t·√π` overflowed to ∞ and erfcx returned 0 for
    /// z ≳ 1.014e308, and the normal range, whose results the `1/√π` division
    /// moves by at most 2 ulp, almost always toward the truth. Truth: mpmath
    /// at 50 dps, `exp(z²)·erfc(z)` below 1e8, above it the asymptotic series
    /// `1/(z√π)·(1 − 1/(2z²) + 3/(4z⁴) − …)` (A&S 7.1.23), exact to f64 there.
    #[test]
    fn erfcx_matches_mpmath_to_f64_max() {
        for (z, want, tol) in [
            (4.0, 0.136_999_457_625_061_39, 3e-16),
            (7.5, 0.074_573_693_062_876_69, 3e-16),
            (30.0, 0.018_795_888_861_416_75, 3e-16),
            (1e8, 5.641_895_835_477_562e-9, 3e-16),
            (1e300, 5.641_895_835_477_562e-301, 3e-16),
            // Subnormal results: spacing 4.9e-324, ~1.5e-15 relative.
            (1.0139e308, 5.564_548_609_801_325e-309, 2e-15),
            (1.7e308, 3.318_762_256_163_27e-309, 2e-15),
            (f64::MAX, 3.138_408_733_985_445e-309, 2e-15),
        ] {
            let got = erfcx(z);
            assert!(
                ((got - want) / want).abs() < tol,
                "erfcx({z:e}) = {got:e}, want {want:e}"
            );
        }
    }
}
