//! Inverse special functions for closed-form critical values.
//!
//! `norm_ppf` uses P. J. Acklam's rational approximation (Acklam 2003,
//! unpublished) refined by one Halley step to ~1e-15 relative: "An algorithm for
//! computing the inverse normal cumulative distribution function," archived at
//! `http://home.online.no/~pjacklam/notes/invnorm/` (Wayback 2015-10-30).
use crate::special::incomplete::{betai_parts, ln_betai, pq_from_parts};
use crate::special::{betai, lbeta};

/// Inverse complementary error function, for p ∈ (0, 2):
/// erfc_inv(p) = −norm_ppf(p/2) / √2. (libm exposes erf/erfc but no inverse.)
/// Matches `scipy.special.erfcinv` (`tests/fixtures/erfcinv.json`) for
/// p ≥ 2·`f64::MIN_POSITIVE`. Below that scipy's p/2 is subnormal: it rounds for
/// odd multiples of the smallest subnormal (inf at 5e-324; 27.1878 at 1.5e-323,
/// where the true value is 27.1931), and elsewhere carries few significant bits
/// (3e-12 relative error at 1e-315). Here that range works from ln(p/2) and
/// matches mpmath.
pub fn erfc_inv(p: f64) -> f64 {
    // Below 2·MIN_POSITIVE, p/2 is subnormal: halving it can round, and its few
    // significant bits break the Halley step in `norm_ppf`, so work from ln(p/2).
    if p > 0.0 && p < 2.0 * f64::MIN_POSITIVE {
        return -norm_ppf_log_tail(libm::log(p) - core::f64::consts::LN_2)
            / core::f64::consts::SQRT_2;
    }
    -norm_ppf(0.5 * p) / core::f64::consts::SQRT_2
}

/// Inverse error function, for x ∈ (−1, 1): erf_inv(x) = Φ⁻¹((1 + x)/2)/√2, odd
/// in x. Matches `scipy.special.erfinv` (`tests/fixtures/erfinv.json`).
pub fn erf_inv(x: f64) -> f64 {
    // Work from |x|: forming 1 − x would round away the low bits of a small x, and
    // of 1 + x as x → −1. The centre takes q = |x|/2 exactly; in the tail 1 − |x|
    // is exact (Sterbenz).
    let a = x.abs();
    let z = if a <= 1.0 - 2.0 * P_LOW {
        norm_ppf_central(0.5 * a) / core::f64::consts::SQRT_2
    } else {
        erfc_inv(1.0 - a)
    };
    z.copysign(x)
}

/// Acklam's central/tail switch point: the central rational covers
/// p ∈ [P_LOW, 1 − P_LOW].
const P_LOW: f64 = 0.02425;

/// Normal quantile Φ⁻¹(p). Acklam rational seed (~1.15e-9) refined by one Halley
/// step to ~1e-15 relative. Kept private.
fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p < P_LOW {
        norm_ppf_lower(p)
    } else if p <= 1.0 - P_LOW {
        norm_ppf_central(p - 0.5)
    } else {
        // Mirror into the lower tail: 1 − p is exact (Sterbenz), whereas a residual
        // Φ(x) − p taken near 1 has only absolute precision ~1e-16.
        -norm_ppf_lower(1.0 - p)
    }
}

/// Φ⁻¹(p) for p < `P_LOW`: Acklam's tail seed plus one Halley step on the residual
/// Φ(x) − p = ½·erfc(−x/√2) − p, which keeps relative precision as p → 0.
fn norm_ppf_lower(p: f64) -> f64 {
    let x = acklam_lower_tail(libm::sqrt(-2.0 * libm::log(p)));
    if x.is_finite() {
        halley(x, 0.5 * libm::erfc(-x / core::f64::consts::SQRT_2) - p)
    } else {
        x
    }
}

/// Φ⁻¹(½ + q) for |q| ≤ ½ − `P_LOW`, from Acklam's central rational in q.
/// The Halley residual is Φ(x) − ½ − q = ½·erf(x/√2) − q. Formed as Φ(x) − p it
/// would carry only absolute precision ~1e-16 (Φ(x) sits near ½), blind to the
/// seed's relative error once |x| ≲ 1e-7; erf keeps full relative precision as
/// its argument → 0.
fn norm_ppf_central(q: f64) -> f64 {
    // Rational-approximation coefficients (Acklam 2003).
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let r = q * q;
    let x = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    halley(x, 0.5 * libm::erf(x / core::f64::consts::SQRT_2) - q)
}

/// One Halley step for Φ(x) = p given the residual e = Φ(x) − p: with
/// u = e/φ(x), x ← x − u/(1 + x·u/2). Cubic convergence takes Acklam's ~1e-9
/// seed to the residual's own precision.
fn halley(x: f64, e: f64) -> f64 {
    let u = e * libm::sqrt(2.0 * core::f64::consts::PI) * libm::exp(x * x / 2.0);
    x - u / (1.0 + x * u / 2.0)
}

/// Acklam's lower-tail rational seed for Φ⁻¹(p), p < `P_LOW`, in terms of
/// q = √(−2 ln p). Negative; the upper tail is its mirror in 1 − p.
fn acklam_lower_tail(q: f64) -> f64 {
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
        / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
}

/// Φ⁻¹(p) from ln p, for subnormal p (x < −37.5). Newton on ln Φ(x) = ln p with
/// ln Φ(x) = −x²/2 − ln(−x) − ½ ln 2π + ln S(x), where
/// S(x) = Σ (−1)ᵏ (2k−1)!!/x²ᵏ is the asymptotic series of Mills' ratio
/// (Abramowitz & Stegun 1964, 26.2.12); eight terms truncate below 1e-18 at
/// |x| ≥ 37. The slope is d ln Φ/dx = φ/Φ = −x/S. One step suffices: Newton
/// squares the seed's ~1e-9 relative error to below f64 resolution.
fn norm_ppf_log_tail(ln_p: f64) -> f64 {
    let mut x = acklam_lower_tail(libm::sqrt(-2.0 * ln_p));
    let t = 1.0 / (x * x);
    let mut s = 0.0; // Horner in −t over (2k−1)!!, k = 7..0
    for c in [135135.0, 10395.0, 945.0, 105.0, 15.0, 3.0, 1.0, 1.0] {
        s = s * -t + c;
    }
    let ln_phi =
        -0.5 * x * x - libm::log(-x) - crate::special::elementary::LN_SQRT_2PI + libm::log(s);
    x += (ln_phi - ln_p) * s / x;
    x
}

/// Inverse regularized incomplete beta I⁻¹_p(a,b) → t/F/beta quantiles. We invert
/// our own accurate `betai` by bisection rather than depend on `puruspe::invbetai`,
/// which loses ~13 digits in the extreme upper tail (a=1,b=30,p=1-1e-9 → 0.64 vs
/// the true 0.499). For p>0.5 we compare the complement 1 − I_x(a,b) = I_{1-x}(b,a)
/// with 1-p: when both I_x(a,b) and p sit near 1 their difference cancels
/// catastrophically, whereas in the complement the residual stays small. The
/// bisection is on x itself, so a small root keeps its relative precision (solving
/// for 1 − x and subtracting from 1 would not). Bisection (not Newton) because I_x
/// is near-flat for skewed params (y³⁰-style) where Newton crawls linearly —
/// sign-based bisection is immune to that. It runs on the f64 bit patterns
/// (`bisect_bits`), so it reaches every root down to the smallest subnormal; for
/// subnormal p it compares ln I with ln p, and a lower-tail root well below the
/// mean is polished by one Newton step on ln I (`tail_newton`).
///
/// `a`, `b`: shape parameters, > 0. `p`: probability; p ≤ 0 returns 0, p ≥ 1
/// returns 1, and a root below half the smallest subnormal returns 0.
///
/// Accuracy is that of `betai` (or of `lbeta` after the polish) divided by the
/// slope of ln I in ln x, a/((1 − x)·S) in the lower tail. Against mpmath, for
/// a, b ≥ 1 and p from 5e-324 to 1 − 1e-16: median 1e-16, worst ~5e-15
/// relative (the few-ulp error of `lbeta` itself, e.g. 2.7e-15 in ln B(2,3)).
/// It grows as 1/a for small a (~2e-13 at a = 0.01), and a subnormal root is
/// exact to its spacing. Matches `scipy.special.betaincinv`
/// (`tests/fixtures/invbetareg.json`).
pub fn inv_beta_reg(a: f64, b: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    if p > 0.5 {
        // 1 − p is exact (Sterbenz).
        return inv_beta_reg_upper(a, b, 1.0 - p);
    }
    // For subnormal p, `betai` and p both keep only a few significant bits, so
    // compare ln I (`ln_betai`, exact in the log) with ln p.
    let x = if p < f64::MIN_POSITIVE {
        let ln_i = |x: f64| ln_betai(a, b, x, 1.0 - x);
        bisect_bits(ln_i, libm::log(p), f64::NEG_INFINITY, 0.0)
    } else {
        bisect_bits(|x| betai(a, b, x), p, 0.0, 1.0)
    };
    if x > 0.0 && x <= 0.125 && (a + b) * x <= 0.125 * (a + 1.0) {
        tail_newton(a, b, p, x)
    } else {
        x
    }
}

/// Root `x` of `1 − I_x(a, b) = q`, the upper-tail inverse (`Beta::isf`, and
/// [`inv_beta_reg`] for p > ½ with q = 1 − p): the same bisection on x, on
/// `−(1 − I_x(a, b))`, which increases with x, with the complement evaluated
/// directly by the kernel (`betai_parts` with `y = 1 − x`, exact for x ≥ ½),
/// so a small q is never formed as 1 − p. For subnormal q it compares
/// `−ln(1 − I_x(a, b)) = −ln I_{1−x}(b, a)` with `−ln q`. q ≤ 0 returns 1,
/// q ≥ 1 returns 0.
pub(crate) fn inv_beta_reg_upper(a: f64, b: f64, q: f64) -> f64 {
    if q <= 0.0 {
        return 1.0;
    }
    if q >= 1.0 {
        return 0.0;
    }
    if q < f64::MIN_POSITIVE {
        let neg_ln_sf = |x: f64| -ln_betai(b, a, 1.0 - x, x);
        return bisect_bits(neg_ln_sf, -libm::log(q), 0.0, f64::INFINITY);
    }
    let neg_sf = |x: f64| {
        let (m, e, complement) = betai_parts(a, b, x, 1.0 - x);
        -pq_from_parts(m, e, complement).1
    };
    bisect_bits(neg_sf, -q, -1.0, 0.0)
}

/// Root of `f(x) = target` for `f` increasing on [0, 1], `f(0) = f0`,
/// `f(1) = f1`, to the nearest float. The bracket starts from 1 and 2^−1, 2^−2,
/// 2^−4, …, 2^−1024 (repeated squaring, exact), which finds the binade of a root
/// near 1 in one to three evaluations and of any root in at most eleven. It is
/// then halved on the bit patterns: for non-negative f64 the IEEE encoding is
/// monotone in the value, so the integer midpoint bisects geometrically across
/// binades and linearly within one, and ends on two adjacent floats after 52
/// to 62 more steps. (Halving [0, 1] in x stalls at a width of 2^−80 ≈ 8e-25
/// and cannot resolve a smaller root.) Of the two, the one whose `f` is nearer
/// `target`.
fn bisect_bits(f: impl Fn(f64) -> f64, target: f64, f0: f64, f1: f64) -> f64 {
    let (mut lo, mut hi, mut f_lo, mut f_hi) = (0.0_f64, 1.0_f64, f0, f1);
    let mut x = 0.5_f64;
    while x > 0.0 {
        let fx = f(x);
        if fx < target {
            (lo, f_lo) = (x, fx);
            break;
        }
        (hi, f_hi) = (x, fx);
        x *= x;
    }
    let (mut lo, mut hi) = (lo.to_bits(), hi.to_bits());
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let fx = f(f64::from_bits(mid));
        if fx < target {
            (lo, f_lo) = (mid, fx);
        } else {
            (hi, f_hi) = (mid, fx);
        }
    }
    f64::from_bits(if target - f_lo < f_hi - target {
        lo
    } else {
        hi
    })
}

/// One Newton step in u = ln x on g(u) = ln I_x(a,b) − ln p, for a bisected
/// root x ≤ 1/8 with (a+b)·x/(a+1) ≤ 1/8. There `betai` forms a·ln x + b·ln y
/// − ln B (`brcomp`) from ln x rounded to |ln x|·ε, so its root is off by
/// ~|ln x|·ε relative (4e-14 at x = 1e-151), and in the subnormal range by up
/// to a unit. Here ln I comes from DLMF 8.17.8,
///   I_x(a,b) = x^a·y^b/(a·B(a,b))·S,  S = Σ_{n≥0} (a+b)_n/(a+1)_n·xⁿ,  y = 1 − x,
/// whose term ratio (a+b+n)/(a+1+n)·x ≤ 1/8 bounds the sum to ~20 terms, and
/// a·ln x − ln p is formed without the large logarithms rounding: each ln is
/// split as k·ln2_hi (exact) + (k·ln2_lo + ln m), m ∈ [√½, √2) (fdlibm's split
/// of ln 2), and a·k·ln2_hi keeps its rounding error by `fma`. What remains is
/// the rounding of ln B(a,b) and ln(S/a), divided by a in x. The slope is
/// dg/du = x·I′/I = a/(y·S). g is linear in u up to the O(x) change of S and y,
/// so a single step lands at the f64 resolution of the residual.
fn tail_newton(a: f64, b: f64, p: f64, x: f64) -> f64 {
    let mut t = x * (a + b) / (a + 1.0);
    let mut s1 = t; // S − 1
    let mut n = 1.0;
    while t > 1e-17 * s1 {
        t *= x * (a + b + n) / (a + 1.0 + n);
        s1 += t;
        n += 1.0;
    }
    let (hx, lx) = ln_split(x);
    let (hp, lp) = ln_split(p);
    let ahx = a * hx;
    let ahx_err = libm::fma(a, hx, -ahx);
    // ln(S/a) as one log: ln a and ln B(a,b) cancel as a → 0.
    let ln_g = b * libm::log1p(-x) - lbeta(a, b) + libm::log((1.0 + s1) / a);
    let g = (ahx - hp) + (ahx_err + a * lx - lp + ln_g);
    let du = -g * (1.0 - x) * (1.0 + s1) / a;
    if x < f64::MIN_POSITIVE {
        // A subnormal x has too few bits for x + x·(e^du − 1) to round once.
        x * libm::exp(du)
    } else {
        x + x * libm::expm1(du)
    }
}

/// ln x = hi + lo for x > 0 (subnormals included), with hi = k·ln2_hi exact
/// (`LN2_HI` has 21 trailing zero bits, |k| ≤ 1075) and lo = k·ln2_lo + ln m,
/// x = m·2^k, m ∈ [√½, √2), so |lo| < 0.35 carries the only rounding.
fn ln_split(x: f64) -> (f64, f64) {
    // fdlibm e_log.c: ln2_hi + ln2_lo = ln 2 to ~1e-26.
    const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
    const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);
    let (mut m, mut k) = libm::frexp(x);
    if m < core::f64::consts::FRAC_1_SQRT_2 {
        m *= 2.0;
        k -= 1;
    }
    let k = k as f64;
    (k * LN2_HI, k * LN2_LO + libm::log(m))
}

#[cfg(test)]
mod tests {
    use super::{erf_inv, erfc_inv, inv_beta_reg};

    /// Φ⁻¹(p) = −√2·erfc_inv(2p), as `dist::norm_quantile` forms it, keeps full
    /// relative precision near ½ (where the Halley residual must come from erf)
    /// and in both tails. Truth is mpmath at 60 dps for the exact f64 p.
    #[test]
    fn norm_quantile_relative_precision_matches_mpmath() {
        for (p, want) in [
            (0.5 + 1e-12, 2.506_572_823_701_860_5e-12),
            (0.5 - 1e-12, -2.506_572_823_701_860_5e-12),
            (0.5 + 1e-15, 2.504_624_782_204_59e-15),
            (0.5 - 1e-15, -2.504_624_782_204_59e-15),
            (0.5 + 1e-6, 2.506_628_274_705_705_2e-6),
            (0.5 - 1e-6, -2.506_628_274_566_559_4e-6),
            (0.51, 0.025_068_908_258_711_058),
            (0.49, -0.025_068_908_258_711_058),
            (1e-10, -6.361_340_902_404_056),
            (0.001, -3.090_232_306_167_813_5),
            (0.0242, -1.973_839_463_313_199_3),
            (0.99, 2.326_347_874_040_840_8),
            (0.999_999_999_9, 6.361_340_889_697_422),
        ] {
            let got = -core::f64::consts::SQRT_2 * erfc_inv(2.0 * p);
            assert!(
                ((got - want) / want).abs() < 2e-15,
                "Φ⁻¹({p:e}) = {got:e}, want {want:e}"
            );
        }
    }

    /// erf_inv near 0 and near −1 works from |x|, never from 1 − x. Truth is
    /// mpmath `erfinv` at 60 dps.
    #[test]
    fn erf_inv_relative_precision_matches_mpmath() {
        for (x, want) in [
            (1e-20, 8.862_269_254_527_58e-21),
            (-2e-12, -1.772_453_850_905_516e-12),
            (0.3, 0.272_462_714_726_754_4),
            (-0.999_999_999_999_999_9, -5.863_584_748_755_168),
        ] {
            let got = erf_inv(x);
            assert!(
                ((got - want) / want).abs() < 1e-15,
                "erf_inv({x:e}) = {got:e}, want {want:e}"
            );
        }
    }

    /// Subnormal arguments take the ln-space tail; truth is mpmath at 60 dps
    /// (root of ln Φ(x) = ln(p/2), erfc_inv(p) = −x/√2). 5e-324 and 1.5e-323 are
    /// odd multiples of the smallest subnormal, where halving p would round.
    #[test]
    fn erfc_inv_subnormal_matches_mpmath() {
        for (p, want) in [
            (5e-324, 27.213_293_210_812_95),
            (1.5e-323, 27.193_114_126_203_97),
            (1e-320, 27.073_153_719_853_041),
            (1e-315, 26.859_832_753_310_736),
        ] {
            let got = erfc_inv(p);
            assert!(
                ((got - want) / want).abs() < 1e-14,
                "erfc_inv({p:e}) = {got}, want {want}"
            );
        }
        // Φ⁻¹(5e-324) = −√2·erfc_inv(1e-323), as `dist::norm_quantile` forms it.
        let z = -core::f64::consts::SQRT_2 * erfc_inv(1e-323);
        assert!(
            (z + 38.467_405_617_144_346).abs() < 1e-13,
            "Φ⁻¹(5e-324) = {z}"
        );
    }

    /// Deep lower tail, where halving [0, 1] in x would stall at ~4e-25
    /// (4.136e-25 for Beta(2,3) at 1e-300) and a subnormal p is easy to lose
    /// bits from (Beta(50,80): 5e-6 relative at 1e-320, 1.3e-2 at 5e-324); the
    /// p > 0.5 side with a small root, which 1 − I⁻¹(b, a, 1 − p) would leave
    /// with absolute precision only; and one
    /// normal-range point. Truth: mpmath at 60 dps, bisection on ln x of the
    /// DLMF 8.17.22 continued fraction for ln I_x (ln(1 − I) past the reflection
    /// point) to 1e-30, at the exact f64 inputs. The tolerance is set by `lbeta`
    /// (2.7e-15 in ln B(2,3)), which enters x divided by a.
    #[test]
    fn inv_beta_reg_tails_match_mpmath() {
        for (a, b, p, want) in [
            (2.0, 3.0, 1e-300, 4.082_482_904_638_63e-151),
            (50.0, 80.0, 1e-320, 7.494_785_532_881_266e-8),
            (50.0, 80.0, 5e-324, 6.436_280_939_579e-8),
            (2.0, 3.0, 1e-310, 4.082_482_904_638_624e-156),
            (0.5, 0.5, 1e-30, 2.467_401_100_272_340_1e-60),
            (3.0, 1e8, 1e-310, 8.434_326_568_674_218e-112),
            (7.5, 0.25, 1e-200, 3.134_404_537_196_548_7e-27),
            (1.0, 1e6, 1e-300, 1e-306),
            (2.0, 1000.0, 0.99, 6.613_073_955_788_385e-3),
            (4.0, 5000.0, 0.999_999, 0.004_259_712_887_593_807),
            (2.0, 3.0, 0.05, 0.097_611_462_886_414_34),
        ] {
            let got = inv_beta_reg(a, b, p);
            assert!(
                ((got - want) / want).abs() < 3e-15,
                "inv_beta_reg({a}, {b}, {p:e}) = {got:e}, want {want:e}"
            );
        }
        // Subnormal root: I_x(1, 30) = 1 − (1 − x)³⁰ ≈ 30x, true root
        // 3.3333e-322, whose nearest float is 67 units of 2^−1074.
        assert_eq!(inv_beta_reg(1.0, 30.0, 1e-320), 3.3e-322);
        // Roots below the f64 range (mpmath: 1.65e-621 and 3.68e-1001) give 0.
        assert_eq!(inv_beta_reg(0.5, 5.0, 1e-310), 0.0);
        assert_eq!(inv_beta_reg(0.001, 2.0, 0.1), 0.0);
    }
}
