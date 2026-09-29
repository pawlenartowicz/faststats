//! Saddle-point forms of the densities and masses with large shapes, and the
//! exact-arithmetic helpers they need.
//!
//! A density such as `x^{a−1}·e^{−x}/Γ(a)` taken as `exp` of a sum of logs
//! adds terms of size `a·|ln x|` whose total is `O(ln a)`, and keeps their
//! rounding, `ε·a·|ln x|` absolute. About the saddle point the exponent is a
//! deviance instead, `a·(e − ln(1 + e))` with `e` the relative distance from
//! the mode, formed without the large terms: DiDonato & Morris's `rlog1` in
//! the gamma and beta prefactors (`rcomp` of ACM TOMS 12(4), 1986, Algorithm
//! 654; `brcomp` of ACM TOMS 18(3), 1992, Algorithm 708), and Loader's `bd0`
//! in the discrete masses (C. Loader, "Fast and accurate computation of
//! binomial probabilities", 2000; R's `dbinom_raw`, `dpois_raw`). The
//! Stirling remainder `δ` supplies the rest of `ln Γ`.
//!
//! The `Binomial`, `Poisson`, `NegBinomial` and `Hypergeometric` masses keep
//! their exponent as a double-double (Dekker 1971), so its size adds ≈ ε
//! relative to `e^{exponent}` however large it is, where one rounded f64
//! exponent carries `|exponent|·ε`. The error of the exponent's own terms
//! remains: `stirlerr` at a real `NegBinomial` size below 1e-50 adds tens of
//! ε (143ε relative at 1e-307).
#![cfg_attr(not(feature = "dist"), allow(dead_code))]

use crate::special::elementary::{LN_SQRT_2PI, bcorr, stirling_del};

/// `ln 2π`.
pub(crate) const LN_2PI: f64 = 2.0 * LN_SQRT_2PI;

/// A double-double `hi + lo`.
pub(crate) type Dd = (f64, f64);

/// Error-free sum `x + y = s + e` (Knuth's TwoSum).
pub(crate) fn two_sum(x: f64, y: f64) -> (f64, f64) {
    let s = x + y;
    let bb = s - x;
    (s, (x - (s - bb)) + (y - bb))
}

/// Error-free product `x·y = h + l` (Dekker 1971, Veltkamp split), valid while
/// `|x|, |y| < 2⁹⁹⁵` (no overflow in the split).
pub(crate) fn two_prod(x: f64, y: f64) -> (f64, f64) {
    fn split(v: f64) -> (f64, f64) {
        let c = 134_217_729.0 * v; // 2²⁷ + 1
        let hi = c - (c - v);
        (hi, v - hi)
    }
    let h = x * y;
    let ((xh, xl), (yh, yl)) = (split(x), split(y));
    (h, ((xh * yh - h) + xh * yl + xl * yh) + xl * yl)
}

/// `x + y` as a double-double for `|x| ≥ |y|` (Dekker's Fast2Sum).
fn quick_two_sum(x: f64, y: f64) -> Dd {
    let s = x + y;
    (s, y - (s - x))
}

/// Double-double `a + b`, to ≈ ε² of the larger operand (the "sloppy"
/// addition of Hida, Li & Bailey's QD library).
pub(crate) fn dd_add(a: Dd, b: Dd) -> Dd {
    let (s, e) = two_sum(a.0, b.0);
    quick_two_sum(s, e + (a.1 + b.1))
}

/// Double-double `−a`.
pub(crate) fn dd_neg(a: Dd) -> Dd {
    (-a.0, -a.1)
}

/// Double-double `a·b`, to ≈ ε² relative.
pub(crate) fn dd_mul(a: Dd, b: Dd) -> Dd {
    let (p, e) = two_prod(a.0, b.0);
    quick_two_sum(p, e + (a.0 * b.1 + a.1 * b.0))
}

/// Double-double `a/b`, to ≈ ε² relative: one correction step on the f64
/// quotient `q`, whose residual `a − q·b` is exact in its leading part
/// (`q·b` is within a factor 2 of `a`, Sterbenz).
fn dd_div(a: Dd, b: Dd) -> Dd {
    let q = a.0 / b.0;
    let (p, e) = two_prod(q, b.0);
    let r = ((a.0 - p) - e) + (a.1 - q * b.1);
    quick_two_sum(q, r / b.0)
}

/// `ln q` for a double-double `q` whose `hi` is positive and finite, to
/// ≈ 1e-21 relative. `q = 2^k·f`, `f ∈ [√½, √2)` (fdlibm's split, with its
/// `ln 2 = LN2_HI + LN2_LO` to 1.2e-26 and `k·LN2_HI` exact), and `ln f =
/// 2·atanh(w) = 2w·Σ_{j≥0} w^{2j}/(2j+1)`, `w = (f − 1)/(f + 1)`, `|w| ≤
/// 0.172`: the first three terms in double-double, the rest (weight
/// `w⁶/7 ≤ 4e-6` of the sum) in f64.
pub(crate) fn ln_dd(q: Dd) -> Dd {
    const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
    const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);
    // 1/3 and 1/5 as double-doubles (mpmath).
    const THIRD: Dd = (0.333_333_333_333_333_3, 1.850_371_707_708_594e-17);
    const FIFTH: Dd = (0.2, -1.110_223_024_625_156_6e-17);
    let (mut f, mut k) = libm::frexp(q.0);
    if f < core::f64::consts::FRAC_1_SQRT_2 {
        f *= 2.0;
        k -= 1;
    }
    let lo = libm::ldexp(q.1, -k);
    // f − 1 is exact (Sterbenz).
    let num = dd_add((f - 1.0, 0.0), (lo, 0.0));
    let den = dd_add(two_sum(f, 1.0), (lo, 0.0));
    let w = dd_div(num, den);
    let w2 = dd_mul(w, w);
    // Σ_{j≥3} w^{2j−6}/(2j+1) to j = 15: w2^13 < 1e-19.
    let mut t = 0.0;
    for j in (3..=15).rev() {
        t = t * w2.0 + 1.0 / (2 * j + 1) as f64;
    }
    let p = dd_add(FIFTH, dd_mul(w2, (t, 0.0)));
    let p = dd_add(THIRD, dd_mul(w2, p));
    let p = dd_add((1.0, 0.0), dd_mul(w2, p));
    let ln_f = dd_mul((2.0 * w.0, 2.0 * w.1), p);
    let k = k as f64;
    dd_add((k * LN2_HI, k * LN2_LO), ln_f)
}

/// `x − ln(1 + x)` for `x > −1`, at full relative precision near 0 where the
/// direct difference cancels. Range reduction as in DiDonato & Morris (1992)
/// `rlog1`: on `[−0.39, −0.18)` and `(0.18, 0.57]` the argument is shifted so
/// `1 + x = (3/4)·(1 + h)` or `(4/3)·(1 + h)` with `|h| < 0.19`, the constant
/// part going into `w1` (the scale is 3/4 rather than the published 0.7 so the
/// shift is exact in f64). The kernel `h − ln(1 + h) = 2r²/(1 − r) −
/// 2r³·Σ_{k≥0} r^{2k}/(2k + 3)`, `r = h/(h + 2)`, follows from
/// `ln(1 + h) = 2·atanh(r)`; `|r| < 0.11`, so ten terms reach f64 resolution.
pub(crate) fn rlog1(x: f64) -> f64 {
    if !(-0.39..=0.57).contains(&x) {
        return x - libm::log1p(x);
    }
    // −1/4 − ln(3/4) and 1/3 − ln(4/3).
    const W_LO: f64 = 0.037_682_072_451_780_93;
    const W_HI: f64 = 0.045_651_260_881_552_41;
    let (h, w1) = if x < -0.18 {
        let h = (x + 0.25) / 0.75;
        (h, W_LO - 0.25 * h)
    } else if x > 0.18 {
        let h = 0.75 * x - 0.25;
        (h, W_HI + h / 3.0)
    } else {
        (x, 0.0)
    };
    let r = h / (h + 2.0);
    let t = r * r;
    let mut s = 0.0;
    for k in (0..10).rev() {
        s = s * t + 1.0 / (2 * k + 3) as f64;
    }
    2.0 * t * (1.0 / (1.0 - r) - r * s) + w1
}

/// `ln(p/q)` for `p, q > 0`; `ln p − ln q` where the quotient is subnormal and
/// would keep only its bits above the subnormal floor.
fn ln_ratio(p: f64, q: f64) -> f64 {
    let r = p / q;
    if r < f64::MIN_POSITIVE {
        libm::log(p) - libm::log(q)
    } else {
        libm::log(r)
    }
}

/// `e − ln(1 + e)` where `1 + e = p/q` and `e` is formed by the caller to full
/// relative precision: `rlog1(e)` for `|e| ≤ 0.6`, beyond it the direct
/// difference with `ln(p/q)` from `ln_ratio`, which no longer cancels.
fn rlog1_ratio(e: f64, p: f64, q: f64) -> f64 {
    if e.abs() > 0.6 {
        e - ln_ratio(p, q)
    } else {
        rlog1(e)
    }
}

/// `ln(1 + e) − e` for `e > −1`, where `1 + e = p/q` with `ln q = c ≤ 0`,
/// given `r = e/(2 + e)` and `inv_1mr = 1/(1 − r)` (formed by the caller).
/// For `e ∈ [−½, 1]` (`|r| ≤ ⅓`) by the series `−2r²/(1 − r) +
/// 2r³·Σ_{k≥0} r^{2k}/(2k + 3)` (from `ln(1 + e) = 2·atanh r`), which keeps
/// full relative precision as `e → 0`, where the direct difference cancels
/// (the difference loses at most ~2 bits outside it). Else `ln(1 + e)` is
/// `log1p(e)`, whose rounding of `1 + e` costs `~ε/(1 + e)`, or the caller's
/// `ln_1pe() = ln p − c`, which costs `~ε·(|ln p| + |c|)` and holds where
/// `1 + e` leaves the f64 range: the first while `(1 + e)·(1 − c) ≥ 2`.
pub(crate) fn log1pmx(r: f64, inv_1mr: f64, e: f64, c: f64, ln_1pe: impl FnOnce() -> f64) -> f64 {
    // NaN takes this branch.
    if !(-1.0 / 3.0..=1.0 / 3.0).contains(&r) {
        let ln_1pe = if (1.0 + e) * (1.0 - c) >= 2.0 {
            libm::log1p(e)
        } else {
            ln_1pe()
        };
        return ln_1pe - e;
    }
    // 1/(2k + 3), k = 0..17.
    const INV_ODD: [f64; 18] = [
        1.0 / 3.0,
        1.0 / 5.0,
        1.0 / 7.0,
        1.0 / 9.0,
        1.0 / 11.0,
        1.0 / 13.0,
        1.0 / 15.0,
        1.0 / 17.0,
        1.0 / 19.0,
        1.0 / 21.0,
        1.0 / 23.0,
        1.0 / 25.0,
        1.0 / 27.0,
        1.0 / 29.0,
        1.0 / 31.0,
        1.0 / 33.0,
        1.0 / 35.0,
        1.0 / 37.0,
    ];
    let t = r * r;
    // Terms kept: the least `n` with `t^n < 1e-17` over each band of `t`
    // (`t ≤ 1/9`).
    let n = if t < 1e-6 {
        3
    } else if t < 1.5e-3 {
        6
    } else if t < 0.0123 {
        9
    } else if t < 0.038 {
        12
    } else {
        18
    };
    let mut s = 0.0;
    for &c in INV_ODD[..n].iter().rev() {
        s = s * t + c;
    }
    2.0 * t * (r * s - inv_1mr)
}

/// `−a·(e − ln(1 + e))`, `1 + e = x/a`, for `a > 0`, `x > 0` normal: the
/// `x`-dependent exponent of `x^a·e^{−x}/Γ(a) = √(a/2π)·e^{this − δ(a)}`
/// (DiDonato & Morris 1986, `rcomp`), where `−x + a·ln x − ln Γ(a)` would
/// keep the `a·ε` rounding of its terms. `e = (x − a)/a` is exact to ε
/// relative (Sterbenz on `[a/2, 2a]`).
pub(crate) fn gamma_saddle_dev(a: f64, x: f64) -> f64 {
    let e = (x - a) / a;
    -a * rlog1_ratio(e, x, a)
}

/// `λ = a − (a+b)·x`, the distance of `x` below the beta mean `a/(a+b)` in
/// units of `1/(a+b)`, from the smaller of `x` and `y = 1 − x`, which is the
/// one held to relative precision (the other is `1 −` it, rounded to ε/2
/// absolute): `(a+b)·y − b` where `y < x`, else `a − (a+b)·x`. Choosing by
/// `a > b` instead would form λ from the rounded one on the side of ½
/// opposite the mean: at `a ≈ b ≥ 1e32` and `x` one ulp below ½, `1 − y`
/// rounds to ½ and λ to 0, and `Beta(1e35, 1e35).sf(0.49999999999999994)`
/// gives ½ for ≈ 1.
pub(crate) fn beta_lambda(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if y < x {
        (a + b) * y - b
    } else {
        a - (a + b) * x
    }
}

/// `−a·rlog1(−λ/a) − b·rlog1(λ/b)`, `λ = beta_lambda(a, b, x, y)`: the
/// `x`-dependent exponent of `x^a·y^b/B(a,b) = √(ab/((a+b)·2π))·e^{this −
/// bcorr(a,b)}` for `min(a, b) ≥ 8` (DiDonato & Morris 1992, `brcomp`):
/// `x/x₀ = 1 − λ/a` and `y/y₀ = 1 + λ/b` about the mean `x₀ = a/(a+b)`,
/// so no large logarithms are formed and cancel.
pub(crate) fn beta_saddle_dev(a: f64, b: f64, x: f64, y: f64, lambda: f64) -> f64 {
    let (x0, y0) = (a / (a + b), b / (a + b));
    let u = rlog1_ratio(-lambda / a, x, x0);
    let v = rlog1_ratio(lambda / b, y, y0);
    -(a * u + b * v)
}

/// `ln √(ab/((a+b)·2π)) − bcorr(a, b)` for `min(a, b) ≥ 8`: the log of the
/// `x`-free factor of the beta saddle form (see `beta_saddle_dev`).
pub(crate) fn ln_beta_saddle_const(a: f64, b: f64) -> f64 {
    0.5 * libm::log(a * (b / (a + b)) / (2.0 * core::f64::consts::PI)) - bcorr(a, b)
}

/// Stirling error `δ(x) = lnΓ(x+1) − [(x+½)·ln x − x + ½·ln 2π]` for `x > 0`
/// (Loader's `stirlerr`; it is `stirling_del(x)`, since `lnΓ(x+1) = ln x +
/// lnΓ(x)`). `stirling_del` from 8, where its fit holds; integers 1–7 from a
/// table (mpmath); other `x < 8` (a real `NegBinomial` size) as the `lgamma`
/// difference itself, whose terms are ≤ 17 there, so ≈ 4e-15 absolute.
pub(crate) fn stirlerr(x: f64) -> f64 {
    const SMALL: [f64; 8] = [
        0.0,
        0.081_061_466_795_327_26,
        0.041_340_695_955_409_3,
        0.027_677_925_684_998_34,
        0.020_790_672_103_765_093,
        0.016_644_691_189_821_193,
        0.013_876_128_823_070_748,
        0.011_896_709_945_891_77,
    ];
    if x >= 8.0 {
        stirling_del(x)
    } else if x >= 1.0 && x == libm::floor(x) {
        SMALL[x as usize]
    } else {
        crate::special::lgamma(x + 1.0) - (x + 0.5) * libm::log(x) + x - LN_SQRT_2PI
    }
}

/// Deviance term `bd0(x, m) = x·ln(x/m) + m − x ≥ 0` (Loader 2000) as a
/// double-double, for `x ≥ 0`, `m > 0`, given `m` and `d = x − m` as
/// double-doubles, so neither is rounded before the cancellation. For
/// `|d| < 0.1·(x + m)` Loader's series `d·v + 2x·Σ_{j≥1} v^{2j+1}/(2j+1)`,
/// `v = d/(x+m)`, whose terms fall by `v² < 0.01` each: in double-double
/// while a term is ≥ 0.01, the rest in f64. Otherwise `x·ln(x/m) − d`,
/// `ln` by `ln_dd`, which cancels by at most a factor ≈ 10. The error is
/// ≈ ε²·(x + m) plus ε·(the first f64 term) absolute. That work is done
/// only for `1/16 ≤ bd0 ≤ 1e3`, at 2–4× the cost of Loader's f64 form, which
/// is taken outside it (`lo = 0`): below, its error, ≈ 2ε·bd0 (the rounding
/// of `d` enters squared), is already under ε/8 absolute; above, a mass
/// `e^{−bd0}` is far below the f64 range (the rest of a hypergeometric
/// exponent adds at most ≈ 21), so only its log is used, to ≈ 20ε relative
/// (Loader's direct form cancels up to ≈ 10×: 4.3e-15 at
/// `bd0(82175, 64409.73)`).
/// Also for `x + m ≥ 1e290`, beyond the split of `two_prod`.
pub(crate) fn bd0(x: f64, m: Dd, d: Dd) -> Dd {
    if x == 0.0 {
        return m;
    }
    let est = bd0_f64(x, m.0, d.0);
    let s = dd_add((x, 0.0), m);
    if !(0.0625..=1e3).contains(&est) || s.0 >= 1e290 {
        return (est, 0.0);
    }
    if d.0.abs() < 0.1 * s.0 {
        let v = dd_div(d, s);
        let v2 = dd_mul(v, v);
        let mut sum = dd_mul(d, v);
        let mut ej = dd_mul((2.0 * x, 0.0), v);
        let mut j = 3.0;
        loop {
            ej = dd_mul(ej, v2);
            let t = dd_div(ej, (j, 0.0));
            sum = dd_add(sum, t);
            j += 2.0;
            if t.0.abs() < 0.01 {
                break;
            }
        }
        let (mut e, mut tail) = (ej.0, 0.0);
        loop {
            e *= v2.0;
            let t1 = tail + e / j;
            if t1 == tail {
                break;
            }
            tail = t1;
            j += 2.0;
        }
        return dd_add(sum, (tail, 0.0));
    }
    let q = dd_div((x, 0.0), m);
    if !(f64::MIN_POSITIVE..f64::INFINITY).contains(&q.0) {
        return (bd0_f64(x, m.0, d.0), 0.0);
    }
    dd_add(dd_mul(ln_dd(q), (x, 0.0)), dd_neg(d))
}

/// Loader's f64 `bd0(x, m)` with `d = x − m`: the series of `bd0` for
/// `|d| < 0.1·(x + m)` (≈ ε relative), else `x·ln(x/m) − d` (≈ 20ε), with
/// `ln(x/m)` split into `ln x − ln m` where `x/m` leaves the normal range.
pub(crate) fn bd0_f64(x: f64, m: f64, d: f64) -> f64 {
    if d.abs() < 0.1 * (x + m) {
        let v = d / (x + m);
        let v2 = v * v;
        let mut s = d * v;
        let mut ej = 2.0 * x * v;
        let mut j = 3.0;
        // |v| < 0.1: each term is < 1% of the last, ≤ 8 terms to converge.
        loop {
            ej *= v2;
            let s1 = s + ej / j;
            if s1 == s {
                return s;
            }
            s = s1;
            j += 2.0;
        }
    }
    let r = x / m;
    let ln_r = if r.is_finite() && r >= f64::MIN_POSITIVE {
        libm::log(r)
    } else {
        libm::log(x) - libm::log(m)
    };
    x * ln_r - d
}

/// `ln[Γ(a+b+1)/(Γ(a+1)·Γ(b+1)) · s^a·(1−s)^b]` for real `a, b ≥ 0` and `s ∈
/// (0, ½]`: the binomial log pmf of `a` successes in `n = a + b` trials, with
/// `1 − s` taken as exact, in Loader's saddle-point form (R's `dbinom_raw`),
/// as `(E, f)` with the log pmf `E − ½·ln f`:
/// `E = δ(n) − δ(a) − δ(b) − bd0(a, n·s) − bd0(b, n·(1−s))` (`δ` =
/// `stirlerr`) a double-double and `f = 2π·a·b/n`. No term is of size
/// `n·ln n`, where a sum of log-gammas keeps ≈ ε·n·ln n. `n·s = a·s + b·s`
/// and `d = a − n·s` are exact double-doubles (`two_prod`), so `d` keeps its
/// digits near the mode, where it is only ≈ √n against `a ≈ n·s`; `n·(1−s)
/// = b + d`. `a = 0` / `b = 0` are `E = b·ln(1−s)` / `a·ln s`, `f = 1`. For
/// `n ≥ 1e290` (beyond the split of `two_prod`) `d` and `E` are f64.
pub(crate) fn binom_saddle(a: f64, b: f64, s: f64) -> (Dd, f64) {
    if a == 0.0 {
        return (dd_mul(ln_dd(two_sum(1.0, -s)), (b, 0.0)), 1.0);
    }
    if b == 0.0 {
        return (dd_mul(ln_dd((s, 0.0)), (a, 0.0)), 1.0);
    }
    let n = a + b;
    // 2π·a·b/n = 2π·lo/(1 + lo/hi): no overflow for either order.
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let f = 2.0 * core::f64::consts::PI * lo / (1.0 + lo / hi);
    let delta = stirlerr(n) - stirlerr(a) - stirlerr(b);
    if n >= 1e290 {
        let d = a - n * s;
        let e = delta - bd0_f64(a, n * s, d) - bd0_f64(b, n * (1.0 - s), -d);
        return ((e, 0.0), f);
    }
    let m1 = dd_add(two_prod(a, s), two_prod(b, s));
    let d = dd_add((a, 0.0), dd_neg(m1));
    let m2 = dd_add((b, 0.0), d);
    let e = dd_add((delta, 0.0), dd_neg(bd0(a, m1, d)));
    (dd_add(e, dd_neg(bd0(b, m2, dd_neg(d)))), f)
}

/// The binomial log pmf of `binom_saddle`, `E − ½·ln f`, as one f64: its
/// absolute error is ≈ ε·(1 + |result|).
pub(crate) fn ln_binom_saddle(a: f64, b: f64, s: f64) -> f64 {
    let (e, f) = binom_saddle(a, b, s);
    e.0 + (e.1 - 0.5 * libm::log(f))
}

#[cfg(test)]
mod tests {
    use super::{Dd, bd0, ln_dd, two_sum};

    /// `ln_dd` and `bd0` against mpmath at 60 digits, `hi + lo` compared as
    /// the rounded difference to a double-double truth `(t_hi, t_lo)`. `bd0`
    /// on both branches (series `|v| < 0.1`, direct) and below its
    /// double-double window, at a hypergeometric far-tail term (21805,
    /// 17393) where the f64 form is 5e3·ε off.
    #[test]
    fn double_double_matches_mpmath() {
        let close = |got: Dd, want: Dd, tol: f64| {
            let err = (got.0 - want.0) + (got.1 - want.1);
            assert!(
                err.abs() <= tol * f64::EPSILON,
                "got {got:?}, want {want:?}, err {err:e}"
            );
        };
        for (q, want) in [
            (
                (0.1, 0.0),
                (-2.302_585_092_994_045_5, -1.715_024_362_805_798_5e-16),
            ),
            (
                two_sum(1.0, -1e-9),
                (-1.000_000_000_500_000_1e-9, 7.427_230_737_157_087e-26),
            ),
            (
                (1.253_665_267_636_405_5, 0.0),
                (0.226_071_474_868_713_6, -2.418_547_968_851_974_2e-18),
            ),
            (
                (1e300, 0.0),
                (690.775_527_898_213_7, 2.374_766_002_880_024_3e-14),
            ),
        ] {
            close(ln_dd(q), want, 1e-3 * want.0.abs());
        }
        for (x, m, want) in [
            (
                21_805.0,
                17_393.0,
                (517.488_509_512_298_8, 4.106_980_399_990_738e-15),
            ),
            (
                81_853.0,
                86_265.0,
                (114.799_494_162_370_32, -1.458_899_993_057_282_2e-15),
            ),
            (
                30_000.0,
                33_000.0,
                (140.694_605_870_254_2, -8.821_075_843_279_108e-15),
            ),
            (
                1.0,
                700.0,
                (692.448_919_664_956_6, -3.849_689_047_928_307_5e-14),
            ),
            (
                1e6,
                1e6 + 10.0,
                (4.999_966_666_916_665e-5, -2.139_387_760_777_603_5e-21),
            ),
        ] {
            let d = two_sum(x, -m);
            close(bd0(x, (m, 0.0), d), want, 0.05);
        }
    }
}
