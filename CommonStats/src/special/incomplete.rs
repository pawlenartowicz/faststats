//! Regularized incomplete gamma/beta. `gammp`/`gammq` follow DiDonato &
//! Morris, "Computation of the incomplete gamma function ratios and their
//! inverse", ACM TOMS 12(4), 1986 (Algorithm 654): the NR §6.2 series and
//! continued fraction with its prefactor (`rcomp`), its small-shape series
//! (`grat1`) for a < 1, and Temme's uniform expansion near x = a for large a.
//! `betai` follows DiDonato & Morris, "Significant digit computation of the
//! incomplete beta function ratios", ACM TOMS 18(3), 1992 (Algorithm 708): its
//! prefactor (`brcomp`), continued fraction (`bfrac`), large-parameter
//! expansion (`basym`), and for a shape ≤ 1 its power series (`bpser`),
//! recurrence (`bup`) and asymptotic expansion (`bgrat`). All log-gamma
//! routes through the crate's `lgamma`/`lbeta` and their Stirling corrections. Every kernel returns its
//! tail as `m·e^{e}`, so the `ln_*` functions keep the log of a tail that
//! underflows.
//!
//! <!-- accuracy:begin -->
//! ## Accuracy
//!
//! Worst error ratio `r` measured by `scripts/accuracy_sweep.py` against
//! mpmath (43495 points): `r` = relative error / (max(κ, 1)·ε), κ the
//! condition number over all real inputs, ε = 2⁻⁵²; for a log output the
//! absolute error over the sensitivity of the log. `r ≤ 10`: at the accuracy
//! the problem allows; `(review)` up to 1000; `(bug)` above it, a structural error;
//! `(hard)` a probability outside [0, 1], a NaN, a non-monotone cdf or a
//! quantile outside the support. Generated from
//! `scripts/accuracy_baseline.json` by `--update-baseline`; not edited by hand.
//!
//! | Kernel regime | `gammp` | `gammq` |
//! |---|---|---|
//! | Temme (a>=100, x/a in [0.6, 1.4]) | 0.26 | 0.21 |
//! | continued fraction (x>=a+1 or a<1, x>=1.1), a<8 | 0.99 | 0.93 |
//! | continued fraction (x>=a+1 or a<1, x>=1.1), a<8, pow | 0.28 | 0.51 |
//! | continued fraction (x>=a+1 or a<1, x>=1.1), a>=8 | 0.25 | 1 |
//! | series (x<a+1), a<8 | 2.4 | 3 |
//! | series (x<a+1), a<8, pow | 0.84 | 0.84 |
//! | series (x<a+1), a>=8 | 0.76 | 0.76 |
//! | small shape (a<1, x<1.1) | 1.4 | 5.3 |
//!
//! | Kernel regime | `betai` |
//! |---|---|
//! | basym (min>100, near mean) | 0.39 |
//! | bfrac complement, min(a,b)<8 | 3 |
//! | bfrac complement, min(a,b)>=8 | 0.63 |
//! | bfrac, min(a,b)<8 | 2.9 |
//! | bfrac, min(a,b)>=8 | 0.87 |
//! | bgrat (shape<=1, complement) | 1.9 |
//! | bgrat (shape<=1, direct) | 0.035 |
//! | bpser (shape<=1, complement) | 0.58 |
//! | bpser (shape<=1, direct) | 0.91 |
//! | bup+bgrat (shape<=1, complement) | 3.4 |
//! | bup+bgrat (shape<=1, direct) | 1.9 |
//! <!-- accuracy:end -->
use crate::special::elementary::{algdiv, bcorr, erfcx, gam1, stirling_del};
use crate::special::lbeta;
use crate::special::saddle::{beta_lambda, beta_saddle_dev, gamma_saddle_dev, rlog1, two_sum};

/// A kernel's `(m, e, flipped)` as `(v, 1 − v)` in the caller's orientation:
/// with `t = m·e^{e}` the tail it evaluated, `(t, 1 − t)`, or `(1 − t, t)`
/// when `flipped`. `gammp`/`betai` take the first, `gammq` the second.
pub(crate) fn pq_from_parts(m: f64, e: f64, flipped: bool) -> (f64, f64) {
    let t = value_of_parts(m, e);
    if flipped { (1.0 - t, t) } else { (t, 1.0 - t) }
}

/// `m·e^{e}` for a kernel's `(m, e)`. Where `e^{e}` alone is subnormal and
/// `m > 1`, that factor would keep only its remaining bits (4.6e-9 relative in
/// `F(5.3e11, 6.9).cdf(0.0047)` ≈ 1e-310); so below `e = −700` it is
/// `(m·e^{e + 700})·e^{−700}`, `e + 700` exact (Sterbenz) down to −1400,
/// with one rounding into the subnormal range at the end.
pub(crate) fn value_of_parts(m: f64, e: f64) -> f64 {
    // e^{−700}, correctly rounded.
    const EXP_M700: f64 = 9.859_676_543_759_77e-305;
    if e < -700.0 {
        m * libm::exp(e + 700.0) * EXP_M700
    } else {
        m * libm::exp(e)
    }
}

/// `ln` of the quantity a kernel's `(m, e)` describes, `ln m + e`, when
/// `direct`; else `ln` of its complement, `log1p(−m·e^{e})`.
fn ln_from_parts(m: f64, e: f64, direct: bool) -> f64 {
    if direct {
        libm::log(m) + e
    } else {
        libm::log1p(-value_of_parts(m, e))
    }
}

/// Euler's constant γ, the limit of `gam1(a)/a` as a → 0.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// `1/Γ(a)` for `0 < a < 16`, within 1e-15: `a·(1 + gam1(a))` below 1, else
/// `(1 + gam1(f))/((f + 1)(f + 2)…(f + n − 1))`, `n = ⌊a⌋`, `f = a − n` (exact),
/// by `Γ(z + 1) = z·Γ(z)`. `exp(−lgamma(a))` would add the rounding of
/// `ln Γ(a)` (and of its exponential) to the result.
fn rgamma_small(a: f64) -> f64 {
    if a < 1.0 {
        return a * (1.0 + gam1(a));
    }
    let n = libm::floor(a);
    let f = a - n;
    let mut p = 1.0;
    let mut k = 1.0;
    while k < n {
        p *= f + k;
        k += 1.0;
    }
    (1.0 + gam1(f)) / p
}

/// `1/Γ(a + b)` for `a + b < 16`, at the exact sum: `s = a + b` rounds by up
/// to half an ulp `t` (recovered exactly by Knuth's TwoSum), which moves
/// `Γ(s)` by `ψ(s)·t`, 5e-15 relative near s = 16. `1/Γ(s + t) = (1 −
/// ψ(s)·t)/Γ(s)` to O(t²), with `ψ(s) ≈ ln(s + ½) − 1/s` (within 0.12 for
/// s > 0), ample for a correction of order ε.
fn rgamma_sum(a: f64, b: f64) -> f64 {
    let (s, t) = two_sum(a, b);
    let psi = libm::log(s + 0.5) - 1.0 / s;
    rgamma_small(s) * (1.0 - psi * t)
}

/// An exponent `hi + lo` kept to twice f64 precision, for a prefactor
/// `scale·e^{hi + lo}`: each added `p·ln v` keeps its product's rounding (by
/// `fma`) and the sum's (by `two_sum`) in `lo`. `e^{hi}` of a single rounded
/// exponent would carry `|hi|·ε/2` relative, 4e-14 at 700; with `lo` folded
/// into the scale as `1 + lo`, only the `|p·ln v|·ε/2` of each `ln v` remains.
#[derive(Clone, Copy)]
struct Expo {
    hi: f64,
    lo: f64,
}

impl Expo {
    fn add(&mut self, t: f64) {
        let (hi, e) = two_sum(self.hi, t);
        self.hi = hi;
        self.lo += e;
    }
    /// Adds `p·l`.
    fn add_prod(&mut self, p: f64, l: f64) {
        let t = p * l;
        self.lo += libm::fma(p, l, -t);
        self.add(t);
    }
    /// `(scale·(1 + lo), hi)`. For `|hi| < 1024`, `lo` is under 1e-13 (half an
    /// ulp of `hi` plus the `fma` residuals), so its square is below
    /// resolution; past that the value is outside the f64 range, only its log
    /// is used, and `lo` (up to half an ulp of `hi`) is dropped.
    fn apply(self, scale: f64) -> (f64, f64) {
        if self.hi.abs() < 1024.0 {
            (scale * (1.0 + self.lo), self.hi)
        } else {
            (scale, self.hi)
        }
    }
}

/// `|p·ln v|` above which a power `v^p` in a prefactor is taken by `pow` rather
/// than as `e^{p·ln v}` in an `Expo`: there the rounding of `ln v` leaves
/// `|p·ln v|·ε/2` relative (1.8e-15 at 16, a third of that typically), where
/// `pow` rounds `v^p` itself to ~1 ulp (libm's `e_pow` carries `log₂ v` in two
/// pieces) at about twice the cost of `exp` and `log`.
const POW_SWITCH: f64 = 16.0;

/// `(scale, expo)` with `x^a·e^{−x}/Γ(a) = scale·e^{expo}`, the prefactor of
/// both NR §6.2 expansions. For a ≥ 8 it is `√(a/2π)·exp(−a·rlog1(x/a − 1) −
/// δ(a))` (DiDonato & Morris 1986, `rcomp`): `−x + a·ln x − ln Γ(a)` would
/// subtract terms of size `a·ln a` and keep their `a·ε` rounding, 1e-10
/// relative at a = 1e5. For a < 8 it is `x^a/Γ(a)·e^{−x}`, `1/Γ(a)` by
/// `rgamma_small`: `e^{a·ln x − x − ln Γ(a)}` keeps the rounding of its
/// exponent, `(|a·ln x| + x)·ε` relative (6e-14 at x = 1e-237, 5e-14 at
/// x = 500), and that of `ln Γ(a)`. Here the exponent is an
/// `Expo`, with `x^a` moved to the scale by `pow` past `POW_SWITCH`; where
/// `x^a` then leaves the normal range, so does the value (`1/Γ(a) < 1.2`,
/// `e^{−x} ≤ 1`) and the one-f64 exponent keeps its log.
fn gamma_prefactor(a: f64, x: f64, use_pow: bool) -> (f64, f64) {
    if a < 8.0 {
        let l = libm::log(x);
        if a < f64::MIN_POSITIVE {
            // `1/Γ(a) = a·(1 + gam1(a))` would be subnormal, short of bits:
            // its factor `a` goes to the exponent instead, where `ln Q` keeps it.
            return (1.0 + gam1(a), libm::log(a) + (a * l - x));
        }
        let r = rgamma_small(a);
        let mut e = Expo { hi: -x, lo: 0.0 };
        if (a * l).abs() <= POW_SWITCH {
            e.add_prod(a, l);
            return e.apply(r);
        }
        // Skips `pow` where `x^a` would underflow, and where `x` alone
        // pushes `e^{hi} = e^{−x}` below the normal range (x > 708): folding
        // `x^a` into the scale would still leave that `e^{−x}` to be formed
        // on its own at the caller's `m·e^{e}`, at a subnormal's degraded
        // precision (relative spacing ~3e-8 near x = 727) instead of
        // cancelling against `a·ln x` in one exponent.
        let w = if use_pow && a * l > -708.0 && e.hi >= -708.0 {
            libm::pow(x, a)
        } else {
            0.0
        };
        if (f64::MIN_POSITIVE..f64::INFINITY).contains(&w) {
            return e.apply(r * w);
        }
        return (r, a * l - x);
    }
    (
        libm::sqrt(a / (2.0 * core::f64::consts::PI)),
        gamma_saddle_dev(a, x) - stirling_del(a),
    )
}

/// P or Q for `a < 1`, `x < 1.1` (DiDonato & Morris 1986, `grat1`). With
/// `P = x^a/Γ(1 + a)·(1 − j)`, `j = a·Σ_{n≥1} (−1)^{n+1} x^n/(n!·(a + n))`
/// (DLMF 8.7.1 integrated termwise), `Q = 1 − P` is rearranged to
/// `(w·j − l)·g − h` with `w = x^a`, `l = w − 1`, `g = 1/Γ(1 + a)`, `h = g − 1`:
/// every term is O(a) as a → 0, so the small `Q ≈ a·E₁(x)` is never formed as
/// `1 −` a P that rounds to 1. P is taken directly on the published side where
/// it is the smaller tail (`x^a ≤ 0.8746` for `x < ¼`, `a ≥ x/2.59` above),
/// with `x^a` kept in the exponent.
fn gamma_small_shape(a: f64, x: f64, use_pow: bool) -> (f64, f64, bool) {
    // j/a = x/(a+1) − x²/(2(a+2)) + …, a positive sum; x^n/n! < 1e-18 by n = 25.
    let mut c = x;
    let mut sum = x / (a + 1.0);
    for n in 2..30 {
        let n = n as f64;
        c *= -x / n;
        let t = c / (a + n);
        sum += t;
        if t.abs() <= 0.5 * f64::EPSILON * sum {
            break;
        }
    }
    let j = a * sum;
    let ln_x = libm::log(x);
    let z = a * ln_x;
    let h = gam1(a);
    let g = 1.0 + h;
    let q_smaller = if x < 0.25 { z > -0.13394 } else { a < x / 2.59 };
    if q_smaller {
        if a < f64::MIN_POSITIVE {
            // Every term above is O(a), subnormal here: `Q/a` is their limit
            // `E₁(x) = Σ − ln x − γ` (DLMF 6.6.2; the O(a) remainder is below
            // resolution), with the factor `a` in the exponent.
            return (sum - ln_x - EULER_GAMMA, libm::log(a), true);
        }
        let l = libm::expm1(z);
        (((l + 1.0) * j - l) * g - h, 0.0, true)
    } else {
        // `x^a = e^{z}` as an `Expo`, by `pow` past `POW_SWITCH`: `e^{z}` of
        // the rounded `z` carried |z|·ε (6e-14 at x = 1e-237, a = 0.95).
        let v = g * (1.0 - j);
        if z < -POW_SWITCH {
            let w = if use_pow && z > -708.0 {
                libm::pow(x, a)
            } else {
                0.0
            };
            if w >= f64::MIN_POSITIVE {
                return (v * w, 0.0, false);
            }
            return (v, z, false);
        }
        let mut e = Expo { hi: 0.0, lo: 0.0 };
        e.add_prod(a, ln_x);
        let (v, e) = e.apply(v);
        (v, e, false)
    }
}

/// P (`x < a`) or Q (`x ≥ a`) for `a ≥ 100`, `|x/a − 1| ≤ 0.4`, by Temme's
/// uniform expansion (N. M. Temme, "The asymptotic expansion of the incomplete
/// gamma functions", SIAM J. Math. Anal. 10(4), 1979; DLMF 8.12.3–8.12.10;
/// DiDonato & Morris 1986 use it on the same `x/a` range from a ≥ 20, where it
/// needs more terms in 1/a): with `φ = x/a − 1 − ln(x/a)`,
/// `η = sign(x − a)·√(2φ)` and `y = a·φ`,
/// `Q = ½·erfc(η·√(a/2)) + e^{−y}/√(2πa)·Σ_k c_k(η)/a^k` and
/// `P = ½·erfc(−η·√(a/2)) − (the same sum)`. Near x = a the series and the
/// continued fraction need O(√a) terms (1e6 at a = 1e10); here the cost is
/// fixed. The smaller tail takes erfc of `√y ≥ 0`; the sum adds to it for P
/// and subtracts at most ~12% of it for Q, so neither cancels. Where `y ≥ 16`,
/// `e^{−y}` goes to the exponent and `erfc` to `erfcx`.
fn gamma_temme(a: f64, x: f64) -> (f64, f64, bool) {
    // Taylor coefficients in η of c_0 … c_7, computed exactly from the
    // recursion of DLMF 8.12.9–8.12.10. Row k is cut where its terms fall below
    // 1e-18·a^k for |η| ≤ 0.48 (x/a = 0.6) and a ≥ 100; c_8/a^8 < 1e-18.
    const TEMME: [&[f64]; 8] = [
        &[
            -0.333_333_333_333_333_3,
            0.083_333_333_333_333_33,
            -0.014_814_814_814_814_815,
            0.001_157_407_407_407_407_3,
            3.527_336_860_670_194e-4,
            -1.787_551_440_329_218e-4,
            3.919_263_178_522_438e-5,
            -2.185_448_510_679_992e-6,
            -1.854_062_210_715_16e-6,
            8.296_711_340_953_087e-7,
            -1.766_595_273_682_607_8e-7,
            6.707_853_543_401_498e-9,
            1.026_180_978_424_030_9e-8,
            -4.382_036_018_453_353e-9,
            9.147_699_582_236_79e-10,
            -2.551_419_399_494_624_8e-11,
            -5.830_772_132_550_426e-11,
            2.436_194_802_066_741_5e-11,
            -5.027_669_280_114_175_5e-12,
        ],
        &[
            -0.001_851_851_851_851_852,
            -0.003_472_222_222_222_222,
            0.002_645_502_645_502_645_4,
            -9.902_263_374_485_596e-4,
            2.057_613_168_724_279_8e-4,
            -4.018_775_720_164_609e-7,
            -1.809_855_033_448_997_7e-5,
            7.649_160_916_081_11e-6,
            -1.612_090_089_456_344_6e-6,
            4.647_127_802_807_434e-9,
            1.378_633_446_915_721e-7,
            -5.752_545_603_517_705e-8,
            1.195_162_859_977_814_8e-8,
            -1.754_324_171_974_764_7e-11,
            -1.009_154_371_060_041_3e-9,
            4.162_792_991_842_583e-10,
            -8.563_907_026_492_98e-11,
        ],
        &[
            0.004_133_597_883_597_883,
            -0.002_681_327_160_493_827_3,
            7.716_049_382_716_049e-4,
            2.009_387_860_082_304_7e-6,
            -1.073_665_322_636_516e-4,
            5.292_344_882_912_012_5e-5,
            -1.276_063_518_861_872_8e-5,
            3.423_578_734_096_138e-8,
            1.372_195_730_906_293_4e-6,
            -6.298_992_138_380_055e-7,
            1.428_061_420_606_424_2e-7,
            -2.047_709_842_199_086_6e-10,
            -1.409_252_991_086_752e-8,
            6.228_974_084_922_022e-9,
            -1.367_048_839_661_711_4e-9,
        ],
        &[
            6.494_341_563_786_008e-4,
            2.294_720_936_213_991_7e-4,
            -4.691_894_943_952_557e-4,
            2.677_206_320_628_388_5e-4,
            -7.561_801_671_883_977e-5,
            -2.396_505_113_867_297e-7,
            1.108_265_411_534_730_2e-5,
            -5.674_952_826_991_596_5e-6,
            1.423_090_073_243_588_3e-6,
            -2.786_108_029_152_814_3e-11,
            -1.695_840_409_193_027_8e-7,
            8.099_464_905_388_083e-8,
            -1.911_116_848_597_365_5e-8,
        ],
        &[
            -8.618_882_909_167_117e-4,
            7.840_392_217_200_666e-4,
            -2.990_724_803_031_902e-4,
            -1.463_845_257_884_341_8e-6,
            6.641_498_215_465_122e-5,
            -3.968_365_047_179_435e-5,
            1.137_572_697_067_841_9e-5,
            2.507_497_226_237_533e-10,
            -1.695_414_953_655_830_5e-6,
            8.907_507_532_205_309e-7,
            -2.292_934_834_000_805e-7,
        ],
        &[
            -3.367_985_533_663_581_3e-4,
            -6.972_813_758_365_857e-5,
            2.772_753_244_959_392e-4,
            -1.993_257_051_618_884_7e-4,
            6.797_780_477_937_208e-5,
            1.419_062_920_643_967e-7,
            -1.359_404_818_976_869_3e-5,
            8.018_470_256_334_202e-6,
        ],
        &[
            5.313_079_364_639_922e-4,
            -5.921_664_373_536_939e-4,
            2.708_782_096_718_045e-4,
            7.902_353_232_660_328e-7,
            -8.153_969_367_561_969e-5,
            5.611_682_753_106_25e-5,
        ],
        &[3.443_676_068_923_776_5e-4],
    ];
    let upper = x >= a;
    // x − a is exact on [0.6a, 1.4a] (Sterbenz).
    let phi = rlog1((x - a) / a);
    let y = a * phi;
    let eta = libm::sqrt(2.0 * phi);
    let eta = if upper { eta } else { -eta };
    let mut sum = 0.0;
    for row in TEMME.iter().rev() {
        let mut c = 0.0;
        for &d in row.iter().rev() {
            c = c * eta + d;
        }
        sum = sum / a + c;
    }
    let s = sum / libm::sqrt(2.0 * core::f64::consts::PI * a);
    let s = if upper { s } else { -s };
    let r = libm::sqrt(y);
    if y < 16.0 {
        (0.5 * libm::erfc(r) + libm::exp(-y) * s, 0.0, upper)
    } else {
        (0.5 * erfcx(r) + s, -y, upper)
    }
}

/// Kernel shared by [`gammp`], [`gammq`], [`ln_gammp`] and [`ln_gammq`]:
/// `(m, e, upper)` with `m·e^{e}` equal to `Q(a, x)` when `upper`, else
/// `P(a, x)`. Each branch yields the tail that is naturally small, so a caller
/// complements at most once and never forms `1 − (1 − small)`. For a < 1 and
/// x < 1.1, `gamma_small_shape`; for a ≥ 100 within 0.4·a of the mean,
/// `gamma_temme`; elsewhere the NR §6.2 series (`x < a+1`: P) or the
/// continued fraction (Q), which also takes a < 1, x ≥ 1.1 (Q ≤ e^{−1.1}
/// there, so `1 − Q` is well conditioned and Q needs no `1 − P`). `ln_of`:
/// `Some(upper)` when the caller takes only the log of that tail (`ln_gammq`:
/// `Some(true)`); where the branch evaluates it, the prefactor skips `pow`, as
/// `ln m + e` of the one-f64 exponent already keeps ~ε relative in the log.
fn gamma_pq(a: f64, x: f64, ln_of: Option<bool>) -> (f64, f64, bool) {
    // Near x ≈ a both expansions need O(√a) terms: the series terms fall off
    // like e^{−n²/(2a)}, and the series stop below needs them under ε·n/a,
    // ~8·√a terms. For a ≥ 100 that region is `gamma_temme`'s; the cap grows
    // with √a so the rest is never truncated. `bgrat` hardcodes this cap at
    // a = 1 as 212 — change together.
    let maxit = 200 + (12.0 * libm::sqrt(a)) as usize;

    if x.is_infinite() {
        // Q(a, ∞) = 0; the prefactor `−x + a·ln x` would be `−∞ + ∞` = NaN.
        return (0.0, 0.0, true);
    }
    if a < 1.0 && x < 1.1 {
        return gamma_small_shape(a, x, ln_of != Some(false));
    }
    if a >= 100.0 && libm::fabs(x - a) <= 0.4 * a {
        return gamma_temme(a, x);
    }
    let upper = x >= a + 1.0 || (a < 1.0 && x >= 1.1);
    let (scale, expo) = gamma_prefactor(a, x, ln_of != Some(upper));
    if !upper {
        // Series: P(a, x) = e^{-x} x^a / Γ(a+1) · Σ_{n≥0} x^n / (a+1)_n.
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut term = sum;
        for _ in 0..maxit {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            // Later ratios x/(ap + k) are all below r = x/(ap + 1), so the
            // tail is under term·r/(1 − r); stopping on term < EPS·sum instead
            // leaves ~(a/n)·EPS behind, 1e-13 at a = 1e5.
            if term * x < sum * (0.5 * f64::EPSILON) * (ap + 1.0 - x) {
                break;
            }
        }
        (sum * scale, expo, false)
    } else {
        (gamma_cf(a, x, maxit) * scale, expo, true)
    }
}

/// `Q(a, x)` over its prefactor `x^a·e^{−x}/Γ(a)`, by the Legendre continued
/// fraction, in at most `maxit` terms; for `x ≥ a + 1`, or `x ≥ 1.1` with
/// `a < 1` (DiDonato & Morris 1986 take it there too; `bgrat` also takes it at
/// `a = 1`).
fn gamma_cf(a: f64, x: f64, maxit: usize) -> f64 {
    const FPMIN: f64 = 1.0e-300;
    // h = 1/(b₀ + a₁/(b₁ + a₂/(b₂ + …))),
    // b_n = x + 1 − a + 2n, a_n = −n(n − a) (NR §6.2, DLMF 8.9.2), summed
    // as the series of its convergents' differences (Euler–Wallis): with
    // d_n = B_{n−1}/B_n, δ_n = h_n − h_{n−1} = −δ_{n−1}·a_n·d_{n−1}·d_n.
    // The modified Lentz product h·Π c_n·d_n keeps every factor's
    // rounding, 1e-14 over the ~100 terms needed near x = 1.1; the sum,
    // compensated by TwoSum, holds the whole to ~1 ulp. The stop leaves a
    // tail under δ_n·r/(1 − r), r (the ratio of successive δ) < 0.8 for
    // x ≥ 1.1, so below ε/2. A NaN argument stops at once.
    let b0 = x + 1.0 - a;
    let mut d = 1.0 / b0;
    let mut delta = d;
    let mut h = d;
    let mut comp = 0.0;
    for n in 1..maxit {
        let n = n as f64;
        let an = -n * (n - a);
        let mut den = an * d + (b0 + 2.0 * n);
        if den.abs() < FPMIN {
            den = FPMIN;
        }
        let dn = 1.0 / den;
        delta = -delta * an * d * dn;
        d = dn;
        let (sum, err) = two_sum(h, delta);
        comp += err;
        h = sum;
        if delta.abs() <= 0.125 * f64::EPSILON * h.abs() || delta.is_nan() {
            break;
        }
    }
    h + comp
}

/// Regularized lower incomplete gamma P(a,x), for a > 0 and x ≥ 0; range [0, 1].
/// (DiDonato & Morris 1986: NR §6.2 series for x < a+1, the Legendre continued
/// fraction otherwise, with their prefactor; their `grat1` series for a < 1,
/// x < 1.1, and the continued fraction (Q, so P = 1 − Q ≥ 0.67) for a < 1,
/// x ≥ 1.1; Temme's expansion for a ≥ 100, |x/a − 1| ≤ 0.4. For a < 8,
/// `gamma_prefactor`'s own bound `(|a·ln x| + x)·ε` relative, also in the far
/// tails (1.7e-13 at a = 7.9, x = 727); for a ≥ 8 a few `|ln P|·ε` where P is
/// small, the rounding of the prefactor's exponent.)
/// Matches `scipy.special.gammainc` (`tests/fixtures/gammp.json`).
pub fn gammp(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    let (m, e, upper) = gamma_pq(a, x, None);
    pq_from_parts(m, e, upper).0
}

/// Regularized upper incomplete gamma Q(a,x) = 1 − P(a,x), for a > 0 and x ≥ 0.
/// Evaluated directly wherever it is the smaller tail (continued fraction for
/// `x ≥ a+1` and for `a < 1`, `x ≥ 1.1`, and the small-shape and Temme
/// branches of [`gammp`]), so the deep upper tail keeps [`gammp`]'s own
/// accuracy rather than losing it to `1 − P` (χ² / Gamma `sf` and `isf` rely
/// on this), as does Q ≈ a·E₁(x) at tiny a. Used by χ² p-values.
/// Matches `scipy.special.gammaincc` (`tests/fixtures/gammq.json`).
pub fn gammq(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return 1.0;
    }
    let (m, e, upper) = gamma_pq(a, x, None);
    pq_from_parts(m, e, upper).1
}

/// `ln P(a, x)`, the natural log of [`gammp`], to full relative precision in the
/// log also where `P` is subnormal or below the f64 range:
/// on the lower-tail side it is `ln m + e` from the kernel, never `ln` of an
/// exponentiated value; on the other side `log1p(−Q)`. `−∞` for x ≤ 0.
// For the `dist` quantile solvers, which need the log of tails below 1e-308.
#[cfg_attr(not(feature = "dist"), allow(dead_code))]
#[doc(hidden)]
pub fn ln_gammp(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let (m, e, upper) = gamma_pq(a, x, Some(false));
    ln_from_parts(m, e, !upper)
}

/// `ln Q(a, x)`, the natural log of [`gammq`], the mirror of [`ln_gammp`]:
/// `ln m + e` on the upper-tail side, `log1p(−P)` on the other. At a
/// subnormal `a` the factor `a` of `Q ≈ a·E₁(x)` is kept in `e`, so the log
/// stays finite where `Q` is below 2⁻¹⁰⁷⁴. `0` for x ≤ 0, `−∞` for x = ∞.
#[cfg_attr(not(feature = "dist"), allow(dead_code))]
#[doc(hidden)]
pub fn ln_gammq(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    let (m, e, upper) = gamma_pq(a, x, Some(true));
    ln_from_parts(m, e, upper)
}

/// Regularized incomplete beta I_x(a,b), for a,b > 0 and x ∈ [0, 1]; range [0, 1].
///
/// DiDonato & Morris (1992): for `min(a, b) > 100` within `0.03·min(a, b)` of
/// the mean in `λ = a − (a+b)·x` units, the asymptotic expansion `basym`;
/// elsewhere the continued fraction `bfrac` on the side below NR §6.4's
/// reflection point `x = (a+1)/(a+b+2)`, the other side as its complement.
/// With a shape ≤ 1, where that complement would be `1 −` a value near 1 and
/// their `bratio` evaluates the side itself, its `bpser`, `bup` and `bgrat`
/// do so here (a small `I` at `b ≪ 1` above the reflection point).
/// Measured against mpmath (`scripts/accuracy_sweep.py`, normal-range
/// values): for `min(a, b) < 8` within 7e-15 relative, also in the far
/// tails (the worst where `1 − x` is 2e-9), and within 1.6e-15 on the
/// shape-≤-1 branches. For `min(a, b) ≥ 8` the error follows the
/// conditioning of `I` in `x`, not `|ln I|`: the rounding of `λ`, formed from
/// the rounded `x`, reaches 2.9e-12 relative at a, b ≈ 1.5e5 (`I` ≈ 3e-296)
/// and 5.6e-8 at a = b = 1e15 within 4e-7 of `x = ½` (`I` ≈ 1e-200).
/// Matches `scipy.special.betainc` (`tests/fixtures/betai.json`).
pub fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let (m, e, complement) = betai_parts(a, b, x, 1.0 - x);
    pq_from_parts(m, e, complement).0
}

/// `ln I_x(a, b)`, the natural log of [`betai`], to full relative precision in
/// the log also where `I` is subnormal or below the f64 range: on the side the
/// kernel evaluates directly it is `ln m + e`, never `ln` of an exponentiated
/// value; on the other side `log1p(−(1 − I))`. `y = 1 − x`, formed by the
/// caller where it is more accurate than `1 − x` (the upper tail `1 − I_x(a, b)
/// = I_y(b, a)` is `ln_betai(b, a, y, x)`). `−∞` for x ≤ 0, `0` for y ≤ 0.
/// On the continued-fraction (`bfrac`) side the prefactor skips `pow` for
/// `a ≥ 1` (`ln m + e` of its one-f64 exponent keeps ~ε relative in the log)
/// and takes it for `a < 1`, where that exponent's `ln a` would cancel
/// against the fraction's `1/a` and leave `|ln a|·ε` absolute in an O(1) log.
/// Measured against mpmath (`scripts/accuracy_sweep.py`): error ratio r ≤ 9.4
/// on every branch but the complement side with `min(a, b) < 8`, which
/// reaches r 12 where `log1p(−(1 − I))` is a small log (1.1e-14 relative at
/// `ln I_{1.01e-10}(0.001, 1e10)` ≈ −2.2e-4).
#[doc(hidden)]
pub fn ln_betai(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if y <= 0.0 {
        return 0.0;
    }
    let (m, e, complement) = betai_kernel(a, b, x, y, true);
    ln_from_parts(m, e, !complement)
}

/// Kernel shared by [`betai`], [`ln_betai`] and the t and F distributions,
/// for `x, y ∈ [0, 1]`, `y = 1 − x`: `(m, e, complement)` with `m·e^{e}` equal
/// to `I_x(a, b)`, or with `complement` to `1 − I_x(a, b)`, whichever the
/// branch evaluates directly. It takes `y` as well as `x`, as TOMS 708
/// `bratio` does (DiDonato & Morris 1992): where `x` is near 1, `1 − x`
/// formed from the rounded `x` keeps only its absolute error, and a caller
/// that forms `y` by its own division (`x²/(df + x²)` beside `df/(df + x²)`)
/// keeps the small side's full relative precision.
pub(crate) fn betai_parts(a: f64, b: f64, x: f64, y: f64) -> (f64, f64, bool) {
    betai_kernel(a, b, x, y, false)
}

/// [`betai_parts`]; with `ln_i` the caller takes only `ln I` (`ln_betai`),
/// so where `bfrac` evaluates `I` itself with `a ≥ 1`, `brcomp` skips `pow`:
/// `ln m + e` of its one-f64 exponent already keeps ~ε relative in the log.
/// For `a < 1` it does not (see the `bfrac` call below).
fn betai_kernel(a: f64, b: f64, x: f64, y: f64, ln_i: bool) -> (f64, f64, bool) {
    let lambda = beta_lambda(a, b, x, y);
    let min_ab = a.min(b);
    if min_ab > 100.0 && lambda.abs() <= 0.03 * min_ab {
        // λ ≥ 0 is the side x ≤ a/(a+b), whose mass is the smaller one.
        let ((m, e), complement) = if lambda >= 0.0 {
            (basym(a, b, lambda), false)
        } else {
            (basym(b, a, -lambda), true)
        };
        return (m, e, complement);
    }
    // Below NR's reflection point `x < (a+1)/(a+b+2)`, that is `λ > 2x − 1`.
    // Tested on λ, formed above from whichever of `x`, `y` holds it to
    // relative precision: where `x` is within an ulp of 1 (`y` tiny, both
    // shapes huge) the ratio rounds onto `x`, and a test on `x` would send it
    // to the side whose fraction needs `λ > −1` with λ ≈ −500 and return `1 −`
    // garbage (1 + 6e-14 for 2.5e-14 at betai(5e19, 5e3, 1 − 2⁻⁵³)). This way each
    // `bfrac` gets the `λ > −1` it needs from the same rounded λ.
    let lower = lambda > 2.0 * x - 1.0;
    // With a shape ≤ 1 DiDonato & Morris's `bratio` may evaluate `1 −
    // I_{x₀}(a₀, b₀)`, `x₀ = min(x, y)`, directly. Where `bfrac`'s side is
    // the other one, it would leave that value as `1 −` its complement, with
    // ε absolute (a small `I` at `b ≪ 1` above the reflection point, or the
    // `ln I` of an `I` near 1 at `a ≪ 1` below it); there `bratio`'s route is
    // taken instead. Where both take the same side, `bfrac` stays.
    if min_ab <= 1.0 && lower != (x > 0.5) {
        let parts = if lower {
            small_shape_upper(a, b, x, y)
        } else {
            small_shape_upper(b, a, y, x)
        };
        if let Some((m, e)) = parts {
            return (m, e, lower);
        }
    }
    // The log form's `−ln B(a, b) ≈ ln a` cancels against the `1/a` in the
    // fraction for `a < 1`, where `ln I` is O(1) and would keep `|ln a|·ε`
    // absolute (1.3e-15 at a = 0.014, ln I = −0.11). For `a ≥ 1` nothing of
    // that size cancels, and `ln_i` skips the `pow` a precise prefactor costs
    // on every step of the t and F quantile Newton iteration (t: `a = df/2`,
    // so df ≥ 2). The complement is exponentiated by its caller
    // (`log1p(−(1 − I))`), so it is always precise.
    let ((m, e), complement) = if lower {
        (bfrac(a, b, x, y, lambda, !ln_i || a < 1.0), false)
    } else {
        (bfrac(b, a, y, x, -lambda, true), true)
    };
    (m, e, complement)
}

/// `(scale, expo)` with `x^a·y^b/B(a,b) = scale·e^{expo}`, `y = 1 − x`
/// (DiDonato & Morris 1992, `brcomp`). For
/// `min(a, b) ≥ 8` it is `√(ab/(a+b))/√(2π)·exp(−a·rlog1(−λ/a) − b·rlog1(λ/b)
/// − bcorr(a, b))`: `x/x₀ = 1 − λ/a` and `y/y₀ = 1 + λ/b` about the mean
/// `x₀ = a/(a+b)`, so no large logarithms are formed and cancel. Otherwise
/// `brcomp_small`; where its scale leaves the normal range (the value is then
/// below `e^{40}` times the smallest normal) or an argument raised to the
/// smaller shape is 0 (`ln 0 = −∞` gives 0), or with `precise` unset (the
/// caller takes only its log and `a ≥ 1`, see `betai_kernel`),
/// `e^{a·ln x + b·ln y − ln B(a,b)}`, whose log keeps ~ε relative.
///
/// Of `x` and `y`, one ≤ 0.375 is taken as exact and the other as `1 −` it,
/// its log by `log1p`; that complement is also kept as `hi·(1 + c)`, `hi` its
/// rounded value and `c` the exact residual (Fast2Sum, as `1 >` the operand),
/// for `brcomp_small`'s `pow`: `(1 − x)^b` from `1 − x` rounded would carry
/// `b·ε`, 1e-11 at the t distribution's `b = df/2 = 1e5`.
fn brcomp(a: f64, b: f64, x: f64, y: f64, lambda: f64, precise: bool) -> (f64, f64) {
    if a.min(b) >= 8.0 {
        let x0 = a / (a + b);
        return (
            libm::sqrt(b * x0 / (2.0 * core::f64::consts::PI)),
            beta_saddle_dev(a, b, x, y, lambda) - bcorr(a, b),
        );
    }
    let complement = |t: f64| {
        let hi = 1.0 - t;
        (hi, ((1.0 - hi) - t) / hi)
    };
    let (xs, ys) = if x <= 0.375 {
        let (hi, c) = complement(x);
        ((x, 0.0, libm::log(x)), (hi, c, libm::log1p(-x)))
    } else if y <= 0.375 {
        let (hi, c) = complement(y);
        ((hi, c, libm::log1p(-y)), (y, 0.0, libm::log(y)))
    } else {
        ((x, 0.0, libm::log(x)), (y, 0.0, libm::log(y)))
    };
    let small = if precise {
        brcomp_small(a, b, xs, ys)
    } else {
        None
    };
    small.unwrap_or_else(|| (1.0, a * xs.2 + b * ys.2 - lbeta(a, b)))
}

/// `x^a·y^b/B(a,b)` for `min(a, b) < 8` as `(scale, expo)`, each argument
/// given as `(hi, c, ln)` with value `hi·(1 + c)` and log `ln` (see
/// `brcomp`); `None` where the scale leaves the normal range or an argument
/// is 0.
/// `e^{a·ln x + b·ln y − ln B}` carries the rounding of its exponent, `|ln I|·ε`
/// relative in a far tail (1e-13 at `I_x(2.5, 3.5)` ≈ 1e-227), and where the
/// larger parameter `S` is big the `s·ln S` in `−ln B(s, S) ≈ s·ln S − ln Γ(s)`
/// cancels against the other argument's power. Here `1/B` is formed without
/// logs and `S`'s share rides with the power it cancels. Both < 8: `1/B =
/// Γ(a+b)/(Γ(a)·Γ(b))` by `rgamma_small`/`rgamma_sum`. Otherwise, `s < 8 ≤ S`
/// with `u`, `v` the arguments raised to `s` and `S`:
/// `u^s·v^S/B(s, S) = (S·u)^s·v^S·e^{ρ}/Γ(s)`,
/// `ρ = ln(Γ(s+S)/Γ(S)) − s·ln S = (s − ½)·ln(1 + h) − S·rlog1(h) − (δ(S) −
/// δ(s+S))`, `h = s/S` (the `algdiv` identity of DiDonato & Morris 1992,
/// through `S·ln(1 + h) − s = −S·rlog1(h)` so no term of size `s` cancels);
/// `S·u` is kept as `p·(1 + c)`, `p` rounded and `c` from the `fma` residual,
/// and where it is subnormal, as `2⁻⁶⁴·p·(1 + c)` with `2^{−64·s}` in the
/// exponent.
/// A power with `|power·ln| > POW_SWITCH` goes to the scale by `pow` of its
/// `hi`, and its `power·c` to the exponent, unless that `pow` leaves the
/// normal range while the product need not (`(S·u)^s` can bring a subnormal
/// `v^S` back: 2.7e-8 relative in `F(5.3e11, 6.9).cdf(0.0047)` ≈ 1e-310);
/// the rest go to the exponent (an `Expo`) as `power·ln`, which there costs
/// `|power·ln|·ε/2` relative, half the value's own sensitivity to `power`.
/// (`c` itself carries ε/2 relative, `|power·c|·ε/2` in the exponent, below
/// 2e-16 for `power < 1e16`.)
fn brcomp_small(
    a: f64,
    b: f64,
    (xh, cx, ln_x): (f64, f64, f64),
    (yh, cy, ln_y): (f64, f64, f64),
) -> Option<(f64, f64)> {
    let (mut scale, mut expo, powers) = if a.max(b) < 8.0 {
        let r = rgamma_small(a) * rgamma_small(b) / rgamma_sum(a, b);
        let e = Expo { hi: 0.0, lo: 0.0 };
        (r, e, [(xh, cx, a, ln_x), (yh, cy, b, ln_y)])
    } else {
        let (s, big, (uh, cu), (vh, cv, ln_v)) = if a < b {
            (a, b, (xh, cx), (yh, cy, ln_y))
        } else {
            (b, a, (yh, cy), (xh, cx, ln_x))
        };
        // `S·u` is subnormal only where `u` is (`S ≥ 8`), and then `u` is the
        // exact argument (the other is ≥ 0.625): `2⁶⁴·u` is exact and normal,
        // and `(S·u)^s = (S·2⁶⁴·u)^s·e^{−64·s·ln 2}`, `64·s` exact. The rounding
        // of `ln 2` adds 1.5e-15·s to the exponent; the value is in range only
        // for `s ≲ 1` there (`s·ln(S·u) > −721`, `ln(S·u) < −708`). A zero `u`
        // is left to the log form, whose `ln 0 = −∞` gives the value 0.
        let (uh, shift) = if big * uh >= f64::MIN_POSITIVE {
            (uh, 0.0)
        } else if uh > 0.0 {
            (uh * 18_446_744_073_709_551_616.0, 64.0)
        } else {
            return None;
        };
        let p = big * uh;
        let c = libm::fma(big, uh, -p) / p + cu;
        let h = s / big;
        let rho = (s - 0.5) * libm::log1p(h)
            - big * rlog1(h)
            - (stirling_del(big) - stirling_del(big + s));
        let mut e = Expo { hi: rho, lo: 0.0 };
        e.add_prod(-shift * s, core::f64::consts::LN_2);
        (
            rgamma_small(s),
            e,
            [(p, c, s, libm::log(p) + c), (vh, cv, big, ln_v)],
        )
    };
    // `1/B ≤ 7e4 < e^{12}` for a, b < 8, `1/Γ(s) < 1.13`: a value whose log
    // is below that of the smallest normal is left to the log form without
    // a `pow` that would only underflow.
    if expo.hi + powers.iter().map(|&(_, _, p, l)| p * l).sum::<f64>() < -721.0 {
        return None;
    }
    for (hi, c, power, ln) in powers {
        let f = if (power * ln).abs() <= POW_SWITCH {
            0.0
        } else {
            libm::pow(hi, power)
        };
        if (f64::MIN_POSITIVE..f64::INFINITY).contains(&f) {
            scale *= f;
            expo.add_prod(power, c);
        } else {
            expo.add_prod(power, ln);
        }
    }
    let (scale, expo) = expo.apply(scale);
    (f64::MIN_POSITIVE..f64::INFINITY)
        .contains(&scale)
        .then_some((scale, expo))
}

/// `I_x(a, b)`, as `(m, e)` with `I = m·e^{e}`, by the continued fraction of
/// DiDonato & Morris (1992), `bfrac`
/// (the NR §6.4 fraction, DLMF 8.17.22, with each step's partial numerator and
/// denominator rescaled). The terms are written through `λ + 1 =
/// (a+1) − (a+b)·x`; forming that as `1 − (a+b)·x/(a+1)` loses `a·ε` to
/// cancellation near the mean at large `a`. Needs `λ > −1`, which holds below
/// the reflection point.
fn bfrac(a: f64, b: f64, x: f64, y: f64, lambda: f64, precise: bool) -> (f64, f64) {
    const EPS: f64 = 1e-15;
    const MAXIT: usize = 10_000;
    let c = lambda + 1.0;
    let c0 = b / a;
    let c1 = 1.0 / a + 1.0;
    let yp1 = y + 1.0;

    let mut p = 1.0;
    let mut s = a + 1.0;
    let (mut an, mut bn, mut anp1, mut bnp1) = (0.0, 1.0, 1.0, c / c1);
    let mut r = c1 / c;
    for n in 1..=MAXIT {
        let n = n as f64;
        let t = n / a;
        let w = n * (b - n) * x;
        let e = a / s;
        let alpha = p * (p + c0) * e * e * (w * x);
        let e = (t + 1.0) / (c1 + t + t);
        let beta = n + w / s + e * (c + n * yp1);
        p = t + 1.0;
        s += 2.0;

        let t = alpha * an + beta * anp1;
        an = anp1;
        anp1 = t;
        let t = alpha * bn + beta * bnp1;
        bn = bnp1;
        bnp1 = t;

        let r0 = r;
        r = anp1 / bnp1;
        // A NaN argument stops at once rather than running MAXIT steps.
        if (r - r0).abs() <= EPS * r || r.is_nan() {
            break;
        }
        an /= bnp1;
        bn /= bnp1;
        anp1 = r;
        bnp1 = 1.0;
    }
    let (scale, expo) = brcomp(a, b, x, y, lambda, precise);
    (scale * r, expo)
}

/// `I_x(a, b)` for large `a, b` near the mean, `λ = a − (a+b)·x ≥ 0`: the
/// asymptotic expansion of DiDonato & Morris (1992), `basym` (after Temme's
/// uniform expansion in `erfc(√f)`, `f = a·rlog1(−λ/a) + b·rlog1(λ/b)`). The
/// published form carries `e^{f}·erfc(√f)` and `e^{−f}` separately; here every
/// `J` term is pre-scaled by `t = e^{−f}`, so only `erfc(√f)` is needed and
/// neither factor overflows. Returns `(m, e)` with `I = m·e^{e}`: where
/// `f ≥ 16`, `e^{−f}` goes to `e` instead (`t = 1`, `erfc` becomes `erfcx`), so
/// the log of a tail below the f64 range stays available.
fn basym(a: f64, b: f64, lambda: f64) -> (f64, f64) {
    const EPS: f64 = 1e-15;
    const NUM: usize = 20;
    // 2/√π and 2^(−3/2).
    const E0: f64 = core::f64::consts::FRAC_2_SQRT_PI;
    const E1: f64 = 0.5 * core::f64::consts::FRAC_1_SQRT_2;

    let f = a * rlog1(-lambda / a) + b * rlog1(lambda / b);
    let (t, expo) = if f < 16.0 {
        (libm::exp(-f), 0.0)
    } else {
        (1.0, -f)
    };
    let z0 = libm::sqrt(f);
    let z = 0.5 * z0 / E1;
    let z2 = f + f;
    let (h, r0, r1, w0) = if a < b {
        let h = a / b;
        (
            h,
            1.0 / (h + 1.0),
            (b - a) / b,
            1.0 / libm::sqrt(a * (h + 1.0)),
        )
    } else {
        let h = b / a;
        (
            h,
            1.0 / (h + 1.0),
            (b - a) / a,
            1.0 / libm::sqrt(b * (h + 1.0)),
        )
    };

    let mut a0 = [0.0; NUM + 1];
    let mut b0 = [0.0; NUM + 1];
    let mut c = [0.0; NUM + 1];
    let mut d = [0.0; NUM + 1];
    a0[0] = r1 * (2.0 / 3.0);
    c[0] = -0.5 * a0[0];
    d[0] = -c[0];
    // t·J₀ and t·J₁ of the published recursion, each carrying the power of
    // `w0` its term is multiplied by (`w0^n` on the even chain, `w0^{n+1}` on
    // the odd one), and the powers of `z2` they add likewise. Kept apart as
    // published, `w0^n` underflows and `z·z2^k` overflows at huge shapes,
    // and their product is `0·∞` = NaN (`betai(1e82, 1e82, 0.49)`), while
    // `w0²·z2 ≈ λ²/min(a, b)²` is at most about 1e-3 in the band.
    let mut j0 = 0.5 / E0 * if f < 16.0 { libm::erfc(z0) } else { erfcx(z0) };
    let mut j1 = w0 * E1 * t;
    let mut sum = j0 + d[0] * j1;

    let mut s = 1.0;
    let h2 = h * h;
    let mut hn = 1.0;
    let w02 = w0 * w0;
    let step = z2 * w02;
    let mut znm1 = z * t;
    let mut zn = w0 * z2 * t;
    for n in (2..=NUM).step_by(2) {
        hn *= h2;
        a0[n - 1] = 2.0 * r0 * (h * hn + 1.0) / (n as f64 + 2.0);
        let np1 = n + 1;
        s += hn;
        a0[np1 - 1] = 2.0 * r1 * s / (n as f64 + 3.0);

        for i in n..=np1 {
            let r = -0.5 * (i as f64 + 1.0);
            b0[0] = r * a0[0];
            for m in 2..=i {
                let mut bsum = 0.0;
                for j in 1..m {
                    let mmj = m - j;
                    bsum += (j as f64 * r - mmj as f64) * a0[j - 1] * b0[mmj - 1];
                }
                b0[m - 1] = r * a0[m - 1] + bsum / m as f64;
            }
            c[i - 1] = b0[i - 1] / (i as f64 + 1.0);
            let mut dsum = 0.0;
            for j in 1..i {
                dsum += d[i - j - 1] * c[j - 1];
            }
            d[i - 1] = -(dsum + c[i - 1]);
        }

        j0 = w02 * (E1 * znm1 + (n as f64 - 1.0) * j0);
        j1 = w02 * (E1 * zn + n as f64 * j1);
        znm1 *= step;
        zn *= step;
        let t0 = d[n - 1] * j0;
        let t1 = d[np1 - 1] * j1;
        sum += t0 + t1;
        if t0.abs() + t1.abs() <= EPS * sum {
            break;
        }
    }
    (E0 * libm::exp(-bcorr(a, b)) * sum, expo)
}

/// For `min(a, b) ≤ 1` and `x ≤ ½`: `1 − I_x(a, b) = I_y(b, a)` as `(m, e)`,
/// `I_y(b, a) = m·e^{e}`, where DiDonato & Morris (1992) `bratio` evaluates
/// that side directly: by `bpser` of the swapped arguments, or `bgrat`,
/// after `bup` when `b ≤ 15`. `None` where `bratio` evaluates `I_x(a, b)`
/// itself (its `bpser` and `fpser`), which the caller leaves to `bfrac`.
/// `bratio`'s `apser`, for `a < 1e-15·min(1, b)`, is not taken: it needs
/// ψ(b) to full precision, and `bpser`/`bgrat` below hold there too while
/// `a` is normal (every O(a) factor is formed without cancellation).
fn small_shape_upper(a: f64, b: f64, x: f64, y: f64) -> Option<(f64, f64)> {
    // `bratio`'s tolerance max(ε, 1e-15) in its branch tests.
    const EPS: f64 = 1e-15;
    if b < EPS * a.min(1.0) {
        return None;
    }
    if a.max(b) > 1.0 {
        if b <= 1.0 || (x < 0.1 && libm::pow(x * b, a) <= 0.7) {
            return None;
        }
        if x >= 0.29 {
            return Some(bpser(b, a, y, x));
        }
        if b > 15.0 {
            return bgrat(b, a, y, x, (0.0, 0.0));
        }
    } else {
        if a >= b.min(0.2) || libm::pow(x, a) <= 0.9 {
            return None;
        }
        if x >= 0.3 {
            return Some(bpser(b, a, y, x));
        }
    }
    const N: usize = 20;
    let w = bup(b, a, y, x, N);
    bgrat(b + N as f64, a, y, x, w)
}

/// `I_x(a, b)` for `b ≤ 1`, `x ≤ 0.71`, as `(m, e)` with `I = m·e^{e}`, by the
/// power series of DiDonato & Morris (1992), `bpser`:
/// `I = x^a/(a·B(a,b))·(1 + a·Σ_{j≥1} (1−b)(2−b)…(j−b)/j!·x^j/(a+j))`.
/// With `b ≤ 1` every term is positive and each is at most `x` times the one
/// before, so the tail after a term `w` is under `2.5·w`. The prefactor is
/// `brcomp`'s `x^a·y^b/B(a,b)` over `a·y^b` (`y ≥ 0.29`, `b ≤ 1`: `y^b` by
/// `pow` to an ulp).
fn bpser(a: f64, b: f64, x: f64, y: f64) -> (f64, f64) {
    // λ is read by `brcomp` only for min(a, b) ≥ 8.
    let (scale, expo) = brcomp(a, b, x, y, 0.0, true);
    let mut sum = 0.0;
    let mut c = 1.0;
    let mut n = 0.0;
    loop {
        n += 1.0;
        c *= (1.0 - b / n) * x;
        let w = c / (a + n);
        sum += w;
        // A NaN argument stops at once.
        if a * w <= 0.125 * f64::EPSILON || w.is_nan() {
            break;
        }
    }
    (scale / (a * libm::pow(y, b)) * (a * sum + 1.0), expo)
}

/// `I_x(a, b) − I_x(a + n, b)` for `b ≤ 1` as `(m, e)` (DiDonato & Morris
/// 1992, `bup`): `x^a·y^b/(a·B(a,b))·Σ_{i<n} d_i`, `d_0 = 1`,
/// `d_i = d_{i−1}·x·(a+b+i−1)/(a+i)`. With `b ≤ 1` the terms fall from the
/// first, so `bup`'s search for the largest term and its `e^{μ}` scaling
/// (the exponent carries the scale here) are not needed.
fn bup(a: f64, b: f64, x: f64, y: f64, n: usize) -> (f64, f64) {
    let (scale, expo) = brcomp(a, b, x, y, 0.0, true);
    let (apb, ap1) = (a + b, a + 1.0);
    let mut d = 1.0;
    let mut w = 1.0;
    for i in 1..n {
        let l = (i - 1) as f64;
        d *= (apb + l) / (ap1 + l) * x;
        w += d;
        if d <= 0.5 * f64::EPSILON * w {
            break;
        }
    }
    (scale * w / a, expo)
}

/// `I_x(a, b) + w` for `a ≥ 15`, `b ≤ 1` as `(m, e)`, with `w = m_w·e^{e_w}`,
/// by the asymptotic expansion of DiDonato & Morris (1992), `bgrat` (their
/// §9, after Temme): with `ν = a + (b − 1)/2` and `z = −ν·ln x`,
/// `I = M·Σ_n d_n·J_n`, `M = Γ(a+b)/(Γ(a)·ν^b)`, `J_0 = Q(b, z)` and each
/// `J_n` from the one before plus a multiple of `r = z^b·e^{−z}/Γ(b)`.
/// Everything is kept in units of `r`, whose `(scale, expo)` is the incomplete
/// gamma kernel's own prefactor (so `J_0/r` of its continued fraction is that
/// fraction exactly); `e^{expo}` stays in the exponent. `None` where the
/// expansion cannot be formed (`b·z` underflows, or a partial sum ≤ 0) or `z`
/// is subnormal: its few remaining bits reach `Q(b, z)` scaled by `b·ln z`
/// (1.2e-10 relative in `betai(4.1e-4, 6.1e5, 5e-324)`).
fn bgrat(a: f64, b: f64, x: f64, y: f64, (mw, ew): (f64, f64)) -> Option<(f64, f64)> {
    const TERMS: usize = 30;
    let bm1 = b - 1.0;
    let nu = a + 0.5 * bm1;
    let lnx = if y > 0.375 {
        libm::log(x)
    } else {
        libm::log1p(-y)
    };
    let z = -nu * lnx;
    if z < f64::MIN_POSITIVE || b * z == 0.0 {
        return None;
    }
    let (scale, expo) = gamma_prefactor(b, z, true);
    // `J_0/r` as `bgrat`'s `grat1` forms it: the continued fraction from
    // z = 1.1, below it `Q` of the small-shape series (or, at b = 1, `1 − P`
    // with P ≤ 0.67).
    let q_r = if z >= 1.1 {
        // `gamma_pq`'s cap at a ≤ 1.
        gamma_cf(b, z, 212)
    } else {
        let (m, e, upper) = gamma_pq(b, z, None);
        pq_from_parts(m, e, upper).1 / scale * libm::exp(-expo)
    };
    // M·r = scale·e^{expo + ln M}; the caller's `w` in those units.
    let ln_m = -(algdiv(b, a) + b * libm::log(nu));
    let l = if mw > 0.0 {
        mw / scale * libm::exp(ew - expo - ln_m)
    } else {
        0.0
    };
    let v = 0.25 / (nu * nu);
    let t2 = 0.25 * lnx * lnx;
    let mut c = [0.0; TERMS];
    let mut d = [0.0; TERMS];
    let mut j = q_r;
    let mut sum = j;
    let (mut t, mut cn, mut n2) = (1.0, 1.0, 0.0);
    for n in 1..=TERMS {
        let bp2n = b + n2;
        j = (bp2n * (bp2n + 1.0) * j + (z + bp2n + 1.0) * t) * v;
        n2 += 2.0;
        t *= t2;
        cn /= n2 * (n2 + 1.0);
        c[n - 1] = cn;
        let mut s = 0.0;
        let mut coef = b - n as f64;
        for i in 1..n {
            s += coef * c[i - 1] * d[n - 1 - i];
            coef += b;
        }
        d[n - 1] = bm1 * cn + s / n as f64;
        let dj = d[n - 1] * j;
        sum += dj;
        if sum <= 0.0 {
            return None;
        }
        if dj.abs() <= 0.5 * f64::EPSILON * (sum + l) {
            break;
        }
    }
    Some((scale * libm::exp(ln_m) * (sum + l), expo))
}

#[cfg(test)]
mod tests {
    use super::{betai, betai_parts, gammp, gammq, ln_betai, ln_gammp, ln_gammq, pq_from_parts};
    use std::format;

    fn check(label: &str, got: f64, want: f64, rel: f64) {
        assert!(
            ((got - want) / want).abs() < rel,
            "{label} = {got:e}, want {want:e}"
        );
    }

    /// `ln P` and `ln Q` where the value is below the smallest subnormal (the
    /// log under −745) and in the normal range. Truth, at the exact f64 inputs:
    /// mpmath `log(gammainc(a, 0, x, regularized=True))` (or `x, inf`) at 80
    /// dps; the Legendre continued fraction at 400 dps for Q(20, 800); for
    /// a ≥ 1e5 Temme's expansion (DLMF 8.12.3–8.12.10) to 14 terms in 1/a at 400
    /// dps, which agrees with the hypergeometric series to 1e-35 at a = 1e3…1e5.
    #[test]
    fn ln_gamma_tails_match_mpmath() {
        const TOL: f64 = 2e-15;
        for (a, x, want) in [
            (167.0, 0.75, -739.971_833_052_438_8),
            (3.0, 1e-300, -2_074.118_343_163_869),
            (0.5, 1e-300, -345.266_981_711_471_6),
            (1e7, 9.5e6, -12_938.926_167_595_113),
            (2.5, 1.5, -1.203_925_591_702_561),
        ] {
            check(&format!("ln P({a}, {x})"), ln_gammp(a, x), want, TOL);
        }
        for (a, x, want) in [
            (20.0, 800.0, -712.308_255_851_346_4),
            (1e5, 1.2e5, -1_772.910_584_581_419),
            (1e-300, 0.5, -691.355_750_770_258_5),
            (2.5, 1.5, -0.356_695_178_602_555_34),
            // Subnormal shape: Q ≈ a·E₁(x) is below 2⁻¹⁰⁷⁴, small-shape series
            // and continued fraction.
            (5e-324, 1.1, -746.122_129_428_189_7),
            (5e-324, 708.0, -1_459.003_925_464_158_4),
        ] {
            check(&format!("ln Q({a}, {x})"), ln_gammq(a, x), want, TOL);
        }
    }

    /// `ln I_x(a, b)` below the f64 range and in the normal range, with `x` and
    /// `y = 1 − x` formed as the t and F distributions form them. Truth: the
    /// continued fraction DLMF 8.17.22 in mpmath at 60 dps, at the exact
    /// complement of whichever of `x`, `y` is smaller.
    #[test]
    fn ln_betai_tails_match_mpmath() {
        // In the normal range `ln I` carries betai's own ~2e-15 relative error
        // divided by |ln I| < 1.
        const TOL: f64 = 3e-15;
        const TOL_NORMAL: f64 = 1e-14;
        // t(1000) upper tail at t = 200: ½·I_z(500, ½), z = df/(df + t²).
        let (df, t2) = (1000.0, 200.0 * 200.0);
        // F(3, 1e6) upper tail at x = 500: I_z(5e5, 1.5), z = dfd/(dfd + dfn·x).
        let (dfd, nx) = (1e6, 3.0 * 500.0);
        for (a, b, x, y, want, tol) in [
            (
                500.0,
                0.5,
                df / (df + t2),
                t2 / (df + t2),
                -1_860.453_630_986_500_2,
                TOL,
            ),
            (
                5e5,
                1.5,
                dfd / (dfd + nx),
                nx / (dfd + nx),
                -746.007_325_715_157_5,
                TOL,
            ),
            (1e7, 1e7, 0.49, 0.51, -4_006.212_875_264_377, TOL),
            (
                50.0,
                80.0,
                1e-10,
                1.0 - 1e-10,
                -1_067.796_120_215_981_5,
                TOL,
            ),
            (2.0, 3.0, 0.4, 0.6, -0.644_738_041_352_257_7, TOL_NORMAL),
            // ln(1 − I_{0.4}(2, 3)) through the swapped arguments.
            (3.0, 2.0, 0.6, 0.4, -0.744_019_510_933_702, TOL_NORMAL),
        ] {
            check(
                &format!("ln I_{x}({a}, {b})"),
                ln_betai(a, b, x, y),
                want,
                tol,
            );
        }
    }

    /// `gammq`'s `a < 8` continued-fraction branch past `x = 708`, where
    /// `gamma_prefactor` skips `pow` and folds `a·ln x − x` into one exponent
    /// rather than forming a subnormal `e^{−x}` on its own, short of bits.
    /// Truth: mpmath `gammainc(7.9, 727.5238762030003, inf, regularized=True)`
    /// at 60 dps, cross-checked against the Legendre continued fraction
    /// (DLMF 8.9.2) evaluated independently at 100 dps (agreement to 90
    /// digits): 1.500000000000150084744152700397…e-300. Tolerance is
    /// `gamma_prefactor`'s own documented bound for this branch,
    /// `(|a·ln x| + x)·ε` (1.73e-13 here).
    #[test]
    fn gammq_pow_path_x_gt_708() {
        const TOL: f64 = 2e-13;
        check(
            "Q(7.9, 727.5238762030003)",
            gammq(7.9, 727.5238762030003),
            1.500_000_000_000_15e-300,
            TOL,
        );
    }

    /// `a < 1`, `1.1 ≤ x < a + 1`: Q comes from the continued fraction, not
    /// `1 − P` of the series, so it joins the small-shape branch (x < 1.1)
    /// with a step against the monotone direction of at most a few ε at
    /// x = 1.1 (4.5ε at a = 0.4600030617710667, checked between adjacent f64
    /// values either side of the seam). Truth: mpmath
    /// `gammainc(a, x, inf, regularized=True)` at 60 dps at the exact f64
    /// inputs: Q(0.4600030617710667, 1.3) = 0.09568592366288373398…,
    /// Q(0.999, 1.9) = 0.14933015391762217169…; P is `1 − Q` of the same.
    #[test]
    fn gammq_small_shape_seam_monotone() {
        const A: f64 = 0.460_003_061_771_066_7;
        let below = f64::from_bits(1.1f64.to_bits() - 1);
        let (q0, q1) = (gammq(A, below), gammq(A, 1.1));
        assert!(
            q1 - q0 <= 10.0 * f64::EPSILON * q1,
            "Q({A}, {below}) = {q0:e} steps up to {q1:e} at 1.1"
        );
        check(
            "Q(0.46, 1.3)",
            gammq(A, 1.3),
            0.095_685_923_662_883_73,
            4e-15,
        );
        check(
            "P(0.46, 1.3)",
            gammp(A, 1.3),
            0.904_314_076_337_116_3,
            4e-15,
        );
        check(
            "Q(0.999, 1.9)",
            gammq(0.999, 1.9),
            0.149_330_153_917_622_17,
            4e-15,
        );
    }

    /// A shape ≤ 1 where `bfrac` would take the other side: `b ≪ 1` above the
    /// reflection point, where `I` comes out as `1 −` a near-1 complement
    /// (8.5e-13 relative at the first row), and `ln I` of an `I` near 1 at
    /// `a ≪ 1` below it (1.4e-11 relative). Truth: mpmath `betainc` at 50
    /// dps at the exact f64 inputs.
    #[test]
    fn betai_small_shape_other_side() {
        const TOL: f64 = 3e-15;
        for (a, b, x, want) in [
            (
                5.0,
                0.001,
                0.857_020_425_653_478_1,
                3.763_324_483_507_106e-4,
            ),
            (
                100.0,
                0.001,
                0.990_186_370_721_855_8,
                2.266_253_801_168_230_7e-4,
            ),
            (
                1e10,
                0.001,
                0.999_999_999_899_900_2,
                2.192_413_047_150_169_8e-4,
            ),
        ] {
            check(&format!("I_{x}({a}, {b})"), betai(a, b, x), want, TOL);
        }
        let (a, b, x) = (
            0.000_230_291_559_577_907_05,
            2.951_345_304_875_987_5,
            0.202_002_427_728_176_5,
        );
        check(
            &format!("ln I_{x}({a}, {b})"),
            ln_betai(a, b, x, 1.0 - x),
            -1.137_967_539_593_608_5e-4,
            1e-15,
        );
    }

    /// `ln I` of an `I` near 1 on `bfrac`'s side at `a < 1`, where the log form
    /// of the prefactor keeps `|ln a|·ε` absolute (1.2e-14 relative at the last
    /// row); the first two at `x = 5e-324`, where `bgrat`'s `z` is subnormal,
    /// the second also with `S·u` subnormal in `brcomp_small`. Truth: mpmath
    /// `log(betainc(a, b, 0, x, regularized=True))` at 80 dps at the exact f64
    /// inputs (the second unchanged at 150 dps).
    #[test]
    fn ln_betai_small_first_shape() {
        const TOL: f64 = 3e-15;
        for (a, b, x, want) in [
            (
                2.302_915_595_779_070_5e-4,
                2.951_345_304_875_987_5,
                5e-324,
                -0.171_097_328_460_452_03,
            ),
            (
                4.057_556_713_422_705_6e-4,
                605_669.684_894_642_4,
                5e-324,
                -0.296_424_440_800_739_26,
            ),
            (
                0.014_104_013_848_231_607,
                0.901_393_322_112_543_3,
                0.000_569_785_581_813_264,
                -0.107_801_632_795_837_83,
            ),
        ] {
            check(
                &format!("ln I_{x}({a}, {b})"),
                ln_betai(a, b, x, 1.0 - x),
                want,
                TOL,
            );
        }
    }

    /// `brcomp_small` with `S·u` subnormal (`S = b`, `u = x = 5e-324`): the
    /// value comes from the rescaled `u`, not the log form (1.1e-15 relative
    /// there). Truth: mpmath `betainc(a, b, 0, x, regularized=True)` at 80
    /// dps at the exact f64 inputs. At `u = 0` (`StudentT::cdf(0)`, which asks
    /// for `I_1(df/2, ½)` with `y = 0`) the log form gives the value, not NaN.
    #[test]
    fn betai_subnormal_scaled_arg() {
        check(
            "I_5e-324(4.06e-4, 605669.7)",
            betai(4.057_556_713_422_705_6e-4, 605_669.684_894_642_4, 5e-324),
            0.743_471_801_275_823_4,
            7e-16,
        );
        let (m, e, complement) = betai_parts(0.5 * 6_457.288_040_284_936, 0.5, 1.0, 0.0);
        assert_eq!(pq_from_parts(m, e, complement), (1.0, 0.0));
    }

    /// `x` within an ulp of 1 with both shapes huge, `(x, y)` as
    /// `FisherF::beta_args` forms them for F(1e4, 1e20) at 1.1 (the sf) and
    /// F(1e20, 1) at 0.001 (the cdf). The reflection point `(a+1)/(a+b+2)`
    /// rounds onto `x`; a test of `x` against it would pick the side whose
    /// fraction needs `λ > −1` at λ ≈ −500, giving an sf of −2e-202 for 1 and
    /// a log of +4e-12 for −26.3. Truth: the continued fraction DLMF 8.17.22 in
    /// mpmath at 400 dps, at `x = 1 − y` with `y` exact. The first row keeps
    /// the `|ln I|·ε` of `min(a, b) ≥ 8` (4e-15 measured).
    #[test]
    fn betai_reflection_side_near_one() {
        for (a, b, x, y, want, ln_want, tol) in [
            (
                5e19,
                5e3,
                0.999_999_999_999_999_9,
                1.099_999_999_999_999_9e-16,
                3.618_329_558_096_489_5e-12,
                -26.345_008_644_630_42,
                1e-14,
            ),
            (
                5e19,
                0.5,
                1.0,
                1e-17,
                1.795_832_784_800_657_4e-219,
                -503.680_666_504_381_75,
                5e-15,
            ),
        ] {
            let (m, e, complement) = betai_parts(a, b, x, y);
            let (i, _) = pq_from_parts(m, e, complement);
            check(&format!("I_{x}({a}, {b})"), i, want, tol);
            check(
                &format!("ln I_{x}({a}, {b})"),
                ln_betai(a, b, x, y),
                ln_want,
                tol,
            );
            // The complement through the swapped arguments.
            let (m, e, complement) = betai_parts(b, a, y, x);
            let (_, j) = pq_from_parts(m, e, complement);
            check(&format!("1 − I_{y}({b}, {a})"), j, want, tol);
        }
    }
}
