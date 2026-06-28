//! Elementary special functions. erf/erfc/gamma via libm (pure-Rust, WASM-clean);
//! lgamma is a Lanczos–Pugh implementation so log-gamma is identical wherever the
//! crate uses it. beta/lbeta are gamma identities.

/// Error function erf(x), defined on all reals, range (−1, 1). libm::erf is
/// Boost-derived, ~1e-15. Matches `scipy.special.erf` (`tests/fixtures/erf.json`).
pub fn erf(x: f64) -> f64 { libm::erf(x) }

/// Complementary error function erfc(x) = 1 − erf(x), all reals → (0, 2).
/// Computed directly (not as 1 − erf) to keep accuracy in the right tail.
/// Matches `scipy.special.erfc` (`tests/fixtures/erfc.json`).
pub fn erfc(x: f64) -> f64 { libm::erfc(x) }

/// Gamma function Γ(x) via libm::tgamma. Defined on the reals except the
/// non-positive integers (poles). Matches `scipy.special.gamma`
/// (`tests/fixtures/gamma.json`).
pub fn gamma(x: f64) -> f64 { libm::tgamma(x) }

/// Natural log of the gamma function, ln Γ(x), used where Γ would overflow.
///
/// Lanczos–Pugh approximation (g = 7, n = 9): for x ≥ 0.5, with z = x − 1,
///   ln Γ(x) = ½·ln(2π) + (z+½)·ln(z+g+½) − (z+g+½) + ln A_g(z),
/// where A_g(z) = P[0] + Σ_{k≥1} P[k]/(z+k). For x < 0.5 the reflection
///   ln Γ(x) = ln(π / sin(πx)) − ln Γ(1−x)
/// maps the argument into the convergent range. ~1e-15 for x > 0.
/// Matches `scipy.special.gammaln` (`tests/fixtures/lgamma.json`).
pub fn lgamma(mut x: f64) -> f64 {
    const G: f64 = 7.0;
    const P: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - lgamma(1.0 - x);
    }
    x -= 1.0;
    let mut a = P[0];
    let t = x + G + 0.5;
    for (i, &pi) in P.iter().enumerate().skip(1) {
        a += pi / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b), via the stable log form exp(lbeta).
/// Defined for a, b > 0. See [`lbeta`] for the validated log form.
pub fn beta(a: f64, b: f64) -> f64 { lbeta(a, b).exp() }

/// Log-beta ln B(a,b) = lgamma(a) + lgamma(b) − lgamma(a+b). Defined for a, b > 0.
/// Matches `scipy.special.betaln` (`tests/fixtures/lbeta.json`).
pub fn lbeta(a: f64, b: f64) -> f64 { lgamma(a) + lgamma(b) - lgamma(a + b) }
