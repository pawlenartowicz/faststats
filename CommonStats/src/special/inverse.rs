//! Inverse special functions for closed-form critical values.
use crate::special::betai;

/// Inverse complementary error function, for p ∈ (0, 2):
/// erfc_inv(p) = −norm_ppf(p/2) / √2. (libm exposes erf/erfc but no inverse.)
/// Matches `scipy.special.erfcinv` (`tests/fixtures/erfcinv.json`).
pub fn erfc_inv(p: f64) -> f64 {
    -norm_ppf(0.5 * p) / core::f64::consts::SQRT_2
}

/// Inverse error function, for x ∈ (−1, 1): erf_inv(x) = erfc_inv(1 − x).
/// Matches `scipy.special.erfinv` (`tests/fixtures/erfinv.json`).
pub fn erf_inv(x: f64) -> f64 { erfc_inv(1.0 - x) }

/// Normal quantile Φ⁻¹(p). Acklam rational seed (~1.15e-9) refined by one Halley
/// step against libm::erf to ~1e-12. Kept private.
fn norm_ppf(p: f64) -> f64 {
    // Rational-approximation coefficients (Acklam).
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
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let mut x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    // One Halley refinement: Φ(x) = 0.5·erfc(-x/√2); drive Φ(x) - p → 0 against
    // libm's ~1e-15 erfc to lift the Acklam seed from ~1e-9 to ~1e-12.
    if x.is_finite() {
        let e = 0.5 * libm::erfc(-x / core::f64::consts::SQRT_2) - p; // Φ(x) - p
        let u = e * (2.0 * core::f64::consts::PI).sqrt() * (x * x / 2.0).exp();
        x -= u / (1.0 + x * u / 2.0);
    }
    x
}

/// Inverse regularized incomplete beta I⁻¹_p(a,b) → t/F/beta quantiles. We invert
/// our own accurate `betai` by bisection rather than depend on `puruspe::invbetai`,
/// which loses ~13 digits in the extreme upper tail (a=1,b=30,p=1-1e-9 → 0.64 vs
/// the true 0.499). For p>0.5 we solve the complement I_{1-x}(b,a)=1-p: when both
/// I_x(a,b) and p sit near 1 their difference cancels catastrophically, whereas in
/// the complement the residual stays small. Bisection (not Newton) because I_x is
/// near-flat for skewed params (y³⁰-style) where Newton crawls linearly — sign-based
/// bisection is immune to that. Matches `scipy.special.betaincinv`
/// (`tests/fixtures/invbetareg.json`).
pub fn inv_beta_reg(a: f64, b: f64, p: f64) -> f64 {
    if p <= 0.0 { return 0.0; }
    if p >= 1.0 { return 1.0; }
    if p > 0.5 {
        return 1.0 - inv_beta_reg_lower(b, a, 1.0 - p);
    }
    inv_beta_reg_lower(a, b, p)
}

/// Solve I_x(a,b) = p for the lower tail (p ≤ 0.5) by bisection on the monotone
/// `betai`. 80 halvings drive the bracket below f64 resolution.
fn inv_beta_reg_lower(a: f64, b: f64, p: f64) -> f64 {
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi { break; } // bracket below f64 resolution
        if betai(a, b, mid) < p { lo = mid; } else { hi = mid; }
    }
    0.5 * (lo + hi)
}
