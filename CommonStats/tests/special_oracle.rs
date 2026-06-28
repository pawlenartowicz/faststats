//! Class-A special-function oracle grids, validated against committed mpmath
//! truth via the shared harness (accuracy-validation spec). Tolerances are the
//! spec §3 bulk ladder: value-kind (erf/gamma/incomplete fns) 1e-12, inverse
//! (quantile-kind) 1e-10. Tail rows assert against a curated band.
mod common;
use common::{assert_monotone, check_grid, measure_band, Tol};

use commonstats::special;

// Bulk ladder (spec §3): pdf/cdf/density-like vs inverse/quantile.
const VAL: Tol = Tol { rel: 1e-12, abs: 1e-14 };
const INV: Tol = Tol { rel: 1e-10, abs: 1e-12 };
// digamma (ψ, the lgamma derivative) is neither value- nor quantile-kind: across
// its grid it achieves ~1e-10 (worst 9.7e-11 at x≈2), mpmath-confirmed and not a
// regression — the pre-rebaseline test already used 1e-10. Tiered on its own.
const DERIV: Tol = Tol { rel: 1e-10, abs: 1e-12 };

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
    check_grid("invbetareg", INV, |a| special::inv_beta_reg(a[0], a[1], a[2]));
}

/// Band calibration (spec §5). Run with `cargo test -- --ignored --nocapture`;
/// the printed values (lightly padded) are curated by a human into the tail rows'
/// `band` field. Tests never write fixtures.
#[test]
#[ignore]
fn measure_tail_bands() {
    let bands = [
        ("erfcinv", measure_band("erfcinv", |a| special::erfc_inv(a[0]))),
        ("invbetareg", measure_band("invbetareg", |a| special::inv_beta_reg(a[0], a[1], a[2]))),
    ];
    for (name, val) in bands {
        match val {
            Some(b) => println!("TAIL BAND  {name:>12} → {b:e}"),
            None => println!("TAIL BAND  {name:>12} → (no tail rows)"),
        }
    }
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
    assert!((special::beta(0.5, 0.5) - core::f64::consts::PI).abs() < 1e-13, "B(.5,.5) {}", special::beta(0.5, 0.5));
    assert!((special::beta(1.0, 1.0) - 1.0).abs() < 1e-13, "B(1,1) {}", special::beta(1.0, 1.0));
    assert!((special::beta(2.0, 3.0) - 1.0 / 12.0).abs() < 1e-13, "B(2,3) {}", special::beta(2.0, 3.0));
}
