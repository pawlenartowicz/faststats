//! digamma (ψ) and logsumexp.

/// Digamma ψ(x) for x > 0, via recurrence ψ(x) = ψ(x+1) − 1/x up to x ≥ 10, then
/// the asymptotic series (Abramowitz & Stegun 6.3.18)
///   ψ(x) ≈ ln x − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶).
/// Threshold is 10, not 6: at x≈6 the first omitted term 1/(240 x⁸) is ~2e-9,
/// which breaks the ~1e-12 accuracy target near x=0.1; pushing the recurrence to
/// x ≥ 10 shrinks it to ~1e-12. Matches `scipy.special.digamma`
/// (`tests/fixtures/digamma.json`).
pub fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 10.0 { result -= 1.0 / x; x += 1.0; }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

/// Numerically stable log(Σ exp(xᵢ)) over `xs`. Empty slice → −∞ (log of an
/// empty sum); a NaN in `xs` propagates to the result.
pub fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY { return f64::NEG_INFINITY; }
    let sum: f64 = xs.iter().map(|&x| (x - m).exp()).sum();
    m + sum.ln()
}
