//! Regularized incomplete gamma/beta (NR §6.2 / §6.4). `gammq` is the upper-tail
//! complement of `gammp`. All log-gamma routes through the crate's `lgamma`/`lbeta`.
use crate::special::{lbeta, lgamma};

/// Regularized lower incomplete gamma P(a,x), for a > 0 and x ≥ 0; range [0, 1].
/// (NR §6.2: series for x < a+1, Lentz continued fraction otherwise; ~1e-14.)
/// Matches `scipy.special.gammainc` (`tests/fixtures/gammp.json`).
pub fn gammp(a: f64, x: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    const MAXIT: usize = 200;
    const EPS: f64 = 3e-15;
    const FPMIN: f64 = 1.0e-300;

    if x < a + 1.0 {
        // Series: P(a, x) = e^{-x} x^a / Γ(a+1) · Σ_{n≥0} x^n / (a+1)_n.
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut term = sum;
        for _ in 0..MAXIT {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * EPS {
                return sum * (-x + a * x.ln() - lgamma(a)).exp();
            }
        }
        sum * (-x + a * x.ln() - lgamma(a)).exp()
    } else {
        // Continued fraction for Q(a, x) (upper); P = 1 - Q.
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..MAXIT {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < EPS {
                let q = (-x + a * x.ln() - lgamma(a)).exp() * h;
                return 1.0 - q;
            }
        }
        let q = (-x + a * x.ln() - lgamma(a)).exp() * h;
        1.0 - q
    }
}

/// Regularized upper incomplete gamma Q(a,x) = 1 − P(a,x), for a > 0 and x ≥ 0.
/// Used by χ² p-values. Matches `scipy.special.gammaincc` (`tests/fixtures/gammq.json`).
pub fn gammq(a: f64, x: f64) -> f64 { 1.0 - gammp(a, x) }

/// Regularized incomplete beta I_x(a,b), for a,b > 0 and x ∈ [0,1]; range [0, 1].
/// (NR §6.4 continued fraction with the x < (a+1)/(a+b+2) symmetry transform; ~1e-15.)
/// Matches `scipy.special.betainc` (`tests/fixtures/betai.json`).
pub fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (a * x.ln() + b * (1.0 - x).ln() - lbeta(a, b)).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for the incomplete beta by Lentz's method (NR §6.4). The
/// coefficient at even step m is d_{2m} = m(b−m)x / [(a+2m−1)(a+2m)] and at odd
/// step d_{2m+1} = −(a+m)(a+b+m)x / [(a+2m)(a+2m+1)]; the loop accumulates the
/// h = Π(d·c) products until successive factors converge to 1.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 200;
    const EPS: f64 = 3e-15;
    const FPMIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        // First step (even index).
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // Second step (odd index).
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            return h;
        }
    }
    h
}
