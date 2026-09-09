//! Closed-form confidence intervals. Critical values from the inverse special functions.
use crate::accum::moments::{checked_variance, comoment_pairs, pooled_var};
use crate::error::StatError;
use crate::htest::result::Ci;
use crate::special::{erfc_inv, inv_beta_reg};

/// Reject a confidence `level` outside the open interval (0, 1).
fn check_level(level: f64) -> Result<(), StatError> {
    if !(0.0..1.0).contains(&level) {
        Err(StatError::ProbabilityOutOfRange(level))
    } else {
        Ok(())
    }
}

/// Two-sided t critical value t_{1-α/2, df} via inverse regularized incomplete beta.
/// For two-sided level, total tail = 1-level; t = sqrt(df*(1/x - 1)) where
/// x = inv_beta_reg(df/2, 1/2, 1-level).
fn t_crit(level: f64, df: f64) -> f64 {
    let alpha_tail = 1.0 - level;
    let x = inv_beta_reg(df / 2.0, 0.5, alpha_tail);
    libm::sqrt(df * (1.0 / x - 1.0))
}

/// Two-sided normal critical value z_{1-α/2} = √2 · erfc_inv(1-level).
/// e.g. level 0.95 → √2·erfc_inv(0.05) = 1.95996 (pinned in the proportion-CI test).
fn z_crit(level: f64) -> f64 {
    core::f64::consts::SQRT_2 * erfc_inv(1.0 - level)
}

/// Two-sided confidence interval for a population mean (Student t, df = n − 1).
///
/// Matches `scipy.stats.t.interval(level, df=n-1, loc=mean, scale=se)`.
///
/// `v`: observations, NaN dropped under Omit, needs ≥ 2 finite values; `level`:
/// coverage in (0, 1). [`StatError::ProbabilityOutOfRange`] if `level` ∉ (0,1);
/// [`StatError::TooFewObservations`] if < 2 finite values.
pub fn ci_mean(v: &[f64], level: f64) -> Result<Ci, StatError> {
    check_level(level)?;
    let s = checked_variance(v)?;
    let n = s.count() as f64;
    let se = s.sd_sample() / libm::sqrt(n);
    let t = t_crit(level, n - 1.0);
    Ok(Ci {
        lower: s.mean() - t * se,
        upper: s.mean() + t * se,
        level,
    })
}

/// Two-sided confidence interval for the difference of two means, pooled-variance
/// Student t (df = nA + nB − 2).
///
/// Matches the CI from `scipy.stats.ttest_ind(a, b, equal_var=True)`.
///
/// `a`, `b`: observations, NaN dropped under Omit, each needs ≥ 2 finite values;
/// `level`: coverage in (0, 1). [`StatError::ProbabilityOutOfRange`] /
/// [`StatError::TooFewObservations`].
pub fn ci_mean_diff(a: &[f64], b: &[f64], level: f64) -> Result<Ci, StatError> {
    check_level(level)?;
    let (sa, sb) = (checked_variance(a)?, checked_variance(b)?);
    let (na, nb) = (sa.count() as f64, sb.count() as f64);
    let se = libm::sqrt(pooled_var(&sa, &sb) * (1.0 / na + 1.0 / nb));
    let t = t_crit(level, na + nb - 2.0);
    let d = sa.mean() - sb.mean();
    Ok(Ci {
        lower: d - t * se,
        upper: d + t * se,
        level,
    })
}

/// Two-sided confidence interval for the difference of two means, Welch
/// (unequal-variance) t with Satterthwaite df.
///
/// Matches `scipy.stats.ttest_ind(a, b, equal_var=False)` CI and
/// R `t.test(a, b, var.equal=FALSE)$conf.int`. SE = √(vA/nA + vB/nB);
/// df from Welch–Satterthwaite.
///
/// Unlike [`ci_mean_diff`], this inverts to the Welch two-sample p-value:
/// matching `t_test_two(…, Welch)`. `a`, `b`: observations, NaN dropped under
/// Omit, each needs ≥ 2 finite values; `level`: coverage in (0, 1).
/// [`StatError::ProbabilityOutOfRange`] / [`StatError::TooFewObservations`].
pub fn ci_mean_diff_welch(a: &[f64], b: &[f64], level: f64) -> Result<Ci, StatError> {
    check_level(level)?;
    let (sa, sb) = (checked_variance(a)?, checked_variance(b)?);
    let (na, nb) = (sa.count() as f64, sb.count() as f64);
    let (va, vb) = (sa.var_sample(), sb.var_sample());
    let se_term = va / na + vb / nb;
    let se = libm::sqrt(se_term);
    let a_term = va / na;
    let b_term = vb / nb;
    let df =
        (se_term * se_term) / ((a_term * a_term) / (na - 1.0) + (b_term * b_term) / (nb - 1.0));
    let t = t_crit(level, df);
    let d = sa.mean() - sb.mean();
    Ok(Ci {
        lower: d - t * se,
        upper: d + t * se,
        level,
    })
}

/// Two-sided Wald (normal-approximation) confidence interval for a proportion.
///
/// Matches `statsmodels.stats.proportion.proportion_confint(successes, n,
/// alpha=1-level, method='normal')`.
///
/// `successes` of `n` trials; `level`: coverage in (0, 1). No continuity
/// correction. [`StatError::EmptyInput`] if `n == 0`;
/// [`StatError::ProbabilityOutOfRange`] if `level` ∉ (0,1).
pub fn ci_proportion(successes: usize, n: usize, level: f64) -> Result<Ci, StatError> {
    if n == 0 {
        return Err(StatError::EmptyInput);
    }
    check_level(level)?;
    let p = successes as f64 / n as f64;
    let se = libm::sqrt(p * (1.0 - p) / n as f64);
    let z = z_crit(level);
    Ok(Ci {
        lower: p - z * se,
        upper: p + z * se,
        level,
    })
}

/// Two-sided confidence interval for a Pearson correlation via the Fisher
/// z-transform (SE = 1/√(n − 3), back-transformed with tanh).
///
/// Matches `scipy.stats.pearsonr(a, b)` Fisher-z CI (confidence_interval method).
///
/// `a`, `b`: equal-length paired data, NaN pairs dropped, needs ≥ 4 complete pairs;
/// `level`: coverage in (0, 1). [`StatError::MismatchedLengths`],
/// [`StatError::TooFewObservations`] (n < 4), [`StatError::ProbabilityOutOfRange`].
pub fn ci_correlation(a: &[f64], b: &[f64], level: f64) -> Result<Ci, StatError> {
    check_level(level)?;
    let c = comoment_pairs(a, b)?;
    let n = c.count() as f64;
    if n < 4.0 {
        return Err(StatError::TooFewObservations {
            needed: 4,
            got: n as usize,
        });
    }
    let r = c.pearson();
    // Fisher z-transform: z = atanh(r), se = 1/sqrt(n-3).
    let zr = libm::atanh(r);
    let se = 1.0 / libm::sqrt(n - 3.0);
    let zc = z_crit(level);
    Ok(Ci {
        lower: libm::tanh(zr - zc * se),
        upper: libm::tanh(zr + zc * se),
        level,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn t_crit_matches_table() {
        // t_{0.975, 9} = 2.262157 (two-sided 95%)
        assert!(
            (t_crit(0.95, 9.0) - 2.262157).abs() < 1e-4,
            "got {}",
            t_crit(0.95, 9.0)
        );
    }
    #[test]
    fn ci_mean_known() {
        // [1..5]: mean 3, s = √2.5; half-width = t_{0.975,4}·s/√5 = 1.96324316
        // (scipy t.ppf(0.975,4)). Pins the width, not just that it brackets the mean.
        let ci = ci_mean(&[1., 2., 3., 4., 5.], 0.95).unwrap();
        assert!(
            (ci.lower - 1.036756838522439).abs() < 1e-6,
            "lower {}",
            ci.lower
        );
        assert!(
            (ci.upper - 4.963243161477561).abs() < 1e-6,
            "upper {}",
            ci.upper
        );
    }
}

#[cfg(test)]
mod ci2_tests {
    use super::*;
    #[test]
    fn proportion_ci_known() {
        // Wald 95% for p̂=0.4, n=100: p̂ ± 1.959964·√(p̂q̂/n) (scipy norm.ppf(0.975)).
        let ci = ci_proportion(40, 100, 0.95).unwrap();
        assert!(
            (ci.lower - 0.3039817664728938).abs() < 1e-9,
            "lower {}",
            ci.lower
        );
        assert!(
            (ci.upper - 0.4960182335271062).abs() < 1e-9,
            "upper {}",
            ci.upper
        );
    }
}
