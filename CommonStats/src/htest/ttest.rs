//! t-tests. Group statistics from the Variance accumulator; p-values from betai.
use crate::accum::moments::{checked_variance, pooled_var};
use crate::error::StatError;
use crate::htest::ci::{ci_mean, ci_mean_diff, ci_mean_diff_welch};
use crate::htest::effect::cohen_d;
use crate::htest::result::{EffectSize, TestResult};
use crate::special::betai;
use alloc::vec::Vec;

/// Variance assumption for the two-sample t-test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarAssumption {
    /// Pooled-variance Student's t (assumes equal population variances).
    Equal,
    /// Welch's t with Satterthwaite df (does not assume equal variances).
    Welch,
}

/// Two-sided survival p-value for a t statistic with df degrees of freedom:
/// p = I_{df/(df+t²)}(df/2, 1/2)  (regularized incomplete beta).
fn t_two_sided_p(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    betai(df / 2.0, 0.5, x)
}

/// One-sample t-test of mean(`v`) against `mu0`.
///
/// Two-sided p-value; df = n − 1; effect size is Cohen's d = (mean − mu0)/sd.
/// `v`: observations, NaN dropped under the Omit default, needs ≥ 2 finite values.
/// [`StatError::TooFewObservations`] when fewer than 2 finite values.
/// Matches `scipy.stats.ttest_1samp` (`tests/fixtures/ttests.json`).
pub fn t_test_one(v: &[f64], mu0: f64) -> Result<TestResult, StatError> {
    let s = checked_variance(v)?;
    let n = s.count() as f64;
    let se = s.sd_sample() / libm::sqrt(n);
    let t = (s.mean() - mu0) / se;
    let df = n - 1.0;
    Ok(TestResult {
        statistic: t,
        df,
        df2: None,
        p_value: t_two_sided_p(t, df),
        effect_size: Some(EffectSize::CohenD((s.mean() - mu0) / s.sd_sample())),
        ci: ci_mean(v, 0.95).ok(),
    })
}

/// Two-sample t-test of mean(`a`) vs mean(`b`), Student (pooled) or Welch per `va`.
///
/// Two-sided p-value. `Equal` pools the variances with df = nA + nB − 2; `Welch`
/// uses the Satterthwaite df. Effect size is Cohen's d (pooled SD); the CI matches
/// the variance assumption (pooled for `Equal`, Welch for `Welch`) so it inverts to
/// the reported p-value. NaN dropped per group under Omit; each
/// group needs ≥ 2 finite values. `statistic` = t, `df` per `va`, two-sided
/// `p_value`. [`StatError::TooFewObservations`] when a group has < 2 finite values.
/// Matches `scipy.stats.ttest_ind(equal_var=…)` (`tests/fixtures/ttests.json`).
pub fn t_test_two(a: &[f64], b: &[f64], va: VarAssumption) -> Result<TestResult, StatError> {
    let (sa, sb) = (checked_variance(a)?, checked_variance(b)?);
    let (na, nb) = (sa.count() as f64, sb.count() as f64);
    let (va_, vb) = (sa.var_sample(), sb.var_sample());
    let (t, df) = match va {
        VarAssumption::Equal => {
            // SE = √(sp²(1/nA+1/nB)); pooled sp² from the shared accumulator helper.
            let se = libm::sqrt(pooled_var(&sa, &sb) * (1.0 / na + 1.0 / nb));
            ((sa.mean() - sb.mean()) / se, na + nb - 2.0)
        }
        VarAssumption::Welch => {
            // Welch–Satterthwaite df = (vA/nA + vB/nB)² /
            //   [ (vA/nA)²/(nA−1) + (vB/nB)²/(nB−1) ]; SE = √(vA/nA + vB/nB).
            let se_term = va_ / na + vb / nb;
            let se = libm::sqrt(se_term);
            let a_term = va_ / na;
            let b_term = vb / nb;
            let df = (se_term * se_term)
                / ((a_term * a_term) / (na - 1.0) + (b_term * b_term) / (nb - 1.0));
            ((sa.mean() - sb.mean()) / se, df)
        }
    };
    let ci = match va {
        VarAssumption::Equal => ci_mean_diff(a, b, 0.95),
        VarAssumption::Welch => ci_mean_diff_welch(a, b, 0.95),
    };
    Ok(TestResult {
        statistic: t,
        df,
        df2: None,
        p_value: t_two_sided_p(t, df),
        effect_size: Some(EffectSize::CohenD(cohen_d(a, b)?)),
        ci: ci.ok(),
    })
}

/// Paired-samples t-test on the differences `a[i] − b[i]` (a one-sample t against 0).
///
/// Two-sided p-value; df = n − 1. [`StatError::MismatchedLengths`] when
/// `a.len() != b.len()`; [`StatError::TooFewObservations`] when < 2 pairs.
/// Matches `scipy.stats.ttest_rel` (`tests/fixtures/ttests.json`).
pub fn t_test_paired(a: &[f64], b: &[f64]) -> Result<TestResult, StatError> {
    if a.len() != b.len() {
        return Err(StatError::MismatchedLengths {
            a: a.len(),
            b: b.len(),
        });
    }
    let diff: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    t_test_one(&diff, 0.0)
}
