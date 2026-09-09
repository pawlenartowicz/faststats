//! One-way ANOVA and the equal-variance F-test. Group stats from Variance; p via betai.
use crate::accum::moments::{Variance, checked_variance};
use crate::error::StatError;
use crate::htest::result::{EffectSize, TestResult};
use crate::special::betai;
use alloc::vec::Vec;

/// Upper-tail p for an F statistic: betai(d2/2, d1/2, d2/(d2 + d1*F)).
fn f_sf(f: f64, d1: f64, d2: f64) -> f64 {
    let x = d2 / (d2 + d1 * f);
    betai(d2 / 2.0, d1 / 2.0, x)
}

/// ANOVA sums for `groups`: (grand_n, SSB, SSW). Each group is NaN-omitted and
/// needs ≥ 2 finite values. Shared by [`anova_one_way`] and
/// [`eta_squared`](crate::htest::effect::eta_squared).
pub(crate) fn anova_sums(groups: &[&[f64]]) -> Result<(f64, f64, f64), StatError> {
    let vars: Vec<Variance> = groups
        .iter()
        .map(|g| checked_variance(g))
        .collect::<Result<_, _>>()?;
    let grand_n: f64 = vars.iter().map(|v| v.count() as f64).sum();
    let grand_mean: f64 = vars
        .iter()
        .map(|v| v.count() as f64 * v.mean())
        .sum::<f64>()
        / grand_n;
    // SSB = Σ nᵢ(meanᵢ - grand)²; SSW = Σ (nᵢ-1)·varᵢ  (var_sample = M2/(n-1) → SSW=Σ M2)
    let ssb: f64 = vars
        .iter()
        .map(|v| {
            let d = v.mean() - grand_mean;
            v.count() as f64 * d * d
        })
        .sum();
    let ssw: f64 = vars
        .iter()
        .map(|v| (v.count() as f64 - 1.0) * v.var_sample())
        .sum();
    Ok((grand_n, ssb, ssw))
}

/// One-way ANOVA F-test across `groups`.
///
/// Upper-tail F p-value; `df` = d1 = k − 1 (between-groups); `df2` = d2 = N − k
/// (within-groups). Effect size is η². `groups`: one slice per group, NaN dropped
/// under Omit, each needs ≥ 2 finite values; needs ≥ 2 groups.
/// [`StatError::TooFewObservations`]. Matches `scipy.stats.f_oneway`
/// (`tests/fixtures/anova.json`).
pub fn anova_one_way(groups: &[&[f64]]) -> Result<TestResult, StatError> {
    if groups.len() < 2 {
        return Err(StatError::TooFewObservations {
            needed: 2,
            got: groups.len(),
        });
    }
    let (grand_n, ssb, ssw) = anova_sums(groups)?;
    let k = groups.len() as f64;
    let (d1, d2) = (k - 1.0, grand_n - k);
    let f = (ssb / d1) / (ssw / d2);
    Ok(TestResult {
        statistic: f,
        df: d1,
        df2: Some(d2),
        p_value: f_sf(f, d1, d2),
        // η² = SSB/(SSB+SSW), reusing the sums already in hand; mirrors htest::effect::eta_squared.
        effect_size: Some(EffectSize::EtaSquared(ssb / (ssb + ssw))),
        ci: None,
    })
}

/// Two-sided F-test for equality of two variances, F = var(`a`)/var(`b`).
///
/// p = 2·min(sf, 1 − sf); the `df` field carries d1 = nA − 1 (d2 = nB − 1). Each
/// group NaN dropped under Omit, needs ≥ 2 finite values.
/// [`StatError::TooFewObservations`]. Matches R `var.test(a, b)$p.value`
/// (`tests/fixtures/ftest.json`).
pub fn f_test_var(a: &[f64], b: &[f64]) -> Result<TestResult, StatError> {
    let (sa, sb) = (checked_variance(a)?, checked_variance(b)?);
    let (d1, d2) = (sa.count() as f64 - 1.0, sb.count() as f64 - 1.0);
    let f = sa.var_sample() / sb.var_sample();
    // two-sided: 2*min(sf, 1-sf)
    let sf = f_sf(f, d1, d2);
    let p = 2.0 * sf.min(1.0 - sf);
    Ok(TestResult {
        statistic: f,
        df: d1,
        df2: None,
        p_value: p,
        effect_size: None,
        ci: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anova_one_way_exposes_df2_within_groups() {
        // 3 groups of 4 → N=12, k=3 → df1=2, df2=9.
        // Validated against scipy.stats.f_oneway: dfn=2, dfd=9.
        let g1 = [89., 88., 97., 92.];
        let g2 = [84., 79., 81., 83.];
        let g3 = [91., 95., 94., 88.];
        let r = anova_one_way(&[&g1, &g2, &g3]).unwrap();
        assert_eq!(r.df, 2.0);
        assert_eq!(r.df2, Some(9.0)); // within-groups df = N - k = 12 - 3
    }
}
