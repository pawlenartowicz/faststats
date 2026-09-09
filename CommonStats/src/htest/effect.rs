//! Effect sizes paired with the tests.
//!
//! `eta_squared` and `anova_one_way` share the private `anova_sums` helper
//! (owner: `htest::anova::anova_sums`) for SSB/SSW computation — change both
//! together when the one-way sums formula changes.
use crate::accum::moments::{checked_variance, pooled_var};
use crate::error::StatError;
use crate::htest::anova::anova_sums;
use crate::htest::chi2::contingency_chi2;

/// Cohen's d for two independent samples, using the pooled standard deviation
/// (ddof = 1) as denominator; sign follows mean(`a`) − mean(`b`).
///
/// `a`, `b`: observations, NaN dropped under the Omit default; each needs ≥ 2
/// finite values, else [`StatError::TooFewObservations`]. No dedicated oracle
/// fixture — the pooled-SD definition matches the d reported alongside
/// `scipy.stats.ttest_ind`.
pub fn cohen_d(a: &[f64], b: &[f64]) -> Result<f64, StatError> {
    let (sa, sb) = (checked_variance(a)?, checked_variance(b)?);
    let sp = libm::sqrt(pooled_var(&sa, &sb));
    Ok((sa.mean() - sb.mean()) / sp)
}

/// η² (eta-squared) for one-way ANOVA: SSB / (SSB + SSW), the proportion of total
/// variance lying between groups; range [0, 1].
///
/// `groups`: one slice per group (`&[&[f64]]`), NaN dropped under Omit; each group
/// needs ≥ 2 finite values, else [`StatError::TooFewObservations`].
///
/// Matches `scipy.stats.f_oneway` (η² = SSB/SST, manual verification).
///
/// Formula is duplicated in `anova_one_way` — mirrors `htest::anova::anova_one_way`,
/// change both together.
pub fn eta_squared(groups: &[&[f64]]) -> Result<f64, StatError> {
    let (_grand_n, ssb, ssw) = anova_sums(groups)?;
    Ok(ssb / (ssb + ssw))
}

/// Cramér's V for an r×c contingency table: √(χ² / (N · min(r−1, c−1))); range
/// [0, 1]. Uncorrected (Cramér 1946), no bias adjustment. Recomputes χ² via the
/// shared [`contingency_chi2`] helper (pure, no shared state) so it stays
/// independent of the test fn.
///
/// `table`: a rectangular ≥2×2 grid of non-negative counts (shape validated by the
/// caller, e.g. [`chi2_independence`](crate::htest::chi2_independence)).
///
/// Matches `scipy.stats.contingency.association(table, method='cramer')`.
pub fn cramers_v(table: &[&[f64]]) -> Result<f64, StatError> {
    let (chi2, grand) = contingency_chi2(table);
    let k = ((table.len() - 1).min(table[0].len() - 1)) as f64;
    Ok(libm::sqrt(chi2 / (grand * k)))
}
