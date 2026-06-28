//! Correlation test. Pearson only; Spearman/Kendall would need rank machinery
//! (not yet implemented).
use crate::accum::moments::comoment_pairs;
use crate::error::StatError;
use crate::htest::ci::ci_correlation;
use crate::htest::result::{EffectSize, TestResult};
use crate::special::betai;

/// Correlation method. Only Pearson is implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorMethod {
    /// Pearson product-moment correlation.
    Pearson,
}

/// Pearson correlation t-test between paired `a` and `b`.
///
/// Two-sided p-value; t = r·√(df/(1 − r²)) with df = n − 2. Reports r as the effect
/// size and a Fisher-z 95% CI; `statistic` is t (r is in `effect_size`). NaN pairs
/// dropped under Omit; needs ≥ 3 complete pairs. [`StatError::MismatchedLengths`],
/// [`StatError::TooFewObservations`] (n < 3). Matches `scipy.stats.pearsonr`
/// (`tests/fixtures/cor.json`).
pub fn cor_test(a: &[f64], b: &[f64], method: CorMethod) -> Result<TestResult, StatError> {
    let CorMethod::Pearson = method; // only Pearson today; add arms when Spearman/Kendall land
    let c = comoment_pairs(a, b)?;
    let n = c.count() as f64;
    if n < 3.0 { return Err(StatError::TooFewObservations { needed: 3, got: n as usize }); }
    let r = c.pearson();
    let df = n - 2.0;
    let t = r * (df / (1.0 - r * r)).sqrt();
    let p = betai(df / 2.0, 0.5, df / (df + t * t));
    Ok(TestResult {
        statistic: t, df, df2: None, p_value: p,
        effect_size: Some(EffectSize::R(r)),
        ci: ci_correlation(a, b, 0.95).ok(),
    })
}
