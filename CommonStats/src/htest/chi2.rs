//! χ² goodness-of-fit and independence. p-values via the upper incomplete gamma gammq.
use crate::error::StatError;
use crate::htest::effect::cramers_v;
use crate::htest::result::{EffectSize, TestResult};
use crate::special::gammq;
use alloc::vec::Vec;

/// χ² upper-tail p-value: Q(df/2, χ²/2).
fn chi2_sf(chi2: f64, df: f64) -> f64 {
    gammq(df / 2.0, chi2 / 2.0)
}

/// Pearson χ² statistic and grand total for a contingency `table`, expected cell
/// e = rowᵢ·colⱼ/grand (cells with e ≤ 0 skipped). Pure — shared by the
/// independence test and Cramér's V, which stay decoupled by each calling it.
/// Assumes the caller has validated the shape (rectangular, ≥ 2×2).
pub(crate) fn contingency_chi2(table: &[&[f64]]) -> (f64, f64) {
    let cols = table[0].len();
    let row_tot: Vec<f64> = table.iter().map(|r| r.iter().sum()).collect();
    let col_tot: Vec<f64> = (0..cols)
        .map(|j| table.iter().map(|r| r[j]).sum())
        .collect();
    let grand: f64 = row_tot.iter().sum();
    let mut chi2 = 0.0;
    for (i, row) in table.iter().enumerate() {
        for (j, &o) in row.iter().enumerate() {
            let e = row_tot[i] * col_tot[j] / grand;
            if e > 0.0 {
                let d = o - e;
                chi2 += d * d / e;
            }
        }
    }
    (chi2, grand)
}

/// Pearson χ² goodness-of-fit test of `observed` against `expected`.
///
/// Upper-tail p-value; df = k − 1. `observed`, `expected`: equal-length non-empty
/// slices; each `expected` entry must be > 0 (a non-positive expected cell yields a
/// NaN statistic). [`StatError::MismatchedLengths`], [`StatError::EmptyInput`].
/// Matches `scipy.stats.chisquare` (`tests/fixtures/chi2.json`).
pub fn chi2_gof(observed: &[f64], expected: &[f64]) -> Result<TestResult, StatError> {
    if observed.len() != expected.len() {
        return Err(StatError::MismatchedLengths {
            a: observed.len(),
            b: expected.len(),
        });
    }
    if observed.is_empty() {
        return Err(StatError::EmptyInput);
    }
    let chi2: f64 = observed
        .iter()
        .zip(expected)
        .map(|(&o, &e)| {
            if e <= 0.0 {
                return f64::NAN;
            }
            (o - e) * (o - e) / e
        })
        .sum();
    let df = (observed.len() - 1) as f64;
    Ok(TestResult {
        statistic: chi2,
        df,
        df2: None,
        p_value: chi2_sf(chi2, df),
        effect_size: None,
        ci: None,
    })
}

/// Pearson χ² test of independence on a contingency `table` (rows × columns).
///
/// Upper-tail p-value; df = (r − 1)(c − 1); no Yates correction. Effect size is
/// Cramér's V. `table`: rectangular, ≥ 2×2, of non-negative counts.
/// [`StatError::TooFewObservations`] (< 2 rows), [`StatError::DomainError`] (ragged
/// or < 2 columns). Matches `scipy.stats.chi2_contingency(correction=False)`
/// (`tests/fixtures/chi2.json`).
pub fn chi2_independence(table: &[&[f64]]) -> Result<TestResult, StatError> {
    let rows = table.len();
    if rows < 2 {
        return Err(StatError::TooFewObservations {
            needed: 2,
            got: rows,
        });
    }
    let cols = table[0].len();
    if cols < 2 || table.iter().any(|r| r.len() != cols) {
        return Err(StatError::DomainError(
            "contingency table must be rectangular, ≥2×2",
        ));
    }
    let (chi2, _) = contingency_chi2(table);
    let df = ((rows - 1) * (cols - 1)) as f64;
    Ok(TestResult {
        statistic: chi2,
        df,
        df2: None,
        p_value: chi2_sf(chi2, df),
        effect_size: Some(EffectSize::CramersV(cramers_v(table)?)),
        ci: None,
    })
}
