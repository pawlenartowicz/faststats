//! Missing-data policy — the single home for how the crate treats NaN.
//!
//! Every batch and streaming entry point routes f64 input through [`clean`], so
//! the rule lives here once and is cited, not re-derived, elsewhere. The default
//! is [`NanPolicy::Omit`]: NaNs are dropped and each statistic reports on the
//! effective count of finite values. [`NanPolicy::Propagate`] is the strict
//! opt-in — any NaN turns the whole result into an error instead of silently
//! dropping data.
use crate::error::StatError;
use alloc::vec::Vec;

/// Missing-data policy, decided once and applied uniformly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NanPolicy {
    /// Drop NaN, report effective n. The crate default.
    #[default]
    Omit,
    /// Any NaN poisons the result (strict callers).
    Propagate,
}

/// Apply the missing-data `policy` to `xs`, returning the values to compute on.
///
/// Accumulators that ingest slices route through this so batch and streaming
/// agree by construction. Returns [`StatError::EmptyInput`] when `xs` is empty;
/// [`StatError::AllNaN`] when no finite value survives `Omit`, or when
/// `Propagate` encounters any NaN.
pub fn clean(xs: &[f64], policy: NanPolicy) -> Result<Vec<f64>, StatError> {
    if xs.is_empty() {
        return Err(StatError::EmptyInput);
    }
    match policy {
        NanPolicy::Propagate => {
            if xs.iter().any(|x| x.is_nan()) {
                Err(StatError::AllNaN)
            } else {
                Ok(xs.to_vec())
            }
        }
        NanPolicy::Omit => {
            let kept: Vec<f64> = xs.iter().copied().filter(|x| !x.is_nan()).collect();
            if kept.is_empty() {
                Err(StatError::AllNaN)
            } else {
                Ok(kept)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::StatError;
    use std::vec;
    #[test]
    fn omit_drops_nan_and_reports_effective_n() {
        let v = clean(&[1.0, f64::NAN, 3.0], NanPolicy::Omit).unwrap();
        assert_eq!(v, vec![1.0, 3.0]);
    }
    #[test]
    fn omit_all_nan_is_error() {
        assert_eq!(clean(&[f64::NAN], NanPolicy::Omit), Err(StatError::AllNaN));
    }
    #[test]
    fn empty_is_error() {
        assert_eq!(clean(&[], NanPolicy::Omit), Err(StatError::EmptyInput));
    }
    #[test]
    fn propagate_poisons_on_any_nan() {
        assert_eq!(
            clean(&[1.0, f64::NAN], NanPolicy::Propagate),
            Err(StatError::AllNaN)
        );
        assert_eq!(
            clean(&[1.0, 2.0], NanPolicy::Propagate).unwrap(),
            vec![1.0, 2.0]
        );
    }
}
