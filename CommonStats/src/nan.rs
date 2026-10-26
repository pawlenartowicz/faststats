//! Missing-data policy — the single home for how the crate treats NaN.
//!
//! Every batch and streaming entry point routes f64 input through [`clean`], so
//! the rule lives here once and is cited, not re-derived, elsewhere. The policy
//! is *Omit*: NaNs are dropped and each statistic reports on the effective count
//! of finite values.
use crate::error::StatError;
use alloc::vec::Vec;

/// Apply the Omit policy to `xs`: drop the NaNs, returning the values to compute on.
///
/// Accumulators that ingest slices route through this so batch and streaming
/// agree by construction. Returns [`StatError::EmptyInput`] when `xs` is empty;
/// [`StatError::AllNaN`] when no value survives.
pub fn clean(xs: &[f64]) -> Result<Vec<f64>, StatError> {
    if xs.is_empty() {
        return Err(StatError::EmptyInput);
    }
    let kept: Vec<f64> = xs.iter().copied().filter(|x| !x.is_nan()).collect();
    if kept.is_empty() {
        Err(StatError::AllNaN)
    } else {
        Ok(kept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::StatError;
    use std::vec;
    #[test]
    fn omit_drops_nan_and_reports_effective_n() {
        let v = clean(&[1.0, f64::NAN, 3.0]).unwrap();
        assert_eq!(v, vec![1.0, 3.0]);
    }
    #[test]
    fn omit_all_nan_is_error() {
        assert_eq!(clean(&[f64::NAN]), Err(StatError::AllNaN));
    }
    #[test]
    fn empty_is_error() {
        assert_eq!(clean(&[]), Err(StatError::EmptyInput));
    }
}
