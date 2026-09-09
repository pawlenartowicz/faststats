//! Bolstad (2003) quantile normalization for expression matrices.
//!
//! This is the genomics/microarray algorithm — **NOT** sklearn's
//! `QuantileTransformer`. sklearn maps each column to a reference distribution
//! (e.g., uniform or normal) using per-column sorted order; this function
//! instead averages the per-column sorted values across columns to build a
//! reference row, then maps each value in each column to the reference at its
//! average rank. The two algorithms produce different outputs even on the same
//! matrix — cite Bolstad 2003, not sklearn, when using this function.

use crate::error::StatError;
use crate::transform::rank::{Ties, rank};
use alloc::{vec, vec::Vec};

/// Column-rank-averaging normalization (Bolstad et al. 2003).
///
/// **Convention:**
/// 1. Average-rank each column independently.
/// 2. Build a reference row: for each rank position `k`, take the mean of the
///    k-th smallest value across all columns.
/// 3. Replace each value in each column by the reference at its average rank,
///    linearly interpolating between adjacent reference entries for fractional
///    (tied) ranks.
///
/// **NaN deviation from crate default:** NaN propagates element-wise; the output
/// matrix has the same shape as the input. NaN positions do not contribute to
/// ranks or the reference row computation. This is the documented exception to
/// the crate's Omit default.
///
/// **Matches** Bolstad et al. (2003), *Bioinformatics* 19(2):185–193.
///
/// # Parameters
/// - `matrix`: slice of column slices; all columns must have equal length.
///
/// # Returns
/// `Vec<Vec<f64>>` with the same shape as `matrix`; NaN positions preserved.
///
/// # Errors
/// - [`StatError::EmptyInput`] if `matrix` is empty.
/// - [`StatError::MismatchedLengths`] if any two columns differ in length.
///
/// # Examples
/// ```
/// use commonstats::transform::quantile_normalize;
/// // Two columns, 2 rows
/// let c0 = [1.0f64, 3.0];
/// let c1 = [2.0f64, 4.0];
/// let out = quantile_normalize(&[&c0, &c1]).unwrap();
/// // After normalization both columns share the same sorted-means distribution
/// assert_eq!(out.len(), 2);
/// assert_eq!(out[0].len(), 2);
/// ```
pub fn quantile_normalize(matrix: &[&[f64]]) -> Result<Vec<Vec<f64>>, StatError> {
    if matrix.is_empty() {
        return Err(StatError::EmptyInput);
    }
    let nrows = matrix[0].len();
    // Validate equal column lengths
    for (_j, col) in matrix.iter().enumerate().skip(1) {
        if col.len() != nrows {
            return Err(StatError::MismatchedLengths {
                a: nrows,
                b: col.len(),
            });
        }
    }
    let ncols = matrix.len();

    // Per-column sorted-finite vectors; NaN elements remain NaN in the output at
    // their original positions (tracked in step 3 via finite_indices).
    let mut finite_sorted: Vec<Vec<f64>> = Vec::with_capacity(ncols);
    for col in matrix.iter() {
        let mut pairs: Vec<(usize, f64)> = col
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_finite())
            .map(|(i, &x)| (i, x))
            .collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        finite_sorted.push(pairs.iter().map(|&(_, v)| v).collect());
    }

    // Reference row: ref[k] = mean of the k-th sorted finite value across columns.
    // Columns shorter than ref_len (NaN-reduced) contribute only up to their length.
    let ref_len = finite_sorted.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut reference: Vec<f64> = vec![0.0; ref_len];
    let mut ref_counts: Vec<usize> = vec![0; ref_len];
    for sorted in &finite_sorted {
        for (k, &val) in sorted.iter().enumerate() {
            reference[k] += val;
            ref_counts[k] += 1;
        }
    }
    for (r, c) in reference.iter_mut().zip(ref_counts.iter()) {
        if *c > 0 {
            *r /= *c as f64;
        }
    }

    // Replace each finite value by the reference at its average rank within the column,
    // linearly interpolating for fractional (tied) ranks.
    let mut output: Vec<Vec<f64>> = matrix.iter().map(|col| col.to_vec()).collect();

    for (j, col) in matrix.iter().enumerate() {
        // Compute average ranks of finite values within this column
        let finite_vals: Vec<f64> = col.iter().copied().filter(|x| x.is_finite()).collect();
        if finite_vals.is_empty() {
            continue; // all-NaN column — leave as-is
        }
        let n_finite = finite_vals.len();
        // rank() returns ranks for each element of finite_vals in their given order
        let avg_ranks = rank(&finite_vals, Ties::Average)
            .expect("rank cannot fail on a non-empty finite slice");

        // Map each finite element's rank to a reference value via linear interpolation.
        // finite_indices[i] = original row index of the i-th finite value in this column.
        let finite_indices: Vec<usize> = col
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_finite())
            .map(|(i, _)| i)
            .collect();

        for (fi, &orig_idx) in finite_indices.iter().enumerate() {
            let r = avg_ranks[fi]; // 1-based fractional rank
            // Interpolate reference at fractional rank r:
            //   floor index = r.floor() as usize - 1  (0-based)
            //   ceil  index = min(r.ceil() as usize - 1, n_finite - 1)
            let r_floor = libm::floor(r);
            let lo_idx = (r_floor as usize).saturating_sub(1);
            let hi_idx = (libm::ceil(r) as usize).saturating_sub(1).min(n_finite - 1);
            let frac = r - r_floor;
            let val = if lo_idx == hi_idx {
                reference[lo_idx]
            } else {
                reference[lo_idx] * (1.0 - frac) + reference[hi_idx] * frac
            };
            output[j][orig_idx] = val;
        }
    }

    Ok(output)
}
