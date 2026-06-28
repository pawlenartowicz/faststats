//! Competitive ranking of a finite slice, configurable tie-breaking.

use crate::error::StatError;
use crate::nan::{clean, NanPolicy};

/// Tie-breaking rule for [`rank`].
///
/// Maps directly to the five `method` values of `scipy.stats.rankdata`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ties {
    /// Mean of the tied 1-based positions (may be fractional).
    Average,
    /// Lowest position in the tie group.
    Min,
    /// Highest position in the tie group.
    Max,
    /// Consecutive group counter (no gaps after ties).
    Dense,
    /// Appearance order within the tie group (stable sort order).
    Ordinal,
}

/// 1-based competitive ranks of the finite values in `v`, with configurable tie-breaking.
///
/// **Convention:** NaN values are omitted (Omit policy). Output length equals the
/// number of finite values in `v`, not `v.len()`.
///
/// **Matches** `scipy.stats.rankdata(v, method=<ties>)` applied to the finite
/// subset of `v`.
///
/// # Parameters
/// - `v`: input slice; may contain NaN.
/// - `ties`: tie-breaking rule (see [`Ties`]).
///
/// # Returns
/// `Vec<f64>` of length = finite count, in the same order as the finite values
/// of `v`. `Average` yields fractional values; all others are exact integers cast
/// to `f64`.
///
/// # Errors
/// - [`StatError::EmptyInput`] if `v` is empty.
/// - [`StatError::AllNaN`] if all values are NaN.
///
/// # Examples
/// ```
/// use commonstats::transform::{rank, Ties};
/// let r = rank(&[3.0, 1.0, 2.0], Ties::Average).unwrap();
/// assert_eq!(r, vec![3.0, 1.0, 2.0]);
/// ```
pub fn rank(v: &[f64], ties: Ties) -> Result<Vec<f64>, StatError> {
    let data = clean(v, NanPolicy::Omit)?;
    let n = data.len();

    // Stable argsort: indices sorted by data[i] ascending, stable (so Ordinal
    // reflects first-appearance order within ties).
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());

    let mut out = vec![0.0f64; n];

    // Scan run-length tie groups in sorted order.
    let mut i = 0usize;
    let mut group_counter = 0usize; // Dense: increments once per distinct value group
    while i < n {
        // Find end of tie group [i, j)
        let mut j = i + 1;
        while j < n && data[idx[j]] == data[idx[i]] {
            j += 1;
        }
        group_counter += 1;
        let lo = i + 1; // 1-based first position of this group
        let hi = j;     // 1-based last position of this group

        match ties {
            Ties::Average => {
                // Mean of 1-based positions lo..=hi
                let avg = (lo + hi) as f64 / 2.0;
                for k in i..j {
                    out[idx[k]] = avg;
                }
            }
            Ties::Min => {
                for k in i..j {
                    out[idx[k]] = lo as f64;
                }
            }
            Ties::Max => {
                for k in i..j {
                    out[idx[k]] = hi as f64;
                }
            }
            Ties::Dense => {
                for k in i..j {
                    out[idx[k]] = group_counter as f64;
                }
            }
            Ties::Ordinal => {
                // Appearance order: assign lo, lo+1, … in stable sorted (=appearance) order.
                for (rank_offset, k) in (i..j).enumerate() {
                    out[idx[k]] = (lo + rank_offset) as f64;
                }
            }
        }
        i = j;
    }

    Ok(out)
}
