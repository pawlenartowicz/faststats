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

/// How [`quantile_normalize_with`] treats ±∞. NaN is missing under both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Infinities {
    /// ±∞ is missing, like NaN: it takes no rank, adds nothing to the target,
    /// and is returned unchanged. The convention of [`quantile_normalize`].
    AsMissing,
    /// ±∞ is an observation beyond every finite value: in each column −∞ takes
    /// the lowest ranks and +∞ the highest (equal infinities share their
    /// average rank, as any tie), and both count toward the column's `m`.
    /// Infinite entries are returned unchanged; see [`quantile_normalize_with`]
    /// for how they shape the target.
    AsExtremes,
}

/// Column-rank-averaging normalization (Bolstad et al. 2003), ±∞ treated as
/// missing.
///
/// [`quantile_normalize_with`] with [`Infinities::AsMissing`]; use that
/// function with [`Infinities::AsExtremes`] to rank ±∞ beyond the finite
/// values instead.
///
/// **Convention:** with `n` rows, and `m` the count of a column's non-missing
/// values,
/// 1. Average-rank each column's non-missing values (ranks 1..m).
/// 2. Build an `n`-point target: each column's sorted non-missing values,
///    stretched onto `n` points by linear interpolation at 0-based position
///    `(m − 1)·i/(n − 1)` when `m < n`, averaged across the columns that have
///    any non-missing value.
/// 3. Replace each non-missing value by the target at 0-based position
///    `(n − 1)·(r − 1)/(m − 1)` for its average rank `r` (`r − 1` when
///    `m = n`; the middle, `(n − 1)/2`, when `m = 1`), interpolating linearly
///    between adjacent target entries. Without missing values this is the
///    target at the average rank.
///
/// **Missing values, a deviation from crate default:** NaN and ±∞ are missing
/// and propagate element-wise; the output matrix has the same shape as the
/// input, with each missing value left in place unchanged. They take no rank
/// and add nothing to the target. A ±∞ is missing rather than ranked here: ranked
/// in and averaged, it would make its target entry infinite and carry that
/// infinity into the finite values of every other column at the same rank
/// ([`Infinities::AsExtremes`] ranks it without averaging it in). This is the
/// documented exception to the crate's NaN-omit policy.
///
/// **Matches** Bolstad et al. (2003), *Bioinformatics* 19(2):185–193
/// (`tests/fixtures/quantile_normalize.json`); with missing values, Bioconductor
/// `preprocessCore::normalize.quantiles` (NA for NaN), including its tie
/// handling (`tests/fixtures/quantile_normalize_missing.json`, mpmath truth),
/// except two cases where preprocessCore's result is an artifact: a column with
/// one value (it takes the target maximum there) and an all-missing column (its
/// target averages in a stale buffer; here the column is left out).
///
/// # Parameters
/// - `matrix`: slice of column slices; all columns must have equal length.
///
/// # Returns
/// `Vec<Vec<f64>>` with the same shape as `matrix`; NaN and ±∞ positions
/// preserved.
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
    quantile_normalize_with(matrix, Infinities::AsMissing)
}

/// Column-rank-averaging normalization (Bolstad et al. 2003) with a chosen
/// treatment of ±∞.
///
/// **Convention:** [`Infinities::AsMissing`] is [`quantile_normalize`]'s
/// convention, step for step. [`Infinities::AsExtremes`] keeps its steps with
/// NaN as the only missing value, so ±∞ is ranked (−∞ lowest, +∞ highest) and
/// counts toward `m`, and changes how the target is formed:
/// - A column adds to the target at a grid position only where its
///   interpolated value there is finite (an infinity, or an interpolation
///   that touches one, adds nothing); the target there is the mean of the
///   finite contributions it gets.
/// - Where those means would decrease from one position to the next (a
///   column's infinities leave its largest or smallest values out there),
///   adjacent positions are pooled to the mean of all their finite
///   contributions (pool-adjacent-violators), so the target is non-decreasing
///   and every column keeps its order.
/// - A position with no finite contribution takes the target interpolated
///   linearly between its nearest positions that have one, or the nearest
///   one's value past either end. If no position has one, the finite values
///   come back NaN: there is no finite target to map them to.
/// - Finite values map through the target at their rank among all non-NaN
///   entries; infinite entries come back unchanged.
///
/// Without infinities in `matrix` the two conventions are bit-identical.
///
/// **Validated against:** `AsMissing` as [`quantile_normalize`]
/// (preprocessCore for missing values). `AsExtremes` has no external
/// implementation to match; it is checked against hand-derived cases, such as
/// A = [2, −∞, 3], B = [0, 5, 7]: sorted A = [−∞, 2, 3] adds nothing at
/// position 0, so the target is [0, (2 + 5)/2, (3 + 7)/2] = [0, 3.5, 5], and
/// A → [3.5, −∞, 5], B → [0, 3.5, 5].
///
/// # Parameters
/// - `matrix`: slice of column slices; all columns must have equal length.
/// - `infinities`: how ±∞ is treated.
///
/// # Returns
/// `Vec<Vec<f64>>` with the same shape as `matrix`; NaN and ±∞ positions
/// preserved.
///
/// # Errors
/// - [`StatError::EmptyInput`] if `matrix` is empty.
/// - [`StatError::MismatchedLengths`] if any two columns differ in length.
///
/// # Examples
/// ```
/// use commonstats::transform::{Infinities, quantile_normalize_with};
/// let a = [2.0, f64::NEG_INFINITY, 3.0];
/// let b = [0.0, 5.0, 7.0];
/// let out = quantile_normalize_with(&[&a, &b], Infinities::AsExtremes).unwrap();
/// assert_eq!(out[0], [3.5, f64::NEG_INFINITY, 5.0]);
/// assert_eq!(out[1], [0.0, 3.5, 5.0]);
/// ```
pub fn quantile_normalize_with(
    matrix: &[&[f64]],
    infinities: Infinities,
) -> Result<Vec<Vec<f64>>, StatError> {
    if matrix.is_empty() {
        return Err(StatError::EmptyInput);
    }
    let n = matrix[0].len();
    for col in matrix.iter().skip(1) {
        if col.len() != n {
            return Err(StatError::MismatchedLengths { a: n, b: col.len() });
        }
    }
    let is_missing = |x: f64| match infinities {
        Infinities::AsMissing => !x.is_finite(),
        Infinities::AsExtremes => x.is_nan(),
    };

    // Per column, its non-missing values in row order with their row indices.
    let observed: Vec<(Vec<usize>, Vec<f64>)> = matrix
        .iter()
        .map(|col| {
            col.iter()
                .enumerate()
                .filter(|&(_, &x)| !is_missing(x))
                .map(|(i, &x)| (i, x))
                .unzip()
        })
        .collect();

    // Per grid position, the sum and count of the finite values the columns'
    // sorted values, stretched onto n points, put there. Without infinities
    // every column with a value adds at every position, so a complete matrix
    // gets the plain mean of the k-th smallest values.
    let mut sum = vec![0.0; n];
    let mut count = vec![0usize; n];
    for (_, vals) in &observed {
        if vals.is_empty() {
            continue;
        }
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = sorted.len();
        for i in 0..n {
            let v = if m == n {
                sorted[i]
            } else {
                interpolate(&sorted, (m - 1) as f64 * i as f64 / (n - 1) as f64)
            };
            if v.is_finite() {
                sum[i] += v;
                count[i] += 1;
            }
        }
    }
    let target = monotone_target(&sum, &count);

    let mut output: Vec<Vec<f64>> = matrix.iter().map(|col| col.to_vec()).collect();
    for ((rows, vals), out) in observed.iter().zip(output.iter_mut()) {
        if vals.is_empty() {
            continue; // all missing: left as is
        }
        let m = vals.len();
        let ranks =
            rank(vals, Ties::Average).expect("rank cannot fail on a non-empty NaN-free slice");
        for ((&row, &r), &x) in rows.iter().zip(ranks.iter()).zip(vals.iter()) {
            if !x.is_finite() {
                continue; // an infinity under `AsExtremes`: left as is
            }
            let pos = if m == n {
                r - 1.0
            } else if m == 1 {
                (n - 1) as f64 / 2.0
            } else {
                (n - 1) as f64 * (r - 1.0) / (m - 1) as f64
            };
            out[row] = interpolate(&target, pos);
        }
    }

    Ok(output)
}

/// The target from each position's `sum` and `count` of finite contributions:
/// `sum/count`, made non-decreasing by pooling adjacent violators (a pooled
/// block takes the mean of all its contributions), and each position without
/// contributions filled linearly between its nearest positions with some (the
/// nearest one's value past either end; NaN everywhere if none has any).
/// Means that already rise are kept as computed, bit for bit: without
/// infinities they always do, since every column adds at every position.
fn monotone_target(sum: &[f64], count: &[usize]) -> Vec<f64> {
    let n = sum.len();
    // Pool-adjacent-violators over the positions with contributions: blocks of
    // (sum, count, first position).
    let mut blocks: Vec<(f64, usize, usize)> = Vec::new();
    for i in (0..n).filter(|&i| count[i] > 0) {
        blocks.push((sum[i], count[i], i));
        while let [.., (s1, c1, first), (s2, c2, _)] = blocks[..] {
            if s1 / c1 as f64 <= s2 / c2 as f64 {
                break;
            }
            blocks.pop();
            *blocks.last_mut().unwrap() = (s1 + s2, c1 + c2, first);
        }
    }
    let mut target = vec![f64::NAN; n];
    for (b, &(s, c, first)) in blocks.iter().enumerate() {
        let end = blocks.get(b + 1).map_or(n, |next| next.2);
        let v = s / c as f64;
        for (t, _) in target[first..end]
            .iter_mut()
            .zip(&count[first..end])
            .filter(|(_, c)| **c > 0)
        {
            *t = v;
        }
    }
    // Fill the positions without contributions.
    let mut prev: Option<usize> = None;
    for i in 0..n {
        if count[i] == 0 {
            continue;
        }
        let lo = prev.map_or(0, |p| p + 1);
        for k in lo..i {
            target[k] = match prev {
                None => target[i],
                Some(p) => {
                    let w = (k - p) as f64 / (i - p) as f64;
                    target[p] + (target[i] - target[p]) * w
                }
            };
        }
        prev = Some(i);
    }
    if let Some(p) = prev {
        for k in p + 1..n {
            target[k] = target[p];
        }
    }
    target
}

/// `s` at 0-based fractional position `pos` in `[0, s.len() − 1]`, linear
/// between adjacent entries; an entry itself where `pos` is whole. `s` is
/// nondecreasing, and so is the result in `pos`: `lo + (hi − lo)·frac` is
/// monotone in `frac` under rounding and capped at `hi`, where
/// `lo·(1 − frac) + hi·frac` is not and can step back by an ulp over a nearly
/// flat stretch. The half position (the only fractional one a complete matrix
/// produces, from tied ranks) keeps the symmetric midpoint `lo·½ + hi·½`,
/// which lies between the two.
fn interpolate(s: &[f64], pos: f64) -> f64 {
    let lo = libm::floor(pos);
    let frac = pos - lo;
    let lo = lo as usize;
    if frac == 0.0 {
        s[lo]
    } else if frac == 0.5 {
        s[lo] * 0.5 + s[lo + 1] * 0.5
    } else {
        let (a, b) = (s[lo], s[lo + 1]);
        // The comparison form, unlike `min`, keeps a NaN (an infinite
        // target pair under AsExtremes) as NaN.
        let v = a + (b - a) * frac;
        if v > b { b } else { v }
    }
}
