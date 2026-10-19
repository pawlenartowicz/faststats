//! The brain store and the nuisance basis it is residualised against.
//!
//! Permutation inference reads the brain data once per batch of draws, one
//! node at a time. [`Store`] is the copy laid out for that: voxel-major `f64`,
//! each node's `n` values contiguous, already residualised against the
//! nuisance columns and scaled to unit norm. With the data in that form a
//! statistic of a draw is a handful of dot products per node and no model is
//! refitted.
//!
//! [`nuisance_basis`] turns the confound columns (and the constant column,
//! when the model has an intercept that is not the tested term) into the
//! orthonormal basis the store is residualised against. Both are plain linear
//! algebra with no random draws, so they live in the `no_std` core.
//!
//! The oracle for both is the textbook least-squares residual from the normal
//! equations on the raw columns
//! (`tests/ols_oracle.rs::store_columns_are_unit_norm_ols_residuals`).

use alloc::vec::Vec;

use crate::error::NeuroError;

/// A column is in the span of the columns before it when its norm after
/// residualising is at most this fraction of its norm before.
const SPAN_TOL: f64 = 1e-10;

/// Nodes gathered and residualised together in [`Store`] construction: a
/// tile of this many columns of `n ≈ 250` values stays inside a typical L2
/// cache while the source rows are read in contiguous runs.
const TILE: usize = 256;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    libm::sqrt(dot(a, a))
}

/// Subtract from `v` its projection on each n-vector of `basis` in turn
/// (modified Gram–Schmidt: each projection is taken from the already updated
/// `v`). `basis` holds orthonormal n-vectors one after another, `n = v.len()`.
fn residualise(v: &mut [f64], basis: &[f64]) {
    for u in basis.chunks_exact(v.len()) {
        let c = dot(u, v);
        for (x, &y) in v.iter_mut().zip(u) {
            *x -= c * y;
        }
    }
}

/// Residualise `col` against `basis` (modified Gram–Schmidt with one
/// re-orthogonalisation pass) and scale it to unit norm. Returns `Ok(false)`
/// when the column is in the span of `basis` (remaining norm at most
/// [`SPAN_TOL`] of the original norm; `col` is then left residualised and
/// unscaled). Errors with [`NeuroError::NonFiniteStat`] when the column's norm
/// is not finite.
pub(crate) fn orthonormalise(col: &mut [f64], basis: &[f64]) -> Result<bool, NeuroError> {
    let before = norm(col);
    if !before.is_finite() {
        return Err(NeuroError::NonFiniteStat);
    }
    residualise(col, basis);
    residualise(col, basis);
    let after = norm(col);
    if after <= SPAN_TOL * before {
        return Ok(false);
    }
    for x in col.iter_mut() {
        *x /= after;
    }
    Ok(true)
}

/// Orthonormal basis `Q` of the nuisance space of a linear model: the
/// confound columns, preceded by a constant column when `intercept` is set.
///
/// Convention: modified Gram–Schmidt with one re-orthogonalisation pass, the
/// constant column first and then the confound columns in their given order.
/// A column whose norm after residualising against the columns already kept
/// is at most `1e-10` of its norm before is dropped (a zero column, a
/// duplicate, a combination of earlier columns, a constant column when the
/// intercept is present). With `intercept` set the first vector is
/// `1/sqrt(n)` and the others are the centred confounds, orthonormalised. Only
/// the span enters a statistic, so the order changes which duplicate is
/// dropped and nothing else. Checked against least-squares residuals from the
/// normal equations on the raw columns
/// (`tests/ols_oracle.rs::store_columns_are_unit_norm_ols_residuals`).
///
/// `confounds`: row-major `n × c`, `confounds[i · c + j]` is confound `j` of
/// subject `i`; empty for no confounds. `n ≥ 1`: number of subjects.
/// `intercept`: `true` when the model has an intercept that is a nuisance
/// term (a regressor is tested), `false` when the intercept itself is tested.
///
/// Returns the `q` kept vectors one after another: vector `k` is
/// `out[k · n .. (k + 1) · n]`, and `q = out.len() / n`.
///
/// Errors: [`NeuroError::TooFewSubjects`] if `n == 0`;
/// [`NeuroError::MismatchedLengths`] if `confounds.len()` is not a multiple of
/// `n` (`expected` = the next multiple); [`NeuroError::NonFiniteStat`] if a
/// column holds a NaN or ±∞, or is so large that its sum of squares overflows.
///
/// ```
/// use neurostats::nuisance_basis;
/// // 4 subjects, one confound, with an intercept.
/// let q = nuisance_basis(&[1.0, 2.0, 3.0, 6.0], 4, true).unwrap();
/// assert_eq!(q.len(), 2 * 4);
/// assert!(q[..4].iter().all(|&x| (x - 0.5).abs() < 1e-15));
/// let dot: f64 = q[..4].iter().zip(&q[4..]).map(|(a, b)| a * b).sum();
/// assert!(dot.abs() < 1e-15);
/// ```
pub fn nuisance_basis(
    confounds: &[f64],
    n: usize,
    intercept: bool,
) -> Result<Vec<f64>, NeuroError> {
    if n == 0 {
        return Err(NeuroError::TooFewSubjects);
    }
    if confounds.len() % n != 0 {
        return Err(NeuroError::MismatchedLengths {
            expected: confounds.len().div_ceil(n) * n,
            got: confounds.len(),
        });
    }
    let c = confounds.len() / n;
    let mut basis = Vec::with_capacity((c + usize::from(intercept)) * n);
    let mut col = alloc::vec![1.0f64; n];
    if intercept && orthonormalise(&mut col, &basis)? {
        basis.extend_from_slice(&col);
    }
    for j in 0..c {
        for (i, x) in col.iter_mut().enumerate() {
            *x = confounds[i * c + j];
        }
        if orthonormalise(&mut col, &basis)? {
            basis.extend_from_slice(&col);
        }
    }
    Ok(basis)
}

/// Voxel-major `f64` copy of the brain data, ready for the draw loop: each
/// node's column of `n` values is residualised against a nuisance basis and
/// scaled to unit norm.
///
/// Convention: column `v` is `(y_v − Q Q' y_v) / ‖y_v − Q Q' y_v‖`, with `Q`
/// the basis passed to the constructor (one modified Gram–Schmidt pass). A
/// degenerate node, one whose norm after residualising is at most `1e-10` of
/// its norm before, is stored as zeros: an all-zero node, a node that is a
/// combination of the nuisance columns, a constant node when the basis holds
/// the intercept. Such a node gets statistic 0 in every draw. Checked against
/// least-squares residuals from the normal equations on the raw columns
/// (`tests/ols_oracle.rs::store_columns_are_unit_norm_ols_residuals`, `rel
/// 1e-10`).
///
/// Memory: `n · V · 8` bytes for the data plus the basis; the source is read
/// in place and never copied whole.
#[derive(Debug, Clone, PartialEq)]
pub struct Store {
    n: usize,
    n_nodes: usize,
    basis: Vec<f64>,
    data: Vec<f64>,
}

impl Store {
    /// Build the store from an `f64` source.
    ///
    /// `src`: subject-major, `src[r · n_nodes + v]` is source row `r` at node
    /// `v` (node order = `Domain` order), `len == n · n_nodes`. `n`: number of
    /// rows of the source. `n_nodes`: number of nodes `V`. `rows`: the source
    /// rows that enter the model, in model order (`0..n` for all of them);
    /// the store has `rows.len()` subjects. `basis`: the nuisance basis from
    /// [`nuisance_basis`] for those subjects, `q` vectors of `rows.len()`
    /// values each; empty for none.
    ///
    /// One pass over the source, in tiles of nodes: gather the selected rows,
    /// convert to `f64`, transpose to voxel-major, residualise and scale.
    ///
    /// Errors: [`NeuroError::MismatchedLengths`] if `src.len() != n · n_nodes`
    /// or `basis.len()` is not a multiple of `rows.len()`;
    /// [`NeuroError::TooFewSubjects`] if `rows` is empty;
    /// [`NeuroError::InvalidParams`] if a row index is `≥ n`;
    /// [`NeuroError::NonFiniteStat`] if a selected value is NaN or ±∞, or a
    /// node's sum of squares overflows.
    pub fn from_f64(
        src: &[f64],
        n: usize,
        n_nodes: usize,
        rows: &[usize],
        basis: &[f64],
    ) -> Result<Self, NeuroError> {
        Self::build(src, n, n_nodes, rows, basis)
    }

    /// Build the store from an `f32` source. Arguments, convention and errors
    /// are those of [`from_f64`](Self::from_f64); each value is widened to
    /// `f64` as it is gathered, so the result is bit-identical to converting
    /// the source first.
    pub fn from_f32(
        src: &[f32],
        n: usize,
        n_nodes: usize,
        rows: &[usize],
        basis: &[f64],
    ) -> Result<Self, NeuroError> {
        Self::build(src, n, n_nodes, rows, basis)
    }

    fn build<T: Copy + Into<f64>>(
        src: &[T],
        n: usize,
        n_nodes: usize,
        rows: &[usize],
        basis: &[f64],
    ) -> Result<Self, NeuroError> {
        if n.checked_mul(n_nodes) != Some(src.len()) {
            return Err(NeuroError::MismatchedLengths {
                expected: n.saturating_mul(n_nodes),
                got: src.len(),
            });
        }
        let m = rows.len();
        if m == 0 {
            return Err(NeuroError::TooFewSubjects);
        }
        if basis.len() % m != 0 {
            return Err(NeuroError::MismatchedLengths {
                expected: basis.len().div_ceil(m) * m,
                got: basis.len(),
            });
        }
        if rows.iter().any(|&r| r >= n) {
            return Err(NeuroError::InvalidParams);
        }
        let mut data = alloc::vec![0.0f64; m * n_nodes];
        for (t, tile) in data.chunks_mut(TILE * m).enumerate() {
            let first = t * TILE;
            let width = tile.len() / m;
            for (i, &r) in rows.iter().enumerate() {
                let row = &src[r * n_nodes + first..][..width];
                for (k, &x) in row.iter().enumerate() {
                    tile[k * m + i] = x.into();
                }
            }
            for col in tile.chunks_exact_mut(m) {
                let before = norm(col);
                if !before.is_finite() {
                    return Err(NeuroError::NonFiniteStat);
                }
                residualise(col, basis);
                let after = norm(col);
                if after <= SPAN_TOL * before {
                    col.fill(0.0);
                } else {
                    for x in col.iter_mut() {
                        *x /= after;
                    }
                }
            }
        }
        Ok(Self {
            n: m,
            n_nodes,
            basis: basis.to_vec(),
            data,
        })
    }

    /// Number of subjects `n` (the selected rows).
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of nodes `V`.
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of vectors `q` in the nuisance basis.
    pub fn q(&self) -> usize {
        self.basis.len() / self.n
    }

    /// The nuisance basis the columns were residualised against: `q` vectors
    /// of `n` values one after another, as [`nuisance_basis`] returns them.
    pub fn basis(&self) -> &[f64] {
        &self.basis
    }

    /// The stored columns, voxel-major: `as_slice()[v · n + i]` is subject
    /// `i` at node `v`, `len == n · V`.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // Columns [3, 4] and [6, 8] (n = 2): the second is twice the first and is
    // dropped; the first becomes [0.6, 0.8].
    #[test]
    fn basis_hand_case_normalises_and_drops_a_multiple() {
        let q = nuisance_basis(&[3.0, 6.0, 4.0, 8.0], 2, false).unwrap();
        assert_eq!(q.len(), 2);
        assert!(
            (q[0] - 0.6).abs() < 1e-15 && (q[1] - 0.8).abs() < 1e-15,
            "{q:?}"
        );
    }

    // n = 2, V = 2, basis [0.6, 0.8]. Node 0 is [3, 4]: in the span, stored as
    // zeros. Node 1 is [4, -3]: orthogonal to the basis, norm 5, stored as
    // [0.8, -0.6].
    #[test]
    fn store_hand_case_residualises_scales_and_zeroes() {
        let st = Store::from_f64(&[3.0, 4.0, 4.0, -3.0], 2, 2, &[0, 1], &[0.6, 0.8]).unwrap();
        assert_eq!((st.n(), st.n_nodes(), st.q()), (2, 2, 1));
        let d = st.as_slice();
        assert_eq!(&d[..2], &[0.0, 0.0]);
        assert!(
            (d[2] - 0.8).abs() < 1e-15 && (d[3] + 0.6).abs() < 1e-15,
            "{d:?}"
        );
        assert_eq!(st.basis(), &[0.6, 0.8]);
    }

    #[test]
    fn tiles_cover_every_node_once() {
        // V = 2·TILE + 3 with one subject and no basis: every stored value is
        // ±1 (a unit-norm column of length 1), in source order.
        let v = 2 * TILE + 3;
        let src: Vec<f64> = (0..v)
            .map(|k| if k % 3 == 0 { -2.0 } else { 5.0 })
            .collect();
        let st = Store::from_f64(&src, 1, v, &[0], &[]).unwrap();
        let want: Vec<f64> = (0..v)
            .map(|k| if k % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        assert_eq!(st.as_slice(), &want[..]);
        assert_eq!(vec![0.0f64; 0], st.basis());
    }
}
