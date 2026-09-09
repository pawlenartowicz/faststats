//! One-sample t-map under sign flips from sufficient statistics.

use alloc::vec::Vec;

use crate::error::NeuroError;

/// One-sample t statistic per node for sign-flipped data, from sufficient
/// statistics: `Q[v] = Σ_i x_iv²` (sign-invariant, computed once) and per
/// realization only `S[v] = Σ_i s_i · x_iv`.
///
/// Convention: `t = mean / sqrt(var / n)` with `var` the `ddof = 1` sample
/// variance, `mean = S/n`, `var = (Q − S²/n) / (n − 1)`; no hat/variance
/// smoothing. Matches `mne.stats.ttest_1samp_no_p(X, sigma=0)` to rounding
/// (`tests/perm_oracle.rs` G8: `rel 1e-12` against a two-pass mean/var).
/// Degenerate node (variance below `4ε·Q`, ε = f64 epsilon: the rounding
/// floor of `Q − S²/n`, whose two terms each carry up to `n·ε` relative error
/// from `n` accumulations — covers both all-equal and all-zero): `t = 0` where
/// MNE returns NaN/±inf, so the map stays finite and such nodes fall below
/// every TFCE threshold.
///
/// Layout: `x` is **subject-major**, `x[i · V + v]` = subject `i` at node `v`
/// (node order = `Domain` order), length `n · V`.
#[derive(Debug, Clone, PartialEq)]
pub struct OneSampleT {
    n: usize,
    v: usize,
    sumsq: Vec<f64>,
}

impl OneSampleT {
    /// Precompute `Q[v]` for `x` (subject-major, `n` subjects, `x.len() / n`
    /// nodes).
    ///
    /// Errors: [`NeuroError::TooFewSubjects`] if `n < 2`;
    /// [`NeuroError::MismatchedLengths`] if `x.len()` is not a multiple of `n`
    /// (`expected` = the next multiple).
    pub fn new(x: &[f64], n: usize) -> Result<Self, NeuroError> {
        if n < 2 {
            return Err(NeuroError::TooFewSubjects);
        }
        if x.len() % n != 0 {
            return Err(NeuroError::MismatchedLengths {
                expected: x.len().div_ceil(n) * n,
                got: x.len(),
            });
        }
        let v = x.len() / n;
        let mut sumsq = alloc::vec![0.0f64; v];
        for row in x.chunks_exact(v) {
            for (q, &xi) in sumsq.iter_mut().zip(row) {
                *q += xi * xi;
            }
        }
        Ok(Self { n, v, sumsq })
    }

    /// Number of subjects `n`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of nodes `V`.
    pub fn n_nodes(&self) -> usize {
        self.v
    }

    /// Write the t-map for realization `signs` into `out` — the only `O(n · V)`
    /// work per draw, allocation-free.
    ///
    /// `x`: the same subject-major data passed to [`new`](Self::new)
    /// (`len == n · V`). `signs`: one of `±1.0` per subject (`len == n`); all
    /// `+1.0` gives the observed map. `out`: `len == V`. Lengths are the
    /// caller's contract (checked by `debug_assert`, out-of-range indexing
    /// panics otherwise).
    pub fn t_map(&self, x: &[f64], signs: &[f64], out: &mut [f64]) {
        let (n, v) = (self.n, self.v);
        debug_assert_eq!(x.len(), n * v);
        debug_assert_eq!(signs.len(), n);
        debug_assert_eq!(out.len(), v);
        out.fill(0.0);
        for (row, &s) in x.chunks_exact(v).zip(signs) {
            for (acc, &xi) in out.iter_mut().zip(row) {
                *acc += s * xi;
            }
        }
        let nf = n as f64;
        for (t, &q) in out.iter_mut().zip(&self.sumsq) {
            let s = *t;
            let var = (q - s * s / nf) / (nf - 1.0);
            *t = if var <= 4.0 * f64::EPSILON * q {
                0.0
            } else {
                (s / nf) / libm::sqrt(var / nf)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn rejects_n_below_two_and_ragged_x() {
        assert_eq!(
            OneSampleT::new(&[1.0, 2.0, 3.0], 1),
            Err(NeuroError::TooFewSubjects)
        );
        assert_eq!(
            OneSampleT::new(&[1.0, 2.0, 3.0], 2),
            Err(NeuroError::MismatchedLengths {
                expected: 4,
                got: 3
            })
        );
    }

    // x = [[1, 0, 5], [3, 0, 5]] (n = 2, V = 3): node 0 mean 2, var 2, t = 2/sqrt(1) = 2;
    // node 1 all-zero → 0; node 2 constant → 0 (degenerate rule).
    #[test]
    fn t_map_identity_and_degenerate_nodes() {
        let x = [1.0, 0.0, 5.0, 3.0, 0.0, 5.0];
        let st = OneSampleT::new(&x, 2).unwrap();
        let mut t = vec![0.0; 3];
        st.t_map(&x, &[1.0, 1.0], &mut t);
        assert!((t[0] - 2.0).abs() < 1e-15, "{t:?}");
        assert_eq!(t[1], 0.0);
        assert_eq!(t[2], 0.0);
        // Flip subject 1: values [1, −3], mean −1, var 8, t = −1/sqrt(4) = −0.5.
        st.t_map(&x, &[1.0, -1.0], &mut t);
        assert!((t[0] + 0.5).abs() < 1e-15, "{t:?}");
    }
}
