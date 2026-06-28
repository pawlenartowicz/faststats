//! The CommonStats half of the frozen resampling seam
//! (permutation-friendly.md §"The frozen seam"): draw-addressable, no-alloc
//! index generation; the [`NullDist`]/[`BootDist`] distribution accumulators; and
//! a serial reference driver that is both the standalone path and SDOC's
//! differential-testing oracle. Orchestration (worker pool, SAB provider,
//! node-hash seeding) is SDOC's and stays out of this crate.
mod dist;

pub use dist::{BootDist, NullDist, Sidedness};

use crate::accum::Accumulator;
use crate::error::StatError;
use crate::rng::CommonStatsRng;

/// Row-index resampling scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// Fisher–Yates shuffle of `0..n` — sampling without replacement.
    Permutation,
    /// `n` draws with replacement (case/pairs bootstrap).
    Bootstrap,
}

/// Fill `out` with the row indices for one resample draw, draw-addressable and
/// allocation-free.
///
/// Convention: **Permutation** is a Fisher–Yates shuffle of `0..n` (each of the
/// `n!` orderings equally likely); **Bootstrap** is `n` independent draws with
/// replacement from `0..n` (case/pairs bootstrap). Both source their randomness
/// from [`CommonStatsRng`] keyed by `(seed, draw_id)` via Lemire unbiased bounded
/// integers, so draw *k* is reproducible and 1-vs-N-thread invariant.
///
/// `scheme`: which resample. `n`: population/resample size. `draw_id`: the draw
/// index in `0..B`. `seed`: the run seed. `out`: caller-owned buffer that
/// receives the indices; **its length must equal `n`**.
///
/// Returns [`StatError::MismatchedLengths`] when `out.len() != n`.
///
/// ```
/// use commonstats::{gen_resample_indices, Scheme};
/// let mut idx = [0u32; 5];
/// gen_resample_indices(Scheme::Permutation, 5, 0, 42, &mut idx).unwrap();
/// let mut sorted = idx;
/// sorted.sort_unstable();
/// assert_eq!(sorted, [0, 1, 2, 3, 4]); // a permutation: every row exactly once
/// ```
pub fn gen_resample_indices(
    scheme: Scheme,
    n: usize,
    draw_id: u64,
    seed: u64,
    out: &mut [u32],
) -> Result<(), StatError> {
    if out.len() != n {
        return Err(StatError::MismatchedLengths { a: n, b: out.len() });
    }
    if n == 0 {
        return Ok(());
    }
    let mut rng = CommonStatsRng::new(seed, draw_id);
    match scheme {
        Scheme::Permutation => {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = i as u32;
            }
            // Fisher–Yates: for i = n-1 down to 1, swap i with a uniform j in [0, i].
            for i in (1..n).rev() {
                let j = rng.bounded((i + 1) as u32) as usize;
                out.swap(i, j);
            }
        }
        Scheme::Bootstrap => {
            let n32 = n as u32;
            for slot in out.iter_mut() {
                *slot = rng.bounded(n32);
            }
        }
    }
    Ok(())
}

/// Serial reference driver: run `b` resample draws, inject the consumer's
/// `statistic` on each, and fold the scalar result into `dist`.
///
/// This is the standalone resampling path **and** the differential-testing oracle
/// for SDOC's parallel driver (same role the interpreter plays for a compiled
/// loop). It reuses one index buffer across draws (RAM flat in `B`) and seeds
/// each draw by `(seed, draw_id)`, so the result set is identical regardless of
/// thread count.
///
/// `statistic(base, idx)` is **injected** by the consumer: `base` is the
/// immutable column set, `idx` the current draw's row indices; it returns the
/// per-draw scalar. (P2 carries the scalar form of the seam's
/// `statistic(base, fit, idx) -> Stat: Mergeable`: the `fit` prelude slot and the
/// vector-valued return are the reserved GLMM-era extensions.)
///
/// `dist` is constructed and owned by the caller (e.g. `NullDist::new(observed,
/// side)` or `BootDist::new(level)`).
///
/// Returns [`StatError::DomainError`] when `b == 0`.
///
/// ```
/// use commonstats::{run_serial, NullDist, Scheme, Sidedness};
/// use commonstats::accum::Accumulator; // brings `finalize` into scope
/// // Permutation test that the mean of one column exceeds a fixed reference.
/// let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let base: &[&[f64]] = &[&data];
/// let stat = |b: &[&[f64]], idx: &[u32]| {
///     let col = b[0];
///     idx.iter().map(|&i| col[i as usize]).sum::<f64>() / idx.len() as f64
/// };
/// let observed = stat(base, &[0, 1, 2, 3, 4, 5]); // mean on the identity index
/// let mut dist = NullDist::new(observed, Sidedness::Greater);
/// run_serial(Scheme::Permutation, base, data.len(), 200, 7, stat, &mut dist).unwrap();
/// let p = dist.finalize();
/// assert!(p > 0.0 && p <= 1.0);
/// ```
pub fn run_serial<D, F>(
    scheme: Scheme,
    base: &[&[f64]],
    n: usize,
    b: u64,
    seed: u64,
    statistic: F,
    dist: &mut D,
) -> Result<(), StatError>
where
    D: Accumulator<Item = f64>,
    F: Fn(&[&[f64]], &[u32]) -> f64,
{
    if b == 0 {
        return Err(StatError::DomainError("run_serial: B must be >= 1"));
    }
    let mut idx = vec![0u32; n];
    for draw_id in 0..b {
        gen_resample_indices(scheme, n, draw_id, seed, &mut idx)?;
        dist.update(statistic(base, &idx));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_is_a_valid_permutation() {
        let n = 50;
        for draw_id in 0..20 {
            let mut idx = vec![0u32; n];
            gen_resample_indices(Scheme::Permutation, n, draw_id, 123, &mut idx).unwrap();
            let mut sorted = idx.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n as u32).collect::<Vec<_>>(), "draw {draw_id} not a permutation");
        }
    }

    #[test]
    fn bootstrap_indices_in_range_and_has_replacement() {
        let n = 40;
        let mut idx = vec![0u32; n];
        gen_resample_indices(Scheme::Bootstrap, n, 0, 9, &mut idx).unwrap();
        assert!(idx.iter().all(|&i| (i as usize) < n));
        // With replacement: over n draws from n rows, collisions are near-certain.
        let distinct: std::collections::HashSet<u32> = idx.iter().copied().collect();
        assert!(distinct.len() < n, "bootstrap of n from n should repeat some rows");
    }

    #[test]
    fn indices_are_draw_addressable() {
        let n = 30;
        let mut a = vec![0u32; n];
        let mut b = vec![0u32; n];
        gen_resample_indices(Scheme::Permutation, n, 5, 77, &mut a).unwrap();
        gen_resample_indices(Scheme::Permutation, n, 5, 77, &mut b).unwrap();
        assert_eq!(a, b, "same (seed, draw_id) must reproduce the index set");
        let mut c = vec![0u32; n];
        gen_resample_indices(Scheme::Permutation, n, 6, 77, &mut c).unwrap();
        assert_ne!(a, c, "different draw_id must differ");
    }

    #[test]
    fn length_mismatch_errors() {
        let mut idx = vec![0u32; 4];
        assert_eq!(
            gen_resample_indices(Scheme::Bootstrap, 5, 0, 1, &mut idx),
            Err(StatError::MismatchedLengths { a: 5, b: 4 }),
        );
    }

    #[test]
    fn run_serial_b_zero_errors() {
        let data = [1.0, 2.0, 3.0];
        let base: &[&[f64]] = &[&data];
        let mut dist = NullDist::new(2.0, Sidedness::Greater);
        let stat = |b: &[&[f64]], idx: &[u32]| b[0][idx[0] as usize];
        assert!(matches!(
            run_serial(Scheme::Permutation, base, 3, 0, 1, stat, &mut dist),
            Err(StatError::DomainError(_)),
        ));
    }

    // Permutation of a constant statistic: every draw gives the same mean (the
    // grand mean is permutation-invariant), so every draw ties the observed and
    // the p-value is exactly 1.0 (all B draws extreme: (B+1)/(B+1)).
    #[test]
    fn run_serial_permutation_mean_is_invariant() {
        let data = [10.0, 20.0, 30.0, 40.0];
        let base: &[&[f64]] = &[&data];
        let stat = |b: &[&[f64]], idx: &[u32]| {
            idx.iter().map(|&i| b[0][i as usize]).sum::<f64>() / idx.len() as f64
        };
        let observed = stat(base, &[0, 1, 2, 3]);
        let mut dist = NullDist::new(observed, Sidedness::TwoSided);
        run_serial(Scheme::Permutation, base, 4, 100, 3, stat, &mut dist).unwrap();
        assert!((dist.finalize() - 1.0).abs() < 1e-15);
    }

    // Bootstrap CI of the mean brackets the data mean (the bootstrap distribution
    // of the sample mean is centred on it).
    #[test]
    fn run_serial_bootstrap_ci_brackets_mean() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let base: &[&[f64]] = &[&data];
        let mean = data.iter().sum::<f64>() / data.len() as f64; // 5.0
        let stat = |b: &[&[f64]], idx: &[u32]| {
            idx.iter().map(|&i| b[0][i as usize]).sum::<f64>() / idx.len() as f64
        };
        let mut dist = BootDist::new(0.95);
        run_serial(Scheme::Bootstrap, base, data.len(), 2000, 1, stat, &mut dist).unwrap();
        let ci = dist.finalize();
        assert!(ci.lower < mean && mean < ci.upper, "CI {:?} must bracket {mean}", ci);
    }
}
