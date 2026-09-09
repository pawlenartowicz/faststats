//! The mergeable-accumulator core. `merge` is the whole game: one op gives
//! one-pass streaming, parallel chunking, and incremental recompute — never
//! downdating.
pub mod histogram;
pub mod moments;
pub mod simple;
pub mod tdigest;

pub use histogram::{HistResult, Histogram};
pub use tdigest::{TDigest, TDigestResult, quantile_edges};

/// Combine two partial states. The merge semigroup the resampling seam requires.
pub trait Mergeable {
    /// Fold `other`'s state into `self`. Associative, so chunked and streaming
    /// folds agree.
    fn merge(&mut self, other: &Self);
}

/// Fold observations in, read a result out. `empty` is the monoid identity.
pub trait Accumulator: Mergeable {
    /// One observation fed to [`update`](Accumulator::update).
    type Item;
    /// The value produced by [`finalize`](Accumulator::finalize).
    type Output;
    /// The identity state: merging it into another changes nothing.
    fn empty() -> Self;
    /// Fold one observation into the running state.
    fn update(&mut self, x: Self::Item);
    /// Read the current result out of the state.
    fn finalize(&self) -> Self::Output;
}

/// Fold an f64 slice into any accumulator, skipping NaN (the Omit policy applied
/// by construction; see [`crate::nan`]). Batch entry points build on this.
pub fn from_slice<A: Accumulator<Item = f64>>(xs: &[f64]) -> A {
    let mut a = A::empty();
    for &x in xs {
        if !x.is_nan() {
            a.update(x);
        }
    }
    a
}

/// Type-7 (R/NumPy default) quantile of an ascending slice by linear interpolation
/// of order statistics: position `q·(n−1)`, interpolate between the bracketing
/// values. Matches `numpy.percentile(method="linear")`. Empty slice → NaN.
/// Shared by `descriptive::describe` and the bootstrap-CI percentiles.
pub(crate) fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let h = (n as f64 - 1.0) * q;
    let lo = libm::floor(h) as usize;
    let hi = libm::ceil(h) as usize;
    let frac = h - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}
