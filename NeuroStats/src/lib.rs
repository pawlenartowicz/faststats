//! NeuroStats — `no_std`, WASM-ready neuroimaging statistics.
//!
//! Four layers, each usable on its own. No I/O: every entry point takes and
//! returns slices, and nothing here opens a file or a network socket.
//!
//! **Adjacency.** [`Domain`] is an undirected graph over the in-mask voxels of
//! a 3-D volume, built with [`Domain::from_volume`] (dense grid) or
//! [`Domain::from_mask`] (a boolean mask), or from a caller's neighbour list
//! with [`Domain::from_csr`] — the route for surface meshes and scipy sparse
//! matrices. [`Domain::to_csr`] hands the adjacency back in that shape,
//! [`Domain::neighbours_into`] reads one row, and [`Domain::n_nodes`] /
//! [`Domain::n_edges`] report its size. Node `i` is the `i`-th in-mask voxel in
//! row-major (C-order) scan of `dims`; every statistic map must use that order.
//!
//! **TFCE.** [`tfce()`] enhances a statistic map over a [`Domain`] and
//! [`tfce_into`] does it into a caller-owned buffer with a reusable
//! [`TfceWorkspace`]; [`tfce_naive`] is the readable re-clustering oracle for
//! both. [`tfce_tail_into`] does the same for a chosen [`Tail`]: the map, the
//! negated map, or the signed difference of the two. [`TfceParams`] carries
//! the exponents, the threshold grid and the [`Weighting`] rule. [`tfce_bands`], [`tfce_bands_into`] and
//! [`tfce_bands_naive`] take an explicit `(threshold, weight)` band list
//! instead of a grid.
//!
//! **Brain store.** [`Store`] is the copy of the brain data that permutation
//! inference reads: voxel-major `f64`, each node's column residualised
//! against a nuisance basis and scaled to unit norm, built in one pass from
//! an `f32` or `f64` source and a row index. [`nuisance_basis`]
//! orthonormalises the confound columns and the intercept into that basis.
//!
//! **Permutation inference** (feature `std`). `tfce_ols` runs TFCE
//! permutation inference for the linear model
//! `brain ~ tested + confounds + intercept` fitted at every node, and returns
//! a `TfceInference`: the observed statistic, the FWE-corrected and
//! uncorrected p-maps, and a `NullState` with the observed TFCE map, the
//! max-TFCE null and the settings of the run. With no tested regressor it
//! tests the intercept, which without confounds is the one-sample sign-flip
//! test. Its pieces are public so a caller can drive the loop itself:
//! `PermOpts` carries the TFCE settings, the tail, the seed, the draw budget
//! and the batch size; `PermProblem::ols` holds everything fixed across
//! draws; `stat_batch` gives the statistic maps of a batch of draws;
//! `run_range` runs a range of draws into a `NullPartial` using a
//! `PermWorkspace`; and `finalize` turns a merged null into p-maps.
//! `draw_permutation` is the row shuffle of one draw. `run_threads` and
//! `tfce_ols_threads` (feature `parallel`) spread the same draws over a rayon
//! pool.
//!
//! Conventions are explicit and never defaulted: the caller states the voxel
//! connectivity ([`Conn`]), the TFCE band weighting ([`Weighting`]) and the
//! tail ([`Tail`]), because the reference tools (MNE vs FSL/PALM/SPM) disagree
//! on the weighting by several × on the same map.
//!
//! Random draws are addressable, not streamed: the row shuffle or sign
//! pattern of draw `b` is a pure function of `(seed, b)`, every dot product of
//! the statistic is one sum in subject order whatever the batch, and merging
//! `NullPartial`s adds counts and concatenates maxima, which `finalize` then
//! sorts. So a run is reproducible from its seed and bit-identical for every
//! thread count, every batch size and every partition of the draws into
//! `run_range` calls.
//!
//! Cargo features, both on by default: `std` pulls in `commonstats` and the
//! standard library and compiles the `permute` module (everything under
//! "permutation inference" above); `parallel` implies `std` and adds `rayon`,
//! which is what `run_threads` and `tfce_ols_threads` run on. With
//! `default-features = false` only the `no_std` core is compiled — [`Domain`],
//! [`tfce()`], [`Store`] and their neighbours — and that is the configuration
//! the `wasm32-unknown-unknown` build uses.
//!
//! ```
//! use neurostats::{Conn, Domain, TfceParams, Weighting, tfce};
//!
//! // 2×2×2 volume, all voxels in-mask, face connectivity.
//! let dom = Domain::from_volume([2, 2, 2], Conn::Face);
//! let stat = [3.0, 3.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0];
//! let p = TfceParams { e: 0.5, h: 2.0, start: 0.4, step: 0.4, weighting: Weighting::SmithNichols };
//! let enh = tfce(&dom, &stat, &p).unwrap();
//! assert!(enh[0] > 0.0 && enh[2] == 0.0);
//! ```
#![no_std]
#![forbid(unsafe_code)]
// Public statistical items must be documented per the convention standard.
#![warn(missing_docs)]

extern crate alloc;

#[cfg(any(test, feature = "std"))]
extern crate std;

pub mod domain;
pub mod error;
#[cfg(feature = "std")]
pub mod permute;
pub mod store;
pub mod tfce;

pub use domain::{Conn, Domain};
pub use error::NeuroError;
#[cfg(feature = "std")]
pub use permute::{
    Engine, NullPartial, NullState, PermOpts, PermProblem, PermWorkspace, TfceInference,
    draw_permutation, finalize, run_range, stat_batch, tfce_ols,
};
#[cfg(feature = "parallel")]
pub use permute::{run_threads, tfce_ols_threads};
pub use store::{Store, nuisance_basis};
pub use tfce::{
    Tail, TfceParams, TfceWorkspace, Weighting, tfce, tfce_bands, tfce_bands_into,
    tfce_bands_naive, tfce_into, tfce_naive, tfce_tail_into,
};
