//! NeuroStats — `no_std`, WASM-ready neuroimaging statistics.
//!
//! Three layers, each usable on its own. No I/O: every entry point takes and
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
//! both. [`TfceParams`] carries the exponents, the threshold grid and the
//! [`Weighting`] rule. [`tfce_bands`], [`tfce_bands_into`] and
//! [`tfce_bands_naive`] take an explicit `(threshold, weight)` band list
//! instead of a grid.
//!
//! **Sign-flip permutation inference** (feature `std`). `tfce_one_sample`
//! runs the one-sample test end to end and returns a `TfceInference` with the
//! observed maps, the FWE-corrected and uncorrected p-maps and the max-TFCE
//! null. Its pieces are public so a caller can drive the loop itself:
//! [`OneSampleT`] is the t-map from sufficient statistics,
//! `sign_flip_plan` picks exhaustive enumeration or Monte Carlo and returns a
//! `SignFlipPlan`, `OneSampleProblem` holds everything fixed across draws,
//! `run_range` runs a range of draws into a `NullPartial` using a
//! `PermWorkspace`, and `finalize` turns a merged null into p-maps.
//! `tfce_one_sample_threads` (feature `parallel`) spreads the same draws over
//! a rayon pool.
//!
//! Conventions are explicit and never defaulted: the caller states the voxel
//! connectivity ([`Conn`]) and the TFCE band weighting ([`Weighting`]), because
//! the reference tools (MNE vs FSL/PALM/SPM) disagree on the latter by several ×
//! on the same map.
//!
//! Random draws are addressable, not streamed: the sign pattern of draw `b` is
//! a pure function of `(seed, b)`, and merging `NullPartial`s adds counts and
//! concatenates maxima, which `finalize` then sorts. So a run is reproducible
//! from its seed and bit-identical for every thread count and every partition
//! of `0..B` into `run_range` calls.
//!
//! Cargo features, both on by default: `std` pulls in `commonstats` and the
//! standard library and compiles the `permute` module (everything under
//! "sign-flip permutation inference" above); `parallel` implies `std` and adds
//! `rayon`, which is what `tfce_one_sample_threads` runs on. With
//! `default-features = false` only the `no_std` core is compiled — [`Domain`],
//! [`tfce()`] and their neighbours — and that is the configuration the
//! `wasm32-unknown-unknown` build uses.
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
pub mod onesample;
#[cfg(feature = "std")]
pub mod permute;
pub mod tfce;

pub use domain::{Conn, Domain};
pub use error::NeuroError;
pub use onesample::OneSampleT;
#[cfg(feature = "parallel")]
pub use permute::tfce_one_sample_threads;
#[cfg(feature = "std")]
pub use permute::{
    NullPartial, OneSampleProblem, PermWorkspace, SignFlipPlan, TfceInference, finalize, run_range,
    sign_flip_plan, tfce_one_sample,
};
pub use tfce::{
    TfceParams, TfceWorkspace, Weighting, tfce, tfce_bands, tfce_bands_into, tfce_bands_naive,
    tfce_into, tfce_naive,
};
