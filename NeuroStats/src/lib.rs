//! NeuroStats — `no_std`, WASM-ready neuroimaging statistics.
//!
//! Present scope: a masked-volume adjacency graph ([`Domain`]) and the
//! threshold-free cluster enhancement operator ([`tfce()`]) over a statistic map
//! on that graph, validated against MNE-Python. Sign-flip permutation inference
//! for the one-sample case (`tfce_one_sample`). No I/O.
//!
//! Conventions are explicit and never defaulted: the caller states the voxel
//! connectivity ([`Conn`]) and the TFCE band weighting ([`Weighting`]), because
//! the reference tools (MNE vs FSL/PALM/SPM) disagree on the latter by several ×
//! on the same map.
//!
//! Cargo features, both on by default: `std` enables the `permute` module
//! (permutation inference, which needs `commonstats` and the standard library);
//! `parallel` adds the rayon-backed `tfce_one_sample_threads`. With
//! `default-features = false` only the `no_std` core ([`Domain`], [`tfce()`])
//! is compiled, which is what the `wasm32-unknown-unknown` build uses.
//!
//! This crate ships only user-facing docs; design notes and specs live in the
//! umbrella dev repo.
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
