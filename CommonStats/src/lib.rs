//! CommonStats — WASM-first statistical core.
//!
//! Base (always compiled): special functions (`special`), the mergeable
//! accumulator core (`accum`), descriptives (`descriptive`), kernel density and
//! auto-histograms (`density`), distribution-free transforms (`transform`), and
//! hypothesis tests (`htest`), plus the fixed-bin histogram + ECDF
//! (`accum::histogram`). Feature-gated: `rng` (counter-based Philox + Lemire
//! bounded ints), `resample` (draw-addressable index generation, the
//! `NullDist`/`BootDist` distribution accumulators, and a serial reference
//! driver), and `dist` (continuous + discrete distribution objects with
//! CDF/SF/PDF/quantile).
//!
//! This crate ships only user-facing docs; design notes and specs live in the
//! umbrella dev repo.
//!
//! ```
//! let g1 = [89., 88., 97., 92.];
//! let g2 = [84., 79., 81., 83.];
//! let r = commonstats::t_test_two(&g1, &g2, commonstats::VarAssumption::Welch).unwrap();
//! assert!(r.p_value < 0.05);
//! ```
#![forbid(unsafe_code)]
// Public statistical items must be documented per the convention standard.
// `warn` for now; promote to `deny` once the backlog is filled.
#![warn(missing_docs)]

pub mod error;
pub mod nan;
pub mod special;
pub mod accum;
pub mod descriptive;
pub mod density;
pub mod htest;
#[cfg(feature = "rng")]
pub mod rng;
#[cfg(feature = "resample")]
pub mod resample;
#[cfg(feature = "dist")]
pub mod dist;
pub mod transform;

pub use error::StatError;
pub use nan::NanPolicy;

pub use accum::{HistResult, Histogram};
#[cfg(feature = "rng")]
pub use rng::{CommonStatsRng, STREAM_TAG_RESAMPLE};
#[cfg(feature = "resample")]
pub use resample::{gen_resample_indices, run_serial, BootDist, NullDist, Scheme, Sidedness};

pub use descriptive::{
    count, cov, describe, kurtosis, max, mean, median, min, pearson, range, sd, skewness,
    sum, var, Ddof, Describe,
};
pub use htest::{
    anova_one_way, chi2_gof, chi2_independence, cor_test, f_test_var, t_test_one, t_test_paired,
    t_test_two, CorMethod, TestResult, VarAssumption,
};
