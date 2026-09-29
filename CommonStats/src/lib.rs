//! CommonStats — WASM-first statistical core.
//!
//! Base (always compiled): special functions (`special`), the mergeable
//! accumulator core (`accum`), descriptives (`descriptive`), kernel density and
//! auto-histograms (`density`), distribution-free transforms (`transform`), and
//! hypothesis tests (`htest`), plus the fixed-bin histogram + ECDF
//! (`accum::histogram`). Feature-gated: `rng` (counter-based Philox + Lemire
//! bounded ints), `resample` (draw-addressable index generation, sign-flip
//! generation, the `NullDist`/`BootDist` distribution accumulators, and a
//! serial reference driver), and `dist` (continuous + discrete distribution
//! objects with CDF/SF/PDF/quantile).
//!
//! This crate ships only user-facing docs; design notes and specs live in the
//! umbrella dev repo.
//!
//! ## Accuracy policy
//!
//! Every function in `special::incomplete`, `dist::continuous`, and
//! `dist::discrete` is checked against mpmath by `scripts/accuracy_sweep.py`
//! and scored by an error ratio r = observed relative error / (max(κ, 1) · ε),
//! where κ is the function's condition number over its real inputs and
//! ε = 2⁻⁵² ≈ 2.2e-16. κ·ε is the error of a backward-stable algorithm, so r
//! counts how many times worse than that this implementation is: r ≤ 10 is
//! `limit` (at the accuracy the problem allows), 10 to 1000 is `review` (fixed
//! when cheap), and above 1000 is `bug` (a structural error such as
//! cancellation, a subnormal intermediate, or a coarse grid).
//!
//! Also a bug regardless of r: a probability outside [0, 1]; a NaN or error
//! for a valid input (a discrete quantile past `i64::MAX` is an error by
//! contract, not a bug); a quantile outside its support; and `quantile(1)` on
//! a bounded support other than its largest point of positive mass. A cdf
//! that decreases (an sf that increases) between two points of one parameter
//! set is a bug once the drop exceeds 10 · max(κ, 1) · ε relative; smaller
//! drops are reported as seam steps, where two branches each accurate to
//! about ε meet, not as violations.
//!
//! Where κ is huge because the function is near a zero of its own (a
//! location-family quantile landing near 0), r hides errors; the harness
//! also reports the absolute error there. Per-function results are in the
//! accuracy tables at the top of `special::incomplete`, `dist::continuous`,
//! and `dist::discrete`.
//!
//! ```
//! let g1 = [89., 88., 97., 92.];
//! let g2 = [84., 79., 81., 83.];
//! let r = commonstats::t_test_two(&g1, &g2, commonstats::VarAssumption::Welch).unwrap();
//! assert!(r.p_value < 0.05);
//! ```
#![no_std]
#![forbid(unsafe_code)]
// Public statistical items must be documented per the convention standard.
// `warn` for now; promote to `deny` once the backlog is filled.
#![warn(missing_docs)]

extern crate alloc;
// `#[cfg(test)] mod tests` blocks in `src/` compile inside this crate and need `std`
// (HashSet, println!); doctests and `tests/` are separate crates and get it for free.
#[cfg(test)]
extern crate std;

pub mod accum;
pub mod density;
pub mod descriptive;
#[cfg(feature = "dist")]
pub mod dist;
pub mod error;
pub mod htest;
pub mod nan;
#[cfg(feature = "resample")]
pub mod resample;
#[cfg(feature = "rng")]
pub mod rng;
pub mod special;
pub mod transform;

pub use error::StatError;
pub use nan::NanPolicy;

pub use accum::{HistResult, Histogram};
#[cfg(feature = "resample")]
pub use resample::{
    BootDist, NullDist, Scheme, Sidedness, gen_resample_indices, gen_sign_flips, run_serial,
};
#[cfg(feature = "rng")]
pub use rng::{
    CommonStatsRng, STREAM_TAG_RESAMPLE, STREAM_TAG_RLRT, STREAM_TAG_SIGNFLIP, STREAM_TAG_SIMULATE,
};

pub use descriptive::{
    Ddof, Describe, count, cov, describe, kurtosis, max, mean, median, min, pearson, range, sd,
    skewness, sum, var,
};
pub use htest::{
    CorMethod, TestResult, VarAssumption, anova_one_way, chi2_gof, chi2_independence, cor_test,
    f_test_var, t_test_one, t_test_paired, t_test_two,
};
