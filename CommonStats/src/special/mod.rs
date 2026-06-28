//! Centralized special-function surface: all transcendental math the crate needs
//! has one home so accuracy and cross-host behavior have a single choke point.
pub mod elementary;
pub mod incomplete;
pub mod inverse;
pub mod misc;

pub use elementary::{beta, erf, erfc, gamma, lbeta, lgamma};
pub use incomplete::{betai, gammp, gammq};
pub use inverse::{erf_inv, erfc_inv, inv_beta_reg};
pub use misc::{digamma, logsumexp};
