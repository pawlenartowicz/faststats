//! Hypothesis tests (named `htest` to avoid Cargo's `tests/` integration dir).
//! Test statistics come off the accumulators; p-values come straight from the
//! `special` functions (no distribution-object suite yet).
pub mod result;
pub mod ttest;
pub mod anova;
pub mod chi2;
pub mod cor;
pub mod effect;
pub mod ci;

pub use result::{Ci, EffectSize, TestResult};
pub use ttest::{t_test_one, t_test_paired, t_test_two, VarAssumption};
pub use anova::{anova_one_way, f_test_var};
pub use chi2::{chi2_gof, chi2_independence};
pub use cor::{cor_test, CorMethod};
pub use effect::{cohen_d, cramers_v, eta_squared};
pub use ci::{ci_correlation, ci_mean, ci_mean_diff, ci_mean_diff_welch, ci_proportion};
