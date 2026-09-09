//! Hypothesis tests (named `htest` to avoid Cargo's `tests/` integration dir).
//! Test statistics come off the accumulators; p-values come straight from the
//! `special` functions (no distribution-object suite yet).
pub mod anova;
pub mod chi2;
pub mod ci;
pub mod cor;
pub mod effect;
pub mod result;
pub mod ttest;

pub use anova::{anova_one_way, f_test_var};
pub use chi2::{chi2_gof, chi2_independence};
pub use ci::{ci_correlation, ci_mean, ci_mean_diff, ci_mean_diff_welch, ci_proportion};
pub use cor::{CorMethod, cor_test};
pub use effect::{cohen_d, cramers_v, eta_squared};
pub use result::{Ci, EffectSize, TestResult};
pub use ttest::{VarAssumption, t_test_one, t_test_paired, t_test_two};
