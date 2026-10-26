//! The result/reporting family. One struct shape for every test.

/// A two-sided confidence interval.
#[derive(Debug, Clone, PartialEq)]
pub struct Ci {
    /// Lower endpoint.
    pub lower: f64,
    /// Upper endpoint.
    pub upper: f64,
    /// Nominal coverage probability (e.g. 0.95).
    pub level: f64,
}

/// Effect-size payload attached to a test, tagged by kind.
#[derive(Debug, Clone, PartialEq)]
pub enum EffectSize {
    /// Cohen's d (pooled SD).
    CohenD(f64),
    /// η² (eta-squared): proportion of variance between groups.
    EtaSquared(f64),
    /// Cramér's V.
    CramersV(f64),
    /// Correlation coefficient r.
    R(f64),
}

/// The outcome of a hypothesis test. The meaning of `statistic`, `df`, `df2`,
/// and the p-value's sidedness depend on the producing function (current
/// p-values are two-sided except the upper-tail F and χ²). Tests with one
/// reference-distribution df (t, χ², F-var, correlation) leave `df2 = None`.
#[derive(Debug, Clone, PartialEq)]
pub struct TestResult {
    /// Test statistic (t, F, χ², …; see the producing function).
    pub statistic: f64,
    /// Primary degrees of freedom. For t-tests and χ²: the single df for the
    /// reference distribution. For one-way ANOVA F: d1 = k − 1 (between-groups).
    /// For F-var: d1 = nA − 1.
    pub df: f64,
    /// Secondary degrees of freedom. `Some(d2)` for two-df tests only:
    /// [`anova_one_way`](crate::htest::anova_one_way) sets `Some(N − k)` (within-groups df). All single-df
    /// tests (t, χ², F-var, correlation) set `None`.
    pub df2: Option<f64>,
    /// p-value; see the producing function for sidedness.
    pub p_value: f64,
    /// Effect size, when the test reports one.
    pub effect_size: Option<EffectSize>,
    /// Confidence interval, when the test reports one.
    pub ci: Option<Ci>,
}
