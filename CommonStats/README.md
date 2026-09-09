# commonstats

[![CI](https://github.com/pawlenartowicz/faststats/actions/workflows/ci.yml/badge.svg)](https://github.com/pawlenartowicz/faststats/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/commonstats.svg)](https://crates.io/crates/commonstats)
[![docs.rs](https://img.shields.io/docsrs/commonstats)](https://docs.rs/commonstats)
[![license](https://img.shields.io/crates/l/commonstats.svg)](https://github.com/pawlenartowicz/faststats#license)
![MSRV](https://img.shields.io/badge/MSRV-1.85-blue.svg)

WASM-first Rust statistical core: special functions, mergeable accumulators,
descriptives, and hypothesis tests. `&[f64]` in, structs out — no binding glue.

## Status

Base: special functions, the `Mergeable`/`Accumulator` core, descriptives,
moment/count-based tests (t / Welch / paired / ANOVA / F / χ² / Pearson), kernel
density + auto-histograms, and distribution-free transforms (rank, normal-scores,
PIT, power, quantile-normalize). Feature-gated: `rng` (Philox), `resample`
(null/bootstrap), and `dist` (continuous + discrete distribution objects).

## Example

```rust
use commonstats::{mean, sd, t_test_two, VarAssumption};

let a = [5.1, 4.9, 6.2, 5.5, 5.8];
let b = [6.1, 5.9, 7.0, 6.5, 6.8];
let m = mean(&a).unwrap();
let result = t_test_two(&a, &b, VarAssumption::Welch).unwrap();
println!("mean {m}, t = {}, p = {}", result.statistic, result.p_value);
```

## API reference

`&[f64]` in, value or struct out. NaNs are dropped by default (the `Omit`
policy). Fallible items return `Result<T, StatError>`, abbreviated `Result<T>`
below; everything is two-sided unless noted.

### A. Descriptives & moments

```rust
count(xs: &[f64])             -> usize          // count of finite values
sum(xs: &[f64])               -> f64            // Neumaier-compensated sum
min(xs: &[f64])               -> Result<f64>    // minimum finite value
max(xs: &[f64])               -> Result<f64>    // maximum finite value
range(xs: &[f64])             -> Result<f64>    // max − min
median(xs: &[f64])            -> Result<f64>    // type-7 sample median
mean(xs: &[f64])              -> Result<f64>    // arithmetic mean
var(xs: &[f64], ddof: Ddof)   -> Result<f64>    // variance; ddof = Sample (n−1) | Population (n)
sd(xs: &[f64], ddof: Ddof)    -> Result<f64>    // standard deviation; ddof = Sample | Population
skewness(xs: &[f64])          -> Result<f64>    // Fisher–Pearson g1 (no bias corr.)
kurtosis(xs: &[f64])          -> Result<f64>    // excess kurtosis (Fisher)
cov(a: &[f64], b: &[f64])        -> Result<f64> // sample covariance (ddof = 1)
pearson(a: &[f64], b: &[f64])    -> Result<f64> // Pearson correlation r
describe(xs: &[f64])          -> Result<Describe> // 5-number summary + moments
```

### B. Hypothesis tests

All return `TestResult { statistic, df, df2, p_value, effect_size, ci }`.

```rust
t_test_one(v: &[f64], mu0: f64)                  -> Result<TestResult> // one-sample t
t_test_two(a: &[f64], b: &[f64], va: VarAssumption) -> Result<TestResult>
                                          // two-sample t; va = Equal (pooled Student) | Welch
t_test_paired(a: &[f64], b: &[f64])              -> Result<TestResult> // paired t on a−b
anova_one_way(groups: &[&[f64]])                 -> Result<TestResult> // one-way ANOVA F
f_test_var(a: &[f64], b: &[f64])                 -> Result<TestResult> // F-test, equal variances
chi2_gof(observed: &[f64], expected: &[f64])     -> Result<TestResult> // χ² goodness-of-fit
chi2_independence(table: &[&[f64]])              -> Result<TestResult> // χ² independence (r×c)
cor_test(a: &[f64], b: &[f64], method: CorMethod) -> Result<TestResult> // correlation test (Pearson)
```

### C. Effect sizes & confidence intervals

All items in this section live in the `commonstats::htest` module (not the crate root).

```rust
htest::cohen_d(a: &[f64], b: &[f64])         -> Result<f64> // pooled-SD standardized mean diff
htest::eta_squared(groups: &[&[f64]])        -> Result<f64> // η² for one-way ANOVA
htest::cramers_v(table: &[&[f64]])           -> Result<f64> // Cramér's V for an r×c table

htest::ci_mean(v: &[f64], level: f64)                -> Result<Ci> // mean CI (Student t)
htest::ci_mean_diff(a: &[f64], b: &[f64], level: f64) -> Result<Ci> // mean-diff CI (pooled)
htest::ci_mean_diff_welch(a: &[f64], b: &[f64], level: f64) -> Result<Ci> // mean-diff CI (Welch)
htest::ci_proportion(successes: usize, n: usize, level: f64) -> Result<Ci> // Wald proportion CI
htest::ci_correlation(a: &[f64], b: &[f64], level: f64) -> Result<Ci> // Pearson r CI (Fisher z)
```

### D. Distributions *(requires feature `dist`)*

All items in this section live in the `commonstats::dist` module (not the crate
root). Each constructor returns `Result<Self>`. All implement `.cdf(x)`, `.sf(x)`,
`.quantile(p) -> Result<_>` (continuous also `.isf(q) -> Result<f64>`, the
upper-tail critical value solved on `q` directly), the moment accessors (`.mean()`, `.variance()`,
`.std_dev()`, `.skewness()`, `.kurtosis()`, `.entropy()`), plus `.density(x)` /
`.log_density(x)` (continuous) or `.mass(k)` / `.log_mass(k)` (discrete).

```rust
// continuous
dist::Normal::new(mean: f64, sd: f64)          // N(μ, σ)
dist::StudentT::new(df: f64)                   // Student's t
dist::ChiSquared::new(k: f64)                  // χ²(k)
dist::FisherF::new(dfn: f64, dfd: f64)         // F(d1, d2)
dist::Uniform::new(a: f64, b: f64)             // Uniform[a, b]
dist::Exponential::new(rate: f64)              // Exp(λ)
dist::Cauchy::new(loc: f64, scale: f64)        // Cauchy (no moments)
dist::Weibull::new(shape: f64, scale: f64)     // Weibull
dist::LogNormal::new(mu: f64, sigma: f64)      // Log-normal
dist::Gamma::new(shape: f64, rate: f64)        // Γ(α, β)
dist::Beta::new(alpha: f64, beta: f64)         // Beta on [0, 1]

// discrete
dist::Bernoulli::new(p: f64)                   // Bernoulli(p)
dist::Binomial::new(n: i64, p: f64)            // Binom(n, p)
dist::Poisson::new(lambda: f64)                // Poisson(λ)
dist::Geometric::new(p: f64)                   // trials until first success
dist::NegBinomial::new(r: f64, p: f64)         // negative binomial
dist::Hypergeometric::new(big_n: i64, k: i64, n: i64) // hypergeometric
```

### E. Transforms

All items in this section live in the `commonstats::transform` module (not the
crate root).

```rust
transform::rank(v: &[f64], ties: Ties)   -> Result<Vec<f64>> // 1-based ranks; ties = Average|Min|Max|Dense|Ordinal
transform::normal_scores(v: &[f64])      -> Result<Vec<f64>> // Blom (1958) rankits
transform::box_cox(v: &[f64], lambda: f64)     -> Result<Vec<f64>> // Box–Cox power (positive data)
transform::yeo_johnson(v: &[f64], lambda: f64) -> Result<Vec<f64>> // Yeo–Johnson power (all reals)
transform::quantile_normalize(matrix: &[&[f64]]) -> Result<Vec<Vec<f64>>> // column rank-averaging

// PIT (requires feature `dist`)
transform::pit(x: f64, dist: &D)                 -> f64         // probability-integral transform
transform::inv_pit(u: f64, dist: &D)             -> Result<f64> // inverse PIT
transform::quantile_map(x: f64, from: &D1, to: &D2) -> Result<f64> // transport between scales
```

### F. Density & histograms

All items in this section live in the `commonstats::density` module (not the
crate root); `Histogram` is the exception — it is a crate-root re-export.

```rust
density::kde(xs: &[f64], kernel: density::Kernel, bandwidth: density::Bandwidth) -> Result<density::Kde>
                          // kernel = Gaussian|Epanechnikov; bandwidth = Silverman|Scott|Fixed(h)
                          // -> Kde { .density(x), .evaluate(xs), .bandwidth(), .n() }
density::histogram_auto(xs: &[f64], bins: density::Bins, norm: density::Norm) -> Result<(Vec<f64>, Vec<f64>)>
                          // (edges, values); bins = Rule(Sturges|FreedmanDiaconis|…)|Fixed(k)|Width(w)|Edges
                          // norm = Count|Density|Probability

Histogram::new(lo: f64, hi: f64, n_bins: usize) -> Result<Histogram> // fixed-bin accumulator
                          // .counts(), .edges(), .underflow(), .overflow(), .ecdf()
```

## Feature flags

| Flag       | Adds                                                                   |
|------------|------------------------------------------------------------------------|
| `dist`     | continuous + discrete distribution objects (CDF/SF/PDF/quantile) and PIT |
| `rng`      | counter-based Philox RNG (`CommonStatsRng`)                            |
| `resample` | permutation/bootstrap index generation + `NullDist` / `BootDist` (implies `rng`) |
