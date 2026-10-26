# Changelog

All notable changes to this crate are documented in this file. Earlier
releases (0.1.0–0.2.2) predate it.

## [0.3.0] — 2026-10-26

### Breaking

- Removed `NanPolicy` (and its `NanPolicy::Propagate` mode). Every slice entry
  point omits NaN.
- `nan::clean(xs, policy)` is now `nan::clean(xs)`. It drops NaN, keeps ±∞,
  and returns `EmptyInput` for an empty slice and `AllNaN` when nothing is
  left.
- Removed `rng::philox`. Use `rand_philox::philox4x32_10` directly.
- New errors where 0.2.2 returned a value or panicked:
  - `htest::ci_proportion` returns `DomainError` when `successes > n`.
  - `htest::eta_squared` returns `TooFewObservations` for fewer than 2 groups.
  - `htest::cramers_v` returns `TooFewObservations` for fewer than 2 rows and
    `DomainError` for a ragged table or fewer than 2 columns.
  - `density::kde` with `Bandwidth::Silverman` or `Bandwidth::Scott` returns
    `DomainError` when the sd is not finite (±∞ in the data, or overflow).
  - `density::histogram_auto` returns `DomainError` when `max − min` is not
    finite (±∞ in the data, or a span past `f64::MAX`), except on the
    `Bins::Edges` path, and when a `Bins::Width` last edge overflows.
  - `density::histogram_auto` and `accum::Histogram::new` return `DomainError`
    when the bin count reaches `isize::MAX / 8`, instead of panicking with
    "capacity overflow".
- `density::histogram_auto` bin rules: `Rule::FreedmanDiaconis` with IQR = 0
  and `Rule::Scott` with sd = 0 now give one bin (FD used to fall back to
  Scott, or error).

### Added

- `dist::StudentizedRange`: Tukey's studentized range `q(k, ν)`, with `cdf`,
  `sf`, `density`, `quantile` and `isf`. `new(k, df)` takes `1 < k ≤ 1e6`
  and `df > 0`, `df = ∞` allowed. Validated against mpmath; no moments and no
  `Sampler`. One `cdf`/`sf`/`density` costs 2–30 ms, one quantile 20–80 ms
  (release build).

### Changed

- `dist::FisherF::density(0)` and `log_density(0)` return the limit: ∞ for
  dfn < 2, 1 for dfn = 2, 0 for dfn > 2 (as R `df(0, ...)`). They used to
  return 0.
- `dist::Uniform` stays finite when `b − a` overflows (e.g.
  `Uniform(-1e308, 1e308)`) in `density`, `cdf`, `sf`, `entropy`, `mean` and
  `variance`. `quantile` and `isf` are clamped to `[a, b]`, with
  `quantile(0) = a` and `quantile(1) = b` exactly.
- `Uniform` sampling now goes through `quantile`, so draws for a given seed
  can differ from 0.2.2 in the last bits.
- `Binomial` sampling (inversion branch, `n·min(p, 1−p) < 10`) starts from
  `exp(n·log1p(−p))`. Draws are now correct for `n ≳ 1e15`; for a given seed
  they can differ from 0.2.2 in rare cases.
- `dist::Beta::log_density` is more accurate for `α < 0.5`, and
  `dist::FisherF::log_density` for tiny `dfn`.
- `dist::Hypergeometric::mean` returns `Some(0.0)` for `N = 0` (was NaN).
- `density::histogram_auto` with `Bins::Width(w)`: the last edge is
  `min + n_bins·w`, so the last bin can extend past the data maximum. Edges
  and densities differ from 0.2.2.
- `density::histogram_auto` with `Bins::Edges` and `Norm::Density` or
  `Norm::Probability` normalises by the count inside the edges, as numpy does
  (was the total count). ±∞ values are dropped on this path.
- `accum::Histogram::merge` with an empty right-hand side is a no-op (it used
  to panic), and `accum::TDigest::merge` with an empty digest of a different
  `delta` no longer panics in debug builds.
