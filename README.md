# faststats

[![CI](https://github.com/pawlenartowicz/faststats/actions/workflows/ci.yml/badge.svg)](https://github.com/pawlenartowicz/faststats/actions/workflows/ci.yml)
[![commonstats](https://img.shields.io/crates/v/commonstats.svg?label=commonstats)](https://crates.io/crates/commonstats)
[![License](https://img.shields.io/badge/license-LGPL--3.0--or--later-blue.svg)](#license)

A Rust statistical ecosystem built WASM-first, from one shared, oracle-validated
core. `&[f64]` in, structs out — no binding glue, minimal dependencies.

This repository is a single git repo (no workspace manifest) holding co-evolving
crates:

| Crate         | Path             | What                                                                              | Status   |
|---------------|------------------|-----------------------------------------------------------------------------------|----------|
| `commonstats` | [`CommonStats/`](CommonStats/) | Common statistics: descriptives, hypothesis tests, distributions, density estimation, transforms, resampling. | `0.1.0`  |
| RobustStats   | `RobustStats/`   | Understandable robust alternatives to popular tests (Brunner–Munzel, energy/kernel, robust regression). | planned  |
| `neurostats`  | [`NeuroStats/`](NeuroStats/) | Neuroimaging statistics, `no_std`: masked-volume adjacency graph, TFCE enhancement, and one-sample sign-flip permutation inference. LGPL-3.0-or-later. | 0.1.0 (unreleased) |

## commonstats

The shipping core. Special functions, mergeable accumulators, descriptives,
moment- and count-based tests (t / Welch / paired / ANOVA / F / χ² / Pearson),
kernel density and auto-histograms, distribution-free transforms, and
feature-gated RNG (`rng`), resampling (`resample`), and distribution objects
(`dist`). Every public statistic states its convention (ddof, definition, df,
sidedness, correction) and is validated against a SciPy/R oracle grid.

```rust
use commonstats::{mean, t_test_two, VarAssumption};

let a = [5.1, 4.9, 6.2, 5.5, 5.8];
let b = [6.1, 5.9, 7.0, 6.5, 6.8];
let m = mean(&a).unwrap();
let r = t_test_two(&a, &b, VarAssumption::Welch).unwrap();
println!("mean {m}, t = {}, p = {}", r.statistic, r.p_value);
```

See [`CommonStats/README.md`](CommonStats/README.md) for the full API reference
and feature flags.

## RobustStats

Planned. Robust, interpretable alternatives to the classical tests, sharing
`commonstats`' lints, documentation gate, and oracle harness. Not yet started.

## neurostats

`#![no_std]` + `alloc`, one dependency (`libm`). Three layers, each usable on
its own: `Domain` builds an adjacency graph over the in-mask voxels of a 3-D
volume or a caller-built neighbour list (L0); `tfce` computes threshold-free
cluster enhancement over a statistic map on that graph with one incremental
union-find sweep, with `tfce_naive` as the readable re-clustering oracle (L1);
and `tfce_one_sample` runs one-sample sign-flip permutation inference end to
end, with `tfce_one_sample_threads` spreading the same draws over a rayon
pool (L2). Band weighting is an explicit parameter (`Weighting::SmithNichols`
for FSL/PALM/SPM, `Weighting::MneStep` for MNE-Python) because the tools
disagree. Frozen MNE 1.12.1 goldens in `NeuroStats/tests/fixtures/`. See
[`NeuroStats/README.md`](NeuroStats/README.md) and the full crate guide,
[`Documentation/NeuroStats/rust/README.md`](Documentation/NeuroStats/rust/README.md).

## Documentation

- [`Documentation/CommonStats/rust/README.md`](Documentation/CommonStats/rust/README.md) — commonstats crate guide (modules, features, conventions policy, oracle harness, CI gates)
- [`Documentation/CommonStats/python/README.md`](Documentation/CommonStats/python/README.md) — faststats.common Python binding status
- [`Documentation/NeuroStats/rust/README.md`](Documentation/NeuroStats/rust/README.md) — neurostats crate guide (L0–L2 API detail)
- [`Documentation/NeuroStats/python/api.md`](Documentation/NeuroStats/python/api.md) — faststats.neuro Python API
- [`Documentation/NeuroStats/python/nilearn_compatibility.md`](Documentation/NeuroStats/python/nilearn_compatibility.md) — faststats.neuro vs nilearn compatibility notes
- [`Documentation/RobustStats/README.md`](Documentation/RobustStats/README.md) — RobustStats status
- [`Documentation/GLMM/README.md`](Documentation/GLMM/README.md) — GLMM pointer to its own repository

## WASM-first

Every crate builds clean for `wasm32-unknown-unknown` with no JS or platform
glue, so the same core can feed native, browser, and downstream tooling. CI
enforces the WASM build on every change.

## License

[LGPL-3.0-or-later](LICENSE) for every crate in this repository (the LGPL is
a set of extra permissions on top of the [GPL-3.0](LICENSE-GPL), so both texts
ship). `commonstats` 0.1.0 as published on crates.io remains under its
original MIT OR Apache-2.0 terms; later releases are LGPL-3.0-or-later.
