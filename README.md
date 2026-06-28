# faststats

A Rust statistical ecosystem built WASM-first, from one shared, oracle-validated
core. `&[f64]` in, structs out — no binding glue, minimal dependencies.

This repository is a Cargo workspace holding two co-evolving crates:

| Crate         | Path             | What                                                                              | Status   |
|---------------|------------------|-----------------------------------------------------------------------------------|----------|
| `commonstats` | [`CommonStats/`](CommonStats/) | Common statistics: descriptives, hypothesis tests, distributions, density estimation, transforms, resampling. | `0.1.0`  |
| RobustStats   | `RobustStats/`   | Understandable robust alternatives to popular tests (Brunner–Munzel, energy/kernel, robust regression). | planned  |

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

## WASM-first

`commonstats` builds clean for `wasm32-unknown-unknown` with no JS or platform
glue, so the same core can feed native, browser, and downstream tooling. CI
enforces the WASM build on every change.

## License

Dual-licensed under either of [Apache-2.0](LICENSE-APACHE) or
[MIT](LICENSE-MIT) at your option.
