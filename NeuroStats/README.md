# neurostats

`no_std`, WASM-ready neuroimaging statistics in Rust. Current scope: a
masked-volume adjacency graph (`Domain`) and threshold-free cluster enhancement
(`tfce`) over a statistic map on that graph, validated against MNE-Python.
Part of the [faststats](https://github.com/pawlenartowicz/faststats) repository.

## Usage

```rust
use neurostats::{Conn, Domain, TfceParams, Weighting, tfce};

// mask: one bool per voxel in row-major (C) order of dims; stat: one f64 per
// in-mask voxel in the same scan order.
let dims = [4, 4, 4];
let mask = vec![true; 64];
let stat: Vec<f64> = (0..64).map(|i| (i % 5) as f64).collect();

let dom = Domain::from_mask(&mask, dims, Conn::Vertex).unwrap();
let p = TfceParams { e: 0.5, h: 2.0, start: 0.5, step: 0.5, weighting: Weighting::SmithNichols };
let enhanced = tfce(&dom, &stat, &p).unwrap();
assert_eq!(enhanced.len(), dom.n_nodes());
```

## Permutation inference (one-sample)

```rust
use neurostats::{Conn, Domain, TfceParams, Weighting, tfce_one_sample};

// x: subject-major, x[i * V + v] = subject i at node v; n subjects.
let dom = Domain::from_volume([3, 1, 1], Conn::Face);
let x = [2.0, 0.3, 0.0, 2.5, -0.4, 0.0, 1.8, 0.1, 0.0, 2.2, -0.2, 0.0];
let p = TfceParams { e: 0.5, h: 2.0, start: 0.0, step: 1.0, weighting: Weighting::Exact };
let r = tfce_one_sample(&dom, &x, 4, &p, 42, 5000).unwrap();
// r.t_obs, r.tfce_obs, r.p_fwe (max-TFCE FWE), r.p_unc, r.null_max (sorted, len B)
```

- **Statistic** — one-sample t, `ddof = 1`, no variance smoothing; degenerate
  nodes (all subjects equal) get `t = 0`.
- **Realizations** — all `2^n` sign patterns when `2^n ≤ B`, else `B` Monte
  Carlo draws (identity + `B − 1` Philox draws keyed by `(seed, draw_id)`).
  `B` counts the identity, so `p ≥ 1/B`.
- **p-values** — `p_fwe(v) = #{b : max_b ≥ tfce_obs(v)} / B`,
  `p_unc(v) = #{b : tfce_b(v) ≥ tfce_obs(v)} / B`. Matches MNE 1.12.1
  `permutation_cluster_1samp_test(threshold=dict(...), tail=1)` by exact
  enumeration on the frozen `tests/fixtures/perm_*.json`.
- **Splitting** — `OneSampleProblem` + `run_range` over any partition of
  `0..B`, merged through `NullPartial`, then `finalize`: bit-identical to the
  serial call regardless of the split.
- **Threads** — `tfce_one_sample_threads(&dom, &x, n, &p, seed, b, threads)`
  (feature `parallel`) does that split over a rayon pool. `threads = 0` means
  every core the machine reports; `threads = 1` runs serially. The result is
  bit-identical to `tfce_one_sample` for every thread count, so a run is
  reproducible from `(seed, b)` alone.

## Conventions

- **Node order** — the `i`-th in-mask voxel in row-major scan of `dims`
  (`numpy.ravel()` order). The caller aligns mask and map.
- **Connectivity** — `Conn::Face` (6), `Conn::Edge` (18), `Conn::Vertex` (26).
- **Thresholds** — under the stepped rules (`SmithNichols`, `MneStep`): NumPy
  `arange(start, max(stat), step)`, start inclusive, stop exclusive. Membership
  at a threshold is strict (`stat > h`).
- **Band weighting** — required, never defaulted, because the reference tools
  disagree:
  - `Weighting::SmithNichols`: `h_i^h · step` (FSL, PALM, SPM TFCE).
  - `Weighting::MneStep`: `|h_0|^h` then `|h_i − h_{i−1}|^h` (MNE-Python ≤ 1.12.1).
  - `Weighting::Exact`: closed-form `∫_{start}^{stat} e(h)^E h^H dh`, no grid;
    recommended for new analyses (`start = 0`).
- **Tail** — positive only.

`tfce` is the incremental union-find product path; `tfce_naive` re-clusters at
every threshold and exists as the readable oracle. They agree within `1e-12`
relative; `tfce_naive` under `MneStep` matches MNE 1.12.1 on the frozen fixtures
in `tests/fixtures/`.

`tfce_bands` / `tfce_bands_into` / `tfce_bands_naive` take an explicit strictly
increasing threshold list, explicit band weights, and a strict/non-strict
membership flag (the form the Python wheel and the nilearn shim use).

`Domain` has two representations. `from_volume` and `from_mask` build a
lattice: the grid is kept and neighbours are derived from voxel coordinates.
`from_csr` builds a domain from a caller-supplied CSR neighbour list
(symmetry is the caller's precondition) and keeps that adjacency as given.
`to_csr() -> Result<(Vec<u32>, Vec<u32>), NeuroError>` materialises the CSR
pair for either representation, in the crate's neighbour order; it returns
`TooManyNodes` if the directed edge count does not fit `u32`.
`neighbours_into(i, &mut Vec<u32>)` fills a caller-provided buffer with the
node indices adjacent to node `i`. `n_edges()` is O(edges) for a CSR domain
and O(voxels) for a lattice domain, since a lattice does not store its edge
count.

## Features

| feature | default | adds |
|---|---|---|
| `std` | yes | the `permute` module — permutation inference; pulls in `commonstats` and the standard library |
| `parallel` | yes | `tfce_one_sample_threads`, the rayon-backed driver; implies `std` |

Opt out with `default-features = false`, which leaves the `no_std` TFCE core
(`Domain`, `tfce`, `tfce_bands`); add `features = ["std"]` back for serial
permutations without rayon. There is no negative feature, because cargo
features are additive and unify across the dependency graph — one consumer
asking for less would strip parallelism from every other consumer in the build.

`RAYON_NUM_THREADS` and rayon's global pool have no effect here:
`tfce_one_sample_threads` builds its own pool per call, sized by its `threads`
argument, so concurrent callers cannot disturb each other's thread counts.

## Portability

The core and `permute` are `#![no_std]` + `alloc`. Only
`tfce_one_sample_threads` (feature `parallel`) links the standard library, for
threads. `#![forbid(unsafe_code)]`. The `wasm32-unknown-unknown` build is
`--no-default-features`: rayon needs threads that target does not have.

## License

LGPL-3.0-or-later. See [LICENSE](LICENSE) and [LICENSE-GPL](LICENSE-GPL).
