# neurostats

`no_std`, WASM-ready neuroimaging statistics in Rust. Three layers, each usable
on its own: an adjacency graph over masked-volume voxels (`Domain`),
threshold-free cluster enhancement (`tfce` and friends) over a statistic map
on that graph, and one-sample sign-flip permutation inference
(`tfce_one_sample`). No I/O anywhere: every entry point takes and returns
slices. Part of the [faststats](https://github.com/pawlenartowicz/faststats)
repository. Full guide with worked examples and API detail:
[`../Documentation/NeuroStats/rust/README.md`](../Documentation/NeuroStats/rust/README.md).

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
  nodes (variance at the rounding floor) get `t = 0`.
- **Realizations** — all `2^n` sign patterns when `2^n ≤ B`, else `B` Monte
  Carlo draws (identity + `B − 1` Philox draws keyed by `(seed, draw_id)`).
  `B` counts the identity, so `p ≥ 1/B`.
- **p-values** — `p_fwe(v) = #{b : max_b ≥ tfce_obs(v)} / B` over the max-TFCE
  null, `p_unc(v) = #{b : tfce_b(v) ≥ tfce_obs(v)} / B` over the node's own
  null (no multiplicity correction). Both one-sided, positive tail; a
  negative effect is tested by negating `x`. `p_fwe` matches MNE 1.12.1
  `permutation_cluster_1samp_test(threshold=dict(...), tail=1)`
  `cluster_pv` by exact enumeration on the frozen fixtures; `p_unc` has no
  external oracle (MNE emits no such map).
- **Splitting** — `OneSampleProblem` + `run_range` over any partition of
  `0..B`, merged through `NullPartial`, then `finalize`: bit-identical to the
  serial call regardless of the split.
- **Threads** — `tfce_one_sample_threads(&dom, &x, n, &p, seed, b, threads)`
  (feature `parallel`) does that split over a rayon pool. `threads = 0` means
  every core the machine reports; `threads = 1` runs serially. The result is
  bit-identical to `tfce_one_sample` for every thread count, so a run is
  reproducible from `(seed, b)` alone.

## `Domain` kinds

Two representations back a `Domain`, and `==` compares them, so a lattice
never equals the CSR domain built from its own `to_csr`.

- **Lattice** — `Domain::from_volume` (dense grid) and `Domain::from_mask`
  (boolean mask) keep `dims` and the in-mask voxel indices, and derive each
  neighbour row from voxel coordinates on demand; no per-node neighbour list
  is stored.
- **CSR** — `Domain::from_csr` keeps the caller's `(offsets, neighbours)`
  arrays as given, the route for surface meshes and scipy sparse matrices.
  Symmetry is a documented precondition, not checked.

Both answer `to_csr`, `neighbours_into`, `n_nodes` and `n_edges` with the same
graph. `to_csr() -> Result<(Vec<u32>, Vec<u32>), NeuroError>` materialises the
CSR pair in the crate's neighbour order, returning `TooManyNodes` if the
directed edge count does not fit `u32`. `neighbours_into(i, &mut Vec<u32>)`
fills a caller-provided buffer with the node indices adjacent to node `i`.
`n_edges()` is O(1) (free) for a CSR domain and O(voxels) for a lattice domain,
since a lattice does not store its edge count.

**Node order** — node `i` is the `i`-th in-mask voxel in row-major (C-order)
scan of `dims` (`numpy.ndarray.ravel()` order). The caller aligns mask and
map. **Connectivity** — `Conn::Face` (6 offsets with one non-zero
coordinate), `Conn::Edge` (18: one or two), `Conn::Vertex` (26: any).

## TFCE weightings

Band weighting is required, never defaulted, because the reference tools
disagree:

- `Weighting::SmithNichols`: `w_i = h_i^H · step` (FSL `fslmaths -tfce`, PALM,
  SPM TFCE toolbox).
- `Weighting::MneStep`: `w_0 = |h_0|^H`, `w_i = |h_i − h_{i−1}|^H` for `i ≥ 1`
  (MNE-Python ≤ 1.12.1).
- `Weighting::Exact`: closed-form `∫_{start}^{stat[v]} e_v(h)^E · h^H dh` — no
  threshold grid, ignores `step` entirely; recommended for new analyses
  (`start = 0`).

Under the stepped rules, thresholds are NumPy `arange(start, max(stat),
step)`, start inclusive, stop exclusive. Membership at a threshold is strict
(`stat > h`).

`tfce` is the incremental union-find product path; `tfce_naive` re-clusters
at every threshold and is the readable oracle. They agree within `1e-12`
relative on every fixture in `tests/fixtures/`; `tfce_naive` under
`Weighting::MneStep` matches MNE 1.12.1 on those same fixtures.

`tfce_bands` / `tfce_bands_into` / `tfce_bands_naive` take an explicit
strictly increasing threshold list, explicit band weights, and a
strict/non-strict membership flag — the form the Python wheel and the
nilearn shim use.

## RNG contract

The sign pattern of draw `b` is a pure function of `(seed, b)` (Philox4x32-10
from `commonstats`, keyed to a sign-flip stream). `tfce_one_sample_threads` is
bit-identical to `tfce_one_sample` for every thread count and every partition
of `0..B` into `run_range` calls, because merging never reorders draws before
sorting. Draw 0 is always the identity realization. Exhaustive enumeration
(every `2^n` sign pattern, `n` = subject count) is used when `2^n ≤ B`,
otherwise `B` Monte Carlo draws.

## Features

| feature | default | adds |
|---|---|---|
| `std` | yes | the `permute` module — permutation inference; pulls in `commonstats` and the standard library |
| `parallel` | yes | `tfce_one_sample_threads`, the rayon-backed driver; implies `std` |

Opt out with `default-features = false`, which leaves the `no_std` TFCE core
(`Domain`, `tfce`, `tfce_bands`); add `features = ["std"]` back for serial
permutations without rayon. There is no negative feature, because cargo
features are additive and unify across the dependency graph — one consumer
asking for less would strip parallelism from every other consumer in the
build.

`RAYON_NUM_THREADS` and rayon's global pool have no effect here:
`tfce_one_sample_threads` builds its own pool per call, sized by its
`threads` argument, so concurrent callers cannot disturb each other's thread
counts.

## CI gates

- `cargo fmt --all -- --check`
- `cargo build --all-features`
- `cargo test --all-features`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --no-default-features --features std`
- `cargo build --no-default-features --target wasm32-unknown-unknown`

## Portability

The core and `permute` are `#![no_std]` + `alloc`. Only
`tfce_one_sample_threads` (feature `parallel`) links the standard library, for
threads. `#![forbid(unsafe_code)]`. The `wasm32-unknown-unknown` build is
`--no-default-features`: rayon needs threads that target does not have.

## License

LGPL-3.0-or-later. See [LICENSE](LICENSE) and [LICENSE-GPL](LICENSE-GPL).
