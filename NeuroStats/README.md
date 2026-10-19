# neurostats

`no_std`, WASM-ready neuroimaging statistics in Rust. Four layers, each usable
on its own: an adjacency graph over masked-volume voxels (`Domain`),
threshold-free cluster enhancement (`tfce` and friends) over a statistic map
on that graph, a residualised copy of the brain data (`Store`), and TFCE
permutation inference for a linear model fitted at every node (`tfce_ols`,
which includes the one-sample test). No I/O anywhere: every entry point takes
and returns slices. Part of the [faststats](https://github.com/pawlenartowicz/faststats)
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

## Permutation inference (`tfce_ols`)

```rust
use neurostats::{Conn, Domain, PermOpts, Tail, TfceParams, Weighting, tfce_ols};

// y: subject-major, y[i * V + v] = subject i at node v; n subjects.
let dom = Domain::from_volume([3, 1, 1], Conn::Face);
let y = [2.0, 0.3, 0.0, 2.5, -0.4, 0.0, 1.8, 0.1, 0.0, 2.2, -0.2, 0.0];
let age = [31.0, 45.0, 28.0, 52.0]; // tested regressor, one value per subject
let sex = [0.0, 1.0, 1.0, 0.0]; // confounds, row-major n × c
let p = TfceParams { e: 0.5, h: 2.0, start: 0.0, step: 1.0, weighting: Weighting::Exact };
let opts = PermOpts::new(p, Tail::TwoSided, 42, 5000); // seed 42, 5000 draws

// brain ~ age + sex + intercept, t of age:
let r = tfce_ols(&dom, &y, 4, Some(&age), &sex, &opts).unwrap();
// brain ~ intercept, t of the intercept: the one-sample test.
let one = tfce_ols(&dom, &y, 4, None, &[], &opts).unwrap();
// r.stat_obs, r.p_fwe (max-TFCE FWE), r.p_unc, r.null.tfce_obs,
// r.null.maxima (the null, sorted, len B)
```

- **Model** — `brain ~ tested + confounds + intercept`, one fit per node with
  the same design; the statistic is the t of `tested`. With `tested = None`
  the model is `brain ~ intercept + confounds` and the statistic is the t of
  the intercept, which without confounds is the one-sample t (`ddof = 1`).
  A node with no residual variance gets `t = 0`. The formula is the one of
  nilearn `permuted_ols`.
- **Draws** — with a tested regressor the rows of the design are shuffled
  together and the brain data stay fixed, which gives the Freedman–Lane
  statistic. With `tested = None` the rows are sign-flipped: all `2^n`
  patterns when `2^n ≤ B`, else `B` Monte Carlo draws. Draw 0 is the identity
  and `B` counts it, so `p ≥ 1/B`.
- **Tails** — `Tail::Positive`, `Tail::Negative` or `Tail::TwoSided`,
  required. The tested quantity `T` is the TFCE of the map, the TFCE of the
  negated map, or `|TFCE(map) − TFCE(−map)|`.
- **p-values** — `p_fwe(v) = #{b : max_b ≥ T(v)} / B` over the max-TFCE null,
  `p_unc(v) = #{b : T_b(v) ≥ T(v)} / B` over the node's own null (no
  multiplicity correction). For the one-sample test under `Tail::Positive`,
  `p_fwe` matches MNE 1.12.1
  `permutation_cluster_1samp_test(threshold=dict(...), tail=1)` `cluster_pv`
  by exact enumeration on the frozen fixtures. The t of every draw and both
  p-maps are checked against a loop that refits every node by the normal
  equations.
- **No refits** — the data are residualised against the confounds and scaled
  once, into a `Store`. A draw then costs a few dot products per node and one
  TFCE. Draws run in batches of `PermOpts::batch`, so the store is read once
  per batch; the batch size never changes a result.
- **Pieces** — `nuisance_basis` + `Store::from_f64` or `Store::from_f32` (an
  `f32` or `f64` source and a row index) + `PermProblem::ols` + `run_range`
  over any partition of `problem.draws()`, merged through `NullPartial`, then
  `finalize`: bit-identical to `tfce_ols` regardless of the split.
- **Threads** — `tfce_ols_threads(.., threads)` and `run_threads` (feature
  `parallel`) do that split over a rayon pool. `threads = 0` means every core
  the machine reports; `threads = 1` runs serially. The result is
  bit-identical to `tfce_ols` for every thread count, so a run is reproducible
  from its seed and budget alone.
- **Extending a null** — a result carries its `NullState` (`r.null`). Set
  `PermOpts::extend = Some(&r.null)` with a larger `n_perm` and the same
  data, design, seed and settings: only the missing draws are run, and the
  result equals one run with the larger budget, bit for bit.

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
- `Weighting::Nilearn`: 100 thresholds in equal steps from 0 up to and
  including `max(stat)`, rebuilt for every map; `w_i = h_i^H` with no step
  factor; membership is `stat >= h`; `start` and `step` are not used (nilearn
  `calculate_tfce` under `dh="auto"`).

Under `SmithNichols` and `MneStep`, thresholds are NumPy `arange(start,
max(stat), step)`, start inclusive, stop exclusive, and membership at a
threshold is strict (`stat > h`).

`tfce` is the incremental union-find product path; `tfce_naive` re-clusters
at every threshold and is the readable oracle. They agree within `1e-12`
relative on every fixture in `tests/fixtures/`; `tfce_naive` under
`Weighting::MneStep` matches MNE 1.12.1 on those same fixtures.

`tfce_tail_into(&dom, &stat, &p, tail, &mut ws, &mut out)` enhances for a
chosen `Tail`: `Positive` is the map, `Negative` the negated map, and
`TwoSided` the first minus the second, which is signed. Under the grid rules
both passes of a two-sided call share one grid, built from `max |stat|`. It
reuses a `TfceWorkspace` and allocates nothing per call.

`tfce_bands` / `tfce_bands_into` / `tfce_bands_naive` take an explicit
strictly increasing threshold list, explicit band weights, and a
strict/non-strict membership flag, for callers that build their own grid.

## RNG contract

Draw `b` is a pure function of `(seed, b)` (Philox4x32-10 from `commonstats`):
a row shuffle from `gen_resample_indices(Scheme::Permutation, ..)` when a
regressor is tested (`draw_permutation` returns it), a sign pattern from
`gen_sign_flips` when the intercept is tested. `tfce_ols_threads` is
bit-identical to `tfce_ols` for every thread count, every batch size and every
partition of the draws into `run_range` calls, because each dot product is
summed in a fixed order and merging never reorders draws before sorting. Draw
0 is always the identity. Every `2^n` sign pattern is enumerated when the
intercept is tested and `2^n ≤ B`; row shuffles are never enumerated.

## Features

| feature | default | adds |
|---|---|---|
| `std` | yes | the `permute` module — permutation inference; pulls in `commonstats` and the standard library |
| `parallel` | yes | `run_threads` and `tfce_ols_threads`, the rayon-backed drivers; implies `std` |

Opt out with `default-features = false`, which leaves the `no_std` core
(`Domain`, `tfce`, `tfce_bands`, `Store`, `nuisance_basis`); add
`features = ["std"]` back for serial permutations without rayon. There is no
negative feature, because cargo features are additive and unify across the
dependency graph — one consumer asking for less would strip parallelism from
every other consumer in the build.

`RAYON_NUM_THREADS` and rayon's global pool have no effect here:
`run_threads` builds its own pool per call, sized by its `threads` argument,
so concurrent callers cannot disturb each other's thread counts.

## CI gates

- `cargo fmt --all -- --check`
- `cargo build --all-features`
- `cargo test --all-features`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --no-default-features --features std`
- `cargo build --no-default-features --target wasm32-unknown-unknown`

## Portability

The core and `permute` are `#![no_std]` + `alloc`. Only `run_threads` and
`tfce_ols_threads` (feature `parallel`) link the standard library, for
threads. `#![forbid(unsafe_code)]`. The `wasm32-unknown-unknown` build is
`--no-default-features`: rayon needs threads that target does not have.

## License

LGPL-3.0-or-later. See [LICENSE](LICENSE) and [LICENSE-GPL](LICENSE-GPL).
