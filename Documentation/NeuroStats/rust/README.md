# neurostats — crate guide

Front page: [`NeuroStats/README.md`](../../../NeuroStats/README.md). API
reference: `cargo doc --open` or docs.rs once published.

## Layers as built

**L0 — adjacency (`no_std` core, no feature needed).** `Domain` is an
undirected graph over the in-mask voxels of a 3-D volume or over a
caller-built neighbour list. Built with `Domain::from_volume`,
`Domain::from_mask`, or `Domain::from_csr`; read back with `Domain::to_csr`,
`Domain::neighbours_into`, `Domain::n_nodes`, `Domain::n_edges`. `Conn`
carries the connectivity choice.

**L1 — TFCE (`no_std` core, no feature needed).** `tfce` / `tfce_into` /
`tfce_naive` enhance a statistic map over a `Domain`; `TfceParams` and
`Weighting` set the rule; `TfceWorkspace` lets a caller reuse allocations
across calls. `tfce_bands` / `tfce_bands_into` / `tfce_bands_naive` take an
explicit band list instead of a threshold grid.

**L2 — one-sample sign-flip permutation (feature `std`; threading needs
`parallel`).** `tfce_one_sample` runs the one-sample test end to end.
`OneSampleT` computes the t-map from sufficient statistics; `sign_flip_plan`
picks exhaustive enumeration or Monte Carlo; `OneSampleProblem` holds
everything fixed across draws; `run_range` runs a range of draws into a
`NullPartial` using a `PermWorkspace`; `finalize` turns a merged null into
`TfceInference`. `tfce_one_sample_threads` (feature `parallel`) spreads the
same draws over a rayon pool.

## `Domain`

```rust
pub fn from_volume(dims: [usize; 3], conn: Conn) -> Self
pub fn from_mask(mask: &[bool], dims: [usize; 3], conn: Conn) -> Result<Self, NeuroError>
pub fn from_csr(offsets: Vec<u32>, neighbours: Vec<u32>) -> Result<Self, NeuroError>
```

`from_volume` is every voxel of `dims` in-mask (dense grid); it panics only
if `dims[0]·dims[1]·dims[2]` exceeds `u32::MAX` voxels. `from_mask` takes one
`bool` per voxel in row-major order of `dims`; `mask.len()` must equal the
`dims` product (`NeuroError::MismatchedLengths` otherwise), and a grid with
more than `u32::MAX` voxels returns `NeuroError::TooManyNodes` before the
length check runs. Both build a *lattice*: `dims`, `conn`, and the in-mask
voxel indices are kept, and each neighbour row is derived from voxel
coordinates on demand — no per-node neighbour list is stored, and the TFCE
sweep runs in voxel-index space for a lattice domain.

`from_csr` builds a *CSR* domain from the caller's own `(offsets,
neighbours)` arrays — the route for surface meshes and scipy sparse matrices.
`n = offsets.len() - 1`; `offsets` must be non-decreasing with `offsets[n] ==
neighbours.len()`; every neighbour index `< n`; no self-loops. Symmetry is a
documented precondition and is **not** checked — the TFCE sweep assumes it.
Violations return `NeuroError::InvalidAdjacency`.

Both representations answer `to_csr`, `neighbours_into`, `n_nodes`, and
`n_edges` with the same graph, but `==` compares representations, so a
lattice domain never equals the CSR domain built from its own `to_csr`.
`to_csr() -> Result<(Vec<u32>, Vec<u32>), NeuroError>` materialises the
adjacency in the crate's neighbour order; a lattice does this in `O(edges ·
log V)` and returns `NeuroError::TooManyNodes` if the directed edge count
does not fit `u32`; a CSR domain clones its arrays. `n_edges()` is O(1) (free)
for a CSR domain (the array length) and O(voxels) for a lattice domain
(recomputed on every call, since a lattice does not store its edge count).

**Node order.** Node `i` is the `i`-th in-mask voxel in row-major (C-order)
scan of `dims` — `linear = (x·dims[1] + y)·dims[2] + z`, the same order as
`numpy.ndarray.ravel()`. Worked example: `Domain::from_volume([2, 2, 1],
Conn::Face)` has all four voxels in-mask (`nx=2, ny=2, nz=1`), so node
indices and their voxel coordinates are:

| node | (x, y, z) |
|---|---|
| 0 | (0, 0, 0) |
| 1 | (0, 1, 0) |
| 2 | (1, 0, 0) |
| 3 | (1, 1, 0) |

With `Conn::Face`, node 0's neighbours are node 1 (`y` differs by 1) and node
2 (`x` differs by 1); node 3's neighbours are node 1 and node 2 by symmetry.

**`Conn` offsets.** `Conn::Face` enumerates 6 `(dx, dy, dz)` offsets (exactly
one non-zero coordinate), `Conn::Edge` 18 (one or two non-zero), `Conn::Vertex`
26 (any non-zero combination in `{-1,0,1}³ \ {0}`) — the full 3-D neighbour
template before boundary clipping; a boundary or in-mask voxel may realize
fewer than that count.

## TFCE

```rust
pub struct TfceParams {
    pub e: f64,       // extent exponent E; Smith–Nichols/MNE default 0.5
    pub h: f64,       // height exponent H; default 2.0; > -1 under Weighting::Exact
    pub start: f64,   // first threshold h_0; nodes with stat <= start get 0; >= 0 under Exact
    pub step: f64,    // threshold spacing; > 0, finite; unused by Weighting::Exact
    pub weighting: Weighting,
}
```

Under the stepped weightings, thresholds are NumPy `arange(start, max(stat),
step)` (start inclusive, stop exclusive), so the largest node clears every
band. Membership at a threshold is strict: `stat > h`.

The three `Weighting` variants:

- `SmithNichols`: `w_i = h_i^H · step` — the rule FSL `fslmaths -tfce`, PALM,
  and the SPM TFCE toolbox implement.
- `MneStep`: `w_0 = |h_0|^H`, `w_i = |h_i − h_{i−1}|^H` for `i ≥ 1` — no
  `· step` factor and no threshold value — matching MNE-Python (`_find_clusters`,
  ≤ 1.12.1).
- `Exact`: closed-form `∫_{h0}^{stat[v]} e_v(h)^E · h^H dh` with `h0 = start`,
  no threshold grid and no `step`; `e_v(h)` is piecewise constant between
  distinct `stat` values, so a run of constant component size `s` over `[a,
  b)` contributes `s^E · (b^{H+1} − a^{H+1}) / (H+1)`. Requires `h > −1` and
  `start ≥ 0` (`NeuroError::InvalidParams` otherwise). Recommended for new
  analyses (`start = 0`).

`tfce(domain, stat, params) -> Result<Vec<f64>, NeuroError>` allocates a fresh
`TfceWorkspace` per call; `tfce_into(domain, stat, params, ws, out)` reuses a
caller-owned `TfceWorkspace` and writes into a caller-owned `out` — the
allocation-free entry point for a permutation loop. Both compute the same
quantity by one descending union-find sweep (nodes sorted by `stat`,
activated as the threshold drops below them, merged in a disjoint-set
forest). `tfce_naive` re-clusters at every threshold from scratch and is the
readable oracle both are checked against; it agrees with `tfce`/`tfce_into`
to `rel 1e-12` (abs floor `1e-12 · max`), not bit equality, because the
summation order differs.

`tfce_bands(domain, stat, thresholds, weights, e, strict) ->
Result<Vec<f64>, NeuroError>` (and `tfce_bands_into`, allocation-free) take an
explicit strictly increasing `thresholds` list, a same-length `weights` list,
and a `strict` flag (`stat > h` when `true`, `stat >= h` otherwise — nilearn's
convention). `tfce_bands_naive` is the readable per-band re-clustering oracle
for `tfce_bands`. With `thresholds = arange(start, max(stat), step)`, the
matching `Weighting` weights, and `strict = true`, `tfce_bands` is
bit-identical to `tfce`.

Errors: `NeuroError::MismatchedLengths` if `stat.len() != domain.n_nodes()`
(or `out.len() != domain.n_nodes()` for the `_into` variants, checked before
`stat`); `NeuroError::NonFiniteStat` on any NaN or ±∞ in `stat`;
`NeuroError::InvalidParams` if `step <= 0` or any `TfceParams` field is
non-finite, or under `Weighting::Exact` if `h <= -1` or `start < 0`;
`NeuroError::InvalidBands` from the `tfce_bands*` family if `thresholds` and
`weights` differ in length, hold a non-finite value, or `thresholds` is not
strictly increasing.

## One-sample permutation

`OneSampleT::new(x, n) -> Result<Self, NeuroError>` precomputes `Q[v] = Σ_i
x_iv²` once (sign-invariant); `n < 2` is `NeuroError::TooFewSubjects`, and
`x.len()` not a multiple of `n` is `NeuroError::MismatchedLengths`.
`t_map(&self, x, signs, out)` then computes, per realization, only `S[v] =
Σ_i s_i · x_iv` and derives `t = (S/n) / sqrt(var/n)` with `var = (Q −
S²/n) / (n − 1)`, `ddof = 1`, no variance smoothing. A degenerate node
(`var` at or below the rounding floor `4ε·Q`) gets `t = 0`. `x` is
subject-major: `x[i · V + v]` is subject `i` at node `v`.

`sign_flip_plan(n, b_requested) -> SignFlipPlan` chooses exhaustive
enumeration (`b = 2^n`, every sign pattern, draw `k`'s bits encode the
flipped subjects, `k = 0` is the identity) when `n < 64 && 2^n <=
b_requested`, else Monte Carlo (`b = b_requested`, draw 0 the identity, draws
`1..b` from `commonstats::gen_sign_flips(n, draw_id, seed)`).

`OneSampleProblem::new(domain, x, n, params, seed) -> Result<Self,
NeuroError>` validates inputs and precomputes the observed t-map and its TFCE
once. Errors: `NeuroError::TooFewSubjects` (`n < 2`),
`NeuroError::MismatchedLengths` (`x.len() != n · domain.n_nodes()`),
`NeuroError::InvalidParams` (bad `params`).

`PermWorkspace::new(n, n_nodes)` holds the per-draw sign buffer, t buffer,
TFCE buffer, and a `TfceWorkspace`.

`run_range(problem, plan, draws, ws) -> NullPartial` runs draws `draws`
(a `Range<u64>` within `0..plan.b`) and returns their `NullPartial`: per-draw
map maxima (unsorted) and, per node, a count of draws whose TFCE reached the
observed TFCE (`≥`). `NullPartial::merge` (the `Mergeable` impl) concatenates
maxima and adds counts — both order-free, so any partition of `0..B` merges
to the same result regardless of order. `finalize(problem, null) ->
Result<TfceInference, NeuroError>` sorts the merged maxima and turns them into
`p_fwe`, `p_unc`, and `null_max`; it errors with `NeuroError::InvalidParams`
if the null holds no draws.

This example splits `0..B` into two ranges and checks the merged result
against one serial call — the same idea as
`tests/perm_oracle.rs::g7_split_merge_is_deterministic` (which does it with
four ranges and asserts on `p_fwe`, `p_unc`, and `null_max` together):

```rust
use commonstats::accum::Mergeable;
use neurostats::{
    Conn, Domain, NullPartial, OneSampleProblem, PermWorkspace, TfceParams, Weighting,
    finalize, run_range, sign_flip_plan,
};

let dom = Domain::from_volume([3, 1, 1], Conn::Face);
let n = 4usize;
let x = [2.0, 0.3, 0.0, 2.5, -0.4, 0.0, 1.8, 0.1, 0.0, 2.2, -0.2, 0.0];
let p = TfceParams { e: 0.5, h: 2.0, start: 0.0, step: 1.0, weighting: Weighting::Exact };
let problem = OneSampleProblem::new(&dom, &x, n, p, 99).unwrap();
let plan = sign_flip_plan(n, 500);
let mut ws = PermWorkspace::new(n, dom.n_nodes());

let serial = finalize(&problem, &run_range(&problem, &plan, 0..plan.b, &mut ws)).unwrap();

let half = plan.b / 2;
let a = run_range(&problem, &plan, 0..half, &mut ws);
let b = run_range(&problem, &plan, half..plan.b, &mut ws);
let mut merged = NullPartial::empty(dom.n_nodes());
merged.merge(&a);
merged.merge(&b);
let split = finalize(&problem, &merged).unwrap();

assert_eq!(split.p_fwe, serial.p_fwe);
assert_eq!(split.null_max, serial.null_max);
```

`TfceInference` carries `t_obs`, `tfce_obs`, `p_fwe`, `p_unc`, and `null_max`
(the sorted max-TFCE null, `len == B`). Both p-maps are one-sided, positive
tail, count with `≥`, and include the identity draw, so every p is at least
`1/B` and never `0`. A negative effect is tested by negating `x` and running
again.

`tfce_one_sample(domain, x, n, params, seed, b_requested) -> Result<TfceInference,
NeuroError>` is `OneSampleProblem::new` + `sign_flip_plan` + one `run_range`
over `0..B` + `finalize`. `b_requested == 0` is `NeuroError::InvalidParams`.
`tfce_one_sample_threads(domain, x, n, params, seed, b_requested, threads)`
(feature `parallel`) is the same computation, spread over a rayon pool built
per call and sized by `threads`; `threads == 0` uses every core
`std::thread::available_parallelism` reports (falling back to 1); `threads ==
1`, a one-draw plan, or a pool-build failure all fall back to the serial path.

## RNG contract

Realizations are draw-addressable, not streamed: the sign pattern of draw `b`
comes only from `(seed, b)` — exhaustive enumeration decodes the bit pattern
of `b` directly, and Monte Carlo draws come from `commonstats::gen_sign_flips`
(Philox4x32-10, keyed to a sign-flip stream tag distinct from any other RNG
use in `commonstats`) with no state carried between draws. This is why the
thread count cannot change the result: `run_range` rebuilds each draw's signs
from scratch, so splitting `0..B` into any set of ranges and running them on
any number of threads visits the same draws with the same signs; merging a
`NullPartial` only concatenates maxima and adds counts (both order-free
operations), and `finalize` sorts the merged maxima before reading off
p-values, so the order ranges finish in cannot matter either.
`tfce_one_sample_threads` is checked bit-identical to `tfce_one_sample` for
every thread count on every permutation fixture
(`tests/perm_oracle.rs::g12_threaded_driver_matches_serial_bitwise`), and
hand-made splits merged through `NullPartial` are checked the same way
(`::g7_split_merge_is_deterministic`).

`B` counts the identity realization: draw 0 is always the all-`+1.0` sign
pattern (the observed data), so `p ≥ 1/B` and `p` is never exactly `0`.

## Oracles and fixtures

| fixture pattern | generator | oracle tool | tolerance |
|---|---|---|---|
| `tests/fixtures/tfce_*.json` (`expected_mne` field) | `scripts/gen_tfce_golden.py` | MNE-Python 1.12.1 `mne.stats.cluster_level._find_clusters`, `tail=1` | rel `1e-12` |
| `tests/fixtures/tfce_*.json` (`expected_smith_nichols` field) | `scripts/gen_tfce_golden.py` | NumPy transcription of this crate's own `SmithNichols` sum (not an independent tool) | rel `1e-12` |
| `tests/fixtures/perm_exact_*.json` | `scripts/gen_perm_golden.py` | MNE-Python 1.12.1 `permutation_cluster_1samp_test(..., threshold=dict(...), tail=1)`, exact enumeration | rel `1e-12` for `null_max`/`t_obs`/`tfce_obs`; `1/B` (plus `1e-15`) for `p_fwe` |
| `tests/fixtures/perm_mc_*.json` | `scripts/gen_perm_golden.py` | same MNE call, Monte Carlo | rel `1e-12` for `t_obs`/`tfce_obs`; statistical for `p_fwe` (4σ band) and `null_max` (Kolmogorov distance `4/sqrt(B)`) |

Every fixture records `mne_version = 1.12.1`, `numpy_version = 2.3.5`,
`scipy_version = 1.16.3`. `Weighting::Exact` and `p_unc` have no fixture
generated from an external tool: `Exact` is checked against `tfce_naive`
(union-find vs re-clustering) and as the `step → 0` limit of `SmithNichols`;
`p_unc` is checked only by its own counting definition inside `run_range`.
`Domain`'s lattice neighbour rows are checked against a transcription of the
plain `dx, dy, dz` triple loop
(`src/domain.rs::lattice_rows_match_transcribed_dxdydz_scan`), and the CSR
round trip is checked over every masked-volume fixture in
`tests/fixtures/tfce_*.json`
(`tests/tfce_oracle.rs::from_csr_round_trips_every_fixture`).

## Features, `no_std`, WASM

| feature | default | adds |
|---|---|---|
| `std` | yes | the `permute` module (`OneSampleT`-driven permutation inference above); pulls in `commonstats` and the standard library |
| `parallel` | yes | `tfce_one_sample_threads`; implies `std`; adds `rayon` |

With `default-features = false`, only the `no_std` core compiles: `Domain`,
`tfce`/`tfce_into`/`tfce_naive`, `tfce_bands*`, `TfceParams`, `Weighting`,
`TfceWorkspace`. That is the configuration the `wasm32-unknown-unknown` build
uses, since rayon needs threads that target does not have:

```
cargo build --no-default-features --target wasm32-unknown-unknown
```

`#![forbid(unsafe_code)]` crate-wide. The core and `permute` are `#![no_std]`
+ `alloc`; only `tfce_one_sample_threads` links the standard library, for
threads.

## Benchmarks

Both use `std::time::Instant`, run the first pass and discard it, then report
the min of the following passes. No timing numbers are given here — run them
locally and pin/lock the machine first.

- `benches/tfce_bench.rs` — single-map `tfce` on the fixture
  `tests/fixtures/tfce_bench.json` (a masked-volume grid; the fixture also
  carries MNE's own wall time for the same map, `mne_seconds`). Run with
  `cargo bench --bench tfce_bench`.
- `benches/perm_bench.rs` — `run_range` over `B = 200` Monte Carlo draws for
  `n = 30` synthetic subjects built from the same bench fixture's map plus
  fixed-LCG noise; reports ms/draw and the `tfce_into` share of that time.
  Run with `cargo bench --bench perm_bench`.

## CI gates

- `cargo fmt --all -- --check`
- `cargo build --all-features`
- `cargo test --all-features`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --no-default-features --features std`
- `cargo build --no-default-features --target wasm32-unknown-unknown`
