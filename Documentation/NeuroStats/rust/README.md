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
`tfce_naive` enhance a statistic map over a `Domain`; `tfce_tail_into` does it
for a chosen `Tail` (the map, the negated map, or their signed difference);
`TfceParams` and `Weighting` set the rule; `TfceWorkspace` lets a caller reuse
allocations across calls. `tfce_bands` / `tfce_bands_into` /
`tfce_bands_naive` take an explicit band list instead of a threshold grid.

**L2 — brain store (`no_std` core, no feature needed).** `Store` is the
voxel-major `f64` copy of the brain data that the permutation loop reads,
each node's column residualised against a nuisance basis and scaled to unit
norm. `nuisance_basis` builds that basis from the confounds and the intercept.

**L3 — TFCE permutation inference for a linear model (feature `std`;
threading needs `parallel`).** `tfce_ols` runs
`brain ~ tested + confounds + intercept` end to end; with no tested regressor
it tests the intercept, which without confounds is the one-sample test.
`PermOpts` carries the settings; `PermProblem::ols` holds everything fixed
across draws; `stat_batch` gives the statistic maps of a batch of draws;
`run_range` runs a range of draws into a `NullPartial` using a
`PermWorkspace`; `finalize` turns a merged null into `TfceInference`, which
carries a `NullState`. `draw_permutation` is the row shuffle of one draw.
`run_threads` and `tfce_ols_threads` (feature `parallel`) spread the same
draws over a rayon pool.

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
    pub start: f64,   // first threshold h_0; nodes with stat <= start get 0; >= 0 under Exact; unused by Nilearn
    pub step: f64,    // threshold spacing; > 0, finite; unused by Weighting::Exact and Weighting::Nilearn
    pub weighting: Weighting,
}
```

Under `SmithNichols` and `MneStep`, thresholds are NumPy `arange(start,
max(stat), step)` (start inclusive, stop exclusive), so the largest node
clears every band, and membership at a threshold is strict: `stat > h`.

The four `Weighting` variants:

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
- `Nilearn`: nilearn's `calculate_tfce` under `dh="auto"`. Thresholds are
  NumPy `linspace(0, max(stat), 101)[1:]`: 100 equal steps, rebuilt for every
  map, the last one being the maximum itself. `w_i = h_i^H` with no step
  factor, and membership is `stat >= h_i`, so the largest node is in every
  band. `start` and `step` are not used but are still validated. A map with
  `max(stat) <= 0` gets all zeros.

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

```rust
pub enum Tail { Positive, Negative, TwoSided }
pub fn tfce_tail_into(domain: &Domain, stat: &[f64], params: &TfceParams, tail: Tail,
                      ws: &mut TfceWorkspace, out: &mut [f64]) -> Result<(), NeuroError>
```

`tfce_tail_into` writes `TFCE(stat)` (`Positive`), `TFCE(−stat)` (`Negative`)
or `TFCE(stat) − TFCE(−stat)` (`TwoSided`, signed) into `out`, reusing `ws`.
`tfce_into` is the `Positive` case. Under the grid rules the thresholds come
from the maximum of the map being enhanced, and the two passes of a two-sided
call share one grid, built from `max |stat|`. That changes no value under
`SmithNichols` and `MneStep`, and it is nilearn's two-sided rule under
`Nilearn`.

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

## Permutation inference

### Model and statistic

One linear model per node, the same design for every node:

```
brain ~ tested + confounds + intercept     (tested given: the t of tested)
brain ~ intercept + confounds              (tested = None: the t of the intercept)
```

Nothing is refitted per draw. The preparation is done once:

1. `nuisance_basis(confounds, n, intercept) -> Result<Vec<f64>, NeuroError>`
   orthonormalises the nuisance columns into `Q`: the constant column first
   when `intercept` is set, then the confound columns in order, by modified
   Gram–Schmidt with one re-orthogonalisation pass. A column whose norm after
   residualising is at most `1e-10` of its norm before is dropped; `q` is the
   number kept. `confounds` is row-major `n × c`. The result holds the `q`
   vectors one after another (`q = len / n`). `intercept` is `true` when a
   regressor is tested and `false` when the intercept is tested.
2. `Store::from_f64(src, n, n_nodes, rows, basis)` and `Store::from_f32(..)`
   build the store in one tiled pass over a subject-major source: gather the
   rows listed in `rows`, widen to `f64`, transpose to voxel-major,
   residualise each node's column against `Q` and scale it to unit norm. A
   node whose norm after residualising is at most `1e-10` of its norm before
   is stored as zeros. `n` is the number of rows of the source; the store has
   `rows.len()` subjects. Memory: `rows.len() · V · 8` bytes.
3. `PermProblem::ols(domain, store, tested, opts)` residualises the tested
   column (`tested`, or a constant column for `tested = None`) against `Q`
   and scales it to unit norm. If nothing is left of it (`1e-10` rule again)
   the call fails with `NeuroError::DegenerateDesign`. The design block is
   `W = [tested column, Q]`, `n × p` with `p = 1 + q` and orthonormal columns.

For draw `b` the rows of `W` are transformed by `R_b` (a row shuffle or a sign
flip, below). With `d` the node's store column:

```
beta = (R_b x)' d              nuisance = (R_b Q)' d
rss  = 1 − beta² − Σ nuisance²
t    = beta · sqrt((n − q − 1) / rss)
```

A node with `rss ≤ 4ε · (n − q − 1)` gets `t = 0` (ε is the `f64` epsilon),
and so does a node the store zeroed. This is the t of the full model fitted
to the nuisance-residualised data with the design rows transformed by `R_b`.
For draw 0 it is the ordinary OLS t of the raw data, and with `tested = None`
and no confounds it is the one-sample t, `mean / sqrt(var / n)` with
`ddof = 1`. nilearn 0.14.1 (`t_score_with_covars_and_normalized_design`) has
the same formula. The nuisance term is part of every draw's statistic: with
confounds `(R_b Q)' d` is not zero for `b ≥ 1`.

### Draws

- `tested` given: the rows of `W` are shuffled together and the brain data
  stay fixed. This gives the Freedman–Lane statistic.
  `draw_permutation(n, draw, seed, out)` writes the shuffle: row `i` of the
  shuffled design is row `out[i]` of the original. Draw 0 is the identity;
  draw `b ≥ 1` is `commonstats::gen_resample_indices(Scheme::Permutation, n,
  b, seed)`.
- `tested = None`: the rows are sign-flipped. When `n < 64 && 2^n ≤ n_perm`
  every pattern is enumerated: `B = 2^n`, and draw `k` flips subject `i` when
  bit `i` of `k` is set. Otherwise draw 0 is the identity and draw `b ≥ 1` is
  `commonstats::gen_sign_flips(n, b, seed)`.

`B` counts the identity draw.

### Options and the loop

```rust
pub struct PermOpts<'a> {
    pub params: TfceParams,              // TFCE settings
    pub tail: Tail,                      // Positive, Negative or TwoSided
    pub seed: u64,
    pub n_perm: u64,                     // draw budget, identity included; 1..=u32::MAX
    pub batch: usize,                    // draws per pass over the store; >= 1
    pub extend: Option<&'a NullState>,   // a null to continue
}
```

`PermOpts::new(params, tail, seed, n_perm)` sets the default `batch` and
`extend = None`. The batch size changes speed and memory, never a result:
each dot product is one sum in subject order whatever the batch.

`PermWorkspace::new(&problem)` holds one thread's buffers: the batch matrix
(`n × batch·p`), `batch` statistic maps, one TFCE map and a `TfceWorkspace`.

`stat_batch(problem, draws, ws) -> &[f64]` returns the statistic maps of up to
`batch` draws, one after another, without TFCE.

`run_range(problem, draws, ws) -> NullPartial` runs the draws `draws` (a
`Range<u64>` within `problem.draws()`) and returns their `NullPartial`: one
null value per draw (unsorted) and, per node, a count of draws whose tested
quantity reached the observed one (`≥`). `NullPartial::merge` (the
`Mergeable` impl) concatenates the null values and adds the counts, both
order-free, so any partition of the draws merges to the same result.
`finalize(problem, null) -> Result<TfceInference, NeuroError>` sorts the
merged null values and turns them into p-maps; it errors with
`NeuroError::InvalidParams` if `null` does not hold exactly the problem's
draws.

| Tail | TFCE per draw | Tested quantity `T` | Null value of a draw |
|---|---|---|---|
| `Positive` | TFCE of the map | `tfce` | map maximum |
| `Negative` | TFCE of the negated map | `tfce` of the negated map | map maximum |
| `TwoSided` | TFCE of the map minus TFCE of the negated map | `\|tfce\|` | maximum of `\|tfce\|` |

```rust
pub struct TfceInference {
    pub stat_obs: Vec<f64>,   // observed statistic, signed
    pub p_fwe: Vec<f64>,      // #{b : max_b >= T(v)} / B
    pub p_unc: Vec<f64>,      // #{b : T_b(v) >= T(v)} / B
    pub null: NullState,
}

pub struct NullState {
    pub seed: u64,
    pub maxima: Vec<f64>,     // one null value per draw, sorted ascending; len B
    pub exceed: Vec<u32>,     // per node: draws with T_b(v) >= T(v)
    pub tfce_obs: Vec<f64>,   // observed TFCE map (signed under TwoSided)
    pub params: TfceParams,
    pub tail: Tail,
    pub n: usize,
    pub n_nodes: usize,
    pub q: usize,
    pub engine: Engine,
}
```

Both p-maps count with `≥` and include the identity draw, so every p is at
least `1/B` and never `0`.

This example builds the pieces by hand from an `f32` source with a row index,
splits the draws into two ranges, and checks the merged result against
`tfce_ols` on the same rows:

```rust
use commonstats::accum::Mergeable;
use neurostats::{
    Conn, Domain, NullPartial, PermOpts, PermProblem, PermWorkspace, Store, Tail, TfceParams,
    Weighting, finalize, nuisance_basis, run_range, tfce_ols,
};

let dom = Domain::from_volume([3, 1, 1], Conn::Face);
// Five source rows of three nodes; row 2 is left out of the model.
let src: [f32; 15] = [
    2.0, 0.3, 0.0, 2.5, -0.4, 0.5, 9.0, 9.0, 9.0, 1.8, 0.1, -0.5, 2.2, -0.2, 1.0,
];
let rows = [0usize, 1, 3, 4];
let n = rows.len();
let tested = [31.0, 45.0, 28.0, 52.0];
let confounds = [0.0, 1.0, 1.0, 0.0]; // row-major n × 1
let p = TfceParams { e: 0.5, h: 2.0, start: 0.0, step: 1.0, weighting: Weighting::Exact };
let opts = PermOpts::new(p, Tail::TwoSided, 99, 500);

let basis = nuisance_basis(&confounds, n, true).unwrap(); // a regressor is tested
let store = Store::from_f32(&src, 5, dom.n_nodes(), &rows, &basis).unwrap();
let problem = PermProblem::ols(&dom, &store, Some(&tested), &opts).unwrap();
let draws = problem.draws(); // 0..500
let mut ws = PermWorkspace::new(&problem);

let half = draws.end / 2;
let a = run_range(&problem, draws.start..half, &mut ws);
let b = run_range(&problem, half..draws.end, &mut ws);
let mut merged = NullPartial::empty(dom.n_nodes());
merged.merge(&a);
merged.merge(&b);
let split = finalize(&problem, &merged).unwrap();

// The same rows as an f64 slice, through the one-call function.
let y: Vec<f64> = rows
    .iter()
    .flat_map(|&r| src[r * 3..r * 3 + 3].iter().map(|&x| f64::from(x)))
    .collect();
let whole = tfce_ols(&dom, &y, n, Some(&tested), &confounds, &opts).unwrap();
assert_eq!(split, whole);
```

`tfce_ols(domain, y, n, tested, confounds, opts) -> Result<TfceInference,
NeuroError>` is `nuisance_basis` + `Store::from_f64` with every row +
`PermProblem::ols` + one `run_range` over `problem.draws()` + `finalize`.
`tfce_ols_threads(.., threads)` (feature `parallel`) is the same computation
with `run_threads(problem, draws, threads)` in place of `run_range`: a rayon
pool built per call and sized by `threads`; `threads == 0` uses every core
`std::thread::available_parallelism` reports (falling back to 1);
`threads == 1`, a single draw, or a pool-build failure all fall back to the
serial path.

### Extending a null

A result carries its `NullState` in `null`. A later call with
`opts.extend = Some(&state)` and a larger `n_perm` continues it:

```rust
let first = tfce_ols(&dom, &y, n, Some(&tested), &confounds, &opts).unwrap(); // 500 draws
let mut more = PermOpts::new(p, Tail::TwoSided, 99, 2000);
more.extend = Some(&first.null);
let longer = tfce_ols(&dom, &y, n, Some(&tested), &confounds, &more).unwrap();
assert_eq!(longer.null.maxima.len(), 2000); // only draws 500..2000 were run
```

- `n_perm` is the new total. The call fails when the stored null already has
  that many draws or more.
- The run recomputes the observed TFCE map. It must equal the stored one bit
  for bit, and the seed, the TFCE settings (every field), the tail, `n`,
  `n_nodes`, `q` and the engine must match.
- With sign flips the call fails when `2^n ≤ n_perm`: a budget that large
  enumerates every pattern, an enumerated draw is not a seeded draw, and an
  enumerated null is already complete.
- Every failure is `NeuroError::InvalidParams`.
- The result equals one run with the larger budget, bit for bit
  (`tests/perm_oracle.rs::g14_extending_a_null_equals_one_run`).

The crate does no I/O. To keep a null between processes, write the fields of
`NullState` and build the struct again from them.

### Errors

`NeuroError::TooFewSubjects` when `n − q − 1 < 1` (so `n < 2` for the
one-sample test) or there is no subject; `NeuroError::DegenerateDesign` when
the tested column is in the span of the nuisance columns;
`NeuroError::NonFiniteStat` on a NaN or ±∞ in the brain data, `tested` or the
confounds; `NeuroError::MismatchedLengths` when `y`, `tested`, `confounds`,
the basis or the store do not fit `n` and the domain;
`NeuroError::InvalidParams` for `n_perm == 0`, `n_perm > u32::MAX`,
`batch == 0`, invalid TFCE settings, a row index past the source, or a null
that cannot be continued.

## RNG contract

Draws are addressable, not streamed: the row shuffle or sign pattern of draw
`b` comes only from `(seed, b)`. Exhaustive enumeration decodes the bit
pattern of `b` directly; Monte Carlo sign flips come from
`commonstats::gen_sign_flips` and row shuffles from
`commonstats::gen_resample_indices` (Philox4x32-10, each keyed to its own
stream), with no state carried between draws. This is why neither the thread
count nor the batch size can change the result: `run_range` rebuilds each
draw's design block from scratch, every dot product of the statistic is one
sum in subject order whatever the batch, and splitting the draws into any set
of ranges on any number of threads visits the same draws with the same
blocks. Merging a `NullPartial` only concatenates null values and adds counts
(both order-free operations), and `finalize` sorts the merged values before
reading off p-values, so the order ranges finish in cannot matter either.
`tfce_ols_threads` is checked bit-identical to `tfce_ols` for thread counts
1, 2, 3 and 7 and several batch sizes on every permutation fixture
(`tests/perm_oracle.rs::g12_threaded_driver_matches_serial_bitwise`), and
hand-made splits merged through `NullPartial` are checked the same way
(`::g7_split_merge_is_deterministic`).

`B` counts the identity draw: draw 0 is always the design as observed, so
`p ≥ 1/B` and `p` is never exactly `0`.

## Oracles and fixtures

| fixture pattern | generator | oracle tool | tolerance |
|---|---|---|---|
| `tests/fixtures/tfce_*.json` (`expected_mne` field) | `scripts/gen_tfce_golden.py` | MNE-Python 1.12.1 `mne.stats.cluster_level._find_clusters`, `tail=1` | rel `1e-12` |
| `tests/fixtures/tfce_*.json` (`expected_smith_nichols` field) | `scripts/gen_tfce_golden.py` | NumPy transcription of this crate's own `SmithNichols` sum (not an independent tool) | rel `1e-12` |
| `tests/fixtures/perm_exact_*.json` | `scripts/gen_perm_golden.py` | MNE-Python 1.12.1 `permutation_cluster_1samp_test(..., threshold=dict(...), tail=1)`, exact enumeration | rel `1e-12` for `null.maxima`/`stat_obs`/`null.tfce_obs`; `1/B` (plus `1e-15`) for `p_fwe` |
| `tests/fixtures/perm_mc_*.json` | `scripts/gen_perm_golden.py` | same MNE call, Monte Carlo | rel `1e-12` for `stat_obs`/`null.tfce_obs`; statistical for `p_fwe` (4σ band) and `null.maxima` (Kolmogorov distance `4/sqrt(B)`) |

Every fixture records `mne_version = 1.12.1`, `numpy_version = 2.3.5`,
`scipy_version = 1.16.3`. The MNE permutation fixtures test the one-sample
case of `tfce_ols` (`tested = None`, no confounds, `Tail::Positive`).
`Weighting::Exact` and `p_unc` have no fixture generated from an external
tool: `Exact` is checked against `tfce_naive` (union-find vs re-clustering)
and as the `step → 0` limit of `SmithNichols`; `p_unc` is checked against a
naive per-draw loop (below).

The linear model has no frozen fixture. `tests/ols_oracle.rs` checks it in
process:

- `Store` and `nuisance_basis` against least-squares residuals from the
  normal equations on the raw columns
  (`store_columns_are_unit_norm_ols_residuals`);
- the t of every draw against one normal-equations fit per node with that
  draw's design, for a tested regressor and for the intercept, with and
  without confounds, on data where the nuisance part of the residual variance
  moves from draw to draw (`o3_draw_statistics_match_per_draw_fits`, rel
  `1e-9`);
- the maps, the null and both p-maps of `tfce_ols` against a loop that refits
  every node for every draw, three tails
  (`o4_inference_matches_a_naive_loop`);
- the shuffled-design statistic against a direct Freedman–Lane loop
  (`o7_shuffled_design_gives_the_freedman_lane_statistic`).

The Python binding compares the same engine in process with nilearn 0.14.1
`permuted_ols(..., tfce=True)`: the t map and the TFCE map to rel `1e-10` under
`Weighting::Nilearn`, and `p_fwe` within a 4σ Monte Carlo band
(`NeuroStats/python/tests/test_ols.py::test_matches_nilearn_permuted_ols`).
`Weighting::Nilearn` has no frozen fixture either: the Python binding checks
it in process against nilearn `calculate_tfce` on 1000 random maps and masks,
each under the three tails, to rel `1e-12` (`NeuroStats/python/tests/test_tfce_parity.py`), and `tfce` is
checked against `tfce_naive` here. The three tails are checked against
`tfce_naive` (`tests/tfce_oracle.rs::g13_tails_match_naive`).
`Domain`'s lattice neighbour rows are checked against a transcription of the
plain `dx, dy, dz` triple loop
(`src/domain.rs::lattice_rows_match_transcribed_dxdydz_scan`), and the CSR
round trip is checked over every masked-volume fixture in
`tests/fixtures/tfce_*.json`
(`tests/tfce_oracle.rs::from_csr_round_trips_every_fixture`).

## Features, `no_std`, WASM

| feature | default | adds |
|---|---|---|
| `std` | yes | the `permute` module (permutation inference above); pulls in `commonstats` and the standard library |
| `parallel` | yes | `run_threads` and `tfce_ols_threads`; implies `std`; adds `rayon` |

With `default-features = false`, only the `no_std` core compiles: `Domain`,
`tfce`/`tfce_into`/`tfce_tail_into`/`tfce_naive`, `tfce_bands*`, `TfceParams`,
`Weighting`, `Tail`, `TfceWorkspace`, `Store`, `nuisance_basis`. That is the
configuration the `wasm32-unknown-unknown` build uses, since rayon needs
threads that target does not have:

```
cargo build --no-default-features --target wasm32-unknown-unknown
```

`#![forbid(unsafe_code)]` crate-wide. The core and `permute` are `#![no_std]`
+ `alloc`; only `run_threads` and `tfce_ols_threads` link the standard
library, for threads.

## Benchmarks

Both use `std::time::Instant`, run the first pass and discard it, then report
the min of the following passes. No timing numbers are given here — run them
locally and pin/lock the machine first.

- `benches/tfce_bench.rs` — single-map `tfce` on the fixture
  `tests/fixtures/tfce_bench.json` (a masked-volume grid; the fixture also
  carries MNE's own wall time for the same map, `mne_seconds`). Run with
  `cargo bench --bench tfce_bench`.
- `benches/perm_bench.rs` — `run_range` over `B = 200` Monte Carlo draws of
  the one-sample test for `n = 30` synthetic subjects built from the same
  bench fixture's map plus fixed-LCG noise; reports ms/draw and the
  `tfce_into` share of that time. Run with `cargo bench --bench perm_bench`.
- `benches/cell_bench.rs` — the OLS permutation loop on a data cell read
  from disk (a mask, an `f32` source with a row index, a tested regressor
  and confounds). `split` times the statistic maps and TFCE apart on one
  thread; `run` times the whole loop at a given thread count and batch size
  and reports the peak anonymous memory. The file's header describes the
  cell format and the arguments. With no arguments it prints its usage.

## CI gates

- `cargo fmt --all -- --check`
- `cargo build --all-features`
- `cargo test --all-features`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --no-default-features --features std`
- `cargo build --no-default-features --target wasm32-unknown-unknown`
