# faststats.neuro — API guide

## Install and import

```bash
pip install faststats
```

```python
import faststats.neuro as ns
```

`faststats.neuro` does not depend on nilearn or scipy — `pyproject.toml` lists only
`numpy>=1.24` as a runtime dependency. Both are pulled in by the `test` extra
(`pip install faststats[test]`), which is meant for running this package's own test
suite, not for using the shim in your own code. If you use the nilearn compatibility
shim (`faststats.neuro.compat.nilearn`), install nilearn and scipy yourself.

## `Domain`

A `Domain` holds the adjacency graph that `tfce` and `tfce_one_sample` enhance over.
Build one once and reuse it across statistic maps.

- `Domain.from_mask(mask, conn=26)` — `mask` is a 3-D boolean array. Builds adjacency
  over the in-mask voxels. `conn` must be `6`, `18`, or `26`; any other value raises
  `ValueError`. Sets `shape` to the mask's shape and `mask` to a copy of it, which
  enables passing `stat`/`x` in 3-D form (see below).
- `Domain.from_volume(shape, conn=26)` — `shape` is a 3-tuple. Builds adjacency over
  every voxel of a full volume (no mask). Same `conn` rule. Sets `shape` and a
  all-`True` `mask`.
- `Domain.from_csr(indptr, indices)` — a caller-built CSR adjacency; symmetry is the
  caller's precondition. `shape` and `mask` are `None`, so only the 1-D node-vector
  form of `tfce`/`tfce_one_sample` applies.
- `Domain.from_adjacency(adj, n=None)` — any object with `.tocoo()` (for example a
  scipy sparse matrix). Symmetrised, self-loops dropped, duplicate edges merged,
  explicitly stored zeros are not edges. `n` defaults to the matrix's own size (it
  must then be square). `shape` and `mask` are `None`.

Attributes: `n_nodes`, `n_edges` (both read from the Rust core), `shape` (the 3-D
shape, or `None`), `mask` (the boolean mask array, or `None`).

## Node order

For a mask-built `Domain`, node order is C-order over the mask — the same order numpy
gives you from `stat[mask]` (`ravel` order). A 1-D `stat`/`x` argument must already be
in this order.

For 3-D/4-D input, `tfce` and `tfce_one_sample` read only the in-mask voxels and scatter
their output back into the full volume shape. Fields with a natural "no data" value
outside the mask are filled with it: `0.0` for enhanced statistic maps and `t_obs`/
`tfce_obs`, `1.0` for `p_fwe`/`p_unc` (a p-value of 1 outside the domain).

## `tfce(stat, domain, *, e=0.5, h=2.0, start, step, weighting, tail="positive")`

Threshold-free cluster enhancement of `stat` over `domain`.

- `stat` — 1-D of shape `(domain.n_nodes,)`, or 3-D of `domain.shape` if the domain
  was built with a shape. Any other shape raises `ValueError`. Values are cast to
  float64. Non-finite values inside the domain raise `ValueError`; for 3-D input,
  voxels outside the mask are never read, so NaN there is ignored.
- `e`, `h` — TFCE extent/height exponents, defaulting to `0.5` and `2.0`.
- `start`, `step`, `weighting` — required keywords; no default is chosen for you,
  because the reference tools disagree on them. `step` may be `None` only when
  `weighting="exact"` (it is then ignored); for the grid weightings it must be a
  positive finite number or `ValueError` is raised.
- `weighting` — one of `"smith_nichols"`, `"mne"`, `"fslmaths"`, `"exact"`. Any other
  value raises `ValueError`.
  - `"smith_nichols"` = `h_i**H * step` (FSL/PALM/SPM convention)
  - `"mne"` = `|h_i - h_{i-1}|**H`, with `|h_0|**H` for the first band (MNE-Python)
  - `"fslmaths"` = `h_i**H`
  - `"exact"` — the closed-form integral from `start`, ignoring `step`
  - the thresholds `h_i` for the three grid weightings are `arange(start, stat.max(), step)`
- `tail` — one of `"positive"`, `"negative"`, `"two_sided"`, `"two_sided_unsigned"`.
  Any other value raises `ValueError`.
  - `"positive"` enhances `stat` as given.
  - `"negative"` enhances `-stat`.
  - `"two_sided"` and `"two_sided_unsigned"` share one grid built from
    `max(|stat|)`, then combine the positive and negative enhancements:
    `"two_sided"` returns `pos - neg` (signed), `"two_sided_unsigned"` returns
    `pos + neg`.

Returns a float64 array: 1-D `(domain.n_nodes,)` for 1-D input, or 3-D `domain.shape`
(zeros outside the mask) for 3-D input.

## `tfce_one_sample(x, domain, *, e=0.5, h=2.0, start, step, weighting, seed, n_perm, n_jobs=-1)`

One-sample sign-flip permutation test with max-TFCE family-wise error correction.

- `x` — `(n_subj, domain.n_nodes)`, or `(n_subj, *domain.shape)` if the domain has a
  shape. Any other shape raises `ValueError`. Cast to float64.
- `e`, `h`, `start`, `step`, `weighting` — same meaning and defaults as `tfce`.
  `weighting="fslmaths"` is not available here and raises `ValueError`.
- `seed` — integer in `[0, 2**64)`; anything else raises `ValueError`.
- `n_perm` — the permutation budget, including the identity permutation. Must be
  `>= 1` or `ValueError` is raised. When `2**n_subj <= n_perm`, every sign pattern is
  enumerated exhaustively; otherwise `n_perm` Monte Carlo sign patterns are drawn from
  `seed`.
- `n_jobs` — follows the nilearn/scikit-learn convention: `-1` (the default) uses
  every core, a positive integer `k` uses exactly `k` OS threads, and `0` or any other
  negative value raises `ValueError`. Results are identical down to the bit for every
  `n_jobs` (`tests/neuro/test_core.py::test_tfce_one_sample_n_jobs_equal_and_validated`).

The statistic tested is the one-sample t (`ddof=1`, `t=0` at nodes with zero variance
across subjects), positive tail. Returns a `TfceInference`:

- `t_obs` — observed one-sample t map.
- `tfce_obs` — observed TFCE-enhanced map.
- `p_fwe` — family-wise error-corrected p-value, `#{max_b >= tfce_obs} / B`.
- `p_unc` — uncorrected p-value per node, `#{tfce_b(v) >= tfce_obs(v)} / B`.
- `null_max` — the sorted max-TFCE null distribution, length `B` (the number of
  permutations actually run).

`t_obs`, `tfce_obs`, `p_fwe`, `p_unc` are 1-D `(domain.n_nodes,)` for 1-D `x`, or 3-D
`domain.shape` for 4-D `x` (0 outside the mask for `t_obs`/`tfce_obs`, 1.0 for the
p-maps). `null_max` is always the flat 1-D array; it has no spatial dimension.

## `grid_and_weights(stat_max, start, step, h, weighting)`

Returns the `(thresholds, weights)` pair that the grid weightings use internally:
`arange(start, stat_max, step)` and the matching band weights (`smith_nichols` =
`h_i**H * step`, `mne` = `|h_i - h_{i-1}|**H` with `|h_0|**H` first, `fslmaths` =
`h_i**H`). `weighting` must be one of the three grid weightings (not `"exact"`);
`step` must be positive and finite; `start` must be `>= 0`. Any violation raises
`ValueError`.

Call it directly when you want to inspect or reuse the threshold grid and per-band
weights that `tfce`/`tfce_one_sample` would use for a given `weighting` — for
example, to plot the weighting curve, or to precompute the grid once for several
calls that share the same `stat_max`.

## Seeds and reproducibility

`seed` is an integer in `[0, 2**64)`, the same range as the Rust crate's RNG. For a
given `seed`, `n_perm`, and input, the sign patterns drawn (or the exhaustive
enumeration, when `2**n_subj <= n_perm`) are fixed — the permutation null and every
value derived from it are reproducible across runs and are unaffected by `n_jobs`.

## Crate behind this

`faststats.neuro` binds the `neurostats` Rust crate through `faststats/python/src/neuro.rs`.
For the domain construction rules, the TFCE algorithm, the sign-flip permutation
scheme, and the crate's own RNG contract, see
[`../rust/README.md`](../rust/README.md).

## Not bound yet

Per `faststats/python/README.md`'s "Not in 0.1.0": `faststats.common`, `faststats.robust`,
NIfTI I/O, GLM contrasts and multi-sample permutation, and surface meshes. The code
confirms there is no I/O helper in this package (`tfce`/`tfce_one_sample` take and
return in-memory numpy arrays only) and no multi-sample or GLM-contrast entry point.
