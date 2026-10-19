# neurostats — Python API guide

## Install and import

```bash
pip install neurostats
```

```python
import neurostats as ns
```

`neurostats` does not depend on nilearn or scipy — `pyproject.toml` lists only
`numpy>=1.24` as a runtime dependency. The `test` extra
(`pip install neurostats[test]`) pulls in what this package's own test suite needs.

## `Domain`

A `Domain` holds the adjacency graph that `tfce` and `tfce_ols` work over. Build one
once and reuse it across statistic maps and inferences.

- `Domain.from_mask(mask, conn=6)` — `mask` is a 3-D boolean array. Builds adjacency
  over the in-mask voxels. `conn` must be `6`, `18`, or `26`; any other value raises
  `ValueError`. Sets `shape` to the mask's shape and `mask` to a copy of it, which
  enables passing `stat` in 3-D form to `tfce` (see below).
- `Domain.from_volume(shape, conn=6)` — `shape` is a 3-tuple. Builds adjacency over
  every voxel of a full volume (no mask). Same `conn` rule. Sets `shape` and an
  all-`True` `mask`.
- `Domain.from_csr(indptr, indices)` — a caller-built CSR adjacency; symmetry is the
  caller's precondition. `shape` and `mask` are `None`, so only the 1-D node-vector
  form of `tfce` applies.
- `Domain.from_adjacency(adj, n=None)` — any object with `.tocoo()` (for example a
  scipy sparse matrix). Symmetrised, self-loops dropped, duplicate edges merged,
  explicitly stored zeros are not edges. `n` defaults to the matrix's own size (it
  must then be square). `shape` and `mask` are `None`.

Attributes: `n_nodes`, `n_edges` (both read from the Rust core), `shape` (the 3-D
shape, or `None`), `mask` (the boolean mask array, or `None`).

## Node order

For a mask-built `Domain`, node order is C-order over the mask — the same order numpy
gives you from `stat[mask]` (`ravel` order), and the column order of a nilearn
`NiftiMasker`. A 1-D `stat` and the columns of the `data` of `tfce_ols` must already be
in this order.

For 3-D input, `tfce` reads only the in-mask voxels and scatters its output back into
the full volume shape, with `0.0` outside the mask. `tfce_ols` returns node vectors;
`export_to` puts them back into images.

## `tfce(stat, domain, *, params="exact", tail="two_sided")`

Threshold-free cluster enhancement of `stat` over `domain`.

- `stat` — 1-D of shape `(domain.n_nodes,)`, or 3-D of `domain.shape` if the domain
  was built with a shape. Any other shape raises `ValueError`. Values are cast to
  float64. Non-finite values inside the domain raise `ValueError`; for 3-D input,
  voxels outside the mask are never read, so NaN there is ignored.
- `params` — the TFCE settings: a preset name or a dict.
  - `"exact"` (the default) — the closed-form integral
    `∫ extent(h)**E * h**H dh` from `start = 0`, with `E = 0.5` and `H = 2.0`. No
    threshold grid.
  - `"nilearn"` — nilearn's `calculate_tfce` rule with `E = 0.5` and `H = 2.0`: 100
    equal steps from 0 up to and including the map's own maximum, weight `h_i**H` per
    threshold with no step factor, and a voxel belongs to a threshold when its value is
    `>= h_i`.
  - a dict with the key `weighting` and any of `e`, `h`, `start`, `step`. `e` and `h`
    default to `0.5` and `2.0`.

    | `weighting` | Band weight | `start` | `step` |
    |---|---|---|---|
    | `"exact"` | closed-form integral | optional, default 0 | not taken |
    | `"nilearn"` | `h_i**H` | not taken | not taken |
    | `"mne"` | `\|h_i - h_{i-1}\|**H`, with `\|h_0\|**H` for the first band (MNE-Python) | required | required |
    | `"smith_nichols"` | `h_i**H * step` (FSL/PALM/SPM convention) | required | required |
    | `"fslmaths"` | `h_i**H` | required | required |

    The thresholds `h_i` of `"mne"`, `"smith_nichols"` and `"fslmaths"` are
    `arange(start, max, step)`, and a voxel belongs to a threshold when its value is
    `> h_i`. `start` must be `>= 0` and `step` `> 0`.
  - An unknown preset name, an unknown key, a key the weighting does not take, a
    missing required key, or a value out of range raises `ValueError` before `stat` is
    read.
- `tail` — one of `"positive"`, `"negative"`, `"two_sided"` (the default). Any other
  value raises `ValueError`.
  - `"positive"` enhances `stat` as given.
  - `"negative"` enhances `-stat`.
  - `"two_sided"` returns the positive enhancement minus the negative one, so the
    output is signed. Under the grid weightings both passes share one grid, built from
    `max(|stat|)`.

Returns a float64 array: 1-D `(domain.n_nodes,)` for 1-D input, or 3-D `domain.shape`
(zeros outside the mask) for 3-D input.

## `grid_and_weights(stat_max, params)`

Returns the `(thresholds, weights)` pair of a grid weighting: `arange(start, stat_max,
step)` and the matching band weights (`smith_nichols` = `h_i**H * step`, `mne` =
`|h_i - h_{i-1}|**H` with `|h_0|**H` first, `fslmaths` = `h_i**H`).

`params` has no default. It is the dict form described under `tfce`, with `weighting`
one of `"smith_nichols"`, `"mne"` or `"fslmaths"`, and with `start` and `step`. Anything
else raises `ValueError`.

Call it when you want to inspect or reuse the threshold grid and per-band weights of a
weighting — for example, to plot the weighting curve.

## `tfce_ols(data, space=None, *, tested, confounds=None, conn=None, params="exact", tail="two_sided", seed=None, n_perm=10000, n_jobs=1, extend=None)`

TFCE permutation inference for one linear model per node, with max-TFCE family-wise
error correction. Returns a `TfceInference`.

### Model and statistic

```
brain ~ tested + confounds + intercept     (tested is an array)
brain ~ intercept + confounds              (tested="intercept")
```

The design is the same at every node. The statistic is the t of `tested`, or the t of
the intercept. With `tested="intercept"` and no confounds it is the one-sample t
(`ddof = 1`, `mean / sqrt(var / n)`).

The t is computed as `b * sqrt((n - q - 1) / (1 - b**2 - a2))`. The node's data and the
tested column are first residualised against the nuisance columns (the confounds, plus
the intercept when a regressor is tested) and scaled to unit length; `b` is their dot
product, `a2` is the part of the node's variance that the nuisance columns take in the
draw, and `q` is the number of nuisance columns kept. For the unpermuted data this is the
ordinary least-squares t. A node with no residual variance (a constant node, or a node
that is a combination of the nuisance columns) gets `t = 0`.

### Draws

- `tested` an array: the rows of the design (`tested` and the confounds together) are
  shuffled and the brain data stay fixed. This gives the Freedman–Lane statistic.
- `tested="intercept"`: the rows are sign-flipped. When `2**n <= n_perm`, every one of
  the `2**n` sign patterns is used once, and the result has `2**n` draws.
- Draw 0 is the data as observed, and it counts as one of the `n_perm` draws.

### Arguments

- `data` — the brain data: an `(n, V)` array with one row per subject and one column
  per node of `space`; images, with a masker or a mask image as `space`; or a
  `MaskedData`. A C-contiguous float32 or float64 array is read in place, without a
  copy; this includes a NumPy memory map (`np.load(..., mmap_mode="r")`). Anything else
  (another dtype, Fortran order, a non-contiguous view, a list) is converted to float64
  first. The engine itself computes in float64 on its own copy, which takes `n * V * 8`
  bytes. A NaN or an infinite value raises `ValueError`.
- `space` — a `Domain` with `V` nodes, a fitted nilearn masker, or a mask image. It is
  left out when `data` is a `MaskedData`.
- `conn` — `6`, `18` or `26`: the voxel connectivity of the `Domain` that is built from a
  masker or a mask image. `None` (the default) is 6. With a `Domain` or a `MaskedData`
  it must be left as `None`: they have their connectivity already.

  `data`, `space` and `conn` go through `import_from`, which lists every accepted
  combination and its errors (see below).
- `tested` — required, keyword-only. A 1-D array with one value per subject, or the
  string `"intercept"`.
- `confounds` — `(n, c)`, or `None`. No constant column is needed. A column that is a
  combination of the columns before it is dropped, and so is a constant column when a
  regressor is tested.

  A row whose `tested` value or any confound value is NaN is left out of the model, and
  the call warns (`UserWarning`) with the number of rows left out; `n` is then the
  number of rows kept. With `tested="intercept"` only the confounds count. The kept rows
  of `data` are read without a copy. An infinite value raises `ValueError`.
- `params` — the TFCE settings, a preset name or a dict, as described under `tfce`.
  `"fslmaths"` is not available here.
- `tail` — `"positive"`, `"negative"` or `"two_sided"` (the default).

  | `tail` | TFCE per draw | Tested quantity `T` | Null value of a draw |
  |---|---|---|---|
  | `"positive"` | TFCE of the map | the TFCE value | the map maximum |
  | `"negative"` | TFCE of the negated map | the TFCE value | the map maximum |
  | `"two_sided"` | TFCE of the map minus TFCE of the negated map | its absolute value | the maximum of the absolute value |

- `seed` — an integer in `[0, 2**64)`, or `None` (the default) for a fresh seed. The
  seed used is stored in the result, so any run can be repeated.
- `n_perm` — the draw budget, default `10000`. It counts the unpermuted data.
- `n_jobs` — `1` (the default) uses one thread, `-1` uses every core, a positive
  integer `k` uses exactly `k` OS threads, and `0` or any other negative value raises
  `ValueError`. A result
  has the same bits for every `n_jobs`
  (`tests/test_ols.py::test_every_n_jobs_gives_the_same_bits`).
- `extend` — an earlier result to continue (see "Extending a null").

### Errors

Every problem with the inputs raises `ValueError`. `params`, `tail`, `seed`, `n_perm`,
`n_jobs`, a `tested` string other than `"intercept"` and the rules of `extend` are
checked before `data` is read. The design errors are:

- `tested` is constant or a combination of the confounds, or, with
  `tested="intercept"`, a confound is constant ("degenerate design");
- too few subjects: `n - q - 1` must be at least 1, counted on the rows kept;
- `tested` or `confounds` does not have one row per subject;
- `tested` or `confounds` holds an infinite value.

## `TfceInference`

The result of `tfce_ols`. The maps are 1-D float64 arrays with one value per node.

| Field | Content |
|---|---|
| `stat_obs` | the observed statistic, signed under every tail |
| `tfce_obs` | its TFCE under `tail` (see the table above); signed for `"two_sided"` |
| `p_fwe` | family-wise error corrected p: `#{b : max_b >= T(v)} / B` over the max-TFCE null |
| `p_unc` | uncorrected p: `#{b : T_b(v) >= T(v)} / B` over the node's own null |
| `null_max` | the null value of every draw, sorted ascending, length `B` |
| `params` | the resolved TFCE settings: `weighting`, `e`, `h`, and the `start` and `step` the weighting uses |
| `tail` | the tail of the run |
| `seed` | the seed of the run; the drawn one when the call had `seed=None` |
| `n_perm` | `B`, the number of draws used. It is `2**n`, below the requested budget, when every sign pattern was enumerated |
| `origin` | the masker or mask image the data came through; `None` when `space` was a `Domain`, and for a loaded result |

`B` counts the unpermuted data, so both p-values are at least `1 / B` and never 0.
nilearn's `(1 + #{null >= observed}) / (n_perm + 1)` is the same rule with
`B = n_perm + 1`.

- `result.save(path)` writes the result to `path` as an `.npz` file. The file name is
  used as given.
- `TfceInference.load(path)` reads it back. A file that is not a saved result raises
  `ValueError`.

A saved result holds every field above except `origin`, and what `extend` needs.

## Extending a null

```python
r = ns.tfce_ols(y, dom, tested=x, confounds=c, n_perm=1000)
r = ns.tfce_ols(y, dom, tested=x, confounds=c, n_perm=5000, extend=r)
```

The second call runs draws 1000 to 4999 only and merges them into the stored null. The
result equals one run of 5000 draws with the same seed, bit for bit
(`tests/test_ols.py::test_extend_equals_one_long_run`). A result read with
`TfceInference.load` extends the same way.

- `n_perm` is the new total. If the stored null already has that many draws or more,
  the call raises a `ValueError` that states the stored number.
- `seed=None` takes the stored seed. A seed that is given must equal it.
- `data`, `tested`, `confounds`, `params` and `tail` must be those of the stored run.
  The call recomputes the observed TFCE map and compares it with the stored one bit
  for bit; any difference raises `ValueError`.
- With `tested="intercept"`, a budget of `2**n` or more cannot continue a null: such a
  budget enumerates every sign pattern, which is a different set of draws, and an
  enumerated null is already complete.

## `import_from(data, space=None, *, conn=None)`

Returns the brain data and their space as a `MaskedData`: an `(n, V)` array and the
`Domain` its columns belong to. Every inference function calls it on its own `data`,
`space` and `conn` arguments, so what is accepted here is accepted there. Call it
yourself to mask once and run several inferences on the result.

| `space` | `data` | What happens |
|---|---|---|
| `Domain` | array `(n, V)` | both are used as they are; `origin` is `None` |
| fitted masker | images | `masker.transform(images)`; the `Domain` is built from `masker.mask_img_` |
| fitted masker | array `(n, V)` | the array as it is; the `Domain` is built from `masker.mask_img_` |
| mask image | images | masked with a nilearn `NiftiMasker` built on that mask; the only case that imports nilearn |
| mask image | array `(n, V)` | the array as it is; the `Domain` is built from the mask |
| left out | `MaskedData` | returned unchanged |

- **Recognition.** A masker is an object with `transform` and `inverse_transform`; it is
  fitted when it also has `mask_img_`. An image is an object with `affine` and `dataobj`
  (a nibabel image). nilearn is not imported to recognise them.
- **Images** as `data` are one image, one file name, or a list of them: whatever the
  masker's `transform` turns into `(n, V)`, such as a list of 3-D images, one 4-D image,
  or a list of file names. One 3-D image alone is masked to a single vector and is
  refused; pass a list of one.
- **Arrays** are `(n, V)`, with `V` the number of in-mask voxels. A C-contiguous float32
  or float64 array is kept without a copy; anything else is converted to float64.
- **The mask.** A voxel is in the mask when its value is non-zero and finite, which is
  the rule of nilearn's `NiftiMasker`. The mask must be 3-D. Node order is C-order over
  the mask, the column order of `NiftiMasker.transform`
  (`tests/test_io.py::test_node_order_is_the_column_order_of_niftimasker_transform`).
- **`conn`** — `6`, `18` or `26`, the connectivity of the `Domain` built from a masker or
  a mask image; `None` (the default) is 6. With a `Domain` or a `MaskedData` it must be
  left as `None`.

`MaskedData` has three fields:

| Field | Content |
|---|---|
| `y` | the `(n, V)` array; column `j` is node `j` of `domain` |
| `domain` | the `Domain` |
| `origin` | the masker or mask image given as `space`; `None` when `space` was a `Domain` |

Errors (`ValueError`): images with no `space`, or with a `Domain`; a masker that is not
fitted; a masker whose `mask_img_` is not an image (a surface masker); a `space` of any
other kind, such as an array or a file name; a mask that is not 3-D; data that are not
`(n, V)`; `conn` given with a `Domain`; `space` or `conn` given with a `MaskedData`.
Masking images through a mask image raises `ImportError` when nilearn is not installed.

`masker.transform` returns the whole `(n, V)` array in memory, and the engine's float64
copy comes on top of it. A float32 or float64 memory map of an array masked earlier
(`np.load(path, mmap_mode="r")`) is read in place and avoids the first of the two.

## `export_to(result, target=None)`

Returns the maps of a `TfceInference` as a dict of images:

| Key | Map | Value outside the mask |
|---|---|---|
| `"stat"` | `stat_obs` | 0 |
| `"tfce"` | `tfce_obs` | 0 |
| `"logp_fwe"` | `-log10(p_fwe)` | 0 |
| `"logp_unc"` | `-log10(p_unc)` | 0 |

- The p-maps are exported as `-log10(p)` because an image is 0 outside the mask, and 0
  then reads as `p = 1`. nilearn's permutation functions use the same form
  (`logp_max_tfce`). No p-value of a result is 0, so the maps are finite. The p-values
  themselves stay on the result as arrays.
- `target` — a fitted masker or a mask image. Left as `None`, the maps go back through
  `result.origin`. It is required when `result.origin` is `None`: a result computed with
  a `Domain`, or one read with `TfceInference.load`. When given, it replaces the origin.
- Through a masker, each image is `masker.inverse_transform(map)`
  (`tests/test_io.py::test_export_equals_inverse_transform`).
- Through a mask image, each image has the class of the mask image and is built on the
  mask's affine and a copy of its header. The data type in that header is set to
  float64, so a saved file holds the values as they are; every other header field is
  the mask's.

Errors (`ValueError`): a `result` that lacks a field of a `TfceInference`; no origin and no `target`; a `target` that is not a fitted masker
or a mask image; a mask whose number of voxels is not the number of nodes of the result.

## Seeds and reproducibility

`seed` is an integer in `[0, 2**64)`, the same range as the Rust crate's RNG. A draw
depends only on the seed and on its own number, so for a given `seed`, `n_perm` and
input the null and every value derived from it are the same on every run, for every
`n_jobs`, and whether the draws were run in one call or added with `extend`. An
enumerated null (`tested="intercept"` with `2**n <= n_perm`) does not depend on the seed
at all.

## Crate behind this

`neurostats` binds the `neurostats` Rust crate through `NeuroStats/python/src/lib.rs`.
For the domain construction rules, the TFCE algorithm, the permutation engine, and the
crate's own RNG contract, see [`../rust/README.md`](../rust/README.md).

## Not bound yet

Reading and writing image files (nilearn and nibabel do that; `import_from` and
`export_to` take and return their objects), surface maskers and surface images, more
than one tested regressor per call, and cluster size and cluster mass inference.
