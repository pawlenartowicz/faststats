# neurostats and nilearn

nilearn prepares the data and displays the result; `neurostats` does the inference in
between. The inference functions take nilearn's maskers and images as they are, and
`export_to` gives the result back as images. nilearn is not a dependency of
`neurostats`: its objects are recognised by their attributes, and `import neurostats`
never imports it.

## Three ways to call

```python
import numpy as np
import neurostats as ns
from nilearn.maskers import NiftiMasker

masker = NiftiMasker(mask_img="mask.nii.gz").fit()

# 1. nilearn objects straight in
res = ns.tfce_ols(subject_files, masker, tested=age, confounds=conf)
out = ns.export_to(res)                       # a dict of images
out["logp_fwe"].to_filename("age_logp_fwe.nii.gz")

# 2. mask once, run several inferences
d = ns.import_from(subject_files, masker)
res_age = ns.tfce_ols(d, tested=age, confounds=conf)
res_one = ns.tfce_ols(d, tested="intercept")

# 3. plain arrays
y = masker.transform(subject_files)           # (n_subjects, n_voxels)
dom = ns.Domain.from_mask(np.asarray(masker.mask_img_.dataobj) != 0, conn=6)
res = ns.tfce_ols(y, dom, tested=age, confounds=conf)
out = ns.export_to(res, target=masker)
```

With the same seed the three forms give the same result, bit for bit.

## Objects in: `import_from`

`import_from(data, space=None, *, conn=None)` is what every inference function runs on its own
`data`, `space` and `conn` arguments. The [API guide](api.md) has the full table; in
short:

- `space` is a fitted masker, a mask image (a nibabel image) or a `Domain`.
- `data` is whatever the masker's `transform` takes (a list of 3-D images, one 4-D
  image, file names), or an `(n, V)` array that is already masked.
- With a fitted masker the images go through `masker.transform`, so the masker's own
  settings (smoothing, standardising, resampling) apply as they do in nilearn.
- With a mask image and images, the images are masked by a `NiftiMasker` built on that
  mask with nilearn's defaults. This is the one step that imports nilearn.
- The `Domain` is built from the mask (`masker.mask_img_`, or the mask image) with
  `conn=6` unless `conn` says 18 or 26. A voxel is in the mask when its value is
  non-zero and finite, as in `NiftiMasker`.
- Column `j` of the masked array is node `j` of the `Domain`: C-order over the mask,
  which is the column order of `NiftiMasker.transform`.

The result is a `MaskedData` with the fields `y`, `domain` and `origin`. `origin` is the
masker or mask image that was given, and the result of an inference keeps it.

`masker.transform` returns the whole `(n, V)` array in memory, and the engine's float64
copy of it comes on top. When that is too much, mask once, save the array with
`np.save`, and pass `np.load(path, mmap_mode="r")` as `data` with the same masker: a
float32 or float64 memory map is read in place.

## Images out: `export_to`

`export_to(result)` returns a dict of four images: `"stat"` (`stat_obs`), `"tfce"`
(`tfce_obs`), `"logp_fwe"` (`-log10(p_fwe)`) and `"logp_unc"` (`-log10(p_unc)`). Every
image is 0 outside the mask. The p-maps are exported as `-log10(p)`, the form of
nilearn's `logp_max_tfce`, so that 0 outside the mask reads as `p = 1`.

- The maps go back through `result.origin`: `masker.inverse_transform` for a masker; for
  a mask image, an image of the same class on the mask's affine and a copy of its
  header, with the data type set to float64.
- `export_to(result, target=masker_or_mask_image)` exports through `target` instead. It
  is needed when the result has no origin: when `space` was a `Domain`, and for a result
  read with `TfceInference.load`, which does not store the origin.

The images are ordinary nibabel images: save them with `to_filename`, or hand them to
nilearn's plotting functions.

## The `"nilearn"` preset

`neurostats.tfce(stat, domain, params="nilearn")` computes the TFCE values of nilearn's
`calculate_tfce` under its default `dh="auto"`:

- 100 thresholds in equal steps from 0 up to and including the map's own maximum,
  rebuilt for every map;
- a voxel belongs to a threshold when its value is `>=` the threshold;
- each cluster adds `size**E * h**H` at each threshold, with no step factor;
- `E = 0.5` and `H = 2`. The dict form `{"weighting": "nilearn", "e": ..., "h": ...}`
  sets other exponents.

The tails map to nilearn's `two_sided_test`:

| `tail` | nilearn call |
|---|---|
| `"two_sided"` | `calculate_tfce(..., two_sided_test=True)` |
| `"positive"` | `calculate_tfce(..., two_sided_test=False)` |
| `"negative"` | `calculate_tfce(...)` on the negated map with `two_sided_test=False` |

In the two-sided case both signs share one threshold grid, built from the largest
absolute value, and the output is signed, as in nilearn.

Connectivity: `conn=6`, `18` and `26` are scipy's
`generate_binary_structure(3, 1)`, `(3, 2)` and `(3, 3)`. nilearn's `permuted_ols` uses
the first, which is the default `conn=6`.

## `tfce_ols` and `permuted_ols`

`neurostats.tfce_ols(y, masker, tested=x, confounds=c, params="nilearn")` fits the model
of nilearn's `permuted_ols(x, y, c, model_intercept=True, masker=masker, tfce=True)` and
uses the same statistic and the same permutation scheme: the rows of the tested
regressor and the confounds are shuffled together, or the signs are flipped when the
intercept is tested.

| nilearn `permuted_ols` | `neurostats.tfce_ols` |
|---|---|
| `tested_vars`, one column | `tested`, a 1-D array |
| a constant `tested_vars` column | `tested="intercept"` |
| `confounding_vars` | `confounds` |
| `model_intercept=True` | always: with a tested regressor the intercept is in the model |
| `two_sided_test=True` | `tail="two_sided"` |
| `two_sided_test=False` | `tail="positive"` |
| the negated regressor with `two_sided_test=False` | `tail="negative"` |
| `masker` | the same masker as `space`; `conn` is 6 unless given |
| `n_perm=B - 1` | `n_perm=B`: the unpermuted data count as one draw |
| `random_state` | `seed`; the draws themselves are not nilearn's |
| output `t` | `stat_obs` |
| output `tfce` | `tfce_obs` |
| output `logp_max_tfce` | `-log10(p_fwe)`; as an image, `export_to(result)["logp_fwe"]` |

The t map and the TFCE map agree to a relative tolerance of 1e-10. The p-values come
from different random draws, so two runs agree only as two Monte Carlo runs do: for
`B` draws each, `|p - p'| <= 4 * sqrt(m * (1 - m) * 2 / B) + 1 / B` with
`m = (p + p') / 2`.

## How this is checked

`NeuroStats/python/tests/test_tfce_parity.py` compares `tfce` and `calculate_tfce` on 1000
random maps and masks, each checked under the three tails (the three connectivities,
three pairs of exponents). The values agree to a relative tolerance of 1e-12.

`NeuroStats/python/tests/test_ols.py::test_matches_nilearn_permuted_ols` compares `tfce_ols`
and `permuted_ols` on five small data sets: a tested regressor and the intercept, with
and without confounds, the three tails. It checks the t map, the TFCE map and, within
the band above, `p_fwe`.

`NeuroStats/python/tests/test_io.py` checks `import_from` and `export_to` against a
`NiftiMasker`: the masked array equals `masker.transform(images)` for a list of 3-D
images, a 4-D image and file names; every exported image equals
`masker.inverse_transform(map)`; and `tfce_ols` gives the same bits from images, from a
`MaskedData` and from the masked array with a `Domain`. One test runs the array paths in
an interpreter where nilearn cannot be imported.

The tests run against the nilearn installed in the test environment; the rules were
read from nilearn 0.14.1.

## Differences

- One tested regressor per call. nilearn takes several columns in `tested_vars`.
- `tfce_ols` has no max-t output (nilearn's `logp_max_t`) and no cluster size or
  cluster mass inference (nilearn's `threshold`).
- A NaN or infinite value inside the domain raises `ValueError`. nilearn computes with
  it.
- Everything is float64. nilearn builds its threshold grid in the dtype of the input, so
  a float32 map can put a voxel on the other side of a threshold there.
- `import neurostats` never imports nilearn, and nilearn is not a dependency.
- Surface maskers and surface images are not taken.
