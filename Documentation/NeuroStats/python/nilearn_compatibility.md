# Using faststats TFCE in a nilearn pipeline

## What you get

`faststats.neuro.compat.nilearn` has three functions with the same names and
signatures as nilearn's: `calculate_tfce`, `permuted_ols`, and
`non_parametric_inference`. `calculate_tfce` is a transcription of nilearn's own
`calculate_tfce` (from nilearn 0.14.1) computed with the Rust TFCE core.
`permuted_ols` and `non_parametric_inference` call nilearn, with nilearn's TFCE
replaced by this one for the duration of the call. Everything else stays nilearn's:
permutation scheme, confound handling, output images, the `-log10 p` encodings,
`random_state` behaviour. Results are the same numbers (see "Checking equality"
below).

## Install

```
pip install faststats nilearn
```

nilearn is not pulled in by `faststats`; install it yourself, as you already
do. `import faststats.neuro` never imports nilearn.

## Change one import line

Before:

```python
from nilearn.mass_univariate import permuted_ols
```

After:

```python
from faststats.neuro.compat.nilearn import permuted_ols
```

The call stays as it was:

```python
out = permuted_ols(tested, target, conf, model_intercept=True, n_perm=500,
                   two_sided_test=True, masker=masker, tfce=True, n_jobs=8,
                   random_state=0, output_type="dict")
```

Same for the image-level front end:

```python
from faststats.neuro.compat.nilearn import non_parametric_inference
out = non_parametric_inference(second_level_input, mask=mask_img,
                               tfce=True, n_perm=5000, n_jobs=8)
```

To go back, change the import line back.

## What happens inside

For the duration of the call, `_patched()` rebinds `calculate_tfce` on
`nilearn.mass_univariate.permuted_least_squares` — the module `permuted_ols`'s
permutation loop imports the name from, so both the observed-data call and every
per-chunk permutation resolve the substitute — and forces joblib's threading
backend with `joblib.parallel_config(backend="threading")`. A `threading.Lock`
guards a small cache of `Domain` objects keyed by `(shape, conn)`, so concurrent
threads share one `Domain` per shape/connectivity pair instead of rebuilding it.
On exit, the original `calculate_tfce` is restored, whether or not the call raised.

The threading backend matters. nilearn's default is loky, which starts separate
processes; those would import nilearn fresh and run nilearn's own TFCE. The Rust
TFCE releases the GIL, so threads can run in parallel.

## Checking equality

Run the same call both ways with the same `random_state` and `n_jobs=1`,
then compare:

```python
import numpy as np
from nilearn.mass_univariate import permuted_ols as nl_permuted_ols
from faststats.neuro.compat.nilearn import permuted_ols as fs_permuted_ols

kw = dict(model_intercept=True, n_perm=50, two_sided_test=True, masker=masker,
          tfce=True, n_jobs=1, random_state=0, output_type="dict")
a = nl_permuted_ols(tested, target, conf, **kw)
b = fs_permuted_ols(tested, target, conf, **kw)
assert np.allclose(a["logp_max_tfce"], b["logp_max_tfce"])
```

`permuted_ols` passes float64 maps to TFCE, so the two agree to floating-point
tolerance (`tests/neuro/test_compat_nilearn.py::test_permuted_ols_end_to_end_matches_nilearn`
checks every output key to `rtol=1e-10`).

## What does not change speed

- `threshold=` (cluster size and cluster mass) still runs nilearn's own code.
  Only TFCE is replaced.
- nilearn rebuilds a full-grid NIfTI image for every permutation before TFCE.
  That step is nilearn's and stays.

## Known differences from nilearn's TFCE

- NaN voxels score 0 and never join a cluster; nilearn keeps them NaN, which
  survives thresholding and is treated as foreground by its cluster labelling.
  Isolated NaN voxels land within float noise of 0 in both implementations
  (`test_nan_voxel_is_zero_and_rest_matches_host`).
- A one-sided regressor whose grid maximum is `<= 0` returns zeros; nilearn
  clusters the negative grid instead and gives those voxels positive scores
  (`test_all_negative_one_sided_regressor_is_zero`).
- A map containing `+inf` or `-inf` raises `ValueError` here, because the band
  weight `h**H` overflows; nilearn produces an all-`inf` map instead (source:
  `faststats/python/python/faststats/neuro/compat/nilearn.py` module docstring;
  no dedicated test in `test_compat_nilearn.py`).
- Float32 input can disagree with nilearn's grid membership at a threshold: nilearn
  builds its threshold grid in the input dtype, so for float32 arrays its interior
  thresholds are float32 roundings of the float64 ones this shim uses. A voxel that
  falls between the two versions of one threshold joins a cluster in one
  implementation and not the other, and no tolerance on the output absorbs that
  (`_membership_differs`, used by `test_calculate_tfce_fuzz` to skip such cases
  rather than assert equality on them). This only affects float32 input; float64
  input matches to `rtol=1e-12`.

All of the above are outside what `permuted_ols`/`non_parametric_inference` actually
produce for real fMRI/statistic maps: those pass float64, non-NaN, two-sided TFCE
input, so none of the four differences applies to them in practice.

## Versions

`calculate_tfce` is transcribed from nilearn 0.14.1
(`faststats/python/python/faststats/neuro/compat/nilearn.py` module docstring).
`test_compat_nilearn.py` does not pin or record a nilearn version; it imports and
runs against whatever nilearn is installed in the test environment, including the
end-to-end tests that call the real `nilearn.mass_univariate.permuted_ols` and
`nilearn.glm.second_level.non_parametric_inference`.
