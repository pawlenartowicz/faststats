# Using faststats TFCE in a nilearn pipeline

**Status:** draft, 2026-09-09, written ahead of the `faststats` wheel.

## What you get

`faststats.neuro.compat.nilearn` has two functions with the same names and
signatures as nilearn's: `permuted_ols` and `non_parametric_inference`. They
call nilearn, with nilearn's TFCE replaced by the Rust one. Everything else
stays nilearn's: permutation scheme, confound handling, output images, the
`-log10 p` encodings, `random_state` behaviour. Results are the same numbers
(see "Checking equality" below).

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

For the duration of the call, the wrapper binds its own `calculate_tfce`
into nilearn's `permuted_least_squares` module and runs nilearn under joblib's
threading backend. When the call returns, nilearn is exactly as it was.

The threading backend matters. nilearn's default is loky, which starts
separate processes; those would import nilearn fresh and run nilearn's own
TFCE. The Rust TFCE releases the GIL, so threads can run in parallel.

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
tolerance.

## What does not change speed

- `threshold=` (cluster size and cluster mass) still runs nilearn's own code.
  Only TFCE is replaced.
- nilearn rebuilds a full-grid NIfTI image for every permutation before TFCE.
  That step is nilearn's and stays.

## Known differences from nilearn's TFCE

Both are deliberate and both are outside what `permuted_ols` produces:

- NaN voxels score 0. nilearn treats them as foreground. `permuted_ols` maps
  never contain NaN.
- A one-sided map with no positive value returns zeros. nilearn gives positive
  scores to negative voxels in that case.

## Versions

Written against nilearn 0.14.1. The wrapper's tests record the installed
nilearn version; check the release notes for the supported range.
