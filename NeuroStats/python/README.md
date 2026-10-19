# neurostats

Neuroimaging statistics for Python on a Rust core: threshold-free cluster enhancement
(TFCE) and TFCE permutation inference for a linear model fitted at every voxel
(`tfce_ols`, which includes the one-sample test), over masked brain volumes or arbitrary
graphs. It is licensed LGPL-3.0-or-later. The `"mne"` weighting reproduces MNE-Python
1.12.1 TFCE goldens to a relative tolerance of 1e-12, and the `"nilearn"` preset
reproduces nilearn 0.14.1 `calculate_tfce` to the same tolerance on random maps and
masks. Under that preset `tfce_ols` reproduces the t map and the TFCE map of nilearn
`permuted_ols` to a relative tolerance of 1e-10.

## Install

```bash
pip install neurostats
```

The only runtime dependency is numpy. nilearn is not needed to import or use the
package; its maskers and images are taken as they are when you have them.

## Documentation

- [API guide](../../Documentation/NeuroStats/python/api.md) — every function, parameter,
  default, and return field.
- [neurostats and nilearn](../../Documentation/NeuroStats/python/nilearn_io.md) —
  maskers and images in, images out (`import_from`, `export_to`), what the `"nilearn"`
  preset reproduces, and `tfce_ols` next to `permuted_ols`.

## Quick start

Enhance a single statistic image, keeping its 3-D shape with zeros outside the mask:

```python
import nibabel as nib, numpy as np, neurostats as ns
from nilearn.image import new_img_like

img, mask_img = nib.load("tstat1.nii.gz"), nib.load("mask.nii.gz")
mask = np.asarray(mask_img.dataobj) > 0
dom = ns.Domain.from_mask(mask, conn=6)
t = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float64))
enh = ns.tfce(t, dom, params="exact", tail="two_sided")
nib.save(new_img_like(img, enh), "tstat1_tfce.nii.gz")
```

Permutation inference across subjects, with family-wise error correction. A fitted
nilearn masker and the subjects' images go in, and the maps come back as images:

```python
from nilearn.maskers import NiftiMasker

masker = NiftiMasker(mask_img="mask.nii.gz").fit()
d = ns.import_from(subject_files, masker)                  # mask once

# brain ~ age + sex + tiv + intercept, t of age at every voxel
r = ns.tfce_ols(d, tested=age, confounds=np.column_stack([sex, tiv]), n_perm=5000)
ns.export_to(r)["logp_fwe"].to_filename("age_logp_fwe.nii.gz")

# brain ~ intercept: the one-sample test
one = ns.tfce_ols(d, tested="intercept", n_perm=5000)
```

`ns.tfce_ols(subject_files, masker, ...)` masks in the same call. Plain arrays work
too: `data` is then `(n_subjects, n_voxels)` and `space` a `Domain`. Such a result has no
masker to go back through, so the export names one: `ns.export_to(r, target=masker)`.

A result can be saved, loaded, and continued with more draws:

```python
r.save("age.npz")
r = ns.TfceInference.load("age.npz")
r = ns.tfce_ols(d, tested=age, confounds=np.column_stack([sex, tiv]),
                n_perm=20000, extend=r)                    # runs draws 5000 .. 19999 only
```

## Conventions

| Concept | Values | Notes |
|---|---|---|
| `params` | `"exact"` (default), `"nilearn"`, or a dict | A dict has the key `weighting` and any of `e`, `h`, `start`, `step`. `e` and `h` default to 0.5 and 2.0 |
| `weighting` | `exact`, `nilearn`, `mne`, `smith_nichols`, `fslmaths` | `exact` is the closed-form integral from `start` (default 0) and takes no `step`; `nilearn` is 100 equal steps up to the map's maximum with weight `h**H` and takes neither `start` nor `step`; `mne` = `\|Δh\|**H` (MNE-Python), `smith_nichols` = `h**H * step` (FSL/PALM/SPM) and `fslmaths` = `h**H` need both `start` and `step`. `fslmaths` is for `tfce` only |
| `tail` | `positive`, `negative`, `two_sided` (default) | `negative` enhances `-stat`; `two_sided` is signed, the positive enhancement minus the negative one, and its tested quantity is the absolute value |
| `conn` | `6` (default), `18`, `26` | voxel connectivity of `Domain.from_mask` and `Domain.from_volume`, and of the `Domain` that `import_from` and `tfce_ols` build from a masker or a mask image (`conn=None` there is 6) |
| `tested` (`tfce_ols`) | a 1-D array, or `"intercept"` | required keyword. An array tests that regressor with an intercept in the model; `"intercept"` tests the intercept (the one-sample test when there are no confounds). A row whose `tested` or confound value is NaN is left out, with a warning |
| `seed` (`tfce_ols`) | `None` (default), or an integer in `[0, 2**64)` | `None` draws a fresh seed, which the result stores |
| `n_perm` (`tfce_ols`) | default `10000` | the draw budget, the unpermuted data included. `p_fwe` and `p_unc` are never below `1 / n_perm` |
| `n_jobs` (`tfce_ols`) | `1` (default), `-1`, or a positive integer | `-1` uses every core, `k` uses exactly `k` OS threads, `0` or any other negative value raises `ValueError`. Results are bit-identical for every `n_jobs` |

### Domains

- `Domain.from_mask(mask, conn=6)` — adjacency over the in-mask voxels of a 3-D volume; a `stat` map may be passed either as a node vector or in the mask's 3-D shape.
- `Domain.from_volume(shape, conn=6)` — adjacency over every voxel of a 3-D volume (no mask).
- `Domain.from_adjacency(adj, n=None)` — any object with `.tocoo()` (e.g. a scipy sparse matrix); symmetrised, self-loops dropped, duplicate edges merged. `shape` and `mask` are `None`, so only the 1-D node-vector form of `tfce` applies.
- `Domain.from_csr(indptr, indices)` — caller-built CSR adjacency; symmetry is the caller's precondition. `shape` and `mask` are `None`, so only the 1-D node-vector form applies.

Node order for a mask-built `Domain` is C-order over the mask (numpy `ravel` order), so
`stat[mask]` is already in node order, and so are the columns a nilearn `NiftiMasker`
returns.

## Not included

Reading and writing image files (nilearn and nibabel do that), surface maskers and
surface images, more than one tested regressor per call, cluster size and cluster mass
inference, and GLM families other than the linear model.

## Build from source

```bash
pip install maturin
maturin develop
```

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
