# faststats

Rust-backed statistics for Python. This release ships `faststats.neuro`: threshold-free
cluster enhancement (TFCE) and one-sample sign-flip permutation inference over masked
brain volumes or arbitrary graphs. It is licensed LGPL-3.0-or-later. The `mne` weighting
reproduces MNE-Python 1.12.1 TFCE goldens to a relative tolerance of 1e-12, and the
nilearn compatibility shim is fuzz-tested against nilearn's own `calculate_tfce`.

## Install

```bash
pip install faststats
```

## Quick start

Enhance a single statistic image, keeping its 3-D shape with zeros outside the mask:

```python
import nibabel as nib, numpy as np, faststats.neuro as ns
from nilearn.image import new_img_like

img, mask_img = nib.load("tstat1.nii.gz"), nib.load("mask.nii.gz")
mask = np.asarray(mask_img.dataobj) > 0
dom = ns.Domain.from_mask(mask, conn=26)
t = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float64))
enh = ns.tfce(t, dom, start=0, step=t[mask].max() / 100, weighting="smith_nichols")
nib.save(new_img_like(img, enh), "tstat1_tfce.nii.gz")
```

One-sample inference across subjects, with family-wise error correction:

```python
imgs = [nib.load(f) for f in subject_files]
mask = np.asarray(nib.load("mask.nii.gz").dataobj) > 0
dom = ns.Domain.from_mask(mask, conn=26)
x = np.stack([np.asarray(i.dataobj, dtype=np.float64) for i in imgs])   # (n_subj, nx, ny, nz)
r = ns.tfce_one_sample(x, dom, e=0.5, h=2.0, start=0, step=None, weighting="exact", seed=1,
                       n_perm=5000, n_jobs=-1)
nib.save(new_img_like(imgs[0], r.p_fwe), "p_fwe.nii.gz")
```

## Drop-in nilearn shim

`faststats.neuro.compat.nilearn` re-implements `calculate_tfce`, `permuted_ols`, and
`non_parametric_inference` with the same call signature as nilearn's own functions, so
switching is one import line:

```python
from faststats.neuro.compat.nilearn import permuted_ols      # was: from nilearn.mass_univariate import permuted_ols

out = permuted_ols(tested, target, conf, model_intercept=True, n_perm=500,
                    two_sided_test=True, masker=masker, tfce=True, n_jobs=8,
                    random_state=0, output_type="dict")
```

The wrapper runs the whole nilearn call under joblib's `parallel_config(backend="threading")`,
so `n_jobs` means threads, not processes. The shim needs `nilearn` and `scipy` installed;
they are not wheel dependencies, so either `pip install faststats[test]` or install them
yourself.

## Conventions

| Concept | Values | Notes |
|---|---|---|
| `weighting` | `smith_nichols`, `mne`, `fslmaths`, `exact` | `smith_nichols` = `h**H * step` (FSL/PALM/SPM); `mne` = `|Δh|**H` (MNE-Python); `fslmaths` = `h**H`; `exact` is the closed-form integral, takes no grid and ignores `step` — recommended for new analyses |
| `tail` | `positive`, `negative`, `two_sided`, `two_sided_unsigned` | `two_sided` is signed (`pos - neg`), `two_sided_unsigned` is `pos + neg` |
| `start`, `step`, `weighting` | required keywords of `tfce`/`tfce_one_sample` | no default is chosen for you, because the reference tools disagree on them |
| `n_jobs` (`tfce_one_sample` only) | `-1` (default), or a positive integer | nilearn/scikit-learn convention: `-1` uses every core, `k` uses exactly `k` OS threads, `0` or any other negative value raises `ValueError`. Results are bit-identical for every `n_jobs` |

### Domains

- `Domain.from_mask(mask, conn=26)` — adjacency over the in-mask voxels of a 3-D volume; `stat`/`x` may be passed either as node vectors or in the mask's 3-D shape.
- `Domain.from_volume(shape, conn=26)` — adjacency over every voxel of a 3-D volume (no mask).
- `Domain.from_adjacency(adj, n=None)` — any object with `.tocoo()` (e.g. a scipy sparse matrix); symmetrised, self-loops dropped, duplicate edges merged. `shape` and `mask` are `None`, so only the 1-D node-vector form of `tfce`/`tfce_one_sample` applies.
- `Domain.from_csr(indptr, indices)` — caller-built CSR adjacency; symmetry is the caller's precondition. `shape` and `mask` are `None`, so only the 1-D node-vector form applies.

Node order for a mask-built `Domain` is C-order over the mask (numpy `ravel` order), so
`stat[mask]` is already in node order. `n_perm` includes the identity permutation: it
covers all `2**n_subj` sign patterns when that fits within `n_perm`, otherwise it draws
`n_perm` Monte Carlo samples from `seed`. `p_fwe` is `#{max_b >= obs} / B`.

## Not in 0.1.0

`faststats.common`, `faststats.robust`, NIfTI I/O, GLM contrasts and multi-sample
permutation, and surface meshes.

## Build from source

```bash
pip install maturin
maturin develop
```

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
