#!/usr/bin/env python3
"""Generate the one-sample TFCE permutation speed dataset as NIfTI files. Run on
a dev machine:

    python scripts/gen_bench_dataset.py

Writes `data.nii` (4D, subjects along the last axis, float32), `mask.nii`
(uint8) and `tmaps.nii` (4D, `N_MAPS` t-maps along the last axis, float32) into
`target/bench_data/`. The files are regenerated from this script and never
committed. `scripts/run_bench_reference.py` times the reference tools on them.

Mask: sphere in a 50^3 grid (~58k in-mask voxels), the same shape as the
`bench` case of `gen_tfce_golden.py`. Subject i = `EFFECT` * blob + smooth unit
noise, so a one-sample t-map has one real cluster and realistic smoothness.

`tmaps.nii` holds the one-sample t-maps (ddof = 1) of the data under `N_MAPS`
sign-flips of the subjects, map 0 unflipped: what a sign-flip permutation loop
passes to TFCE.

NumPy/SciPy/nibabel are generation-time tools only — never crate or CI
dependencies.
"""
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

OUT = os.path.join(os.path.dirname(__file__), "..", "target", "bench_data")

SEED = 20261014
DIMS = (50, 50, 50)
RADIUS = 24.0
N = 20
SIGMA = 2.5          # noise smoothing, voxels
EFFECT = 0.5         # blob peak, in noise standard deviations
BLOB_CENTRE = (30.0, 24.0, 20.0)
BLOB_WIDTH = 4.0     # Gaussian sd, voxels
N_MAPS = 100


def dataset():
    """Return (data, mask): data is (*DIMS, N) float32, zero outside the mask."""
    rng = np.random.default_rng(SEED)
    g = np.indices(DIMS).astype(float)
    mask = ((g - 24.5) ** 2).sum(0) <= RADIUS ** 2
    blob = np.exp(-sum((g[k] - BLOB_CENTRE[k]) ** 2 for k in range(3)) / (2 * BLOB_WIDTH ** 2))
    data = np.zeros(DIMS + (N,), np.float32)
    for i in range(N):
        noise = gaussian_filter(rng.normal(0, 1, DIMS), sigma=SIGMA)
        noise /= noise[mask].std()
        data[..., i] = np.where(mask, EFFECT * blob + noise, 0.0)
    return data, mask


def tmaps(data, mask):
    """Return the (*DIMS, N_MAPS) float32 sign-flipped t-maps, zero outside the mask."""
    rng = np.random.default_rng(SEED + 1)
    signs = rng.choice([-1.0, 1.0], size=(N_MAPS, N))
    signs[0] = 1.0
    x = data[mask].astype(float)  # (V, N)
    out = np.zeros(DIMS + (N_MAPS,), np.float32)
    for m in range(N_MAPS):
        y = x * signs[m]
        out[mask, m] = y.mean(1) / (y.std(1, ddof=1) / np.sqrt(N))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    data, mask = dataset()
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(data, affine), os.path.join(OUT, "data.nii"))
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), os.path.join(OUT, "mask.nii"))
    nib.save(nib.Nifti1Image(tmaps(data, mask), affine), os.path.join(OUT, "tmaps.nii"))
    print(f"dims={DIMS} n={N} in-mask voxels={int(mask.sum())} t-maps={N_MAPS} seed={SEED} "
          f"-> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
