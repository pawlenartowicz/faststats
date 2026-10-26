import subprocess
import sys

import pytest

# Run in a fresh interpreter where `import nilearn` fails, as it does in an
# environment that does not have it installed.
CODE = """
import sys
sys.modules["nilearn"] = None
import numpy as np
import neurostats as ns

dom = ns.Domain.from_mask(np.ones((3, 3, 2), dtype=bool))
stat = np.arange(18, dtype=float) - 6.0
grid = {"weighting": "mne", "start": 0.0, "step": 0.5}
for params in ("exact", "nilearn", grid):
    for tail in ("positive", "negative", "two_sided"):
        assert ns.tfce(stat, dom, params=params, tail=tail).shape == (18,)
ns.grid_and_weights(3.0, {"weighting": "smith_nichols", "start": 0.0, "step": 0.5})
rng = np.random.default_rng(0)
y = rng.standard_normal((8, 18))
r = ns.tfce_ols(y, dom, tested=rng.standard_normal(8), confounds=rng.standard_normal((8, 1)),
                seed=1, n_perm=20, n_jobs=1)
ns.tfce_ols(y, dom, tested="intercept", params="nilearn", tail="positive", n_perm=20)
r.save(sys.argv[1])
back = ns.TfceInference.load(sys.argv[1])
assert back.n_perm == 20 and back.origin is None
d = ns.import_from(y, dom)
assert d.y is y and d.domain is dom and d.origin is None
assert ns.import_from(d) is d
assert ns.tfce_ols(d, tested="intercept", n_perm=20).origin is None
try:
    ns.export_to(r)
except ValueError:
    pass
else:
    raise AssertionError("a result with no origin was exported without a target")
print("ok")
"""


def test_plain_array_paths_run_without_nilearn(tmp_path):
    r = subprocess.run([sys.executable, "-c", CODE, str(tmp_path / "result.npz")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


# Run in a fresh interpreter where `import nilearn` fails. nibabel alone
# supplies the images.
WITHOUT_NILEARN = """
import sys
sys.modules["nilearn"] = None
import nibabel as nib
import numpy as np
import neurostats as ns

rng = np.random.default_rng(0)
mask = rng.random((4, 3, 3)) < 0.7
mask_img = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
vols = rng.standard_normal((8, 4, 3, 3))
y = vols[:, mask]


class Masker:
    # Not a nilearn class: recognised by its attributes alone.
    mask_img_ = mask_img
    calls = []

    def transform(self, imgs):
        self.calls.append("transform")
        return np.stack([np.asarray(img.dataobj)[mask] for img in imgs])

    def inverse_transform(self, vec):
        self.calls.append("inverse_transform")
        vol = np.zeros(mask.shape)
        vol[mask] = vec
        return nib.Nifti1Image(vol, np.eye(4))


imgs = [nib.Nifti1Image(v, np.eye(4)) for v in vols]
kw = dict(tested=rng.standard_normal(8), seed=1, n_perm=20, n_jobs=1)
want = ns.tfce_ols(y, ns.Domain.from_mask(mask), **kw)
for data, space in ((y, mask_img), (y, Masker()), (imgs, Masker())):
    res = ns.tfce_ols(data, space, **kw)
    assert np.array_equal(res.tfce_obs, want.tfce_obs) and np.array_equal(res.p_fwe, want.p_fwe)
    out = ns.export_to(res)
    assert np.array_equal(np.asarray(out["tfce"].dataobj)[mask], want.tfce_obs)
# Masking and unmasking are the masker's own: one transform for the images,
# one inverse_transform for each of the four maps of an export.
assert Masker.calls == ["inverse_transform"] * 4 + ["transform"] + ["inverse_transform"] * 4
# Masking images through a bare mask image is the one step that needs nilearn.
try:
    ns.import_from(imgs, mask_img)
except ImportError:
    print("ok")
"""


def test_nilearn_is_imported_only_to_mask_images_through_a_mask_image():
    pytest.importorskip("nibabel")
    r = subprocess.run([sys.executable, "-c", WITHOUT_NILEARN], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
