import numpy as np
import pytest

pytest.importorskip("nilearn")

# nilearn splits n_perm into one chunk per job and draws a fresh seed per chunk,
# so its permutation null (and every p-value derived from it) legitimately
# changes with n_jobs. Only the observed-data maps are n_jobs-invariant; the
# rest is checked against nilearn's own run at the same n_jobs.
_OBSERVED_KEYS = ("t", "tfce")


def test_wrapper_outputs_equal_across_n_jobs(monkeypatch):
    from nilearn.maskers import NiftiMasker
    from nilearn.mass_univariate import permuted_ols as host_permuted_ols
    import nibabel as nib
    from faststats.neuro.compat import nilearn as shim

    rng = np.random.default_rng(5)
    n_subj, shape = 12, (6, 6, 5)
    mask_img = nib.Nifti1Image(np.ones(shape, dtype=np.uint8), np.eye(4))
    masker = NiftiMasker(mask_img=mask_img).fit()
    data = rng.standard_normal((n_subj, *shape))
    imgs = nib.Nifti1Image(np.moveaxis(data, 0, -1), np.eye(4))
    target = masker.transform(imgs)
    tested = np.ones((n_subj, 1))
    kw = dict(model_intercept=False, n_perm=40, two_sided_test=True, masker=masker,
              tfce=True, random_state=0, output_type="dict")
    one = shim.permuted_ols(tested, target, n_jobs=1, **kw)
    calls = []
    orig = shim._core.tfce_bands

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(shim._core, "tfce_bands", counting)
    two = shim.permuted_ols(tested, target, n_jobs=2, **kw)
    for key in _OBSERVED_KEYS:
        np.testing.assert_allclose(two[key], one[key], rtol=1e-10, atol=1e-10, err_msg=key)
    want = host_permuted_ols(tested, target, n_jobs=2, **kw)
    for key in want:
        np.testing.assert_allclose(two[key], want[key], rtol=1e-10, atol=1e-10, err_msg=key)
    assert calls, "shim was not called inside nilearn's joblib workers"
