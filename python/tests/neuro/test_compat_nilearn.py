import warnings

import numpy as np
import pytest

pytest.importorskip("nilearn")
from nilearn.mass_univariate import _utils as nl_utils  # noqa: E402
from scipy.ndimage import generate_binary_structure  # noqa: E402

from faststats.neuro.compat import nilearn as shim  # noqa: E402


def _host_grid(arr3d, dh, two_sided):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(nl_utils._return_score_threshs(arr3d, dh=dh, two_sided_test=two_sided))


def _random_case(rng, dtype):
    shape = tuple(rng.integers(3, 8, size=3)) + (int(rng.integers(1, 3)),)
    arr = rng.standard_normal(shape) * rng.uniform(0.5, 4.0)
    mask = rng.random(shape[:3]) < 0.7
    arr[~mask] = 0.0
    return arr.astype(dtype)


def _membership_differs(arr4d, dh, two_sided):
    """Skip a float32 case when the host's grid and the shim's grid disagree
    about which voxels clear a threshold. nilearn builds the grid in the input
    dtype, so for float32 input its interior thresholds are the float32
    roundings of the float64 ones the shim uses; a voxel between the two
    versions of one threshold joins a cluster in one implementation and not the
    other, and no tolerance on the output can absorb that. Grid endpoints are
    exact in both dtypes, so the maximum voxel is never a reason to skip.
    """
    for r in range(arr4d.shape[3]):
        a = arr4d[..., r]
        m = np.nanmax(np.abs(a)) if two_sided else np.nanmax(a)
        if m <= 0:
            return True
        th_host = _host_grid(a, dh, two_sided).astype(np.float64)
        th_ref = _host_grid(a.astype(np.float64), dh, two_sided)
        vals = np.abs(a[a != 0]) if two_sided else a[a > 0]
        if vals.size == 0:
            return True
        v = vals.astype(np.float64)[:, None]
        if np.any((v >= th_host[None, :]) != (v >= th_ref[None, :])):
            return True
    return False


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_calculate_tfce_fuzz(dtype):
    rng = np.random.default_rng(20260909)
    n_cases = 1000 if dtype is np.float64 else 300
    done = 0
    while done < n_cases:
        conn_k = int(rng.integers(1, 4))
        bin_struct = generate_binary_structure(3, conn_k)
        dh = rng.choice(["auto", 0.001, 0.5, 5.0])
        dh = dh if dh == "auto" else float(dh)
        two_sided = bool(rng.integers(0, 2))
        arr = _random_case(rng, dtype)
        if dtype is np.float32 and _membership_differs(arr, dh, two_sided):
            continue
        if not two_sided and any(arr[..., r].max() <= 0 for r in range(arr.shape[3])):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want = nl_utils.calculate_tfce(arr, bin_struct, E=0.5, H=2, dh=dh, two_sided_test=two_sided)
            got = shim.calculate_tfce(arr, bin_struct, E=0.5, H=2, dh=dh, two_sided_test=two_sided)
        assert got.dtype == arr.dtype and got.shape == arr.shape
        scale = float(np.max(np.abs(want))) or 1.0
        if dtype is np.float64:
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * scale)
        else:
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6 * scale)
        done += 1


def test_calculate_tfce_all_zero_map_is_zero():
    arr = np.zeros((4, 4, 4, 1))
    got = shim.calculate_tfce(arr, generate_binary_structure(3, 1))
    assert np.all(got == 0)


def test_clamp_warning_is_attributed_to_the_caller():
    # dh=5.0 on a map whose maximum is ~1 gives round(max/dh) == 0 steps, which
    # trips nilearn's lower clamp. The warning must name this module, not the
    # shim, the way the host's names whoever called calculate_tfce.
    arr = np.zeros((4, 4, 4, 1))
    arr[1:3, 1:3, 1:3, 0] = 1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shim.calculate_tfce(arr, generate_binary_structure(3, 1), dh=5.0)
    steps = [w for w in caught if "Not enough steps for TFCE" in str(w.message)]
    assert len(steps) == 1
    assert steps[0].filename == __file__


def test_all_negative_one_sided_regressor_is_zero():
    # Documented deviation: the host would build a negative threshold grid here;
    # the shim returns zeros.
    arr = -np.abs(np.random.default_rng(11).standard_normal((5, 5, 5, 1))) - 0.5
    got = shim.calculate_tfce(arr, generate_binary_structure(3, 1), two_sided_test=False)
    assert np.all(got == 0)


def test_nan_voxel_is_zero_and_rest_matches_host():
    # Documented deviation: a NaN voxel is set to 0 and so never joins a cluster.
    # The host keeps it NaN, and since `NaN < thresh` is False it survives every
    # thresholding step and scipy's label treats it as foreground. The NaN is
    # therefore isolated here by a shell of zeros: left touching real data it
    # would bridge clusters in the host and the two maps would diverge far from
    # the NaN itself. Isolated, it forms a one-voxel cluster at every threshold
    # whose two signed passes cancel, so the host lands within float noise of the
    # shim's exact 0.
    rng = np.random.default_rng(12)
    arr = rng.standard_normal((7, 7, 7, 1)) * 2.0
    nan_pos = (3, 3, 3, 0)
    arr[2:5, 2:5, 2:5, 0] = 0.0
    arr[nan_pos] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = nl_utils.calculate_tfce(arr, generate_binary_structure(3, 1))
        got = shim.calculate_tfce(arr, generate_binary_structure(3, 1))
    assert got[nan_pos] == 0.0
    scale = float(np.max(np.abs(want)))
    assert abs(want[nan_pos]) < 1e-12 * scale
    keep = np.ones(arr.shape, dtype=bool)
    keep[nan_pos] = False
    np.testing.assert_allclose(got[keep], want[keep], rtol=1e-12, atol=1e-12 * scale)


def test_calculate_tfce_rejects_unknown_structure():
    with pytest.raises(ValueError):
        shim.calculate_tfce(np.zeros((3, 3, 3, 1)), np.zeros((3, 3, 3), dtype=bool))


def test_permuted_ols_end_to_end_matches_nilearn():
    from nilearn.mass_univariate import permuted_ols as host_permuted_ols
    from nilearn.maskers import NiftiMasker
    import nibabel as nib

    rng = np.random.default_rng(3)
    n_subj, shape = 12, (6, 6, 5)
    mask = np.ones(shape, dtype=bool)
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
    masker = NiftiMasker(mask_img=mask_img).fit()
    data = rng.standard_normal((n_subj, *shape))
    data[:, 1:4, 1:4, 1:3] += 1.0
    imgs = nib.Nifti1Image(np.moveaxis(data, 0, -1), np.eye(4))
    target = masker.transform(imgs)
    tested = np.ones((n_subj, 1))
    kw = dict(model_intercept=False, n_perm=50, two_sided_test=True, masker=masker,
              tfce=True, n_jobs=1, random_state=0, output_type="dict")
    want = host_permuted_ols(tested, target, **kw)
    got = shim.permuted_ols(tested, target, **kw)
    for key in want:
        np.testing.assert_allclose(got[key], want[key], rtol=1e-10, atol=1e-10,
                                   err_msg=key)


def test_non_parametric_inference_end_to_end_matches_nilearn():
    from nilearn.glm.second_level import non_parametric_inference as host_npi
    import nibabel as nib
    import pandas as pd

    rng = np.random.default_rng(4)
    n_subj, shape = 10, (6, 6, 5)
    mask_img = nib.Nifti1Image(np.ones(shape, dtype=np.uint8), np.eye(4))
    data = rng.standard_normal((n_subj, *shape))
    data[:, 1:4, 1:4, 1:3] += 1.0
    imgs = [nib.Nifti1Image(data[i], np.eye(4)) for i in range(n_subj)]
    # nilearn 0.14 requires a design matrix when the input is a list of images.
    design = pd.DataFrame({"intercept": np.ones(n_subj)})
    kw = dict(mask=mask_img, design_matrix=design, second_level_contrast="intercept",
              tfce=True, n_perm=30, n_jobs=1, random_state=0)
    want = host_npi(imgs, **kw)
    got = shim.non_parametric_inference(imgs, **kw)
    for key in want:
        np.testing.assert_allclose(got[key].get_fdata(), want[key].get_fdata(),
                                   rtol=1e-10, atol=1e-10, err_msg=key)
