"""`import_from` and `export_to` against nilearn itself: node order against
`NiftiMasker.transform`, exported images against `NiftiMasker.inverse_transform`,
the same `tfce_ols` result from images, from a `MaskedData` and from plain
arrays, every kind of input, and every refusal."""

import dataclasses
import subprocess
import sys

import numpy as np
import pytest

import neurostats as ns

pytest.importorskip("nilearn")
import nibabel as nib  # noqa: E402
from nilearn.maskers import NiftiMasker, SurfaceMasker  # noqa: E402
from nilearn.surface import InMemoryMesh, PolyMesh, SurfaceImage  # noqa: E402

# nilearn 0.14 warns on every `transform` of a masker left at its default
# `standardize=False`; the warning is about a renamed option value.
pytestmark = pytest.mark.filterwarnings("ignore:boolean values for 'standardize':FutureWarning")

MAPS = ("stat_obs", "tfce_obs", "p_fwe", "p_unc", "null_max")
SHAPE = (6, 5, 4)
N = 12


def assert_same(a, b):
    """Two results with the same bits in every array and the same settings."""
    for name in MAPS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert (a.params, a.tail, a.seed, a.n_perm) == (b.params, b.tail, b.seed, b.n_perm)


def data_of(img):
    return np.asarray(img.dataobj)


@dataclasses.dataclass
class Scene:
    mask: np.ndarray  # (6, 5, 4) bool
    affine: np.ndarray
    mask_img: nib.Nifti1Image  # uint8
    masker: NiftiMasker  # fitted on mask_img
    vols: np.ndarray  # (N, 6, 5, 4) float32
    imgs: list  # N 3-D images
    y: np.ndarray  # (N, V) float32, the in-mask voxels in C-order
    tested: np.ndarray


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(20261014)
    mask = rng.random(SHAPE) < 0.7
    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    affine[:3, 3] = [-6.0, 4.0, 1.5]
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
    tested = rng.standard_normal(N)
    vols = rng.standard_normal((N, *SHAPE)).astype(np.float32)
    vols[:, :3] += 0.9 * tested[:, None, None, None].astype(np.float32)
    imgs = [nib.Nifti1Image(v, affine) for v in vols]
    return Scene(mask, affine, mask_img, NiftiMasker(mask_img=mask_img).fit(), vols, imgs,
                 np.ascontiguousarray(vols[:, mask]), tested)


def run(data, space=None, **kw):
    return ns.tfce_ols(data, space, tested=kw.pop("tested"), seed=3, n_perm=40, n_jobs=1, **kw)


# --- import_from -----------------------------------------------------------


def test_node_order_is_the_column_order_of_niftimasker_transform(scene, tmp_path):
    s = scene
    names = []
    for i, img in enumerate(s.imgs):
        names.append(str(tmp_path / f"subject{i:02d}.nii.gz"))
        img.to_filename(names[-1])
    forms = {
        "a list of 3-D images": s.imgs,
        "one 4-D image": nib.Nifti1Image(np.moveaxis(s.vols, 0, -1), s.affine),
        "a list of file names": names,
        "a tuple of paths": tuple(tmp_path / f"subject{i:02d}.nii.gz" for i in range(N)),
    }
    for label, images in forms.items():
        d = ns.import_from(images, s.masker)
        want = s.masker.transform(images)
        np.testing.assert_array_equal(d.y, want, err_msg=label)
        # numpy's own C-order over the mask, stated without nilearn.
        np.testing.assert_array_equal(d.y, s.y, err_msg=label)
        assert d.y.shape == (N, int(s.mask.sum())) and d.origin is s.masker
        assert d.domain.n_nodes == int(s.mask.sum()) == s.masker.n_elements_
        np.testing.assert_array_equal(d.domain.mask, s.mask)


def test_conn_builds_the_domain_and_none_is_6(scene):
    s = scene
    for space in (s.masker, s.mask_img):
        assert ns.import_from(s.y, space).domain.n_edges == ns.Domain.from_mask(s.mask, conn=6).n_edges
        for conn in (6, 18, 26):
            got = ns.import_from(s.y, space, conn=conn).domain
            assert got.n_edges == ns.Domain.from_mask(s.mask, conn=conn).n_edges
    # The connectivity reaches the inference: 26 is not 6.
    six = run(s.imgs, s.masker, tested=s.tested)
    assert_same(six, run(s.imgs, s.masker, tested=s.tested, conn=6))
    assert not np.array_equal(six.tfce_obs, run(s.imgs, s.masker, tested=s.tested, conn=26).tfce_obs)


def test_every_kind_of_space_and_data(scene, tmp_path):
    s = scene
    dom = ns.Domain.from_mask(s.mask)
    # A Domain and an array: both used as they are.
    d = ns.import_from(s.y, dom)
    assert d.y is s.y and d.domain is dom and d.origin is None
    # A masker or a mask image and an array: the array as it is, a memory map
    # included.
    np.save(tmp_path / "y.npy", s.y)
    mapped = np.load(tmp_path / "y.npy", mmap_mode="r")
    for space in (s.masker, s.mask_img):
        d = ns.import_from(s.y, space)
        assert d.y is s.y and d.origin is space
        np.testing.assert_array_equal(d.domain.mask, s.mask)
        assert ns.import_from(mapped, space).y is mapped
    # A mask image and images: what a NiftiMasker built on the mask returns.
    d = ns.import_from(s.imgs, s.mask_img)
    np.testing.assert_array_equal(d.y, NiftiMasker(mask_img=s.mask_img).fit().transform(s.imgs))
    assert d.origin is s.mask_img
    # A MaskedData comes back unchanged.
    assert ns.import_from(d) is d
    # An array in another layout is converted, as with a Domain.
    np.testing.assert_array_equal(ns.import_from(np.asfortranarray(s.y).tolist(), s.masker).y, s.y)


@pytest.mark.filterwarnings("ignore:Non-finite values detected:UserWarning")
def test_a_mask_that_is_not_zeros_and_ones_selects_nilearns_voxels(scene):
    s = scene
    values = np.where(s.mask, 255.0, 0.0).astype(np.float32)
    in_mask = np.argwhere(s.mask)
    values[tuple(in_mask[0])] = np.nan  # nilearn reads a non-finite voxel as outside
    values[tuple(in_mask[1])] = -3.0

    def odd_img():
        # nilearn sets the non-finite voxels of a mask image it is given to 0,
        # in place, so every use gets an image of its own.
        return nib.Nifti1Image(values.copy(), s.affine)

    masker = NiftiMasker(mask_img=odd_img()).fit()
    want = masker.transform(s.imgs)
    assert want.shape[1] == int(s.mask.sum()) - 1
    for space, data in ((odd_img(), s.imgs), (odd_img(), want), (masker, s.imgs)):
        d = ns.import_from(data, space)
        np.testing.assert_array_equal(d.y, want)
        np.testing.assert_array_equal(d.domain.mask, data_of(masker.mask_img_) != 0)


@pytest.mark.filterwarnings("ignore:Non-finite values detected:UserWarning")
def test_masking_through_a_mask_image_leaves_the_callers_image_alone(scene):
    s = scene
    values = np.where(s.mask, 1.0, 0.0).astype(np.float32)
    in_mask = np.argwhere(s.mask)
    values[tuple(in_mask[0])] = np.nan
    values[tuple(in_mask[1])] = np.inf
    mask_img = nib.Nifti1Image(values, s.affine)
    before = data_of(mask_img).copy()
    d = ns.import_from(s.imgs, mask_img)
    np.testing.assert_array_equal(data_of(mask_img), before)
    assert not np.isfinite(data_of(mask_img)).all()
    fresh = ns.import_from(s.imgs, nib.Nifti1Image(values.copy(), s.affine))
    np.testing.assert_array_equal(d.y, fresh.y)
    np.testing.assert_array_equal(d.domain.mask, fresh.domain.mask)


def test_same_result_three_ways(scene):
    s = scene
    kw = dict(tested=s.tested, confounds=np.cos(np.arange(N))[:, None], params="nilearn")
    for space in (s.masker, s.mask_img):
        straight = run(s.imgs, space, **kw)
        masked = ns.import_from(s.imgs, space, conn=6)
        plain = run(masked.y, ns.Domain.from_mask(s.mask, conn=6), **kw)
        assert_same(straight, run(masked, **kw))
        assert_same(straight, plain)
        assert straight.origin is space and run(masked, **kw).origin is space and plain.origin is None
    assert_same(straight, run(s.y, s.masker, **kw))
    # A null continues across the forms: the origin is that of the last call.
    longer = ns.tfce_ols(s.imgs, s.masker, n_perm=70, extend=plain, **kw)
    assert longer.n_perm == 70 and longer.origin is s.masker


# --- what import_from refuses ---------------------------------------------


def surface_masker():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    mesh = PolyMesh(left=InMemoryMesh(coords, faces), right=InMemoryMesh(coords, faces))
    mask = SurfaceImage(mesh, {"left": np.ones(4, dtype=bool), "right": np.ones(4, dtype=bool)})
    return SurfaceMasker(mask_img=mask).fit()


def test_import_refusals(scene):
    s = scene
    dom = ns.Domain.from_mask(s.mask)
    v = dom.n_nodes
    refused = [
        # Images with no space, and with a space that cannot mask them.
        ((s.imgs, None), {}, "images need a space"),
        ((s.imgs[0], None), {}, "images need a space"),
        (("subject.nii.gz", None), {}, "images need a space"),
        ((s.imgs, dom), {}, "images need a fitted masker or a mask image"),
        # A masker that is not fitted, with images and with an array.
        ((s.imgs, NiftiMasker(mask_img=s.mask_img)), {}, "not fitted"),
        ((s.y, NiftiMasker(mask_img=s.mask_img)), {}, "not fitted"),
        # A masker whose mask is not an image.
        ((np.zeros((N, 8)), surface_masker()), {}, "mask_img_ is not an image"),
        # A space of no known kind.
        ((s.y, None), {}, "space must be a Domain, a fitted masker or a mask image"),
        ((s.y, s.mask), {}, "space must be a Domain, a fitted masker or a mask image"),
        ((s.y, "mask.nii.gz"), {}, "space must be a Domain, a fitted masker or a mask image"),
        # An array whose second dimension is not the number of in-mask voxels.
        ((s.y[:, :-1], s.masker), {}, rf"data must have shape \(n, {v}\)"),
        ((s.y[:, :-1], s.mask_img), {}, rf"data must have shape \(n, {v}\)"),
        ((s.vols, s.masker), {}, rf"data must have shape \(n, {v}\)"),
        # One 3-D image is masked to one vector, not to one row per subject.
        ((s.imgs[0], s.masker), {}, rf"data must have shape \(n, {v}\), got \({v},\)"),
        # conn.
        ((s.y, dom), dict(conn=6), "conn cannot be given with a Domain"),
        ((s.y, s.masker), dict(conn=7), "conn must be one of"),
        ((s.imgs, s.mask_img), dict(conn=0), "conn must be one of"),
        # A mask image that is not 3-D.
        ((s.y, nib.Nifti1Image(s.mask.astype(np.uint8)[..., None], s.affine)), {}, "mask must be 3-D"),
    ]
    for (data, space), kw, match in refused:
        with pytest.raises(ValueError, match=match):
            ns.import_from(data, space, **kw)
        # The inference function refuses the same input with the same words.
        with pytest.raises(ValueError, match=match):
            run(data, space, tested=s.tested, **kw)
    # A MaskedData has its Domain: nothing more can be given with it.
    d = ns.import_from(s.y, s.masker)
    for kw in (dict(space=s.masker), dict(space=dom), dict(conn=6)):
        with pytest.raises(ValueError, match="cannot be given with a MaskedData"):
            ns.import_from(d, **kw)
        with pytest.raises(ValueError, match="cannot be given with a MaskedData"):
            ns.tfce_ols(d, tested=s.tested, n_perm=10, **kw)
    # A list of one image is one row.
    assert ns.import_from(s.imgs[:1], s.masker).y.shape == (1, v)


# --- export_to -------------------------------------------------------------

KEYS = ("stat", "tfce", "logp_fwe", "logp_unc")


def test_export_equals_inverse_transform(scene):
    s = scene
    res = run(s.imgs, s.masker, tested=s.tested)
    out = ns.export_to(res)
    assert tuple(out) == KEYS
    vectors = (res.stat_obs, res.tfce_obs, -np.log10(res.p_fwe), -np.log10(res.p_unc))
    for key, vec in zip(KEYS, vectors):
        want = s.masker.inverse_transform(vec)
        np.testing.assert_array_equal(data_of(out[key]), data_of(want), err_msg=key)
        np.testing.assert_array_equal(out[key].affine, s.affine)
        vol = data_of(out[key])
        assert vol.shape == SHAPE and np.all(vol[~s.mask] == 0)
        np.testing.assert_array_equal(vol[s.mask], vec)
        # Masked again, an exported map is the map.
        np.testing.assert_array_equal(s.masker.transform(out[key]), vec)
    # No p-value is 0, so the exported p-maps are finite, and p = 1 reads as 0.
    assert res.p_fwe.max() == 1.0
    for key in ("logp_fwe", "logp_unc"):
        vol = data_of(out[key])
        assert np.all(np.isfinite(vol)) and vol.min() == 0 and vol.max() <= np.log10(res.n_perm)
    # The exported maps come back through import_from as they went out.
    back = ns.import_from([out["stat"], out["tfce"]], s.masker)
    np.testing.assert_array_equal(back.y, np.stack([res.stat_obs, res.tfce_obs]))


def test_an_image_imported_and_exported_is_the_image_inside_the_mask(scene):
    s = scene
    res = run(s.imgs, s.masker, tested=s.tested)
    d = ns.import_from(s.imgs, s.masker)
    for space in (s.masker, s.mask_img):
        carried = dataclasses.replace(res, stat_obs=d.y[4].astype(np.float64), origin=space)
        vol = data_of(ns.export_to(carried)["stat"])
        np.testing.assert_array_equal(vol[s.mask], s.vols[4][s.mask])
        assert np.all(vol[~s.mask] == 0)


def test_export_through_a_mask_image(scene, tmp_path):
    s = scene
    res = run(s.imgs, s.mask_img, tested=s.tested)
    assert res.origin is s.mask_img
    out = ns.export_to(res)
    through_masker = ns.export_to(res, target=s.masker)
    assert tuple(out) == KEYS
    for key in KEYS:
        assert type(out[key]) is nib.Nifti1Image
        np.testing.assert_array_equal(data_of(out[key]), data_of(through_masker[key]), err_msg=key)
        np.testing.assert_array_equal(out[key].affine, s.affine)
        # The header is the mask's, with one change: the maps are float64, so
        # a file holds them as they are and not rounded to the mask's uint8.
        assert out[key].get_data_dtype() == np.float64
        out[key].to_filename(tmp_path / f"{key}.nii.gz")
        np.testing.assert_array_equal(nib.load(tmp_path / f"{key}.nii.gz").get_fdata(),
                                      data_of(out[key]), err_msg=key)
    assert s.mask_img.get_data_dtype() == np.uint8
    # A mask read from a file works the same.
    s.mask_img.to_filename(tmp_path / "mask.nii.gz")
    from_file = ns.export_to(res, target=nib.load(tmp_path / "mask.nii.gz"))
    np.testing.assert_array_equal(data_of(from_file["tfce"]), data_of(out["tfce"]))


def test_target_is_needed_without_an_origin_and_replaces_one(scene, tmp_path):
    s = scene
    dom = ns.Domain.from_mask(s.mask)
    plain = run(s.y, dom, tested=s.tested)
    with_origin = run(s.imgs, s.masker, tested=s.tested)
    assert_same(plain, with_origin)
    with_origin.save(tmp_path / "result.npz")
    loaded = ns.TfceInference.load(tmp_path / "result.npz")
    want = ns.export_to(with_origin)
    for res in (plain, loaded):
        assert res.origin is None
        with pytest.raises(ValueError, match="pass target"):
            ns.export_to(res)
        for target in (s.masker, s.mask_img):
            out = ns.export_to(res, target=target)
            for key in KEYS:
                np.testing.assert_array_equal(data_of(out[key]), data_of(want[key]), err_msg=key)
    # A target replaces the origin: the same voxels on another grid.
    moved = s.affine.copy()
    moved[:3, 3] += 10.0
    for target in (nib.Nifti1Image(s.mask.astype(np.uint8), moved),
                   NiftiMasker(mask_img=nib.Nifti1Image(s.mask.astype(np.uint8), moved)).fit()):
        out = ns.export_to(with_origin, target=target)
        np.testing.assert_array_equal(out["stat"].affine, moved)
        np.testing.assert_array_equal(data_of(out["stat"]), data_of(want["stat"]))


def test_export_refusals(scene):
    s = scene
    res = run(s.imgs, s.masker, tested=s.tested)
    for target, match in [
        (ns.Domain.from_mask(s.mask), "target must be a fitted masker or a mask image, got Domain"),
        (s.mask, "target must be a fitted masker or a mask image, got ndarray"),
        ("mask.nii.gz", "target must be a fitted masker or a mask image, got str"),
        (NiftiMasker(mask_img=s.mask_img), "not fitted"),
        (surface_masker(), "mask_img_ is not an image"),
    ]:
        with pytest.raises(ValueError, match=match):
            ns.export_to(res, target=target)
    # A mask with another number of voxels, as an image and inside a masker.
    smaller = s.mask.copy()
    smaller[tuple(np.argwhere(s.mask)[0])] = False
    smaller_img = nib.Nifti1Image(smaller.astype(np.uint8), s.affine)
    v = int(s.mask.sum())
    with pytest.raises(ValueError, match=f"the result has {v} nodes, the mask of target has {v - 1} voxels"):
        ns.export_to(res, target=smaller_img)
    with pytest.raises(ValueError):
        ns.export_to(res, target=NiftiMasker(mask_img=smaller_img).fit())
    with pytest.raises(ValueError, match="result must be a TfceInference, got object"):
        ns.export_to(object(), target=s.masker)


# --- without nilearn -------------------------------------------------------

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
    r = subprocess.run([sys.executable, "-c", WITHOUT_NILEARN], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
