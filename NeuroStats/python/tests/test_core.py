import json

import numpy as np
import pytest

import neurostats as ns
from neurostats import _neurostats as core
from neurostats._tfce import INFERENCE_WEIGHTINGS, resolve_params

SMALL = ["grid4", "blob", "ties", "flat", "single", "empty_grid", "two_clusters"]


def load(fixtures_dir, name):
    with open(fixtures_dir / f"tfce_{name}.json") as f:
        return json.load(f)


def fixture_mask(fx):
    return (np.frombuffer(fx["mask"].encode(), dtype=np.uint8) == ord("1")).reshape(fx["dims"])


def grid_params(fx, weighting):
    p = fx["params"]
    return {"weighting": weighting, "e": p["e"], "h": p["h"], "start": p["start"], "step": p["step"]}


def allclose_map(got, want):
    want = np.asarray(want)
    scale = np.max(np.abs(want)) if want.size else 0.0
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * scale)


def test_version_is_exposed():
    assert isinstance(ns.__version__, str) and ns.__version__


def test_domain_from_volume_counts():
    d = core.Domain.from_volume((2, 2, 2), 6)
    assert d.n_nodes == 8
    assert d.n_edges == 8 * 3


def test_conn_defaults_to_6():
    mask = np.ones((2, 2, 2), dtype=bool)
    assert ns.Domain.from_volume((2, 2, 2)).n_edges == ns.Domain.from_volume((2, 2, 2), conn=6).n_edges
    assert ns.Domain.from_mask(mask).n_edges == ns.Domain.from_mask(mask, conn=6).n_edges
    assert ns.Domain.from_volume((2, 2, 2)).n_edges != ns.Domain.from_volume((2, 2, 2), conn=26).n_edges


def test_from_csr_round_trip(fixtures_dir):
    fx = load(fixtures_dir, "blob")
    mask = np.ascontiguousarray(fixture_mask(fx).ravel())
    dom = core.Domain.from_mask(mask, tuple(fx["dims"]), fx["conn"])
    back = core.Domain.from_csr(dom.indptr(), dom.indices())
    assert back.n_nodes == dom.n_nodes and back.n_edges == dom.n_edges
    np.testing.assert_array_equal(back.indptr(), dom.indptr())


def test_errors_are_value_errors():
    dom = core.Domain.from_volume((2, 1, 1), 6)
    with pytest.raises(ValueError, match="length mismatch"):
        core.tfce_bands(dom, np.array([1.0]), np.array([0.5]), np.array([1.0]), 0.5, True)
    with pytest.raises(ValueError, match="NaN"):
        core.tfce_bands(dom, np.array([1.0, np.nan]), np.array([0.5]), np.array([1.0]), 0.5, True)
    with pytest.raises(ValueError, match="conn"):
        core.Domain.from_volume((2, 1, 1), 7)
    with pytest.raises(ValueError, match="weighting"):
        core.tfce(dom, np.zeros(2), 0.5, 2.0, 0.0, 1.0, "bogus", "positive")
    with pytest.raises(ValueError, match="tail"):
        core.tfce(dom, np.zeros(2), 0.5, 2.0, None, None, "exact", "up")


@pytest.mark.parametrize("name", SMALL)
def test_tfce_mne_weighting_reproduces_fixture(fixtures_dir, name):
    fx = load(fixtures_dir, name)
    mask = fixture_mask(fx)
    dom = ns.Domain.from_mask(mask, conn=fx["conn"])
    stat = np.asarray(fx["stat"], dtype=np.float64)
    params = grid_params(fx, "mne")
    got = ns.tfce(stat, dom, params=params, tail="positive")
    allclose_map(got, fx["expected_mne"])
    # 3-D in -> 3-D out, zeros outside the mask, equals the 1-D result scattered.
    vol = np.zeros(fx["dims"])
    vol[mask] = stat
    got3 = ns.tfce(vol, dom, params=params, tail="positive")
    assert got3.shape == tuple(fx["dims"])
    assert np.all(got3[~mask] == 0)
    np.testing.assert_array_equal(got3[mask], got)


@pytest.mark.parametrize("name", SMALL)
def test_tfce_smith_nichols_weighting_reproduces_fixture(fixtures_dir, name):
    fx = load(fixtures_dir, name)
    dom = ns.Domain.from_mask(fixture_mask(fx), conn=fx["conn"])
    stat = np.asarray(fx["stat"], dtype=np.float64)
    got = ns.tfce(stat, dom, params=grid_params(fx, "smith_nichols"), tail="positive")
    allclose_map(got, fx["expected_smith_nichols"])


@pytest.mark.parametrize("weighting", ["smith_nichols", "mne"])
def test_tails_on_the_grid_of_grid_and_weights(fixtures_dir, weighting):
    # `tfce` runs these weightings in the crate. Here the same sum is rebuilt
    # from the numpy grid of `grid_and_weights` and the explicit band operator.
    # The two-sided grid comes from max(|stat|) and both passes share it.
    fx = load(fixtures_dir, "blob")
    dom = ns.Domain.from_mask(fixture_mask(fx), conn=fx["conn"])
    stat = np.random.default_rng(0).standard_normal(dom.n_nodes)
    params = {"weighting": weighting, "start": 0.0, "step": 0.1}

    def bands(vec, stat_max):
        ths, w = ns.grid_and_weights(stat_max, params)
        return core.tfce_bands(dom._core, vec, ths, w, 0.5, True)

    pos = bands(stat, stat.max())
    neg = bands(-stat, (-stat).max())
    two = bands(stat, np.abs(stat).max()) - bands(-stat, np.abs(stat).max())
    for tail, want in (("positive", pos), ("negative", neg), ("two_sided", two)):
        allclose_map(ns.tfce(stat, dom, params=params, tail=tail), want)
    assert np.all(pos >= 0) and np.all(neg >= 0) and np.any(two < 0)


def test_fslmaths_is_smith_nichols_without_the_step_factor(fixtures_dir):
    fx = load(fixtures_dir, "blob")
    dom = ns.Domain.from_mask(fixture_mask(fx), conn=fx["conn"])
    stat = np.random.default_rng(2).standard_normal(dom.n_nodes)
    grid = {"start": 0.0, "step": 0.25}
    for tail in ("positive", "negative", "two_sided"):
        a = ns.tfce(stat, dom, params={"weighting": "fslmaths", **grid}, tail=tail)
        b = ns.tfce(stat, dom, params={"weighting": "smith_nichols", **grid}, tail=tail)
        allclose_map(a, b / 0.25)


def test_exact_preset_and_defaults():
    dom = ns.Domain.from_volume((2, 1, 1))
    # E = 1, H = 0, start = 0: both nodes form a pair on (0, 1], node 0 is
    # alone on (1, 3]: node 0 -> 2*1 + 1*2 = 4, node 1 -> 2*1 = 2.
    got = ns.tfce(np.array([3.0, 1.0]), dom, params={"weighting": "exact", "e": 1.0, "h": 0.0})
    np.testing.assert_allclose(got, [4.0, 2.0], rtol=1e-12)
    # The defaults are params="exact" (e 0.5, h 2, start 0) and tail="two_sided".
    stat = np.array([3.0, -1.0])
    full = {"weighting": "exact", "e": 0.5, "h": 2.0, "start": 0.0}
    np.testing.assert_array_equal(ns.tfce(stat, dom), ns.tfce(stat, dom, params=full, tail="two_sided"))
    np.testing.assert_array_equal(ns.tfce(stat, dom, params="exact"), ns.tfce(stat, dom))
    assert ns.tfce(stat, dom)[1] < 0


def test_nilearn_preset_equals_its_dict_form():
    dom = ns.Domain.from_volume((3, 3, 2))
    stat = np.random.default_rng(3).standard_normal(dom.n_nodes)
    full = {"weighting": "nilearn", "e": 0.5, "h": 2.0}
    for tail in ("positive", "negative", "two_sided"):
        np.testing.assert_array_equal(
            ns.tfce(stat, dom, params="nilearn", tail=tail), ns.tfce(stat, dom, params=full, tail=tail)
        )


def test_resolve_params_returns_the_settings_each_weighting_uses():
    assert resolve_params("exact") == {"weighting": "exact", "e": 0.5, "h": 2.0, "start": 0.0}
    assert resolve_params("nilearn") == {"weighting": "nilearn", "e": 0.5, "h": 2.0}
    assert resolve_params({"weighting": "exact", "start": 1, "h": 1}) == {
        "weighting": "exact", "e": 0.5, "h": 1.0, "start": 1.0}
    assert resolve_params({"weighting": "mne", "start": 0, "step": np.float64(0.1), "e": 1}) == {
        "weighting": "mne", "e": 1.0, "h": 2.0, "start": 0.0, "step": 0.1}
    assert INFERENCE_WEIGHTINGS == ("exact", "nilearn", "mne", "smith_nichols")
    with pytest.raises(ValueError, match="weighting"):
        resolve_params({"weighting": "fslmaths", "start": 0.0, "step": 0.1})


@pytest.mark.parametrize(
    "params, match",
    [
        ("bogus", "preset"),
        ("mne", "preset"),
        (None, "preset"),
        ({"e": 0.5}, "weighting"),
        ({"weighting": "palm"}, "weighting"),
        ({"weighting": "exact", "dh": 0.1}, "unknown key"),
        ({"weighting": "exact", "step": 0.1}, "does not use"),
        ({"weighting": "nilearn", "start": 0.0}, "does not use"),
        ({"weighting": "nilearn", "step": 0.1}, "does not use"),
        ({"weighting": "mne", "start": 0.0}, "needs the key 'step'"),
        ({"weighting": "smith_nichols", "step": 0.1}, "needs the key 'start'"),
        ({"weighting": "fslmaths"}, "needs the key"),
        ({"weighting": "mne", "start": 0.0, "step": 0.0}, "step"),
        ({"weighting": "mne", "start": 0.0, "step": float("nan")}, "step"),
        ({"weighting": "mne", "start": -1.0, "step": 0.1}, "start"),
        ({"weighting": "exact", "h": -1.0}, "h must be"),
        ({"weighting": "exact", "e": "big"}, "e must be a number"),
    ],
)
def test_bad_params_raise_before_stat_is_read(params, match):
    dom = ns.Domain.from_volume((2, 2, 1))
    # `stat` is not an array at all: the params error must come first.
    with pytest.raises(ValueError, match=match):
        ns.tfce(object(), dom, params=params)


def test_grid_and_weights_takes_grid_weightings_only():
    ths, w = ns.grid_and_weights(1.0, {"weighting": "fslmaths", "start": 0.25, "step": 0.25, "h": 2.0})
    np.testing.assert_allclose(ths, [0.25, 0.5, 0.75])
    np.testing.assert_allclose(w, [0.0625, 0.25, 0.5625])
    for params in ("exact", "nilearn", {"weighting": "exact"}, {"weighting": "nilearn"}):
        with pytest.raises(ValueError, match="weighting"):
            ns.grid_and_weights(1.0, params)
    with pytest.raises(TypeError):
        ns.grid_and_weights(1.0)


def test_python_api_errors():
    dom = ns.Domain.from_volume((2, 2, 1))
    with pytest.raises(ValueError, match="shape"):
        ns.tfce(np.zeros(3), dom)
    for tail in ("up", "two_sided_unsigned"):
        with pytest.raises(ValueError, match="tail"):
            ns.tfce(np.zeros(4), dom, tail=tail)
    with pytest.raises(ValueError, match="NaN"):
        ns.tfce(np.array([1.0, np.nan, 0, 0]), dom)
    with pytest.raises(ValueError, match="conn"):
        ns.Domain.from_volume((2, 2, 1), conn=8)


@pytest.mark.parametrize("params", ["exact", "nilearn", {"weighting": "fslmaths", "start": 0.0, "step": 0.5}])
def test_stat_layouts_and_dtypes_give_the_same_map(params):
    rng = np.random.default_rng(4)
    mask = rng.random((4, 3, 5)) < 0.8
    dom = ns.Domain.from_mask(mask)
    vol = np.round(rng.standard_normal(mask.shape) * 4)
    vol[~mask] = np.nan  # outside the mask is never read
    want = ns.tfce(vol[mask], dom, params=params)
    variants = {
        "3-D": vol,
        "3-D Fortran order": np.asfortranarray(vol),
        "float32": vol[mask].astype(np.float32),
        "int64": vol[mask].astype(np.int64),
        "non-contiguous": np.repeat(vol[mask], 2)[::2],
        "list": vol[mask].tolist(),
    }
    for label, stat in variants.items():
        got = ns.tfce(stat, dom, params=params)
        np.testing.assert_array_equal(got[mask] if got.ndim == 3 else got, want, err_msg=label)


@pytest.mark.parametrize(
    "params",
    ["exact", "nilearn", {"weighting": "mne", "start": 0.0, "step": 0.5},
     {"weighting": "fslmaths", "start": 0.0, "step": 0.5}],
)
def test_empty_domain_gives_empty_map(params):
    mask = np.zeros((2, 2, 2), dtype=bool)
    dom = ns.Domain.from_mask(mask)
    for tail in ("positive", "negative", "two_sided"):
        assert ns.tfce(np.zeros(0), dom, params=params, tail=tail).shape == (0,)
        assert np.all(ns.tfce(np.zeros(mask.shape), dom, params=params, tail=tail) == 0)


def test_from_adjacency_lattice_equals_from_volume():
    pytest.importorskip("scipy")
    from scipy import sparse

    shape = (3, 4, 2)
    dom_v = ns.Domain.from_volume(shape, conn=6)
    # Build the 6-connected lattice as a sparse matrix, upper triangle only, to
    # check symmetrisation.
    n = int(np.prod(shape))
    idx = np.arange(n).reshape(shape)
    rows, cols = [], []
    for ax in range(3):
        a = np.moveaxis(idx, ax, 0)
        rows.append(a[:-1].ravel())
        cols.append(a[1:].ravel())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    adj = sparse.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    dom_a = ns.Domain.from_adjacency(adj)
    assert dom_a.n_nodes == dom_v.n_nodes and dom_a.n_edges == dom_v.n_edges
    stat = np.random.default_rng(1).standard_normal(n)
    params = {"weighting": "mne", "start": 0.0, "step": 0.2}
    np.testing.assert_allclose(ns.tfce(stat, dom_a, params=params), ns.tfce(stat, dom_v, params=params),
                               rtol=1e-12)
    assert dom_a.shape is None and dom_a.mask is None



# --- the compiled tfce_ols -------------------------------------------------
# What the public function cannot reach: a row index out of order, and the
# sources the Python layer converts before the call.

EXACT = (0.5, 2.0, 0.0, None, "exact")


def ols_data(n=26, seed=5):
    rng = np.random.default_rng(seed)
    dom = core.Domain.from_volume((4, 3, 3), 6)
    tested = rng.standard_normal(n)
    conf = rng.standard_normal((n, 2))
    y = rng.standard_normal((n, dom.n_nodes)) + 0.8 * tested[:, None] + conf @ rng.standard_normal((2, 36))
    return dom, y, tested, conf


def test_compiled_tfce_ols_takes_the_listed_rows_in_the_listed_order():
    dom, y, tested, conf = ols_data()
    picked = [3, 0, 25] + list(range(7, 24))
    rows = np.array(picked, dtype=np.uint64)
    sub_t, sub_c = tested[:20].copy(), conf[:20].copy()
    args = (*EXACT, "two_sided", 3, 50, 1, None)
    stat_obs, _, _, p_unc, null_max, exceed, n, q = core.tfce_ols(dom, y, rows, sub_t, sub_c, *args)
    # The t of the tested column from one least-squares fit per node, on the
    # picked rows of y in the order listed.
    design = np.column_stack([sub_t, sub_c, np.ones(20)])
    beta, *_ = np.linalg.lstsq(design, y[picked], rcond=None)
    resid = y[picked] - design @ beta
    var = (resid**2).sum(axis=0) / (20 - design.shape[1])
    want = beta[0] / np.sqrt(var * np.linalg.inv(design.T @ design)[0, 0])
    np.testing.assert_allclose(stat_obs, want, rtol=1e-9)
    assert (n, q) == (20, 3)  # nuisance columns: the intercept and two confounds
    assert null_max.shape == (50,) and exceed.dtype == np.uint32
    np.testing.assert_array_equal(p_unc, exceed / 50)
    # No row index is every row, in order.
    every = np.arange(26, dtype=np.uint64)
    for a, b in zip(core.tfce_ols(dom, y, None, tested, conf, *args),
                    core.tfce_ols(dom, y, every, tested, conf, *args)):
        np.testing.assert_array_equal(a, b)


def test_compiled_tfce_ols_refuses_what_the_python_layer_never_passes():
    dom, y, tested, conf = ols_data()
    args = (*EXACT, "two_sided", 3, 20, 1, None)
    for bad in (np.asfortranarray(y), y.astype(np.int64), y[:, ::2], y[0], y.tolist()):
        with pytest.raises(ValueError, match="2-D C-contiguous float32 or float64"):
            core.tfce_ols(dom, bad, None, tested, conf, *args)
    with pytest.raises(ValueError, match="confounds must be C-contiguous"):
        core.tfce_ols(dom, y, None, tested, np.asfortranarray(conf), *args)
    with pytest.raises(ValueError, match="invalid parameters"):
        core.tfce_ols(dom, y, np.array([0, 1, 2, 3, 26], dtype=np.uint64), None, None, *args)
    with pytest.raises(ValueError, match="too few subjects"):
        core.tfce_ols(dom, y, np.array([], dtype=np.uint64), None, None, *args)
