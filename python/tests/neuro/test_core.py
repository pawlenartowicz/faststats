import faststats
from faststats._faststats import neuro as core


def test_version():
    assert faststats.__version__ == "0.1.0"


def test_domain_from_volume_counts():
    d = core.Domain.from_volume((2, 2, 2), 6)
    assert d.n_nodes == 8
    assert d.n_edges == 8 * 3


import json

import numpy as np
import pytest

SMALL = ["grid4", "blob", "ties", "flat", "single", "empty_grid", "two_clusters"]


def load(fixtures_dir, name):
    with open(fixtures_dir / f"tfce_{name}.json") as f:
        return json.load(f)


def fixture_domain(fx):
    mask = np.frombuffer(fx["mask"].encode(), dtype=np.uint8) == ord("1")
    return core.Domain.from_mask(mask, tuple(fx["dims"]), fx["conn"])


def mne_weights(ths, h):
    if ths.size == 0:
        return np.array([])
    dh = np.empty_like(ths)
    dh[0] = abs(ths[0])
    dh[1:] = np.abs(np.diff(ths))
    return dh**h


def allclose_map(got, want):
    want = np.asarray(want)
    scale = np.max(np.abs(want)) if want.size else 0.0
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * scale)


@pytest.mark.parametrize("name", SMALL)
def test_tfce_bands_reproduces_mne_fixture(fixtures_dir, name):
    fx = load(fixtures_dir, name)
    dom = fixture_domain(fx)
    stat = np.asarray(fx["stat"], dtype=np.float64)
    p = fx["params"]
    ths = np.arange(p["start"], stat.max(), p["step"]) if stat.size else np.array([])
    got = core.tfce_bands(dom, stat, ths, mne_weights(ths, p["h"]), p["e"], True)
    allclose_map(got, fx["expected_mne"])


@pytest.mark.parametrize("name", SMALL)
def test_tfce_bands_reproduces_smith_nichols_fixture(fixtures_dir, name):
    fx = load(fixtures_dir, name)
    dom = fixture_domain(fx)
    stat = np.asarray(fx["stat"], dtype=np.float64)
    p = fx["params"]
    ths = np.arange(p["start"], stat.max(), p["step"]) if stat.size else np.array([])
    got = core.tfce_bands(dom, stat, ths, ths ** p["h"] * p["step"], p["e"], True)
    allclose_map(got, fx["expected_smith_nichols"])


def test_from_csr_round_trip(fixtures_dir):
    fx = load(fixtures_dir, "blob")
    dom = fixture_domain(fx)
    back = core.Domain.from_csr(dom.indptr(), dom.indices())
    assert back.n_nodes == dom.n_nodes and back.n_edges == dom.n_edges
    np.testing.assert_array_equal(back.indptr(), dom.indptr())


def test_tfce_exact_hand_case():
    dom = core.Domain.from_volume((2, 1, 1), 6)
    got = core.tfce_exact(dom, np.array([3.0, 1.0]), 1.0, 0.0, 0.0)
    np.testing.assert_allclose(got, [4.0, 2.0], rtol=1e-12)


def test_errors_are_value_errors():
    dom = core.Domain.from_volume((2, 1, 1), 6)
    with pytest.raises(ValueError, match="length mismatch"):
        core.tfce_bands(dom, np.array([1.0]), np.array([0.5]), np.array([1.0]), 0.5, True)
    with pytest.raises(ValueError, match="NaN"):
        core.tfce_bands(dom, np.array([1.0, np.nan]), np.array([0.5]), np.array([1.0]), 0.5, True)
    with pytest.raises(ValueError, match="conn"):
        core.Domain.from_volume((2, 1, 1), 7)
    with pytest.raises(ValueError, match="weighting"):
        core.tfce_one_sample(dom, np.zeros(4), 2, 0.5, 2.0, 0.0, 1.0, "bogus", 1, 8)


def test_tfce_one_sample_matches_crate_doc_example():
    dom = core.Domain.from_volume((3, 1, 1), 6)
    x = np.array([2.0, 0.3, 0.0, 2.5, -0.4, 0.0, 1.8, 0.1, 0.0, 2.2, -0.2, 0.0])
    t_obs, tfce_obs, p_fwe, p_unc, null_max = core.tfce_one_sample(
        dom, x, 4, 0.5, 2.0, 0.0, 1.0, "exact", 42, 1000
    )
    assert null_max.shape == (16,)
    assert np.all(np.diff(null_max) >= 0)
    assert p_fwe[0] <= 2.0 / 16.0 and p_fwe[2] == 1.0
    assert t_obs.shape == tfce_obs.shape == p_unc.shape == (3,)


import faststats.neuro as ns


@pytest.mark.parametrize("name", SMALL)
def test_tfce_mne_weighting_reproduces_fixture(fixtures_dir, name):
    fx = load(fixtures_dir, name)
    mask = (np.frombuffer(fx["mask"].encode(), dtype=np.uint8) == ord("1")).reshape(fx["dims"])
    dom = ns.Domain.from_mask(mask, conn=fx["conn"])
    stat = np.asarray(fx["stat"], dtype=np.float64)
    p = fx["params"]
    got = ns.tfce(stat, dom, e=p["e"], h=p["h"], start=p["start"], step=p["step"], weighting="mne")
    allclose_map(got, fx["expected_mne"])
    # 3-D in -> 3-D out, zeros outside the mask, equals the 1-D result scattered.
    vol = np.zeros(fx["dims"])
    vol[mask] = stat
    got3 = ns.tfce(vol, dom, e=p["e"], h=p["h"], start=p["start"], step=p["step"], weighting="mne")
    assert got3.shape == tuple(fx["dims"])
    assert np.all(got3[~mask] == 0)
    np.testing.assert_array_equal(got3[mask], got)


def test_tail_identities(fixtures_dir):
    fx = load(fixtures_dir, "blob")
    mask = (np.frombuffer(fx["mask"].encode(), dtype=np.uint8) == ord("1")).reshape(fx["dims"])
    dom = ns.Domain.from_mask(mask, conn=fx["conn"])
    rng = np.random.default_rng(0)
    stat = rng.standard_normal(dom.n_nodes)
    kw = dict(e=0.5, h=2.0, start=0.0, step=0.1, weighting="smith_nichols")
    neg = ns.tfce(stat, dom, tail="negative", **kw)
    pos_of_neg = ns.tfce(-stat, dom, tail="positive", **kw)
    np.testing.assert_array_equal(neg, pos_of_neg)
    # two-sided tails share one grid from max(|stat|): rebuild them by hand.
    ths, w = ns.grid_and_weights(np.abs(stat).max(), 0.0, 0.1, 2.0, "smith_nichols")
    p = core.tfce_bands(dom._core, stat, ths, w, 0.5, True)
    n = core.tfce_bands(dom._core, -stat, ths, w, 0.5, True)
    np.testing.assert_array_equal(ns.tfce(stat, dom, tail="two_sided", **kw), p - n)
    np.testing.assert_array_equal(ns.tfce(stat, dom, tail="two_sided_unsigned", **kw), p + n)


def test_exact_weighting_ignores_step():
    dom = ns.Domain.from_volume((2, 1, 1), conn=6)
    a = ns.tfce(np.array([3.0, 1.0]), dom, e=1.0, h=0.0, start=0.0, step=None, weighting="exact")
    b = ns.tfce(np.array([3.0, 1.0]), dom, e=1.0, h=0.0, start=0.0, step=5.0, weighting="exact")
    np.testing.assert_allclose(a, [4.0, 2.0], rtol=1e-12)
    np.testing.assert_array_equal(a, b)


def test_from_adjacency_lattice_equals_from_volume():
    scipy = pytest.importorskip("scipy")
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
    kw = dict(e=0.5, h=2.0, start=0.0, step=0.2, weighting="mne")
    np.testing.assert_allclose(ns.tfce(stat, dom_a, **kw), ns.tfce(stat, dom_v, **kw), rtol=1e-12)
    assert dom_a.shape is None and dom_a.mask is None


def test_tfce_one_sample_python_api(fixtures_dir):
    with open(fixtures_dir / "perm_exact_n6_grid4.json") as f:
        fx = json.load(f)
    mask = (np.frombuffer(fx["mask"].encode(), dtype=np.uint8) == ord("1")).reshape(fx["dims"])
    dom = ns.Domain.from_mask(mask, conn=fx["conn"])
    n = fx["n"]
    x1 = np.asarray(fx["x"], dtype=np.float64).reshape(n, dom.n_nodes)
    p = fx["params"]
    r1 = ns.tfce_one_sample(x1, dom, e=p["e"], h=p["h"], start=p["start"], step=p["step"],
                            weighting="mne", seed=fx["seed"], n_perm=fx["n_permutations"])
    assert r1.null_max.shape == (2**n,)
    allclose_map(r1.t_obs, fx["t_obs"])
    allclose_map(r1.tfce_obs, fx["tfce_obs"])
    # MNE's enumeration differs by at most one draw in the count (crate G5 rule).
    assert np.max(np.abs(r1.p_fwe - np.asarray(fx["p_fwe"]))) <= 1.0 / (2**n) + 1e-15
    # 4-D form: same numbers, 3-D outputs with 0 / 1.0 outside the mask.
    x4 = np.zeros((n, *fx["dims"]))
    x4[:, mask] = x1
    r4 = ns.tfce_one_sample(x4, dom, e=p["e"], h=p["h"], start=p["start"], step=p["step"],
                            weighting="mne", seed=fx["seed"], n_perm=fx["n_permutations"])
    assert r4.p_fwe.shape == tuple(fx["dims"])
    np.testing.assert_array_equal(r4.p_fwe[mask], r1.p_fwe)
    assert np.all(r4.p_fwe[~mask] == 1.0) and np.all(r4.t_obs[~mask] == 0.0)
    np.testing.assert_array_equal(r4.null_max, r1.null_max)


def test_tfce_one_sample_n_jobs_equal_and_validated(fixtures_dir):
    with open(fixtures_dir / "perm_exact_n6_grid4.json") as f:
        fx = json.load(f)
    mask = (np.frombuffer(fx["mask"].encode(), dtype=np.uint8) == ord("1")).reshape(fx["dims"])
    dom = ns.Domain.from_mask(mask, conn=fx["conn"])
    n = fx["n"]
    x1 = np.asarray(fx["x"], dtype=np.float64).reshape(n, dom.n_nodes)
    p = fx["params"]
    kw = dict(e=p["e"], h=p["h"], start=p["start"], step=p["step"], weighting="mne",
              seed=fx["seed"], n_perm=fx["n_permutations"])
    r1 = ns.tfce_one_sample(x1, dom, n_jobs=1, **kw)
    r_all = ns.tfce_one_sample(x1, dom, n_jobs=-1, **kw)
    r3 = ns.tfce_one_sample(x1, dom, n_jobs=3, **kw)
    for other in (r_all, r3):
        for field in ("t_obs", "tfce_obs", "p_fwe", "p_unc", "null_max"):
            np.testing.assert_array_equal(getattr(r1, field), getattr(other, field))
    with pytest.raises(ValueError, match="n_jobs"):
        ns.tfce_one_sample(x1, dom, n_jobs=0, **kw)
    with pytest.raises(ValueError, match="n_jobs"):
        ns.tfce_one_sample(x1, dom, n_jobs=-2, **kw)


def test_python_api_errors():
    dom = ns.Domain.from_volume((2, 2, 1), conn=6)
    with pytest.raises(ValueError, match="shape"):
        ns.tfce(np.zeros(3), dom, start=0.0, step=0.1, weighting="mne")
    with pytest.raises(ValueError, match="step"):
        ns.tfce(np.zeros(4), dom, start=0.0, step=None, weighting="mne")
    with pytest.raises(ValueError, match="tail"):
        ns.tfce(np.zeros(4), dom, start=0.0, step=0.1, weighting="mne", tail="up")
    with pytest.raises(ValueError, match="weighting"):
        ns.tfce(np.zeros(4), dom, start=0.0, step=0.1, weighting="palm")
    with pytest.raises(ValueError, match="NaN"):
        ns.tfce(np.array([1.0, np.nan, 0, 0]), dom, start=0.0, step=0.1, weighting="mne")
    with pytest.raises(ValueError, match="conn"):
        ns.Domain.from_volume((2, 2, 1), conn=8)
