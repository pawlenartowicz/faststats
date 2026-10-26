"""`tfce_ols`: the result against its stated rules, parity with nilearn
`permuted_ols`, and `n_jobs`, `extend`, `save`/`load` (with a file saved by
0.2.0), the source layouts, the rows left out for a NaN and the error paths."""

from pathlib import Path

import numpy as np
import pytest

import neurostats as ns
from neurostats import _arrays

MAPS = ("stat_obs", "tfce_obs", "p_fwe", "p_unc", "null_max")
TAILS = ("positive", "negative", "two_sided")


def make(seed=0, n=20, shape=(5, 4, 3), n_conf=2):
    """A domain, brain data with an effect of `tested` and of the confounds in
    the first nodes, the tested regressor and the confounds."""
    rng = np.random.default_rng(seed)
    dom = ns.Domain.from_volume(shape)
    tested = rng.standard_normal(n)
    conf = rng.standard_normal((n, n_conf))
    y = rng.standard_normal((n, dom.n_nodes))
    y[:, :8] += 0.8 * tested[:, None] + 0.5
    y += conf @ rng.standard_normal((n_conf, dom.n_nodes))
    return dom, y, tested, conf


def assert_same(a, b):
    """Two results with the same bits in every array and the same settings."""
    for name in MAPS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert (a.params, a.tail, a.seed, a.n_perm) == (b.params, b.tail, b.seed, b.n_perm)


# --- the statistic and the result ------------------------------------------


@pytest.mark.parametrize("tail", TAILS)
@pytest.mark.parametrize("params", ["exact", {"weighting": "mne", "start": 0.0, "step": 0.3}])
def test_result_follows_the_stated_rules(tail, params):
    dom, y, tested, conf = make()
    b = 120
    r = ns.tfce_ols(y, dom, tested=tested, confounds=conf, params=params, tail=tail,
                    seed=5, n_perm=b, n_jobs=1)
    assert r.n_perm == b and r.null_max.shape == (b,) and r.seed == 5 and r.tail == tail
    assert r.origin is None
    assert r.params == ({"weighting": "exact", "e": 0.5, "h": 2.0, "start": 0.0} if params == "exact"
                        else {"weighting": "mne", "e": 0.5, "h": 2.0, "start": 0.0, "step": 0.3})
    for name in MAPS[:4]:
        assert getattr(r, name).shape == (dom.n_nodes,) and getattr(r, name).dtype == np.float64
    # tfce_obs is the plain TFCE of stat_obs under the same tail and settings.
    np.testing.assert_array_equal(r.tfce_obs, ns.tfce(r.stat_obs, dom, params=params, tail=tail))
    # The tested quantity, the null of maxima and the two counting rules.
    t = np.abs(r.tfce_obs) if tail == "two_sided" else r.tfce_obs
    assert np.all(np.diff(r.null_max) >= 0)
    assert t.max() in r.null_max  # draw 0 is the data as observed
    np.testing.assert_array_equal(r.p_fwe, (r.null_max[None, :] >= t[:, None]).sum(axis=1) / b)
    counts = r.p_unc * b
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-9)
    assert np.all(r.p_unc >= 1 / b) and np.all(r.p_unc <= r.p_fwe)


def test_seed_none_draws_a_seed_and_stores_it():
    dom, y, tested, conf = make()
    kw = dict(tested=tested, confounds=conf, n_perm=40, n_jobs=1)
    a = ns.tfce_ols(y, dom, **kw)
    b = ns.tfce_ols(y, dom, **kw)
    assert isinstance(a.seed, int) and 0 <= a.seed < 2**64
    assert a.seed != b.seed
    assert_same(a, ns.tfce_ols(y, dom, seed=a.seed, **kw))


def test_every_n_jobs_gives_the_same_bits():
    dom, y, tested, conf = make()
    kw = dict(tested=tested, confounds=conf, seed=7, n_perm=150)
    one = ns.tfce_ols(y, dom, n_jobs=1, **kw)
    for n_jobs in (2, 3, 7, -1):
        assert_same(one, ns.tfce_ols(y, dom, n_jobs=n_jobs, **kw))


# --- extend, save and load -------------------------------------------------


@pytest.mark.parametrize("intercept", [False, True])
def test_extend_equals_one_long_run(intercept):
    dom, y, tested, conf = make()
    kw = dict(tested="intercept" if intercept else tested, confounds=conf, params="nilearn")
    long = ns.tfce_ols(y, dom, seed=11, n_perm=130, n_jobs=1, **kw)
    short = ns.tfce_ols(y, dom, seed=11, n_perm=40, n_jobs=1, **kw)
    assert short.n_perm == 40
    # seed=None takes the stored seed; n_perm is the new total.
    ext = ns.tfce_ols(y, dom, n_perm=130, n_jobs=3, extend=short, **kw)
    assert ext.seed == 11 and ext.n_perm == 130
    assert_same(ext, long)
    # In two steps, the first of one draw, with the seed given again.
    mid = ns.tfce_ols(y, dom, seed=11, n_perm=41, n_jobs=1, extend=short, **kw)
    assert mid.n_perm == 41
    assert_same(ns.tfce_ols(y, dom, seed=11, n_perm=130, extend=mid, **kw), long)
    # The stored result is not changed by the call.
    assert short.n_perm == 40 and short.null_max.shape == (40,)


def test_save_and_load_round_trip_then_extend(tmp_path):
    dom, y, tested, conf = make()
    seed = 2**64 - 3
    params = {"weighting": "smith_nichols", "start": 0.1, "step": 0.25, "e": 0.6, "h": 1.5}
    kw = dict(tested=tested, confounds=conf, tail="negative", params=params)
    short = ns.tfce_ols(y, dom, seed=seed, n_perm=30, n_jobs=1, **kw)
    path = tmp_path / "null.result"
    short.save(path)
    assert [p.name for p in tmp_path.iterdir()] == ["null.result"]
    back = ns.TfceInference.load(path)
    assert_same(back, short)
    assert back.origin is None and back.seed == seed
    # What extend needs came back too: the loaded result continues to the
    # bits of one long run.
    long = ns.tfce_ols(y, dom, seed=seed, n_perm=90, n_jobs=1, **kw)
    assert_same(ns.tfce_ols(y, dom, n_perm=90, extend=back, **kw), long)
    # A string path works as well, and a loaded result saves again.
    back.save(str(tmp_path / "again.npz"))
    assert_same(ns.TfceInference.load(str(tmp_path / "again.npz")), short)


def test_load_refuses_a_file_that_is_not_a_saved_result(tmp_path):
    np.savez(tmp_path / "other.npz", stat_obs=np.zeros(3))
    with pytest.raises(ValueError, match="not a saved result"):
        ns.TfceInference.load(tmp_path / "other.npz")
    np.save(tmp_path / "plain.npy", np.zeros(3))
    with pytest.raises(ValueError, match="not a saved result"):
        ns.TfceInference.load(tmp_path / "plain.npy")
    (tmp_path / "text.npz").write_text("not an archive")
    with pytest.raises(ValueError):
        ns.TfceInference.load(tmp_path / "text.npz")
    # Every array present, but the meta is not the dict `save` writes.
    for meta in ("{}", "[]"):
        np.savez(tmp_path / "meta.npz", meta=np.array(meta), exceed=np.zeros(3),
                 **{k: np.zeros(3) for k in MAPS})
        with pytest.raises(ValueError, match="not a saved result"):
            ns.TfceInference.load(tmp_path / "meta.npz")


# A result written by `TfceInference.save` of neurostats 0.2.0. It pins the
# on-disk format, which every other save/load test writes and reads with the
# same code. Made from `dom, y, tested, conf = make()` with
# `ns.tfce_ols(y, dom, tested=tested, confounds=conf, seed=SAVED_SEED,
# n_perm=20, n_jobs=1, **SAVED_KW).save(path)` (4.8 KB). Regenerate it
# that way only for an intended format change.
SAVED_0_2_0 = Path(__file__).parent / "data" / "tfce_inference_0.2.0.npz"
SAVED_SEED = 2**64 - 3
SAVED_KW = dict(tail="two_sided", params={"weighting": "mne", "start": 0.0, "step": 0.3})


def test_a_result_saved_by_0_2_0_loads_and_extends():
    dom, y, tested, conf = make()
    kw = dict(tested=tested, confounds=conf, n_jobs=1, **SAVED_KW)
    old = ns.TfceInference.load(SAVED_0_2_0)
    assert_same(old, ns.tfce_ols(y, dom, seed=SAVED_SEED, n_perm=20, **kw))
    assert_same(ns.tfce_ols(y, dom, n_perm=40, extend=old, **kw),
                ns.tfce_ols(y, dom, seed=SAVED_SEED, n_perm=40, **kw))


def test_extend_refusals():
    dom, y, tested, conf = make()
    kw = dict(tested=tested, confounds=conf)
    stored = ns.tfce_ols(y, dom, seed=11, n_perm=40, n_jobs=1, **kw)
    # The budget is the total, and the message states the stored number.
    for n_perm in (40, 39, 1):
        with pytest.raises(ValueError, match="already has 40 draws"):
            ns.tfce_ols(y, dom, n_perm=n_perm, extend=stored, **kw)
    with pytest.raises(ValueError, match="seed 12 is not the seed of the stored null"):
        ns.tfce_ols(y, dom, seed=12, n_perm=80, extend=stored, **kw)
    with pytest.raises(ValueError, match="tail 'positive' is not the tail of the stored null"):
        ns.tfce_ols(y, dom, tail="positive", n_perm=80, extend=stored, **kw)
    for params in ("nilearn", {"weighting": "exact", "e": 0.6}, {"weighting": "exact", "start": 0.5}):
        with pytest.raises(ValueError, match="are not the settings of the stored null"):
            ns.tfce_ols(y, dom, params=params, n_perm=80, extend=stored, **kw)
    with pytest.raises(ValueError, match="extend must be a TfceInference"):
        ns.tfce_ols(y, dom, n_perm=80, extend="null.npz", **kw)
    # All of those are settings: they are refused before data is read.
    with pytest.raises(ValueError, match="already has 40 draws"):
        ns.tfce_ols(object(), dom, n_perm=40, extend=stored, **kw)
    # Other data, another design, another number of subjects or nodes.
    y2 = y.copy()
    y2[0, 0] += 1e-9
    cannot = "cannot be continued"
    with pytest.raises(ValueError, match=cannot):
        ns.tfce_ols(y2, dom, n_perm=80, extend=stored, **kw)
    with pytest.raises(ValueError, match=cannot):
        ns.tfce_ols(y, dom, tested=tested, confounds=conf[:, :1], n_perm=80, extend=stored)
    with pytest.raises(ValueError, match=cannot):
        ns.tfce_ols(y, dom, tested=tested[::-1].copy(), confounds=conf, n_perm=80, extend=stored)
    with pytest.raises(ValueError, match=cannot):
        ns.tfce_ols(y, dom, tested="intercept", confounds=conf, n_perm=80, extend=stored)
    with pytest.raises(ValueError, match=cannot):
        ns.tfce_ols(y[:-1], dom, tested=tested[:-1], confounds=conf[:-1], n_perm=80, extend=stored)
    small = ns.Domain.from_volume((5, 4, 2))
    with pytest.raises(ValueError, match=cannot):
        ns.tfce_ols(y[:, : small.n_nodes], small, n_perm=80, extend=stored, **kw)
    # Nothing above changed the stored result: it still extends.
    assert ns.tfce_ols(y, dom, n_perm=41, extend=stored, **kw).n_perm == 41


def test_extend_refuses_a_budget_that_enumerates_every_sign_pattern():
    rng = np.random.default_rng(3)
    dom = ns.Domain.from_volume((3, 2, 2))
    x = rng.standard_normal((7, dom.n_nodes)) + 0.4  # 2**7 = 128 sign patterns
    kw = dict(tested="intercept", tail="positive")
    stored = ns.tfce_ols(x, dom, seed=5, n_perm=100, **kw)
    assert stored.n_perm == 100
    assert ns.tfce_ols(x, dom, n_perm=127, extend=stored, **kw).n_perm == 127
    for n_perm in (128, 5000):
        with pytest.raises(ValueError, match="enumerate every sign pattern"):
            ns.tfce_ols(x, dom, n_perm=n_perm, extend=stored, **kw)
    # An enumerated null is complete: no budget continues it.
    full = ns.tfce_ols(x, dom, seed=5, n_perm=128, **kw)
    assert full.n_perm == 128
    with pytest.raises(ValueError, match="enumerate every sign pattern"):
        ns.tfce_ols(x, dom, n_perm=1000, extend=full, **kw)
    with pytest.raises(ValueError, match="already has 128 draws"):
        ns.tfce_ols(x, dom, n_perm=100, extend=full, **kw)
    # A budget above 2**n enumerates as well: n_perm is the number of draws used.
    assert ns.tfce_ols(x, dom, seed=5, n_perm=1000, **kw).n_perm == 128


# --- the source ------------------------------------------------------------


def test_float32_source_equals_its_float64_copy(tmp_path):
    dom, y, tested, conf = make()
    kw = dict(tested=tested, confounds=conf, seed=8, n_perm=60, n_jobs=1)
    y32 = y.astype(np.float32)
    kept = y32.copy()
    want = ns.tfce_ols(y32.astype(np.float64), dom, **kw)
    assert_same(ns.tfce_ols(y32, dom, **kw), want)
    np.testing.assert_array_equal(y32, kept)
    # The same bytes read through a read-only memory map.
    y32.tofile(tmp_path / "y.f32")
    mapped = np.memmap(tmp_path / "y.f32", dtype=np.float32, mode="r", shape=y32.shape)
    assert_same(ns.tfce_ols(mapped, dom, **kw), want)
    # No copy is made on the way: such a source is handed over as the object
    # it is. Nothing public shows a copy, so this is checked on the helper.
    assert _arrays.source(y32, dom.n_nodes) is y32
    assert _arrays.source(mapped, dom.n_nodes) is mapped


def test_other_layouts_and_dtypes_give_the_same_result():
    dom, y, tested, conf = make()
    y = np.round(y * 8)
    kw = dict(seed=9, n_perm=40, n_jobs=1)
    want = ns.tfce_ols(y, dom, tested=tested, confounds=conf, **kw)
    sources = {
        "Fortran order": np.asfortranarray(y),
        "int64": y.astype(np.int64),
        "big-endian float32": y.astype(">f4"),
        "non-contiguous": np.repeat(y, 2, axis=1)[:, ::2],
        "nested list": y.tolist(),
    }
    for label, data in sources.items():
        assert_same(ns.tfce_ols(data, dom, tested=tested, confounds=conf, **kw), want)
    designs = {
        "lists": (tested.tolist(), conf.tolist()),
        "Fortran order": (tested, np.asfortranarray(conf)),
        "non-contiguous": (np.repeat(tested, 2)[::2], np.repeat(conf, 2, axis=0)[::2]),
    }
    for label, (t, c) in designs.items():
        assert_same(ns.tfce_ols(y, dom, tested=t, confounds=c, **kw), want)


def test_a_domain_with_no_nodes_gives_empty_maps():
    empty = ns.Domain.from_mask(np.zeros((2, 2, 2), dtype=bool))
    r = ns.tfce_ols(np.zeros((6, 0)), empty, tested=np.arange(6.0), seed=1, n_perm=10)
    for name in MAPS[:4]:
        assert getattr(r, name).shape == (0,)
    assert r.n_perm == 10 and np.all(r.null_max == 0)


# --- rows with a NaN -------------------------------------------------------


@pytest.mark.parametrize("intercept", [False, True])
def test_rows_with_a_nan_in_tested_or_confounds_are_left_out(intercept):
    dom, y, tested, conf = make(n=24)
    t, c = tested.copy(), conf.copy()
    t[3] = np.nan
    c[5, 0] = c[17, 1] = np.nan
    # With tested="intercept" only the confounds count.
    left_out = [5, 17] if intercept else [3, 5, 17]
    keep = np.setdiff1d(np.arange(24), left_out)
    design = "intercept" if intercept else t
    kw = dict(seed=4, n_jobs=1)

    def by_hand(n_perm):
        sub = "intercept" if intercept else tested[keep]
        return ns.tfce_ols(y[keep], dom, tested=sub, confounds=conf[keep], n_perm=n_perm, **kw)

    warned = f"{len(left_out)} of 24 rows have a NaN in tested or confounds"
    with pytest.warns(UserWarning, match=warned):
        got = ns.tfce_ols(y, dom, tested=design, confounds=c, n_perm=60, **kw)
    assert_same(got, by_hand(60))
    # extend finds the same rows in the same arrays and continues the null.
    with pytest.warns(UserWarning, match=warned):
        assert_same(ns.tfce_ols(y, dom, tested=design, confounds=c, n_perm=90, extend=got, **kw),
                    by_hand(90))
    # The arrays without the NaN are another design: every row is in it.
    with pytest.raises(ValueError, match="cannot be continued"):
        ns.tfce_ols(y, dom, tested="intercept" if intercept else tested, confounds=conf,
                    n_perm=90, extend=got, **kw)


# --- errors ----------------------------------------------------------------


def test_settings_are_checked_before_data_is_read():
    dom = ns.Domain.from_volume((2, 2, 1))
    unread = object()  # not an array: any error about it means data was read
    good = dict(tested=np.zeros(5), seed=1, n_perm=10)
    bad = [
        (dict(params="bogus"), "preset"),
        (dict(params={"weighting": "fslmaths", "start": 0.0, "step": 0.1}), "weighting"),
        (dict(tail="two_sided_unsigned"), "tail"),
        (dict(tested="slope"), "tested"),
        (dict(tested=None), "tested"),
        (dict(seed=-1), "seed"),
        (dict(seed=2**64), "seed"),
        (dict(n_perm=0), "n_perm"),
        (dict(n_perm=2**32), "n_perm"),
        (dict(n_jobs=0), "n_jobs"),
        (dict(n_jobs=-2), "n_jobs"),
        (dict(n_perm=None), "must be integers"),
        (dict(n_jobs=None), "must be integers"),
        (dict(seed=[1]), "must be integers"),
    ]
    for change, match in bad:
        with pytest.raises(ValueError, match=match):
            ns.tfce_ols(unread, dom, **{**good, **change})
    with pytest.raises(TypeError):
        ns.tfce_ols(unread, dom)  # tested is required
    with pytest.raises(TypeError):
        ns.tfce_ols(unread, dom, np.zeros(5))  # and keyword-only


def test_bad_data_space_and_conn_are_value_errors():
    dom, y, tested, conf = make()
    kw = dict(tested=tested, seed=1, n_perm=10)
    with pytest.raises(ValueError, match="space must be a Domain"):
        ns.tfce_ols(y, **kw)
    with pytest.raises(ValueError, match="space must be a Domain"):
        ns.tfce_ols(y, np.ones(dom.shape, dtype=bool), **kw)
    with pytest.raises(ValueError, match="conn cannot be given with a Domain"):
        ns.tfce_ols(y, dom, conn=6, **kw)
    # One column short, 1-D, one volume per subject, and not an array at all.
    for data in (y[:, :-1], y[0], y.reshape(y.shape[0], *dom.shape), object()):
        with pytest.raises(ValueError, match="data must have shape"):
            ns.tfce_ols(data, dom, **kw)
    y_nan = y.astype(np.float32)
    y_nan[3, 5] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        ns.tfce_ols(y_nan, dom, **kw)


def test_bad_designs_are_value_errors():
    dom, y, tested, conf = make()
    n = len(tested)
    kw = dict(seed=1, n_perm=10)
    with pytest.raises(ValueError, match="tested must be 1-D"):
        ns.tfce_ols(y, dom, tested=tested.reshape(-1, 1), **kw)
    with pytest.raises(ValueError, match=f"tested has {n - 1} values, the model has {n} subjects"):
        ns.tfce_ols(y, dom, tested=tested[:-1], **kw)
    with pytest.raises(ValueError, match="confounds must be 2-D"):
        ns.tfce_ols(y, dom, tested=tested, confounds=conf[:, 0], **kw)
    with pytest.raises(ValueError, match=f"confounds has {n - 1} rows, the model has {n} subjects"):
        ns.tfce_ols(y, dom, tested=tested, confounds=conf[:-1], **kw)
    # 2 x 10 values would pass for one column of 20.
    with pytest.raises(ValueError, match="confounds has 2 rows"):
        ns.tfce_ols(y, dom, tested=tested, confounds=np.ones((2, 10)), **kw)
    # An infinity in the design is refused, also next to a NaN.
    for name in ("tested", "confounds"):
        args = dict(tested=tested.copy(), confounds=conf.copy())
        args[name][2, ...] = np.inf
        args[name][4, ...] = np.nan
        with pytest.raises(ValueError, match=f"{name} holds an infinite value"):
            ns.tfce_ols(y, dom, **args, **kw)
    # Each error of the engine arrives as a ValueError with its own text.
    with pytest.raises(ValueError, match="degenerate design"):
        ns.tfce_ols(y, dom, tested=conf[:, 0] - 2 * conf[:, 1] + 1, confounds=conf, **kw)
    with pytest.raises(ValueError, match="degenerate design"):
        ns.tfce_ols(y, dom, tested="intercept", confounds=np.ones((n, 1)), **kw)
    with pytest.raises(ValueError, match="too few subjects"):
        ns.tfce_ols(y[:4], dom, tested=tested[:4], confounds=conf[:4], **kw)
    # So does a model with too few rows once the rows with a NaN are left out:
    # three rows for three nuisance columns, and no row at all.
    for kept in (3, 0):
        t = tested.copy()
        t[kept:] = np.nan
        with pytest.warns(UserWarning, match=f"{n - kept} of {n} rows"):
            with pytest.raises(ValueError, match="too few subjects"):
                ns.tfce_ols(y, dom, tested=t, confounds=conf, **kw)


# --- nilearn ---------------------------------------------------------------


def band(p, q, b):
    """4-sigma band for two Monte Carlo p-values from `b` draws each."""
    mean = (p + q) / 2
    return 4 * np.sqrt(mean * (1 - mean) * (2 / b)) + 1 / b


@pytest.mark.parametrize(
    "tail, n_conf, intercept",
    [
        ("two_sided", 2, False),
        ("positive", 0, False),
        ("negative", 2, False),
        ("two_sided", 0, True),
        ("positive", 2, True),
    ],
)
def test_matches_nilearn_permuted_ols(tail, n_conf, intercept):
    pytest.importorskip("nilearn")
    import nibabel as nib
    from nilearn.maskers import NiftiMasker
    from nilearn.mass_univariate import permuted_ols

    rng = np.random.default_rng(20261014)
    n, b = 24, 400
    mask = rng.random((6, 6, 5)) < 0.8
    dom = ns.Domain.from_mask(mask, conn=6)
    v = dom.n_nodes
    tested = rng.standard_normal(n)
    conf = rng.standard_normal((n, n_conf)) if n_conf else None
    y = rng.standard_normal((n, v))
    # An effect in the tested direction, so that small p-values are compared too.
    y[:, : v // 5] += 0.6 if intercept else (-0.7 if tail == "negative" else 0.7) * tested[:, None]
    if n_conf:
        y += conf @ rng.standard_normal((n_conf, v))

    # nilearn has no negative tail: it is the positive tail of the negated
    # regressor. Its n_perm does not count the unpermuted data.
    if intercept:
        tested_vars = np.ones((n, 1))
    else:
        tested_vars = (-tested if tail == "negative" else tested).reshape(-1, 1)
    masker = NiftiMasker(mask_img=nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))).fit()
    out = permuted_ols(tested_vars, y, conf, model_intercept=True, n_perm=b - 1,
                       two_sided_test=(tail == "two_sided"), random_state=0, n_jobs=1,
                       masker=masker, tfce=True, output_type="dict", verbose=0)
    t_nilearn = -out["t"][0] if tail == "negative" else out["t"][0]
    tfce_nilearn = out["tfce"][0]
    p_nilearn = 10.0 ** -out["logp_max_tfce"][0]

    r = ns.tfce_ols(y, dom, tested="intercept" if intercept else tested, confounds=conf,
                    params="nilearn", tail=tail, seed=1, n_perm=b, n_jobs=1)
    np.testing.assert_allclose(r.stat_obs, t_nilearn, rtol=1e-10)
    np.testing.assert_allclose(r.tfce_obs, tfce_nilearn, rtol=1e-10,
                               atol=1e-10 * np.max(np.abs(tfce_nilearn)))
    excess = np.abs(r.p_fwe - p_nilearn) - band(r.p_fwe, p_nilearn, b)
    assert np.all(excess <= 0), f"largest excess over the band: {excess.max()}"
    assert r.p_fwe.min() < 0.05 and p_nilearn.min() < 0.05
