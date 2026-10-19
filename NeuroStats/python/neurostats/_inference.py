"""Permutation inference with max-TFCE family-wise error correction."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field

from . import _arrays
from . import _neurostats as _core
from ._arrays import Array
from ._io import import_from
from ._tfce import check_tail, resolve_params

# The per-node draw counts are 32-bit.
_MAX_DRAWS = 2**32 - 1
_ARRAYS = ("stat_obs", "tfce_obs", "p_fwe", "p_unc", "null_max")


@dataclass(frozen=True, eq=False)
class TfceInference:
    """Result of a TFCE permutation inference function.

    Maps are 1-D float64, one value per node of the ``Domain``.

    - ``stat_obs``: the observed statistic, signed under every tail.
    - ``tfce_obs``: its TFCE under ``tail``: of the map (``"positive"``), of
      the negated map (``"negative"``), or the first minus the second, signed
      (``"two_sided"``).
    - ``p_fwe``: ``#{b : max_b >= T(v)} / B`` over the max-TFCE null, where
      ``T`` is ``tfce_obs``, or ``|tfce_obs|`` under ``"two_sided"``.
    - ``p_unc``: ``#{b : T_b(v) >= T(v)} / B`` over the node's own null, with
      no correction for the number of nodes.
    - ``null_max``: the null value of every draw (the maximum of ``T`` over
      the nodes), sorted ascending, length ``B``.
    - ``params``: the resolved TFCE settings: ``weighting``, ``e``, ``h`` and
      the ``start`` and ``step`` the weighting uses.
    - ``tail``, ``seed``: as the run used them. ``seed`` is the drawn seed
      when the call had ``seed=None``.
    - ``n_perm``: ``B``, the number of draws used, the identity draw included.
      It is below the requested budget when every sign pattern was enumerated.
    - ``origin``: the masker or mask image the data came through, or ``None``
      for plain arrays and for a loaded result.

    Both p-values count the identity draw, so neither is ever below ``1/B``.
    A result also keeps what a later call with ``extend=`` needs to continue
    the null; :meth:`save` writes that too.
    """

    stat_obs: Array
    tfce_obs: Array
    p_fwe: Array
    p_unc: Array
    null_max: Array
    params: dict
    tail: str
    seed: int
    n_perm: int
    origin: object
    # What `extend` needs beyond the fields above: `exceed` (per node, the
    # number of draws with T_b(v) >= T(v)), `n` (subjects), `q` (nuisance
    # columns kept) and `engine` (which function made the null).
    _null: dict = field(repr=False)

    def save(self, path) -> None:
        """Write the result to ``path`` as an ``.npz`` file. The name is used
        as given. ``origin`` is not written."""
        arrays = {name: getattr(self, name) for name in _ARRAYS}
        arrays["exceed"] = self._null["exceed"]
        meta = {
            "params": self.params,
            "tail": self.tail,
            "seed": self.seed,
            "n_perm": self.n_perm,
            "n": self._null["n"],
            "q": self._null["q"],
            "engine": self._null["engine"],
        }
        _arrays.save_npz(path, arrays, json.dumps(meta))

    @classmethod
    def load(cls, path) -> "TfceInference":
        """Read a result written by :meth:`save`. ``origin`` is ``None``. The
        result can be passed as ``extend=`` like the one that was saved. A
        file that is not a saved result raises ``ValueError``."""
        arrays, meta = _arrays.load_npz(path, (*_ARRAYS, "exceed"))
        meta = json.loads(meta)
        null = {"exceed": arrays.pop("exceed"), "n": meta["n"], "q": meta["q"], "engine": meta["engine"]}
        return cls(**arrays, params=meta["params"], tail=meta["tail"], seed=meta["seed"],
                   n_perm=meta["n_perm"], origin=None, _null=null)


def _resolve_data(data, space, conn):
    """``(y, domain, origin)`` from the ``data``, ``space`` and ``conn``
    arguments of an inference function.

    ``y``: the brain data as :func:`_arrays.source` returns them,
    ``(source rows, domain.n_nodes)``. ``origin``: what
    ``TfceInference.origin`` is set to.
    """
    masked = import_from(data, space, conn=conn)
    return masked.y, masked.domain, masked.origin


def _run_settings(params, tail, seed, n_perm, n_jobs, extend, engine: str):
    """Check the settings of an inference call and apply the rules of
    ``extend``, before any data is read. ``engine`` names the calling
    function (``"ols"``).

    Returns ``(p, seed, n_perm, threads, stored)``: the resolved ``params``
    dict, the seed of the run (drawn when ``seed`` is ``None`` and nothing is
    continued), the draw budget, the thread count for the compiled module
    (0 for every core), and the stored null as the compiled module takes it
    (``None`` for a fresh run).
    """
    p = resolve_params(params)
    check_tail(tail)
    try:
        n_perm, n_jobs = int(n_perm), int(n_jobs)
        seed = None if seed is None else int(seed)
    except (TypeError, OverflowError) as e:
        raise ValueError(f"seed, n_perm and n_jobs must be integers: {e}") from e
    if not 1 <= n_perm <= _MAX_DRAWS:
        raise ValueError(f"n_perm must be in [1, {_MAX_DRAWS}], got {n_perm}")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be -1 (all cores) or a positive integer, got {n_jobs}")
    threads = 0 if n_jobs == -1 else n_jobs
    if seed is not None and not 0 <= seed < 2**64:
        raise ValueError(f"seed must be in [0, 2**64), got {seed}")
    if extend is None:
        if seed is None:
            seed = secrets.randbits(64)
        return p, seed, n_perm, threads, None
    if not isinstance(extend, TfceInference):
        raise ValueError(f"extend must be a TfceInference, got {type(extend).__name__}")
    null = extend._null
    if null["engine"] != engine:
        raise ValueError(f"extend is a result of tfce_{null['engine']}, not of tfce_{engine}")
    if seed is None:
        seed = extend.seed
    elif seed != extend.seed:
        raise ValueError(
            f"seed {seed} is not the seed of the stored null ({extend.seed}); "
            "leave seed=None to continue it"
        )
    if p != extend.params:
        raise ValueError(f"params {p} are not the settings of the stored null ({extend.params})")
    if tail != extend.tail:
        raise ValueError(f"tail {tail!r} is not the tail of the stored null ({extend.tail!r})")
    if n_perm <= extend.n_perm:
        raise ValueError(
            f"the stored null already has {extend.n_perm} draws; n_perm is the total "
            f"number of draws and must be larger, got {n_perm}"
        )
    stored = (
        _arrays.f64(extend.null_max),
        _arrays.u32(null["exceed"]),
        _arrays.f64(extend.tfce_obs),
        null["n"],
        null["q"],
    )
    return p, seed, n_perm, threads, stored


def _result(out, p: dict, tail: str, seed: int, origin, engine: str) -> TfceInference:
    """The ``TfceInference`` of what an inference function of the compiled
    module returned."""
    stat_obs, tfce_obs, p_fwe, p_unc, null_max, exceed, n, q = out
    null = {"exceed": exceed, "n": n, "q": q, "engine": engine}
    return TfceInference(stat_obs, tfce_obs, p_fwe, p_unc, null_max, p, tail, seed,
                         len(null_max), origin, null)


def tfce_ols(data, space=None, *, tested, confounds=None, conn=None, params="exact",
             tail: str = "two_sided", seed: int | None = None, n_perm: int = 10000,
             n_jobs: int = 1, extend: TfceInference | None = None) -> TfceInference:
    """TFCE permutation inference for one linear model per node, with
    max-TFCE family-wise error correction.

    Model: ``brain ~ tested + confounds + intercept``, the same design at
    every node; the statistic is the t of ``tested``. With
    ``tested="intercept"`` the model is ``brain ~ intercept + confounds`` and
    the statistic is the t of the intercept, which without confounds is the
    one-sample t (``ddof = 1``). The t is
    ``b * sqrt((n - q - 1) / (1 - b**2 - a2))``, with ``b`` the correlation of
    the node's residualised data with the residualised tested column, ``a2``
    the part of the node's variance the nuisance columns take in that draw,
    and ``q`` the number of nuisance columns (the confounds kept, plus the
    intercept when a regressor is tested). A node with no residual variance
    gets ``t = 0``.

    Draws: with a tested regressor the rows of the design are shuffled
    together and the brain data stay fixed (the Freedman-Lane statistic).
    With ``tested="intercept"`` the rows are sign-flipped; every one of the
    ``2**n`` patterns is enumerated when ``2**n <= n_perm``. Draw 0 is the
    data as observed.

    Checked against nilearn 0.14.1 ``permuted_ols(..., tfce=True)``: under
    ``params="nilearn"`` the t map and the TFCE map agree to a relative
    1e-10, and ``p_fwe`` agrees within the 4-sigma Monte Carlo band of two
    runs (``tests/test_ols.py::test_matches_nilearn_permuted_ols``).
    nilearn's ``n_perm`` does not count the unpermuted data, so ``n_perm=B``
    here corresponds to ``n_perm=B - 1`` there. The observed t is also
    checked against one least-squares fit per node
    (``::test_stat_obs_is_the_t_of_a_least_squares_fit``), and the
    one-sample case against MNE-Python 1.12.1
    (``::test_intercept_test_reproduces_the_mne_fixture``).

    ``data``, ``space`` and ``conn`` go through :func:`import_from`, which
    lists every accepted form. ``data``: an ``(n, V)`` array with one row per
    subject and one column per node of ``space``, images, or a
    ``MaskedData``. A C-contiguous float32 or float64 array (a NumPy memory
    map too) is read in place; anything else is converted to float64 first.
    ``space``: a ``Domain``, a fitted masker or a mask image; left out with a
    ``MaskedData``. ``conn``: 6, 18 or 26 for the ``Domain`` built from a
    masker or a mask image, ``None`` for 6; it must be left as ``None`` with
    a ``Domain`` or a ``MaskedData``.

    ``tested``: one value per subject, 1-D, or the string ``"intercept"``.
    ``confounds``: ``(n, c)`` or ``None``; no constant column is needed. A
    column that is a combination of earlier ones is dropped, and so is a
    constant column when a regressor is tested. A row whose ``tested`` value
    or any confound value is NaN is left out, with a ``UserWarning`` that
    states how many; ``n`` is then the number of rows kept, and the kept rows
    of ``data`` are read without a copy. An infinite value in ``tested`` or
    ``confounds`` raises ``ValueError``.

    ``params``: the TFCE settings, a preset name or a dict (see :func:`tfce`;
    ``"fslmaths"`` is not available here). ``tail``: ``"positive"``,
    ``"negative"`` or ``"two_sided"``.

    ``seed``: an integer in ``[0, 2**64)``, or ``None`` for a fresh seed,
    which is stored in the result. ``n_perm``: the draw budget, the identity
    draw included. ``n_jobs``: ``1`` (the default) uses one thread, ``-1``
    uses every core, a positive integer ``k`` uses ``k`` threads, ``0`` or
    any other negative value raises ``ValueError``. A result has the same
    bits for every ``n_jobs``.

    ``extend``: an earlier result of the same call (same data, design,
    ``params`` and ``tail``). ``n_perm`` is then the new total: only the
    draws the stored null lacks are run, and the result equals one run of
    ``n_perm`` draws bit for bit. ``seed=None`` takes the stored seed; a
    given seed must equal it.

    Every problem with the inputs raises ``ValueError``; ``params``,
    ``tail``, ``seed``, ``n_perm``, ``n_jobs`` and the rules of ``extend``
    are checked before ``data`` is read.
    """
    intercept = isinstance(tested, str)
    if tested is None or (intercept and tested != "intercept"):
        raise ValueError(f"tested must be an array or the string 'intercept', got {tested!r}")
    p, seed, n_perm, threads, stored = _run_settings(params, tail, seed, n_perm, n_jobs, extend, "ols")
    y, domain, origin = _resolve_data(data, space, conn)
    rows, tested, confounds = _arrays.complete_rows(
        y.shape[0],
        None if intercept else _arrays.column(tested, "tested"),
        None if confounds is None else _arrays.columns(confounds, "confounds"),
    )
    out = _core.tfce_ols(
        domain._core,
        y,
        rows,
        tested,
        confounds,
        p["e"], p["h"], p.get("start"), p.get("step"), p["weighting"],
        tail, seed, n_perm, threads, stored,
    )
    return _result(out, p, tail, seed, origin, "ols")
