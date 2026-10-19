"""``Domain``, the TFCE settings (``params``, ``tail``) and the plain TFCE
functions ``tfce`` and ``grid_and_weights``."""

from __future__ import annotations

import math

from . import _arrays
from . import _neurostats as _core
from ._arrays import Array

# Weightings the permutation inference functions accept.
INFERENCE_WEIGHTINGS = ("exact", "nilearn", "mne", "smith_nichols")
_TFCE_WEIGHTINGS = INFERENCE_WEIGHTINGS + ("fslmaths",)
_GRID_WEIGHTINGS = ("smith_nichols", "mne", "fslmaths")
_PRESETS = ("exact", "nilearn")
# Which of `start` and `step` each weighting takes.
_GRID_KEYS = {
    "exact": ("start",),
    "nilearn": (),
    "mne": ("start", "step"),
    "smith_nichols": ("start", "step"),
    "fslmaths": ("start", "step"),
}
TAILS = ("positive", "negative", "two_sided")
_CONN = (6, 18, 26)


def check_conn(conn) -> None:
    if conn not in _CONN:
        raise ValueError(f"conn must be one of {_CONN}, got {conn}")


def check_tail(tail) -> None:
    if tail not in TAILS:
        raise ValueError(f"tail must be one of {TAILS}, got {tail!r}")


def _number(value, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x


def resolve_params(params, weightings=INFERENCE_WEIGHTINGS) -> dict:
    """The ``params`` argument as a dict of TFCE settings.

    ``params`` is a preset name (``"exact"`` or ``"nilearn"``) or a dict with
    the key ``weighting`` and any of ``e``, ``h``, ``start``, ``step``.
    ``weightings`` are the weighting names the caller accepts. Returns
    ``{"weighting", "e", "h"}`` plus the ``start`` and ``step`` the weighting
    uses: ``start`` for ``"exact"``, neither for ``"nilearn"``, both for
    ``"mne"``, ``"smith_nichols"`` and ``"fslmaths"``. ``e`` and ``h`` default
    to 0.5 and 2.0, and the ``start`` of ``"exact"`` to 0. Every problem
    raises ``ValueError``.
    """
    if isinstance(params, str):
        if params not in _PRESETS:
            raise ValueError(f"params must be a dict or one of the presets {_PRESETS}, got {params!r}")
        params = {"weighting": params}
    elif not isinstance(params, dict):
        raise ValueError(f"params must be a dict or one of the presets {_PRESETS}, got {params!r}")
    if "weighting" not in params:
        raise ValueError("params needs the key 'weighting'")
    weighting = params["weighting"]
    if weighting not in weightings:
        raise ValueError(f"weighting must be one of {weightings}, got {weighting!r}")
    unknown = [k for k in params if k not in ("weighting", "e", "h", "start", "step")]
    if unknown:
        raise ValueError(f"unknown key in params: {unknown}")
    uses = _GRID_KEYS[weighting]
    unused = [k for k in ("start", "step") if k in params and k not in uses]
    if unused:
        raise ValueError(f"weighting={weighting!r} does not use {unused}")
    out = {
        "weighting": weighting,
        "e": _number(params.get("e", 0.5), "e"),
        "h": _number(params.get("h", 2.0), "h"),
    }
    for key in uses:
        if key in params:
            out[key] = _number(params[key], key)
        elif weighting == "exact":
            out[key] = 0.0
        else:
            raise ValueError(f"weighting={weighting!r} needs the key {key!r}")
    if "start" in out and out["start"] < 0:
        raise ValueError(f"start must be >= 0, got {out['start']}")
    if "step" in out and not out["step"] > 0:
        raise ValueError(f"step must be > 0, got {out['step']}")
    if weighting == "exact" and not out["h"] > -1:
        raise ValueError(f"h must be > -1 for weighting='exact', got {out['h']}")
    return out


class Domain:
    """Adjacency over the in-mask voxels of a 3-D volume, or any CSR graph.

    Node order for mask-built domains is C-order over the mask (numpy
    ``ravel`` order), so ``stat[mask]`` is already in node order.
    """

    __slots__ = ("_core", "shape", "mask")

    def __init__(self, core, shape, mask):
        self._core = core
        self.shape = shape
        self.mask = mask

    @classmethod
    def from_mask(cls, mask, conn: int = 6) -> "Domain":
        flat, shape, mask = _arrays.mask_volume(mask)
        check_conn(conn)
        return cls(_core.Domain.from_mask(flat, shape, conn), shape, mask)

    @classmethod
    def from_volume(cls, shape, conn: int = 6) -> "Domain":
        shape = tuple(int(s) for s in shape)
        if len(shape) != 3:
            raise ValueError(f"shape must have 3 entries, got {shape}")
        check_conn(conn)
        return cls(_core.Domain.from_volume(shape, conn), shape, _arrays.full_mask(shape))

    @classmethod
    def from_csr(cls, indptr, indices) -> "Domain":
        """Caller-built CSR adjacency; symmetry is the caller's precondition."""
        return cls(_core.Domain.from_csr(*_arrays.csr(indptr, indices)), None, None)

    @classmethod
    def from_adjacency(cls, adj, n: int | None = None) -> "Domain":
        """Any object with ``.tocoo()`` (scipy sparse). Symmetrised, self-loops
        dropped, duplicate edges merged, explicitly stored zeros are not edges;
        ``shape`` and ``mask`` are ``None``."""
        return cls(_core.Domain.from_csr(*_arrays.csr_from_adjacency(adj, n)), None, None)

    @property
    def n_nodes(self) -> int:
        return self._core.n_nodes

    @property
    def n_edges(self) -> int:
        return self._core.n_edges

    def __repr__(self) -> str:
        return f"Domain(n_nodes={self.n_nodes}, n_edges={self.n_edges}, shape={self.shape})"


def grid_and_weights(stat_max: float, params) -> tuple[Array, Array]:
    """``(thresholds, weights)`` of a grid weighting for a map whose maximum
    is ``stat_max``.

    ``params`` is the dict form of ``"smith_nichols"``, ``"mne"`` or
    ``"fslmaths"`` (see :func:`tfce`), with ``start`` and ``step``. Thresholds
    are ``arange(start, stat_max, step)``. Weights: ``smith_nichols`` =
    ``h_i^H * step``, ``mne`` = ``|h_i - h_{i-1}|^H`` (with ``|h_0|^H``
    first), ``fslmaths`` = ``h_i^H``.

    Checked in ``tests/test_core.py::test_tails_on_the_grid_of_grid_and_weights``:
    the band sum over this grid equals :func:`tfce` for ``smith_nichols`` and
    ``mne``.
    """
    p = resolve_params(params, _GRID_WEIGHTINGS)
    return _arrays.grid_and_weights(float(stat_max), p["start"], p["step"], p["h"], p["weighting"])


def _tfce_fslmaths(vec, domain: Domain, p: dict, tail: str):
    maps, stat_max = _arrays.tail_maps(vec, tail, p["start"])
    ths, w = _arrays.grid_and_weights(stat_max, p["start"], p["step"], p["h"], "fslmaths")
    out = [_core.tfce_bands(domain._core, m, ths, w, p["e"], True) for m in maps]
    return out[0] if len(out) == 1 else _arrays.difference(out[0], out[1])


def tfce(stat, domain: Domain, *, params="exact", tail: str = "two_sided") -> Array:
    """Threshold-free cluster enhancement of ``stat`` over ``domain``.

    ``stat``: 1-D of ``n_nodes``, or 3-D of ``domain.shape`` for a domain
    built from a mask or a volume (the output is then 3-D with zeros outside
    the mask). Non-finite values inside the domain raise ``ValueError``; for
    3-D input, voxels outside the mask are never read.

    ``params``: a preset name or a dict of TFCE settings.

    - ``"exact"``: the closed-form integral from ``start = 0`` with
      ``e = 0.5``, ``h = 2.0``. No threshold grid.
    - ``"nilearn"``: nilearn's ``calculate_tfce`` rule with ``e = 0.5``,
      ``h = 2.0``: 100 equal steps from 0 up to and including the map's
      maximum, weight ``h_i^H``, membership ``stat >= h_i``.
    - a dict with the key ``weighting`` and any of ``e``, ``h``, ``start``,
      ``step``. ``"exact"`` takes ``start`` (default 0) and no ``step``;
      ``"nilearn"`` takes neither; ``"mne"`` (``|h_i - h_{i-1}|^H``),
      ``"smith_nichols"`` (``h_i^H * step``) and ``"fslmaths"`` (``h_i^H``)
      need both and use the thresholds ``arange(start, max, step)`` with
      membership ``stat > h_i``.

    ``tail``: ``"positive"`` enhances ``stat``, ``"negative"`` enhances
    ``-stat``, and ``"two_sided"`` returns the first minus the second, which
    is signed. Both passes of a two-sided call share one grid built from
    ``max |stat|``.

    Checked against nilearn 0.14.1 ``calculate_tfce`` for ``"nilearn"``
    (``tests/test_tfce_parity.py::test_tfce_nilearn_matches_calculate_tfce``)
    and against MNE 1.12.1 fixtures for ``"mne"``
    (``tests/test_core.py::test_tfce_mne_weighting_reproduces_fixture``).

    A bad ``params`` or ``tail`` raises ``ValueError`` before ``stat`` is read.
    """
    p = resolve_params(params, _TFCE_WEIGHTINGS)
    check_tail(tail)
    vec, was_3d = _arrays.to_nodes(stat, domain, "stat")
    _arrays.check_finite(vec)
    if p["weighting"] == "fslmaths":
        out = _tfce_fslmaths(vec, domain, p, tail)
    else:
        out = _core.tfce(
            domain._core, vec, p["e"], p["h"], p.get("start"), p.get("step"), p["weighting"], tail
        )
    return _arrays.scatter(out, domain, 0.0) if was_3d else out
