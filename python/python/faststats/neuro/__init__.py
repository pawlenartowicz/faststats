"""faststats.neuro — TFCE and one-sample permutation inference over masked volumes.

Conventions are never defaulted where the reference tools disagree: ``start``,
``step`` and ``weighting`` are required keywords of :func:`tfce`. Build a
:class:`Domain` once and reuse it across maps; every call takes float64 and
returns float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faststats._faststats import neuro as _core

__all__ = ["Domain", "TfceInference", "tfce", "tfce_one_sample", "grid_and_weights"]

_CONN = (6, 18, 26)
_GRID_WEIGHTINGS = ("smith_nichols", "mne", "fslmaths")
_WEIGHTINGS = _GRID_WEIGHTINGS + ("exact",)
_TAILS = ("positive", "negative", "two_sided", "two_sided_unsigned")


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


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
    def from_mask(cls, mask, conn: int = 26) -> "Domain":
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 3:
            raise ValueError(f"mask must be 3-D, got shape {mask.shape}")
        _check_conn(conn)
        flat = np.ascontiguousarray(mask.ravel())
        core = _core.Domain.from_mask(flat, tuple(int(s) for s in mask.shape), conn)
        return cls(core, tuple(int(s) for s in mask.shape), mask.copy())

    @classmethod
    def from_volume(cls, shape, conn: int = 26) -> "Domain":
        shape = tuple(int(s) for s in shape)
        if len(shape) != 3:
            raise ValueError(f"shape must have 3 entries, got {shape}")
        _check_conn(conn)
        return cls(_core.Domain.from_volume(shape, conn), shape, np.ones(shape, dtype=bool))

    @classmethod
    def from_csr(cls, indptr, indices) -> "Domain":
        """Caller-built CSR adjacency; symmetry is the caller's precondition."""
        core = _core.Domain.from_csr(
            np.ascontiguousarray(indptr, dtype=np.uint32),
            np.ascontiguousarray(indices, dtype=np.uint32),
        )
        return cls(core, None, None)

    @classmethod
    def from_adjacency(cls, adj, n: int | None = None) -> "Domain":
        """Any object with ``.tocoo()`` (scipy sparse). Symmetrised, self-loops
        dropped, duplicate edges merged, explicitly stored zeros are not edges;
        ``shape`` and ``mask`` are ``None``."""
        coo = adj.tocoo()
        if n is None:
            if coo.shape[0] != coo.shape[1]:
                raise ValueError(f"adjacency must be square, got shape {coo.shape}")
            n = int(coo.shape[0])
        r = np.asarray(coo.row, dtype=np.int64)
        c = np.asarray(coo.col, dtype=np.int64)
        keep = (r != c) & (np.asarray(coo.data) != 0)
        r, c = r[keep], c[keep]
        rows = np.concatenate([r, c])
        cols = np.concatenate([c, r])
        if rows.size and (rows.max() >= n or cols.max() >= n):
            raise ValueError("adjacency index out of range")
        pairs = np.unique(np.stack([rows, cols], axis=1), axis=0)
        counts = np.bincount(pairs[:, 0], minlength=n)
        indptr = np.zeros(n + 1, dtype=np.uint32)
        np.cumsum(counts, out=indptr[1:])
        return cls(_core.Domain.from_csr(indptr, pairs[:, 1].astype(np.uint32)), None, None)

    @property
    def n_nodes(self) -> int:
        return self._core.n_nodes

    @property
    def n_edges(self) -> int:
        return self._core.n_edges

    def __repr__(self) -> str:
        return f"Domain(n_nodes={self.n_nodes}, n_edges={self.n_edges}, shape={self.shape})"


def _check_conn(conn: int) -> None:
    if conn not in _CONN:
        raise ValueError(f"conn must be one of {_CONN}, got {conn}")


def grid_and_weights(stat_max: float, start: float, step: float, h: float, weighting: str):
    """``arange(start, stat_max, step)`` and the band weights of ``weighting``:
    ``smith_nichols`` = ``h_i^H * step``, ``mne`` = ``|h_i - h_{i-1}|^H`` (with
    ``|h_0|^H`` first), ``fslmaths`` = ``h_i^H``."""
    if weighting not in _GRID_WEIGHTINGS:
        raise ValueError(f"weighting must be one of {_GRID_WEIGHTINGS}, got {weighting!r}")
    if step is None or not (step > 0) or not np.isfinite(step):
        raise ValueError(f"step must be a positive finite number for weighting={weighting!r}")
    if start < 0:
        raise ValueError(f"start must be >= 0 for weighting={weighting!r}, got {start}")
    ths = np.arange(start, stat_max, step, dtype=np.float64)
    if weighting == "smith_nichols":
        w = ths**h * step
    elif weighting == "fslmaths":
        w = ths**h
    else:
        dh = np.empty_like(ths)
        if ths.size:
            dh[0] = abs(ths[0])
            dh[1:] = np.abs(np.diff(ths))
        w = dh**h
    return ths, w


def _to_nodes(stat, domain: Domain, name: str) -> tuple[np.ndarray, bool]:
    """Return ``(1-D node vector, was_3d)`` under the spec's shape rules."""
    a = np.asarray(stat)
    if a.ndim == 1 and a.shape[0] == domain.n_nodes:
        return _f64(a), False
    if a.ndim == 3 and domain.shape is not None and a.shape == domain.shape:
        return _f64(a[domain.mask]), True
    raise ValueError(
        f"{name} must have shape ({domain.n_nodes},)"
        + (f" or {domain.shape}" if domain.shape is not None else "")
        + f", got {a.shape}"
    )


def _scatter(vec: np.ndarray, domain: Domain, fill: float) -> np.ndarray:
    out = np.full(domain.shape, fill, dtype=np.float64)
    out[domain.mask] = vec
    return out


def _tfce_positive(vec, domain, e, h, start, step, weighting):
    if weighting == "exact":
        return _core.tfce_exact(domain._core, vec, e, h, start)
    stat_max = float(vec.max()) if vec.size else start
    ths, w = grid_and_weights(stat_max, start, step, h, weighting)
    return _core.tfce_bands(domain._core, vec, ths, w, e, True)


def tfce(stat, domain: Domain, *, e: float = 0.5, h: float = 2.0, start: float,
         step: float | None, weighting: str, tail: str = "positive") -> np.ndarray:
    """Threshold-free cluster enhancement of ``stat`` over ``domain``.

    ``stat``: 1-D of ``n_nodes`` or 3-D of ``domain.shape`` (mask-built
    domains; output is 3-D with zeros outside the mask). Non-finite values
    inside the domain raise ``ValueError``; for 3-D input, voxels outside the
    mask are never read, so NaN there is ignored. Thresholds are ``arange(start, max, step)`` for the
    grid weightings; ``weighting="exact"`` is the closed-form integral from
    ``start`` and ignores ``step``. Tails: ``"negative"`` enhances ``-stat``;
    the two-sided tails share one grid from ``max(|stat|)`` and return
    ``pos - neg`` (signed) or ``pos + neg`` (unsigned).
    """
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"weighting must be one of {_WEIGHTINGS}, got {weighting!r}")
    if tail not in _TAILS:
        raise ValueError(f"tail must be one of {_TAILS}, got {tail!r}")
    vec, was_3d = _to_nodes(stat, domain, "stat")
    if not np.all(np.isfinite(vec)):
        raise ValueError("statistic map contains NaN or infinity")
    if tail == "positive":
        out = _tfce_positive(vec, domain, e, h, start, step, weighting)
    elif tail == "negative":
        out = _tfce_positive(-vec, domain, e, h, start, step, weighting)
    elif weighting == "exact":
        pos = _core.tfce_exact(domain._core, vec, e, h, start)
        neg = _core.tfce_exact(domain._core, -vec, e, h, start)
        out = pos - neg if tail == "two_sided" else pos + neg
    else:
        stat_max = float(np.abs(vec).max()) if vec.size else start
        ths, w = grid_and_weights(stat_max, start, step, h, weighting)
        pos = _core.tfce_bands(domain._core, vec, ths, w, e, True)
        neg = _core.tfce_bands(domain._core, -vec, ths, w, e, True)
        out = pos - neg if tail == "two_sided" else pos + neg
    return _scatter(out, domain, 0.0) if was_3d else out


@dataclass(frozen=True)
class TfceInference:
    """Output of :func:`tfce_one_sample`. Maps are 1-D over nodes, or 3-D for
    a 4-D input (0 outside the mask for ``t_obs``/``tfce_obs``, 1.0 for the
    p-maps). ``null_max`` is the sorted max-TFCE null, length ``B``."""

    t_obs: np.ndarray
    tfce_obs: np.ndarray
    p_fwe: np.ndarray
    p_unc: np.ndarray
    null_max: np.ndarray


def tfce_one_sample(x, domain: Domain, *, e: float = 0.5, h: float = 2.0, start: float,
                    step: float | None, weighting: str, seed: int, n_perm: int,
                    n_jobs: int = -1) -> TfceInference:
    """One-sample sign-flip permutation test with max-TFCE FWE correction.

    ``x``: ``(n_subj, n_nodes)`` or ``(n_subj, nx, ny, nz)``. One-sample t
    (ddof 1, ``t = 0`` at degenerate nodes), positive tail. ``n_perm`` is the
    budget including the identity: all ``2**n_subj`` sign patterns when that
    fits, else ``n_perm`` Monte Carlo draws from ``seed``. ``p_fwe = #{max_b >=
    tfce_obs} / B``, ``p_unc = #{tfce_b(v) >= tfce_obs(v)} / B``.

    ``n_jobs`` follows the nilearn/scikit-learn convention: ``-1`` uses every
    core, a positive integer ``k`` uses exactly ``k`` OS threads, and ``0`` or
    any other negative value raises ``ValueError``. Results are identical (down
    to the bit) for every ``n_jobs``.
    """
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"weighting must be one of {_WEIGHTINGS}, got {weighting!r}")
    if weighting == "fslmaths":
        raise ValueError("weighting='fslmaths' is not available in tfce_one_sample")
    a = np.asarray(x)
    if a.ndim == 2 and a.shape[1] == domain.n_nodes:
        flat, was_4d = _f64(a), False
    elif a.ndim == 4 and domain.shape is not None and a.shape[1:] == domain.shape:
        flat, was_4d = _f64(a[:, domain.mask]), True
    else:
        raise ValueError(
            f"x must have shape (n_subj, {domain.n_nodes})"
            + (f" or (n_subj, *{domain.shape})" if domain.shape is not None else "")
            + f", got {a.shape}"
        )
    n_subj = int(flat.shape[0])
    if weighting == "exact":
        step_val = 1.0
    else:
        if step is None or not (step > 0) or not np.isfinite(step):
            raise ValueError(f"step must be a positive finite number for weighting={weighting!r}")
        step_val = float(step)
    seed, n_perm = int(seed), int(n_perm)
    if not 0 <= seed < 2**64:
        raise ValueError(f"seed must be in [0, 2**64), got {seed}")
    if n_perm < 1:
        raise ValueError(f"n_perm must be >= 1, got {n_perm}")
    n_jobs = int(n_jobs)
    if n_jobs == 0 or (n_jobs < 0 and n_jobs != -1):
        raise ValueError(f"n_jobs must be -1 (all cores) or a positive integer, got {n_jobs}")
    threads = 0 if n_jobs == -1 else n_jobs
    t_obs, tfce_obs, p_fwe, p_unc, null_max = _core.tfce_one_sample(
        domain._core, flat.ravel(), n_subj, float(e), float(h), float(start), step_val,
        weighting, seed, n_perm, threads,
    )
    if was_4d:
        return TfceInference(_scatter(t_obs, domain, 0.0), _scatter(tfce_obs, domain, 0.0),
                             _scatter(p_fwe, domain, 1.0), _scatter(p_unc, domain, 1.0), null_max)
    return TfceInference(t_obs, tfce_obs, p_fwe, p_unc, null_max)
