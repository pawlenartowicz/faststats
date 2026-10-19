"""Every numpy call of the package.

The other modules hand arrays to these helpers and never import numpy
themselves. A ``domain`` argument is a ``neurostats.Domain``; only its
``n_nodes``, ``shape`` and ``mask`` attributes are read.
"""

from __future__ import annotations

import warnings

import numpy as np

Array = np.ndarray


def f64(a) -> Array:
    """``a`` as a C-contiguous float64 array (a copy only when needed)."""
    return np.ascontiguousarray(a, dtype=np.float64)


def u32(a) -> Array:
    """``a`` as a C-contiguous uint32 array (a copy only when needed)."""
    return np.ascontiguousarray(a, dtype=np.uint32)


def mask_volume(mask):
    """A 3-D boolean mask as ``(flat C-order bool array, shape, own copy)``."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D, got shape {mask.shape}")
    shape = tuple(int(s) for s in mask.shape)
    return np.ascontiguousarray(mask.ravel()), shape, mask.copy()


def full_mask(shape) -> Array:
    """The all-true mask of a volume."""
    return np.ones(shape, dtype=bool)


def image_mask(img) -> Array:
    """The in-mask voxels of a mask image as a boolean array of the image's
    shape. A voxel is in the mask when its value is non-zero and finite, the
    rule of nilearn's ``NiftiMasker``."""
    a = np.asarray(img.dataobj)
    return (a != 0) & np.isfinite(a)


def csr(indptr, indices):
    """CSR row pointers and column indices as C-contiguous uint32 arrays."""
    return (
        np.ascontiguousarray(indptr, dtype=np.uint32),
        np.ascontiguousarray(indices, dtype=np.uint32),
    )


def csr_from_adjacency(adj, n):
    """CSR ``(indptr, indices)`` of any object with ``.tocoo()``: symmetrised,
    self-loops dropped, duplicate edges merged, stored zeros are not edges."""
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
    return indptr, pairs[:, 1].astype(np.uint32)


def to_nodes(stat, domain, name: str):
    """``(1-D float64 node vector, was_3d)``. ``stat`` is 1-D of ``n_nodes``,
    or 3-D of ``domain.shape`` for a domain built from a mask or a volume."""
    a = np.asarray(stat)
    if a.ndim == 1 and a.shape[0] == domain.n_nodes:
        return f64(a), False
    if a.ndim == 3 and domain.shape is not None and a.shape == domain.shape:
        return f64(a[domain.mask]), True
    raise ValueError(
        f"{name} must have shape ({domain.n_nodes},)"
        + (f" or {domain.shape}" if domain.shape is not None else "")
        + f", got {a.shape}"
    )


def source(data, n_nodes: int) -> Array:
    """The brain data as the engine reads them: 2-D ``(n, n_nodes)``,
    C-contiguous, float32 or float64. An array that already has that form
    (an ``np.memmap`` included) is returned as it is, with no copy; anything
    else is converted to C-contiguous float64."""
    a = data if isinstance(data, np.ndarray) else np.asarray(data)
    if a.ndim != 2 or a.shape[1] != n_nodes:
        raise ValueError(f"data must have shape (n, {n_nodes}), got {a.shape}")
    if a.dtype in (np.float32, np.float64) and a.flags.c_contiguous:
        return a
    return f64(a)


def column(x, name: str) -> Array:
    """``x`` as a 1-D C-contiguous float64 array."""
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, one value per row of data, got shape {a.shape}")
    return f64(a)


def columns(x, name: str) -> Array:
    """``x`` as a 2-D C-contiguous float64 array."""
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D, one row per row of data, got shape {a.shape}")
    return f64(a)


def complete_rows(n: int, tested, confounds):
    """``(rows, tested, confounds)`` for data with ``n`` rows, without the
    rows whose ``tested`` value or any confound value is NaN.

    ``tested`` is 1-D float64 or ``None``, ``confounds`` 2-D float64 or
    ``None``. When rows are left out, ``rows`` is the uint64 index of the
    rows kept, in ascending order, ``tested`` and ``confounds`` hold those
    rows only, and a ``UserWarning`` states how many rows were left out.
    When none is, ``rows`` is ``None`` and the arrays are returned as they
    are. An infinite value raises ``ValueError``. An array whose length is
    not ``n`` is returned as it is, with ``rows=None``: the compiled function
    reports the mismatch."""
    given = [a for a in (tested, confounds) if a is not None]
    for name, a in (("tested", tested), ("confounds", confounds)):
        if a is not None and np.isinf(a).any():
            raise ValueError(f"{name} holds an infinite value; a NaN marks a missing value")
    if any(a.shape[0] != n for a in given):
        return None, tested, confounds
    nan = np.zeros(n, dtype=bool)
    for a in given:
        nan |= np.isnan(a) if a.ndim == 1 else np.isnan(a).any(axis=1)
    if not nan.any():
        return None, tested, confounds
    warnings.warn(
        f"{int(nan.sum())} of {n} rows have a NaN in tested or confounds and are left out",
        UserWarning,
        stacklevel=3,
    )
    keep = ~nan
    return (
        np.flatnonzero(keep).astype(np.uint64),
        None if tested is None else tested[keep],
        None if confounds is None else confounds[keep],
    )


def scatter(vec, domain, fill: float) -> Array:
    """A node vector as a volume of ``domain.shape``, ``fill`` outside the mask."""
    out = np.full(domain.shape, fill, dtype=np.float64)
    out[domain.mask] = vec
    return out


def check_finite(vec) -> None:
    if not np.all(np.isfinite(vec)):
        raise ValueError("statistic map contains NaN or infinity")


def tail_maps(vec, tail: str, empty_max: float):
    """The maps one tail enhances and the maximum their shared grid is built
    from: ``([vec], max)``, ``([-vec], max of -vec)`` or
    ``([vec, -vec], max |vec|)``. ``empty_max`` is the maximum of an empty map."""
    if tail == "positive":
        maps = [vec]
    elif tail == "negative":
        maps = [-vec]
    else:
        maps = [vec, -vec]
    stat_max = max(float(m.max()) for m in maps) if vec.size else empty_max
    return maps, stat_max


def difference(a, b) -> Array:
    return a - b


def neg_log10(p) -> Array:
    """``-log10(p)``, elementwise."""
    return -np.log10(p)


def save_npz(path, arrays: dict, meta: str) -> None:
    """Write ``arrays`` and the text ``meta`` to ``path`` as an uncompressed
    ``.npz``. The file name is used as given: no extension is added."""
    with open(path, "wb") as f:
        np.savez(f, meta=np.array(meta), **arrays)


def load_npz(path, keys):
    """``({key: array}, meta)`` of a file written by :func:`save_npz`.
    ``ValueError`` when the file is not an ``.npz`` or lacks an entry."""
    z = np.load(path, allow_pickle=False)
    if not hasattr(z, "files"):
        raise ValueError(f"{path} is not a saved result: not an .npz file")
    with z:
        missing = [k for k in (*keys, "meta") if k not in z.files]
        if missing:
            raise ValueError(f"{path} is not a saved result: no entry {missing}")
        return {k: z[k] for k in keys}, str(z["meta"])


def grid_and_weights(stat_max: float, start: float, step: float, h: float, weighting: str):
    """``arange(start, stat_max, step)`` and its band weights: ``smith_nichols``
    = ``h_i^H * step``, ``mne`` = ``|h_i - h_{i-1}|^H`` (with ``|h_0|^H``
    first), ``fslmaths`` = ``h_i^H``."""
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
