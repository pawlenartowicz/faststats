"""nilearn drop-in: ``calculate_tfce`` reproducing
``nilearn.mass_univariate._utils.calculate_tfce`` (transcribed from nilearn
0.14.1), and ``permuted_ols`` / ``non_parametric_inference`` wrappers that
run nilearn with this TFCE for the duration of the call.

Deliberate deviations from the host (all harmless for masked inputs):
NaN voxels are set to 0 (never a member); a regressor whose grid maximum is
``<= 0`` returns zeros instead of clustering a negative grid; a map containing
``+inf`` or ``-inf`` raises ``ValueError`` instead of producing an all-``inf``
map, because the band weight ``h**H`` overflows.
"""

from __future__ import annotations

import contextlib
import threading
import warnings

import numpy as np

from faststats._faststats import neuro as _core
from faststats.neuro import Domain

__all__ = ["calculate_tfce", "permuted_ols", "non_parametric_inference"]

_lock = threading.Lock()
_domain_cache: dict[tuple[tuple[int, int, int], int], Domain] = {}


def _conn_of(bin_struct) -> int:
    from scipy.ndimage import generate_binary_structure

    bs = np.asarray(bin_struct, dtype=bool)
    for k, conn in ((1, 6), (2, 18), (3, 26)):
        if bs.shape == (3, 3, 3) and np.array_equal(bs, generate_binary_structure(3, k)):
            return conn
    raise ValueError("bin_struct must equal scipy.ndimage.generate_binary_structure(3, k) for k in 1..3")


def _domain(shape, conn) -> Domain:
    key = (tuple(int(s) for s in shape), conn)
    with _lock:
        dom = _domain_cache.get(key)
        if dom is None:
            _domain_cache.clear()
            dom = Domain.from_volume(key[0], conn=conn)
            _domain_cache[key] = dom
    return dom


def _score_threshs(arr3d, dh, two_sided_test):
    """Transcription of nilearn ``_return_score_threshs``: same grid, same
    clamps, same warnings. Returns a float64 array of at least 10 thresholds.

    nilearn picks its ``stacklevel`` with a helper that walks out of the nilearn
    package; from here that helper stops at the first frame, so the count is
    fixed instead: 3 frames out of this function, through ``calculate_tfce``,
    lands on whoever called ``calculate_tfce`` — where the host attributes it.
    """
    max_score = np.nanmax(np.abs(arr3d)) if two_sided_test else np.nanmax(arr3d)

    number_steps = 100 if dh == "auto" else round(max_score / dh)
    if number_steps < 10:
        warnings.warn(
            f"Not enough steps for TFCE. Got: {number_steps=}. Setting it to 10.",
            stacklevel=3,
        )
        number_steps = 10
    if number_steps > 1000:
        warnings.warn(
            f"Too many steps for TFCE. Got: {number_steps=}. Setting it to 1000.",
            stacklevel=3,
        )
        number_steps = 1000

    threshs = np.linspace(0, max_score, number_steps + 1)[1:]
    return np.asarray(threshs, dtype=np.float64)


def calculate_tfce(arr4d, bin_struct, E=0.5, H=2, dh="auto", two_sided_test=True):
    """nilearn's ``calculate_tfce``. Returns a 4-D array of ``arr4d``'s dtype."""
    arr4d = np.asarray(arr4d)
    if arr4d.ndim != 4:
        raise ValueError(f"arr4d must be 4-D, got shape {arr4d.shape}")
    conn = _conn_of(bin_struct)
    dom = _domain(arr4d.shape[:3], conn)
    out = np.zeros(arr4d.shape, dtype=np.float64)
    for r in range(arr4d.shape[3]):
        a = np.nan_to_num(arr4d[..., r].astype(np.float64, copy=True), nan=0.0)
        m = float(np.max(np.abs(a))) if two_sided_test else float(np.max(a))
        if not m > 0:
            continue
        ths = _score_threshs(a, dh, two_sided_test)
        w = ths**H
        vec = np.ascontiguousarray(a.ravel())
        pos = _core.tfce_bands(dom._core, vec, ths, w, float(E), False)
        if two_sided_test:
            neg = _core.tfce_bands(dom._core, -vec, ths, w, float(E), False)
            pos = pos - neg
        out[..., r] = pos.reshape(arr4d.shape[:3])
    return out.astype(arr4d.dtype, copy=False)


@contextlib.contextmanager
def _patched():
    """Rebind ``calculate_tfce`` where nilearn's permutation loop looks it up
    (``nilearn.mass_univariate.permuted_least_squares`` imports it by name from
    ``._utils``, so the module attribute is what both the observed-data call and
    the per-chunk permutation loop resolve), and force joblib's threading
    backend so workers see the rebinding (the default loky workers import
    nilearn fresh). Undone on exit."""
    import joblib
    from nilearn.mass_univariate import permuted_least_squares as pls

    original = pls.calculate_tfce
    pls.calculate_tfce = calculate_tfce
    try:
        with joblib.parallel_config(backend="threading"):
            yield
    finally:
        pls.calculate_tfce = original


def permuted_ols(*args, **kwargs):
    """``nilearn.mass_univariate.permuted_ols`` with TFCE computed by faststats.

    Runs inside nilearn's joblib workers under the threading backend it
    forces, so the substituted ``calculate_tfce`` is visible to every worker
    (tested with ``n_jobs=2``)."""
    from nilearn.mass_univariate import permuted_ols as host

    with _patched():
        return host(*args, **kwargs)


def non_parametric_inference(*args, **kwargs):
    """``nilearn.glm.second_level.non_parametric_inference`` with TFCE computed by faststats.

    Runs inside nilearn's joblib workers under the threading backend it
    forces, so the substituted ``calculate_tfce`` is visible to every worker
    (tested with ``n_jobs=2``)."""
    from nilearn.glm.second_level import non_parametric_inference as host

    with _patched():
        return host(*args, **kwargs)
