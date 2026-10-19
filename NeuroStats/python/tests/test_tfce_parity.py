"""`tfce(..., params="nilearn")` against nilearn's own `calculate_tfce` on
random maps and masks, for the three tails."""

import numpy as np
import pytest

pytest.importorskip("nilearn")
from nilearn.mass_univariate._utils import calculate_tfce  # noqa: E402
from scipy.ndimage import generate_binary_structure  # noqa: E402

import neurostats as ns  # noqa: E402

N_CASES = 1000
# scipy's structuring-element rank for each `conn`.
RANK_OF_CONN = {6: 1, 18: 2, 26: 3}
EXPONENTS = [(0.5, 2.0), (1.0, 1.0), (0.6, 1.5)]


def nilearn_tfce(vol, conn, e, h, two_sided):
    out = calculate_tfce(
        vol[..., np.newaxis],
        generate_binary_structure(3, RANK_OF_CONN[conn]),
        E=e,
        H=h,
        dh="auto",
        two_sided_test=two_sided,
    )
    return out[..., 0]


@pytest.mark.parametrize("tail", ["positive", "negative", "two_sided"])
def test_tfce_nilearn_matches_calculate_tfce(tail):
    rng = np.random.default_rng(20261014)
    for case in range(N_CASES):
        shape = tuple(int(s) for s in rng.integers(3, 8, size=3))
        conn = int(rng.choice([6, 18, 26]))
        e, h = EXPONENTS[int(rng.integers(len(EXPONENTS)))]
        mask = rng.random(shape) < 0.7
        vol = rng.standard_normal(shape) * rng.uniform(0.5, 4.0)
        # nilearn works on the full grid; a zero voxel never joins a cluster,
        # which is what leaving it out of the domain does.
        vol[~mask] = 0.0
        if tail == "two_sided":
            want = nilearn_tfce(vol, conn, e, h, two_sided=True)
        elif tail == "positive":
            want = nilearn_tfce(vol, conn, e, h, two_sided=False)
        else:
            want = nilearn_tfce(-vol, conn, e, h, two_sided=False)
        dom = ns.Domain.from_mask(mask, conn=conn)
        params = "nilearn" if (e, h) == (0.5, 2.0) else {"weighting": "nilearn", "e": e, "h": h}
        got = ns.tfce(vol, dom, params=params, tail=tail)
        scale = float(np.max(np.abs(want))) or 1.0
        np.testing.assert_allclose(
            got, want, rtol=1e-12, atol=1e-12 * scale,
            err_msg=f"case {case}: shape {shape}, conn {conn}, e {e}, h {h}",
        )
