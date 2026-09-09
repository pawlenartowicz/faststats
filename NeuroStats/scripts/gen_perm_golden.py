#!/usr/bin/env python3
"""Generate neurostats sign-flip permutation goldens from MNE-Python. Run on a dev
machine:

    python scripts/gen_perm_golden.py

MNE/NumPy/SciPy are generation-time tools only — never crate or CI dependencies.

Per case the fixture stores the subject-major data `x` (n·V), the mask/dims/conn
and TFCE params, MNE's raw one-sample t (`ttest_1samp_no_p`, sigma=0), MNE's
TFCE-enhanced observed map, its FWE p per node (`cluster_pv`) and its sorted
max-TFCE null (`h0`, identity included), from
`permutation_cluster_1samp_test(..., threshold=dict(...), tail=1)`.

Exact cases request n_permutations = 2^n so MNE enumerates every non-identity
pattern (2^n − 1) and prepends the identity: len(h0) == 2^n, RNG-independent.
The Monte Carlo case is RNG-dependent by nature and is compared statistically.
"""
import json, os, warnings
import numpy as np
import scipy
from scipy import sparse
import mne
from mne.stats import permutation_cluster_1samp_test, ttest_1samp_no_p

from gen_tfce_golden import adjacency

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)
TFCE_DIR = OUT


def load_tfce(name):
    with open(os.path.join(TFCE_DIR, f"tfce_{name}.json")) as f:
        return json.load(f)


def case(name, base, n, n_permutations, seed, effect, noise, start=0.4, step=0.4, e=0.5, h=2.0):
    """`base`: an L1 tfce fixture whose mask/conn/stat shape the data; subject i =
    effect * stat + noise * N(0,1)."""
    rng = np.random.default_rng(seed)
    dims = tuple(base["dims"])
    mask = np.array([c == "1" for c in base["mask"]], bool).reshape(dims)
    stat = np.asarray(base["stat"], float)
    V = stat.size
    X = effect * stat[None, :] + noise * rng.standard_normal((n, V))
    adj = sparse.coo_matrix(adjacency(mask, base["conn"]))
    thr = dict(start=start, step=step, e_power=e, h_power=h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_tfce, _, pv, h0 = permutation_cluster_1samp_test(
            X, threshold=thr, tail=1, adjacency=adj, n_permutations=n_permutations,
            seed=seed, out_type="indices", n_jobs=1, verbose=False)
    exact = (2 ** n - 1) < n_permutations
    assert len(h0) == (2 ** n if exact else n_permutations), (name, len(h0))
    t_raw = ttest_1samp_no_p(X, sigma=0)
    fx = dict(
        mne_version=mne.__version__, numpy_version=np.__version__, scipy_version=scipy.__version__,
        dims=list(dims), conn=base["conn"], mask=base["mask"],
        n=n, x=X.ravel().tolist(),
        params=dict(e=e, h=h, start=start, step=step),
        n_permutations=n_permutations, seed=seed, exact=bool(exact),
        t_obs=t_raw.tolist(),
        tfce_obs=np.abs(t_tfce).tolist(),   # MNE returns tfce * sign(t); positive tail → |.| drops the -0.0
        p_fwe=np.asarray(pv, float).tolist(),
        h0=np.sort(np.asarray(h0, float)).tolist(),
    )
    with open(os.path.join(OUT, f"perm_{name}.json"), "w") as f:
        json.dump(fx, f)
    print(f"{name:18s} n={n:2d} V={V:4d} B={len(h0):5d} exact={exact} "
          f"min p={min(pv):.4f} max tfce={np.abs(t_tfce).max():.4g}")


def main():
    case("exact_n6_grid4", load_tfce("grid4"), n=6, n_permutations=64, seed=1, effect=0.5, noise=1.0)
    case("exact_n8_blob", load_tfce("blob"), n=8, n_permutations=256, seed=2, effect=0.4, noise=1.0)
    case("mc_n20_blob", load_tfce("blob"), n=20, n_permutations=4000, seed=3, effect=0.25, noise=1.0)


if __name__ == "__main__":
    main()
