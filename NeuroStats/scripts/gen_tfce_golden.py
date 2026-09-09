#!/usr/bin/env python3
"""Generate neurostats TFCE golden fixtures from MNE-Python. Run on a dev machine:

    python scripts/gen_tfce_golden.py

MNE/NumPy/SciPy are generation-time tools only — never crate or CI dependencies
(CI runs `cargo test` against the committed fixtures).

Per case the fixture stores the input (dims, conn, mask, stat, params), MNE's
enhanced map (`expected_mne`, from `mne.stats.cluster_level._find_clusters`,
tail=1, cross-checked once against the public
`permutation_cluster_1samp_test` route), an independent NumPy transcription of
the stepped Smith–Nichols sum (`expected_smith_nichols`, same thresholds and
strict membership, weight h_i^H * step — a self-reference, not an oracle), and
MNE's best-of-3 wall time (`mne_seconds`).

Fixture node order = row-major (C) scan of the in-mask voxels, the same order
`neurostats::Domain::from_mask` uses.
"""
import json, os, time, warnings
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import mne
from mne.stats.cluster_level import _find_clusters
from mne.stats import permutation_cluster_1samp_test

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

OFFSETS = {
    6: [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if (dx != 0) + (dy != 0) + (dz != 0) == 1],
    18: [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
         if 1 <= (dx != 0) + (dy != 0) + (dz != 0) <= 2],
    26: [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
         if (dx, dy, dz) != (0, 0, 0)],
}


def adjacency(mask, conn):
    """Sparse COO adjacency over in-mask voxels, node = compacted C-order ordinal."""
    dims = mask.shape
    node_of = -np.ones(mask.shape, dtype=np.int64)
    node_of[mask] = np.arange(mask.sum())
    rows, cols = [], []
    xs, ys, zs = np.nonzero(mask)
    for dx, dy, dz in OFFSETS[conn]:
        qx, qy, qz = xs + dx, ys + dy, zs + dz
        ok = (qx >= 0) & (qy >= 0) & (qz >= 0) & (qx < dims[0]) & (qy < dims[1]) & (qz < dims[2])
        q = node_of[qx[ok], qy[ok], qz[ok]]
        keep = q >= 0
        rows.append(node_of[xs[ok], ys[ok], zs[ok]][keep])
        cols.append(q[keep])
    n = int(mask.sum())
    r = np.concatenate(rows) if rows else np.array([], int)
    c = np.concatenate(cols) if cols else np.array([], int)
    return sparse.coo_array((np.ones(len(r)), (r, c)), shape=(n, n))


def smith_nichols(stat, adj, start, step, e, h):
    """Independent stepped Smith–Nichols sum: sum_i h_i^H * step * extent_i^E,
    thresholds np.arange(start, max, step), strict membership."""
    ths = np.arange(start, stat.max(), step)
    out = np.zeros(stat.size)
    adj = adj.tocsr()
    for th in ths:
        m = stat > th
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        sub = adj[idx][:, idx]
        _, lab = connected_components(sub, directed=False)
        sizes = np.bincount(lab)
        out[idx] += th ** h * step * sizes[lab] ** e
    return out


def mne_tfce(stat, adj, start, step, e, h):
    thr = dict(start=start, step=step, e_power=e, h_power=h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            _, scores = _find_clusters(stat, thr, tail=1, adjacency=adj)
            best = min(best, time.perf_counter() - t0)
    return np.asarray(scores, float), best


def mne_public_route(stat, adj, start, step, e, h):
    """Public API cross-check: one 'sample' whose stat_fun returns the map itself."""
    thr = dict(start=start, step=step, e_power=e, h_power=h)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_obs, _, _, _ = permutation_cluster_1samp_test(
            stat[None, :], threshold=thr, tail=1, adjacency=sparse.coo_matrix(adj),
            n_permutations=1, stat_fun=lambda X: X[0], out_type="indices", verbose=False)
    return np.asarray(t_obs, float)


def case(name, mask, conn, stat, start, step, e=0.5, h=2.0, public_check=True):
    mask = np.asarray(mask, bool)
    stat = np.asarray(stat, float)
    assert stat.size == mask.sum()
    adj = adjacency(mask, conn)
    exp_mne, secs = mne_tfce(stat, adj, start, step, e, h)
    if public_check and stat.size > 0:
        pub = mne_public_route(stat, adj, start, step, e, h)
        assert np.allclose(pub, exp_mne, rtol=0, atol=0), f"{name}: public route != _find_clusters"
    exp_sn = smith_nichols(stat, adj, start, step, e, h)
    fx = dict(
        mne_version=mne.__version__, numpy_version=np.__version__, scipy_version=scipy.__version__,
        dims=list(mask.shape), conn=conn,
        mask="".join("1" if m else "0" for m in mask.ravel()),
        stat=stat.tolist(),
        params=dict(e=e, h=h, start=start, step=step),
        n_thresholds=int(len(np.arange(start, stat.max(), step))) if stat.size else 0,
        expected_mne=exp_mne.tolist(),
        expected_smith_nichols=exp_sn.tolist(),
        mne_seconds=secs,
    )
    path = os.path.join(OUT, f"tfce_{name}.json")
    with open(path, "w") as f:
        json.dump(fx, f)
    print(f"{name:14s} V={stat.size:6d} conn={conn:2d} thresholds={fx['n_thresholds']:4d} "
          f"mne={secs:.4f}s max_mne={exp_mne.max():.4g}")


def main():
    rng = np.random.default_rng(20260905)

    # 4x4x4 dense grid, face connectivity, smooth blob + noise.
    g = np.indices((4, 4, 4)).astype(float)
    blob = 4.0 * np.exp(-((g[0] - 1.5) ** 2 + (g[1] - 1.5) ** 2 + (g[2] - 1.5) ** 2) / 2.0)
    case("grid4", np.ones((4, 4, 4), bool), 6, (blob + rng.normal(0, 0.3, blob.shape)).ravel(), 0.4, 0.4)

    # Irregular mask on 6x5x7, vertex connectivity, two blobs.
    dims = (6, 5, 7)
    mask = rng.random(dims) < 0.7
    mask[0, 0, 0] = True
    g = np.indices(dims).astype(float)
    s = (3.0 * np.exp(-((g[0] - 1) ** 2 + (g[1] - 1) ** 2 + (g[2] - 1) ** 2) / 1.5)
         + 5.0 * np.exp(-((g[0] - 4) ** 2 + (g[1] - 3) ** 2 + (g[2] - 5) ** 2) / 2.5)
         + rng.normal(0, 0.4, dims))
    case("blob", mask, 26, s[mask], 0.4, 0.4)

    # Ties: stat quantized to 4 levels so many nodes share values; edge connectivity.
    dims = (5, 5, 5)
    q = np.round(rng.random(dims) * 3) * 0.9
    case("ties", np.ones(dims, bool), 18, q.ravel(), 0.3, 0.6)

    # Flat map: all equal -> one component at every threshold.
    case("flat", np.ones((3, 3, 3), bool), 6, np.full(27, 2.0), 0.4, 0.4)

    # Single node.
    case("single", np.ones((1, 1, 1), bool), 26, [1.7], 0.5, 0.5)

    # start above max: empty threshold grid -> all zeros (MNE warns).
    case("empty_grid", np.ones((2, 2, 2), bool), 6, rng.random(8), 2.0, 0.5, public_check=False)

    # Two separated clusters of different size on a line-like mask (checks extent^E).
    dims = (9, 1, 1)
    case("two_clusters", np.ones(dims, bool), 6, [3, 3, 3, 3, 0, 0, 3, 3, 0.1], 0.5, 0.5)

    # Realistic bench mask: sphere in a 50^3 grid (~58k in-mask voxels), vertex
    # connectivity, smooth random field. Same fixture drives benches/tfce_bench.rs.
    dims = (50, 50, 50)
    g = np.indices(dims).astype(float) - 24.5
    sphere = (g ** 2).sum(0) <= 24.0 ** 2
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(rng.normal(0, 1, dims), sigma=2.5)
    field = field / field[sphere].std() * 1.5 + 1.0
    case("bench", sphere, 26, field[sphere], 0.4, 0.4, public_check=False)


if __name__ == "__main__":
    main()
