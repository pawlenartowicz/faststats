import subprocess
import sys

# Run in a fresh interpreter where `import nilearn` fails, as it does in an
# environment that does not have it installed.
CODE = """
import sys
sys.modules["nilearn"] = None
import numpy as np
import neurostats as ns

dom = ns.Domain.from_mask(np.ones((3, 3, 2), dtype=bool))
stat = np.arange(18, dtype=float) - 6.0
grid = {"weighting": "mne", "start": 0.0, "step": 0.5}
for params in ("exact", "nilearn", grid):
    for tail in ("positive", "negative", "two_sided"):
        assert ns.tfce(stat, dom, params=params, tail=tail).shape == (18,)
ns.grid_and_weights(3.0, {"weighting": "smith_nichols", "start": 0.0, "step": 0.5})
rng = np.random.default_rng(0)
y = rng.standard_normal((8, 18))
r = ns.tfce_ols(y, dom, tested=rng.standard_normal(8), confounds=rng.standard_normal((8, 1)),
                seed=1, n_perm=20, n_jobs=1)
ns.tfce_ols(y, dom, tested="intercept", params="nilearn", tail="positive", n_perm=20)
r.save(sys.argv[1])
back = ns.TfceInference.load(sys.argv[1])
assert back.n_perm == 20 and back.origin is None
d = ns.import_from(y, dom)
assert d.y is y and d.domain is dom and d.origin is None
assert ns.import_from(d) is d
assert ns.tfce_ols(d, tested="intercept", n_perm=20).origin is None
try:
    ns.export_to(r)
except ValueError:
    pass
else:
    raise AssertionError("a result with no origin was exported without a target")
print("ok")
"""


def test_plain_array_paths_run_without_nilearn(tmp_path):
    r = subprocess.run([sys.executable, "-c", CODE, str(tmp_path / "result.npz")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
