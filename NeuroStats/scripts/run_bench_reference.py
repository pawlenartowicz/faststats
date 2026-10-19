#!/usr/bin/env python3
"""Time the reference tools on the TFCE speed dataset and record their timings
and output maps. Run on a dev machine, clock locked:

    python scripts/gen_bench_dataset.py
    python scripts/run_bench_reference.py [perm] [tfce] [nilearn] [neurostats] [fsl] [palm]

Arguments pick cases (`perm`, `tfce`) and tools. With no case named, every case
is run. With no tool named, only `neurostats` is run; name the reference tools
to time them again. After each case the recorded reference timings are printed
as a ratio to neurostats. `randomise`, `fslmaths` and `palm` are looked up on
PATH; a tool that is not found is skipped. Each run rewrites only what it ran.

Both cases use TFCE with E=0.5, H=2, face (6) connectivity, positive tail, one
thread pinned to `CORE`. Each tool is run `RUNS + 1` times; the first pass is
discarded and the minimum of the rest is recorded.

Case `tfce` — the bare TFCE transform of the `N_MAPS` t-maps in `tmaps.nii`.
  nilearn    — `calculate_tfce` on the in-memory 4D array.
  neurostats — `neurostats.tfce(map, domain, params="nilearn", tail="positive")`
               per map, on the in-memory 3-D maps and a prebuilt `Domain`.
  palm       — `palm_tfce` in a loop inside Octave, timed there (no Octave
               start, maps in memory).
  fsl        — the whole `fslmaths -tfce` command (process start, reading
               `tmaps.nii`, writing the output).

Case `perm` — one-sample t, sign-flipping, `N_PERM` permutations.
  nilearn — reading and masking the data plus `permuted_ols`, inside a warm
            interpreter (no interpreter start, no imports).
  fsl     — the whole `randomise` command (process start, file reading, writing
            outputs).
  palm    — the whole `palm` command (Octave start included). PALM also
            computes voxelwise and uncorrected p-values, which it does by
            default. It runs with `-noniiclass`, which reads and writes NIfTI
            through FSL, so FSL must be on PATH for PALM too.

What was timed is recorded per tool as `timed`. The tools choose their TFCE
threshold grid themselves (about 100 steps per map), and PALM multiplies the
sum by the step where nilearn and FSL do not, so maps differ in scale.

Outputs, in `tests/fixtures/`:
  bench_reference.json — the dataset parameters and, per case and tool: seconds
                         (min), all pass times, version, command, settings.
  bench_reference.npz  — float32 maps in row-major (C) scan order of the
                         in-mask voxels, the order `neurostats::Domain::from_mask`
                         uses.
                         `tfce_<tool>`: (n_voxels, KEEP_MAPS), TFCE of the first
                         `KEEP_MAPS` t-maps.
                         `perm_<tool>_t`, `perm_<tool>_tfce`, `perm_<tool>_p_fwe`:
                         (n_voxels,); `p_fwe` is the plain FWE-corrected p-value
                         for every tool.

nilearn/FSL/PALM are generation-time tools only — never crate or CI
dependencies.
"""
import os

CORE = 1
os.sched_setaffinity(0, {CORE})  # inherited by the child processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"  # FSL's default; fixes the output format that gets timed

import glob, importlib.metadata, json, shutil, subprocess, sys, time
import numpy as np
import nibabel as nib
import scipy.io
from scipy.ndimage import generate_binary_structure

import gen_bench_dataset as ds

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.abspath(ds.OUT)
DATA = os.path.join(DATA_DIR, "data.nii")
MASK = os.path.join(DATA_DIR, "mask.nii")
TMAPS = os.path.join(DATA_DIR, "tmaps.nii")
WORK = os.path.join(DATA_DIR, "out")
FIX = os.path.join(HERE, "..", "tests", "fixtures")
JSON_PATH = os.path.join(FIX, "bench_reference.json")
NPZ_PATH = os.path.join(FIX, "bench_reference.npz")

N_PERM = 100
RUNS = 3
SEED = 1
E, H, CONN = 0.5, 2.0, 6
KEEP_MAPS = 10  # all N_MAPS maps of four tools would be ~38 MB of fixture


def timed_passes(run):
    """Call `run()` RUNS + 1 times; return (pass seconds without the first, last result)."""
    secs, result = [], None
    for _ in range(RUNS + 1):
        t0 = time.perf_counter()
        result = run()
        secs.append(time.perf_counter() - t0)
    return secs[1:], result


def shown(cmd):
    """`cmd` as recorded: paths relative to the crate directory, so the fixture
    does not carry the generating machine's layout."""
    return " ".join(cmd).replace(ROOT + os.sep, "")


def in_mask(stem, mask):
    # reshape: PALM writes a trailing singleton fourth dimension
    return np.asarray(nib.load(stem + ".nii.gz").get_fdata(), np.float32).reshape(mask.shape)[mask]


def set_fsldir(exe):
    # A conda install of FSL has FSLDIR = the environment prefix.
    os.environ.setdefault("FSLDIR", os.path.dirname(os.path.dirname(shutil.which(exe))))


def palm_dir():
    return os.path.dirname(os.path.realpath(shutil.which("palm")))


def palm_version():
    commit = subprocess.run(["git", "-C", palm_dir(), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    octave = subprocess.run(["octave", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    return f"{open(os.path.join(palm_dir(), 'palm_version.txt')).read().strip()} commit {commit}, {octave}"


# ---- case `tfce`: each runner returns (pass seconds, meta, (n_voxels, N_MAPS) maps)

def tfce_python(calculate_tfce, mask):
    arr = nib.load(TMAPS).get_fdata()
    bin_struct = generate_binary_structure(3, 1)
    secs, out = timed_passes(
        lambda: calculate_tfce(arr, bin_struct, E=E, H=H, dh="auto", two_sided_test=False))
    return secs, out[mask]


def tfce_nilearn(mask):
    import nilearn
    from nilearn.mass_univariate._utils import calculate_tfce
    secs, maps = tfce_python(calculate_tfce, mask)
    meta = dict(
        version=nilearn.__version__,
        command=f"nilearn.mass_univariate._utils.calculate_tfce(arr4d, bin_struct, E={E}, H={H}, "
                "dh='auto', two_sided_test=False)",
        timed="calculate_tfce on the in-memory 4D array",
        tfce=f"E={E} H={H} conn={CONN}, 100 thresholds per map (dh='auto')",
    )
    return secs, meta, maps


def tfce_neurostats(mask):
    import neurostats
    arr = nib.load(TMAPS).get_fdata()
    domain = neurostats.Domain.from_mask(mask, conn=CONN)
    secs, out = timed_passes(
        lambda: [neurostats.tfce(arr[..., i], domain, params="nilearn", tail="positive")
                 for i in range(arr.shape[-1])])
    meta = dict(
        version=importlib.metadata.version("neurostats"),
        command=f"neurostats.tfce(map, domain, params='nilearn', tail='positive') per map, "
                f"domain = Domain.from_mask(mask, conn={CONN})",
        timed="tfce per map on the in-memory 3-D maps, Domain prebuilt",
        tfce=f"E={E} H={H} conn={CONN}, nilearn's 100 thresholds per map",
    )
    return secs, meta, np.stack([m[mask] for m in out], axis=1)


def tfce_fsl(mask):
    set_fsldir("fslmaths")
    out = os.path.join(WORK, "tfce_fsl")
    cmd = ["fslmaths", TMAPS, "-tfce", str(H), str(E), str(CONN), out]
    secs, _ = timed_passes(lambda: subprocess.run(cmd, check=True))
    package = glob.glob(os.path.join(os.environ["FSLDIR"], "conda-meta", "fsl-avwutils-*.json"))
    meta = dict(
        version=os.path.basename(package[0])[:-len(".json")] if package else None,
        command=shown(cmd),
        timed="whole fslmaths command",
        tfce=f"E={E} H={H} conn={CONN}, fslmaths' own threshold grid",
    )
    return secs, meta, np.asarray(nib.load(out + ".nii.gz").get_fdata(), np.float32)[mask]


PALM_TFCE_M = """\
addpath('{palm_dir}'); pkg load image;
load('{inp}');  % x: (n_voxels, n_maps) in column-major scan order of the mask; mask: 3D
opts.npcmod = false; opts.MV = false; opts.forcemaskinter = false;
opts.tfce.deltah = 0; opts.tfce.conn = {conn}; opts.tfce.E = {e}; opts.tfce.H = {h};
plm.nmasks = 1; plm.masks{{1}}.data = logical(mask); plm.Yisvol = true;
out = zeros(size(x)); secs = zeros(1, {passes});
for r = 1:{passes}
  tic;
  for m = 1:columns(x)
    out(:, m) = palm_tfce(x(:, m)', 1, opts, plm);
  end
  secs(r) = toc;
end
save('-v6', '{out}', 'out', 'secs');
"""


def tfce_palm(mask):
    script, inp, out = (os.path.join(WORK, f) for f in ("tfce_palm.m", "tfce_palm_in.mat", "tfce_palm_out.mat"))
    # PALM fills the mask in column-major (Fortran) scan order; `fortran` maps
    # that order to the row-major node index used everywhere else here.
    node = np.zeros(mask.shape, np.int64)
    node[mask] = np.arange(mask.sum())
    fortran = node.T[mask.T]
    x = nib.load(TMAPS).get_fdata()[mask]
    scipy.io.savemat(inp, dict(x=x[fortran], mask=mask))
    with open(script, "w") as f:
        f.write(PALM_TFCE_M.format(palm_dir=palm_dir(), inp=inp, out=out, conn=CONN, e=E, h=H,
                                   passes=RUNS + 1))
    subprocess.run(["octave", "-q", "--no-window-system", script], check=True)
    res = scipy.io.loadmat(out)
    maps = np.empty_like(x)
    maps[fortran] = res["out"]
    meta = dict(
        version=palm_version(),
        command=f"palm_tfce(x(:, m)', 1, opts, plm) for each map, opts.tfce: deltah=0 conn={CONN} E={E} H={H}",
        timed="palm_tfce loop inside Octave (tic/toc), maps in memory",
        tfce=f"E={E} H={H} conn={CONN}, step max/100 per map, sum multiplied by the step",
    )
    return res["secs"].ravel()[1:].tolist(), meta, maps


# ---- case `perm`: each runner returns (pass seconds, meta, dict of (n_voxels,) maps)

def perm_nilearn(mask):
    import nilearn
    from nilearn.maskers import NiftiMasker
    from nilearn.mass_univariate import permuted_ols

    def run():
        masker = NiftiMasker(mask_img=MASK, standardize=None).fit()
        y = masker.transform(DATA)
        out = permuted_ols(
            np.ones((y.shape[0], 1)), y, model_intercept=False, n_perm=N_PERM,
            two_sided_test=False, random_state=SEED, n_jobs=1, masker=masker,
            tfce=True, output_type="dict")
        return masker, out

    secs, (masker, out) = timed_passes(run)

    def nodes(a):  # masker order -> C-scan order of the in-mask voxels
        return np.asarray(masker.inverse_transform(a).get_fdata(), np.float32)[..., 0][mask]

    meta = dict(
        version=nilearn.__version__,
        command=f"permuted_ols(ones, y, model_intercept=False, n_perm={N_PERM}, "
                f"two_sided_test=False, random_state={SEED}, n_jobs=1, masker=masker, tfce=True)",
        timed="NiftiMasker fit + transform + permuted_ols, warm interpreter",
        tfce=f"E={E} H={H} conn={CONN}, 100 thresholds (dh='auto')",
    )
    maps = dict(t=nodes(out["t"]), tfce=nodes(out["tfce"]),
                p_fwe=10.0 ** -nodes(out["logp_max_tfce"]))
    return secs, meta, maps


def perm_fsl(mask):
    set_fsldir("randomise")
    root = os.path.join(WORK, "fsl")
    cmd = ["randomise", "-i", DATA, "-o", root, "-m", MASK, "-1", "-T", "-R", "-n", str(N_PERM),
           f"--tfce_H={H}", f"--tfce_E={E}", f"--tfce_C={CONN}", f"--seed={SEED}", "--quiet"]
    secs, _ = timed_passes(lambda: subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL))
    banner = subprocess.run(["randomise"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True).stdout.split("\n")
    meta = dict(
        version=" ".join(line.strip() for line in banner[:3] if line.strip()),
        command=shown(cmd),
        timed="whole randomise command",
        tfce=f"E={E} H={H} conn={CONN}, randomise's own threshold grid",
    )
    maps = dict(t=in_mask(root + "_tstat1", mask),
                tfce=in_mask(root + "_tfce_tstat1", mask),
                p_fwe=1.0 - in_mask(root + "_tfce_corrp_tstat1", mask))  # stored as 1-p
    return secs, meta, maps


def perm_palm(mask):
    root = os.path.join(WORK, "palm")
    # Explicit design and contrast: with none given PALM tests both the positive
    # and the negative contrast, twice the work of the other tools.
    design, contrast = os.path.join(WORK, "design.csv"), os.path.join(WORK, "contrast.csv")
    np.savetxt(design, np.ones(ds.N), fmt="%d")
    np.savetxt(contrast, [1], fmt="%d")
    cmd = ["palm", "-i", DATA, "-m", MASK, "-d", design, "-t", contrast, "-o", root,
           "-n", str(N_PERM), "-ise", "-T", "-tfce_H", str(H), "-tfce_E", str(E),
           "-tfce_C", str(CONN), "-seed", str(SEED), "-noniiclass"]
    secs, _ = timed_passes(lambda: subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    meta = dict(
        version=palm_version(),
        command=shown(cmd),
        timed="whole palm command",
        tfce=f"E={E} H={H} conn={CONN}, tfce_dh auto",
    )
    maps = dict(t=in_mask(root + "_vox_tstat", mask),
                tfce=in_mask(root + "_tfce_tstat", mask),
                p_fwe=in_mask(root + "_tfce_tstat_fwep", mask))
    return secs, meta, maps


def on_path(exe):
    return lambda: shutil.which(exe) is not None


# case -> (unit that is counted, number of units, {tool: (runner, is available)})
CASES = {
    "tfce": ("map", ds.N_MAPS, {
        "nilearn": (tfce_nilearn, lambda: True),
        "neurostats": (tfce_neurostats, lambda: True),
        "fsl": (tfce_fsl, on_path("fslmaths")),
        "palm": (tfce_palm, on_path("palm")),
    }),
    "perm": ("perm", N_PERM, {
        "nilearn": (perm_nilearn, lambda: True),
        "fsl": (perm_fsl, on_path("randomise")),
        "palm": (perm_palm, on_path("palm")),
    }),
}


def main():
    args = sys.argv[1:]
    cases = [a for a in args if a in CASES] or list(CASES)
    wanted = [a for a in args if a not in CASES] or ["neurostats"]
    os.makedirs(WORK, exist_ok=True)
    mask = nib.load(MASK).get_fdata() > 0

    report = json.load(open(JSON_PATH)) if os.path.exists(JSON_PATH) else {}
    maps = dict(np.load(NPZ_PATH)) if os.path.exists(NPZ_PATH) else {}
    report["dataset"] = dict(
        script="scripts/gen_bench_dataset.py", seed=ds.SEED, dims=list(ds.DIMS), n=ds.N,
        n_voxels=int(mask.sum()), sigma=ds.SIGMA, effect=ds.EFFECT, n_maps=ds.N_MAPS)
    settings = dict(e=E, h=H, conn=CONN, tail="positive", core=CORE, runs=RUNS)

    for case in cases:
        unit, count, tools = CASES[case]
        entry = report.setdefault(case, {})
        entry["test"] = dict(settings, **{f"n_{unit}s": count})
        if case == "perm":
            entry["test"]["design"] = "one-sample t, sign-flipping"
        else:
            entry["test"]["keep_maps"] = KEEP_MAPS
        for name in wanted:
            if name not in tools:
                continue
            run, available = tools[name]
            if not available():
                print(f"{case:5s} {name:10s} not found — skipped")
                continue
            secs, meta, tool_maps = run(mask)
            entry.setdefault("tools", {})[name] = dict(seconds=min(secs), pass_seconds=secs, **meta)
            if case == "perm":
                for key, arr in tool_maps.items():
                    maps[f"perm_{name}_{key}"] = np.asarray(arr, np.float32)
            else:
                maps[f"tfce_{name}"] = np.asarray(tool_maps[:, :KEEP_MAPS], np.float32)
            print(f"{case:5s} {name:10s} min {min(secs):8.3f} s = {1e3 * min(secs) / count:8.2f} ms/{unit}   "
                  f"passes {[round(s, 3) for s in secs]}")
        rows = entry.get("tools", {})
        if "neurostats" in rows:
            for name, row in rows.items():
                if name != "neurostats":
                    print(f"{case:5s} {name:10s} recorded {row['seconds']:8.3f} s = "
                          f"{row['seconds'] / rows['neurostats']['seconds']:5.1f}x neurostats")

    with open(JSON_PATH, "w") as f:
        json.dump(report, f, indent=1)
    np.savez_compressed(NPZ_PATH, **maps)


if __name__ == "__main__":
    main()
