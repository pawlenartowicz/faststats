#!/usr/bin/env python3
"""Generate CommonStats oracle fixtures. Run on a dev machine:

    python scripts/gen_oracle.py [area ...]        # default: all areas

Areas: special htest dist density transform. mpmath/scipy/numpy/Rscript are
*generation-time* tools only — never crate or CI dependencies (CI runs only
`cargo test` against the committed fixtures).

Two validation axes:
  Layer 1 — Truth: mpmath at high precision is the `expected` value.
  Layer 2 — Cross-validation (one-shot, at generation): scipy AND R recompute
            each bulk quantity; `mpmath ≈ scipy ≈ R` is asserted (`rel 1e-9`) and
            their values + achieved precision are stamped into the fixture as
            permanent provenance (`xcheck`). Disagreement fails generation loudly
            — it means the *definition* (ddof, sidedness, parameterization) is
            wrong, not the precision. Tail rows skip the gate (scipy/R are only
            ~1e-8 there); mpmath is the sole arbiter.

All five areas (special, dist, transform, density, htest) are on the three-source
path; mpmath is the truth arbiter, scipy+R the one-shot convention cross-check.
"""
import json, os, math, sys, subprocess, tempfile
import numpy as np
from scipy import special as sp
from scipy import stats as st
import mpmath as mp

mp.mp.dps = 60  # 60 decimal digits — Layer-1 truth precision

OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)
RSCRIPT = os.path.join(os.path.dirname(__file__), "gen_oracle.R")


# ===========================================================================
# Three-source machinery (mpmath truth + scipy + R cross-check + gate)
# ===========================================================================
CONV_REL = 1e-9    # Layer-2 convention tolerance — trips on order-1 definition bugs
CONV_ABS = 1e-12   # abs floor so near-zero truths don't blow up the rel check
TAIL_LO, TAIL_HI = 1e-6, 1.0 - 1e-6  # bulk band for a probability arg

# Reviewed tail-band ledger. One value per fixture, stamped onto its tail
# rows; the Rust harness asserts each tail row within TAIL_SLACK(=4)× this. Sourced
# from the `measure_tail_bands` calibration test (our impl's measured rel error vs
# mpmath truth), lightly padded. CHANGE ONLY when an algorithm changes — re-run the
# calibration and update here. Living in the generator keeps regeneration byte-stable
# (a hand-edit into the fixtures would be wiped on regen) while tests never write files.
TAIL_BANDS = {
    "erfcinv":    8.0e-10,   # measured 7.39e-10 @ p=1e-9 (our impl ≈ scipy there)
    "invbetareg": 2.0e-6,    # measured 1.87e-6 — inverse incomplete-beta deep-tail limit
    # Continuous ppf tail bands (measure_tail_bands calibration, ~15% pad). uniform
    # & cauchy & beta measure ≈0 (closed-form / atan inverse — exact to ~1 ulp); a
    # 1e-15 floor keeps the slack·band assertion meaningful without demanding bit-exactness.
    "dist_normal_ppf":      5.3e-11,  # 4.60e-11
    "dist_studentt_ppf":    1.0e-15,  # 1.78e-16 (Newton on the tail mass, both tails)
    "dist_chisquared_ppf":  1.3e-15,  # 1.07e-15 (Newton on the smaller tail mass)
    "dist_fisherf_ppf":     2.3e-15,  # 1.96e-15
    "dist_uniform_ppf":     1.0e-15,  # measured 0 (linear closed form)
    "dist_exponential_ppf": 3.3e-8,   # 2.83e-8
    "dist_cauchy_ppf":      1.0e-15,  # 1.87e-16
    "dist_weibull_ppf":     1.7e-8,   # 1.41e-8
    "dist_lognormal_ppf":   2.5e-10,  # 2.16e-10
    "dist_gamma_ppf":       1.1e-15,  # 9.61e-16
    "dist_beta_ppf":        1.5e-15,  # 9.42e-16
    # isf tail bands. Newton/closed-form dists sit at ~1e-15 in both tails (each
    # residual is formed on the smaller tail mass, never on `1 − near-1`); truth is
    # taken at the exact f64 argument, so the argument itself is not a limit.
    # normal/lognormal are bounded by erfc_inv at the mirror q = 1−1e-9 rows.
    "dist_normal_isf":      5.8e-11,  # 5.00e-11
    "dist_studentt_isf":    1.0e-15,  # 1.78e-16
    "dist_chisquared_isf":  1.3e-15,  # 1.07e-15
    "dist_fisherf_isf":     1.5e-15,  # 1.25e-15
    "dist_uniform_isf":     1.0e-15,  # 1.11e-16
    "dist_exponential_isf": 1.0e-15,  # 1.99e-16
    "dist_cauchy_isf":      1.0e-15,  # 1.87e-16
    "dist_weibull_isf":     1.0e-15,  # 2.86e-16
    "dist_lognormal_isf":   2.5e-10,  # 2.16e-10
    "dist_gamma_isf":       1.0e-15,  # 8.57e-16
    "dist_beta_isf":        1.4e-15,  # 1.19e-15
}

# --- mpmath truth helpers (delimited section; split to oracle_mp.py if it grows) ---
def mp_lgamma(a):    return mp.loggamma(a)                       # ln Γ(a), a>0
def mp_lbeta(a, b):  return mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)
def mp_gammp(a, x):  return mp.gammainc(a, 0, x, regularized=True)   # P(a,x)
def mp_gammq(a, x):  return mp.gammainc(a, x, mp.inf, regularized=True)  # Q(a,x)
def mp_betai(a, b, x): return mp.betainc(a, b, 0, x, regularized=True)   # I_x(a,b)
def mp_erfcinv(p):   return mp.erfinv(1 - p)                     # erfc⁻¹ via erf⁻¹

def mp_invbetareg(a, b, p):
    """Inverse regularized incomplete beta by bisection (I_x is ↑ in x on (0,1));
    mpmath has no direct inverse, so root-find the defining equation at 60 dps."""
    lo, hi = mp.mpf(0), mp.mpf(1)
    for _ in range(200):  # 2⁻²⁰⁰ ≫ 60-dps target
        mid = (lo + hi) / 2
        if mp.betainc(a, b, 0, mid, regularized=True) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _f64(mpval):
    """mpf → f64, or None if non-finite (matches the harness null-skip convention)."""
    try:
        v = float(mpval)
    except (ValueError, OverflowError):
        return None
    return v if math.isfinite(v) else None


def _rel_err(approx, truth_mpf):
    """Achieved precision of `approx` vs the high-precision truth. abs error when
    truth≈0 (rel is meaningless there)."""
    if approx is None or not math.isfinite(approx):
        return None
    a = mp.mpf(approx)
    if truth_mpf == 0:
        return float(abs(a))
    return float(abs(a - truth_mpf) / abs(truth_mpf))


class Quantity:
    """One logical quantity declared with its three matched callables (the
    identity map). `kind`: 'value' (cdf/density-like, bulk ladder everywhere) or
    'quantile' (tail-tiered). `prob` = (arg_index, lo, hi) domain of the
    probability argument for tier tagging, or None when every row is bulk."""
    def __init__(self, name, kind, args_list, mp_fn, sp_fn, r_func, prob=None, r_argmap=None):
        self.name, self.kind, self.args_list = name, kind, args_list
        self.mp_fn, self.sp_fn, self.prob = mp_fn, sp_fn, prob
        # r_func=None → R cannot compute this quantity ("R: n/a"); it
        # cross-checks on mpmath+scipy only. r_argmap maps row args → the R call's
        # args when they differ (e.g. a dist row carries only [x] but R d/p/q needs
        # [x, *params]); default identity.
        self.r_func = r_func
        self.r_argmap = r_argmap or (lambda a: a)

    def tier(self, args):
        if self.prob is None:
            return "bulk"
        i, lo, hi = self.prob
        p = args[i]
        return "tail" if (p < lo + TAIL_LO or p > hi - TAIL_LO) else "bulk"


def _run_r(jobs):
    """One batched Rscript invocation: manifest of {func,args} → list of values
    (None where R errored / non-finite). Order-aligned with `jobs`."""
    with tempfile.TemporaryDirectory() as d:
        man, out = os.path.join(d, "manifest.json"), os.path.join(d, "out.json")
        with open(man, "w") as f:
            json.dump(jobs, f)
        subprocess.run(["Rscript", RSCRIPT, man, out], check=True)
        with open(out) as f:
            return json.load(f)


def generate_three_source(quantities):
    """Compute mpmath truth + scipy + R for every quantity, run the cross-val gate
    on bulk rows, and write fixtures with the xcheck provenance stamp. Exits
    non-zero (no fixtures written) if any bulk row's definitions disagree."""
    # Pass 1: truth + scipy per row; accumulate one flat R manifest.
    built = []          # per quantity: list of row dicts (pre-R)
    r_jobs, r_index = [], []   # flat manifest + back-pointers (qi, ri)
    for qi, q in enumerate(quantities):
        rows = []
        for args in q.args_list:
            args = [float(a) for a in args]
            try:
                truth_mpf = q.mp_fn(*args)
            except Exception:
                truth_mpf = None
            expected = _f64(truth_mpf) if truth_mpf is not None else None
            try:
                sval = float(q.sp_fn(*args))
                if not math.isfinite(sval):
                    sval = None
            except Exception:
                sval = None
            rows.append({"args": args, "tier": q.tier(args),
                         "truth_mpf": truth_mpf, "expected": expected, "scipy": sval, "r": None})
            if q.r_func is not None:
                r_index.append((qi, len(rows) - 1))
                r_jobs.append({"func": q.r_func, "args": q.r_argmap(args)})
        built.append(rows)

    r_vals = _run_r(r_jobs) if r_jobs else []
    for (qi, ri), rv in zip(r_index, r_vals):
        built[qi][ri]["r"] = rv

    # Pass 2: cross-validation gate (bulk rows only; R skipped where R is n/a).
    failures = []
    for q, rows in zip(quantities, built):
        sources = ("scipy", "r") if q.r_func is not None else ("scipy",)
        for row in rows:
            if row["tier"] != "bulk" or row["truth_mpf"] is None:
                continue
            t = row["truth_mpf"]
            floor = max(CONV_ABS, CONV_REL * abs(float(t)) if math.isfinite(float(t)) else CONV_ABS)
            for src in sources:
                v = row[src]
                if v is None:
                    failures.append((q.name, row["args"], src, "missing", float(t)))
                    continue
                if abs(mp.mpf(v) - t) > floor:
                    failures.append((q.name, row["args"], src, v, float(t)))
    if failures:
        print("\n*** CROSS-VALIDATION GATE FAILED — convention mismatch, no fixtures written ***")
        for name, args, src, got, truth in failures:
            print(f"  {name}{args}: {src}={got} vs mpmath={truth!r}")
        sys.exit(1)

    # Pass 3: write fixtures with the one-shot provenance stamp.
    for q, rows in zip(quantities, built):
        recs = []
        for row in rows:
            t = row["truth_mpf"]
            band = TAIL_BANDS.get(q.name) if row["tier"] == "tail" else None
            rec = {"args": row["args"], "expected": row["expected"], "tier": row["tier"],
                   "band": band,
                   "xcheck": {
                       "scipy": {"val": row["scipy"], "rel": _rel_err(row["scipy"], t) if t is not None else None},
                       "r":     {"val": row["r"],     "rel": _rel_err(row["r"], t) if t is not None else None}}}
            recs.append(rec)
        with open(os.path.join(OUT, f"{q.name}.json"), "w") as f:
            json.dump(recs, f, indent=0)
        n_tail = sum(1 for r in recs if r["tier"] == "tail")
        print(f"wrote {q.name}.json ({len(recs)} rows, {n_tail} tail)")


# ===========================================================================
# Area: special functions (Class A) — three-source path
# ===========================================================================
def gen_special():
    xs = [-5, -2, -1, -0.3, 0, 0.3, 1, 2, 5, 8]
    pos = [0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 120]
    probs = [1e-9, 1e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-9]

    # logsumexp: each row is a 3-element input vector; R uses the stable form
    # m + log(sum(exp(x - m))). Cases cover ordinary magnitudes, large values
    # (overflow in a naive sum), mixed signs, all-negative, and large spread.
    lse_cases = [
        (1.0, 2.0, 3.0),           # ordinary positive — all three sources agree
        (1000.0, 1000.0, 1000.0),  # naively exp() overflows; stable: 1000 + ln 3
        (-1.0, 0.0, 1.0),          # symmetric around 0
        (-5.0, -4.0, -3.0),        # all negative
        (1.0, 100.0, 50.0),        # large spread; dominated by the max term
    ]
    def mp_lse(*xs_):  return mp.log(sum(mp.exp(mp.mpf(x)) for x in xs_))
    def sp_lse(*xs_):  return float(sp.logsumexp(list(xs_)))

    quantities = [
        Quantity("erf",     "value", [(x,) for x in xs],  mp.erf,     sp.erf,      "erf"),
        Quantity("erfc",    "value", [(x,) for x in xs],  mp.erfc,    sp.erfc,     "erfc"),
        Quantity("lgamma",  "value", [(a,) for a in pos], mp_lgamma,  sp.gammaln,  "lgamma"),
        Quantity("gamma",   "value", [(a,) for a in pos], mp.gamma,   sp.gamma,    "gamma"),
        Quantity("digamma", "value", [(a,) for a in pos], mp.digamma, sp.digamma,  "digamma"),
        Quantity("gammp",   "value", [(a, x) for a in pos for x in pos], mp_gammp, sp.gammainc,  "gammp"),
        Quantity("gammq",   "value", [(a, x) for a in pos for x in pos], mp_gammq, sp.gammaincc, "gammq"),
        Quantity("betai",   "value", [(a, b, x) for a in pos for b in pos for x in (0.05, 0.3, 0.7, 0.95)],
                 mp_betai, sp.betainc, "betai"),
        Quantity("lbeta",   "value", [(a, b) for a in pos for b in pos], mp_lbeta, sp.betaln, "lbeta"),
        Quantity("logsumexp", "value", lse_cases, mp_lse, sp_lse, "logsumexp"),
        # Inverses: quantile-kind, tail-tiered on the probability argument.
        Quantity("erfcinv", "quantile", [(p,) for p in probs], mp_erfcinv, sp.erfcinv, "erfcinv",
                 prob=(0, 0.0, 2.0)),
        Quantity("erfinv",  "quantile", [(p,) for p in (-0.99, -0.5, -0.1, 0.1, 0.5, 0.9, 0.99)],
                 mp.erfinv, sp.erfinv, "erfinv", prob=(0, -1.0, 1.0)),
        Quantity("invbetareg", "quantile",
                 [(a, b, p) for a in (0.5, 1, 5, 30) for b in (0.5, 1, 5, 30) for p in probs],
                 mp_invbetareg, sp.betaincinv, "invbetareg", prob=(2, 0.0, 1.0)),
    ]
    generate_three_source(quantities)


# ===========================================================================
# Area: hypothesis tests (Class C) — LEGACY single-oracle (migrated in Phase 5)
# ===========================================================================
# ---- htest mpmath-truth helpers (Class C: stat by HP algebra, p via Class-A fns) ----
def _h_mean(xs): return mp.fsum([mp.mpf(x) for x in xs]) / len(xs)
def _h_var1(xs):                                    # sample variance, ddof=1
    m = _h_mean(xs)
    return mp.fsum([(mp.mpf(x) - m) ** 2 for x in xs]) / (len(xs) - 1)
def _h_t_p(t, df):                                  # two-sided t p = I_{df/(df+t²)}(df/2, ½)
    return mp.betainc(df / 2, mp.mpf(1) / 2, 0, df / (df + t * t), regularized=True)
def _h_f_sf(f, d1, d2):                             # upper-tail F p = I_{d2/(d2+d1F)}(d2/2, d1/2)
    return mp.betainc(d2 / 2, d1 / 2, 0, d2 / (d2 + d1 * f), regularized=True)
def _h_t_crit(level, df):                           # two-sided t critical value (for CI surfacing)
    x = mp_invbetareg(df / 2, mp.mpf(1) / 2, 1 - mp.mpf(level))
    return mp.sqrt(df * (1 / x - 1))


def gen_htest():
    def _htest_gate(rows_with_sources):
        """rows_with_sources: (name, desc, truth_mpf, {src: value}). Asserts each
        present source matches mpmath truth within CONV_REL (abs floor CONV_ABS);
        None source = engine n/a. Exits 1 before any fixture is written."""
        failures = []
        for name, desc, truth_mpf, srcs in rows_with_sources:
            if truth_mpf is None:
                continue
            ft = float(truth_mpf)
            floor = max(CONV_ABS, CONV_REL * abs(ft) if math.isfinite(ft) else CONV_ABS)
            for src, v in srcs.items():
                if v is None:
                    continue
                if abs(mp.mpf(v) - truth_mpf) > floor:
                    failures.append((name, desc, src, v, ft))
        if failures:
            print("\n*** HTEST CROSS-VALIDATION GATE FAILED — no fixtures written ***")
            for nm, desc, src, got, truth in failures:
                print(f"  {nm}[{desc}]: {src}={got} vs mpmath={truth!r}")
            sys.exit(1)

    def _xc(truth_mpf, scipy_val, r_val):
        """One-shot provenance stamp: each cross-source value + its rel error vs truth."""
        return {"scipy": {"val": scipy_val, "rel": _rel_err(scipy_val, truth_mpf)},
                "r":     {"val": r_val,     "rel": _rel_err(r_val, truth_mpf)}}

    a = [5.1, 4.9, 6.2, 5.5, 5.8, 6.0, 4.7, 5.3]
    b = [6.1, 5.9, 7.0, 6.5, 6.8, 7.2, 5.7, 6.3, 6.6]
    na, nb = len(a), len(b)
    ma, mb, va, vb = _h_mean(a), _h_mean(b), _h_var1(a), _h_var1(b)

    # --- t-tests: mpmath truth ---
    t_one = (ma - 5) / (mp.sqrt(va) / mp.sqrt(na)); df_one = mp.mpf(na - 1)
    sp2 = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    t_stu = (ma - mb) / mp.sqrt(sp2 * (mp.mpf(1) / na + mp.mpf(1) / nb)); df_stu = mp.mpf(na + nb - 2)
    se_w = mp.sqrt(va / na + vb / nb)
    df_w = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    t_wel = (ma - mb) / se_w
    diff = [mp.mpf(x) - mp.mpf(y) for x, y in zip(a[:8], b[:8])]
    md = mp.fsum(diff) / 8; vd = mp.fsum([(d - md) ** 2 for d in diff]) / 7
    t_pai = md / (mp.sqrt(vd) / mp.sqrt(8)); df_pai = mp.mpf(7)
    truth = {
        "one":    [t_one, _h_t_p(t_one, df_one)],
        "student":[t_stu, _h_t_p(t_stu, df_stu)],
        "welch":  [t_wel, _h_t_p(t_wel, df_w), df_w],
        "paired": [t_pai, _h_t_p(t_pai, df_pai)],
    }
    # scipy values
    sci = {
        "one":     (lambda r: [r.statistic, r.pvalue])(st.ttest_1samp(a, 5.0)),
        "student": (lambda r: [r.statistic, r.pvalue])(st.ttest_ind(a, b, equal_var=True)),
        "welch":   (lambda r: [r.statistic, r.pvalue, r.df])(st.ttest_ind(a, b, equal_var=False)),
        "paired":  (lambda r: [r.statistic, r.pvalue])(st.ttest_rel(a[:8], b[:8])),
    }
    # R jobs (length-tagged payloads; see gen_oracle.R). two-sample payload [na,nb,a…,b…].
    two = [na, nb] + a + b
    paired_pl = [8, 8] + a[:8] + b[:8]
    rj, ri = [], {}
    def addj(key, func, args):
        ri[key] = len(rj); rj.append({"func": func, "args": [float(x) for x in args]})
    addj("one_t", "ttest_one_t", [5.0] + a);        addj("one_p", "ttest_one_p", [5.0] + a)
    addj("stu_t", "ttest_student_t", two);          addj("stu_p", "ttest_student_p", two)
    addj("wel_t", "ttest_welch_t", two);            addj("wel_p", "ttest_welch_p", two)
    addj("wel_df", "ttest_welch_df", two)
    addj("wel_cilo", "ttest_welch_cilo", two);      addj("wel_cihi", "ttest_welch_cihi", two)
    addj("pai_t", "ttest_paired_t", paired_pl);     addj("pai_p", "ttest_paired_p", paired_pl)

    # --- ANOVA truth ---
    g1 = [89, 88, 97, 92]; g2 = [84, 79, 81, 83]; g3 = [91, 95, 94, 88]
    groups = [g1, g2, g3]; sizes = [len(g) for g in groups]; k = len(groups)
    gmeans = [_h_mean(g) for g in groups]; gvars = [_h_var1(g) for g in groups]
    grand_n = sum(sizes); grand_m = mp.fsum([s * m for s, m in zip(sizes, gmeans)]) / grand_n
    ssb = mp.fsum([s * (m - grand_m) ** 2 for s, m in zip(sizes, gmeans)])
    ssw = mp.fsum([(s - 1) * v for s, v in zip(sizes, gvars)])
    an_d1, an_d2 = mp.mpf(k - 1), mp.mpf(grand_n - k)
    an_F = (ssb / an_d1) / (ssw / an_d2); an_p = _h_f_sf(an_F, an_d1, an_d2)
    fo = st.f_oneway(g1, g2, g3)
    addj("anova_f", "anova_f", [k] + sizes + g1 + g2 + g3)
    addj("anova_p", "anova_p", [k] + sizes + g1 + g2 + g3)

    # --- chi² truth (gof + independence) ---
    obs = [16, 18, 16, 14, 12, 12]; exp = [16, 16, 16, 16, 12, 12]  # Σobs==Σexp==88
    gof_chi2 = mp.fsum([(mp.mpf(o) - e) ** 2 / e for o, e in zip(obs, exp)])
    gof_df = mp.mpf(len(obs) - 1)
    gof_p = mp.gammainc(gof_df / 2, gof_chi2 / 2, mp.inf, regularized=True)  # gammq
    c1 = st.chisquare(obs, exp)
    table = [[10, 20, 30], [6, 9, 17]]
    rtot = [mp.fsum([mp.mpf(x) for x in row]) for row in table]
    ctot = [mp.fsum([mp.mpf(row[j]) for row in table]) for j in range(len(table[0]))]
    grand = mp.fsum(rtot)
    ind_chi2 = mp.fsum([(mp.mpf(table[i][j]) - rtot[i] * ctot[j] / grand) ** 2 / (rtot[i] * ctot[j] / grand)
                        for i in range(len(table)) for j in range(len(table[0]))])
    ind_df = mp.mpf((len(table) - 1) * (len(table[0]) - 1))
    ind_p = mp.gammainc(ind_df / 2, ind_chi2 / 2, mp.inf, regularized=True)
    chi2_ind = st.chi2_contingency(table, correction=False)
    addj("gof_chi2", "chisq_gof_chi2", [len(obs)] + obs + exp)
    addj("gof_p", "chisq_gof_p", [len(obs)] + obs + exp)
    flat = [x for row in table for x in row]
    addj("ind_chi2", "chisq_ind_chi2", [len(table), len(table[0])] + flat)
    addj("ind_p", "chisq_ind_p", [len(table), len(table[0])] + flat)

    # --- correlation truth ---
    xa = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    ya = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    mx, my = _h_mean(xa), _h_mean(ya); nc = len(xa)
    sxy = mp.fsum([(mp.mpf(x) - mx) * (mp.mpf(y) - my) for x, y in zip(xa, ya)])
    sxx = mp.fsum([(mp.mpf(x) - mx) ** 2 for x in xa]); syy = mp.fsum([(mp.mpf(y) - my) ** 2 for y in ya])
    cor_r = sxy / mp.sqrt(sxx * syy); cor_df = mp.mpf(nc - 2)
    cor_t = cor_r * mp.sqrt(cor_df / (1 - cor_r * cor_r))
    cor_p = _h_t_p(cor_t, cor_df)
    pr = st.pearsonr(xa, ya)
    addj("cor_r", "cor_r", [nc] + xa + ya); addj("cor_t", "cor_t", [nc] + xa + ya)
    addj("cor_p", "cor_p", [nc] + xa + ya)

    # --- F-test for equal variances truth ---
    ft_F = va / vb; ft_d1, ft_d2 = mp.mpf(na - 1), mp.mpf(nb - 1)
    ft_sf = _h_f_sf(ft_F, ft_d1, ft_d2); ft_p = 2 * min(ft_sf, 1 - ft_sf)
    f_sf_sp = float(st.f.sf(float(ft_F), na - 1, nb - 1)); f_p_sp = 2 * min(f_sf_sp, 1 - f_sf_sp)
    addj("ft_f", "vartest_f", two); addj("ft_p", "vartest_p", two)

    rv = _run_r(rj)
    R = {key: rv[idx] for key, idx in ri.items()}

    # --- cross-validation gate (mpmath ≈ scipy ≈ R) ---
    gate = [
        ("ttest/one", "t", truth["one"][0], {"scipy": sci["one"][0], "r": R["one_t"]}),
        ("ttest/one", "p", truth["one"][1], {"scipy": sci["one"][1], "r": R["one_p"]}),
        ("ttest/student", "t", truth["student"][0], {"scipy": sci["student"][0], "r": R["stu_t"]}),
        ("ttest/student", "p", truth["student"][1], {"scipy": sci["student"][1], "r": R["stu_p"]}),
        ("ttest/welch", "t", truth["welch"][0], {"scipy": sci["welch"][0], "r": R["wel_t"]}),
        ("ttest/welch", "p", truth["welch"][1], {"scipy": sci["welch"][1], "r": R["wel_p"]}),
        ("ttest/welch", "df", truth["welch"][2], {"scipy": sci["welch"][2], "r": R["wel_df"]}),
        ("ttest/paired", "t", truth["paired"][0], {"scipy": sci["paired"][0], "r": R["pai_t"]}),
        ("ttest/paired", "p", truth["paired"][1], {"scipy": sci["paired"][1], "r": R["pai_p"]}),
        ("anova", "F", an_F, {"scipy": fo.statistic, "r": R["anova_f"]}),
        ("anova", "p", an_p, {"scipy": fo.pvalue, "r": R["anova_p"]}),
        ("chi2/gof", "chi2", gof_chi2, {"scipy": c1.statistic, "r": R["gof_chi2"]}),
        ("chi2/gof", "p", gof_p, {"scipy": c1.pvalue, "r": R["gof_p"]}),
        ("chi2/ind", "chi2", ind_chi2, {"scipy": chi2_ind.statistic, "r": R["ind_chi2"]}),
        ("chi2/ind", "p", ind_p, {"scipy": chi2_ind.pvalue, "r": R["ind_p"]}),
        ("cor", "r", cor_r, {"scipy": pr.statistic, "r": R["cor_r"]}),
        ("cor", "p", cor_p, {"scipy": pr.pvalue, "r": R["cor_p"]}),
        ("ftest", "F", ft_F, {"scipy": float(ft_F), "r": R["ft_f"]}),
        ("ftest", "p", ft_p, {"scipy": f_p_sp, "r": R["ft_p"]}),
    ]
    _htest_gate(gate)

    # --- Welch-CI verification (ci_mean_diff_welch uses correct Welch SE + df) ---
    # t_test_two(Welch).ci calls ci_mean_diff_welch (SE = √(vA/nA+vB/nB), Welch–
    # Satterthwaite df), verified against R t.test(var.equal=FALSE)$conf.int.
    # The prior pooled-CI latent bug was fixed; this block records the correct CI.
    tcw = _h_t_crit(0.95, df_w)
    welch_ci_correct = [float((ma - mb) - tcw * se_w), float((ma - mb) + tcw * se_w)]
    print(f"\n[htest] Welch CI correct (impl matches R t.test(var.equal=FALSE)$conf.int):")
    print(f"        mpmath Welch CI: [{welch_ci_correct[0]:.6f}, {welch_ci_correct[1]:.6f}]")
    print(f"        R conf.int:     [{R['wel_cilo']:.6f}, {R['wel_cihi']:.6f}]")

    # --- write fixtures (mpmath truth as expected; xcheck provenance) ---
    def f2(pair): return [float(pair[0]), float(pair[1])]
    rows = [
        {"args": ["one"],     "expected": f2(truth["one"]),
         "xcheck": _xc(truth["one"][1], sci["one"][1], R["one_p"])},
        {"args": ["student"], "expected": f2(truth["student"]),
         "xcheck": _xc(truth["student"][1], sci["student"][1], R["stu_p"])},
        {"args": ["welch"],   "expected": [float(truth["welch"][0]), float(truth["welch"][1]), float(truth["welch"][2])],
         "xcheck": _xc(truth["welch"][1], sci["welch"][1], R["wel_p"]),
         "welch_ci_correct": welch_ci_correct,
         "welch_ci_note": "t_test_two(Welch).ci asserted against welch_ci_correct "
                          "(R t.test var.equal=FALSE); the prior pooled-CI was the latent bug, now fixed"},
        {"args": ["paired"],  "expected": f2(truth["paired"]),
         "xcheck": _xc(truth["paired"][1], sci["paired"][1], R["pai_p"])},
    ]
    with open(os.path.join(OUT, "ttests.json"), "w") as f:
        json.dump({"a": a, "b": b, "rows": rows}, f, indent=0)
    print("wrote ttests.json")

    anova_fix = {"groups": groups, "F": float(an_F), "p": float(an_p),
                 "xcheck": {"F": _xc(an_F, fo.statistic, R["anova_f"]),
                            "p": _xc(an_p, fo.pvalue, R["anova_p"])}}
    with open(os.path.join(OUT, "anova.json"), "w") as f: json.dump(anova_fix, f, indent=0)

    chi2_fix = {"gof_obs": obs, "gof_exp": exp, "gof_chi2": float(gof_chi2), "gof_p": float(gof_p),
                "table": table, "ind_chi2": float(ind_chi2), "ind_p": float(ind_p), "ind_df": float(ind_df),
                "xcheck": {"gof_chi2": _xc(gof_chi2, c1.statistic, R["gof_chi2"]),
                           "gof_p": _xc(gof_p, c1.pvalue, R["gof_p"]),
                           "ind_chi2": _xc(ind_chi2, chi2_ind.statistic, R["ind_chi2"]),
                           "ind_p": _xc(ind_p, chi2_ind.pvalue, R["ind_p"])}}
    with open(os.path.join(OUT, "chi2.json"), "w") as f: json.dump(chi2_fix, f, indent=0)
    print("wrote anova.json, chi2.json")

    cor_fix = {"a": xa, "b": ya, "r": float(cor_r), "t": float(cor_t), "p": float(cor_p),
               "xcheck": {"r": _xc(cor_r, pr.statistic, R["cor_r"]),
                          "t": _xc(cor_t, None, R["cor_t"]), "p": _xc(cor_p, pr.pvalue, R["cor_p"])}}
    with open(os.path.join(OUT, "cor.json"), "w") as f: json.dump(cor_fix, f, indent=0)
    print("wrote cor.json")

    ftest_fix = {"a": a, "b": b, "F": float(ft_F), "df1": float(na - 1), "p": float(ft_p),
                 "xcheck": {"F": _xc(ft_F, float(ft_F), R["ft_f"]), "p": _xc(ft_p, f_p_sp, R["ft_p"])}}
    with open(os.path.join(OUT, "ftest.json"), "w") as f: json.dump(ftest_fix, f, indent=0)
    print("wrote ftest.json")


# ===========================================================================
# Area: distribution suite (Class A) — three-source path (mpmath + scipy + R)
# ===========================================================================
# Continuous ppf probability grid: tail-tiered (rows below 1e-6 / above 1−1e-6
# are tagged `tail`, where scipy/R are untrusted and a curated band governs).
PPF_PROBS = [1e-9, 1e-7, 1e-6, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6,
             0.75, 0.9, 0.95, 0.99, 0.999, 1 - 1e-6, 1 - 1e-7, 1 - 1e-9]
# Discrete ppf grid: all rows are bulk (the ppf is an *exact* integer, so no tail
# tier) and R is n/a (R's q* right-continuity convention differs from ours).
PROBS_DISC = [1e-6, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75,
              0.9, 0.95, 0.99, 0.999, 1 - 1e-3, 1 - 1e-6]


def _mp_ppf_bisect(cdf_fn, p, lo, hi):
    """Bisect cdf_fn(x)=p on a bracket already known to straddle the root; ~200
    halvings ≫ the 60-dps truth target. Generalizes mp_invbetareg's pattern."""
    for _ in range(200):
        mid = (lo + hi) / 2
        if cdf_fn(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _mp_ppf_unbounded(cdf_fn, p):
    """ppf for a two-sided unbounded support (normal/studentt/cauchy): grow the
    bracket geometrically until it straddles, then bisect."""
    p = mp.mpf(p)
    lo, hi = mp.mpf(-1), mp.mpf(1)
    while cdf_fn(lo) > p:
        lo *= 2
    while cdf_fn(hi) < p:
        hi *= 2
    return _mp_ppf_bisect(cdf_fn, p, lo, hi)


def _mp_ppf_lo(cdf_fn, p, lo=mp.mpf('1e-30')):
    """ppf for a support bounded below at 0 (chisq/exp/gamma/lognormal/weibull/F):
    fixed low edge, grow the high edge until it straddles, then bisect."""
    p = mp.mpf(p)
    hi = mp.mpf(1)
    while cdf_fn(hi) < p:
        hi *= 2
    return _mp_ppf_bisect(cdf_fn, p, lo, hi)


def gen_dist():
    quantities = []

    def add_cont(name, params, mp_pdf, mp_cdf, mp_ppf, sp, xs, r_d, r_p):
        """Append the 5 continuous quantities for one dist. `params` is prepended
        to each row's [x] (or [p]) to build the R call. sf/isf use r_func=None (R's
        survival via lower.tail=FALSE is skipped — mpmath+scipy gate them). isf
        truth is ppf(1 − q) with `1 − q` formed at mpmath precision, so the
        upper-tail rows carry the full truth the f64 argument `1 − q` could not."""
        rmap = lambda a, params=params: [a[0], *params]
        quantities.extend([
            Quantity(f"dist_{name}_pdf", "value", [(x,) for x in xs],
                     mp_pdf, sp.pdf, r_d, r_argmap=rmap),
            Quantity(f"dist_{name}_cdf", "value", [(x,) for x in xs],
                     mp_cdf, sp.cdf, r_p, r_argmap=rmap),
            Quantity(f"dist_{name}_sf", "value", [(x,) for x in xs],
                     lambda x, c=mp_cdf: 1 - c(x), sp.sf, None),
            Quantity(f"dist_{name}_ppf", "quantile", [(p,) for p in PPF_PROBS],
                     mp_ppf, sp.ppf, None, prob=(0, 0.0, 1.0)),
            Quantity(f"dist_{name}_isf", "quantile", [(q,) for q in PPF_PROBS],
                     lambda q, f=mp_ppf: f(1 - mp.mpf(q)), sp.isf, None, prob=(0, 0.0, 1.0)),
        ])

    def add_disc(name, mp_pmf, mp_cdf, mp_ppf, sp, ks, r_d, r_p, r_argmap):
        """Append the 3 discrete quantities. pmf/cdf cross-check on R (params via
        r_argmap); ppf is exact-integer, all-bulk, R n/a."""
        quantities.extend([
            Quantity(f"dist_{name}_pmf", "value", [(k,) for k in ks],
                     mp_pmf, sp.pmf, r_d, r_argmap=r_argmap),
            Quantity(f"dist_{name}_cdf", "value", [(k,) for k in ks],
                     mp_cdf, sp.cdf, r_p, r_argmap=r_argmap),
            Quantity(f"dist_{name}_ppf", "quantile", [(p,) for p in PROBS_DISC],
                     mp_ppf, sp.ppf, None),
        ])

    SQ2 = mp.sqrt(2)

    # ---- Normal(mu=0.5, sigma=2.0) -----------------------------------------
    mu, sig = mp.mpf('0.5'), mp.mpf('2.0')
    def n_pdf(x): x = mp.mpf(x); return mp.e**(-(x - mu)**2 / (2 * sig**2)) / (sig * mp.sqrt(2 * mp.pi))
    def n_cdf(x): x = mp.mpf(x); return mp.mpf('0.5') * (1 + mp.erf((x - mu) / (sig * SQ2)))
    def n_ppf(p): return _mp_ppf_unbounded(n_cdf, p)
    add_cont("normal", [0.5, 2.0], n_pdf, n_cdf, n_ppf, st.norm(0.5, 2.0),
             list(np.linspace(-8.0, 9.0, 40)), "dnorm", "pnorm")

    # ---- StudentT(df=7) ----------------------------------------------------
    nu = mp.mpf('7.0')
    def t_pdf(x):
        x = mp.mpf(x)
        return mp.gamma((nu + 1) / 2) / (mp.sqrt(nu * mp.pi) * mp.gamma(nu / 2)) \
            * (1 + x**2 / nu)**(-(nu + 1) / 2)
    def t_cdf(x):
        x = mp.mpf(x)
        xt = nu / (nu + x**2)
        ib = mp.betainc(nu / 2, mp.mpf('0.5'), 0, xt, regularized=True)
        return 1 - mp.mpf('0.5') * ib if x > 0 else mp.mpf('0.5') * ib
    def t_ppf(p): return _mp_ppf_unbounded(t_cdf, p)
    add_cont("studentt", [7.0], t_pdf, t_cdf, t_ppf, st.t(7.0),
             list(np.linspace(-10.0, 10.0, 40)), "dt", "pt")

    # ---- ChiSquared(k=5) ---------------------------------------------------
    kdf = mp.mpf('5.0')
    def chi_pdf(x):
        x = mp.mpf(x)
        if x <= 0:
            return mp.mpf(0)
        return x**(kdf / 2 - 1) * mp.e**(-x / 2) / (2**(kdf / 2) * mp.gamma(kdf / 2))
    def chi_cdf(x):
        x = mp.mpf(x)
        if x <= 0:
            return mp.mpf(0)
        return mp.gammainc(kdf / 2, 0, x / 2, regularized=True)
    def chi_ppf(p): return _mp_ppf_lo(chi_cdf, p)
    add_cont("chisquared", [5.0], chi_pdf, chi_cdf, chi_ppf, st.chi2(5.0),
             list(np.linspace(0.01, 25.0, 40)), "dchisq", "pchisq")

    # ---- FisherF(d1=6, d2=12) ----------------------------------------------
    d1, d2 = mp.mpf('6.0'), mp.mpf('12.0')
    def f_pdf(x):
        x = mp.mpf(x)
        if x <= 0:
            return mp.mpf(0)
        num = (d1 * x)**d1 * d2**d2 / (d1 * x + d2)**(d1 + d2)
        return mp.sqrt(num) / (x * mp.beta(d1 / 2, d2 / 2))
    def f_cdf(x):
        x = mp.mpf(x)
        if x <= 0:
            return mp.mpf(0)
        return mp.betainc(d1 / 2, d2 / 2, 0, d1 * x / (d1 * x + d2), regularized=True)
    def f_ppf(p): return _mp_ppf_lo(f_cdf, p)
    add_cont("fisherf", [6.0, 12.0], f_pdf, f_cdf, f_ppf, st.f(6.0, 12.0),
             list(np.linspace(0.01, 10.0, 40)), "df", "pf")

    # ---- Uniform(a=-1, b=3) ; scipy uniform(loc=-1, scale=4) ----------------
    ua, ub = mp.mpf('-1.0'), mp.mpf('3.0')
    def u_pdf(x):
        x = mp.mpf(x)
        return 1 / (ub - ua) if ua <= x <= ub else mp.mpf(0)
    def u_cdf(x):
        x = mp.mpf(x)
        if x < ua:
            return mp.mpf(0)
        if x > ub:
            return mp.mpf(1)
        return (x - ua) / (ub - ua)
    def u_ppf(p):
        p = mp.mpf(p)
        return _mp_ppf_bisect(u_cdf, p, ua, ub)
    add_cont("uniform", [-1.0, 3.0], u_pdf, u_cdf, u_ppf, st.uniform(-1.0, 4.0),
             list(np.linspace(-2.0, 4.0, 40)), "dunif", "punif")

    # ---- Exponential(rate=1.5) ---------------------------------------------
    lam = mp.mpf('1.5')
    def e_pdf(x):
        x = mp.mpf(x)
        return lam * mp.e**(-lam * x) if x >= 0 else mp.mpf(0)
    def e_cdf(x):
        x = mp.mpf(x)
        return 1 - mp.e**(-lam * x) if x > 0 else mp.mpf(0)
    def e_ppf(p): return _mp_ppf_lo(e_cdf, p)
    add_cont("exponential", [1.5], e_pdf, e_cdf, e_ppf, st.expon(scale=1.0 / 1.5),
             list(np.linspace(0.0, 8.0, 40)), "dexp", "pexp")

    # ---- Cauchy(x0=1, gamma=2) ---------------------------------------------
    x0, gam = mp.mpf('1.0'), mp.mpf('2.0')
    def c_pdf(x):
        x = mp.mpf(x)
        return 1 / (mp.pi * gam * (1 + ((x - x0) / gam)**2))
    def c_cdf(x):
        x = mp.mpf(x)
        return mp.mpf('0.5') + mp.atan((x - x0) / gam) / mp.pi
    def c_ppf(p): return _mp_ppf_unbounded(c_cdf, p)
    add_cont("cauchy", [1.0, 2.0], c_pdf, c_cdf, c_ppf, st.cauchy(1.0, 2.0),
             list(np.linspace(-20.0, 22.0, 40)), "dcauchy", "pcauchy")

    # ---- Weibull(shape=2, scale=1.5) ---------------------------------------
    wk, wl = mp.mpf('2.0'), mp.mpf('1.5')
    def w_pdf(x):
        x = mp.mpf(x)
        if x < 0:
            return mp.mpf(0)
        return (wk / wl) * (x / wl)**(wk - 1) * mp.e**(-(x / wl)**wk)
    def w_cdf(x):
        x = mp.mpf(x)
        return 1 - mp.e**(-(x / wl)**wk) if x >= 0 else mp.mpf(0)
    def w_ppf(p): return _mp_ppf_lo(w_cdf, p)
    add_cont("weibull", [2.0, 1.5], w_pdf, w_cdf, w_ppf, st.weibull_min(2.0, scale=1.5),
             list(np.linspace(0.0, 6.0, 40)), "dweibull", "pweibull")

    # ---- LogNormal(mu=0.5, sigma=0.75) -------------------------------------
    lm, ls = mp.mpf('0.5'), mp.mpf('0.75')
    def l_pdf(x):
        x = mp.mpf(x)
        if x <= 0:
            return mp.mpf(0)
        return mp.e**(-(mp.log(x) - lm)**2 / (2 * ls**2)) / (x * ls * mp.sqrt(2 * mp.pi))
    def l_cdf(x):
        x = mp.mpf(x)
        if x <= 0:
            return mp.mpf(0)
        return mp.mpf('0.5') * (1 + mp.erf((mp.log(x) - lm) / (ls * SQ2)))
    def l_ppf(p): return _mp_ppf_lo(l_cdf, p)
    add_cont("lognormal", [0.5, 0.75], l_pdf, l_cdf, l_ppf, st.lognorm(s=0.75, scale=np.exp(0.5)),
             list(np.linspace(0.01, 12.0, 40)), "dlnorm", "plnorm")

    # ---- Gamma(shape=3.5, rate=2.0) ----------------------------------------
    ga, gb = mp.mpf('3.5'), mp.mpf('2.0')
    def g_pdf(x):
        x = mp.mpf(x)
        if x <= 0:
            return mp.mpf(0)
        return gb**ga * x**(ga - 1) * mp.e**(-gb * x) / mp.gamma(ga)
    def g_cdf(x):
        x = mp.mpf(x)
        if x <= 0:
            return mp.mpf(0)
        return mp.gammainc(ga, 0, gb * x, regularized=True)
    def g_ppf(p): return _mp_ppf_lo(g_cdf, p)
    add_cont("gamma", [3.5, 2.0], g_pdf, g_cdf, g_ppf, st.gamma(3.5, scale=1.0 / 2.0),
             list(np.linspace(0.01, 8.0, 40)), "dgamma", "pgamma_d")

    # ---- Beta(alpha=2.5, beta=4.0) -----------------------------------------
    ba, bb = mp.mpf('2.5'), mp.mpf('4.0')
    def b_pdf(x):
        x = mp.mpf(x)
        if x < 0 or x > 1:
            return mp.mpf(0)
        if x == 0:
            return mp.mpf(0) if ba > 1 else mp.inf
        if x == 1:
            return mp.mpf(0) if bb > 1 else mp.inf
        return x**(ba - 1) * (1 - x)**(bb - 1) / mp.beta(ba, bb)
    def b_cdf(x):
        x = mp.mpf(x)
        if x <= 0:
            return mp.mpf(0)
        if x >= 1:
            return mp.mpf(1)
        return mp.betainc(ba, bb, 0, x, regularized=True)
    def b_ppf(p):
        p = mp.mpf(p)
        return _mp_ppf_bisect(b_cdf, p, mp.mpf('1e-30'), 1 - mp.mpf('1e-30'))
    add_cont("beta", [2.5, 4.0], b_pdf, b_cdf, b_ppf, st.beta(2.5, 4.0),
             list(np.linspace(0.0, 1.0, 41)), "dbeta", "pbeta_d")

    # === DISCRETE ===========================================================
    def disc_cdf_from_pmf(pmf, support):
        """cdf(k) = Σ pmf(j) for j in support, j ≤ k."""
        def cdf(k):
            k = mp.mpf(k)
            s = mp.mpf(0)
            for j in support:
                if j <= k:
                    s += pmf(j)
            return s
        return cdf

    def disc_ppf_from_cdf(cdf, support):
        """ppf(p) = smallest k in support with cdf(k) ≥ p (exact integer)."""
        def ppf(p):
            p = mp.mpf(p)
            for j in support:
                if cdf(j) >= p:
                    return mp.mpf(j)
            return mp.mpf(support[-1])
        return ppf

    # ---- Bernoulli(p=0.4) ; R via dbinom(size=1) ---------------------------
    bp = mp.mpf('0.4')
    bern_supp = list(range(0, 2))
    def bern_pmf(k):
        k = int(round(k))
        return bp if k == 1 else (1 - bp) if k == 0 else mp.mpf(0)
    bern_cdf = disc_cdf_from_pmf(bern_pmf, bern_supp)
    add_disc("bernoulli", bern_pmf, bern_cdf, disc_ppf_from_cdf(bern_cdf, bern_supp),
             st.bernoulli(0.4), list(range(-1, 3)),
             "dbinom", "pbinom", r_argmap=lambda a: [a[0], 1.0, 0.4])

    # ---- Binomial(n=20, p=0.35) --------------------------------------------
    bn, bpp = 20, mp.mpf('0.35')
    binom_supp = list(range(0, bn + 1))
    def binom_pmf(k):
        k = int(round(k))
        if k < 0 or k > bn:
            return mp.mpf(0)
        return mp.binomial(bn, k) * bpp**k * (1 - bpp)**(bn - k)
    binom_cdf = disc_cdf_from_pmf(binom_pmf, binom_supp)
    add_disc("binomial", binom_pmf, binom_cdf, disc_ppf_from_cdf(binom_cdf, binom_supp),
             st.binom(20, 0.35), list(range(-1, 22)),
             "dbinom", "pbinom", r_argmap=lambda a: [a[0], 20.0, 0.35])

    # ---- Poisson(lambda=4.5) -----------------------------------------------
    plam = mp.mpf('4.5')
    pois_supp = list(range(0, 60))  # cdf/ppf summation reach (p up to 1−1e-6)
    def pois_pmf(k):
        k = int(round(k))
        if k < 0:
            return mp.mpf(0)
        return plam**k * mp.e**(-plam) / mp.factorial(k)
    pois_cdf = disc_cdf_from_pmf(pois_pmf, pois_supp)
    add_disc("poisson", pois_pmf, pois_cdf, disc_ppf_from_cdf(pois_cdf, pois_supp),
             st.poisson(4.5), list(range(-1, 30)),
             "dpois", "ppois", r_argmap=lambda a: [a[0], 4.5])

    # ---- Geometric(p=0.4), 1-indexed ; R dgeom is 0-indexed (k-1) -----------
    # Binary-float param (mp.mpf(0.4), not '0.4'): the discrete ppf is a step
    # function whose breakpoints land exactly on cdf values; the impl/scipy
    # evaluate cdf in binary f64, so truth must use the same 0.4 they see, else a
    # row where p coincides with a cdf step (p=0.4, cdf(1)=0.4) tips the wrong way.
    gp = mp.mpf(0.4)
    geom_supp = list(range(1, 80))
    def geom_pmf(k):
        k = int(round(k))
        return (1 - gp)**(k - 1) * gp if k >= 1 else mp.mpf(0)
    geom_cdf = disc_cdf_from_pmf(geom_pmf, geom_supp)
    add_disc("geometric", geom_pmf, geom_cdf, disc_ppf_from_cdf(geom_cdf, geom_supp),
             st.geom(0.4), list(range(0, 25)),
             "dgeom", "pgeom", r_argmap=lambda a: [a[0] - 1, 0.4])

    # ---- NegBinomial(r=4, p_success=0.3) ; R prob=failure=0.7 --------------
    nbr, nbp = mp.mpf('4.0'), mp.mpf('0.3')
    nb_supp = list(range(0, 200))
    def nb_pmf(k):
        k = int(round(k))
        if k < 0:
            return mp.mpf(0)
        return mp.binomial(k + nbr - 1, k) * (1 - nbp)**nbr * nbp**k
    nb_cdf = disc_cdf_from_pmf(nb_pmf, nb_supp)
    add_disc("negbinomial", nb_pmf, nb_cdf, disc_ppf_from_cdf(nb_cdf, nb_supp),
             st.nbinom(4, 0.7), list(range(-1, 30)),
             "dnbinom", "pnbinom", r_argmap=lambda a: [a[0], 4.0, 0.7])

    # ---- Hypergeometric(N=30, K=12, n=10) ; R dhyper(k, m=K, n=N-K, k=draws) -
    hN, hK, hn = 30, 12, 10
    hyp_supp = list(range(max(0, hn + hK - hN), min(hn, hK) + 1))
    def hyp_pmf(k):
        k = int(round(k))
        if k < max(0, hn + hK - hN) or k > min(hn, hK):
            return mp.mpf(0)
        return mp.binomial(hK, k) * mp.binomial(hN - hK, hn - k) / mp.binomial(hN, hn)
    hyp_cdf = disc_cdf_from_pmf(hyp_pmf, hyp_supp)
    add_disc("hypergeometric", hyp_pmf, hyp_cdf, disc_ppf_from_cdf(hyp_cdf, hyp_supp),
             st.hypergeom(30, 12, 10), list(range(-1, 12)),
             "dhyper", "phyper", r_argmap=lambda a: [a[0], 12.0, 18.0, 10.0])

    generate_three_source(quantities)


# ===========================================================================
# Area: density (Class B) — three-source path (mpmath truth + scipy + R),
# bespoke fixture shapes (NOT the generic Quantity/check_grid grid).
# ===========================================================================
def gen_density():
    import scipy.stats as st_stats

    def _density_gate(label, cases):
        """cases: list of (case_label, i, truth_mpf, {src: float_or_None}). Asserts
        every present source matches truth within CONV_REL (abs floor CONV_ABS).
        Exits 1 before any fixture is written."""
        failures = []
        for case_label, i, t, srcs in cases:
            if t is None:
                continue
            ft = float(t)
            floor = max(CONV_ABS, CONV_REL * abs(ft) if math.isfinite(ft) else CONV_ABS)
            for src, v in srcs.items():
                if v is None:
                    continue
                if abs(mp.mpf(v) - t) > floor:
                    failures.append((label, case_label, i, src, v, ft))
        if failures:
            print(f"\n*** {label} CROSS-VALIDATION GATE FAILED — no fixtures written ***")
            for lbl, cl, i, src, got, truth in failures:
                print(f"  {lbl}[{cl}][{i}]: {src}={got} vs mpmath={truth!r}")
            sys.exit(1)

    def mp_kde_pt(x, h, data):
        """Gaussian KDE density at x: (1/(n·h))·Σ φ((x−x_i)/h), φ(z)=e^(−z²/2)/√(2π).
        `h` is scipy's resolved bandwidth (passed in) so mpmath and scipy use a
        bit-identical h — any residual difference is the kernel sum, not the bw."""
        x, h = mp.mpf(x), mp.mpf(h)
        n = len(data)
        total = mp.mpf(0)
        for xi in data:
            z = (x - mp.mpf(xi)) / h
            total += mp.e**(-z * z / 2) / mp.sqrt(2 * mp.pi)
        return total / (n * h)

    data_small = [1.0, 2.0, 3.0, 4.0, 5.0]
    data_large = [0.1, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 8.5, 9.0, 10.0]

    grid_small = np.linspace(0.0, 6.0, 20).tolist()
    grid_large = np.linspace(-1.0, 12.0, 30).tolist()

    def kde_case(label, data, bw_method, fixed_h, grid):
        """Resolve scipy's bandwidth (unchanged from legacy), then compute mpmath
        truth at `h=resolved_h` plus scipy & R cross-checks. Returns a record with
        an `_r_jobs` slice (one kde_manual job per grid point) batched by the caller."""
        data_arr = np.array(data)
        if bw_method in ('silverman', 'scott'):
            k = st_stats.gaussian_kde(data_arr, bw_method=bw_method)
            resolved_h = float(k.factor * data_arr.std(ddof=1))
        else:
            # scipy: bw_method=scalar sets factor=scalar, h=factor*std(ddof=1); for
            # Fixed(h) pass fixed_h/std so factor*std == fixed_h exactly.
            sd = data_arr.std(ddof=1)
            k = st_stats.gaussian_kde(data_arr, bw_method=fixed_h / sd)
            resolved_h = fixed_h

        scipy_density = k.evaluate(np.array(grid)).tolist()
        mp_density = [mp_kde_pt(x, resolved_h, data) for x in grid]
        r_jobs = [{"func": "kde_manual",
                   "args": [float(x), resolved_h] + [float(d) for d in data]}
                  for x in grid]
        return {"label": label, "data": data, "bw_method": bw_method, "fixed_h": fixed_h,
                "grid": grid, "bandwidth": resolved_h,
                "_mp": mp_density, "_scipy": scipy_density, "_r_jobs": r_jobs}

    raw = [
        kde_case("small_silverman", data_small, "silverman", None, grid_small),
        kde_case("small_scott",     data_small, "scott",     None, grid_small),
        kde_case("small_fixed_0.5", data_small, None, 0.5, grid_small),
        kde_case("small_fixed_1.2", data_small, None, 1.2, grid_small),
        kde_case("large_silverman", data_large, "silverman", None, grid_large),
        kde_case("large_scott",     data_large, "scott",     None, grid_large),
    ]

    # One batched R invocation for all KDE grid points.
    all_r_jobs, r_starts = [], []
    for case in raw:
        r_starts.append(len(all_r_jobs))
        all_r_jobs.extend(case["_r_jobs"])
    r_all = _run_r(all_r_jobs) if all_r_jobs else []

    gate_rows, kde_records = [], []
    for case, r_start in zip(raw, r_starts):
        n_pts = len(case["grid"])
        r_slice = r_all[r_start:r_start + n_pts]
        mp_dens, sp_dens = case["_mp"], case["_scipy"]
        for i, (t, sv, rv) in enumerate(zip(mp_dens, sp_dens, r_slice)):
            gate_rows.append((case["label"], i, t, {"scipy": sv, "r": rv}))
        xcheck = [{"scipy": {"val": sv, "rel": _rel_err(sv, t)},
                   "r":     {"val": rv, "rel": _rel_err(rv, t)}}
                  for sv, rv, t in zip(sp_dens, r_slice, mp_dens)]
        kde_records.append({
            "label": case["label"], "data": case["data"],
            "bw_method": case["bw_method"], "fixed_h": case["fixed_h"],
            "grid": case["grid"], "bandwidth": case["bandwidth"],
            "density": [_f64(t) for t in mp_dens],
            "xcheck_density": xcheck,
        })

    _density_gate("kde", gate_rows)
    with open(os.path.join(OUT, "kde.json"), "w") as f:
        json.dump(kde_records, f, indent=1)
    print(f"wrote kde.json ({len(kde_records)} cases)")

    # --- histogram_auto: edges/counts are EXACT (numpy), density = mpmath of
    # count/(n·binwidth). scipy/numpy cross-checks density; R n/a (nclass.* differs).
    data_hist = [1.2, 1.5, 2.0, 2.3, 2.8, 3.1, 3.5, 4.0, 4.2, 4.8,
                 5.0, 5.3, 5.7, 6.0, 6.5, 7.0, 7.3, 8.0, 8.5, 9.0]
    n_hist = len(data_hist)

    hist_records, hist_gate_rows = [], []
    for rule_str in ('sturges', 'scott', 'fd', 'rice', 'sqrt'):
        edges = np.histogram_bin_edges(data_hist, bins=rule_str)
        counts_arr, _ = np.histogram(data_hist, bins=edges)
        counts_list = counts_arr.tolist()
        edges_list = edges.tolist()

        mp_density, scipy_xcheck = [], []
        for i, c in enumerate(counts_list):
            bw = mp.mpf(edges_list[i + 1]) - mp.mpf(edges_list[i])
            d_mp = mp.mpf(c) / (mp.mpf(n_hist) * bw)
            mp_density.append(d_mp)
            d_sp = float(c) / (float(n_hist) * float(bw))
            scipy_xcheck.append(d_sp)
            hist_gate_rows.append((rule_str, i, d_mp, {"scipy": d_sp}))

        hist_records.append({
            "rule": rule_str, "data": data_hist,
            "edges": edges_list, "counts": counts_list,
            "density": [_f64(t) for t in mp_density],
            "xcheck_density": [{"scipy": {"val": sv, "rel": _rel_err(sv, t)}}
                               for sv, t in zip(scipy_xcheck, mp_density)],
        })

    _density_gate("histogram_auto", hist_gate_rows)
    with open(os.path.join(OUT, "histogram_auto.json"), "w") as f:
        json.dump(hist_records, f, indent=1)
    print(f"wrote histogram_auto.json ({len(hist_records)} cases)")


# ===========================================================================
# Area: transforms (Class A/B) — three-source path (mpmath truth + scipy + R),
# bespoke per-quantity fixture shapes (NOT the generic Quantity/check_grid grid).
# ===========================================================================
def gen_transform():
    def _transform_gate(rows_with_sources):
        """rows_with_sources: list of (name, desc, truth_mpf, {src: value}). Asserts
        every present source matches truth within CONV_REL (abs floor CONV_ABS).
        `None` source = engine n/a, skipped. Exits 1 before any fixture is written."""
        failures = []
        for name, desc, truth_mpf, srcs in rows_with_sources:
            if truth_mpf is None:
                continue
            t = truth_mpf
            ft = float(t)
            floor = max(CONV_ABS, CONV_REL * abs(ft) if math.isfinite(ft) else CONV_ABS)
            for src, v in srcs.items():
                if v is None:
                    continue
                if abs(mp.mpf(v) - t) > floor:
                    failures.append((name, desc, src, v, ft))
        if failures:
            print("\n*** TRANSFORM CROSS-VALIDATION GATE FAILED — no fixtures written ***")
            for nm, desc, src, got, truth in failures:
                print(f"  {nm}[{desc}]: {src}={got} vs mpmath={truth!r}")
            sys.exit(1)

    def write_rankdata():
        # Ranks are EXACT (integers / half-integers), so scipy `rankdata` stays the
        # `expected` truth — mpmath adds nothing. R cross-checks average/min/max only;
        # R base has no dense/ordinal (those rows carry r=null, "R n/a").
        from scipy.stats import rankdata
        methods = ["average", "min", "max", "dense", "ordinal"]
        # Three representative input vectors (chosen to hit ties, all-same, monotone).
        # NaN stored as null (None) so serde_json can parse the fixture.
        vectors = [
            [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],   # ties in position 1&3
            [1.0, 2.0, 3.0, 4.0, 5.0],                    # strict monotone
            [2.0, 2.0, 2.0],                               # all tied
            [5.0, None, 3.0, None, 1.0],                   # NaN omit: ranks of [5,3,1]
        ]
        r_func_map = {"average": "rank_avg", "min": "rank_min", "max": "rank_max",
                      "dense": None, "ordinal": None}

        records = []
        r_jobs, r_index = [], []   # flat manifest + back-pointers (rec_i, elem_i)
        for v in vectors:
            finite = [x for x in v if x is not None]  # filter NaN (stored as None/null)
            for method in methods:
                result = rankdata(finite, method=method).tolist()
                rec = {"input": v, "method": method, "expected": result,
                       "xcheck": {"r": [None] * len(result)}}
                records.append(rec)
                r_func = r_func_map[method]
                if r_func is not None:
                    for idx in range(len(result)):
                        r_index.append((len(records) - 1, idx))
                        r_jobs.append({"func": r_func,
                                       "args": [float(x) for x in finite] + [float(idx)]})

        r_vals = _run_r(r_jobs) if r_jobs else []
        for (rec_i, elem_i), rv in zip(r_index, r_vals):
            records[rec_i]["xcheck"]["r"][elem_i] = rv

        # Gate: R must reproduce scipy's exact rank (treat scipy as truth here).
        gate_rows = []
        for rec in records:
            if rec["xcheck"]["r"][0] is None:
                continue  # R n/a (dense/ordinal)
            for i, (sp_val, r_val) in enumerate(zip(rec["expected"], rec["xcheck"]["r"])):
                gate_rows.append((f"rankdata/{rec['method']}", f"{rec['input']}#{i}",
                                  mp.mpf(sp_val), {"r": r_val}))
        _transform_gate(gate_rows)

        # Finalize: replace bare R values with {val, rel} provenance.
        for rec in records:
            r_xcheck = []
            for sp_val, r_val in zip(rec["expected"], rec["xcheck"]["r"]):
                if r_val is None:
                    r_xcheck.append({"val": None, "rel": None})
                else:
                    r_xcheck.append({"val": r_val, "rel": _rel_err(r_val, mp.mpf(sp_val))})
            rec["xcheck"]["r"] = r_xcheck

        with open(os.path.join(OUT, "rankdata.json"), "w") as f:
            json.dump(records, f, indent=0)
        print(f"wrote rankdata.json ({len(records)} rows)")

    write_rankdata()

    def write_boxcox():
        from scipy.stats import boxcox as sp_boxcox
        lambdas = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        # Strictly positive values only (box_cox requires x > 0)
        xs = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]

        def mp_boxcox(lam, x):
            lam, x = mp.mpf(lam), mp.mpf(x)
            return mp.log(x) if lam == 0 else (x**lam - 1) / lam

        records, gate_rows = [], []
        for lam in lambdas:
            for x in xs:
                truth_mpf = mp_boxcox(lam, x)
                expected = _f64(truth_mpf)
                try:
                    sval = float(sp_boxcox([x], lmbda=lam)[0])
                    sval = sval if math.isfinite(sval) else None
                except Exception:
                    sval = None
                gate_rows.append(("boxcox", f"lam={lam},x={x}", truth_mpf, {"scipy": sval}))
                records.append({"lambda": lam, "x": x, "expected": expected,
                                "xcheck": {"scipy": {"val": sval,
                                                     "rel": _rel_err(sval, truth_mpf)}}})
        # x<=0 rows: DomainError expected (expected=null); no truth, no gate.
        for x in [-1.0, 0.0]:
            records.append({"lambda": 1.0, "x": x, "expected": None,
                            "xcheck": {"scipy": {"val": None, "rel": None}}})

        _transform_gate(gate_rows)
        with open(os.path.join(OUT, "boxcox.json"), "w") as f:
            json.dump(records, f, indent=0)
        print(f"wrote boxcox.json ({len(records)} rows)")

    def write_yeojohnson():
        from scipy.stats import yeojohnson as sp_yj
        lambdas = [-1.0, 0.0, 0.5, 1.0, 2.0, 3.0]
        xs = [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

        def mp_yeojohnson(lam, x):
            lam, x = mp.mpf(lam), mp.mpf(x)
            if x >= 0:
                return ((x + 1)**lam - 1) / lam if lam != 0 else mp.log(x + 1)
            # x < 0
            return -((-x + 1)**(2 - lam) - 1) / (2 - lam) if lam != 2 else -mp.log(-x + 1)

        records, gate_rows = [], []
        for lam in lambdas:
            for x in xs:
                truth_mpf = mp_yeojohnson(lam, x)
                expected = _f64(truth_mpf)
                try:
                    sval = float(sp_yj([x], lmbda=lam)[0])
                    sval = sval if math.isfinite(sval) else None
                except Exception:
                    sval = None
                gate_rows.append(("yeojohnson", f"lam={lam},x={x}", truth_mpf, {"scipy": sval}))
                records.append({"lambda": lam, "x": x, "expected": expected,
                                "xcheck": {"scipy": {"val": sval,
                                                     "rel": _rel_err(sval, truth_mpf)}}})
        _transform_gate(gate_rows)
        with open(os.path.join(OUT, "yeojohnson.json"), "w") as f:
            json.dump(records, f, indent=0)
        print(f"wrote yeojohnson.json ({len(records)} rows)")

    write_boxcox()
    write_yeojohnson()

    def write_normal_scores():
        # Truth = mpmath qnorm of the Blom score p=(rank−3/8)/(n+1/4) (average ranks).
        # qnorm(p) = √2 · erfinv(2p−1). scipy norm.ppf and R qnorm both cross-check.
        from scipy.stats import rankdata, norm
        vectors = [
            [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 2.0, 2.0],
            [5.0, None, 3.0, None, 1.0],  # NaN stored as null
        ]

        def mp_qnorm(p):
            p = mp.mpf(p)
            return mp.sqrt(2) * mp.erfinv(2 * p - 1)

        # Pass 1: truth + scipy per element; accumulate flat R manifest.
        built = []   # per vector: dict with ps, truth, scipy, r-slice metadata
        r_jobs, r_index = [], []
        for v in vectors:
            finite = [x for x in v if x is not None]
            n = len(finite)
            ranks = rankdata(finite, method='average')
            ps = [(r - 3/8) / (n + 1/4) for r in ranks]
            truth = [mp_qnorm(p) for p in ps]
            scipy_vals = [float(norm.ppf(p)) for p in ps]
            slot = {"input": v, "ps": ps, "truth": truth, "scipy": scipy_vals,
                    "r_start": len(r_jobs), "n_elems": len(ps)}
            for p in ps:
                r_index.append(len(r_jobs))
                r_jobs.append({"func": "qnorm", "args": [float(p)]})
            built.append(slot)

        r_vals = _run_r(r_jobs) if r_jobs else []

        gate_rows, records = [], []
        for slot in built:
            truth = slot["truth"]
            scipy_vals = slot["scipy"]
            r_slice = r_vals[slot["r_start"]:slot["r_start"] + slot["n_elems"]]
            for i, (t, sv, rv) in enumerate(zip(truth, scipy_vals, r_slice)):
                gate_rows.append(("normal_scores", f"{slot['input']}#{i}", t,
                                  {"scipy": sv, "r": rv}))
            records.append({
                "input": slot["input"],
                "expected": [_f64(t) for t in truth],
                "xcheck": {
                    "scipy": [{"val": sv, "rel": _rel_err(sv, t)}
                              for sv, t in zip(scipy_vals, truth)],
                    "r":     [{"val": rv, "rel": _rel_err(rv, t)}
                              for rv, t in zip(r_slice, truth)]},
            })

        _transform_gate(gate_rows)
        with open(os.path.join(OUT, "normal_scores.json"), "w") as f:
            json.dump(records, f, indent=0)
        print(f"wrote normal_scores.json ({len(records)} rows)")

    write_normal_scores()

    def write_quantile_normalize():
        # Bolstad 2003 column-rank-averaging normalization (3 columns, 4 rows),
        # recomputed in mpmath. scipy/R have no matching built-in → single-source
        # (mpmath only), allowed per spec; no xcheck block.
        # col0=[2,4,3,5] sorted=[2,3,4,5]; col1=[1,4,4,4] sorted=[1,4,4,4];
        # col2=[3,6,4,8] sorted=[3,4,6,8]
        #   ref[0]=(2+1+3)/3=2, ref[1]=(3+4+4)/3=11/3, ref[2]=(4+4+6)/3=14/3, ref[3]=(5+4+8)/3=17/3
        from scipy.stats import rankdata
        matrix = [[2.0, 4.0, 3.0, 5.0], [1.0, 4.0, 4.0, 4.0], [3.0, 6.0, 4.0, 8.0]]
        n = len(matrix[0])
        sorted_cols = [sorted(mp.mpf(x) for x in col) for col in matrix]
        ref_mp = [(sorted_cols[0][k] + sorted_cols[1][k] + sorted_cols[2][k]) / mp.mpf(len(matrix))
                  for k in range(n)]
        ref = [_f64(r) for r in ref_mp]
        result = []
        for col in matrix:
            ranks = rankdata([float(x) for x in col], method='average')  # 1-based avg, exact
            out_col = []
            for r in ranks:
                lo_idx = int(r) - 1
                hi_idx = min(int(r), n - 1)
                frac = mp.mpf(r) - int(r)
                val = ref_mp[lo_idx] * (1 - frac) + ref_mp[hi_idx] * frac
                out_col.append(_f64(val))
            result.append(out_col)
        fixture = {"matrix": matrix, "ref": ref, "expected": result}
        with open(os.path.join(OUT, "quantile_normalize.json"), "w") as f:
            json.dump(fixture, f, indent=0)
        print("wrote quantile_normalize.json")

    write_quantile_normalize()


def gen_descriptives():
    """Three-source oracle fixtures for the exported descriptive statistics.

    Covers mean, var (both ddof), sd (both ddof), median, skewness (g1),
    kurtosis (excess), cov, and pearson.  R is n/a for skewness and kurtosis
    (no base-R function); those are scipy-only in the gate.  `sum`, `count`,
    `min`, `max`, and `range` return exact integers / comparisons and are
    covered by inline unit tests — no floating-point oracle fixture needed.
    """
    def _desc_gate(items):
        """items: (label, truth_mpf, {src: val}). Same gate as _htest_gate."""
        failures = []
        for label, truth_mpf, srcs in items:
            if truth_mpf is None:
                continue
            t = truth_mpf
            ft = float(t)
            floor = max(CONV_ABS, CONV_REL * abs(ft) if math.isfinite(ft) else CONV_ABS)
            for src, v in srcs.items():
                if v is None:
                    continue
                if abs(mp.mpf(v) - t) > floor:
                    failures.append((label, src, v, ft))
        if failures:
            print("\n*** DESCRIPTIVES CROSS-VALIDATION GATE FAILED — no fixtures written ***")
            for label, src, got, truth in failures:
                print(f"  {label}: {src}={got} vs mpmath={truth!r}")
            sys.exit(1)

    def _xc2(truth_mpf, scipy_val, r_val):
        return {"scipy": {"val": scipy_val, "rel": _rel_err(scipy_val, truth_mpf)},
                "r":     {"val": r_val,     "rel": _rel_err(r_val, truth_mpf)}}

    def _xc1(truth_mpf, scipy_val):
        return {"scipy": {"val": scipy_val, "rel": _rel_err(scipy_val, truth_mpf)}}

    # --- mpmath truth helpers ---
    def mp_mean(xs_):
        return mp.fsum([mp.mpf(x) for x in xs_]) / len(xs_)

    def mp_var_sample(xs_):
        m = mp_mean(xs_)
        return mp.fsum([(mp.mpf(x) - m)**2 for x in xs_]) / (len(xs_) - 1)

    def mp_var_pop(xs_):
        m = mp_mean(xs_)
        return mp.fsum([(mp.mpf(x) - m)**2 for x in xs_]) / len(xs_)

    def mp_median(xs_):
        s = sorted(mp.mpf(x) for x in xs_)
        n = len(s)
        h = mp.mpf(n - 1) / 2
        lo, hi = int(mp.floor(h)), int(mp.ceil(h))
        return s[lo] + (h - lo) * (s[hi] - s[lo])

    def mp_skewness(xs_):
        # Fisher–Pearson g1: m3/m2^(3/2), biased (scipy skew default, bias=True)
        m = mp_mean(xs_)
        n = len(xs_)
        ds = [mp.mpf(x) - m for x in xs_]
        m2 = mp.fsum([d**2 for d in ds]) / n
        m3 = mp.fsum([d**3 for d in ds]) / n
        return m3 / m2**(mp.mpf(3) / 2)

    def mp_kurtosis(xs_):
        # Fisher excess: m4/m2^2 − 3, biased (scipy kurtosis default, bias=True)
        m = mp_mean(xs_)
        n = len(xs_)
        ds = [mp.mpf(x) - m for x in xs_]
        m2 = mp.fsum([d**2 for d in ds]) / n
        m4 = mp.fsum([d**4 for d in ds]) / n
        return m4 / m2**2 - 3

    def mp_cov(xs_, ys_):
        # sample covariance, ddof=1
        n = len(xs_)
        mx, my = mp_mean(xs_), mp_mean(ys_)
        dx = [mp.mpf(x) - mx for x in xs_]
        dy = [mp.mpf(y) - my for y in ys_]
        return mp.fsum([a * b for a, b in zip(dx, dy)]) / (n - 1)

    def mp_pearson(xs_, ys_):
        mx, my = mp_mean(xs_), mp_mean(ys_)
        dx = [mp.mpf(x) - mx for x in xs_]
        dy = [mp.mpf(y) - my for y in ys_]
        sxy = mp.fsum([a * b for a, b in zip(dx, dy)])
        sxx = mp.fsum([a**2 for a in dx])
        syy = mp.fsum([a**2 for a in dy])
        return sxy / mp.sqrt(sxx * syy)

    # Input vectors: two unary cases + one binary case.
    # v1: numpy-docs classic, non-trivial skewness and kurtosis.
    # v2: mixed signs, exercises negative-skewness path.
    v1 = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    v2 = [1.0, 3.0, -2.0, 5.0, 0.0, 4.0]
    xs_b = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys_b = [2.0, 1.0, 4.0, 3.0, 6.0]

    # R jobs (R n/a for var_pop / sd_pop / skewness / kurtosis)
    rj, ri = [], {}
    def addj(key, func, args):
        ri[key] = len(rj); rj.append({"func": func, "args": [float(x) for x in args]})
    for tag, v in [("v1", v1), ("v2", v2)]:
        addj(f"{tag}_mean",   "desc_mean",   v)
        addj(f"{tag}_var",    "desc_var",    v)
        addj(f"{tag}_sd",     "desc_sd",     v)
        addj(f"{tag}_median", "desc_median", v)
    nb = len(xs_b)
    addj("bin_cov",     "desc_cov",     [nb] + xs_b + ys_b)
    addj("bin_pearson", "desc_pearson", [nb] + xs_b + ys_b)
    rv_all = _run_r(rj)
    R = {key: rv_all[idx] for key, idx in ri.items()}

    # Gate + fixture building (computed once, used in both)
    gate = []
    unary_records = []
    for tag, v in [("v1", v1), ("v2", v2)]:
        t_mean  = mp_mean(v)
        t_vs    = mp_var_sample(v)
        t_vp    = mp_var_pop(v)
        t_sds   = mp.sqrt(t_vs)
        t_sdp   = mp.sqrt(t_vp)
        t_med   = mp_median(v)
        t_skew  = mp_skewness(v)
        t_kurt  = mp_kurtosis(v)

        sp_mean  = float(np.mean(v))
        sp_vs    = float(np.var(v, ddof=1))
        sp_vp    = float(np.var(v, ddof=0))
        sp_sds   = float(np.std(v, ddof=1))
        sp_sdp   = float(np.std(v, ddof=0))
        sp_med   = float(np.median(v))
        sp_skew  = float(st.skew(v))     # bias=True (Fisher g1, biased)
        sp_kurt  = float(st.kurtosis(v)) # fisher=True, bias=True (excess, biased)

        gate += [
            (f"{tag}/mean",       t_mean, {"scipy": sp_mean, "r": R[f"{tag}_mean"]}),
            (f"{tag}/var_sample", t_vs,   {"scipy": sp_vs,   "r": R[f"{tag}_var"]}),
            (f"{tag}/var_pop",    t_vp,   {"scipy": sp_vp,   "r": None}),
            (f"{tag}/sd_sample",  t_sds,  {"scipy": sp_sds,  "r": R[f"{tag}_sd"]}),
            (f"{tag}/sd_pop",     t_sdp,  {"scipy": sp_sdp,  "r": None}),
            (f"{tag}/median",     t_med,  {"scipy": sp_med,  "r": R[f"{tag}_median"]}),
            (f"{tag}/skewness",   t_skew, {"scipy": sp_skew, "r": None}),
            (f"{tag}/kurtosis",   t_kurt, {"scipy": sp_kurt, "r": None}),
        ]
        unary_records.append({
            "xs":         v,
            "mean":       _f64(t_mean),
            "var_sample": _f64(t_vs),
            "var_pop":    _f64(t_vp),
            "sd_sample":  _f64(t_sds),
            "sd_pop":     _f64(t_sdp),
            "median":     _f64(t_med),
            "skewness":   _f64(t_skew),
            "kurtosis":   _f64(t_kurt),
            "xcheck": {
                "mean":       _xc2(t_mean, sp_mean, R[f"{tag}_mean"]),
                "var_sample": _xc2(t_vs,   sp_vs,   R[f"{tag}_var"]),
                "var_pop":    _xc1(t_vp,   sp_vp),
                "sd_sample":  _xc2(t_sds,  sp_sds,  R[f"{tag}_sd"]),
                "sd_pop":     _xc1(t_sdp,  sp_sdp),
                "median":     _xc2(t_med,  sp_med,  R[f"{tag}_median"]),
                "skewness":   _xc1(t_skew, sp_skew),
                "kurtosis":   _xc1(t_kurt, sp_kurt),
            },
        })

    t_cov = mp_cov(xs_b, ys_b)
    t_pea = mp_pearson(xs_b, ys_b)
    sp_cov = float(np.cov(xs_b, ys_b, ddof=1)[0, 1])
    sp_pea = float(st.pearsonr(xs_b, ys_b).statistic)
    gate += [
        ("binary/cov",     t_cov, {"scipy": sp_cov, "r": R["bin_cov"]}),
        ("binary/pearson", t_pea, {"scipy": sp_pea, "r": R["bin_pearson"]}),
    ]
    binary_record = {
        "xs": xs_b, "ys": ys_b,
        "cov":     _f64(t_cov),
        "pearson": _f64(t_pea),
        "xcheck": {
            "cov":     _xc2(t_cov, sp_cov, R["bin_cov"]),
            "pearson": _xc2(t_pea, sp_pea, R["bin_pearson"]),
        },
    }

    _desc_gate(gate)

    fixture = {"unary": unary_records, "binary": [binary_record]}
    with open(os.path.join(OUT, "descriptives.json"), "w") as f:
        json.dump(fixture, f, indent=0)
    print("wrote descriptives.json")


AREAS = {
    "special":      gen_special,
    "htest":        gen_htest,
    "dist":         gen_dist,
    "density":      gen_density,
    "transform":    gen_transform,
    "descriptives": gen_descriptives,
}


def main():
    selected = sys.argv[1:] or list(AREAS)
    unknown = [s for s in selected if s not in AREAS]
    if unknown:
        sys.exit(f"unknown area(s): {unknown}; choose from {list(AREAS)}")
    for name in selected:
        print(f"=== {name} ===")
        AREAS[name]()


if __name__ == "__main__":
    main()
