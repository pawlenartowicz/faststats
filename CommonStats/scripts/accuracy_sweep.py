#!/usr/bin/env python3
"""Accuracy harness: measures CommonStats against mpmath truth.

    python3 scripts/accuracy_sweep.py [--only NAMES] [--check] [--update-baseline]
                                      [--export-fixtures] [--jobs N]

Covers every function and distribution method of `src/special/incomplete.rs`,
`src/dist/continuous.rs` and `src/dist/discrete.rs` (with the crate-private log
variants the quantile solvers use), plus `lgamma`, `lbeta` and `inv_beta_reg`.
The crate is evaluated by `examples/accuracy_probe.rs` (built here with
`cargo build --release --example accuracy_probe --features dist`, a no-op when
current); mpmath is a development tool only, never a crate or CI dependency.

Error ratio of one point, with ε = 2⁻⁵² and K = Σᵢ |xᵢ·∂f/∂xᵢ| over the real
inputs (integer inputs are exact and add nothing):

    r = |observed − truth| / (ε · max(K, |truth|, 2⁻¹⁰²²))

With the condition number κ = K/|truth| this is rel. error / (max(κ, 1)·ε), the
number of times the error exceeds that of a backward-stable algorithm; the
2⁻¹⁰²² floor scores a subnormal result against its own spacing. For a log
output (`log_density`, `log_mass`, `ln_*`, `lgamma`, `lbeta`) f is the log
itself, so its absolute error is scored against the sensitivity of ln f. A
quantile's K is (p + Σ|θ·∂M/∂θ|)/pdf at the root, M the tail mass solved for. A
discrete quantile scores the probability gap that makes the returned k wrong,
|F(k) − p|/(ε·max(p, K_F)) with F the cdf (sf above p = ½) at the misjudged k
and K_F its sensitivity to the real parameters there.

Classes: r ≤ LIMIT_R limit, up to BUG_R review, above it bug. Also a bug
whatever r: a probability outside [0, 1], a NaN or error for a valid input
(except a discrete quantile past i64::MAX, an error by contract), a cdf that
decreases (an sf that increases) between the points of one parameter set by
more than LIMIT_R·max(κ, 1)·ε relative (smaller steps are reported as seam
steps), a quantile outside the support, and `quantile(1)` on a bounded support
other than its largest point of positive mass.

Flags:
  --only NAMES        comma-separated function names (`gammq`, `gamma.cdf`) or
                      distribution names (`gamma` = all its methods); the run,
                      the report, `--check`, `--update-baseline` and
                      `--export-fixtures` then cover only those functions.
  --check             exit 1 if, for a function and kernel regime, the worst r
                      moves into a worse class than in
                      `scripts/accuracy_baseline.json` or, being above 1, grows
                      by more than 2x; or if a function's count of hard
                      violations grows.
  --update-baseline   write this run's per-function results into
                      `scripts/accuracy_baseline.json` (replacing only the
                      functions run) and regenerate the `//!` accuracy tables
                      of the three source files from it, between
                      `//! <!-- accuracy:begin -->` and `//! <!-- accuracy:end -->`.
  --export-fixtures   merge the worst FIXTURE_N points per function and regime
                      into `tests/fixtures/accuracy_<group>.json`, the points
                      `tests/accuracy.rs` replays. Points already there are kept
                      and re-scored by every run (they are added to the sweep).
  --jobs N            truth worker processes (default: CPUs − 2).

Truth (mpmath, 60 significant digits, more where parameters are large) is
cached in `scripts/.accuracy_cache/` (git-ignored), keyed by the exact inputs,
so a rerun computes only new points and can be interrupted and resumed. Points
are drawn from fixed seeds, so reruns see the same points. The text report is
written to `scripts/.accuracy_cache/report.txt`.
"""
import argparse
import json
import math
import multiprocessing as mproc
import os
import random
import signal
import subprocess
import sys
import time
import zlib

import mpmath

mp = mpmath.mp
mpf = mpmath.mpf

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
CACHE = os.path.join(HERE, ".accuracy_cache")
BASELINE = os.path.join(HERE, "accuracy_baseline.json")
FIXTURES = os.path.join(CRATE, "tests", "fixtures")
PROBE = os.path.join(CRATE, "target", "release", "examples", "accuracy_probe")

EPS = 2.0 ** -52
TINY = 2.0 ** -1022
F64_MAX = sys.float_info.max
SEED = 20260927

# Class thresholds on r. BUG_R sits in the one gap the functions' worst r
# show (680 → 3000); LIMIT_R is provisional, the worst r run densely
# through 10.
LIMIT_R = 10.0
BUG_R = 1e3

# Points per (function, regime) written by --export-fixtures.
FIXTURE_N = 20

# Truth: working digits, the initial relative step of the differences for K
# (`z_deriv` shrinks it where the curvature needs), and the per-point time
# limit (a point over it is recorded as a truth failure).
DPS = 60
FD_STEP = mpf(10) ** -20
TRUTH_TIMEOUT = 120
# Digits of a quadrature whose integrand is formed without cancellation.
QUAD_DPS = 50


# ============================================================================
# Truth: mpmath evaluation at the exact f64/i64 inputs
# ============================================================================

def to_mp(s):
    """An input string as an exact mpf (via its f64 value) or a Python int."""
    t = s.lstrip("-")
    if t.isdigit():
        return int(s)
    return mpf(float(s))


def extra_digits(vals):
    """Digits to add for inputs up to 10^d: lnΓ and log terms of size v·ln v
    cancel to O(1) in several formulas, costing ~log10(v) digits."""
    m = 1.0
    for v in vals:
        try:
            m = max(m, abs(float(v)))
        except (OverflowError, TypeError):
            m = 1e308
    if not math.isfinite(m):
        m = 1e308
    return min(330, max(0, int(math.ceil(math.log10(m)))))


class TruthError(Exception):
    pass


def _quad_log(logf, lo, hi, pts, peak, base=None, resolve=0):
    """∫ e^{logf(t)} dt over [lo, hi] with breakpoints `pts`, as
    e^{L}·∫ e^{logf(t) − L} dt, L = logf(peak): mpmath's quad stops on an
    absolute error estimate (~10^−dps), so the integrand is scaled to O(1) at
    its largest value on the interval, and an integral that still comes out
    below 1 (a peak narrower than 1, or an interval like [0, 1e-61] in a deep
    tail) is integrated again divided by that first estimate, which makes the
    stopping test relative. With `base`, logf is the log relative to e^{base}
    (terms of O(1), free of the large cancelling logs), so the quadrature runs
    at QUAD_DPS digits (plus the digits `z_deriv` adds for a small step, plus
    `resolve`: the decades by which the integrand's width is below its distance
    from 0, so the nodes and knots near the peak stay distinct) and only base
    carries the extra ones."""
    ctx = mp.workdps(QUAD_DPS + resolve + _fd_extra[0]) if base is not None else mp.extradps(0)
    with ctx:
        top = logf(peak)

        def f(t):
            if t <= lo or t >= hi:
                return mpf(0)
            return mp.exp(logf(t) - top)
        knots = sorted(set([lo, hi] + [p for p in pts if lo < p < hi]))
        q = mp.quad(f, knots)
        if 0 < q < 1:
            q0 = q
            q = q0 * mp.quad(lambda t: f(t) / q0, knots)
    return mp.exp((base or 0) + top) * q


def gamma_tail(a, x, upper):
    """Regularized P(a, x) (upper=False) or Q(a, x), each to full relative
    precision (mpmath evaluates the requested tail itself). For a ≥ 1e6, where
    mpmath's series near x ≈ a takes seconds to minutes, and wherever it does not
    converge, quadrature of the smaller tail."""
    if x <= 0:
        return mpf(1) if upper else mpf(0)
    if mpmath.isinf(x):
        return mpf(0) if upper else mpf(1)
    if 1e-10 <= a < 1e6:
        try:
            if upper:
                return mp.gammainc(a, x, mp.inf, regularized=True)
            return mp.gammainc(a, 0, x, regularized=True)
        except (mpmath.libmp.NoConvergence, ValueError, ZeroDivisionError):
            pass
    small_upper = x >= a
    with mp.extradps(extra_digits([a, x]) + 10):
        lg = mp.loggamma(a)

        base = (a - 1) * mp.log(x) - x - lg

        def rel(t):
            return (a - 1) * mp.log1p((t - x) / x) - (t - x)
        mode = max(a - 1, mpf(0))
        sa = mp.sqrt(a)
        slope = abs((a - 1) / x - 1)
        w = min(1 / slope if slope else mp.inf, sa + 1)
        resolve = max(0, int(mp.ceil(mp.log10(x / min(w, sa))))) + 1
        if small_upper and a < 1:
            # t = e^s: Γ(a, x) = ∫ e^{a·s − e^s} ds from ln x, smooth where the
            # t-integrand spreads over hundreds of decades (a → 0).
            ls = mp.log(x)

            def logh(u):
                return a * u - mp.exp(u) - lg
            # Past s = ln x + 40 the integrand is below e^{−x·e^{40}}.
            top_s = max(mpf(8), ls + 40)
            pts = [ls + 20 * k for k in range(1, int(-ls / 20) + 1)] + [mpf(j) for j in (0, 1, 2, 3, 4, 5, 6, 8)]
            pts += [ls + mpf(2) ** k / x for k in range(-2, 8)]
            peak = mp.log(a) if mp.log(a) > ls else ls
            t = _quad_log(logh, ls, top_s, pts, peak)
        elif small_upper:
            pts = [x + w * 2 ** k for k in range(-2, 16)] + [mode + sa * j for j in (1, 2, 4, 8, 16, 32)]
            peak = mode if mode > x else x
            t = _quad_log(rel, x, mp.inf, pts, peak, base, resolve)
        elif a < 1:
            # t = x·v^{1/a} (t^{a−1}dt = (x^a/a)·dv) removes the singularity at 0.
            def logg(v):
                return -x * v ** (1 / a)
            pts = [1 - mpf(2) ** -k for k in range(1, 64)]
            t = mp.exp(a * mp.log(x) - mp.log(a) - lg) * _quad_log(logg, mpf(0), mpf(1), pts, mpf(0))
        else:
            pts = [x - w * 2 ** k for k in range(-2, 16)] + [mode - sa * j for j in (1, 2, 4, 8, 16, 32)]
            pts += [x * mpf(2) ** -k for k in range(1, 9)]
            peak = mode if 0 < mode < x else x
            t = _quad_log(rel, mpf(0), x, pts, peak, base, resolve)
        v = t if small_upper == upper else 1 - t
    return +v


def beta_lower(a, b, x, y):
    """I_x(a, b) with y = 1 − x (both exact), to full relative precision.
    Where x is within 1e-25 of 1, x at the working precision keeps too few
    digits of y, so the value is 1 − I_y(b, a), at a precision that keeps the
    difference's digits."""
    if x <= 0:
        return mpf(0)
    if y <= 0:
        return mpf(1)
    if x > 0.5 and y < mpf(10) ** -25:
        return difference(lambda: (mpf(1), _beta_direct(b, a, y, x)))
    return _beta_direct(a, b, x, y)


def _beta_direct(a, b, x, y):
    # mpmath's hypergeometric series takes ~20 s (or fails) once the larger
    # shape passes ~1e6; there the quadrature (validated against mpmath to
    # 1e-46 at shapes up to 5e6) is used.
    if min(a, b) < 1000 and max(a, b) < 1e6:
        try:
            return mp.betainc(a, b, 0, x, regularized=True)
        except (mpmath.libmp.NoConvergence, ValueError, ZeroDivisionError):
            pass
    # Quadrature of the smaller tail, always from 0 (the upper tail as
    # I_y(b, a)), so no 1 − t is formed near t = 1. For a ≤ 1 the substitution
    # in `_beta_quad` integrates I_x itself even where x is near 1, so `1 − x`
    # is never rebuilt from an x rounded to 1 (which the upper-tail form does).
    if x * (a + b) <= a or a <= 1:
        return _beta_quad(a, b, x)
    with mp.extradps(10):
        return 1 - _beta_quad(b, a, y)


def _beta_quad(a, b, x):
    """∫₀ˣ t^{a−1}(1−t)^{b−1} dt / B(a, b) by tanh-sinh. For a ≤ 1 through
    t = x·v^{1/a} (t^{a−1}dt = (x^a/a)·dv), which removes the singularity at 0
    (and at a = 1 the mode at 0, where the relative form below reads
    0·log1p(−1)); else with breakpoints on the integrand's own scales: 1/slope
    and 1/√curvature at x, and the mode ± multiples of the standard deviation."""
    if a <= 1:
        # The integrand has no cancelling terms, so the quadrature runs at the
        # caller's precision (which `difference` raises); only B(a, b) needs
        # the extra digits.
        with mp.extradps(extra_digits([a, b]) + 15):
            lnb = mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)
            pre = mp.exp(a * mp.log(x) - mp.log(a) - lnb)

        def logg(v):
            return (b - 1) * mp.log1p(-x * v ** (1 / a))
        pts = [1 - mpf(2) ** -k for k in range(1, 64)]
        return +(pre * _quad_log(logg, mpf(0), mpf(1), pts, mpf(0) if b >= 1 else mpf(1)))

    with mp.extradps(extra_digits([a, b]) + 15):
        lnb = mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)
        y = 1 - x
        base = (a - 1) * mp.log(x) + (b - 1) * mp.log(y) - lnb

        def rel(t):
            return (a - 1) * mp.log1p((t - x) / x) + (b - 1) * mp.log1p((x - t) / y)
        slope = abs((a - 1) / x - (b - 1) / y)
        curv = abs((a - 1) / x ** 2 + (b - 1) / y ** 2)
        w = min(1 / slope if slope else mp.inf, 1 / mp.sqrt(curv) if curv else mp.inf, x)
        # Past ~40 widths below x (or 12 sd below the mode) the integrand is
        # below 1e-17 of its peak on a log-concave integrand.
        pts = [x - w * 2 ** k for k in range(-1, 7)]
        s = a + b
        sd = mp.sqrt(a * b / (s * s * (s + 1)))
        peak = x
        if b > 1:
            mode = (a - 1) / (s - 2)
            pts += [mode + sd * j for j in (-12, -6, -3, -1, 0, 1, 3, 6, 12)]
            if mode < x:
                peak = mode
        resolve = max(0, int(mp.ceil(mp.log10(x / min(w, sd))))) + 1
        return +_quad_log(rel, mpf(0), x, pts, peak, base, resolve)


def log_beta(a, b):
    with mp.extradps(extra_digits([a, b]) + 5):
        return +(mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b))


def ncdf(z):
    """Φ(z). mpmath's erfc fails beyond ~1e154; from |w| = 1e10 on, the
    asymptotic series erfc(w) = e^{−w²}/(w√π)·Σ (−1)ⁿ(2n−1)!!/(2w²)ⁿ reaches
    the working precision within a few terms."""
    w = -z / mp.sqrt(2)
    if abs(w) < 10 ** 10:
        return mp.erfc(w) / 2
    aw = abs(w)
    s, term, n = mpf(1), mpf(1), 1
    while abs(term) > mpf(10) ** (-(mp.dps + 5)):
        term *= -mpf(2 * n - 1) / (2 * aw * aw)
        s += term
        n += 1
    tail = mp.exp(-aw * aw) / (aw * mp.sqrt(mp.pi)) * s / 2
    return tail if w > 0 else 1 - tail


def difference(fn):
    """fn() → (t1, t2); t1 − t2 at a precision where the difference keeps at
    least 30 digits (raised until it does)."""
    extra = 0
    while True:
        with mp.extradps(extra):
            t1, t2 = fn()
            d = t1 - t2
            scale = max(abs(t1), abs(t2))
            if d == 0 and scale == 0:
                return mpf(0)
            if scale == 0 or abs(d) >= scale * mpf(10) ** (-(mp.dps - 30)) or extra > 600:
                return +d
        extra += 60


# ---- continuous distributions ---------------------------------------------
# Each: cdf, sf, logpdf over mpf params P and x; `coord` picks the quantile
# solver's variable (`real`: x = c + s·sinh u, `pos`: x = e^u, `unit`: x = e^u
# below the median and 1 − e^u above it); `closed` gives an exact quantile.

class Cont:
    coord = "pos"
    lo, hi = 0, mp.inf

    def support(self, P):
        return (self.lo, self.hi)

    def center(self, P):
        return mpf(0), mpf(1)

    def closed(self, P, t, upper):
        return None

    def pdf(self, P, x):
        return mp.exp(self.logpdf(P, x))


class NormalT(Cont):
    coord = "real"
    lo, hi = -mp.inf, mp.inf

    def center(self, P):
        return P[0], P[1]

    def cdf(self, P, x):
        return ncdf((x - P[0]) / P[1])

    def sf(self, P, x):
        return ncdf(-(x - P[0]) / P[1])

    def logpdf(self, P, x):
        z = (x - P[0]) / P[1]
        return -z * z / 2 - mp.log(P[1]) - mp.log(2 * mp.pi) / 2


class StudentTT(Cont):
    coord = "real"
    lo, hi = -mp.inf, mp.inf

    def _half(self, df, x):
        # ½·I_{df/(df+x²)}(df/2, ½) = P(X > |x|), and its x²/(df + x²) mirror.
        d = df + x * x
        return beta_lower(df / 2, mpf(1) / 2, df / d, x * x / d) / 2

    def cdf(self, P, x):
        t = self._half(P[0], x)
        return t if x < 0 else 1 - t

    def sf(self, P, x):
        t = self._half(P[0], x)
        return t if x > 0 else 1 - t

    def logpdf(self, P, x):
        df = P[0]
        with mp.extradps(extra_digits([df]) + 5):
            return +(-log_beta(mpf(1) / 2, df / 2) - mp.log(df) / 2
                     - (df + 1) / 2 * mp.log1p(x * x / df))


class GammaT(Cont):
    def cdf(self, P, x):
        return gamma_tail(P[0], P[1] * x, False) if x > 0 else mpf(0)

    def sf(self, P, x):
        return gamma_tail(P[0], P[1] * x, True) if x > 0 else mpf(1)

    def logpdf(self, P, x):
        a, rate = P
        with mp.extradps(extra_digits([a, x * rate]) + 5):
            return +(a * mp.log(rate) + (a - 1) * mp.log(x) - rate * x - mp.loggamma(a))


class ChiSquaredT(GammaT):
    def _g(self, P):
        return [P[0] / 2, mpf(1) / 2]

    def cdf(self, P, x):
        return GammaT.cdf(self, self._g(P), x)

    def sf(self, P, x):
        return GammaT.sf(self, self._g(P), x)

    def logpdf(self, P, x):
        return GammaT.logpdf(self, self._g(P), x)


class FisherFT(Cont):
    def cdf(self, P, x):
        m, n = P
        if x <= 0:
            return mpf(0)
        d = n + m * x
        return beta_lower(m / 2, n / 2, m * x / d, n / d)

    def sf(self, P, x):
        m, n = P
        if x <= 0:
            return mpf(1)
        d = n + m * x
        return beta_lower(n / 2, m / 2, n / d, m * x / d)

    def logpdf(self, P, x):
        m, n = P
        with mp.extradps(extra_digits([m, n]) + 5):
            return +(m / 2 * mp.log(m / n) + (m / 2 - 1) * mp.log(x)
                     - (m + n) / 2 * mp.log1p(m * x / n) - log_beta(m / 2, n / 2))


class UniformT(Cont):
    coord = "closed"

    def support(self, P):
        return (P[0], P[1])

    def cdf(self, P, x):
        a, b = P
        return mpf(0) if x <= a else mpf(1) if x >= b else (x - a) / (b - a)

    def sf(self, P, x):
        a, b = P
        return mpf(1) if x <= a else mpf(0) if x >= b else (b - x) / (b - a)

    def logpdf(self, P, x):
        return -mp.log(P[1] - P[0])

    def closed(self, P, t, upper):
        a, b = P
        return b - t * (b - a) if upper else a + t * (b - a)


class ExponentialT(Cont):
    coord = "closed"

    def cdf(self, P, x):
        return -mp.expm1(-P[0] * x) if x > 0 else mpf(0)

    def sf(self, P, x):
        return mp.exp(-P[0] * x) if x > 0 else mpf(1)

    def logpdf(self, P, x):
        return mp.log(P[0]) - P[0] * x

    def closed(self, P, t, upper):
        if upper:
            return -mp.log(t) / P[0] if t > 0 else mp.inf
        return -mp.log1p(-t) / P[0] if t < 1 else mp.inf


class CauchyT(Cont):
    coord = "closed"
    lo, hi = -mp.inf, mp.inf

    def cdf(self, P, x):
        z = (x - P[0]) / P[1]
        return mp.acot(-z) / mp.pi if z < 0 else 1 - mp.acot(z) / mp.pi if z > 0 else mpf(1) / 2

    def sf(self, P, x):
        z = (x - P[0]) / P[1]
        return mp.acot(z) / mp.pi if z > 0 else 1 - mp.acot(-z) / mp.pi if z < 0 else mpf(1) / 2

    def logpdf(self, P, x):
        z = (x - P[0]) / P[1]
        return -mp.log(mp.pi * P[1]) - mp.log1p(z * z)

    def closed(self, P, t, upper):
        loc, s = P
        if t <= 0 or t >= 1:
            return (mp.inf if upper else -mp.inf) if t <= 0 else (-mp.inf if upper else mp.inf)
        # Lower-tail offset −s·cot(πt) for mass t below; the upper tail mirrors.
        # cot(π/2) is not exactly 0 at working precision.
        if t == mpf(1) / 2:
            return loc
        if t < mpf(1) / 2:
            off = -s * mp.cot(mp.pi * t)
        else:
            off = s * mp.cot(mp.pi * (1 - t))
        return loc - off if upper else loc + off


class WeibullT(Cont):
    coord = "closed"

    def cdf(self, P, x):
        return -mp.expm1(-(x / P[1]) ** P[0]) if x > 0 else mpf(0)

    def sf(self, P, x):
        return mp.exp(-(x / P[1]) ** P[0]) if x > 0 else mpf(1)

    def logpdf(self, P, x):
        k, lam = P
        return mp.log(k) - mp.log(lam) + (k - 1) * mp.log(x / lam) - (x / lam) ** k

    def closed(self, P, t, upper):
        k, lam = P
        e = -mp.log(t) if upper else -mp.log1p(-t)
        return lam * e ** (1 / k)


class LogNormalT(Cont):
    def cdf(self, P, x):
        return ncdf((mp.log(x) - P[0]) / P[1]) if x > 0 else mpf(0)

    def sf(self, P, x):
        return ncdf(-(mp.log(x) - P[0]) / P[1]) if x > 0 else mpf(1)

    def logpdf(self, P, x):
        if x == 0:
            return -mp.inf
        z = (mp.log(x) - P[0]) / P[1]
        return -z * z / 2 - mp.log(x) - mp.log(P[1]) - mp.log(2 * mp.pi) / 2


class BetaT(Cont):
    coord = "unit"
    lo, hi = 0, 1

    def cdf(self, P, x):
        return beta_lower(P[0], P[1], x, 1 - x)

    def sf(self, P, x):
        return beta_lower(P[1], P[0], 1 - x, x)

    def logpdf(self, P, x):
        a, b = P
        with mp.extradps(extra_digits([a, b]) + 5):
            # (shape − 1)·ln 0 is 0 at shape 1 (the factor is 1, not 0·∞).
            lx = (a - 1) * mp.log(x) if a != 1 else mpf(0)
            ly = (b - 1) * mp.log1p(-x) if b != 1 else mpf(0)
            return +(lx + ly - log_beta(a, b))


class InverseGaussianT(Cont):
    """e^{2λ/μ} needs its exponent to log10(λ/μ) digits beyond the result's."""

    def _ab(self, P, x):
        mu, lam = P
        r = mp.sqrt(lam / x)
        return r * (x / mu - 1), r * (x / mu + 1), mp.exp(2 * lam / mu)

    def cdf(self, P, x):
        with mp.extradps(extra_digits([P[1] / P[0]]) + 5):
            a, b, e = self._ab(P, x)
            return +(ncdf(a) + e * ncdf(-b))

    def sf(self, P, x):
        def terms():
            a, b, e = self._ab(P, x)
            return ncdf(-a), e * ncdf(-b)
        with mp.extradps(extra_digits([P[1] / P[0]]) + 5):
            return +difference(terms)

    def logpdf(self, P, x):
        mu, lam = P
        return mp.log(lam / (2 * mp.pi)) / 2 - mpf(3) / 2 * mp.log(x) - lam * (x - mu) ** 2 / (2 * mu * mu * x)


CONT = {
    "normal": NormalT(), "studentt": StudentTT(), "chisquared": ChiSquaredT(),
    "fisherf": FisherFT(), "uniform": UniformT(), "exponential": ExponentialT(),
    "cauchy": CauchyT(), "weibull": WeibullT(), "lognormal": LogNormalT(),
    "gamma": GammaT(), "beta": BetaT(), "inversegaussian": InverseGaussianT(),
}


# ---- discrete distributions -------------------------------------------------
# Each: support(P) → (lo, hi or None), logmass, cdf, sf over mpf/int params
# and an int k; `real` lists the indices of the real parameters.

class Disc:
    real = ()

    def mass(self, P, k):
        return mp.exp(self.logmass(P, k))


class BernoulliT(Disc):
    real = (0,)

    def support(self, P):
        return 0, 1

    def logmass(self, P, k):
        p = P[0]
        if k == 0:
            # log1p: 1 − p at the working precision is 1 for p < 1e-60.
            return mp.log1p(-p) if p < 1 else -mp.inf
        return mp.log(p) if k == 1 and p > 0 else -mp.inf

    def cdf(self, P, k):
        return mpf(0) if k < 0 else 1 - P[0] if k == 0 else mpf(1)

    def sf(self, P, k):
        return mpf(1) if k < 0 else P[0] if k == 0 else mpf(0)


class BinomialT(Disc):
    real = (1,)

    def support(self, P):
        return 0, P[0]

    def logmass(self, P, k):
        n, p = P
        if k < 0 or k > n:
            return -mp.inf
        with mp.extradps(extra_digits([n]) + 5):
            return +(mp.loggamma(n + 1) - mp.loggamma(k + 1) - mp.loggamma(n - k + 1)
                     + (k * mp.log(p) if k else 0) + ((n - k) * mp.log1p(-p) if n - k else 0))

    def cdf(self, P, k):
        n, p = P
        if k < 0:
            return mpf(0)
        if k >= n:
            return mpf(1)
        return beta_lower(mpf(n - k), mpf(k + 1), 1 - p, p)

    def sf(self, P, k):
        n, p = P
        if k < 0:
            return mpf(1)
        if k >= n:
            return mpf(0)
        return beta_lower(mpf(k + 1), mpf(n - k), p, 1 - p)


class PoissonT(Disc):
    real = (0,)

    def support(self, P):
        return 0, None

    def logmass(self, P, k):
        lam = P[0]
        if k < 0:
            return -mp.inf
        with mp.extradps(extra_digits([lam, k]) + 5):
            return +(-lam + k * mp.log(lam) - mp.loggamma(k + 1))

    def cdf(self, P, k):
        return gamma_tail(mpf(k + 1), P[0], True) if k >= 0 else mpf(0)

    def sf(self, P, k):
        return gamma_tail(mpf(k + 1), P[0], False) if k >= 0 else mpf(1)


class GeometricT(Disc):
    real = (0,)

    def support(self, P):
        return 1, (1 if P[0] == 1 else None)

    def logmass(self, P, k):
        p = P[0]
        if k < 1:
            return -mp.inf
        if p == 1:
            return mpf(0) if k == 1 else -mp.inf
        return (k - 1) * mp.log1p(-p) + mp.log(p)

    def cdf(self, P, k):
        p = P[0]
        if k < 1:
            return mpf(0)
        if p == 1:
            return mpf(1)
        return -mp.expm1(k * mp.log1p(-p))

    def sf(self, P, k):
        p = P[0]
        if k < 1:
            return mpf(1)
        if p == 1:
            return mpf(0)
        return mp.exp(k * mp.log1p(-p))


class NegBinomialT(Disc):
    """Params (r, p), success probability p; `ms` for (mean μ, size θ):
    r = θ, p = μ/(μ + θ), q = θ/(μ + θ) exactly."""
    real = (0, 1)

    def __init__(self, ms=False):
        self.ms = ms

    def rpq(self, P):
        if self.ms:
            mu, size = P
            return size, mu / (mu + size), size / (mu + size)
        r, p = P
        return r, p, 1 - p

    def support(self, P):
        return 0, None

    def logmass(self, P, k):
        r, p, q = self.rpq(P)
        if k < 0:
            return -mp.inf
        with mp.extradps(extra_digits([r, k]) + 5):
            return +(mp.loggamma(k + r) - mp.loggamma(r) - mp.loggamma(k + 1)
                     + r * mp.log(q) + (k * mp.log(p) if k else 0))

    def cdf(self, P, k):
        r, p, q = self.rpq(P)
        return beta_lower(r, mpf(k + 1), q, p) if k >= 0 else mpf(0)

    def sf(self, P, k):
        r, p, q = self.rpq(P)
        return beta_lower(mpf(k + 1), r, p, q) if k >= 0 else mpf(1)


class HypergeometricT(Disc):
    def support(self, P):
        nn, kk, n = P
        return max(0, n + kk - nn), min(n, kk)

    def logmass(self, P, k):
        nn, kk, n = P
        lo, hi = self.support(P)
        if k < lo or k > hi:
            return -mp.inf

        def lc(a, b):
            return mp.loggamma(a + 1) - mp.loggamma(b + 1) - mp.loggamma(a - b + 1)
        with mp.extradps(extra_digits([nn]) + 5):
            return +(lc(kk, k) + lc(nn - kk, n - k) - lc(nn, n))

    def _sum(self, P, a, b):
        """Σ_{j=a..b} mass(j) by the pmf ratio from mass(a)."""
        nn, kk, n = P
        if a > b:
            return mpf(0)
        rest = nn - kk - n
        f = self.mass(P, a)
        s = f
        for j in range(a, b):
            f = f * (kk - j) * (n - j) / ((j + 1) * (rest + j + 1))
            s += f
        return s

    def cdf(self, P, k):
        lo, hi = self.support(P)
        return mpf(0) if k < lo else mpf(1) if k >= hi else self._sum(P, lo, k)

    def sf(self, P, k):
        lo, hi = self.support(P)
        return mpf(1) if k < lo else mpf(0) if k >= hi else self._sum(P, k + 1, hi)


DISC = {
    "bernoulli": BernoulliT(), "binomial": BinomialT(), "poisson": PoissonT(),
    "geometric": GeometricT(), "negbinomial": NegBinomialT(),
    "negbinomial_ms": NegBinomialT(ms=True), "hypergeometric": HypergeometricT(),
}


# ---- truth of one point -----------------------------------------------------

def ln_tail(v, other):
    """ln of a tail value v, as log1p(−other()) where v > ½: ln v of a value
    within 10⁻⁶⁰ of 1 would read 0 at the working precision."""
    if v <= 0:
        raise TruthError("tail value not positive")
    return mp.log1p(-other()) if v > 0.5 else mp.log(v)


def _gamma_dx(P, x):
    """|x·∂P/∂x| = x^a·e^{−x}/Γ(a)."""
    a = P[0]
    with mp.extradps(extra_digits([a, x]) + 5):
        return +mp.exp(a * mp.log(x) - x - mp.loggamma(a))


def _beta_dx(P, x):
    """|x·∂I_x(a, b)/∂x| = x^a·(1 − x)^{b−1}/B(a, b)."""
    a, b = P
    with mp.extradps(extra_digits([a, b]) + 5):
        return +mp.exp(a * mp.log(x) + (b - 1) * mp.log1p(-x) - log_beta(a, b))


def _gp(P, x):
    return gamma_tail(P[0], x, False)


def _gq(P, x):
    return gamma_tail(P[0], x, True)


def _bi(P, x):
    return beta_lower(P[0], P[1], x, 1 - x)


def _bc(P, x):
    return beta_lower(P[1], P[0], 1 - x, x)


# name → (f(P, x), |x·∂f/∂x|(P, x) or None for a finite difference).
SPECIAL_VALUE = {
    "gammp": (_gp, _gamma_dx),
    "gammq": (_gq, _gamma_dx),
    "ln_gammp": (lambda P, x: ln_tail(_gp(P, x), lambda: _gq(P, x)), lambda P, x: _gamma_dx(P, x) / _gp(P, x)),
    "ln_gammq": (lambda P, x: ln_tail(_gq(P, x), lambda: _gp(P, x)), lambda P, x: _gamma_dx(P, x) / _gq(P, x)),
    "betai": (_bi, _beta_dx),
    "ln_betai": (lambda P, x: ln_tail(_bi(P, x), lambda: _bc(P, x)), lambda P, x: _beta_dx(P, x) / _bi(P, x)),
    "lgamma": (lambda P, x: mp.loggamma(x), lambda P, x: abs(x * mp.digamma(x))),
    "lbeta": (lambda P, x: log_beta(P[0], x), lambda P, x: abs(x * (mp.digamma(x) - mp.digamma(P[0] + x)))),
}


def value_fn(name):
    """(f(P, x), indices of the real params (None: all), x real?, analytic
    |x·∂f/∂x| or None) for a non-quantile name."""
    if name in SPECIAL_VALUE:
        f, dx = SPECIAL_VALUE[name]
        return f, None, True, dx
    ty, method = name.split(".")
    if ty in CONT:
        d = CONT[ty]

        def xpdf(P, x):
            return abs(x) * d.pdf(P, x)
        # At or past an end of the support the tail is exactly 0 and its ln −∞;
        # elsewhere a zero tail is a truth failure (`ln_tail`).
        f, dx = {
            "cdf": (d.cdf, xpdf), "sf": (d.sf, xpdf), "pdf": (d.pdf, None), "log_density": (d.logpdf, None),
            "ln_cdf": (lambda P, x: -mp.inf if x <= d.support(P)[0] else ln_tail(d.cdf(P, x), lambda: d.sf(P, x)),
                       lambda P, x: xpdf(P, x) / d.cdf(P, x)),
            "ln_sf": (lambda P, x: -mp.inf if x >= d.support(P)[1] else ln_tail(d.sf(P, x), lambda: d.cdf(P, x)),
                      lambda P, x: xpdf(P, x) / d.sf(P, x)),
        }[method]
        return f, None, True, dx
    d = DISC[ty]
    f = {"mass": d.mass, "log_mass": d.logmass, "cdf": d.cdf, "sf": d.sf}[method]
    return f, d.real, False, None


# Digits `z_deriv` currently adds to the working precision (one slot, per
# process). The quadratures follow only these: the callers' own extra digits
# serve formulas that cancel, which the relative integrand does not.
_fd_extra = [0]


def _safe(g, z):
    try:
        v = g(z)
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return v if mpmath.isfinite(v) else None


def z_deriv(g, z, v0):
    """z·dg/dz at z (g of one variable, v0 = g(z)) by central differences of
    relative step h. The forward and backward quotients differ by h·z²·g''
    (to leading order), so a one-sided quotient carries the curvature, not the
    slope, where g'' is large (a log density at a shape of 1e300 has curvature
    ~1e300 in ln k). h shrinks, with the working precision raised to keep
    ~40 digits of the differences, until the two quotients agree to 1 % of
    max(|slope|, |g|): K is then good to that fraction of the scale r uses.
    One-sided (Richardson on h and h/2) where a step leaves the domain."""
    h = FD_STEP
    extra = 0
    best = None
    saved = _fd_extra[0]
    try:
        return _z_deriv_loop(g, z, v0, h, extra, best)
    finally:
        _fd_extra[0] = saved


def _z_deriv_loop(g, z, v0, h, extra, best):
    for _ in range(12):
        _fd_extra[0] = extra
        with mp.extradps(extra):
            v = v0 if extra == 0 else g(z)
            fp, fm = _safe(g, z * (1 + h)), _safe(g, z * (1 - h))
            if fp is not None and fm is not None:
                dp, dm = (fp - v) / h, (v - fm) / h
                c, spread = (dp + dm) / 2, abs(dp - dm)
            else:
                sgn = 1 if fp is not None else -1
                f1 = fp if fp is not None else fm
                f2 = _safe(g, z * (1 + sgn * h / 2))
                if f1 is None or f2 is None:
                    return best
                d1, d2 = sgn * (f1 - v) / h, sgn * (f2 - v) / (h / 2)
                c, spread = 2 * d2 - d1, abs(d1 - d2)
            best = +c
            scale = max(abs(c), abs(v))
            if spread <= scale / 100:
                return best
        # The spread is linear in h: aim at 1e-4 of the scale.
        shrink = min(mpf(10) ** -3, scale / spread / 10 ** 4) if scale > 0 else mpf(10) ** -10
        h *= shrink
        if h < mpf(10) ** -700:
            return best
        extra = max(0, int(-mp.log10(h)) + 40 - DPS)
    return best


def sensitivity(f, P, x, real_idx, x_real, v0, dx=None):
    """K = Σ |z·∂f/∂z| over the real inputs z: `dx` for x where given, else
    `z_deriv`. v0 = f(P, x)."""
    K = mpf(0)
    idx = range(len(P)) if real_idx is None else real_idx
    slots = [("p", i) for i in idx if not isinstance(P[i], int)] + ([("x", 0)] if x_real else [])
    for kind, i in slots:
        z = P[i] if kind == "p" else x
        if z == 0 or mpmath.isinf(z):
            continue
        if kind == "x" and dx is not None:
            K += abs(dx(P, x))
            continue
        if kind == "p":
            def g(zz, i=i):
                Q = list(P)
                Q[i] = zz
                return f(Q, x)
        else:
            def g(zz):
                return f(P, zz)
        d = z_deriv(g, z, v0)
        if d is not None:
            K += abs(d)
    return K


def solve_cont(ty, P, t, upper, seed):
    """Root x of cdf(x) = t (upper=False) or sf(x) = t, solved on the smaller
    tail: (root, mass fn, pdf at root). Root ±inf beyond ±f64 max, 0 below
    2⁻¹⁰⁷⁵ on a positive support."""
    d = CONT[ty]
    if t <= 0 or t >= 1:
        lo, hi = d.support(P)
        at_hi = (t >= 1) != upper
        return (mpf(hi) if at_hi else mpf(lo)), None
    # Smaller tail: `side_upper` true when solving sf(x) = m.
    if t <= mpf(1) / 2:
        side_upper, m = upper, t
    else:
        side_upper, m = not upper, 1 - t
    mass = d.sf if side_upper else d.cdf
    closed = d.closed(P, t, upper)
    if closed is not None:
        return closed, mass
    ln_m = mp.log(m)

    # u → x, and whether x rises with u. On [0, 1] the variable follows the
    # side of ½ the root is on (e^u below, 1 − e^u above), so a root near
    # either end keeps its digits whichever tail mass is solved for.
    near_one = False
    if d.coord == "unit":
        # The crate's answer places the root where it is in (0, 1): cdf(½)
        # at extreme shapes costs more than the whole solve.
        if seed is not None and 0 < seed < 1:
            near_one = seed > 0.5
        else:
            near_one = d.cdf(P, mpf(1) / 2) < (1 - t if upper else t)
    if d.coord == "real":
        c, s = d.center(P)

        def xof(u):
            return c + s * mp.sinh(u)

        def uof(x):
            return mp.asinh((x - c) / s)
    elif near_one:
        def xof(u):
            return 1 - mp.exp(u)

        def uof(x):
            return mp.log(1 - x) if x < 1 else mpf(-2000)
    else:
        def xof(u):
            return mp.exp(u)

        def uof(x):
            return mp.log(x) if x > 0 else mpf(-2000)
    rising = not near_one

    def g(u):
        # > 0 when u is past the root in the direction of growing mass.
        v = mass(P, xof(u))
        if v <= 0:
            return -mp.inf
        return mp.log(v) - ln_m
    # g increases with u iff the mass increases with u.
    inc = rising != side_upper

    # Range checks for a positive support: root above f64 max or below 2^-1075.
    if d.coord == "pos" or (d.coord == "unit" and not near_one):
        for xe, beyond_hi in ((mpf(F64_MAX), True), (mpf(2) ** -1075, False)):
            if d.coord == "unit" and beyond_hi:
                continue
            ge = g(uof(xe))
            # Root beyond xe: at xe the mass has not yet reached m on the way.
            root_beyond = (ge < 0) == inc if beyond_hi else (ge > 0) == inc
            if root_beyond:
                return (mp.inf if beyond_hi else mpf(0)), mass
    if d.coord == "real":
        for xe, beyond_hi in ((mpf(F64_MAX), True), (-mpf(F64_MAX), False)):
            ge = g(uof(xe))
            root_beyond = (ge < 0) == inc if beyond_hi else (ge > 0) == inc
            if root_beyond:
                return (mp.inf if beyond_hi else -mp.inf), mass

    u0 = None
    if seed is not None and mpmath.isfinite(seed):
        lo, hi = d.support(P)
        if lo < seed < hi:
            u0 = uof(seed)
    if u0 is None:
        u0 = mpf(0) if d.coord != "unit" else mpf(-1)
    def gd(u):
        """(g, dg/du): d ln M/du = ±pdf·x'(u)/M, + for the cdf, − for the sf."""
        x = xof(u)
        v = mass(P, x)
        if v <= 0:
            return -mp.inf, mp.nan
        dxdu = s * mp.cosh(u) if d.coord == "real" else (-mp.exp(u) if near_one else mp.exp(u))
        return mp.log(v) - ln_m, (-1 if side_upper else 1) * d.pdf(P, x) * dxdu / v

    # Bracket: step outward from u0 (the crate's own answer where it is in
    # the support, so the bracket is usually a few ulp wide), growing ×8.
    g0 = g(u0)
    if g0 == 0:
        return xof(u0), mass
    direction = -1 if (g0 > 0) == inc else 1
    ua = u0
    step = max(abs(u0), mpf(1)) * mpf(10) ** -12
    for _ in range(80):
        ub = ua + direction * step
        if d.coord == "unit" and ub >= 0:
            ub = ua / 2 if ua < 0 else mpf(-1e-30)
        gb = g(ub)
        if (gb > 0) != (g0 > 0) or gb == 0:
            break
        ua = ub
        step *= 8
    else:
        raise TruthError("no bracket")
    lo_u, hi_u = (ua, ub) if ua < ub else (ub, ua)
    lo_pos = g(lo_u) > 0
    # Newton on g, kept inside the bracket (bisection where a step leaves it,
    # as NR's rtsafe), to 40 digits in u.
    tol = mpf(10) ** -40
    u = (lo_u + hi_u) / 2
    for _ in range(200):
        gv, gp = gd(u)
        if gv == 0:
            break
        if (gv > 0) == lo_pos:
            lo_u = u
        else:
            hi_u = u
        nu = u - gv / gp if mpmath.isfinite(gv) and mpmath.isfinite(gp) and gp != 0 else None
        if nu is None or not (lo_u < nu < hi_u):
            nu = (lo_u + hi_u) / 2
        if abs(nu - u) <= tol * max(1, abs(u)) or hi_u - lo_u <= tol * max(1, abs(u)):
            u = nu
            break
        u = nu
    else:
        raise TruthError("root did not converge")
    return xof(u), mass


def cont_quantile_truth(ty, P, t, upper, seed, root=None):
    """(root, K) of quantile(t) (upper=False) or isf(t); with `root` (a
    cached one) only K is computed."""
    d = CONT[ty]
    if root is None:
        x, mass = solve_cont(ty, P, t, upper, seed)
    else:
        x = root
        side_upper = upper if t <= mpf(1) / 2 else not upper
        mass = None if t <= 0 or t >= 1 else d.sf if side_upper else d.cdf
    if mass is None or not mpmath.isfinite(x) or x == 0 and d.coord == "pos":
        return x, mpf(0)
    lo, hi = d.support(P)
    if not (lo < x < hi):
        return x, mpf(0)
    pdf = d.pdf(P, x)
    if pdf == 0:
        return x, mpf(0)

    def fm(Q, _x):
        return mass(Q, x)
    kp = sensitivity(fm, P, x, None, False, mass(P, x))
    return x, (abs(t) + kp) / pdf


def disc_quantile_truth(ty, P, p, seed):
    """Smallest k in the support with cdf(k) ≥ p, and the tail values at k−1
    and k on the side used (cdf for p ≤ ½, else sf, compared with 1 − p)."""
    d = DISC[ty]
    lo, hi = d.support(P)
    use_sf = p > mpf(1) / 2
    q = 1 - p

    memo = {}

    def tail(P_, k):
        if k not in memo:
            memo[k] = d.sf(P, k) if use_sf else d.cdf(P, k)
        return memo[k]

    def ok(k):
        if hi is not None and k >= hi:
            return True
        return tail(P, k) <= q if use_sf else tail(P, k) >= p

    if p <= 0:
        return lo, None, None, use_sf
    k0 = seed if seed is not None and lo <= seed and (hi is None or seed <= hi) else lo
    if ok(k0):
        good, step = k0, 1
        bad = None
        while good > lo:
            k = max(lo, good - step)
            if ok(k):
                good = k
                step *= 2
            else:
                bad = k
                break
        if bad is None:
            k_star = good
        else:
            while good - bad > 1:
                mid = (good + bad) // 2
                if ok(mid):
                    good = mid
                else:
                    bad = mid
            k_star = good
    else:
        bad, step = k0, 1
        while True:
            k = bad + step
            if hi is not None and k > hi:
                k = hi
            if ok(k):
                good = k
                break
            bad = k
            step *= 2
            if step > 2 ** 70:
                raise TruthError("no upper bracket")
        while good - bad > 1:
            mid = (good + bad) // 2
            if ok(mid):
                good = mid
            else:
                bad = mid
        k_star = good
    before = tail(P, k_star - 1) if k_star > lo else None
    at = tail(P, k_star)
    return k_star, before, at, use_sf


def mstr(v):
    if v is None:
        return None
    if isinstance(v, int):
        return str(v)
    if mpmath.isnan(v):
        return "nan"
    if mpmath.isinf(v):
        return "inf" if v > 0 else "-inf"
    return mpmath.nstr(v, 30, min_fixed=0, max_fixed=0)


def truth_job(job):
    """Truth of one point; `job` = (fn, params, x, seed) with the inputs as the
    probe reads them. Returns (fn, key, entry)."""
    name, params, x, seed, prior = job
    key = cache_key(params, x)
    entry = {}
    old = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(TRUTH_TIMEOUT)
    try:
        P = [to_mp(s) for s in params]
        X = to_mp(x)
        # Formulas that cancel at large inputs add their own digits.
        mp.dps = DPS
        kind = FN[name]["kind"]
        if kind == "quantile":
            if name == "inv_beta_reg":
                ty, upper = "beta", False
            else:
                ty, method = name.split(".")
                upper = method == "isf"
            sd = mpf(float(seed)) if seed not in (None, "") else None
            known = None
            if prior is not None:
                known = {"inf": mp.inf, "-inf": -mp.inf}.get(prior) or mpf(prior)
            root, K = cont_quantile_truth(ty, P, mpf(X), upper, sd, known)
            entry = {"t": prior if prior is not None else mstr(root), "K": mstr(K)}
        elif kind == "dquantile":
            ty = name.split(".")[0]
            sd = int(seed) if seed not in (None, "") else None
            k, before, at, use_sf = disc_quantile_truth(ty, P, mpf(X), sd)
            entry = {"t": str(k), "before": mstr(before), "at": mstr(at), "sf": use_sf}
        else:
            f, real_idx, x_real, dx = value_fn(name)
            v = f(P, X)
            K = sensitivity(f, P, X, real_idx, x_real, v, dx) if mpmath.isfinite(v) else mpf(0)
            entry = {"t": mstr(v), "K": mstr(K)}
        if "K" in entry:
            entry["kv"] = K_VERSION
    except TruthTimeout:
        entry = {"fail": "timeout", "tv": TIMEOUT_VERSION}
    except Exception as e:  # noqa: BLE001 - any mpmath failure is recorded, not fatal
        entry = {"fail": f"{type(e).__name__}: {str(e)[:120]}"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return name, key, entry


class TruthTimeout(Exception):
    pass


def _timeout(signum, frame):
    raise TruthTimeout()


def cache_key(params, x):
    return json.dumps([list(params), x], separators=(",", ":"))


# ============================================================================
# Functions and point generation
# ============================================================================
# FN[name] = {group, kind, dist}. group: the fixture file / table
# (`incomplete`, `special`, `continuous`, `discrete`). kind: `prob` (a
# probability), `value`, `log`, `quantile`, `dquantile` (discrete quantile).

CONT_METHODS = {"cdf": "prob", "sf": "prob", "pdf": "value", "log_density": "log",
                "quantile": "quantile", "isf": "quantile"}
LN_VARIANTS = {"fisherf": ("ln_cdf", "ln_sf"), "gamma": ("ln_cdf",),
               "inversegaussian": ("ln_cdf", "ln_sf")}
DISC_METHODS = {"mass": "prob", "log_mass": "log", "cdf": "prob", "sf": "prob",
                "quantile": "dquantile"}
FN = {}
for _n, _k in (("gammp", "prob"), ("gammq", "prob"), ("ln_gammp", "log"), ("ln_gammq", "log"),
               ("betai", "prob"), ("ln_betai", "log")):
    FN[_n] = {"group": "incomplete", "kind": _k, "dist": None}
for _n, _k in (("lgamma", "log"), ("lbeta", "log"), ("inv_beta_reg", "quantile")):
    FN[_n] = {"group": "special", "kind": _k, "dist": None}
for _t in CONT:
    for _m, _k in CONT_METHODS.items():
        FN[f"{_t}.{_m}"] = {"group": "continuous", "kind": _k, "dist": _t}
    for _m in LN_VARIANTS.get(_t, ()):
        FN[f"{_t}.{_m}"] = {"group": "continuous", "kind": "log", "dist": _t}
for _t in DISC:
    for _m, _k in DISC_METHODS.items():
        FN[f"{_t}.{_m}"] = {"group": "discrete", "kind": _k, "dist": _t}

# Probabilities at which points are placed (lower tail by `quantile`, upper by
# `isf`), and upper-tail masses q for `quantile(1 − q)` / `isf(1 − q)`.
P_LEVELS = [0.5, 0.1, 1e-3, 1e-6, 1e-10, 1e-20, 1e-50, 1e-100, 1e-200, 1e-300]
UPPER_Q = [0.1, 1e-3, 1e-6, 1e-10, 1e-15]
SUBNORMAL_P = [1e-310, 5e-324]  # quantile arguments only
TAIL_P = 1e-6  # a point whose smaller tail mass is below this is a tail point
N_RANDOM = 24  # random parameter sets per distribution


def lu(rng, lo, hi):
    """Log-uniform draw on [lo, hi]."""
    return 10 ** rng.uniform(math.log10(lo), math.log10(hi))


def signed(rng, v):
    return v if rng.random() < 0.5 else -v


def near(t):
    """A threshold t with its f64 neighbours and ±1 % (the sides of a branch)."""
    if not math.isfinite(t):
        return []
    return [t, math.nextafter(t, math.inf), math.nextafter(t, -math.inf), t * 1.01, t * 0.99]


def fstr(v):
    """An f64 input as the probe reads it: `repr` (shortest round-trip,
    always with a `.`, an exponent, `inf` or `nan`)."""
    return repr(float(v))


# ---- regimes: the kernel branch (or parameter range) a point exercises -----
# Mirrors of the branch conditions in the source; each names the branch.

POW_SWITCH = 16.0


def gamma_regime(a, x):
    if not math.isfinite(x):
        return "x=inf"
    if a < 1 and x < 1.1:
        return "small shape (a<1, x<1.1)"
    if a >= 100 and abs(x - a) <= 0.4 * a:
        return "Temme (a>=100, x/a in [0.6, 1.4])"
    upper = x >= a + 1 or (a < 1 and x >= 1.1)
    lx = math.log(x) if x > 0 else -math.inf
    pow_ = a < 8 and abs(a * lx) > POW_SWITCH
    side = "continued fraction (x>=a+1 or a<1, x>=1.1)" if upper else "series (x<a+1)"
    if a >= 8:
        return f"{side}, a>=8"
    return f"{side}, a<8" + (", pow" if pow_ else "")


def bratio_upper(a, b, x):
    """incomplete.rs `small_shape_upper`: for min(a, b) <= 1 and x <= 1/2, the
    route by which it evaluates 1 − I_x(a, b) directly, or None (mirrors
    that function; change together)."""
    if b < 1e-15 * min(a, 1.0):
        return None
    if max(a, b) > 1:
        if b <= 1 or (x < 0.1 and (x * b) ** a <= 0.7):
            return None
        if x >= 0.29:
            return "bpser"
        if b > 15:
            return "bgrat"
    else:
        if a >= min(b, 0.2) or x ** a <= 0.9:
            return None
        if x >= 0.3:
            return "bpser"
    return "bup+bgrat"


def beta_regime(a, b, x):
    y = 1.0 - x
    lam = (a + b) * y - b if a > b else a - (a + b) * x
    m = min(a, b)
    if m > 100 and abs(lam) <= 0.03 * m:
        return "basym (min>100, near mean)"
    lower = lam > 2 * x - 1
    if m <= 1 and lower != (x > 0.5):
        route = bratio_upper(a, b, x) if lower else bratio_upper(b, a, y)
        if route:
            return f"{route} (shape<=1, " + ("complement)" if lower else "direct)")
    side = "bfrac" if lower else "bfrac complement"
    return f"{side}, min(a,b)" + ("<8" if m < 8 else ">=8")


def shape_bucket(a, cuts=(1, 8, 100)):
    lo = None
    for c in cuts:
        if a < c:
            return f"<{c}" if lo is None else f"{lo}-{c}"
        lo = c
    return f">={cuts[-1]}"


DIST_REGIME = {
    "normal": lambda P: "all",
    "studentt": lambda P: "df " + shape_bucket(P[0], (1, 200)),
    "chisquared": lambda P: "k/2 " + shape_bucket(P[0] / 2),
    "fisherf": lambda P: "dfn/2 " + ("<8" if P[0] < 16 else ">=8") + ", dfd/2 " + ("<8" if P[1] < 16 else ">=8"),
    "uniform": lambda P: "all",
    "exponential": lambda P: "all",
    "cauchy": lambda P: "all",
    "weibull": lambda P: "k<1" if P[0] < 1 else "k>=1",
    "lognormal": lambda P: "all",
    "gamma": lambda P: "shape " + shape_bucket(P[0]),
    "beta": lambda P: "min " + shape_bucket(min(P), (8, 100)),
    "inversegaussian": lambda P: "shape/mean<1" if P[1] < P[0] else "shape/mean>=1",
    "bernoulli": lambda P: "all",
    "binomial": lambda P: "n<1e6" if P[0] < 1e6 else "n>=1e6",
    "poisson": lambda P: "lambda<10" if P[0] < 10 else "lambda>=10",
    "geometric": lambda P: "all",
    "negbinomial": lambda P: "r<8" if P[0] < 8 else "r>=8",
    "negbinomial_ms": lambda P: "size<8" if P[1] < 8 else "size>=8",
    "hypergeometric": lambda P: "N<1e6" if P[0] < 1e6 else "N>=1e6",
}


def regime_of(fn, P, x):
    """P: the parameters as floats/ints; x: the argument as float/int."""
    if fn in ("gammp", "gammq", "ln_gammp", "ln_gammq"):
        return gamma_regime(P[0], x)
    if fn in ("betai", "ln_betai"):
        return beta_regime(P[0], P[1], x) if 0 < x < 1 else "x at an end"
    if fn == "inv_beta_reg":
        side = "subnormal p" if x < 2.0 ** -1022 else "p<=1/2" if x <= 0.5 else "p>1/2"
        return f"{side}, min(a,b)" + ("<8" if min(P) < 8 else ">=8")
    if fn == "lgamma":
        return ("ln_gamma_1p - ln x (x<=0.8)" if x <= 0.8 else "ln_gamma_1p (x<=2.25)" if x <= 2.25
                else "recurrence (x<10)" if x < 10 else "Stirling (x>=10)")
    if fn == "lbeta":
        a, b = sorted((P[0], x))
        return "both>=8" if a >= 8 else "larger>=8" if b >= 8 else "both<8"
    return DIST_REGIME[FN[fn]["dist"]](P)


# ---- parameter sets ---------------------------------------------------------

def cont_params(ty, rng):
    """Random draws, then the extremes and branch-boundary parameter sets."""
    R = []
    for _ in range(N_RANDOM):
        if ty == "normal":
            R.append([0.0 if rng.random() < 0.3 else signed(rng, lu(rng, 1e-10, 1e10)), lu(rng, 1e-10, 1e10)])
        elif ty == "studentt":
            R.append([lu(rng, 1e-3, 1e12)])
        elif ty == "chisquared":
            R.append([lu(rng, 1e-3, 1e12)])
        elif ty == "fisherf":
            R.append([lu(rng, 1e-3, 1e12), lu(rng, 1e-3, 1e12)])
        elif ty == "uniform":
            a = signed(rng, lu(rng, 1e-10, 1e10))
            R.append([a, a + lu(rng, 1e-6, 1e10) * max(1.0, abs(a)) * 1e-6])
        elif ty == "exponential":
            R.append([lu(rng, 1e-100, 1e100)])
        elif ty == "cauchy":
            R.append([0.0 if rng.random() < 0.3 else signed(rng, lu(rng, 1e-10, 1e10)), lu(rng, 1e-100, 1e100)])
        elif ty == "weibull":
            R.append([lu(rng, 1e-2, 1e3), lu(rng, 1e-100, 1e100)])
        elif ty == "lognormal":
            R.append([rng.uniform(-700, 700), lu(rng, 1e-3, 30)])
        elif ty == "gamma":
            R.append([lu(rng, 1e-3, 1e12), lu(rng, 1e-100, 1e100)])
        elif ty == "beta":
            R.append([lu(rng, 1e-3, 1e12), lu(rng, 1e-3, 1e12)])
        elif ty == "inversegaussian":
            R.append([lu(rng, 1e-50, 1e50), lu(rng, 1e-50, 1e50)])
    E = {
        "normal": [[0.0, 1.0], [0.5, 2.0], [0.0, 1e-300], [0.0, 1e300], [1e300, 1.0], [-1e300, 1e290]],
        "studentt": [[1e-10], [0.1], [1.0], [1.0 + 1e-10], [2.0], [3.0], [199.9], [200.0], [200.1], [1e15], [1e20]],
        "chisquared": [[1e-300], [0.0027530089082596562], [1.0], [2.0], [15.99], [16.0], [199.99], [200.0],
                       [486622.4754497569], [1e15], [1e20]],
        "fisherf": [[0.0015205765296369873, 2057.918596622136], [534536453686.36383, 6.913288301642686],
                    [1e4, 1e20], [1e20, 1.0], [1.0, 1.0], [5.0, 10.0], [16.0, 16.0], [15.99, 16.01],
                    [1e-3, 1e-3], [0.01, 10.0], [10.0, 0.01], [1e20, 1e20],
                    [1e-5, 1e4], [1e-20, 5.0], [1e-100, 16.0], [1e-300, 15.99], [1.99, 5.0], [2.0, 5.0],
                    [5.0, 1.99], [5.0, 2.0], [16.0, 1e-100], [15.99, 1e-300]],
        "uniform": [[-383.03635179613127, -304.1942570061313], [0.0, 1.0], [-1e300, 1e300], [1.0, 1.0 + 2.0 ** -40]],
        "exponential": [[1.0], [2.0], [1e-300], [1e300]],
        "cauchy": [[0.0, 1.0], [0.0, 1e-100], [1e10, 1.0], [0.0, 1e300], [-5.0, 1e-300]],
        "weibull": [[2.0, 1.0], [1.5, 2.0], [0.3, 1e200], [0.1, 1e100], [50.0, 1.0], [1e3, 1.0], [1e-3, 1.0]],
        "lognormal": [[0.0, 1.0], [700.0, 0.1], [-700.0, 1.0], [0.0, 1e-10], [0.0, 30.0]],
        "gamma": [[0.0019089211433534496, 1.361085851280448e-88], [3.69e14, 4.11e-21], [1e15, 1e-84],
                  [1.0, 1.0], [2.0, 1.0], [0.5, 1.0], [7.9, 1.0], [8.0, 1.0], [100.0, 1.0], [1e-10, 1.0],
                  [1e5, 1e5], [2.0, 1e300], [2.0, 1e-300]],
        "beta": [[2.0, 3.0], [1.0, 1.0], [0.5, 0.5], [1.0, 1e6], [0.0018300780372745866, 0.01999572019743655],
                 [0.598598177922815, 760412215611.1234], [8.0, 8.0], [7.99, 8.01], [100.0, 100.0],
                 [1e-3, 5.0], [5.0, 1e-3], [0.01, 1.0], [1e5, 10.0]],
        "inversegaussian": [[1.0, 1.0], [1.5, 2.0], [1.0, 1e-3], [1.0, 1e3], [1e-100, 1.0], [1.0, 1e100]],
    }[ty]
    return R + E


# Supports whose end points are excluded from the x points: the density there
# is a convention (0, a limit or ∞), not a measurement.
OPEN_SUPPORT = {"chisquared": (0, math.inf), "fisherf": (0, math.inf), "exponential": (0, math.inf),
                "weibull": (0, math.inf), "lognormal": (0, math.inf), "gamma": (0, math.inf),
                "beta": (0, 1), "inversegaussian": (0, math.inf)}


# Quantile/isf arguments at a distribution's own branch thresholds in p:
# Cauchy's leading-term cot at p < 1e-9.
CONT_PBOUNDS = {"cauchy": near(1e-9)}


def cont_xbounds(ty, P):
    """Arguments at the distribution's own branch thresholds: those of its
    kernel (`gamma_xbounds`, `beta_xbounds`) mapped to x, plus its own."""
    xs = []
    if ty in ("gamma", "chisquared"):
        a, rate = (P[0], P[1]) if ty == "gamma" else (P[0] / 2, 0.5)
        xs = [y / rate for y in gamma_xbounds(a)]
        # cdf/sf take the log route where rate·x is subnormal.
        xs += near(TINY / rate)
    elif ty == "beta":
        xs = beta_xbounds(*P)
    elif ty == "fisherf":
        m, n = P
        # FISHERF_SMALL_ARG on y and on z, and the subnormal y and z of
        # ln_cdf/ln_sf's log routes. Divided in turn, as `m·small` underflows
        # at tiny dfn; a threshold past the f64 range is inf, which `near` drops.
        for small in (1e-8, TINY):
            xs += near(small * n / (m * (1 - small)))
            xs += near(n * (1 - small) / m / small)
    elif ty == "cauchy":
        loc, scale = P
        # |x − loc| = scale, the split into the tail forms.
        xs += near(loc - scale) + near(loc + scale)
    elif ty == "weibull":
        # x/scale leaving the normal range (the log route of (x/scale)^k).
        xs += near(TINY * P[1]) + near(F64_MAX * P[1])
    elif ty == "lognormal":
        # x·σ·√(2π) subnormal in the density.
        xs += near(TINY / (P[1] * math.sqrt(2 * math.pi)))
    elif ty == "inversegaussian":
        mu, lam = P
        # sf switch `h ≤ ¼·max(u, 1)`, h = √(2λ/x), u = a/√2: x = 9μ where
        # u ≥ 1, x = 32λ where u < 1; erfcx_drop's u = 1.5 (a = 1.5·√2, a
        # quadratic in √x).
        xs += near(9 * mu) + near(32 * lam)
        c = 1.5 * math.sqrt(2)
        sq = (c + math.sqrt(c * c + 4 * lam / mu)) / (2 * math.sqrt(lam) / mu)
        xs += near(sq * sq)
    hi = 1.0 if ty == "beta" else math.inf
    return [x for x in xs if math.isfinite(x) and 0 < x < hi]


def disc_params(ty, rng):
    R = []
    for _ in range(N_RANDOM):
        if ty == "bernoulli":
            u = rng.random()
            R.append([u if u < 0.4 else lu(rng, 1e-300, 0.5) if u < 0.7 else 1 - lu(rng, 1e-16, 0.5)])
        elif ty == "binomial":
            p = lu(rng, 1e-20, 0.5) if rng.random() < 0.6 else 1 - lu(rng, 1e-15, 0.5)
            R.append([int(lu(rng, 1, 1e15)), p])
        elif ty == "poisson":
            R.append([lu(rng, 1e-10, 1e15)])
        elif ty == "geometric":
            R.append([lu(rng, 1e-15, 1.0)])
        elif ty == "negbinomial":
            p = lu(rng, 1e-15, 0.5) if rng.random() < 0.5 else 1 - lu(rng, 1e-12, 0.5)
            R.append([lu(rng, 1e-10, 1e15), p])
        elif ty == "negbinomial_ms":
            R.append([lu(rng, 1e-3, 1e8), lu(rng, 1e-3, 1e15)])
        elif ty == "hypergeometric":
            nn = int(lu(rng, 2, 1e15))
            kk = rng.randint(0, nn) if nn < 10 ** 6 else int(nn * rng.random())
            if rng.random() < 0.5:
                n = int(min(nn, lu(rng, 1, 2e4)))
            else:
                n = rng.randint(0, nn) if nn < 10 ** 6 else int(nn * rng.random())
            R.append([nn, kk, n])
    E = {
        "bernoulli": [[0.0], [1.0], [0.5], [1e-300], [1 - 2.0 ** -53], [0.3]],
        "binomial": [[35, 0.12636357047271568], [111329, 0.015901893389398805], [10 ** 12, 1e-6],
                     [10 ** 15, 1e-20], [2 ** 53 + 1, 0.3], [2 ** 53 - 1, 0.5], [1, 0.5], [10, 0.3],
                     [1000, 1 - 1e-10]],
        "poisson": [[64409.7283739895], [1e-300], [10.0], [9.99], [1e6], [1e15], [3.0]],
        "geometric": [[1.0], [0.5], [0.25], [1e-10], [1e-300]],
        "negbinomial": [[1e15, 1e-20], [5.0, 0.4], [1e-10, 0.5], [1.0, 1 - 1e-9], [8.0, 0.3]],
        "negbinomial_ms": [[0.37, 2.5e13], [1.0, 1e10], [0.004423702810547923, 830930.3407766076], [3.0, 0.5]],
        "hypergeometric": [[10 ** 7, 5 * 10 ** 6, 10 ** 5], [3543, 2298, 2770], [20, 7, 12], [10 ** 15, 5 * 10 ** 14, 1000],
                           [100, 100, 50], [10, 0, 5], [10 ** 6, 10, 10 ** 5]],
    }[ty]
    return R + E


def disc_moments(ty, P):
    """(mean, sd, lo, hi) as floats (hi None when unbounded)."""
    if ty == "bernoulli":
        p = P[0]
        return p, math.sqrt(p * (1 - p)), 0, 1
    if ty == "binomial":
        n, p = P
        return n * p, math.sqrt(n * p * (1 - p)), 0, n
    if ty == "poisson":
        return P[0], math.sqrt(P[0]), 0, None
    if ty == "geometric":
        p = P[0]
        return 1 / p, math.sqrt(1 - p) / p, 1, (1 if p == 1 else None)
    if ty in ("negbinomial", "negbinomial_ms"):
        if ty == "negbinomial":
            r, p = P
            q = 1 - p
        else:
            mu, r = P
            p, q = mu / (mu + r), r / (mu + r)
        return r * p / q, math.sqrt(r * p) / q, 0, None
    nn, kk, n = P
    mean = n * kk / nn if nn else 0
    var = n * (kk / nn) * ((nn - kk) / nn) * ((nn - n) / max(nn - 1, 1)) if nn > 1 else 0
    return mean, math.sqrt(max(var, 0)), max(0, n + kk - nn), min(n, kk)


def hyper_summable(P):
    """Hypergeometric cdf/sf sum over the support (in the crate and here), so
    they and the quantile are swept only where it is short."""
    lo, hi = max(0, P[2] + P[1] - P[0]), min(P[2], P[1])
    return hi - lo <= 2 * 10 ** 5


GAMMA_A_EXTREMES = [5e-324, 1e-300, 1e-10, 1e-3, 0.1, 0.5, 0.99, 1.0, 1.5, 2.5, 5.0, 6.5, 7.9, 7.999999, 8.0,
                    8.000001, 20.0, 99.99, 100.0, 100.01, 1e3, 1e5, 243311.23772487845, 1e8, 1e12, 1e15, 2.0 ** 53, 1e20]
BETA_AB_EXTREMES = [[5.0, 1e-3], [1.0, 1e-3], [100.0, 1e-3], [1e-3, 5.0], [159137.46250199742, 149651.09756472873],
                    [1e9, 1e9], [1e9, 2e9], [2.0, 3.0], [1.0, 1.0], [0.5, 0.5], [8.0, 8.0], [7.99, 8.01],
                    [100.5, 100.5], [1e5, 10.0], [10.0, 1e5], [1e-3, 1e10], [1e10, 1e-3], [5e19, 5e3], [2.5, 3.5],
                    [0.5, 1e6], [1e6, 0.5], [1e15, 1e15]]


def special_params(fam, rng):
    if fam == "gamma":
        return [[lu(rng, 1e-6, 1e14)] for _ in range(70)] + [[a] for a in GAMMA_A_EXTREMES]
    if fam == "beta":
        R = []
        for _ in range(60):
            u = rng.random()
            if u < 0.4:
                R.append([lu(rng, 1e-4, 1e12), lu(rng, 1e-4, 1e12)])
            elif u < 0.7:
                R.append([lu(rng, 1e-2, 20), lu(rng, 1e-2, 20)])
            else:
                a = lu(rng, 10, 1e12)
                R.append([a, a * lu(rng, 0.3, 3)])
        return R + BETA_AB_EXTREMES
    raise ValueError(fam)


def gamma_xbounds(a):
    """The incomplete gamma kernel's thresholds in x (x < 1.1 for a < 1,
    x = a + 1, the Temme band, the pow switch |a·ln x| = POW_SWITCH, e^{−x}
    leaving the normal range at 708–745) and the subnormal/series edges the
    distributions test (1e-17, 2⁻¹⁰²², 1e-300, 1e-310)."""
    xs = []
    for t in [1.1, a + 1, 708.0, 709.0, 745.0, 1e-17, 2.0 ** -1022, 1e-300, 1e-310, a, 0.6 * a, 1.4 * a,
              math.exp(-POW_SWITCH / a) if a > 0.03 else 0.0, math.exp(POW_SWITCH / a) if a > 0.03 else 0.0]:
        if t > 0:
            xs += near(t)
    return [x for x in xs if math.isfinite(x) and x > 0]


def beta_xbounds(a, b):
    """The incomplete beta kernel's thresholds in x: the reflection point, the
    mean, brcomp's 0.375 on x and y, the basym band |λ| = 0.03·min(a, b), and
    for a shape <= 1 the branch tests of `small_shape_upper` on x0 = min(x, y)
    (0.1, 0.29, 0.3, (x0·b0)^a0 = 0.7, x0^a0 = 0.9) on both sides of 1/2."""
    xs = []
    for t in [(a + 1) / (a + b + 2), a / (a + b), 0.375, 0.625]:
        xs += near(t)
    m = min(a, b)
    if m <= 1:
        for a0, b0, flip in ((a, b, False), (b, a, True)):
            for t in [0.1, 0.29, 0.3, 0.7 ** (1 / a0) / b0, 0.9 ** (1 / a0)]:
                if 0 < t < 0.5:
                    xs += near(1 - t if flip else t)
    if m > 100:
        for lam in (0.03 * m, -0.03 * m):
            xs += near((a - lam) / (a + b))
    return [x for x in xs if 0 < x < 1]


# Inputs at which known defects were reported, measured as points of their own.
CITED_POINTS = [
    ("lgamma", [], 1.0), ("lbeta", [2.0], 3.0), ("inv_beta_reg", [0.01, 1.0], 0.5),
    ("inv_beta_reg", [0.01, 5.0], 1e-10), ("beta.quantile", [2.0, 3.0], 0.3),
    ("betai", [159137.46250199742, 149651.09756472873], 0.48228411628494705),
    ("gammq", [7.9], 727.5238762030003),
    ("binomial.cdf", [10 ** 12, 1e-6], 997000), ("binomial.cdf", [111329, 0.015901893389398805], 1645),
    ("binomial.cdf", [10 ** 15, 1e-20], 0),
    ("negbinomial_ms.cdf", [0.37, 2.5e13], 0), ("negbinomial_ms.cdf", [1.0, 1e10], 0),
    ("negbinomial_ms.sf", [0.004423702810547923, 830930.3407766076], 0), ("negbinomial.cdf", [1e15, 1e-20], 0),
    ("hypergeometric.sf", [10 ** 7, 5 * 10 ** 6, 10 ** 5], 55000),
    ("binomial.quantile", [35, 0.12636357047271568], 1.0), ("hypergeometric.quantile", [3543, 2298, 2770], 1.0),
    ("poisson.log_mass", [64409.7283739895], 82175),
    ("geometric.quantile", [1e-18], 0.999999), ("negbinomial.quantile", [7565433.3, 0.9999999999993723], 0.5),
    ("bernoulli.quantile", [1e-17], 1.0), ("hypergeometric.quantile", [52053, 11609, 43033], 1.0),
    ("hypergeometric.sf", [52053, 11609, 43033], 9880), ("hypergeometric.sf", [1000, 5, 100], 0),
    ("exponential.quantile", [1.0], 1e-10), ("exponential.quantile", [1.0], 1e-20),
    ("weibull.quantile", [2.0, 1.0], 1e-20),
    ("cauchy.cdf", [0.0, 1.0], -1e8), ("cauchy.cdf", [0.0, 1.0], -1e17), ("cauchy.sf", [0.0, 1.0], 1e17),
    ("cauchy.log_density", [0.0, 1.0], 1e155), ("cauchy.isf", [0.0, 1e-100], 1e-310),
    ("uniform.sf", [-383.03635179613127, -304.1942570061313], -304.19425700644064),
    ("beta.sf", [2.0, 3.0], 0.999), ("beta.sf", [2.0, 3.0], 0.99999999),
    # Mean within an ulp of ½ at huge shapes: beta_lambda's side choice at
    # x = ½ ± ulp (λ from the rounded one of x, 1 − x rounds to 0).
    ("beta.sf", [1e35, 1e35], 0.49999999999999994), ("beta.cdf", [1e35, 1e35], 0.49999999999999994),
    ("beta.sf", [1e35, 1e35], 0.5000000000000001), ("beta.cdf", [1e35, 1e35], 0.5000000000000001),
    ("beta.isf", [1.0, 1e6], 0.5), ("beta.isf", [0.0018300780372745866, 0.01999572019743655], 0.19541583110766808),
    ("chisquared.log_density", [486622.4754497569], 487311.41223330377),
    ("chisquared.log_density", [1e300], 1e300), ("gamma.log_density", [3.69e14, 4.11e-21], 8.98e34),
    ("gamma.quantile", [1e15, 1e-84], 0.5),
    ("gamma.cdf", [0.0019089211433534496, 1.361085851280448e-88], 1.9384513246510335e-279),
    ("chisquared.cdf", [0.0027530089082596562], 1.53e-322),
    ("fisherf.quantile", [0.0015205765296369873, 2057.918596622136], 0.56),
    ("fisherf.isf", [0.0015205765296369873, 2057.918596622136], 0.44),
    ("fisherf.cdf", [534536453686.36383, 6.913288301642686], 0.00474263815705448),
    ("beta.log_density", [0.598598177922815, 760412215611.1234], 2.3840156456241725e-13),
]


class Gen:
    """Collects points; places arguments by probability through the probe."""

    def __init__(self, probe, want):
        self.probe = probe
        self.want = want
        self.points = []
        self.seen = set()

    def add(self, fn, params, x, pl=None, gid=None):
        if fn not in self.want:
            return
        key = (fn, tuple(params), x)
        if key in self.seen:
            return
        self.seen.add(key)
        self.points.append({"fn": fn, "params": list(params), "x": x, "pl": pl, "gid": gid})

    def place(self, fn_q, fn_isf, params, levels):
        """(x, tail mass) from quantile(p) and isf(p) at each level."""
        reqs = []
        for p in levels:
            reqs.append((fn_q, params, fstr(p)))
            if fn_isf:
                reqs.append((fn_isf, params, fstr(p)))
        out = self.probe(reqs)
        res = []
        for (f, _, p), v in zip(reqs, out):
            if v is None:
                continue
            try:
                xv = float(v)
            except ValueError:
                continue
            if math.isfinite(xv):
                res.append((xv, float(p)))
        return res


def generate(probe, want):
    g = Gen(probe, want)
    gid = 0

    def wants_any(names):
        return any(n in want for n in names)

    # Incomplete gamma family.
    gam = ("gammp", "gammq", "ln_gammp", "ln_gammq")
    if wants_any(gam):
        rng = random.Random(SEED ^ zlib.crc32(b"gamma-family"))
        for (a,) in special_params("gamma", rng):
            gid += 1
            ps = [fstr(a)]
            xs = [(x, pl) for x, pl in g.place("gamma.quantile", "gamma.isf", [fstr(a), "1.0"], P_LEVELS)]
            xs += [(x, None) for x in gamma_xbounds(a)]
            xs += [(lu(rng, 1e-10, 1e3) * max(a, 1e-3), None) for _ in range(3)]
            if 0.05 < a < 1:
                xs += [(x, None) for x in (1.1, 1.3, 1.5, 1.7, 1.9) if x < a + 1]
            for x, pl in xs:
                if x > 0 and math.isfinite(x):
                    for fn in gam:
                        g.add(fn, ps, fstr(x), pl, gid)
        # Q = 1 − P for a < 1 and 1.1 ≤ x < a + 1: a dense grid of that region.
        for a in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999):
            gid += 1
            for i in range(12):
                x = 1.1 + (a - 0.1) * i / 11
                for fn in gam:
                    g.add(fn, [fstr(a)], fstr(x), None, gid)

    # Incomplete beta family and inv_beta_reg.
    bet = ("betai", "ln_betai")
    if wants_any(bet + ("inv_beta_reg",)):
        rng = random.Random(SEED ^ zlib.crc32(b"beta-family"))
        for a, b in special_params("beta", rng):
            gid += 1
            ps = [fstr(a), fstr(b)]
            if wants_any(bet):
                xs = g.place("beta.quantile", "beta.isf", ps, P_LEVELS)
                xs += [(x, None) for x in beta_xbounds(a, b)]
                for x, pl in xs:
                    if 0 < x < 1:
                        for fn in bet:
                            g.add(fn, ps, fstr(x), pl, gid)
            for p in P_LEVELS + SUBNORMAL_P + [1 - q for q in UPPER_Q]:
                g.add("inv_beta_reg", ps, fstr(p), min(p, 1 - p), gid)

    # lgamma, lbeta.
    if "lgamma" in want:
        rng = random.Random(SEED ^ zlib.crc32(b"lgamma"))
        xs = [lu(rng, 1e-300, 1e300) for _ in range(100)] + [lu(rng, 0.1, 30) for _ in range(100)]
        xs += [1 + s * 2.0 ** -k for k in (1, 5, 10, 20, 30, 40, 52) for s in (1, -1)]
        xs += [2 + s * 2.0 ** -k for k in (1, 5, 10, 20, 30, 40, 51) for s in (1, -1)]
        xs += [float(i) for i in range(1, 31)] + [i + 0.5 for i in range(0, 30)]
        xs += near(0.5) + near(8.0) + [5e-324, 1e-300, 1e-10]
        # lgamma's branch switches, and ln_gamma_1p's at 0.7 (x = 0.7, 1.7).
        for t in (0.7, 0.8, 1.7, 2.25, 3.25, 10.0):
            xs += near(t)
        for x in xs:
            g.add("lgamma", [], fstr(x), None, None)
    if "lbeta" in want:
        rng = random.Random(SEED ^ zlib.crc32(b"lbeta"))
        pairs = [[lu(rng, 1e-4, 1e15), lu(rng, 1e-4, 1e15)] for _ in range(80)]
        pairs += [[lu(rng, 0.1, 20), lu(rng, 0.1, 20)] for _ in range(60)]
        pairs += [[2.0, 3.0], [8.0, 8.0], [7.99, 8.0], [7.99, 1e5], [1e-300, 1.0], [1e15, 1e15], [0.5, 0.5],
                  [1.0, 1.0], [1e-3, 1e10]]
        for a, b in pairs:
            g.add("lbeta", [fstr(a)], fstr(b), None, None)

    # Continuous distributions.
    for ty in CONT:
        names = [f"{ty}.{m}" for m in list(CONT_METHODS) + list(LN_VARIANTS.get(ty, ()))]
        if not wants_any(names):
            continue
        rng = random.Random(SEED ^ zlib.crc32(ty.encode()))
        for P in cont_params(ty, rng):
            gid += 1
            ps = [fstr(v) for v in P]
            xs = g.place(f"{ty}.quantile", f"{ty}.isf", ps, P_LEVELS)
            xs += [(x, None) for x in cont_xbounds(ty, P)]
            lo, hi = OPEN_SUPPORT.get(ty, (-math.inf, math.inf))
            for x, pl in xs:
                if not lo < x < hi:
                    continue
                for m in ("cdf", "sf", "pdf", "log_density") + LN_VARIANTS.get(ty, ()):
                    g.add(f"{ty}.{m}", ps, fstr(x), pl, gid)
            ps_levels = P_LEVELS + SUBNORMAL_P + [1 - q for q in UPPER_Q] + [rng.random(), rng.random()]
            ps_levels += CONT_PBOUNDS.get(ty, [])
            for p in ps_levels:
                for m in ("quantile", "isf"):
                    g.add(f"{ty}.{m}", ps, fstr(p), min(p, 1 - p), gid)

    # Discrete distributions.
    for ty in DISC:
        names = [f"{ty}.{m}" for m in DISC_METHODS]
        if not wants_any(names):
            continue
        rng = random.Random(SEED ^ zlib.crc32(ty.encode()))
        for P in disc_params(ty, rng):
            gid += 1
            ps = [str(v) if isinstance(v, int) else fstr(v) for v in P]
            mean, sd, lo, hi = disc_moments(ty, P)
            summable = ty != "hypergeometric" or hyper_summable(P)
            levels = P_LEVELS + [1 - q for q in UPPER_Q]
            ks = []
            if summable:
                for k, p in g.place(f"{ty}.quantile", None, ps, levels):
                    ks += [(int(k), min(p, 1 - p)), (int(k) - 1, None), (int(k) + 1, None)]
            for j in (10, 20, 40, 100):
                k = int(mean + j * sd)
                if hi is None or k < hi:
                    ks.append((k, None))
            ks += [(lo, None), (lo + 1, None)]
            if hi is not None:
                ks += [(hi, None), (hi - 1, None), (hi - 2, None)]
            if ty == "hypergeometric":
                # cdf/sf switch to the other tail's sum at the support midpoint.
                ks += [((lo + hi) // 2 + d, None) for d in (-1, 0, 1)]
            ks.append((int(mean), None))
            for k, pl in ks:
                if k < lo or (hi is not None and k > hi) or k > 2 ** 63 - 1:
                    continue
                for m in ("mass", "log_mass") + (("cdf", "sf") if summable else ()):
                    g.add(f"{ty}.{m}", ps, str(k), pl, gid)
            if summable:
                qs = levels + SUBNORMAL_P + [rng.random(), rng.random()] + ([1.0] if hi is not None else [])
                for p in qs:
                    g.add(f"{ty}.quantile", ps, fstr(p), min(p, 1 - p), gid)
    for fn, P, x in CITED_POINTS:
        ps = [str(v) if isinstance(v, int) else fstr(v) for v in P]
        g.add(fn, ps, str(x) if isinstance(x, int) else fstr(x), min(x, 1 - x) if FN.get(fn, {}).get("kind") in
              ("quantile", "dquantile") else None, None)
    return g.points


# ============================================================================
# Probe, truth cache, scoring
# ============================================================================

def build_probe():
    r = subprocess.run(["cargo", "build", "--release", "--example", "accuracy_probe", "--features", "dist"],
                       cwd=CRATE, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("probe build failed:\n" + r.stderr)


def run_probe(reqs):
    """reqs: [(fn, params, x)] → one output string per request (None for an
    `err:` line)."""
    if not reqs:
        return []
    lines = "".join(json.dumps({"fn": f, "params": list(p), "x": x}) + "\n" for f, p, x in reqs)
    r = subprocess.run([PROBE], input=lines, capture_output=True, text=True, timeout=3600)
    out = r.stdout.split("\n")[:-1]
    if r.returncode != 0 or len(out) != len(reqs):
        sys.exit(f"probe failed ({r.returncode}, {len(out)}/{len(reqs)} lines): {r.stderr[:500]}")
    return [None if o.startswith("err:") else o for o in out]


# Version of the K computation; cached entries with another version (or none)
# have their K recomputed (a continuous quantile keeps its cached root).
K_VERSION = 2
# Version of the truth code for timed-out points: one that timed out under an
# earlier version is retried.
TIMEOUT_VERSION = 2


def stale(e):
    """An entry to recompute: its K is of another version, or it timed out
    under another version of the truth code."""
    if e is None:
        return False
    if e.get("fail") == "timeout":
        return e.get("tv") != TIMEOUT_VERSION
    return "K" in e and e.get("kv") != K_VERSION


def load_cache(names):
    cache = {}
    for n in names:
        path = os.path.join(CACHE, n + ".jsonl")
        d = {}
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        e = json.loads(line)
                        d[e.pop("k")] = e
        cache[n] = d
    return cache


def compute_truth(jobs, cache, nproc):
    """Runs the missing truth jobs in a pool, appending each result to the
    cache file of its function as it arrives (so an interrupted run resumes)."""
    todo = []
    seen = set()
    for j in jobs:
        key = cache_key(j[1], j[2])
        e = cache[j[0]].get(key)
        if (e is not None and not stale(e)) or (j[0], key) in seen:
            continue
        seen.add((j[0], key))
        prior = e["t"] if e is not None and "t" in e and FN[j[0]]["kind"] == "quantile" else None
        todo.append((*j, prior))
    if not todo:
        return
    os.makedirs(CACHE, exist_ok=True)
    # Slow jobs (large parameters) first, so the pool does not end on them.
    todo.sort(key=lambda j: -max([abs(float(s)) for s in j[1]] + [1.0]))
    print(f"truth: {len(todo)} new points on {nproc} processes", flush=True)
    files = {}
    t0 = last = time.time()
    done = 0
    ctx = mproc.get_context("fork")
    # Workers are recycled: mpmath caches quadrature nodes per precision, and
    # the precision varies with the parameters.
    with ctx.Pool(nproc, maxtasksperchild=400) as pool:
        try:
            for name, key, entry in pool.imap_unordered(truth_job, todo, chunksize=2):
                cache[name][key] = entry
                fh = files.get(name)
                if fh is None:
                    fh = files[name] = open(os.path.join(CACHE, name + ".jsonl"), "a")
                fh.write(json.dumps({"k": key, **entry}) + "\n")
                done += 1
                if time.time() - last > 30:
                    for f in files.values():
                        f.flush()
                    last = time.time()
                    rate = done / (last - t0)
                    print(f"  {done}/{len(todo)}  {rate:.1f}/s  eta {(len(todo) - done) / max(rate, 1e-9) / 60:.1f} min",
                          flush=True)
        finally:
            for f in files.values():
                f.close()
    print(f"truth: done in {(time.time() - t0) / 60:.1f} min", flush=True)


def as_num(s):
    """A float input string as a float, a digit string as an int."""
    return int(s) if s.lstrip("-").isdigit() else float(s)


def classify(r, hard):
    if hard or r is None or r > BUG_R:
        return "bug"
    return "limit" if r <= LIMIT_R else "review"


IG_CLAMPED = ("inversegaussian.ln_cdf", "inversegaussian.ln_sf")
LN_HALF_MIN_SUBNORMAL = -1075 * math.log(2)


def past_i64(tr):
    """A discrete quantile's truth beyond i64::MAX, where the crate returns an
    error by contract: a root above 2⁶³ − 1, or no upper bracket (the search
    gives up past 2⁷⁰)."""
    if "fail" in tr:
        return "no upper bracket" in tr["fail"]
    return int(tr["t"]) > 2 ** 63 - 1


def score_point(pt, obs, tr, cache):
    """Adds r, abs error, hard violation (if any), truth and K to the point.
    The hard checks that need no truth (an error or NaN for a valid input, a
    probability outside [0, 1], a quantile outside the support) run also where
    the truth failed."""
    fn = pt["fn"]
    meta = FN[fn]
    kind = meta["kind"]
    pt["obs"] = obs
    pt["hard"] = None
    P = [as_num(s) for s in pt["params"]]
    x = as_num(pt["x"])
    O = None
    beyond_i64 = kind == "dquantile" and past_i64(tr)
    if obs is None:
        if not beyond_i64:
            pt["hard"] = "error for a valid input"
    elif kind == "dquantile":
        lo, hi = disc_moments(meta["dist"], P)[2:]
        if int(obs) < lo or (hi is not None and int(obs) > hi):
            pt["hard"] = f"quantile outside the support [{lo}, {hi}]"
    else:
        O = float(obs)
        if math.isnan(O):
            pt["hard"] = "NaN for a valid input"
        elif kind == "prob" and not (0.0 <= O <= 1.0):
            pt["hard"] = "probability outside [0, 1]"
        elif kind == "quantile":
            lo, hi = CONT[meta["dist"] or "beta"].support([mpf(v) for v in P])
            if not (lo <= O <= hi):
                pt["hard"] = "quantile outside the support"
    if "fail" in tr:
        pt["r"] = None
        pt["fail"] = tr["fail"]
        return
    if kind == "dquantile":
        k_star = int(tr["t"])
        pt["truth"] = str(k_star)
        pt["K"] = "0"
        p = x
        if obs is None:
            pt["r"] = 0.0 if beyond_i64 else math.inf
            return
        k = int(obs)
        pt["abs"] = abs(k - k_star)
        lo, hi = disc_moments(meta["dist"], P)[2:]
        if p == 1.0 and k != k_star:
            pt["hard"] = pt["hard"] or f"quantile(1) = {k}, largest k with positive mass {k_star}"
        pt["r"] = 0.0 if k == k_star else disc_gap_r(pt, tr, k, k_star, p, cache)
        pt["alt_r"] = {str(j): disc_gap_r(pt, tr, j, k_star, p, cache) for j in (k_star - 1, k_star + 1)
                       if j >= lo and (hi is None or j <= hi) and tr.get("before" if j < k_star else "at")}
        return
    t = tr["t"]
    T = mpf(t) if t not in ("inf", "-inf", "nan") else {"inf": mp.inf, "-inf": -mp.inf, "nan": mp.nan}[t]
    K = mpf(tr["K"])
    pt["truth"], pt["K"] = t, tr["K"]
    if mpmath.isnan(T):
        pt["r"] = None
        pt["fail"] = "truth undefined"
        return
    if O is None or math.isnan(O):
        pt["r"] = math.inf
        return
    if mpmath.isfinite(T):
        pt["kappa"] = float(K / abs(T)) if T != 0 else (math.inf if K > 0 else 0.0)
    if fn in IG_CLAMPED and T < LN_HALF_MIN_SUBNORMAL and O == -math.inf:
        # Documented: below half the smallest subnormal the mass reads as 0.
        pt["r"], pt["abs"] = 0.0, 0.0
        return
    if mpmath.isinf(T) or abs(T) > F64_MAX:
        same = (math.isinf(O) or abs(O) == F64_MAX) and (O > 0) == (T > 0)
        pt["r"] = 0.0 if same else math.inf
        pt["abs"] = 0.0 if same else math.inf
        return
    err = abs(mpf(O) - T)
    if err <= abs(T) * mpf(10) ** -28:
        # Below the resolution of a truth stored to 30 digits.
        err = mpf(0)
    den = EPS * max(K, abs(T), mpf(TINY))
    pt["r"] = float(err / den)
    pt["abs"] = float(err)


def disc_gap_r(pt, tr, k, k_star, p, cache):
    """r of a discrete quantile k ≠ k*: the probability gap at the misjudged
    point (the cdf, or above p = ½ the sf, at k if k < k*, else at k − 1),
    over ε·max(p, K) with K that tail's sensitivity to the real parameters
    there, the error a backward-stable cdf is allowed. The tail value comes
    from the solve (k* − 1, k*) or the cdf/sf truth; K only from the latter
    (0 where it is not computed). None where the value is missing."""
    use_sf = tr["sf"]
    q = 1 - mpf(p)
    at = k if k < k_star else k - 1
    side = "sf" if use_sf else "cdf"
    e = cache.get(f"{FN[pt['fn']]['dist']}.{side}", {}).get(cache_key(pt["params"], str(at)))
    e = e if e is not None and "fail" not in e else None
    if at == k_star - 1 and tr.get("before"):
        v = mpf(tr["before"])
    elif at == k_star and tr.get("at"):
        v = mpf(tr["at"])
    elif e is not None:
        v = mpf(e["t"])
    else:
        return None
    K = mpf(e["K"]) if e is not None else mpf(0)
    if k < k_star:
        gap = v - q if use_sf else mpf(p) - v
    else:
        gap = q - v if use_sf else v - mpf(p)
    return float(max(gap, mpf(0)) / (EPS * max(mpf(p), K, mpf(TINY))))


def aux_jobs(points, cache):
    """cdf/sf truth (value and K) at the point a wrong discrete quantile
    misjudged and at k* − 1, k*, for the gaps of the neighbours."""
    jobs = []
    for pt in points:
        if FN[pt["fn"]]["kind"] != "dquantile" or pt["obs"] is None:
            continue
        tr = cache[pt["fn"]].get(cache_key(pt["params"], pt["x"]))
        if not tr or "fail" in tr:
            continue
        k, k_star = int(pt["obs"]), int(tr["t"])
        side = "sf" if tr["sf"] else "cdf"
        # The misjudged point of the returned k, and of the neighbours k* ± 1
        # (their gaps decide the fixture's accepted answers).
        ats = {k_star - 1, k_star} | ({k if k < k_star else k - 1} if k != k_star else set())
        for at in ats:
            if at >= 0:
                jobs.append((f"{FN[pt['fn']]['dist']}.{side}", tuple(pt["params"]), str(at), None))
    return jobs


def monotone_checks(points):
    """cdf non-decreasing and sf non-increasing in x within one parameter set
    (gammp/betai likewise, gammq decreasing). A step against the direction is
    a hard violation when it exceeds LIMIT_R·max(κ, 1)·ε relative to the
    values (κ the larger of the two points'), i.e. more than the limit class
    allows each value; a smaller one is a seam step, reported apart."""
    groups = {}
    for pt in points:
        fn = pt["fn"]
        if pt.get("gid") is None or pt.get("obs") is None:
            continue
        if fn.endswith(".cdf") or fn.endswith(".sf") or fn in ("gammp", "gammq", "betai"):
            groups.setdefault((fn, tuple(pt["params"])), []).append(pt)
    for (fn, _), pts in groups.items():
        rising = not (fn.endswith(".sf") or fn == "gammq")
        pts = sorted(pts, key=lambda p: as_num(p["x"]))
        for a, b in zip(pts, pts[1:]):
            va, vb = float(a["obs"]), float(b["obs"])
            if as_num(a["x"]) == as_num(b["x"]) or math.isnan(va) or math.isnan(vb):
                continue
            if (vb < va) if rising else (vb > va):
                step = abs(vb - va) / max(abs(va), abs(vb), TINY) / EPS
                allowed = LIMIT_R * max(a.get("kappa", 1.0), b.get("kappa", 1.0), 1.0)
                msg = (f"{'cdf decreases' if rising else 'sf increases'} by {step:.3g} eps (allowed "
                       f"{allowed:.3g}) from x={a['x']} ({va!r}) to x={b['x']} ({vb!r})")
                if step > allowed:
                    b["hard"] = b["hard"] or msg
                else:
                    b["seam"] = msg


# ============================================================================
# Summary, report, baseline, tables, fixtures
# ============================================================================

# Known open defects by the input region and mechanism of the points they
# explain: (title, predicate on a point), first match wins. Titles are stable
# short names of the defects, not source lines. A bug-class point outside
# every region is `untracked`, so it is not skipped silently. Exported
# bug-class points carry the title as `skip`.

def _pf(pt, i):
    return float(pt["params"][i])


def _x(pt):
    return float(pt["x"])


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _method(pt):
    return pt["fn"].split(".")[-1]


def _kernel_ax(pt):
    """(shape, argument) of the incomplete gamma P/Q behind a point, or None."""
    fn, ty, m = pt["fn"], FN[pt["fn"]]["dist"], _method(pt)
    if ty is None and fn in ("gammp", "gammq", "ln_gammp", "ln_gammq"):
        return _pf(pt, 0), _x(pt)
    if ty == "gamma" and m in ("cdf", "sf", "ln_cdf"):
        return _pf(pt, 0), _pf(pt, 1) * _x(pt)
    if ty == "chisquared" and m in ("cdf", "sf"):
        return _pf(pt, 0) / 2, _x(pt) / 2
    if ty == "poisson" and m in ("cdf", "sf"):
        return _x(pt) + 1, _pf(pt, 0)
    return None


def _beta_args(pt):
    """(a, b, x) of the incomplete beta behind a point, or None."""
    fn, ty, m = pt["fn"], FN[pt["fn"]]["dist"], _method(pt)
    if fn in ("betai", "ln_betai"):
        return _pf(pt, 0), _pf(pt, 1), _x(pt)
    if ty == "beta" and m == "cdf":
        return _pf(pt, 0), _pf(pt, 1), _x(pt)
    if ty == "fisherf" and m in ("cdf", "sf", "ln_cdf", "ln_sf"):
        dfn, dfd, x = _pf(pt, 0), _pf(pt, 1), _x(pt)
        return dfn / 2, dfd / 2, dfn * x / (dfd + dfn * x)
    return None


def _known_seam(pt):
    """A beyond-allowance monotone step at a known kernel seam: incomplete
    gamma at x = 1.1 (small-shape series) or x = a + 1 (series / continued
    fraction), incomplete beta at its reflection point (a+1)/(a+b+2), each
    within the ±1 % the sweep samples around it."""
    if not (pt.get("hard") and " eps (allowed " in pt["hard"]):
        return False
    ax = _kernel_ax(pt)
    if ax is not None:
        a, x = ax
        return any(abs(x - t) <= 0.015 * t for t in (1.1, a + 1))
    ab = _beta_args(pt)
    if ab is not None:
        a, b, x = ab
        t = (a + 1) / (a + b + 2)
        return abs(x - t) <= 0.015 * t
    return False


def _bgrat_underflow(pt):
    """FisherF's small tail from the incomplete-beta kernel at a normal
    argument with a tiny shape: sf/ln_sf at y with shape dfn/2, and the
    mirror cdf/ln_cdf at z = 1 − y with shape dfd/2. The shape-≤-1 route's
    `bgrat` gets z' = −ν·ln(1 − arg), ν = other/2 + 20 + (shape − 1)/2 after
    `bup`, and returns None where `shape·z'` underflows to 0; the kernel then
    gives that tail as 1 − I. A monotone step counts when either of its two
    points is such a point."""
    m = _method(pt)
    if FN[pt["fn"]]["dist"] != "fisherf" or m not in ("sf", "ln_sf", "cdf", "ln_cdf"):
        return False

    def at(x):
        dfn, dfd, x = _pf(pt, 0), _pf(pt, 1), float(x)
        if m in ("sf", "ln_sf"):
            a, b, t = dfn / 2, dfd / 2, dfn * x / (dfd + dfn * x)
        else:
            a, b, t = dfd / 2, dfn / 2, dfd / (dfd + dfn * x)
        nu = b + 20 + (a - 1) / 2
        return TINY <= t < 1e-8 and a * (-nu * math.log1p(-t)) == 0.0
    hard = pt.get("hard") or ""
    prev = hard.split(" from x=")[1].split(" ")[0] if " from x=" in hard else None
    return at(pt["x"]) or (prev is not None and at(prev))


def _fisherf_root_out_of_range(pt):
    """FisherF isf (quantile) whose root lies below (above) the f64 range:
    truth 0 (inf), the tail mass at every representable x is under the target,
    and a finite positive x is returned instead."""
    if pt["fn"] not in ("fisherf.isf", "fisherf.quantile"):
        return False
    want = 0.0 if pt["fn"] == "fisherf.isf" else math.inf
    obs = _num(pt.get("obs"))
    return _num(pt.get("truth")) == want and obs is not None and 0 < obs < math.inf


def _fisherf_density_small_df_logs(pt):
    """FisherF pdf/log_density on the both-halves-below-8 branch, which sums
    `a·ln(dfn/dfd) + (a − 1)·ln x − (a + b)·ln(1 + u) − ln B(a, b)` directly:
    where more than 2000 of the terms' total size C cancels (a tiny dfd makes
    `a·ln(dfn/dfd)` ~5e3 against `(a − 1)·ln x`), their rounding survives:
    ~C·ε/2 absolute in the log, so r up to ~C/2, the bug class from C = 2000."""
    if pt["fn"] not in ("fisherf.pdf", "fisherf.log_density"):
        return False
    dfn, dfd, x = _pf(pt, 0), _pf(pt, 1), _x(pt)
    a, b = dfn / 2, dfd / 2
    if not (a < 8 and b < 8 and x > 0):
        return False
    u = dfn / dfd * x
    ln1pu = math.log1p(u) if math.isfinite(u) else math.log(dfn) - math.log(dfd) + math.log(x)
    terms = [a * (math.log(dfn) - math.log(dfd)), (a - 1) * math.log(x), -(a + b) * ln1pu,
             math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)]
    return sum(abs(t) for t in terms) - abs(sum(terms)) > 2000


TRACKER = [
    ("non-monotone step beyond the limit at a kernel branch seam", _known_seam),
    ("incomplete-beta bgrat returns None where tiny shape·z underflows; FisherF small tail becomes 1 − I",
     _bgrat_underflow),
    ("FisherF isf/quantile returns a finite x where the root is outside the f64 range",
     _fisherf_root_out_of_range),
    ("FisherF log_density small-df branch: over 2000 of its terms' size cancels", _fisherf_density_small_df_logs),
]


def tracker_label(pt):
    for label, pred in TRACKER:
        if pred(pt):
            return label
    return "untracked"


def is_tail(pt):
    kind = FN[pt["fn"]]["kind"]
    if kind in ("quantile", "dquantile") or pt.get("pl") is not None:
        return pt.get("pl") is not None and pt["pl"] < TAIL_P
    t = pt.get("truth")
    if t is None or t in ("inf", "-inf", "nan"):
        return False
    v = float(t)
    if kind == "prob":
        return min(v, 1 - v) < TAIL_P
    if pt["fn"].split(".")[-1].startswith("ln_"):
        return v < math.log(TAIL_P) or v > math.log1p(-TAIL_P)
    return False


def rfmt(r):
    if r is None:
        return "-"
    if math.isinf(r):
        return "inf"
    return f"{r:.2g}" if r < 99.5 else f"{r:.1e}".replace("e+0", "e").replace("e+", "e")


def jnum(v):
    return "inf" if isinstance(v, float) and math.isinf(v) else v


def summarize(points, want):
    by = {}
    for pt in points:
        by.setdefault(pt["fn"], []).append(pt)
    S = {}
    for fn in sorted(want):
        pts = by.get(fn, [])
        if not pts:
            continue
        scored = [p for p in pts if p.get("r") is not None]
        hard = [p for p in pts if p.get("hard")]
        fail = [p for p in pts if p.get("fail")]
        rs = sorted(p["r"] for p in scored)

        def q(f):
            return rs[min(len(rs) - 1, int(f * len(rs)))] if rs else None
        worst = max(scored, key=lambda p: p["r"]) if scored else None
        worst_r = worst["r"] if worst else None
        regimes = {}
        for p in scored:
            reg = regimes.setdefault(p["regime"], {"n": 0, "worst_r": 0.0, "bulk_r": None, "tail_r": None,
                                                    "hard": 0})
            reg["n"] += 1
            reg["worst_r"] = max(reg["worst_r"], p["r"])
            k = "tail_r" if is_tail(p) else "bulk_r"
            reg[k] = p["r"] if reg[k] is None else max(reg[k], p["r"])
            reg["hard"] += 1 if p.get("hard") else 0
        for reg in regimes.values():
            reg["class"] = classify(reg["worst_r"], reg["hard"] > 0)
        S[fn] = {
            "n": len(pts), "scored": len(scored), "fail": len(fail), "hard": len(hard),
            "worst_r": worst_r, "p50": q(0.5), "p90": q(0.9), "p99": q(0.99),
            "class": classify(worst_r, bool(hard)) if scored else "no truth",
            "worst": {k: worst.get(k) for k in ("params", "x", "obs", "truth", "K", "regime")} if worst else None,
            "regimes": regimes,
        }
    return S


def write_report(points, S, path):
    L = []
    npts = sum(s["n"] for s in S.values())
    L.append(f"CommonStats accuracy sweep: {npts} points, {len(S)} functions. "
             f"r = |obs − truth| / (ε·max(K, |truth|, 2^-1022)), ε = 2^-52. "
             f"Classes: r ≤ {LIMIT_R:g} limit, ≤ {BUG_R:g} review, above bug.")
    L.append("")
    L.append(f"{'function':30s} {'n':>6s} {'fail':>5s} {'hard':>5s} {'p50':>8s} {'p90':>8s} {'p99':>8s} "
             f"{'max':>8s}  class")
    order = sorted(S, key=lambda f: -(S[f]["worst_r"] if S[f]["worst_r"] is not None else -1))
    for fn in order:
        s = S[fn]
        L.append(f"{fn:30s} {s['n']:6d} {s['fail']:5d} {s['hard']:5d} {rfmt(s['p50']):>8s} {rfmt(s['p90']):>8s} "
                 f"{rfmt(s['p99']):>8s} {rfmt(s['worst_r']):>8s}  {s['class']}")
    L.append("")
    L.append("Distribution of r per function: counts with log10(r) in (−inf,0] (0,1] (1,2] (2,3] (3,6] (6,∞]")
    edges = [1, 10, 100, 1e3, 1e6]
    by = {}
    for pt in points:
        if pt.get("r") is not None:
            by.setdefault(pt["fn"], []).append(pt["r"])
    for fn in order:
        c = [0] * 6
        for r in by.get(fn, []):
            i = 0
            while i < 5 and r > edges[i]:
                i += 1
            c[i] += 1
        L.append(f"  {fn:30s} " + " ".join(f"{v:6d}" for v in c))
    L.append("")
    L.append("Largest absolute error per function among points with κ > 1/ε (where r hides errors):")
    for fn in order:
        big = [p for p in points if p["fn"] == fn and p.get("kappa", 0) > 1 / EPS and p.get("abs") is not None
               and math.isfinite(p["abs"])]
        if big:
            p = max(big, key=lambda p: p["abs"])
            L.append(f"  {fn:30s} n={len(big):5d}  abs {p['abs']:.3g}  κ {p['kappa']:.3g}  r {rfmt(p['r'])}  "
                     f"params={p['params']} x={p['x']} obs={p['obs']} truth={p.get('truth')}")
    L.append("")
    seams = [p for p in points if p.get("seam")]
    L.append(f"Seam steps: {len(seams)} steps against the monotone direction within LIMIT_R·max(κ, 1)·ε "
             f"(not hard violations):")
    for p in sorted(seams, key=lambda p: (p["fn"], p["params"], as_num(p["x"]))):
        L.append(f"  {p['fn']:22s} params={p['params']}  {p['seam']}")
    L.append("")
    L.append("Per function and regime: worst r (bulk / tail), then the 5 worst points.")
    for fn in order:
        s = S[fn]
        L.append("")
        L.append(f"== {fn}  worst r {rfmt(s['worst_r'])}  [{s['class']}]")
        for reg, v in sorted(s["regimes"].items()):
            L.append(f"   {reg:45s} n={v['n']:5d}  worst {rfmt(v['worst_r']):>8s}  bulk {rfmt(v['bulk_r']):>8s}  "
                     f"tail {rfmt(v['tail_r']):>8s}  {v['class']}")
        pts = sorted((p for p in points if p["fn"] == fn and p.get("r") is not None), key=lambda p: -p["r"])[:5]
        for p in pts:
            L.append(f"   r={rfmt(p['r']):>8s} abs={p.get('abs', 0):.3g}  params={p['params']} x={p['x']} "
                     f"obs={p['obs']} truth={p.get('truth')} K={float(p.get('K', 0)):.3g}  [{p['regime']}, "
                     f"{'tail' if is_tail(p) else 'bulk'}]" + (f"  HARD: {p['hard']}" if p.get("hard") else ""))
        hard = [p for p in points if p["fn"] == fn and p.get("hard")]
        for p in hard[:10]:
            L.append(f"   hard: {p['hard']}  params={p['params']} x={p['x']} obs={p['obs']} truth={p.get('truth')}")
        if len(hard) > 10:
            L.append(f"   ... {len(hard) - 10} more hard violations")
        fails = [p for p in points if p["fn"] == fn and p.get("fail")]
        for p in fails[:3]:
            L.append(f"   truth failed: {p['fail']}  params={p['params']} x={p['x']}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return L


def load_baseline():
    if os.path.exists(BASELINE):
        with open(BASELINE) as fh:
            return json.load(fh)
    return {"meta": {}, "functions": {}}


def rnum(v):
    return math.inf if v == "inf" else v


def check(S, base):
    """Regressions against the baseline, per (function, regime): a worse
    class, or a worst r above 1 that grew by more than 2x, or a baseline
    regime missing from this run; per function, a worse class (including "no
    truth") or more hard violations. A regime new to this run is printed, not
    failed."""
    rank = {"limit": 0, "review": 1, "bug": 2, "no truth": 3}
    bad = []
    for fn, s in S.items():
        b = base["functions"].get(fn)
        if b is None:
            print(f"check: {fn} not in the baseline")
            continue
        if rank[s["class"]] > rank[b["class"]]:
            bad.append(f"{fn}: class {b['class']} -> {s['class']}")
        for reg in b["regimes"]:
            if reg not in s["regimes"]:
                bad.append(f"{fn} [{reg}]: regime in the baseline but not in this run")
        for reg, cur in s["regimes"].items():
            old = b["regimes"].get(reg)
            if old is None:
                print(f"check: {fn} [{reg}] not in the baseline")
                continue
            br, sr = rnum(old["worst_r"]), cur["worst_r"]
            if rank[cur["class"]] > rank[old["class"]]:
                bad.append(f"{fn} [{reg}]: class {old['class']} -> {cur['class']} (worst r {rfmt(br)} -> {rfmt(sr)})")
            elif sr > 1 and sr > 2 * br:
                bad.append(f"{fn} [{reg}]: worst r {rfmt(br)} -> {rfmt(sr)} (> 2x)")
        if s["hard"] > b["hard"]:
            bad.append(f"{fn}: hard violations {b['hard']} -> {s['hard']}")
    return bad


def update_baseline(S, base):
    for fn, s in S.items():
        e = dict(s)
        for k in ("worst_r", "p50", "p90", "p99"):
            e[k] = jnum(e[k])
        e["regimes"] = {r: {k: jnum(v) for k, v in d.items()} for r, d in s["regimes"].items()}
        base["functions"][fn] = e
    base["meta"] = {"limit_r": LIMIT_R, "bug_r": BUG_R, "eps": "2^-52", "truth_dps": DPS, "seed": SEED}
    with open(BASELINE, "w") as fh:
        json.dump(base, fh, indent=1, sort_keys=True)
        fh.write("\n")


# ---- generated rustdoc tables -----------------------------------------------

BEGIN, END = "//! <!-- accuracy:begin -->", "//! <!-- accuracy:end -->"
TYPE_NAME = {"normal": "Normal", "studentt": "StudentT", "chisquared": "ChiSquared", "fisherf": "FisherF",
             "uniform": "Uniform", "exponential": "Exponential", "cauchy": "Cauchy", "weibull": "Weibull",
             "lognormal": "LogNormal", "gamma": "Gamma", "beta": "Beta", "inversegaussian": "InverseGaussian",
             "bernoulli": "Bernoulli", "binomial": "Binomial", "poisson": "Poisson", "geometric": "Geometric",
             "negbinomial": "NegBinomial", "negbinomial_ms": "NegBinomial (`from_mean_size`)",
             "hypergeometric": "Hypergeometric"}


def cell(e):
    """One table cell from a baseline entry (function or regime)."""
    if e is None:
        return "–"
    r = rnum(e["worst_r"])
    if e.get("hard"):
        return f"{rfmt(r)} (hard)"
    c = e["class"]
    return rfmt(r) + ("" if c == "limit" else f" ({c})")


def table_block(group, F):
    npts = sum(F[f]["n"] for f in F if FN.get(f, {}).get("group") == group)
    head = [
        BEGIN,
        "//! ## Accuracy",
        "//!",
        "//! Worst error ratio `r` measured by `scripts/accuracy_sweep.py` against",
        f"//! mpmath ({npts} points): `r` = relative error / (max(κ, 1)·ε), κ the",
        "//! condition number over all real inputs, ε = 2⁻⁵²; for a log output the",
        "//! absolute error over the sensitivity of the log. `r ≤ " + f"{LIMIT_R:g}" + "`: at the accuracy",
        "//! the problem allows; `(review)` up to " + f"{BUG_R:g}" + "; `(bug)` above it, a structural error;",
        "//! `(hard)` a probability outside [0, 1], a NaN, a non-monotone cdf or a",
        "//! quantile outside the support. Generated from",
        "//! `scripts/accuracy_baseline.json` by `--update-baseline`; not edited by hand.",
        "//!",
    ]
    rows = []
    if group == "continuous":
        cols = ["cdf", "sf", "pdf", "log_density", "quantile", "isf"]
        rows.append("//! | Distribution | " + " | ".join(cols) + " |")
        rows.append("//! |---|" + "---|" * len(cols))
        for ty in CONT:
            rows.append(f"//! | {TYPE_NAME[ty]} | " + " | ".join(cell(F.get(f"{ty}.{c}")) for c in cols) + " |")
    elif group == "discrete":
        cols = ["mass", "log_mass", "cdf", "sf", "quantile"]
        rows.append("//! | Distribution | " + " | ".join(cols) + " |")
        rows.append("//! |---|" + "---|" * len(cols))
        for ty in DISC:
            rows.append(f"//! | {TYPE_NAME[ty]} | " + " | ".join(cell(F.get(f"{ty}.{c}")) for c in cols) + " |")
    else:
        for fams, cols in ((("gammp", "gammq"), ("gammp", "gammq")), (("betai",), ("betai",))):
            regs = sorted(set().union(*[F[f]["regimes"].keys() for f in fams if f in F]))
            rows.append("//! | Kernel regime | " + " | ".join(f"`{c}`" for c in cols) + " |")
            rows.append("//! |---|" + "---|" * len(cols))
            for reg in regs:
                rows.append(f"//! | {reg} | " + " | ".join(cell(F.get(c, {}).get("regimes", {}).get(reg))
                                                         for c in cols) + " |")
            rows.append("//!")
        rows.pop()
    return head + rows + [END]


def write_tables(base):
    F = base["functions"]
    for group, rel in (("incomplete", "src/special/incomplete.rs"), ("continuous", "src/dist/continuous.rs"),
                       ("discrete", "src/dist/discrete.rs")):
        if not any(FN.get(f, {}).get("group") == group for f in F):
            continue
        path = os.path.join(CRATE, rel)
        with open(path) as fh:
            lines = fh.read().split("\n")
        i, j = lines.index(BEGIN), lines.index(END)
        lines[i:j + 1] = table_block(group, F)
        with open(path, "w") as fh:
            fh.write("\n".join(lines))


# ---- fixtures -----------------------------------------------------------------

def fixture_path(group):
    return os.path.join(FIXTURES, f"accuracy_{group}.json")


def load_fixture_points(want):
    out = []
    for group in ("incomplete", "special", "continuous", "discrete"):
        path = fixture_path(group)
        if os.path.exists(path):
            with open(path) as fh:
                for e in json.load(fh):
                    if e["fn"] in want:
                        out.append(e)
    return out


def fixture_entry(pt):
    c = classify(pt["r"], bool(pt.get("hard")))
    e = {"fn": pt["fn"], "params": pt["params"], "x": pt["x"], "truth": pt["truth"], "regime": pt["regime"],
         "class": c}
    if pt.get("pl") is not None:
        e["pl"] = pt["pl"]
    kind = FN[pt["fn"]]["kind"]
    if kind == "dquantile":
        lim = LIMIT_R if c == "limit" else BUG_R
        e["alt"] = [int(j) for j, r in sorted(pt.get("alt_r", {}).items()) if r is not None and r <= lim]
        # The crate's own answer, scored within the limit, is accepted too.
        if c != "bug" and pt["obs"] is not None and int(pt["obs"]) != int(pt["truth"]) \
                and int(pt["obs"]) not in e["alt"]:
            e["alt"].append(int(pt["obs"]))
    else:
        t = pt["truth"]
        e["truth"] = t if t in ("inf", "-inf") else repr(float(mpf(t)) if abs(mpf(t)) <= F64_MAX else
                                                         (math.inf if mpf(t) > 0 else -math.inf))
        e["k"] = repr(float(mpf(pt["K"])))
    if c == "bug":
        e["skip"] = tracker_label(pt)
    else:
        e["max_r"] = LIMIT_R if c == "limit" else BUG_R
    return e


def export_fixtures(points, want):
    new = {}
    groups = {}
    for pt in points:
        if pt.get("r") is None:
            continue
        groups.setdefault((pt["fn"], pt["regime"]), []).append(pt)
    for pts in groups.values():
        pts.sort(key=lambda p: -p["r"])
        for p in pts[:FIXTURE_N]:
            e = fixture_entry(p)
            new[(p["fn"], tuple(p["params"]), p["x"])] = e
    # Re-scored earlier fixture points are replaced too; unscored ones kept.
    rescored = {(p["fn"], tuple(p["params"]), p["x"]): p for p in points if p.get("from_fixture") and
                p.get("r") is not None}
    for key, p in rescored.items():
        new.setdefault(key, fixture_entry(p))
    for group in ("incomplete", "special", "continuous", "discrete"):
        path = fixture_path(group)
        old = []
        if os.path.exists(path):
            with open(path) as fh:
                old = json.load(fh)
        merged = {}
        for e in old:
            k = (e["fn"], tuple(e["params"]), e["x"])
            if e["fn"] in want and k in rescored:
                continue
            merged[k] = e
        for k, e in new.items():
            if FN[k[0]]["group"] == group:
                merged[k] = e
        if not merged:
            continue
        out = sorted(merged.values(), key=lambda e: (e["fn"], e["regime"], e["params"], e["x"]))
        with open(path, "w") as fh:
            fh.write("[\n" + ",\n".join(json.dumps(e, ensure_ascii=False) for e in out) + "\n]\n")
        print(f"fixtures: {len(out)} points in {os.path.relpath(path, CRATE)}")


# ============================================================================
# Main
# ============================================================================

def select(only):
    if not only:
        return set(FN)
    want = set()
    for name in only.split(","):
        name = name.strip()
        hit = {f for f in FN if f == name or f.split(".")[0] == name}
        if not hit:
            sys.exit(f"--only: unknown function {name}")
        want |= hit
    return want


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--export-fixtures", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    want = select(args.only)
    # Scoring subtracts truths near an observed f64 in mpmath.
    mp.dps = DPS

    build_probe()
    t0 = time.time()
    points = generate(run_probe, want)
    index = {(p["fn"], tuple(p["params"]), p["x"]): p for p in points}
    for e in load_fixture_points(want):
        k = (e["fn"], tuple(e["params"]), e["x"])
        if k in index:
            index[k]["from_fixture"] = True
        else:
            index[k] = {"fn": e["fn"], "params": e["params"], "x": e["x"], "pl": e.get("pl"), "gid": None,
                        "from_fixture": True}
            points.append(index[k])
    print(f"points: {len(points)} ({(time.time() - t0):.0f} s to generate)", flush=True)
    obs = run_probe([(p["fn"], p["params"], p["x"]) for p in points])
    for p, o in zip(points, obs):
        p["obs"] = o
        p["regime"] = regime_of(p["fn"], [as_num(s) for s in p["params"]], as_num(p["x"]))

    names = set(want)
    for f in want:
        if FN[f]["kind"] == "dquantile":
            names |= {f"{FN[f]['dist']}.cdf", f"{FN[f]['dist']}.sf"}
    cache = load_cache(sorted(names))
    jobs = [(p["fn"], tuple(p["params"]), p["x"],
             p["obs"] if FN[p["fn"]]["kind"] in ("quantile", "dquantile") else None) for p in points]
    compute_truth(jobs, cache, args.jobs)
    compute_truth(aux_jobs(points, cache), cache, args.jobs)

    for p in points:
        score_point(p, p["obs"], cache[p["fn"]][cache_key(p["params"], p["x"])], cache)
    monotone_checks(points)
    S = summarize(points, want)
    lines = write_report(points, S, os.path.join(CACHE, "report.txt"))
    print("\n".join(lines[:len(S) + 4]))
    print(f"report: {os.path.relpath(os.path.join(CACHE, 'report.txt'), CRATE)}")

    status = 0
    if args.check:
        bad = check(S, load_baseline())
        if bad:
            print("check FAILED:\n  " + "\n  ".join(bad))
            status = 1
        else:
            print("check passed")
    if args.update_baseline:
        base = load_baseline()
        update_baseline(S, base)
        write_tables(base)
        print(f"baseline: {os.path.relpath(BASELINE, CRATE)} and the //! tables updated")
    if args.export_fixtures:
        export_fixtures(points, want)
    sys.exit(status)


if __name__ == "__main__":
    main()
