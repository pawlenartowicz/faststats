//! Threshold-free cluster enhancement (TFCE) over a [`Domain`].
//!
//! Two implementations of one sum: stepped for [`Weighting::SmithNichols`] /
//! [`Weighting::MneStep`] / [`Weighting::Nilearn`], closed-form integral for
//! [`Weighting::Exact`] (see its docs). [`tfce_naive`] re-labels connected components at every band
//! (the readable oracle, a transcription of MNE-Python's loop for the stepped
//! rules), and [`tfce()`] sweeps the bands once with an incremental
//! union-find (the product path). Both take the same [`TfceParams`] and agree
//! within `1e-12` relative on every fixture in `tests/fixtures/tfce_*.json`.
//!
//! Which of the two the caller gets is not observable in the numbers beyond
//! that tolerance, but the band rule is: [`Weighting::SmithNichols`] and
//! [`Weighting::MneStep`] evaluate the `arange(start, max(stat), step)` grid,
//! so `step` changes the answer; [`Weighting::Nilearn`] evaluates 100 equal
//! steps up to `max(stat)` and ignores `start` and `step`; and
//! [`Weighting::Exact`] integrates over the distinct `stat` values and ignores
//! `step` entirely. [`tfce_bands`] exposes the band list directly for callers
//! that want none of these grids.
//!
//! The module's oracle is MNE-Python 1.12.1
//! (`mne.stats.cluster_level._find_clusters`, `tail=1`), frozen into
//! `tests/fixtures/tfce_*.json` as `expected_mne` by
//! `scripts/gen_tfce_golden.py` and checked by
//! `tests/tfce_oracle.rs::g2_naive_matches_mne` and
//! `::g3_union_find_matches_mne`. The same fixtures carry an independent NumPy
//! transcription of the stepped Smith–Nichols sum (`expected_smith_nichols`),
//! which is a cross-check on this crate's own definition rather than a second
//! tool's answer.

use alloc::vec::Vec;

use crate::domain::{Domain, Graph, conn_offsets};
use crate::error::NeuroError;

/// Weight `w_i` given to the threshold band ending at `h_i`. The reference
/// tools disagree here by several × on the same map, so the choice is a
/// required parameter, never a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weighting {
    /// Smith & Nichols (2009) stepped integral: `w_i = h_i^h · step`. The rule
    /// FSL `fslmaths -tfce`, PALM, and the SPM TFCE toolbox implement. No
    /// external oracle: no fixture is generated from FSL or PALM. Checked
    /// against a NumPy transcription of the same sum
    /// (`expected_smith_nichols` in `tests/fixtures/tfce_*.json`, written by
    /// `scripts/gen_tfce_golden.py`; `tests/tfce_oracle.rs::g4_naive_matches_numpy_smith_nichols`),
    /// which restates this crate's definition rather than a second tool's.
    SmithNichols,
    /// MNE-Python (`mne.stats.cluster_level._find_clusters`, ≤ 1.12.1):
    /// `w_0 = |h_0|^h`, `w_i = |h_i − h_{i−1}|^h` for `i ≥ 1` — the band
    /// *height* raised to `h`, with no `· step` factor and no threshold value.
    /// Matches MNE 1.12.1 `_find_clusters` (`tail=1`) to `rel 1e-12` on every
    /// `expected_mne` map in `tests/fixtures/tfce_*.json`
    /// (`tests/tfce_oracle.rs::g2_naive_matches_mne`,
    /// `::g3_union_find_matches_mne`).
    MneStep,
    /// Closed-form Smith–Nichols integral `∫_{h0}^{stat[v]} e_v(h)^E · h^H dh`
    /// with `h0 = start` — no threshold grid, no `step` (still validated,
    /// pass any positive value). `e_v(h)` is piecewise constant between
    /// distinct `stat` values, so each run of constant component size `s` over
    /// `[a, b)` contributes `s^E · (b^{H+1} − a^{H+1}) / (H+1)`. The
    /// recommended choice for new analyses (`start = 0`); `SmithNichols` /
    /// `MneStep` exist for parity with FSL/PALM and MNE pipelines. Requires
    /// `h > −1` and `start ≥ 0` ([`NeuroError::InvalidParams`]). Nodes with
    /// `stat ≤ start` get 0. No external oracle: no reference tool emits this
    /// quantity in a form that can be frozen here. Validated union-find against
    /// naive to `rel 1e-12`
    /// (`tests/tfce_oracle.rs::g1_union_find_matches_naive_on_fixtures`,
    /// `::g1_union_find_matches_naive_synthetic`) and as the `step → 0` limit
    /// of `SmithNichols`
    /// (`tests/perm_oracle.rs::g11_exact_is_limit_of_smith_nichols`).
    Exact,
    /// nilearn `calculate_tfce` under `dh="auto"` (nilearn 0.14.1,
    /// `mass_univariate/_utils.py`): 100 thresholds
    /// `np.linspace(0, max(stat), 101)[1:]`, rebuilt for every map, so the
    /// grid scales with the map and its last threshold is the maximum itself.
    /// `w_i = h_i^h` with no `· step` factor, and membership is `stat >= h_i`
    /// (not strict), so the largest node is a member of every band.
    /// `start` and `step` are not used (still validated: pass any finite
    /// `start` and any positive `step`). A map with `max(stat) <= 0` gets all
    /// zeros, as in nilearn. The map is also all zeros when `max(stat) / 100`
    /// is below `f64::MIN_POSITIVE`: the step is then subnormal and the
    /// thresholds are no longer strictly increasing. Matches nilearn's
    /// `calculate_tfce` to `rel 1e-12` on random maps and masks, one- and
    /// two-sided (`NeuroStats/python/tests/test_tfce_parity.py` in the `faststats`
    /// repository); union-find against naive to `rel 1e-12`
    /// (`tests/tfce_oracle.rs::g1_union_find_matches_naive_on_fixtures`,
    /// `::g1_union_find_matches_naive_synthetic`).
    Nilearn,
}

/// TFCE parameters. Under [`Weighting::SmithNichols`] and
/// [`Weighting::MneStep`], thresholds follow NumPy
/// `arange(start, max(stat), step)` (start inclusive, stop exclusive), so the
/// largest node clears every band; [`Weighting::Nilearn`] builds its own grid
/// from `max(stat)` and [`Weighting::Exact`] uses no grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TfceParams {
    /// Extent exponent `E`. Smith–Nichols / MNE default `0.5`.
    pub e: f64,
    /// Height exponent `H`. Smith–Nichols / MNE default `2.0`. `> −1` under
    /// [`Weighting::Exact`].
    pub h: f64,
    /// First threshold `h_0`; nodes with `stat <= start` get 0. Finite;
    /// `≥ 0` under [`Weighting::Exact`]. Unused by [`Weighting::Nilearn`]
    /// (pass any finite value).
    pub start: f64,
    /// Threshold spacing; must be `> 0` and finite. Unused by
    /// [`Weighting::Exact`] and [`Weighting::Nilearn`] (pass any positive
    /// value).
    pub step: f64,
    /// Band weight rule — see [`Weighting`].
    pub weighting: Weighting,
}

impl TfceParams {
    fn validate(&self) -> Result<(), NeuroError> {
        let mut ok = self.e.is_finite()
            && self.h.is_finite()
            && self.start.is_finite()
            && self.step.is_finite()
            && self.step > 0.0;
        if self.weighting == Weighting::Exact {
            // Closed form divides by H+1 and evaluates h^{H+1} at h0; H ≤ −1
            // diverges at h0 = 0 and h^{H+1} is NaN for negative h with
            // fractional exponent.
            ok = ok && self.h > -1.0 && self.start >= 0.0;
        }
        if ok {
            Ok(())
        } else {
            Err(NeuroError::InvalidParams)
        }
    }
}

/// Which sign of the statistic counts as evidence, and so which map TFCE
/// enhances. A required argument of [`tfce_tail_into`], never a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tail {
    /// TFCE of the map. Large positive values of `stat` are the evidence.
    Positive,
    /// TFCE of the negated map. Large negative values of `stat` are the
    /// evidence.
    Negative,
    /// TFCE of the map minus TFCE of the negated map. The output is signed:
    /// positive where `stat` is positive, negative where it is negative. The
    /// tested quantity is its absolute value.
    TwoSided,
}

/// Length and finiteness checks on `stat`; returns `max(stat)` (`None` when
/// the map is empty).
fn check_stat(domain: &Domain, stat: &[f64]) -> Result<Option<f64>, NeuroError> {
    if stat.len() != domain.n_nodes() {
        return Err(NeuroError::MismatchedLengths {
            expected: domain.n_nodes(),
            got: stat.len(),
        });
    }
    let mut max: Option<f64> = None;
    for &s in stat {
        if !s.is_finite() {
            return Err(NeuroError::NonFiniteStat);
        }
        max = Some(max.map_or(s, |m| if s > m { s } else { m }));
    }
    Ok(max)
}

/// Checks shared by both implementations; returns `max(stat)` (or `None` when
/// the map is empty).
fn check_inputs(
    domain: &Domain,
    stat: &[f64],
    params: &TfceParams,
) -> Result<Option<f64>, NeuroError> {
    params.validate()?;
    check_stat(domain, stat)
}

/// `thresholds` strictly increasing and finite, `weights` same length and
/// finite, `e` finite.
fn check_bands(thresholds: &[f64], weights: &[f64], e: f64) -> Result<(), NeuroError> {
    if !e.is_finite() {
        return Err(NeuroError::InvalidParams);
    }
    let ok = thresholds.len() == weights.len()
        && thresholds.iter().all(|h| h.is_finite())
        && weights.iter().all(|w| w.is_finite())
        && thresholds.windows(2).all(|w| w[0] < w[1]);
    if ok {
        Ok(())
    } else {
        Err(NeuroError::InvalidBands)
    }
}

/// Band membership: `stat > h` (strict, the `tfce` rule) or `stat >= h`
/// (nilearn's `x >= h` after zeroing).
#[inline]
fn member(s: f64, h: f64, strict: bool) -> bool {
    if strict { s > h } else { s >= h }
}

/// NumPy `np.arange(start, stop, step)` for `f64`, reproduced bit-exactly
/// (NumPy 2.3.5): `n = ceil((stop − start) / step)`, `h_0 = start`,
/// `h_1 = start + step`, `h_i = start + i·delta` with `delta = (start + step) − start`.
/// Neither `start + i·step` nor repeated accumulation matches `arange` on every grid.
/// Writes into `out` (cleared first) so the sweep can reuse one buffer.
fn arange_into(start: f64, stop: f64, step: f64, out: &mut Vec<f64>) {
    out.clear();
    let n_f = libm::ceil((stop - start) / step);
    if n_f.is_nan() || n_f <= 0.0 {
        return;
    }
    let n = n_f as usize;
    let delta = (start + step) - start;
    out.extend((0..n).map(|i| match i {
        0 => start,
        1 => start + step,
        _ => start + (i as f64) * delta,
    }));
}

fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let mut v = Vec::new();
    arange_into(start, stop, step, &mut v);
    v
}

/// Number of thresholds of [`Weighting::Nilearn`] (nilearn's `dh="auto"`).
const NILEARN_STEPS: usize = 100;

/// NumPy `np.linspace(0, max, 101)[1:]` for `f64`, reproduced bit-exactly
/// (NumPy 2.5.3): `h_k = k · (max / 100)` for `k = 1..=99`, and `h_100 = max`
/// itself. Bit equality matters because membership is `stat >= h_k`. Left
/// empty when `max / 100 < f64::MIN_POSITIVE` (`max <= 0`, or a step so small
/// that the grid is not strictly increasing). Writes into `out` (cleared
/// first) so the sweep can reuse one buffer.
fn nilearn_thresholds_into(max: f64, out: &mut Vec<f64>) {
    out.clear();
    let step = max / NILEARN_STEPS as f64;
    if step < f64::MIN_POSITIVE {
        return;
    }
    out.extend((1..NILEARN_STEPS).map(|k| k as f64 * step));
    out.push(max);
}

/// Weight `w_i` of band `i` under `params.weighting` (see [`Weighting`]).
/// Only called for the grid rules — [`Weighting::Exact`] builds its bands
/// with the closed-form weight instead (see [`tfce_naive`], [`sweep_exact`]).
fn band_weight(i: usize, ths: &[f64], params: &TfceParams) -> f64 {
    let h = ths[i];
    match params.weighting {
        Weighting::SmithNichols => libm::pow(h, params.h) * params.step,
        Weighting::Nilearn => libm::pow(h, params.h),
        Weighting::Exact => unreachable!("Exact never calls band_weight"),
        Weighting::MneStep => {
            let dh = if i == 0 {
                libm::fabs(h)
            } else {
                libm::fabs(h - ths[i - 1])
            };
            libm::pow(dh, params.h)
        }
    }
}

/// TFCE-enhanced map — naive stepped re-clustering (the oracle path).
///
/// `TFCE(v) = Σ_{i : h_i < stat[v]} w_i · extent_i(v)^e`, where `extent_i(v)` is
/// the node count of the connected component (under `domain` adjacency,
/// restricted to nodes with `stat > h_i`, **strict**) containing `v`, thresholds
/// `h_i` are NumPy `arange(start, max(stat), step)`, and `w_i` follows
/// `params.weighting`. Under [`Weighting::Nilearn`] the thresholds are its own
/// 100-step grid and membership is `stat >= h_i`. Positive tail only.
/// Summation is ascending in `i`, one
/// addition per band per node — a line-for-line transcription of
/// `mne.stats.cluster_level._find_clusters` (MNE 1.12.1, `tail=1`), so under
/// [`Weighting::MneStep`] it matches MNE to `pow` rounding
/// (`tests/fixtures/tfce_*.json`, `expected_mne`, rel `1e-12`).
///
/// `stat`: one finite value per node in `domain` node order (`len == n_nodes`).
/// Cost `O(V · n_thresholds)`; use [`tfce()`] for anything but small validation maps.
/// Under [`Weighting::Exact`] the bands are the distinct `stat` values
/// `u_0 = start < u_1 < … < u_K` with weight `(u_{k+1}^{H+1} − u_k^{H+1}) / (H+1)`
/// on band `k` — the closed-form integral, `O(V · K)`.
///
/// Errors: [`NeuroError::MismatchedLengths`] if `stat.len() != domain.n_nodes()`,
/// [`NeuroError::NonFiniteStat`] on any NaN/±∞, [`NeuroError::InvalidParams`] if
/// `step <= 0` or any parameter is non-finite, or, under [`Weighting::Exact`],
/// `h <= -1` or `start < 0`.
pub fn tfce_naive(
    domain: &Domain,
    stat: &[f64],
    params: &TfceParams,
) -> Result<Vec<f64>, NeuroError> {
    let n = stat.len();
    let mut out = alloc::vec![0.0f64; n];
    let Some(max) = check_inputs(domain, stat, params)? else {
        return Ok(out);
    };
    // (threshold, weight) bands, contribution `weight · extent^E`. SmithNichols
    // and MneStep take the `arange` grid, Nilearn its 100-step grid, Exact the
    // distinct stat values above h0 with the closed-form interval weight.
    let bands: Vec<(f64, f64)> = match params.weighting {
        Weighting::Exact => {
            let mut u: Vec<f64> = stat.iter().copied().filter(|&s| s > params.start).collect();
            u.sort_unstable_by(f64::total_cmp);
            u.dedup();
            let hp1 = params.h + 1.0;
            let mut prev = params.start;
            let mut bands = Vec::with_capacity(u.len());
            for &next in &u {
                let w = (libm::pow(next, hp1) - libm::pow(prev, hp1)) / hp1;
                bands.push((prev, w));
                prev = next;
            }
            bands
        }
        Weighting::Nilearn => {
            let mut ths = Vec::new();
            nilearn_thresholds_into(max, &mut ths);
            grid_bands(&ths, params)
        }
        Weighting::SmithNichols | Weighting::MneStep => {
            grid_bands(&arange(params.start, max, params.step), params)
        }
    };

    let strict = params.weighting != Weighting::Nilearn;
    naive_sum(domain, stat, &bands, params.e, strict, &mut out);
    Ok(out)
}

/// `(threshold, weight)` pairs of a threshold grid under `params.weighting`.
fn grid_bands(ths: &[f64], params: &TfceParams) -> Vec<(f64, f64)> {
    (0..ths.len())
        .map(|i| (ths[i], band_weight(i, ths, params)))
        .collect()
}

/// Naive re-clustering sum over `(threshold, weight)` bands into `out`
/// (must be zeroed by the caller). Membership per `strict`.
fn naive_sum(
    domain: &Domain,
    stat: &[f64],
    bands: &[(f64, f64)],
    e: f64,
    strict: bool,
    out: &mut [f64],
) {
    let n = stat.len();
    let mut visited = alloc::vec![false; n];
    let mut queue: Vec<u32> = Vec::new();
    let mut nb: Vec<u32> = Vec::new();
    for &(h, wi) in bands {
        visited.iter_mut().for_each(|v| *v = false);
        for seed in 0..n {
            if visited[seed] || !member(stat[seed], h, strict) {
                continue;
            }
            // BFS one component; `queue` doubles as the member list.
            queue.clear();
            queue.push(seed as u32);
            visited[seed] = true;
            let mut head = 0;
            while head < queue.len() {
                let v = queue[head] as usize;
                head += 1;
                domain.neighbours_into(v, &mut nb);
                for &u in &nb {
                    let u = u as usize;
                    if !visited[u] && member(stat[u], h, strict) {
                        visited[u] = true;
                        queue.push(u as u32);
                    }
                }
            }
            let add = wi * libm::pow(queue.len() as f64, e);
            for &v in &queue {
                out[v as usize] += add;
            }
        }
    }
}

/// TFCE-enhanced map — incremental union-find (the product path).
///
/// Same quantity and conventions as [`tfce_naive`] (strict `stat > h_i`
/// membership and NumPy `arange` thresholds under [`Weighting::SmithNichols`]
/// and [`Weighting::MneStep`], `stat >= h_i` on the 100-step grid under
/// [`Weighting::Nilearn`], `params.weighting` band weights, positive tail),
/// computed in one
/// descending sweep: nodes are sorted by `stat`, activated as the threshold
/// drops below them, and merged with active neighbours in a disjoint-set
/// forest (union by size, path compression). Contributions are banked at
/// component roots and paid lazily per run of bands with constant size, so
/// the summation order differs from the naive ascending sum: agreement is
/// `rel 1e-12` (abs floor `1e-12 · max`), not bit equality. Validated against
/// MNE 1.12.1 under [`Weighting::MneStep`] (`tests/fixtures/tfce_*.json`,
/// `expected_mne`).
///
/// `stat`: one finite value per node in `domain` node order (`len == n_nodes`).
/// Cost `O(V log V + V·α(V) + n_thresholds)` under the stepped rules,
/// `O(V log V + V·α(V))` under [`Weighting::Exact`]; allocates `O(V)` scratch.
///
/// Errors: as [`tfce_naive`].
///
/// ```
/// use neurostats::{Conn, Domain, TfceParams, Weighting, tfce, tfce_naive};
/// let dom = Domain::from_volume([3, 3, 3], Conn::Vertex);
/// let stat: Vec<f64> = (0..27).map(|i| (i % 7) as f64 * 0.7).collect();
/// let p = TfceParams { e: 0.5, h: 2.0, start: 0.5, step: 0.5, weighting: Weighting::MneStep };
/// let fast = tfce(&dom, &stat, &p).unwrap();
/// let slow = tfce_naive(&dom, &stat, &p).unwrap();
/// for (a, b) in fast.iter().zip(&slow) {
///     assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0));
/// }
/// ```
/// Allocates a fresh [`TfceWorkspace`] per call — use [`tfce_into`] in a loop.
pub fn tfce(domain: &Domain, stat: &[f64], params: &TfceParams) -> Result<Vec<f64>, NeuroError> {
    let mut ws = TfceWorkspace::new(domain.n_nodes());
    let mut out = alloc::vec![0.0f64; domain.n_nodes()];
    tfce_into(domain, stat, params, &mut ws, &mut out)?;
    Ok(out)
}

/// Scratch buffers for [`tfce_into`] and [`tfce_tail_into`]: the sweep order,
/// the disjoint-set forest, the threshold grid with its cumulative band
/// weights, and for the negative and two-sided tails the negated map and its
/// TFCE (two node-sized buffers).
/// Grown on demand (the threshold count follows `max(stat)`), never shrunk, so
/// a permutation loop that reuses one workspace reallocates only when a draw
/// needs a longer threshold grid than any earlier one (stepped rules); never
/// under [`Weighting::Exact`].
///
/// A lattice sweep works in voxel-index space, so the forest is sized by the
/// grid, not by the node count: on a mask that fills a small part of its grid
/// a workspace costs several times what the node-indexed CSR path does.
#[derive(Debug, Clone)]
pub struct TfceWorkspace {
    /// The sweep order and its heights, see [`build_order`].
    order: Vec<u64>,
    heights: Vec<u64>,
    forest: Forest,
    ths: Vec<f64>,
    cum: Vec<f64>,
    /// `-stat` in node order. Empty until a [`Tail::Negative`] or
    /// [`Tail::TwoSided`] call.
    neg_stat: Vec<f64>,
    /// TFCE of `-stat` in node order. Empty until a [`Tail::TwoSided`] call.
    neg_out: Vec<f64>,
}

impl TfceWorkspace {
    /// Workspace pre-sized for `n_nodes` nodes (any size works; buffers grow to
    /// fit the map actually passed to [`tfce_into`]).
    pub fn new(n_nodes: usize) -> Self {
        Self {
            order: Vec::with_capacity(n_nodes),
            heights: Vec::new(),
            forest: Forest::new(n_nodes),
            ths: Vec::new(),
            cum: Vec::new(),
            neg_stat: Vec::new(),
            neg_out: Vec::new(),
        }
    }
}

/// [`tfce()`] into a caller-owned `out`, reusing `ws` — the allocation-free
/// entry point for permutation loops. Same quantity, conventions, tolerances,
/// and errors as [`tfce()`], bit-identical to it. Equal to [`tfce_tail_into`]
/// with [`Tail::Positive`].
///
/// `out.len()` must equal `domain.n_nodes()`, else
/// [`NeuroError::MismatchedLengths`] (checked before `stat`).
pub fn tfce_into(
    domain: &Domain,
    stat: &[f64],
    params: &TfceParams,
    ws: &mut TfceWorkspace,
    out: &mut [f64],
) -> Result<(), NeuroError> {
    tfce_tail_into(domain, stat, params, Tail::Positive, ws, out)
}

/// TFCE of `stat` under `tail` into a caller-owned `out`, reusing `ws`. No
/// allocation once `ws` has grown to the map, so a permutation loop calls
/// this once per draw.
///
/// - [`Tail::Positive`]: `out = TFCE(stat)`, bit-identical to [`tfce()`].
/// - [`Tail::Negative`]: `out = TFCE(−stat)`.
/// - [`Tail::TwoSided`]: `out = TFCE(stat) − TFCE(−stat)`, signed.
///
/// Each `TFCE(·)` is the positive-tail quantity of [`tfce()`] under `params`.
/// The threshold grid of the grid rules is built from the maximum of the map
/// being enhanced: `max(stat)`, `max(−stat)`, or, for the two-sided tail,
/// `max |stat|`, one grid shared by both passes. Under
/// [`Weighting::SmithNichols`] and [`Weighting::MneStep`] the shared grid
/// changes no value (a band above a map's own maximum is empty). Under
/// [`Weighting::Nilearn`] it does, because that grid scales with its maximum;
/// this is nilearn's `calculate_tfce(two_sided_test=True)`, matched to
/// `rel 1e-12` on random maps and masks
/// (`NeuroStats/python/tests/test_tfce_parity.py` in the `faststats` repository).
/// [`Weighting::Exact`] uses no grid. With `start ≥ 0` at most one of the two
/// terms of the two-sided difference is non-zero at a node, so the
/// subtraction loses no digits. The tails are checked against [`tfce_naive`]
/// (`tests/tfce_oracle.rs::g13_tails_match_naive`).
///
/// `stat`: one finite value per node in `domain` node order. `out`: same
/// length; every entry is overwritten.
///
/// Errors: [`NeuroError::MismatchedLengths`] if `out.len()` (checked first)
/// or `stat.len()` differs from `domain.n_nodes()`;
/// [`NeuroError::NonFiniteStat`] and [`NeuroError::InvalidParams`] as
/// [`tfce()`].
///
/// ```
/// use neurostats::{Conn, Domain, Tail, TfceParams, TfceWorkspace, Weighting, tfce, tfce_tail_into};
/// let dom = Domain::from_volume([4, 1, 1], Conn::Face);
/// let stat = [3.0, 1.0, -2.0, -2.0];
/// let neg: Vec<f64> = stat.iter().map(|s| -s).collect();
/// let p = TfceParams { e: 0.5, h: 2.0, start: 0.0, step: 1.0, weighting: Weighting::Exact };
/// let mut ws = TfceWorkspace::new(dom.n_nodes());
/// let mut out = vec![0.0; 4];
/// tfce_tail_into(&dom, &stat, &p, Tail::TwoSided, &mut ws, &mut out).unwrap();
/// let (up, down) = (tfce(&dom, &stat, &p).unwrap(), tfce(&dom, &neg, &p).unwrap());
/// for v in 0..4 {
///     assert_eq!(out[v], up[v] - down[v]);
/// }
/// assert!(out[0] > 0.0 && out[2] < 0.0);
/// ```
pub fn tfce_tail_into(
    domain: &Domain,
    stat: &[f64],
    params: &TfceParams,
    tail: Tail,
    ws: &mut TfceWorkspace,
    out: &mut [f64],
) -> Result<(), NeuroError> {
    let n = domain.n_nodes();
    if out.len() != n {
        return Err(NeuroError::MismatchedLengths {
            expected: n,
            got: out.len(),
        });
    }
    let Some(max) = check_inputs(domain, stat, params)? else {
        return Ok(());
    };
    if tail == Tail::Positive {
        tfce_pass(domain, stat, params, max, ws, out);
        return Ok(());
    }
    // The negated map is moved out of the workspace for the passes, which
    // borrow the rest of it mutably, and moved back afterwards.
    let mut neg = core::mem::take(&mut ws.neg_stat);
    neg.clear();
    let mut neg_max = f64::NEG_INFINITY;
    for &s in stat {
        neg.push(-s);
        if -s > neg_max {
            neg_max = -s;
        }
    }
    if tail == Tail::Negative {
        tfce_pass(domain, &neg, params, neg_max, ws, out);
    } else {
        let grid_max = if max > neg_max { max } else { neg_max };
        tfce_pass(domain, stat, params, grid_max, ws, out);
        let mut neg_out = core::mem::take(&mut ws.neg_out);
        neg_out.clear();
        neg_out.resize(n, 0.0);
        tfce_pass(domain, &neg, params, grid_max, ws, &mut neg_out);
        for (o, &m) in out.iter_mut().zip(&neg_out) {
            *o -= m;
        }
        ws.neg_out = neg_out;
    }
    ws.neg_stat = neg;
    Ok(())
}

/// One positive-tail TFCE pass over `stat` into `out`, with the grid rules'
/// thresholds built from `grid_max` (the map's own maximum, or `max |stat|`
/// when two passes share a grid). Inputs are already checked.
fn tfce_pass(
    domain: &Domain,
    stat: &[f64],
    params: &TfceParams,
    grid_max: f64,
    ws: &mut TfceWorkspace,
    out: &mut [f64],
) {
    let strict = match params.weighting {
        Weighting::Exact => {
            sweep_exact(domain, stat, params, ws, out);
            return;
        }
        Weighting::Nilearn => {
            nilearn_thresholds_into(grid_max, &mut ws.ths);
            false
        }
        Weighting::SmithNichols | Weighting::MneStep => {
            arange_into(params.start, grid_max, params.step, &mut ws.ths);
            true
        }
    };
    ws.cum.clear();
    ws.cum.push(0.0f64);
    for i in 0..ws.ths.len() {
        let last = *ws.cum.last().expect("non-empty");
        ws.cum.push(last + band_weight(i, &ws.ths, params));
    }
    sweep_bands(domain, stat, params.e, strict, ws, out);
}

/// The index space a sweep's forest lives in, and the neighbour relation on
/// it. A CSR domain sweeps node indices; a lattice sweeps linear voxel indices,
/// where a neighbour is an arithmetic step instead of a random read into a
/// 26-column table — the sweep is memory-bound and that table is its dominant
/// cost. `stat` and `out` stay in node order either way.
trait Neighbours {
    /// Size of the index space the forest is allocated over.
    fn len(&self) -> usize;
    /// Index-space index of node `i` (the identity for a CSR domain).
    fn index_of(&self, i: usize) -> usize;
    /// Calls `f(u)` for every neighbour `u` of `v`, both in this index space.
    fn for_each<F: FnMut(usize)>(&self, v: usize, f: F);
    /// Largest neighbour count of any node: the size of the sweep's per-event
    /// buffers. A CSR row may be any length, not only a lattice's 26.
    fn max_degree(&self) -> usize;
}

struct CsrView<'a> {
    offsets: &'a [u32],
    neighbours: &'a [u32],
}

impl Neighbours for CsrView<'_> {
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn index_of(&self, i: usize) -> usize {
        i
    }

    fn for_each<F: FnMut(usize)>(&self, v: usize, mut f: F) {
        let (lo, hi) = (self.offsets[v] as usize, self.offsets[v + 1] as usize);
        for &u in &self.neighbours[lo..hi] {
            f(u as usize);
        }
    }

    fn max_degree(&self) -> usize {
        self.offsets
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0)
    }
}

struct LatticeView<'a> {
    dims: [usize; 3],
    lin_of: &'a [u32],
    /// Coordinate step per allowed offset (`n_off` entries used), for the bound test.
    off: [[i32; 3]; 26],
    /// The same offsets as linear-index steps, so a neighbour is one addition.
    delta: [i64; 26],
    n_off: usize,
}

impl<'a> LatticeView<'a> {
    fn new(dims: [usize; 3], conn: crate::Conn, lin_of: &'a [u32]) -> Self {
        let (off, n_off) = conn_offsets(conn);
        let [_, ny, nz] = dims;
        let mut delta = [0i64; 26];
        for (d, o) in delta.iter_mut().zip(&off) {
            *d = (o[0] as i64 * ny as i64 + o[1] as i64) * nz as i64 + o[2] as i64;
        }
        Self {
            dims,
            lin_of,
            off,
            delta,
            n_off,
        }
    }
}

impl Neighbours for LatticeView<'_> {
    fn len(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    fn index_of(&self, i: usize) -> usize {
        self.lin_of[i] as usize
    }

    fn for_each<F: FnMut(usize)>(&self, v: usize, mut f: F) {
        let [nx, ny, nz] = self.dims;
        let z = v % nz;
        let t = v / nz;
        let (y, x) = (t % ny, t / ny);
        for k in 0..self.n_off {
            let [dx, dy, dz] = self.off[k];
            let (qx, qy, qz) = (
                x as i64 + dx as i64,
                y as i64 + dy as i64,
                z as i64 + dz as i64,
            );
            if qx < 0 || qy < 0 || qz < 0 || qx >= nx as i64 || qy >= ny as i64 || qz >= nz as i64 {
                continue;
            }
            // Out-of-mask voxels are never activated, so no mask test is needed:
            // the sweep skips them exactly as it skips an inactive node.
            f((v as i64 + self.delta[k]) as usize);
        }
    }

    fn max_degree(&self) -> usize {
        self.n_off
    }
}

/// One descending union-find sweep over explicit bands, on whichever index
/// space `domain` prefers. `ws.ths` strictly increasing,
/// `ws.cum[k] = Σ_{j<k} w_j` (`len == ths.len() + 1`). Membership per `strict`.
/// Shared by `tfce_pass` (the grid of `params.weighting`) and
/// `tfce_bands_into` (caller grid, either rule).
fn sweep_bands(
    domain: &Domain,
    stat: &[f64],
    e: f64,
    strict: bool,
    ws: &mut TfceWorkspace,
    out: &mut [f64],
) {
    let TfceWorkspace {
        order,
        heights,
        forest,
        ths,
        cum,
        ..
    } = ws;
    match domain.graph() {
        Graph::Csr {
            offsets,
            neighbours,
        } => {
            let g = CsrView {
                offsets,
                neighbours,
            };
            sweep_bands_impl(&g, stat, ths, cum, e, strict, order, heights, forest, out);
        }
        Graph::Lattice { dims, conn, lin_of } => {
            let g = LatticeView::new(*dims, *conn, lin_of);
            sweep_bands_impl(&g, stat, ths, cum, e, strict, order, heights, forest, out);
        }
    }
}

/// Order-preserving `f64` → `u64`: flip every bit of a negative value, flip
/// only the sign bit of a positive one. `a < b` as `f64` iff `key(a) < key(b)`
/// as `u64` for all finite inputs and for `±inf`, which lets the sweep order
/// come from an integer radix sort instead of a float comparison sort.
/// `-0.0` and `+0.0` keep distinct keys (`-0.0` below `+0.0`); they are equal
/// stats and either order is a valid sweep order.
#[inline]
fn sort_key(x: f64) -> u64 {
    let b = x.to_bits();
    const SIGN: u64 = 1u64 << 63;
    if b & SIGN != 0 { !b } else { b ^ SIGN }
}

/// Node index carried in the low half of a sort entry of [`build_order`].
#[inline]
fn node_of(entry: u64) -> usize {
    (entry as u32) as usize
}

/// The sweep order: the nodes whose stat `keep` accepts, by [`sort_key`]
/// descending, equal keys in node order. On return `order[i]` is the `i`-th
/// node's index in `g`'s index space and `heights[i]` the bits of its stat,
/// so the sweep reads both in sequence. The sweep stops at the first
/// non-member anyway, so dropping the rest changes no output and shortens the
/// sort to the activating count.
///
/// A sort entry packs the top 32 bits of the node's key above its node index.
/// A stable LSD radix sort orders the entries by those 32 bits in three
/// passes (11, 11 and 10 bits), with all three histograms counted in the one
/// scan that builds the entries. That leaves only runs of nodes whose keys
/// share their top half out of order; each run is then sorted by the full
/// key. On a statistic map the runs are short, so this moves a third of the
/// bytes of a radix sort over the whole key in three eighths of the passes.
/// A map whose values differ only in their low bits degrades to the
/// comparison sort of the runs, `O(V log V)`.
///
/// Tie order matters: equal keys are equal stats, and keeping them in node
/// order makes the activation sequence among ties deterministic.
fn build_order<G: Neighbours, F: Fn(f64) -> bool>(
    g: &G,
    stat: &[f64],
    keep: F,
    order: &mut Vec<u64>,
    heights: &mut Vec<u64>,
) {
    const MASK: u64 = 0x7ff;
    order.clear();
    let mut count = [[0u32; 2048]; 3];
    for (k, &s) in stat.iter().enumerate() {
        if keep(s) {
            let top = sort_key(s) >> 32;
            count[0][(top & MASK) as usize] += 1;
            count[1][((top >> 11) & MASK) as usize] += 1;
            count[2][(top >> 22) as usize] += 1;
            order.push(top << 32 | k as u64);
        }
    }
    let n = order.len();
    // `heights` doubles as the radix scratch. Every pass overwrites its first
    // `n` slots before reading them, so it is only grown, never cleared.
    if heights.len() < n {
        heights.resize(n, 0);
    }
    for (pass, count) in count.iter().enumerate() {
        // A digit every entry shares cannot change the order.
        if count.iter().any(|&c| c as usize == n) {
            continue;
        }
        let shift = 32 + 11 * pass;
        // Descending: prefix-sum from the highest digit value down.
        let mut start = [0u32; 2048];
        let mut acc = 0u32;
        for (s, &c) in start.iter_mut().zip(count.iter()).rev() {
            *s = acc;
            acc += c;
        }
        for &entry in &order[..n] {
            let b = ((entry >> shift) & MASK) as usize;
            heights[start[b] as usize] = entry;
            start[b] += 1;
        }
        core::mem::swap(order, heights);
    }
    if heights.len() < n {
        heights.resize(n, 0);
    }
    let mut i = 0usize;
    while i < n {
        let top = order[i] >> 32;
        let mut j = i + 1;
        while j < n && order[j] >> 32 == top {
            j += 1;
        }
        if j - i > 1 {
            // Entries of a run share their top half, so comparing two of them
            // as integers compares their node indices.
            order[i..j].sort_unstable_by(|&a, &b| {
                sort_key(stat[node_of(b)])
                    .cmp(&sort_key(stat[node_of(a)]))
                    .then(a.cmp(&b))
            });
        }
        i = j;
    }
    // Node → index space and stat here, in a loop whose iterations are
    // independent, so the two random reads per node overlap across nodes
    // instead of sitting at the head of each activation in the sweep.
    order.truncate(n);
    heights.truncate(n);
    for (o, h) in order.iter_mut().zip(heights.iter_mut()) {
        let k = node_of(*o);
        *o = g.index_of(k) as u64;
        *h = stat[k].to_bits();
    }
}

/// `sweep_bands` over one index space. `stat` and `out` are in node order.
#[allow(clippy::too_many_arguments)]
fn sweep_bands_impl<G: Neighbours>(
    g: &G,
    stat: &[f64],
    ths: &[f64],
    cum: &[f64],
    e: f64,
    strict: bool,
    order: &mut Vec<u64>,
    heights: &mut Vec<u64>,
    uf: &mut Forest,
    out: &mut [f64],
) {
    let nt = ths.len();
    if nt == 0 {
        out.fill(0.0);
        return;
    }
    // Only nodes that clear the lowest band can ever activate.
    build_order(g, stat, |s| member(s, ths[0], strict), order, heights);
    uf.reset(g.len());
    let d = g.max_degree();
    let (mut near, mut above) = (alloc::vec![0u32; d], alloc::vec![0u32; d]);
    let mut ptr = 0usize;
    for i in (0..nt).rev() {
        let h = ths[i];
        while ptr < order.len() && member(f64::from_bits(heights[ptr]), h, strict) {
            let v = order[ptr] as usize;
            ptr += 1;
            let n = uf.active_neighbours(g, v, &mut near, &mut above);
            uf.activate(v, i);
            let mut rv = v; // v is its own root at activation
            for (&u, &p) in near[..n].iter().zip(&above) {
                let ru = uf.find_from(u as usize, p as usize);
                if ru != rv {
                    rv = uf.link(rv, ru, i, cum, e);
                }
            }
        }
    }
    // Walk the nodes, not the index space: on a sparse mask the voxel-count
    // walk would cost several times the node count for nothing.
    for (k, o) in out.iter_mut().enumerate() {
        *o = uf.resolve(g.index_of(k), |uf, r| uf.pay(r, 0, cum, e));
    }
}

/// TFCE over an explicit band list, for callers that build their own
/// threshold grid.
///
/// `TFCE(v) = Σ_i weights[i] · extent_i(v)^e` over bands `i` where `v` is a
/// member: `stat[v] > thresholds[i]` when `strict`, `stat[v] >= thresholds[i]`
/// otherwise. Nodes in no band get 0. Positive tail only; the caller composes
/// tails by negating `stat`. `thresholds` strictly increasing and finite,
/// `weights` same length and finite; an empty list gives all zeros. Node order
/// and `stat` rules as [`tfce()`].
///
/// Oracle: with `thresholds = arange(start, max(stat), step)`, the matching
/// `Weighting` weights, and `strict = true` this is bit-identical to
/// [`tfce()`] on every fixture
/// (`g5_bands_of_arange_grid_equals_tfce_bitwise`, this module's tests); under
/// `strict = false` and an irregular grid it agrees with [`tfce_bands_naive`]
/// to `rel 1e-12` (`tests/tfce_oracle.rs::g6_bands_union_find_matches_naive`).
///
/// Errors: [`NeuroError::InvalidBands`] if `thresholds` and `weights` differ
/// in length, hold a non-finite value, or `thresholds` is not strictly
/// increasing; [`NeuroError::InvalidParams`] if `e` is non-finite;
/// [`NeuroError::MismatchedLengths`] if `stat.len() != domain.n_nodes()`;
/// [`NeuroError::NonFiniteStat`] on any NaN or ±∞ in `stat`.
///
/// ```
/// use neurostats::{Conn, Domain, tfce_bands};
/// let dom = Domain::from_volume([2, 2, 1], Conn::Face);
/// // Nodes 0 and 1 are face neighbours; 2 and 3 sit below both thresholds.
/// let stat = [3.0, 2.0, 0.0, 0.0];
/// let thresholds = [1.0, 2.5];
/// let weights = [1.0, 4.0];
/// let enh = tfce_bands(&dom, &stat, &thresholds, &weights, 1.0, true).unwrap();
/// // Node 0: band 0 (component {0, 1}, extent 2) + band 1 (alone, extent 1).
/// assert_eq!(enh[0], 1.0 * 2.0 + 4.0 * 1.0);
/// // Node 1 clears band 0 only; nodes 2 and 3 clear neither.
/// assert_eq!(enh[1], 2.0);
/// assert_eq!(enh[2], 0.0);
/// ```
pub fn tfce_bands(
    domain: &Domain,
    stat: &[f64],
    thresholds: &[f64],
    weights: &[f64],
    e: f64,
    strict: bool,
) -> Result<Vec<f64>, NeuroError> {
    let mut ws = TfceWorkspace::new(domain.n_nodes());
    let mut out = alloc::vec![0.0f64; domain.n_nodes()];
    tfce_bands_into(
        domain, stat, thresholds, weights, e, strict, &mut ws, &mut out,
    )?;
    Ok(out)
}

/// [`tfce_bands`] into a caller-owned `out`, reusing `ws`. Same quantity,
/// conventions and errors; `out.len() != domain.n_nodes()` is
/// [`NeuroError::MismatchedLengths`] (checked first).
#[allow(clippy::too_many_arguments)]
pub fn tfce_bands_into(
    domain: &Domain,
    stat: &[f64],
    thresholds: &[f64],
    weights: &[f64],
    e: f64,
    strict: bool,
    ws: &mut TfceWorkspace,
    out: &mut [f64],
) -> Result<(), NeuroError> {
    let n = domain.n_nodes();
    if out.len() != n {
        return Err(NeuroError::MismatchedLengths {
            expected: n,
            got: out.len(),
        });
    }
    check_bands(thresholds, weights, e)?;
    if check_stat(domain, stat)?.is_none() {
        return Ok(());
    }
    ws.ths.clear();
    ws.ths.extend_from_slice(thresholds);
    ws.cum.clear();
    ws.cum.push(0.0f64);
    for &w in weights {
        let last = *ws.cum.last().expect("non-empty");
        ws.cum.push(last + w);
    }
    sweep_bands(domain, stat, e, strict, ws, out);
    Ok(())
}

/// [`tfce_bands`] by naive per-band re-clustering — the readable oracle for
/// `tests/tfce_oracle.rs::g6_bands_union_find_matches_naive`. Same quantity,
/// conventions and errors as [`tfce_bands`]; `O(V · n_bands)`.
pub fn tfce_bands_naive(
    domain: &Domain,
    stat: &[f64],
    thresholds: &[f64],
    weights: &[f64],
    e: f64,
    strict: bool,
) -> Result<Vec<f64>, NeuroError> {
    check_bands(thresholds, weights, e)?;
    let mut out = alloc::vec![0.0f64; domain.n_nodes()];
    if check_stat(domain, stat)?.is_none() {
        return Ok(out);
    }
    let bands: Vec<(f64, f64)> = thresholds
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .collect();
    naive_sum(domain, stat, &bands, e, strict, &mut out);
    Ok(out)
}

/// The `Exact` sweep: same forest, no grid. Events are the distinct `stat`
/// values in descending order; a root's bank is paid for the height run
/// `[h_now, a]` (`a` the height of its last size change) at its current size
/// each time its size changes, and finally down to `h0 = start`. One activation per node, one pay per size
/// change: `O(V log V + V·α(V))`.
fn sweep_exact(
    domain: &Domain,
    stat: &[f64],
    params: &TfceParams,
    ws: &mut TfceWorkspace,
    out: &mut [f64],
) {
    let TfceWorkspace {
        order,
        heights,
        forest,
        ..
    } = ws;
    match domain.graph() {
        Graph::Csr {
            offsets,
            neighbours,
        } => {
            let g = CsrView {
                offsets,
                neighbours,
            };
            sweep_exact_impl(&g, stat, params, order, heights, forest, out);
        }
        Graph::Lattice { dims, conn, lin_of } => {
            let g = LatticeView::new(*dims, *conn, lin_of);
            sweep_exact_impl(&g, stat, params, order, heights, forest, out);
        }
    }
}

/// `sweep_exact` over one index space (see [`sweep_bands_impl`]).
fn sweep_exact_impl<G: Neighbours>(
    g: &G,
    stat: &[f64],
    params: &TfceParams,
    order: &mut Vec<u64>,
    heights: &mut Vec<u64>,
    uf: &mut Forest,
    out: &mut [f64],
) {
    let hp1 = params.h + 1.0;
    let h0 = params.start;
    let p0 = height_pow(h0, hp1);
    // Only nodes above `start` can ever activate.
    build_order(g, stat, |s| s > h0, order, heights);
    uf.reset(g.len());
    let d = g.max_degree();
    let (mut near, mut above) = (alloc::vec![0u32; d], alloc::vec![0u32; d]);
    for (&v, &h) in order.iter().zip(heights.iter()) {
        let v = v as usize;
        // One power per event: every settlement at this height reuses it.
        let p = height_pow(f64::from_bits(h), hp1);
        let n = uf.active_neighbours(g, v, &mut near, &mut above);
        uf.activate_exact(v, p);
        let mut rv = v; // v is its own root at activation
        for (&u, &up) in near[..n].iter().zip(&above) {
            let ru = uf.find_from(u as usize, up as usize);
            if ru != rv {
                rv = uf.link_exact(rv, ru, p, hp1, params.e);
            }
        }
    }
    for (k, o) in out.iter_mut().enumerate() {
        *o = uf.resolve(g.index_of(k), |uf, r| uf.pay_exact(r, p0, hp1, params.e));
    }
}

/// `x^{H+1}` for a height `x ≥ 0`, `hp1 = H + 1`. The usual `H = 2` is two
/// multiplications, within 1 ulp of the correctly rounded cube, in place of a
/// general `pow`; every other exponent takes `pow`.
#[inline]
fn height_pow(x: f64, hp1: f64) -> f64 {
    if hp1 == 3.0 {
        x * x * x
    } else {
        libm::pow(x, hp1)
    }
}

/// `size^E`. The usual `E = 0.5` is the correctly rounded square root, the
/// same bits `pow(size, 0.5)` returns.
#[inline]
fn extent_pow(size: u32, e: f64) -> f64 {
    if e == 0.5 {
        libm::sqrt(size as f64)
    } else {
        libm::pow(size as f64, e)
    }
}

/// A node's share of the forest that only settlements and the final read
/// touch. One cache line holds two, and none straddles a line.
#[derive(Debug, Clone, Copy)]
#[repr(align(32))]
struct Account {
    /// TFCE sum: absolute for a root, relative to its parent otherwise.
    bank: f64,
    /// Root only, `Exact`: height of the last size change, raised to `H+1`.
    since_p: f64,
    /// Root only, stepped rules: one past the band of the last size change.
    unpaid_end: u32,
    /// Root only: node count of the component.
    size: u32,
}

/// Disjoint-set forest with lazily paid per-root TFCE banks.
///
/// Invariant: for a root `r`, `bank` is the absolute TFCE sum every member
/// has earned in bands `unpaid_end..`; bands `current..unpaid_end` are still
/// unpaid and will be paid with the current `size`. For a non-root `v`,
/// `bank` is relative to its parent — a node's value is the sum of `bank`
/// along its path to the root. Linking `a` under `b` therefore sets
/// `bank[a] -= bank[b]` so `a`'s subtree does not inherit `b`'s past.
/// `unpaid_end` (band index, stepped rules) or `since_p` (height raised to
/// `H+1`, `Exact`) marks where the root's size last changed.
///
/// `parent` and `active` are all a neighbour probe or a root search reads, so
/// they stay in arrays of their own, dense enough to keep in cache — `active`
/// is one bit per index, so every probe of an inactive neighbour stays in the
/// nearest caches. The rest of a node sits together in its [`Account`].
#[derive(Debug, Clone)]
struct Forest {
    parent: Vec<u32>,
    account: Vec<Account>,
    /// Bit `v % 64` of word `v / 64` is set once `v` has activated.
    active: Vec<u64>,
    /// Scratch for iterative path compression (depth ≤ log₂ V under union by size).
    path: Vec<u32>,
}

impl Forest {
    const EMPTY: Account = Account {
        bank: 0.0,
        since_p: 0.0,
        unpaid_end: 0,
        size: 0,
    };

    fn new(n: usize) -> Self {
        Self {
            parent: alloc::vec![0; n],
            account: alloc::vec![Self::EMPTY; n],
            active: alloc::vec![0; n.div_ceil(64)],
            path: Vec::new(),
        }
    }

    /// Size every array to `n` and deactivate all nodes; other fields are
    /// overwritten by `activate` before they are read.
    fn reset(&mut self, n: usize) {
        self.parent.resize(n, 0);
        self.account.resize(n, Self::EMPTY);
        self.active.clear();
        self.active.resize(n.div_ceil(64), 0);
    }

    #[inline]
    fn is_active(&self, v: usize) -> bool {
        self.active[v / 64] >> (v % 64) & 1 != 0
    }

    #[inline]
    fn set_active(&mut self, v: usize) {
        self.active[v / 64] |= 1 << (v % 64);
    }

    fn activate(&mut self, v: usize, band: usize) {
        self.set_active(v);
        self.parent[v] = v as u32;
        self.account[v] = Account {
            bank: 0.0,
            since_p: 0.0,
            unpaid_end: band as u32 + 1,
            size: 1,
        };
    }

    /// Pay root `r` for bands `lo..unpaid_end` at its current size (callers
    /// pass `lo = current + 1` on a merge, `lo = 0` at the end). Paying a
    /// root twice down to the same `lo` pays nothing the second time.
    fn pay(&mut self, r: usize, lo: usize, cum: &[f64], e: f64) {
        let a = &mut self.account[r];
        let end = a.unpaid_end as usize;
        if end > lo {
            let wsum = cum[end] - cum[lo];
            a.bank += wsum * extent_pow(a.size, e);
        }
        a.unpaid_end = lo as u32;
    }

    /// The active neighbours of `v` into `near` and each one's parent into
    /// `above`, in neighbour order; returns their count. Collecting them
    /// before the first merge costs no branch per neighbour, and the parent
    /// reads do not depend on one another, so their cache misses overlap
    /// instead of queueing behind each merge.
    fn active_neighbours<G: Neighbours>(
        &self,
        g: &G,
        v: usize,
        near: &mut [u32],
        above: &mut [u32],
    ) -> usize {
        let mut n = 0usize;
        g.for_each(v, |u| {
            near[n] = u as u32;
            n += self.is_active(u) as usize;
        });
        for (a, &u) in above.iter_mut().zip(&near[..n]) {
            *a = self.parent[u as usize];
        }
        n
    }

    /// [`Forest::find`] of `v` given `p`, its parent as
    /// [`Forest::active_neighbours`] read it earlier in the same event. If
    /// `p` is a root now it was one then, no merge or compression since has
    /// moved `v` off it, and it is the answer; otherwise the forest changed
    /// under the read and the search starts over.
    fn find_from(&mut self, v: usize, p: usize) -> usize {
        if self.parent[p] as usize == p {
            return p;
        }
        self.find(v)
    }

    /// Root of `v` with offset-preserving path compression.
    fn find(&mut self, v: usize) -> usize {
        // A root, or a child of one, has nothing to compress: the usual case
        // once paths have been compressed.
        let p = self.parent[v] as usize;
        let mut r = self.parent[p] as usize;
        if r == p {
            return p;
        }
        while self.parent[r] as usize != r {
            r = self.parent[r] as usize;
        }
        // Compress top-down so each node's parent is already root-relative.
        self.path.clear();
        let mut x = v;
        while self.parent[x] as usize != r && x != r {
            self.path.push(x as u32);
            x = self.parent[x] as usize;
        }
        for k in (0..self.path.len()).rev() {
            let u = self.path[k] as usize;
            let p = self.parent[u] as usize;
            if p != r {
                self.account[u].bank += self.account[p].bank;
                self.parent[u] = r as u32;
            }
        }
        r
    }

    /// Attach root `small` under root `big`, both settled to the current
    /// event; the caller then records the event as `big`'s last size change.
    fn attach(&mut self, small: usize, big: usize) {
        self.parent[small] = big as u32;
        self.account[small].bank -= self.account[big].bank;
        self.account[big].size += self.account[small].size;
    }

    /// Merge two known roots `ra`, `rb` during band `band`; returns the
    /// surviving root. Ties (equal size) send `rb` under `ra`, so the caller's
    /// running root wins deterministically.
    fn link(&mut self, ra: usize, rb: usize, band: usize, cum: &[f64], e: f64) -> usize {
        // Both roots' sizes change now: settle the bands above `band` first.
        self.pay(ra, band + 1, cum, e);
        self.pay(rb, band + 1, cum, e);
        let (small, big) = if self.account[ra].size < self.account[rb].size {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.attach(small, big);
        self.account[big].unpaid_end = band as u32 + 1;
        big
    }

    /// `p` is the activation height raised to `H+1`.
    fn activate_exact(&mut self, v: usize, p: f64) {
        self.set_active(v);
        self.parent[v] = v as u32;
        self.account[v] = Account {
            bank: 0.0,
            since_p: p,
            unpaid_end: 0,
            size: 1,
        };
    }

    /// Pay root `r` for the height run down to the current event at its
    /// current size: `size^E · (since_p − p_now) / (H+1)`, both heights
    /// already raised to `H+1`. Heights are non-negative, so the powers order
    /// as the heights do; two heights whose powers round equal pay zero.
    fn pay_exact(&mut self, r: usize, p_now: f64, hp1: f64, e: f64) {
        let a = &mut self.account[r];
        if a.since_p > p_now {
            let w = (a.since_p - p_now) / hp1;
            a.bank += w * extent_pow(a.size, e);
        }
        a.since_p = p_now;
    }

    /// Merge two known roots `ra`, `rb` at an event whose height raised to
    /// `H+1` is `p_now`; returns the surviving root (tie-breaking as
    /// [`Forest::link`]).
    fn link_exact(&mut self, ra: usize, rb: usize, p_now: f64, hp1: f64, e: f64) -> usize {
        self.pay_exact(ra, p_now, hp1, e);
        self.pay_exact(rb, p_now, hp1, e);
        let (small, big) = if self.account[ra].size < self.account[rb].size {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.attach(small, big);
        self.account[big].since_p = p_now;
        big
    }

    /// Node `v`'s TFCE value once the sweep is done: its own bank plus its
    /// root's (the bank of a non-root is relative to its parent, and `find`
    /// has just compressed the path to the root). `settle` pays a root down
    /// to the end of the sweep; it runs on every read of a member, and every
    /// call after a root's first pays nothing.
    fn resolve<S: Fn(&mut Self, usize)>(&mut self, v: usize, settle: S) -> f64 {
        if !self.is_active(v) {
            return 0.0;
        }
        let r = self.find(v);
        settle(self, r);
        if r == v {
            self.account[v].bank
        } else {
            self.account[v].bank + self.account[r].bank
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `build_order` over a path graph of `vals.len()` nodes, as `(node, stat)`.
    fn ordered(
        vals: &[f64],
        keep: impl Fn(f64) -> bool,
        order: &mut Vec<u64>,
        heights: &mut Vec<u64>,
    ) -> Vec<(usize, f64)> {
        let offsets: Vec<u32> = (0..=vals.len() as u32).collect();
        let g = CsrView {
            offsets: &offsets,
            neighbours: &[],
        };
        build_order(&g, vals, keep, order, heights);
        order
            .iter()
            .zip(heights.iter())
            .map(|(&v, &h)| (v as usize, f64::from_bits(h)))
            .collect()
    }

    #[test]
    fn build_order_matches_comparison_sort() {
        // Negatives, zeros of both signs and repeats exercise ties and the
        // sign-bit branch of `sort_key`; the `1.0 + k·ε` block shares the top
        // half of its key, so only the per-run sort can order it; subnormals
        // and far-apart exponents exercise every radix digit.
        let mut vals: Vec<f64> = alloc::vec![
            0.0, -0.0, 1.5, -1.5, 3.0, -3.0, 1.5, 0.25, -0.25, 2.0, -2.0, 0.0, 7.5, -7.5, 1e-300,
            -1e-300, 1e300, -1e300, 42.0, 42.0, 5e-324, 1e-310, -5e-324,
        ];
        for k in [7u64, 2, 9, 2, 0, 31, 5, 31, 1] {
            vals.push(f64::from_bits(1.0f64.to_bits() + k));
        }
        for keep_all in [true, false] {
            let keep = |s: f64| keep_all || s > 0.25;
            let got = ordered(&vals, keep, &mut Vec::new(), &mut alloc::vec![0u64; 3]);

            // Stable comparison sort: equal keys stay in node order.
            let mut want: Vec<usize> = (0..vals.len()).filter(|&k| keep(vals[k])).collect();
            want.sort_by(|&a, &b| sort_key(vals[b]).cmp(&sort_key(vals[a])));
            assert_eq!(got.len(), want.len());
            for (&(node, stat), &k) in got.iter().zip(&want) {
                assert_eq!(node, k);
                assert_eq!(stat.to_bits(), vals[k].to_bits());
            }
        }
    }

    #[test]
    fn build_order_reuses_longer_scratch() {
        // A second, shorter map must not see entries left by the first.
        let (mut order, mut heights) = (Vec::new(), Vec::new());
        let long: Vec<f64> = (0..300).map(|i| ((i * 37) % 101) as f64 * 0.5).collect();
        ordered(&long, |_| true, &mut order, &mut heights);
        let short = [2.0, 9.0, 1e-3, 9.0, 4.0];
        let got = ordered(&short, |_| true, &mut order, &mut heights);
        assert_eq!(
            got,
            alloc::vec![(1, 9.0), (3, 9.0), (4, 4.0), (0, 2.0), (2, 1e-3)]
        );
    }

    #[test]
    fn sort_key_is_order_preserving() {
        let vals = [
            f64::NEG_INFINITY,
            -1e300,
            -1.0,
            -0.0,
            0.0,
            1.0,
            1e300,
            f64::INFINITY,
        ];
        for w in vals.windows(2) {
            assert!(sort_key(w[0]) < sort_key(w[1]));
        }
    }

    #[test]
    fn arange_matches_numpy_rule() {
        // np.arange(0.4, 2.0, 0.4) -> [0.4, 0.8, 1.2, 1.6]
        let t = arange(0.4, 2.0, 0.4);
        assert_eq!(t.len(), 4);
        assert_eq!(t[0], 0.4);
        assert_eq!(t[1], 0.8);
        assert!(arange(2.0, 1.0, 0.5).is_empty());
        assert!(arange(1.0, 1.0, 0.5).is_empty());
    }

    #[test]
    fn empty_and_flat_maps() {
        let dom = Domain::from_volume([0, 1, 1], crate::Conn::Face);
        let p = TfceParams {
            e: 0.5,
            h: 2.0,
            start: 0.4,
            step: 0.4,
            weighting: Weighting::MneStep,
        };
        assert!(tfce(&dom, &[], &p).unwrap().is_empty());
        // start above max: every node 0.
        let dom = Domain::from_volume([2, 2, 1], crate::Conn::Face);
        let z = tfce(&dom, &[0.1, 0.2, 0.3, 0.35], &p).unwrap();
        assert_eq!(z, alloc::vec![0.0; 4]);
    }

    #[test]
    fn exact_rejects_h_le_minus_one_and_negative_start() {
        let dom = Domain::from_volume([2, 1, 1], crate::Conn::Face);
        let ok = TfceParams {
            e: 0.5,
            h: 2.0,
            start: 0.0,
            step: 1.0,
            weighting: Weighting::Exact,
        };
        assert!(tfce(&dom, &[1.0, 2.0], &ok).is_ok());
        let bad_h = TfceParams { h: -1.0, ..ok };
        assert_eq!(
            tfce(&dom, &[1.0, 2.0], &bad_h),
            Err(NeuroError::InvalidParams)
        );
        let bad_start = TfceParams { start: -0.5, ..ok };
        assert_eq!(
            tfce(&dom, &[1.0, 2.0], &bad_start),
            Err(NeuroError::InvalidParams)
        );
        // step is unused by Exact but still validated.
        let bad_step = TfceParams { step: 0.0, ..ok };
        assert_eq!(
            tfce(&dom, &[1.0, 2.0], &bad_step),
            Err(NeuroError::InvalidParams)
        );
    }

    // Two nodes on a line, stat = [3, 1], E = 1, H = 0 (so TFCE = ∫ size dh), h0 = 0:
    // both nodes form a 2-component on (0,1] and node 0 is alone on (1,3]:
    // node 0 → 2·1 + 1·2 = 4, node 1 → 2·1 = 2.
    #[test]
    fn exact_closed_form_hand_case() {
        let dom = Domain::from_volume([2, 1, 1], crate::Conn::Face);
        let p = TfceParams {
            e: 1.0,
            h: 0.0,
            start: 0.0,
            step: 1.0,
            weighting: Weighting::Exact,
        };
        let fast = tfce(&dom, &[3.0, 1.0], &p).unwrap();
        let slow = tfce_naive(&dom, &[3.0, 1.0], &p).unwrap();
        assert!(
            (fast[0] - 4.0).abs() < 1e-12 && (fast[1] - 2.0).abs() < 1e-12,
            "{fast:?}"
        );
        assert!(
            (slow[0] - 4.0).abs() < 1e-12 && (slow[1] - 2.0).abs() < 1e-12,
            "{slow:?}"
        );
    }

    /// G5 — `tfce_bands` with the arange grid and `band_weight` weights is
    /// bit-equal to `tfce` on every frozen fixture (same sweep, same order).
    #[test]
    fn g5_bands_of_arange_grid_equals_tfce_bitwise() {
        #[derive(serde::Deserialize)]
        struct Fx {
            dims: [usize; 3],
            conn: u32,
            mask: std::string::String,
            stat: Vec<f64>,
            params: FxParams,
        }
        #[derive(serde::Deserialize)]
        struct FxParams {
            e: f64,
            h: f64,
            start: f64,
            step: f64,
        }
        for name in [
            "grid4",
            "blob",
            "ties",
            "flat",
            "single",
            "empty_grid",
            "two_clusters",
            "bench",
        ] {
            let path = std::format!(
                "{}/tests/fixtures/tfce_{name}.json",
                env!("CARGO_MANIFEST_DIR")
            );
            let fx: Fx = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let mask: Vec<bool> = fx.mask.bytes().map(|b| b == b'1').collect();
            let conn = match fx.conn {
                6 => crate::Conn::Face,
                18 => crate::Conn::Edge,
                _ => crate::Conn::Vertex,
            };
            let dom = Domain::from_mask(&mask, fx.dims, conn).unwrap();
            let max = fx.stat.iter().cloned().fold(f64::MIN, f64::max);
            for w in [Weighting::MneStep, Weighting::SmithNichols] {
                let p = TfceParams {
                    e: fx.params.e,
                    h: fx.params.h,
                    start: fx.params.start,
                    step: fx.params.step,
                    weighting: w,
                };
                let ths = arange(p.start, max, p.step);
                let wts: Vec<f64> = (0..ths.len()).map(|i| band_weight(i, &ths, &p)).collect();
                let a = tfce(&dom, &fx.stat, &p).unwrap();
                let b = tfce_bands(&dom, &fx.stat, &ths, &wts, p.e, true).unwrap();
                assert_eq!(a, b, "G5 {name} {w:?}");
            }
        }
    }

    // Two nodes on a line, stat = [100, 50], E = 1, H = 2. The grid is
    // h_k = k for k = 1..=100. Node 1 is a member of bands 1..=50 (`>=`: band
    // 50 sits exactly on its value) and node 0 of all 100 (band 100 sits on
    // the maximum). Bands 1..=50 hold both nodes, bands 51..=100 node 0 alone:
    // node 1 → 2·Σ_{k≤50} k² = 85850, node 0 → 85850 + Σ_{51≤k≤100} k² = 381275.
    // Under strict membership the values would be 80850 and 368775.
    #[test]
    fn nilearn_hand_case_counts_ties_and_the_maximum() {
        let dom = Domain::from_volume([2, 1, 1], crate::Conn::Face);
        let p = TfceParams {
            e: 1.0,
            h: 2.0,
            start: 0.0,
            step: 1.0,
            weighting: Weighting::Nilearn,
        };
        for got in [
            tfce(&dom, &[100.0, 50.0], &p).unwrap(),
            tfce_naive(&dom, &[100.0, 50.0], &p).unwrap(),
        ] {
            assert!(
                (got[0] - 381275.0).abs() < 1e-7 && (got[1] - 85850.0).abs() < 1e-7,
                "{got:?}"
            );
        }
        // A flat map: every node sits on the maximum, so all three are in all
        // 100 bands. E = 1, H = 0 → 100 · 3.
        let line = Domain::from_volume([3, 1, 1], crate::Conn::Face);
        let flat = TfceParams { h: 0.0, ..p };
        assert_eq!(
            tfce(&line, &[7.0, 7.0, 7.0], &flat).unwrap(),
            alloc::vec![300.0; 3]
        );
    }

    #[test]
    fn nilearn_thresholds_follow_numpy_linspace() {
        // np.linspace(0, 3.7, 101)[1:] — first, a middle one, the last two.
        let mut t = Vec::new();
        nilearn_thresholds_into(3.7, &mut t);
        assert_eq!(t.len(), 100);
        assert_eq!(t[0], 0.037000000000000005);
        assert_eq!(t[49], 1.8500000000000003);
        assert_eq!(t[98], 3.6630000000000007);
        assert_eq!(t[99], 3.7);
        assert!(t.windows(2).all(|w| w[0] < w[1]));
    }

    // No positive value, or a maximum too small for a strictly increasing
    // grid: every node 0, and `start` / `step` change nothing.
    #[test]
    fn nilearn_degenerate_maps_are_zero() {
        let dom = Domain::from_volume([3, 1, 1], crate::Conn::Face);
        let p = TfceParams {
            e: 0.5,
            h: 2.0,
            start: 0.0,
            step: 1.0,
            weighting: Weighting::Nilearn,
        };
        for stat in [
            [-1.0, -2.0, -0.5],
            [0.0, 0.0, 0.0],
            [0.0, -3.0, 0.0],
            [1e-320, 5e-321, 0.0],
        ] {
            assert_eq!(tfce(&dom, &stat, &p).unwrap(), alloc::vec![0.0; 3]);
            assert_eq!(tfce_naive(&dom, &stat, &p).unwrap(), alloc::vec![0.0; 3]);
        }
        let other = TfceParams {
            start: 7.5,
            step: 0.01,
            ..p
        };
        let stat = [2.0, 1.0, 3.0];
        assert_eq!(
            tfce(&dom, &stat, &p).unwrap(),
            tfce(&dom, &stat, &other).unwrap()
        );
    }

    #[test]
    fn rejects_bad_inputs() {
        let dom = Domain::from_volume([2, 1, 1], crate::Conn::Face);
        let p = TfceParams {
            e: 0.5,
            h: 2.0,
            start: 0.4,
            step: 0.4,
            weighting: Weighting::MneStep,
        };
        assert_eq!(
            tfce(&dom, &[1.0], &p),
            Err(NeuroError::MismatchedLengths {
                expected: 2,
                got: 1
            })
        );
        assert_eq!(
            tfce(&dom, &[1.0, f64::NAN], &p),
            Err(NeuroError::NonFiniteStat)
        );
        let bad = TfceParams { step: 0.0, ..p };
        assert_eq!(
            tfce(&dom, &[1.0, 2.0], &bad),
            Err(NeuroError::InvalidParams)
        );
    }
}
