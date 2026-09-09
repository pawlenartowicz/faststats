//! Threshold-free cluster enhancement (TFCE) over a [`Domain`].
//!
//! Two implementations of one sum: stepped for [`Weighting::SmithNichols`] /
//! [`Weighting::MneStep`], closed-form integral for [`Weighting::Exact`] (see
//! its docs). [`tfce_naive`] re-labels connected components at every band
//! (the readable oracle, a transcription of MNE-Python's loop for the stepped
//! rules), and [`tfce()`] sweeps the bands once with an incremental
//! union-find (the product path). Both take the same [`TfceParams`] and agree
//! within `1e-12` relative on every fixture in `tests/fixtures/tfce_*.json`.

use alloc::vec::Vec;

use crate::domain::{Domain, Graph, conn_offsets};
use crate::error::NeuroError;

/// Weight `w_i` given to the threshold band ending at `h_i`. The reference
/// tools disagree here by several × on the same map, so the choice is a
/// required parameter, never a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weighting {
    /// Smith & Nichols (2009) stepped integral: `w_i = h_i^h · step`. The rule
    /// FSL `fslmaths -tfce`, PALM, and the SPM TFCE toolbox implement. No local
    /// oracle yet — checked against an independent NumPy transcription only.
    SmithNichols,
    /// MNE-Python (`mne.stats.cluster_level._find_clusters`, ≤ 1.12.1):
    /// `w_0 = |h_0|^h`, `w_i = |h_i − h_{i−1}|^h` for `i ≥ 1` — the band
    /// *height* raised to `h`, with no `· step` factor and no threshold value.
    /// Matches `tests/fixtures/tfce_*.json` `expected_mne`.
    MneStep,
    /// Closed-form Smith–Nichols integral `∫_{h0}^{stat[v]} e_v(h)^E · h^H dh`
    /// with `h0 = start` — no threshold grid, no `step` (still validated,
    /// pass any positive value). `e_v(h)` is piecewise constant between
    /// distinct `stat` values, so each run of constant component size `s` over
    /// `[a, b)` contributes `s^E · (b^{H+1} − a^{H+1}) / (H+1)`. The
    /// recommended choice for new analyses (`start = 0`); `SmithNichols` /
    /// `MneStep` exist for parity with FSL/PALM and MNE pipelines. Same quantity
    /// as Gaser's SPM TFCE toolbox (`tfce_maxtree`) and eTFCE. Requires
    /// `h > −1` and `start ≥ 0` ([`NeuroError::InvalidParams`]). Nodes with
    /// `stat ≤ start` get 0. No external oracle: validated union-find vs naive
    /// (`rel 1e-12`) and as the `step → 0` limit of `SmithNichols`.
    Exact,
}

/// TFCE parameters. Under the stepped rules, thresholds follow NumPy
/// `arange(start, max(stat), step)` (start inclusive, stop exclusive), so the
/// largest node clears every band; [`Weighting::Exact`] uses no grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TfceParams {
    /// Extent exponent `E`. Smith–Nichols / MNE default `0.5`.
    pub e: f64,
    /// Height exponent `H`. Smith–Nichols / MNE default `2.0`. `> −1` under
    /// [`Weighting::Exact`].
    pub h: f64,
    /// First threshold `h_0`; nodes with `stat <= start` get 0. Finite;
    /// `≥ 0` under [`Weighting::Exact`].
    pub start: f64,
    /// Threshold spacing; must be `> 0` and finite. Unused by
    /// [`Weighting::Exact`] (pass any positive value).
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

/// Weight `w_i` of band `i` under `params.weighting` (see [`Weighting`]).
/// Only called for the stepped rules — [`Weighting::Exact`] builds its bands
/// with the closed-form weight instead (see [`tfce_naive`], [`sweep_exact`]).
fn band_weight(i: usize, ths: &[f64], params: &TfceParams) -> f64 {
    let h = ths[i];
    match params.weighting {
        Weighting::SmithNichols => libm::pow(h, params.h) * params.step,
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
/// `params.weighting`. Positive tail only. Summation is ascending in `i`, one
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
    // (threshold, weight) bands: membership is `stat > threshold`, contribution
    // `weight · extent^E`. Stepped rules take the `arange` grid; Exact takes the
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
        _ => {
            let ths = arange(params.start, max, params.step);
            (0..ths.len())
                .map(|i| (ths[i], band_weight(i, &ths, params)))
                .collect()
        }
    };

    naive_sum(domain, stat, &bands, params.e, true, &mut out);
    Ok(out)
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
/// membership, NumPy `arange` thresholds under the stepped rules,
/// `params.weighting` band weights, positive tail), computed in one
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

/// Scratch buffers for [`tfce_into`]: the sweep order, the disjoint-set forest,
/// the threshold grid and its cumulative band weights, and — for lattice
/// domains only — the voxel-indexed input and output the sweep runs over.
/// Grown on demand (the threshold count follows `max(stat)`), never shrunk, so
/// a permutation loop that reuses one workspace reallocates only when a draw
/// needs a longer threshold grid than any earlier one (stepped rules); never
/// under [`Weighting::Exact`].
///
/// A lattice sweep works in voxel-index space, so the forest and the two
/// voxel buffers are sized by the grid, not by the node count: on a mask that
/// fills a small part of its grid a workspace costs several times what the
/// node-indexed CSR path did.
#[derive(Debug, Clone)]
pub struct TfceWorkspace {
    order: Vec<u32>,
    /// Radix-sort scratch: the `u64` sort keys of `order` and the two
    /// ping-pong halves the eight passes swap through.
    keys: Vec<u64>,
    keys_tmp: Vec<u64>,
    order_tmp: Vec<u32>,
    forest: Forest,
    ths: Vec<f64>,
    cum: Vec<f64>,
    /// `stat` scattered to voxel index, `-inf` outside the mask. Empty for CSR.
    stat_v: Vec<f64>,
    /// The sweep's voxel-indexed output, gathered back to node order. Empty for CSR.
    out_v: Vec<f64>,
}

impl TfceWorkspace {
    /// Workspace pre-sized for `n_nodes` nodes (any size works; buffers grow to
    /// fit the map actually passed to [`tfce_into`]).
    pub fn new(n_nodes: usize) -> Self {
        Self {
            order: Vec::with_capacity(n_nodes),
            keys: Vec::new(),
            keys_tmp: Vec::new(),
            order_tmp: Vec::new(),
            forest: Forest::new(n_nodes),
            ths: Vec::new(),
            cum: Vec::new(),
            stat_v: Vec::new(),
            out_v: Vec::new(),
        }
    }
}

/// [`tfce()`] into a caller-owned `out`, reusing `ws` — the allocation-free
/// entry point for permutation loops. Same quantity, conventions, tolerances,
/// and errors as [`tfce()`], bit-identical to it.
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
    if params.weighting == Weighting::Exact {
        sweep_exact(domain, stat, params, ws, out);
        return Ok(());
    }
    arange_into(params.start, max, params.step, &mut ws.ths);
    ws.cum.clear();
    ws.cum.push(0.0f64);
    for i in 0..ws.ths.len() {
        let last = *ws.cum.last().expect("non-empty");
        ws.cum.push(last + band_weight(i, &ws.ths, params));
    }
    sweep_bands(domain, stat, params.e, true, ws, out);
    Ok(())
}

/// The index space a sweep runs over, and the neighbour relation on it. A CSR
/// domain sweeps node indices; a lattice sweeps linear voxel indices, where a
/// neighbour is an arithmetic step instead of a random read into a 26-column
/// table — the sweep is memory-bound and that table is its dominant cost.
trait Neighbours {
    /// Size of the index space the forest is allocated over.
    fn len(&self) -> usize;
    /// Number of nodes, i.e. the length of `stat` and `out` in node order.
    fn n_nodes(&self) -> usize;
    /// Index-space index of node `i` (the identity for a CSR domain).
    fn index_of(&self, i: usize) -> usize;
    /// Calls `f(u)` for every neighbour `u` of `v`, both in this index space.
    fn for_each<F: FnMut(usize)>(&self, v: usize, f: F);
}

struct CsrView<'a> {
    offsets: &'a [u32],
    neighbours: &'a [u32],
}

impl Neighbours for CsrView<'_> {
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn n_nodes(&self) -> usize {
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

    fn n_nodes(&self) -> usize {
        self.lin_of.len()
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
}

/// One descending union-find sweep over explicit bands, on whichever index
/// space `domain` prefers. `ws.ths` strictly increasing,
/// `ws.cum[k] = Σ_{j<k} w_j` (`len == ths.len() + 1`). Membership per `strict`.
/// Shared by `tfce_into` (arange grid, strict) and `tfce_bands_into` (caller
/// grid, either rule).
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
        keys,
        keys_tmp,
        order_tmp,
        forest,
        ths,
        cum,
        stat_v,
        out_v,
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
            sweep_bands_impl(
                &g, stat, ths, cum, e, strict, order, keys, keys_tmp, order_tmp, forest, out,
            );
        }
        Graph::Lattice { dims, conn, lin_of } => {
            let g = LatticeView::new(*dims, *conn, lin_of);
            scatter(stat, lin_of, g.len(), stat_v);
            out_v.resize(g.len(), 0.0);
            sweep_bands_impl(
                &g, stat_v, ths, cum, e, strict, order, keys, keys_tmp, order_tmp, forest, out_v,
            );
            gather(out_v, lin_of, out);
        }
    }
}

/// `stat` (node order) into voxel order, `-inf` outside the mask so an
/// out-of-mask voxel is a member of no band under either rule.
fn scatter(stat: &[f64], lin_of: &[u32], n_vox: usize, stat_v: &mut Vec<f64>) {
    stat_v.clear();
    stat_v.resize(n_vox, f64::NEG_INFINITY);
    for (i, &s) in stat.iter().enumerate() {
        stat_v[lin_of[i] as usize] = s;
    }
}

/// The sweep's voxel-order output back to node order. Reads only in-mask
/// entries, every one of which the resolve loop has written.
fn gather(out_v: &[f64], lin_of: &[u32], out: &mut [f64]) {
    for (o, &l) in out.iter_mut().zip(lin_of) {
        *o = out_v[l as usize];
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

/// Stable LSD radix sort of the `(keys, order)` pairs by key, descending.
/// Eight one-byte passes: a byte histogram is 256 counters, which stays in L1,
/// and eight of them cover the whole `u64` key. Descending order comes from
/// prefix-summing each histogram from byte value 255 down. The pass count is
/// even, so after the last swap the sorted data is back in `keys` / `order`.
///
/// Stability matters: equal keys are equal stats, and keeping them in index
/// order makes the activation sequence among ties deterministic.
fn radix_sort_desc(
    keys: &mut Vec<u64>,
    order: &mut Vec<u32>,
    keys_tmp: &mut Vec<u64>,
    order_tmp: &mut Vec<u32>,
) {
    debug_assert_eq!(keys.len(), order.len());
    let n = keys.len();
    keys_tmp.clear();
    keys_tmp.resize(n, 0);
    order_tmp.clear();
    order_tmp.resize(n, 0);
    for pass in 0..8 {
        let shift = pass * 8;
        let mut count = [0usize; 256];
        for &k in keys.iter() {
            count[((k >> shift) & 0xff) as usize] += 1;
        }
        let mut start = [0usize; 256];
        let mut acc = 0usize;
        for (s, &c) in start.iter_mut().zip(count.iter()).rev() {
            *s = acc;
            acc += c;
        }
        for (&k, &o) in keys.iter().zip(order.iter()) {
            let b = ((k >> shift) & 0xff) as usize;
            keys_tmp[start[b]] = k;
            order_tmp[start[b]] = o;
            start[b] += 1;
        }
        core::mem::swap(keys, keys_tmp);
        core::mem::swap(order, order_tmp);
    }
}

/// Fill `order` with the indices whose stat `keep` accepts, sorted by stat
/// descending. The sweep stops at the first non-member anyway, so dropping the
/// rest changes no output and shortens the sort to the activating count.
fn build_order<F: Fn(f64) -> bool>(
    stat: &[f64],
    keep: F,
    order: &mut Vec<u32>,
    keys: &mut Vec<u64>,
    keys_tmp: &mut Vec<u64>,
    order_tmp: &mut Vec<u32>,
) {
    order.clear();
    keys.clear();
    for (v, &s) in stat.iter().enumerate() {
        if keep(s) {
            order.push(v as u32);
            keys.push(sort_key(s));
        }
    }
    radix_sort_desc(keys, order, keys_tmp, order_tmp);
}

/// `sweep_bands` over one index space. `stat` and `out` are indexed by it and
/// have length `g.len()`.
#[allow(clippy::too_many_arguments)]
fn sweep_bands_impl<G: Neighbours>(
    g: &G,
    stat: &[f64],
    ths: &[f64],
    cum: &[f64],
    e: f64,
    strict: bool,
    order: &mut Vec<u32>,
    keys: &mut Vec<u64>,
    keys_tmp: &mut Vec<u64>,
    order_tmp: &mut Vec<u32>,
    uf: &mut Forest,
    out: &mut [f64],
) {
    let n = g.len();
    let nt = ths.len();
    if nt == 0 {
        out.fill(0.0);
        return;
    }
    // Only indices that clear the lowest band can ever activate.
    build_order(
        stat,
        |s| member(s, ths[0], strict),
        order,
        keys,
        keys_tmp,
        order_tmp,
    );
    uf.reset(n);
    let mut ptr = 0usize;
    for i in (0..nt).rev() {
        let h = ths[i];
        while ptr < order.len() && member(stat[order[ptr] as usize], h, strict) {
            let v = order[ptr] as usize;
            ptr += 1;
            uf.activate(v, i);
            let mut rv = v; // v is its own root at activation
            g.for_each(v, |u| {
                if !uf.active[u] {
                    return;
                }
                let ru = uf.find(u);
                if ru == rv {
                    return;
                }
                rv = uf.link(rv, ru, i, cum, e);
            });
        }
    }
    // Walk the nodes, not the index space: on a sparse mask the voxel-count
    // walk would cost several times the node count for nothing.
    for k in 0..g.n_nodes() {
        let v = g.index_of(k);
        if uf.active[v] && uf.parent[v] as usize == v {
            uf.pay(v, 0, cum, e);
        }
    }
    for k in 0..g.n_nodes() {
        let v = g.index_of(k);
        out[v] = resolve(uf, v);
    }
}

/// A node's TFCE value once the sweep is done: its own bank plus its root's
/// (the bank of a non-root is relative to its parent, and `find` has just
/// compressed the path to the root).
fn resolve(uf: &mut Forest, v: usize) -> f64 {
    if !uf.active[v] {
        return 0.0;
    }
    let r = uf.find(v);
    if r == v {
        uf.bank[v]
    } else {
        uf.bank[v] + uf.bank[r]
    }
}

/// TFCE over an explicit band list — the low-level operator behind the Python
/// grid weightings and the nilearn shim.
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
/// [`tfce()`] (unit gate G5, every fixture); under `strict = false` and an
/// irregular grid it agrees with [`tfce_bands_naive`] to `rel 1e-12` (G6).
///
/// Errors: [`NeuroError::InvalidBands`], [`NeuroError::InvalidParams`] (`e`
/// non-finite), [`NeuroError::MismatchedLengths`], [`NeuroError::NonFiniteStat`].
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
/// G6. Same conventions and errors; `O(V · n_bands)`.
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
/// `[h_now, since_h[r]]` at its current size each time its size changes, and
/// finally down to `h0 = start`. One activation per node, one pay per size
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
        keys,
        keys_tmp,
        order_tmp,
        forest,
        stat_v,
        out_v,
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
            sweep_exact_impl(
                &g, stat, params, order, keys, keys_tmp, order_tmp, forest, out,
            );
        }
        Graph::Lattice { dims, conn, lin_of } => {
            let g = LatticeView::new(*dims, *conn, lin_of);
            scatter(stat, lin_of, g.len(), stat_v);
            out_v.resize(g.len(), 0.0);
            sweep_exact_impl(
                &g, stat_v, params, order, keys, keys_tmp, order_tmp, forest, out_v,
            );
            gather(out_v, lin_of, out);
        }
    }
}

/// `sweep_exact` over one index space (see [`sweep_bands_impl`]).
#[allow(clippy::too_many_arguments)]
fn sweep_exact_impl<G: Neighbours>(
    g: &G,
    stat: &[f64],
    params: &TfceParams,
    order: &mut Vec<u32>,
    keys: &mut Vec<u64>,
    keys_tmp: &mut Vec<u64>,
    order_tmp: &mut Vec<u32>,
    uf: &mut Forest,
    out: &mut [f64],
) {
    let n = g.len();
    let h0 = params.start;
    let hp1 = params.h + 1.0;
    // Only indices above `start` can ever activate.
    build_order(stat, |s| s > h0, order, keys, keys_tmp, order_tmp);
    uf.reset(n);
    for &v32 in &*order {
        let v = v32 as usize;
        let h = stat[v];
        uf.activate_exact(v, h);
        let mut rv = v; // v is its own root at activation
        g.for_each(v, |u| {
            if !uf.active[u] {
                return;
            }
            let ru = uf.find(u);
            if ru == rv {
                return;
            }
            rv = uf.link_exact(rv, ru, h, hp1, params.e);
        });
    }
    for k in 0..g.n_nodes() {
        let v = g.index_of(k);
        if uf.active[v] && uf.parent[v] as usize == v {
            uf.pay_exact(v, h0, hp1, params.e);
        }
    }
    for k in 0..g.n_nodes() {
        let v = g.index_of(k);
        out[v] = resolve(uf, v);
    }
}

/// Disjoint-set forest with lazily paid per-root TFCE banks.
///
/// Invariant: for a root `r`, `bank[r]` is the absolute TFCE sum every member
/// has earned through band `since[r] + 1`; bands `since[r]..=current` are still
/// unpaid and will be paid with the current `size[r]`. For a non-root `v`,
/// `bank[v]` is relative to its parent — a node's value is the sum of `bank`
/// along its path to the root. Linking `a` under `b` therefore sets
/// `bank[a] -= bank[b]` so `a`'s subtree does not inherit `b`'s past.
/// `since[r]` (band index, stepped rules) or `since_h[r]` (height, `Exact`)
/// marks where the root's size last changed.
#[derive(Debug, Clone)]
struct Forest {
    parent: Vec<u32>,
    size: Vec<u32>,
    bank: Vec<f64>,
    since: Vec<u32>,
    since_h: Vec<f64>,
    active: Vec<bool>,
    /// Scratch for iterative path compression (depth ≤ log₂ V under union by size).
    path: Vec<u32>,
}

impl Forest {
    fn new(n: usize) -> Self {
        Self {
            parent: alloc::vec![0; n],
            size: alloc::vec![0; n],
            bank: alloc::vec![0.0; n],
            since: alloc::vec![0; n],
            since_h: alloc::vec![0.0; n],
            active: alloc::vec![false; n],
            path: Vec::new(),
        }
    }

    /// Size every array to `n` and deactivate all nodes; other fields are
    /// overwritten by `activate` before they are read.
    fn reset(&mut self, n: usize) {
        self.parent.resize(n, 0);
        self.size.resize(n, 0);
        self.bank.resize(n, 0.0);
        self.since.resize(n, 0);
        self.since_h.resize(n, 0.0);
        self.active.clear();
        self.active.resize(n, false);
    }

    fn activate(&mut self, v: usize, band: usize) {
        self.active[v] = true;
        self.parent[v] = v as u32;
        self.size[v] = 1;
        self.bank[v] = 0.0;
        self.since[v] = band as u32;
    }

    /// Pay root `r` for bands `lo..=since[r]` at its current size, then mark
    /// `lo − 1` as the next unpaid band (callers pass `lo = current + 1` on a
    /// merge, `lo = 0` at the end).
    fn pay(&mut self, r: usize, lo: usize, cum: &[f64], e: f64) {
        let hi = self.since[r] as usize;
        if hi >= lo {
            let wsum = cum[hi + 1] - cum[lo];
            self.bank[r] += wsum * libm::pow(self.size[r] as f64, e);
        }
        self.since[r] = lo.saturating_sub(1) as u32;
    }

    /// Root of `v` with offset-preserving path compression.
    fn find(&mut self, v: usize) -> usize {
        let mut r = v;
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
                self.bank[u] += self.bank[p];
                self.parent[u] = r as u32;
            }
        }
        r
    }

    /// Merge two known roots `ra`, `rb` during band `band`; returns the
    /// surviving root. Ties (equal size) send `rb` under `ra`, so the caller's
    /// running root wins deterministically.
    fn link(&mut self, ra: usize, rb: usize, band: usize, cum: &[f64], e: f64) -> usize {
        // Both roots' sizes change now: settle the bands above `band` first.
        self.pay(ra, band + 1, cum, e);
        self.pay(rb, band + 1, cum, e);
        let (small, big) = if self.size[ra] < self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big as u32;
        self.bank[small] -= self.bank[big];
        self.size[big] += self.size[small];
        self.since[big] = band as u32;
        big
    }

    fn activate_exact(&mut self, v: usize, h: f64) {
        self.active[v] = true;
        self.parent[v] = v as u32;
        self.size[v] = 1;
        self.bank[v] = 0.0;
        self.since_h[v] = h;
    }

    /// Pay root `r` for heights `[h_now, since_h[r]]` at its current size:
    /// `size^E · (since_h^{H+1} − h_now^{H+1}) / (H+1)`.
    fn pay_exact(&mut self, r: usize, h_now: f64, hp1: f64, e: f64) {
        let a = self.since_h[r];
        if a > h_now {
            let w = (libm::pow(a, hp1) - libm::pow(h_now, hp1)) / hp1;
            self.bank[r] += w * libm::pow(self.size[r] as f64, e);
        }
        self.since_h[r] = h_now;
    }

    /// Merge two known roots `ra`, `rb` at event height `h_now`; returns the
    /// surviving root (tie-breaking as [`Forest::link`]).
    fn link_exact(&mut self, ra: usize, rb: usize, h_now: f64, hp1: f64, e: f64) -> usize {
        self.pay_exact(ra, h_now, hp1, e);
        self.pay_exact(rb, h_now, hp1, e);
        let (small, big) = if self.size[ra] < self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big as u32;
        self.bank[small] -= self.bank[big];
        self.size[big] += self.size[small];
        self.since_h[big] = h_now;
        big
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_sort_desc_matches_comparison_sort() {
        // Negatives, zeros of both signs, and repeats, so ties and the
        // sign-bit branch of `sort_key` are both exercised.
        let vals: Vec<f64> = alloc::vec![
            0.0, -0.0, 1.5, -1.5, 3.0, -3.0, 1.5, 0.25, -0.25, 2.0, -2.0, 0.0, 7.5, -7.5, 1e-300,
            -1e-300, 1e300, -1e300, 42.0, 42.0,
        ];
        let mut order: Vec<u32> = (0..vals.len() as u32).collect();
        let mut keys: Vec<u64> = vals.iter().map(|&x| sort_key(x)).collect();
        let (mut kt, mut ot) = (Vec::new(), Vec::new());
        radix_sort_desc(&mut keys, &mut order, &mut kt, &mut ot);

        let mut want: Vec<u32> = (0..vals.len() as u32).collect();
        want.sort_by(|&a, &b| sort_key(vals[b as usize]).cmp(&sort_key(vals[a as usize])));
        assert_eq!(order, want);
        // Keys travel with their index and come out descending.
        for (k, &o) in keys.iter().zip(order.iter()) {
            assert_eq!(*k, sort_key(vals[o as usize]));
        }
        assert!(keys.windows(2).all(|w| w[0] >= w[1]));
        // Stable: the two 42.0 entries keep input order, and so do the 1.5s.
        let p42: Vec<u32> = order
            .iter()
            .copied()
            .filter(|&o| o == 18 || o == 19)
            .collect();
        assert_eq!(p42, alloc::vec![18, 19]);
        let p15: Vec<u32> = order
            .iter()
            .copied()
            .filter(|&o| o == 2 || o == 6)
            .collect();
        assert_eq!(p15, alloc::vec![2, 6]);
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
