//! Permutation driver: a statistic map per draw → TFCE → max-statistic null →
//! FWE-corrected and uncorrected p-maps.
//!
//! The loop does not know which model made the statistic. A [`PermProblem`]
//! holds what is fixed across draws: the [`Store`], the [`Domain`], the TFCE
//! settings, the tail and the observed maps. Its constructor is the
//! engine-specific part. [`PermProblem::ols`] is the linear model
//! `brain ~ tested + confounds + intercept`, or, with no tested regressor, the
//! test of the intercept, which without confounds is the one-sample t-test.
//!
//! For a draw the engine supplies an `n × p` block of coefficients, the
//! product kernel takes the `p` dot products of that block with every node's
//! column of the store, and a finishing step turns them into one statistic.
//! No model is refitted: the store and the design are orthonormalised once,
//! and the t of a draw follows from the dot products in closed form. Draws
//! are grouped in batches of [`PermOpts::batch`] so that the store is read
//! once per batch.
//!
//! The loop is draw-addressable (draw `b` depends only on `(seed, b)`),
//! allocation-free after the workspace is built, and splittable: any
//! partition of the draws into [`run_range`] calls merged through
//! [`NullPartial`] finalizes bit-identically to one serial pass, for every
//! batch size. Under the `parallel` feature (on by default) `run_threads`
//! performs that split over a rayon pool.
//!
//! Concretely: [`run_range`] rebuilds each draw's block from `(seed, draw)`
//! alone, so no two ranges share state; every dot product is one sequential
//! sum over the subjects, so its bits do not depend on the batch it is
//! computed in; [`NullPartial`] merges by concatenating per-draw maxima and
//! adding per-node counts, both order-free; and [`finalize`] sorts the merged
//! maxima before reading p-values off them. [`tfce_ols`] is those steps run
//! serially.
//!
//! A finished run returns its [`NullState`]. A later run that is given the
//! state in [`PermOpts::extend`] and a larger budget runs only the draws the
//! state does not hold and merges, which gives the bits of one run with the
//! larger budget (`tests/perm_oracle.rs::g14_extending_a_null_equals_one_run`).
//!
//! Oracles. The t of every draw is checked against one least-squares fit per
//! node from the normal equations, and the p-maps against a per-draw loop on
//! the same draws (`tests/ols_oracle.rs::o3_draw_statistics_match_per_draw_fits`,
//! `::o4_inference_matches_a_naive_loop`). The one-sample case is checked
//! against MNE-Python 1.12.1
//! `permutation_cluster_1samp_test(..., threshold=dict(...), tail=1)`, frozen
//! into `tests/fixtures/perm_*.json` by `scripts/gen_perm_golden.py`: the two
//! exact-enumeration fixtures are RNG-independent and are compared map for map
//! (`tests/perm_oracle.rs::g5_exact_enumeration_matches_mne`); the Monte Carlo
//! fixture is compared statistically
//! (`::g6_monte_carlo_matches_mne_statistically`). Split, batch and thread
//! determinism have no external oracle and are checked against the serial run
//! (`::g7_split_merge_is_deterministic`,
//! `::g12_threaded_driver_matches_serial_bitwise`).

use alloc::vec::Vec;
use core::ops::Range;

use commonstats::accum::Mergeable;
use commonstats::{Scheme, gen_resample_indices, gen_sign_flips};

use crate::domain::Domain;
use crate::error::NeuroError;
use crate::store::{Store, nuisance_basis, orthonormalise};
use crate::tfce::{Tail, TfceParams, TfceWorkspace, tfce_tail_into};

/// Draws per batch that [`PermOpts::new`] sets.
const DEFAULT_BATCH: usize = 8;

/// Settings of a permutation run. Every convention is stated by the caller:
/// there is no default tail, weighting or seed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PermOpts<'a> {
    /// TFCE settings, applied to the statistic map of every draw.
    pub params: TfceParams,
    /// Which sign of the statistic counts as evidence (see [`Tail`] and
    /// [`TfceInference`]).
    pub tail: Tail,
    /// Run seed. Draw `b ≥ 1` is a pure function of `(seed, b)`.
    pub seed: u64,
    /// Draw budget `≥ 1`, the identity draw included, at most `u32::MAX`.
    /// With sign flips, all `2^n` patterns are enumerated when
    /// `2^n ≤ n_perm`, and the run then has `2^n` draws.
    pub n_perm: u64,
    /// Batch size `K ≥ 1`: how many draws share one pass over the store. It
    /// changes speed and memory (`K` statistic maps per thread), never a
    /// result.
    pub batch: usize,
    /// A null to continue, from the `null` field of an earlier
    /// [`TfceInference`] of the same test; `None` for a fresh run. `n_perm`
    /// is then the new total: only the draws from the stored count up to
    /// `n_perm` are run. The rules are on [`NullState`].
    pub extend: Option<&'a NullState>,
}

impl PermOpts<'_> {
    /// Options for a fresh run with the default batch size (8 draws).
    pub fn new(params: TfceParams, tail: Tail, seed: u64, n_perm: u64) -> Self {
        Self {
            params,
            tail,
            seed,
            n_perm,
            batch: DEFAULT_BATCH,
            extend: None,
        }
    }
}

/// Row permutation of draw `draw` into `out`: row `i` of the shuffled design
/// is row `out[i]` of the original.
///
/// Convention: draw 0 is the identity; draw `b ≥ 1` is
/// `commonstats::gen_resample_indices(Scheme::Permutation, n, b, seed)`, a
/// Fisher–Yates shuffle on Philox4x32-10 keyed by `(seed, b)`, so a draw does
/// not depend on any other draw, on the batch it is computed in, or on the
/// thread that computes it. This is the function [`PermProblem::ols`] draws
/// its row shuffles from
/// (`tests/ols_oracle.rs::o3_draw_statistics_match_per_draw_fits` rebuilds
/// every draw from it).
///
/// `n`: number of rows. `out`: `len == n`.
///
/// Errors: [`NeuroError::MismatchedLengths`] if `out.len() != n`;
/// [`NeuroError::InvalidParams`] if `n > u32::MAX`.
pub fn draw_permutation(n: usize, draw: u64, seed: u64, out: &mut [u32]) -> Result<(), NeuroError> {
    if out.len() != n {
        return Err(NeuroError::MismatchedLengths {
            expected: n,
            got: out.len(),
        });
    }
    if u32::try_from(n).is_err() {
        return Err(NeuroError::InvalidParams);
    }
    if draw == 0 {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = i as u32;
        }
    } else {
        gen_resample_indices(Scheme::Permutation, n, draw, seed, out)
            .expect("length and range checked above");
    }
    Ok(())
}

/// Signs of draw `draw` into `out` (`len == n`). `exact`: the draw id is the
/// pattern, bit `i` set ⇒ subject `i` flipped. Otherwise draw 0 is the
/// identity and draw `b ≥ 1` is `commonstats::gen_sign_flips(n, b, seed)`
/// (sampling with replacement from the sign group, so a repeated pattern is a
/// valid draw).
fn fill_signs(exact: bool, n: usize, seed: u64, draw: u64, out: &mut [f64]) {
    if exact {
        for (i, s) in out.iter_mut().enumerate() {
            *s = if (draw >> i) & 1 == 1 { -1.0 } else { 1.0 };
        }
    } else if draw == 0 {
        out.fill(1.0);
    } else {
        gen_sign_flips(n, draw, seed, out).expect("sign buffer has length n");
    }
}

/// The model behind a null, recorded in [`NullState`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// The linear model of [`PermProblem::ols`], statistic t.
    Ols,
}

/// The engine-specific part of a [`PermProblem`]: how the coefficient block
/// of a draw is made and how a node's dot products become a statistic.
#[derive(Debug, Clone)]
enum Blocks {
    /// Linear model. `w`: the design block `[tested column, Q]`, row-major
    /// `n × p` with orthonormal columns. `flip`: the rows are sign-flipped
    /// (the intercept is tested), else shuffled. `exact`: every sign pattern
    /// is enumerated. `dof = n − q − 1`.
    Ols {
        w: Vec<f64>,
        flip: bool,
        exact: bool,
        dof: f64,
    },
}

/// Everything fixed across draws: the store, the domain, the TFCE settings,
/// the tail, the draws to run, the null being continued (if any), and the
/// observed statistic and TFCE maps (draw 0, computed once at construction).
#[derive(Debug, Clone)]
pub struct PermProblem<'a> {
    domain: &'a Domain,
    store: &'a Store,
    params: TfceParams,
    tail: Tail,
    seed: u64,
    batch: usize,
    p: usize,
    blocks: Blocks,
    draws: Range<u64>,
    prior: Option<&'a NullState>,
    stat_obs: Vec<f64>,
    tfce_obs: Vec<f64>,
}

impl<'a> PermProblem<'a> {
    /// The OLS engine: one linear model per node with the same design,
    /// `brain ~ tested + confounds + intercept`, statistic t of `tested`; or
    /// with `tested = None` the model `brain ~ intercept + confounds`,
    /// statistic t of the intercept.
    ///
    /// Statistic: with `d` the node's store column, `x` the tested column
    /// (residualised against the nuisance basis `Q` and scaled to unit norm)
    /// and `R` the draw's row shuffle or sign flip,
    /// `b = (R x)' d`, `a² = Σ ((R Q)' d)²`, `rss = 1 − b² − a²` and
    /// `t = b · sqrt((n − q − 1) / rss)`. This is the t of the full model
    /// fitted to the nuisance-residualised data with the design rows
    /// transformed by `R`; for draw 0 it is the ordinary OLS t of the raw
    /// data. With `tested = None` and no confounds it is the one-sample t,
    /// `mean / sqrt(var / n)` with `ddof = 1`. A node with
    /// `rss ≤ 4ε · (n − q − 1)` gets `t = 0` (ε = f64 epsilon: no residual
    /// variance is left to rounding), and so does a node the store zeroed.
    /// nilearn 0.14.1 `t_score_with_covars_and_normalized_design` has the
    /// same formula. Checked to `rel 1e-9` against one normal-equations fit
    /// per node and draw
    /// (`tests/ols_oracle.rs::o3_draw_statistics_match_per_draw_fits`) and,
    /// for the one-sample case, against MNE-Python
    /// (`tests/perm_oracle.rs::g5_exact_enumeration_matches_mne`).
    ///
    /// Draws: with `tested` given, the rows of `[tested, Q]` are shuffled
    /// together by [`draw_permutation`] and the brain data stay fixed, which
    /// gives the Freedman–Lane statistic. With `tested = None` the rows are
    /// sign-flipped by `commonstats::gen_sign_flips(n, b, seed)`, or every
    /// one of the `2^n` patterns is enumerated when `2^n ≤ opts.n_perm`
    /// (draw `k` then flips subject `i` when bit `i` of `k` is set). Draw 0
    /// is the identity in each case.
    ///
    /// `store`: built with `nuisance_basis(confounds, n, tested.is_some())`,
    /// so that the intercept is a nuisance term exactly when a regressor is
    /// tested. `tested`: one value per subject, `len == store.n()`.
    ///
    /// Errors: [`NeuroError::InvalidParams`] if `opts.n_perm` is 0 or above
    /// `u32::MAX`, `opts.batch == 0`, `opts.params` is invalid, or
    /// `opts.extend` holds a null this run cannot continue (see
    /// [`NullState`]);
    /// [`NeuroError::MismatchedLengths`] if `store.n_nodes()` differs from
    /// `domain.n_nodes()` or `tested.len()` from `store.n()`;
    /// [`NeuroError::TooFewSubjects`] if `n − q − 1 < 1`;
    /// [`NeuroError::NonFiniteStat`] if `tested` holds a NaN or ±∞;
    /// [`NeuroError::DegenerateDesign`] if the tested column is in the span
    /// of the nuisance basis (norm after residualising at most `1e-10` of the
    /// norm before).
    pub fn ols(
        domain: &'a Domain,
        store: &'a Store,
        tested: Option<&[f64]>,
        opts: &PermOpts<'a>,
    ) -> Result<Self, NeuroError> {
        if opts.n_perm == 0 || opts.n_perm > u64::from(u32::MAX) || opts.batch == 0 {
            return Err(NeuroError::InvalidParams);
        }
        if store.n_nodes() != domain.n_nodes() {
            return Err(NeuroError::MismatchedLengths {
                expected: domain.n_nodes(),
                got: store.n_nodes(),
            });
        }
        let (n, q) = (store.n(), store.q());
        if let Some(t) = tested {
            if t.len() != n {
                return Err(NeuroError::MismatchedLengths {
                    expected: n,
                    got: t.len(),
                });
            }
        }
        if n < q + 2 {
            return Err(NeuroError::TooFewSubjects);
        }
        let mut x = tested.map_or_else(|| alloc::vec![1.0f64; n], <[f64]>::to_vec);
        if !orthonormalise(&mut x, store.basis())? {
            return Err(NeuroError::DegenerateDesign);
        }
        let p = q + 1;
        let mut w = alloc::vec![0.0f64; n * p];
        for (i, row) in w.chunks_exact_mut(p).enumerate() {
            row[0] = x[i];
            for (k, slot) in row[1..].iter_mut().enumerate() {
                *slot = store.basis()[k * n + i];
            }
        }
        let flip = tested.is_none();
        let exact = flip && n < 64 && (1u64 << n) <= opts.n_perm;
        let b = if exact { 1u64 << n } else { opts.n_perm };
        // An enumerated draw id is a bit pattern, not a seeded draw: such a
        // run cannot continue a Monte Carlo null.
        if exact && opts.extend.is_some() {
            return Err(NeuroError::InvalidParams);
        }
        let mut problem = Self {
            domain,
            store,
            params: opts.params,
            tail: opts.tail,
            seed: opts.seed,
            batch: opts.batch.min(b as usize),
            p,
            blocks: Blocks::Ols {
                w,
                flip,
                exact,
                dof: (n - q - 1) as f64,
            },
            draws: 0..b,
            prior: None,
            stat_obs: Vec::new(),
            tfce_obs: Vec::new(),
        };
        problem.observe()?;
        if let Some(state) = opts.extend {
            problem.resume(state)?;
        }
        Ok(problem)
    }

    /// Draw 0 through the kernel and TFCE: the observed maps.
    fn observe(&mut self) -> Result<(), NeuroError> {
        let v = self.domain.n_nodes();
        let mut ws = PermWorkspace::sized(self, 1);
        fill_stats(self, 0..1, &mut ws);
        let mut tfce_obs = alloc::vec![0.0f64; v];
        tfce_tail_into(
            self.domain,
            &ws.stat[..v],
            &self.params,
            self.tail,
            &mut ws.ws,
            &mut tfce_obs,
        )?;
        self.stat_obs = ws.stat;
        self.tfce_obs = tfce_obs;
        Ok(())
    }

    /// Continue `state`: check that it is a null of this very test with fewer
    /// draws than the budget, and start the run after its last draw.
    fn resume(&mut self, state: &'a NullState) -> Result<(), NeuroError> {
        let v = self.n_nodes();
        let b_old = state.maxima.len() as u64;
        let same = state.seed == self.seed
            && state.params == self.params
            && state.tail == self.tail
            && state.n == self.n()
            && state.n_nodes == v
            && state.q == self.store.q()
            && state.engine == self.engine()
            && state.exceed.len() == v
            && state.exceed.iter().all(|&c| u64::from(c) <= b_old)
            && state.tfce_obs.len() == v
            && state
                .tfce_obs
                .iter()
                .zip(&self.tfce_obs)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if !same || b_old == 0 || b_old >= self.draws.end {
            return Err(NeuroError::InvalidParams);
        }
        self.draws.start = b_old;
        self.batch = self.batch.min((self.draws.end - b_old) as usize);
        self.prior = Some(state);
        Ok(())
    }

    fn engine(&self) -> Engine {
        match self.blocks {
            Blocks::Ols { .. } => Engine::Ols,
        }
    }

    /// Observed statistic per node (draw 0), signed.
    pub fn stat_obs(&self) -> &[f64] {
        &self.stat_obs
    }

    /// TFCE of the observed statistic under the problem's tail, as
    /// [`tfce_tail_into`] returns it (signed for [`Tail::TwoSided`]).
    pub fn tfce_obs(&self) -> &[f64] {
        &self.tfce_obs
    }

    /// Number of subjects `n`.
    pub fn n(&self) -> usize {
        self.store.n()
    }

    /// Number of nodes `V`.
    pub fn n_nodes(&self) -> usize {
        self.domain.n_nodes()
    }

    /// The draws this run adds: `0..B`, or `B_old..B` when it continues a
    /// null of `B_old` draws. `B` is the budget `opts.n_perm`, or `2^n` when
    /// every sign pattern is enumerated.
    pub fn draws(&self) -> Range<u64> {
        self.draws.clone()
    }

    /// Batch size in use: `opts.batch`, capped at the number of draws.
    pub fn batch(&self) -> usize {
        self.batch
    }
}

/// Per-thread scratch for [`run_range`]: the row index and sign buffers (`n`
/// each), the batch matrix (`n × K·p`), the dot products of one node
/// (`K·p`), `K` statistic maps, one TFCE map, and a [`TfceWorkspace`].
#[derive(Debug, Clone)]
pub struct PermWorkspace {
    perm: Vec<u32>,
    signs: Vec<f64>,
    batch: Vec<f64>,
    acc: Vec<f64>,
    stat: Vec<f64>,
    tfce: Vec<f64>,
    ws: TfceWorkspace,
}

impl PermWorkspace {
    /// Workspace for `problem`, sized by its batch.
    pub fn new(problem: &PermProblem<'_>) -> Self {
        Self::sized(problem, problem.batch)
    }

    fn sized(problem: &PermProblem<'_>, batch: usize) -> Self {
        let (n, v) = (problem.n(), problem.n_nodes());
        let width = batch * problem.p;
        Self {
            perm: alloc::vec![0; n],
            signs: alloc::vec![1.0; n],
            batch: alloc::vec![0.0; n * width],
            acc: alloc::vec![0.0; width],
            stat: alloc::vec![0.0; batch * v],
            tfce: alloc::vec![0.0; v],
            ws: TfceWorkspace::new(v),
        }
    }
}

/// The product kernel: for every node, the dot products of its store column
/// with the `width` columns of `batch` (row-major `n × width`), handed to
/// `finish` as `(node, dots)`. Each dot product is one sum over the subjects
/// in row order, so its value does not depend on `width` or on which thread
/// runs it; the inner loop runs across the columns, which is what lets the
/// compiler vectorise it without reordering any single sum.
fn product(store: &Store, batch: &[f64], acc: &mut [f64], mut finish: impl FnMut(usize, &[f64])) {
    let width = acc.len();
    for (node, col) in store.as_slice().chunks_exact(store.n()).enumerate() {
        acc.fill(0.0);
        for (&d, row) in col.iter().zip(batch.chunks_exact(width)) {
            for (a, &x) in acc.iter_mut().zip(row) {
                *a += d * x;
            }
        }
        finish(node, acc);
    }
}

/// Statistic maps of `draws` into `ws.stat`, map `j` at `[j · V, (j + 1) · V)`.
fn fill_stats(problem: &PermProblem<'_>, draws: Range<u64>, ws: &mut PermWorkspace) {
    let k = (draws.end - draws.start) as usize;
    let (n, p, v) = (problem.n(), problem.p, problem.n_nodes());
    let width = k * p;
    assert!(
        width <= ws.acc.len() && k * v <= ws.stat.len(),
        "more draws than the workspace's batch"
    );
    if k == 0 {
        return;
    }
    let batch = &mut ws.batch[..n * width];
    let stat = &mut ws.stat[..k * v];
    match &problem.blocks {
        Blocks::Ols {
            w,
            flip,
            exact,
            dof,
        } => {
            for (j, draw) in draws.enumerate() {
                let rows = batch.chunks_exact_mut(width);
                if *flip {
                    fill_signs(*exact, n, problem.seed, draw, &mut ws.signs);
                    for ((row, src), &s) in rows.zip(w.chunks_exact(p)).zip(&ws.signs) {
                        for (slot, &x) in row[j * p..(j + 1) * p].iter_mut().zip(src) {
                            *slot = s * x;
                        }
                    }
                } else {
                    draw_permutation(n, draw, problem.seed, &mut ws.perm)
                        .expect("workspace sized for n rows");
                    for (row, &r) in rows.zip(&ws.perm) {
                        let r = r as usize;
                        row[j * p..(j + 1) * p].copy_from_slice(&w[r * p..(r + 1) * p]);
                    }
                }
            }
            // No residual variance left to rounding: the statistic is 0.
            let floor = 4.0 * f64::EPSILON * dof;
            product(problem.store, batch, &mut ws.acc[..width], |node, dots| {
                for (j, d) in dots.chunks_exact(p).enumerate() {
                    let b = d[0];
                    let mut ss = b * b;
                    for &a in &d[1..] {
                        ss += a * a;
                    }
                    let rss = 1.0 - ss;
                    stat[j * v + node] = if rss <= floor {
                        0.0
                    } else {
                        b * libm::sqrt(dof / rss)
                    };
                }
            });
        }
    }
}

/// Statistic maps of the draws `draws`, without TFCE: the product and the
/// finishing step of the problem's engine (for [`PermProblem::ols`], the t
/// defined there). Returns `k = draws.len()` maps one after another: map `j`,
/// of draw `draws.start + j`, is `out[j · V .. (j + 1) · V]`. The bits of a
/// map do not depend on which other draws share the call.
///
/// `draws`: at most [`PermProblem::batch`] draw ids. `ws`: built by
/// [`PermWorkspace::new`] for this `problem`.
///
/// Panics if `draws` holds more draws than the batch.
pub fn stat_batch<'w>(
    problem: &PermProblem<'_>,
    draws: Range<u64>,
    ws: &'w mut PermWorkspace,
) -> &'w [f64] {
    let k = (draws.end - draws.start) as usize;
    fill_stats(problem, draws, ws);
    &ws.stat[..k * problem.n_nodes()]
}

/// Mergeable null partial over a range of draws: one null value per draw and,
/// per node, the count of draws whose tested TFCE quantity reached the
/// observed one (`≥`). The tested quantity is the TFCE map for the one-sided
/// tails and its absolute value for [`Tail::TwoSided`]; the null value of a
/// draw is the maximum of that quantity over the nodes.
/// [`Mergeable::merge`] concatenates maxima and adds counts — order-free, so
/// any partition of the draws finalizes identically. Counts are `u32`, which
/// is why a run holds at most `u32::MAX` draws.
#[derive(Debug, Clone, PartialEq)]
pub struct NullPartial {
    maxima: Vec<f64>,
    exceed: Vec<u32>,
}

impl NullPartial {
    /// The identity partial for `n_nodes` nodes: no draws, zero counts.
    pub fn empty(n_nodes: usize) -> Self {
        Self {
            maxima: Vec::new(),
            exceed: alloc::vec![0; n_nodes],
        }
    }

    /// Per-draw null values in the order the draws were appended (unsorted);
    /// merge concatenates them.
    pub fn maxima(&self) -> &[f64] {
        &self.maxima
    }

    /// Per node: number of draws in this partial whose tested TFCE quantity
    /// is `≥` the observed one.
    pub fn exceed(&self) -> &[u32] {
        &self.exceed
    }
}

impl Mergeable for NullPartial {
    /// Panics if the node counts differ (partials from different problems).
    fn merge(&mut self, other: &Self) {
        assert_eq!(
            self.exceed.len(),
            other.exceed.len(),
            "NullPartial::merge: node count mismatch"
        );
        self.maxima.extend_from_slice(&other.maxima);
        for (a, &b) in self.exceed.iter_mut().zip(&other.exceed) {
            *a += b;
        }
    }
}

/// Run the draws `draws` and return their [`NullPartial`]. Per batch of
/// [`PermProblem::batch`] draws: the statistic maps ([`stat_batch`]), then
/// per draw [`tfce_tail_into`], the null value and the per-node `≥` count. A
/// draw with no node above `start` has null value `0`. No allocation inside
/// the loop (after the workspace has grown to the longest threshold grid
/// seen). `draws` must lie within [`PermProblem::draws`] (`debug_assert`);
/// the caller splits that range across threads as it likes.
pub fn run_range(
    problem: &PermProblem<'_>,
    draws: Range<u64>,
    ws: &mut PermWorkspace,
) -> NullPartial {
    debug_assert!(
        problem.draws.start <= draws.start && draws.end <= problem.draws.end,
        "draw range outside the problem's draws"
    );
    let v = problem.n_nodes();
    let two_sided = problem.tail == Tail::TwoSided;
    let mut null = NullPartial::empty(v);
    null.maxima
        .reserve(draws.end.saturating_sub(draws.start) as usize);
    let mut start = draws.start;
    while start < draws.end {
        let end = draws.end.min(start + problem.batch as u64);
        fill_stats(problem, start..end, ws);
        for j in 0..(end - start) as usize {
            tfce_tail_into(
                problem.domain,
                &ws.stat[j * v..(j + 1) * v],
                &problem.params,
                problem.tail,
                &mut ws.ws,
                &mut ws.tfce,
            )
            .expect("params validated and statistic finite by construction");
            let mut max = 0.0f64;
            for ((&tb, &to), cnt) in ws.tfce.iter().zip(&problem.tfce_obs).zip(&mut null.exceed) {
                let (tb, to) = if two_sided {
                    (libm::fabs(tb), libm::fabs(to))
                } else {
                    (tb, to)
                };
                if tb > max {
                    max = tb;
                }
                if tb >= to {
                    *cnt += 1;
                }
            }
            null.maxima.push(max);
        }
        start = end;
    }
    null
}

/// A finished null and what it was computed for: enough to check that a later
/// run continues the same test.
///
/// Continuing: pass the state in [`PermOpts::extend`] together with the same
/// data, design, seed, TFCE settings and tail, and a larger `n_perm`. The run
/// recomputes the observed TFCE map, runs draws `B..n_perm` only and merges
/// them with the stored null. The result equals, bit for bit, one run with
/// budget `n_perm` (`tests/perm_oracle.rs::g14_extending_a_null_equals_one_run`).
///
/// The call fails with [`NeuroError::InvalidParams`] when
/// - `n_perm` is not larger than the stored `B`;
/// - the draws are sign flips and `2^n ≤ n_perm`: a budget that large
///   enumerates every pattern, an enumerated draw is not a seeded draw, and
///   an enumerated null is already complete;
/// - the recomputed observed TFCE map differs from `tfce_obs` in any bit, or
///   `seed`, `params` (every field, also one the weighting does not use),
///   `tail`, `n`, `n_nodes`, `q` or `engine` differ;
/// - the state is not a null at all: no draws, `exceed` of another length, a
///   count above `B`.
///
/// The crate does no I/O: a caller that stores a state writes these fields
/// and builds the struct again from them.
#[derive(Debug, Clone, PartialEq)]
pub struct NullState {
    /// Run seed.
    pub seed: u64,
    /// The FWE null: one null value per draw, sorted ascending. Its length is
    /// `B`, the number of draws, identity included.
    pub maxima: Vec<f64>,
    /// Per node: number of draws whose tested TFCE quantity is `≥` the
    /// observed one, identity included.
    pub exceed: Vec<u32>,
    /// TFCE of the observed statistic under `tail`, as [`tfce_tail_into`]
    /// returns it: signed for [`Tail::TwoSided`], the TFCE of the negated map
    /// for [`Tail::Negative`].
    pub tfce_obs: Vec<f64>,
    /// TFCE settings of the run.
    pub params: TfceParams,
    /// Tail of the run.
    pub tail: Tail,
    /// Number of subjects `n`.
    pub n: usize,
    /// Number of nodes `V`.
    pub n_nodes: usize,
    /// Number of vectors in the nuisance basis of the store.
    pub q: usize,
    /// Which model made the statistic.
    pub engine: Engine,
}

/// Observed maps and permutation p-maps. `B = null.maxima.len()` is the
/// number of draws, identity included.
///
/// The tested quantity `T` follows the tail of the run: `tfce_obs` for
/// [`Tail::Positive`] (TFCE of the map) and [`Tail::Negative`] (TFCE of the
/// negated map), `|tfce_obs|` for [`Tail::TwoSided`] (TFCE of the map minus
/// TFCE of the negated map). The null value of a draw is the maximum of its
/// `T` over the nodes. Both p-maps count with `≥` and include the identity
/// draw, so every p is at least `1/B` and never 0. nilearn's
/// `(1 + #{null ≥ observed}) / (n_perm + 1)` is the same rule with
/// `B = n_perm + 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct TfceInference {
    /// Observed statistic per node, signed under every tail.
    pub stat_obs: Vec<f64>,
    /// FWE-corrected p per node: `#{b : max_b ≥ T(v)} / B` over the max-TFCE
    /// null. For the one-sample test under [`Tail::Positive`] this matches
    /// MNE `permutation_cluster_1samp_test(..., threshold=dict(...), tail=1)`
    /// `cluster_pv` (`tests/perm_oracle.rs::g5_exact_enumeration_matches_mne`,
    /// exact enumeration).
    pub p_fwe: Vec<f64>,
    /// Uncorrected p per node: `#{b : T_b(v) ≥ T(v)} / B` — the node's own
    /// null, not the max null, so it carries no multiplicity correction. FSL
    /// `randomise` `_tfce_p` semantics; checked against a naive per-draw loop
    /// (`tests/ols_oracle.rs::o4_inference_matches_a_naive_loop`).
    pub p_unc: Vec<f64>,
    /// The null and its settings: the observed TFCE map (`null.tfce_obs`),
    /// the sorted per-draw null values (`null.maxima`, `len == B`), the
    /// per-node counts, the seed, the TFCE settings and the tail.
    pub null: NullState,
}

/// Turn the null of a run (every draw of [`PermProblem::draws`] accounted for
/// once) into p-maps, merging it first with the null the run continues, if
/// any. `p_fwe` by binary search over the sorted maxima, `O(V log B)`. Errors
/// with [`NeuroError::InvalidParams`] when the number of draws `null` holds
/// differs from the number of draws the problem still needs (it does not check
/// which draws they are), and with [`NeuroError::MismatchedLengths`] when it has another node count.
pub fn finalize(
    problem: &PermProblem<'_>,
    null: &NullPartial,
) -> Result<TfceInference, NeuroError> {
    let v = problem.n_nodes();
    if null.exceed.len() != v {
        return Err(NeuroError::MismatchedLengths {
            expected: v,
            got: null.exceed.len(),
        });
    }
    if null.maxima.len() as u64 != problem.draws.end - problem.draws.start {
        return Err(NeuroError::InvalidParams);
    }
    let mut maxima = null.maxima.clone();
    let mut exceed = null.exceed.clone();
    if let Some(prior) = problem.prior {
        maxima.extend_from_slice(&prior.maxima);
        for (a, &b) in exceed.iter_mut().zip(&prior.exceed) {
            *a += b;
        }
    }
    maxima.sort_unstable_by(f64::total_cmp);
    let two_sided = problem.tail == Tail::TwoSided;
    let b = maxima.len() as f64;
    let p_fwe = problem
        .tfce_obs
        .iter()
        .map(|&t| {
            let t = if two_sided { libm::fabs(t) } else { t };
            let below = maxima.partition_point(|&m| m < t);
            (maxima.len() - below) as f64 / b
        })
        .collect();
    let p_unc = exceed.iter().map(|&c| f64::from(c) / b).collect();
    Ok(TfceInference {
        stat_obs: problem.stat_obs.clone(),
        p_fwe,
        p_unc,
        null: NullState {
            seed: problem.seed,
            maxima,
            exceed,
            tfce_obs: problem.tfce_obs.clone(),
            params: problem.params,
            tail: problem.tail,
            n: problem.n(),
            n_nodes: v,
            q: problem.store.q(),
            engine: problem.engine(),
        },
    })
}

/// The store of [`tfce_ols`]: every row of `y`, residualised against the
/// confounds and, when a regressor is tested, the intercept.
fn ols_store(
    domain: &Domain,
    y: &[f64],
    n: usize,
    tested: Option<&[f64]>,
    confounds: &[f64],
) -> Result<Store, NeuroError> {
    let basis = nuisance_basis(confounds, n, tested.is_some())?;
    let rows: Vec<usize> = (0..n).collect();
    Store::from_f64(y, n, domain.n_nodes(), &rows, &basis)
}

/// TFCE permutation inference for the linear model
/// `brain ~ tested + confounds + intercept` fitted at every node, with
/// max-TFCE FWE correction — the serial reference path:
/// [`nuisance_basis`] + [`Store::from_f64`] + [`PermProblem::ols`] + one
/// [`run_range`] over [`PermProblem::draws`] + [`finalize`].
///
/// Convention: statistic t of `tested` per node; with `tested = None` the
/// model is `brain ~ intercept + confounds` and the statistic is the t of the
/// intercept (the one-sample t, `ddof = 1`, when there are no confounds).
/// Statistic, degenerate nodes and draws are those of [`PermProblem::ols`];
/// the tail, the tested quantity and the p-values those of
/// [`TfceInference`]. The run is reproducible from `opts.seed` and
/// independent of the batch size and of how the draws are split. Matches a
/// naive loop that refits every node by the normal equations for every draw
/// (`tests/ols_oracle.rs::o4_inference_matches_a_naive_loop`), and for
/// `tested = None`, no confounds, [`Tail::Positive`] it matches MNE 1.12.1
/// `permutation_cluster_1samp_test(X, threshold=dict(start, step, e_power,
/// h_power), tail=1, adjacency)` under `Weighting::MneStep`
/// (`tests/perm_oracle.rs::g5_exact_enumeration_matches_mne`: exact
/// enumeration, `null.maxima` set-equal within `rel 1e-12` once MNE's
/// enumeration is corrected — it counts the identity twice and skips the
/// full-negation pattern — and `|p_fwe − p_mne| ≤ 1/B`;
/// `::g6_monte_carlo_matches_mne_statistically`: Monte Carlo within a 4σ
/// band).
///
/// `y`: brain data, subject-major, `y[i · V + v]` is subject `i` at node `v`,
/// `len == n · domain.n_nodes()`; every row is used. `n`: number of subjects,
/// `n − q − 1 ≥ 1` with `q` the number of nuisance columns kept (so `n ≥ 2`
/// for the one-sample test). `tested`: the tested regressor, one value per
/// subject, or `None` to test the intercept. `confounds`: row-major `n × c`,
/// no constant column needed; empty for none. `opts`: TFCE settings, tail,
/// seed, draw budget, batch size and an optional null to continue (see
/// [`PermOpts`] and [`NullState`]); `Weighting::Exact` with `start = 0` is
/// the recommended TFCE rule.
///
/// Errors: those of [`nuisance_basis`], [`Store::from_f64`] and
/// [`PermProblem::ols`]: [`NeuroError::TooFewSubjects`],
/// [`NeuroError::MismatchedLengths`], [`NeuroError::NonFiniteStat`],
/// [`NeuroError::DegenerateDesign`], [`NeuroError::InvalidParams`].
///
/// ```
/// use neurostats::{Conn, Domain, PermOpts, Tail, TfceParams, Weighting, tfce_ols};
/// let dom = Domain::from_volume([3, 1, 1], Conn::Face);
/// // 4 subjects × 3 nodes, subject-major; node 0 carries a consistent effect.
/// let y = [2.0, 0.3, 0.0, 2.5, -0.4, 0.0, 1.8, 0.1, 0.0, 2.2, -0.2, 0.0];
/// let p = TfceParams { e: 0.5, h: 2.0, start: 0.0, step: 1.0, weighting: Weighting::Exact };
/// let opts = PermOpts::new(p, Tail::Positive, 42, 1000);
/// // No tested regressor and no confounds: the one-sample test.
/// let r = tfce_ols(&dom, &y, 4, None, &[], &opts).unwrap();
/// assert_eq!(r.null.maxima.len(), 16); // 2^4 ≤ 1000: exact enumeration
/// assert!(r.p_fwe[0] <= 2.0 / 16.0 && r.p_fwe[2] == 1.0);
/// ```
pub fn tfce_ols(
    domain: &Domain,
    y: &[f64],
    n: usize,
    tested: Option<&[f64]>,
    confounds: &[f64],
    opts: &PermOpts<'_>,
) -> Result<TfceInference, NeuroError> {
    let store = ols_store(domain, y, n, tested, confounds)?;
    let problem = PermProblem::ols(domain, &store, tested, opts)?;
    let mut ws = PermWorkspace::new(&problem);
    let null = run_range(&problem, problem.draws(), &mut ws);
    finalize(&problem, &null)
}

/// Split `draws` into at most `4 · threads` contiguous ranges, each a whole
/// number of batches of `batch` draws (the last range takes the remainder),
/// with batch counts differing by at most one. Whole batches, so that a batch
/// is not cut short at a range boundary. Four chunks per thread, not one:
/// draws differ in cost (a draw whose statistic map has more suprathreshold
/// nodes does more union-find work), so the extra chunks let rayon re-balance
/// to the end of the run. `batch ≥ 1` and `threads ≥ 1`.
#[cfg(feature = "parallel")]
fn chunk_ranges(draws: Range<u64>, batch: usize, threads: usize) -> Vec<Range<u64>> {
    let k = batch as u64;
    let batches = (draws.end - draws.start).div_ceil(k);
    let chunks = batches.min((threads as u64).saturating_mul(4)).max(1);
    let (base, rem) = (batches / chunks, batches % chunks);
    let mut out = Vec::with_capacity(chunks as usize);
    let mut start = draws.start;
    for i in 0..chunks {
        let end = draws.end.min(start + (base + u64::from(i < rem)) * k);
        out.push(start..end);
        start = end;
    }
    out
}

/// [`run_range`] with the draws spread over `threads` OS threads.
/// **Bit-identical to [`run_range`] after [`finalize`] for any `threads`**:
/// a draw depends only on `(seed, draw)`, and [`finalize`] sorts the merged
/// maxima, so the split cannot move a single bit of the result.
///
/// `threads == 0` means every core rayon can see
/// (`std::thread::available_parallelism`, falling back to 1). `threads == 1`,
/// a single draw, or a pool that fails to build all run the serial path — a
/// pool build failure is a fallback, not an error to the caller. Each thread
/// owns one [`PermWorkspace`].
///
/// The pool is built per call and sized by `threads`, so `RAYON_NUM_THREADS`
/// and rayon's global pool have no effect here and two concurrent callers do
/// not fight over one setting.
///
/// Checked on every permutation fixture and thread count by
/// `tests/perm_oracle.rs::g12_threaded_driver_matches_serial_bitwise`.
#[cfg(feature = "parallel")]
pub fn run_threads(problem: &PermProblem<'_>, draws: Range<u64>, threads: usize) -> NullPartial {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let serial = || {
        let mut ws = PermWorkspace::new(problem);
        run_range(problem, draws.clone(), &mut ws)
    };
    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(1, |t| t.get())
    } else {
        threads
    };
    if threads == 1 || draws.end.saturating_sub(draws.start) <= 1 {
        return serial();
    }
    // A pool per call rather than rayon's global one: `threads` then means
    // exactly what this caller passed, and two concurrent callers (Python
    // threads, say) cannot overwrite each other's thread count. Building it
    // costs milliseconds — nothing against a permutation run.
    match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
        Err(_) => serial(),
        Ok(pool) => pool.install(|| {
            chunk_ranges(draws.clone(), problem.batch, threads)
                .into_par_iter()
                .map_init(
                    || PermWorkspace::new(problem),
                    |ws, r| run_range(problem, r, ws),
                )
                .reduce(
                    || NullPartial::empty(problem.n_nodes()),
                    |mut a, b| {
                        a.merge(&b);
                        a
                    },
                )
        }),
    }
}

/// [`tfce_ols`] with the draws spread over `threads` OS threads by
/// `run_threads`. **Bit-identical to [`tfce_ols`] for any `threads`.**
/// `threads == 0` means every core. Arguments, conventions and errors are
/// [`tfce_ols`]'s.
///
/// ```
/// use neurostats::{Conn, Domain, PermOpts, Tail, TfceParams, Weighting, tfce_ols, tfce_ols_threads};
/// let dom = Domain::from_volume([3, 1, 1], Conn::Face);
/// let y = [2.0, 0.3, 0.0, 2.5, -0.4, 0.0, 1.8, 0.1, 0.0, 2.2, -0.2, 0.0];
/// let tested = [0.5, 1.5, -1.0, 2.0];
/// let p = TfceParams { e: 0.5, h: 2.0, start: 0.0, step: 1.0, weighting: Weighting::Exact };
/// let opts = PermOpts::new(p, Tail::TwoSided, 42, 200);
/// let serial = tfce_ols(&dom, &y, 4, Some(&tested), &[], &opts).unwrap();
/// let threaded = tfce_ols_threads(&dom, &y, 4, Some(&tested), &[], &opts, 0).unwrap();
/// assert_eq!(serial, threaded);
/// ```
#[cfg(feature = "parallel")]
pub fn tfce_ols_threads(
    domain: &Domain,
    y: &[f64],
    n: usize,
    tested: Option<&[f64]>,
    confounds: &[f64],
    opts: &PermOpts<'_>,
    threads: usize,
) -> Result<TfceInference, NeuroError> {
    let store = ols_store(domain, y, n, tested, confounds)?;
    let problem = PermProblem::ols(domain, &store, tested, opts)?;
    let null = run_threads(&problem, problem.draws(), threads);
    finalize(&problem, &null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tfce::Weighting;
    use crate::{Conn, Domain};
    use alloc::vec;

    #[test]
    fn exact_draw_bits_encode_signs_and_draw_zero_is_identity() {
        let mut s = vec![0.0; 4];
        fill_signs(true, 4, 0, 0, &mut s);
        assert_eq!(s, [1.0, 1.0, 1.0, 1.0]);
        fill_signs(true, 4, 0, 0b1010, &mut s);
        assert_eq!(s, [1.0, -1.0, 1.0, -1.0]);
        fill_signs(false, 4, 9, 0, &mut s);
        assert_eq!(s, [1.0, 1.0, 1.0, 1.0]);
    }

    fn toy() -> (Domain, Vec<f64>, PermOpts<'static>) {
        let dom = Domain::from_volume([3, 1, 1], Conn::Face);
        // n = 4 subjects, V = 3: node 0 strong positive, node 1 noise, node 2 zero.
        let y = vec![
            2.0, 0.3, 0.0, //
            2.5, -0.4, 0.0, //
            1.8, 0.1, 0.0, //
            2.2, -0.2, 0.0,
        ];
        let p = TfceParams {
            e: 0.5,
            h: 2.0,
            start: 0.0,
            step: 1.0,
            weighting: Weighting::Exact,
        };
        (dom, y, PermOpts::new(p, Tail::Positive, 1, 1000))
    }

    // Exact enumeration on n = 4: B = 16, identity included, p ≥ 1/16, and the
    // strong node's FWE p is the smallest attainable (1/16: the all-flipped
    // realization mirrors the observed max only if it is also positive, and
    // here it is not).
    #[test]
    fn one_sample_exact_toy() {
        let (dom, y, opts) = toy();
        let r = tfce_ols(&dom, &y, 4, None, &[], &opts).unwrap();
        assert_eq!(r.null.maxima.len(), 16);
        assert!(r.null.maxima.windows(2).all(|w| w[0] <= w[1]));
        assert!(r.p_fwe.iter().all(|&p| (1.0 / 16.0..=1.0).contains(&p)));
        assert!(r.p_unc.iter().all(|&p| (1.0 / 16.0..=1.0).contains(&p)));
        assert!((r.p_fwe[0] - 1.0 / 16.0).abs() < 1e-15, "{:?}", r.p_fwe);
        assert_eq!(r.null.tfce_obs[2], 0.0);
        assert_eq!(r.p_fwe[2], 1.0);
    }

    // The split covers the draws exactly once, in contiguous pieces of whole
    // batches, with the count the driver promises.
    #[cfg(feature = "parallel")]
    #[test]
    fn chunk_ranges_partition_the_draws_in_whole_batches() {
        for first in [0u64, 37] {
            for b in [1u64, 2, 5, 16, 17, 200, 1000] {
                for batch in [1usize, 3, 8, 64] {
                    for threads in [1usize, 2, 3, 7, 64] {
                        let label = alloc::format!("b {b} batch {batch} threads {threads}");
                        let r = chunk_ranges(first..first + b, batch, threads);
                        let batches = b.div_ceil(batch as u64);
                        assert_eq!(r.len() as u64, batches.min(4 * threads as u64), "{label}");
                        assert_eq!(r[0].start, first);
                        assert_eq!(r[r.len() - 1].end, first + b);
                        for w in r.windows(2) {
                            assert_eq!(w[0].end, w[1].start, "{label}: gap/overlap");
                            assert_eq!((w[0].end - w[0].start) % batch as u64, 0, "{label}");
                        }
                        let counts: Vec<u64> = r
                            .iter()
                            .map(|x| (x.end - x.start).div_ceil(batch as u64))
                            .collect();
                        let (lo, hi) =
                            (*counts.iter().min().unwrap(), *counts.iter().max().unwrap());
                        assert!(hi - lo <= 1, "{label}: batches per chunk {counts:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn zero_budget_and_zero_batch_are_invalid() {
        let (dom, y, opts) = toy();
        for bad in [
            PermOpts { n_perm: 0, ..opts },
            PermOpts { batch: 0, ..opts },
            PermOpts {
                n_perm: u64::from(u32::MAX) + 1,
                ..opts
            },
        ] {
            assert_eq!(
                tfce_ols(&dom, &y, 4, None, &[], &bad),
                Err(NeuroError::InvalidParams)
            );
        }
    }

    #[test]
    fn y_length_must_match_domain() {
        let (dom, y, opts) = toy();
        assert_eq!(
            tfce_ols(&dom, &y[..8], 4, None, &[], &opts),
            Err(NeuroError::MismatchedLengths {
                expected: 12,
                got: 8
            })
        );
    }
}
