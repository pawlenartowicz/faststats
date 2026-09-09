//! Sign-flip permutation driver: one-sample t → TFCE per realization →
//! max-statistic null → FWE-corrected and uncorrected p-maps.
//!
//! The draw loop is draw-addressable (realization `b` depends only on
//! `(seed, b)`), allocation-free after the workspace is built, and splittable:
//! any partition of `0..B` into [`run_range`] calls merged through
//! [`NullPartial`] finalizes bit-identically to one serial pass. Under the
//! `parallel` feature (on by default) `tfce_one_sample_threads` performs that
//! split over a rayon pool; without it the caller owns any threading.
//!
//! Concretely: [`run_range`] takes a `Range<u64>` of draw ids and rebuilds each
//! draw's signs from `(seed, draw_id)` alone, so no two ranges share state;
//! [`NullPartial`] merges by concatenating per-draw maxima and adding per-node
//! counts, both order-free; and [`finalize`] sorts the merged maxima before
//! reading p-values off them, so the answer cannot depend on which range
//! finished first. [`tfce_one_sample`] is those three steps run serially.
//!
//! The module's oracle is MNE-Python 1.12.1
//! `permutation_cluster_1samp_test(..., threshold=dict(...), tail=1)`, frozen
//! into `tests/fixtures/perm_*.json` by `scripts/gen_perm_golden.py`. The two
//! exact-enumeration fixtures are RNG-independent and are compared map for map
//! (`tests/perm_oracle.rs::g5_exact_enumeration_matches_mne`); the Monte Carlo
//! fixture is compared statistically
//! (`::g6_monte_carlo_matches_mne_statistically`). Split/merge determinism has
//! no external oracle — it is checked against the serial run
//! (`::g7_split_merge_is_deterministic`,
//! `::g12_threaded_driver_matches_serial_bitwise`).

use alloc::vec::Vec;
use core::ops::Range;

use commonstats::accum::Mergeable;
use commonstats::gen_sign_flips;

use crate::domain::Domain;
use crate::error::NeuroError;
use crate::onesample::OneSampleT;
use crate::tfce::{TfceParams, TfceWorkspace, tfce_into};

/// How the `B` realizations are produced. `B` counts the identity realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignFlipPlan {
    /// Number of realizations actually run, identity included.
    pub b: u64,
    /// `true`: all `2^n` sign patterns, `b = 2^n`, draw `k` encodes the
    /// pattern bit-wise (bit `i` of `k` set ⇒ subject `i` flipped; `k = 0` is
    /// the identity). `false`: Monte Carlo, `b = b_requested`, draw 0 is the
    /// identity and draws `1..b` come from `commonstats::gen_sign_flips`
    /// (sampling with replacement from the sign group — a repeated pattern is a
    /// valid draw; expected duplicate pairs `B(B−1)/2^(n+1)`).
    pub exact: bool,
}

/// Choose exact enumeration when every pattern fits in the requested budget
/// (`2^n ≤ b_requested`, MNE's `_get_1samp_orders` rule with the identity
/// counted), else Monte Carlo with exactly `b_requested` realizations.
pub fn sign_flip_plan(n: usize, b_requested: u64) -> SignFlipPlan {
    if n < 64 && (1u64 << n) <= b_requested {
        SignFlipPlan {
            b: 1u64 << n,
            exact: true,
        }
    } else {
        SignFlipPlan {
            b: b_requested,
            exact: false,
        }
    }
}

/// Signs for draw `draw_id` under `plan` into `out` (`len == n`).
fn fill_signs(plan: &SignFlipPlan, n: usize, seed: u64, draw_id: u64, out: &mut [f64]) {
    if plan.exact {
        for (i, s) in out.iter_mut().enumerate() {
            *s = if (draw_id >> i) & 1 == 1 { -1.0 } else { 1.0 };
        }
    } else if draw_id == 0 {
        out.fill(1.0);
    } else {
        gen_sign_flips(n, draw_id, seed, out).expect("sign buffer has length n");
    }
}

/// Everything fixed across draws: the data, the domain, the TFCE parameters,
/// the seed, and the observed t and TFCE maps (computed once here).
#[derive(Debug, Clone)]
pub struct OneSampleProblem<'a> {
    domain: &'a Domain,
    x: &'a [f64],
    stat: OneSampleT,
    params: TfceParams,
    seed: u64,
    t_obs: Vec<f64>,
    tfce_obs: Vec<f64>,
}

impl<'a> OneSampleProblem<'a> {
    /// Validate and precompute. `x`: subject-major, `len == n · domain.n_nodes()`
    /// (see [`OneSampleT`]). `params`: TFCE operator settings; `Weighting::Exact`
    /// with `start = 0` is the recommended choice. `seed`: run seed for the
    /// Monte Carlo sign stream.
    ///
    /// Errors: [`NeuroError::TooFewSubjects`] (`n < 2`),
    /// [`NeuroError::MismatchedLengths`] (`x.len() != n · V`, `expected = n · V`),
    /// [`NeuroError::InvalidParams`] (bad `params`).
    pub fn new(
        domain: &'a Domain,
        x: &'a [f64],
        n: usize,
        params: TfceParams,
        seed: u64,
    ) -> Result<Self, NeuroError> {
        if n < 2 {
            return Err(NeuroError::TooFewSubjects);
        }
        let v = domain.n_nodes();
        if x.len() != n * v {
            return Err(NeuroError::MismatchedLengths {
                expected: n * v,
                got: x.len(),
            });
        }
        let stat = OneSampleT::new(x, n)?;
        let ones = alloc::vec![1.0f64; n];
        let mut t_obs = alloc::vec![0.0f64; v];
        stat.t_map(x, &ones, &mut t_obs);
        let mut ws = TfceWorkspace::new(v);
        let mut tfce_obs = alloc::vec![0.0f64; v];
        tfce_into(domain, &t_obs, &params, &mut ws, &mut tfce_obs)?;
        Ok(Self {
            domain,
            x,
            stat,
            params,
            seed,
            t_obs,
            tfce_obs,
        })
    }

    /// Observed one-sample t-map (identity realization).
    pub fn t_obs(&self) -> &[f64] {
        &self.t_obs
    }

    /// TFCE of the observed t-map.
    pub fn tfce_obs(&self) -> &[f64] {
        &self.tfce_obs
    }

    /// Number of subjects.
    pub fn n(&self) -> usize {
        self.stat.n()
    }
}

/// Per-thread scratch for [`run_range`]: sign buffer (`n`), t buffer (`V`),
/// TFCE buffer (`V`), and a [`TfceWorkspace`].
#[derive(Debug, Clone)]
pub struct PermWorkspace {
    signs: Vec<f64>,
    t: Vec<f64>,
    tfce: Vec<f64>,
    ws: TfceWorkspace,
}

impl PermWorkspace {
    /// Workspace for `n` subjects over `n_nodes` nodes.
    pub fn new(n: usize, n_nodes: usize) -> Self {
        Self {
            signs: alloc::vec![1.0; n],
            t: alloc::vec![0.0; n_nodes],
            tfce: alloc::vec![0.0; n_nodes],
            ws: TfceWorkspace::new(n_nodes),
        }
    }
}

/// Mergeable null partial over a range of draws: one max-TFCE per draw and,
/// per node, the count of draws whose TFCE reached the observed TFCE (`≥`).
/// [`Mergeable::merge`] concatenates maxima and adds counts — order-free, so
/// any partition of `0..B` finalizes identically. Counts are `u32`, so a null
/// with more than `u32::MAX` draws per node is unsupported.
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

    /// Per-draw map maxima in the order the draws were appended (unsorted);
    /// merge concatenates maxima.
    pub fn maxima(&self) -> &[f64] {
        &self.maxima
    }

    /// Per node: number of draws in this partial with `tfce_b(v) ≥ tfce_obs(v)`.
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

/// Run draws `draws` of `plan` and return their [`NullPartial`]. Per draw:
/// signs → t-map ([`OneSampleT::t_map`]) → [`tfce_into`] → map max and
/// per-node `≥` count. A draw with no node above `start` has max `0`. No
/// allocation inside the loop (after the workspace has grown to the longest
/// grid seen). `draws` must lie within `0..plan.b`
/// (`debug_assert`); the caller splits `0..plan.b` across threads as it likes.
pub fn run_range(
    problem: &OneSampleProblem<'_>,
    plan: &SignFlipPlan,
    draws: Range<u64>,
    ws: &mut PermWorkspace,
) -> NullPartial {
    debug_assert!(draws.end <= plan.b, "draw range exceeds plan.b");
    let n = problem.n();
    let v = problem.domain.n_nodes();
    let mut null = NullPartial::empty(v);
    null.maxima
        .reserve(draws.end.saturating_sub(draws.start) as usize);
    for draw_id in draws {
        fill_signs(plan, n, problem.seed, draw_id, &mut ws.signs);
        problem.stat.t_map(problem.x, &ws.signs, &mut ws.t);
        tfce_into(
            problem.domain,
            &ws.t,
            &problem.params,
            &mut ws.ws,
            &mut ws.tfce,
        )
        .expect("params validated and t finite by construction");
        let mut max = 0.0f64;
        for ((&tb, &to), cnt) in ws.tfce.iter().zip(&problem.tfce_obs).zip(&mut null.exceed) {
            if tb > max {
                max = tb;
            }
            if tb >= to {
                *cnt += 1;
            }
        }
        null.maxima.push(max);
    }
    null
}

/// Observed maps and permutation p-maps. `B` = number of realizations,
/// identity included. Both p-maps are **one-sided, positive tail**: only large
/// positive TFCE values count as evidence, and a negative effect is tested by
/// negating `x` and running again. Both count with `≥` and both include the
/// identity realization, so every p is at least `1/B` and never 0.
#[derive(Debug, Clone, PartialEq)]
pub struct TfceInference {
    /// Observed one-sample t per node (see [`OneSampleT`]).
    pub t_obs: Vec<f64>,
    /// TFCE of `t_obs`.
    pub tfce_obs: Vec<f64>,
    /// FWE-corrected p per node: `#{b : max_b ≥ tfce_obs(v)} / B` over the
    /// max-TFCE null (`≥`, identity included, so `p ≥ 1/B`). Matches MNE
    /// `permutation_cluster_1samp_test(..., threshold=dict(...), tail=1)`
    /// `cluster_pv` (`tests/perm_oracle.rs::g5_exact_enumeration_matches_mne`,
    /// exact enumeration).
    pub p_fwe: Vec<f64>,
    /// Uncorrected p per node: `#{b : tfce_b(v) ≥ tfce_obs(v)} / B` — the
    /// node's own null, not the max null, so it carries no multiplicity
    /// correction. Positive tail, identity included, `p ≥ 1/B`. FSL
    /// `randomise` `_tfce_p` semantics; MNE emits no such map, so no fixture
    /// validates it against an external tool.
    pub p_unc: Vec<f64>,
    /// The FWE null: per-draw map maxima, sorted ascending, `len == B`.
    pub null_max: Vec<f64>,
}

/// Turn a complete null (every draw in `0..B` accounted for once) into p-maps.
/// `B = null.maxima().len()`; `p_fwe` by binary search over the sorted maxima,
/// `O(V log B)`. Errors with `InvalidParams` when the null holds no draws
/// (`B = 0`, e.g. an unmerged [`NullPartial::empty`]).
pub fn finalize(
    problem: &OneSampleProblem<'_>,
    null: &NullPartial,
) -> Result<TfceInference, NeuroError> {
    if null.maxima.is_empty() {
        return Err(NeuroError::InvalidParams);
    }
    let mut null_max = null.maxima.clone();
    null_max.sort_unstable_by(f64::total_cmp);
    let b = null_max.len() as f64;
    let p_fwe = problem
        .tfce_obs
        .iter()
        .map(|&t| {
            let below = null_max.partition_point(|&m| m < t);
            (null_max.len() - below) as f64 / b
        })
        .collect();
    let p_unc = null.exceed.iter().map(|&c| c as f64 / b).collect();
    Ok(TfceInference {
        t_obs: problem.t_obs.clone(),
        tfce_obs: problem.tfce_obs.clone(),
        p_fwe,
        p_unc,
        null_max,
    })
}

/// One-sample sign-flip permutation test with max-TFCE FWE correction — the
/// serial reference path: [`OneSampleProblem::new`] + [`sign_flip_plan`] +
/// one [`run_range`] over `0..B` + [`finalize`].
///
/// Convention: one-sample t (`ddof = 1`, no hat smoothing) per node, positive
/// tail; TFCE per realization with `params`; null = per-realization map
/// maximum with the identity realization included; `p = mean(null ≥ obs)`.
/// Realizations: all `2^n` when `2^n ≤ b_requested`, else `b_requested`
/// Monte Carlo draws (identity + `b_requested − 1` from
/// `commonstats::gen_sign_flips(n, draw_id, seed)`), reproducible from `seed`
/// and independent of how `0..B` is split. Matches MNE 1.12.1
/// `permutation_cluster_1samp_test(X, threshold=dict(start, step, e_power,
/// h_power), tail=1, adjacency)` under `Weighting::MneStep`
/// (`tests/perm_oracle.rs::g5_exact_enumeration_matches_mne`: exact
/// enumeration, `null_max` set-equal within `rel 1e-12` once MNE's
/// enumeration is corrected — it counts the identity twice and skips the
/// full-negation pattern — and `|p_fwe − p_mne| ≤ 1/B`;
/// `::g6_monte_carlo_matches_mne_statistically`: Monte Carlo within a 4σ
/// band).
///
/// `x`: subject-major, `len == n · domain.n_nodes()`. `n ≥ 2`. `params`: see
/// [`TfceParams`]; `Weighting::Exact`, `start = 0` recommended. `seed`: any
/// `u64`. `b_requested ≥ 1`: the permutation budget including the identity.
///
/// Errors: [`NeuroError::InvalidParams`] if `b_requested == 0` or `params`
/// invalid; [`NeuroError::TooFewSubjects`]; [`NeuroError::MismatchedLengths`].
///
/// ```
/// use neurostats::{Conn, Domain, TfceParams, Weighting, tfce_one_sample};
/// let dom = Domain::from_volume([3, 1, 1], Conn::Face);
/// // 4 subjects × 3 nodes, subject-major; node 0 carries a consistent effect.
/// let x = [2.0, 0.3, 0.0, 2.5, -0.4, 0.0, 1.8, 0.1, 0.0, 2.2, -0.2, 0.0];
/// let p = TfceParams { e: 0.5, h: 2.0, start: 0.0, step: 1.0, weighting: Weighting::Exact };
/// let r = tfce_one_sample(&dom, &x, 4, &p, 42, 1000).unwrap();
/// assert_eq!(r.null_max.len(), 16); // 2^4 ≤ 1000: exact enumeration
/// assert!(r.p_fwe[0] <= 2.0 / 16.0 && r.p_fwe[2] == 1.0);
/// ```
pub fn tfce_one_sample(
    domain: &Domain,
    x: &[f64],
    n: usize,
    params: &TfceParams,
    seed: u64,
    b_requested: u64,
) -> Result<TfceInference, NeuroError> {
    if b_requested == 0 {
        return Err(NeuroError::InvalidParams);
    }
    let problem = OneSampleProblem::new(domain, x, n, *params, seed)?;
    let plan = sign_flip_plan(n, b_requested);
    let mut ws = PermWorkspace::new(n, domain.n_nodes());
    let null = run_range(&problem, &plan, 0..plan.b, &mut ws);
    finalize(&problem, &null)
}

/// Split `0..b` into `min(b, 4 · threads)` contiguous ranges of near-equal
/// length, the first `b % chunks` one draw longer. Four chunks per thread, not
/// one: draws differ in cost (a draw whose t-map has more suprathreshold nodes
/// does more union-find work), so the extra chunks let rayon re-balance to the
/// end of the run, while the chunk count still bounds how many
/// [`PermWorkspace`]s get allocated. `b ≥ 1` and `threads ≥ 1`.
#[cfg(feature = "parallel")]
fn chunk_ranges(b: u64, threads: usize) -> Vec<Range<u64>> {
    let chunks = b.min((threads as u64).saturating_mul(4)).max(1);
    let (base, rem) = (b / chunks, b % chunks);
    let mut out = Vec::with_capacity(chunks as usize);
    let mut start = 0u64;
    for i in 0..chunks {
        let len = base + u64::from(i < rem);
        out.push(start..start + len);
        start += len;
    }
    out
}

/// [`tfce_one_sample`] with the draws spread over `threads` OS threads.
/// **Bit-identical to [`tfce_one_sample`] for any `threads`**: the draw stream
/// depends only on `(seed, draw_id)`, and [`finalize`] sorts the merged maxima,
/// so the split cannot move a single bit of the result.
///
/// `threads == 0` means every core rayon can see
/// (`std::thread::available_parallelism`, falling back to 1). `threads == 1`,
/// a one-draw plan, or a pool that fails to build all run the serial path — a
/// pool build failure is a fallback, not an error to the caller.
///
/// The pool is built per call and sized by `threads`, so `RAYON_NUM_THREADS`
/// and rayon's global pool have no effect here and two concurrent callers do
/// not fight over one setting. Arguments, conventions and errors are
/// [`tfce_one_sample`]'s.
///
/// The bit-identity claim is checked on every permutation fixture and thread
/// count by `tests/perm_oracle.rs::g12_threaded_driver_matches_serial_bitwise`;
/// `::g7_split_merge_is_deterministic` checks the same for hand-made splits
/// merged through [`NullPartial`].
///
/// ```
/// use neurostats::{Conn, Domain, TfceParams, Weighting, tfce_one_sample, tfce_one_sample_threads};
/// let dom = Domain::from_volume([3, 1, 1], Conn::Face);
/// let x = [2.0, 0.3, 0.0, 2.5, -0.4, 0.0, 1.8, 0.1, 0.0, 2.2, -0.2, 0.0];
/// let p = TfceParams { e: 0.5, h: 2.0, start: 0.0, step: 1.0, weighting: Weighting::Exact };
/// let serial = tfce_one_sample(&dom, &x, 4, &p, 42, 1000).unwrap();
/// let threaded = tfce_one_sample_threads(&dom, &x, 4, &p, 42, 1000, 0).unwrap();
/// assert_eq!(serial, threaded);
/// ```
#[cfg(feature = "parallel")]
pub fn tfce_one_sample_threads(
    domain: &Domain,
    x: &[f64],
    n: usize,
    params: &TfceParams,
    seed: u64,
    b_requested: u64,
    threads: usize,
) -> Result<TfceInference, NeuroError> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    if b_requested == 0 {
        return Err(NeuroError::InvalidParams);
    }
    let problem = OneSampleProblem::new(domain, x, n, *params, seed)?;
    let plan = sign_flip_plan(n, b_requested);
    let v = domain.n_nodes();

    let serial = || {
        let mut ws = PermWorkspace::new(n, v);
        run_range(&problem, &plan, 0..plan.b, &mut ws)
    };

    let threads = if threads == 0 {
        std::thread::available_parallelism().map_or(1, |t| t.get())
    } else {
        threads
    };
    let null = if threads == 1 || plan.b == 1 {
        serial()
    } else {
        // A pool per call rather than rayon's global one: `threads` then means
        // exactly what this caller passed, and two concurrent callers (Python
        // threads, say) cannot overwrite each other's thread count. Building it
        // costs milliseconds — nothing against a permutation run.
        match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
            Err(_) => serial(),
            Ok(pool) => pool.install(|| {
                chunk_ranges(plan.b, threads)
                    .into_par_iter()
                    .map_init(
                        || PermWorkspace::new(n, v),
                        |ws, r| run_range(&problem, &plan, r, ws),
                    )
                    .reduce(
                        || NullPartial::empty(v),
                        |mut a, b| {
                            a.merge(&b);
                            a
                        },
                    )
            }),
        }
    };
    finalize(&problem, &null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tfce::Weighting;
    use crate::{Conn, Domain};
    use alloc::vec;

    #[test]
    fn plan_is_exact_iff_two_pow_n_fits() {
        assert_eq!(sign_flip_plan(3, 8), SignFlipPlan { b: 8, exact: true });
        assert_eq!(sign_flip_plan(3, 100), SignFlipPlan { b: 8, exact: true });
        assert_eq!(sign_flip_plan(3, 7), SignFlipPlan { b: 7, exact: false });
        assert_eq!(
            sign_flip_plan(70, 5000),
            SignFlipPlan {
                b: 5000,
                exact: false
            }
        );
    }

    #[test]
    fn exact_draw_bits_encode_signs_and_draw_zero_is_identity() {
        let mut s = vec![0.0; 4];
        fill_signs(&SignFlipPlan { b: 16, exact: true }, 4, 0, 0, &mut s);
        assert_eq!(s, [1.0, 1.0, 1.0, 1.0]);
        fill_signs(&SignFlipPlan { b: 16, exact: true }, 4, 0, 0b1010, &mut s);
        assert_eq!(s, [1.0, -1.0, 1.0, -1.0]);
        fill_signs(
            &SignFlipPlan {
                b: 100,
                exact: false,
            },
            4,
            9,
            0,
            &mut s,
        );
        assert_eq!(s, [1.0, 1.0, 1.0, 1.0]);
    }

    fn toy() -> (Domain, Vec<f64>, TfceParams) {
        let dom = Domain::from_volume([3, 1, 1], Conn::Face);
        // n = 4 subjects, V = 3: node 0 strong positive, node 1 noise, node 2 zero.
        let x = vec![
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
        (dom, x, p)
    }

    // Exact enumeration on n = 4: B = 16, identity included, p ≥ 1/16, and the
    // strong node's FWE p is the smallest attainable (1/16 or 2/16: the
    // all-flipped realization mirrors the observed max under Exact/positive tail
    // only if it is also positive — here it is not).
    #[test]
    fn tfce_one_sample_exact_toy() {
        let (dom, x, p) = toy();
        let r = tfce_one_sample(&dom, &x, 4, &p, 1, 1000).unwrap();
        assert_eq!(r.null_max.len(), 16);
        assert!(r.null_max.windows(2).all(|w| w[0] <= w[1]));
        assert!(r.p_fwe.iter().all(|&p| (1.0 / 16.0..=1.0).contains(&p)));
        assert!(r.p_unc.iter().all(|&p| (1.0 / 16.0..=1.0).contains(&p)));
        assert!((r.p_fwe[0] - 1.0 / 16.0).abs() < 1e-15, "{:?}", r.p_fwe);
        assert_eq!(r.tfce_obs[2], 0.0);
        assert_eq!(r.p_fwe[2], 1.0);
    }

    // The split covers 0..b exactly once, in near-equal contiguous
    // pieces, with the count the driver promises.
    #[cfg(feature = "parallel")]
    #[test]
    fn chunk_ranges_partition_the_draws() {
        for b in [1u64, 2, 5, 16, 17, 200, 1000] {
            for threads in [1usize, 2, 3, 7, 64] {
                let r = chunk_ranges(b, threads);
                let want_chunks = b.min(4 * threads as u64);
                assert_eq!(r.len() as u64, want_chunks, "b {b} threads {threads}");
                assert_eq!(r[0].start, 0);
                assert_eq!(r[r.len() - 1].end, b);
                for w in r.windows(2) {
                    assert_eq!(w[0].end, w[1].start, "b {b} threads {threads}: gap/overlap");
                }
                let lens: Vec<u64> = r.iter().map(|x| x.end - x.start).collect();
                let (lo, hi) = (*lens.iter().min().unwrap(), *lens.iter().max().unwrap());
                assert!(hi - lo <= 1, "b {b} threads {threads}: lens {lens:?}");
                assert_eq!(lens.iter().sum::<u64>(), b);
            }
        }
    }

    #[test]
    fn b_zero_is_invalid() {
        let (dom, x, p) = toy();
        assert_eq!(
            tfce_one_sample(&dom, &x, 4, &p, 1, 0),
            Err(NeuroError::InvalidParams)
        );
    }

    #[test]
    fn x_length_must_match_domain() {
        let (dom, x, p) = toy();
        assert_eq!(
            tfce_one_sample(&dom, &x[..8], 4, &p, 1, 16),
            Err(NeuroError::MismatchedLengths {
                expected: 12,
                got: 8
            })
        );
    }
}
