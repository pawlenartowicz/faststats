//! Permutation-loop gates:
//!   G5  — one-sample test, exact enumeration ≈ MNE 1.12.1 (RNG-independent)
//!   G6  — one-sample test, Monte Carlo ≈ MNE, statistically
//!   G7  — split, merge and batch-size determinism of the loop
//!   G8  — one-sample t of the engine == direct two-pass t on flipped data
//!   G9  — `tfce_into` == `tfce` bit-for-bit, workspace reused across fixtures
//!   G11 — `Exact` is the limit of `SmithNichols` as `step → 0`
//!   G12 — threaded driver == serial driver bit-for-bit
//!   G14 — a null extended to `B` draws == one run of `B` draws
mod common;

use common::{Fixture, assert_map_close, load, load_perm};
use commonstats::accum::Mergeable;
use commonstats::gen_sign_flips;
use neurostats::{
    Conn, Domain, NullPartial, PermOpts, PermProblem, PermWorkspace, Store, Tail, TfceInference,
    TfceParams, TfceWorkspace, Weighting, finalize, nuisance_basis, run_range, stat_batch, tfce,
    tfce_into, tfce_ols,
};

const REL: f64 = 1e-12;
const ABS_FLOOR: f64 = 1e-12;

const SMALL: [&str; 7] = [
    "grid4",
    "blob",
    "ties",
    "flat",
    "single",
    "empty_grid",
    "two_clusters",
];

fn params(fx: &Fixture, weighting: Weighting) -> TfceParams {
    TfceParams {
        e: fx.params.e,
        h: fx.params.h,
        start: fx.params.start,
        step: fx.params.step,
        weighting,
    }
}

/// The engine's one-sample t (no tested regressor, no confounds) against a
/// direct two-pass mean/var on the sign-flipped data, for the engine's own
/// draws rebuilt from `commonstats::gen_sign_flips`. Prints the worst relative
/// error seen.
#[test]
fn g8_one_sample_t_matches_direct() {
    let mut seed: u64 = 0xD1B5_4A32_D192_ED03;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let (n, v) = (17usize, 200usize);
    // Signal so some nodes have |t| ≫ 1 (where 1 − b² loses digits).
    let y: Vec<f64> = (0..n * v)
        .map(|k| {
            let node = k % v;
            let shift = if node < 40 { 1.5 } else { 0.0 };
            shift + (next() - 0.5) * 2.0
        })
        .collect();
    let dom = Domain::from_volume([v, 1, 1], Conn::Face);
    let p = TfceParams {
        e: 0.5,
        h: 2.0,
        start: 0.0,
        step: 1.0,
        weighting: Weighting::Exact,
    };
    let rows: Vec<usize> = (0..n).collect();
    let store = Store::from_f64(&y, n, v, &rows, &[]).unwrap();
    let (run_seed, b) = (11u64, 51u64);
    let mut opts = PermOpts::new(p, Tail::Positive, run_seed, b);
    opts.batch = 7;
    let problem = PermProblem::ols(&dom, &store, None, &opts).unwrap();
    assert_eq!(problem.draws(), 0..b);
    let mut ws = PermWorkspace::new(&problem);
    let mut signs = vec![1.0; n];
    let mut worst = 0.0f64;
    for first in (0..b).step_by(7) {
        let last = b.min(first + 7);
        let maps = stat_batch(&problem, first..last, &mut ws);
        for draw in first..last {
            if draw == 0 {
                signs.fill(1.0);
            } else {
                gen_sign_flips(n, draw, run_seed, &mut signs).unwrap();
            }
            let t = &maps[(draw - first) as usize * v..][..v];
            for node in 0..v {
                let vals: Vec<f64> = (0..n).map(|i| signs[i] * y[i * v + node]).collect();
                let mean = vals.iter().sum::<f64>() / n as f64;
                let var = vals.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
                let want = mean / (var / n as f64).sqrt();
                let err = (t[node] - want).abs() / want.abs().max(1e-300);
                worst = worst.max(err);
                assert!(
                    err <= REL,
                    "G8 draw {draw} node {node}: got {} want {want} rel {err:e}",
                    t[node]
                );
            }
        }
    }
    println!("G8 worst rel error {worst:.3e}");
}

/// One workspace across every fixture (node counts and threshold counts all
/// differ, so grow-on-demand and reuse are both exercised).
#[test]
fn g9_tfce_into_matches_tfce_bitwise() {
    let mut ws = TfceWorkspace::new(0);
    for name in SMALL.iter().copied().chain(["bench"]) {
        let fx = load(name);
        let dom = fx.domain();
        for w in [
            Weighting::MneStep,
            Weighting::SmithNichols,
            Weighting::Nilearn,
        ] {
            let p = params(&fx, w);
            let want = tfce(&dom, &fx.stat, &p).unwrap();
            let mut got = vec![f64::NAN; dom.n_nodes()];
            tfce_into(&dom, &fx.stat, &p, &mut ws, &mut got).unwrap();
            assert_eq!(got, want, "G9 {name} {w:?}");
        }
    }
}

/// Left Riemann sum → integral: node-wise max |SmithNichols(step) − Exact| is
/// O(step). Bound C·step with C taken from the larger of the two coarse-step
/// ratios (0.4 and 0.2): `2 · max(err(0.4)/0.4, err(0.2)/0.2)` — a single
/// coarse step can land a grid edge exactly on a tie value and understate the
/// true per-step error, so two steps guard against one of them being lucky.
/// No per-step ordering asserted (band edges land differently on each grid).
#[test]
fn g11_exact_is_limit_of_smith_nichols() {
    for name in SMALL {
        let fx = load(name);
        let dom = fx.domain();
        let exact = tfce(&dom, &fx.stat, &params(&fx, Weighting::Exact)).unwrap();
        let err = |step: f64| -> f64 {
            let p = TfceParams {
                step,
                ..params(&fx, Weighting::SmithNichols)
            };
            let sn = tfce(&dom, &fx.stat, &p).unwrap();
            sn.iter()
                .zip(&exact)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
        };
        let c = 2.0 * (err(0.4) / 0.4).max(err(0.2) / 0.2);
        for step in [0.1, 0.025] {
            let e = err(step);
            assert!(
                e <= c * step + 1e-12,
                "G11 {name}: err({step}) = {e:e} > C·step = {:e}",
                c * step
            );
        }
    }
}

/// Synthetic data on the `blob` fixture's domain: `n` subjects, the fixture
/// map as a shared effect plus uniform noise.
fn blob_data(fx: &Fixture, n: usize, v: usize) -> Vec<f64> {
    let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n * v)
        .map(|k| 0.4 * fx.stat[k % v] + (next() - 0.5) * 2.0)
        .collect()
}

/// A tested regressor and two confounds for `n` subjects (fixed LCG).
fn design(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let tested: Vec<f64> = (0..n).map(|_| next()).collect();
    let confounds: Vec<f64> = (0..2 * n).map(|_| 1.0 + next()).collect();
    (tested, confounds)
}

/// Any partition of the draws, merged in any order, with any batch size,
/// finalizes bit-identically to the serial run with batch 1. Sign flips (the
/// intercept is tested) and row shuffles (a regressor with confounds), three
/// tails.
#[test]
fn g7_split_merge_is_deterministic() {
    let fx = load("blob");
    let dom = fx.domain();
    let (n, v) = (12usize, dom.n_nodes());
    let y = blob_data(&fx, n, v);
    let (tested, confounds) = design(n);
    let p = params(&fx, Weighting::Exact);
    let rows: Vec<usize> = (0..n).collect();
    for (label, tested, confounds) in [
        ("sign flips", None, &[][..]),
        ("row shuffles", Some(&tested[..]), &confounds[..]),
    ] {
        let basis = nuisance_basis(confounds, n, tested.is_some()).unwrap();
        let store = Store::from_f64(&y, n, v, &rows, &basis).unwrap();
        for tail in [Tail::Positive, Tail::Negative, Tail::TwoSided] {
            let run = |batch: usize, cuts: &[u64], order: &[usize]| -> TfceInference {
                let mut opts = PermOpts::new(p, tail, 99, 120);
                opts.batch = batch;
                let problem = PermProblem::ols(&dom, &store, tested, &opts).unwrap();
                assert_eq!(problem.draws(), 0..120);
                let mut ws = PermWorkspace::new(&problem);
                let parts: Vec<NullPartial> = cuts
                    .windows(2)
                    .map(|w| run_range(&problem, w[0]..w[1], &mut ws))
                    .collect();
                let mut acc = NullPartial::empty(v);
                for &k in order {
                    acc.merge(&parts[k]);
                }
                finalize(&problem, &acc).unwrap()
            };
            let serial = run(1, &[0, 120], &[0]);
            for batch in [3usize, 8, 1000] {
                assert_eq!(
                    run(batch, &[0, 120], &[0]),
                    serial,
                    "G7 {label} {tail:?} batch {batch}"
                );
                for order in [[2usize, 0, 3, 1], [3, 2, 1, 0]] {
                    assert_eq!(
                        run(batch, &[0, 7, 60, 61, 120], &order),
                        serial,
                        "G7 {label} {tail:?} batch {batch} order {order:?}"
                    );
                }
            }
        }
    }
}

/// A null that does not hold exactly the run's draws is refused.
#[test]
fn finalize_refuses_an_incomplete_null() {
    let dom = Domain::from_volume([3, 1, 1], Conn::Face);
    let y = [2.0, 0.3, 0.0, 2.5, -0.4, 0.0, 1.8, 0.1, 0.0, 2.2, -0.2, 0.0];
    let p = TfceParams {
        e: 0.5,
        h: 2.0,
        start: 0.0,
        step: 1.0,
        weighting: Weighting::Exact,
    };
    let store = Store::from_f64(&y, 4, 3, &[0, 1, 2, 3], &[]).unwrap();
    let opts = PermOpts::new(p, Tail::Positive, 1, 10);
    let problem = PermProblem::ols(&dom, &store, None, &opts).unwrap();
    let mut ws = PermWorkspace::new(&problem);
    assert_eq!(
        finalize(&problem, &NullPartial::empty(3)),
        Err(neurostats::NeuroError::InvalidParams)
    );
    assert_eq!(
        finalize(&problem, &run_range(&problem, 0..9, &mut ws)),
        Err(neurostats::NeuroError::InvalidParams)
    );
    assert_eq!(
        finalize(&problem, &NullPartial::empty(2)),
        Err(neurostats::NeuroError::MismatchedLengths {
            expected: 3,
            got: 2
        })
    );
    assert!(finalize(&problem, &run_range(&problem, 0..10, &mut ws)).is_ok());
}

fn perm_params(fx: &common::PermFixture) -> TfceParams {
    TfceParams {
        e: fx.params.e,
        h: fx.params.h,
        start: fx.params.start,
        step: fx.params.step,
        weighting: Weighting::MneStep,
    }
}

/// The one-sample test of `tfce_ols`: no tested regressor, no confounds,
/// positive tail.
fn one_sample(fx: &common::PermFixture) -> TfceInference {
    let opts = PermOpts::new(perm_params(fx), Tail::Positive, fx.seed, fx.n_permutations);
    tfce_ols(&fx.domain(), &fx.x, fx.n, None, &[], &opts).unwrap()
}

/// Exact enumeration: both sides run every sign pattern, so the null is
/// RNG-independent. MNE 1.12.1 (`_get_1samp_orders`, tail=1) takes
/// `bin_perm_rep` rows `1..2^n−1` with `signs = 2·order − 1`, so it never
/// evaluates the full-negation pattern (row 0) and counts the identity twice
/// (row `2^n−1` plus the prepended observed statistic). neurostats
/// enumerates each pattern once, so MNE's `h0` is compared after swapping one
/// copy of the identity max for the full-negation max; that swapped-in value
/// is not MNE-sourced, it is computed by neurostats itself (`tfce(-t_obs)`,
/// exact because negating every subject negates `t` exactly). (a) sorted
/// maxima == corrected MNE H0, rel 1e-12; (b) |p_fwe − p_mne| ≤ 1/B per node
/// (a one-element swap moves any p by at most 1/B); (c) stat_obs rel 1e-12;
/// (d) tfce_obs rel 1e-12.
#[test]
fn g5_exact_enumeration_matches_mne() {
    for name in ["exact_n6_grid4", "exact_n8_blob"] {
        let fx = load_perm(name);
        assert!(fx.exact);
        let dom = fx.domain();
        let r = one_sample(&fx);
        let b = fx.h0.len();
        assert_eq!(r.null.maxima.len(), b, "G5 {name}: B");

        let neg: Vec<f64> = r.stat_obs.iter().map(|t| -t).collect();
        let full_neg_max = tfce(&dom, &neg, &perm_params(&fx))
            .unwrap()
            .iter()
            .fold(0.0f64, |m, &v| m.max(v));
        let ident_max = r.null.tfce_obs.iter().fold(0.0f64, |m, &v| m.max(v));

        let mut want = fx.h0.clone();
        let scale = want.iter().fold(0.0f64, |m, &w| m.max(w.abs()));
        let i = want
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (**a - ident_max).abs().total_cmp(&(**b - ident_max).abs()))
            .map(|(i, _)| i)
            .expect("h0 non-empty");
        assert!(
            (want[i] - ident_max).abs() <= REL * ident_max + ABS_FLOOR * scale,
            "G5 {name}: no h0 entry close to ident_max {ident_max}, closest was {}",
            want[i]
        );
        want.remove(i);
        want.push(full_neg_max);
        want.sort_unstable_by(f64::total_cmp);

        assert_map_close(
            &format!("G5 {name} maxima"),
            &r.null.maxima,
            &want,
            REL,
            ABS_FLOOR,
        );
        assert_map_close(
            &format!("G5 {name} stat_obs"),
            &r.stat_obs,
            &fx.t_obs,
            REL,
            ABS_FLOOR,
        );
        assert_map_close(
            &format!("G5 {name} tfce_obs"),
            &r.null.tfce_obs,
            &fx.tfce_obs,
            REL,
            ABS_FLOOR,
        );
        for (v, (&p, &q)) in r.p_fwe.iter().zip(&fx.p_fwe).enumerate() {
            assert!(
                (p - q).abs() <= 1.0 / b as f64 + 1e-15,
                "G5 {name} p_fwe[{v}]: got {p} want {q}"
            );
        }
    }
}

/// Monte Carlo: different RNGs, and both p-maps are B-draw estimates, so the
/// difference has variance 2·p(1−p)/B; a 4σ band node-wise plus a Kolmogorov
/// distance of 4/sqrt(B) between the two sorted null-max curves.
#[test]
fn g6_monte_carlo_matches_mne_statistically() {
    let fx = load_perm("mc_n20_blob");
    assert!(!fx.exact);
    let r = one_sample(&fx);
    let null_max = &r.null.maxima;
    let b = fx.h0.len();
    assert_eq!(null_max.len(), b);
    let bf = b as f64;
    assert_map_close("G6 stat_obs", &r.stat_obs, &fx.t_obs, REL, ABS_FLOOR);
    assert_map_close(
        "G6 tfce_obs",
        &r.null.tfce_obs,
        &fx.tfce_obs,
        REL,
        ABS_FLOOR,
    );
    let mut worst_ratio = 0.0f64;
    for (v, (&p, &q)) in r.p_fwe.iter().zip(&fx.p_fwe).enumerate() {
        let pbar = 0.5 * (p + q);
        let tol = 4.0 * (2.0 * pbar * (1.0 - pbar) / bf).sqrt() + 1.0 / bf;
        worst_ratio = worst_ratio.max((p - q).abs() / tol);
        assert!(
            (p - q).abs() <= tol,
            "G6 p_fwe[{v}]: got {p} want {q} tol {tol:.4}"
        );
    }
    println!("G6 worst |Δp|/tol ratio {worst_ratio:.3}");
    // Kolmogorov distance between the two empirical null-max CDFs.
    let mut all: Vec<f64> = null_max.iter().chain(&fx.h0).copied().collect();
    all.sort_unstable_by(f64::total_cmp);
    let mut ks = 0.0f64;
    for &x in &all {
        let fa = null_max.partition_point(|&m| m <= x) as f64 / bf;
        let fb = fx.h0.partition_point(|&m| m <= x) as f64 / bf;
        ks = ks.max((fa - fb).abs());
    }
    println!("G6 KS distance {ks:.4} (bound {:.4})", 4.0 / bf.sqrt());
    assert!(
        ks <= 4.0 / bf.sqrt(),
        "G6 null_max KS distance {ks:.4} > {:.4}",
        4.0 / bf.sqrt()
    );
}

/// One G12 case: a model plus its permutation settings.
#[cfg(feature = "parallel")]
struct Case<'a> {
    label: String,
    dom: &'a Domain,
    y: &'a [f64],
    n: usize,
    tested: Option<&'a [f64]>,
    confounds: &'a [f64],
    p: TfceParams,
    tail: Tail,
    seed: u64,
    b: u64,
}

/// G12 — the threaded driver is bit-identical to the serial one for thread
/// counts 1, 2, 3 and 7 and for several batch sizes: on every permutation
/// fixture (exact enumeration and Monte Carlo), on two toy problems with
/// fewer draws than chunks, and on a model with a tested regressor and
/// confounds (row shuffles, two-sided, the nilearn weighting, whose two
/// passes share one threshold grid per draw).
#[cfg(feature = "parallel")]
#[test]
fn g12_threaded_driver_matches_serial_bitwise() {
    use neurostats::tfce_ols_threads;

    let toy_dom = Domain::from_volume([3, 1, 1], Conn::Face);
    // 4 subjects × 3 nodes, subject-major; node 0 strong positive, node 2 zero.
    let toy_y = vec![
        2.0, 0.3, 0.0, //
        2.5, -0.4, 0.0, //
        1.8, 0.1, 0.0, //
        2.2, -0.2, 0.0,
    ];
    let toy_p = TfceParams {
        e: 0.5,
        h: 2.0,
        start: 0.0,
        step: 1.0,
        weighting: Weighting::Exact,
    };
    // B = 16 (2^4 ≤ 1000, exact) and B = 10 (Monte Carlo); both < 4 · 7.
    let mut cases = vec![
        Case {
            label: "toy exact B=16".into(),
            dom: &toy_dom,
            y: &toy_y,
            n: 4,
            tested: None,
            confounds: &[],
            p: toy_p,
            tail: Tail::Positive,
            seed: 1,
            b: 1000,
        },
        Case {
            label: "toy mc B=10".into(),
            dom: &toy_dom,
            y: &toy_y,
            n: 4,
            tested: None,
            confounds: &[],
            p: toy_p,
            tail: Tail::Positive,
            seed: 1,
            b: 10,
        },
    ];

    let fx: Vec<common::PermFixture> = ["exact_n6_grid4", "exact_n8_blob", "mc_n20_blob"]
        .iter()
        .map(|name| load_perm(name))
        .collect();
    let doms: Vec<Domain> = fx.iter().map(|f| f.domain()).collect();
    for (f, dom) in fx.iter().zip(&doms) {
        cases.push(Case {
            label: format!("fixture n={} B={}", f.n, f.n_permutations),
            dom,
            y: &f.x,
            n: f.n,
            tested: None,
            confounds: &[],
            p: perm_params(f),
            tail: Tail::Positive,
            seed: f.seed,
            b: f.n_permutations,
        });
    }

    let blob = load("blob");
    let blob_dom = blob.domain();
    let n = 12usize;
    let blob_y = blob_data(&blob, n, blob_dom.n_nodes());
    let (tested, confounds) = design(n);
    cases.push(Case {
        label: "regressor with confounds, two-sided, nilearn weighting, B=120".into(),
        dom: &blob_dom,
        y: &blob_y,
        n,
        tested: Some(&tested),
        confounds: &confounds,
        p: params(&blob, Weighting::Nilearn),
        tail: Tail::TwoSided,
        seed: 5,
        b: 120,
    });

    for c in cases {
        let mut opts = PermOpts::new(c.p, c.tail, c.seed, c.b);
        let want = tfce_ols(c.dom, c.y, c.n, c.tested, c.confounds, &opts).unwrap();
        for batch in [3usize, opts.batch] {
            opts.batch = batch;
            for threads in [1usize, 2, 3, 7] {
                let got = tfce_ols_threads(c.dom, c.y, c.n, c.tested, c.confounds, &opts, threads)
                    .unwrap();
                assert_eq!(got, want, "G12 {} batch {batch} threads {threads}", c.label);
            }
        }
    }
}

/// G14 — extending a null. A run of `B_old` draws continued to `B` equals one
/// run of `B` draws, bit for bit: in one step or two, serial or threaded, and
/// with another batch size in the continuation. Row shuffles with confounds
/// and Monte Carlo sign flips, three tails.
#[test]
fn g14_extending_a_null_equals_one_run() {
    let fx = load("blob");
    let dom = fx.domain();
    let (n, v) = (12usize, dom.n_nodes());
    let y = blob_data(&fx, n, v);
    let (tested, confounds) = design(n);
    let p = params(&fx, Weighting::Exact);
    for (label, tested, confounds) in [
        ("sign flips", None, &[][..]),
        ("row shuffles", Some(&tested[..]), &confounds[..]),
    ] {
        for tail in [Tail::Positive, Tail::Negative, Tail::TwoSided] {
            let run = |b: u64, batch: usize, extend: Option<&neurostats::NullState>| {
                let mut opts = PermOpts::new(p, tail, 31, b);
                opts.batch = batch;
                opts.extend = extend;
                tfce_ols(&dom, &y, n, tested, confounds, &opts).unwrap()
            };
            let whole = run(100, 8, None);
            let first = run(40, 8, None);
            assert_eq!(first.null.maxima.len(), 40);
            assert_eq!(
                run(100, 8, Some(&first.null)),
                whole,
                "G14 {label} {tail:?}"
            );
            assert_eq!(
                run(100, 3, Some(&first.null)),
                whole,
                "G14 {label} {tail:?} batch 3"
            );
            let second = run(71, 5, Some(&first.null));
            assert_eq!(second, run(71, 8, None), "G14 {label} {tail:?} 40 -> 71");
            assert_eq!(
                run(100, 8, Some(&second.null)),
                whole,
                "G14 {label} {tail:?} 40 -> 71 -> 100"
            );
            // A continuation of one draw.
            assert_eq!(run(41, 8, Some(&first.null)), run(41, 8, None));

            #[cfg(feature = "parallel")]
            for threads in [1usize, 2, 3, 7] {
                let mut opts = PermOpts::new(p, tail, 31, 100);
                opts.extend = Some(&first.null);
                let got =
                    neurostats::tfce_ols_threads(&dom, &y, n, tested, confounds, &opts, threads)
                        .unwrap();
                assert_eq!(got, whole, "G14 {label} {tail:?} threads {threads}");
            }
        }
    }
}

/// G14 — through the loop pieces: a problem that continues a null runs only
/// the new draws, and `finalize` merges them with the stored ones.
#[test]
fn g14_a_continued_problem_runs_only_the_new_draws() {
    let dom = Domain::from_volume([3, 2, 1], Conn::Face);
    let (n, v) = (10usize, dom.n_nodes());
    let (tested, confounds) = design(n);
    let mut seed: u64 = 0x1234_5678_9ABC_DEF1;
    let y: Vec<f64> = (0..n * v)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect();
    let p = TfceParams {
        e: 0.5,
        h: 2.0,
        start: 0.0,
        step: 1.0,
        weighting: Weighting::Exact,
    };
    let basis = nuisance_basis(&confounds, n, true).unwrap();
    let rows: Vec<usize> = (0..n).collect();
    let store = Store::from_f64(&y, n, v, &rows, &basis).unwrap();
    let first = tfce_ols(
        &dom,
        &y,
        n,
        Some(&tested),
        &confounds,
        &PermOpts::new(p, Tail::TwoSided, 4, 30),
    )
    .unwrap();

    let mut opts = PermOpts::new(p, Tail::TwoSided, 4, 50);
    opts.extend = Some(&first.null);
    let problem = PermProblem::ols(&dom, &store, Some(&tested), &opts).unwrap();
    assert_eq!(problem.draws(), 30..50);
    let mut ws = PermWorkspace::new(&problem);
    let added = run_range(&problem, problem.draws(), &mut ws);
    assert_eq!(added.maxima().len(), 20);
    let merged = finalize(&problem, &added).unwrap();
    assert_eq!(merged.null.maxima.len(), 50);
    let whole = tfce_ols(
        &dom,
        &y,
        n,
        Some(&tested),
        &confounds,
        &PermOpts::new(p, Tail::TwoSided, 4, 50),
    )
    .unwrap();
    assert_eq!(merged, whole);
}

/// G14 — a null is continued only by the test that made it. Every other case
/// is `InvalidParams`.
#[test]
fn g14_a_null_that_does_not_fit_is_refused() {
    let dom = Domain::from_volume([3, 2, 1], Conn::Face);
    let (n, v) = (6usize, dom.n_nodes());
    let (tested, confounds) = design(n);
    let mut seed: u64 = 0x0FED_CBA9_8765_4321;
    let y: Vec<f64> = (0..n * v)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect();
    let p = TfceParams {
        e: 0.5,
        h: 2.0,
        start: 0.0,
        step: 1.0,
        weighting: Weighting::Exact,
    };
    let confound = &confounds[..n];
    let base = PermOpts::new(p, Tail::TwoSided, 4, 30);
    let first = tfce_ols(&dom, &y, n, Some(&tested), confound, &base).unwrap();
    let refused = Err(neurostats::NeuroError::InvalidParams);
    let extend = |y: &[f64], confounds: &[f64], opts: PermOpts| {
        tfce_ols(&dom, y, n, Some(&tested), confounds, &opts).map(|r| r.null.maxima.len())
    };
    fn with(p: TfceParams, state: &neurostats::NullState, n_perm: u64) -> PermOpts<'_> {
        let mut opts = PermOpts::new(p, Tail::TwoSided, 4, n_perm);
        opts.extend = Some(state);
        opts
    }
    let state = &first.null;
    assert_eq!(extend(&y, confound, with(p, state, 31)), Ok(31));

    // The budget is the total number of draws and must grow.
    assert_eq!(extend(&y, confound, with(p, state, 30)), refused);
    assert_eq!(extend(&y, confound, with(p, state, 7)), refused);
    // Another seed, tail or TFCE setting.
    assert_eq!(
        extend(
            &y,
            confound,
            PermOpts {
                seed: 5,
                ..with(p, state, 60)
            }
        ),
        refused
    );
    assert_eq!(
        extend(
            &y,
            confound,
            PermOpts {
                tail: Tail::Positive,
                ..with(p, state, 60)
            }
        ),
        refused
    );
    let other = TfceParams { e: 0.6, ..p };
    assert_eq!(
        extend(
            &y,
            confound,
            PermOpts {
                params: other,
                ..with(p, state, 60)
            }
        ),
        refused
    );
    // Other data: the observed TFCE map differs.
    let mut y2 = y.clone();
    y2[0] += 1e-9;
    assert_eq!(extend(&y2, confound, with(p, state, 60)), refused);
    // Another design: no confound (other q), or another confound (same q).
    assert_eq!(extend(&y, &[], with(p, state, 60)), refused);
    assert_eq!(extend(&y, &confounds[n..], with(p, state, 60)), refused);
    // A state that is not a null of this problem.
    let mut broken = state.clone();
    broken.exceed.pop();
    assert_eq!(extend(&y, confound, with(p, &broken, 60)), refused);
    let mut broken = state.clone();
    broken.exceed[0] = 31;
    assert_eq!(extend(&y, confound, with(p, &broken, 60)), refused);
    let mut broken = state.clone();
    broken.maxima.clear();
    assert_eq!(extend(&y, confound, with(p, &broken, 60)), refused);

    // Sign flips, n = 6: a budget of 2^6 = 64 or more enumerates the
    // patterns, so it cannot continue a Monte Carlo null of 50 draws, and an
    // enumerated null cannot be continued at all.
    let flips = |n_perm: u64, state: Option<&neurostats::NullState>| {
        let mut opts = PermOpts::new(p, Tail::Positive, 4, n_perm);
        opts.extend = state;
        tfce_ols(&dom, &y, n, None, &[], &opts)
    };
    let mc = flips(50, None).unwrap();
    assert_eq!(flips(63, Some(&mc.null)).unwrap(), flips(63, None).unwrap());
    assert_eq!(
        flips(64, Some(&mc.null)).map(|_| ()),
        refused.map(|_: usize| ())
    );
    assert_eq!(
        flips(5000, Some(&mc.null)).map(|_| ()),
        refused.map(|_: usize| ())
    );
    let enumerated = flips(1000, None).unwrap();
    assert_eq!(enumerated.null.maxima.len(), 64);
    assert_eq!(
        flips(2000, Some(&enumerated.null)).map(|_| ()),
        refused.map(|_: usize| ())
    );
}
