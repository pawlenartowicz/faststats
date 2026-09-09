//! L2 parity gates (design spec §5):
//!   G5  — exact enumeration ≈ MNE 1.12.1 (RNG-independent)
//!   G6  — Monte Carlo ≈ MNE, statistically
//!   G7  — split/merge determinism of `NullPartial`
//!   G8  — sufficient-statistic t == direct two-pass t
//!   G9  — `tfce_into` == `tfce` bit-for-bit, workspace reused across fixtures
//!   G10 — `Weighting::Exact`: union-find ≈ naive
//!   G11 — `Exact` is the limit of `SmithNichols` as `step → 0`
mod common;

use common::{Fixture, assert_map_close, load, load_perm};
use commonstats::accum::Mergeable;
use neurostats::{
    Conn, Domain, NullPartial, OneSampleProblem, OneSampleT, PermWorkspace, TfceParams,
    TfceWorkspace, Weighting, finalize, run_range, sign_flip_plan, tfce, tfce_into, tfce_naive,
    tfce_one_sample,
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

/// Sufficient-statistic t (Q − S²/n) vs a direct two-pass mean/var on the
/// flipped data. Prints the worst relative error seen.
#[test]
fn g8_sufficient_stat_t_matches_direct() {
    let mut seed: u64 = 0xD1B5_4A32_D192_ED03;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let (n, v) = (17usize, 200usize);
    // Signal so some nodes have |t| ≫ 1 (where Q − S²/n loses digits).
    let x: Vec<f64> = (0..n * v)
        .map(|k| {
            let node = k % v;
            let shift = if node < 40 { 1.5 } else { 0.0 };
            shift + (next() - 0.5) * 2.0
        })
        .collect();
    let st = OneSampleT::new(&x, n).unwrap();
    let mut t = vec![0.0; v];
    let mut worst = 0.0f64;
    for draw in 0..50 {
        let signs: Vec<f64> = (0..n)
            .map(|_| if next() < 0.5 { -1.0 } else { 1.0 })
            .collect();
        st.t_map(&x, &signs, &mut t);
        for node in 0..v {
            let vals: Vec<f64> = (0..n).map(|i| signs[i] * x[i * v + node]).collect();
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
        for w in [Weighting::MneStep, Weighting::SmithNichols] {
            let p = params(&fx, w);
            let want = tfce(&dom, &fx.stat, &p).unwrap();
            let mut got = vec![f64::NAN; dom.n_nodes()];
            tfce_into(&dom, &fx.stat, &p, &mut ws, &mut got).unwrap();
            assert_eq!(got, want, "G9 {name} {w:?}");
        }
    }
}

#[test]
fn g10_exact_union_find_matches_naive() {
    for name in SMALL {
        let fx = load(name);
        let dom = fx.domain();
        let p = params(&fx, Weighting::Exact);
        let fast = tfce(&dom, &fx.stat, &p).unwrap();
        let slow = tfce_naive(&dom, &fx.stat, &p).unwrap();
        assert_map_close(&format!("G10 {name}"), &fast, &slow, REL, ABS_FLOOR);
    }
    // Synthetic: same LCG/mask/quantisation recipe as G1 synthetic.
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for (dims, conn) in [
        ([6, 5, 4], Conn::Face),
        ([5, 5, 5], Conn::Edge),
        ([4, 7, 3], Conn::Vertex),
    ] {
        for trial in 0..4 {
            let n_vox = dims[0] * dims[1] * dims[2];
            let mask: Vec<bool> = (0..n_vox).map(|_| next() < 0.75).collect();
            let dom = Domain::from_mask(&mask, dims, conn).unwrap();
            let stat: Vec<f64> = (0..dom.n_nodes())
                .map(|_| (next() * 6.0 * 4.0).round() / 4.0)
                .collect();
            let p = TfceParams {
                e: 0.5,
                h: 2.0,
                start: 0.25,
                step: 0.5,
                weighting: Weighting::Exact,
            };
            let fast = tfce(&dom, &stat, &p).unwrap();
            let slow = tfce_naive(&dom, &stat, &p).unwrap();
            assert_map_close(
                &format!("G10 synth {dims:?} {conn:?} t{trial}"),
                &fast,
                &slow,
                REL,
                ABS_FLOOR,
            );
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

/// Any partition of 0..B merged in any order finalizes bit-identically to the
/// serial run.
#[test]
fn g7_split_merge_is_deterministic() {
    let fx = load("blob");
    let dom = fx.domain();
    let (n, v) = (12usize, dom.n_nodes());
    let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let x: Vec<f64> = (0..n * v)
        .map(|k| 0.4 * fx.stat[k % v] + (next() - 0.5) * 2.0)
        .collect();
    let p = params(&fx, Weighting::Exact);
    let problem = OneSampleProblem::new(&dom, &x, n, p, 99).unwrap();
    let plan = sign_flip_plan(n, 500);
    assert!(!plan.exact && plan.b == 500);
    let mut ws = PermWorkspace::new(n, v);
    let serial = finalize(&problem, &run_range(&problem, &plan, 0..plan.b, &mut ws)).unwrap();
    let parts = [
        run_range(&problem, &plan, 0..7, &mut ws),
        run_range(&problem, &plan, 7..250, &mut ws),
        run_range(&problem, &plan, 250..251, &mut ws),
        run_range(&problem, &plan, 251..500, &mut ws),
    ];
    for order in [[2usize, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]] {
        let mut acc = NullPartial::empty(v);
        for &k in &order {
            acc.merge(&parts[k]);
        }
        let merged = finalize(&problem, &acc).unwrap();
        assert_eq!(merged.p_fwe, serial.p_fwe, "G7 p_fwe order {order:?}");
        assert_eq!(merged.p_unc, serial.p_unc, "G7 p_unc order {order:?}");
        assert_eq!(
            merged.null_max, serial.null_max,
            "G7 null_max order {order:?}"
        );
    }
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

/// Exact enumeration: both sides run every sign pattern, so the null is
/// RNG-independent. MNE 1.12.1 (`_get_1samp_orders`, tail=1) takes
/// `bin_perm_rep` rows `1..2^n−1` with `signs = 2·order − 1`, so it never
/// evaluates the full-negation pattern (row 0) and counts the identity twice
/// (row `2^n−1` plus the prepended observed statistic). neurostats
/// enumerates each pattern once, so MNE's `h0` is compared after swapping one
/// copy of the identity max for the full-negation max; that swapped-in value
/// is not MNE-sourced, it is computed by neurostats itself (`tfce(-t_obs)`,
/// exact because negating every subject negates `t` exactly). (a) sorted
/// null_max == corrected MNE H0, rel 1e-12; (b) |p_fwe − p_mne| ≤ 1/B per node
/// (a one-element swap moves any p by at most 1/B); (c) t_obs rel 1e-12; (d)
/// tfce_obs rel 1e-12.
#[test]
fn g5_exact_enumeration_matches_mne() {
    for name in ["exact_n6_grid4", "exact_n8_blob"] {
        let fx = load_perm(name);
        assert!(fx.exact);
        let dom = fx.domain();
        let r = tfce_one_sample(
            &dom,
            &fx.x,
            fx.n,
            &perm_params(&fx),
            fx.seed,
            fx.n_permutations,
        )
        .unwrap();
        let b = fx.h0.len();
        assert_eq!(r.null_max.len(), b, "G5 {name}: B");

        let neg: Vec<f64> = r.t_obs.iter().map(|t| -t).collect();
        let full_neg_max = tfce(&dom, &neg, &perm_params(&fx))
            .unwrap()
            .iter()
            .fold(0.0f64, |m, &v| m.max(v));
        let ident_max = r.tfce_obs.iter().fold(0.0f64, |m, &v| m.max(v));

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
            &format!("G5 {name} null_max"),
            &r.null_max,
            &want,
            REL,
            ABS_FLOOR,
        );
        assert_map_close(
            &format!("G5 {name} t_obs"),
            &r.t_obs,
            &fx.t_obs,
            REL,
            ABS_FLOOR,
        );
        assert_map_close(
            &format!("G5 {name} tfce_obs"),
            &r.tfce_obs,
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
    let dom = fx.domain();
    let r = tfce_one_sample(
        &dom,
        &fx.x,
        fx.n,
        &perm_params(&fx),
        fx.seed,
        fx.n_permutations,
    )
    .unwrap();
    let b = fx.h0.len();
    assert_eq!(r.null_max.len(), b);
    let bf = b as f64;
    assert_map_close("G6 t_obs", &r.t_obs, &fx.t_obs, REL, ABS_FLOOR);
    assert_map_close("G6 tfce_obs", &r.tfce_obs, &fx.tfce_obs, REL, ABS_FLOOR);
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
    let mut all: Vec<f64> = r.null_max.iter().chain(&fx.h0).copied().collect();
    all.sort_unstable_by(f64::total_cmp);
    let mut ks = 0.0f64;
    for &x in &all {
        let fa = r.null_max.partition_point(|&m| m <= x) as f64 / bf;
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

/// One G12 case: a problem plus its permutation settings.
#[cfg(feature = "parallel")]
struct Case<'a> {
    label: String,
    dom: &'a Domain,
    x: &'a [f64],
    n: usize,
    p: TfceParams,
    seed: u64,
    b: u64,
}

/// G12 — the threaded driver is bit-identical to the serial one, on every
/// permutation fixture (exact enumeration and Monte Carlo) and on two toy
/// problems whose `B` is below `4 · threads`, so every chunk is a single draw.
#[cfg(feature = "parallel")]
#[test]
fn g12_threaded_driver_matches_serial_bitwise() {
    use neurostats::tfce_one_sample_threads;

    let toy_dom = Domain::from_volume([3, 1, 1], Conn::Face);
    // 4 subjects × 3 nodes, subject-major; node 0 strong positive, node 2 zero.
    let toy_x = vec![
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
            x: &toy_x,
            n: 4,
            p: toy_p,
            seed: 1,
            b: 1000,
        },
        Case {
            label: "toy mc B=10".into(),
            dom: &toy_dom,
            x: &toy_x,
            n: 4,
            p: toy_p,
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
            x: &f.x,
            n: f.n,
            p: perm_params(f),
            seed: f.seed,
            b: f.n_permutations,
        });
    }

    for c in cases {
        let want = tfce_one_sample(c.dom, c.x, c.n, &c.p, c.seed, c.b).unwrap();
        for threads in [1usize, 2, 3, 7] {
            let got = tfce_one_sample_threads(c.dom, c.x, c.n, &c.p, c.seed, c.b, threads).unwrap();
            assert_eq!(got, want, "G12 {} threads {threads}", c.label);
        }
    }
}
