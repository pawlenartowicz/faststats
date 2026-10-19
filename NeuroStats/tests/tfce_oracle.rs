//! TFCE parity gates:
//!   G1 — union-find `tfce` ≈ `tfce_naive`, all four weightings (internal, no oracle)
//!   G2 — `tfce_naive` ≈ MNE 1.12.1 golden (`MneStep`)
//!   G3 — `tfce` ≈ MNE golden (`MneStep`)
//!   G4 — `tfce_naive` ≈ independent NumPy Smith–Nichols transcription (`SmithNichols`)
//!   G13 — `tfce_tail_into` ≈ `tfce_naive` for the three tails; nilearn hand case
//! Tolerance everywhere: rel 1e-12 with abs floor 1e-12 · max|expected|.
mod common;

use common::{Fixture, assert_map_close, load};
use neurostats::{
    Conn, Domain, Tail, TfceParams, TfceWorkspace, Weighting, tfce, tfce_bands, tfce_bands_naive,
    tfce_naive, tfce_tail_into,
};

const REL: f64 = 1e-12;
const ABS_FLOOR: f64 = 1e-12;

/// Frozen cases; `bench` is excluded from the naive gates (O(V·|H|) on 58k nodes
/// is slow) and covered by G3 only.
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

#[test]
fn g1_union_find_matches_naive_on_fixtures() {
    for name in SMALL {
        let fx = load(name);
        let dom = fx.domain();
        for w in [
            Weighting::MneStep,
            Weighting::SmithNichols,
            Weighting::Exact,
            Weighting::Nilearn,
        ] {
            let p = params(&fx, w);
            let fast = tfce(&dom, &fx.stat, &p).unwrap();
            let slow = tfce_naive(&dom, &fx.stat, &p).unwrap();
            assert_map_close(&format!("G1 {name} {w:?}"), &fast, &slow, REL, ABS_FLOOR);
        }
    }
}

/// G1 on synthetic maps with no Python in the loop: random masks and maps from
/// a fixed LCG, all three connectivities, coarse steps to force many ties per band.
#[test]
fn g1_union_find_matches_naive_synthetic() {
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
            for w in [
                Weighting::MneStep,
                Weighting::SmithNichols,
                Weighting::Exact,
                Weighting::Nilearn,
            ] {
                let p = TfceParams {
                    e: 0.5,
                    h: 2.0,
                    start: 0.25,
                    step: 0.5,
                    weighting: w,
                };
                let fast = tfce(&dom, &stat, &p).unwrap();
                let slow = tfce_naive(&dom, &stat, &p).unwrap();
                assert_map_close(
                    &format!("G1 synth {dims:?} {conn:?} t{trial} {w:?}"),
                    &fast,
                    &slow,
                    REL,
                    ABS_FLOOR,
                );
            }
        }
    }
}

#[test]
fn g2_naive_matches_mne() {
    for name in SMALL {
        let fx = load(name);
        let dom = fx.domain();
        let got = tfce_naive(&dom, &fx.stat, &params(&fx, Weighting::MneStep)).unwrap();
        assert_map_close(
            &format!("G2 {name} (MNE {})", fx.mne_version),
            &got,
            &fx.expected_mne,
            REL,
            ABS_FLOOR,
        );
    }
}

#[test]
fn g3_union_find_matches_mne() {
    for name in SMALL.iter().copied().chain(["bench"]) {
        let fx = load(name);
        let dom = fx.domain();
        let got = tfce(&dom, &fx.stat, &params(&fx, Weighting::MneStep)).unwrap();
        assert_map_close(
            &format!("G3 {name} (MNE {})", fx.mne_version),
            &got,
            &fx.expected_mne,
            REL,
            ABS_FLOOR,
        );
    }
}

#[test]
fn g4_naive_matches_numpy_smith_nichols() {
    for name in SMALL {
        let fx = load(name);
        let dom = fx.domain();
        let got = tfce_naive(&dom, &fx.stat, &params(&fx, Weighting::SmithNichols)).unwrap();
        assert_map_close(
            &format!("G4 {name}"),
            &got,
            &fx.expected_smith_nichols,
            REL,
            ABS_FLOOR,
        );
    }
}

/// Prints the worst observed error as a fraction of tolerance per gate — run with
/// `cargo test -- --ignored --nocapture measure_headroom` when re-justifying REL.
#[test]
#[ignore]
fn measure_headroom() {
    for name in SMALL.iter().copied().chain(["bench"]) {
        let fx = load(name);
        let dom = fx.domain();
        let p = params(&fx, Weighting::MneStep);
        let fast = tfce(&dom, &fx.stat, &p).unwrap();
        let g3 = assert_map_close("G3", &fast, &fx.expected_mne, REL, ABS_FLOOR);
        println!("{name:<13} G3 worst = {g3:.3e} × tol");
        if name != "bench" {
            let slow = tfce_naive(&dom, &fx.stat, &p).unwrap();
            let g1 = assert_map_close("G1", &fast, &slow, REL, ABS_FLOOR);
            let g2 = assert_map_close("G2", &slow, &fx.expected_mne, REL, ABS_FLOOR);
            println!("{name:<13} G1 worst = {g1:.3e} × tol   G2 worst = {g2:.3e} × tol");
        }
    }
}

/// `from_csr` of the CSR read back from `from_mask` reproduces the same graph
/// on every fixture mask. Compared through `to_csr`, not `==`: the two carry
/// different representations of it.
#[test]
fn from_csr_round_trips_every_fixture() {
    for name in SMALL.iter().copied().chain(["bench"]) {
        let fx = load(name);
        let dom = fx.domain();
        let csr = dom.to_csr().unwrap();
        let back = Domain::from_csr(csr.0.clone(), csr.1.clone()).unwrap();
        assert_eq!(back.to_csr().unwrap(), csr, "from_csr round trip {name}");
        assert_eq!(back.n_nodes(), dom.n_nodes(), "round trip n_nodes {name}");
        assert_eq!(back.n_edges(), dom.n_edges(), "round trip n_edges {name}");
    }
}

/// A CSR domain over the same graph as `lat`.
fn as_csr(lat: &Domain) -> Domain {
    let (offsets, neighbours) = lat.to_csr().unwrap();
    Domain::from_csr(offsets, neighbours).unwrap()
}

/// The fixture with every third in-mask voxel dropped, keeping the stat values
/// of the survivors. Several fixture masks are all-true, where `lin_of` is the
/// identity and the lattice sweep's scatter, gather and node walk degenerate;
/// thinning is what puts them under test.
fn thinned(fx: &Fixture) -> (Domain, Vec<f64>) {
    let conn = match fx.conn {
        6 => Conn::Face,
        18 => Conn::Edge,
        _ => Conn::Vertex,
    };
    let mut mask: Vec<bool> = fx.mask.bytes().map(|b| b == b'1').collect();
    let mut stat = Vec::new();
    let mut node = 0usize;
    for m in mask.iter_mut() {
        if *m {
            if node % 3 == 2 {
                *m = false;
            } else {
                stat.push(fx.stat[node]);
            }
            node += 1;
        }
    }
    (Domain::from_mask(&mask, fx.dims, conn).unwrap(), stat)
}

/// The lattice sweep and the CSR sweep are the same computation on the same
/// graph: `tfce` on `from_mask` matches `tfce` on the CSR read out of it, for
/// every weighting and every fixture, dense and thinned. (Not bit equality —
/// the two index spaces visit tied stats in different orders, so bank
/// additions group differently.)
#[test]
fn lattice_matches_csr_on_fixtures() {
    for name in SMALL.iter().copied().chain(["bench"]) {
        let fx = load(name);
        let (thin_dom, thin_stat) = thinned(&fx);
        for (kind, lat, stat) in [
            ("dense", fx.domain(), fx.stat.clone()),
            ("thinned", thin_dom, thin_stat),
        ] {
            let csr = as_csr(&lat);
            // `SmithNichols` is this crate's name for the FSL/PALM stepped rule.
            for w in [
                Weighting::MneStep,
                Weighting::SmithNichols,
                Weighting::Exact,
                Weighting::Nilearn,
            ] {
                let p = params(&fx, w);
                let a = tfce(&lat, &stat, &p).unwrap();
                let b = tfce(&csr, &stat, &p).unwrap();
                assert_map_close(
                    &format!("lattice-vs-csr {name} {kind} {w:?}"),
                    &a,
                    &b,
                    REL,
                    ABS_FLOOR,
                );
            }
        }
    }
}

/// The same for `tfce_bands` under the non-strict membership and irregular
/// grid of G6.
#[test]
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn lattice_matches_csr_on_bands() {
    for name in SMALL {
        let fx = load(name);
        let (thin_dom, thin_stat) = thinned(&fx);
        for (kind, lat, stat) in [
            ("dense", fx.domain(), fx.stat.clone()),
            ("thinned", thin_dom, thin_stat),
        ] {
            let csr = as_csr(&lat);
            let max = stat.iter().cloned().fold(f64::MIN, f64::max);
            if !(max > 0.0) {
                continue;
            }
            let ths: Vec<f64> = [0.05, 0.2, 0.21, 0.5, 0.8, 0.95]
                .iter()
                .map(|f| f * max)
                .collect();
            let ws: Vec<f64> = vec![1.0, 0.5, 2.0, 0.25, 3.0, 0.1];
            let a = tfce_bands(&lat, &stat, &ths, &ws, 0.5, false).unwrap();
            let b = tfce_bands(&csr, &stat, &ths, &ws, 0.5, false).unwrap();
            assert_map_close(
                &format!("lattice-vs-csr bands {name} {kind}"),
                &a,
                &b,
                REL,
                ABS_FLOOR,
            );
        }
    }
}

/// G6 — `tfce_bands` (union-find) ≈ `tfce_bands_naive` under non-strict
/// membership and an irregular hand-written grid, on every small fixture.
#[test]
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn g6_bands_union_find_matches_naive() {
    for name in SMALL {
        let fx = load(name);
        let dom = fx.domain();
        let max = fx.stat.iter().cloned().fold(f64::MIN, f64::max);
        if !(max > 0.0) {
            continue;
        }
        // Irregular, strictly increasing grid ending below max; weights
        // deliberately not a power law so the sum is not a special case.
        let ths: Vec<f64> = [0.05, 0.2, 0.21, 0.5, 0.8, 0.95]
            .iter()
            .map(|f| f * max)
            .collect();
        let ws: Vec<f64> = vec![1.0, 0.5, 2.0, 0.25, 3.0, 0.1];
        for strict in [true, false] {
            let fast = tfce_bands(&dom, &fx.stat, &ths, &ws, 0.5, strict).unwrap();
            let slow = tfce_bands_naive(&dom, &fx.stat, &ths, &ws, 0.5, strict).unwrap();
            assert_map_close(
                &format!("G6 {name} strict={strict}"),
                &fast,
                &slow,
                REL,
                ABS_FLOOR,
            );
        }
    }
}

/// Non-strict membership must differ from strict when a node sits exactly on
/// a threshold (that is the whole reason the flag exists).
#[test]
fn g6_strict_flag_changes_membership_on_ties() {
    let dom = Domain::from_volume([3, 1, 1], Conn::Face);
    let stat = [1.0, 2.0, 3.0];
    let ths = [1.0, 2.0];
    let ws = [1.0, 1.0];
    let strict = tfce_bands(&dom, &stat, &ths, &ws, 1.0, true).unwrap();
    let loose = tfce_bands(&dom, &stat, &ths, &ws, 1.0, false).unwrap();
    // strict: band 1.0 = {2,3} (size 2), band 2.0 = {3} (size 1)
    assert_eq!(strict, vec![0.0, 2.0, 3.0]);
    // loose: band 1.0 = {1,2,3} (size 3), band 2.0 = {2,3} (size 2)
    assert_eq!(loose, vec![3.0, 5.0, 5.0]);
}

#[test]
fn tfce_bands_rejects_bad_bands() {
    let dom = Domain::from_volume([2, 1, 1], Conn::Face);
    let s = [1.0, 2.0];
    let bad = neurostats::NeuroError::InvalidBands;
    assert_eq!(
        tfce_bands(&dom, &s, &[1.0, 1.0], &[1.0, 1.0], 0.5, true),
        Err(bad)
    );
    assert_eq!(
        tfce_bands(&dom, &s, &[1.0], &[1.0, 1.0], 0.5, true),
        Err(bad)
    );
    assert_eq!(
        tfce_bands(&dom, &s, &[1.0, f64::NAN], &[1.0, 1.0], 0.5, true),
        Err(bad)
    );
    assert_eq!(
        tfce_bands(&dom, &s, &[1.0], &[1.0], f64::INFINITY, true),
        Err(neurostats::NeuroError::InvalidParams)
    );
    // Empty band list is legal: every node 0.
    assert_eq!(
        tfce_bands(&dom, &s, &[], &[], 0.5, true).unwrap(),
        vec![0.0, 0.0]
    );
}

const WEIGHTINGS: [Weighting; 4] = [
    Weighting::MneStep,
    Weighting::SmithNichols,
    Weighting::Exact,
    Weighting::Nilearn,
];

/// The fixture map with every other node negated. With `balanced`, one node
/// is then forced to `+m` and another to `-m`, `m = 1.5 · max |stat|`, so both
/// signs have the same maximum.
fn signed(fx: &Fixture, balanced: bool) -> Vec<f64> {
    let mut stat: Vec<f64> = fx
        .stat
        .iter()
        .enumerate()
        .map(|(i, &s)| if i % 2 == 0 { s } else { -s })
        .collect();
    if balanced {
        let m = 1.5 * stat.iter().fold(0.0f64, |a, s| a.max(s.abs()));
        let last = stat.len() - 1;
        stat[0] = m;
        stat[last] = -m;
    }
    stat
}

/// G13 — the three tails against `tfce_naive`: positive is the naive map,
/// negative is the naive map of `-stat`, two-sided is their difference.
/// `tfce_naive` builds each grid from its own map's maximum, so it is the
/// two-sided oracle where the shared grid changes nothing: for `MneStep`,
/// `SmithNichols` and `Exact` on any map, and for `Nilearn` on a map whose
/// two signs have the same maximum.
#[test]
fn g13_tails_match_naive() {
    let mut ws = TfceWorkspace::new(0);
    for name in ["grid4", "blob", "ties", "two_clusters"] {
        let fx = load(name);
        let dom = fx.domain();
        for balanced in [false, true] {
            let stat = signed(&fx, balanced);
            let neg: Vec<f64> = stat.iter().map(|s| -s).collect();
            for w in WEIGHTINGS {
                if w == Weighting::Nilearn && !balanced {
                    continue;
                }
                let p = params(&fx, w);
                let up = tfce_naive(&dom, &stat, &p).unwrap();
                let down = tfce_naive(&dom, &neg, &p).unwrap();
                let both: Vec<f64> = up.iter().zip(&down).map(|(a, b)| a - b).collect();
                let mut got = vec![f64::NAN; dom.n_nodes()];
                for (tail, want) in [
                    (Tail::Positive, &up),
                    (Tail::Negative, &down),
                    (Tail::TwoSided, &both),
                ] {
                    tfce_tail_into(&dom, &stat, &p, tail, &mut ws, &mut got).unwrap();
                    assert_map_close(
                        &format!("G13 {name} balanced={balanced} {w:?} {tail:?}"),
                        &got,
                        want,
                        REL,
                        ABS_FLOOR,
                    );
                }
            }
        }
    }
}

/// Under `Nilearn` both passes of a two-sided call use the grid of
/// `max |stat|`. Two adjacent nodes, stat = [100, -50], E = 1, H = 2: the grid
/// is h_k = k for k = 1..=100. Node 0 is alone in all 100 bands of the
/// positive pass: Σ k² = 338350. Node 1 is alone in bands 1..=50 of the
/// negative pass: Σ_{k≤50} k² = 42925. On its own grid (h_k = k/2) the
/// negative pass would give 84587.5. nilearn 0.14.1
/// `calculate_tfce(E=1, H=2, two_sided_test=True)` returns [338350, -42925].
#[test]
fn g13_two_sided_nilearn_grid_comes_from_max_abs() {
    let dom = Domain::from_volume([2, 1, 1], Conn::Face);
    let p = TfceParams {
        e: 1.0,
        h: 2.0,
        start: 0.0,
        step: 1.0,
        weighting: Weighting::Nilearn,
    };
    let mut ws = TfceWorkspace::new(2);
    let mut got = vec![f64::NAN; 2];
    tfce_tail_into(&dom, &[100.0, -50.0], &p, Tail::TwoSided, &mut ws, &mut got).unwrap();
    assert_map_close("G13 hand", &got, &[338350.0, -42925.0], REL, ABS_FLOOR);
}

/// One workspace reused across tails, weightings and domains of different
/// size gives the bits a fresh workspace gives: nothing of an earlier call
/// (the negated map, its TFCE, the grid) leaks into a later one.
#[test]
fn g13_workspace_reuse_across_tails_is_bitwise() {
    let mut shared = TfceWorkspace::new(0);
    for name in [
        "bench",
        "grid4",
        "blob",
        "single",
        "empty_grid",
        "two_clusters",
    ] {
        let fx = load(name);
        let dom = fx.domain();
        let stat: Vec<f64> = fx
            .stat
            .iter()
            .enumerate()
            .map(|(i, &s)| if i % 3 == 0 { -0.6 * s } else { s })
            .collect();
        for w in WEIGHTINGS {
            let p = params(&fx, w);
            for tail in [Tail::TwoSided, Tail::Negative, Tail::Positive] {
                let mut want = vec![f64::NAN; dom.n_nodes()];
                let mut fresh = TfceWorkspace::new(dom.n_nodes());
                tfce_tail_into(&dom, &stat, &p, tail, &mut fresh, &mut want).unwrap();
                let mut got = vec![f64::NAN; dom.n_nodes()];
                tfce_tail_into(&dom, &stat, &p, tail, &mut shared, &mut got).unwrap();
                assert_eq!(got, want, "reuse {name} {w:?} {tail:?}");
            }
        }
    }
}

/// A map with no negative value: the negative tail is all zeros and the
/// two-sided tail equals the positive tail. An all-zero map is all zeros
/// under every tail. No NaN and no negative zero-crossing artefacts.
#[test]
fn g13_one_signed_and_zero_maps() {
    let dom = Domain::from_volume([3, 2, 1], Conn::Face);
    let stat = [0.5, 2.0, 0.0, 1.5, 3.0, 0.25];
    let mut ws = TfceWorkspace::new(dom.n_nodes());
    for w in WEIGHTINGS {
        let p = TfceParams {
            e: 0.5,
            h: 2.0,
            start: 0.0,
            step: 0.25,
            weighting: w,
        };
        let up = tfce(&dom, &stat, &p).unwrap();
        let mut got = vec![f64::NAN; 6];
        tfce_tail_into(&dom, &stat, &p, Tail::Negative, &mut ws, &mut got).unwrap();
        assert_eq!(got, vec![0.0; 6], "{w:?} negative");
        tfce_tail_into(&dom, &stat, &p, Tail::TwoSided, &mut ws, &mut got).unwrap();
        assert_eq!(got, up, "{w:?} two-sided");
        for tail in [Tail::Positive, Tail::Negative, Tail::TwoSided] {
            tfce_tail_into(&dom, &[0.0; 6], &p, tail, &mut ws, &mut got).unwrap();
            assert_eq!(got, vec![0.0; 6], "{w:?} {tail:?} zero map");
        }
    }
}

#[test]
fn tfce_tail_into_rejects_bad_inputs() {
    let dom = Domain::from_volume([2, 1, 1], Conn::Face);
    let p = TfceParams {
        e: 0.5,
        h: 2.0,
        start: 0.0,
        step: 1.0,
        weighting: Weighting::Nilearn,
    };
    let mut ws = TfceWorkspace::new(2);
    let mut out = vec![0.0; 2];
    let mut short = vec![0.0; 1];
    for tail in [Tail::Positive, Tail::Negative, Tail::TwoSided] {
        assert_eq!(
            tfce_tail_into(&dom, &[1.0, 2.0], &p, tail, &mut ws, &mut short),
            Err(neurostats::NeuroError::MismatchedLengths {
                expected: 2,
                got: 1
            })
        );
        assert_eq!(
            tfce_tail_into(&dom, &[1.0, f64::NAN], &p, tail, &mut ws, &mut out),
            Err(neurostats::NeuroError::NonFiniteStat)
        );
    }
}
