//! TFCE parity gates (design spec §5):
//!   G1 — union-find `tfce` ≈ `tfce_naive`, both weightings (internal, no oracle)
//!   G2 — `tfce_naive` ≈ MNE 1.12.1 golden (`MneStep`)
//!   G3 — `tfce` ≈ MNE golden (`MneStep`)
//!   G4 — `tfce_naive` ≈ independent NumPy Smith–Nichols transcription (`SmithNichols`)
//! Tolerance everywhere: rel 1e-12 with abs floor 1e-12 · max|expected|.
mod common;

use common::{Fixture, assert_map_close, load};
use neurostats::{
    Conn, Domain, TfceParams, Weighting, tfce, tfce_bands, tfce_bands_naive, tfce_naive,
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
        for w in [Weighting::MneStep, Weighting::SmithNichols] {
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
            for w in [Weighting::MneStep, Weighting::SmithNichols] {
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
