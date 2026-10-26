//! Shared TFCE fixture harness — the map-shaped counterpart of CommonStats'
//! row-grid harness: load a frozen `tests/fixtures/tfce_*.json`, build the
//! `Domain`, and compare whole maps node-wise under a rel tolerance with an abs
//! floor scaled to the map's maximum.
#![allow(dead_code)]

use neurostats::{Conn, Domain, TfceParams, Weighting};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Params {
    pub e: f64,
    pub h: f64,
    pub start: f64,
    pub step: f64,
}

impl Params {
    /// The fixture's TFCE settings under `weighting`.
    pub fn with(&self, weighting: Weighting) -> TfceParams {
        TfceParams {
            e: self.e,
            h: self.h,
            start: self.start,
            step: self.step,
            weighting,
        }
    }
}

#[derive(Deserialize)]
pub struct Fixture {
    pub mne_version: String,
    pub dims: [usize; 3],
    pub conn: u32,
    /// One `'0'`/`'1'` char per voxel, row-major over `dims`.
    pub mask: String,
    pub stat: Vec<f64>,
    pub params: Params,
    pub expected_mne: Vec<f64>,
    pub expected_smith_nichols: Vec<f64>,
    pub mne_seconds: f64,
}

/// Shared `conn` code → [`Conn`] mapping used by both fixture kinds' `domain()`.
pub fn conn_of(c: u32) -> Conn {
    match c {
        6 => Conn::Face,
        18 => Conn::Edge,
        26 => Conn::Vertex,
        c => panic!("fixture conn {c} not in {{6,18,26}}"),
    }
}

impl Fixture {
    /// One `bool` per voxel of `dims`, row-major.
    pub fn mask_bools(&self) -> Vec<bool> {
        self.mask.bytes().map(|b| b == b'1').collect()
    }

    pub fn domain(&self) -> Domain {
        Domain::from_mask(&self.mask_bools(), self.dims, conn_of(self.conn))
            .expect("fixture mask/dims")
    }
}

pub fn load(name: &str) -> Fixture {
    let path = format!(
        "{}/tests/fixtures/tfce_{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let txt = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {path}: {e} — run scripts/gen_tfce_golden.py"));
    serde_json::from_str(&txt).expect("fixture parse")
}

#[derive(Deserialize)]
pub struct PermFixture {
    pub dims: [usize; 3],
    pub conn: u32,
    pub mask: String,
    pub n: usize,
    pub x: Vec<f64>,
    pub params: Params,
    pub n_permutations: u64,
    pub seed: u64,
    pub exact: bool,
    pub t_obs: Vec<f64>,
    pub tfce_obs: Vec<f64>,
    pub p_fwe: Vec<f64>,
    pub h0: Vec<f64>,
}

impl PermFixture {
    pub fn domain(&self) -> Domain {
        let mask: Vec<bool> = self.mask.bytes().map(|b| b == b'1').collect();
        Domain::from_mask(&mask, self.dims, conn_of(self.conn)).expect("fixture mask/dims")
    }
}

pub fn load_perm(name: &str) -> PermFixture {
    let path = format!(
        "{}/tests/fixtures/perm_{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let txt = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {path}: {e} — run scripts/gen_perm_golden.py"));
    serde_json::from_str(&txt).expect("fixture parse")
}

/// Node-wise `|got − want| ≤ max(abs_floor · max|want|, rel · |want|)`.
pub fn assert_map_close(label: &str, got: &[f64], want: &[f64], rel: f64, abs_floor: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let scale = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{label}[{i}]: non-finite {g:e}");
        let tol = (abs_floor * scale).max(rel * w.abs());
        let diff = (g - w).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: got {g:e}, want {w:e}, diff {diff:e} > tol {tol:e}"
        );
    }
}
