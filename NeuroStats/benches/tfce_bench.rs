//! Single-map TFCE timing on the realistic bench fixture vs the MNE wall time
//! recorded in the same fixture by `scripts/gen_tfce_golden.py`.
//!
//! Run: `cargo bench --bench tfce_bench` (pin and lock the machine first; the
//! first pass is discarded, the min of the rest is reported).

use std::time::Instant;

use neurostats::{Conn, Domain, TfceParams, Weighting, tfce};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    dims: [usize; 3],
    conn: u32,
    mask: String,
    stat: Vec<f64>,
    params: Params,
    mne_seconds: f64,
}

#[derive(Deserialize)]
struct Params {
    e: f64,
    h: f64,
    start: f64,
    step: f64,
}

fn main() {
    let path = format!(
        "{}/tests/fixtures/tfce_bench.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fx: Fixture = serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("fixture parse");
    let mask: Vec<bool> = fx.mask.bytes().map(|b| b == b'1').collect();
    let conn = match fx.conn {
        6 => Conn::Face,
        18 => Conn::Edge,
        26 => Conn::Vertex,
        c => panic!("bad conn {c}"),
    };
    let dom = Domain::from_mask(&mask, fx.dims, conn).unwrap();
    let p = TfceParams {
        e: fx.params.e,
        h: fx.params.h,
        start: fx.params.start,
        step: fx.params.step,
        weighting: Weighting::MneStep,
    };

    let runs = 10;
    let mut best = f64::INFINITY;
    for r in 0..=runs {
        let t = Instant::now();
        let out = tfce(&dom, &fx.stat, &p).unwrap();
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(out);
        if r > 0 {
            best = best.min(dt);
        }
    }
    println!(
        "nodes {}  edges {}  tfce(union-find) min {:.4} s over {runs} runs  |  MNE 1.12.1 {:.4} s  |  speedup {:.1}x",
        dom.n_nodes(),
        dom.n_edges() / 2,
        best,
        fx.mne_seconds,
        fx.mne_seconds / best
    );
}
