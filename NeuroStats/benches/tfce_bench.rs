//! Single-map TFCE timing on the realistic bench fixture vs the MNE wall time
//! recorded in the same fixture by `scripts/gen_tfce_golden.py`.
//!
//! Run: `cargo bench --bench tfce_bench` (pin and lock the machine first; the
//! first pass is discarded, the min of the rest is reported).

use std::time::Instant;

use neurostats::{TfceParams, Weighting, tfce};

#[path = "../tests/common/mod.rs"]
mod common;

fn main() {
    let fx = common::load("bench");
    let dom = fx.domain();
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
        "nodes {}  edges {}  tfce(union-find) min {:.4} s over {runs} runs  |  MNE {} {:.4} s  |  speedup {:.1}x",
        dom.n_nodes(),
        dom.n_edges() / 2,
        best,
        fx.mne_version,
        fx.mne_seconds,
        fx.mne_seconds / best
    );
}
