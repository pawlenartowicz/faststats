//! Sign-flip permutation timing on the realistic bench fixture: `B = 200`
//! Monte Carlo draws over `n = 30` synthetic subjects (fixture map + unit
//! noise, fixed LCG), reporting ms/draw and the `tfce_into` share.
//!
//! Run: `cargo bench --bench perm_bench` (pin and lock the machine first; the
//! first pass is discarded, the min of the rest is reported).

use std::time::Instant;

use neurostats::{
    Conn, Domain, OneSampleProblem, PermWorkspace, TfceParams, TfceWorkspace, Weighting, run_range,
    sign_flip_plan, tfce_into,
};
#[cfg(feature = "parallel")]
use neurostats::{tfce_one_sample, tfce_one_sample_threads};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    dims: [usize; 3],
    conn: u32,
    mask: String,
    stat: Vec<f64>,
    params: Params,
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
    let (n, v, b) = (30usize, dom.n_nodes(), 200u64);
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let x: Vec<f64> = (0..n * v)
        .map(|k| 0.3 * fx.stat[k % v] + (next() - 0.5) * 3.4641)
        .collect();
    let runs = 3;
    for weighting in [Weighting::MneStep, Weighting::Exact] {
        let p = TfceParams {
            e: fx.params.e,
            h: fx.params.h,
            start: fx.params.start,
            step: fx.params.step,
            weighting,
        };
        let problem = OneSampleProblem::new(&dom, &x, n, p, 7).unwrap();
        let plan = sign_flip_plan(n, b);
        let mut ws = PermWorkspace::new(n, v);
        let mut best = f64::INFINITY;
        for r in 0..=runs {
            let t = Instant::now();
            let null = run_range(&problem, &plan, 0..plan.b, &mut ws);
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(null);
            if r > 0 {
                best = best.min(dt);
            }
        }
        // tfce_into alone on the observed t-map, same B, same workspace reuse.
        let mut tws = TfceWorkspace::new(v);
        let mut out = vec![0.0; v];
        let mut best_tfce = f64::INFINITY;
        for r in 0..=runs {
            let t = Instant::now();
            for _ in 0..b {
                tfce_into(&dom, problem.t_obs(), &p, &mut tws, &mut out).unwrap();
            }
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(&out);
            if r > 0 {
                best_tfce = best_tfce.min(dt);
            }
        }
        println!(
            "{weighting:?}: nodes {v}  n {n}  B {b}  min {:.3} s = {:.2} ms/draw  |  tfce_into {:.2} ms/draw ({:.0}%)",
            best,
            1e3 * best / b as f64,
            1e3 * best_tfce / b as f64,
            100.0 * best_tfce / best
        );
    }

    scaling_section();
}

/// Brain-mask-sized scaling check: `tfce_one_sample_threads` at `threads` in
/// {1, 2, 4, 6} against serial `tfce_one_sample`, on a ~250k-node ellipsoid
/// mask (the shape of a real brain-mask problem, unlike the small fixture
/// above). Report seconds and the ratio to serial — spec A gate 5 turns this
/// into a pass/fail (>= 4x at 6 threads) but only under a locked clock, so
/// this bench just prints the numbers.
#[cfg(feature = "parallel")]
fn scaling_section() {
    let dims = [128usize, 128, 92];
    let [nx, ny, nz] = dims;
    let (cx, cy, cz) = (nx as f64 / 2.0, ny as f64 / 2.0, nz as f64 / 2.0);
    // Semi-axes picked by counting the discretised mask directly (the
    // closed-form ellipsoid volume undercounts once voxels are discretised):
    // (44, 44, 31) lands at 251,376 nodes on this grid.
    let (rx, ry, rz) = (44.0f64, 44.0, 31.0);
    let mut mask = vec![false; nx * ny * nz];
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let dx = (x as f64 + 0.5 - cx) / rx;
                let dy = (y as f64 + 0.5 - cy) / ry;
                let dz = (z as f64 + 0.5 - cz) / rz;
                // Row-major lin index matches `Domain::from_mask`'s convention.
                mask[(x * ny + y) * nz + z] = dx * dx + dy * dy + dz * dz <= 1.0;
            }
        }
    }
    let dom = Domain::from_mask(&mask, dims, Conn::Vertex).unwrap();
    let (n, v, b) = (30usize, dom.n_nodes(), 200u64);

    let mut seed: u64 = 0xD1B5_4A32_D192_ED03;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let x: Vec<f64> = (0..n * v).map(|_| (next() - 0.5) * 3.4641).collect();
    let p = TfceParams {
        e: 0.5,
        h: 2.0,
        start: 0.0,
        step: 1.0,
        weighting: Weighting::Exact,
    };

    let runs = 5;

    println!(
        "scaling: nodes {v}  n {n}  B {b}  weighting {:?}",
        p.weighting
    );

    // Serial baseline: the dedicated single-threaded function, not
    // `tfce_one_sample_threads(.., 1)` — the two are expected to be
    // bit-identical, but the ratio below is against the serial code path
    // callers actually use without the `parallel` feature.
    let mut serial_best = f64::INFINITY;
    for r in 0..=runs {
        let t = Instant::now();
        let out = tfce_one_sample(&dom, &x, n, &p, 7, b).unwrap();
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(&out);
        if r > 0 {
            serial_best = serial_best.min(dt);
        }
    }
    println!("  serial (tfce_one_sample): min {:.3} s", serial_best);

    for threads in [1usize, 2, 4, 6] {
        let mut best = f64::INFINITY;
        for r in 0..=runs {
            let t = Instant::now();
            let out = tfce_one_sample_threads(&dom, &x, n, &p, 7, b, threads).unwrap();
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(&out);
            if r > 0 {
                best = best.min(dt);
            }
        }
        println!(
            "  threads {threads}: min {:.3} s  ({:.2}x serial)",
            best,
            serial_best / best
        );
    }
}

#[cfg(not(feature = "parallel"))]
fn scaling_section() {}
