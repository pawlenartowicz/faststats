//! Time and memory of the OLS permutation loop on a data cell read from disk:
//! brain data too large for a fixture, with its mask and its design.
//!
//! A cell is a directory with three files:
//! - `cell.json`: `dims` (grid shape), `source_rows` (rows of the source),
//!   `rows` (the source rows that enter the model), `tested` (one value per
//!   model row) and `confounds` (row-major, one row per model row; may be
//!   empty);
//! - `mask.u8`: one byte per grid voxel in row-major order, non-zero = in mask;
//! - `y.f32`: the source, subject-major `source_rows × V` little-endian `f32`,
//!   `V` being the number of mask voxels.
//!
//! The model is `brain ~ tested + confounds + intercept` with
//! `Weighting::Exact` (`E = 0.5`, `H = 2`, `start = 0`), the two-sided tail,
//! 6-connectivity and seed 1.
//!
//! Two modes, one process per configuration, so that a memory peak belongs to
//! one configuration:
//!
//! ```text
//! cargo bench --bench cell_bench -- split <dir> <batch>
//! cargo bench --bench cell_bench -- run <dir> <threads> <batch> <draws> <passes>
//! ```
//!
//! `split` runs on one thread and times the two halves of a draw apart, on
//! the same `batch` null draws: the statistic maps (`stat_batch`) and their
//! TFCE (`tfce_tail_into`). The first pass is discarded and the minimum of the
//! next three is reported, per draw.
//!
//! `run` runs the whole loop (`run_threads` + `finalize`) `passes` times and
//! prints the seconds of each pass, and the peak anonymous memory (`RssAnon`
//! of `/proc/self/status`, sampled every 10 ms) during the store build and
//! during the loop. The source is held in memory while the store is built,
//! so the build peak includes it; `rss_base_gb` is read after the source is
//! dropped and before the loop starts.
//!
//! Pin and lock the machine first.

use std::fs;
use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use neurostats::{
    Conn, Domain, PermOpts, PermProblem, PermWorkspace, Store, Tail, TfceParams, TfceWorkspace,
    Weighting, finalize, nuisance_basis, run_threads, stat_batch, tfce_tail_into,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Cell {
    dims: [usize; 3],
    source_rows: usize,
    rows: Vec<usize>,
    tested: Vec<f64>,
    confounds: Vec<f64>,
}

const PARAMS: TfceParams = TfceParams {
    e: 0.5,
    h: 2.0,
    start: 0.0,
    step: 1.0,
    weighting: Weighting::Exact,
};
const TAIL: Tail = Tail::TwoSided;
const SEED: u64 = 1;
const GB: f64 = 1024.0 * 1024.0;

/// `RssAnon` of this process in kB; 0 where `/proc` has no such line.
fn rss_anon_kb() -> u64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("RssAnon:"))
                .and_then(|rest| rest.split_whitespace().next()?.parse().ok())
        })
        .unwrap_or(0)
}

/// Background sampler of `RssAnon`. `take_gb` returns the peak since the last
/// call and starts a new window.
struct Sampler(Arc<AtomicU64>);

impl Sampler {
    fn start() -> Self {
        let peak = Arc::new(AtomicU64::new(0));
        let shared = Arc::clone(&peak);
        std::thread::spawn(move || {
            loop {
                shared.fetch_max(rss_anon_kb(), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(10));
            }
        });
        Self(peak)
    }

    fn take_gb(&self) -> f64 {
        self.0.swap(0, Ordering::Relaxed).max(rss_anon_kb()) as f64 / GB
    }
}

/// `len` little-endian `f32` values, the whole of the file at `path`.
fn read_f32(path: &str, len: usize) -> Vec<f32> {
    let mut file = fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let bytes = file.metadata().expect("metadata").len();
    assert_eq!(bytes, 4 * len as u64, "{path}: expected {len} f32 values");
    let mut out = Vec::with_capacity(len);
    let mut buf = vec![0u8; 1 << 22];
    let mut left = 4 * len;
    while left > 0 {
        let take = left.min(buf.len());
        file.read_exact(&mut buf[..take]).expect("read y.f32");
        out.extend(
            buf[..take]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
        left -= take;
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--bench")
        .collect();
    let num = |i: usize| -> usize {
        args.get(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or_else(|| panic!("argument {i} must be a number: {args:?}"))
    };
    let (mode, dir) = match args.as_slice() {
        [mode, dir, ..] if mode == "split" || mode == "run" => (mode.as_str(), dir.as_str()),
        _ => {
            println!(
                "usage: cell_bench split <dir> <batch> | run <dir> <threads> <batch> <draws> <passes>"
            );
            return;
        }
    };
    let (threads, batch, draws, passes) = if mode == "split" {
        (1, num(2), 1 + num(2) as u64, 0)
    } else {
        (num(2), num(3), num(4) as u64, num(5))
    };

    let cell: Cell =
        serde_json::from_str(&fs::read_to_string(format!("{dir}/cell.json")).expect("cell.json"))
            .expect("cell.json parse");
    let mask: Vec<bool> = fs::read(format!("{dir}/mask.u8"))
        .expect("mask.u8")
        .iter()
        .map(|&b| b != 0)
        .collect();
    let dom = Domain::from_mask(&mask, cell.dims, Conn::Face).expect("mask and dims");
    let (n, v) = (cell.rows.len(), dom.n_nodes());

    let sampler = Sampler::start();
    let src = read_f32(&format!("{dir}/y.f32"), cell.source_rows * v);
    let t = Instant::now();
    let basis = nuisance_basis(&cell.confounds, n, true).expect("confounds");
    let store = Store::from_f32(&src, cell.source_rows, v, &cell.rows, &basis).expect("store");
    let build_s = t.elapsed().as_secs_f64();
    drop(src);
    let peak_build = sampler.take_gb();

    let mut opts = PermOpts::new(PARAMS, TAIL, SEED, draws);
    opts.batch = batch;
    let problem = PermProblem::ols(&dom, &store, Some(&cell.tested), &opts).expect("problem");
    let head = format!("n={n} V={v} q={} batch={batch}", store.q());

    if mode == "split" {
        let mut ws = PermWorkspace::new(&problem);
        let mut tws = TfceWorkspace::new(v);
        let mut out = vec![0.0f64; v];
        let (mut product, mut tfce) = (f64::INFINITY, f64::INFINITY);
        for pass in 0..4 {
            let t = Instant::now();
            let maps = stat_batch(&problem, 1..1 + batch as u64, &mut ws);
            let dt_product = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for map in maps.chunks_exact(v) {
                tfce_tail_into(&dom, map, &PARAMS, TAIL, &mut tws, &mut out).unwrap();
            }
            let dt_tfce = t.elapsed().as_secs_f64();
            std::hint::black_box(&out);
            if pass > 0 {
                product = product.min(dt_product);
                tfce = tfce.min(dt_tfce);
            }
        }
        println!(
            "split {head} store_build_s={build_s:.2} product_ms_per_draw={:.2} tfce_ms_per_draw={:.2}",
            1e3 * product / batch as f64,
            1e3 * tfce / batch as f64
        );
        return;
    }

    sampler.take_gb();
    let base = rss_anon_kb() as f64 / GB;
    let mut seconds = Vec::new();
    let mut summary = String::new();
    for _ in 0..passes {
        let t = Instant::now();
        let null = run_threads(&problem, problem.draws(), threads);
        let r = finalize(&problem, &null).expect("finalize");
        seconds.push(format!("{:.2}", t.elapsed().as_secs_f64()));
        summary = format!(
            "min_p_fwe={} null_max_top={:.6e}",
            r.p_fwe.iter().fold(1.0f64, |a, &p| a.min(p)),
            r.null.maxima[r.null.maxima.len() - 1]
        );
    }
    let peak_loop = sampler.take_gb();
    println!(
        "run {head} threads={threads} draws={draws} store_build_s={build_s:.2} rss_base_gb={base:.3} rss_peak_build_gb={peak_build:.3} rss_peak_loop_gb={peak_loop:.3} loop_s={} {summary}",
        seconds.join(",")
    );
}
