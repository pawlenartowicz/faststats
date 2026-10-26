#![cfg(feature = "std")]
//! OLS engine gates. The oracle throughout is the textbook path: one
//! least-squares fit per node from the normal equations on the raw columns.
//!   O1 — `nuisance_basis`: orthonormal, same span, dependent columns dropped
//!   O2 — `Store`: columns are unit-norm least-squares residuals; f32 source
//!        and row index; degenerate nodes
//!   O3 — the t of every draw ≈ a direct fit with that draw's design
//!   O4 — `tfce_ols` maps and p-maps ≈ a naive per-draw loop, three tails;
//!        Monte Carlo draws and exhaustive sign enumeration
//!   O5 — design edge cases and inputs the loop must refuse or survive
//!   O7 — the shuffled-design statistic ≈ a direct Freedman–Lane loop
use commonstats::gen_sign_flips;
use neurostats::{
    Conn, Domain, NeuroError, PermOpts, PermProblem, PermWorkspace, Store, Tail, TfceParams,
    Weighting, draw_permutation, nuisance_basis, stat_batch, tfce, tfce_ols,
};

/// Deterministic uniform numbers in `[-1, 1)` (64-bit LCG, top 53 bits).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    fn vec(&mut self, len: usize) -> Vec<f64> {
        (0..len).map(|_| self.next()).collect()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Least-squares fit of `y` on the columns of `x` (row-major `n × p`) from the
/// normal equations, solved by Gaussian elimination with partial pivoting.
/// Returns the coefficients, the inverse of `X'X` (row-major `p × p`) and the
/// residuals.
fn fit(x: &[f64], y: &[f64], n: usize, p: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Augmented system [X'X | X'y | I].
    let w = 2 * p + 1;
    let mut a = vec![0.0f64; p * w];
    for j in 0..p {
        for k in 0..p {
            a[j * w + k] = (0..n).map(|i| x[i * p + j] * x[i * p + k]).sum();
        }
        a[j * w + p] = (0..n).map(|i| x[i * p + j] * y[i]).sum();
        a[j * w + p + 1 + j] = 1.0;
    }
    for col in 0..p {
        let pivot = (col..p)
            .max_by(|&r, &s| a[r * w + col].abs().total_cmp(&a[s * w + col].abs()))
            .unwrap();
        for k in 0..w {
            a.swap(col * w + k, pivot * w + k);
        }
        let d = a[col * w + col];
        for k in 0..w {
            a[col * w + k] /= d;
        }
        for r in 0..p {
            if r != col {
                let f = a[r * w + col];
                for k in 0..w {
                    a[r * w + k] -= f * a[col * w + k];
                }
            }
        }
    }
    let beta: Vec<f64> = (0..p).map(|j| a[j * w + p]).collect();
    let inv: Vec<f64> = (0..p * p).map(|k| a[(k / p) * w + p + 1 + k % p]).collect();
    let resid: Vec<f64> = (0..n)
        .map(|i| y[i] - (0..p).map(|j| x[i * p + j] * beta[j]).sum::<f64>())
        .collect();
    (beta, inv, resid)
}

/// Row-major `n × (c + 1)` matrix `[confounds, 1]`.
fn with_ones(confounds: &[f64], n: usize) -> Vec<f64> {
    let c = confounds.len() / n;
    let mut out = Vec::with_capacity(n * (c + 1));
    for i in 0..n {
        out.extend_from_slice(&confounds[i * c..(i + 1) * c]);
        out.push(1.0);
    }
    out
}

/// Column `v` of a subject-major `n × n_nodes` matrix.
fn column(y: &[f64], n: usize, n_nodes: usize, v: usize) -> Vec<f64> {
    (0..n).map(|i| y[i * n_nodes + v]).collect()
}

/// O1 — the basis is orthonormal, starts with the constant vector, and spans
/// the confounds and the intercept: projecting any of those columns on it
/// gives the column back.
#[test]
fn basis_is_orthonormal_and_spans_the_columns() {
    let (n, c) = (12usize, 3usize);
    let mut rng = Lcg(0x5EED_0001);
    // A large offset makes the confounds far from centred: one Gram–Schmidt
    // pass loses orthogonality to ~1e-12 here, two keep it at ~1e-16.
    let confounds: Vec<f64> = rng.vec(n * c).iter().map(|x| 1e3 + x).collect();
    for intercept in [false, true] {
        let basis = nuisance_basis(&confounds, n, intercept).unwrap();
        let q = c + usize::from(intercept);
        assert_eq!(basis.len(), q * n);
        for a in 0..q {
            for b in 0..q {
                let g = dot(&basis[a * n..(a + 1) * n], &basis[b * n..(b + 1) * n]);
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((g - want).abs() < 1e-14, "Q'Q[{a},{b}] = {g}");
            }
        }
        if intercept {
            let s = 1.0 / (n as f64).sqrt();
            assert!(basis[..n].iter().all(|&x| (x - s).abs() < 1e-15));
        }
        let mut columns: Vec<Vec<f64>> = (0..c)
            .map(|j| (0..n).map(|i| confounds[i * c + j]).collect())
            .collect();
        if intercept {
            columns.push(vec![1.0; n]);
        }
        for col in &columns {
            let mut rest = col.clone();
            for u in basis.chunks_exact(n) {
                let k = dot(u, col);
                for (r, &ui) in rest.iter_mut().zip(u) {
                    *r -= k * ui;
                }
            }
            assert!(dot(&rest, &rest).sqrt() < 1e-12 * dot(col, col).sqrt());
        }
    }
}

/// O1 — the `1e-10` drop rule: a duplicate, a zero column, a constant column
/// next to the intercept and a combination of earlier columns are dropped; a
/// column that leaves the span by `1e-6` of its norm is kept.
#[test]
fn basis_drops_dependent_columns() {
    let n = 10usize;
    let mut rng = Lcg(0x5EED_0002);
    let a = rng.vec(n);
    let b = rng.vec(n);
    let noise = rng.vec(n);
    let rows = |cols: &[&[f64]]| -> Vec<f64> {
        (0..n)
            .flat_map(|i| cols.iter().map(move |col| col[i]))
            .collect()
    };
    let zero = vec![0.0; n];
    let seven = vec![7.0; n];
    let combo: Vec<f64> = (0..n).map(|i| 2.0 * a[i] - 3.0 * b[i] + 0.5).collect();
    let nearly: Vec<f64> = (0..n).map(|i| combo[i] + 1e-13 * noise[i]).collect();
    let apart: Vec<f64> = (0..n).map(|i| combo[i] + 1e-6 * noise[i]).collect();

    let q = |cols: &[&[f64]], intercept: bool| {
        nuisance_basis(&rows(cols), n, intercept).unwrap().len() / n
    };
    assert_eq!(q(&[&a, &b], true), 3);
    assert_eq!(q(&[&a, &a, &b], true), 3, "duplicate");
    assert_eq!(q(&[&a, &zero, &b], true), 3, "zero column");
    assert_eq!(
        q(&[&a, &seven, &b], true),
        3,
        "constant next to the intercept"
    );
    assert_eq!(q(&[&a, &seven, &b], false), 3, "constant with no intercept");
    assert_eq!(q(&[&a, &b, &combo], true), 3, "combination");
    assert_eq!(q(&[&a, &b, &nearly], true), 3, "combination within 1e-13");
    assert_eq!(q(&[&a, &b, &apart], true), 4, "1e-6 outside the span");
    assert_eq!(q(&[], true), 1);
    assert_eq!(q(&[], false), 0);

    // The duplicate is dropped, not the first copy: the basis equals the one
    // built without it, bit for bit.
    assert_eq!(
        nuisance_basis(&rows(&[&a, &a, &b]), n, true).unwrap(),
        nuisance_basis(&rows(&[&a, &b]), n, true).unwrap()
    );
}

#[test]
fn basis_rejects_bad_input() {
    assert_eq!(
        nuisance_basis(&[], 0, true),
        Err(NeuroError::TooFewSubjects)
    );
    assert_eq!(
        nuisance_basis(&[1.0, 2.0, 3.0, 4.0, 5.0], 3, true),
        Err(NeuroError::MismatchedLengths {
            expected: 6,
            got: 5
        })
    );
    for bad in [f64::NAN, f64::INFINITY, 1e200] {
        assert_eq!(
            nuisance_basis(&[1.0, bad, 3.0], 3, true),
            Err(NeuroError::NonFiniteStat),
            "{bad}"
        );
    }
}

/// O2 — every stored column is the residual of a least-squares fit of the
/// node on the raw nuisance columns, scaled to unit norm. With and without an
/// intercept, with and without confounds.
#[test]
fn store_columns_are_unit_norm_ols_residuals() {
    let (n, n_nodes) = (14usize, 9usize);
    let mut rng = Lcg(0x5EED_0003);
    let rows: Vec<usize> = (0..n).collect();
    // Brain data with a large mean, so centring matters.
    let y: Vec<f64> = rng.vec(n * n_nodes).iter().map(|x| 40.0 + x).collect();
    for c in [0usize, 3] {
        let confounds: Vec<f64> = rng.vec(n * c).iter().map(|x| 2.0 + x).collect();
        for intercept in [false, true] {
            let basis = nuisance_basis(&confounds, n, intercept).unwrap();
            let store = Store::from_f64(&y, n, n_nodes, &rows, &basis).unwrap();
            assert_eq!((store.n(), store.n_nodes()), (n, n_nodes));
            assert_eq!(store.q(), c + usize::from(intercept));
            let design = if intercept {
                with_ones(&confounds, n)
            } else {
                confounds.clone()
            };
            let p = store.q();
            for v in 0..n_nodes {
                let yv = column(&y, n, n_nodes, v);
                let resid = if p == 0 {
                    yv.clone()
                } else {
                    fit(&design, &yv, n, p).2
                };
                let norm = dot(&resid, &resid).sqrt();
                for (i, r) in resid.iter().enumerate() {
                    let got = store.as_slice()[v * n + i];
                    let want = r / norm;
                    assert!(
                        (got - want).abs() <= 1e-10,
                        "c {c} intercept {intercept} node {v} row {i}: {got} vs {want}"
                    );
                }
            }
        }
    }
}

/// O2 — an f32 source with a row index gives the bits of an f64 source that
/// holds the selected rows already. More nodes than one tile, rows out of
/// order.
#[test]
fn store_from_f32_with_a_row_index_matches_the_gathered_f64_source() {
    let (n_src, n_nodes) = (9usize, 700usize);
    let mut rng = Lcg(0x5EED_0004);
    let src32: Vec<f32> = (0..n_src * n_nodes)
        .map(|_| (3.0 + rng.next()) as f32)
        .collect();
    let rows = [7usize, 0, 3, 8, 2, 5];
    let n = rows.len();
    let confounds = rng.vec(n * 2);
    let basis = nuisance_basis(&confounds, n, true).unwrap();
    let gathered: Vec<f64> = rows
        .iter()
        .flat_map(|&r| {
            src32[r * n_nodes..(r + 1) * n_nodes]
                .iter()
                .map(|&x| f64::from(x))
        })
        .collect();
    let all: Vec<usize> = (0..n).collect();
    let want = Store::from_f64(&gathered, n, n_nodes, &all, &basis).unwrap();
    let got = Store::from_f32(&src32, n_src, n_nodes, &rows, &basis).unwrap();
    assert_eq!(got, want);
    assert_eq!(got.n(), n);
    // An f64 source with the same row index gives the same store too.
    let src64: Vec<f64> = src32.iter().map(|&x| f64::from(x)).collect();
    assert_eq!(
        Store::from_f64(&src64, n_src, n_nodes, &rows, &basis).unwrap(),
        want
    );
}

/// O2 — the degenerate-node rule: a node inside the span of the basis is
/// stored as zeros. With no basis nothing is removed, so a constant node stays
/// (it is caught later, by the residual-variance gate of the statistic).
#[test]
fn store_zeroes_nodes_in_the_span_of_the_basis() {
    let n = 8usize;
    let mut rng = Lcg(0x5EED_0005);
    let confound = rng.vec(n);
    let noise = rng.vec(n);
    // Nodes: constant, all zero, 2·confound + 3, ordinary.
    let n_nodes = 4;
    let mut y = vec![0.0f64; n * n_nodes];
    for i in 0..n {
        y[i * n_nodes] = 5.0;
        y[i * n_nodes + 2] = 2.0 * confound[i] + 3.0;
        y[i * n_nodes + 3] = noise[i];
    }
    let rows: Vec<usize> = (0..n).collect();
    let basis = nuisance_basis(&confound, n, true).unwrap();
    let store = Store::from_f64(&y, n, n_nodes, &rows, &basis).unwrap();
    let d = store.as_slice();
    for v in 0..3 {
        assert_eq!(&d[v * n..(v + 1) * n], &vec![0.0; n][..], "node {v}");
    }
    let last = &d[3 * n..];
    assert!((dot(last, last) - 1.0).abs() < 1e-14);

    let plain = Store::from_f64(&y, n, n_nodes, &rows, &[]).unwrap();
    let s = 1.0 / (n as f64).sqrt();
    assert!(plain.as_slice()[..n].iter().all(|&x| (x - s).abs() < 1e-15));
    assert_eq!(&plain.as_slice()[n..2 * n], &vec![0.0; n][..]);
}

#[test]
fn store_rejects_bad_input() {
    let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let all = [0usize, 1, 2];
    assert_eq!(
        Store::from_f64(&y, 3, 3, &all, &[]),
        Err(NeuroError::MismatchedLengths {
            expected: 9,
            got: 6
        })
    );
    assert_eq!(
        Store::from_f64(&y, 3, 2, &[], &[]),
        Err(NeuroError::TooFewSubjects)
    );
    assert_eq!(
        Store::from_f64(&y, 3, 2, &[0, 3], &[]),
        Err(NeuroError::InvalidParams)
    );
    // A basis for 2 subjects with 3 selected rows.
    assert_eq!(
        Store::from_f64(&y, 3, 2, &all, &[0.6, 0.8]),
        Err(NeuroError::MismatchedLengths {
            expected: 3,
            got: 2
        })
    );
    for bad in [f64::NAN, f64::NEG_INFINITY, 1e200] {
        let mut z = y;
        z[3] = bad;
        assert_eq!(
            Store::from_f64(&z, 3, 2, &all, &[]),
            Err(NeuroError::NonFiniteStat),
            "{bad}"
        );
    }
    // A bad value in a row that is not selected is never read.
    let mut z = y;
    z[4] = f64::NAN;
    assert!(Store::from_f64(&z, 3, 2, &[0, 1], &[]).is_ok());
    assert_eq!(
        Store::from_f32(&[1.0f32, f32::NAN], 2, 1, &[0, 1], &[]),
        Err(NeuroError::NonFiniteStat)
    );
    // No nodes: an empty store.
    let empty = Store::from_f64(&[], 3, 0, &all, &[]).unwrap();
    assert_eq!(
        (empty.n(), empty.n_nodes(), empty.as_slice().len()),
        (3, 0, 0)
    );
}

const EXACT: TfceParams = TfceParams {
    e: 0.5,
    h: 2.0,
    start: 0.0,
    step: 1.0,
    weighting: Weighting::Exact,
};

/// One linear model for the tests: brain data with an effect of the tested
/// term on the first nodes, a tested regressor correlated with the confounds,
/// and confounds with a non-zero mean.
struct Model {
    n: usize,
    n_nodes: usize,
    y: Vec<f64>,
    tested: Option<Vec<f64>>,
    confounds: Vec<f64>,
}

impl Model {
    fn new(seed: u64, n: usize, n_nodes: usize, c: usize, with_tested: bool) -> Self {
        let mut rng = Lcg(seed);
        let confounds: Vec<f64> = rng.vec(n * c).iter().map(|x| 1.0 + x).collect();
        let tested: Option<Vec<f64>> = with_tested.then(|| {
            (0..n)
                .map(|i| rng.next() + 0.3 * confounds[i * c..(i + 1) * c].iter().sum::<f64>())
                .collect()
        });
        let slopes = rng.vec(c * n_nodes);
        let mut y = vec![0.0f64; n * n_nodes];
        for i in 0..n {
            for v in 0..n_nodes {
                let nuisance: f64 = (0..c)
                    .map(|j| confounds[i * c + j] * slopes[j * n_nodes + v])
                    .sum();
                let effect = if v < n_nodes / 3 {
                    tested.as_ref().map_or(0.9, |t| 0.8 * t[i])
                } else {
                    0.0
                };
                y[i * n_nodes + v] = 2.0 + nuisance + effect + rng.next();
            }
        }
        Self {
            n,
            n_nodes,
            y,
            tested,
            confounds,
        }
    }

    fn c(&self) -> usize {
        self.confounds.len() / self.n
    }

    fn store(&self) -> Store {
        let basis = nuisance_basis(&self.confounds, self.n, self.tested.is_some()).unwrap();
        let rows: Vec<usize> = (0..self.n).collect();
        Store::from_f64(&self.y, self.n, self.n_nodes, &rows, &basis).unwrap()
    }

    /// The raw design, row-major: `[tested, confounds, 1]`, or
    /// `[1, confounds]` when the intercept is tested. The tested term is
    /// column 0.
    fn design(&self) -> (Vec<f64>, usize) {
        let (n, c) = (self.n, self.c());
        let p = c + 1 + usize::from(self.tested.is_some());
        let mut x = Vec::with_capacity(n * p);
        for i in 0..n {
            match &self.tested {
                Some(t) => {
                    x.push(t[i]);
                    x.extend_from_slice(&self.confounds[i * c..(i + 1) * c]);
                    x.push(1.0);
                }
                None => {
                    x.push(1.0);
                    x.extend_from_slice(&self.confounds[i * c..(i + 1) * c]);
                }
            }
        }
        (x, p)
    }

    /// The nuisance columns alone, row-major: `[confounds, 1]`, or
    /// `[confounds]` when the intercept is tested.
    fn nuisance(&self) -> (Vec<f64>, usize) {
        if self.tested.is_some() {
            (with_ones(&self.confounds, self.n), self.c() + 1)
        } else {
            (self.confounds.clone(), self.c())
        }
    }

    /// The brain data with the nuisance fit removed, subject-major.
    fn residualised(&self) -> Vec<f64> {
        let (z, pz) = self.nuisance();
        if pz == 0 {
            return self.y.clone();
        }
        let mut out = vec![0.0f64; self.n * self.n_nodes];
        for v in 0..self.n_nodes {
            let resid = fit(&z, &column(&self.y, self.n, self.n_nodes, v), self.n, pz).2;
            for (i, r) in resid.iter().enumerate() {
                out[i * self.n_nodes + v] = *r;
            }
        }
        out
    }

    /// The design of draw `draw`: rows shuffled by `draw_permutation`, or
    /// sign-flipped by `commonstats::gen_sign_flips`, as the engine draws them.
    fn design_of_draw(&self, draw: u64, seed: u64) -> Vec<f64> {
        let (x, p) = self.design();
        let n = self.n;
        let mut out = vec![0.0f64; n * p];
        if self.tested.is_some() {
            let mut perm = vec![0u32; n];
            draw_permutation(n, draw, seed, &mut perm).unwrap();
            for i in 0..n {
                let r = perm[i] as usize;
                out[i * p..(i + 1) * p].copy_from_slice(&x[r * p..(r + 1) * p]);
            }
        } else {
            let mut signs = vec![1.0f64; n];
            if draw > 0 {
                gen_sign_flips(n, draw, seed, &mut signs).unwrap();
            }
            for i in 0..n {
                for j in 0..p {
                    out[i * p + j] = signs[i] * x[i * p + j];
                }
            }
        }
        out
    }
}

/// The t of column 0 in a least-squares fit of every column of `y`
/// (subject-major `n × n_nodes`) on `x` (row-major `n × p`):
/// `beta / sqrt(rss / (n − p) · (X'X)⁻¹[0, 0])`.
fn direct_t(x: &[f64], p: usize, y: &[f64], n: usize, n_nodes: usize) -> Vec<f64> {
    (0..n_nodes)
        .map(|v| {
            let (beta, inv, resid) = fit(x, &column(y, n, n_nodes, v), n, p);
            let sigma2 = dot(&resid, &resid) / (n - p) as f64;
            beta[0] / (sigma2 * inv[0]).sqrt()
        })
        .collect()
}

fn assert_close(label: &str, got: &[f64], want: &[f64], rel: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (v, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= rel * w.abs().max(1.0),
            "{label}[{v}]: got {g}, want {w}"
        );
    }
}

/// The four designs: a tested regressor or the intercept, with and without
/// confounds.
fn models() -> Vec<(&'static str, Model)> {
    vec![
        ("regressor + confounds", Model::new(0x0A11, 16, 12, 2, true)),
        ("regressor", Model::new(0x0A12, 16, 12, 0, true)),
        (
            "intercept + confounds",
            Model::new(0x0A13, 16, 12, 2, false),
        ),
        ("intercept", Model::new(0x0A14, 16, 12, 0, false)),
    ]
}

/// O3 — the statistic of every draw equals the t of a direct fit: the design
/// rows transformed by the draw, fitted to the nuisance-residualised data by
/// the normal equations. Draw 0 also equals the plain OLS t of the raw data.
/// With confounds the nuisance part of the residual variance changes from
/// draw to draw, so a loop that kept the observed denominator would fail
/// here; the test checks that the data make that difference visible.
#[test]
fn o3_draw_statistics_match_per_draw_fits() {
    let seed = 77u64;
    let b = 24u64;
    for (label, m) in models() {
        let dom = Domain::from_volume([m.n_nodes, 1, 1], Conn::Face);
        let store = m.store();
        let mut opts = PermOpts::new(EXACT, Tail::TwoSided, seed, b);
        opts.batch = 5;
        let problem = PermProblem::ols(&dom, &store, m.tested.as_deref(), &opts).unwrap();
        let mut ws = PermWorkspace::new(&problem);
        let (x, p) = m.design();
        let (n, v) = (m.n, m.n_nodes);
        let residualised = m.residualised();

        assert_close(
            &format!("{label} stat_obs vs raw fit"),
            problem.stat_obs(),
            &direct_t(&x, p, &m.y, n, v),
            1e-9,
        );
        let mut moved = 0.0f64;
        for first in (0..b).step_by(5) {
            let last = b.min(first + 5);
            let maps = stat_batch(&problem, first..last, &mut ws).to_vec();
            for draw in first..last {
                let xb = m.design_of_draw(draw, seed);
                assert_close(
                    &format!("{label} draw {draw}"),
                    &maps[(draw - first) as usize * v..][..v],
                    &direct_t(&xb, p, &residualised, n, v),
                    1e-9,
                );
                // Squared length of the node's store column inside the span
                // of the draw's nuisance columns (columns 1.. of the design).
                if p > 1 {
                    let nuisance: Vec<f64> = (0..n)
                        .flat_map(|i| xb[i * p + 1..(i + 1) * p].to_vec())
                        .collect();
                    let d = &store.as_slice()[..n];
                    let (_, _, resid) = fit(&nuisance, d, n, p - 1);
                    let inside = 1.0 - dot(&resid, &resid);
                    if draw == 0 {
                        assert!(inside.abs() < 1e-12, "{label}: observed {inside}");
                    } else {
                        moved = moved.max(inside);
                    }
                }
            }
        }
        if m.c() > 0 {
            assert!(moved > 0.05, "{label}: nuisance projections moved {moved}");
        }
    }
}

/// `tfce_ols` on `m` against a loop written the slow way. `designs[b]` is the
/// design of draw `b` (row-major, the tested term in column 0), and the run
/// has `designs.len()` draws for the budget `n_perm`. For every draw: one
/// direct fit per node on the nuisance-residualised data, TFCE of the map and
/// of its negation, the null value and the per-node counts. Maps within
/// `rel 1e-8`; with them equal, the p-maps are equal.
fn assert_matches_naive_loop(
    label: &str,
    dom: &Domain,
    m: &Model,
    designs: &[Vec<f64>],
    seed: u64,
    n_perm: u64,
) {
    let v = dom.n_nodes();
    let b = designs.len();
    let (x, p) = m.design();
    let residualised = m.residualised();
    // Per draw: TFCE(t) and TFCE(−t).
    let draws: Vec<(Vec<f64>, Vec<f64>)> = designs
        .iter()
        .map(|design| {
            let t = direct_t(design, p, &residualised, m.n, v);
            let neg: Vec<f64> = t.iter().map(|s| -s).collect();
            (
                tfce(dom, &t, &EXACT).unwrap(),
                tfce(dom, &neg, &EXACT).unwrap(),
            )
        })
        .collect();
    for tail in [Tail::Positive, Tail::Negative, Tail::TwoSided] {
        // (signed TFCE map, tested quantity) of a draw under `tail`.
        let quantity = |up: &[f64], down: &[f64]| -> (Vec<f64>, Vec<f64>) {
            match tail {
                Tail::Positive => (up.to_vec(), up.to_vec()),
                Tail::Negative => (down.to_vec(), down.to_vec()),
                Tail::TwoSided => {
                    let signed: Vec<f64> = up.iter().zip(down).map(|(a, b)| a - b).collect();
                    let abs = signed.iter().map(|s| s.abs()).collect();
                    (signed, abs)
                }
            }
        };
        let (tfce_obs, tested_obs) = quantity(&draws[0].0, &draws[0].1);
        let mut maxima = Vec::new();
        let mut exceed = vec![0u32; v];
        for (up, down) in &draws {
            let (_, q) = quantity(up, down);
            maxima.push(q.iter().fold(0.0f64, |a, &s| a.max(s)));
            for (cnt, (qb, qo)) in exceed.iter_mut().zip(q.iter().zip(&tested_obs)) {
                if qb >= qo {
                    *cnt += 1;
                }
            }
        }
        maxima.sort_unstable_by(f64::total_cmp);
        let p_fwe: Vec<f64> = tested_obs
            .iter()
            .map(|t| maxima.iter().filter(|m| *m >= t).count() as f64 / b as f64)
            .collect();
        let p_unc: Vec<f64> = exceed.iter().map(|&k| f64::from(k) / b as f64).collect();

        let opts = PermOpts::new(EXACT, tail, seed, n_perm);
        let r = tfce_ols(dom, &m.y, m.n, m.tested.as_deref(), &m.confounds, &opts).unwrap();
        let name = format!("{label} {tail:?}");
        assert_eq!(r.null.maxima.len(), b, "{name}: number of draws");
        assert_close(
            &format!("{name} stat_obs"),
            &r.stat_obs,
            &direct_t(&x, p, &m.y, m.n, v),
            1e-9,
        );
        assert_close(
            &format!("{name} tfce_obs"),
            &r.null.tfce_obs,
            &tfce_obs,
            1e-8,
        );
        assert_close(&format!("{name} maxima"), &r.null.maxima, &maxima, 1e-8);
        assert_eq!(r.p_fwe, p_fwe, "{name} p_fwe");
        assert_eq!(r.p_unc, p_unc, "{name} p_unc");
        assert_eq!(r.null.exceed, exceed, "{name} exceed");
        assert_eq!(
            (r.null.seed, r.null.n, r.null.n_nodes, r.null.tail),
            (seed, m.n, v, tail)
        );
        assert_eq!(r.null.q, m.c() + usize::from(m.tested.is_some()));
        assert!(r.p_fwe.iter().chain(&r.p_unc).all(|&p| p >= 1.0 / b as f64));
    }
}

/// O4 — Monte Carlo draws: a tested regressor with confounds (row shuffles)
/// and the intercept with confounds (sign flips), 60 draws each.
#[test]
fn o4_inference_matches_a_naive_loop() {
    let dom = Domain::from_volume([3, 2, 2], Conn::Face);
    let (seed, b) = (2026u64, 60u64);
    for (label, with_tested) in [
        ("regressor + confounds", true),
        ("intercept + confounds", false),
    ] {
        let m = Model::new(
            0x0B02 + u64::from(with_tested),
            13,
            dom.n_nodes(),
            2,
            with_tested,
        );
        let designs: Vec<Vec<f64>> = (0..b).map(|draw| m.design_of_draw(draw, seed)).collect();
        assert_matches_naive_loop(label, &dom, &m, &designs, seed, b);
    }
}

/// O4 — exhaustive enumeration on a tiny `n`: the intercept with one confound,
/// `n = 5`, so a budget of 100 runs all `2^5 = 32` sign patterns. The naive
/// loop builds pattern `k` from the bits of `k` (bit `i` set flips subject
/// `i`), which needs no random draw at all.
#[test]
fn o4_exhaustive_sign_enumeration_matches_a_naive_loop() {
    let dom = Domain::from_volume([3, 2, 1], Conn::Face);
    let m = Model::new(0x0B10, 5, dom.n_nodes(), 1, false);
    let (x, p) = m.design();
    let designs: Vec<Vec<f64>> = (0..32u32)
        .map(|k| {
            let mut design = x.clone();
            for (i, row) in design.chunks_exact_mut(p).enumerate() {
                if (k >> i) & 1 == 1 {
                    row.iter_mut().for_each(|value| *value = -*value);
                }
            }
            design
        })
        .collect();
    // Draw k of the enumerated run is pattern k (bit i of k flips subject i;
    // draw 0 is the identity), map by map through the public `stat_batch`.
    let v = dom.n_nodes();
    let store = m.store();
    let problem = PermProblem::ols(
        &dom,
        &store,
        None,
        &PermOpts::new(EXACT, Tail::Positive, 9, 100),
    )
    .unwrap();
    assert_eq!(problem.draws(), 0..32);
    let mut ws = PermWorkspace::new(&problem);
    let residualised = m.residualised();
    let k = problem.batch() as u64;
    for first in (0..32u64).step_by(k as usize) {
        let last = 32.min(first + k);
        let maps = stat_batch(&problem, first..last, &mut ws).to_vec();
        for draw in first..last {
            assert_close(
                &format!("exhaustive draw {draw}"),
                &maps[(draw - first) as usize * v..][..v],
                &direct_t(&designs[draw as usize], p, &residualised, m.n, v),
                1e-9,
            );
        }
    }
    assert_matches_naive_loop("exhaustive", &dom, &m, &designs, 9, 100);
}

/// The public row permutation: draw 0 is the identity, and draw `b ≥ 1` is
/// `commonstats::gen_resample_indices(Scheme::Permutation, n, b, seed)` for
/// the caller's seed (the permutation properties of that stream are
/// commonstats' own tests).
#[test]
fn draw_permutation_is_a_pure_function_of_seed_and_draw() {
    let n = 30usize;
    let mut a = vec![0u32; n];
    let mut b = vec![0u32; n];
    draw_permutation(n, 0, 9, &mut a).unwrap();
    assert_eq!(a, (0..n as u32).collect::<Vec<u32>>());
    for seed in [9u64, 10] {
        for draw in 1..40u64 {
            draw_permutation(n, draw, seed, &mut a).unwrap();
            commonstats::gen_resample_indices(
                commonstats::Scheme::Permutation,
                n,
                draw,
                seed,
                &mut b,
            )
            .unwrap();
            assert_eq!(a, b, "seed {seed} draw {draw}");
        }
    }
    assert_eq!(
        draw_permutation(n, 1, 9, &mut a[..n - 1]),
        Err(NeuroError::MismatchedLengths {
            expected: n,
            got: n - 1
        })
    );
}

/// O5 — a design the model cannot be fitted with is refused.
#[test]
fn o5_degenerate_designs_are_refused() {
    let m = Model::new(0x0C01, 10, 6, 2, true);
    let dom = Domain::from_volume([m.n_nodes, 1, 1], Conn::Face);
    let opts = PermOpts::new(EXACT, Tail::TwoSided, 1, 20);
    let (n, c) = (m.n, m.c());
    let run = |tested: Option<&[f64]>, confounds: &[f64]| {
        tfce_ols(&dom, &m.y, n, tested, confounds, &opts).map(|r| r.null.q)
    };
    assert_eq!(run(m.tested.as_deref(), &m.confounds), Ok(3));

    // Tested regressor in the span of the confounds and the intercept.
    let in_span: Vec<f64> = (0..n)
        .map(|i| 2.0 * m.confounds[i * c] - m.confounds[i * c + 1] + 4.0)
        .collect();
    assert_eq!(
        run(Some(&in_span), &m.confounds),
        Err(NeuroError::DegenerateDesign)
    );
    // A constant tested regressor is the intercept.
    assert_eq!(
        run(Some(&vec![3.0; n]), &[]),
        Err(NeuroError::DegenerateDesign)
    );
    assert_eq!(
        run(Some(&vec![0.0; n]), &[]),
        Err(NeuroError::DegenerateDesign)
    );
    // The intercept is tested and a confound is constant.
    let with_constant: Vec<f64> = (0..n).flat_map(|i| [m.confounds[i * c], 2.0]).collect();
    assert_eq!(run(None, &with_constant), Err(NeuroError::DegenerateDesign));
    // The same confounds are fine when a regressor is tested: the constant
    // column is dropped next to the intercept.
    assert_eq!(run(m.tested.as_deref(), &with_constant), Ok(2));

    // A duplicated confound column changes nothing, bit for bit.
    let doubled: Vec<f64> = (0..n)
        .flat_map(|i| {
            [
                m.confounds[i * c],
                m.confounds[i * c],
                m.confounds[i * c + 1],
            ]
        })
        .collect();
    assert_eq!(
        tfce_ols(&dom, &m.y, n, m.tested.as_deref(), &doubled, &opts),
        tfce_ols(&dom, &m.y, n, m.tested.as_deref(), &m.confounds, &opts)
    );

    // n too small: n − q − 1 must be at least 1.
    let small_dom = Domain::from_volume([2, 1, 1], Conn::Face);
    let small = |n: usize, tested: Option<&[f64]>, confounds: &[f64]| {
        let y: Vec<f64> = (0..2 * n).map(|k| (k * k % 7) as f64).collect();
        tfce_ols(&small_dom, &y, n, tested, confounds, &opts).map(|r| r.null.q)
    };
    assert_eq!(small(1, None, &[]), Err(NeuroError::TooFewSubjects));
    assert_eq!(small(2, None, &[]), Ok(0));
    assert_eq!(
        small(2, Some(&[1.0, 2.0]), &[]),
        Err(NeuroError::TooFewSubjects)
    );
    assert_eq!(small(3, Some(&[1.0, 2.0, 4.0]), &[]), Ok(1));
    assert_eq!(
        small(3, Some(&[1.0, 2.0, 4.0]), &[0.0, 1.0, 0.0]),
        Err(NeuroError::TooFewSubjects)
    );
    assert_eq!(small(0, None, &[]), Err(NeuroError::TooFewSubjects));
}

/// O5 — wrong lengths and non-finite values are errors, never a panic in the
/// draw loop.
#[test]
fn o5_bad_inputs_are_errors() {
    let m = Model::new(0x0C02, 8, 4, 1, true);
    let dom = Domain::from_volume([m.n_nodes, 1, 1], Conn::Face);
    let opts = PermOpts::new(EXACT, Tail::Positive, 1, 20);
    let tested = m.tested.clone().unwrap();
    let n = m.n;
    let run = |y: &[f64], tested: Option<&[f64]>, confounds: &[f64], opts: &PermOpts| {
        tfce_ols(&dom, y, n, tested, confounds, opts).map(|_| ())
    };
    assert_eq!(run(&m.y, Some(&tested), &m.confounds, &opts), Ok(()));
    assert_eq!(
        run(&m.y[1..], Some(&tested), &m.confounds, &opts),
        Err(NeuroError::MismatchedLengths {
            expected: n * m.n_nodes,
            got: n * m.n_nodes - 1
        })
    );
    assert_eq!(
        run(&m.y, Some(&tested[1..]), &m.confounds, &opts),
        Err(NeuroError::MismatchedLengths {
            expected: n,
            got: n - 1
        })
    );
    assert_eq!(
        run(&m.y, Some(&tested), &m.confounds[1..], &opts),
        Err(NeuroError::MismatchedLengths {
            expected: n,
            got: n - 1
        })
    );
    for bad in [f64::NAN, f64::INFINITY] {
        let mut y = m.y.clone();
        y[5] = bad;
        assert_eq!(
            run(&y, Some(&tested), &m.confounds, &opts),
            Err(NeuroError::NonFiniteStat)
        );
        let mut t = tested.clone();
        t[2] = bad;
        assert_eq!(
            run(&m.y, Some(&t), &m.confounds, &opts),
            Err(NeuroError::NonFiniteStat)
        );
        let mut z = m.confounds.clone();
        z[3] = bad;
        assert_eq!(
            run(&m.y, Some(&tested), &z, &opts),
            Err(NeuroError::NonFiniteStat)
        );
    }
    // A draw budget of 0 or above u32::MAX, a batch of 0 (which would never
    // advance the draw loop).
    for bad in [
        PermOpts { n_perm: 0, ..opts },
        PermOpts { batch: 0, ..opts },
        PermOpts {
            n_perm: u64::from(u32::MAX) + 1,
            ..opts
        },
    ] {
        assert_eq!(
            run(&m.y, Some(&tested), &m.confounds, &bad),
            Err(NeuroError::InvalidParams)
        );
    }
    let bad_params = TfceParams { h: -2.0, ..EXACT };
    assert_eq!(
        run(
            &m.y,
            Some(&tested),
            &m.confounds,
            &PermOpts::new(bad_params, Tail::Positive, 1, 20)
        ),
        Err(NeuroError::InvalidParams)
    );
    // A store for another domain.
    let other = Domain::from_volume([m.n_nodes + 1, 1, 1], Conn::Face);
    assert_eq!(
        PermProblem::ols(&other, &m.store(), Some(&tested), &opts).map(|_| ()),
        Err(NeuroError::MismatchedLengths {
            expected: m.n_nodes + 1,
            got: m.n_nodes
        })
    );
}

/// O5 — maps with nothing to find: constant nodes get statistic 0 and p = 1;
/// a run of one draw is the identity alone; a domain with no nodes gives
/// empty maps and a null of zeros.
#[test]
fn o5_empty_and_constant_data_give_defined_results() {
    let (n, v) = (9usize, 5usize);
    let dom = Domain::from_volume([v, 1, 1], Conn::Face);
    let mut rng = Lcg(0x0C03);
    let tested = rng.vec(n);
    // Node 2 is constant, node 4 all zero, the others noise.
    let mut y = rng.vec(n * v);
    for i in 0..n {
        y[i * v + 2] = 7.5;
        y[i * v + 4] = 0.0;
    }
    for tested in [None, Some(&tested[..])] {
        for tail in [Tail::Positive, Tail::Negative, Tail::TwoSided] {
            let r = tfce_ols(&dom, &y, n, tested, &[], &PermOpts::new(EXACT, tail, 3, 50)).unwrap();
            for node in [2usize, 4] {
                assert_eq!(r.stat_obs[node], 0.0, "{tail:?} node {node}");
                assert_eq!(r.null.tfce_obs[node], 0.0);
                assert_eq!((r.p_fwe[node], r.p_unc[node]), (1.0, 1.0));
            }
            assert!(r.stat_obs.iter().all(|s| s.is_finite()));
        }
    }
    // Every node constant. With a tested regressor the intercept is a nuisance
    // term and the store holds zeros: every draw is a map of zeros. When the
    // intercept is tested a sign-flipped constant is no longer constant, so
    // only the observed map is zero; p is 1 either way.
    let flat = vec![1.25f64; n * v];
    let opts = PermOpts::new(EXACT, Tail::TwoSided, 3, 50);
    let r = tfce_ols(&dom, &flat, n, Some(&tested), &[], &opts).unwrap();
    assert_eq!(r.stat_obs, vec![0.0; v]);
    assert_eq!(r.null.maxima, vec![0.0; 50]);
    assert_eq!(r.p_fwe, vec![1.0; v]);
    let r = tfce_ols(&dom, &flat, n, None, &[], &opts).unwrap();
    assert_eq!(r.stat_obs, vec![0.0; v]);
    assert_eq!(r.null.maxima[0], 0.0);
    assert_eq!(r.p_fwe, vec![1.0; v]);
    assert_eq!(r.p_unc, vec![1.0; v]);

    // One draw, and a batch larger than the run.
    let mut opts = PermOpts::new(EXACT, Tail::Positive, 3, 1);
    opts.batch = 64;
    let one = tfce_ols(&dom, &y, n, Some(&tested), &[], &opts).unwrap();
    assert_eq!(one.null.maxima.len(), 1);
    assert_eq!(one.p_fwe, vec![1.0; v]);
    assert_eq!(one.p_unc, vec![1.0; v]);

    // No nodes.
    let none = Domain::from_mask(&[false; 4], [4, 1, 1], Conn::Face).unwrap();
    let r = tfce_ols(
        &none,
        &[],
        n,
        Some(&tested),
        &[],
        &PermOpts::new(EXACT, Tail::TwoSided, 3, 12),
    )
    .unwrap();
    assert!(r.stat_obs.is_empty() && r.p_fwe.is_empty() && r.p_unc.is_empty());
    assert_eq!(r.null.maxima, vec![0.0; 12]);
}

/// O7 — the Freedman–Lane loop written out. For every draw: take the
/// residuals of the confound-only model, permute them, add the confound fit
/// back, fit the full model with the design as observed, and read the t of
/// the tested regressor. The engine shuffles the design rows instead and
/// leaves the data alone; both give the same statistic. With `perm` the
/// engine's row permutation (shuffled row `i` is original row `perm[i]`),
/// the permuted data are `y*[perm[i]] = r[i] + fit[perm[i]]`.
#[test]
fn o7_shuffled_design_gives_the_freedman_lane_statistic() {
    let (seed, b) = (404u64, 16u64);
    for (label, m) in models().into_iter().filter(|(_, m)| m.tested.is_some()) {
        let dom = Domain::from_volume([m.n_nodes, 1, 1], Conn::Face);
        let store = m.store();
        let opts = PermOpts::new(EXACT, Tail::TwoSided, seed, b);
        let problem = PermProblem::ols(&dom, &store, m.tested.as_deref(), &opts).unwrap();
        let mut ws = PermWorkspace::new(&problem);
        let (n, v) = (m.n, m.n_nodes);
        let (x, p) = m.design();
        let residuals = m.residualised();
        let mut perm = vec![0u32; n];
        for first in (0..b).step_by(problem.batch()) {
            let last = b.min(first + problem.batch() as u64);
            let maps = stat_batch(&problem, first..last, &mut ws).to_vec();
            for draw in first..last {
                draw_permutation(n, draw, seed, &mut perm).unwrap();
                let mut permuted = vec![0.0f64; n * v];
                for i in 0..n {
                    let to = perm[i] as usize;
                    for node in 0..v {
                        let fitted = m.y[to * v + node] - residuals[to * v + node];
                        permuted[to * v + node] = residuals[i * v + node] + fitted;
                    }
                }
                assert_close(
                    &format!("{label} draw {draw}"),
                    &maps[(draw - first) as usize * v..][..v],
                    &direct_t(&x, p, &permuted, n, v),
                    1e-9,
                );
            }
        }
    }
}
