//! t-digest statistical-tolerance suite (area 2). No oracle fixture: the digest
//! is the crate's sole approximation, so references come from exact (sort-based)
//! computation on deterministic, rng-free data. The bound is the k1 rank error
//! `3/(4δ)` = 0.75% at δ = 100.

use commonstats::accum::{quantile_edges, Mergeable, TDigest};

/// Deterministic data, no `rng` feature: an in-test LCG over [0, 1)·scale.
/// (Glibc-style constants; purely to get spread-out, reproducible values.)
fn lcg_sample(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 1000.0
        })
        .collect()
}

/// Exact type-7 quantile on a sorted slice — mirrors `accum::quantile_sorted`
/// (which is `pub(crate)` and not reachable from an integration test): linear
/// interpolation at `q·(n−1)`.
fn exact_quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = q * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

fn digest_of(xs: &[f64]) -> TDigest {
    let mut d = TDigest::new(TDigest::default_delta()).unwrap();
    for &x in xs {
        d.update(x);
    }
    d
}

// (1) accuracy: |cdf(exact_q_value) − q| ≤ 0.0075 over a q grid.
#[test]
fn accuracy_rank_error_within_bound() {
    let xs = lcg_sample(20_000, 0xDEADBEEF);
    let mut sorted = xs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let d = digest_of(&xs);
    for &q in &[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
        let value = exact_quantile_sorted(&sorted, q);
        let est_rank = d.cdf(value); // estimated cumulative prob at the true q-value
        assert!(
            (est_rank - q).abs() <= 0.0075,
            "q={q}: cdf={est_rank}, |err|={} > 0.0075",
            (est_rank - q).abs()
        );
    }
}

// (2) merge associativity (split A|B|C, left vs right fold).
#[test]
fn merge_associativity() {
    let xs = lcg_sample(9000, 7);
    let (a, b, c) = (&xs[0..3000], &xs[3000..6000], &xs[6000..9000]);
    let (da, db, dc) = (digest_of(a), digest_of(b), digest_of(c));
    let mut left = da.clone();
    left.merge(&db);
    left.merge(&dc);
    let mut bc = db.clone();
    bc.merge(&dc);
    let mut right = da.clone();
    right.merge(&bc);
    for &q in &[0.05, 0.25, 0.5, 0.75, 0.95] {
        let l = left.quantile(q).unwrap();
        let r = right.quantile(q).unwrap();
        assert!((left.cdf(l) - right.cdf(r)).abs() <= 0.0075, "q={q}");
    }
    assert_eq!(left.count(), right.count());
}

// (3) chunked == streaming.
#[test]
fn chunked_equals_streaming() {
    let xs = lcg_sample(8000, 99);
    let streamed = digest_of(&xs);
    let mut chunked = TDigest::empty();
    for chunk in xs.chunks(1000) {
        chunked.merge(&digest_of(chunk));
    }
    assert_eq!(chunked.count(), streamed.count());
    assert_eq!(chunked.min(), streamed.min());
    assert_eq!(chunked.max(), streamed.max());
    for &q in &[0.05, 0.25, 0.5, 0.75, 0.95] {
        let target = streamed.quantile(q).unwrap();
        assert!((chunked.cdf(target) - q).abs() <= 0.0075, "q={q}");
    }
}

// (4) empty / identity.
#[test]
fn empty_identity() {
    let d = TDigest::empty();
    assert_eq!(d.count(), 0);
    assert_eq!(d.min(), f64::INFINITY);
    assert_eq!(d.max(), f64::NEG_INFINITY);
    assert!(d.cdf(0.0).is_nan());
    assert!(d.quantile(0.5).is_err());
    // empty is the merge identity
    let full = digest_of(&lcg_sample(500, 3));
    let mut e = TDigest::empty();
    e.merge(&full);
    assert_eq!(e.count(), full.count());
    assert_eq!(e.quantile(0.5).unwrap(), full.quantile(0.5).unwrap());
}

// (5) single value.
#[test]
fn single_value() {
    let d = digest_of(&[42.0]);
    assert_eq!(d.count(), 1);
    assert_eq!(d.min(), 42.0);
    assert_eq!(d.max(), 42.0);
    assert_eq!(d.quantile(0.0).unwrap(), 42.0);
    assert_eq!(d.quantile(0.5).unwrap(), 42.0);
    assert_eq!(d.quantile(1.0).unwrap(), 42.0);
    assert_eq!(d.cdf(42.0), 1.0);
    assert_eq!(d.cdf(10.0), 0.0);
}

// (6) duplicates.
#[test]
fn duplicates() {
    let mut xs = vec![5.0; 1000];
    xs.extend(vec![5.0; 1000]);
    let d = digest_of(&xs);
    assert_eq!(d.count(), 2000);
    assert_eq!(d.quantile(0.5).unwrap(), 5.0);
    assert_eq!(d.min(), 5.0);
    assert_eq!(d.max(), 5.0);
}

// (7) NaN omitted.
#[test]
fn nan_omitted() {
    let mut d = TDigest::new(100.0).unwrap();
    d.update(1.0);
    d.update(f64::NAN);
    d.update(3.0);
    d.update(f64::NAN);
    assert_eq!(d.count(), 2);
    assert_eq!(d.min(), 1.0);
    assert_eq!(d.max(), 3.0);
}

// (8) new rejects δ<1 / NaN.
#[test]
fn new_rejects_bad_delta() {
    assert!(TDigest::new(0.5).is_err());
    assert!(TDigest::new(0.0).is_err());
    assert!(TDigest::new(f64::NAN).is_err());
    assert!(TDigest::new(1.0).is_ok());
}

// (9) cdf monotone in [0,1].
#[test]
fn cdf_monotone() {
    let d = digest_of(&lcg_sample(5000, 123));
    let mut prev = f64::NEG_INFINITY;
    for i in 0..=400 {
        let x = i as f64 / 400.0 * 1000.0;
        let c = d.cdf(x);
        assert!(c >= prev - 1e-12, "cdf decreased at x={x}: {c} < {prev}");
        assert!((0.0..=1.0).contains(&c), "cdf out of [0,1] at x={x}: {c}");
        prev = c;
    }
}

// quantile_edges integration (area-3 contract).
#[test]
fn quantile_edges_for_area3() {
    let d = digest_of(&lcg_sample(5000, 55));
    let edges = quantile_edges(&d, 10).unwrap();
    assert_eq!(edges.len(), 11);
    assert_eq!(edges[0], d.min());
    assert_eq!(edges[10], d.max());
    for w in edges.windows(2) {
        assert!(w[1] >= w[0]);
    }
}
