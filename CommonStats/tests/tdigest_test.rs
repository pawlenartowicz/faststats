//! t-digest statistical-tolerance suite (area 2). No oracle fixture: the digest
//! is the crate's sole approximation, so references come from exact (sort-based)
//! computation on deterministic, rng-free data. The bound is the k1 rank error
//! `3/(4δ)` = 0.75% at δ = 100.

use commonstats::accum::{Mergeable, TDigest, quantile_edges};

/// Deterministic data, no `rng` feature: an in-test LCG over [0, 1)·scale.
/// (Glibc-style constants; purely to get spread-out, reproducible values.)
fn lcg_sample(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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

/// Fraction of the sorted sample ≤ `v`: the exact empirical CDF.
fn exact_rank(sorted: &[f64], v: f64) -> f64 {
    sorted.partition_point(|&x| x <= v) as f64 / sorted.len() as f64
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
        let rank = exact_rank(&sorted, d.quantile(q).unwrap());
        assert!(
            (rank - q).abs() <= 0.0075,
            "q={q}: exact rank of quantile = {rank}"
        );
    }
}

// (2) merge associativity (split A|B|C, left vs right fold).
#[test]
fn merge_associativity() {
    let xs = lcg_sample(9000, 7);
    let mut sorted = xs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
        for d in [&left, &right] {
            let rank = exact_rank(&sorted, d.quantile(q).unwrap());
            assert!((rank - q).abs() <= 0.0075, "q={q}: rank {rank}");
        }
        let cross = right.cdf(left.quantile(q).unwrap());
        assert!(
            (cross - q).abs() <= 0.0075,
            "q={q}: right.cdf(left.quantile) = {cross}"
        );
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
    assert_eq!(d.quantile(0.5), Err(commonstats::StatError::EmptyInput));
    // empty is the merge identity on both sides
    let full = digest_of(&lcg_sample(2000, 3));
    let mut left = TDigest::empty();
    left.merge(&full);
    let mut right = full.clone();
    right.merge(&TDigest::empty());
    for q in [0.1, 0.5, 0.9] {
        assert_eq!(left.quantile(q).unwrap(), full.quantile(q).unwrap());
        assert_eq!(right.quantile(q).unwrap(), full.quantile(q).unwrap());
    }
    assert_eq!(left.count(), full.count());
    assert_eq!(right.count(), full.count());
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
    // Constant sample through two flushes: equal-mean centroids.
    let d = digest_of(&vec![5.0; 2000]);
    assert_eq!(d.count(), 2000);
    assert_eq!((d.min(), d.max()), (5.0, 5.0));
    assert_eq!(d.quantile(0.5).unwrap(), 5.0);
    assert_eq!(d.cdf(5.0), 1.0); // value-space span 0 at the end: t = 1
    assert_eq!(d.cdf(4.999), 0.0);
    // Two atoms: quantiles land on the atoms, cdf between them is the mass below.
    // (At an atom the cdf may sit anywhere in its jump, so it is not asserted there.)
    let mut xs = vec![5.0; 1000];
    xs.extend(vec![7.0; 1000]);
    let d = digest_of(&xs);
    assert_eq!(d.quantile(0.25).unwrap(), 5.0);
    assert_eq!(d.quantile(0.75).unwrap(), 7.0);
    assert!((d.cdf(6.0) - 0.5).abs() <= 0.0075);
    assert_eq!(d.cdf(7.0), 1.0);
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

// (8) cdf monotone in [0,1].
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

// quantile_edges: k + 1 ascending edges spanning the exact sample min and max.
#[test]
fn quantile_edges_span_sample_and_ascend() {
    let xs = lcg_sample(5000, 55);
    let mut sorted = xs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let d = digest_of(&xs);
    for k in [1usize, 10] {
        let edges = quantile_edges(&d, k).unwrap();
        assert_eq!(edges.len(), k + 1);
        assert_eq!(edges[0], sorted[0]); // exact sample min
        assert_eq!(edges[k], sorted[sorted.len() - 1]); // exact sample max
        assert!(edges.windows(2).all(|w| w[1] >= w[0]));
    }
}
