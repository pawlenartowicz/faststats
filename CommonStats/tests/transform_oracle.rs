//! Oracle-based integration tests for src/transform/*.
//! Run: cargo test (base transforms); cargo test --features dist (PIT family).

mod common;
use common::{Tol, assert_close};

use commonstats::transform::{Ties, rank};

fn fixture_path(name: &str) -> String {
    format!("{}/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"))
}

fn load_str(name: &str) -> String {
    let path = fixture_path(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {path}: {e} — run scripts/gen_oracle.py"))
}

// ── rank ─────────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct RankRecord {
    // NaN stored as null in fixture so serde_json can parse it; None → f64::NAN on use.
    input: Vec<Option<f64>>,
    method: String,
    expected: Vec<f64>,
}

fn opt_vec_to_f64(v: &[Option<f64>]) -> Vec<f64> {
    v.iter().map(|&x| x.unwrap_or(f64::NAN)).collect()
}

fn method_to_ties(m: &str) -> Ties {
    match m {
        "average" => Ties::Average,
        "min" => Ties::Min,
        "max" => Ties::Max,
        "dense" => Ties::Dense,
        "ordinal" => Ties::Ordinal,
        other => panic!("unknown ties method: {other}"),
    }
}

#[test]
fn rank_oracle() {
    let records: Vec<RankRecord> =
        serde_json::from_str(&load_str("rankdata")).expect("rankdata fixture parse");
    for r in &records {
        let ties = method_to_ties(&r.method);
        let input_f64 = opt_vec_to_f64(&r.input);
        let got = rank(&input_f64, ties)
            .unwrap_or_else(|e| panic!("rank({:?}, {:?}): {:?}", r.input, r.method, e));
        assert_eq!(
            got.len(),
            r.expected.len(),
            "rank output length mismatch for input {:?} method {}",
            r.input,
            r.method
        );
        for (i, (&g, &w)) in got.iter().zip(r.expected.iter()).enumerate() {
            assert_close(
                &format!("rank[{}] input={:?} method={}", i, r.input, r.method),
                g,
                w,
                Tol {
                    rel: 1e-14,
                    abs: 1e-14,
                },
            );
        }
    }
}

// ── box_cox ──────────────────────────────────────────────────────────────────

use commonstats::transform::box_cox;

#[derive(serde::Deserialize)]
struct PowerRecord {
    lambda: f64,
    x: f64,
    expected: Option<f64>,
}

#[test]
fn boxcox_oracle() {
    let records: Vec<PowerRecord> =
        serde_json::from_str(&load_str("boxcox")).expect("boxcox fixture parse");
    for r in &records {
        let result = box_cox(&[r.x], r.lambda);
        match r.expected {
            None => {
                // x<=0 rows: expect DomainError
                assert!(
                    result.is_err(),
                    "box_cox([{}], lambda={}) should be Err, got {:?}",
                    r.x,
                    r.lambda,
                    result
                );
            }
            Some(w) => {
                let got = result
                    .unwrap_or_else(|e| panic!("box_cox([{}], lambda={}): {:?}", r.x, r.lambda, e));
                assert_close(
                    &format!("box_cox x={} lambda={}", r.x, r.lambda),
                    got[0],
                    w,
                    Tol {
                        rel: 1e-12,
                        abs: 1e-14,
                    },
                );
            }
        }
    }
}

#[test]
fn boxcox_domain_error_on_nonpositive() {
    use commonstats::error::StatError;
    // First non-positive x triggers DomainError — no partial output
    let r = box_cox(&[1.0, 0.0, 2.0], 1.0);
    assert!(matches!(r, Err(StatError::DomainError(_))));
}

// ── yeo_johnson ──────────────────────────────────────────────────────────────

use commonstats::transform::yeo_johnson;

#[derive(serde::Deserialize)]
struct YjRecord {
    lambda: f64,
    x: f64,
    expected: f64,
}

#[test]
fn yeojohnson_oracle() {
    let records: Vec<YjRecord> =
        serde_json::from_str(&load_str("yeojohnson")).expect("yeojohnson fixture parse");
    for r in &records {
        let got = yeo_johnson(&[r.x], r.lambda)
            .unwrap_or_else(|e| panic!("yeo_johnson([{}], lambda={}): {:?}", r.x, r.lambda, e));
        assert_close(
            &format!("yeo_johnson x={} lambda={}", r.x, r.lambda),
            got[0],
            r.expected,
            Tol {
                rel: 1e-12,
                abs: 1e-14,
            },
        );
    }
}

// ── normal_scores ─────────────────────────────────────────────────────────────

use commonstats::transform::normal_scores;

#[derive(serde::Deserialize)]
struct NsRecord {
    // NaN stored as null in fixture; None → f64::NAN on use.
    input: Vec<Option<f64>>,
    expected: Vec<f64>,
}

#[test]
fn normal_scores_oracle() {
    let records: Vec<NsRecord> =
        serde_json::from_str(&load_str("normal_scores")).expect("normal_scores fixture parse");
    for r in &records {
        let input_f64 = opt_vec_to_f64(&r.input);
        let got = normal_scores(&input_f64)
            .unwrap_or_else(|e| panic!("normal_scores({:?}): {:?}", r.input, e));
        assert_eq!(
            got.len(),
            r.expected.len(),
            "length mismatch for {:?}",
            r.input
        );
        for (i, (&g, &w)) in got.iter().zip(r.expected.iter()).enumerate() {
            assert_close(
                &format!("normal_scores[{}] input={:?}", i, input_f64),
                g,
                w,
                Tol {
                    rel: 1e-12,
                    abs: 1e-14,
                },
            );
        }
    }
}

#[test]
fn normal_scores_single_value() {
    // n=1: rank=1, p=(1-3/8)/(1+1/4)=0.5 → Φ⁻¹(0.5)=0
    let got = normal_scores(&[42.0]).unwrap();
    assert_close(
        "normal_scores[0]",
        got[0],
        0.0,
        Tol {
            rel: 1e-12,
            abs: 1e-14,
        },
    );
}

// ── empty / all-NaN input ────────────────────────────────────────────────────

/// Every element-wise transform rejects an empty slice and an all-NaN slice
/// with the crate's two input errors.
#[test]
fn elementwise_transforms_reject_empty_and_all_nan() {
    use commonstats::error::StatError;
    type Transform = fn(&[f64]) -> Result<Vec<f64>, StatError>;
    let transforms: [(&str, Transform); 4] = [
        ("rank", |xs| rank(xs, Ties::Average)),
        ("box_cox", |xs| box_cox(xs, 1.0)),
        ("yeo_johnson", |xs| yeo_johnson(xs, 1.0)),
        ("normal_scores", normal_scores),
    ];
    for (name, f) in transforms {
        assert_eq!(f(&[]), Err(StatError::EmptyInput), "{name}");
        assert_eq!(f(&[f64::NAN, f64::NAN]), Err(StatError::AllNaN), "{name}");
    }
}

// ── quantile_normalize ────────────────────────────────────────────────────────

use commonstats::transform::{Infinities, quantile_normalize, quantile_normalize_with};

#[derive(serde::Deserialize)]
struct QnFixture {
    matrix: Vec<Vec<f64>>,
    #[allow(dead_code)]
    r#ref: Vec<f64>,
    expected: Vec<Vec<f64>>,
}

#[test]
fn quantile_normalize_bolstad_example() {
    let fixture: QnFixture =
        serde_json::from_str(&load_str("quantile_normalize")).expect("qn fixture parse");
    let cols: Vec<&[f64]> = fixture.matrix.iter().map(|c| c.as_slice()).collect();
    let got = quantile_normalize(&cols).unwrap();
    assert_eq!(got.len(), fixture.expected.len(), "column count mismatch");
    for (j, (gc, ec)) in got.iter().zip(fixture.expected.iter()).enumerate() {
        assert_eq!(gc.len(), ec.len(), "row count mismatch col {j}");
        for (i, (&g, &w)) in gc.iter().zip(ec.iter()).enumerate() {
            assert_close(
                &format!("qn[col={j}][row={i}]"),
                g,
                w,
                Tol {
                    rel: 1e-12,
                    abs: 1e-14,
                },
            );
        }
    }
}

#[test]
fn quantile_normalize_missing_value_keeps_column_order() {
    // B's NaN must not let A's tail reference entry average fewer columns: A
    // was [5.5, 11, 16.5, 4] (its largest value mapped lowest). Want:
    // preprocessCore `normalize.quantiles(cbind(1:4, c(10, 20, 30, NA)))`.
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [10.0, 20.0, 30.0, f64::NAN];
    let got = quantile_normalize(&[&a, &b]).unwrap();
    let want_a = [5.5, 9.333_333_333_333_334, 13.166_666_666_666_666, 17.0];
    let want_b = [5.5, 11.25, 17.0];
    let tol = || Tol {
        rel: 1e-12,
        abs: 1e-14,
    };
    for (i, (&g, &w)) in got[0].iter().zip(want_a.iter()).enumerate() {
        assert_close(&format!("a[{i}]"), g, w, tol());
    }
    for (i, (&g, &w)) in got[1].iter().zip(want_b.iter()).enumerate() {
        assert_close(&format!("b[{i}]"), g, w, tol());
    }
    assert!(got[1][3].is_nan());
}

#[derive(serde::Deserialize)]
struct QnMissingCase {
    label: String,
    // Columns; `null` is a missing (NaN) entry, in `matrix` and in `expected`.
    matrix: Vec<Vec<Option<f64>>>,
    expected: Vec<Vec<Option<f64>>>,
}

#[test]
fn quantile_normalize_missing_values_oracle() {
    let cases: Vec<QnMissingCase> = serde_json::from_str(&load_str("quantile_normalize_missing"))
        .expect("qn missing fixture parse");
    for case in &cases {
        let cols: Vec<Vec<f64>> = case
            .matrix
            .iter()
            .map(|c| c.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
            .collect();
        let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
        let got = quantile_normalize(&refs).unwrap();
        for (j, (gc, ec)) in got.iter().zip(case.expected.iter()).enumerate() {
            for (i, (&g, e)) in gc.iter().zip(ec.iter()).enumerate() {
                let label = format!("{}[col={j}][row={i}]", case.label);
                match e {
                    Some(w) => assert_close(
                        &label,
                        g,
                        *w,
                        Tol {
                            rel: 1e-12,
                            abs: 1e-14,
                        },
                    ),
                    None => assert!(g.is_nan(), "{label}: got {g}, want NaN"),
                }
            }
        }
    }
}

fn assert_cols_close(got: &[Vec<f64>], want: &[&[f64]]) {
    for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        for (i, (&g, &w)) in g.iter().zip(w.iter()).enumerate() {
            if w.is_finite() {
                let tol = Tol {
                    rel: 1e-15,
                    abs: 0.0,
                };
                assert_close(&format!("col {j} row {i}"), g, w, tol);
            } else {
                assert_eq!(g, w, "col {j} row {i}");
            }
        }
    }
}

#[test]
fn quantile_normalize_as_extremes_hand_case() {
    // The case in `quantile_normalize_with`'s doc. Sorted A = [−∞, 2, 3], B =
    // [0, 5, 7]; A's −∞ adds nothing at position 0, so the target is
    // [0, (2 + 5)/2, (3 + 7)/2] = [0, 3.5, 5].
    let a = [2.0, f64::NEG_INFINITY, 3.0];
    let b = [0.0, 5.0, 7.0];
    let got = quantile_normalize_with(&[&a, &b], Infinities::AsExtremes).unwrap();
    assert_cols_close(&got, &[&[3.5, f64::NEG_INFINITY, 5.0], &[0.0, 3.5, 5.0]]);
}

#[test]
fn quantile_normalize_as_extremes_pools_a_decreasing_target() {
    // B's +∞ adds nothing at position 2, so the plain means [50.5, 101, 3]
    // decrease; pooling positions 1 and 2 gives (2 + 200 + 3)/3 = 205/3 for
    // both, and A keeps its order.
    let a = [1.0, 2.0, 3.0];
    let b = [100.0, 200.0, f64::INFINITY];
    let got = quantile_normalize_with(&[&a, &b], Infinities::AsExtremes).unwrap();
    let p = 205.0 / 3.0;
    assert_cols_close(&got, &[&[50.5, p, p], &[50.5, p, f64::INFINITY]]);
}

#[test]
fn quantile_normalize_as_extremes_without_finite_target_gives_nan() {
    // One column [−∞, 5, +∞] with a NaN: stretched onto 4 points every grid
    // value touches an infinity, so no position has a finite target.
    let a = [f64::NEG_INFINITY, 5.0, f64::INFINITY, f64::NAN];
    let got = quantile_normalize_with(&[&a], Infinities::AsExtremes).unwrap();
    assert_eq!(got[0][0], f64::NEG_INFINITY);
    assert!(got[0][1].is_nan());
    assert_eq!(got[0][2], f64::INFINITY);
    assert!(got[0][3].is_nan());
}

/// Deterministic matrices with ties, NaN and (when `infinities`) ±∞.
fn qn_test_matrices(infinities: bool) -> Vec<Vec<Vec<f64>>> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as u32
    };
    let mut out = Vec::new();
    for _ in 0..300 {
        let n = 1 + (next() % 9) as usize;
        let ncols = 1 + (next() % 4) as usize;
        let m = (0..ncols)
            .map(|_| {
                (0..n)
                    .map(|_| match next() % 10 {
                        0 => f64::NAN,
                        1 if infinities => f64::INFINITY,
                        2 if infinities => f64::NEG_INFINITY,
                        // Few distinct values, so ties are common.
                        _ => f64::from(next() % 6) * 1.5 - 2.0,
                    })
                    .collect()
            })
            .collect();
        out.push(m);
    }
    out
}

#[test]
fn quantile_normalize_order_holds_at_ulp_spacing() {
    // Values a few ulps apart, one column NaN-reduced, so the second column's
    // ranks are stretched onto fractional target positions over a nearly flat
    // target: `a·(1 − f) + b·f` is not monotone in `f` at that spacing and
    // mapped 3.700000000000001 above 3.7000000000000015.
    let a = [
        3.700000000000002,
        3.700000000000002,
        3.700000000000002,
        3.700000000000002,
        3.7000000000000015,
    ];
    let b = [
        3.7000000000000015,
        3.700000000000002,
        3.7,
        3.700000000000001,
        f64::NAN,
    ];
    for inf in [Infinities::AsMissing, Infinities::AsExtremes] {
        let out = quantile_normalize_with(&[&a, &b], inf).unwrap();
        for (col, o) in [(&a[..], &out[0]), (&b[..], &out[1])] {
            for (i, (&x, &ox)) in col.iter().zip(o.iter()).enumerate() {
                for (j, (&y, &oy)) in col.iter().zip(o.iter()).enumerate() {
                    if x < y {
                        assert!(ox <= oy, "{inf:?} rows {i}, {j}: {x} < {y} but {ox} > {oy}");
                    }
                }
            }
        }
    }
}

#[test]
fn quantile_normalize_preserves_column_order() {
    for inf in [Infinities::AsMissing, Infinities::AsExtremes] {
        for m in qn_test_matrices(true) {
            let refs: Vec<&[f64]> = m.iter().map(|c| c.as_slice()).collect();
            let got = quantile_normalize_with(&refs, inf).unwrap();
            // Finite values map to NaN only where no grid position has a
            // finite target (documented), and then all of them do.
            let finite_out = || {
                m.iter()
                    .flatten()
                    .zip(got.iter().flatten())
                    .filter(|(x, _)| x.is_finite())
                    .map(|(_, &o)| o)
            };
            let no_target = finite_out().any(f64::is_nan);
            assert!(!no_target || finite_out().all(f64::is_nan), "{inf:?} {m:?}");
            assert!(!no_target || inf == Infinities::AsExtremes, "{m:?}");
            for (col, out) in m.iter().zip(got.iter()) {
                for (i, (&x, &ox)) in col.iter().zip(out.iter()).enumerate() {
                    // NaN and ±∞ come back unchanged under both conventions.
                    if !x.is_finite() {
                        assert_eq!(ox.to_bits(), x.to_bits(), "{inf:?} {m:?}: row {i}");
                        continue;
                    }
                    if no_target {
                        continue;
                    }
                    for (&y, &oy) in col.iter().zip(out.iter()) {
                        if y.is_nan() || (inf == Infinities::AsMissing && y.is_infinite()) {
                            continue;
                        }
                        if x < y {
                            assert!(ox <= oy, "{inf:?} {m:?}: {x} < {y} but {ox} > {oy}");
                        } else if x == y {
                            assert_eq!(ox, oy, "{inf:?} {m:?}: tie {x}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn quantile_normalize_as_extremes_without_infinities_is_as_missing() {
    for m in qn_test_matrices(false) {
        let refs: Vec<&[f64]> = m.iter().map(|c| c.as_slice()).collect();
        let a = quantile_normalize(&refs).unwrap();
        let b = quantile_normalize_with(&refs, Infinities::AsExtremes).unwrap();
        let bits = |v: &[Vec<f64>]| -> Vec<Vec<u64>> {
            v.iter()
                .map(|c| c.iter().map(|x| x.to_bits()).collect())
                .collect()
        };
        assert_eq!(bits(&a), bits(&b), "{m:?}");
    }
}

#[test]
fn quantile_normalize_infinities_are_kept_out_like_nan() {
    // ±∞ stays in place and, like NaN, is left out of the ranks and the
    // reference row: the finite outputs equal those with NaN in its place.
    let c0 = [2.0, f64::INFINITY, 3.0, 5.0];
    let c1 = [1.0, 4.0, f64::NEG_INFINITY, 4.0];
    let c2 = [3.0, 6.0, 4.0, f64::NAN];
    let got = quantile_normalize(&[&c0, &c1, &c2]).unwrap();
    let n0 = [2.0, f64::NAN, 3.0, 5.0];
    let n1 = [1.0, 4.0, f64::NAN, 4.0];
    let want = quantile_normalize(&[&n0, &n1, &c2]).unwrap();
    assert_eq!(got[0][1], f64::INFINITY);
    assert_eq!(got[1][2], f64::NEG_INFINITY);
    assert!(got[2][3].is_nan());
    for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        for (i, (&g, &w)) in g.iter().zip(w.iter()).enumerate() {
            assert!(
                !g.is_finite() || g == w,
                "col {j} row {i}: {g} vs NaN-substituted {w}"
            );
            assert!(g.is_finite() || !w.is_finite(), "col {j} row {i} not kept");
        }
    }
}

#[test]
fn quantile_normalize_empty_is_err() {
    use commonstats::error::StatError;
    assert_eq!(quantile_normalize(&[]), Err(StatError::EmptyInput));
}

#[test]
fn quantile_normalize_unequal_columns_is_err() {
    use commonstats::error::StatError;
    let col0 = [1.0f64, 2.0];
    let col1 = [1.0f64, 2.0, 3.0];
    let r = quantile_normalize(&[&col0, &col1]);
    assert!(matches!(r, Err(StatError::MismatchedLengths { .. })));
}

#[test]
fn quantile_normalize_nan_propagates() {
    // NaN in input propagates in output at the same position (shape preserved)
    let col0 = [1.0f64, f64::NAN, 3.0];
    let col1 = [2.0f64, 4.0, 6.0];
    let got = quantile_normalize(&[&col0, &col1]).unwrap();
    assert!(got[0][1].is_nan(), "NaN should propagate at col0[1]");
    // col0's two values stretched onto 3 points are [1, 2, 3], so the target is
    // [(1+2)/2, (2+4)/2, (3+6)/2] = [1.5, 3, 4.5]; col0 ranks [1, 2] map to
    // positions 0 and 2 → 1.5, 4.5 (preprocessCore `normalize.quantiles` agrees).
    assert_close(
        "qn_nan/col0[0]",
        got[0][0],
        1.5,
        Tol {
            rel: 1e-12,
            abs: 1e-14,
        },
    );
    assert_close(
        "qn_nan/col0[2]",
        got[0][2],
        4.5,
        Tol {
            rel: 1e-12,
            abs: 1e-14,
        },
    );
}

// ── PIT family (requires --features dist) ────────────────────────────────────

#[cfg(feature = "dist")]
mod pit_tests {
    use commonstats::dist::continuous::Normal;
    use commonstats::error::StatError;
    use commonstats::transform::{inv_pit, pit, quantile_map};

    fn normal() -> Normal {
        Normal::new(0.0, 1.0).unwrap()
    }

    #[test]
    fn pit_normal_cdf() {
        let n = normal();
        // pit is just cdf — verify against known values
        let p = pit(0.0, &n);
        assert!(
            (p - 0.5).abs() < 1e-12,
            "pit(0, N(0,1)) should be 0.5, got {p}"
        );
        let p2 = pit(1.6448536269514729, &n); // Φ(1.645) ≈ 0.95
        assert!(
            (p2 - 0.95).abs() < 1e-12,
            "pit(1.645, N(0,1)) ≈ 0.95, got {p2}"
        );
    }

    #[test]
    fn inv_pit_normal_quantile() {
        let n = normal();
        let x = inv_pit(0.5, &n).unwrap();
        assert!(
            (x - 0.0).abs() < 1e-10,
            "inv_pit(0.5, N(0,1)) should be 0, got {x}"
        );
        assert!(matches!(
            inv_pit(-0.1, &n),
            Err(StatError::ProbabilityOutOfRange(_))
        ));
        assert!(matches!(
            inv_pit(1.1, &n),
            Err(StatError::ProbabilityOutOfRange(_))
        ));
    }

    #[test]
    fn quantile_map_normal_to_normal() {
        // quantile_map(x, N(0,1), N(0,1)) = x (identity)
        let n = normal();
        let x = 1.5f64;
        let mapped = quantile_map(x, &n, &n).unwrap();
        assert!(
            (mapped - x).abs() < 1e-8,
            "identity map got {mapped}, want {x}"
        );
    }

    #[test]
    fn quantile_map_shifts_distribution() {
        // quantile_map(0, N(0,1), N(1,1)) should give 1.0
        // because Φ₀₁(0) = 0.5 → Φ₁₁⁻¹(0.5) = 1.0
        let from = Normal::new(0.0, 1.0).unwrap();
        let to = Normal::new(1.0, 1.0).unwrap();
        let mapped = quantile_map(0.0, &from, &to).unwrap();
        assert!(
            (mapped - 1.0).abs() < 1e-8,
            "shifted map got {mapped}, want 1.0"
        );
    }
}
