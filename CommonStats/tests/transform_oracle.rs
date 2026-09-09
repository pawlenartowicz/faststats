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

#[test]
fn rank_empty_is_err() {
    use commonstats::error::StatError;
    assert_eq!(rank(&[], Ties::Average), Err(StatError::EmptyInput));
}

#[test]
fn rank_all_nan_is_err() {
    use commonstats::error::StatError;
    assert_eq!(
        rank(&[f64::NAN, f64::NAN], Ties::Average),
        Err(StatError::AllNaN)
    );
}

#[test]
fn rank_nan_omit_length() {
    // NaN rows are filtered; output length = finite count
    let v = [5.0, f64::NAN, 3.0, f64::NAN, 1.0];
    let got = rank(&v, Ties::Average).unwrap();
    assert_eq!(got.len(), 3, "expected 3 finite values");
    // ranks of [5,3,1] with Average: 3,2,1
    assert_close(
        "rank[0]",
        got[0],
        3.0,
        Tol {
            rel: 1e-14,
            abs: 1e-14,
        },
    );
    assert_close(
        "rank[1]",
        got[1],
        2.0,
        Tol {
            rel: 1e-14,
            abs: 1e-14,
        },
    );
    assert_close(
        "rank[2]",
        got[2],
        1.0,
        Tol {
            rel: 1e-14,
            abs: 1e-14,
        },
    );
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
fn boxcox_empty_is_err() {
    use commonstats::error::StatError;
    assert_eq!(box_cox(&[], 1.0), Err(StatError::EmptyInput));
}

#[test]
fn boxcox_all_nan_is_err() {
    use commonstats::error::StatError;
    assert_eq!(box_cox(&[f64::NAN], 1.0), Err(StatError::AllNaN));
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
    expected: Option<f64>,
}

#[test]
fn yeojohnson_oracle() {
    let records: Vec<YjRecord> =
        serde_json::from_str(&load_str("yeojohnson")).expect("yeojohnson fixture parse");
    for r in &records {
        if let Some(w) = r.expected {
            let got = yeo_johnson(&[r.x], r.lambda)
                .unwrap_or_else(|e| panic!("yeo_johnson([{}], lambda={}): {:?}", r.x, r.lambda, e));
            assert_close(
                &format!("yeo_johnson x={} lambda={}", r.x, r.lambda),
                got[0],
                w,
                Tol {
                    rel: 1e-12,
                    abs: 1e-14,
                },
            );
        } else {
            // YJ is defined for all reals, so None rows represent overflow cases; assert Err.
            assert!(
                yeo_johnson(&[r.x], r.lambda).is_err(),
                "yeo_johnson([{}], lambda={}) should be Err on a None-expected row",
                r.x,
                r.lambda
            );
        }
    }
}

#[test]
fn yeojohnson_empty_is_err() {
    use commonstats::error::StatError;
    assert_eq!(yeo_johnson(&[], 1.0), Err(StatError::EmptyInput));
}

#[test]
fn yeojohnson_all_nan_is_err() {
    use commonstats::error::StatError;
    assert_eq!(yeo_johnson(&[f64::NAN], 1.0), Err(StatError::AllNaN));
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
fn normal_scores_empty_is_err() {
    use commonstats::error::StatError;
    assert_eq!(normal_scores(&[]), Err(StatError::EmptyInput));
}

#[test]
fn normal_scores_all_nan_is_err() {
    use commonstats::error::StatError;
    assert_eq!(normal_scores(&[f64::NAN, f64::NAN]), Err(StatError::AllNaN));
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

// ── quantile_normalize ────────────────────────────────────────────────────────

use commonstats::transform::quantile_normalize;

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
    // Non-NaN positions: reference = [(1+2)/2, (3+4)/2, 6/1] = [1.5, 3.5, 6.0];
    // col0 ranks [1,2] → col0[0]=1.5, col0[2]=3.5.
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
        3.5,
        Tol {
            rel: 1e-12,
            abs: 1e-14,
        },
    );
}

#[test]
fn quantile_normalize_both_cols_map_to_reference() {
    // Distinct input from the Bolstad fixture: no NaN, clean arithmetic.
    // reference = [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5];
    // both columns share ranks [1,2,3] → both normalize to [2.5, 3.5, 4.5].
    let col0 = [1.0f64, 2.0, 3.0];
    let col1 = [4.0f64, 5.0, 6.0];
    let got = quantile_normalize(&[&col0, &col1]).unwrap();
    let want = [2.5, 3.5, 4.5];
    for (j, col) in got.iter().enumerate() {
        for i in 0..3 {
            assert_close(
                &format!("qn_shape/col{j}[{i}]"),
                col[i],
                want[i],
                Tol {
                    rel: 1e-12,
                    abs: 1e-14,
                },
            );
        }
    }
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
