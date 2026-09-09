use commonstats::htest::{
    CorMethod, EffectSize, VarAssumption, anova_one_way, chi2_gof, chi2_independence, cor_test,
    f_test_var, t_test_one, t_test_paired, t_test_two,
};
use serde::Deserialize;

mod common;
use common::{Tol, assert_close};

fn read<T: serde::de::DeserializeOwned>(name: &str) -> T {
    let p = format!("{}/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

// Fixtures now carry mpmath truth (see scripts/gen_oracle.py gen_htest). Spec §3
// Class-C ladder: statistic/df 1e-12, p-value 1e-10 — relative with an abs floor.
const STAT: Tol = Tol {
    rel: 1e-12,
    abs: 1e-14,
}; // t / F / χ² / r
const DF: Tol = Tol {
    rel: 1e-12,
    abs: 1e-12,
};
const CI: Tol = Tol {
    rel: 1e-12,
    abs: 1e-9,
};
const PVAL: Tol = Tol {
    rel: 1e-10,
    abs: 1e-14,
};

#[derive(Deserialize)]
struct TtestFixture {
    a: Vec<f64>,
    b: Vec<f64>,
    rows: Vec<Row>,
}
#[derive(Deserialize)]
struct Row {
    args: Vec<String>,
    expected: Vec<f64>,
    // R t.test(var.equal=FALSE)$conf.int (matches mpmath); only on the welch row.
    #[serde(default)]
    welch_ci_correct: Option<Vec<f64>>,
}

#[test]
fn ttests_match_scipy() {
    let f: TtestFixture = read("ttests");
    for row in &f.rows {
        let (got_t, got_p, got_df) = match row.args[0].as_str() {
            "one" => {
                let r = t_test_one(&f.a, 5.0).unwrap();
                (r.statistic, r.p_value, r.df)
            }
            "student" => {
                let r = t_test_two(&f.a, &f.b, VarAssumption::Equal).unwrap();
                (r.statistic, r.p_value, r.df)
            }
            "welch" => {
                let r = t_test_two(&f.a, &f.b, VarAssumption::Welch).unwrap();
                (r.statistic, r.p_value, r.df)
            }
            "paired" => {
                let r = t_test_paired(&f.a[..8], &f.b[..8]).unwrap();
                (r.statistic, r.p_value, r.df)
            }
            other => panic!("unknown row {other}"),
        };
        assert_close(&format!("{}: t", row.args[0]), got_t, row.expected[0], STAT);
        assert_close(&format!("{}: p", row.args[0]), got_p, row.expected[1], PVAL);
        if row.expected.len() > 2 {
            assert_close(&format!("{}: df", row.args[0]), got_df, row.expected[2], DF);
        }
    }
    // Welch CI must invert to the Welch p-value (the pooled-CI inconsistency was the
    // prior latent bug). welch_ci_correct = R t.test(var.equal=FALSE)$conf.int.
    let welch_row = f.rows.iter().find(|r| r.args[0] == "welch").unwrap();
    let want_ci = welch_row
        .welch_ci_correct
        .as_ref()
        .expect("welch_ci_correct fixture");
    let ci = t_test_two(&f.a, &f.b, VarAssumption::Welch)
        .unwrap()
        .ci
        .expect("welch ci");
    assert_close("welch ci lower", ci.lower, want_ci[0], CI);
    assert_close("welch ci upper", ci.upper, want_ci[1], CI);

    // Cohen's d (pooled SD) paired with Student's t — scipy/numpy pooled value.
    let student = t_test_two(&f.a, &f.b, VarAssumption::Equal).unwrap();
    match student.effect_size {
        Some(EffectSize::CohenD(d)) => {
            assert!((d - (-1.9663081137313068)).abs() < 1e-9, "cohen_d {d}")
        }
        _ => panic!("expected CohenD effect size"),
    }
}

#[derive(Deserialize)]
struct AnovaFix {
    groups: Vec<Vec<f64>>,
    #[serde(rename = "F")]
    f: f64,
    p: f64,
}
#[derive(Deserialize)]
struct Chi2Fix {
    gof_obs: Vec<f64>,
    gof_exp: Vec<f64>,
    gof_chi2: f64,
    gof_p: f64,
    table: Vec<Vec<f64>>,
    ind_chi2: f64,
    ind_p: f64,
    ind_df: f64,
}

#[test]
fn anova_matches_scipy() {
    let fx: AnovaFix = read("anova");
    let groups: Vec<&[f64]> = fx.groups.iter().map(|g| g.as_slice()).collect();
    let r = anova_one_way(&groups).unwrap();
    assert_close("F", r.statistic, fx.f, STAT);
    assert_close("p", r.p_value, fx.p, PVAL);
    match r.effect_size {
        Some(EffectSize::EtaSquared(e)) => {
            assert!((e - 0.7402447471715539).abs() < 1e-9, "eta2 {e}")
        }
        _ => panic!("expected EtaSquared effect size"),
    }
}
#[test]
fn chi2_matches_scipy() {
    let fx: Chi2Fix = read("chi2");
    let g = chi2_gof(&fx.gof_obs, &fx.gof_exp).unwrap();
    assert_close("gof χ²", g.statistic, fx.gof_chi2, STAT);
    assert_close("gof p", g.p_value, fx.gof_p, PVAL);
    let table: Vec<&[f64]> = fx.table.iter().map(|r| r.as_slice()).collect();
    let i = chi2_independence(&table).unwrap();
    assert_close("ind χ²", i.statistic, fx.ind_chi2, STAT);
    assert_close("ind p", i.p_value, fx.ind_p, PVAL);
    assert_close("ind df", i.df, fx.ind_df, DF);
    match i.effect_size {
        Some(EffectSize::CramersV(v)) => {
            assert!((v - 0.05433137570422292).abs() < 1e-9, "cramers_v {v}")
        }
        _ => panic!("expected CramersV effect size"),
    }
}

#[derive(Deserialize)]
struct CorFix {
    a: Vec<f64>,
    b: Vec<f64>,
    r: f64,
    t: f64,
    p: f64,
}

#[test]
fn pearson_cor_matches_scipy() {
    let fx: CorFix = read("cor");
    let res = cor_test(&fx.a, &fx.b, CorMethod::Pearson).unwrap();
    let r = match res.effect_size {
        Some(EffectSize::R(r)) => r,
        _ => panic!("no r"),
    };
    assert_close("r", r, fx.r, STAT);
    assert_close("p", res.p_value, fx.p, PVAL);
    assert_close("t", res.statistic, fx.t, STAT);
}

#[derive(Deserialize)]
struct FtestFix {
    a: Vec<f64>,
    b: Vec<f64>,
    #[serde(rename = "F")]
    f: f64,
    df1: f64,
    p: f64,
}

#[test]
fn f_test_var_matches_scipy() {
    let fx: FtestFix = read("ftest");
    let r = f_test_var(&fx.a, &fx.b).unwrap();
    assert_close("F", r.statistic, fx.f, STAT);
    assert_close("df1", r.df, fx.df1, DF);
    assert_close("p", r.p_value, fx.p, PVAL);
}
