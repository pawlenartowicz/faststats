use commonstats::htest::{
    anova_one_way, chi2_gof, chi2_independence, cor_test, f_test_var, t_test_one, t_test_paired, t_test_two,
    CorMethod, EffectSize, VarAssumption,
};
use serde::Deserialize;

fn read<T: serde::de::DeserializeOwned>(name: &str) -> T {
    let p = format!("{}/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

// Fixtures now carry mpmath truth (see scripts/gen_oracle.py gen_htest). Spec §3
// Class-C ladder: statistic/df 1e-12, p-value 1e-10 — relative with an abs floor.
fn close(got: f64, want: f64, rel: f64, abs: f64) -> bool {
    (got - want).abs() <= abs.max(rel * want.abs())
}
const STAT: f64 = 1e-12; // t / F / χ² / r / df
const PVAL: f64 = 1e-10;

#[derive(Deserialize)]
struct TtestFixture { a: Vec<f64>, b: Vec<f64>, rows: Vec<Row> }
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
            "one"     => { let r = t_test_one(&f.a, 5.0).unwrap(); (r.statistic, r.p_value, r.df) }
            "student" => { let r = t_test_two(&f.a, &f.b, VarAssumption::Equal).unwrap(); (r.statistic, r.p_value, r.df) }
            "welch"   => { let r = t_test_two(&f.a, &f.b, VarAssumption::Welch).unwrap(); (r.statistic, r.p_value, r.df) }
            "paired"  => { let r = t_test_paired(&f.a[..8], &f.b[..8]).unwrap(); (r.statistic, r.p_value, r.df) }
            other => panic!("unknown row {other}"),
        };
        assert!(close(got_t, row.expected[0], STAT, 1e-14), "{}: t {got_t} vs {}", row.args[0], row.expected[0]);
        assert!(close(got_p, row.expected[1], PVAL, 1e-14), "{}: p {got_p} vs {}", row.args[0], row.expected[1]);
        if row.expected.len() > 2 {
            assert!(close(got_df, row.expected[2], STAT, 1e-12), "{}: df {got_df} vs {}", row.args[0], row.expected[2]);
        }
    }
    // Welch CI must invert to the Welch p-value (the pooled-CI inconsistency was the
    // prior latent bug). welch_ci_correct = R t.test(var.equal=FALSE)$conf.int.
    let welch_row = f.rows.iter().find(|r| r.args[0] == "welch").unwrap();
    let want_ci = welch_row.welch_ci_correct.as_ref().expect("welch_ci_correct fixture");
    let ci = t_test_two(&f.a, &f.b, VarAssumption::Welch).unwrap().ci.expect("welch ci");
    assert!(close(ci.lower, want_ci[0], STAT, 1e-9), "welch ci lower {} vs {}", ci.lower, want_ci[0]);
    assert!(close(ci.upper, want_ci[1], STAT, 1e-9), "welch ci upper {} vs {}", ci.upper, want_ci[1]);

    // Cohen's d (pooled SD) paired with Student's t — scipy/numpy pooled value.
    let student = t_test_two(&f.a, &f.b, VarAssumption::Equal).unwrap();
    match student.effect_size {
        Some(EffectSize::CohenD(d)) => assert!((d - (-1.9663081137313068)).abs() < 1e-9, "cohen_d {d}"),
        _ => panic!("expected CohenD effect size"),
    }
}

#[derive(Deserialize)]
struct AnovaFix { groups: Vec<Vec<f64>>, #[serde(rename="F")] f: f64, p: f64 }
#[derive(Deserialize)]
struct Chi2Fix { gof_obs: Vec<f64>, gof_exp: Vec<f64>, gof_chi2: f64, gof_p: f64,
                 table: Vec<Vec<f64>>, ind_chi2: f64, ind_p: f64, ind_df: f64 }

#[test]
fn anova_matches_scipy() {
    let fx: AnovaFix = read("anova");
    let groups: Vec<&[f64]> = fx.groups.iter().map(|g| g.as_slice()).collect();
    let r = anova_one_way(&groups).unwrap();
    assert!(close(r.statistic, fx.f, STAT, 1e-14), "F {} vs {}", r.statistic, fx.f);
    assert!(close(r.p_value, fx.p, PVAL, 1e-14), "p {} vs {}", r.p_value, fx.p);
    match r.effect_size {
        Some(EffectSize::EtaSquared(e)) => assert!((e - 0.7402447471715539).abs() < 1e-9, "eta2 {e}"),
        _ => panic!("expected EtaSquared effect size"),
    }
}
#[test]
fn chi2_matches_scipy() {
    let fx: Chi2Fix = read("chi2");
    let g = chi2_gof(&fx.gof_obs, &fx.gof_exp).unwrap();
    assert!(close(g.statistic, fx.gof_chi2, STAT, 1e-14), "gof χ² {} vs {}", g.statistic, fx.gof_chi2);
    assert!(close(g.p_value, fx.gof_p, PVAL, 1e-14), "gof p {} vs {}", g.p_value, fx.gof_p);
    let table: Vec<&[f64]> = fx.table.iter().map(|r| r.as_slice()).collect();
    let i = chi2_independence(&table).unwrap();
    assert!(close(i.statistic, fx.ind_chi2, STAT, 1e-14), "ind χ² {} vs {}", i.statistic, fx.ind_chi2);
    assert!(close(i.p_value, fx.ind_p, PVAL, 1e-14), "ind p {} vs {}", i.p_value, fx.ind_p);
    assert!(close(i.df, fx.ind_df, STAT, 1e-12), "ind df {} vs {}", i.df, fx.ind_df);
    match i.effect_size {
        Some(EffectSize::CramersV(v)) => assert!((v - 0.05433137570422292).abs() < 1e-9, "cramers_v {v}"),
        _ => panic!("expected CramersV effect size"),
    }
}

#[derive(Deserialize)]
struct CorFix { a: Vec<f64>, b: Vec<f64>, r: f64, t: f64, p: f64 }

#[test]
fn pearson_cor_matches_scipy() {
    let fx: CorFix = read("cor");
    let res = cor_test(&fx.a, &fx.b, CorMethod::Pearson).unwrap();
    let r = match res.effect_size { Some(EffectSize::R(r)) => r, _ => panic!("no r") };
    assert!(close(r, fx.r, STAT, 1e-14), "r {r} vs {}", fx.r);
    assert!(close(res.p_value, fx.p, PVAL, 1e-14), "p {} vs {}", res.p_value, fx.p);
    assert!(close(res.statistic, fx.t, STAT, 1e-14), "t {} vs {}", res.statistic, fx.t);
}

#[derive(Deserialize)]
struct FtestFix { a: Vec<f64>, b: Vec<f64>, #[serde(rename = "F")] f: f64, df1: f64, p: f64 }

#[test]
fn f_test_var_matches_scipy() {
    let fx: FtestFix = read("ftest");
    let r = f_test_var(&fx.a, &fx.b).unwrap();
    assert!(close(r.statistic, fx.f, STAT, 1e-14), "F {} vs {}", r.statistic, fx.f);
    assert!(close(r.df, fx.df1, STAT, 1e-12), "df1 {} vs {}", r.df, fx.df1);
    assert!(close(r.p_value, fx.p, PVAL, 1e-14), "p {} vs {}", r.p_value, fx.p);
}
