//! Replays the points `scripts/accuracy_sweep.py --export-fixtures` wrote to
//! `tests/fixtures/accuracy_*.json` (the worst points per function and kernel
//! regime of the last sweep, with their mpmath truth). Each point's error
//! ratio
//!
//! `r = |got − truth| / (ε · max(k, |truth|, 2⁻¹⁰²²))`, ε = 2⁻⁵²,
//!
//! with `k` the sensitivity `Σ|xᵢ·∂f/∂xᵢ|` over the real inputs, must stay
//! within `max_r`, the limit of the class it was exported in. The truth is
//! stored rounded to f64, which moves `r` by at most ½, hence the `+ 0.5`. A
//! discrete quantile must return the true `k` or one of `alt` (neighbours the
//! sweep scored within the limit). Points still in the bug class carry `skip`
//! (the open bug they reproduce) and are not asserted until a fix re-exports
//! them. No mpmath at test time.
#[path = "common/accuracy_eval.rs"]
mod accuracy_eval;

use accuracy_eval::{Arg, Val, eval};
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Point {
    #[serde(rename = "fn")]
    name: String,
    params: Vec<String>,
    x: String,
    truth: String,
    #[serde(default)]
    k: Option<String>,
    #[serde(default)]
    max_r: Option<f64>,
    #[serde(default)]
    skip: Option<String>,
    #[serde(default)]
    alt: Vec<i64>,
}

/// `r` of `got` against `truth` (an `inf` truth lies beyond the f64 range,
/// where `±f64::MAX` also counts as exact).
fn ratio(got: f64, truth: f64, k: f64) -> f64 {
    if truth.is_infinite() {
        let same = (got.is_infinite() || got.abs() == f64::MAX) && got.signum() == truth.signum();
        return if same { 0.0 } else { f64::INFINITY };
    }
    if got.is_nan() {
        return f64::INFINITY;
    }
    (got - truth).abs() / (EPS * k.max(truth.abs()).max(f64::MIN_POSITIVE))
}

fn check(p: &Point) -> Result<(), String> {
    let args = |s: &str| Arg::parse(s);
    let params = p
        .params
        .iter()
        .map(|s| args(s))
        .collect::<Result<Vec<_>, _>>()?;
    let got = eval(&p.name, &params, args(&p.x)?)?;
    match got {
        Val::I(k) => {
            let want: i64 = p
                .truth
                .parse()
                .map_err(|e| format!("truth {}: {e}", p.truth))?;
            if k == want || p.alt.contains(&k) {
                Ok(())
            } else {
                Err(format!("got {k}, want {want} (or {:?})", p.alt))
            }
        }
        Val::F(v) => {
            let truth: f64 = p
                .truth
                .parse()
                .map_err(|e| format!("truth {}: {e}", p.truth))?;
            let k: f64 =
                p.k.as_deref()
                    .unwrap_or("0")
                    .parse()
                    .map_err(|e| format!("k: {e}"))?;
            let max_r = p.max_r.ok_or("point without max_r or skip")?;
            let r = ratio(v, truth, k);
            if r <= max_r + 0.5 {
                Ok(())
            } else {
                Err(format!("got {v:e}, truth {truth:e}, r = {r:.3e} > {max_r}"))
            }
        }
    }
}

fn replay(group: &str) {
    let path = format!(
        "{}/tests/fixtures/accuracy_{group}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let txt = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("missing {path}: {e} — run scripts/accuracy_sweep.py --export-fixtures")
    });
    let points: Vec<Point> = serde_json::from_str(&txt).expect("fixture parse");
    let mut failures = Vec::new();
    for p in points.iter().filter(|p| p.skip.is_none()) {
        if let Err(e) = check(p) {
            failures.push(format!("{}({:?}; {}): {e}", p.name, p.params, p.x));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} points outside their class limit:\n{}",
        failures.len(),
        points.len(),
        failures.join("\n")
    );
}

#[test]
fn incomplete_points() {
    replay("incomplete");
}

#[test]
fn special_points() {
    replay("special");
}

#[cfg(feature = "dist")]
#[test]
fn continuous_points() {
    replay("continuous");
}

#[cfg(feature = "dist")]
#[test]
fn discrete_points() {
    replay("discrete");
}
