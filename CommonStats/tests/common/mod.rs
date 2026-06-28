//! Shared oracle-grid harness (accuracy-validation spec §5). Centralizes the
//! fixture load + tiered close-check that was copy-pasted across the `*_oracle.rs`
//! test files. `expected` is the committed mpmath-truth value; the `xcheck`
//! provenance block in the fixture is intentionally not modeled here — serde
//! ignores it (it is the human audit trail, not part of the recurring test).
//!
//! `allow(dead_code)`: each `*_oracle.rs` is a separate test binary that pulls in
//! the whole harness but uses only its slice — the grid binaries (special/dist)
//! use `check_grid` et al.; the bespoke binaries (transform/density) use only
//! `assert_close`/`Tol`. Every item is live in some binary, dead in others.
#![allow(dead_code)]

use serde::Deserialize;

/// Region of a grid row. `bulk` = the probability argument lies in
/// `[1e-6, 1−1e-6]` (or the row is not probability-indexed); `tail` otherwise,
/// where only mpmath truth is trusted and the assertion uses a measured band.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    #[default]
    Bulk,
    Tail,
}

#[derive(Deserialize)]
pub struct Row {
    pub args: Vec<f64>,
    /// mpmath truth; `null` where the quantity is undefined/±∞ (row is skipped).
    pub expected: Option<f64>,
    #[serde(default)]
    pub tier: Tier,
    /// Tail rows only: the reviewed accuracy band (our impl's measured rel error
    /// vs truth). `None` until curated — such rows fall back to the graceful check.
    #[serde(default)]
    pub band: Option<f64>,
}

/// Bulk relative/absolute tolerance for a quantity-kind (spec §3 ladder).
pub struct Tol {
    pub rel: f64,
    pub abs: f64,
}

/// Point-wise accuracy check for bespoke-shape fixtures (quantities that do not
/// use the generic check_grid grid). Mirrors the grid check's bulk-row assertion.
pub fn assert_close(label: &str, got: f64, want: f64, tol: Tol) {
    let diff = (got - want).abs();
    let t = tol.abs.max(tol.rel * want.abs());
    assert!(
        diff <= t,
        "{label}: got {got:e}, want {want:e}, diff {diff:e} > tol {t:e}"
    );
}

/// Multiplier on a tail row's curated band — absorbs cross-platform libm jitter.
pub const TAIL_SLACK: f64 = 4.0;

/// Below this magnitude a truth value is effectively zero, so its sign carries no
/// information; the graceful sign check is skipped there to avoid false failures
/// on near-zero numerical noise (underflowed incomplete-gamma/beta rows).
const SIGN_FLOOR: f64 = 1e-12;

pub fn load(name: &str) -> Vec<Row> {
    let path = format!("{}/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let txt = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {path}: {e} — run scripts/gen_oracle.py"));
    serde_json::from_str(&txt).expect("fixture parse")
}

/// Always-on check (spec §3): result is finite, and its sign matches truth where
/// truth is meaningfully non-zero. Catches NaN/Inf and wrong-branch even where the
/// numeric band is loose or absent.
fn graceful(name: &str, args: &[f64], got: f64, want: f64) {
    assert!(got.is_finite(), "{name}{args:?}: non-finite result {got:e}");
    if want.abs() > SIGN_FLOOR {
        assert!(
            got.signum() == want.signum(),
            "{name}{args:?}: sign mismatch — got {got:e}, want {want:e}"
        );
    }
}

fn rel_err(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

/// Per fixture: null-skip → graceful check → bulk rows within `tol`; tail rows
/// within `TAIL_SLACK × band` (or graceful-only while `band` is uncurated).
pub fn check_grid(name: &str, tol: Tol, eval: impl Fn(&[f64]) -> f64) {
    for r in load(name) {
        let Some(want) = r.expected else { continue }; // expected==null → skip
        let got = eval(&r.args);
        graceful(name, &r.args, got, want);
        match r.tier {
            Tier::Bulk => {
                let t = tol.abs.max(tol.rel * want.abs());
                let diff = (got - want).abs();
                assert!(
                    diff <= t,
                    "{name}{:?}: got {got:e}, want {want:e}, diff {diff:e} > tol {t:e}",
                    r.args
                );
            }
            Tier::Tail => {
                if let Some(band) = r.band {
                    let allowed = TAIL_SLACK * band;
                    let rel = rel_err(got, want);
                    assert!(
                        rel <= allowed,
                        "{name}{:?} tail: rel {rel:e} > slack·band {allowed:e} (band {band:e})",
                        r.args
                    );
                }
                // Uncurated band → graceful-only; already asserted above.
            }
        }
    }
}

/// Opt-in monotonicity check for cdf-like (↑ in x) and quantile (↑ in p) grids
/// whose fixture rows are ordered by the varying argument. A small slack absorbs
/// float noise at flat segments.
pub fn assert_monotone(name: &str, eval: impl Fn(&[f64]) -> f64) {
    let mut prev: Option<f64> = None;
    for r in load(name) {
        let y = eval(&r.args);
        if let Some(p) = prev {
            assert!(
                y >= p - 1e-12,
                "{name}{:?}: not monotone↑ — {y:e} < previous {p:e}",
                r.args
            );
        }
        prev = Some(y);
    }
}

/// Max observed relative error of `eval` vs truth over a fixture's *tail* rows —
/// drives the `#[ignore]`d band-calibration tests (spec §5). `None` if the fixture
/// has no (non-null) tail rows. The generator cannot compute this — it is a
/// property of the Rust impl vs truth — so a human curates the printed values
/// (lightly padded) into the fixtures' `band` field.
pub fn measure_band(name: &str, eval: impl Fn(&[f64]) -> f64) -> Option<f64> {
    let mut worst: Option<f64> = None;
    for r in load(name) {
        if r.tier != Tier::Tail {
            continue;
        }
        let Some(want) = r.expected else { continue };
        let rel = rel_err(eval(&r.args), want);
        worst = Some(worst.map_or(rel, |w| w.max(rel)));
    }
    worst
}
