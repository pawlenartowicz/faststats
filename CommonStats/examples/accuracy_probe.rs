//! Evaluator of the accuracy harness (`scripts/accuracy_sweep.py`), built once
//! and fed every point of a sweep on stdin.
//!
//! Input: one JSON object per line, `{"fn": name, "params": [..], "x": ..}`,
//! numbers as strings (see `Arg` in `tests/common/accuracy_eval.rs`: JSON
//! numbers would pass through a parser that need not round correctly). Output:
//! one line per input line, in order: an `f64` in `{:?}` form (shortest
//! round-trip; `inf`, `NaN`, `-0.0` included), an integer for a discrete
//! quantile, or `err: <message>` (a constructor or quantile `Err`, a panic, a
//! malformed line).
#[path = "../tests/common/accuracy_eval.rs"]
mod accuracy_eval;

use accuracy_eval::{Arg, Val, eval};
use std::io::{BufRead, BufWriter, Write};

fn run(line: &str) -> Result<Val, String> {
    let v: serde_json::Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
    let name = v["fn"].as_str().ok_or("missing \"fn\"")?;
    let params = v["params"]
        .as_array()
        .ok_or("missing \"params\"")?
        .iter()
        .map(|p| {
            p.as_str()
                .ok_or("parameter not a string".to_string())
                .and_then(Arg::parse)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let x = Arg::parse(v["x"].as_str().ok_or("missing \"x\"")?)?;
    eval(name, &params, x)
}

fn main() {
    // A panic is reported on its line as `err: panic`; the default hook would
    // also print it to stderr.
    std::panic::set_hook(Box::new(|_| {}));
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("stdin is not UTF-8");
        let res =
            std::panic::catch_unwind(|| run(&line)).unwrap_or_else(|_| Err("panic".to_string()));
        match res {
            Ok(Val::F(v)) => writeln!(out, "{v:?}"),
            Ok(Val::I(k)) => writeln!(out, "{k}"),
            Err(e) => writeln!(out, "err: {}", e.replace('\n', " ")),
        }
        .expect("stdout closed");
    }
}
