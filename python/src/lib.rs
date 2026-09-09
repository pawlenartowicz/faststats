//! `faststats._faststats` — the compiled half of the `faststats` wheel.
//! Submodules mirror the core crates; only `neuro` exists in 0.1.0. Every
//! user-facing convention lives in the Python layer (`python/faststats/`);
//! this crate is a thin numpy-in / numpy-out shim over `neurostats`.

use pyo3::prelude::*;

mod neuro;

#[pymodule]
fn _faststats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let neuro = PyModule::new(m.py(), "neuro")?;
    neuro::register(&neuro)?;
    m.add_submodule(&neuro)?;
    // Make `from faststats._faststats.neuro import X` importable: PyO3
    // submodules are attributes, not packages, until registered in sys.modules.
    m.py()
        .import("sys")?
        .getattr("modules")?
        .set_item("faststats._faststats.neuro", &neuro)?;
    Ok(())
}
