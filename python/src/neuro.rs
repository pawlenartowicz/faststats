//! `faststats._faststats.neuro` — `Domain`, `tfce_bands`, `tfce_exact`,
//! `tfce_one_sample` over `neurostats`. Arrays are float64 / uint32 and
//! C-contiguous (the Python layer converts); the sweeps run with the GIL
//! released. Errors map `NeuroError` to `ValueError` with the Rust message.

use neurostats::{Conn, NeuroError, TfceParams, Weighting};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn err(e: NeuroError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn conn_of(conn: u32) -> PyResult<Conn> {
    match conn {
        6 => Ok(Conn::Face),
        18 => Ok(Conn::Edge),
        26 => Ok(Conn::Vertex),
        c => Err(PyValueError::new_err(format!(
            "conn must be 6, 18 or 26, got {c}"
        ))),
    }
}

fn weighting_of(name: &str) -> PyResult<Weighting> {
    match name {
        "smith_nichols" => Ok(Weighting::SmithNichols),
        "mne" => Ok(Weighting::MneStep),
        "exact" => Ok(Weighting::Exact),
        w => Err(PyValueError::new_err(format!(
            "weighting must be 'smith_nichols', 'mne' or 'exact', got {w:?}"
        ))),
    }
}

/// Adjacency over in-mask nodes (see `neurostats::Domain`). Node order is
/// C-order over the mask; the Python `Domain` wrapper owns shape and mask.
#[pyclass(frozen, name = "Domain")]
pub struct PyDomain {
    inner: neurostats::Domain,
}

#[pymethods]
impl PyDomain {
    #[staticmethod]
    fn from_volume(shape: (usize, usize, usize), conn: u32) -> PyResult<Self> {
        Ok(Self {
            inner: neurostats::Domain::from_volume([shape.0, shape.1, shape.2], conn_of(conn)?),
        })
    }

    /// `mask`: bool, 1-D, length `nx*ny*nz`, C-order over `shape`.
    #[staticmethod]
    fn from_mask(
        mask: PyReadonlyArray1<'_, bool>,
        shape: (usize, usize, usize),
        conn: u32,
    ) -> PyResult<Self> {
        let m = mask.as_slice()?;
        let inner = neurostats::Domain::from_mask(m, [shape.0, shape.1, shape.2], conn_of(conn)?)
            .map_err(err)?;
        Ok(Self { inner })
    }

    /// CSR `indptr` (`n+1`) and `indices`; symmetry is the caller's job.
    #[staticmethod]
    fn from_csr(
        indptr: PyReadonlyArray1<'_, u32>,
        indices: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Self> {
        let inner =
            neurostats::Domain::from_csr(indptr.as_slice()?.to_vec(), indices.as_slice()?.to_vec())
                .map_err(err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    #[getter]
    fn n_edges(&self) -> usize {
        self.inner.n_edges()
    }

    /// CSR row pointers. A masked-volume domain stores no neighbour list, so
    /// this materialises one on every call — `O(edges · log V)` time and memory.
    fn indptr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let (offsets, _) = self.inner.to_csr().map_err(err)?;
        Ok(PyArray1::from_vec(py, offsets))
    }

    /// CSR column indices; same cost as [`PyDomain::indptr`].
    fn indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let (_, neighbours) = self.inner.to_csr().map_err(err)?;
        Ok(PyArray1::from_vec(py, neighbours))
    }
}

/// `neurostats::tfce_bands` over numpy arrays; GIL released for the sweep.
#[pyfunction]
fn tfce_bands<'py>(
    py: Python<'py>,
    domain: &PyDomain,
    stat: PyReadonlyArray1<'py, f64>,
    thresholds: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    e: f64,
    strict: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (s, t, w) = (
        stat.as_slice()?,
        thresholds.as_slice()?,
        weights.as_slice()?,
    );
    let dom = &domain.inner;
    let out = py
        .detach(|| neurostats::tfce_bands(dom, s, t, w, e, strict))
        .map_err(err)?;
    Ok(PyArray1::from_vec(py, out))
}

/// `neurostats::tfce` under `Weighting::Exact` (`step` is unused there; a
/// positive placeholder satisfies `TfceParams` validation).
#[pyfunction]
fn tfce_exact<'py>(
    py: Python<'py>,
    domain: &PyDomain,
    stat: PyReadonlyArray1<'py, f64>,
    e: f64,
    h: f64,
    start: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s = stat.as_slice()?;
    let dom = &domain.inner;
    let p = TfceParams {
        e,
        h,
        start,
        step: 1.0,
        weighting: Weighting::Exact,
    };
    let out = py.detach(|| neurostats::tfce(dom, s, &p)).map_err(err)?;
    Ok(PyArray1::from_vec(py, out))
}

/// `neurostats::tfce_one_sample_threads`; `x` subject-major `(n_subj * n_nodes)`.
/// Returns `(t_obs, tfce_obs, p_fwe, p_unc, null_max)`. `threads == 0` means
/// every core rayon can see; results are bit-identical for any `threads`.
#[pyfunction]
#[pyo3(signature = (domain, x, n_subj, e, h, start, step, weighting, seed, n_perm, threads = 0))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn tfce_one_sample<'py>(
    py: Python<'py>,
    domain: &PyDomain,
    x: PyReadonlyArray1<'py, f64>,
    n_subj: usize,
    e: f64,
    h: f64,
    start: f64,
    step: f64,
    weighting: &str,
    seed: u64,
    n_perm: u64,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let xs = x.as_slice()?;
    let dom = &domain.inner;
    let p = TfceParams {
        e,
        h,
        start,
        step,
        weighting: weighting_of(weighting)?,
    };
    let r = py
        .detach(|| neurostats::tfce_one_sample_threads(dom, xs, n_subj, &p, seed, n_perm, threads))
        .map_err(err)?;
    Ok((
        PyArray1::from_vec(py, r.t_obs),
        PyArray1::from_vec(py, r.tfce_obs),
        PyArray1::from_vec(py, r.p_fwe),
        PyArray1::from_vec(py, r.p_unc),
        PyArray1::from_vec(py, r.null_max),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDomain>()?;
    m.add_function(wrap_pyfunction!(tfce_bands, m)?)?;
    m.add_function(wrap_pyfunction!(tfce_exact, m)?)?;
    m.add_function(wrap_pyfunction!(tfce_one_sample, m)?)?;
    Ok(())
}
