//! `neurostats._neurostats` — the compiled half of the `neurostats` wheel:
//! `Domain`, `tfce`, `tfce_bands` and `tfce_ols` over the `neurostats` crate.
//! Every user-facing convention lives in the Python layer
//! (`neurostats/`); this crate is a thin numpy-in / numpy-out shim.
//! Arrays are float64 / uint32 and C-contiguous (the Python layer converts);
//! the brain data of `tfce_ols` may also be float32 and are read in place.
//! The sweeps run with the GIL released. Errors map `NeuroError` to
//! `ValueError` with the Rust message.

use neurostats::{
    Conn, Engine, NeuroError, NullState, PermOpts, PermProblem, Store, Tail, TfceInference,
    TfceParams, TfceWorkspace, Weighting, finalize, nuisance_basis, run_threads,
};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
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
        "nilearn" => Ok(Weighting::Nilearn),
        w => Err(PyValueError::new_err(format!(
            "weighting must be 'smith_nichols', 'mne', 'exact' or 'nilearn', got {w:?}"
        ))),
    }
}

fn tail_of(name: &str) -> PyResult<Tail> {
    match name {
        "positive" => Ok(Tail::Positive),
        "negative" => Ok(Tail::Negative),
        "two_sided" => Ok(Tail::TwoSided),
        t => Err(PyValueError::new_err(format!(
            "tail must be 'positive', 'negative' or 'two_sided', got {t:?}"
        ))),
    }
}

/// `TfceParams` from the resolved Python settings. `start` and `step` are
/// `None` for a weighting that does not use them; the crate validates both
/// fields under every weighting, so a value that passes is filled in.
fn params_of(
    e: f64,
    h: f64,
    start: Option<f64>,
    step: Option<f64>,
    weighting: &str,
) -> PyResult<TfceParams> {
    Ok(TfceParams {
        e,
        h,
        start: start.unwrap_or(0.0),
        step: step.unwrap_or(1.0),
        weighting: weighting_of(weighting)?,
    })
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

/// `neurostats::tfce_tail_into` over numpy arrays with a fresh workspace; GIL
/// released for the sweep. `weighting` and `tail` are the Python names.
#[pyfunction]
#[pyo3(signature = (domain, stat, e, h, start, step, weighting, tail))]
#[allow(clippy::too_many_arguments)]
fn tfce<'py>(
    py: Python<'py>,
    domain: &PyDomain,
    stat: PyReadonlyArray1<'py, f64>,
    e: f64,
    h: f64,
    start: Option<f64>,
    step: Option<f64>,
    weighting: &str,
    tail: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s = stat.as_slice()?;
    let dom = &domain.inner;
    let p = params_of(e, h, start, step, weighting)?;
    let tail = tail_of(tail)?;
    let out = py
        .detach(|| {
            let mut ws = TfceWorkspace::new(dom.n_nodes());
            let mut out = vec![0.0f64; dom.n_nodes()];
            neurostats::tfce_tail_into(dom, s, &p, tail, &mut ws, &mut out).map(|()| out)
        })
        .map_err(err)?;
    Ok(PyArray1::from_vec(py, out))
}

/// A null to continue, as the Python layer hands it over:
/// `(null_max, exceed, tfce_obs, n, q)`.
type StoredNull<'py> = (
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, u32>,
    PyReadonlyArray1<'py, f64>,
    usize,
    usize,
);

/// What an inference function returns:
/// `(stat_obs, tfce_obs, p_fwe, p_unc, null_max, exceed, n, q)`.
type InferenceOut<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u32>>,
    usize,
    usize,
);

/// The `NullState` of a stored null. The Python layer has already checked
/// that the run has the stored seed, settings and tail, so those three are
/// the run's own; the crate still compares the observed TFCE map bit for bit.
fn null_state(
    stored: &StoredNull<'_>,
    seed: u64,
    params: TfceParams,
    tail: Tail,
    engine: Engine,
) -> PyResult<NullState> {
    let (maxima, exceed, tfce_obs, n, q) = stored;
    let tfce_obs = tfce_obs.as_slice()?.to_vec();
    Ok(NullState {
        seed,
        maxima: maxima.as_slice()?.to_vec(),
        exceed: exceed.as_slice()?.to_vec(),
        n_nodes: tfce_obs.len(),
        tfce_obs,
        params,
        tail,
        n: *n,
        q: *q,
        engine,
    })
}

fn inference_out(py: Python<'_>, r: TfceInference) -> InferenceOut<'_> {
    (
        PyArray1::from_vec(py, r.stat_obs),
        PyArray1::from_vec(py, r.null.tfce_obs),
        PyArray1::from_vec(py, r.p_fwe),
        PyArray1::from_vec(py, r.p_unc),
        PyArray1::from_vec(py, r.null.maxima),
        PyArray1::from_vec(py, r.null.exceed),
        r.null.n,
        r.null.q,
    )
}

/// The brain data as numpy holds them, borrowed without a copy.
enum Source<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

/// TFCE permutation inference for `brain ~ tested + confounds + intercept`
/// (`neurostats::PermProblem::ols`), built from the pieces of the crate so
/// that the source is read in place.
///
/// `y`: 2-D, C-contiguous, float32 or float64, `(source rows, n_nodes)`; an
/// `np.memmap` is read through the page cache. `rows`: the source rows that
/// enter the model, in model order; `None` for every row. `tested`: one value
/// per model row, or `None` to test the intercept. `confounds`:
/// `(model rows, c)` or `None`. `threads == 0` means every core rayon can
/// see; results are bit-identical for any `threads`. `extend`: a stored null
/// to continue; `n_perm` is then the new total.
///
/// The store is built on the calling thread; the store, the draws and the
/// TFCE sweeps all run with the GIL released.
#[pyfunction]
#[pyo3(signature = (domain, y, rows, tested, confounds, e, h, start, step, weighting, tail, seed, n_perm, threads, extend))]
#[allow(clippy::too_many_arguments)]
fn tfce_ols<'py>(
    py: Python<'py>,
    domain: &PyDomain,
    y: &Bound<'py, PyAny>,
    rows: Option<PyReadonlyArray1<'py, u64>>,
    tested: Option<PyReadonlyArray1<'py, f64>>,
    confounds: Option<PyReadonlyArray2<'py, f64>>,
    e: f64,
    h: f64,
    start: Option<f64>,
    step: Option<f64>,
    weighting: &str,
    tail: &str,
    seed: u64,
    n_perm: u64,
    threads: usize,
    extend: Option<StoredNull<'py>>,
) -> PyResult<InferenceOut<'py>> {
    let dom = &domain.inner;
    let p = params_of(e, h, start, step, weighting)?;
    let tail = tail_of(tail)?;
    let y32 = y.cast::<PyArray2<f32>>().ok().map(|a| a.readonly());
    let y64 = y.cast::<PyArray2<f64>>().ok().map(|a| a.readonly());
    let (n_source, source) = match (&y32, &y64) {
        (Some(a), _) if a.is_c_contiguous() => (a.shape()[0], Source::F32(a.as_slice()?)),
        (_, Some(a)) if a.is_c_contiguous() => (a.shape()[0], Source::F64(a.as_slice()?)),
        _ => {
            return Err(PyValueError::new_err(
                "y must be a 2-D C-contiguous float32 or float64 array",
            ));
        }
    };
    let rows: Vec<usize> = match &rows {
        Some(r) => r.as_slice()?.iter().map(|&i| i as usize).collect(),
        None => (0..n_source).collect(),
    };
    let m = rows.len();
    let tested = tested.as_ref().map(|t| t.as_slice()).transpose()?;
    if let Some(t) = tested {
        if t.len() != m {
            return Err(PyValueError::new_err(format!(
                "tested has {} values, the model has {m} subjects",
                t.len()
            )));
        }
    }
    let conf: &[f64] = match &confounds {
        Some(c) if c.shape()[0] != m => {
            return Err(PyValueError::new_err(format!(
                "confounds has {} rows, the model has {m} subjects",
                c.shape()[0]
            )));
        }
        Some(c) if !c.is_c_contiguous() => {
            return Err(PyValueError::new_err("confounds must be C-contiguous"));
        }
        Some(c) => c.as_slice()?,
        None => &[],
    };
    let state = extend
        .as_ref()
        .map(|s| null_state(s, seed, p, tail, Engine::Ols))
        .transpose()?;
    let mut opts = PermOpts::new(p, tail, seed, n_perm);
    opts.extend = state.as_ref();
    let r = py.detach(|| {
        let basis = nuisance_basis(conf, m, tested.is_some()).map_err(err)?;
        let store = match source {
            Source::F32(s) => Store::from_f32(s, n_source, dom.n_nodes(), &rows, &basis),
            Source::F64(s) => Store::from_f64(s, n_source, dom.n_nodes(), &rows, &basis),
        }
        .map_err(err)?;
        let problem = PermProblem::ols(dom, &store, tested, &opts).map_err(|e| match e {
            // Settings were checked before the call, so with a null to
            // continue this is the crate refusing to continue it.
            NeuroError::InvalidParams if opts.extend.is_some() => PyValueError::new_err(
                "the stored null cannot be continued: it does not belong to this data, \
                 design, seed, tail and TFCE settings, or the new budget would enumerate \
                 every sign pattern",
            ),
            e => err(e),
        })?;
        let null = run_threads(&problem, problem.draws(), threads);
        finalize(&problem, &null).map_err(err)
    })?;
    Ok(inference_out(py, r))
}

#[pymodule]
fn _neurostats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDomain>()?;
    m.add_function(wrap_pyfunction!(tfce, m)?)?;
    m.add_function(wrap_pyfunction!(tfce_bands, m)?)?;
    m.add_function(wrap_pyfunction!(tfce_ols, m)?)?;
    Ok(())
}
