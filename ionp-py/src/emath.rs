//! PyO3 bindings for `numpy.emath` (a.k.a. `numpy.lib.scimath`): the nine
//! branch-cut-aware functions (`sqrt`/`log`/`log2`/`log10`/`logn`/`power`/
//! `arccos`/`arcsin`/`arctanh`) that promote real input to complex instead
//! of returning NaN. This file only parses Python arguments and marshals
//! them into `ionp_core::emath`'s pure-Rust promotion logic -- every actual
//! numeric decision (the branch-cut trigger conditions, the dtype-promotion
//! table) lives there, per the crate's "Rust does arithmetic, Python does
//! dispatch" rule (mirrors `manip.rs`'s own module doc, including reusing
//! its `extract_or_ingest_ndarray` pattern locally since `manip.rs`'s copy
//! is not `pub`).

use pyo3::prelude::*;

use ionp_core::{emath, NdArray};

use crate::{to_py_err, PyArray};

fn extract_or_ingest_ndarray(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(pyref.inner.clone());
    }
    crate::array_impl(obj, None)
}

/// `emath.*` functions must obey the same full-reduction-to-scalar contract
/// as every other anionpy entry point: numpy's `emath.sqrt`/`log`/etc. return a
/// genuine numpy scalar (`np.float64`, `np.complex128`, ...) when applied to
/// 0-d input, never a 0-d `ndarray`. This mirrors `linalg.rs`'s
/// `wrap_norm_result` and `reductions.rs`'s `wrap_reduction` -- same
/// `numpy_scalar_from_0d` helper, applied here since this file previously
/// had zero references to it and always returned a bare `PyArray`.
fn wrap_emath_result(py: Python<'_>, arr: NdArray) -> PyResult<Py<PyAny>> {
    if arr.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &arr);
    }
    Ok(Py::new(py, PyArray { inner: arr })?.into_any())
}

#[pyfunction]
fn sqrt(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let a = extract_or_ingest_ndarray(x)?;
    wrap_emath_result(py, emath::emath_sqrt(&a).map_err(to_py_err)?)
}

#[pyfunction]
fn log(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let a = extract_or_ingest_ndarray(x)?;
    wrap_emath_result(py, emath::emath_log(&a).map_err(to_py_err)?)
}

#[pyfunction]
fn log2(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let a = extract_or_ingest_ndarray(x)?;
    wrap_emath_result(py, emath::emath_log2(&a).map_err(to_py_err)?)
}

#[pyfunction]
fn log10(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let a = extract_or_ingest_ndarray(x)?;
    wrap_emath_result(py, emath::emath_log10(&a).map_err(to_py_err)?)
}

/// numpy's own signature is `logn(n, x)` -- base first, argument second
/// (verified from `numpy/lib/_scimath_impl.py`'s `def logn(n, x):`) -- kept
/// in the same order here.
#[pyfunction]
fn logn(py: Python<'_>, n: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let n = extract_or_ingest_ndarray(n)?;
    let x = extract_or_ingest_ndarray(x)?;
    wrap_emath_result(py, emath::emath_logn(&n, &x).map_err(to_py_err)?)
}

#[pyfunction]
fn power(py: Python<'_>, x: &Bound<'_, PyAny>, p: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let x = extract_or_ingest_ndarray(x)?;
    let p = extract_or_ingest_ndarray(p)?;
    wrap_emath_result(py, emath::emath_power(&x, &p).map_err(to_py_err)?)
}

#[pyfunction]
fn arccos(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let a = extract_or_ingest_ndarray(x)?;
    wrap_emath_result(py, emath::emath_arccos(&a).map_err(to_py_err)?)
}

#[pyfunction]
fn arcsin(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let a = extract_or_ingest_ndarray(x)?;
    wrap_emath_result(py, emath::emath_arcsin(&a).map_err(to_py_err)?)
}

#[pyfunction]
fn arctanh(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let a = extract_or_ingest_ndarray(x)?;
    wrap_emath_result(py, emath::emath_arctanh(&a).map_err(to_py_err)?)
}

/// Registers a real `emath` submodule (mirrors `linalg.rs`'s own
/// `PyModule::new` + `add_submodule` pattern) so `anionpy.emath.sqrt(...)`
/// resolves exactly like `numpy.emath.sqrt(...)` -- these are NOT
/// top-level `anionpy.*` names, matching numpy's own namespacing.
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "emath")?;
    m.add_function(wrap_pyfunction!(sqrt, &m)?)?;
    m.add_function(wrap_pyfunction!(log, &m)?)?;
    m.add_function(wrap_pyfunction!(log2, &m)?)?;
    m.add_function(wrap_pyfunction!(log10, &m)?)?;
    m.add_function(wrap_pyfunction!(logn, &m)?)?;
    m.add_function(wrap_pyfunction!(power, &m)?)?;
    m.add_function(wrap_pyfunction!(arccos, &m)?)?;
    m.add_function(wrap_pyfunction!(arcsin, &m)?)?;
    m.add_function(wrap_pyfunction!(arctanh, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
