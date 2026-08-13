//! PyO3 bindings for the contraction family (`dot`, `vdot`, `inner`,
//! `tensordot`, `cross`). All arithmetic lives in `ionp_core::products`,
//! composed there entirely out of already-verified `ufunc` primitives (see
//! that module's doc comment for why, and for the known BLAS-ordering
//! precision caveat on float dot products). This file only parses Python
//! arguments and marshals `NdArray`s across the FFI boundary.
//!
//! Explicitly NOT implemented here: `matmul`/`vecdot`/`matvec`/`vecmat`
//! (owned by `matmul.rs`, another task) and `einsum`/`einsum_path` (a
//! general string-parsed contraction planner; deferred -- see the
//! products-cluster report for why).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{IntoPyObject, IntoPyObjectExt};

use ionp_core::{products, NdArray};

use crate::{to_py_err, PyArray};

fn extract_or_ingest_ndarray(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(pyref.inner.clone());
    }
    crate::array_impl(obj, None)
}

fn wrap_result(py: Python<'_>, inner: NdArray) -> PyResult<Py<PyAny>> {
    if inner.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &inner);
    }
    Py::new(py, PyArray { inner })?.into_py_any(py)
}

#[pyfunction]
#[pyo3(signature = (a, b, out=None))]
fn dot(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, out: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    if out.is_some() {
        return Err(PyValueError::new_err("anionpy.dot: out= is not supported"));
    }
    let aa = extract_or_ingest_ndarray(a)?;
    let bb = extract_or_ingest_ndarray(b)?;
    let inner = products::dot(&aa, &bb).map_err(to_py_err)?;
    wrap_result(py, inner)
}

#[pyfunction]
#[pyo3(signature = (a, b))]
fn vdot(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let aa = extract_or_ingest_ndarray(a)?;
    let bb = extract_or_ingest_ndarray(b)?;
    let inner = products::vdot(&aa, &bb).map_err(to_py_err)?;
    wrap_result(py, inner)
}

#[pyfunction]
#[pyo3(signature = (a, b))]
fn inner(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let aa = extract_or_ingest_ndarray(a)?;
    let bb = extract_or_ingest_ndarray(b)?;
    let inner = products::inner(&aa, &bb).map_err(to_py_err)?;
    wrap_result(py, inner)
}

/// Parses numpy's flexible `axes=` argument for `tensordot`: an int N
/// (contract the last N axes of `a` against the first N axes of `b`), or a
/// pair of int-or-sequence-of-int (explicit axis lists per operand).
fn parse_tensordot_axes(
    axes: &Bound<'_, PyAny>,
    a_ndim: usize,
    b_ndim: usize,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    let norm = |v: isize, ndim: usize| -> PyResult<usize> {
        let n = if v < 0 { v + ndim as isize } else { v };
        if n < 0 || n as usize >= ndim {
            return Err(PyValueError::new_err(format!("axis {} is out of bounds for array of dimension {}", v, ndim)));
        }
        Ok(n as usize)
    };
    if let Ok(n) = axes.extract::<isize>() {
        if n < 0 {
            return Err(PyValueError::new_err("tensordot axes must be non-negative"));
        }
        let n = n as usize;
        let axes_a: Vec<usize> = ((a_ndim - n)..a_ndim).collect();
        let axes_b: Vec<usize> = (0..n).collect();
        return Ok((axes_a, axes_b));
    }
    let seq = axes.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
    let one_side = |obj: &Bound<'_, PyAny>, ndim: usize| -> PyResult<Vec<usize>> {
        if let Ok(i) = obj.extract::<isize>() {
            return Ok(vec![norm(i, ndim)?]);
        }
        let items: Vec<isize> = obj.extract()?;
        items.into_iter().map(|i| norm(i, ndim)).collect()
    };
    let axes_a = one_side(&seq.0, a_ndim)?;
    let axes_b = one_side(&seq.1, b_ndim)?;
    Ok((axes_a, axes_b))
}

#[pyfunction]
#[pyo3(signature = (a, b, axes=None))]
fn tensordot(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    axes: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let aa = extract_or_ingest_ndarray(a)?;
    let bb = extract_or_ingest_ndarray(b)?;
    let two = 2isize.into_pyobject(py)?.into_any();
    let axes = axes.unwrap_or(&two);
    let (axes_a, axes_b) = parse_tensordot_axes(axes, aa.ndim(), bb.ndim())?;
    let inner = products::tensordot(&aa, &bb, &axes_a, &axes_b).map_err(to_py_err)?;
    wrap_result(py, inner)
}

#[pyfunction]
#[pyo3(signature = (a, b, axisa=-1, axisb=-1, axisc=-1, axis=None))]
fn cross(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    axisa: isize,
    axisb: isize,
    axisc: isize,
    axis: Option<isize>,
) -> PyResult<Py<PyAny>> {
    let aa = extract_or_ingest_ndarray(a)?;
    let bb = extract_or_ingest_ndarray(b)?;
    let (axisa, axisb, axisc) = match axis {
        Some(ax) => (ax, ax, ax),
        None => (axisa, axisb, axisc),
    };
    let norm = |v: isize, ndim: usize| -> PyResult<usize> {
        let n = if v < 0 { v + ndim as isize } else { v };
        if n < 0 || n as usize >= ndim {
            return Err(PyValueError::new_err(format!("axis {} is out of bounds for array of dimension {}", v, ndim)));
        }
        Ok(n as usize)
    };
    let axis_a = norm(axisa, aa.ndim())?;
    let axis_b = norm(axisb, bb.ndim())?;
    let out_ndim = aa.ndim().max(bb.ndim());
    let axis_c = if axisc < 0 { (axisc + out_ndim as isize).max(0) as usize } else { axisc as usize };
    let inner = products::cross(&aa, &bb, axis_a, axis_b, axis_c).map_err(to_py_err)?;
    wrap_result(py, inner)
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(vdot, m)?)?;
    m.add_function(wrap_pyfunction!(inner, m)?)?;
    m.add_function(wrap_pyfunction!(tensordot, m)?)?;
    m.add_function(wrap_pyfunction!(cross, m)?)?;
    Ok(())
}
