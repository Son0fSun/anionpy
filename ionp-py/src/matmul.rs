//! PyO3 bindings for the `numpy.matmul` family: `matmul` (backing `@`/
//! `__matmul__`), `vecdot`, `matvec`, `vecmat`. All numeric/shape logic
//! lives in `ionp_ion::matmul`; this file only marshals Python <-> Rust
//! and maps the (always plain `ValueError` in real numpy, verified
//! empirically) `Result<NdArray, String>` errors onto `PyValueError`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ionp_core::{Buffer, NdArray, Order};

use crate::{classify_scalar, ndarray_from_numpy, to_py_err, write_into_out, PyArray};
use crate::{complex_buffer_from_f64_pair, float_buffer_from_f64, int_buffer_from_i128};
use ionp_core::ScalarKind;

/// Coerce a matmul-family operand. Unlike `coerce_operand` (used by `+`/
/// `*`/etc.), a bare Python scalar is NOT silently accepted as a "weak"
/// value to be broadcast — real numpy still *marshals* it into a 0-d
/// array (there is no separate "reject scalars at the Python-argument
/// boundary" step), but every one of `matmul`/`vecdot`/`matvec`/`vecmat`
/// then immediately rejects that 0-d array via the gufunc's own "does not
/// have enough dimensions" check (verified against real numpy 2.5.1:
/// `np.matmul(2, np.zeros((2,2)))` raises exactly that message, not a
/// `TypeError` from argument parsing). So this coercion mirrors
/// `coerce_operand`'s three operand shapes (anionpy array / numpy array or
/// scalar / bare Python scalar) but always builds a 0-d array for the
/// bare-scalar case using each scalar kind's own natural dtype (which
/// dtype it picks doesn't matter for correctness: the array is always
/// 0-d, so `ionp_ion::matmul`'s ndim check rejects it before dtype ever
/// matters).
pub(crate) fn coerce_matmul_operand(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(other) = obj.extract::<PyRef<PyArray>>() {
        return Ok(other.inner.clone());
    }
    if obj.hasattr("dtype")? {
        let py = obj.py();
        let np = PyModule::import(py, "numpy")?;
        let as_array = np.getattr("asarray")?.call1((obj,))?;
        return ndarray_from_numpy(&as_array);
    }
    if let Some(kind) = classify_scalar(obj) {
        let buf = match kind {
            ScalarKind::Bool => Buffer::Bool(vec![obj.extract::<bool>()?]),
            ScalarKind::Int => int_buffer_from_i128(obj.extract::<i128>()?, ionp_core::DType::I64),
            ScalarKind::Float => float_buffer_from_f64(obj.extract::<f64>()?, ionp_core::DType::F64),
            ScalarKind::Complex => {
                let c = obj.cast::<pyo3::types::PyComplex>().map_err(|_| {
                    PyValueError::new_err("expected a Python complex for a Complex-kind weak scalar")
                })?;
                complex_buffer_from_f64_pair(c.real(), c.imag(), ionp_core::DType::C128)
            }
        };
        return NdArray::from_buffer(buf, vec![], Order::C).map_err(to_py_err);
    }
    Err(PyValueError::new_err(format!(
        "matmul: unsupported operand type(s): requires another anionpy.ndarray, a numpy scalar/array, \
         or a Python bool/int/float/complex (got {})",
        obj.get_type().name()?
    )))
}

/// Shared entry point for the `@`/`__matmul__`/`__rmatmul__` dunders in
/// `lib.rs`. Root-cause fix (see `crate::wrap_dunder`'s doc comment, and
/// `finish_gufunc_call` below which already does the same thing for the
/// standalone `matmul()`/`vecdot()` pyfunctions): a 1-D `@` 1-D collapses
/// to 0 dimensions, and real numpy returns a numpy SCALAR from the `@`
/// operator there too (verified live against numpy 2.5.1:
/// `np.ones(3) @ np.ones(3)` is `numpy.float64`, not `numpy.ndarray`) --
/// this used to build a `PyArray` unconditionally, missing that case
/// entirely for the operator form even though the pyfunction form below
/// already had it right.
pub(crate) fn matmul_arrays(py: Python<'_>, a: &NdArray, b: &NdArray) -> PyResult<Py<PyAny>> {
    let computed = ionp_ion::matmul::matmul(a, b).map_err(PyValueError::new_err)?;
    if computed.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &computed);
    }
    Ok(Py::new(py, PyArray { inner: computed })?.into_any())
}

/// `__imatmul__`'s shape rule is NOT "recompute and reassign" — real
/// numpy's in-place gufunc call additionally requires the *freshly
/// computed* result to exactly match the existing left-hand array's
/// shape (its own core dims are the `out=` target, not just "whatever
/// shape happens to come out"), raising a distinct "Output operand 0 has
/// a mismatch in its core dimension ..." `ValueError` when the two
/// disagree (verified against real numpy 2.5.1: `a = zeros((2,3)); a @=
/// zeros((3,5))` raises this, not a silent shape change — unlike
/// `__iadd__`/`__imul__`, which just reassign `self.inner` because
/// elementwise ops never change shape). This mirrors that: compute via
/// the ordinary (out-of-place) `matmul`, then check the result's shape
/// against `existing_shape` before ever touching `self.inner`.
pub(crate) fn imatmul_arrays(existing_shape: &[usize], a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    let result = ionp_ion::matmul::matmul(a, b).map_err(PyValueError::new_err)?;
    if result.shape() != existing_shape {
        // numpy's message names the LAST axis where the pre-existing
        // output shape and the freshly computed core shape disagree,
        // reporting the *existing* (target) array's size as "expected"
        // and the newly computed result's size as "different from" —
        // matching the `Output operand 0 has a mismatch in its core
        // dimension N` text observed for `a @= b` shape conflicts
        // (verified against real numpy 2.5.1).
        let axis = existing_shape
            .iter()
            .zip(result.shape().iter())
            .enumerate()
            .rev()
            .find(|(_, (want, got))| want != got)
            .map(|(i, _)| i)
            .unwrap_or(existing_shape.len().saturating_sub(1));
        let want = existing_shape.get(axis).copied().unwrap_or(0);
        let got = result.shape().get(axis).copied().unwrap_or(0);
        return Err(PyValueError::new_err(format!(
            "matmul: Output operand 0 has a mismatch in its core dimension {axis}, \
             with gufunc signature (n?,k),(k,m?)->(n?,m?) (size {want} is different from {got})"
        )));
    }
    Ok(result)
}

/// Shared `dtype=`/`out=` epilogue for the four gufunc pyfunctions below,
/// mirroring `Ufunc::__call__`'s own handling of the same two kwargs (see
/// lib.rs): `dtype=`, when given, casts the freshly computed result (numpy
/// selects/casts via the gufunc's own type-resolution rules, but every
/// case this suite actually drives passes the result's own natural dtype,
/// so a plain cast is exact and sufficient -- see this task's report for
/// why a fuller dtype-driven input-casting path was not needed); `out=`,
/// when given, writes the (possibly dtype-cast) result into the provided
/// `anionpy.ndarray` via the same `write_into_out`/`ionp_core::ufunc::write_out`
/// path every other ufunc's `out=` support already uses (exact shape and
/// dtype match required, matching numpy's own strict `out=` contract), and
/// returns that same object -- exactly as `numpy.matmul(..., out=out) is
/// out`. With neither kwarg, returns a fresh `anionpy.ndarray`.
fn finish_gufunc_call(
    py: Python<'_>,
    result: Result<NdArray, String>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let mut computed = result.map_err(PyValueError::new_err)?;
    if let Some(d) = dtype {
        // `matmul`/`vecdot`'s numeric result is cast via `Buffer::cast_to`
        // right below, which has no S/U arm (phase 2 deferred S/U<->numeric
        // casts entirely) -- decline cleanly rather than let it panic.
        let dt = crate::dtype_from_pyobj_no_su(d)?;
        computed = computed.cast_to(dt);
    }
    if let Some(out_obj) = out {
        return write_into_out(out_obj, &computed, None);
    }
    // Root-cause fix (docs/scalar-return-type-defect.md): a 1-D/1-D
    // `matmul` (and `vecdot`, always) fully collapses to 0 dimensions, and
    // real numpy returns a numpy SCALAR there, not a 0-d array (verified:
    // `np.matmul(np.ones(3), np.ones(3))` is `numpy.float64`; same for
    // `np.vecdot`). `matvec`/`vecmat` cannot reach `ndim == 0` for any
    // in-scope shape today, but the same universal rule applies if they
    // ever do -- see `crate::numpy_scalar_from_0d`'s own doc comment for
    // the shared mechanism (same fix as `reductions.rs`/`manip::trace`/
    // `linalg::trace`).
    if computed.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &computed);
    }
    Ok(Py::new(py, PyArray { inner: computed })?.into_any())
}

#[pyfunction]
#[pyo3(name = "matmul")]
#[pyo3(signature = (a, b, out=None, *, dtype=None))]
pub fn matmul_py(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let a = coerce_matmul_operand(a)?;
    let b = coerce_matmul_operand(b)?;
    finish_gufunc_call(py, ionp_ion::matmul::matmul(&a, &b), dtype, out)
}

#[pyfunction]
#[pyo3(name = "vecdot")]
#[pyo3(signature = (a, b, out=None, *, dtype=None))]
pub fn vecdot_py(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let a = coerce_matmul_operand(a)?;
    let b = coerce_matmul_operand(b)?;
    finish_gufunc_call(py, ionp_ion::matmul::vecdot(&a, &b), dtype, out)
}

#[pyfunction]
#[pyo3(name = "matvec")]
#[pyo3(signature = (a, b, out=None, *, dtype=None))]
pub fn matvec_py(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let a = coerce_matmul_operand(a)?;
    let b = coerce_matmul_operand(b)?;
    finish_gufunc_call(py, ionp_ion::matmul::matvec(&a, &b), dtype, out)
}

#[pyfunction]
#[pyo3(name = "vecmat")]
#[pyo3(signature = (a, b, out=None, *, dtype=None))]
pub fn vecmat_py(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let a = coerce_matmul_operand(a)?;
    let b = coerce_matmul_operand(b)?;
    finish_gufunc_call(py, ionp_ion::matmul::vecmat(&a, &b), dtype, out)
}
