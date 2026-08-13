//! PyO3 bindings for `ionp_core::poly_legacy` -- the numeric kernels
//! behind numpy's LEGACY polynomial API (`np.poly1d`/`np.polyval`/
//! `np.polyadd`/`np.polysub`/`np.polyder`/`np.polyint`/`np.polydiv`/
//! `np.poly`/`np.roots`). See that module's doc comment for why this is
//! a SEPARATE file/module from `poly.rs` (which is `numpy.polynomial.
//! polynomial`, the modern power-series basis, concurrently owned by a
//! different part of this task).
//!
//! Deliberately self-contained (own small `ingest`/coercion helpers,
//! not shared with `poly.rs`'s `Series`/`coerce_series*`) for the same
//! reason `poly.rs` itself gives for not sharing with `linalg.rs`: this
//! file must never need to touch a file another concurrent agent may
//! also be editing.
//!
//! `polyadd`/`polysub`/`poly1d` construction/`roots`'s companion-matrix
//! assembly are NOT bound here -- built instead in
//! `anionpy/_polynomial_legacy.py` by composing already-shipped,
//! already-verified `anionpy` primitives (`concatenate`, `zeros`,
//! array `+`/`-`/`/`, `diag`, 2-D slice assignment, `anionpy.linalg.
//! eigvals`). Real numpy's OWN legacy implementations do the same thing
//! (`_polynomial_impl.py`'s `polyadd`/`polysub`/`roots` are themselves
//! just `NX.concatenate`/`NX.zeros`/`diag`/slice-assignment calls, no
//! bespoke loop) -- see `ionp/docs/TICKET-72-POLY1D-2026-08-08.md` for
//! the full reasoning on why that split does not violate this
//! codebase's "no arithmetic in `.py`" rule: every actual floating-point
//! operation involved already lives in a previously-shipped Rust ufunc,
//! this file/the Python module only sequences calls to those ufuncs.

use num_complex::Complex64 as C128;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use ionp_core::{poly_legacy, Buffer, DType, NdArray, Order};

use crate::PyArray;

fn err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

enum Series {
    R(Vec<f64>),
    C(Vec<C128>),
    I(Vec<i64>),
}

impl Series {
    fn is_complex(&self) -> bool {
        matches!(self, Series::C(_))
    }
    fn is_int(&self) -> bool {
        matches!(self, Series::I(_))
    }
    fn to_real(self) -> Vec<f64> {
        match self {
            Series::R(v) => v,
            Series::I(v) => v.into_iter().map(|x| x as f64).collect(),
            Series::C(_) => unreachable!("caller must check is_complex first"),
        }
    }
    fn to_complex(self) -> Vec<C128> {
        match self {
            Series::R(v) => v.into_iter().map(|x| C128::new(x, 0.0)).collect(),
            Series::I(v) => v.into_iter().map(|x| C128::new(x as f64, 0.0)).collect(),
            Series::C(v) => v,
        }
    }
}

fn ingest(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(pyref.inner.clone());
    }
    crate::array_impl(obj, None)
}

/// `p`/`x` coercion for the legacy family: unlike the modern module's
/// `as_series`, legacy numpy does NOT force int/bool input to float --
/// `np.polyval(np.array([1,2,3]), 2)` stays `int64` (measured directly,
/// 2026-08-08: `np.polyval([1,2,3], 2)` returns Python `int` 11, and
/// `np.polyval(np.array([1,2,3]), np.array(2))` returns `np.int64(11)`).
/// Preserved here ONLY for the pure-int64 fast path (`Series::I`);
/// every other integer width and bool are promoted to `f64` on ingest
/// (a deliberate, documented scope narrowing -- see the module doc
/// comment and the ticket doc's "declined" section).
fn coerce(arr: &NdArray) -> PyResult<Series> {
    if arr.dtype().is_complex() {
        let contig = arr.cast_to(DType::C128).to_contiguous();
        let data = match contig.buffer() {
            Buffer::C128(v) => v.clone(),
            _ => unreachable!(),
        };
        return Ok(Series::C(data));
    }
    if arr.dtype() == DType::I64 {
        let contig = arr.to_contiguous();
        let data = match contig.buffer() {
            Buffer::I64(v) => v.clone(),
            _ => unreachable!(),
        };
        return Ok(Series::I(data));
    }
    let contig = arr.cast_to(DType::F64).to_contiguous();
    let data = match contig.buffer() {
        Buffer::F64(v) => v.clone(),
        _ => unreachable!(),
    };
    Ok(Series::R(data))
}

fn mk_real(py: Python<'_>, data: Vec<f64>) -> PyResult<Py<PyAny>> {
    let n = data.len();
    let inner = NdArray::from_buffer(Buffer::F64(data), vec![n], Order::C).map_err(crate::to_py_err)?;
    Ok(Py::new(py, PyArray { inner })?.into_any())
}
fn mk_complex(py: Python<'_>, data: Vec<C128>) -> PyResult<Py<PyAny>> {
    let n = data.len();
    let inner = NdArray::from_buffer(Buffer::C128(data), vec![n], Order::C).map_err(crate::to_py_err)?;
    Ok(Py::new(py, PyArray { inner })?.into_any())
}
fn mk_int64(py: Python<'_>, data: Vec<i64>) -> PyResult<Py<PyAny>> {
    let n = data.len();
    let inner = NdArray::from_buffer(Buffer::I64(data), vec![n], Order::C).map_err(crate::to_py_err)?;
    Ok(Py::new(py, PyArray { inner })?.into_any())
}

fn mk_shaped_real(py: Python<'_>, data: Vec<f64>, shape: Vec<usize>) -> PyResult<Py<PyAny>> {
    let inner = NdArray::from_buffer(Buffer::F64(data), shape.clone(), Order::C).map_err(crate::to_py_err)?;
    if shape.is_empty() {
        return crate::numpy_scalar_from_0d(py, &inner);
    }
    Ok(Py::new(py, PyArray { inner })?.into_any())
}
fn mk_shaped_complex(py: Python<'_>, data: Vec<C128>, shape: Vec<usize>) -> PyResult<Py<PyAny>> {
    let inner = NdArray::from_buffer(Buffer::C128(data), shape.clone(), Order::C).map_err(crate::to_py_err)?;
    if shape.is_empty() {
        return crate::numpy_scalar_from_0d(py, &inner);
    }
    Ok(Py::new(py, PyArray { inner })?.into_any())
}
fn mk_shaped_int64(py: Python<'_>, data: Vec<i64>, shape: Vec<usize>) -> PyResult<Py<PyAny>> {
    let inner = NdArray::from_buffer(Buffer::I64(data), shape.clone(), Order::C).map_err(crate::to_py_err)?;
    if shape.is_empty() {
        return crate::numpy_scalar_from_0d(py, &inner);
    }
    Ok(Py::new(py, PyArray { inner })?.into_any())
}

// ─────────────────────────── polyval ────────────────────────────

#[pyfunction]
#[pyo3(name = "_polyval_legacy")]
pub fn polyval_legacy_py(py: Python<'_>, p: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let parr = ingest(p)?;
    if parr.ndim() != 1 {
        return Err(err("anionpy.polyval: p must be 1-d in this build"));
    }
    let ps = coerce(&parr)?;
    let xarr = ingest(x)?;
    let out_shape = xarr.shape().to_vec();

    // Pure-int64 fast path: preserves numpy's own dtype-preserving
    // behaviour (see `coerce`'s doc comment) using wrapping i64
    // arithmetic, matching numpy's own (practically two's-complement)
    // integer overflow behaviour instead of promoting through f64.
    if ps.is_int() && xarr.dtype() == DType::I64 {
        let pv = match ps {
            Series::I(v) => v,
            _ => unreachable!(),
        };
        let contig = xarr.to_contiguous();
        let xs = match contig.buffer() {
            Buffer::I64(v) => v.clone(),
            _ => unreachable!(),
        };
        let out = poly_legacy::polyval_legacy_i64(&pv, &xs);
        return mk_shaped_int64(py, out, out_shape);
    }

    if !xarr.dtype().is_complex() && !ps.is_complex() {
        let contig = xarr.cast_to(DType::F64).to_contiguous();
        let xs = match contig.buffer() {
            Buffer::F64(v) => v.clone(),
            _ => unreachable!(),
        };
        let cv = ps.to_real();
        let out = poly_legacy::polyval_legacy(&cv, &xs);
        return mk_shaped_real(py, out, out_shape);
    }
    let contig = xarr.cast_to(DType::C128).to_contiguous();
    let xs = match contig.buffer() {
        Buffer::C128(v) => v.clone(),
        _ => unreachable!(),
    };
    let cv = ps.to_complex();
    let out = poly_legacy::polyval_legacy(&cv, &xs);
    mk_shaped_complex(py, out, out_shape)
}

// ───────────────────── N-D fallbacks (ticket #76) ──────────────────────
//
// `p.ndim() == 1` keeps using the dtype-specialized fast paths above
// (`_polyval_legacy`/`_polyder_legacy`/`_polyint_legacy`) unchanged --
// already independently verified bit-exact. These three bindings are
// only reached from `_polynomial_legacy.py` when `p.ndim() != 1`
// (0-d or ndim>=2): they call straight into the generic, dtype-agnostic
// `NdArray`-level kernels in `ionp_core::poly_legacy`, which reuse
// already-broadcast-correct primitives (`ufunc::binary_op`, `manip::
// concatenate`) instead of a bespoke reimplementation. None of the
// three ever need `numpy_scalar_from_0d`: `polyval`'s N-D Horner output
// is never 0-d (p.ndim() >= 2 here, so the broadcast result always
// keeps at least one of p's trailing axes); `polyder`'s N-D path never
// succeeds on a 0-d p (numpy's own `len()` call always raises first);
// and `polyint`'s `m == 0` 0-d success case returns a genuine 0-d
// `ndarray` in real numpy (`np.polyint(np.array(5), 0)` is `<class
// 'numpy.ndarray'>`, NOT a numpy scalar -- measured directly, unlike
// `polyval`'s scalar-returning convention), so plain `PyArray` wrapping
// is the byte-correct choice here, not a bug to fix.

fn wrap_plain(py: Python<'_>, inner: NdArray) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, PyArray { inner })?.into_any())
}

#[pyfunction]
#[pyo3(name = "_polyval_legacy_nd")]
pub fn polyval_legacy_nd_py(py: Python<'_>, p: &Bound<'_, PyAny>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let parr = ingest(p)?;
    let xarr = ingest(x)?;
    let out = poly_legacy::polyval_legacy_horner_nd(&parr, &xarr).map_err(crate::to_py_err)?;
    wrap_plain(py, out)
}

#[pyfunction]
#[pyo3(name = "_polyder_legacy_nd")]
pub fn polyder_legacy_nd_py(py: Python<'_>, p: &Bound<'_, PyAny>, m: i64) -> PyResult<Py<PyAny>> {
    let arr = ingest(p)?;
    let out = poly_legacy::polyder_legacy_nd(&arr, m).map_err(crate::to_py_err)?;
    wrap_plain(py, out)
}

#[pyfunction]
#[pyo3(name = "_polyint_legacy_nd")]
pub fn polyint_legacy_nd_py(py: Python<'_>, p: &Bound<'_, PyAny>, m: i64, k: Vec<f64>) -> PyResult<Py<PyAny>> {
    let arr = ingest(p)?;
    let out = poly_legacy::polyint_legacy_nd(&arr, m, &k).map_err(crate::to_py_err)?;
    wrap_plain(py, out)
}

// ─────────────────────────── polyder / polyint ────────────────────────────

#[pyfunction]
#[pyo3(name = "_polyder_legacy")]
pub fn polyder_legacy_py(py: Python<'_>, p: &Bound<'_, PyAny>, m: i64) -> PyResult<Py<PyAny>> {
    if m < 0 {
        return Err(err("Order of derivative must be positive (see polyint)"));
    }
    let arr = ingest(p)?;
    if arr.ndim() != 1 {
        return Err(err("anionpy.polyder: p must be 1-d in this build"));
    }
    match coerce(&arr)? {
        Series::I(v) => mk_int64(py, poly_legacy::polyder_legacy_i64(&v, m as usize)),
        Series::R(v) => mk_real(py, poly_legacy::polyder_legacy(&v, m as usize)),
        Series::C(v) => mk_complex(py, poly_legacy::polyder_legacy(&v, m as usize)),
    }
}

#[pyfunction]
#[pyo3(name = "_polyint_legacy")]
pub fn polyint_legacy_py(py: Python<'_>, p: &Bound<'_, PyAny>, k: Vec<f64>) -> PyResult<Py<PyAny>> {
    let arr = ingest(p)?;
    if arr.ndim() != 1 {
        return Err(err("anionpy.polyint: p must be 1-d in this build"));
    }
    // `k` always arrives here as an already-length-`m`, zero-padded f64
    // vector (built in Python -- see `_polynomial_legacy.py`'s
    // `polyint`); integration constants add a float offset regardless
    // of `p`'s own dtype, so the int64-preserving fast path `polyder`
    // has does not apply here (numpy's own `polyint` promotes through
    // `NX.zeros(m, float)`-typed `k` too -- verified: `np.polyint([1,2,
    // 3]).dtype` is `float64` even though the input is a Python int
    // list).
    match coerce(&arr)? {
        Series::C(v) => {
            let kk: Vec<C128> = k.iter().map(|&x| C128::new(x, 0.0)).collect();
            mk_complex(py, poly_legacy::polyint_legacy(&v, &kk))
        }
        other => {
            let v = other.to_real();
            mk_real(py, poly_legacy::polyint_legacy(&v, &k))
        }
    }
}

// ─────────────────────────── polydiv ────────────────────────────

#[pyfunction]
#[pyo3(name = "_polydiv_legacy")]
pub fn polydiv_legacy_py(
    py: Python<'_>,
    u: &Bound<'_, PyAny>,
    v: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let uarr = ingest(u)?;
    let varr = ingest(v)?;
    if uarr.ndim() != 1 || varr.ndim() != 1 {
        return Err(err("anionpy.polydiv: u and v must be 1-d in this build"));
    }
    if uarr.shape()[0] == 0 || varr.shape()[0] == 0 {
        return Err(PyIndexError::new_err("index -1 is out of bounds for axis 0 with size 0"));
    }
    let us = coerce(&uarr)?;
    let vs = coerce(&varr)?;
    let complex_path = us.is_complex() || vs.is_complex();
    if complex_path {
        let uc = us.to_complex();
        let vc = vs.to_complex();
        let (q, r) = poly_legacy::polydiv_legacy(&uc, &vc);
        return Ok((mk_complex(py, q)?, mk_complex(py, r)?));
    }
    let ur = us.to_real();
    let vr = vs.to_real();
    let (q, r) = poly_legacy::polydiv_legacy(&ur, &vr);
    Ok((mk_real(py, q)?, mk_real(py, r)?))
}

// ─────────────────────────── poly (from roots) ────────────────────────────

#[pyfunction]
#[pyo3(name = "_poly_from_roots_legacy")]
pub fn poly_from_roots_legacy_py(py: Python<'_>, roots: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = ingest(roots)?;
    if arr.ndim() != 1 {
        return Err(err("anionpy.poly: seq_of_zeros must be 1-d in this build"));
    }
    if arr.shape()[0] == 0 {
        // Matches real numpy's own `if len(seq_of_zeros) == 0: return 1.0`
        // -- a bare Python float, not a 1-element array. The Python
        // caller special-cases this (checks length before calling in).
        return Err(err("anionpy internal: empty roots must be handled by the caller"));
    }
    match coerce(&arr)? {
        Series::C(v) => mk_complex(py, poly_legacy::poly_from_roots_legacy(&v)),
        other => {
            let v = other.to_real();
            mk_real(py, poly_legacy::poly_from_roots_legacy(&v))
        }
    }
}

pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(polyval_legacy_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(polyder_legacy_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(polyint_legacy_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(polyval_legacy_nd_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(polyder_legacy_nd_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(polyint_legacy_nd_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(polydiv_legacy_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_from_roots_legacy_py, m)?)?;
    Ok(())
}
