//! PyO3 bindings for the numeric kernels of `numpy.polynomial.legendre`.
//! Sibling to `ionp-py/src/poly.rs` (the power-series basis binding) —
//! reuses that file's `Series` enum and coercion helpers (`coerce_series`,
//! `coerce_series_strict`, `coerce_pair`, `mk_real`/`mk_complex`/
//! `mk_series`) rather than duplicating them, since the dtype-promotion
//! story (`as_series`-family bool-rejecting vs `legder`/`legint`/`legval`/
//! `legvander`'s own bool-promoting dtype path) is IDENTICAL between the
//! two bases — verified directly against `numpy/polynomial/legendre.py`:
//! `legder`/`legint`/`legval`/`legvander` each do their own `if c.dtype.char
//! in '?bBhHiIlLqQpP': c = c.astype(np.double)` (bool promoted), while
//! `legadd`/`legsub`/`legmulx`/`legmul`/`legdiv`/`legfromroots`/
//! `legcompanion`/`legpow` all route through `pu.as_series` (bool rejected)
//! -- the exact same split `poly.rs`'s `coerce_series_ex` doc comment
//! documents for the power basis.
//!
//! Only `legmulx`, `legcompanion`, `legmul`/`legdiv`, `legder`/`legint`,
//! `legval`, and `legvander` need dedicated Rust kernels
//! (`ionp_core::legendre`) -- `legadd`/`legsub` are the literal same
//! `crate::poly::add_trim`/`sub_trim` numpy itself shares across bases (see
//! `ionp_core::legendre`'s `legadd_trim`/`legsub_trim` re-exports).

use num_complex::Complex64 as C128;
use pyo3::exceptions::PyZeroDivisionError;
use pyo3::prelude::*;

use ionp_core::{legendre as leg, Buffer, DType, NdArray, Order};

use crate::poly::{coerce_series, coerce_series_strict, Series};

fn leg_err(msg: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.into())
}

fn mk_real(py: Python<'_>, data: Vec<f64>) -> PyResult<Py<PyAny>> {
    let n = data.len();
    let inner = NdArray::from_buffer(Buffer::F64(data), vec![n], Order::C).map_err(crate::to_py_err)?;
    Ok(Py::new(py, crate::PyArray { inner })?.into_any())
}
fn mk_complex(py: Python<'_>, data: Vec<C128>) -> PyResult<Py<PyAny>> {
    let n = data.len();
    let inner = NdArray::from_buffer(Buffer::C128(data), vec![n], Order::C).map_err(crate::to_py_err)?;
    Ok(Py::new(py, crate::PyArray { inner })?.into_any())
}
#[allow(dead_code)]
fn mk_series(py: Python<'_>, s: Series) -> PyResult<Py<PyAny>> {
    match s {
        Series::R(v) => mk_real(py, v),
        Series::C(v) => mk_complex(py, v),
    }
}
fn mk_int64(py: Python<'_>, data: Vec<i64>) -> PyResult<Py<PyAny>> {
    let n = data.len();
    let inner = NdArray::from_buffer(Buffer::I64(data), vec![n], Order::C).map_err(crate::to_py_err)?;
    Ok(Py::new(py, crate::PyArray { inner })?.into_any())
}

fn ingest(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, crate::PyArray>>() {
        return Ok(pyref.inner.clone());
    }
    crate::array_impl(obj, None)
}

fn coerce_pair(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<(Series, Series)> {
    let sa = coerce_series_strict(a)?;
    let sb = coerce_series_strict(b)?;
    if sa.is_complex() || sb.is_complex() {
        Ok((
            Series::C(match sa {
                Series::R(v) => v.into_iter().map(|x| C128::new(x, 0.0)).collect(),
                Series::C(v) => v,
            }),
            Series::C(match sb {
                Series::R(v) => v.into_iter().map(|x| C128::new(x, 0.0)).collect(),
                Series::C(v) => v,
            }),
        ))
    } else {
        Ok((sa, sb))
    }
}

// ─────────────────────────── constants ────────────────────────────

#[pyfunction]
#[pyo3(name = "_leg_domain")]
pub fn leg_domain_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_real(py, vec![-1.0, 1.0])
}
#[pyfunction]
#[pyo3(name = "_leg_zero")]
pub fn leg_zero_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![0])
}
#[pyfunction]
#[pyo3(name = "_leg_one")]
pub fn leg_one_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![1])
}
#[pyfunction]
#[pyo3(name = "_leg_x")]
pub fn leg_x_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![0, 1])
}

// ─────────────────────────── legline ────────────────────────────

#[pyfunction]
#[pyo3(name = "_leg_line")]
pub fn leg_line_py(py: Python<'_>, off: f64, scl: f64) -> PyResult<Py<PyAny>> {
    if scl != 0.0 {
        mk_real(py, vec![off, scl])
    } else {
        mk_real(py, vec![off])
    }
}

// ─────────────────────────── legval ────────────────────────────

fn mk_shaped_real(py: Python<'_>, data: Vec<f64>, shape: Vec<usize>) -> PyResult<Py<PyAny>> {
    let inner = NdArray::from_buffer(Buffer::F64(data), shape.clone(), Order::C).map_err(crate::to_py_err)?;
    if shape.is_empty() {
        return crate::numpy_scalar_from_0d(py, &inner);
    }
    Ok(Py::new(py, crate::PyArray { inner })?.into_any())
}
fn mk_shaped_complex(py: Python<'_>, data: Vec<C128>, shape: Vec<usize>) -> PyResult<Py<PyAny>> {
    let inner = NdArray::from_buffer(Buffer::C128(data), shape.clone(), Order::C).map_err(crate::to_py_err)?;
    if shape.is_empty() {
        return crate::numpy_scalar_from_0d(py, &inner);
    }
    Ok(Py::new(py, crate::PyArray { inner })?.into_any())
}

#[pyfunction]
#[pyo3(name = "_leg_val")]
pub fn leg_val_py(py: Python<'_>, x: &Bound<'_, PyAny>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // `legval` does its OWN dtype handling (`c = np.array(c, ndmin=1); if
    // c.dtype.char in ...: c = c.astype(np.double)`), same non-`as_series`
    // path `polyval` uses -- empty `c` is likewise an `IndexError` from
    // indexing `c[-2]`/`c[-1]` on an empty array in real numpy's own body
    // for len(c)==1/2 special cases... actually verified directly: an
    // EMPTY c hits `len(c) == 1` as false and falls to the `else` branch's
    // `c[-2]` -- `IndexError: index -2 is out of bounds for axis 0 with
    // size 0`. Matched explicitly here, same shape as `poly_val_py`.
    {
        let probe = ingest(c)?;
        if probe.shape().iter().product::<usize>() == 0 && probe.ndim() >= 1 {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "index -2 is out of bounds for axis 0 with size 0",
            ));
        }
    }
    let cs = coerce_series(c)?;
    let xarr = ingest(x)?;
    let x_is_complex = xarr.dtype().is_complex();
    let out_shape = xarr.shape().to_vec();

    if !x_is_complex && !cs.is_complex() {
        let contig = xarr.cast_to(DType::F64).to_contiguous();
        let xs = match contig.buffer() {
            Buffer::F64(v) => v.clone(),
            _ => unreachable!(),
        };
        let cv = match cs {
            Series::R(v) => v,
            _ => unreachable!(),
        };
        let out = leg::leg_eval(&cv, &xs);
        return mk_shaped_real(py, out, out_shape);
    }
    let contig = xarr.cast_to(DType::C128).to_contiguous();
    let xs = match contig.buffer() {
        Buffer::C128(v) => v.clone(),
        _ => unreachable!(),
    };
    let cv = cs.to_complex();
    let out = leg::leg_eval(&cv, &xs);
    mk_shaped_complex(py, out, out_shape)
}

// ─────────────────────── add / sub / mulx / mul / div ────────────────────────

#[pyfunction]
#[pyo3(name = "_leg_add")]
pub fn leg_add_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, leg::legadd_trim(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, leg::legadd_trim(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_leg_sub")]
pub fn leg_sub_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, leg::legsub_trim(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, leg::legsub_trim(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_leg_mulx")]
pub fn leg_mulx_py(py: Python<'_>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    match coerce_series_strict(c)? {
        Series::R(v) => mk_real(py, leg::legmulx(&v)),
        Series::C(v) => mk_complex(py, leg::legmulx(&v)),
    }
}

#[pyfunction]
#[pyo3(name = "_leg_mul")]
pub fn leg_mul_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, leg::legmul(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, leg::legmul(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_leg_div")]
pub fn leg_div_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => match leg::legdiv(&a, &b) {
            Ok((q, r)) => Ok((mk_real(py, q)?, mk_real(py, r)?)),
            Err(()) => Err(PyZeroDivisionError::new_err(())),
        },
        (Series::C(a), Series::C(b)) => match leg::legdiv(&a, &b) {
            Ok((q, r)) => Ok((mk_complex(py, q)?, mk_complex(py, r)?)),
            Err(()) => Err(PyZeroDivisionError::new_err(())),
        },
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

// ─────────────────────────── legder / legint ────────────────────────────

#[pyfunction]
#[pyo3(name = "_leg_der")]
pub fn leg_der_py(py: Python<'_>, c: &Bound<'_, PyAny>, m: i64, scl: f64) -> PyResult<Py<PyAny>> {
    if m < 0 {
        return Err(leg_err("The order of derivation must be non-negative"));
    }
    match coerce_series(c)? {
        Series::R(v) => mk_real(py, leg::legder(&v, m as usize, scl)),
        Series::C(v) => mk_complex(py, leg::legder(&v, m as usize, C128::new(scl, 0.0))),
    }
}

#[pyfunction]
#[pyo3(name = "_leg_int")]
pub fn leg_int_py(
    py: Python<'_>,
    c: &Bound<'_, PyAny>,
    m: i64,
    k: Vec<f64>,
    lbnd: f64,
    scl: f64,
) -> PyResult<Py<PyAny>> {
    if m < 0 {
        return Err(leg_err("The order of integration must be non-negative"));
    }
    if k.len() > m as usize {
        return Err(leg_err("Too many integration constants"));
    }
    let mut kk = k.clone();
    kk.resize(m as usize, 0.0);
    match coerce_series(c)? {
        Series::R(v) => mk_real(py, leg::legint(&v, m as usize, &kk, lbnd, scl)),
        Series::C(v) => {
            let kkc: Vec<C128> = kk.iter().map(|&x| C128::new(x, 0.0)).collect();
            mk_complex(
                py,
                leg::legint(&v, m as usize, &kkc, C128::new(lbnd, 0.0), C128::new(scl, 0.0)),
            )
        }
    }
}

// ─────────────────────────── legvander ────────────────────────────

#[pyfunction]
#[pyo3(name = "_leg_vander")]
pub fn leg_vander_py(py: Python<'_>, x: &Bound<'_, PyAny>, deg: i64) -> PyResult<Py<PyAny>> {
    if deg < 0 {
        return Err(leg_err("deg must be non-negative"));
    }
    let xarr = ingest(x)?;
    if xarr.ndim() > 1 {
        return Err(leg_err(format!(
            "legvander: expected a 1-d array of sample points, got {} dimensions \
             (batched multi-dim x is out of scope for this build)",
            xarr.ndim()
        )));
    }
    let n: usize = xarr.shape().iter().product::<usize>().max(1);
    let width = (deg as usize) + 1;
    if xarr.dtype().is_complex() {
        let contig = xarr.cast_to(DType::C128).to_contiguous();
        let xs = match contig.buffer() {
            Buffer::C128(v) => v.clone(),
            _ => unreachable!(),
        };
        let out = leg::legvander(&xs, deg as usize);
        let inner = NdArray::from_buffer(Buffer::C128(out), vec![n, width], Order::C).map_err(crate::to_py_err)?;
        Ok(Py::new(py, crate::PyArray { inner })?.into_any())
    } else {
        let contig = xarr.cast_to(DType::F64).to_contiguous();
        let xs = match contig.buffer() {
            Buffer::F64(v) => v.clone(),
            _ => unreachable!(),
        };
        let out = leg::legvander(&xs, deg as usize);
        let inner = NdArray::from_buffer(Buffer::F64(out), vec![n, width], Order::C).map_err(crate::to_py_err)?;
        Ok(Py::new(py, crate::PyArray { inner })?.into_any())
    }
}

// ─────────────────────────── legcompanion ────────────────────────────

#[pyfunction]
#[pyo3(name = "_leg_companion")]
pub fn leg_companion_py(py: Python<'_>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let s = coerce_series_strict(c)?;
    if s.len() < 2 {
        return Err(leg_err("Series must have maximum degree of at least 1."));
    }
    let n = s.len() - 1;
    match s {
        Series::R(v) => {
            let m = leg::legcompanion(&v);
            let inner = NdArray::from_buffer(Buffer::F64(m), vec![n, n], Order::C).map_err(crate::to_py_err)?;
            Ok(Py::new(py, crate::PyArray { inner })?.into_any())
        }
        Series::C(v) => {
            let m = leg::legcompanion(&v);
            let inner = NdArray::from_buffer(Buffer::C128(m), vec![n, n], Order::C).map_err(crate::to_py_err)?;
            Ok(Py::new(py, crate::PyArray { inner })?.into_any())
        }
    }
}

// ─────────────────────────── legtrim ────────────────────────────

#[pyfunction]
#[pyo3(name = "_leg_trim")]
#[pyo3(signature = (c, tol=0.0))]
pub fn leg_trim_py(py: Python<'_>, c: &Bound<'_, PyAny>, tol: f64) -> PyResult<Py<PyAny>> {
    if tol < 0.0 {
        return Err(leg_err("tol must be non-negative"));
    }
    match coerce_series_strict(c)? {
        Series::R(v) => mk_real(py, ionp_core::poly::trim_coef_f64(&v, tol)),
        Series::C(v) => mk_complex(py, ionp_core::poly::trim_coef_c128(&v, tol)),
    }
}

pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(leg_domain_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_zero_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_one_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_x_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_line_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_trim_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_val_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_add_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_sub_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_mulx_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_mul_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_div_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_der_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_int_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_vander_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(leg_companion_py, m)?)?;
    Ok(())
}
