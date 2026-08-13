//! PyO3 bindings for the numeric kernels of `numpy.polynomial.hermite`
//! (physicists' Hermite). Structural template borrowed from
//! `ionp-py/src/laguerre.rs` -- see `ionp_core::hermite`'s module doc
//! comment for the arithmetic-grouping divergences this binding layer
//! must not silently erase. `herm2poly`/`poly2herm`/`hermroots`/
//! `hermfit`/`hermgauss`/`hermweight`/`hermfromroots`/`hermpow` are NOT
//! bound here -- per this task's architecture rule, their numpy sources
//! are themselves bounded-degree Python orchestration loops built on top
//! of the primitives bound below (`hermmul`, `hermline`-equivalent,
//! `hermcompanion`, ...), so they live in
//! `anionpy/polynomial/hermite.py` directly, exactly like
//! `lagroots`/`lagfit`/`laggauss`/`lag2poly`/`poly2lag` do in
//! `laguerre.py` rather than here.

use num_complex::Complex64 as C128;
use pyo3::exceptions::PyZeroDivisionError;
use pyo3::prelude::*;

use ionp_core::{hermite as herm, Buffer, DType, NdArray, Order};

use crate::poly::{coerce_series, coerce_series_strict, Series};

fn herm_err(msg: impl Into<String>) -> PyErr {
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
// `hermdomain = np.array([-1., 1.])` (float); `hermzero = np.array([0])`,
// `hermone = np.array([1])`, `hermx = np.array([0, 0.5])` -- verified
// directly against `hermite.py:203-232`. Note `hermx` is NOT an int array
// (unlike `lagx`/`legx`): `0.5` forces float dtype.

#[pyfunction]
#[pyo3(name = "_herm_domain")]
pub fn herm_domain_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_real(py, vec![-1.0, 1.0])
}
#[pyfunction]
#[pyo3(name = "_herm_zero")]
pub fn herm_zero_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![0])
}
#[pyfunction]
#[pyo3(name = "_herm_one")]
pub fn herm_one_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![1])
}
#[pyfunction]
#[pyo3(name = "_herm_x")]
pub fn herm_x_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_real(py, vec![0.0, 0.5])
}

// ─────────────────────────── hermline ────────────────────────────
// `[off, scl/2]` -- verified against `hermite.py:257-259`.

#[pyfunction]
#[pyo3(name = "_herm_line")]
pub fn herm_line_py(py: Python<'_>, off: f64, scl: f64) -> PyResult<Py<PyAny>> {
    if scl != 0.0 {
        mk_real(py, vec![off, scl / 2.0])
    } else {
        mk_real(py, vec![off])
    }
}

// ─────────────────────────── hermval ────────────────────────────

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
#[pyo3(name = "_herm_val")]
pub fn herm_val_py(py: Python<'_>, x: &Bound<'_, PyAny>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // `hermval` also indexes `c[-2]`/`c[-1]` directly in its `len(c) >= 3`
    // branch (same shape as `lagval`/`legval`) -- an empty `c` hits the
    // same `IndexError` before any arithmetic.
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
        let out = herm::herm_eval(&cv, &xs);
        return mk_shaped_real(py, out, out_shape);
    }
    let contig = xarr.cast_to(DType::C128).to_contiguous();
    let xs = match contig.buffer() {
        Buffer::C128(v) => v.clone(),
        _ => unreachable!(),
    };
    let cv = cs.to_complex();
    let out = herm::herm_eval(&cv, &xs);
    mk_shaped_complex(py, out, out_shape)
}

// ─────────────────────── add / sub / mulx / mul / div ────────────────────────

#[pyfunction]
#[pyo3(name = "_herm_add")]
pub fn herm_add_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, herm::hermadd_trim(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, herm::hermadd_trim(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_herm_sub")]
pub fn herm_sub_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, herm::hermsub_trim(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, herm::hermsub_trim(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_herm_mulx")]
pub fn herm_mulx_py(py: Python<'_>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    match coerce_series_strict(c)? {
        Series::R(v) => mk_real(py, herm::hermmulx(&v)),
        Series::C(v) => mk_complex(py, herm::hermmulx(&v)),
    }
}

#[pyfunction]
#[pyo3(name = "_herm_mul")]
pub fn herm_mul_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, herm::hermmul(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, herm::hermmul(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_herm_div")]
pub fn herm_div_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => match herm::hermdiv(&a, &b) {
            Ok((q, r)) => Ok((mk_real(py, q)?, mk_real(py, r)?)),
            Err(()) => Err(PyZeroDivisionError::new_err(())),
        },
        (Series::C(a), Series::C(b)) => match herm::hermdiv(&a, &b) {
            Ok((q, r)) => Ok((mk_complex(py, q)?, mk_complex(py, r)?)),
            Err(()) => Err(PyZeroDivisionError::new_err(())),
        },
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

// ─────────────────────────── hermder / hermint ────────────────────────────

#[pyfunction]
#[pyo3(name = "_herm_der")]
pub fn herm_der_py(py: Python<'_>, c: &Bound<'_, PyAny>, m: i64, scl: f64) -> PyResult<Py<PyAny>> {
    if m < 0 {
        return Err(herm_err("The order of derivation must be non-negative"));
    }
    match coerce_series(c)? {
        Series::R(v) => mk_real(py, herm::hermder(&v, m as usize, scl)),
        Series::C(v) => mk_complex(py, herm::hermder(&v, m as usize, C128::new(scl, 0.0))),
    }
}

/// `k` arrives as `(re, im)` pairs, not `Vec<f64>` -- same rationale as
/// `ionp-py/src/laguerre.rs`'s `lag_int_py`: a blind `float()` cast would
/// silently truncate a genuinely complex integration constant. See that
/// function's doc comment for the full precedent (caught by an
/// out-of-corpus sweep, not the differential corpus itself).
#[pyfunction]
#[pyo3(name = "_herm_int")]
pub fn herm_int_py(
    py: Python<'_>,
    c: &Bound<'_, PyAny>,
    m: i64,
    k: Vec<(f64, f64)>,
    lbnd: f64,
    scl: f64,
) -> PyResult<Py<PyAny>> {
    if m < 0 {
        return Err(herm_err("The order of integration must be non-negative"));
    }
    if k.len() > m as usize {
        return Err(herm_err("Too many integration constants"));
    }
    match coerce_series(c)? {
        Series::R(v) => {
            let mut kk: Vec<f64> = Vec::with_capacity(m as usize);
            for &(re, im) in &k {
                if im != 0.0 {
                    return Err(herm_err(
                        "hermint: a genuinely complex integration constant `k` \
                         requires a complex-valued `c` (matches real numpy's \
                         in-place same-kind casting rule for `tmp[0] += k[i]`)",
                    ));
                }
                kk.push(re);
            }
            kk.resize(m as usize, 0.0);
            mk_real(py, herm::hermint(&v, m as usize, &kk, lbnd, scl))
        }
        Series::C(v) => {
            let mut kkc: Vec<C128> = k.iter().map(|&(re, im)| C128::new(re, im)).collect();
            kkc.resize(m as usize, C128::new(0.0, 0.0));
            mk_complex(
                py,
                herm::hermint(&v, m as usize, &kkc, C128::new(lbnd, 0.0), C128::new(scl, 0.0)),
            )
        }
    }
}

// ─────────────────────────── hermvander ────────────────────────────

#[pyfunction]
#[pyo3(name = "_herm_vander")]
pub fn herm_vander_py(py: Python<'_>, x: &Bound<'_, PyAny>, deg: i64) -> PyResult<Py<PyAny>> {
    if deg < 0 {
        return Err(herm_err("deg must be non-negative"));
    }
    let xarr = ingest(x)?;
    if xarr.ndim() > 1 {
        return Err(herm_err(format!(
            "hermvander: expected a 1-d array of sample points, got {} dimensions \
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
        let out = herm::hermvander(&xs, deg as usize);
        let inner = NdArray::from_buffer(Buffer::C128(out), vec![n, width], Order::C).map_err(crate::to_py_err)?;
        Ok(Py::new(py, crate::PyArray { inner })?.into_any())
    } else {
        let contig = xarr.cast_to(DType::F64).to_contiguous();
        let xs = match contig.buffer() {
            Buffer::F64(v) => v.clone(),
            _ => unreachable!(),
        };
        let out = herm::hermvander(&xs, deg as usize);
        let inner = NdArray::from_buffer(Buffer::F64(out), vec![n, width], Order::C).map_err(crate::to_py_err)?;
        Ok(Py::new(py, crate::PyArray { inner })?.into_any())
    }
}

// ─────────────────────────── hermcompanion ────────────────────────────

#[pyfunction]
#[pyo3(name = "_herm_companion")]
pub fn herm_companion_py(py: Python<'_>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let s = coerce_series_strict(c)?;
    if s.len() < 2 {
        return Err(herm_err("Series must have maximum degree of at least 1."));
    }
    let n = s.len() - 1;
    match s {
        Series::R(v) => {
            let m = herm::hermcompanion(&v);
            let inner = NdArray::from_buffer(Buffer::F64(m), vec![n, n], Order::C).map_err(crate::to_py_err)?;
            Ok(Py::new(py, crate::PyArray { inner })?.into_any())
        }
        Series::C(v) => {
            let m = herm::hermcompanion(&v);
            let inner = NdArray::from_buffer(Buffer::C128(m), vec![n, n], Order::C).map_err(crate::to_py_err)?;
            Ok(Py::new(py, crate::PyArray { inner })?.into_any())
        }
    }
}

// ─────────────────────────── hermtrim ────────────────────────────

#[pyfunction]
#[pyo3(name = "_herm_trim")]
#[pyo3(signature = (c, tol=0.0))]
pub fn herm_trim_py(py: Python<'_>, c: &Bound<'_, PyAny>, tol: f64) -> PyResult<Py<PyAny>> {
    if tol < 0.0 {
        return Err(herm_err("tol must be non-negative"));
    }
    match coerce_series_strict(c)? {
        Series::R(v) => mk_real(py, ionp_core::poly::trim_coef_f64(&v, tol)),
        Series::C(v) => mk_complex(py, ionp_core::poly::trim_coef_c128(&v, tol)),
    }
}

pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(herm_domain_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_zero_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_one_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_x_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_line_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_trim_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_val_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_add_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_sub_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_mulx_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_mul_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_div_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_der_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_int_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_vander_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(herm_companion_py, m)?)?;
    Ok(())
}
