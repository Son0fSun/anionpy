//! PyO3 bindings for the numeric kernels of `numpy.polynomial.chebyshev`
//! (first-kind Chebyshev series). Structural template borrowed from
//! `ionp-py/src/hermite_e.rs` -- see `ionp_core::chebyshev`'s module doc
//! comment for the z-series-vs-direct-recurrence split this binding layer
//! must not silently erase. `cheb2poly`/`poly2cheb`/`chebroots`/
//! `chebfit`/`chebgauss`/`chebweight`/`chebfromroots`/`chebpts1`/
//! `chebpts2`/`chebinterpolate` are NOT bound here -- per this task's
//! architecture rule, their numpy sources are themselves bounded-degree
//! Python orchestration loops built on top of the primitives bound below
//! (`chebmul`, `chebline`, `chebvander`, ...), so they live in
//! `anionpy/polynomial/chebyshev.py` directly.

use num_complex::Complex64 as C128;
use pyo3::exceptions::PyZeroDivisionError;
use pyo3::prelude::*;

use ionp_core::{chebyshev as cheb, Buffer, DType, NdArray, Order};

use crate::poly::{coerce_series, coerce_series_strict, Series};

fn cheb_err(msg: impl Into<String>) -> PyErr {
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
// `chebdomain = np.array([-1., 1.])` (float); `chebzero = np.array([0])`,
// `chebone = np.array([1])`, `chebx = np.array([0, 1])` -- verified
// directly against `chebyshev.py`.

#[pyfunction]
#[pyo3(name = "_cheb_domain")]
pub fn cheb_domain_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_real(py, vec![-1.0, 1.0])
}
#[pyfunction]
#[pyo3(name = "_cheb_zero")]
pub fn cheb_zero_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![0])
}
#[pyfunction]
#[pyo3(name = "_cheb_one")]
pub fn cheb_one_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![1])
}
#[pyfunction]
#[pyo3(name = "_cheb_x")]
pub fn cheb_x_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![0, 1])
}

// ─────────────────────────── chebline ────────────────────────────

#[pyfunction]
#[pyo3(name = "_cheb_line")]
pub fn cheb_line_py(py: Python<'_>, off: f64, scl: f64) -> PyResult<Py<PyAny>> {
    if scl != 0.0 {
        mk_real(py, vec![off, scl])
    } else {
        mk_real(py, vec![off])
    }
}

// ─────────────────────────── chebval ────────────────────────────

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
#[pyo3(name = "_cheb_val")]
pub fn cheb_val_py(py: Python<'_>, x: &Bound<'_, PyAny>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // `chebval` indexes `c[-2]`/`c[-1]` directly in its `len(c) >= 3`
    // branch -- an empty `c` hits the same `IndexError` before any
    // arithmetic, same shape as the sibling bases' `*val`.
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
        let out = cheb::cheb_eval(&cv, &xs);
        return mk_shaped_real(py, out, out_shape);
    }
    let contig = xarr.cast_to(DType::C128).to_contiguous();
    let xs = match contig.buffer() {
        Buffer::C128(v) => v.clone(),
        _ => unreachable!(),
    };
    let cv = cs.to_complex();
    let out = cheb::cheb_eval(&cv, &xs);
    mk_shaped_complex(py, out, out_shape)
}

// ─────────────────── add / sub / mulx / mul / div / pow ───────────────────

#[pyfunction]
#[pyo3(name = "_cheb_add")]
pub fn cheb_add_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, cheb::chebadd_trim(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, cheb::chebadd_trim(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_cheb_sub")]
pub fn cheb_sub_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, cheb::chebsub_trim(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, cheb::chebsub_trim(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_cheb_mulx")]
pub fn cheb_mulx_py(py: Python<'_>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    match coerce_series_strict(c)? {
        Series::R(v) => mk_real(py, cheb::chebmulx(&v)),
        Series::C(v) => mk_complex(py, cheb::chebmulx(&v)),
    }
}

#[pyfunction]
#[pyo3(name = "_cheb_mul")]
pub fn cheb_mul_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, cheb::chebmul(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, cheb::chebmul(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_cheb_div")]
pub fn cheb_div_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => match cheb::chebdiv(&a, &b) {
            Ok((q, r)) => Ok((mk_real(py, q)?, mk_real(py, r)?)),
            Err(()) => Err(PyZeroDivisionError::new_err(())),
        },
        (Series::C(a), Series::C(b)) => match cheb::chebdiv(&a, &b) {
            Ok((q, r)) => Ok((mk_complex(py, q)?, mk_complex(py, r)?)),
            Err(()) => Err(PyZeroDivisionError::new_err(())),
        },
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

/// `chebpow(c, pow, maxpower=16)`: the `maxpower` bound check itself is
/// tiny/degree-bound Python-orchestration-shaped, so it's enforced in the
/// Python shim (`anionpy/polynomial/chebyshev.py`) before this is called
/// -- this binding itself just runs the (degree-bounded) z-series loop
/// entirely in Rust, per `ionp_core::chebyshev::chebpow`'s doc comment.
#[pyfunction]
#[pyo3(name = "_cheb_pow")]
pub fn cheb_pow_py(py: Python<'_>, c: &Bound<'_, PyAny>, power: u64) -> PyResult<Py<PyAny>> {
    match coerce_series_strict(c)? {
        Series::R(v) => mk_real(py, cheb::chebpow(&v, power as usize)),
        Series::C(v) => mk_complex(py, cheb::chebpow(&v, power as usize)),
    }
}

// ─────────────────────────── chebder / chebint ────────────────────────────

#[pyfunction]
#[pyo3(name = "_cheb_der")]
pub fn cheb_der_py(py: Python<'_>, c: &Bound<'_, PyAny>, m: i64, scl: f64) -> PyResult<Py<PyAny>> {
    if m < 0 {
        return Err(cheb_err("The order of derivation must be non-negative"));
    }
    match coerce_series(c)? {
        Series::R(v) => mk_real(py, cheb::chebder(&v, m as usize, scl)),
        Series::C(v) => mk_complex(py, cheb::chebder(&v, m as usize, C128::new(scl, 0.0))),
    }
}

/// `k` arrives as `(re, im)` pairs, not `Vec<f64>` -- same rationale as
/// `ionp-py/src/hermite_e.rs`'s `herme_int_py`: a blind `float()` cast
/// would silently truncate a genuinely complex integration constant.
#[pyfunction]
#[pyo3(name = "_cheb_int")]
pub fn cheb_int_py(
    py: Python<'_>,
    c: &Bound<'_, PyAny>,
    m: i64,
    k: Vec<(f64, f64)>,
    lbnd: f64,
    scl: f64,
) -> PyResult<Py<PyAny>> {
    if m < 0 {
        return Err(cheb_err("The order of integration must be non-negative"));
    }
    if k.len() > m as usize {
        return Err(cheb_err("Too many integration constants"));
    }
    match coerce_series(c)? {
        Series::R(v) => {
            let mut kk: Vec<f64> = Vec::with_capacity(m as usize);
            for &(re, im) in &k {
                if im != 0.0 {
                    return Err(cheb_err(
                        "chebint: a genuinely complex integration constant `k` \
                         requires a complex-valued `c` (matches real numpy's \
                         in-place same-kind casting rule for `tmp[0] += k[i]`)",
                    ));
                }
                kk.push(re);
            }
            kk.resize(m as usize, 0.0);
            mk_real(py, cheb::chebint(&v, m as usize, &kk, lbnd, scl))
        }
        Series::C(v) => {
            let mut kkc: Vec<C128> = k.iter().map(|&(re, im)| C128::new(re, im)).collect();
            kkc.resize(m as usize, C128::new(0.0, 0.0));
            mk_complex(
                py,
                cheb::chebint(&v, m as usize, &kkc, C128::new(lbnd, 0.0), C128::new(scl, 0.0)),
            )
        }
    }
}

// ─────────────────────────── chebvander ────────────────────────────

#[pyfunction]
#[pyo3(name = "_cheb_vander")]
pub fn cheb_vander_py(py: Python<'_>, x: &Bound<'_, PyAny>, deg: i64) -> PyResult<Py<PyAny>> {
    if deg < 0 {
        return Err(cheb_err("deg must be non-negative"));
    }
    let xarr = ingest(x)?;
    if xarr.ndim() > 1 {
        return Err(cheb_err(format!(
            "chebvander: expected a 1-d array of sample points, got {} dimensions \
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
        let out = cheb::chebvander(&xs, deg as usize);
        let inner = NdArray::from_buffer(Buffer::C128(out), vec![n, width], Order::C).map_err(crate::to_py_err)?;
        Ok(Py::new(py, crate::PyArray { inner })?.into_any())
    } else {
        let contig = xarr.cast_to(DType::F64).to_contiguous();
        let xs = match contig.buffer() {
            Buffer::F64(v) => v.clone(),
            _ => unreachable!(),
        };
        let out = cheb::chebvander(&xs, deg as usize);
        let inner = NdArray::from_buffer(Buffer::F64(out), vec![n, width], Order::C).map_err(crate::to_py_err)?;
        Ok(Py::new(py, crate::PyArray { inner })?.into_any())
    }
}

// ─────────────────────────── chebcompanion ────────────────────────────

#[pyfunction]
#[pyo3(name = "_cheb_companion")]
pub fn cheb_companion_py(py: Python<'_>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let s = coerce_series_strict(c)?;
    if s.len() < 2 {
        return Err(cheb_err("Series must have maximum degree of at least 1."));
    }
    let n = s.len() - 1;
    match s {
        Series::R(v) => {
            let m = cheb::chebcompanion(&v);
            let inner = NdArray::from_buffer(Buffer::F64(m), vec![n, n], Order::C).map_err(crate::to_py_err)?;
            Ok(Py::new(py, crate::PyArray { inner })?.into_any())
        }
        Series::C(v) => {
            let m = cheb::chebcompanion(&v);
            let inner = NdArray::from_buffer(Buffer::C128(m), vec![n, n], Order::C).map_err(crate::to_py_err)?;
            Ok(Py::new(py, crate::PyArray { inner })?.into_any())
        }
    }
}

// ─────────────────────────── chebtrim ────────────────────────────

#[pyfunction]
#[pyo3(name = "_cheb_trim")]
#[pyo3(signature = (c, tol=0.0))]
pub fn cheb_trim_py(py: Python<'_>, c: &Bound<'_, PyAny>, tol: f64) -> PyResult<Py<PyAny>> {
    if tol < 0.0 {
        return Err(cheb_err("tol must be non-negative"));
    }
    match coerce_series_strict(c)? {
        Series::R(v) => mk_real(py, ionp_core::poly::trim_coef_f64(&v, tol)),
        Series::C(v) => mk_complex(py, ionp_core::poly::trim_coef_c128(&v, tol)),
    }
}

pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(cheb_domain_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_zero_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_one_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_x_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_line_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_trim_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_val_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_add_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_sub_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_mulx_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_mul_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_div_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_pow_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_der_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_int_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_vander_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cheb_companion_py, m)?)?;
    Ok(())
}
