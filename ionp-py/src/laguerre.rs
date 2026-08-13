//! PyO3 bindings for the numeric kernels of `numpy.polynomial.laguerre`.
//! Sibling to `ionp-py/src/legendre.rs` -- structural template only, see
//! `ionp_core::laguerre`'s module doc comment for the arithmetic-grouping
//! divergences from Legendre this binding layer must not silently erase.
//! Reuses `poly.rs`'s `Series`/coercion helpers exactly like `legendre.rs`
//! does: the bool-rejecting (`as_series`) vs bool-promoting (own dtype
//! handling) split is the same shape in `laguerre.py` as in `legendre.py`
//! (`lagder`/`lagint`/`lagval`/`lagvander` each do their own
//! `if c.dtype.char in '?bBhHiIlLqQpP': c = c.astype(np.double)`, while
//! `lagadd`/`lagsub`/`lagmulx`/`lagmul`/`lagdiv`/`lagfromroots`/
//! `lagcompanion`/`lagpow` route through `pu.as_series`).

use num_complex::Complex64 as C128;
use pyo3::exceptions::PyZeroDivisionError;
use pyo3::prelude::*;

use ionp_core::{laguerre as lag, Buffer, DType, NdArray, Order};

use crate::poly::{coerce_series, coerce_series_strict, Series};

fn lag_err(msg: impl Into<String>) -> PyErr {
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
// `lagdomain = np.array([0., 1.])` (float); `lagzero`/`lagone`/`lagx` are
// int arrays (`np.array([0])`, `np.array([1])`, `np.array([1, -1])`) --
// verified directly against `laguerre.py:202-231`.

#[pyfunction]
#[pyo3(name = "_lag_domain")]
pub fn lag_domain_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_real(py, vec![0.0, 1.0])
}
#[pyfunction]
#[pyo3(name = "_lag_zero")]
pub fn lag_zero_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![0])
}
#[pyfunction]
#[pyo3(name = "_lag_one")]
pub fn lag_one_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![1])
}
#[pyfunction]
#[pyo3(name = "_lag_x")]
pub fn lag_x_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![1, -1])
}

// ─────────────────────────── lagline ────────────────────────────
// `[off + scl, -scl]` -- a genuinely different affine map from `legline`'s
// `[off, scl]`, not a relabeled constant. Verified against
// `laguerre.py:249-251`.

#[pyfunction]
#[pyo3(name = "_lag_line")]
pub fn lag_line_py(py: Python<'_>, off: f64, scl: f64) -> PyResult<Py<PyAny>> {
    if scl != 0.0 {
        mk_real(py, vec![off + scl, -scl])
    } else {
        mk_real(py, vec![off])
    }
}

// ─────────────────────────── lagval ────────────────────────────

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
#[pyo3(name = "_lag_val")]
pub fn lag_val_py(py: Python<'_>, x: &Bound<'_, PyAny>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Same empty-`c` `IndexError` shape as `leg_val_py` -- `lagval` also
    // indexes `c[-2]`/`c[-1]` directly for its `len(c) >= 3` branch, and an
    // empty array hits that same `IndexError` before any arithmetic.
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
        let out = lag::lag_eval(&cv, &xs);
        return mk_shaped_real(py, out, out_shape);
    }
    let contig = xarr.cast_to(DType::C128).to_contiguous();
    let xs = match contig.buffer() {
        Buffer::C128(v) => v.clone(),
        _ => unreachable!(),
    };
    // Real coefficients evaluated at complex points: route through the
    // dedicated hybrid-typed kernel (`lag_eval_real_coef_complex_x`), NOT
    // the generic `lag_eval::<C128>` after upcasting `cv` to complex --
    // upcasting loses numpy's own "real until touched by x" scalar-typing
    // behaviour partway through the Clenshaw recursion, which is a real,
    // measured bit-exactness gap (see that function's doc comment in
    // `ionp-core/src/laguerre.rs` for the full mechanism and evidence).
    if !cs.is_complex() {
        let cv = match cs {
            Series::R(v) => v,
            _ => unreachable!(),
        };
        let out = lag::lag_eval_real_coef_complex_x(&cv, &xs);
        return mk_shaped_complex(py, out, out_shape);
    }
    let cv = cs.to_complex();
    let out = lag::lag_eval(&cv, &xs);
    mk_shaped_complex(py, out, out_shape)
}

// ─────────────────────── add / sub / mulx / mul / div ────────────────────────

#[pyfunction]
#[pyo3(name = "_lag_add")]
pub fn lag_add_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, lag::lagadd_trim(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, lag::lagadd_trim(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_lag_sub")]
pub fn lag_sub_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, lag::lagsub_trim(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, lag::lagsub_trim(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_lag_mulx")]
pub fn lag_mulx_py(py: Python<'_>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    match coerce_series_strict(c)? {
        Series::R(v) => mk_real(py, lag::lagmulx(&v)),
        Series::C(v) => mk_complex(py, lag::lagmulx(&v)),
    }
}

#[pyfunction]
#[pyo3(name = "_lag_mul")]
pub fn lag_mul_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, lag::lagmul(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, lag::lagmul(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_lag_div")]
pub fn lag_div_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => match lag::lagdiv(&a, &b) {
            Ok((q, r)) => Ok((mk_real(py, q)?, mk_real(py, r)?)),
            Err(()) => Err(PyZeroDivisionError::new_err(())),
        },
        (Series::C(a), Series::C(b)) => match lag::lagdiv(&a, &b) {
            Ok((q, r)) => Ok((mk_complex(py, q)?, mk_complex(py, r)?)),
            Err(()) => Err(PyZeroDivisionError::new_err(())),
        },
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

// ─────────────────────────── lagder / lagint ────────────────────────────

#[pyfunction]
#[pyo3(name = "_lag_der")]
pub fn lag_der_py(py: Python<'_>, c: &Bound<'_, PyAny>, m: i64, scl: f64) -> PyResult<Py<PyAny>> {
    if m < 0 {
        return Err(lag_err("The order of derivation must be non-negative"));
    }
    match coerce_series(c)? {
        Series::R(v) => mk_real(py, lag::lagder(&v, m as usize, scl)),
        Series::C(v) => mk_complex(py, lag::lagder(&v, m as usize, C128::new(scl, 0.0))),
    }
}

/// `k` arrives as `(re, im)` pairs, not `Vec<f64>` -- an earlier version
/// of this binding (and the `anionpy/polynomial/laguerre.py` wrapper
/// feeding it) took `Vec<f64>` and blind-`float()`-cast every element of
/// `k` in Python, silently discarding the imaginary part whenever a
/// complex integration constant was passed alongside a complex `c`. Real
/// numpy's `lagint` (`laguerre.py:756-789`) allocates its working array
/// with `dtype=c.dtype`, so a complex `c` accepts a genuinely complex
/// `k[i]` (`tmp[0] += k[i] - lagval(lbnd, tmp)`, ordinary complex
/// addition, no special-casing). Caught by
/// `/private/tmp/lag_bitexact_sweep.py`'s out-of-corpus random sweep
/// (mandated step 3), NOT by the differential corpus itself (which never
/// exercised complex `c` with `m > 0` and a nonzero, genuinely-complex
/// `k`). Passing `(re, im)` pairs keeps `anionpy/polynomial/laguerre.py`
/// as pure marshaling (splitting `complex(v)` into its two float fields
/// is not arithmetic), while the actual real-vs-complex promotion
/// decision -- and all arithmetic -- stays here / in `ionp-core`.
#[pyfunction]
#[pyo3(name = "_lag_int")]
pub fn lag_int_py(
    py: Python<'_>,
    c: &Bound<'_, PyAny>,
    m: i64,
    k: Vec<(f64, f64)>,
    lbnd: f64,
    scl: f64,
) -> PyResult<Py<PyAny>> {
    if m < 0 {
        return Err(lag_err("The order of integration must be non-negative"));
    }
    if k.len() > m as usize {
        return Err(lag_err("Too many integration constants"));
    }
    match coerce_series(c)? {
        Series::R(v) => {
            let mut kk: Vec<f64> = Vec::with_capacity(m as usize);
            for &(re, im) in &k {
                if im != 0.0 {
                    return Err(lag_err(
                        "lagint: a genuinely complex integration constant `k` \
                         requires a complex-valued `c` (matches real numpy's \
                         in-place same-kind casting rule for `tmp[0] += k[i]`)",
                    ));
                }
                kk.push(re);
            }
            kk.resize(m as usize, 0.0);
            mk_real(py, lag::lagint(&v, m as usize, &kk, lbnd, scl))
        }
        Series::C(v) => {
            let mut kkc: Vec<C128> = k.iter().map(|&(re, im)| C128::new(re, im)).collect();
            kkc.resize(m as usize, C128::new(0.0, 0.0));
            mk_complex(
                py,
                lag::lagint(&v, m as usize, &kkc, C128::new(lbnd, 0.0), C128::new(scl, 0.0)),
            )
        }
    }
}

// ─────────────────────────── lagvander ────────────────────────────

#[pyfunction]
#[pyo3(name = "_lag_vander")]
pub fn lag_vander_py(py: Python<'_>, x: &Bound<'_, PyAny>, deg: i64) -> PyResult<Py<PyAny>> {
    if deg < 0 {
        return Err(lag_err("deg must be non-negative"));
    }
    let xarr = ingest(x)?;
    if xarr.ndim() > 1 {
        return Err(lag_err(format!(
            "lagvander: expected a 1-d array of sample points, got {} dimensions \
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
        let out = lag::lagvander(&xs, deg as usize);
        let inner = NdArray::from_buffer(Buffer::C128(out), vec![n, width], Order::C).map_err(crate::to_py_err)?;
        Ok(Py::new(py, crate::PyArray { inner })?.into_any())
    } else {
        let contig = xarr.cast_to(DType::F64).to_contiguous();
        let xs = match contig.buffer() {
            Buffer::F64(v) => v.clone(),
            _ => unreachable!(),
        };
        let out = lag::lagvander(&xs, deg as usize);
        let inner = NdArray::from_buffer(Buffer::F64(out), vec![n, width], Order::C).map_err(crate::to_py_err)?;
        Ok(Py::new(py, crate::PyArray { inner })?.into_any())
    }
}

// ─────────────────────────── lagcompanion ────────────────────────────

#[pyfunction]
#[pyo3(name = "_lag_companion")]
pub fn lag_companion_py(py: Python<'_>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let s = coerce_series_strict(c)?;
    if s.len() < 2 {
        return Err(lag_err("Series must have maximum degree of at least 1."));
    }
    let n = s.len() - 1;
    match s {
        Series::R(v) => {
            let m = lag::lagcompanion(&v);
            let inner = NdArray::from_buffer(Buffer::F64(m), vec![n, n], Order::C).map_err(crate::to_py_err)?;
            Ok(Py::new(py, crate::PyArray { inner })?.into_any())
        }
        Series::C(v) => {
            let m = lag::lagcompanion(&v);
            let inner = NdArray::from_buffer(Buffer::C128(m), vec![n, n], Order::C).map_err(crate::to_py_err)?;
            Ok(Py::new(py, crate::PyArray { inner })?.into_any())
        }
    }
}

// ─────────────────────────── lagtrim ────────────────────────────

#[pyfunction]
#[pyo3(name = "_lag_trim")]
#[pyo3(signature = (c, tol=0.0))]
pub fn lag_trim_py(py: Python<'_>, c: &Bound<'_, PyAny>, tol: f64) -> PyResult<Py<PyAny>> {
    if tol < 0.0 {
        return Err(lag_err("tol must be non-negative"));
    }
    match coerce_series_strict(c)? {
        Series::R(v) => mk_real(py, ionp_core::poly::trim_coef_f64(&v, tol)),
        Series::C(v) => mk_complex(py, ionp_core::poly::trim_coef_c128(&v, tol)),
    }
}

pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(lag_domain_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_zero_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_one_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_x_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_line_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_trim_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_val_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_add_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_sub_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_mulx_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_mul_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_div_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_der_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_int_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_vander_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lag_companion_py, m)?)?;
    Ok(())
}
