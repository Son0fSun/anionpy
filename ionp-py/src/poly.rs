//! PyO3 bindings for the numeric kernels of
//! `numpy.polynomial.polynomial` (power-series basis). All arithmetic
//! lives in `ionp_core::poly`; this file only marshals Python arguments
//! (anionpy arrays, numpy arrays, or bare Python lists/scalars) into
//! `Vec<f64>`/`Vec<C128>` and marshals results back into `PyArray`.
//!
//! Scope (deliberate, see this task's report): only the 1-D coefficient-
//! array call forms are bound here. `polyval`'s `tensor=` broadcasting
//! over multi-dimensional `x` IS supported (the sample points, not the
//! coefficients, are what varies in shape); the N-D composition helpers
//! `polyval2d`/`polyval3d`/`polyvalnd`/`polygrid2d`/`polygrid3d`/
//! `polyvander2d`/`polyvander3d` are NOT bound in this pass (time-boxed
//! out, not a measured decline -- see the task report).
//!
//! Deliberately does NOT reuse `linalg.rs`'s private extraction helpers
//! (`as_real1`/`as_complex1`/...): this file owns its own small,
//! self-contained copies so it never needs to touch a file another
//! concurrent agent may also be editing.

use num_complex::Complex64 as C128;
use pyo3::exceptions::{PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;

use ionp_core::{poly, Buffer, DType, NdArray, Order};

use crate::PyArray;

fn poly_err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// One coefficient/sample-point series, already promoted the way numpy's
/// `polyutils.as_series` promotes: any complex input yields `C128`,
/// everything else (bool/int/float) yields `f64`. Mirrors numpy's own
/// `np.common_type` rule -- verified live: `polyadd([1,2],[1.0,2.0])` and
/// `polyadd([True,False],[1,2])` both come back `float64` in real numpy.
pub(crate) enum Series {
    R(Vec<f64>),
    C(Vec<C128>),
}

impl Series {
    // `pub(crate)`: shared with `ionp-py/src/legendre.rs`, which uses the
    // identical `as_series`/own-dtype-path split (see that file's module
    // doc comment) and would otherwise need to duplicate this enum's
    // accessor methods verbatim.
    pub(crate) fn len(&self) -> usize {
        match self {
            Series::R(v) => v.len(),
            Series::C(v) => v.len(),
        }
    }
    pub(crate) fn is_complex(&self) -> bool {
        matches!(self, Series::C(_))
    }
    pub(crate) fn to_complex(self) -> Vec<C128> {
        match self {
            Series::R(v) => v.into_iter().map(|x| C128::new(x, 0.0)).collect(),
            Series::C(v) => v,
        }
    }
}

/// Ingest any 1-D array-like (anionpy array, numpy array, or Python
/// list/tuple/scalar) into an `NdArray`, WITHOUT ever calling real numpy
/// to do so -- `crate::array_impl` is anionpy's own array-construction
/// entry point (used by `anionpy.array()` itself), already numpy-free.
fn ingest(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(pyref.inner.clone());
    }
    crate::array_impl(obj, None)
}

/// Pull a 1-D coefficient/point series out of `obj`, promoted per
/// `Series`'s doc comment above. Empty input is rejected the way numpy's
/// `as_series` rejects it (`ValueError: coefficient array is empty`).
///
/// `reject_bool`: real numpy's `polynomial.py` uses TWO different dtype
/// paths, verified by direct probe (2026-08-07, see
/// tests/differential/polynomial_cases.py's module comment history):
///   - `polyder`/`polyint`/`polyval`/`polyvalfromroots`/`polyvander` do
///     their OWN `c = np.array(c, ndmin=1); if c.dtype.char in
///     '?bBhHiIlLqQpP': c = c + 0.0` -- bool is silently promoted to
///     float64, never rejected. `reject_bool=false` for these.
///   - everything that goes through `polyutils.as_series` (`polytrim`,
///     `polyadd`, `polysub`, `polymulx`, `polymul`, `polydiv`,
///     `polyfromroots`, `polycompanion`, and transitively `polypow`/
///     `polyroots`) calls `np.common_type(*arrays)`, which raises
///     `TypeError` outright for a bool-dtype array (bool is not an
///     "inexact" or even numeric-enough type for `common_type`);
///     `as_series` catches that and re-raises
///     `ValueError("Coefficient arrays have no common type")` (the
///     plural wording is used even for a single-array call -- verified
///     directly: `np.polynomial.polynomial.polytrim([True, False])`
///     raises the same plural message). `reject_bool=true` for these.
pub(crate) fn coerce_series_ex(obj: &Bound<'_, PyAny>, reject_bool: bool) -> PyResult<Series> {
    let arr = ingest(obj)?;
    // Match numpy's `as_series` exactly: empty is checked BEFORE the
    // ndim check (verified against real numpy 2.5.1: a 0-d empty-typed
    // array is unreachable via any real construction, but a genuinely
    // empty 2-D array like `np.zeros((0, 3))` raises "Coefficient array
    // is empty", not "...is not 1-d" -- `a.size == 0` is checked first
    // in `as_series`'s loop).
    if arr.shape().iter().product::<usize>() == 0 {
        return Err(poly_err("Coefficient array is empty"));
    }
    if arr.ndim() != 1 {
        return Err(poly_err("Coefficient array is not 1-d"));
    }
    if reject_bool && arr.dtype() == DType::Bool {
        return Err(poly_err("Coefficient arrays have no common type"));
    }
    if arr.dtype().is_complex() {
        let contig = arr.cast_to(DType::C128).to_contiguous();
        let data = match contig.buffer() {
            Buffer::C128(v) => v.clone(),
            _ => unreachable!("cast_to(C128) always yields Buffer::C128"),
        };
        Ok(Series::C(data))
    } else {
        let contig = arr.cast_to(DType::F64).to_contiguous();
        let data = match contig.buffer() {
            Buffer::F64(v) => v.clone(),
            _ => unreachable!("cast_to(F64) always yields Buffer::F64"),
        };
        Ok(Series::R(data))
    }
}

/// Non-rejecting series coercion: the `polyder`/`polyint`/`polyval`/
/// `polyvalfromroots`/`polyvander` path -- bool silently promoted to
/// float64. See `coerce_series_ex`'s doc comment for the full split.
pub(crate) fn coerce_series(obj: &Bound<'_, PyAny>) -> PyResult<Series> {
    coerce_series_ex(obj, false)
}

/// `as_series`-family series coercion: the `polytrim`/`polyadd`/`polysub`/
/// `polymulx`/`polymul`/`polydiv`/`polyfromroots`/`polycompanion` path --
/// bool is rejected with `ValueError("Coefficient arrays have no common
/// type")`. See `coerce_series_ex`'s doc comment for the full split.
pub(crate) fn coerce_series_strict(obj: &Bound<'_, PyAny>) -> PyResult<Series> {
    coerce_series_ex(obj, true)
}

/// `polyvalfromroots`-only coercion: NOT `as_series`-based (see that
/// function's call site for the full explanation) -- accepts an empty
/// roots array (no error), rejects only ndim != 1 (out of this build's
/// scope), and promotes bool/int/complex the same way `coerce_series`
/// does otherwise.
fn coerce_roots(obj: &Bound<'_, PyAny>) -> PyResult<Series> {
    let arr = ingest(obj)?;
    if arr.ndim() != 1 {
        return Err(poly_err("polyvalfromroots: roots array must be 1-d in this build"));
    }
    if arr.dtype().is_complex() {
        let contig = arr.cast_to(DType::C128).to_contiguous();
        let data = match contig.buffer() {
            Buffer::C128(v) => v.clone(),
            _ => unreachable!("cast_to(C128) always yields Buffer::C128"),
        };
        Ok(Series::C(data))
    } else {
        let contig = arr.cast_to(DType::F64).to_contiguous();
        let data = match contig.buffer() {
            Buffer::F64(v) => v.clone(),
            _ => unreachable!("cast_to(F64) always yields Buffer::F64"),
        };
        Ok(Series::R(data))
    }
}

/// Promote a pair of series to a common representation (real unless
/// EITHER is complex), matching `as_series([c1, c2])`'s joint promotion.
/// Always uses the bool-rejecting (`as_series`-family) coercion: every
/// caller of `coerce_pair` (`polyadd`/`polysub`/`polymul`/`polydiv`) is
/// itself an `as_series`-family function.
fn coerce_pair(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<(Series, Series)> {
    let sa = coerce_series_strict(a)?;
    let sb = coerce_series_strict(b)?;
    if sa.is_complex() || sb.is_complex() {
        Ok((Series::C(match sa {
            Series::R(v) => v.into_iter().map(|x| C128::new(x, 0.0)).collect(),
            Series::C(v) => v,
        }), Series::C(match sb {
            Series::R(v) => v.into_iter().map(|x| C128::new(x, 0.0)).collect(),
            Series::C(v) => v,
        })))
    } else {
        Ok((sa, sb))
    }
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
fn mk_series(py: Python<'_>, s: Series) -> PyResult<Py<PyAny>> {
    match s {
        Series::R(v) => mk_real(py, v),
        Series::C(v) => mk_complex(py, v),
    }
}

/// Real numpy builds `polyzero`/`polyone`/`polyx` as `np.array([0])`,
/// `np.array([1])`, `np.array([0, 1])` -- plain Python-int literals, so
/// they carry the platform default integer dtype (`int64` on this
/// machine), NOT float64. Verified directly (2026-08-07):
/// `np.polynomial.polynomial.polyone.dtype` is `int64`. `polydomain`
/// alone is `np.array([-1, 1.])` (note the literal float `1.`), which IS
/// float64 -- left on `mk_real` below, unchanged.
fn mk_int64(py: Python<'_>, data: Vec<i64>) -> PyResult<Py<PyAny>> {
    let n = data.len();
    let inner = NdArray::from_buffer(Buffer::I64(data), vec![n], Order::C).map_err(crate::to_py_err)?;
    Ok(Py::new(py, PyArray { inner })?.into_any())
}

// ─────────────────────────── constants ────────────────────────────

#[pyfunction]
#[pyo3(name = "_poly_domain")]
pub fn poly_domain_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_real(py, vec![-1.0, 1.0])
}
#[pyfunction]
#[pyo3(name = "_poly_zero")]
pub fn poly_zero_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![0])
}
#[pyfunction]
#[pyo3(name = "_poly_one")]
pub fn poly_one_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![1])
}
#[pyfunction]
#[pyo3(name = "_poly_x")]
pub fn poly_x_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    mk_int64(py, vec![0, 1])
}

// ─────────────────────────── polyline ────────────────────────────

#[pyfunction]
#[pyo3(name = "_poly_line")]
pub fn poly_line_py(py: Python<'_>, off: f64, scl: f64) -> PyResult<Py<PyAny>> {
    // numpy: `if scl != 0: return array([off, scl]); else: return array([off])`
    if scl != 0.0 {
        mk_real(py, vec![off, scl])
    } else {
        mk_real(py, vec![off])
    }
}

// ─────────────────────────── trimming ────────────────────────────

#[pyfunction]
#[pyo3(name = "_poly_trim")]
#[pyo3(signature = (c, tol=0.0))]
pub fn poly_trim_py(py: Python<'_>, c: &Bound<'_, PyAny>, tol: f64) -> PyResult<Py<PyAny>> {
    if tol < 0.0 {
        return Err(poly_err("tol must be non-negative"));
    }
    match coerce_series_strict(c)? {
        Series::R(v) => mk_real(py, poly::trim_coef_f64(&v, tol)),
        Series::C(v) => mk_complex(py, poly::trim_coef_c128(&v, tol)),
    }
}

// ─────────────────────────── polyval ────────────────────────────

#[pyfunction]
#[pyo3(name = "_poly_val")]
pub fn poly_val_py(py: Python<'_>, x: &Bound<'_, PyAny>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // `polyval` does NOT go through `as_series` in real numpy (verified
    // against numpy 2.5.1 source, `polynomial.py`'s `polyval`): it does
    // its own `c = np.array(c, ndmin=1); ... c0 = c[-1] + x*0`, so an
    // empty `c` raises a bare `IndexError` ("index -1 is out of bounds
    // for axis 0 with size 0"), not the `ValueError("Coefficient array
    // is empty")` every OTHER function in this file raises via
    // `as_series`/`coerce_series`. Matched here explicitly rather than
    // routing through `coerce_series`'s general empty-check.
    {
        let probe = ingest(c)?;
        if probe.shape().iter().product::<usize>() == 0 && probe.ndim() >= 1 {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "index -1 is out of bounds for axis 0 with size 0",
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
        let out = poly::horner_eval(&cv, &xs);
        return mk_shaped_real(py, out, out_shape);
    }
    // Either x or c (or both) is complex: promote everything to complex.
    let contig = xarr.cast_to(DType::C128).to_contiguous();
    let xs = match contig.buffer() {
        Buffer::C128(v) => v.clone(),
        _ => unreachable!(),
    };
    let cv = cs.to_complex();
    let out = poly::horner_eval(&cv, &xs);
    mk_shaped_complex(py, out, out_shape)
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

// ─────────────────────── polyvalfromroots ────────────────────────

#[pyfunction]
#[pyo3(name = "_poly_valfromroots")]
pub fn poly_valfromroots_py(py: Python<'_>, x: &Bound<'_, PyAny>, r: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // `polyvalfromroots` does NOT go through `as_series`/`coerce_series` at
    // all in real numpy: its own body is `r = np.array(r, ndmin=1); if
    // r.dtype.char in bool/int-chars: r = r.astype(np.double)` -- no empty
    // check, no ndim!=1 check, no complex-vs-real joint promotion beyond
    // the bool/int cast. An empty `r` is valid and meaningful (`np.prod`
    // over zero factors is the empty product, 1.0 -- verified directly:
    // `P.polyvalfromroots(2.0, [])` -> `1.0`,
    // `P.polyvalfromroots([1,2,3], [])` -> `array([1., 1., 1.])`).
    // `coerce_roots` below mirrors that: promoting, non-rejecting on
    // empty, ndim!=1 still rejected (multi-dim roots are out of this
    // build's scope, same as everywhere else in this module).
    let rs = coerce_roots(r)?;
    let xarr = ingest(x)?;
    let out_shape = xarr.shape().to_vec();
    if !xarr.dtype().is_complex() && !rs.is_complex() {
        let contig = xarr.cast_to(DType::F64).to_contiguous();
        let xs = match contig.buffer() {
            Buffer::F64(v) => v.clone(),
            _ => unreachable!(),
        };
        let rv = match rs {
            Series::R(v) => v,
            _ => unreachable!(),
        };
        let out = poly::val_from_roots(&xs, &rv);
        return mk_shaped_real(py, out, out_shape);
    }
    let contig = xarr.cast_to(DType::C128).to_contiguous();
    let xs = match contig.buffer() {
        Buffer::C128(v) => v.clone(),
        _ => unreachable!(),
    };
    let rv = rs.to_complex();
    let out = poly::val_from_roots(&xs, &rv);
    mk_shaped_complex(py, out, out_shape)
}

// ─────────────────────── add / sub / mulx / mul ────────────────────────

#[pyfunction]
#[pyo3(name = "_poly_add")]
pub fn poly_add_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, poly::add_trim(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, poly::add_trim(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_poly_sub")]
pub fn poly_sub_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => mk_real(py, poly::sub_trim(&a, &b)),
        (Series::C(a), Series::C(b)) => mk_complex(py, poly::sub_trim(&a, &b)),
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

#[pyfunction]
#[pyo3(name = "_poly_mulx")]
pub fn poly_mulx_py(py: Python<'_>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // 2026-08-07 (Monday): this used to duplicate `poly::mulx`'s body
    // inline with a LITERAL zero fill (`vec![0.0; ...]` /
    // `vec![C128::new(0.0, 0.0); ...]`), which is why the `poly.rs` core
    // fix (`out[0] = c[0].poly_mul(T::zero())`, matching numpy's own
    // derived-zero `prd[0] = c[0] * 0` idiom) never reached this binding
    // -- the exposed `_poly_mulx` symbol never called `poly::mulx` at
    // all. Routing through the core kernel here is the actual fix; the
    // core kernel's own signed-zero derivation was correct but dead code
    // until now.
    //
    // Ticket #84 (2026-08-08, Monday): the `v.len() == 1 && v[0] == 0.0`
    // check ABOVE was itself checking an UNTRIMMED `v` -- numpy's own
    // `polymulx` starts with `[c] = pu.as_series([c])`, which trims
    // trailing zeros BEFORE that same length/value check, so a longer
    // all-(signed-)zero coefficient array (e.g. `[-0j,-0j,-0j]`, which
    // arises legitimately from underflow) fell through the `len()==1`
    // check here (false, len is 3) straight into `poly::mulx`'s general
    // path -- same missing-trim defect, same measured shape+sign fallout
    // as `legendre::legmulx`'s identical fix (see that function's doc
    // comment for the full trace). `poly::mulx` itself still documents
    // "the caller handles trimming" -- this IS that handling, now
    // actually done.
    match coerce_series_strict(c)? {
        Series::R(v) => {
            let v = poly::trim_trailing_zeros(&v);
            if v.len() == 1 && v[0] == 0.0 {
                return mk_real(py, v);
            }
            mk_real(py, poly::mulx(&v))
        }
        Series::C(v) => {
            let v = poly::trim_trailing_zeros(&v);
            if v.len() == 1 && v[0] == C128::new(0.0, 0.0) {
                return mk_complex(py, v);
            }
            mk_complex(py, poly::mulx(&v))
        }
    }
}

#[pyfunction]
#[pyo3(name = "_poly_mul")]
pub fn poly_mul_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => {
            let out = poly::convolve_full(&a, &b);
            mk_real(py, poly::trim_trailing_zeros(&out))
        }
        (Series::C(a), Series::C(b)) => {
            let out = poly::convolve_full(&a, &b);
            mk_complex(py, poly::trim_trailing_zeros(&out))
        }
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

// ─────────────────────────── polydiv ────────────────────────────

#[pyfunction]
#[pyo3(name = "_poly_div")]
pub fn poly_div_py(py: Python<'_>, c1: &Bound<'_, PyAny>, c2: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (a, b) = coerce_pair(c1, c2)?;
    match (a, b) {
        (Series::R(a), Series::R(b)) => match poly::div(&a, &b) {
            Ok((q, r)) => Ok((mk_real(py, q)?, mk_real(py, r)?)),
            Err(()) => Err(PyZeroDivisionError::new_err(())),
        },
        (Series::C(a), Series::C(b)) => match poly::div(&a, &b) {
            Ok((q, r)) => Ok((mk_complex(py, q)?, mk_complex(py, r)?)),
            Err(()) => Err(PyZeroDivisionError::new_err(())),
        },
        _ => unreachable!("coerce_pair always returns a matching kind"),
    }
}

// ─────────────────────────── polyder / polyint ────────────────────────────

#[pyfunction]
#[pyo3(name = "_poly_der")]
pub fn poly_der_py(py: Python<'_>, c: &Bound<'_, PyAny>, m: i64, scl: f64) -> PyResult<Py<PyAny>> {
    if m < 0 {
        return Err(poly_err("The order of derivation must be non-negative"));
    }
    match coerce_series(c)? {
        Series::R(v) => mk_real(py, poly::der(&v, m as usize, scl)),
        Series::C(v) => mk_complex(py, poly::der(&v, m as usize, C128::new(scl, 0.0))),
    }
}

#[pyfunction]
#[pyo3(name = "_poly_int")]
pub fn poly_int_py(
    py: Python<'_>,
    c: &Bound<'_, PyAny>,
    m: i64,
    k: Vec<f64>,
    lbnd: f64,
    scl: f64,
) -> PyResult<Py<PyAny>> {
    if m < 0 {
        return Err(poly_err("The order of integration must be non-negative"));
    }
    if k.len() > m as usize {
        return Err(poly_err("k must have length <= m"));
    }
    let mut kk = k.clone();
    kk.resize(m as usize, 0.0);
    match coerce_series(c)? {
        Series::R(v) => mk_real(py, poly::int_(&v, m as usize, &kk, lbnd, scl)),
        Series::C(v) => {
            let kkc: Vec<C128> = kk.iter().map(|&x| C128::new(x, 0.0)).collect();
            mk_complex(py, poly::int_(&v, m as usize, &kkc, C128::new(lbnd, 0.0), C128::new(scl, 0.0)))
        }
    }
}

// ─────────────────────────── polyvander ────────────────────────────

#[pyfunction]
#[pyo3(name = "_poly_vander")]
pub fn poly_vander_py(py: Python<'_>, x: &Bound<'_, PyAny>, deg: i64) -> PyResult<Py<PyAny>> {
    if deg < 0 {
        return Err(poly_err("deg must be non-negative"));
    }
    let xarr = ingest(x)?;
    // numpy's own `polyvander` does `x = np.array(x, copy=False, ndmin=1) +
    // 0.0` -- a 0-d/scalar `x` is silently promoted to a 1-element 1-D
    // array (verified directly: `P.polyvander(2.0, 3).shape ==
    // (1, 4)`), not rejected. ndim > 1 genuinely is out of scope here
    // (batched multi-dim x support was never built).
    if xarr.ndim() > 1 {
        return Err(poly_err(format!(
            "polyvander: expected a 1-d array of sample points, got {} dimensions \
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
        let out = poly::vander(&xs, deg as usize);
        let inner =
            NdArray::from_buffer(Buffer::C128(out), vec![n, width], Order::C).map_err(crate::to_py_err)?;
        Ok(Py::new(py, PyArray { inner })?.into_any())
    } else {
        let contig = xarr.cast_to(DType::F64).to_contiguous();
        let xs = match contig.buffer() {
            Buffer::F64(v) => v.clone(),
            _ => unreachable!(),
        };
        let out = poly::vander(&xs, deg as usize);
        let inner = NdArray::from_buffer(Buffer::F64(out), vec![n, width], Order::C).map_err(crate::to_py_err)?;
        Ok(Py::new(py, PyArray { inner })?.into_any())
    }
}

// ─────────────────────────── polycompanion ────────────────────────────

#[pyfunction]
#[pyo3(name = "_poly_companion")]
pub fn poly_companion_py(py: Python<'_>, c: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let s = coerce_series_strict(c)?;
    if s.len() < 2 {
        return Err(poly_err("Series must have maximum degree of at least 1."));
    }
    let n = s.len() - 1;
    match s {
        Series::R(v) => {
            let m = poly::companion(&v);
            let inner = NdArray::from_buffer(Buffer::F64(m), vec![n, n], Order::C).map_err(crate::to_py_err)?;
            Ok(Py::new(py, PyArray { inner })?.into_any())
        }
        Series::C(v) => {
            let m = poly::companion(&v);
            let inner = NdArray::from_buffer(Buffer::C128(m), vec![n, n], Order::C).map_err(crate::to_py_err)?;
            Ok(Py::new(py, PyArray { inner })?.into_any())
        }
    }
}

pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(poly_domain_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_zero_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_one_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_x_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_line_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_trim_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_val_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_valfromroots_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_add_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_sub_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_mulx_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_mul_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_div_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_der_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_int_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_vander_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(poly_companion_py, m)?)?;
    Ok(())
}
