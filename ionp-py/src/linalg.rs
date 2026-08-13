//! `anionpy.linalg.*` — PyO3 bindings over the already-implemented, already-
//! tested (66/66 unit tests, residual/invariant-based, see
//! `ionp-ion/src/dense_linalg.rs`) dense linear algebra core. This file
//! does no numerics of its own: every routine below marshals numpy arrays
//! across the FFI boundary, calls straight into `ionp_ion::dense_linalg`,
//! and marshals the result back. All arithmetic happens in Rust via
//! Apple Accelerate LAPACK, exactly like the rest of `ionp-py`.
//!
//! ## Scope (deliberate, documented, not a placeholder)
//! Only the 2-D (single matrix, no batching over leading axes) call forms
//! are bound, because `ionp-ion`'s core operates on a single `m x n`
//! buffer. Where a numpy parameter changes the *algorithm* used (not just
//! performance) and the core has no independent code path for it, the
//! binding raises `NotImplementedError` rather than silently ignoring the
//! parameter and returning a value that looks right but was computed a
//! different way than numpy would compute it:
//!   - `svd(..., full_matrices=True)` (the numpy default!) — the core is
//!     economy-only (`dgesdd`/`zgesdd` job `'S'`); only
//!     `full_matrices=False` is bound.
//!   - `eigh`/`eigvalsh(..., UPLO='U')` — the core always reads the lower
//!     triangle; only the default `UPLO='L'` is bound.
//!   - `matrix_rank`/`pinv(..., hermitian=True)` — numpy's `hermitian=True`
//!     takes a genuinely different (eigendecomposition-based) algorithm
//!     path with a different answer, not just a speed difference; the core
//!     has no such path, so only `hermitian=False` (the default) is bound.
//!   - `qr(..., mode=...)` other than `'reduced'` (the default) — not
//!     bound.
//!   - `trace(..., offset!=0)` — the core only sums the main diagonal.
//!   - `cond(..., p=...)` for any `p` other than `None`/`1`/`2`/`inf` — the
//!     core only implements those four.
//!   - `matrix_power`/`cond` on complex input — the core only implements
//!     these two for `f64`, not `Complex64`.
//! Every one of these is a real, named gap, not silently-wrong output.

use num_complex::Complex64;
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

use ionp_core::{manip, Buffer, DType, NdArray, Order, SliceItem};

use crate::{check_full_broadcast, errors, write_into_out, write_into_out_ufunc, PyArray};

/// numpy's `det`/`slogdet`/`trace`/`cond`/`matrix_rank`/`lstsq` (rank
/// output) return genuine numpy scalars (`np.float64`/`np.complex128`/
/// `np.int32`/`np.int64`, ...), which carry a `.dtype` -- a bare Rust
/// `f64`/`Complex64` marshaled via `into_py_any`/`PyComplex::from_doubles`
/// becomes a plain Python `float`/`complex` with NO `.dtype` at all, wrong
/// in a different way than an anionpy 0-d `ndarray` is wrong (see
/// `docs/scalar-return-type-defect.md`). `mk_scalar_*` below build a real
/// 0-d `NdArray` first (so the value/dtype-selection logic is unchanged)
/// and then hand it to `crate::numpy_scalar_from_0d`, the shared
/// scalar-return-type fix, which mints the matching genuine numpy scalar
/// type (`.dtype` present and correct, and `type(...) is np.float64`
/// etc, not merely an anionpy `ndarray` lookalike or a bare Python
/// float/complex/int).
fn linalg_ionp_err(e: ionp_core::IonpError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn mk_scalar_f64(py: Python<'_>, v: f64) -> PyResult<Py<PyAny>> {
    let inner = NdArray::from_buffer(Buffer::F64(vec![v]), vec![], Order::C).map_err(linalg_ionp_err)?;
    crate::numpy_scalar_from_0d(py, &inner)
}
fn mk_scalar_c128(py: Python<'_>, v: Complex64) -> PyResult<Py<PyAny>> {
    let inner = NdArray::from_buffer(Buffer::C128(vec![v]), vec![], Order::C).map_err(linalg_ionp_err)?;
    crate::numpy_scalar_from_0d(py, &inner)
}

/// `trace(..., dtype=...)`: measured against real numpy 2.5.1 (see
/// `linalg.rs` `trace` fn below) that `np.trace(a, dtype=X)` sums the
/// diagonal at full `float64`/`complex128` precision first and only casts
/// the *final scalar* to `X` -- NOT accumulating at `X`'s precision the
/// whole way through (verified with a float32-vs-float64-accumulator
/// discriminating case: a 50x50 matrix with one diagonal entry `1e-3`
/// against others `~1e7`; `np.trace(a, dtype=np.float32)` matched
/// `float32(float64_sum)` bit-for-bit and did NOT match a running-float32
/// accumulation, which loses the small entry earlier). So this always
/// calls the existing full-precision `rc::trace`/`zc::trace` core fn
/// first, then casts the resulting scalar's `Buffer` -- no new numeric
/// code path, just an output-dtype cast, matching the "cast after" rule
/// measured above.
fn mk_scalar_f64_dtype(py: Python<'_>, v: f64, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let buf = match dtype {
        None => Buffer::F64(vec![v]),
        Some(d) => Buffer::F64(vec![v]).cast_to(crate::dtype_from_pyobj_no_su(d)?),
    };
    let inner = NdArray::from_buffer(buf, vec![], Order::C).map_err(linalg_ionp_err)?;
    crate::numpy_scalar_from_0d(py, &inner)
}
fn mk_scalar_c128_dtype(py: Python<'_>, v: Complex64, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let buf = match dtype {
        None => Buffer::C128(vec![v]),
        Some(d) => Buffer::C128(vec![v]).cast_to(crate::dtype_from_pyobj_no_su(d)?),
    };
    let inner = NdArray::from_buffer(buf, vec![], Order::C).map_err(linalg_ionp_err)?;
    crate::numpy_scalar_from_0d(py, &inner)
}

use ionp_ion::dense_linalg::{complex as zc, real as rc, LinAlgError};

// ─────────────────────────────── errors ────────────────────────────────

/// Raise anionpy's own `LinAlgError` (subclasses real `numpy.linalg.LinAlgError`
/// when numpy is importable, so identity-based `except
/// numpy.linalg.LinAlgError:` still catches it; subclasses `ValueError`
/// directly when numpy is absent -- see `errors.rs`, which owns the class
/// itself and its numpy-present/-absent construction). Unlike `lib.rs`'s
/// `no_ufunc_loop_err`/`ufunc_output_casting_err` (which construct native
/// Rust `TypeError`s instead, because their numpy originals --
/// `_UFuncNoLoopError`/`_UFuncOutputCastingError` -- are PRIVATE, so raising
/// the genuine class would make the differential comparison circular),
/// `LinAlgError` is PUBLIC API (`numpy.linalg.LinAlgError`, documented,
/// legitimately caught by identity by real callers), so subclassing the
/// genuine class here is correct rather than a defect: this is project
/// policy's "Answer B" other branch, not the same pattern as the
/// private-class shims. Never uses a bare `?` on the numpy import (that was
/// the single site in the whole crate making `import anionpy` require numpy;
/// `errors.rs` always falls back instead).
fn linalg_err(e: LinAlgError) -> PyErr {
    Python::attach(|py| errors::raise_linalg_error(py, &e.to_string()))
        .unwrap_or_else(|_| PyValueError::new_err(e.to_string()))
}

/// Same as `linalg_err`, but raises with `msg` verbatim instead of routing
/// it through `LinAlgError`'s `Display` (which always prepends
/// `"shape mismatch: "` for the `ShapeMismatch` variant -- appropriate for
/// most of `LinAlgError::ShapeMismatch`'s callers, but numpy's own
/// `lstsq`/`svd` "Incompatible dimensions" / "Last 2 dimensions of the
/// array must be square" `LinAlgError`s carry no such prefix, verified
/// against real numpy 2.5.1). Kept local to this file since
/// `ionp_ion::dense_linalg`'s `LinAlgError` Display impl is out of this
/// task's file ownership.
fn linalg_err_raw(msg: &str) -> PyErr {
    Python::attach(|py| errors::raise_linalg_error(py, msg))
        .unwrap_or_else(|_| PyValueError::new_err(msg.to_string()))
}

// ─────────────────────────── extraction helpers ────────────────────────

/// Pull a real `Vec<f64>` + shape out of an `anionpy.ndarray` of dtype
/// `float64`, if `obj` is one. `Ok(None)` (not an error) means "this isn't
/// an anionpy.ndarray of this dtype, try numpy / the complex path next" --
/// same linear-probe contract `as_real2`/`as_complex2` already use for the
/// numpy side. Any strided/non-contiguous anionpy.ndarray is materialized
/// contiguous first (`to_contiguous`) so callers always get a plain,
/// logically-C-ordered `Vec<f64>` regardless of how the input array was
/// produced (slice, transpose, ...).
fn ionp_real_buf(obj: &Bound<'_, PyAny>) -> Option<(Vec<f64>, Vec<usize>)> {
    let pa = obj.extract::<PyRef<'_, PyArray>>().ok()?;
    if pa.inner.dtype() != DType::F64 {
        return None;
    }
    let shape = pa.inner.shape().to_vec();
    let contig = pa.inner.to_contiguous();
    let data = match contig.buffer() {
        Buffer::F64(v) => v.clone(),
        _ => unreachable!("to_contiguous preserves dtype"),
    };
    Some((data, shape))
}

fn ionp_complex_buf(obj: &Bound<'_, PyAny>) -> Option<(Vec<Complex64>, Vec<usize>)> {
    let pa = obj.extract::<PyRef<'_, PyArray>>().ok()?;
    if pa.inner.dtype() != DType::C128 {
        return None;
    }
    let shape = pa.inner.shape().to_vec();
    let contig = pa.inner.to_contiguous();
    let data = match contig.buffer() {
        Buffer::C128(v) => v.clone(),
        _ => unreachable!("to_contiguous preserves dtype"),
    };
    Some((data, shape))
}

/// Pull a 2-D `f64` matrix out of an arbitrary array argument -- either a
/// real `anionpy.ndarray` (checked first, see `ionp_real_buf`; this is the
/// 2026-08-01 fix: every one of the 18 `anionpy.linalg.*` bindings used to
/// bind ONLY `numpy.ndarray`, silently rejecting the library's own array
/// type, which made these items unusable as part of a numpy REPLACEMENT --
/// see the task brief this fixes) or a numpy array (existing path,
/// unchanged). `None` (not an error) means "try the complex path next" --
/// mirrors `lib.rs::ndarray_from_numpy`'s linear-probe style.
fn as_real2(obj: &Bound<'_, PyAny>) -> PyResult<Option<(Vec<f64>, usize, usize)>> {
    if let Some((data, shape)) = ionp_real_buf(obj) {
        if shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "anionpy.linalg: expected a 2-D array, got shape {:?} (batched/1-D inputs are out of scope)",
                shape
            )));
        }
        return Ok(Some((data, shape[0], shape[1])));
    }
    match obj.extract::<PyReadonlyArrayDyn<f64>>() {
        Ok(arr) => {
            let shape = arr.shape().to_vec();
            if shape.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "anionpy.linalg: expected a 2-D array, got shape {:?} (batched/1-D inputs are out of scope)",
                    shape
                )));
            }
            let data: Vec<f64> = arr.as_array().iter().copied().collect();
            Ok(Some((data, shape[0], shape[1])))
        }
        Err(_) => Ok(None),
    }
}

fn as_complex2(obj: &Bound<'_, PyAny>) -> PyResult<Option<(Vec<Complex64>, usize, usize)>> {
    if let Some((data, shape)) = ionp_complex_buf(obj) {
        if shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "anionpy.linalg: expected a 2-D array, got shape {:?} (batched/1-D inputs are out of scope)",
                shape
            )));
        }
        return Ok(Some((data, shape[0], shape[1])));
    }
    match obj.extract::<PyReadonlyArrayDyn<Complex64>>() {
        Ok(arr) => {
            let shape = arr.shape().to_vec();
            if shape.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "anionpy.linalg: expected a 2-D array, got shape {:?} (batched/1-D inputs are out of scope)",
                    shape
                )));
            }
            let data: Vec<Complex64> = arr.as_array().iter().copied().collect();
            Ok(Some((data, shape[0], shape[1])))
        }
        Err(_) => Ok(None),
    }
}

fn as_real1(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<f64>>> {
    if let Some((data, shape)) = ionp_real_buf(obj) {
        return Ok(if shape.len() == 1 { Some(data) } else { None });
    }
    match obj.extract::<PyReadonlyArrayDyn<f64>>() {
        Ok(arr) if arr.shape().len() == 1 => Ok(Some(arr.as_array().iter().copied().collect())),
        Ok(_) => Ok(None),
        Err(_) => Ok(None),
    }
}

fn as_complex1(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<Complex64>>> {
    if let Some((data, shape)) = ionp_complex_buf(obj) {
        return Ok(if shape.len() == 1 { Some(data) } else { None });
    }
    match obj.extract::<PyReadonlyArrayDyn<Complex64>>() {
        Ok(arr) if arr.shape().len() == 1 => Ok(Some(arr.as_array().iter().copied().collect())),
        Ok(_) => Ok(None),
        Err(_) => Ok(None),
    }
}

/// Pull a real `NdArray` of ANY dtype/shape out of an arbitrary array
/// argument -- either a real `anionpy.ndarray` (checked first, same
/// linear-probe order `as_real2`/`as_complex2` use) or a numpy array (via
/// `crate::ndarray_from_numpy`, the same conversion `matmul.rs`'s
/// `coerce_matmul_operand` uses for its ionp-array-or-numpy-array branch).
/// Used by the 2026-08-01 array-API items below (`diagonal`,
/// `matrix_transpose`, `tensordot`, `matmul`, `vecdot`, `outer`) whose
/// underlying `ionp_core`/`ionp_ion` implementations are already
/// dtype-polymorphic (unlike the LAPACK-backed 2-D-float64/complex128-only
/// items above), so restricting the Python-argument boundary to those two
/// dtypes here would be a narrower gap than the Rust code actually has.
/// Row-major transpose of a square `n x n` real matrix, used by
/// `eigvalsh(UPLO='U')` (see that function's doc comment for the proof).
fn transpose_f64(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j];
        }
    }
    out
}

/// Row-major conjugate-transpose of a square `n x n` complex matrix, used
/// by `eigvalsh(UPLO='U')` (see that function's doc comment for the proof).
fn conj_transpose_c128(data: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j].conj();
        }
    }
    out
}

fn as_ndarray_any(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(pa) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(pa.inner.clone());
    }
    crate::ndarray_from_numpy(obj)
}

fn unsupported_dtype(name: &str) -> PyErr {
    PyValueError::new_err(format!(
        "anionpy.linalg.{name}: unsupported dtype (only float64 and complex128 2-D arrays are bound)"
    ))
}

// ═══════════════ dtype promotion (2026-08-01 widen-coverage fix) ════════
//
// Real numpy's `numpy.linalg._linalg._commonType` promotes every array
// argument through a fixed table (verified via `inspect.getsource` against
// numpy 2.5.1): float32 -> float32 ("single"), complex64 -> complex64
// ("csingle"), float64/complex128 stay as-is, and EVERY OTHER real dtype
// (bool, int8-64, uint8-64, float16) -> float64 ("double", there is no
// lower-precision LAPACK driver family). When more than one array
// participates (`solve(a, b)`, `lstsq(a, b)`), the combined precision is
// single only if EVERY participating array is single-precision-family,
// otherwise double -- the same "any double array wins" rule numpy's
// `_commonType` implements by taking the max over all arrays' `_array_precision`.
//
// anionpy's Rust core (`ionp_ion::dense_linalg`) only has `f64`/`Complex64`
// (f64-pair) LAPACK call paths -- no `f32`/`Complex32` (f32-pair) driver.
// Rather than duplicate every LAPACK routine at single precision (a much
// larger, core-crate-touching change), this promotes EVERY real input to
// f64 / EVERY complex input to Complex64 (f64-pair) before calling the
// existing core, and downcasts the OUTPUT to f32/complex64 at the end when
// `Prec::Single` was requested. This is not "the s/c LAPACK driver" bit-
// for-bit -- it is strictly MORE accurate internally (full double-precision
// arithmetic, rounded once at the end) than numpy's real single-precision
// call, so it does not introduce a defect, but it is also not guaranteed
// bit-identical to numpy's actual `sgetrf`/`cgesdd`/etc. output. Each item
// re-declared using this path carries a per-dtype `float32`/`complex64`
// epsilon_tolerance in `linalg_cases.py`, measured the same way (seeded
// sweep, `ulp_sweep.sweep_matrix_item_evidence`) the existing float64/
// complex128 non-associativity tolerances are -- not asserted without
// evidence.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Prec {
    Single,
    Double,
    // Added for `vector_norm`/`matrix_norm`/`norm` only: every OTHER
    // caller of `classify_prec` in this file first calls `reject_float16`
    // (or routes through `svd_singular_values_batched`, which does), so
    // `Half` is unreachable there and those call sites' existing
    // two-way `if prec == Prec::Single {..} else {..}` casts are
    // unaffected. Needed because, verified against real numpy 2.5.1,
    // `vector_norm`/`matrix_norm`/`norm` on float16 input return a
    // float16 (not float64-promoted) result for every non-SVD-based ord
    // -- unlike every other linalg function here, which promotes to
    // double.
    Half,
}

fn classify_prec(dt: DType) -> Prec {
    match dt {
        DType::F16 => Prec::Half,
        DType::F32 | DType::C64 => Prec::Single,
        _ => Prec::Double,
    }
}

fn combine_prec(a: Prec, b: Prec) -> Prec {
    if a == Prec::Single && b == Prec::Single {
        Prec::Single
    } else {
        Prec::Double
    }
}

fn is_complex_dtype(dt: DType) -> bool {
    matches!(dt, DType::C64 | DType::C128)
}

// 2026-08-01: real numpy's `_commonType` (numpy/linalg/_linalg.py) does NOT
// promote float16 -- it explicitly rejects it (verified via
// `inspect.getsource`): every OTHER real dtype maps to float64, but
// float16 alone raises `TypeError("array type float16 is unsupported in
// linalg")`. This was missed when the promotion table above was written
// (float16 was folded into the same "everything else -> double" bucket as
// bool/int*/uint*), and the widened dtype-sweep corpus this task built
// caught it directly: anionpy was silently computing on float16 input where
// numpy raises. Matching numpy's exact exception TYPE and MESSAGE here is
// not "calling numpy at runtime for an error message" (forbidden) -- it is
// reimplementing a known, static, verified-from-source numpy behavior in
// Rust, the same as every other error path in this file.
fn reject_float16(dt: DType) -> PyResult<()> {
    if dt == DType::F16 {
        return Err(PyTypeError::new_err(
            "array type float16 is unsupported in linalg",
        ));
    }
    Ok(())
}

fn try_ndarray_any(obj: &Bound<'_, PyAny>) -> Option<NdArray> {
    if let Ok(pa) = obj.extract::<PyRef<'_, PyArray>>() {
        return Some(pa.inner.clone());
    }
    crate::ndarray_from_numpy(obj).ok()
}

/// 2-D promoted extraction: any of anionpy's/numpy's 13 dtypes in, either a
/// `Vec<f64>` (real path: bool/int*/uint*/float16/float32/float64) or a
/// `Vec<Complex64>` (complex path: complex64/complex128) out, tagged with
/// the `Prec` the caller must downcast results to. `Ok(None)` (not an
/// error) means "not an array-like `anionpy`/numpy can extract at all" --
/// mirrors `as_real2`'s existing linear-probe contract.
#[allow(clippy::type_complexity)]
fn as_promoted_real2(obj: &Bound<'_, PyAny>) -> PyResult<Option<(Vec<f64>, usize, usize, Prec)>> {
    let Some(arr) = try_ndarray_any(obj) else { return Ok(None) };
    let dt = arr.dtype();
    if is_complex_dtype(dt) {
        return Ok(None);
    }
    reject_float16(dt)?;
    let shape = arr.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "anionpy.linalg: expected a 2-D array, got shape {:?} (batched/1-D inputs are out of scope)",
            shape
        )));
    }
    let prec = classify_prec(dt);
    let contig = arr.to_contiguous();
    let buf = contig.buffer().cast_to(DType::F64);
    let data = match buf {
        Buffer::F64(v) => v,
        _ => unreachable!("cast_to(F64) always yields Buffer::F64"),
    };
    Ok(Some((data, shape[0], shape[1], prec)))
}

#[allow(clippy::type_complexity)]
fn as_promoted_complex2(obj: &Bound<'_, PyAny>) -> PyResult<Option<(Vec<Complex64>, usize, usize, Prec)>> {
    let Some(arr) = try_ndarray_any(obj) else { return Ok(None) };
    let dt = arr.dtype();
    if !is_complex_dtype(dt) {
        return Ok(None);
    }
    let shape = arr.shape().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "anionpy.linalg: expected a 2-D array, got shape {:?} (batched/1-D inputs are out of scope)",
            shape
        )));
    }
    let prec = classify_prec(dt);
    let contig = arr.to_contiguous();
    let buf = contig.buffer().cast_to(DType::C128);
    let data = match buf {
        Buffer::C128(v) => v,
        _ => unreachable!("cast_to(C128) always yields Buffer::C128"),
    };
    Ok(Some((data, shape[0], shape[1], prec)))
}

/// numpy's `_assert_stacked_2d` + `_assert_stacked_square`
/// (`numpy/linalg/_linalg.py`), run BEFORE any LAPACK-backed dispatch for
/// every square-matrix-family item (`det`/`slogdet`/`inv`/`cholesky`/
/// `eigvals`/`eigvalsh`/`eig`/`eigh`/`matrix_power`). Verified against real
/// numpy 2.5.1: ndim < 2 raises `LinAlgError("{ndim}-dimensional array
/// given. Array must be at least two-dimensional")`; ndim >= 2 with
/// unequal trailing two dims raises `LinAlgError("Last 2 dimensions of the
/// array must be square")` -- checked in that order, and ONLY the trailing
/// two dims need be square (arbitrary batch dims ahead of them are fine).
/// This is the up-front gate the panic audit found missing: `to_col_major`
/// in `ionp-ion/src/dense_linalg.rs` assumes its `rows*cols == a.len()`
/// invariant via `debug_assert_eq!` alone (compiled out in `--release`), so
/// any caller that reaches it with a non-square/mismatched shape indexes
/// out of bounds. Calling this BEFORE ever constructing the flat buffer for
/// LAPACK closes that gap at the boundary instead of relying on the
/// LAPACK-adjacent code to defend itself. Returns the shared trailing
/// dimension `m` (== rows == cols) on success.
/// numpy's `_assert_stacked_2d` alone, without the squareness half -- for
/// the non-square-friendly LAPACK items (`qr`, `pinv`, `matrix_rank`,
/// `svd`) whose trailing two dims may legitimately differ.
fn check_stacked_2d(shape: &[usize]) -> PyResult<()> {
    if shape.len() < 2 {
        return Err(linalg_err_raw(&format!(
            "{}-dimensional array given. Array must be at least two-dimensional",
            shape.len()
        )));
    }
    Ok(())
}

fn check_stacked_square(shape: &[usize]) -> PyResult<usize> {
    if shape.len() < 2 {
        return Err(linalg_err_raw(&format!(
            "{}-dimensional array given. Array must be at least two-dimensional",
            shape.len()
        )));
    }
    let m = shape[shape.len() - 2];
    let n = shape[shape.len() - 1];
    if m != n {
        return Err(linalg_err_raw("Last 2 dimensions of the array must be square"));
    }
    Ok(m)
}

/// N-D promoted real extraction -- same dtype-promotion/float16-rejection
/// rules as `as_promoted_real2`, but does NOT collapse to exactly 2-D:
/// returns the full shape (any ndim, including 0/1) so callers can run
/// `check_stacked_square` themselves and then batch over any leading
/// dimensions, matching numpy's own batched `linalg` semantics (`det` on a
/// `(2,3,3)` input returns shape `(2,)`, etc). `anionpy`'s own `NdArray` is
/// always C-contiguous after `to_contiguous()`, so each batch element's
/// `m*n` matrix is a contiguous slice of `data` at `data[b*m*n..(b+1)*m*n]`.
fn as_promoted_real_nd(obj: &Bound<'_, PyAny>) -> PyResult<Option<(Vec<f64>, Vec<usize>, Prec)>> {
    let Some(arr) = try_ndarray_any(obj) else { return Ok(None) };
    let dt = arr.dtype();
    if is_complex_dtype(dt) {
        return Ok(None);
    }
    reject_float16(dt)?;
    let shape = arr.shape().to_vec();
    let prec = classify_prec(dt);
    let contig = arr.to_contiguous();
    let buf = contig.buffer().cast_to(DType::F64);
    let data = match buf {
        Buffer::F64(v) => v,
        _ => unreachable!("cast_to(F64) always yields Buffer::F64"),
    };
    Ok(Some((data, shape, prec)))
}

/// Complex counterpart of `as_promoted_real_nd` -- see its doc comment.
fn as_promoted_complex_nd(obj: &Bound<'_, PyAny>) -> PyResult<Option<(Vec<Complex64>, Vec<usize>, Prec)>> {
    let Some(arr) = try_ndarray_any(obj) else { return Ok(None) };
    let dt = arr.dtype();
    if !is_complex_dtype(dt) {
        return Ok(None);
    }
    let shape = arr.shape().to_vec();
    let prec = classify_prec(dt);
    let contig = arr.to_contiguous();
    let buf = contig.buffer().cast_to(DType::C128);
    let data = match buf {
        Buffer::C128(v) => v,
        _ => unreachable!("cast_to(C128) always yields Buffer::C128"),
    };
    Ok(Some((data, shape, prec)))
}

/// Build an arbitrary-shape f64 output (e.g. a batched `det` result of
/// shape `batch_shape`, or a batched `inv`/`cholesky` result of shape
/// `batch_shape + [m, m]`), downcasting to f32 for `Prec::Single` the same
/// way the fixed-shape `mkN_f64_prec` constructors do.
fn mk_nd_f64_prec(py: Python<'_>, data: Vec<f64>, shape: Vec<usize>, prec: Prec) -> PyResult<Py<PyAny>> {
    match prec {
        Prec::Double => {
            let inner = NdArray::from_buffer(Buffer::F64(data), shape, Order::C).map_err(linalg_ionp_err)?;
            PyArray { inner }.into_py_any(py)
        }
        Prec::Single => {
            let buf = Buffer::F64(data).cast_to(DType::F32);
            let inner = NdArray::from_buffer(buf, shape, Order::C).map_err(linalg_ionp_err)?;
            PyArray { inner }.into_py_any(py)
        }
        Prec::Half => unreachable!("mk_nd_f64_prec: float16 must be rejected upstream"),
    }
}

/// Complex counterpart of `mk_nd_f64_prec` -- see its doc comment.
fn mk_nd_c128_prec(py: Python<'_>, data: Vec<Complex64>, shape: Vec<usize>, prec: Prec) -> PyResult<Py<PyAny>> {
    match prec {
        Prec::Double => {
            let inner = NdArray::from_buffer(Buffer::C128(data), shape, Order::C).map_err(linalg_ionp_err)?;
            PyArray { inner }.into_py_any(py)
        }
        Prec::Single => {
            let buf = Buffer::C128(data).cast_to(DType::C64);
            let inner = NdArray::from_buffer(buf, shape, Order::C).map_err(linalg_ionp_err)?;
            PyArray { inner }.into_py_any(py)
        }
        Prec::Half => unreachable!("mk_nd_c128_prec: float16 must be rejected upstream"),
    }
}

fn as_promoted_real1(obj: &Bound<'_, PyAny>) -> PyResult<Option<(Vec<f64>, Prec)>> {
    let Some(arr) = try_ndarray_any(obj) else { return Ok(None) };
    let dt = arr.dtype();
    if is_complex_dtype(dt) || arr.ndim() != 1 {
        return Ok(None);
    }
    reject_float16(dt)?;
    let prec = classify_prec(dt);
    let contig = arr.to_contiguous();
    let buf = contig.buffer().cast_to(DType::F64);
    let data = match buf {
        Buffer::F64(v) => v,
        _ => unreachable!(),
    };
    Ok(Some((data, prec)))
}

fn as_promoted_complex1(obj: &Bound<'_, PyAny>) -> PyResult<Option<(Vec<Complex64>, Prec)>> {
    let Some(arr) = try_ndarray_any(obj) else { return Ok(None) };
    let dt = arr.dtype();
    if !is_complex_dtype(dt) || arr.ndim() != 1 {
        return Ok(None);
    }
    let prec = classify_prec(dt);
    let contig = arr.to_contiguous();
    let buf = contig.buffer().cast_to(DType::C128);
    let data = match buf {
        Buffer::C128(v) => v,
        _ => unreachable!(),
    };
    Ok(Some((data, prec)))
}

// ─── precision-aware output constructors (downcast to f32/complex64 when
// `Prec::Single`, matching numpy's "single input family -> single output"
// rule) ───

fn mk2_f64_prec(py: Python<'_>, data: Vec<f64>, rows: usize, cols: usize, prec: Prec) -> PyResult<Py<PyAny>> {
    match prec {
        Prec::Double => mk2_f64(py, data, rows, cols),
        Prec::Single => {
            let buf = Buffer::F64(data).cast_to(DType::F32);
            let inner = NdArray::from_buffer(buf, vec![rows, cols], Order::C).map_err(linalg_ionp_err)?;
            PyArray { inner }.into_py_any(py)
        }
        // `Half` is unreachable here: every caller of this generic
        // precision-output constructor rejects float16 upstream via
        // `reject_float16` before `classify_prec` ever runs (this
        // constructor predates -- and is unrelated to -- the
        // vector_norm/matrix_norm/norm float16-passthrough behavior).
        Prec::Half => unreachable!("mk2_f64_prec: float16 must be rejected upstream"),
    }
}
fn mk1_f64_prec(py: Python<'_>, data: Vec<f64>, prec: Prec) -> PyResult<Py<PyAny>> {
    match prec {
        Prec::Double => mk1_f64(py, data),
        Prec::Single => {
            let n = data.len();
            let buf = Buffer::F64(data).cast_to(DType::F32);
            let inner = NdArray::from_buffer(buf, vec![n], Order::C).map_err(linalg_ionp_err)?;
            PyArray { inner }.into_py_any(py)
        }
        Prec::Half => unreachable!("mk1_f64_prec: float16 must be rejected upstream"),
    }
}
fn mk2_c64_prec(py: Python<'_>, data: Vec<Complex64>, rows: usize, cols: usize, prec: Prec) -> PyResult<Py<PyAny>> {
    match prec {
        Prec::Double => mk2_c64(py, data, rows, cols),
        Prec::Single => {
            let buf = Buffer::C128(data).cast_to(DType::C64);
            let inner = NdArray::from_buffer(buf, vec![rows, cols], Order::C).map_err(linalg_ionp_err)?;
            PyArray { inner }.into_py_any(py)
        }
        Prec::Half => unreachable!("mk2_c64_prec: float16 must be rejected upstream"),
    }
}
fn mk1_c64_prec(py: Python<'_>, data: Vec<Complex64>, prec: Prec) -> PyResult<Py<PyAny>> {
    match prec {
        Prec::Double => mk1_c64(py, data),
        Prec::Single => {
            let n = data.len();
            let buf = Buffer::C128(data).cast_to(DType::C64);
            let inner = NdArray::from_buffer(buf, vec![n], Order::C).map_err(linalg_ionp_err)?;
            PyArray { inner }.into_py_any(py)
        }
        Prec::Half => unreachable!("mk1_c64_prec: float16 must be rejected upstream"),
    }
}
fn mk_scalar_f64_prec(py: Python<'_>, v: f64, prec: Prec) -> PyResult<Py<PyAny>> {
    match prec {
        Prec::Double => mk_scalar_f64(py, v),
        Prec::Single => {
            let buf = Buffer::F64(vec![v]).cast_to(DType::F32);
            let inner = NdArray::from_buffer(buf, vec![], Order::C).map_err(linalg_ionp_err)?;
            crate::numpy_scalar_from_0d(py, &inner)
        }
        Prec::Half => unreachable!("mk_scalar_f64_prec: float16 must be rejected upstream"),
    }
}
fn mk_scalar_c128_prec(py: Python<'_>, v: Complex64, prec: Prec) -> PyResult<Py<PyAny>> {
    match prec {
        Prec::Double => mk_scalar_c128(py, v),
        Prec::Single => {
            let buf = Buffer::C128(vec![v]).cast_to(DType::C64);
            let inner = NdArray::from_buffer(buf, vec![], Order::C).map_err(linalg_ionp_err)?;
            crate::numpy_scalar_from_0d(py, &inner)
        }
        Prec::Half => unreachable!("mk_scalar_c128_prec: float16 must be rejected upstream"),
    }
}

// ─── hand-rolled complex helpers for items whose core (`dense_linalg.rs`)
// has no complex path at all (`matrix_power`, `cond`) -- built here at the
// binding layer from complex primitives the core DOES already expose
// (`zc::inv`, `zc::svd`), not a new core algorithm. ───

fn complex_matmul_square(a: &[Complex64], b: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == Complex64::new(0.0, 0.0) {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    out
}

fn complex_matrix_power(a: &[Complex64], n: usize, power: i32) -> Result<Vec<Complex64>, LinAlgError> {
    if n == 0 {
        return Ok(vec![]);
    }
    let base = if power < 0 { zc::inv(a, n)? } else { a.to_vec() };
    let mut p = power.unsigned_abs();
    if p == 0 {
        let mut ident = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            ident[i * n + i] = Complex64::new(1.0, 0.0);
        }
        return Ok(ident);
    }
    let mut result: Option<Vec<Complex64>> = None;
    let mut cur = base;
    while p > 0 {
        if p & 1 == 1 {
            result = Some(match result {
                None => cur.clone(),
                Some(r) => complex_matmul_square(&r, &cur, n),
            });
        }
        p >>= 1;
        if p > 0 {
            cur = complex_matmul_square(&cur, &cur, n);
        }
    }
    Ok(result.unwrap())
}

fn complex_norm_1(a: &[Complex64], n: usize) -> f64 {
    (0..n)
        .map(|j| (0..n).map(|i| a[i * n + j].norm()).sum::<f64>())
        .fold(0.0, f64::max)
}
fn complex_norm_inf(a: &[Complex64], n: usize) -> f64 {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j].norm()).sum::<f64>())
        .fold(0.0, f64::max)
}
/// `p=-1` counterpart of `complex_norm_1` -- added 2026-08-02 for
/// `cond(p=-1)` on complex input.
fn complex_norm_neg1(a: &[Complex64], n: usize) -> f64 {
    (0..n)
        .map(|j| (0..n).map(|i| a[i * n + j].norm()).sum::<f64>())
        .fold(f64::INFINITY, f64::min)
}
/// `p=-inf` counterpart of `complex_norm_inf` -- added 2026-08-02 for
/// `cond(p=-inf)` on complex input.
fn complex_norm_neginf(a: &[Complex64], n: usize) -> f64 {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j].norm()).sum::<f64>())
        .fold(f64::INFINITY, f64::min)
}

/// `cond` for complex input: the core (`dense_linalg.rs`) only implements
/// `real::cond`. Built here directly from `zc::svd`/`zc::inv` (already
/// bound, already tested) -- mirroring `real::cond`'s own branching
/// exactly (verified by reading that fn, see its source and its 2026-08-02
/// doc comment above `real::cond` for why `-1`/`-inf`/`fro` are genuine
/// "not yet implemented" gaps rather than "genuinely unimplementable"
/// ones, now filled in on both the real and complex paths together).
fn complex_cond(a: &[Complex64], n: usize, p: Option<f64>, fro: bool) -> Result<f64, LinAlgError> {
    if fro {
        return match zc::inv(a, n) {
            Ok(ai) => Ok(zc::norm_fro(a) * zc::norm_fro(&ai)),
            Err(LinAlgError::Singular) => Ok(f64::INFINITY),
            Err(e) => Err(e),
        };
    }
    match p {
        None | Some(2.0) => {
            let (_u, s, _vt) = zc::svd(a, n, n)?;
            let smax = s.iter().cloned().fold(0.0f64, f64::max);
            let smin = s.iter().cloned().fold(f64::INFINITY, f64::min);
            if smin == 0.0 {
                Ok(f64::INFINITY)
            } else {
                Ok(smax / smin)
            }
        }
        Some(v) if v == 1.0 => match zc::inv(a, n) {
            Ok(ai) => Ok(complex_norm_1(a, n) * complex_norm_1(&ai, n)),
            Err(LinAlgError::Singular) => Ok(f64::INFINITY),
            Err(e) => Err(e),
        },
        Some(v) if v == -1.0 => match zc::inv(a, n) {
            Ok(ai) => Ok(complex_norm_neg1(a, n) * complex_norm_neg1(&ai, n)),
            Err(LinAlgError::Singular) => Ok(f64::INFINITY),
            Err(e) => Err(e),
        },
        Some(v) if v.is_infinite() && v > 0.0 => match zc::inv(a, n) {
            Ok(ai) => Ok(complex_norm_inf(a, n) * complex_norm_inf(&ai, n)),
            Err(LinAlgError::Singular) => Ok(f64::INFINITY),
            Err(e) => Err(e),
        },
        Some(v) if v.is_infinite() && v < 0.0 => match zc::inv(a, n) {
            Ok(ai) => Ok(complex_norm_neginf(a, n) * complex_norm_neginf(&ai, n)),
            Err(LinAlgError::Singular) => Ok(f64::INFINITY),
            Err(e) => Err(e),
        },
        Some(v) => Err(LinAlgError::InvalidInput(format!("cond: unsupported ord {}", v))),
    }
}

// ─────────────────────────── construction helpers ──────────────────────
//
// All results are built as real `anionpy.ndarray` (via `NdArray::from_buffer`
// + `PyArray { inner }.into_py_any(py)`), not `numpy.ndarray` -- the
// 2026-08-01 fix. Returning numpy here (the original implementation) is
// exactly the other half of the defect `as_real2`/`as_complex2` above fix
// on the input side: it made every `anionpy.linalg.*` result un-chainable
// with any other `anionpy` operation, a numpy *accessory* rather than a numpy
// *replacement*.

fn mk2_f64(py: Python<'_>, data: Vec<f64>, rows: usize, cols: usize) -> PyResult<Py<PyAny>> {
    let inner = NdArray::from_buffer(Buffer::F64(data), vec![rows, cols], Order::C).map_err(linalg_ionp_err)?;
    PyArray { inner }.into_py_any(py)
}
fn mk2_c64(py: Python<'_>, data: Vec<Complex64>, rows: usize, cols: usize) -> PyResult<Py<PyAny>> {
    let inner = NdArray::from_buffer(Buffer::C128(data), vec![rows, cols], Order::C).map_err(linalg_ionp_err)?;
    PyArray { inner }.into_py_any(py)
}
fn mk1_f64(py: Python<'_>, data: Vec<f64>) -> PyResult<Py<PyAny>> {
    let n = data.len();
    let inner = NdArray::from_buffer(Buffer::F64(data), vec![n], Order::C).map_err(linalg_ionp_err)?;
    PyArray { inner }.into_py_any(py)
}
fn mk1_c64(py: Python<'_>, data: Vec<Complex64>) -> PyResult<Py<PyAny>> {
    let n = data.len();
    let inner = NdArray::from_buffer(Buffer::C128(data), vec![n], Order::C).map_err(linalg_ionp_err)?;
    PyArray { inner }.into_py_any(py)
}

/// numpy's `matrix_rank`/`lstsq` don't return a plain Python int for their
/// rank output -- they return a genuine numpy scalar (`np.int64` for
/// `matrix_rank`, `np.int32` for `lstsq`, confirmed empirically against
/// numpy 2.5.1), which carries a `.dtype`. A bare Rust `i32`/`i64` marshaled
/// via `into_py_any` becomes a plain Python `int` with NO `.dtype` attribute
/// at all, which the differential harness (`compare_values`'s `_dtype_of`)
/// reports as a dtype-presence mismatch even when the integer VALUE is
/// correct -- so this constructs a real 0-d `anionpy.ndarray` of the matching
/// dtype instead (shape `()`, `.dtype` present and correct), which
/// `compare_values` treats as equivalent to numpy's own scalar for its
/// dtype/shape/value checks.
fn mk_scalar_i32(py: Python<'_>, v: i32) -> PyResult<Py<PyAny>> {
    let inner = NdArray::from_buffer(Buffer::I32(vec![v]), vec![], Order::C).map_err(linalg_ionp_err)?;
    crate::numpy_scalar_from_0d(py, &inner)
}
fn mk_scalar_i64(py: Python<'_>, v: i64) -> PyResult<Py<PyAny>> {
    let inner = NdArray::from_buffer(Buffer::I64(vec![v]), vec![], Order::C).map_err(linalg_ionp_err)?;
    crate::numpy_scalar_from_0d(py, &inner)
}
fn mk_nd_i64(py: Python<'_>, data: Vec<i64>, shape: Vec<usize>) -> PyResult<Py<PyAny>> {
    let inner = NdArray::from_buffer(Buffer::I64(data), shape, Order::C).map_err(linalg_ionp_err)?;
    PyArray { inner }.into_py_any(py)
}

// ══════════════════════════════ det / slogdet ══════════════════════════

#[pyfunction]
#[pyo3(signature = (a))]
fn det(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Some((data, shape, prec)) = as_promoted_real_nd(a)? {
        let m = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let v = rc::det(&data, m).map_err(linalg_err)?;
            return mk_scalar_f64_prec(py, v, prec);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let slice = &data[b * m * m..(b + 1) * m * m];
            out.push(rc::det(slice, m).map_err(linalg_err)?);
        }
        return mk_nd_f64_prec(py, out, batch_shape, prec);
    }
    if let Some((data, shape, prec)) = as_promoted_complex_nd(a)? {
        let m = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let v = zc::det(&data, m).map_err(linalg_err)?;
            return mk_scalar_c128_prec(py, v, prec);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let slice = &data[b * m * m..(b + 1) * m * m];
            out.push(zc::det(slice, m).map_err(linalg_err)?);
        }
        return mk_nd_c128_prec(py, out, batch_shape, prec);
    }
    Err(unsupported_dtype("det"))
}

#[pyfunction]
#[pyo3(signature = (a))]
fn slogdet(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Some((data, shape, prec)) = as_promoted_real_nd(a)? {
        let m = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let (sign, logdet) = rc::slogdet(&data, m).map_err(linalg_err)?;
            let signpy = mk_scalar_f64_prec(py, sign, prec)?;
            let logdetpy = mk_scalar_f64_prec(py, logdet, prec)?;
            return Ok((signpy, logdetpy).into_py_any(py)?);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut signs = Vec::with_capacity(batch);
        let mut logdets = Vec::with_capacity(batch);
        for b in 0..batch {
            let slice = &data[b * m * m..(b + 1) * m * m];
            let (sign, logdet) = rc::slogdet(slice, m).map_err(linalg_err)?;
            signs.push(sign);
            logdets.push(logdet);
        }
        let signpy = mk_nd_f64_prec(py, signs, batch_shape.clone(), prec)?;
        let logdetpy = mk_nd_f64_prec(py, logdets, batch_shape, prec)?;
        return Ok((signpy, logdetpy).into_py_any(py)?);
    }
    if let Some((data, shape, prec)) = as_promoted_complex_nd(a)? {
        let m = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let (sign, logdet) = zc::slogdet(&data, m).map_err(linalg_err)?;
            let signpy = mk_scalar_c128_prec(py, sign, prec)?;
            let logdetpy = mk_scalar_f64_prec(py, logdet, prec)?;
            return Ok((signpy, logdetpy).into_py_any(py)?);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut signs = Vec::with_capacity(batch);
        let mut logdets = Vec::with_capacity(batch);
        for b in 0..batch {
            let slice = &data[b * m * m..(b + 1) * m * m];
            let (sign, logdet) = zc::slogdet(slice, m).map_err(linalg_err)?;
            signs.push(sign);
            logdets.push(logdet);
        }
        let signpy = mk_nd_c128_prec(py, signs, batch_shape.clone(), prec)?;
        let logdetpy = mk_nd_f64_prec(py, logdets, batch_shape, prec)?;
        return Ok((signpy, logdetpy).into_py_any(py)?);
    }
    Err(unsupported_dtype("slogdet"))
}

// ══════════════════════════════════ inv ═════════════════════════════════

#[pyfunction]
#[pyo3(signature = (a))]
fn inv(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Some((data, shape, prec)) = as_promoted_real_nd(a)? {
        let m = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let out = rc::inv(&data, m).map_err(linalg_err)?;
            return mk2_f64_prec(py, out, m, m, prec);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch * m * m);
        for b in 0..batch {
            let slice = &data[b * m * m..(b + 1) * m * m];
            out.extend(rc::inv(slice, m).map_err(linalg_err)?);
        }
        let mut full_shape = batch_shape;
        full_shape.push(m);
        full_shape.push(m);
        return mk_nd_f64_prec(py, out, full_shape, prec);
    }
    if let Some((data, shape, prec)) = as_promoted_complex_nd(a)? {
        let m = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let out = zc::inv(&data, m).map_err(linalg_err)?;
            return mk2_c64_prec(py, out, m, m, prec);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch * m * m);
        for b in 0..batch {
            let slice = &data[b * m * m..(b + 1) * m * m];
            out.extend(zc::inv(slice, m).map_err(linalg_err)?);
        }
        let mut full_shape = batch_shape;
        full_shape.push(m);
        full_shape.push(m);
        return mk_nd_c128_prec(py, out, full_shape, prec);
    }
    Err(unsupported_dtype("inv"))
}

// ═══════════════════════════════ solve ══════════════════════════════════

/// Real numpy's `solve` broadcasts `a`'s leading (batch) dims against
/// `b`'s -- verified live via `inspect.getsource(numpy.linalg._linalg.solve)`
/// 2026-08-02: the choice between the "vector" gufunc (`(m,m),(m)->(m)`)
/// and the "matrix" gufunc (`(m,m),(m,n)->(m,n)`) is decided SOLELY by
/// `b.ndim == 1` (not by any relationship to `a.ndim`, and not by the
/// historical `b.ndim == a.ndim - 1` rule numpy dropped in 2.0), and once
/// the matrix gufunc is chosen, `a`'s and `b`'s leading dims broadcast
/// against each other with full numpy broadcasting (not just "equal or
/// unbatched"). This local helper mirrors `ionp-py/src/strings.rs`'s
/// `broadcast_shapes` (that one is private to its own module and out of
/// this task's edit scope, so duplicated here rather than imported).
fn solve_broadcast_shapes(a: &[usize], b: &[usize]) -> PyResult<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ia = i + a.len();
        let ib = i + b.len();
        let da = if ia >= n { a[ia - n] } else { 1 };
        let db = if ib >= n { b[ib - n] } else { 1 };
        if da == db {
            out.push(da);
        } else if da == 1 {
            out.push(db);
        } else if db == 1 {
            out.push(da);
        } else {
            return Err(PyValueError::new_err(format!(
                "operands could not be broadcast together with shapes {a:?} {b:?}"
            )));
        }
    }
    Ok(out)
}

/// Flat batch index into an array of shape `shape` (row-major/C-contig,
/// right-aligned against `out_shape` the way numpy broadcasting aligns
/// shapes), given a full multi-index `idx` in `out_shape`'s coordinate
/// space. A `shape` dim of `1` (a broadcast dim) always contributes
/// coordinate `0`, matching numpy's own broadcast-stride-0 semantics.
fn solve_batch_index(idx: &[usize], shape: &[usize], out_shape: &[usize]) -> usize {
    let offset = out_shape.len() - shape.len();
    let mut strides = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    let mut flat = 0usize;
    for i in 0..shape.len() {
        let coord = if shape[i] == 1 { 0 } else { idx[offset + i] };
        flat += coord * strides[i];
    }
    flat
}

fn solve_next_index(idx: &mut [usize], shape: &[usize]) {
    for i in (0..shape.len()).rev() {
        idx[i] += 1;
        if idx[i] < shape[i] {
            return;
        }
        idx[i] = 0;
    }
}

#[pyfunction]
#[pyo3(signature = (a, b))]
fn solve(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Some((adata, ashape, aprec)) = as_promoted_real_nd(a)? {
        let n = check_stacked_square(&ashape)?;
        let a_loop = ashape[..ashape.len() - 2].to_vec();
        if let Some((bdata, bshape, bprec)) = as_promoted_real_nd(b)? {
            let prec = combine_prec(aprec, bprec);
            if bshape.len() == 1 {
                if bshape[0] != n {
                    return Err(PyValueError::new_err(format!(
                        "solve1: Input operand 1 has a mismatch in its core dimension 0, \
                         with gufunc signature (m,m),(m)->(m) (size {} is different from {})",
                        bshape[0], n
                    )));
                }
                let batch: usize = a_loop.iter().product::<usize>().max(1);
                let mut out = Vec::with_capacity(batch * n);
                for i in 0..batch {
                    let aslice = &adata[i * n * n..(i + 1) * n * n];
                    let r = rc::solve(aslice, n, &bdata, 1).map_err(linalg_err)?;
                    out.extend(r);
                }
                let mut shp = a_loop;
                shp.push(n);
                return mk_nd_f64_prec(py, out, shp, prec);
            }
            let k = bshape[bshape.len() - 1];
            let bm = bshape[bshape.len() - 2];
            if bm != n {
                return Err(PyValueError::new_err(format!(
                    "solve: Input operand 1 has a mismatch in its core dimension 0, \
                     with gufunc signature (m,m),(m,n)->(m,n) (size {} is different from {})",
                    bm, n
                )));
            }
            let b_loop = bshape[..bshape.len() - 2].to_vec();
            let loop_shape = solve_broadcast_shapes(&a_loop, &b_loop)?;
            let batch: usize = loop_shape.iter().product::<usize>().max(1);
            let ndim = loop_shape.len();
            let mut idx = vec![0usize; ndim];
            let mut out = Vec::with_capacity(batch * n * k);
            for _ in 0..batch {
                let ai = solve_batch_index(&idx, &a_loop, &loop_shape);
                let bi = solve_batch_index(&idx, &b_loop, &loop_shape);
                let aslice = &adata[ai * n * n..(ai + 1) * n * n];
                let bslice = &bdata[bi * n * k..(bi + 1) * n * k];
                let r = rc::solve(aslice, n, bslice, k).map_err(linalg_err)?;
                out.extend(r);
                solve_next_index(&mut idx, &loop_shape);
            }
            let mut shp = loop_shape;
            shp.push(n);
            shp.push(k);
            return mk_nd_f64_prec(py, out, shp, prec);
        }
        return Err(unsupported_dtype("solve"));
    }
    if let Some((adata, ashape, aprec)) = as_promoted_complex_nd(a)? {
        let n = check_stacked_square(&ashape)?;
        let a_loop = ashape[..ashape.len() - 2].to_vec();
        if let Some((bdata, bshape, bprec)) = as_promoted_complex_nd(b)? {
            let prec = combine_prec(aprec, bprec);
            if bshape.len() == 1 {
                if bshape[0] != n {
                    return Err(PyValueError::new_err(format!(
                        "solve1: Input operand 1 has a mismatch in its core dimension 0, \
                         with gufunc signature (m,m),(m)->(m) (size {} is different from {})",
                        bshape[0], n
                    )));
                }
                let batch: usize = a_loop.iter().product::<usize>().max(1);
                let mut out = Vec::with_capacity(batch * n);
                for i in 0..batch {
                    let aslice = &adata[i * n * n..(i + 1) * n * n];
                    let r = zc::solve(aslice, n, &bdata, 1).map_err(linalg_err)?;
                    out.extend(r);
                }
                let mut shp = a_loop;
                shp.push(n);
                return mk_nd_c128_prec(py, out, shp, prec);
            }
            let k = bshape[bshape.len() - 1];
            let bm = bshape[bshape.len() - 2];
            if bm != n {
                return Err(PyValueError::new_err(format!(
                    "solve: Input operand 1 has a mismatch in its core dimension 0, \
                     with gufunc signature (m,m),(m,n)->(m,n) (size {} is different from {})",
                    bm, n
                )));
            }
            let b_loop = bshape[..bshape.len() - 2].to_vec();
            let loop_shape = solve_broadcast_shapes(&a_loop, &b_loop)?;
            let batch: usize = loop_shape.iter().product::<usize>().max(1);
            let ndim = loop_shape.len();
            let mut idx = vec![0usize; ndim];
            let mut out = Vec::with_capacity(batch * n * k);
            for _ in 0..batch {
                let ai = solve_batch_index(&idx, &a_loop, &loop_shape);
                let bi = solve_batch_index(&idx, &b_loop, &loop_shape);
                let aslice = &adata[ai * n * n..(ai + 1) * n * n];
                let bslice = &bdata[bi * n * k..(bi + 1) * n * k];
                let r = zc::solve(aslice, n, bslice, k).map_err(linalg_err)?;
                out.extend(r);
                solve_next_index(&mut idx, &loop_shape);
            }
            let mut shp = loop_shape;
            shp.push(n);
            shp.push(k);
            return mk_nd_c128_prec(py, out, shp, prec);
        }
        return Err(unsupported_dtype("solve"));
    }
    Err(unsupported_dtype("solve"))
}

// ═════════════════════════════ cholesky ═════════════════════════════════

#[pyfunction]
#[pyo3(signature = (a, upper=None))]
fn cholesky(py: Python<'_>, a: &Bound<'_, PyAny>, upper: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    // `upper=` is plain Python truthiness in real numpy (measured
    // 2026-08-02, same mechanism as `svd`'s `full_matrices`/`compute_uv`/
    // `hermitian` fixed in 58d5e6a): `np.linalg.cholesky(a, upper=1.5)`,
    // `upper="x"`, `upper=[1]` etc all dispatch on `if upper:`, not
    // `isinstance(upper, bool)`. PyO3's strict `bool` extraction rejected
    // all of these with `TypeError` where numpy accepts them.
    let upper = match upper {
        None => false,
        Some(v) => v.is_truthy()?,
    };
    if let Some((data, shape, prec)) = as_promoted_real_nd(a)? {
        let n = check_stacked_square(&shape)?;
        let transform = |l: &[f64]| -> Vec<f64> {
            if !upper {
                return l.to_vec();
            }
            // upper=True: numpy returns L^T (a genuine transpose of the
            // same lower factor the core computes), not an independently
            // computed upper factorization -- verified this is what real
            // numpy does (cholesky(upper=True) satisfies A = U^T U with
            // U = L^T).
            let mut u = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    u[i * n + j] = l[j * n + i];
                }
            }
            u
        };
        if shape.len() == 2 {
            let l = rc::cholesky(&data, n).map_err(linalg_err)?;
            return mk2_f64_prec(py, transform(&l), n, n, prec);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let slice = &data[b * n * n..(b + 1) * n * n];
            let l = rc::cholesky(slice, n).map_err(linalg_err)?;
            out.extend(transform(&l));
        }
        let mut full_shape = batch_shape;
        full_shape.push(n);
        full_shape.push(n);
        return mk_nd_f64_prec(py, out, full_shape, prec);
    }
    if let Some((data, shape, prec)) = as_promoted_complex_nd(a)? {
        let n = check_stacked_square(&shape)?;
        let transform = |l: &[Complex64]| -> Vec<Complex64> {
            if !upper {
                return l.to_vec();
            }
            // upper=True (complex): A = U^H U with U = L^H (conjugate
            // transpose).
            let mut u = vec![Complex64::new(0.0, 0.0); n * n];
            for i in 0..n {
                for j in 0..n {
                    u[i * n + j] = l[j * n + i].conj();
                }
            }
            u
        };
        if shape.len() == 2 {
            let l = zc::cholesky(&data, n).map_err(linalg_err)?;
            return mk2_c64_prec(py, transform(&l), n, n, prec);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let slice = &data[b * n * n..(b + 1) * n * n];
            let l = zc::cholesky(slice, n).map_err(linalg_err)?;
            out.extend(transform(&l));
        }
        let mut full_shape = batch_shape;
        full_shape.push(n);
        full_shape.push(n);
        return mk_nd_c128_prec(py, out, full_shape, prec);
    }
    Err(unsupported_dtype("cholesky"))
}

// ═══════════════════════════ matrix_power ═══════════════════════════════

#[pyfunction]
#[pyo3(signature = (a, n))]
fn matrix_power(py: Python<'_>, a: &Bound<'_, PyAny>, n: i32) -> PyResult<Py<PyAny>> {
    // numpy preserves the ORIGINAL input dtype for n>=0 -- verified against
    // real numpy 2.5.1: `matrix_power` on bool/int8/int32/int64/uint8/
    // uint32/float16/float32/float64 all round-trip their own dtype for
    // n=0 (identity) and n>=1 (repeated matmul); float16 is NOT rejected
    // for n>=0 either. This is UNLIKE every other linalg function in this
    // file (and unlike this same function's own n<0 branch below), which
    // promote to float64/complex128. So n>=0 is handled natively here via
    // `ionp_ion::matmul::matmul` (which itself already preserves dtype per
    // ufunc semantics), bypassing `as_promoted_real2`/`reject_float16`
    // entirely -- only n<0 (which genuinely needs `inv`, and DOES match
    // numpy's usual float64-promotion / float16-rejection rules, verified
    // separately) still routes through the promoted path.
    // numpy's `matrix_power` runs `_assert_stacked_2d` (ndim < 2 -> distinct
    // "{ndim}-dimensional array given. Array must be at least two-dimensional"
    // LinAlgError) BEFORE `_assert_stacked_square` (non-square trailing 2 dims
    // -> "Last 2 dimensions of the array must be square") -- verified against
    // real numpy 2.5.1 that these are two DIFFERENT messages for two DIFFERENT
    // conditions; the previous code collapsed both into the square message,
    // which is wrong for a true 1-D/0-D input. `_assert_stacked_2d` allows
    // ndim > 2 (batched/stacked matrices) too, which this file's 2-D-only
    // extraction helpers do not support -- that is a separate, pre-existing,
    // out-of-scope architectural gap (shared with `svd`, reported honestly
    // rather than silently worked around), so ndim > 2 still hits the same
    // "must be square"-shaped rejection path below via the shape[0]!=shape[1]
    // check once ndim==2 is otherwise established; this fix only corrects the
    // ndim<2 message class/text, not the batching gap itself.
    if n >= 0 {
        let arr = as_ndarray_any(a)?;
        let shape = arr.shape().to_vec();
        if shape.len() < 2 {
            return Err(linalg_err_raw(&format!(
                "{}-dimensional array given. Array must be at least two-dimensional",
                shape.len()
            )));
        }
        // numpy's `_assert_stacked_square` checks only the LAST 2 dimensions
        // -- `matrix_power` supports arbitrary batch/stack dimensions ahead
        // of them (verified: `np.linalg.matrix_power(np.zeros((2,0,0)), 2)`
        // succeeds with output shape (2,0,0), a batch of 2 empty 0x0
        // matrices). The n>=0 path here already delegates squaring to
        // `ionp_ion::matmul::matmul`, which is itself already batch-general
        // (verified separately: `anionpy.matmul` on two (2,3,3) operands
        // succeeds), so only this validation and the n==0 identity
        // construction below needed to stop assuming exactly-2-D.
        if shape[shape.len() - 2] != shape[shape.len() - 1] {
            return Err(linalg_err_raw("Last 2 dimensions of the array must be square"));
        }
        let m = shape[shape.len() - 1];
        if n == 0 {
            let ident_f64 = ionp_core::creation::eye_values(m, m, 0);
            let ident2d = NdArray::from_buffer(Buffer::F64(ident_f64), vec![m, m], Order::C)
                .map_err(linalg_ionp_err)?
                .cast_to(arr.dtype());
            // Broadcast the single (m, m) identity up to the full batch
            // shape (batch_dims..., m, m) -- `n == 0` needs a per-batch-
            // element identity, not just one, for any batch dim count != 1.
            let ident = if shape.len() == 2 {
                ident2d
            } else {
                ionp_core::creation::broadcast_to(&ident2d, &shape)
                    .map_err(linalg_ionp_err)?
                    .to_contiguous()
            };
            return Ok(Py::new(py, PyArray { inner: ident })?.into_any());
        }
        // Repeated squaring, native dtype throughout, using the same
        // `ionp_ion::matmul::matmul` binding the top-level `matmul`
        // ufunc uses.
        let mut acc: Option<NdArray> = None;
        let mut base = arr.to_contiguous();
        let mut k = n as u32;
        while k > 0 {
            if k & 1 == 1 {
                acc = Some(match acc {
                    None => base.clone(),
                    Some(r) => ionp_ion::matmul::matmul(&r, &base).map_err(PyValueError::new_err)?,
                });
            }
            k >>= 1;
            if k > 0 {
                base = ionp_ion::matmul::matmul(&base, &base).map_err(PyValueError::new_err)?;
            }
        }
        let out = acc.expect("n >= 1 guarantees at least one squaring iteration ran");
        return Ok(Py::new(py, PyArray { inner: out })?.into_any());
    }
    // Same `_assert_stacked_2d`-then-`_assert_stacked_square` ordering as the
    // n>=0 branch above, applied before falling into the promoted-2-D
    // extraction helpers (whose own ndim!=2 check raises the generic,
    // non-numpy-matching "expected a 2-D array..." message, and which never
    // checks squareness at all -- `rc::matrix_power`/`complex_matrix_power`
    // both require a square matrix for the n<0 inverse-power path).
    if let Some(arr) = try_ndarray_any(a) {
        let shape = arr.shape().to_vec();
        if shape.len() < 2 {
            return Err(linalg_err_raw(&format!(
                "{}-dimensional array given. Array must be at least two-dimensional",
                shape.len()
            )));
        }
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(linalg_err_raw("Last 2 dimensions of the array must be square"));
        }
    }
    if let Some((data, m, _, prec)) = as_promoted_real2(a)? {
        let out = rc::matrix_power(&data, m, n).map_err(linalg_err)?;
        return mk2_f64_prec(py, out, m, m, prec);
    }
    if let Some((data, m, _, prec)) = as_promoted_complex2(a)? {
        let out = complex_matrix_power(&data, m, n).map_err(linalg_err)?;
        return mk2_c64_prec(py, out, m, m, prec);
    }
    Err(unsupported_dtype("matrix_power"))
}

// ═══════════════════════════ eig / eigvals ══════════════════════════════

#[pyfunction]
#[pyo3(signature = (a))]
fn eig(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Was `as_promoted_real2`/`as_promoted_complex2` (2-D-only, discarded
    // the second dim entirely -- `(data, n, _, prec)` bound the SECOND
    // tuple element as "n" and threw the actual column count away). This
    // let `eig` reach `rc::eig`/`zc::eig` -> `dense_linalg.rs`'s
    // `to_col_major`/`_c` with `rows=cols=n` for genuinely non-square
    // input, which is exactly the debug_assert-guarded (release-mode-live)
    // panic this task's audit was asked to find: confirmed directly,
    // `eig(zeros((3,2)))` panics at dense_linalg.rs:94 ("index out of
    // bounds: the len is 6 but the index is 6") on this build, and
    // `eig(zeros((2,3)))` silently computed a WRONG answer from a
    // truncated read of the buffer instead of panicking or raising. Fixed
    // with the same `check_stacked_square` + N-D batching pattern already
    // applied to `eigvals`/`eigvalsh`/etc. this task.
    if let Some((data, shape, prec)) = as_promoted_real_nd(a)? {
        let n = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let (w, v) = rc::eig(&data, n).map_err(linalg_err)?;
            let wpy = mk1_c64_prec(py, w, prec)?;
            let vpy = mk2_c64_prec(py, v, n, n, prec)?;
            return Ok((wpy, vpy).into_py_any(py)?);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut wout = Vec::with_capacity(batch * n);
        let mut vout = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let slice = &data[b * n * n..(b + 1) * n * n];
            let (w, v) = rc::eig(slice, n).map_err(linalg_err)?;
            wout.extend(w);
            vout.extend(v);
        }
        let mut wshape = batch_shape.clone();
        wshape.push(n);
        let mut vshape = batch_shape;
        vshape.push(n);
        vshape.push(n);
        let wpy = mk_nd_c128_prec(py, wout, wshape, prec)?;
        let vpy = mk_nd_c128_prec(py, vout, vshape, prec)?;
        return Ok((wpy, vpy).into_py_any(py)?);
    }
    if let Some((data, shape, prec)) = as_promoted_complex_nd(a)? {
        let n = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let (w, v) = zc::eig(&data, n).map_err(linalg_err)?;
            let wpy = mk1_c64_prec(py, w, prec)?;
            let vpy = mk2_c64_prec(py, v, n, n, prec)?;
            return Ok((wpy, vpy).into_py_any(py)?);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut wout = Vec::with_capacity(batch * n);
        let mut vout = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let slice = &data[b * n * n..(b + 1) * n * n];
            let (w, v) = zc::eig(slice, n).map_err(linalg_err)?;
            wout.extend(w);
            vout.extend(v);
        }
        let mut wshape = batch_shape.clone();
        wshape.push(n);
        let mut vshape = batch_shape;
        vshape.push(n);
        vshape.push(n);
        let wpy = mk_nd_c128_prec(py, wout, wshape, prec)?;
        let vpy = mk_nd_c128_prec(py, vout, vshape, prec)?;
        return Ok((wpy, vpy).into_py_any(py)?);
    }
    Err(unsupported_dtype("eig"))
}

#[pyfunction]
#[pyo3(signature = (a))]
fn eigvals(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Some((data, shape, prec)) = as_promoted_real_nd(a)? {
        let n = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let w = rc::eigvals(&data, n).map_err(linalg_err)?;
            return mk1_c64_prec(py, w, prec);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch * n);
        for b in 0..batch {
            let slice = &data[b * n * n..(b + 1) * n * n];
            out.extend(rc::eigvals(slice, n).map_err(linalg_err)?);
        }
        let mut full_shape = batch_shape;
        full_shape.push(n);
        return mk_nd_c128_prec(py, out, full_shape, prec);
    }
    if let Some((data, shape, prec)) = as_promoted_complex_nd(a)? {
        let n = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let w = zc::eigvals(&data, n).map_err(linalg_err)?;
            return mk1_c64_prec(py, w, prec);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch * n);
        for b in 0..batch {
            let slice = &data[b * n * n..(b + 1) * n * n];
            out.extend(zc::eigvals(slice, n).map_err(linalg_err)?);
        }
        let mut full_shape = batch_shape;
        full_shape.push(n);
        return mk_nd_c128_prec(py, out, full_shape, prec);
    }
    Err(unsupported_dtype("eigvals"))
}

// ═══════════════════════════ eigh / eigvalsh ════════════════════════════

#[pyfunction]
#[allow(non_snake_case)]
#[pyo3(signature = (*args, **kwargs))]
fn eigh(py: Python<'_>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    // Raw `*args`/`**kwargs` capture (mirrors `pinv` above): `UPLO` ABSENT
    // and `UPLO=None` explicit are two DIFFERENT reachable states here, not
    // the same one -- a plain typed `UPLO: Option<&Bound<PyAny>> = None`
    // pyo3 parameter cannot tell an omitted argument apart from an
    // explicitly-passed Python `None` (pyo3 maps both to Rust `None`
    // identically), yet real numpy's `UPLO.upper()` treats them completely
    // differently: omitted defaults to `'L'` and never calls `.upper()` at
    // all, while `UPLO=None` explicit calls `None.upper()` and raises
    // `AttributeError` (verified live, and by the coordinator's
    // /tmp/uplocheck.py grid this fix responds to).
    const NAMES: [&str; 2] = ["a", "UPLO"];
    if args.len() > NAMES.len() {
        return Err(PyTypeError::new_err(format!(
            "eigh() takes from 1 to {} positional arguments but {} were given",
            NAMES.len(),
            args.len()
        )));
    }
    let mut slots: [Option<Bound<'_, PyAny>>; 2] = [None, None];
    for (i, item) in args.iter().enumerate() {
        slots[i] = Some(item);
    }
    if let Some(kw) = kwargs {
        for (idx, name) in NAMES.iter().enumerate() {
            if let Some(v) = kw.get_item(name)? {
                if slots[idx].is_some() {
                    return Err(PyTypeError::new_err(format!(
                        "eigh() got multiple values for argument '{name}'"
                    )));
                }
                slots[idx] = Some(v);
            }
        }
    }
    let [a_slot, uplo_slot] = slots;
    let a_owned = a_slot
        .ok_or_else(|| PyTypeError::new_err("eigh() missing 1 required positional argument: 'a'"))?;
    let a: &Bound<'_, PyAny> = &a_owned;
    // numpy's real body is exactly `UPLO = UPLO.upper(); if UPLO not in
    // ('L', 'U'): raise ValueError(...)` (verified via `inspect.getsource`)
    // -- it does NOT type-check UPLO first, it just calls `.upper()` on
    // whatever was passed and lets that fail on its own. That leaks THREE
    // observable behaviors a drop-in replacement must reproduce, not
    // improve on: (1) `UPLO='l'`/`'u'` (or any object whose `.upper()`
    // returns 'L'/'U') is silently ACCEPTED, same as 'L'/'U'; (2) a
    // non-string lacking `.upper()` (`None`, `0`, ...) raises numpy's own
    // generic `AttributeError` from the failed attribute lookup itself,
    // not a hand-written type check; (3) an object that HAS `.upper()`
    // but whose result is not `'L'`/`'U'` (e.g. `bytes`/`bytearray`,
    // whose `.upper()` succeeds and returns `b'L'`/`b'U'`, which then
    // compares UNEQUAL to the str `'L'`/`'U'` via `not in ('L', 'U')`)
    // must fall through to the SAME `ValueError` as an invalid string --
    // never a type error. This third case (round 3, 2026-08-02 coordinator
    // grid `/tmp/uplovals.py`: 40 eigh + 20 eigvalsh mismatches) was missed
    // by an earlier draft that extracted the `.upper()` result straight to
    // a Rust `String` -- that extraction itself raises `TypeError` for a
    // non-str return, which numpy never does: numpy only ever compares
    // (`==`/`in`), it never coerces the `.upper()` result to `str`. So the
    // call itself may raise (only `AttributeError`, only when `.upper()`
    // genuinely doesn't exist), but a *successful* call has exactly two
    // outcomes -- matches 'L'/'U' (via Python `==`, not Rust string
    // equality after a fallible extract) or falls into `ValueError` --
    // never a third, TypeError, outcome. ABSENT (uplo_slot is `None` here
    // because the ARGUMENT was never supplied, not because `None` was
    // passed) defaults straight to `"L"` without ever calling `.upper()`,
    // matching numpy's own `UPLO='L'` default exactly.
    let uplo: &str = match &uplo_slot {
        Some(obj) => {
            let upped = obj.call_method0("upper")?;
            if upped.eq("L")? {
                "L"
            } else if upped.eq("U")? {
                "U"
            } else {
                // Exact numpy message text (verified: `numpy.linalg.eigh(x,
                // UPLO='X')` raises `ValueError("UPLO argument must be 'L'
                // or 'U'")` -- NO interpolated `got` value).
                return Err(PyValueError::new_err("UPLO argument must be 'L' or 'U'"));
            }
        }
        None => "L",
    };
    let UPLO: &str = uplo;
    // Same defect as `eig` above (fixed this task): was 2-D-only
    // `as_promoted_real2`/`as_promoted_complex2` with the real column count
    // discarded, reachable with a genuinely non-square shape and
    // panicking/silently-wrong via `dense_linalg.rs`'s `to_col_major`/`_c`
    // debug_assert-guarded invariant. Fixed with `check_stacked_square` +
    // N-D batching, matching `eigvalsh`.
    //
    // `UPLO='U'` (2026-08-02): was previously a blanket
    // `NotImplementedError` on this function specifically, even though
    // `eigvalsh` right below already implements and documents the exact
    // transpose-input trick this needs (see that function's doc comment
    // for the full algebraic proof: `lower((conjugate-)transpose(a))`'s
    // mirrored effective matrix is IDENTICAL, entry-for-entry, to the
    // effective Hermitian matrix numpy's `UPLO='U'` builds from `a`'s
    // upper triangle). That proof is about the EFFECTIVE MATRIX being
    // fed to the eigendecomposition, not merely its eigenvalues, so it
    // extends unchanged to the full `eigh` (eigenvectors too, not just
    // `eigvalsh`'s eigenvalues) -- the eigenvectors of an identical
    // effective matrix are identical regardless of which LAPACK entry
    // point (`syevd`/`heevd` here) computes them.
    if let Some((data, shape, prec)) = as_promoted_real_nd(a)? {
        let n = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let src = if UPLO == "U" { transpose_f64(&data, n, n) } else { data };
            let (w, v) = rc::eigh(&src, n).map_err(linalg_err)?;
            let wpy = mk1_f64_prec(py, w, prec)?;
            let vpy = mk2_f64_prec(py, v, n, n, prec)?;
            return Ok((wpy, vpy).into_py_any(py)?);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut wout = Vec::with_capacity(batch * n);
        let mut vout = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let slice = &data[b * n * n..(b + 1) * n * n];
            let src = if UPLO == "U" { transpose_f64(slice, n, n) } else { slice.to_vec() };
            let (w, v) = rc::eigh(&src, n).map_err(linalg_err)?;
            wout.extend(w);
            vout.extend(v);
        }
        let mut wshape = batch_shape.clone();
        wshape.push(n);
        let mut vshape = batch_shape;
        vshape.push(n);
        vshape.push(n);
        let wpy = mk_nd_f64_prec(py, wout, wshape, prec)?;
        let vpy = mk_nd_f64_prec(py, vout, vshape, prec)?;
        return Ok((wpy, vpy).into_py_any(py)?);
    }
    if let Some((data, shape, prec)) = as_promoted_complex_nd(a)? {
        let n = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let src = if UPLO == "U" { conj_transpose_c128(&data, n, n) } else { data };
            let (w, v) = zc::eigh(&src, n).map_err(linalg_err)?;
            let wpy = mk1_f64_prec(py, w, prec)?;
            let vpy = mk2_c64_prec(py, v, n, n, prec)?;
            return Ok((wpy, vpy).into_py_any(py)?);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut wout = Vec::with_capacity(batch * n);
        let mut vout = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let slice = &data[b * n * n..(b + 1) * n * n];
            let src = if UPLO == "U" { conj_transpose_c128(slice, n, n) } else { slice.to_vec() };
            let (w, v) = zc::eigh(&src, n).map_err(linalg_err)?;
            wout.extend(w);
            vout.extend(v);
        }
        let mut wshape = batch_shape.clone();
        wshape.push(n);
        let mut vshape = batch_shape;
        vshape.push(n);
        vshape.push(n);
        let wpy = mk_nd_f64_prec(py, wout, wshape, prec)?;
        let vpy = mk_nd_c128_prec(py, vout, vshape, prec)?;
        return Ok((wpy, vpy).into_py_any(py)?);
    }
    Err(unsupported_dtype("eigh"))
}

/// `UPLO='U'`: the core (`rc::eigvalsh`/`zc::eigvalsh`) always reads the
/// LOWER triangle. 2026-08-01 fix, proved algebraically (see this task's
/// report): for any square `a`, the symmetric/Hermitian matrix numpy's
/// `UPLO='U'` implicitly builds by mirroring `a`'s upper triangle
/// (`A_eff[i,j] = a[i,j]` for `i<=j`, `= conj(a[j,i])` for `i>j`) is
/// EXACTLY the same effective matrix the core builds by reading the LOWER
/// triangle of `a`'s (conjugate-)transpose: `lower(a^H)[i,j] = conj(a[j,i])`
/// for `i>=j`, which mirrors to `conj(a[j,i])` for `i>j` and `a[i,i]` on
/// the diagonal (real-valued for a valid Hermitian input) -- identical to
/// `A_eff` at every entry. So `UPLO='U'` is bound by transposing (real) /
/// conjugate-transposing (complex) the input before handing it to the
/// existing lower-triangle-only core, not a second LAPACK code path.
#[pyfunction]
#[allow(non_snake_case)]
#[pyo3(signature = (*args, **kwargs))]
fn eigvalsh(py: Python<'_>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    // Same fix, same reason, as `eigh` above (2026-08-02, coordinator
    // ruling) -- measured this function independently before fixing (not
    // assumed clean from eigh's fix): `UPLO=None`/`UPLO=0` raised this
    // crate's own hand-written `TypeError: 'None' is not an instance of
    // 'str'`/`'int' object is not an instance of 'str'` where real numpy
    // raises `AttributeError: 'NoneType'/'int' object has no attribute
    // 'upper'`, and `UPLO='l'`/`'u'` were wrongly rejected where numpy
    // silently accepts them (`UPLO.upper()` then membership-checks). Same
    // raw `*args`/`**kwargs` ABSENT-vs-None disambiguation as `eigh` above
    // (a plain `Option<&Bound<PyAny>>` typed parameter cannot tell the two
    // apart, and they are observably different states here).
    const NAMES: [&str; 2] = ["a", "UPLO"];
    if args.len() > NAMES.len() {
        return Err(PyTypeError::new_err(format!(
            "eigvalsh() takes from 1 to {} positional arguments but {} were given",
            NAMES.len(),
            args.len()
        )));
    }
    let mut slots: [Option<Bound<'_, PyAny>>; 2] = [None, None];
    for (i, item) in args.iter().enumerate() {
        slots[i] = Some(item);
    }
    if let Some(kw) = kwargs {
        for (idx, name) in NAMES.iter().enumerate() {
            if let Some(v) = kw.get_item(name)? {
                if slots[idx].is_some() {
                    return Err(PyTypeError::new_err(format!(
                        "eigvalsh() got multiple values for argument '{name}'"
                    )));
                }
                slots[idx] = Some(v);
            }
        }
    }
    let [a_slot, uplo_slot] = slots;
    let a_owned = a_slot.ok_or_else(|| {
        PyTypeError::new_err("eigvalsh() missing 1 required positional argument: 'a'")
    })?;
    let a: &Bound<'_, PyAny> = &a_owned;
    // Same three-way duck-typed dispatch as `eigh` above (see that
    // function's doc comment for the full rationale, including the
    // `bytes`/`bytearray` case that a naive `.extract::<String>()` gets
    // wrong by raising `TypeError` where numpy raises `ValueError`):
    // a successful `.upper()` call has exactly two outcomes -- matches
    // 'L'/'U' via Python `==`, or falls into the same `ValueError` as an
    // invalid string, never a third (type-error) outcome. The call
    // itself may only raise `AttributeError`, only for objects that
    // genuinely lack `.upper()`. Measured directly for `eigvalsh` (not
    // assumed from `eigh`'s fix) per 2026-08-02 coordinator ruling.
    let uplo: &str = match &uplo_slot {
        Some(obj) => {
            let upped = obj.call_method0("upper")?;
            if upped.eq("L")? {
                "L"
            } else if upped.eq("U")? {
                "U"
            } else {
                return Err(PyValueError::new_err("UPLO argument must be 'L' or 'U'"));
            }
        }
        None => "L",
    };
    let UPLO: &str = uplo;
    if let Some((data, shape, prec)) = as_promoted_real_nd(a)? {
        let n = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let src = if UPLO == "U" { transpose_f64(&data, n, n) } else { data };
            let w = rc::eigvalsh(&src, n).map_err(linalg_err)?;
            return mk1_f64_prec(py, w, prec);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch * n);
        for b in 0..batch {
            let slice = &data[b * n * n..(b + 1) * n * n];
            let src = if UPLO == "U" { transpose_f64(slice, n, n) } else { slice.to_vec() };
            out.extend(rc::eigvalsh(&src, n).map_err(linalg_err)?);
        }
        let mut full_shape = batch_shape;
        full_shape.push(n);
        return mk_nd_f64_prec(py, out, full_shape, prec);
    }
    if let Some((data, shape, prec)) = as_promoted_complex_nd(a)? {
        let n = check_stacked_square(&shape)?;
        if shape.len() == 2 {
            let src = if UPLO == "U" { conj_transpose_c128(&data, n, n) } else { data };
            let w = zc::eigvalsh(&src, n).map_err(linalg_err)?;
            return mk1_f64_prec(py, w, prec);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch * n);
        for b in 0..batch {
            let slice = &data[b * n * n..(b + 1) * n * n];
            let src = if UPLO == "U" { conj_transpose_c128(slice, n, n) } else { slice.to_vec() };
            out.extend(zc::eigvalsh(&src, n).map_err(linalg_err)?);
        }
        let mut full_shape = batch_shape;
        full_shape.push(n);
        return mk_nd_f64_prec(py, out, full_shape, prec);
    }
    Err(unsupported_dtype("eigvalsh"))
}

// ═══════════════════════════════ qr ═════════════════════════════════════

#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn qr(py: Python<'_>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    // `mode='r'`/`'complete'`/`'raw'` -- added 2026-08-02. Real numpy's
    // `qr` supports all 4 (verified against real numpy 2.5.1, including
    // N-D batching for every mode: `qr(stack_of_matrices, mode='raw')`
    // returns `h.shape==(batch...,N,M)`, `tau.shape==(batch...,K)`, etc).
    // These were previously a blanket `NotImplementedError` even though
    // the only missing pieces were: (a) three new core LAPACK-wrapper
    // functions (`qr_r`/`qr_complete`/`qr_raw`, added to `dense_linalg.rs`
    // alongside the existing `qr`, reusing the identical `dgeqrf`/`zgeqrf`
    // step) and (b) this dispatch -- not a genuine gap.
    //
    // `mode` type handling rewritten 2026-08-02 (coordinator's UPLO round
    // 3 survey: "check whether any OTHER string-valued parameter in
    // linalg.rs has the same defect"). numpy's real body is `if mode not
    // in ('reduced', 'complete', 'r', 'raw'): ... raise ValueError(
    // f"Unrecognized mode '{mode}'")` -- a plain `==`/`in` MEMBERSHIP
    // test, never a method call on `mode`, so it accepts (rejects, really)
    // ANY object with zero prior type check: measured directly, passing
    // `mode=None`/`0`/`b"reduced"`/`["reduced"]`/`1.5` all produce numpy's
    // `ValueError("Unrecognized mode '<str(mode)>'")` (the message
    // interpolates `str(mode)`, NOT `repr(mode)` -- verified: `mode=None`
    // gives the message text `"Unrecognized mode 'None'"`, and
    // `mode=['reduced']` gives `"Unrecognized mode '['reduced']'"`, both
    // consistent only with literal quote characters wrapped around
    // `str(mode)`, not `!r`). The previous `mode: &str` signature let
    // PyO3's own extraction raise a hand-shaped `TypeError` for any
    // non-str value instead -- same defect class as `eigh`/`eigvalsh`'s
    // `UPLO`, different mechanism (membership test, not a duck-typed
    // method call). Raw `*args`/`**kwargs` capture used here (not
    // `Option<&PyAny>` with a `None` default) for the same reason as
    // `eigh`: `mode=None` is a normal, distinct, ERRORING value in numpy,
    // not a signal to fall back to the `"reduced"` default -- an
    // `Option<T>` parameter cannot tell "omitted" from "explicit None"
    // apart, and would wrongly treat explicit `mode=None` as the default.
    const NAMES: [&str; 2] = ["a", "mode"];
    if args.len() > NAMES.len() {
        return Err(PyTypeError::new_err(format!(
            "qr() takes from 1 to {} positional arguments but {} were given",
            NAMES.len(),
            args.len()
        )));
    }
    let mut slots: [Option<Bound<'_, PyAny>>; 2] = [None, None];
    for (i, item) in args.iter().enumerate() {
        slots[i] = Some(item);
    }
    if let Some(kw) = kwargs {
        for (idx, name) in NAMES.iter().enumerate() {
            if let Some(v) = kw.get_item(name)? {
                if slots[idx].is_some() {
                    return Err(PyTypeError::new_err(format!(
                        "qr() got multiple values for argument '{name}'"
                    )));
                }
                slots[idx] = Some(v);
            }
        }
    }
    let [a_slot, mode_slot] = slots;
    let a_owned =
        a_slot.ok_or_else(|| PyTypeError::new_err("qr() missing 1 required positional argument: 'a'"))?;
    let a: &Bound<'_, PyAny> = &a_owned;
    let mode: &str = match &mode_slot {
        None => "reduced",
        Some(obj) => {
            if obj.eq("reduced")? {
                "reduced"
            } else if obj.eq("r")? {
                "r"
            } else if obj.eq("complete")? {
                "complete"
            } else if obj.eq("raw")? {
                "raw"
            } else {
                // `str(mode)`, computed through Python itself (not Rust
                // `Display`), so a list/bytes/etc. formats exactly like
                // numpy's own f-string would (e.g. `str(['reduced'])`,
                // `str(b'reduced')`), not a Rust-side guess.
                let rendered = obj.str()?.to_string();
                return Err(PyValueError::new_err(format!("Unrecognized mode '{rendered}'")));
            }
        }
    };
    if let Some((data, shape, prec)) = as_promoted_real_nd(a)? {
        check_stacked_2d(&shape)?;
        let m = shape[shape.len() - 2];
        let n = shape[shape.len() - 1];
        let k = m.min(n);
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        match mode {
            "r" => {
                if shape.len() == 2 {
                    let r = rc::qr_r(&data, m, n).map_err(linalg_err)?;
                    return mk2_f64_prec(py, r, k, n, prec);
                }
                let mut r_all = Vec::with_capacity(batch * k * n);
                for b in 0..batch {
                    let slice = &data[b * m * n..(b + 1) * m * n];
                    r_all.extend(rc::qr_r(slice, m, n).map_err(linalg_err)?);
                }
                let mut r_shape = batch_shape;
                r_shape.push(k);
                r_shape.push(n);
                return mk_nd_f64_prec(py, r_all, r_shape, prec);
            }
            "complete" => {
                if shape.len() == 2 {
                    let (q, r) = rc::qr_complete(&data, m, n).map_err(linalg_err)?;
                    let qpy = mk2_f64_prec(py, q, m, m, prec)?;
                    let rpy = mk2_f64_prec(py, r, m, n, prec)?;
                    return Ok((qpy, rpy).into_py_any(py)?);
                }
                let mut q_all = Vec::with_capacity(batch * m * m);
                let mut r_all = Vec::with_capacity(batch * m * n);
                for b in 0..batch {
                    let slice = &data[b * m * n..(b + 1) * m * n];
                    let (q, r) = rc::qr_complete(slice, m, n).map_err(linalg_err)?;
                    q_all.extend(q);
                    r_all.extend(r);
                }
                let mut q_shape = batch_shape.clone();
                q_shape.push(m);
                q_shape.push(m);
                let mut r_shape = batch_shape;
                r_shape.push(m);
                r_shape.push(n);
                let qpy = mk_nd_f64_prec(py, q_all, q_shape, prec)?;
                let rpy = mk_nd_f64_prec(py, r_all, r_shape, prec)?;
                return Ok((qpy, rpy).into_py_any(py)?);
            }
            "raw" => {
                if shape.len() == 2 {
                    let (h, tau) = rc::qr_raw(&data, m, n).map_err(linalg_err)?;
                    let hpy = mk2_f64_prec(py, h, n, m, prec)?;
                    let taupy = mk1_f64_prec(py, tau, prec)?;
                    return Ok((hpy, taupy).into_py_any(py)?);
                }
                let mut h_all = Vec::with_capacity(batch * m * n);
                let mut tau_all = Vec::with_capacity(batch * k);
                for b in 0..batch {
                    let slice = &data[b * m * n..(b + 1) * m * n];
                    let (h, tau) = rc::qr_raw(slice, m, n).map_err(linalg_err)?;
                    h_all.extend(h);
                    tau_all.extend(tau);
                }
                let mut h_shape = batch_shape.clone();
                h_shape.push(n);
                h_shape.push(m);
                let mut tau_shape = batch_shape;
                tau_shape.push(k);
                let hpy = mk_nd_f64_prec(py, h_all, h_shape, prec)?;
                let taupy = mk_nd_f64_prec(py, tau_all, tau_shape, prec)?;
                return Ok((hpy, taupy).into_py_any(py)?);
            }
            _ => {
                if shape.len() == 2 {
                    let (q, r) = rc::qr(&data, m, n).map_err(linalg_err)?;
                    let qpy = mk2_f64_prec(py, q, m, k, prec)?;
                    let rpy = mk2_f64_prec(py, r, k, n, prec)?;
                    return Ok((qpy, rpy).into_py_any(py)?);
                }
                let mut q_all = Vec::with_capacity(batch * m * k);
                let mut r_all = Vec::with_capacity(batch * k * n);
                for b in 0..batch {
                    let slice = &data[b * m * n..(b + 1) * m * n];
                    let (q, r) = rc::qr(slice, m, n).map_err(linalg_err)?;
                    q_all.extend(q);
                    r_all.extend(r);
                }
                let mut q_shape = batch_shape.clone();
                q_shape.push(m);
                q_shape.push(k);
                let mut r_shape = batch_shape;
                r_shape.push(k);
                r_shape.push(n);
                let qpy = mk_nd_f64_prec(py, q_all, q_shape, prec)?;
                let rpy = mk_nd_f64_prec(py, r_all, r_shape, prec)?;
                return Ok((qpy, rpy).into_py_any(py)?);
            }
        }
    }
    if let Some((data, shape, prec)) = as_promoted_complex_nd(a)? {
        check_stacked_2d(&shape)?;
        let m = shape[shape.len() - 2];
        let n = shape[shape.len() - 1];
        let k = m.min(n);
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        match mode {
            "r" => {
                if shape.len() == 2 {
                    let r = zc::qr_r(&data, m, n).map_err(linalg_err)?;
                    return mk2_c64_prec(py, r, k, n, prec);
                }
                let mut r_all = Vec::with_capacity(batch * k * n);
                for b in 0..batch {
                    let slice = &data[b * m * n..(b + 1) * m * n];
                    r_all.extend(zc::qr_r(slice, m, n).map_err(linalg_err)?);
                }
                let mut r_shape = batch_shape;
                r_shape.push(k);
                r_shape.push(n);
                return mk_nd_c128_prec(py, r_all, r_shape, prec);
            }
            "complete" => {
                if shape.len() == 2 {
                    let (q, r) = zc::qr_complete(&data, m, n).map_err(linalg_err)?;
                    let qpy = mk2_c64_prec(py, q, m, m, prec)?;
                    let rpy = mk2_c64_prec(py, r, m, n, prec)?;
                    return Ok((qpy, rpy).into_py_any(py)?);
                }
                let mut q_all = Vec::with_capacity(batch * m * m);
                let mut r_all = Vec::with_capacity(batch * m * n);
                for b in 0..batch {
                    let slice = &data[b * m * n..(b + 1) * m * n];
                    let (q, r) = zc::qr_complete(slice, m, n).map_err(linalg_err)?;
                    q_all.extend(q);
                    r_all.extend(r);
                }
                let mut q_shape = batch_shape.clone();
                q_shape.push(m);
                q_shape.push(m);
                let mut r_shape = batch_shape;
                r_shape.push(m);
                r_shape.push(n);
                let qpy = mk_nd_c128_prec(py, q_all, q_shape, prec)?;
                let rpy = mk_nd_c128_prec(py, r_all, r_shape, prec)?;
                return Ok((qpy, rpy).into_py_any(py)?);
            }
            "raw" => {
                if shape.len() == 2 {
                    let (h, tau) = zc::qr_raw(&data, m, n).map_err(linalg_err)?;
                    let hpy = mk2_c64_prec(py, h, n, m, prec)?;
                    let taupy = mk1_c64_prec(py, tau, prec)?;
                    return Ok((hpy, taupy).into_py_any(py)?);
                }
                let mut h_all = Vec::with_capacity(batch * m * n);
                let mut tau_all = Vec::with_capacity(batch * k);
                for b in 0..batch {
                    let slice = &data[b * m * n..(b + 1) * m * n];
                    let (h, tau) = zc::qr_raw(slice, m, n).map_err(linalg_err)?;
                    h_all.extend(h);
                    tau_all.extend(tau);
                }
                let mut h_shape = batch_shape.clone();
                h_shape.push(n);
                h_shape.push(m);
                let mut tau_shape = batch_shape;
                tau_shape.push(k);
                let hpy = mk_nd_c128_prec(py, h_all, h_shape, prec)?;
                let taupy = mk_nd_c128_prec(py, tau_all, tau_shape, prec)?;
                return Ok((hpy, taupy).into_py_any(py)?);
            }
            _ => {
                if shape.len() == 2 {
                    let (q, r) = zc::qr(&data, m, n).map_err(linalg_err)?;
                    let qpy = mk2_c64_prec(py, q, m, k, prec)?;
                    let rpy = mk2_c64_prec(py, r, k, n, prec)?;
                    return Ok((qpy, rpy).into_py_any(py)?);
                }
                let mut q_all = Vec::with_capacity(batch * m * k);
                let mut r_all = Vec::with_capacity(batch * k * n);
                for b in 0..batch {
                    let slice = &data[b * m * n..(b + 1) * m * n];
                    let (q, r) = zc::qr(slice, m, n).map_err(linalg_err)?;
                    q_all.extend(q);
                    r_all.extend(r);
                }
                let mut q_shape = batch_shape.clone();
                q_shape.push(m);
                q_shape.push(k);
                let mut r_shape = batch_shape;
                r_shape.push(k);
                r_shape.push(n);
                let qpy = mk_nd_c128_prec(py, q_all, q_shape, prec)?;
                let rpy = mk_nd_c128_prec(py, r_all, r_shape, prec)?;
                return Ok((qpy, rpy).into_py_any(py)?);
            }
        }
    }
    Err(unsupported_dtype("qr"))
}

// ═════════════════════════════ svd / svdvals ════════════════════════════

/// Batched (ndim > 2) non-Hermitian SVD: loops `rc`/`zc`'s existing
/// per-matrix `svd`/`svd_full` LAPACK calls over each of the leading batch
/// dimensions' flattened (m, n) slices (the array is already contiguous and
/// C-ordered, so slice `b` of `data` is exactly `data[b*m*n..(b+1)*m*n]`, no
/// transpose needed -- unlike `svd_singular_values_batched` below, whose
/// `row_axis`/`col_axis` can be arbitrary for `matrix_norm`; top-level `svd`
/// always operates on the trailing two axes), then stacks each output back
/// into a single `(batch_dims..., ...)`-shaped `NdArray`. `rc::svd`/
/// `rc::svd_full`/`zc::svd`/`zc::svd_full` already handle the `m==0||n==0`
/// per-slice case correctly (fixed earlier this task), so no additional
/// empty-matrix special-casing is needed here.
fn svd_batched(py: Python<'_>, arr: &NdArray, full_matrices: bool, compute_uv: bool) -> PyResult<Py<PyAny>> {
    let dt = arr.dtype();
    let contig = arr.to_contiguous();
    let shape = contig.shape().to_vec();
    let ndim = shape.len();
    let m = shape[ndim - 2];
    let n = shape[ndim - 1];
    let batch: usize = shape[..ndim - 2].iter().product();
    let batch_shape = shape[..ndim - 2].to_vec();
    let k = m.min(n);
    let (uk, vtk) = if full_matrices { (m, n) } else { (k, k) };

    if is_complex_dtype(dt) {
        let prec = classify_prec(dt);
        let data = match contig.buffer().cast_to(DType::C128) {
            Buffer::C128(v) => v,
            _ => unreachable!("cast_to(C128) always yields Buffer::C128"),
        };
        let mut s_all = Vec::with_capacity(batch * k);
        if !compute_uv {
            for b in 0..batch {
                let mat = &data[b * m * n..(b + 1) * m * n];
                let (_u, s, _vt) = zc::svd(mat, m, n).map_err(linalg_err)?;
                s_all.extend(s);
            }
            let mut s_shape = batch_shape;
            s_shape.push(k);
            let s_arr = NdArray::from_buffer(Buffer::F64(s_all), s_shape, Order::C).map_err(linalg_ionp_err)?;
            let s_arr = cast_real_by_prec(s_arr, prec);
            return Ok(Py::new(py, PyArray { inner: s_arr })?.into_any());
        }
        let mut u_all = Vec::with_capacity(batch * m * uk);
        let mut vt_all = Vec::with_capacity(batch * vtk * n);
        for b in 0..batch {
            let mat = &data[b * m * n..(b + 1) * m * n];
            let (u, s, vt) = if full_matrices {
                zc::svd_full(mat, m, n).map_err(linalg_err)?
            } else {
                zc::svd(mat, m, n).map_err(linalg_err)?
            };
            u_all.extend(u);
            s_all.extend(s);
            vt_all.extend(vt);
        }
        let mut u_shape = batch_shape.clone();
        u_shape.push(m);
        u_shape.push(uk);
        let mut s_shape = batch_shape.clone();
        s_shape.push(k);
        let mut vt_shape = batch_shape;
        vt_shape.push(vtk);
        vt_shape.push(n);
        let u_arr = NdArray::from_buffer(Buffer::C128(u_all), u_shape, Order::C).map_err(linalg_ionp_err)?;
        let u_arr = if prec == Prec::Single { u_arr.cast_to(DType::C64) } else { u_arr };
        let s_arr = NdArray::from_buffer(Buffer::F64(s_all), s_shape, Order::C).map_err(linalg_ionp_err)?;
        let s_arr = cast_real_by_prec(s_arr, prec);
        let vt_arr = NdArray::from_buffer(Buffer::C128(vt_all), vt_shape, Order::C).map_err(linalg_ionp_err)?;
        let vt_arr = if prec == Prec::Single { vt_arr.cast_to(DType::C64) } else { vt_arr };
        let upy = Py::new(py, PyArray { inner: u_arr })?.into_any();
        let spy = Py::new(py, PyArray { inner: s_arr })?.into_any();
        let vtpy = Py::new(py, PyArray { inner: vt_arr })?.into_any();
        return Ok((upy, spy, vtpy).into_py_any(py)?);
    }

    reject_float16(dt)?;
    let prec = classify_prec(dt);
    let data = match contig.buffer().cast_to(DType::F64) {
        Buffer::F64(v) => v,
        _ => unreachable!("cast_to(F64) always yields Buffer::F64"),
    };
    let mut s_all = Vec::with_capacity(batch * k);
    if !compute_uv {
        for b in 0..batch {
            let mat = &data[b * m * n..(b + 1) * m * n];
            let (_u, s, _vt) = rc::svd(mat, m, n).map_err(linalg_err)?;
            s_all.extend(s);
        }
        let mut s_shape = batch_shape;
        s_shape.push(k);
        let s_arr = NdArray::from_buffer(Buffer::F64(s_all), s_shape, Order::C).map_err(linalg_ionp_err)?;
        let s_arr = cast_real_by_prec(s_arr, prec);
        return Ok(Py::new(py, PyArray { inner: s_arr })?.into_any());
    }
    let mut u_all = Vec::with_capacity(batch * m * uk);
    let mut vt_all = Vec::with_capacity(batch * vtk * n);
    for b in 0..batch {
        let mat = &data[b * m * n..(b + 1) * m * n];
        let (u, s, vt) = if full_matrices {
            rc::svd_full(mat, m, n).map_err(linalg_err)?
        } else {
            rc::svd(mat, m, n).map_err(linalg_err)?
        };
        u_all.extend(u);
        s_all.extend(s);
        vt_all.extend(vt);
    }
    let mut u_shape = batch_shape.clone();
    u_shape.push(m);
    u_shape.push(uk);
    let mut s_shape = batch_shape.clone();
    s_shape.push(k);
    let mut vt_shape = batch_shape;
    vt_shape.push(vtk);
    vt_shape.push(n);
    let u_arr = NdArray::from_buffer(Buffer::F64(u_all), u_shape, Order::C).map_err(linalg_ionp_err)?;
    let u_arr = cast_real_by_prec(u_arr, prec);
    let s_arr = NdArray::from_buffer(Buffer::F64(s_all), s_shape, Order::C).map_err(linalg_ionp_err)?;
    let s_arr = cast_real_by_prec(s_arr, prec);
    let vt_arr = NdArray::from_buffer(Buffer::F64(vt_all), vt_shape, Order::C).map_err(linalg_ionp_err)?;
    let vt_arr = cast_real_by_prec(vt_arr, prec);
    let upy = Py::new(py, PyArray { inner: u_arr })?.into_any();
    let spy = Py::new(py, PyArray { inner: s_arr })?.into_any();
    let vtpy = Py::new(py, PyArray { inner: vt_arr })?.into_any();
    Ok((upy, spy, vtpy).into_py_any(py)?)
}

#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn svd(py: Python<'_>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    // `full_matrices`/`compute_uv`/`hermitian` type handling rewritten
    // 2026-08-02 (coordinator's UPLO round-3 survey named "svd/cond's
    // arguments" as candidates to measure). numpy's real body never type
    // checks these -- it does plain Python TRUTH tests (`if hermitian:`,
    // `if compute_uv:`, `if full_matrices:`, verified via
    // `inspect.getsource(numpy.linalg.svd)`), so ANY object works: measured
    // directly, `full_matrices=1`/`0`/`"yes"`/`""`/`None`/`[]` all behave
    // in real numpy exactly as their Python truthiness dictates, while the
    // previous `bool`-typed parameters here made PyO3 itself raise
    // `TypeError: '<type>' object is not an instance of 'bool'` for every
    // one of them -- same defect class as `eigh`'s `UPLO`, different
    // mechanism (truthiness, not a duck-typed method call or a membership
    // test). Raw `*args`/`**kwargs` capture used (not `Option<&PyAny>`
    // params with `None` defaults) for the same ABSENT-vs-explicit-None
    // reason as `eigh`/`qr`: omitting `full_matrices` must default to
    // `true`, but explicitly passing `full_matrices=None` is a normal,
    // DIFFERENT, falsy value in numpy (`if None:` is `False`) -- an
    // `Option<T>` parameter cannot tell those two cases apart.
    const NAMES: [&str; 4] = ["a", "full_matrices", "compute_uv", "hermitian"];
    if args.len() > NAMES.len() {
        return Err(PyTypeError::new_err(format!(
            "svd() takes from 1 to {} positional arguments but {} were given",
            NAMES.len(),
            args.len()
        )));
    }
    let mut slots: [Option<Bound<'_, PyAny>>; 4] = [None, None, None, None];
    for (i, item) in args.iter().enumerate() {
        slots[i] = Some(item);
    }
    if let Some(kw) = kwargs {
        for (idx, name) in NAMES.iter().enumerate() {
            if let Some(v) = kw.get_item(name)? {
                if slots[idx].is_some() {
                    return Err(PyTypeError::new_err(format!(
                        "svd() got multiple values for argument '{name}'"
                    )));
                }
                slots[idx] = Some(v);
            }
        }
    }
    let [a_slot, full_matrices_slot, compute_uv_slot, hermitian_slot] = slots;
    let a_owned =
        a_slot.ok_or_else(|| PyTypeError::new_err("svd() missing 1 required positional argument: 'a'"))?;
    let a: &Bound<'_, PyAny> = &a_owned;
    let full_matrices = match &full_matrices_slot {
        Some(obj) => obj.is_truthy()?,
        None => true,
    };
    let compute_uv = match &compute_uv_slot {
        Some(obj) => obj.is_truthy()?,
        None => true,
    };
    let hermitian = match &hermitian_slot {
        Some(obj) => obj.is_truthy()?,
        None => false,
    };
    // numpy's `svd` runs `_assert_stacked_2d` before anything else, raising
    // `LinAlgError: "{ndim}-dimensional array given. Array must be at least
    // two-dimensional"` for ndim < 2 -- verified against real numpy 2.5.1.
    // `as_real2`/`as_complex2` below only check `shape.len() != 2` and raise
    // a generic, non-numpy-matching `ValueError` for the ndim<2 case (wrong
    // class and text), so that case is fixed here up front.
    if let Some(arr) = try_ndarray_any(a) {
        let ndim = arr.shape().len();
        if ndim < 2 {
            return Err(linalg_err_raw(&format!(
                "{}-dimensional array given. Array must be at least two-dimensional",
                ndim
            )));
        }
        // 2026-08-02: numpy's `svd` supports arbitrary batch/stack
        // dimensions ahead of the trailing (m, n) pair (verified:
        // `np.linalg.svd(np.zeros((2,0,0)))` succeeds, returning
        // per-batch-element U/S/Vh). `svd_hermitian` below is NOT
        // extended to batching here (out of scope for this pass, still
        // 2-D-only, an honestly-reported narrower remaining gap) -- only
        // the general (non-Hermitian) path, which is what the empty-shape
        // `(2,0,0)` crossed-axis case exercises, is batched, reusing the
        // exact same `rc::svd`/`rc::svd_full`/`zc::svd`/`zc::svd_full`
        // core calls the 2-D path above already uses, looped per batch
        // slice -- mirroring the already-existing, already-verified
        // `svd_singular_values_batched` helper below (used by
        // `matrix_norm`'s P2/PNeg2/Nuc ords), extended here to also
        // produce U/Vt when `compute_uv` is requested.
        if ndim > 2 && !hermitian {
            return svd_batched(py, &arr, full_matrices, compute_uv);
        }
    }
    if hermitian {
        // A Hermitian/symmetric matrix is necessarily square, so
        // k = min(m, n) = m = n: the economy and full SVD coincide here
        // (verified empirically -- `full_matrices=True` and `=False` gave
        // bit-identical U/S/Vh for the same Hermitian input). So this path
        // is bound for both values of `full_matrices`, unlike the general
        // path below.
        return svd_hermitian(py, a, compute_uv);
    }
    // 2026-08-02: switched from `as_real2`/`as_complex2` (raw float64/
    // complex128-only extraction, no promotion) to the same
    // `as_promoted_real2`/`as_promoted_complex2` helpers `det`/`inv`/`qr`/
    // `eigh`/etc. already use elsewhere in this file. `as_real2` rejected
    // every OTHER real dtype (bool/int*/uint*/float32/float16) outright
    // with `unsupported_dtype("svd")`, disagreeing with numpy's own
    // `_commonType` promotion table (verified: `np.linalg.svd(np.zeros((0,
    // 0), dtype=np.int64))` succeeds, promoting to float64) -- this was a
    // real, if narrow, dtype-coverage gap, not something specific to empty
    // shapes; it simply surfaced there because that is what the empty-
    // shape crossed-axis sweep tests. `svd_full`/`svd`'s core signatures
    // are unchanged (`&[f64]`/`&[Complex64]` in, `Vec<f64>`/`Vec<Complex64>`
    // out); only the extraction/output-downcast wrapper changes, mirroring
    // `qr`'s existing `_prec`-suffixed output pattern.
    if let Some((data, m, n, prec)) = as_promoted_real2(a)? {
        // 2026-08-06: NaN-input convergence gap. Apple Accelerate's
        // `dgesdd` (this crate's real-SVD LAPACK backend) does NOT reliably
        // report `info != 0` for NaN-containing input at small (1x1/2x2)
        // sizes -- measured directly: `rc::svd`/`rc::svd_full` return
        // silently with NaN-propagated output instead of the
        // `LinAlgError::DidNotConverge` real numpy raises (verified against
        // real numpy 2.5.1: `np.linalg.svd([[nan]])` and
        // `np.linalg.svd([[nan,1],[3,4]])` both raise
        // `LinAlgError("SVD did not converge")`). Checked and confirmed
        // ONLY at these small sizes for this task's scope -- NOT extended
        // to Inf-containing input, whose real-numpy behavior is unclear
        // (a live probe of `np.linalg.svd` on an Inf-containing matrix did
        // not return within two minutes, suggesting a genuine long-running
        // LAPACK iteration rather than a clean raise; scoping this fix to
        // NaN only rather than guessing at Inf's contract). Message text
        // "SVD did not converge" copied verbatim from the real-numpy
        // probe above (not `LinAlgError::DidNotConverge`'s own `Display`,
        // which renders "Eigenvalues/SVD did not converge" -- wrong
        // prefix), so this raises via `linalg_err_raw`, mirroring how
        // `lstsq`/this same function's "Last 2 dimensions..." messages
        // already bypass `Display` for the same reason (see
        // `linalg_err_raw`'s doc comment above).
        if data.iter().any(|v| v.is_nan()) {
            return Err(linalg_err_raw("SVD did not converge"));
        }
        let k = m.min(n);
        if !compute_uv {
            let (_u, s, _vt) = rc::svd(&data, m, n).map_err(linalg_err)?;
            return mk1_f64_prec(py, s, prec);
        }
        if full_matrices {
            let (u, s, vt) = rc::svd_full(&data, m, n).map_err(linalg_err)?;
            let upy = mk2_f64_prec(py, u, m, m, prec)?;
            let spy = mk1_f64_prec(py, s, prec)?;
            let vtpy = mk2_f64_prec(py, vt, n, n, prec)?;
            return Ok((upy, spy, vtpy).into_py_any(py)?);
        }
        let (u, s, vt) = rc::svd(&data, m, n).map_err(linalg_err)?;
        let upy = mk2_f64_prec(py, u, m, k, prec)?;
        let spy = mk1_f64_prec(py, s, prec)?;
        let vtpy = mk2_f64_prec(py, vt, k, n, prec)?;
        return Ok((upy, spy, vtpy).into_py_any(py)?);
    }
    if let Some((data, m, n, prec)) = as_promoted_complex2(a)? {
        // Same NaN-convergence gap as the real branch above (see that
        // block's doc comment for the full rationale) -- verified
        // separately against real numpy 2.5.1 for the complex-dtype case
        // too (`np.linalg.svd([[nan+0j]])` and a 2x2 complex NaN case both
        // raise the identical `LinAlgError("SVD did not converge")`).
        if data.iter().any(|c| c.re.is_nan() || c.im.is_nan()) {
            return Err(linalg_err_raw("SVD did not converge"));
        }
        let k = m.min(n);
        if !compute_uv {
            let (_u, s, _vt) = zc::svd(&data, m, n).map_err(linalg_err)?;
            return mk1_f64_prec(py, s, prec);
        }
        if full_matrices {
            let (u, s, vt) = zc::svd_full(&data, m, n).map_err(linalg_err)?;
            let upy = mk2_c64_prec(py, u, m, m, prec)?;
            let spy = mk1_f64_prec(py, s, prec)?;
            let vtpy = mk2_c64_prec(py, vt, n, n, prec)?;
            return Ok((upy, spy, vtpy).into_py_any(py)?);
        }
        let (u, s, vt) = zc::svd(&data, m, n).map_err(linalg_err)?;
        let upy = mk2_c64_prec(py, u, m, k, prec)?;
        let spy = mk1_f64_prec(py, s, prec)?;
        let vtpy = mk2_c64_prec(py, vt, k, n, prec)?;
        return Ok((upy, spy, vtpy).into_py_any(py)?);
    }
    Err(unsupported_dtype("svd"))
}

/// `svd(a, hermitian=True)`: numpy's own docstring: "If `a` is Hermitian
/// or real symmetric, then the singular values are `abs(eigenvalues)`" --
/// computed via `eigh` (real `dsyev`/complex `zheev`, already bound as
/// `rc::eigh`/`zc::eigh`), not the general SVD LAPACK routine. Algorithm,
/// verified by hand against real numpy 2.5.1 for a real indefinite 2x2
/// case (eigenvalues of opposite sign), a real positive-definite case, and
/// a complex-Hermitian case, all matching numpy's own singular values and
/// `U @ diag(S) @ Vh` reconstruction:
///   `eigh` returns eigenvalues `w` ascending and eigenvectors as the
///   COLUMNS of an n x n row-major matrix `v`. For each eigenpair
///   `(w_i, v_i)`, `A = sum_i w_i * v_i * v_i^H`. Sorting by `|w_i|`
///   descending gives the singular values `S = |w|`; for each such
///   eigenpair, set `U`'s column to `v_i` unchanged and `Vh`'s row to
///   `sign(w_i) * v_i^H` (conjugate-transposed for the complex case) --
///   this reproduces `A = U @ diag(S) @ Vh` exactly for both the positive
///   and negative eigenvalue case, since
///   `w_i * v_i * v_i^H = |w_i| * v_i * (sign(w_i) * v_i^H)`.
fn svd_hermitian(py: Python<'_>, a: &Bound<'_, PyAny>, compute_uv: bool) -> PyResult<Py<PyAny>> {
    // 2026-08-02: was `as_real2`/`as_complex2` (raw f64/complex128-only
    // extraction, no dtype promotion, 2-D-only, no batching) -- the same
    // defect class fixed elsewhere this task for `eigh`/
    // `pinv(hermitian=True)`/`matrix_rank(hermitian=True)`, all of which
    // now share the `hermitian_svd_real`/`_complex` helpers this function
    // is rewritten to reuse (previously NOT reused, to keep this call path
    // independently verified per that helper's own doc comment -- now
    // reused deliberately since both call paths have since been
    // independently probed and matched, so the coupling risk that
    // justified the duplication no longer applies). Verified via a
    // shape x dtype x full_matrices x compute_uv sweep against real numpy
    // 2.5.1: 4/32 mismatches were exactly this gap (batched hermitian
    // input), 0 after this rewrite.
    if let Some((data, shape, prec)) = as_promoted_real_nd(a)? {
        let n = check_stacked_square(&shape)?;
        let compute = |slice: &[f64]| -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
            hermitian_svd_real(slice, n).map_err(linalg_err)
        };
        if shape.len() == 2 {
            let (u, s, vt) = compute(&data)?;
            if !compute_uv {
                return mk1_f64_prec(py, s, prec);
            }
            let upy = mk2_f64_prec(py, u, n, n, prec)?;
            let spy = mk1_f64_prec(py, s, prec)?;
            let vtpy = mk2_f64_prec(py, vt, n, n, prec)?;
            return Ok((upy, spy, vtpy).into_py_any(py)?);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        if !compute_uv {
            let mut sout = Vec::with_capacity(batch * n);
            for b in 0..batch {
                let slice = &data[b * n * n..(b + 1) * n * n];
                let (_u, s, _vt) = compute(slice)?;
                sout.extend(s);
            }
            let mut s_shape = batch_shape;
            s_shape.push(n);
            return mk_nd_f64_prec(py, sout, s_shape, prec);
        }
        let mut uout = Vec::with_capacity(batch * n * n);
        let mut sout = Vec::with_capacity(batch * n);
        let mut vtout = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let slice = &data[b * n * n..(b + 1) * n * n];
            let (u, s, vt) = compute(slice)?;
            uout.extend(u);
            sout.extend(s);
            vtout.extend(vt);
        }
        let mut uv_shape = batch_shape.clone();
        uv_shape.push(n);
        uv_shape.push(n);
        let mut s_shape = batch_shape;
        s_shape.push(n);
        let upy = mk_nd_f64_prec(py, uout, uv_shape.clone(), prec)?;
        let spy = mk_nd_f64_prec(py, sout, s_shape, prec)?;
        let vtpy = mk_nd_f64_prec(py, vtout, uv_shape, prec)?;
        return Ok((upy, spy, vtpy).into_py_any(py)?);
    }
    if let Some((data, shape, prec)) = as_promoted_complex_nd(a)? {
        let n = check_stacked_square(&shape)?;
        let compute = |slice: &[Complex64]| -> PyResult<(Vec<Complex64>, Vec<f64>, Vec<Complex64>)> {
            hermitian_svd_complex(slice, n).map_err(linalg_err)
        };
        if shape.len() == 2 {
            let (u, s, vt) = compute(&data)?;
            if !compute_uv {
                return mk1_f64_prec(py, s, prec);
            }
            let upy = mk2_c64_prec(py, u, n, n, prec)?;
            let spy = mk1_f64_prec(py, s, prec)?;
            let vtpy = mk2_c64_prec(py, vt, n, n, prec)?;
            return Ok((upy, spy, vtpy).into_py_any(py)?);
        }
        let batch_shape = shape[..shape.len() - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        if !compute_uv {
            let mut sout = Vec::with_capacity(batch * n);
            for b in 0..batch {
                let slice = &data[b * n * n..(b + 1) * n * n];
                let (_u, s, _vt) = compute(slice)?;
                sout.extend(s);
            }
            let mut s_shape = batch_shape;
            s_shape.push(n);
            return mk_nd_f64_prec(py, sout, s_shape, prec);
        }
        let mut uout = Vec::with_capacity(batch * n * n);
        let mut sout = Vec::with_capacity(batch * n);
        let mut vtout = Vec::with_capacity(batch * n * n);
        for b in 0..batch {
            let slice = &data[b * n * n..(b + 1) * n * n];
            let (u, s, vt) = compute(slice)?;
            uout.extend(u);
            sout.extend(s);
            vtout.extend(vt);
        }
        let mut uv_shape = batch_shape.clone();
        uv_shape.push(n);
        uv_shape.push(n);
        let mut s_shape = batch_shape;
        s_shape.push(n);
        let upy = mk_nd_c128_prec(py, uout, uv_shape.clone(), prec)?;
        let spy = mk_nd_f64_prec(py, sout, s_shape, prec)?;
        let vtpy = mk_nd_c128_prec(py, vtout, uv_shape, prec)?;
        return Ok((upy, spy, vtpy).into_py_any(py)?);
    }
    Err(unsupported_dtype("svd"))
}

#[pyfunction]
#[pyo3(signature = (x))]
fn svdvals(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Some((data, m, n, prec)) = as_promoted_real2(x)? {
        let (_u, s, _vt) = rc::svd(&data, m, n).map_err(linalg_err)?;
        return mk1_f64_prec(py, s, prec);
    }
    if let Some((data, m, n, prec)) = as_promoted_complex2(x)? {
        let (_u, s, _vt) = zc::svd(&data, m, n).map_err(linalg_err)?;
        return mk1_f64_prec(py, s, prec);
    }
    Err(unsupported_dtype("svdvals"))
}

/// Raw (non-PyO3) Hermitian-SVD singular values + factors for a real
/// symmetric `n x n` input, used by `pinv(hermitian=True)` and
/// `matrix_rank(hermitian=True)`. Same algorithm as `svd_hermitian` above
/// (not a call into it, to avoid coupling two independently-verified call
/// paths -- see that function's doc comment for the proof): eigendecompose
/// via `eigh`, sort by `|eigenvalue|` descending, `S = |w|`, `U`'s columns
/// are the eigenvectors unchanged, `Vh`'s rows are `sign(w) * eigenvector^H`.
fn hermitian_svd_real(data: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), LinAlgError> {
    let (w, v) = rc::eigh(data, n)?;
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| w[j].abs().partial_cmp(&w[i].abs()).unwrap());
    let s: Vec<f64> = idx.iter().map(|&i| w[i].abs()).collect();
    let mut u = vec![0.0f64; n * n];
    let mut vt = vec![0.0f64; n * n];
    for (col, &i) in idx.iter().enumerate() {
        let sign = if w[i] < 0.0 { -1.0 } else { 1.0 };
        for row in 0..n {
            let val = v[row * n + i];
            u[row * n + col] = val;
            vt[col * n + row] = sign * val;
        }
    }
    Ok((u, s, vt))
}

/// Complex counterpart of `hermitian_svd_real`.
fn hermitian_svd_complex(
    data: &[Complex64],
    n: usize,
) -> Result<(Vec<Complex64>, Vec<f64>, Vec<Complex64>), LinAlgError> {
    let (w, v) = zc::eigh(data, n)?;
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| w[j].abs().partial_cmp(&w[i].abs()).unwrap());
    let s: Vec<f64> = idx.iter().map(|&i| w[i].abs()).collect();
    let mut u = vec![Complex64::new(0.0, 0.0); n * n];
    let mut vt = vec![Complex64::new(0.0, 0.0); n * n];
    for (col, &i) in idx.iter().enumerate() {
        let sign = if w[i] < 0.0 { -1.0 } else { 1.0 };
        for row in 0..n {
            let val = v[row * n + i];
            u[row * n + col] = val;
            vt[col * n + row] = val.conj() * sign;
        }
    }
    Ok((u, s, vt))
}

// ═══════════════════════════════ pinv ═══════════════════════════════════

/// `data type dtype('int64') not compatible with finfo`-style eps lookup,
/// used only by pinv's `rtol=None` (explicit) branch below. Real numpy's
/// `finfo(a.dtype).eps` (verified via `inspect.getsource`/live probes on
/// numpy 2.5.1) succeeds for every float/complex dtype -- including
/// float16 (`eps == 0.0009765625`, i.e. `2**-10`, verified via
/// `np.finfo(np.float16).eps`) -- and raises a plain `ValueError` for
/// every integer/bool dtype (`"data type dtype('int64') not compatible
/// with finfo"` / `"...dtype('bool')..."`, exact text verified live).
/// `f32::EPSILON`/`f64::EPSILON` match `np.finfo(float32/64).eps` bit for
/// bit; complex64/128 share their real counterpart's eps (verified live).
fn finfo_eps(dt: DType) -> PyResult<f64> {
    match dt {
        DType::F16 => Ok(0.0009765625_f64),
        DType::F32 | DType::C64 => Ok(f32::EPSILON as f64),
        DType::F64 | DType::C128 => Ok(f64::EPSILON),
        _ => Err(PyValueError::new_err(format!(
            "data type dtype('{}') not compatible with finfo",
            dt.name()
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn pinv(py: Python<'_>, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    // Real signature: `pinv(a, rcond=None, hermitian=False, *, rtol=_NoValue)`.
    // `rtol` is keyword-only in real numpy AND (see doc block below) its
    // real default is a private `_NoValue` sentinel, not `None` -- so
    // "rtol omitted" and "rtol=None passed explicitly" are two different,
    // both-reachable, both-load-bearing states that a plain typed
    // `rtol: Option<f64> = None` pyo3 parameter cannot tell apart (pyo3's
    // `Option<T>` maps a Python `None` argument to Rust `None` identically
    // to an omitted argument). This raw `*args`/`**kwargs` technique
    // mirrors `diff()` in `ionp-py/src/setops.rs` (see its own doc comment
    // for the general derivation) and `Ufunc::__call__`'s `out=` handling
    // in `ionp-py/src/lib.rs`.
    const NAMES: [&str; 3] = ["a", "rcond", "hermitian"];
    if args.len() > NAMES.len() {
        return Err(PyTypeError::new_err(format!(
            "pinv() takes from 1 to {} positional arguments but {} were given",
            NAMES.len(),
            args.len()
        )));
    }
    let mut slots: [Option<Bound<'_, PyAny>>; 3] = [None, None, None];
    for (i, item) in args.iter().enumerate() {
        slots[i] = Some(item);
    }
    let mut rtol_slot: Option<Bound<'_, PyAny>> = None;
    if let Some(kw) = kwargs {
        for (idx, name) in NAMES.iter().enumerate() {
            if let Some(v) = kw.get_item(name)? {
                if slots[idx].is_some() {
                    return Err(PyTypeError::new_err(format!(
                        "pinv() got multiple values for argument '{name}'"
                    )));
                }
                slots[idx] = Some(v);
            }
        }
        if let Some(v) = kw.get_item("rtol")? {
            rtol_slot = Some(v);
        }
        for key in kw.keys().iter() {
            let key_str: String = key.extract()?;
            if !NAMES.contains(&key_str.as_str()) && key_str != "rtol" {
                return Err(PyTypeError::new_err(format!(
                    "pinv() got an unexpected keyword argument '{key_str}'"
                )));
            }
        }
    }
    let [a_slot, rcond_slot, hermitian_slot] = slots;
    let a_obj = a_slot
        .ok_or_else(|| PyTypeError::new_err("pinv() missing 1 required positional argument: 'a'"))?;
    let a = &a_obj;
    // `rcond=None` explicit and `rcond` omitted collapse to the SAME real
    // numpy state (the documented default literally IS `None`), unlike
    // `rtol` below -- so a plain `Option<f64>` is fine here.
    let rcond: Option<f64> = match &rcond_slot {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => Some(v.extract::<f64>()?),
    };
    // `hermitian=` is plain Python truthiness in real numpy (measured
    // 2026-08-02, same as `svd`'s flags above) -- `v.extract::<bool>()`
    // wrongly required `isinstance(v, bool)`; `v.is_truthy()` matches
    // numpy's `if hermitian:` dispatch exactly.
    let hermitian: bool = match &hermitian_slot {
        None => false,
        Some(v) => v.is_truthy()?,
    };

    // Mirror numpy's exact rcond/rtol defaulting (verified against numpy
    // 2.5.1's `inspect.getsource(np.linalg.pinv)`, and live-probed for
    // execution ORDER, not just outcome -- see below):
    //
    //   if rcond is None:
    //       if rtol is _NoValue: rcond = 1e-15
    //       elif rtol is None: rcond = max(a.shape[-2:]) * finfo(a.dtype).eps
    //       else: rcond = rtol
    //   elif rtol is not _NoValue:
    //       raise ValueError("`rtol` and `rcond` can't be both set.")
    //
    // Two live-verified, non-obvious consequences reproduced here:
    // 1. The "both set" error fires whenever rcond is given (non-None) AND
    //    rtol is anything other than absent -- INCLUDING `rtol=None`
    //    explicit (verified: `pinv(eye(2), rcond=1e-8, rtol=None)` raises
    //    the same "can't be both set" ValueError as `rcond=1e-8,
    //    rtol=1e-8`). `rcond=None` explicit does NOT count as "given".
    // 2. `max(a.shape[-2:]) * finfo(a.dtype).eps` runs BEFORE the 1-D
    //    `(0,)` unpack special-case and BEFORE the general `_is_empty_2d`
    //    early return below -- so `pinv(zeros(0), rtol=None)` with an
    //    int64/bool dtype raises finfo's ValueError, not the unpack
    //    ValueError (verified live: int64 shape (0,) with rtol=None gives
    //    "data type dtype('int64') not compatible with finfo", while
    //    float64 shape (0,) with rtol=None gives "not enough values to
    //    unpack..."). `max()` on a Python tuple works for any length
    //    (including the 1-element `shape[-2:]` of a 1-D array), so only
    //    the dtype check can fail here, not an unpack; a genuinely 0-d
    //    array's `shape[-2:]` is `()`, and `max(())` raises
    //    "max() iterable argument is empty" (also reproduced below,
    //    verified live to fire before any dtype/finfo check since it's
    //    the left operand of `max(...) * finfo(...).eps`).
    let rtol_is_none_or_absent_ok = rcond.is_none();
    if !rtol_is_none_or_absent_ok && rtol_slot.is_some() {
        return Err(PyValueError::new_err("`rtol` and `rcond` can't be both set."));
    }
    let resolved_rcond = if let Some(r) = rcond {
        r
    } else {
        match &rtol_slot {
            None => 1e-15,
            Some(v) if v.is_none() => {
                let peek = try_ndarray_any(a).ok_or_else(|| {
                    PyTypeError::new_err("pinv() argument must be array-like")
                })?;
                let shape = peek.shape();
                let tail: &[usize] = if shape.len() >= 2 { &shape[shape.len() - 2..] } else { shape };
                let m = tail.iter().copied().max().ok_or_else(|| {
                    PyValueError::new_err("max() iterable argument is empty")
                })?;
                (m as f64) * finfo_eps(peek.dtype())?
            }
            Some(v) => v.extract::<f64>()?,
        }
    };
    // Deliberate replication of a real numpy 2.5.1 implementation accident
    // (verified via `inspect.getsource(np.linalg.pinv)`), not a feature:
    // numpy's `_is_empty_2d(a)` short-circuits on `a.size == 0`, and only
    // THEN evaluates `m, n = a.shape[-2:]` -- for a 1-D empty array
    // (shape (0,)), `a.shape[-2:]` is the one-element tuple `(0,)`, so the
    // 2-tuple unpack itself fails with a plain `ValueError`, BEFORE numpy's
    // usual `_assert_stacked_2d` dimensionality check ever runs (that check
    // lives inside `svd`, reached only past this branch). This is a
    // numpy source-order accident, not a documented contract -- but anionpy is
    // a drop-in replacement, and caller code written against numpy may
    // catch `ValueError` around `pinv`; silently substituting our own (more
    // correct) `LinAlgError` here would let that catch miss and the
    // program fail somewhere unrelated. So: reproduce numpy exactly, not
    // "improve" on it. Confirmed this is the ONLY shape affected: 0-d
    // input, 1-D non-empty input, and every 2-D-or-higher empty shape
    // ((2,0), (0,3), (0,0), (2,0,0), ...) all take numpy's normal path with
    // no unpack error (verified against real numpy 2.5.1 across all of
    // those shapes) because `_is_empty_2d`'s `arr.size == 0` guard is false
    // for 0-d/non-empty-1-D, and `a.shape[-2:]` already has exactly 2
    // elements for ndim >= 2 regardless of emptiness. Also verified this
    // triggers identically for `hermitian=True` (the check runs before the
    // `hermitian` branch in numpy's source, so it is not specific to the
    // general-SVD path below).
    if let Some(peek) = try_ndarray_any(a) {
        let shape = peek.shape();
        if shape.len() == 1 && shape[0] == 0 {
            return Err(PyValueError::new_err("not enough values to unpack (expected 2, got 1)"));
        }
        // General `_is_empty_2d(a)` early return (verified via
        // `inspect.getsource`: `arr.size == 0 and prod(arr.shape[-2:]) ==
        // 0`), same numpy source-order accident as the 1-D case above but
        // for ndim >= 2: `res = empty(a.shape[:-2] + (n, m), dtype=a.dtype)`
        // runs and returns BEFORE the `hermitian` branch, before `svd`, and
        // before ANY dtype-support check -- so this applies identically for
        // hermitian=True/False, and (verified directly against real numpy
        // 2.5.1) even for float16 (`pinv(zeros((0,3),dtype=f16))` succeeds
        // with a float16 result, while `pinv(zeros((2,3),dtype=f16))`
        // raises the usual "unsupported in linalg" TypeError) and for
        // bool/int64 (result dtype is the INPUT dtype exactly, not
        // promoted to float64 -- e.g. `pinv(zeros((0,3),dtype=bool)).dtype
        // == bool`). The output shape/dtype-swap math mirrors numpy's own
        // `m, n = a.shape[-2:]; ... (n, m)`. Every empty shape this branch
        // is reachable for has 0 total output elements (trailing product
        // == 0 implies n*m == 0), so `empty_buffer`'s zero-filled contents
        // (see that function's own doc comment: numpy's `empty` contents
        // are documented-garbage anyway, shape/dtype is the entire
        // contract) never actually holds any data to disagree over.
        if shape.len() >= 2 {
            let total: usize = shape.iter().product();
            let trailing: usize = shape[shape.len() - 2..].iter().product();
            if total == 0 && trailing == 0 {
                let m = shape[shape.len() - 2];
                let n = shape[shape.len() - 1];
                let mut out_shape = shape[..shape.len() - 2].to_vec();
                out_shape.push(n);
                out_shape.push(m);
                let dt = peek.dtype();
                let buf = ionp_core::creation::empty_buffer(dt, 0);
                let out = NdArray::from_buffer(buf, out_shape, Order::C).map_err(linalg_ionp_err)?;
                return PyArray { inner: out }.into_py_any(py);
            }
        }
    }
    if hermitian {
        // numpy's pinv(hermitian=True) (`inspect.getsource`): `u, s, vt =
        // svd(a, hermitian=True)`, `cutoff = rcond * s.max()`,
        // `res = vt^H @ diag(1/s where s>cutoff else 0) @ u^H`. Built here
        // directly from `hermitian_svd_real`/`_complex` (same math as
        // `svd_hermitian`, see that helper's doc comment) rather than the
        // core's `rc::pinv`/`zc::pinv`, which are general-SVD-only.
        //
        // 2026-08-02: was 2-D-only (`as_promoted_real2`/`_complex2`), the
        // same defect class as `eig`/`eigh` before their fix this task --
        // reachable via `pinv(zeros((5,)), hermitian=True)` returning the
        // generic "batched/1-D inputs are out of scope" `ValueError`
        // instead of numpy's `LinAlgError: 1-dimensional array given...`.
        // Rewritten with `check_stacked_square` + N-D batching, matching
        // `eigh`'s pattern; the empty-input case is already handled above
        // (this point is only reached for non-empty input), so squareness
        // is genuinely required here same as numpy's own `svd(...,
        // hermitian=True)` path.
        if let Some((data, shape, prec)) = as_promoted_real_nd(a)? {
            let n = check_stacked_square(&shape)?;
            let compute = |slice: &[f64]| -> PyResult<Vec<f64>> {
                let (u, s, vt) = hermitian_svd_real(slice, n).map_err(linalg_err)?;
                let smax = s.iter().cloned().fold(0.0f64, f64::max);
                let cutoff = resolved_rcond * smax;
                let mut out = vec![0.0f64; n * n];
                for i in 0..n {
                    for j in 0..n {
                        let mut acc = 0.0f64;
                        for k in 0..n {
                            if s[k] > cutoff {
                                acc += vt[k * n + i] * (1.0 / s[k]) * u[j * n + k];
                            }
                        }
                        out[i * n + j] = acc;
                    }
                }
                Ok(out)
            };
            if shape.len() == 2 {
                return mk2_f64_prec(py, compute(&data)?, n, n, prec);
            }
            let batch_shape = shape[..shape.len() - 2].to_vec();
            let batch: usize = batch_shape.iter().product();
            let mut out = Vec::with_capacity(batch * n * n);
            for b in 0..batch {
                let slice = &data[b * n * n..(b + 1) * n * n];
                out.extend(compute(slice)?);
            }
            let mut out_shape = batch_shape;
            out_shape.push(n);
            out_shape.push(n);
            return mk_nd_f64_prec(py, out, out_shape, prec);
        }
        if let Some((data, shape, prec)) = as_promoted_complex_nd(a)? {
            let n = check_stacked_square(&shape)?;
            let compute = |slice: &[Complex64]| -> PyResult<Vec<Complex64>> {
                let (u, s, vt) = hermitian_svd_complex(slice, n).map_err(linalg_err)?;
                let smax = s.iter().cloned().fold(0.0f64, f64::max);
                let cutoff = resolved_rcond * smax;
                let mut out = vec![Complex64::new(0.0, 0.0); n * n];
                for i in 0..n {
                    for j in 0..n {
                        let mut acc = Complex64::new(0.0, 0.0);
                        for k in 0..n {
                            if s[k] > cutoff {
                                acc += vt[k * n + i].conj() * (1.0 / s[k]) * u[j * n + k].conj();
                            }
                        }
                        out[i * n + j] = acc;
                    }
                }
                Ok(out)
            };
            if shape.len() == 2 {
                return mk2_c64_prec(py, compute(&data)?, n, n, prec);
            }
            let batch_shape = shape[..shape.len() - 2].to_vec();
            let batch: usize = batch_shape.iter().product();
            let mut out = Vec::with_capacity(batch * n * n);
            for b in 0..batch {
                let slice = &data[b * n * n..(b + 1) * n * n];
                out.extend(compute(slice)?);
            }
            let mut out_shape = batch_shape;
            out_shape.push(n);
            out_shape.push(n);
            return mk_nd_c128_prec(py, out, out_shape, prec);
        }
        return Err(unsupported_dtype("pinv"));
    }
    // General (non-hermitian) path: `pinv` does NOT require square input
    // (that's the whole point of a pseudo-inverse) -- only `_assert_stacked_2d`,
    // not `_assert_stacked_square`, per numpy's own source. Verified: numpy
    // batches over leading dims same as the other linalg ufuncs.
    if let Some((data, shape, prec)) = as_promoted_real_nd(a)? {
        check_stacked_2d(&shape)?;
        let len = shape.len();
        let m = shape[len - 2];
        let n = shape[len - 1];
        if len == 2 {
            let out = rc::pinv(&data, m, n, resolved_rcond).map_err(linalg_err)?;
            return mk2_f64_prec(py, out, n, m, prec);
        }
        let batch_shape = shape[..len - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch * m * n);
        for b in 0..batch {
            let slice = &data[b * m * n..(b + 1) * m * n];
            out.extend(rc::pinv(slice, m, n, resolved_rcond).map_err(linalg_err)?);
        }
        let mut out_shape = batch_shape;
        out_shape.push(n);
        out_shape.push(m);
        return mk_nd_f64_prec(py, out, out_shape, prec);
    }
    if let Some((data, shape, prec)) = as_promoted_complex_nd(a)? {
        check_stacked_2d(&shape)?;
        let len = shape.len();
        let m = shape[len - 2];
        let n = shape[len - 1];
        if len == 2 {
            let out = zc::pinv(&data, m, n, resolved_rcond).map_err(linalg_err)?;
            return mk2_c64_prec(py, out, n, m, prec);
        }
        let batch_shape = shape[..len - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch * m * n);
        for b in 0..batch {
            let slice = &data[b * m * n..(b + 1) * m * n];
            out.extend(zc::pinv(slice, m, n, resolved_rcond).map_err(linalg_err)?);
        }
        let mut out_shape = batch_shape;
        out_shape.push(n);
        out_shape.push(m);
        return mk_nd_c128_prec(py, out, out_shape, prec);
    }
    Err(unsupported_dtype("pinv"))
}

// ═══════════════════════════ matrix_rank ════════════════════════════════

#[pyfunction]
#[allow(non_snake_case)]
#[pyo3(signature = (A, tol=None, hermitian=None, rtol=None))]
fn matrix_rank(
    py: Python<'_>,
    A: &Bound<'_, PyAny>,
    tol: Option<f64>,
    hermitian: Option<&Bound<'_, PyAny>>,
    rtol: Option<f64>,
) -> PyResult<Py<PyAny>> {
    // `hermitian=` is plain Python truthiness in real numpy (measured
    // 2026-08-02, same as `svd`/`pinv` above); no ABSENT-vs-explicit-None
    // ambiguity here (unlike `rtol`'s `_NoValue` sentinel). PyO3 cannot
    // combine a bare `&Bound<PyAny>` param with a literal `bool` default
    // in `#[pyo3(signature=...)]` (macro-generated default-value code
    // can't convert a `bool` literal into a `&Bound<PyAny>`), so this is
    // `Option<&Bound<PyAny>>` with `hermitian=None`, defaulted by hand.
    let hermitian = match hermitian {
        None => false,
        Some(v) => v.is_truthy()?,
    };
    if tol.is_some() && rtol.is_some() {
        return Err(PyValueError::new_err("`tol` and `rtol` can't be both set."));
    }
    // Real numpy (`inspect.getsource`): `A.ndim < 2` is checked FIRST,
    // unconditionally -- before `hermitian` is ever looked at -- and
    // returns a plain Python `int(not all(A == 0))`, not an ndarray.
    // 2026-08-02: `hermitian=True` never received this check at all (it
    // went straight to a 2-D-only extractor), so e.g.
    // `matrix_rank(zeros(5), hermitian=True)` raised anionpy's own
    // "batched/1-D inputs are out of scope" ValueError where real numpy
    // returns `0`. Confirmed via a 936-case shape x dtype x hermitian x
    // tol/rtol probe: every 1-D/0-d shape crossed with hermitian=True
    // mismatched (210/936 total). Hoisted here, shared by both branches,
    // exactly mirroring numpy's own check order.
    if let Some(peek) = try_ndarray_any(A) {
        if peek.shape().len() < 2 {
            if let Some((data, shape, _prec)) = as_promoted_real_nd(A)? {
                let _ = shape;
                let rank: i64 = if data.iter().all(|&v| v == 0.0) { 0 } else { 1 };
                return rank.into_py_any(py);
            }
            if let Some((data, shape, _prec)) = as_promoted_complex_nd(A)? {
                let _ = shape;
                let rank: i64 =
                    if data.iter().all(|&v| v == Complex64::new(0.0, 0.0)) { 0 } else { 1 };
                return rank.into_py_any(py);
            }
            return Err(unsupported_dtype("matrix_rank"));
        }
    }
    if hermitian {
        // Same threshold formula as the general path (`rc::matrix_rank`'s
        // default: `smax * max(m,n) * eps`; explicit `tol`/`rtol` resolved
        // the same way numpy does), just fed singular values from the
        // Hermitian eigendecomposition (`hermitian_svd_real`/`_complex`,
        // same math as `svd_hermitian`) instead of the general SVD.
        //
        // 2026-08-02: was 2-D-only (`as_promoted_real2`/`_complex2`, no
        // batching), same defect class `eig`/`eigh`/`pinv(hermitian=True)`
        // all had before their own fixes -- rewritten with
        // `check_stacked_square` + N-D batching. Also fixed: the `eps`
        // factor was hardcoded to `f64::EPSILON` regardless of input
        // precision; real numpy's default tolerance is
        // `finfo(S.dtype).eps`, and `S.dtype` tracks float32/complex64
        // input precision (verified live: `f32::EPSILON` vs
        // `f64::EPSILON` genuinely flip the rank of a matrix with a
        // singular value between the two thresholds).
        if let Some((data, shape, prec)) = as_promoted_real_nd(A)? {
            let n = check_stacked_square(&shape)?;
            let eps = if prec == Prec::Single { f32::EPSILON as f64 } else { f64::EPSILON };
            let resolve = |slice: &[f64]| -> PyResult<i64> {
                let (_u, s, _vt) = hermitian_svd_real(slice, n).map_err(linalg_err)?;
                let smax = s.iter().cloned().fold(0.0f64, f64::max);
                let t = match (tol, rtol) {
                    (Some(t), None) => t,
                    (None, Some(rt)) => smax * rt,
                    (None, None) => smax * (n as f64) * eps,
                    (Some(_), Some(_)) => unreachable!(),
                };
                Ok(s.iter().filter(|&&v| v > t).count() as i64)
            };
            if shape.len() == 2 {
                return mk_scalar_i64(py, resolve(&data)?);
            }
            let batch_shape = shape[..shape.len() - 2].to_vec();
            let batch: usize = batch_shape.iter().product();
            let mut out = Vec::with_capacity(batch);
            for b in 0..batch {
                let slice = &data[b * n * n..(b + 1) * n * n];
                out.push(resolve(slice)?);
            }
            return mk_nd_i64(py, out, batch_shape);
        }
        if let Some((data, shape, prec)) = as_promoted_complex_nd(A)? {
            let n = check_stacked_square(&shape)?;
            let eps = if prec == Prec::Single { f32::EPSILON as f64 } else { f64::EPSILON };
            let resolve = |slice: &[Complex64]| -> PyResult<i64> {
                let (_u, s, _vt) = hermitian_svd_complex(slice, n).map_err(linalg_err)?;
                let smax = s.iter().cloned().fold(0.0f64, f64::max);
                let t = match (tol, rtol) {
                    (Some(t), None) => t,
                    (None, Some(rt)) => smax * rt,
                    (None, None) => smax * (n as f64) * eps,
                    (Some(_), Some(_)) => unreachable!(),
                };
                Ok(s.iter().filter(|&&v| v > t).count() as i64)
            };
            if shape.len() == 2 {
                return mk_scalar_i64(py, resolve(&data)?);
            }
            let batch_shape = shape[..shape.len() - 2].to_vec();
            let batch: usize = batch_shape.iter().product();
            let mut out = Vec::with_capacity(batch);
            for b in 0..batch {
                let slice = &data[b * n * n..(b + 1) * n * n];
                out.push(resolve(slice)?);
            }
            return mk_nd_i64(py, out, batch_shape);
        }
        return Err(unsupported_dtype("matrix_rank"));
    }
    // General (non-hermitian) path: `matrix_rank` does NOT require square
    // input (per numpy's own `_assert_stacked_2d`-only check) and numpy
    // batches over leading dims, collapsing the trailing 2 into a scalar
    // rank per matrix. The `ndim < 2` special case is handled above,
    // shared with the hermitian branch.
    if let Some((data, shape, prec)) = as_promoted_real_nd(A)? {
        let eps = if prec == Prec::Single { f32::EPSILON as f64 } else { f64::EPSILON };
        let len = shape.len();
        let m = shape[len - 2];
        let n = shape[len - 1];
        let resolve = |slice: &[f64]| -> PyResult<usize> {
            let resolved_tol = match (tol, rtol) {
                (Some(t), None) => Some(t),
                (None, Some(rt)) => {
                    let (_u, s, _vt) = rc::svd(slice, m, n).map_err(linalg_err)?;
                    let smax = s.iter().cloned().fold(0.0f64, f64::max);
                    Some(smax * rt)
                }
                (None, None) => None, // core computes smax*max(m,n)*eps itself
                (Some(_), Some(_)) => unreachable!(),
            };
            rc::matrix_rank(slice, m, n, resolved_tol, eps).map_err(linalg_err)
        };
        if len == 2 {
            return mk_scalar_i64(py, resolve(&data)? as i64);
        }
        let batch_shape = shape[..len - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let slice = &data[b * m * n..(b + 1) * m * n];
            out.push(resolve(slice)? as i64);
        }
        return mk_nd_i64(py, out, batch_shape);
    }
    if let Some((data, shape, prec)) = as_promoted_complex_nd(A)? {
        let eps = if prec == Prec::Single { f32::EPSILON as f64 } else { f64::EPSILON };
        let len = shape.len();
        let m = shape[len - 2];
        let n = shape[len - 1];
        let resolve = |slice: &[Complex64]| -> PyResult<usize> {
            let resolved_tol = match (tol, rtol) {
                (Some(t), None) => Some(t),
                (None, Some(rt)) => {
                    let (_u, s, _vt) = zc::svd(slice, m, n).map_err(linalg_err)?;
                    let smax = s.iter().cloned().fold(0.0f64, f64::max);
                    Some(smax * rt)
                }
                (None, None) => None,
                (Some(_), Some(_)) => unreachable!(),
            };
            zc::matrix_rank(slice, m, n, resolved_tol, eps).map_err(linalg_err)
        };
        if len == 2 {
            return mk_scalar_i64(py, resolve(&data)? as i64);
        }
        let batch_shape = shape[..len - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let slice = &data[b * m * n..(b + 1) * m * n];
            out.push(resolve(slice)? as i64);
        }
        return mk_nd_i64(py, out, batch_shape);
    }
    Err(unsupported_dtype("matrix_rank"))
}

// ════════════════════════════════ cond ══════════════════════════════════

#[pyfunction]
#[pyo3(signature = (x, p=None))]
fn cond(py: Python<'_>, x: &Bound<'_, PyAny>, p: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    // `p='fro'` is a string sentinel, not extractable as f64 -- detect it
    // explicitly before falling into the "unsupported" branch. Verified
    // 2026-08-02 against `inspect.getsource(numpy.linalg._linalg.cond)`:
    // every `p` value numpy's `cond` recognizes (`None`, `1`, `-1`, `2`,
    // `-2`, `inf`, `-inf`, `'fro'`) is now implemented on both the real
    // and complex paths (see `real::cond` in dense_linalg.rs and
    // `complex_cond` above for the shared `norm(x,p)*norm(inv(x),p)`
    // formula `fro`/`-1`/`-inf` route through).
    let mut fro = false;
    let p_f64: Option<f64> = match p {
        None => None,
        Some(obj) => {
            if let Ok(v) = obj.extract::<f64>() {
                Some(v)
            } else if let Ok(s) = obj.extract::<String>() {
                if s == "fro" {
                    fro = true;
                    None
                } else {
                    return Err(PyNotImplementedError::new_err(format!(
                        "anionpy.linalg.cond: unsupported ord {s:?} (only None, 1, -1, 2, -2, inf, \
                         -inf, 'fro' are implemented)"
                    )));
                }
            } else {
                return Err(PyNotImplementedError::new_err(
                    "anionpy.linalg.cond: p must be a real number or the string 'fro'",
                ));
            }
        }
    };
    // numpy's `cond` checks `x.size == 0` FIRST, before any dimensionality
    // check -- verified directly: `cond(np.zeros((0,)))` raises
    // `LinAlgError('cond is not defined on empty arrays')`, not the
    // "1-dimensional array given" message a plain `_assert_stacked_2d`
    // would produce. Peek the shape via the same ungated `try_ndarray_any`
    // the promoted extractors use, before promoting/validating further.
    if let Some(peek) = try_ndarray_any(x) {
        let shape = peek.shape();
        if shape.iter().product::<usize>() == 0 {
            return Err(linalg_err_raw("cond is not defined on empty arrays"));
        }
    }
    if let Some((data, shape, prec)) = as_promoted_real_nd(x)? {
        check_stacked_2d(&shape)?;
        let len = shape.len();
        let m = shape[len - 2];
        let n = shape[len - 1];
        // numpy's default `p=None`, AND `p in {2, -2}`, all use the SVD
        // directly (`if p is None or p in {2, -2}: s = svd(x, ...)`,
        // verified via `inspect.getsource(numpy.linalg._linalg.cond)`
        // 2026-08-02) -- which is defined for ANY m×n shape, not just
        // square, and was WRONGLY routed through the square-only
        // `rc::cond`/`complex_cond` path for `p=2`/`p=-2` before this fix
        // (a genuine correctness bug, found via a p-value sweep: `cond`
        // on a non-square matrix with the default `p=None` succeeded, but
        // explicit `p=2` on the SAME matrix incorrectly raised "Last 2
        // dimensions of the array must be square"). For SQUARE matrices
        // the two code paths are mathematically equivalent (numpy's own
        // `p` not in `{None,2,-2}` branch computes `norm(x,p)*norm(inv(x),p)`,
        // and for `p=2` that's `smax(x) * (1/smin(x))` = the same ratio
        // the SVD path computes directly), so this fix only changes
        // behavior for the previously-incorrectly-rejected non-square
        // case. `p=-2` uses `s.min()/s.max()` (the reciprocal ratio, per
        // numpy's own `if p == -2: r = s[..., -1] / s[..., 0]` with `s`
        // descending) rather than `p=2`'s `s.max()/s.min()`.
        let is_neg2 = p_f64 == Some(-2.0);
        if !fro && (p_f64.is_none() || p_f64 == Some(2.0) || is_neg2) {
            let compute = |slice: &[f64]| -> PyResult<f64> {
                let (_u, s, _vt) = rc::svd(slice, m, n).map_err(linalg_err)?;
                let smax = s.iter().cloned().fold(f64::MIN, f64::max);
                let smin = s.iter().cloned().fold(f64::MAX, f64::min);
                Ok(if is_neg2 { smin / smax } else { smax / smin })
            };
            if len == 2 {
                return mk_scalar_f64_prec(py, compute(&data)?, prec);
            }
            let batch_shape = shape[..len - 2].to_vec();
            let batch: usize = batch_shape.iter().product();
            let mut out = Vec::with_capacity(batch);
            for b in 0..batch {
                out.push(compute(&data[b * m * n..(b + 1) * m * n])?);
            }
            return mk_nd_f64_prec(py, out, batch_shape, prec);
        }
        let msquare = check_stacked_square(&shape)?;
        if len == 2 {
            let v = rc::cond(&data, msquare, p_f64, fro).map_err(linalg_err)?;
            return mk_scalar_f64_prec(py, v, prec);
        }
        let batch_shape = shape[..len - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let slice = &data[b * msquare * msquare..(b + 1) * msquare * msquare];
            out.push(rc::cond(slice, msquare, p_f64, fro).map_err(linalg_err)?);
        }
        return mk_nd_f64_prec(py, out, batch_shape, prec);
    }
    if let Some((data, shape, prec)) = as_promoted_complex_nd(x)? {
        check_stacked_2d(&shape)?;
        let len = shape.len();
        let m = shape[len - 2];
        let n = shape[len - 1];
        let is_neg2 = p_f64 == Some(-2.0);
        if !fro && (p_f64.is_none() || p_f64 == Some(2.0) || is_neg2) {
            let compute = |slice: &[Complex64]| -> PyResult<f64> {
                let (_u, s, _vt) = zc::svd(slice, m, n).map_err(linalg_err)?;
                let smax = s.iter().cloned().fold(f64::MIN, f64::max);
                let smin = s.iter().cloned().fold(f64::MAX, f64::min);
                Ok(if is_neg2 { smin / smax } else { smax / smin })
            };
            if len == 2 {
                return mk_scalar_f64_prec(py, compute(&data)?, prec);
            }
            let batch_shape = shape[..len - 2].to_vec();
            let batch: usize = batch_shape.iter().product();
            let mut out = Vec::with_capacity(batch);
            for b in 0..batch {
                out.push(compute(&data[b * m * n..(b + 1) * m * n])?);
            }
            return mk_nd_f64_prec(py, out, batch_shape, prec);
        }
        let msquare = check_stacked_square(&shape)?;
        if len == 2 {
            let v = complex_cond(&data, msquare, p_f64, fro).map_err(linalg_err)?;
            return mk_scalar_f64_prec(py, v, prec);
        }
        let batch_shape = shape[..len - 2].to_vec();
        let batch: usize = batch_shape.iter().product();
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let slice = &data[b * msquare * msquare..(b + 1) * msquare * msquare];
            out.push(complex_cond(slice, msquare, p_f64, fro).map_err(linalg_err)?);
        }
        return mk_nd_f64_prec(py, out, batch_shape, prec);
    }
    Err(unsupported_dtype("cond"))
}

// ═══════════════════════════════ lstsq ══════════════════════════════════

#[pyfunction]
#[pyo3(signature = (a, b, rcond=None))]
fn lstsq(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, rcond: Option<f64>) -> PyResult<Py<PyAny>> {
    // 2026-08-01: real numpy's `lstsq` validates a/b shape COMPATIBILITY
    // before it ever reaches the dtype-support check inside the LAPACK
    // dispatch (verified directly: `np.linalg.lstsq` on a float16 `a` with
    // a mismatched-length `b` raises `LinAlgError("Incompatible
    // dimensions")`, NOT the float16 TypeError) -- but `as_promoted_real2`
    // below rejects float16 as its very first step, so calling it on `a`
    // first would raise the wrong exception for this combination. Peeking
    // shapes here (via the same ungated `try_ndarray_any` the promoted
    // extractors use internally) reproduces numpy's actual check order
    // without duplicating the rest of the promotion logic.
    if let (Some(pa), Some(pb)) = (try_ndarray_any(a), try_ndarray_any(b)) {
        if pa.shape().len() == 2 {
            let m = pa.shape()[0];
            let b_leading = pb.shape().first().copied();
            if let Some(bl) = b_leading {
                if bl != m {
                    return Err(linalg_err_raw("Incompatible dimensions"));
                }
            }
        }
    }
    if let Some((adata, m, n, aprec)) = as_promoted_real2(a)? {
        let eps = if aprec == Prec::Single { f32::EPSILON as f64 } else { f64::EPSILON };
        let resolved_rcond = rcond.unwrap_or(eps * (m.max(n) as f64));
        let (bdata, nrhs, is_1d, bprec) = if let Some((v, bprec)) = as_promoted_real1(b)? {
            if v.len() != m {
                return Err(linalg_err_raw("Incompatible dimensions"));
            }
            (v, 1usize, true, bprec)
        } else if let Some((bd, bm, bk, bprec)) = as_promoted_real2(b)? {
            if bm != m {
                return Err(linalg_err_raw("Incompatible dimensions"));
            }
            (bd, bk, false, bprec)
        } else {
            return Err(unsupported_dtype("lstsq"));
        };
        let prec = combine_prec(aprec, bprec);
        let (x, s, rank) = rc::lstsq(&adata, m, n, &bdata, nrhs, resolved_rcond).map_err(linalg_err)?;
        // numpy's residuals rule (verified against numpy 2.5.1 source):
        // populated only when rank == n and m > n; empty otherwise.
        let residuals: Vec<f64> = if (rank as usize) == n && m > n {
            // sum of squared residuals per rhs column: ||b - a@x||^2
            (0..nrhs)
                .map(|j| {
                    (0..m)
                        .map(|i| {
                            let bax: f64 = (0..n).map(|k| adata[i * n + k] * x[k * nrhs + j]).sum();
                            let r = bdata[i * nrhs + j] - bax;
                            r * r
                        })
                        .sum()
                })
                .collect()
        } else {
            vec![]
        };
        let xpy = if is_1d {
            mk1_f64_prec(py, x, prec)?
        } else {
            mk2_f64_prec(py, x, n, nrhs, prec)?
        };
        let respy = mk1_f64_prec(py, residuals, prec)?;
        let spy = mk1_f64_prec(py, s, prec)?;
        let rankpy = mk_scalar_i32(py, rank)?;
        return Ok((xpy, respy, rankpy, spy).into_py_any(py)?);
    }
    if let Some((adata, m, n, aprec)) = as_promoted_complex2(a)? {
        let eps = if aprec == Prec::Single { f32::EPSILON as f64 } else { f64::EPSILON };
        let resolved_rcond = rcond.unwrap_or(eps * (m.max(n) as f64));
        let (bdata, nrhs, is_1d, bprec) = if let Some((v, bprec)) = as_promoted_complex1(b)? {
            if v.len() != m {
                return Err(linalg_err_raw("Incompatible dimensions"));
            }
            (v, 1usize, true, bprec)
        } else if let Some((bd, bm, bk, bprec)) = as_promoted_complex2(b)? {
            if bm != m {
                return Err(linalg_err_raw("Incompatible dimensions"));
            }
            (bd, bk, false, bprec)
        } else {
            return Err(unsupported_dtype("lstsq"));
        };
        let prec = combine_prec(aprec, bprec);
        let (x, s, rank) = zc::lstsq(&adata, m, n, &bdata, nrhs, resolved_rcond).map_err(linalg_err)?;
        let residuals: Vec<f64> = if (rank as usize) == n && m > n {
            (0..nrhs)
                .map(|j| {
                    (0..m)
                        .map(|i| {
                            let bax: Complex64 = (0..n).map(|k| adata[i * n + k] * x[k * nrhs + j]).sum();
                            let r = bdata[i * nrhs + j] - bax;
                            r.norm_sqr()
                        })
                        .sum()
                })
                .collect()
        } else {
            vec![]
        };
        let xpy = if is_1d {
            mk1_c64_prec(py, x, prec)?
        } else {
            mk2_c64_prec(py, x, n, nrhs, prec)?
        };
        let respy = mk1_f64_prec(py, residuals, prec)?;
        let spy = mk1_f64_prec(py, s, prec)?;
        let rankpy = mk_scalar_i32(py, rank)?;
        return Ok((xpy, respy, rankpy, spy).into_py_any(py)?);
    }
    Err(unsupported_dtype("lstsq"))
}

// ═══════════════════════════════ trace ══════════════════════════════════

/// `offset != 0`: the LAPACK-backed `rc::trace`/`zc::trace` core (used for
/// the already-declared `offset=0` path above, unchanged) only ever sums
/// the main diagonal, so any other offset used to be a flat
/// `NotImplementedError`. 2026-08-01 fix: `offset != 0` is not a LAPACK
/// question at all -- it is pure array-shape bookkeeping, which
/// `ionp_core::manip::trace` (already implemented, generalized to N-D and
/// reused as-is here, not a new numeric code path invented for this
/// binding) already does: `linalg.trace`'s own signature has no
/// axis1/axis2 parameter (verified against real numpy 2.5.1's
/// `inspect.signature` -- unlike bare `numpy.trace`, the `linalg.` variant
/// is Array-API-shaped and always operates on the last two axes), so
/// `axis1=-2, axis2=-1` is passed unconditionally. `manip::trace` sums at
/// the array's own dtype and then casts, matching the "sum at full
/// precision, cast only the final scalar" rule the `dtype=` kwarg already
/// follows on the `offset=0` path (mk_scalar_f64_dtype/mk_scalar_c128_dtype
/// above) -- so for a 2-D input this is the exact same accumulate-then-cast
/// contract, just routed through the generic core instead of LAPACK's own
/// trace loop.
///
/// 2026-08-01: verified against real numpy 2.5.1 across bool/int8-64/
/// float32/float64/complex64/complex128 at offset in {1,-1,2} -- all
/// bit-exact. `float16` is the one exception: found a genuine 1-ULP
/// mismatch (`np.linalg.trace(a16, offset=-1)` -> `1.135`, this path ->
/// `1.137`, for a hand-inspected 3-element diagonal where
/// `diag.astype(f64).sum()` rounds to `1.135` at f16 too -- so `manip::
/// trace`'s own float16 accumulation, not a dtype-cast-order issue here,
/// diverges from numpy's). `manip.rs` is not an owned file this task, so
/// rather than patch that shared accumulator, float16 is explicitly
/// rejected below as a documented scope gap instead of silently returning
/// the wrong value.
fn trace_offset(py: Python<'_>, x: &Bound<'_, PyAny>, offset: i64, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let arr = as_ndarray_any(x)?;
    if arr.ndim() < 2 {
        // numpy's `np.trace`/`np.linalg.trace` on a <2-D array raises this
        // exact `ValueError` (from `diagonal()`'s own C-level guard, which
        // `trace` calls internally) -- verified against real numpy 2.5.1:
        // `np.linalg.trace(np.zeros((0,)))` gives
        // `ValueError: diag requires an array of at least two dimensions`,
        // not a class- or text-generic message. A caller matching on
        // message text must not be able to tell anionpy and numpy apart here.
        return Err(PyValueError::new_err(
            "diag requires an array of at least two dimensions",
        ));
    }
    if arr.dtype() == DType::F16 && offset != 0 {
        return Err(PyNotImplementedError::new_err(
            "anionpy.linalg.trace: offset!=0 on a float16 array is not implemented (manip::trace's \
             float16 accumulation has a measured 1-ULP divergence from numpy's own; not a dtype-cast \
             gap, an unowned-file accumulation-order gap)",
        ));
    }
    let target = match dtype {
        Some(d) => Some(crate::dtype_from_pyobj_no_su(d)?),
        None => None,
    };
    let out = manip::trace(&arr, offset as isize, -2, -1, target).map_err(linalg_ionp_err)?;
    // 2026-08-01: `manip::trace` (unowned file, `ionp-core/src/manip.rs`)
    // passes an explicit `dtype` through to its internal
    // `ufunc::reduce_axis` call as the CAST-BEFORE-REDUCE dtype, but
    // `reduce_axis` (also unowned, `ionp-core/src/ufunc.rs`) applies its
    // own accumulator-widening rule on top of that regardless (e.g.
    // int32 -> int64), which is correct for a bare `.sum()`-style
    // reduction but wrong for `trace(..., dtype=X)`, where numpy's own
    // `np.linalg.trace`/`np.trace` return EXACTLY the requested dtype
    // (verified: `np.linalg.trace(a, dtype=np.int32).dtype == int32`,
    // measured directly, not upcast). Re-casting the OUTPUT to the
    // explicitly requested target here (a local fix, inside this owned
    // file, not touching manip.rs/ufunc.rs) restores that exact-dtype
    // contract without altering either unowned function's general
    // reduction behavior for anything else that calls them.
    let out = match target {
        Some(t) if out.dtype() != t => out.cast_to(t),
        _ => out,
    };
    // `manip::trace` on a plain 2-D input returns a 0-d NdArray (batch dims
    // removed, none left). CORRECTED 2026-08-02 (scalar-return-type audit,
    // docs/scalar-return-type-defect.md): real numpy does NOT keep that as
    // an array here -- `np.linalg.trace(np.eye(2))` is `numpy.float64`, a
    // genuine numpy scalar, same defect class/fix as the `reductions.rs`
    // family (see `crate::numpy_scalar_from_0d`). Only a batched (>2-D)
    // input, which leaves real batch dimensions behind, stays an array.
    if out.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &out);
    }
    PyArray { inner: out }.into_py_any(py)
}

#[pyfunction]
#[pyo3(signature = (x, offset=0, *, dtype=None))]
fn trace(py: Python<'_>, x: &Bound<'_, PyAny>, offset: i64, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    // 2026-08-01: previously this `offset == 0` (numpy's default) branch
    // used the old `as_real2`/`as_complex2` extraction (f64/c128 only),
    // while `offset != 0` already routed through `trace_offset` ->
    // `manip::trace`, which is dtype-general (matches numpy's exact
    // accumulator-promotion rule: bool/int8-64 -> int64, uint8-64 ->
    // uint64, float16/32/64 and complex64/128 unchanged -- see
    // `manip::trace_accum_dtype`). That meant the *default* call was the
    // one still restricted to two dtypes. Routing unconditionally through
    // `trace_offset` (offset=0 is just the identity offset for the same
    // code path) fixes the reported "9 of 11 dtypes raise" defect using
    // the already-verified generic path, with the same float16 gap
    // documented on `trace_offset` above (not introduced here).
    trace_offset(py, x, offset, dtype)
}

// ═══════════════ 2026-08-02: net-new Array-API-shaped items ═════════════
//
// `cross`, `tensordot`, `multi_dot`, `tensorinv`, `tensorsolve`,
// `vector_norm`, `matrix_norm`, `norm` were entirely absent (verified via
// `register()` below, before this pass -- no partial stub existed for any
// of them). Each is implemented by composing already-existing, already-
// tested primitives (`NdArray::get_view`/`transpose_axes`/`reshape`,
// `ionp_core::ufunc::binary_op`/`outer_binary`, `ionp_core::creation::
// stack`, `ionp_ion::matmul::matmul`, and the existing promoted-2-D `rc`/
// `zc` `inv`/`solve`/`svd` LAPACK core this file's `inv`/`solve`/`svd`
// bindings above already use) rather than any new LAPACK call.

fn normalize_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    let nd = ndim as isize;
    let a = if axis < 0 { axis + nd } else { axis };
    if a < 0 || a >= nd {
        return Err(PyValueError::new_err(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis, ndim
        )));
    }
    Ok(a as usize)
}

fn full_slices(ndim: usize) -> Vec<SliceItem> {
    vec![
        SliceItem::Slice {
            start: None,
            stop: None,
            step: None
        };
        ndim
    ]
}

/// Extract the (ndim-1)-D slice of `arr` at `index` along `axis` (removes
/// that axis, numpy fancy-single-index semantics), used by `cross` to pull
/// each of the 3 components.
fn index_along_axis(arr: &NdArray, axis: usize, index: isize) -> PyResult<NdArray> {
    let mut items = full_slices(arr.ndim());
    items[axis] = SliceItem::Index(index);
    arr.get_view(&items).map_err(linalg_ionp_err)
}

fn wrap_ndarray<'py>(py: Python<'py>, arr: NdArray) -> PyResult<Bound<'py, PyAny>> {
    Ok(Py::new(py, PyArray { inner: arr })?.into_bound(py).into_any())
}

// ─────────────────────────────── cross ───────────────────────────────────

#[pyfunction]
#[pyo3(signature = (x1, x2, /, *, axis=-1))]
fn cross(py: Python<'_>, x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>, axis: isize) -> PyResult<Py<PyAny>> {
    let a = as_ndarray_any(x1)?;
    let b = as_ndarray_any(x2)?;
    let axis_a = normalize_axis(axis, a.ndim())?;
    let axis_b = normalize_axis(axis, b.ndim())?;
    if a.shape()[axis_a] != 3 || b.shape()[axis_b] != 3 {
        // numpy's exact text interpolates the ACTUAL found axis lengths
        // (not ndim) for both operands, always in `a, b` order regardless
        // of which one was the bad one -- verified against real numpy
        // 2.5.1: `np.linalg.cross(np.zeros((2,)), np.zeros((2,)))` raises
        // `'...but they are 2 and 2 dimensional instead.'`.
        return Err(PyValueError::new_err(format!(
            "Both input arrays must be (arrays of) 3-dimensional vectors, but they are {} and {} dimensional instead.",
            a.shape()[axis_a],
            b.shape()[axis_b]
        )));
    }
    let a0 = index_along_axis(&a, axis_a, 0)?;
    let a1 = index_along_axis(&a, axis_a, 1)?;
    let a2 = index_along_axis(&a, axis_a, 2)?;
    let b0 = index_along_axis(&b, axis_b, 0)?;
    let b1 = index_along_axis(&b, axis_b, 1)?;
    let b2 = index_along_axis(&b, axis_b, 2)?;

    let mul = |x: &NdArray, y: &NdArray| {
        ionp_core::ufunc::binary_op(ionp_core::ufunc::BinaryOp::Multiply, x, y).map_err(crate::to_py_err)
    };
    let sub = |x: &NdArray, y: &NdArray| {
        // `crate::to_py_err` (not the local `linalg_ionp_err`, which always
        // raises `ValueError`) so that a bool-dtype subtract correctly
        // raises numpy's real `TypeError` ("numpy boolean subtract, the `-`
        // operator, is not supported...") -- verified against real numpy
        // 2.5.1 that `np.linalg.cross` on bool input reaches this exact
        // check and raises TypeError, not ValueError.
        ionp_core::ufunc::binary_op(ionp_core::ufunc::BinaryOp::Subtract, x, y).map_err(crate::to_py_err)
    };

    let c0 = sub(&mul(&a1, &b2)?, &mul(&a2, &b1)?)?;
    let c1 = sub(&mul(&a2, &b0)?, &mul(&a0, &b2)?)?;
    let c2 = sub(&mul(&a0, &b1)?, &mul(&a1, &b0)?)?;

    let out_ndim = c0.ndim() + 1;
    let stack_axis = normalize_axis(axis, out_ndim)? as isize;
    let out = ionp_core::manip::stack(&[&c0, &c1, &c2], stack_axis).map_err(linalg_ionp_err)?;
    Ok(Py::new(py, PyArray { inner: out })?.into_any())
}

// ─────────────────────────────── tensordot ────────────────────────────────

/// Parse `axes=`: either a single non-negative int `N` (contract the last
/// `N` axes of x1 with the first `N` axes of x2 -- numpy's default,
/// `axes=2`) or a pair of int-sequences (contract `axes_a[i]` of x1 with
/// `axes_b[i]` of x2, matched pairwise) -- numpy's own documented
/// `tensordot` signature.
/// numpy's `tensordot` indexes the *raw shape tuple* with each axis value
/// (`as_[axes_a[k]]`/`bs[axes_b[k]]` in `numpy/_core/numeric.py`) -- plain
/// Python tuple-indexing semantics, NOT the `normalize_axis`/`AxisError`
/// semantics used elsewhere in this file (e.g. by `cross`/`tensorsolve`).
/// An out-of-range axis (from either the default `axes=N` int form, where
/// numpy builds `axes_a = list(range(-N, 0))` unclamped against `a.ndim`,
/// or an explicit tuple-of-sequences form) therefore raises Python's own
/// `IndexError: tuple index out of range`, verified against real numpy for
/// both forms. This must be bounds-checked HERE, before any of these
/// indices are used to index `a.shape()`/`b.shape()` downstream in
/// `tensordot` -- doing the subtraction/indexing unchecked (the previous
/// implementation) panics via `usize` underflow or out-of-bounds slice
/// indexing for any axis count/index exceeding the array's ndim.
fn tensordot_index(idx: i64, ndim: usize) -> PyResult<usize> {
    let normalized = if idx < 0 { idx + ndim as i64 } else { idx };
    if normalized < 0 || normalized >= ndim as i64 {
        return Err(PyIndexError::new_err("tuple index out of range"));
    }
    Ok(normalized as usize)
}

fn parse_tensordot_axes(axes: &Bound<'_, PyAny>, a_ndim: usize, b_ndim: usize) -> PyResult<(Vec<usize>, Vec<usize>)> {
    if let Ok(n) = axes.extract::<i64>() {
        // numpy's real `tensordot` builds `axes_a = list(range(-axes, 0))`
        // and `axes_b = list(range(0, axes))` -- plain CPython `range()`
        // semantics, not a normalize-then-reject rule. For ANY negative
        // `n`, `range(-n, 0)` has start `-n > 0` and stop `0`, which is an
        // EMPTY range (Python never rejects this), and likewise
        // `range(0, n)` for negative `n` is empty too -- so a negative
        // `axes=` contracts ZERO axes (a pure outer product, `A.shape +
        // B.shape`), not an error. Verified directly against real numpy
        // 2.5.1 for axes=-1, -2, -5 (uniform across negative N, not just
        // -1): `np.tensordot(A, B, axes=-1)` on A=(2,3,4)/B=(4,3,2)
        // succeeds with shape (2,3,4,4,3,2). The previous code rejected
        // every negative `n` with a `ValueError` that has no real numpy
        // counterpart. `n >= 0` keeps the exact prior (already-correct)
        // Rust `range` transliteration.
        let axes_a: Vec<usize> = if n < 0 {
            Vec::new()
        } else {
            (-n..0).map(|i| tensordot_index(i, a_ndim)).collect::<PyResult<_>>()?
        };
        let axes_b: Vec<usize> = if n < 0 {
            Vec::new()
        } else {
            (0..n).map(|i| tensordot_index(i, b_ndim)).collect::<PyResult<_>>()?
        };
        return Ok((axes_a, axes_b));
    }
    if let Ok((ra, rb)) = axes.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>() {
        let to_axes = |obj: &Bound<'_, PyAny>, ndim: usize| -> PyResult<Vec<usize>> {
            if let Ok(i) = obj.extract::<i64>() {
                return Ok(vec![tensordot_index(i, ndim)?]);
            }
            let raw: Vec<i64> = obj.extract()?;
            raw.into_iter().map(|v| tensordot_index(v, ndim)).collect()
        };
        let axes_a = to_axes(&ra, a_ndim)?;
        let axes_b = to_axes(&rb, b_ndim)?;
        if axes_a.len() != axes_b.len() {
            return Err(PyValueError::new_err("shape-mismatch for sum"));
        }
        return Ok((axes_a, axes_b));
    }
    Err(PyValueError::new_err("axes must be an integer or a tuple of two sequences"))
}

#[pyfunction]
#[pyo3(signature = (x1, x2, /, *, axes=None))]
fn tensordot(py: Python<'_>, x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>, axes: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let a = as_ndarray_any(x1)?;
    let b = as_ndarray_any(x2)?;
    let two = 2i64.into_pyobject(py)?.into_any();
    let axes_obj: &Bound<'_, PyAny> = axes.unwrap_or(&two);
    let (axes_a, axes_b) = parse_tensordot_axes(axes_obj, a.ndim(), b.ndim())?;

    let free_a: Vec<usize> = (0..a.ndim()).filter(|i| !axes_a.contains(i)).collect();
    let free_b: Vec<usize> = (0..b.ndim()).filter(|i| !axes_b.contains(i)).collect();

    let contracted_shape_a: Vec<usize> = axes_a.iter().map(|&i| a.shape()[i]).collect();
    let contracted_shape_b: Vec<usize> = axes_b.iter().map(|&i| b.shape()[i]).collect();
    if contracted_shape_a != contracted_shape_b {
        return Err(PyValueError::new_err("shape-mismatch for sum"));
    }
    let contract_size: usize = contracted_shape_a.iter().product();
    let free_a_shape: Vec<usize> = free_a.iter().map(|&i| a.shape()[i]).collect();
    let free_b_shape: Vec<usize> = free_b.iter().map(|&i| b.shape()[i]).collect();
    let free_a_size: usize = free_a_shape.iter().product();
    let free_b_size: usize = free_b_shape.iter().product();

    let mut order_a: Vec<isize> = free_a.iter().map(|&i| i as isize).collect();
    order_a.extend(axes_a.iter().map(|&i| i as isize));
    let mut order_b: Vec<isize> = axes_b.iter().map(|&i| i as isize).collect();
    order_b.extend(free_b.iter().map(|&i| i as isize));

    let a_t = a.transpose_axes(&order_a).map_err(linalg_ionp_err)?;
    let b_t = b.transpose_axes(&order_b).map_err(linalg_ionp_err)?;
    let a_2d = a_t
        .to_contiguous()
        .reshape(&[free_a_size, contract_size])
        .map_err(linalg_ionp_err)?;
    let b_2d = b_t
        .to_contiguous()
        .reshape(&[contract_size, free_b_size])
        .map_err(linalg_ionp_err)?;

    let out2d = ionp_ion::matmul::matmul(&a_2d, &b_2d).map_err(PyValueError::new_err)?;
    let mut out_shape = free_a_shape;
    out_shape.extend(free_b_shape);
    let out = out2d.reshape(&out_shape).map_err(linalg_ionp_err)?;
    Ok(Py::new(py, PyArray { inner: out })?.into_any())
}

// ─────────────────────────────── multi_dot ────────────────────────────────
//
// Direct port of numpy's own matrix-chain-order DP
// (`numpy.linalg._linalg._multi_dot_matrix_chain_order`) plus its recursive
// evaluator (`_multi_dot`) and its 3-matrix closed-form shortcut
// (`_multi_dot_three`) -- needed VERBATIM (not just "a correct
// parenthesization") because floating point multiplication is not
// associative: a different (still mathematically valid) evaluation order
// produces a bit-different answer, and the differential harness compares
// bits.

/// `multi_dot`'s chain-order DP and its `_multi_dot_three` shortcut assume
/// every intermediate product is a plain 2-D matmul, and numpy's own
/// `multi_dot` reaches this via `np.dot` internally -- so a shape mismatch
/// here must raise `np.dot`'s exact message (`"shapes (m,n) and (p,q) not
/// aligned: n (dim 1) != p (dim 0)"`), NOT `ionp_ion::matmul::matmul`'s own
/// gufunc-signature-style message (which is correct and unchanged for the
/// *general* `matmul` binding elsewhere in this crate -- this check is
/// local to `multi_dot`'s call sites only). Verified against real numpy
/// 2.5.1: `np.linalg.multi_dot([a, b])` with mismatched inner shapes
/// raises exactly that `np.dot`-style `ValueError`.
/// numpy's own `multi_dot` (for chains of 3+ arrays) asserts every
/// participant is exactly 2-D via `_assert_2d` BEFORE any shape-alignment
/// check runs (`numpy/linalg/_linalg.py`), raising `LinAlgError:
/// "{ndim}-dimensional array given. Array must be two-dimensional"` --
/// verified against real numpy 2.5.1 for both a 0-D and a 1-D array in a
/// non-first/non-last chain position. This must be checked here, before
/// `ash[1]`/`bsh[0]` are indexed below, since `multi_dot`'s caller only
/// promotes a 1-D array at the very first/last chain position (a 1-D or
/// 0-D array anywhere else -- or a 0-D array anywhere -- reaches this
/// function with ndim < 2, which previously indexed `shape()[1]`/`[0]`
/// unchecked and panicked instead of raising this error).
fn check_dot_aligned(a: &NdArray, b: &NdArray) -> PyResult<()> {
    let (ash, bsh) = (a.shape(), b.shape());
    if ash.len() != 2 {
        return Err(linalg_err_raw(&format!(
            "{}-dimensional array given. Array must be two-dimensional",
            ash.len()
        )));
    }
    if bsh.len() != 2 {
        return Err(linalg_err_raw(&format!(
            "{}-dimensional array given. Array must be two-dimensional",
            bsh.len()
        )));
    }
    if ash[1] != bsh[0] {
        return Err(PyValueError::new_err(format!(
            "shapes {} and {} not aligned: {} (dim 1) != {} (dim 0)",
            fmt_shape_tuple(ash),
            fmt_shape_tuple(bsh),
            ash[1],
            bsh[0]
        )));
    }
    Ok(())
}

fn fmt_shape_tuple(shape: &[usize]) -> String {
    if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","))
    }
}

/// numpy's own `multi_dot`, for exactly 2 arrays, shortcuts straight to
/// `return dot(arrays[0], arrays[1])` -- NOT matmul, and crucially WITHOUT
/// any dimensionality restriction to exactly 2-D (`numpy/linalg/_linalg.py`,
/// verified against real numpy 2.5.1 directly: `multi_dot([zeros((2,0,0)),
/// zeros((2,0,0))])` SUCCEEDS with shape `(2,0,2,0)`, matching `np.dot`'s
/// own contraction rule -- sum-product over `a`'s last axis and `b`'s
/// second-to-last axis, broadcast (NOT batched, unlike `matmul`) over every
/// other axis -- rather than raising the "Array must be two-dimensional"
/// `LinAlgError` the 3+-array chain path (`check_dot_aligned`,
/// `multi_dot_chain_order`) raises for a non-2-D participant. This was a
/// real, previously-undetected mismatch: the old code ran `check_dot_aligned`
/// (a 2-D-only check) unconditionally even for the 2-array shortcut, which
/// is only correct for the common ndim==2 case and silently wrong (an
/// incorrect LinAlgError, not merely a missing feature) for any higher-ndim
/// 2-array `multi_dot` call, discovered via the new empty-shape (2,0,0)
/// crossed-axis corpus case added this task. Mirrors `tensordot`'s own
/// transpose-then-reshape-then-matmul technique above, generalized to a
/// single fixed contraction axis pair. `multi_dot`'s caller has already
/// promoted a 1-D first/last operand to a 2-D row/column vector before this
/// is ever reached with such an operand, so only the `a.ndim() == 0` /
/// `b.ndim() == 0` edge (not exercised by any real numpy call site this
/// function is reachable from, and not part of this task's empty-shape
/// corpus) is guarded against explicitly below to avoid an underflow panic,
/// rather than replicating numpy's separate 0-d-scalar-multiply `dot`
/// behavior (an honest, narrow, out-of-scope gap for a genuinely 0-d input
/// specifically).
fn multi_dot_two(a: &NdArray, b: &NdArray) -> PyResult<NdArray> {
    if a.ndim() == 0 || b.ndim() == 0 {
        let bad = if a.ndim() == 0 { a } else { b };
        return Err(linalg_err_raw(&format!(
            "{}-dimensional array given. Array must be two-dimensional",
            bad.ndim()
        )));
    }
    let a_ax = a.ndim() - 1;
    let b_ax = if b.ndim() >= 2 { b.ndim() - 2 } else { 0 };
    if a.shape()[a_ax] != b.shape()[b_ax] {
        return Err(PyValueError::new_err(format!(
            "shapes {} and {} not aligned: {} (dim {}) != {} (dim {})",
            fmt_shape_tuple(a.shape()),
            fmt_shape_tuple(b.shape()),
            a.shape()[a_ax],
            a_ax,
            b.shape()[b_ax],
            b_ax
        )));
    }
    let free_a: Vec<usize> = (0..a.ndim()).filter(|&i| i != a_ax).collect();
    let free_b: Vec<usize> = (0..b.ndim()).filter(|&i| i != b_ax).collect();
    let contract_size = a.shape()[a_ax];
    let free_a_shape: Vec<usize> = free_a.iter().map(|&i| a.shape()[i]).collect();
    let free_b_shape: Vec<usize> = free_b.iter().map(|&i| b.shape()[i]).collect();
    let free_a_size: usize = free_a_shape.iter().product();
    let free_b_size: usize = free_b_shape.iter().product();

    let mut order_a: Vec<isize> = free_a.iter().map(|&i| i as isize).collect();
    order_a.push(a_ax as isize);
    let mut order_b: Vec<isize> = vec![b_ax as isize];
    order_b.extend(free_b.iter().map(|&i| i as isize));

    let a_t = a.transpose_axes(&order_a).map_err(linalg_ionp_err)?;
    let b_t = b.transpose_axes(&order_b).map_err(linalg_ionp_err)?;
    let a_2d = a_t
        .to_contiguous()
        .reshape(&[free_a_size, contract_size])
        .map_err(linalg_ionp_err)?;
    let b_2d = b_t
        .to_contiguous()
        .reshape(&[contract_size, free_b_size])
        .map_err(linalg_ionp_err)?;
    let out2d = ionp_ion::matmul::matmul(&a_2d, &b_2d).map_err(PyValueError::new_err)?;
    let mut out_shape = free_a_shape;
    out_shape.extend(free_b_shape);
    out2d.reshape(&out_shape).map_err(linalg_ionp_err)
}

fn multi_dot_three(a: &NdArray, b: &NdArray, c: &NdArray) -> PyResult<NdArray> {
    check_dot_aligned(a, b)?;
    check_dot_aligned(b, c)?;
    let (p0, p1) = (a.shape()[0] as u128, a.shape()[1] as u128);
    let p2 = b.shape()[1] as u128;
    let p3 = c.shape()[1] as u128;
    let cost1 = p0 * p1 * p2 + p0 * p2 * p3;
    let cost2 = p1 * p2 * p3 + p0 * p1 * p3;
    if cost1 <= cost2 {
        let ab = ionp_ion::matmul::matmul(a, b).map_err(PyValueError::new_err)?;
        ionp_ion::matmul::matmul(&ab, c).map_err(PyValueError::new_err)
    } else {
        let bc = ionp_ion::matmul::matmul(b, c).map_err(PyValueError::new_err)?;
        ionp_ion::matmul::matmul(a, &bc).map_err(PyValueError::new_err)
    }
}

fn multi_dot_chain_order(arrays: &[NdArray]) -> PyResult<Vec<Vec<usize>>> {
    // Same `_assert_2d` guard as `check_dot_aligned` -- this DP runs over
    // every chain participant eagerly (unlike `check_dot_aligned`, which is
    // only reached lazily per-pair during evaluation), so a non-2-D middle
    // array (or the reshaped-but-still-non-2-D first/last one) must be
    // rejected here too, before `shape()[0]`/`shape()[1]` are indexed.
    for arr in arrays {
        if arr.ndim() != 2 {
            return Err(linalg_err_raw(&format!(
                "{}-dimensional array given. Array must be two-dimensional",
                arr.ndim()
            )));
        }
    }
    let n = arrays.len();
    let mut p = vec![0usize; n + 1];
    p[0] = arrays[0].shape()[0];
    for (i, arr) in arrays.iter().enumerate() {
        p[i + 1] = arr.shape()[1];
    }
    let mut m = vec![vec![0u128; n]; n];
    let mut s = vec![vec![0usize; n]; n];
    for l in 2..=n {
        for i in 0..(n - l + 1) {
            let j = i + l - 1;
            m[i][j] = u128::MAX;
            for k in i..j {
                let q = m[i][k] + m[k + 1][j] + (p[i] as u128) * (p[k + 1] as u128) * (p[j + 1] as u128);
                if q < m[i][j] {
                    m[i][j] = q;
                    s[i][j] = k;
                }
            }
        }
    }
    Ok(s)
}

fn multi_dot_eval(arrays: &[NdArray], s: &[Vec<usize>], i: usize, j: usize) -> PyResult<NdArray> {
    if i == j {
        Ok(arrays[i].clone())
    } else {
        let left = multi_dot_eval(arrays, s, i, s[i][j])?;
        let right = multi_dot_eval(arrays, s, s[i][j] + 1, j)?;
        check_dot_aligned(&left, &right)?;
        ionp_ion::matmul::matmul(&left, &right).map_err(PyValueError::new_err)
    }
}

#[pyfunction]
#[pyo3(signature = (arrays, *, out=None))]
fn multi_dot(py: Python<'_>, arrays: &Bound<'_, PyAny>, out: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let items: Vec<Bound<'_, PyAny>> = arrays.try_iter()?.collect::<PyResult<_>>()?;
    if items.len() < 2 {
        return Err(PyValueError::new_err("Expecting at least two arrays."));
    }
    let mut mats: Vec<NdArray> = items.iter().map(as_ndarray_any).collect::<PyResult<_>>()?;

    // numpy's multi_dot treats a 1-D first/last operand as a row/column
    // vector for the DP's cost model, then squeezes it back out of the
    // final result (`inspect.getsource(numpy.linalg.multi_dot)`).
    let first_1d = mats[0].ndim() == 1;
    if first_1d {
        let n = mats[0].shape()[0];
        mats[0] = mats[0].reshape(&[1, n]).map_err(linalg_ionp_err)?;
    }
    let li = mats.len() - 1;
    let last_1d = mats[li].ndim() == 1;
    if last_1d {
        let n = mats[li].shape()[0];
        mats[li] = mats[li].reshape(&[n, 1]).map_err(linalg_ionp_err)?;
    }

    let result = if mats.len() == 2 {
        multi_dot_two(&mats[0], &mats[1])?
    } else if mats.len() == 3 {
        multi_dot_three(&mats[0], &mats[1], &mats[2])?
    } else {
        let s = multi_dot_chain_order(&mats)?;
        multi_dot_eval(&mats, &s, 0, mats.len() - 1)?
    };

    let result = if first_1d && last_1d {
        result.reshape(&[]).map_err(linalg_ionp_err)?
    } else if first_1d {
        let n = result.shape()[1];
        result.reshape(&[n]).map_err(linalg_ionp_err)?
    } else if last_1d {
        let n = result.shape()[0];
        result.reshape(&[n]).map_err(linalg_ionp_err)?
    } else {
        result
    };

    // numpy's own `multi_dot` always finishes by delegating to `np.dot(...,
    // out=out)` for its final multiplication (`np.linalg._linalg.multi_dot`
    // source), whose C-level `out=` validation is: (1) dtype must match
    // EXACTLY (no casting -- verified against real numpy 2.5.1: even a
    // safely-castable float64-result-into-float32-`out=` raises), checked
    // BEFORE shape, raising `ValueError: "output array is not acceptable
    // (must have the right datatype, number of dimensions, and be a
    // C-Array)"`; (2) only once dtype matches, shape must match exactly,
    // else `ValueError: "output array has wrong dimensions"`. This is
    // deliberately NOT the same check `write_into_out` normally performs
    // for other `out=` call sites in this crate (which allows broadcasting
    // `computed` up into a higher-rank/differently-shaped `out`) -- that
    // broadcast-tolerant behavior is correct for elementwise ufuncs but
    // wrong here, so both checks are done explicitly first and
    // `write_into_out` is only reached once they already hold (making its
    // own internal shape/dtype handling a no-op broadcast-to-self).
    if let Some(out_obj) = out {
        let out_arr = as_ndarray_any(out_obj)?;
        if out_arr.dtype() != result.dtype() {
            return Err(PyValueError::new_err(
                "output array is not acceptable (must have the right datatype, number of dimensions, and be a C-Array)",
            ));
        }
        if out_arr.shape() != result.shape() {
            return Err(PyValueError::new_err("output array has wrong dimensions"));
        }
        return write_into_out(out_obj, &result, None);
    }

    Ok(Py::new(py, PyArray { inner: result })?.into_any())
}

// ─────────────────────────────── tensorinv ────────────────────────────────

#[pyfunction]
#[pyo3(signature = (a, ind=2))]
fn tensorinv(py: Python<'_>, a: &Bound<'_, PyAny>, ind: i64) -> PyResult<Py<PyAny>> {
    if ind <= 0 {
        return Err(PyValueError::new_err("Invalid ind argument."));
    }
    let ind = ind as usize;
    let arr = as_ndarray_any(a)?;
    let old_shape = arr.shape().to_vec();
    // numpy's own `tensorinv` (`numpy/linalg/_linalg.py`) does NOT bounds-
    // check `ind` against `a.ndim` at all -- it slices the shape TUPLE with
    // plain Python semantics (`oldshape[ind:]`/`oldshape[:ind]`), which
    // CLAMP silently for an out-of-range `ind` (`(0,)[2:]` is `()`, not an
    // error) rather than raising. The previous code raised a bespoke
    // "Invalid ind argument." `ValueError` here instead, which numpy does
    // not: verified that `tensorinv(zeros((0,)))` (ind=2 default, 1
    // dimension) instead reaches numpy's `reshape(prod, -1)` -> `inv(...)`
    // path with `prod = 1` (empty-tuple product) against a total size of 0,
    // producing a non-square (1, 0) reshape that `inv` rejects with
    // `LinAlgError: "Last 2 dimensions of the array must be square"` -- a
    // completely different exception class AND text than the old bespoke
    // message. Clamping `ind` to `old_shape.len()` here (matching Python's
    // slice-clamping, not raising) reproduces that exact real path instead
    // of shortcutting around it; this also avoids the out-of-bounds Rust
    // slice-index panic the previous unclamped `old_shape[ind..]` risked
    // for any `ind > old_shape.len()`.
    let ind_clamped = ind.min(old_shape.len());
    let prod: usize = old_shape[ind_clamped..].iter().product();
    let mut inv_shape: Vec<usize> = old_shape[ind_clamped..].to_vec();
    inv_shape.extend_from_slice(&old_shape[..ind_clamped]);

    let total: usize = old_shape.iter().product();
    // numpy's own `tensorinv` reshapes via `a.reshape(prod, -1)` -- the
    // second dimension is numpy's own `-1`-infers-the-rest reshape
    // mechanism, not a precomputed `total / prod`. When `prod == 0`, that
    // inference is genuinely AMBIGUOUS (numpy: "cannot reshape array of
    // size 0 into shape (0,newaxis)") rather than silently resolving to 0
    // -- verified against real numpy 2.5.1 for `tensorinv(zeros((2,0,0)))`
    // (`ind=2` default): `oldshape[2:] = (0,)` gives `prod = 0`, and
    // `zeros((2,0,0)).reshape(0, -1)` itself raises exactly that
    // `ValueError`, distinct from (and reached BEFORE) the square-check
    // `LinAlgError` below. `prod == 0` here always implies `total == 0`
    // too (the same zero dimension that makes the suffix product 0 is
    // also part of the full-shape product), so this branch is reached
    // consistently, not just for this one shape.
    if prod == 0 {
        return Err(PyValueError::new_err(format!(
            "cannot reshape array of size {} into shape ({},newaxis)",
            total, prod
        )));
    }
    let other = total / prod;
    // numpy checks squareness BEFORE any dtype-support check (verified:
    // `np.linalg.tensorinv(np.zeros((2,3), dtype=np.float16), ind=1)` raises
    // `LinAlgError: Last 2 dimensions of the array must be square`, not the
    // float16-unsupported TypeError). `as_promoted_real2` below calls
    // `reject_float16` up front, so this check must happen first or a
    // non-square float16 input would wrongly surface the dtype error instead.
    if prod != other {
        return Err(linalg_err_raw("Last 2 dimensions of the array must be square"));
    }
    let a2d = arr.to_contiguous().reshape(&[prod, other]).map_err(linalg_ionp_err)?;
    let a2d_bound = wrap_ndarray(py, a2d)?;

    if let Some((data, n, m, prec)) = as_promoted_real2(&a2d_bound)? {
        debug_assert_eq!(n, m, "square check above already guarantees this");
        let out = rc::inv(&data, n).map_err(linalg_err)?;
        let out_arr = NdArray::from_buffer(Buffer::F64(out), vec![n, n], Order::C).map_err(linalg_ionp_err)?;
        let out_arr = if prec == Prec::Single { out_arr.cast_to(DType::F32) } else { out_arr };
        let final_arr = out_arr.reshape(&inv_shape).map_err(linalg_ionp_err)?;
        return Ok(Py::new(py, PyArray { inner: final_arr })?.into_any());
    }
    if let Some((data, n, m, prec)) = as_promoted_complex2(&a2d_bound)? {
        if n != m {
            return Err(linalg_err_raw("Last 2 dimensions of the array must be square"));
        }
        let out = zc::inv(&data, n).map_err(linalg_err)?;
        let out_arr = NdArray::from_buffer(Buffer::C128(out), vec![n, n], Order::C).map_err(linalg_ionp_err)?;
        let out_arr = if prec == Prec::Single { out_arr.cast_to(DType::C64) } else { out_arr };
        let final_arr = out_arr.reshape(&inv_shape).map_err(linalg_ionp_err)?;
        return Ok(Py::new(py, PyArray { inner: final_arr })?.into_any());
    }
    Err(unsupported_dtype("tensorinv"))
}

// ─────────────────────────────── tensorsolve ───────────────────────────────

#[pyfunction]
#[pyo3(signature = (a, b, axes=None))]
fn tensorsolve(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, axes: Option<Vec<isize>>) -> PyResult<Py<PyAny>> {
    let mut arr_a = as_ndarray_any(a)?;
    let arr_b = as_ndarray_any(b)?;
    let an = arr_a.ndim();
    if let Some(ax) = axes {
        let mut norm_axes = Vec::with_capacity(ax.len());
        for k in ax {
            norm_axes.push(normalize_axis(k, an)?);
        }
        let mut all_axes: Vec<usize> = (0..an).filter(|x| !norm_axes.contains(x)).collect();
        all_axes.extend(norm_axes);
        let order: Vec<isize> = all_axes.iter().map(|&x| x as isize).collect();
        arr_a = arr_a.transpose_axes(&order).map_err(linalg_ionp_err)?;
    }
    let bn = arr_b.ndim();
    if an < bn {
        return Err(linalg_err_raw(
            "Input arrays must satisfy the requirement             prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])",
        ));
    }
    // numpy computes `oldshape = a.shape[-(an - b.ndim):]` -- plain Python
    // slicing, where a NEGATIVE-ZERO start index (`an == bn`, so
    // `-(an-bn)` is `-0`, which Python normalizes to plain `0`, NOT "from
    // the end") means the slice is the FULL shape tuple, not an empty one.
    // The direct Rust transliteration `shape()[ndim - (an-bn)..]` gets this
    // wrong specifically when `an == bn`: `ndim - 0 == ndim`, giving an
    // EMPTY slice instead of the full one -- verified against real numpy
    // 2.5.1 (`tensorsolve(zeros((0,0)), zeros((0,0)))` succeeds with
    // result shape `(0,0)`, i.e. `oldshape == (0,0)`, the full shape, not
    // `()`). This diff-based branch reproduces Python's actual slice
    // semantics instead of the direct-but-wrong arithmetic transliteration.
    let diff = an - bn;
    let start = if diff == 0 { 0 } else { arr_a.ndim() - diff };
    let old_shape: Vec<usize> = arr_a.shape()[start..].to_vec();
    let prod: usize = old_shape.iter().product();
    let total: usize = arr_a.shape().iter().product();
    // numpy's own check is exactly `a.size != prod ** 2` (`numpy/linalg/
    // _linalg.py::tensorsolve`) -- nothing else. The previous `prod == 0`
    // guard rejected EVERY empty-shape input outright (verified against
    // real numpy 2.5.1: `tensorsolve(zeros(sh), zeros(sh))` for all of
    // `(0,0) (0,3) (3,0) (0,) (2,0,0)` actually SUCCEEDS, because `prod`
    // and `a.size` are both 0 there, so `0 != 0**2` is false and numpy
    // proceeds to `a.reshape(prod, prod)` -- a valid, trivially-empty
    // `(0, 0)` reshape -- followed by `solve` on a 0x0 system, which is
    // vacuously solvable). Matching that means checking `total !=
    // prod*prod` alone, and reshaping directly to `(prod, prod)` rather
    // than `(total / prod, prod)` -- the old `total / prod` form would
    // itself panic with a Rust integer division-by-zero whenever `prod ==
    // 0` (even though, by the check above, `total` is then guaranteed to
    // also be 0 and `(prod, prod)` is already the correct target shape
    // without needing the division at all).
    if total != prod.saturating_mul(prod) {
        return Err(linalg_err_raw(
            "Input arrays must satisfy the requirement             prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])",
        ));
    }
    let a2d = arr_a.to_contiguous().reshape(&[prod, prod]).map_err(linalg_ionp_err)?;
    let b_total: usize = arr_b.shape().iter().product();
    let b1d = arr_b.to_contiguous().reshape(&[b_total]).map_err(linalg_ionp_err)?;
    let a2d_bound = wrap_ndarray(py, a2d)?;
    let b1d_bound = wrap_ndarray(py, b1d)?;

    // numpy's `tensorsolve` reshapes `a` to `(prod, prod)` and `b` to
    // `(prod,)`, then calls its `solve1` gufunc (signature `(m,m),(m)->(m)`)
    // -- which independently validates that `b`'s own flattened size
    // matches `a`'s core dimension `prod`, and raises if not. The
    // `total != prod**2` check above only validates `a`'s OWN shape
    // self-consistency; it says nothing about `b`, so a `b` with the wrong
    // total size previously reached `rc::solve`/`zc::solve` with a length
    // mismatch (`bdata.len() != n`), reading past the end of `bdata` and
    // returning a silently-wrong result instead of raising. Verified
    // directly against real numpy 2.5.1: `tensorsolve(rng((3,4,3,4)),
    // rng((3,5)))` (solve-dim size 12, b.size 15) raises `ValueError:
    // solve1: Input operand 1 has a mismatch in its core dimension 0, with
    // gufunc signature (m,m),(m)->(m) (size {b_total} is different from
    // {prod})`, byte-for-byte reproduced here.
    //
    // 2026-08-06: this check MUST run after dtype resolution
    // (`as_promoted_real2`/`_complex2` below), not before it -- numpy's own
    // `solve1` gufunc resolves/validates the operand DTYPE first (raising
    // `TypeError: array type <dtype> is unsupported in linalg` for e.g.
    // float16) and only checks the core-dimension size second. Originally
    // placed here, before the dtype branches, this check fired first for
    // an unsupported-dtype + mismatched-b-size input (e.g. float16 A/b of
    // mismatched size), raising `ValueError` where real numpy raises
    // `TypeError` -- caught by this task's own differential corpus sweep
    // (`b_size_mismatch_raises` swept across dtypes including float16) and
    // fixed by moving the check into each dtype branch below, after that
    // branch's own `as_promoted_real2`/`_complex2` call has already
    // confirmed the dtype is supported.
    macro_rules! check_b_size {
        () => {
            if b_total != prod {
                // Plain `ValueError`, NOT `linalg_err_raw`/`LinAlgError` --
                // verified against real numpy 2.5.1 (`type(e) is
                // ValueError`, not a `LinAlgError` subclass instance) since
                // this comes from the internal `solve1` gufunc's own
                // core-dimension check, not from `numpy.linalg`'s own
                // `LinAlgError`-raising validation layer.
                return Err(PyValueError::new_err(format!(
                    "solve1: Input operand 1 has a mismatch in its core dimension 0, \
                     with gufunc signature (m,m),(m)->(m) (size {} is different from {})",
                    b_total, prod
                )));
            }
        };
    }

    if let Some((adata, n, m, aprec)) = as_promoted_real2(&a2d_bound)? {
        if n != m {
            return Err(linalg_err_raw(
                "Input arrays must satisfy the requirement             prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])",
            ));
        }
        check_b_size!();
        let (bdata, bprec) = as_promoted_real1(&b1d_bound)?.ok_or_else(|| unsupported_dtype("tensorsolve"))?;
        let out = rc::solve(&adata, n, &bdata, 1).map_err(linalg_err)?;
        let out_arr = NdArray::from_buffer(Buffer::F64(out), vec![n], Order::C).map_err(linalg_ionp_err)?;
        let out_arr = if combine_prec(aprec, bprec) == Prec::Single { out_arr.cast_to(DType::F32) } else { out_arr };
        let final_arr = out_arr.reshape(&old_shape).map_err(linalg_ionp_err)?;
        return Ok(Py::new(py, PyArray { inner: final_arr })?.into_any());
    }
    if let Some((adata, n, m, aprec)) = as_promoted_complex2(&a2d_bound)? {
        if n != m {
            return Err(linalg_err_raw(
                "Input arrays must satisfy the requirement             prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])",
            ));
        }
        check_b_size!();
        let (bdata, bprec) = as_promoted_complex1(&b1d_bound)?.ok_or_else(|| unsupported_dtype("tensorsolve"))?;
        let out = zc::solve(&adata, n, &bdata, 1).map_err(linalg_err)?;
        let out_arr = NdArray::from_buffer(Buffer::C128(out), vec![n], Order::C).map_err(linalg_ionp_err)?;
        let out_arr = if combine_prec(aprec, bprec) == Prec::Single { out_arr.cast_to(DType::C64) } else { out_arr };
        let final_arr = out_arr.reshape(&old_shape).map_err(linalg_ionp_err)?;
        return Ok(Py::new(py, PyArray { inner: final_arr })?.into_any());
    }
    Err(unsupported_dtype("tensorsolve"))
}

// ─────────────────── vector_norm / matrix_norm / norm ─────────────────────
//
// All three reduce to the same primitive: a magnitude array (`abs(x)` for
// real, `|z|` for complex -- valid for every ord below, including the
// SVD-based matrix ords, since a singular value is itself a magnitude), a
// row-major shape, and a generic "reduce over a set of axes with a fold /
// init / finish" combinator. `matrix_norm`'s two-step (sum over one axis,
// then max/min over the other, with numpy's own axis-index-shift-after-
// first-reduction bookkeeping) and the SVD-based ords (2/-2/'nuc', batched
// over any leading axes via a transpose-to-the-end + per-matrix economy-SVD
// loop, reusing the existing 2-D `rc::svd`/`zc::svd` this file's `svd`/
// `svdvals` bindings already call) are all built from the one combinator.

fn all_indices(shape: &[usize]) -> Vec<Vec<usize>> {
    let mut out = vec![vec![]];
    for &dim in shape {
        let mut next = Vec::with_capacity(out.len() * dim.max(1));
        for idx in &out {
            for d in 0..dim {
                let mut v = idx.clone();
                v.push(d);
                next.push(v);
            }
        }
        out = next;
    }
    out
}

fn flat_index(idx: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut pos = 0usize;
    for i in (0..shape.len()).rev() {
        pos += idx[i] * stride;
        stride *= shape[i];
    }
    pos
}

fn reduce_generic(
    mag: &[f64],
    shape: &[usize],
    axes: &[usize],
    init: f64,
    fold: impl Fn(f64, f64) -> f64,
    finish: impl Fn(f64) -> f64,
) -> (Vec<f64>, Vec<usize>) {
    let ndim = shape.len();
    let mut reduced = vec![false; ndim];
    for &a in axes {
        reduced[a] = true;
    }
    let kept_axes: Vec<usize> = (0..ndim).filter(|&i| !reduced[i]).collect();
    let reduced_axes: Vec<usize> = (0..ndim).filter(|&i| reduced[i]).collect();
    let kept_shape: Vec<usize> = kept_axes.iter().map(|&i| shape[i]).collect();
    let reduced_shape: Vec<usize> = reduced_axes.iter().map(|&i| shape[i]).collect();

    let kept_idxs = all_indices(&kept_shape);
    let reduced_idxs = all_indices(&reduced_shape);

    let mut out = Vec::with_capacity(kept_idxs.len());
    for kidx in &kept_idxs {
        let mut acc = init;
        for ridx in &reduced_idxs {
            let mut full = vec![0usize; ndim];
            for (pos, &ax) in kept_axes.iter().enumerate() {
                full[ax] = kidx[pos];
            }
            for (pos, &ax) in reduced_axes.iter().enumerate() {
                full[ax] = ridx[pos];
            }
            let flat = flat_index(&full, shape);
            acc = fold(acc, mag[flat]);
        }
        out.push(finish(acc));
    }
    (out, kept_shape)
}

enum NormOrdKind {
    P(f64),
    Inf,
    NegInf,
    Zero,
}

// Validation-order fix (2026-08-02): numpy's real `vector_norm` special-cases
// ANY string ord (recognized or not -- there is no valid string ord for a
// vector, unlike a matrix) with `ValueError: Invalid norm order '{ord}' for
// vectors` BEFORE ever attempting a numeric interpretation. The previous
// version of this function called `o.extract::<f64>()` unconditionally,
// which for a `str` fails with PyO3's own generic `TypeError: must be real
// number, not str` -- using the value (attempting numeric coercion) before
// validating it (checking whether it's a string first), the wrong-order bug
// this task's brief named. Checking `extract::<String>()` first (which also
// accepts a `str` subclass and numpy's `str_` scalar, both real `str`
// instances) and raising numpy's exact message fixes that one shape.
//
// NOTE this does NOT make `vector_norm`/`norm`'s vector path fully numpy-
// exact for every ord type: verified directly against live numpy 2.5.1 that
// non-string, non-numeric ord values (bytes, bytearray, list, arbitrary
// object) do NOT uniformly raise this same ValueError the way a matrix ord
// does -- numpy's vector norm literally computes `abs(x) ** ord` via its
// generic ufunc machinery for anything that isn't a recognized string, so
// the result depends on the value's own type in ways that range from a
// successful (if bizarre) broadcast (`bytearray` succeeds, treated as a
// small int array) to `TypeError`/`UFuncTypeError` with messages that have
// nothing to do with this function's own numeric-ord domain. Replicating
// that would mean reimplementing numpy's generic array power-ufunc
// dispatch, not fixing a validation-order bug -- out of scope here, and
// left as a genuine (documented, not silently excluded) second blocker on
// `vector_norm`/`norm`'s vector branch; see `anionpy/_state/linalg.py`.
fn parse_ord_general(obj: Option<&Bound<'_, PyAny>>) -> PyResult<NormOrdKind> {
    match obj {
        None => Ok(NormOrdKind::P(2.0)),
        Some(o) => {
            if let Ok(s) = o.extract::<String>() {
                return Err(PyValueError::new_err(format!("Invalid norm order '{}' for vectors", s)));
            }
            let v: f64 = o.extract()?;
            if v.is_infinite() {
                Ok(if v > 0.0 { NormOrdKind::Inf } else { NormOrdKind::NegInf })
            } else if v == 0.0 {
                Ok(NormOrdKind::Zero)
            } else {
                Ok(NormOrdKind::P(v))
            }
        }
    }
}

fn apply_p_reduce(mag: &[f64], shape: &[usize], axes: &[usize], kind: &NormOrdKind) -> (Vec<f64>, Vec<usize>) {
    match kind {
        NormOrdKind::Inf => reduce_generic(mag, shape, axes, f64::NEG_INFINITY, f64::max, |a| a),
        NormOrdKind::NegInf => reduce_generic(mag, shape, axes, f64::INFINITY, f64::min, |a| a),
        NormOrdKind::Zero => reduce_generic(mag, shape, axes, 0.0, |acc, v| acc + if v != 0.0 { 1.0 } else { 0.0 }, |a| a),
        NormOrdKind::P(p) => {
            let p = *p;
            reduce_generic(mag, shape, axes, 0.0, move |acc, v| acc + v.powf(p), move |acc| acc.powf(1.0 / p))
        }
    }
}

fn ndarray_and_magnitudes(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>, Prec)> {
    let arr = try_ndarray_any(obj).ok_or_else(|| PyValueError::new_err("expected an array"))?;
    let dt = arr.dtype();
    if is_complex_dtype(dt) {
        let prec = classify_prec(dt);
        let contig = arr.to_contiguous();
        let buf = contig.buffer().cast_to(DType::C128);
        let data = match buf {
            Buffer::C128(v) => v,
            _ => unreachable!(),
        };
        let mag: Vec<f64> = data.iter().map(|c| c.norm()).collect();
        Ok((mag, arr.shape().to_vec(), prec))
    } else {
        // NOTE: numpy allows float16 through every non-SVD-based norm ord
        // (this shared magnitude-extraction path); it only rejects float16
        // for the SVD-based ords (2, -2, 'nuc'), which route through
        // `svd_singular_values_batched` instead and reject it there. Do
        // NOT add a `reject_float16` call here -- verified against real
        // numpy: `vector_norm`/`matrix_norm(ord='fro'/1/-1/inf/-inf)` all
        // succeed on float16 input.
        let prec = classify_prec(dt);
        let contig = arr.to_contiguous();
        let buf = contig.buffer().cast_to(DType::F64);
        let data = match buf {
            Buffer::F64(v) => v,
            _ => unreachable!(),
        };
        let mag: Vec<f64> = data.iter().map(|v| v.abs()).collect();
        Ok((mag, arr.shape().to_vec(), prec))
    }
}

fn parse_axis_list(axis: Option<&Bound<'_, PyAny>>, ndim: usize) -> PyResult<Vec<usize>> {
    match axis {
        None => Ok((0..ndim).collect()),
        Some(a) => {
            if let Ok(i) = a.extract::<isize>() {
                Ok(vec![normalize_axis(i, ndim)?])
            } else {
                let raw: Vec<isize> = a.extract()?;
                raw.into_iter().map(|v| normalize_axis(v, ndim)).collect()
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (x, /, *, axis=None, keepdims=None, ord=None))]
fn vector_norm(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    ord: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // `keepdims=` here is plain Python truthiness in real numpy (measured
    // 2026-08-02) -- unlike the C-format-parsed `keepdims=` on
    // `sum`/`mean`/etc in reductions.rs, `linalg.vector_norm`/
    // `matrix_norm`/`norm` dispatch on `if keepdims:` directly.
    let keepdims = match keepdims {
        None => false,
        Some(v) => v.is_truthy()?,
    };
    let (mag, shape, prec) = ndarray_and_magnitudes(x)?;
    let ndim = shape.len();
    let axes = parse_axis_list(axis, ndim)?;
    let kind = parse_ord_general(ord)?;
    let (data, kept_shape) = apply_p_reduce(&mag, &shape, &axes, &kind);

    let out_shape = if keepdims {
        let mut s = shape.clone();
        for &a in &axes {
            s[a] = 1;
        }
        s
    } else {
        kept_shape
    };
    let arr = NdArray::from_buffer(Buffer::F64(data), out_shape, Order::C).map_err(linalg_ionp_err)?;
    let arr = cast_real_by_prec(arr, prec);
    wrap_norm_result(py, arr)
}

/// Same scalar-return-type fix as `mk_scalar_*` above (see
/// `crate::numpy_scalar_from_0d`'s doc comment / `docs/scalar-return-type-defect.md`),
/// applied at `vector_norm`/`matrix_norm`/`norm`'s own result-wrapping sites:
/// a full reduction to 0 dimensions (`axis=None` with no `keepdims`, or every
/// axis explicitly reduced) must come back as a genuine numpy scalar, not a
/// 0-d `anionpy.ndarray` -- verified directly (`np.linalg.norm(np.array([3.,4.]))`
/// is `numpy.float64`, not `numpy.ndarray`). None of these three functions
/// accept `out=`, so unlike `wrap_reduction`/`do_reduce_axis` there is no
/// `out=` branch to special-case here.
fn wrap_norm_result(py: Python<'_>, arr: NdArray) -> PyResult<Py<PyAny>> {
    if arr.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &arr);
    }
    Ok(Py::new(py, PyArray { inner: arr })?.into_any())
}

/// Cast a freshly-built (always-F64-buffer) real result down to match the
/// input's precision class -- `vector_norm`/`matrix_norm`/`norm`-specific,
/// since these (uniquely among this file's linalg functions) preserve
/// float16 output for float16 input rather than promoting to double.
fn cast_real_by_prec(arr: NdArray, prec: Prec) -> NdArray {
    match prec {
        Prec::Half => arr.cast_to(DType::F16),
        Prec::Single => arr.cast_to(DType::F32),
        Prec::Double => arr,
    }
}

enum MatOrd {
    Fro,
    Nuc,
    P1,
    PNeg1,
    P2,
    PNeg2,
    Inf,
    NegInf,
}

// Validation-order fix (2026-08-02), same root cause and same task as
// `parse_ord_general` above: numpy's real `matrix_norm` raises the exact
// same `ValueError: Invalid norm order for matrices.` (note, verified
// directly: NO value interpolated into this message, unlike the vector
// case -- confirmed for an invalid string, bytes, bytearray, an
// out-of-domain float, a list, and an arbitrary object, all producing this
// identical literal string) for EVERY ord that is neither a recognized
// string (`"fro"`/`"f"`/`"nuc"`) nor a recognized numeric value (`1`, `-1`,
// `2`, `-2`, `inf`, `-inf`) -- regardless of the rejected value's TYPE.
// Unlike the vector case, this really is just a validation-order bug: the
// old code called `o.extract::<f64>()` unconditionally in the non-string
// branch, which fails with PyO3's own generic `TypeError` for any
// non-numeric non-string type (bytes/bytearray/list/object) instead of
// numpy's uniform domain `ValueError` -- and it also wrongly interpolated
// the rejected string's value into the message on the string branch, which
// live numpy never does for matrices. Both are fixed here: the string
// branch's error no longer carries the value, and the numeric branch now
// treats an `extract::<f64>()` failure (wrong type entirely) exactly like
// an out-of-domain numeric value -- both are "not a recognized matrix
// ord", both raise the one uniform message. Verified directly against live
// numpy 2.5.1 across this exact type grid (str/invalid-str/str-subclass/
// np.str_/bytes/bytearray/float/list/None/object/complex/NaN/bool); see
// `anionpy/_state/linalg.py` and `tests/differential/strparam_cases.py` for
// the corpus this was checked against.
fn parse_matrix_ord(obj: Option<&Bound<'_, PyAny>>) -> PyResult<MatOrd> {
    match obj {
        None => Ok(MatOrd::Fro),
        Some(o) => {
            if let Ok(s) = o.extract::<String>() {
                match s.as_str() {
                    "fro" | "f" => Ok(MatOrd::Fro),
                    "nuc" => Ok(MatOrd::Nuc),
                    _ => Err(PyValueError::new_err("Invalid norm order for matrices.")),
                }
            } else {
                match o.extract::<f64>() {
                    Ok(v) => {
                        if v.is_infinite() {
                            Ok(if v > 0.0 { MatOrd::Inf } else { MatOrd::NegInf })
                        } else if v == 1.0 {
                            Ok(MatOrd::P1)
                        } else if v == -1.0 {
                            Ok(MatOrd::PNeg1)
                        } else if v == 2.0 {
                            Ok(MatOrd::P2)
                        } else if v == -2.0 {
                            Ok(MatOrd::PNeg2)
                        } else {
                            Err(PyValueError::new_err("Invalid norm order for matrices."))
                        }
                    }
                    Err(_) => Err(PyValueError::new_err("Invalid norm order for matrices.")),
                }
            }
        }
    }
}

/// Batched singular values: move `row_axis`/`col_axis` to the trailing two
/// positions (via a view + materialize), then loop the existing 2-D
/// `rc::svd`/`zc::svd` economy core over every leading "batch" matrix --
/// only the singular VALUES are needed here (norms 2/-2/'nuc'), so the
/// economy (job 'S') core already bound by `svdvals` above is sufficient,
/// with no full_matrices=True gap to worry about.
fn svd_singular_values_batched(arr: &NdArray, row_axis: usize, col_axis: usize) -> PyResult<(Vec<f64>, Vec<usize>, Prec)> {
    let ndim = arr.ndim();
    let mut order: Vec<usize> = (0..ndim).filter(|&i| i != row_axis && i != col_axis).collect();
    order.push(row_axis);
    order.push(col_axis);
    let order_isize: Vec<isize> = order.iter().map(|&x| x as isize).collect();
    let moved = arr.transpose_axes(&order_isize).map_err(linalg_ionp_err)?.to_contiguous();
    let shape = moved.shape().to_vec();
    let m = shape[ndim - 2];
    let n = shape[ndim - 1];
    let batch: usize = shape[..ndim - 2].iter().product();
    let k = m.min(n);
    let dt = moved.dtype();

    if is_complex_dtype(dt) {
        let prec = classify_prec(dt);
        let buf = moved.buffer().cast_to(DType::C128);
        let data = match buf {
            Buffer::C128(v) => v,
            _ => unreachable!(),
        };
        let mut out = Vec::with_capacity(batch * k);
        for b in 0..batch {
            let mat = &data[b * m * n..(b + 1) * m * n];
            let (_u, s, _vt) = zc::svd(mat, m, n).map_err(linalg_err)?;
            out.extend(s);
        }
        let mut out_shape = shape[..ndim - 2].to_vec();
        out_shape.push(k);
        Ok((out, out_shape, prec))
    } else {
        reject_float16(dt)?;
        let prec = classify_prec(dt);
        let buf = moved.buffer().cast_to(DType::F64);
        let data = match buf {
            Buffer::F64(v) => v,
            _ => unreachable!(),
        };
        let mut out = Vec::with_capacity(batch * k);
        for b in 0..batch {
            let mat = &data[b * m * n..(b + 1) * m * n];
            let (_u, s, _vt) = rc::svd(mat, m, n).map_err(linalg_err)?;
            out.extend(s);
        }
        let mut out_shape = shape[..ndim - 2].to_vec();
        out_shape.push(k);
        Ok((out, out_shape, prec))
    }
}

#[allow(clippy::too_many_arguments)]
fn matrix_norm_core(
    py: Python<'_>,
    orig_obj: &Bound<'_, PyAny>,
    arr: &NdArray,
    row_axis: usize,
    col_axis: usize,
    ord: MatOrd,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    let (data, out_shape, out_prec): (Vec<f64>, Vec<usize>, Prec) = match ord {
        MatOrd::Fro => {
            let (mag, shape, prec) = ndarray_and_magnitudes(orig_obj)?;
            let (d, s) = apply_p_reduce(&mag, &shape, &[row_axis, col_axis], &NormOrdKind::P(2.0));
            (d, s, prec)
        }
        MatOrd::P1 | MatOrd::PNeg1 => {
            let (mag, shape, prec) = ndarray_and_magnitudes(orig_obj)?;
            let (d1, s1) = reduce_generic(&mag, &shape, &[row_axis], 0.0, |a, v| a + v, |a| a);
            let new_col = if col_axis > row_axis { col_axis - 1 } else { col_axis };
            let kind = if matches!(ord, MatOrd::P1) { NormOrdKind::Inf } else { NormOrdKind::NegInf };
            let (d, s) = apply_p_reduce(&d1, &s1, &[new_col], &kind);
            (d, s, prec)
        }
        MatOrd::Inf | MatOrd::NegInf => {
            let (mag, shape, prec) = ndarray_and_magnitudes(orig_obj)?;
            let (d1, s1) = reduce_generic(&mag, &shape, &[col_axis], 0.0, |a, v| a + v, |a| a);
            let new_row = if row_axis > col_axis { row_axis - 1 } else { row_axis };
            let kind = if matches!(ord, MatOrd::Inf) { NormOrdKind::Inf } else { NormOrdKind::NegInf };
            let (d, s) = apply_p_reduce(&d1, &s1, &[new_row], &kind);
            (d, s, prec)
        }
        MatOrd::P2 | MatOrd::PNeg2 | MatOrd::Nuc => {
            let (svals, sshape, prec) = svd_singular_values_batched(arr, row_axis, col_axis)?;
            let last = sshape.len() - 1;
            let kind = match ord {
                MatOrd::P2 => NormOrdKind::Inf,
                MatOrd::PNeg2 => NormOrdKind::NegInf,
                MatOrd::Nuc => NormOrdKind::P(1.0),
                _ => unreachable!(),
            };
            let (d, s) = apply_p_reduce(&svals, &sshape, &[last], &kind);
            (d, s, prec)
        }
    };

    let final_shape = if keepdims {
        let mut s = arr.shape().to_vec();
        s[row_axis] = 1;
        s[col_axis] = 1;
        s
    } else {
        out_shape
    };
    let out_arr = NdArray::from_buffer(Buffer::F64(data), final_shape, Order::C).map_err(linalg_ionp_err)?;
    let out_arr = cast_real_by_prec(out_arr, out_prec);
    wrap_norm_result(py, out_arr)
}

#[pyfunction]
#[pyo3(signature = (x, /, *, keepdims=None, ord=None))]
fn matrix_norm(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    keepdims: Option<&Bound<'_, PyAny>>,
    ord: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Plain Python truthiness, same as `vector_norm` above (measured
    // 2026-08-02).
    let keepdims = match keepdims {
        None => false,
        Some(v) => v.is_truthy()?,
    };
    let arr = as_ndarray_any(x)?;
    if arr.ndim() < 2 {
        // numpy's `matrix_norm` doesn't check dimensionality directly --
        // it calls `normalize_axis_tuple((-2, -1), ndim)` first, which
        // fails on an ndim<2 array with a real `numpy.exceptions.AxisError`
        // (axis -2 is out of bounds), NOT a generic `ValueError`. Verified
        // against real numpy 2.5.1: `np.linalg.matrix_norm(np.zeros(3))`
        // raises `AxisError: axis -2 is out of bounds for array of
        // dimension 1`.
        return Err(crate::axis_error(
            -2,
            Some(arr.ndim()),
            &format!("axis -2 is out of bounds for array of dimension {}", arr.ndim()),
        ));
    }
    let ndim = arr.ndim();
    let ord_parsed = parse_matrix_ord(ord)?;
    matrix_norm_core(py, x, &arr, ndim - 2, ndim - 1, ord_parsed, keepdims)
}

#[pyfunction]
#[pyo3(signature = (x, ord=None, axis=None, keepdims=None))]
fn norm(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    ord: Option<&Bound<'_, PyAny>>,
    axis: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Plain Python truthiness, same as `vector_norm`/`matrix_norm` above
    // (measured 2026-08-02).
    let keepdims = match keepdims {
        None => false,
        Some(v) => v.is_truthy()?,
    };
    let arr = as_ndarray_any(x)?;
    let ndim = arr.ndim();

    // axis=None fast paths (numpy's own shortcuts): (ord is None) OR
    // (ord in {'fro','f'} and ndim==2) OR (ord==2 and ndim==1) all collapse
    // to "flatten and take the 2-norm of the flattened vector".
    if axis.is_none() {
        let ord_is_none = ord.is_none();
        let ord_is_fro = ord
            .and_then(|o| o.extract::<String>().ok())
            .map(|s| (s == "fro" || s == "f") && ndim == 2)
            .unwrap_or(false);
        let ord_is_2 = ord
            .and_then(|o| o.extract::<f64>().ok())
            .map(|v| v == 2.0 && ndim == 1)
            .unwrap_or(false);
        if ord_is_none || ord_is_fro || ord_is_2 {
            let (mag, _shape, prec) = ndarray_and_magnitudes(x)?;
            // NOTE: deliberately NOT `mag.iter().map(|v| v*v).sum()` -- Rust's
            // `Iterator::sum::<f64>()` folds from a `-0.0` identity (verified:
            // `[].iter().sum::<f64>()` is `-0.0`, not `+0.0`), so an all-zero or
            // empty `mag` would silently produce `-0.0` here, disagreeing with
            // numpy's `+0.0`. Fold from an explicit `+0.0` instead.
            let sq: f64 = mag.iter().fold(0.0f64, |acc, v| acc + v * v);
            let val = sq.sqrt();
            let out_shape = if keepdims { vec![1usize; ndim] } else { vec![] };
            let out_arr = NdArray::from_buffer(Buffer::F64(vec![val]), out_shape, Order::C).map_err(linalg_ionp_err)?;
            let out_arr = cast_real_by_prec(out_arr, prec);
            return wrap_norm_result(py, out_arr);
        }
    }

    let axes = parse_axis_list(axis, ndim)?;

    if axes.len() == 1 {
        let kind = parse_ord_general(ord)?;
        let (mag, shape, prec) = ndarray_and_magnitudes(x)?;
        let (data, kept_shape) = apply_p_reduce(&mag, &shape, &axes, &kind);
        let out_shape = if keepdims {
            let mut s = shape.clone();
            s[axes[0]] = 1;
            s
        } else {
            kept_shape
        };
        let out_arr = NdArray::from_buffer(Buffer::F64(data), out_shape, Order::C).map_err(linalg_ionp_err)?;
        let out_arr = cast_real_by_prec(out_arr, prec);
        wrap_norm_result(py, out_arr)
    } else if axes.len() == 2 {
        if axes[0] == axes[1] {
            return Err(PyValueError::new_err("Duplicate axes given."));
        }
        let ord_parsed = parse_matrix_ord(ord)?;
        matrix_norm_core(py, x, &arr, axes[0], axes[1], ord_parsed, keepdims)
    } else {
        Err(PyValueError::new_err("Improper number of dimensions to norm."))
    }
}

// ═══════════════ Array-API-restricted thin wrappers ═════════════════════
//
// `numpy.linalg.matmul`/`vecdot`/`matrix_transpose`/`diagonal`/`outer` are
// each, per numpy's own source (`numpy/linalg/_linalg.py`, verified via
// `inspect.getsource` against numpy 2.5.1), a THIN positional-only,
// no-`dtype=`/`out=` wrapper over the exact same core numpy already uses
// for the top-level `numpy.matmul`/`numpy.vecdot`/`numpy.matrix_transpose`/
// `numpy.diagonal`/`numpy.outer` (`linalg.diagonal` additionally pins
// `axis1=-2, axis2=-1`, and `linalg.outer`/`linalg.matrix_transpose`
// additionally validate ndim before delegating -- both differences are
// implemented explicitly below, not inherited "for free"). anionpy already
// has fully general, already-tested N-D/batched implementations of all
// five underlying algorithms (`ionp_ion::matmul::{matmul,vecdot}`,
// `ionp_core::array::NdArray::transpose_axes`, `ionp_core::manip::diagonal`,
// `ionp_core::ufunc::outer_binary`), so these bindings call straight into
// them rather than re-deriving anything -- unlike the rest of this file,
// they are NOT restricted to 2-D-only inputs, since the reused core
// genuinely already handles the N-D/batched case correctly.

#[pyfunction]
#[pyo3(signature = (x1, x2, /))]
fn matmul(py: Python<'_>, x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let a = crate::matmul::coerce_matmul_operand(x1)?;
    let b = crate::matmul::coerce_matmul_operand(x2)?;
    let out = ionp_ion::matmul::matmul(&a, &b).map_err(PyValueError::new_err)?;
    // Same 1-D/1-D scalar-return-type fix as top-level `anionpy.matmul`
    // (`matmul.rs::finish_gufunc_call`) -- see
    // `crate::numpy_scalar_from_0d`'s doc comment.
    if out.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &out);
    }
    Ok(Py::new(py, PyArray { inner: out })?.into_any())
}

#[pyfunction]
#[pyo3(signature = (x1, x2, /, *, axis=-1))]
fn vecdot(py: Python<'_>, x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    // `ionp_ion::matmul::vecdot` is fixed at the last axis (numpy's own
    // default and, per that module's own scope note, its only bound
    // axis). `axis=-1` is the default and the only value verified here;
    // anything else is a genuine scope gap, not silently wrong output.
    if axis != -1 {
        return Err(PyNotImplementedError::new_err(
            "anionpy.linalg.vecdot: only the default axis=-1 is implemented",
        ));
    }
    let a = crate::matmul::coerce_matmul_operand(x1)?;
    let b = crate::matmul::coerce_matmul_operand(x2)?;
    let out = ionp_ion::matmul::vecdot(&a, &b).map_err(PyValueError::new_err)?;
    // Same scalar-return-type fix as top-level `anionpy.vecdot`
    // (`matmul.rs::finish_gufunc_call`) -- `vecdot` always fully collapses
    // its dot-product axis, so this branch is unconditionally taken; see
    // `crate::numpy_scalar_from_0d`'s doc comment.
    if out.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &out);
    }
    Ok(Py::new(py, PyArray { inner: out })?.into_any())
}

#[pyfunction]
#[pyo3(signature = (x, /))]
fn matrix_transpose(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = as_ndarray_any(x)?;
    let ndim = arr.ndim();
    if ndim < 2 {
        return Err(PyValueError::new_err(format!(
            "Input array must be at least 2-dimensional, but it is {}",
            ndim
        )));
    }
    let mut axes: Vec<isize> = (0..ndim as isize).collect();
    axes.swap(ndim - 1, ndim - 2);
    let out = arr.transpose_axes(&axes).map_err(linalg_ionp_err)?;
    // A transpose is a strided VIEW: `.base` must point at the input and
    // `OWNDATA` must be False, exactly as for `ndarray.mT`, which this is
    // the array-API free-function spelling of.
    Ok(crate::wrap_shape_view(py, x, &arr, out)?.into_any())
}

#[pyfunction]
#[pyo3(signature = (x, /, *, offset=0))]
fn diagonal(py: Python<'_>, x: &Bound<'_, PyAny>, offset: isize) -> PyResult<Py<PyAny>> {
    let arr = as_ndarray_any(x)?;
    let out = manip::diagonal(&arr, offset, -2, -1).map_err(linalg_ionp_err)?;
    Ok(Py::new(py, PyArray { inner: out })?.into_any())
}

#[pyfunction]
#[pyo3(signature = (x1, x2, /))]
fn outer(py: Python<'_>, x1: &Bound<'_, PyAny>, x2: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let a = as_ndarray_any(x1)?;
    let b = as_ndarray_any(x2)?;
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "Input arrays must be one-dimensional, but they are x1.ndim={} and x2.ndim={}.",
            a.ndim(),
            b.ndim()
        )));
    }
    let out = ionp_core::ufunc::outer_binary(ionp_core::ufunc::BinaryOp::Multiply, &a, &b)
        .map_err(linalg_ionp_err)?;
    Ok(Py::new(py, PyArray { inner: out })?.into_any())
}

/// Top-level `numpy.outer(a, b, out=None)` -- NOT the same contract as
/// `linalg.outer` above (Array API, strictly 1-D only): numpy's top-level
/// `outer` ravels ANY-ndim input first (`a.ravel()`, `b.ravel()`), verified
/// directly against live numpy 2.5.1 (`np.outer(arange(6).reshape(2,3),
/// arange(4).reshape(2,2)).shape == (6, 4)`, while `np.linalg.outer` on the
/// same 2-D inputs raises `ValueError`). Reusing `linalg::outer`'s function
/// unchanged for the top-level name would therefore be a real behavioral
/// divergence (a spurious ValueError on any non-1-D input), not a harmless
/// rename -- so this is a separate function: ravel via `NdArray::ravel_order`
/// (the same value-correct flatten `ravel`/`flatten`'s Rust core uses; not
/// the Python-exposed `ravel()` binding, which has its own documented
/// view-vs-copy gap unrelated to the VALUES this needs) then delegate to the
/// same `outer_binary(Multiply)` core `linalg::outer` uses.
///
/// `out=` (2026-08-03 fix): the previous `"out= is not implemented (matching
/// linalg.outer's own declared scope)"` justification does not hold --
/// `numpy.linalg.outer` (Array API) genuinely has no `out=` parameter, but
/// top-level `numpy.outer` does (`inspect.signature(np.outer) ==
/// (a, b, out=None)`), and `linalg.outer`'s scope is therefore irrelevant to
/// this binding. Reading numpy's own source
/// (`inspect.getsource(numpy.outer)`) shows top-level `outer` is not a
/// standalone `out=` implementation at all -- it is a thin wrapper that
/// ravels both operands and delegates straight to the `multiply` ufunc:
/// `return multiply(a.ravel()[:, newaxis], b.ravel()[newaxis, :], out)`.
/// So `out=` here has EXACTLY `multiply`'s own ufunc `out=` contract, not a
/// bespoke one, verified live against numpy 2.5.1:
///   - correct shape/dtype `out=`: returns the SAME object (`is`), mutated
///     in place.
///   - `out=` positional (3rd positional arg) and keyword both work, and
///     `out=None` explicitly is legal shorthand for "allocate a fresh array"
///     (same as omitting it); passing a 4th positional argument is a
///     `TypeError` (numpy's real `outer()` is a plain 2-python-arg-plus-out
///     function, not a ufunc-`__call__`-style `*args`).
///   - wrong-shape `out=`: `ValueError`, `"operands could not be broadcast
///     together with shapes {a_col} {b_row} {out} "` where `a_col`/`b_row`
///     are the shapes of the *reshaped* operands actually fed to `multiply`
///     (`(M, 1)` and `(1, N)`), NOT the original 1-D input shapes -- e.g.
///     `np.outer([1,2,3], [4,5], out=empty((2,2)))` raises with shapes
///     `(3,1) (1,2) (2,2)`, matching `check_full_broadcast`'s existing
///     all-operands-plus-out convention used by every other ufunc's `out=`
///     path in this crate (same function, reused unchanged).
///   - same-kind-castable dtype mismatch (e.g. float64 result into a
///     float32 `out=`): SUCCEEDS, casting down, same object identity
///     returned.
///   - non-same-kind mismatch (e.g. float64 result into an int64 `out=`):
///     raises numpy's real `_UFuncOutputCastingError`,
///     `"Cannot cast ufunc 'multiply' output from dtype('float64') to
///     dtype('int64') with casting rule 'same_kind'"` -- note the message
///     names `'multiply'`, not `'outer'`, because that IS the real ufunc
///     doing the cast; reusing `write_into_out_ufunc` with
///     `ufunc_name = "multiply"` reproduces this exactly rather than
///     inventing a divergent `'outer'`-named message.
///   - non-`anionpy.ndarray` `out=` (e.g. a list): `TypeError`,
///     `"return arrays must be of ArrayType"`.
///   - F-order / non-contiguous (strided) `out=`: writes through the
///     existing strides, same object identity returned -- no special
///     casing needed, `write_out`'s `NdIter`-based write already handles
///     arbitrary strides.
///   - N-D (ravel) input path and the empty-input case: both just feed a
///     smaller/larger/zero-sized `computed`, no different from the 1-D
///     case; `out=` support does not depend on input rank.
/// This reuses the SAME `write_into_out_ufunc`/`check_full_broadcast`
/// helpers every other ufunc's `out=` already goes through (both already
/// crate-visible from this module's ancestor, `lib.rs`'s crate root, and
/// already imported here) -- not a new pattern, and specifically not the
/// plain `write_into_out` this file uses elsewhere for the genuinely
/// bespoke (non-ufunc-delegating) linalg routines like `multi_dot`, which
/// do NOT get numpy's casting/broadcast-message behavior because they are
/// not secretly a ufunc call underneath. `outer` is, so it gets that
/// treatment.
#[pyfunction]
#[pyo3(name = "outer")]
#[pyo3(signature = (a, b, /, out=None))]
pub fn outer_toplevel(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let a = as_ndarray_any(a)?;
    let b = as_ndarray_any(b)?;
    let a_flat = a.ravel_order("C").map_err(linalg_ionp_err)?;
    let b_flat = b.ravel_order("C").map_err(linalg_ionp_err)?;
    let m = a_flat.shape()[0];
    let n = b_flat.shape()[0];
    let computed = ionp_core::ufunc::outer_binary(ionp_core::ufunc::BinaryOp::Multiply, &a_flat, &b_flat)
        .map_err(linalg_ionp_err)?;
    if let Some(out_obj) = out {
        // Full-operand broadcast pre-check, same convention as
        // `Ufunc::__call__`'s own `out=` handling in lib.rs (see
        // `check_full_broadcast`'s doc): real numpy's error lists the
        // RESHAPED operands actually fed to `multiply` (`(M,1)`, `(1,N)`),
        // not the original 1-D `a`/`b` shapes, plus the `out=` buffer's
        // shape. Skipped when `out_obj` isn't even an `anionpy.ndarray` --
        // `write_into_out_ufunc` below raises the correct
        // "return arrays must be of ArrayType" for that case unchanged.
        if let Ok(out_bound) = out_obj.cast::<PyArray>() {
            let out_shape = out_bound.borrow().inner.shape().to_vec();
            let a_col_shape = [m, 1usize];
            let b_row_shape = [1usize, n];
            let shapes: Vec<&[usize]> = vec![&a_col_shape, &b_row_shape, &out_shape];
            check_full_broadcast(&shapes)?;
        }
        return write_into_out_ufunc(out_obj, &computed, None, "multiply", "same_kind", None);
    }
    Ok(Py::new(py, PyArray { inner: computed })?.into_any())
}

// ═══════════════════════════ module registration ═══════════════════════

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "linalg")?;
    m.add_function(wrap_pyfunction!(det, &m)?)?;
    m.add_function(wrap_pyfunction!(slogdet, &m)?)?;
    m.add_function(wrap_pyfunction!(inv, &m)?)?;
    m.add_function(wrap_pyfunction!(solve, &m)?)?;
    m.add_function(wrap_pyfunction!(cholesky, &m)?)?;
    m.add_function(wrap_pyfunction!(matrix_power, &m)?)?;
    m.add_function(wrap_pyfunction!(eig, &m)?)?;
    m.add_function(wrap_pyfunction!(eigvals, &m)?)?;
    m.add_function(wrap_pyfunction!(eigh, &m)?)?;
    m.add_function(wrap_pyfunction!(eigvalsh, &m)?)?;
    m.add_function(wrap_pyfunction!(qr, &m)?)?;
    m.add_function(wrap_pyfunction!(svd, &m)?)?;
    m.add_function(wrap_pyfunction!(svdvals, &m)?)?;
    m.add_function(wrap_pyfunction!(pinv, &m)?)?;
    m.add_function(wrap_pyfunction!(matrix_rank, &m)?)?;
    m.add_function(wrap_pyfunction!(cond, &m)?)?;
    m.add_function(wrap_pyfunction!(lstsq, &m)?)?;
    m.add_function(wrap_pyfunction!(trace, &m)?)?;
    m.add_function(wrap_pyfunction!(matmul, &m)?)?;
    m.add_function(wrap_pyfunction!(vecdot, &m)?)?;
    m.add_function(wrap_pyfunction!(matrix_transpose, &m)?)?;
    m.add_function(wrap_pyfunction!(diagonal, &m)?)?;
    m.add_function(wrap_pyfunction!(outer, &m)?)?;
    m.add_function(wrap_pyfunction!(cross, &m)?)?;
    m.add_function(wrap_pyfunction!(tensordot, &m)?)?;
    m.add_function(wrap_pyfunction!(multi_dot, &m)?)?;
    m.add_function(wrap_pyfunction!(tensorinv, &m)?)?;
    m.add_function(wrap_pyfunction!(tensorsolve, &m)?)?;
    m.add_function(wrap_pyfunction!(vector_norm, &m)?)?;
    m.add_function(wrap_pyfunction!(matrix_norm, &m)?)?;
    m.add_function(wrap_pyfunction!(norm, &m)?)?;
    // anionpy's OWN `LinAlgError` (see `errors.rs`): a genuine subclass of
    // `numpy.linalg.LinAlgError` when numpy is importable (so `except
    // numpy.linalg.LinAlgError:` still catches instances anionpy raises,
    // identity-sensitive callers keep working), or a plain `ValueError`
    // subclass when numpy is absent. Every error already raised here (via
    // `linalg_err`/`linalg_err_raw`) already constructs this exact class --
    // so exposing it as an attribute is just publishing what the module
    // already uses, not new logic. This is what "expose the class as an
    // attribute" now means without a bare `?` on a numpy import: this used
    // to be the ONE runtime numpy borrow in the whole crate with no
    // numpy-absent fallback, which alone made `import anionpy` require numpy.
    m.add("LinAlgError", errors::linalg_error_class(py)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
