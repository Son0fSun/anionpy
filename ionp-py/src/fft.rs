//! PyO3 bindings for the `fft` namespace (`anionpy.fft.*`), mirroring
//! `numpy.fft` as a genuine dotted submodule (same
//! `PyModule::new(py, "fft")` + `add_submodule` pattern `linalg.rs` uses
//! for `anionpy.linalg`, not the flat-re-export-from-a-`.py`-file pattern
//! `strings.py`/`char.py` use -- `fft.fft`/`fft.ifft`/etc. are already
//! naturally dotted in numpy's own namespace).
//!
//! This file only parses Python arguments and marshals them into
//! `ionp_core::fft`'s pure-Rust implementations -- every actual numeric
//! loop lives there (see that module's own doc comment for the full
//! semantics/precision-policy writeup), per this crate's "Rust does
//! arithmetic, Python does dispatch" rule.

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::PyInt;

use ionp_core::fft as core_fft;
use ionp_core::{Buffer, DType, NdArray};

use crate::{to_py_err, PyArray};

// ---------------------------------------------------------------------------
// out= / device= support
//
// numpy's `out=` for every `numpy.fft.*` transform is NOT a generic
// dispatch-time convenience -- these 14 functions are thin Python wrappers
// (`numpy/fft/_pocketfft.py`) around exactly 5 real `numpy.ufunc` objects
// living in the compiled submodule `numpy.fft._pocketfft_umath` (aliased
// `pfu` inside `_pocketfft.py`): `fft`, `ifft`, `irfft`, `rfft_n_even`,
// `rfft_n_odd`. Confirmed directly against live numpy 2.5.1:
//   fft/fft2/fftn                       -> ufunc 'fft'
//   ifft/ifft2/ifftn                    -> ufunc 'ifft'
//   irfft/irfft2/irfftn/hfft            -> ufunc 'irfft' (always)
//   rfft/rfft2/rfftn/ihfft              -> ufunc 'rfft_n_even' or
//                                           'rfft_n_odd', selected by the
//                                           PARITY of the effective length
//                                           of the last transformed axis
//                                           (the explicit `n`/`s[-1]` if
//                                           given, else that axis's input
//                                           size) -- NOT by the output
//                                           array's size (which is
//                                           ambiguous: n=4 and n=5 both
//                                           produce an n//2+1==3 output).
// This identity leaks into the real exception numpy raises for a
// dtype-incompatible `out=`: `numpy._core._exceptions._UFuncOutputCastingError`,
// whose repr embeds the actual ufunc object (`<ufunc 'rfft_n_odd'>`, not
// e.g. `<ufunc 'rfft'>`), so reproducing it byte-for-byte requires picking
// the SAME one numpy would have used, not just any fft-shaped ufunc name.
// Verified empirically (not guessed) via `.venv/bin/python3` probes against
// 1-D and N-D cases, several explicit `s=`/`axes=` combinations, and one
// 3-D case, confirming: (a) the mapping above, (b) the parity rule depends
// on the LAST entry of the resolved `axes` (defaulting to the array's own
// last axis when `axes=None`, matching pocketfft's real-transform-always-
// last placement -- true for `rfft2`'s `axes=(-2,-1)` default AND
// `rfftn`'s `axes=range(ndim)` default, since both end at axis -1 when
// unspecified), and (c) the exception constructor's 5th positional
// argument (`_UFuncOutputCastingError(ufunc, "same_kind", from, to, i)`)
// is the constant `2` in every fft out=-casting case tested -- NOT the `0`
// the pre-existing `ufunc_output_casting_err` helper in `lib.rs` uses for
// its own (different) in-place-binary-op scenario. (As of 2026-08-01 that
// helper is also a native Rust `TypeError` construction, not a numpy
// import -- see its doc comment in `lib.rs` -- but it still isn't reusable
// here verbatim: it hard-codes index `0` and has no notion of
// `numpy.fft._pocketfft_umath`'s 5 distinct ufunc names / the rfft parity
// rule below.) This module writes its own analogous helper below rather
// than editing `lib.rs` (out of scope for this task).

/// Which real `numpy.fft._pocketfft_umath` ufunc backs a given transform's
/// `out=` casting-error identity. `rfft`/`rfft2`/`rfftn`/`ihfft` need the
/// effective last-axis length's parity; every other transform's ufunc name
/// is fixed regardless of shape.
enum FftUfunc {
    Fixed(&'static str),
    RfftParity,
}

/// numpy's real exception here, `numpy._core._exceptions._UFuncOutputCastingError`,
/// is PRIVATE (verified: `hasattr(np.exceptions, 'UFuncTypeError')` is
/// `False` -- there is no public alias; the class lives only under
/// `numpy._core._exceptions`) and its MRO includes `TypeError`. Per this
/// project's settled policy for numpy-PRIVATE exception classes ("Answer
/// B" -- public classes like `AxisError`/`LinAlgError`/`ValueError` still
/// get built from the real class; only private ones get this treatment):
/// construct a native Rust `TypeError` with byte-identical message text
/// and declare a narrow, documented type-vs-message equivalence, rather
/// than importing numpy at raise-time to build the real private instance.
/// `except TypeError` on the caller's side catches both, exactly as it
/// would catch `_UFuncOutputCastingError` (a `TypeError` subclass).
///
/// This was ORIGINALLY implemented by importing `numpy`,
/// `numpy.fft._pocketfft_umath`, and `numpy._core._exceptions` at
/// raise-time to construct a genuine instance, with this native
/// construction only as a fallback. That was wrong on two independent
/// grounds, both confirmed by the coordinator's 2026-08-01 review and
/// not merely asserted: (1) it made anionpy's own error path depend on
/// numpy being importable at all -- a "replacement" that requires the
/// thing it replaces to raise its own errors is not a replacement; (2)
/// it made the differential `out=` dtype-casting cases CIRCULAR --
/// numpy's side raises the real numpy instance, and anionpy's side would
/// ALSO raise that exact same real numpy instance (same class, same
/// `.args`), so the comparison is structurally incapable of ever
/// failing regardless of whether anionpy's own casting logic is correct.
/// This is the same failure shape ("a tautological receiver that made
/// dunder tests unfalsifiable") that cost this project 32 items
/// elsewhere; a test that cannot fail reports green while proving
/// nothing. The message text below was verified byte-for-byte against
/// live numpy 2.5.1 for all 5 real ufunc names (`fft`, `ifft`, `irfft`,
/// `rfft_n_even`, `rfft_n_odd`) via a direct
/// `_UFuncOutputCastingError(ufunc, 'same_kind', from_dtype, to_dtype, 2)`
/// probe -- e.g. for `rfft_n_odd`:
///   "Cannot cast ufunc 'rfft_n_odd' output from dtype('complex128') to
///    dtype('int64') with casting rule 'same_kind'"
/// `DType::name()` (this crate's `dtype.rs`) already returns numpy's own
/// `dtype.name` short strings (`"float64"`, `"complex128"`, ...), so no
/// separate name-mapping table is needed.
fn fft_output_casting_err(ufunc_name: &str, from_: DType, to: DType) -> PyErr {
    // BUG FOUND AND FIXED 2026-08-02: this built a plain `PyTypeError` --
    // byte-identical text to real numpy's `_UFuncOutputCastingError` but the
    // wrong CLASS. `UFuncTypeError` (a `TypeError` subclass, `_anionpy`-local,
    // defined in `lib.rs`) is what real numpy's private class DISPLAYS as
    // (`type(e).__name__ == "UFuncTypeError"`); a plain `PyTypeError` here
    // meant `np.fft.fft(a, out=int64_arr)` (and `ifft`) diverged from numpy
    // on exact class even though the text always matched -- confirmed live:
    // real numpy raises `UFuncTypeError` for this exact case, anionpy raised
    // `TypeError`. This was the divergence `tests/differential/fft_cases.py`'s
    // `_OUT_CASTING_EXC_EQUIV` was papering over; now fixed at the source,
    // that equivalence is deleted rather than kept as a standing licence.
    crate::UFuncTypeError::new_err(format!(
        "Cannot cast ufunc '{ufunc_name}' output from dtype('{}') to dtype('{}') with casting rule 'same_kind'",
        from_.name(),
        to.name()
    ))
}

/// Resolve which real ufunc name backs a 1-D transform's `out=` casting
/// error, given the effective (already-defaulted) transform length `n_eff`
/// for the `RfftParity` cases.
fn resolve_ufunc_name(which: &FftUfunc, n_eff: i64) -> &'static str {
    match which {
        FftUfunc::Fixed(name) => name,
        FftUfunc::RfftParity => {
            if n_eff.rem_euclid(2) == 0 {
                "rfft_n_even"
            } else {
                "rfft_n_odd"
            }
        }
    }
}

/// Shared `out=` finishing logic for every fft transform. `inner` is the
/// freshly computed (natural-shape, natural-dtype) result; `out`, if
/// given, must be exactly-shaped (fft's `out=` -- unlike a generic ufunc's
/// -- does NOT support broadcasting a smaller natural result up into a
/// larger `out`; verified directly: `np.fft.fft(a, out=bigger)` raises
/// `ValueError: output array has wrong shape.` exactly like a
/// wrong-*smaller*-shape `out` does) and dtype-compatible under numpy's
/// `same_kind` casting rule. On success, returns the identical Python
/// object identity `out` itself is (`result is out` -- verified this is
/// numpy's own real contract: `np.fft.fft(a, out=o); _ is o == True`), not
/// a fresh wrapper -- which is why this returns `Py<PyAny>` rather than
/// `PyArray` (a `-> PyArray` pyfunction return always constructs a NEW
/// Python object via PyO3's `IntoPyObject` conversion, which cannot
/// preserve the caller's own object identity).
fn finish_fft_result(
    py: Python<'_>,
    inner: NdArray,
    out: Option<&Bound<'_, PyAny>>,
    ufunc: FftUfunc,
    n_eff: i64,
) -> PyResult<Py<PyAny>> {
    let out = match out {
        None => return Ok(Py::new(py, PyArray { inner })?.into_any()),
        Some(o) => o,
    };
    // numpy's real check (`return arrays must be of ArrayType`) fires for
    // ANY non-ndarray `out=` value (verified: a list, an int, ... all hit
    // this exact TypeError, before shape/dtype are even examined) --
    // reproduced here rather than reusing `lib.rs::write_into_out`'s
    // generic "out= must be an anionpy.ndarray" message, which is the wrong
    // text for this family.
    let out_ref = out
        .extract::<PyRef<'_, PyArray>>()
        .map_err(|_| PyTypeError::new_err("return arrays must be of ArrayType"))?;
    if out_ref.inner.shape() != inner.shape() {
        return Err(PyValueError::new_err("output array has wrong shape."));
    }
    let out_dtype = out_ref.inner.dtype();
    drop(out_ref);
    let casted = if out_dtype == inner.dtype() {
        inner
    } else {
        if !crate::same_kind_castable(inner.dtype(), out_dtype) {
            let ufunc_name = resolve_ufunc_name(&ufunc, n_eff);
            return Err(fft_output_casting_err(ufunc_name, inner.dtype(), out_dtype));
        }
        inner.cast_to(out_dtype)
    };
    {
        let mut out_mut = out.extract::<PyRefMut<'_, PyArray>>()?;
        ionp_core::ufunc::write_out(&mut out_mut.inner, &casted, None).map_err(to_py_err)?;
    }
    Ok(out.clone().unbind())
}

// ---------------------------------------------------------------------------
// Local argument-parsing helpers (own copy, per this codebase's established
// per-module-copy convention -- see `manip.rs`'s near-identical helpers,
// which this file cannot import since they are not `pub` and `manip.rs` is
// not owned by this task).
// ---------------------------------------------------------------------------

fn extract_or_ingest_ndarray(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(pyref.inner.clone());
    }
    crate::array_impl(obj, None)
}

fn isize_list_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        let ints = pyref.inner.cast_to(DType::I64);
        let vals = match ints.buffer() {
            Buffer::I64(v) => v.clone(),
            _ => unreachable!("cast_to(DType::I64) always yields Buffer::I64"),
        };
        return Ok(vals.into_iter().map(|v| v as isize).collect());
    }
    if let Ok(seq) = obj.extract::<Vec<isize>>() {
        Ok(seq)
    } else {
        Ok(vec![obj.extract::<isize>()?])
    }
}

fn i64_list_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        let ints = pyref.inner.cast_to(DType::I64);
        let vals = match ints.buffer() {
            Buffer::I64(v) => v.clone(),
            _ => unreachable!("cast_to(DType::I64) always yields Buffer::I64"),
        };
        return Ok(vals);
    }
    if let Ok(seq) = obj.extract::<Vec<i64>>() {
        Ok(seq)
    } else {
        Ok(vec![obj.extract::<i64>()?])
    }
}

fn axes_opt_from_pyobj(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<isize>>> {
    match obj {
        None => Ok(None),
        Some(a) if a.is_none() => Ok(None),
        Some(a) => Ok(Some(isize_list_from_pyobj(a)?)),
    }
}

fn s_opt_from_pyobj(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<i64>>> {
    match obj {
        None => Ok(None),
        Some(a) if a.is_none() => Ok(None),
        Some(a) => Ok(Some(i64_list_from_pyobj(a)?)),
    }
}

/// numpy's `fftfreq`/`rfftfreq` explicitly check `isinstance(n, (int,
/// numpy.integer))` and raise `ValueError: n should be an integer` on
/// anything else (notably: a Python `float` like `5.0`, even though it is
/// numerically integral, is rejected). Approximated here via: real Python
/// `int` (which also covers `bool`, since `bool` subclasses `int` in
/// CPython, exactly like numpy accepts it), or any object exposing a
/// numpy-style `.dtype.kind` of `'i'`/`'u'`.
fn n_as_integer(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    if obj.is_instance_of::<PyInt>() {
        return obj.extract::<i64>();
    }
    if let Ok(dtype_obj) = obj.getattr("dtype") {
        if let Ok(kind) = dtype_obj.getattr("kind") {
            if let Ok(k) = kind.extract::<String>() {
                if k == "i" || k == "u" {
                    return obj.extract::<i64>();
                }
            }
        }
    }
    Err(PyValueError::new_err("n should be an integer"))
}

/// numpy's 1-D transforms (`fft`/`ifft`/`rfft`/`irfft`/`hfft`/`ihfft`) select
/// the transform axis via plain Python tuple indexing on `a.shape` internally
/// (`numpy/fft/_pocketfft.py`'s `_raw_fft` does `n = a.shape[axis]`), NOT via
/// numpy's fancy-indexing-based `normalize_axis_index`/real `AxisError` path
/// that most other ufunc/reduction axis arguments use elsewhere in this
/// codebase. An out-of-range axis therefore raises a plain built-in
/// `IndexError: tuple index out of range` -- always this exact fixed message,
/// regardless of the offending axis value or the array's ndim. Verified
/// directly against live numpy 2.5.1 across axis values {5, -10, 3, 100} on a
/// (2,3,4)-shaped array for both `fft` and `rfft`: all four raise this exact
/// `(IndexError, "tuple index out of range")` pair. Reproduced here (rather
/// than reusing `IonpError::AxisError`'s real-`numpy.exceptions.AxisError`
/// conversion in `to_py_err`, which is correct for ndarray reductions but
/// wrong for `numpy.fft`) so the exception TYPE and message match exactly.
fn check_1d_axis(ndim: usize, axis: isize) -> PyResult<()> {
    let nd = ndim as isize;
    if axis < -nd || axis >= nd {
        return Err(PyIndexError::new_err("tuple index out of range"));
    }
    Ok(())
}

/// numpy's N-D transforms (`fftn`/`ifftn`/`fft2`/`ifft2`/`rfftn`/`rfft2`/
/// `irfftn`/`irfft2`) validate an explicit `axes=` argument via fancy-index
/// lookup into `numpy.array(a.shape)` (`shapeless = numpy.take(a.shape,
/// axes)`-style access in `numpy/fft/_pocketfft.py`), which raises
/// `IndexError: index {axis} is out of bounds for axis 0 with size {ndim}`
/// -- citing "axis 0" (of the internal 1-D shape array, not of `a` itself)
/// and `a.ndim` as the size, using the FIRST out-of-range value in the
/// user-supplied `axes` order (not sorted, not normalized). Verified
/// directly against live numpy 2.5.1 on a (2,3,4)-shaped array: axes
/// `(0,5)`, `(5,6)`, `(-10,0)`, `(0,1,5)`, `(0,5,1)`, `(5,0,1)` all raise
/// "index 5 is out of bounds for axis 0 with size 3" (or "index -10 ..."
/// for the negative case) -- always reporting the first invalid entry
/// verbatim (unnormalized) and `ndim`, never the position within `axes`.
fn check_nd_axes(ndim: usize, axes: &[isize]) -> PyResult<()> {
    let nd = ndim as isize;
    for &axis in axes {
        if axis < -nd || axis >= nd {
            return Err(PyIndexError::new_err(format!(
                "index {axis} is out of bounds for axis 0 with size {ndim}"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 1-D transforms
// ---------------------------------------------------------------------------

macro_rules! def_1d {
    ($name:ident, $core:path, $ufunc:expr) => {
        #[pyfunction]
        #[pyo3(signature = (a, n=None, axis=-1, norm=None, out=None))]
        fn $name(
            py: Python<'_>,
            a: &Bound<'_, PyAny>,
            n: Option<i64>,
            axis: isize,
            norm: Option<&str>,
            out: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Py<PyAny>> {
            let arr = extract_or_ingest_ndarray(a)?;
            let ndim = arr.shape().len();
            check_1d_axis(ndim, axis)?;
            let inner = $core(&arr, n, axis, norm).map_err(to_py_err)?;
            // Effective transform length for the `RfftParity` ufunc-name
            // selection (irrelevant, but harmless to compute, for the
            // `Fixed` variants) -- the explicit `n=` if given, else the
            // (already axis-range-checked) transform axis's own input
            // size. See the module-level doc comment above for why this
            // must be the INPUT length, not derived from the output shape.
            let axis_norm = if axis < 0 { axis + ndim as isize } else { axis } as usize;
            let n_eff = n.unwrap_or(arr.shape()[axis_norm] as i64);
            finish_fft_result(py, inner, out, $ufunc, n_eff)
        }
    };
}

def_1d!(fft, core_fft::fft, FftUfunc::Fixed("fft"));
def_1d!(ifft, core_fft::ifft, FftUfunc::Fixed("ifft"));
def_1d!(rfft, core_fft::rfft, FftUfunc::RfftParity);
def_1d!(irfft, core_fft::irfft, FftUfunc::Fixed("irfft"));
def_1d!(hfft, core_fft::hfft, FftUfunc::Fixed("irfft"));
def_1d!(ihfft, core_fft::ihfft, FftUfunc::RfftParity);

// ---------------------------------------------------------------------------
// N-D / 2-D transforms
// ---------------------------------------------------------------------------

macro_rules! def_nd {
    ($name:ident, $core:path, $ufunc:expr) => {
        #[pyfunction]
        #[pyo3(signature = (a, s=None, axes=None, norm=None, out=None))]
        fn $name(
            py: Python<'_>,
            a: &Bound<'_, PyAny>,
            s: Option<&Bound<'_, PyAny>>,
            axes: Option<&Bound<'_, PyAny>>,
            norm: Option<&str>,
            out: Option<&Bound<'_, PyAny>>,
        ) -> PyResult<Py<PyAny>> {
            let arr = extract_or_ingest_ndarray(a)?;
            let ndim = arr.shape().len();
            let s_vec = s_opt_from_pyobj(s)?;
            let axes_vec = axes_opt_from_pyobj(axes)?;
            if let Some(ref av) = axes_vec {
                check_nd_axes(ndim, av)?;
            }
            let inner = $core(&arr, s_vec.as_deref(), axes_vec.as_deref(), norm).map_err(to_py_err)?;
            // See `finish_fft_result`'s doc comment / this module's
            // top-of-file writeup: the real transform is always applied
            // along the LAST entry of the (possibly-defaulted) `axes`
            // list, which is the array's own last axis whenever `axes` is
            // omitted -- true for both `rfftn`'s "all axes" default and
            // `rfft2`'s explicit `(-2,-1)` default, since both end at
            // axis -1. `s`'s last entry overrides that axis's input size
            // when given, exactly like the 1-D case's explicit `n=`.
            let last_axis: isize = axes_vec
                .as_ref()
                .and_then(|a| a.last().copied())
                .unwrap_or(ndim as isize - 1);
            let last_axis_norm = if last_axis < 0 { last_axis + ndim as isize } else { last_axis } as usize;
            let n_eff = s_vec
                .as_ref()
                .and_then(|s| s.last().copied())
                .unwrap_or(arr.shape()[last_axis_norm] as i64);
            finish_fft_result(py, inner, out, $ufunc, n_eff)
        }
    };
}

def_nd!(fftn, core_fft::fftn, FftUfunc::Fixed("fft"));
def_nd!(ifftn, core_fft::ifftn, FftUfunc::Fixed("ifft"));
def_nd!(fft2, core_fft::fft2, FftUfunc::Fixed("fft"));
def_nd!(ifft2, core_fft::ifft2, FftUfunc::Fixed("ifft"));
def_nd!(rfftn, core_fft::rfftn, FftUfunc::RfftParity);
def_nd!(rfft2, core_fft::rfft2, FftUfunc::RfftParity);
def_nd!(irfftn, core_fft::irfftn, FftUfunc::Fixed("irfft"));
def_nd!(irfft2, core_fft::irfft2, FftUfunc::Fixed("irfft"));

// ---------------------------------------------------------------------------
// fftshift / ifftshift / fftfreq / rfftfreq
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (x, axes=None))]
fn fftshift(x: &Bound<'_, PyAny>, axes: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let arr = extract_or_ingest_ndarray(x)?;
    let axes_vec = axes_opt_from_pyobj(axes)?;
    let inner = core_fft::fftshift(&arr, axes_vec.as_deref()).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (x, axes=None))]
fn ifftshift(x: &Bound<'_, PyAny>, axes: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let arr = extract_or_ingest_ndarray(x)?;
    let axes_vec = axes_opt_from_pyobj(axes)?;
    let inner = core_fft::ifftshift(&arr, axes_vec.as_deref()).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

/// numpy's `fftfreq`/`rfftfreq` compute `val = 1.0 / (n * d)` as a plain
/// Python float division BEFORE building the (possibly zero-length,
/// e.g. `n=0`) output array -- Python's float division raises
/// `ZeroDivisionError: division by zero` on `1.0 / 0.0` (unlike Rust/IEEE
/// 754, which would silently produce `inf`). Verified directly against
/// live numpy 2.5.1: `fftfreq(0, 1.0)`, `fftfreq(5, 0.0)`, and
/// `fftfreq(0, 0.0)` all raise this exact exception, regardless of the
/// resulting array being empty either way for `n=0`. Reproduced here at
/// the binding boundary (not in `ionp_core::fft`, which has no Python
/// float-division-by-zero semantics to inherit) so both callers see the
/// same exception TYPE numpy raises, not merely a same-shaped message.
fn check_zero_division(n: i64, d: f64) -> PyResult<()> {
    if n as f64 * d == 0.0 {
        return Err(PyZeroDivisionError::new_err("division by zero"));
    }
    Ok(())
}

// `device=` reuses `crate::check_device` verbatim -- an ALREADY-EXISTING
// `lib.rs` helper (used today by the array-creation family: `zeros`/
// `ones`/`empty`/etc.) implementing exactly the contract numpy's own
// `numpy/_core/numeric.py` source gives `device=`: accept `None` (numpy's
// own default) or the literal string `"cpu"`, else raise
// `ValueError('Device not understood. Only "cpu" is allowed, but
// received: {device}')` with `{device}` interpolated via `str()`, not
// `repr()`. `numpy.fft.fftfreq`/`rfftfreq` carry this same Array-API kwarg
// (`inspect.signature` confirms `(n, d=1.0, device=None)` on live numpy
// 2.5.1) with identical semantics -- verified directly, not assumed: numpy
// accepts `device=None`/`device="cpu"` and raises this exact message for
// e.g. `device="gpu"`/`device=1`. This module cannot edit `lib.rs` (out of
// scope), but `check_device` is a private (non-`pub`) fn at the crate
// root, which Rust's privacy rules make visible to child modules of the
// same crate (confirmed precedent: `reductions.rs` already imports
// several non-`pub` `lib.rs` fns via `use crate::{...}`) -- so calling
// `crate::check_device` here is reuse, not reimplementation.
#[pyfunction]
#[pyo3(signature = (n, d=1.0, device=None))]
fn fftfreq(n: &Bound<'_, PyAny>, d: f64, device: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let n_val = n_as_integer(n)?;
    check_zero_division(n_val, d)?;
    crate::check_device(device)?;
    let inner = core_fft::fftfreq(n_val, d).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (n, d=1.0, device=None))]
fn rfftfreq(n: &Bound<'_, PyAny>, d: f64, device: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let n_val = n_as_integer(n)?;
    check_zero_division(n_val, d)?;
    crate::check_device(device)?;
    let inner = core_fft::rfftfreq(n_val, d).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

// ═══════════════════════════ module registration ═══════════════════════

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "fft")?;
    m.add_function(wrap_pyfunction!(fft, &m)?)?;
    m.add_function(wrap_pyfunction!(ifft, &m)?)?;
    m.add_function(wrap_pyfunction!(rfft, &m)?)?;
    m.add_function(wrap_pyfunction!(irfft, &m)?)?;
    m.add_function(wrap_pyfunction!(hfft, &m)?)?;
    m.add_function(wrap_pyfunction!(ihfft, &m)?)?;
    m.add_function(wrap_pyfunction!(fftn, &m)?)?;
    m.add_function(wrap_pyfunction!(ifftn, &m)?)?;
    m.add_function(wrap_pyfunction!(fft2, &m)?)?;
    m.add_function(wrap_pyfunction!(ifft2, &m)?)?;
    m.add_function(wrap_pyfunction!(rfftn, &m)?)?;
    m.add_function(wrap_pyfunction!(rfft2, &m)?)?;
    m.add_function(wrap_pyfunction!(irfftn, &m)?)?;
    m.add_function(wrap_pyfunction!(irfft2, &m)?)?;
    m.add_function(wrap_pyfunction!(fftshift, &m)?)?;
    m.add_function(wrap_pyfunction!(ifftshift, &m)?)?;
    m.add_function(wrap_pyfunction!(fftfreq, &m)?)?;
    m.add_function(wrap_pyfunction!(rfftfreq, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
