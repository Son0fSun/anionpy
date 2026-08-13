//! numpy's scalar-type hierarchy: 10 abstract, non-instantiable base
//! classes (`generic`, `number`, `integer`, `signedinteger`,
//! `unsignedinteger`, `inexact`, `floating`, `complexfloating`, `flexible`,
//! `character`) plus 14 concrete scalar classes (`bool_`, `int8`..`int64`,
//! `uint8`..`uint64`, `float16`..`float64`, `complex64`/`complex128`).
//!
//! Every constructor is built entirely from Rust-side conversion rules --
//! `construct_int_buffer`/`construct_float_buffer`/`construct_complex_buffer`/
//! `construct_bool_value` below -- reusing the exact `int_buffer_from_i128`/
//! `int_buffer_from_f64`/`float_buffer_from_f64`/`complex_buffer_from_f64_pair`/
//! `check_int_bounds` helpers `lib.rs` already uses for weak-scalar array
//! arithmetic, plus CPython's OWN `int()`/`float()`/`complex()` builtins for
//! the "parse this Python value" step (string parsing, complex-source
//! TypeError text, float truncation semantics) -- never numpy. No arithmetic
//! happens in Python; every builtin call here is a lossless read of the
//! *input*, not a computed *answer* for anionpy.
//!
//! DERIVED RULE (verified offline against real numpy 2.5.1, not looked up in
//! numpy source): a scalar constructor's source is either
//!   - "weak": a bare Python bool/int/float/complex/str with no `.dtype`.
//!     Integer targets: bounds-checked (`OverflowError` on failure, exactly
//!     `check_int_bounds`'s existing message). Nan/inf floats raise via
//!     Python's own `int()` (`ValueError`/`OverflowError`), matching numpy.
//!   - "strong": anything with a `.dtype` attribute (a numpy or anionpy scalar,
//!     or a 0-d array). Integer targets: an UNSAFE wrapping/saturating cast
//!     (`int_buffer_from_i128` for int/bool sources, `int_buffer_from_f64`'s
//!     ARM64-FCVTZS-derived rule for float sources), never bounds-checked
//!     (`np.int8(np.int32(300))` -> `44`, `np.int8(np.float64(inf))` -> `-1`,
//!     both verified).
//! Float/complex targets need no strong/weak distinction at all: narrowing
//! never raises, so a uniform `float(obj)`/`complex(obj)` (CPython builtin,
//! which also correctly raises `TypeError` for a complex source into a float
//! target -- verified: `np.float32(3+4j)` raises) covers every source kind.
//!
//! KNOWN, DISCLOSED GAP (not fixable within PyO3's single-inheritance
//! `pyclass` model, not just unattempted): real numpy's `float64` and
//! `complex128` additionally subclass Python's builtin `float`/`complex`
//! (`np.float64.__mro__` includes `<class 'float'>`; `np.int8.__mro__` does
//! NOT include `int` -- verified, this is asymmetric in real numpy, not a
//! measurement error). That is C-level multiple inheritance; a PyO3
//! `#[pyclass]` has exactly one `extends=` base, so `anionpy.float64`/
//! `anionpy.complex128` cannot also inherit from `float`/`complex`.
//! `isinstance(anionpy.float64(1.0), float)` is `False` where numpy's is `True`.
//! This module does not attempt to hide that: the numpy-internal MRO chain
//! (`generic`/`number`/`inexact`/`floating`/... and every `int8`..`bool_`
//! leaf) is verified exact and is what `tests/differential/scalar_cases.py`
//! tests; `isinstance(x, float)`/`isinstance(x, complex)` against the Python
//! builtins are deliberately NOT exercised or declared anywhere.
//!
//! ARRAY-LIKE INPUT (`anionpy.int8([1, 2])`, `anionpy.float32(some_ndarray)`):
//! real numpy's scalar-type constructors silently build an *array* for
//! array-like input instead of a scalar (`type(np.int8([1, 2]))` is
//! `numpy.ndarray`, shape `(2,)`). `#[new]` itself is structurally locked
//! to return `PyClassInitializer<Self>` and can never do this. Instead,
//! `install_new_overrides` (near the bottom of this file, called from
//! `register()`) monkeypatches each of the 14 concrete leaf classes'
//! `__new__` slot with a Rust closure (`make_new_override`) built via
//! `PyCFunction::new_closure`: array-like input (`is_array_like`) is routed
//! through `build_array_for_dtype`/`flatten_weak*` into a fresh
//! `anionpy.ndarray`; everything else falls through unchanged to the
//! `#[new]`-generated associated function below (`$rust::new`). This is
//! sound, sanctioned CPython behavior, not UB or an exploit: PyO3 pyclasses
//! are heap types, whose `__new__` slot is a plain mutable class attribute,
//! and CPython's `type_call()` never requires `tp_new`'s return value to be
//! an instance of `cls` -- it only skips calling `__init__` when it isn't.

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyCFunction, PyComplex, PyDict, PyFloat, PyInt, PyList, PyTuple,
};
use pyo3::PyClassInitializer;

use ionp_core::buffer::{C128, C64};
use ionp_core::dtype::{weak_target_dtype, ScalarKind};
use ionp_core::ufunc::{binary_op, math_binary_op, BinaryOp, MathBinaryOp};
use ionp_core::{Buffer, DType, NdArray, Order};

use crate::{
    axis_error, check_int_bounds, check_ragged, classify_scalar, complex_buffer_from_f64_pair,
    dtype_name_to_dtype, float_buffer_from_f64, int_buffer_from_f64,
    int_buffer_from_i128, ndarray_from_numpy, no_ufunc_loop_err_dtypes, retag_divmod_name,
    shape_of_nested, to_py_err, weak_scalar_buffer, PyArray, PyDType, UFuncTypeError,
};

/// TICKET #90a: real numpy's `np.longdouble(str)` parses via a dedicated C
/// `strtold`-based routine, NOT the same lenient path `np.float64(str)`
/// uses (which is Python's own `float(str)`, stripping leading AND
/// trailing whitespace) -- verified live: `np.longdouble("  3.5  ")` raises
/// `ValueError: invalid literal for long double:   3.5  ` while
/// `np.float64("  3.5  ")` succeeds (`3.5`). Also verified: LEADING
/// whitespace is fine either way (`np.longdouble("  3.5")` -> `3.5`,
/// `np.longdouble("\t3.5")` -> `3.5`) -- `strtold` itself skips leading
/// whitespace before parsing; it is characters left over AFTER the parsed
/// number (trailing whitespace, or any other trailing garbage) that numpy
/// rejects, matching C's `strtold`-then-"is anything left?" pattern.
/// `np.clongdouble(str)` was ALSO measured live and found to NOT have this
/// restriction (`np.clongdouble("1+2j  ")` succeeds) -- it goes through
/// Python's own lenient `complex(str)`, same as `np.complex128`, so this
/// check is applied to `LongDouble` only, never `CLongDouble`.
fn reject_trailing_whitespace_str(obj: &Bound<'_, PyAny>, typename: &str) -> PyResult<()> {
    if let Ok(s) = obj.cast::<pyo3::types::PyString>() {
        let s: String = s.extract()?;
        if s.trim_end() != s {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid literal for {typename}: {s}"
            )));
        }
    }
    Ok(())
}

fn reject_sequence(obj: &Bound<'_, PyAny>, typename: &str) -> PyResult<()> {
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        return Err(PyTypeError::new_err(format!(
            "anionpy.{typename}() does not support array-like/multi-element input \
             (real numpy builds an array for this input instead of a scalar); \
             use anionpy.array(..., dtype='{typename}') instead"
        )));
    }
    Ok(())
}

// ---- 0-d `squeeze(axis=...)` / `transpose(*axes)` axis-argument handling
// -----------------------------------------------------------------------
//
// This section reproduces numpy 2.5.1's real (live-verified, and genuinely
// INCONSISTENT) axis-argument acceptance for a 0-dimensional scalar. The
// two methods use COMPLETELY DIFFERENT C-level dispatchers in real numpy,
// confirmed by their divergent behavior, so they get separate rule sets
// below rather than a shared "axis parser":
//
// `squeeze(axis=None)`: single positional-or-keyword param named `axis`.
//   - absent or `None` -> identity (no-op).
//   - a single int-like value: ONLY `0` or `-1` succeed (identity); every
//     other in-range-for-a-C-`int` value raises `AxisError`; out-of-range
//     magnitude raises `OverflowError`/`ValueError` first (see
//     `axis_int_value` below). This `{0, -1}` acceptance on an object with
//     NO axes at all is numpy's own bug/inconsistency (confirmed live) --
//     reproduced exactly, not tidied away.
//   - a tuple: only `()` succeeds; ANY non-empty tuple raises `AxisError`
//     for its first element's value, even `(0,)`/`(-1,)` -- i.e. the SAME
//     values that succeed as a bare int fail inside a tuple. Also a real
//     inconsistency, also reproduced exactly.
//   - a `bool` (or `bool_` scalar) is explicitly rejected with its own
//     `TypeError` (message differs for tuple-element vs bare-value form).
//   - anything else without `__index__` (float/str/list/dict/...) raises
//     `TypeError` naming its Python type.
//
// `transpose(*axes)`: pure variadic positional (`*axes`), NO keyword
//   accepted at all -- `transpose(axis=0)` raises `TypeError:
//   generic.transpose() takes no keyword arguments` regardless of the
//   value. Given the resulting `axes` tuple:
//   - zero args, OR the single arg is `None` -> identity.
//   - the single arg is a tuple/list/bytes (a "sequence"): empty -> OK;
//     non-empty -> convert elements left-to-right, stopping at the first
//     one that fails (either `TypeError` for a bad element type, or an
//     i64-range overflow raising `Maximum allowed dimension exceeded`);
//     if every element converts, `ValueError: axes don't match array`
//     (numpy never returns a valid *permutation* for `ndim == 0` unless
//     the sequence was empty).
//   - `axes` has >= 2 elements (the bare `*axes` spelling, e.g.
//     `transpose(0, 1)`): same left-to-right element conversion as above.
//   - the single arg is a bare int-like (non-bool) value: `ValueError:
//     axes don't match array` if it fits an `i64`, else `Maximum allowed
//     dimension exceeded` (note: NO `AxisError` anywhere in `transpose`'s
//     path -- confirmed live, `transpose` and `squeeze` really do use
//     different error machinery on the same 0-d object).
//   - the single arg is `str` -> `TypeError: 'str' object cannot be
//     interpreted as an integer`.
//   - the single arg is anything else (bool, float, complex, dict, ...)
//     -> `TypeError: expected a sequence of integers or a single integer,
//     got '{str(value)}'` (note: `str(value)`, not `repr(value)`).

/// True for a Python `bool` or any scalar (numpy's or anionpy's own) whose
/// type name is exactly `"bool"` -- numpy 2.x's own scalar type name for
/// `bool_`/`np.bool` (anionpy's `Bool_` pyclass registers the same Python name,
/// see its `#[pyclass(name = "bool", ...)]` above). Used because real numpy
/// explicitly special-cases `bool` for axis arguments (rejecting it with
/// its OWN message even though `bool` is an `int` subclass and would
/// otherwise convert just fine) -- live-verified for both `squeeze` and
/// `transpose`.
fn is_axis_bool_like(obj: &Bound<'_, PyAny>) -> bool {
    if obj.is_instance_of::<PyBool>() {
        return true;
    }
    obj.get_type().name().map(|n| n == "bool").unwrap_or(false)
}

/// numpy's Python type name for a `TypeError` axis message, e.g. `'float'
/// object cannot be interpreted as an integer` -- uses the object's actual
/// runtime type name (`obj.get_type().name()`), NOT a hardcoded label, so
/// this reads correctly for anionpy's own scalar types too (e.g. `'float64'
/// object cannot be interpreted as an integer` for an anionpy float scalar,
/// matching real numpy's `'numpy.float64' object cannot be interpreted...`
/// -- close enough in spirit; exact numpy-internal type-name strings for
/// OTHER numpy types are out of scope here since anionpy never receives real
/// numpy objects as axis arguments in its own test corpus).
fn axis_type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "object".to_string())
}

fn axis_cannot_interpret_err(obj: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "'{}' object cannot be interpreted as an integer",
        axis_type_name(obj)
    ))
}

/// Convert `obj` to an `i128` via Python's `__index__` protocol (exactly
/// what `PyArray_PyIntAsIntp`/`PyNumber_Index` use in real numpy -- NOT
/// `__int__`, which is why a `float` axis raises `TypeError` instead of
/// silently truncating, live-verified: `np.float64.squeeze` has no
/// `__index__`, only `np.int*`/`np.uint*`/Python `int`/`bool` do). Returns
/// `Err` for anything without `__index__` (bool is handled by the caller
/// BEFORE this, since numpy's bool-rejection message differs by context).
fn axis_index_i128(obj: &Bound<'_, PyAny>) -> PyResult<i128> {
    obj.extract::<i128>()
}

/// `astype`'s `subok=`/`copy=` on anionpy's scalar types (both accepted and
/// immediately discarded, see the doc comment on `astype` below) --
/// measured live 2026-08-02 to be parsed via real numpy scalar
/// `.astype()`'s C-level integer-format argument machinery, NOT plain
/// Python truthiness: `np.int32(5).astype(np.int64, subok=1.5, ...)`
/// raises `TypeError: integer argument expected, got float` (a DISTINCT
/// message from the generic `__index__`-protocol failure), while
/// `subok="x"`/`[1]`/`None` raise the generic `'<type>' object cannot be
/// interpreted as an integer`. Also confirmed `np.bool_(True)` itself is
/// REJECTED here (no `__index__` on `numpy.bool`) even though plain
/// Python `bool`/`int` ARE accepted -- same defect shape as
/// `reductions.rs`'s `keepdims_index_like` for `sum`/`mean`/etc; this is
/// a deliberate near-duplicate of that helper (not a shared import) per
/// this crate's existing module-duplication precedent (see that file's
/// own doc comment on `do_reduce_axis`).
fn scalar_index_like_check(obj: &Bound<'_, PyAny>) -> PyResult<()> {
    if obj.is_instance_of::<pyo3::types::PyFloat>() {
        return Err(PyTypeError::new_err("integer argument expected, got float"));
    }
    match obj.getattr("__index__") {
        Err(_) => Err(axis_cannot_interpret_err(obj)),
        Ok(idx_fn) => {
            idx_fn.call0()?;
            Ok(())
        }
    }
}

/// `squeeze`'s single-axis-value pipeline (used for both the bare-value
/// form and a tuple's first element, which raise the SAME magnitude
/// errors but different bool/type messages -- see `in_tuple`). Mirrors
/// real numpy's own two-stage `PyLong_AsLong`-then-C-`int`-range check
/// (same shape as `weak_int_overflow_check` elsewhere in this file):
/// first make sure the Python int fits a C `long` (`i64` here) at all
/// (else `OverflowError`), then that it ALSO fits a C `int` (`i32`, else
/// `ValueError`) -- numpy's axis argument is parsed as a C `int`, not an
/// `intp`, unlike most other axis parameters in this codebase.
fn axis_int_value(obj: &Bound<'_, PyAny>, in_tuple: bool) -> PyResult<i32> {
    if is_axis_bool_like(obj) {
        return Err(PyTypeError::new_err(if in_tuple {
            "integers are required for the axis tuple elements"
        } else {
            "an integer is required for the axis"
        }));
    }
    let v = axis_index_i128(obj).map_err(|_| axis_cannot_interpret_err(obj))?;
    if i64::try_from(v).is_err() {
        return Err(PyOverflowError::new_err(
            "Python int too large to convert to C long",
        ));
    }
    let v64 = v as i64;
    i32::try_from(v64)
        .map_err(|_| PyValueError::new_err("integer won't fit into a C int"))
}

/// numpy's real, disclosed-inconsistent 0-d `squeeze(axis=...)` check --
/// see this section's top comment. `axis` is `None` for BOTH "argument
/// absent" and "argument explicitly `None`" (pyo3 collapses these the same
/// way real numpy does: both mean "no axis restriction").
fn squeeze_axis_ok(axis: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(ax) = axis else { return Ok(()) };
    if ax.is_none() {
        return Ok(());
    }
    if let Ok(tup) = ax.cast::<PyTuple>() {
        if tup.is_empty() {
            return Ok(());
        }
        let first = tup.get_item(0)?;
        let n = axis_int_value(&first, true)?;
        return Err(axis_error(
            n as isize,
            Some(0),
            &format!("axis {n} is out of bounds for array of dimension 0"),
        ));
    }
    let n = axis_int_value(ax, false)?;
    if n == 0 || n == -1 {
        return Ok(());
    }
    Err(axis_error(
        n as isize,
        Some(0),
        &format!("axis {n} is out of bounds for array of dimension 0"),
    ))
}

/// Resolves `squeeze`'s single `axis` parameter from raw `(*args,
/// **kwargs)`, matching numpy's exact `TypeError` text for every
/// malformed-call shape (arity, unknown keyword, positional+keyword
/// collision) -- live-verified against numpy 2.5.1. Using raw
/// `*args`/`**kwargs` instead of pyo3's own `#[pyo3(signature = (axis=
/// None))]` arg-count/kwarg-name enforcement is deliberate: pyo3's
/// auto-generated messages prefix the class name (`"int8.squeeze() got an
/// unexpected keyword..."`), but real numpy's is unprefixed (`"squeeze()
/// got an unexpected keyword..."`) since `squeeze` is numpy's single
/// shared `generic` method across every scalar dtype.
fn resolve_squeeze_axis<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if args.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "squeeze() takes from 0 to 1 positional arguments but {} were given",
            args.len()
        )));
    }
    if let Some(kw) = kwargs {
        for (key, _) in kw.iter() {
            let key_str = key.extract::<String>().unwrap_or_default();
            if key_str != "axis" {
                return Err(PyTypeError::new_err(format!(
                    "squeeze() got an unexpected keyword argument '{key_str}'"
                )));
            }
        }
        if !args.is_empty() && kw.contains("axis")? {
            return Err(PyTypeError::new_err(
                "argument for squeeze() given by name ('axis') and position (position 0)",
            ));
        }
    }
    if !args.is_empty() {
        return Ok(Some(args.get_item(0)?));
    }
    if let Some(kw) = kwargs {
        if let Some(v) = kw.get_item("axis")? {
            return Ok(Some(v));
        }
    }
    Ok(None)
}

/// numpy's real `str(value)` (not `repr`) rendering for the
/// `"expected a sequence of integers or a single integer, got '{}'"`
/// message -- calls Python's own `str()` rather than reimplementing
/// per-type formatting, so it's byte-identical for every object type numpy
/// could ever be handed.
fn axis_pystr(obj: &Bound<'_, PyAny>) -> String {
    obj.str()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "?".to_string())
}

fn transpose_expected_seq_err(obj: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "expected a sequence of integers or a single integer, got '{}'",
        axis_pystr(obj)
    ))
}

/// `transpose`'s per-element conversion for a sequence (tuple/list/`*args`)
/// of axis values on a 0-d scalar -- see this section's top comment. Only
/// checks TYPE/magnitude per element (never `AxisError`: `transpose` has no
/// per-element validity notion, only an aggregate length check performed by
/// the caller after every element converts successfully).
fn transpose_seq_element_ok(obj: &Bound<'_, PyAny>) -> PyResult<()> {
    if is_axis_bool_like(obj) {
        return Err(PyTypeError::new_err("an integer is required"));
    }
    let v = axis_index_i128(obj).map_err(|_| axis_cannot_interpret_err(obj))?;
    if i64::try_from(v).is_err() {
        return Err(PyValueError::new_err("Maximum allowed dimension exceeded"));
    }
    Ok(())
}

/// Validate a whole sequence of transpose axis elements (tuple, list, the
/// bare `*args` tuple, or a `bytes` object's byte values) left-to-right,
/// stopping at the first element that fails; an all-valid non-empty
/// sequence always mismatches on a 0-d scalar (`ndim == 0` accepts no
/// permutation except the empty one).
fn transpose_seq_ok<'a>(items: impl Iterator<Item = Bound<'a, PyAny>>) -> PyResult<()> {
    let mut any = false;
    for item in items {
        any = true;
        transpose_seq_element_ok(&item)?;
    }
    if any {
        Err(PyValueError::new_err("axes don't match array"))
    } else {
        Ok(())
    }
}

/// numpy's real 0-d `transpose(*axes)` check -- see this section's top
/// comment for the full disclosed rule set (deliberately NOT reusing
/// `squeeze`'s helpers: real numpy genuinely dispatches these two methods
/// through unrelated C paths on the same 0-d object, confirmed by their
/// divergent error types -- `AxisError` for `squeeze`, plain `ValueError`
/// for `transpose`).
fn transpose_axes_ok(axes: &Bound<'_, PyTuple>) -> PyResult<()> {
    if axes.is_empty() {
        return Ok(());
    }
    if axes.len() >= 2 {
        return transpose_seq_ok(axes.iter());
    }
    // axes.len() == 1
    let val = axes.get_item(0)?;
    if val.is_none() {
        return Ok(());
    }
    if let Ok(tup) = val.cast::<PyTuple>() {
        return transpose_seq_ok(tup.iter());
    }
    if let Ok(list) = val.cast::<PyList>() {
        return transpose_seq_ok(list.iter());
    }
    if let Ok(bytes) = val.cast::<PyBytes>() {
        let n = bytes.as_bytes().len();
        return if n == 0 {
            Ok(())
        } else {
            Err(PyValueError::new_err("axes don't match array"))
        };
    }
    if is_axis_bool_like(&val) {
        return Err(transpose_expected_seq_err(&val));
    }
    if let Ok(v) = axis_index_i128(&val) {
        return if i64::try_from(v).is_ok() {
            Err(PyValueError::new_err("axes don't match array"))
        } else {
            Err(PyValueError::new_err("Maximum allowed dimension exceeded"))
        };
    }
    if val.is_instance_of::<pyo3::types::PyString>() {
        return Err(axis_cannot_interpret_err(&val));
    }
    Err(transpose_expected_seq_err(&val))
}

/// Raised for `transpose(axis=...)`/`transpose(axes=...)` -- real numpy's
/// `generic.transpose()` accepts ONLY the bare `*axes` spelling, no keyword
/// whatsoever (live-verified: even `transpose(axes=0)` on the FIRST
/// positional-style name raises this, not a normal "unexpected keyword"
/// naming that keyword).
///
/// DISCLOSED, PERMANENT GAP (not fixable, minimized rather than papered
/// over): real numpy's message text for this exact error depends on the
/// caller's Python bytecode form, live-verified against numpy 2.5.1 on all
/// 14 dtypes, all 4 combinations of {literal `name=value` vs. `**kwargs`
/// unpacking} x {direct attribute access vs. `getattr()`}:
///   - `v.transpose(axis=0)` (literal syntax, DIRECT attribute access,
///     i.e. `LOAD_METHOD`/`CALL_METHOD`) -> `"generic.transpose() takes no
///     keyword arguments"` (prefixed).
///   - `v.transpose(**kw)` (kwargs-unpacked, `CALL_FUNCTION_EX`) ->
///     `"transpose() takes no keyword arguments"` (unprefixed).
///   - `getattr(v, "transpose")(axis=0)` (literal syntax via a fetched
///     bound method) -> unprefixed.
///   - `getattr(v, "transpose")(**kw)` -> unprefixed.
/// The prefixed wording is the OUTLIER: it fires only on the single literal
/// direct-attribute-call form. Every other form -- including the one this
/// project's differential harness actually exercises (`resolve_ionp`/
/// `resolve_numpy` dispatch members through `getattr`, per harness.py's own
/// docstring) -- is unprefixed. PyO3's `#[pymethods]` trampoline has no
/// hook to observe the caller's bytecode form at all, so anionpy cannot be
/// correct on all 4 forms; this function deliberately emits the UNPREFIXED
/// wording to be correct on 3 of 4 (including the harness path), rather
/// than the earlier, wrong choice of always emitting the prefixed wording
/// (correct on only 1 of 4, and wrong on the harness path specifically).
fn transpose_no_kwargs_err() -> PyErr {
    PyTypeError::new_err("transpose() takes no keyword arguments")
}

/// Resolve a "strong" source object's dtype (anything with a `.dtype`
/// attribute: a numpy scalar, an anionpy scalar, or a 0-d array of either) --
/// preferring anionpy's own `PyDType` directly (no numpy call at all) and
/// falling back to reading the numpy-side `.dtype.name` string attribute
/// (a plain attribute read, not asking numpy to compute anything) through
/// the same `dtype_name_to_dtype` lookup table used everywhere else.
fn strong_source_dtype(obj: &Bound<'_, PyAny>) -> PyResult<DType> {
    let dt_obj = obj.getattr("dtype")?;
    if let Ok(d) = dt_obj.extract::<PyRef<'_, PyDType>>() {
        return Ok(d.inner);
    }
    let name: String = dt_obj.getattr("name")?.extract()?;
    dtype_name_to_dtype(&name)
}

// ---------------------------------------------------------------------
// PART 1 shared infrastructure: numpy scalars carry the full 0-d array
// surface (`.shape`, `.flags`, `.item()`, `.astype()`, `.reshape()`, ...).
// `ScalarFlags` mirrors `numpy.core.multiarray.flagsobj` for a scalar --
// numpy hardcodes the SAME flag values for every dtype regardless of
// contents (verified live: `np.int8(5).flags` == `np.complex128(3).flags`
// textually: C_CONTIGUOUS/F_CONTIGUOUS/OWNDATA/ALIGNED True, WRITEABLE/
// WRITEBACKIFCOPY False).
// ---------------------------------------------------------------------

#[pyclass(name = "_ScalarFlags", module = "anionpy")]
pub struct ScalarFlags;

#[pymethods]
impl ScalarFlags {
    #[getter(c_contiguous)]
    fn c_contiguous(&self) -> bool {
        true
    }
    #[getter(f_contiguous)]
    fn f_contiguous(&self) -> bool {
        true
    }
    #[getter(owndata)]
    fn owndata(&self) -> bool {
        true
    }
    #[getter(writeable)]
    fn writeable(&self) -> bool {
        false
    }
    #[getter(aligned)]
    fn aligned(&self) -> bool {
        true
    }
    #[getter(writebackifcopy)]
    fn writebackifcopy(&self) -> bool {
        false
    }
    #[getter(fnc)]
    fn fnc(&self) -> bool {
        false
    }
    #[getter(forc)]
    fn forc(&self) -> bool {
        true
    }
    #[getter(behaved)]
    fn behaved(&self) -> bool {
        false
    }
    #[getter(carray)]
    fn carray(&self) -> bool {
        false
    }
    #[getter(farray)]
    fn farray(&self) -> bool {
        false
    }
    fn __getitem__(&self, key: &str) -> PyResult<bool> {
        match key {
            "C_CONTIGUOUS" | "CONTIGUOUS" | "C" => Ok(true),
            "F_CONTIGUOUS" | "FORTRAN" | "F" => Ok(true),
            "OWNDATA" | "O" => Ok(true),
            "WRITEABLE" | "W" => Ok(false),
            "ALIGNED" | "A" => Ok(true),
            "WRITEBACKIFCOPY" | "X" => Ok(false),
            "FNC" => Ok(false),
            "FORC" => Ok(true),
            "BEHAVED" | "B" => Ok(false),
            "CARRAY" | "CA" => Ok(false),
            "FARRAY" | "FA" => Ok(false),
            _ => Err(PyValueError::new_err(format!("flag not found: '{key}'"))),
        }
    }
    fn __repr__(&self) -> String {
        "  C_CONTIGUOUS : True\n  F_CONTIGUOUS : True\n  OWNDATA : True\n  WRITEABLE : False\n  ALIGNED : True\n  WRITEBACKIFCOPY : False\n".to_string()
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// numpy's 0-d `.nonzero()` error message. Live-verified against numpy
/// 2.5.1 across all 14 scalar types: `bool_` alone gets the LONG form
/// (with the third "If the context..." sentence); every other dtype gets
/// the SHORT form. Not a guess -- re-checked in a loop across all 14
/// dtypes specifically because a uniform assumption would have been wrong.
const NONZERO_MSG_SHORT: &str = "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.";
const NONZERO_MSG_LONG: &str = "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead. If the context of this error is of the form `arr[nonzero(cond)]`, just use `arr[cond]`.";

// ---------------------------------------------------------------------
// PART 5 (coordinator audit, 2026-08-02): a scalar's `.flat` attribute.
// numpy's is a genuine `numpy.flatiter` (a C-level fixed-position iterator
// with `.index`, `__getitem__`, `__iter__`/`__next__`, `__len__`); anionpy's
// is a small dedicated pyclass mirroring the single-element case (all
// scalars are shape `()`, size 1, so `.flat` always has exactly one
// element). Live-verified against numpy 2.5.1: `list(np.int8(5).flat) ==
// [np.int8(5)]`; `f[0]` returns the element, `f[1]`/`f[-2]` raise
// `IndexError: index {i} is out of bounds for size 1`; `f.index == 0`
// before consuming the single element via `next()`, `== 1` (== len) after;
// iterating past the single element raises `StopIteration`.
// ---------------------------------------------------------------------

#[pyclass(name = "flatiter", module = "anionpy")]
pub struct ScalarFlatIter {
    value: Py<PyAny>,
    // `pyclass` requires `Send + Sync`, so a plain `Cell<bool>` (Send but
    // NOT Sync) doesn't qualify -- `AtomicBool` gives the same single-writer
    // interior mutability under `&self` while staying thread-safe.
    consumed: std::sync::atomic::AtomicBool,
}

impl ScalarFlatIter {
    fn new(value: Py<PyAny>) -> Self {
        ScalarFlatIter {
            value,
            consumed: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

#[pymethods]
impl ScalarFlatIter {
    fn __len__(&self) -> usize {
        1
    }
    #[getter]
    fn index(&self) -> usize {
        if self.consumed.load(std::sync::atomic::Ordering::SeqCst) {
            1
        } else {
            0
        }
    }
    fn __getitem__(&self, py: Python<'_>, idx: isize) -> PyResult<Py<PyAny>> {
        if idx == 0 || idx == -1 {
            Ok(self.value.clone_ref(py))
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {idx} is out of bounds for size 1"
            )))
        }
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        if self.consumed.swap(true, std::sync::atomic::Ordering::SeqCst) {
            None
        } else {
            Some(self.value.clone_ref(py))
        }
    }
}

/// Wrap a freshly computed single-element `Buffer` back up into the
/// matching concrete scalar pyclass -- used by every leaf's `.astype()`
/// (strong-source unsafe cast, reusing `Buffer::cast_to`) and by
/// `.copy()`/`.conjugate()`/`.min()`/`.max()`/etc. that must return a
/// same-or-different-dtype scalar rather than an `anionpy.ndarray`.
fn scalar_from_buffer(py: Python<'_>, buf: &Buffer) -> PyResult<Py<PyAny>> {
    Ok(match buf {
        Buffer::Bool(v) => Py::new(py, Bool_::init_chain(v[0]))?.into_any(),
        Buffer::I8(v) => Py::new(py, Int8::init_chain(v[0]))?.into_any(),
        Buffer::I16(v) => Py::new(py, Int16::init_chain(v[0]))?.into_any(),
        Buffer::I32(v) => Py::new(py, Int32::init_chain(v[0]))?.into_any(),
        Buffer::I64(v) => Py::new(py, Int64::init_chain(v[0]))?.into_any(),
        Buffer::U8(v) => Py::new(py, UInt8::init_chain(v[0]))?.into_any(),
        Buffer::U16(v) => Py::new(py, UInt16::init_chain(v[0]))?.into_any(),
        Buffer::U32(v) => Py::new(py, UInt32::init_chain(v[0]))?.into_any(),
        Buffer::U64(v) => Py::new(py, UInt64::init_chain(v[0]))?.into_any(),
        Buffer::F16(v) => Py::new(py, Float16::init_chain(v[0]))?.into_any(),
        Buffer::F32(v) => Py::new(py, Float32::init_chain(v[0]))?.into_any(),
        Buffer::F64(v) => Py::new(py, Float64::init_chain(v[0]))?.into_any(),
        Buffer::C64(v) => Py::new(py, Complex64::init_chain(v[0]))?.into_any(),
        Buffer::C128(v) => Py::new(py, Complex128::init_chain(v[0]))?.into_any(),
        // No `str_`/`bytes_` scalar pyclass exists yet -- phase 2 shipped
        // 0-d/1-d S/U array construction, indexing (`elem_to_py` in
        // `ndarray_attrs.rs`), and repr, but not a dedicated numpy-scalar
        // wrapper type for `.astype()`/`.copy()`/etc. to return here.
        Buffer::S(_, _) | Buffer::U(_, _) => {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "anionpy: no scalar wrapper type for S/U string dtypes yet",
            ))
        }
    })
}

/// Shared `.reshape()` implementation for all 14 leaf scalars: numpy always
/// returns an `anionpy.ndarray` on success (never another scalar), regardless
/// of target shape (verified: `np.int8(5).reshape(1)` is `numpy.ndarray`),
/// rejects a zero-arg call with `TypeError: reshape() takes exactly 1
/// argument (0 given)`, accepts either varargs ints or a single tuple/list,
/// supports one `-1` inferred dim, and raises numpy's bespoke
/// no-space-in-tuple `ValueError` (`cannot reshape array of size 1 into
/// shape (4,5)`) when the resolved size isn't 1.
fn scalar_reshape(
    py: Python<'_>,
    shape: &Bound<'_, PyTuple>,
    order: Option<&str>,
    buf: Buffer,
) -> PyResult<Py<PyAny>> {
    if shape.is_empty() {
        return Err(PyTypeError::new_err(
            "reshape() takes exactly 1 argument (0 given)",
        ));
    }
    let dims: Vec<isize> = if shape.len() == 1 {
        let first = shape.get_item(0)?;
        match first.extract::<Vec<isize>>() {
            Ok(t) => t,
            Err(_) => vec![first.extract::<isize>()?],
        }
    } else {
        shape
            .iter()
            .map(|x| x.extract::<isize>())
            .collect::<PyResult<Vec<_>>>()?
    };
    let neg_count = dims.iter().filter(|&&d| d < 0).count();
    if neg_count > 1 {
        return Err(PyValueError::new_err("can only specify one unknown dimension"));
    }
    let resolved: Vec<usize> = dims
        .iter()
        .map(|&d| if d < 0 { 1usize } else { d as usize })
        .collect();
    let total: usize = resolved.iter().product();
    if total != 1 {
        return Err(to_py_err(ionp_core::IonpError::Reshape {
            size: 1,
            shape: resolved,
        }));
    }
    let ord = if order == Some("F") { Order::F } else { Order::C };
    let arr = NdArray::from_buffer(buf, resolved, ord).map_err(to_py_err)?;
    Ok(Py::new(py, PyArray { inner: arr })?.into_any())
}

// ---------------------------------------------------------------------
// Comparison dunders (`__eq__`/`__ne__`/`__lt__`/`__le__`/`__gt__`/`__ge__`)
// and `__hash__`, shared by every concrete leaf type via the macros below.
//
// THE BUG THIS SECTION FIXES: none of the 14 concrete scalar pyclasses used
// to define ANY comparison dunder, so every comparison (including
// self-equality on a freshly built pair, `anionpy.int8(3) == 3`,
// `anionpy.float64(1.5) == anionpy.float64(1.5)`) fell through to CPython's
// default identity check and was always `False`. See `scalars.py`'s
// WITHDRAWN note on `True_`/`False_` for the measurement that found this.
//
// STRATEGY: do not reimplement comparison semantics here at all -- marshal
// both operands into `ionp_core::Buffer`s and hand them to
// `ionp_core::ufunc::binary_op`, the EXACT SAME function
// `ionp-py/src/lib.rs`'s `PyArray::__eq__`/`__lt__`/etc. already call for
// `anionpy.ndarray` comparisons (verified by reading those methods directly:
// `coerce_operand_for_compare` + `binary_op`). That function already
// implements, per-dtype-pair: `promote_dtype`-based cross-width value
// comparison (`anionpy.int8(-1) == anionpy.uint8(255)` -> promotes to `int16`,
// compares `-1 == 255` -> `False`, matching `np.int8(-1) ==
// np.uint8(255)`), IEEE NaN semantics (`x != x`), and complex ordering.
// Reusing it means this file carries zero duplicate comparison logic to
// drift out of sync with the array path.
//
// MEASURED, NOT ASSUMED: two things the task brief suggested but that this
// implementation deliberately does NOT do, because live measurement against
// real numpy 2.5.1 contradicted them --
//   1. "Ordering comparisons on complex must raise TypeError" -- FALSE for
//      numpy 2.5.1 scalars. Measured live: `np.complex128(1+2j) >
//      np.complex128(3+4j)` returns `False` (no exception, no warning, `-W
//      error` does not turn it into one either); `np.complex128(1+2j) <
//      np.complex128(3+4j)` returns `True`. This is exactly the
//      lexicographic-by-real-then-imaginary rule (NaN component -> False
//      unconditionally) that `ionp_core::ufunc::cmp_complex` ALREADY
//      implements for array comparisons -- verified by reading its doc
//      comment, which independently documents the identical NaN rule as
//      measured against numpy 2.5.1. Reusing `binary_op` therefore matches
//      real numpy on this axis for free; hand-raising `TypeError` here
//      would have been the actual divergence.
//   2. "Weak Python-int comparison casts into self's own (possibly narrow)
//      dtype" -- FALSE. `np.int8(3) == 259` is `False`, not `True`. If
//      comparison wrapped `259` into `int8` the way scalar CONSTRUCTION's
//      "strong" path does, `259 as i8` wraps to `3` and the comparison
//      would wrongly come back `True`. Real numpy instead promotes a bare
//      Python `int`/`float`/`complex`/`bool` operand to its KIND's default
//      dtype (`int` -> `int64`, `float` -> `float64`, `complex` ->
//      `complex128`, `bool` -> `bool`) and lets ordinary `promote_dtype`
//      resolve the final comparison domain against `self`'s dtype from
//      there -- the SAME rule `ionp-py/src/lib.rs`'s own
//      `coerce_operand_for_compare` already documents and implements for
//      `ndarray` comparisons (see that function's doc comment), now
//      confirmed to also hold for scalar-vs-scalar comparison, not just
//      array comparison.
//
// DISCLOSED, NOT ATTEMPTED: `other` values this module cannot classify
// (a string, `None`, a list, an arbitrary object) make `compare_operand`
// return `Ok(None)`, which every dunder below turns into Python's
// `NotImplemented`. Real numpy's scalar `__eq__`/`__ne__` against a
// STRING specifically does NOT do this -- it returns `np.False_`/
// `np.True_` (verified: `np.int8(3) == 'abc'` -> `np.False_`, not a
// `NotImplemented`-driven identity fallback), via an internal
// asarray-into-a-string-dtype-then-cross-dtype-compare path that requires
// numpy's string dtype machinery. anionpy has no string/object dtype at all
// (see this crate's other disclosed gaps), so there is nothing to build
// that path on; `NotImplemented` is the closest correct behavior available
// and happens to match numpy's OWN behavior for a `None` right-hand side
// exactly (`np.int8(3) == None` -> plain Python `False` via the identical
// `NotImplemented` fallback, verified), just not numpy's string case.
// Comparing a scalar against a MULTI-ELEMENT numpy/anionpy array is a second,
// separate disclosed gap for the identical structural reason the
// list/tuple constructor gap exists: real numpy broadcasts to an array
// result, but a PyO3 dunder has one fixed return type, so this raises
// (surfaced from Python's own `int()`/`float()`/`complex()` builtin
// refusing a multi-element source) rather than returning an array.
// ---------------------------------------------------------------------

/// Marshal a comparison right-hand side `obj` into a 1-element `Buffer`
/// carrying its own resolved dtype, or `Ok(None)` if `obj` is not a
/// comparable numeric kind at all (see the module-level doc comment above
/// for exactly what that covers and excludes).
fn compare_operand(obj: &Bound<'_, PyAny>) -> PyResult<Option<Buffer>> {
    if obj.hasattr("dtype")? {
        let dtype = match strong_source_dtype(obj) {
            Ok(d) => d,
            Err(_) => return Ok(None),
        };
        let buf = if dtype.is_bool() {
            Buffer::Bool(vec![obj.is_truthy()?])
        } else if dtype.is_integer() {
            construct_int_buffer(obj, dtype)?
        } else if dtype.is_floating() {
            construct_float_buffer(obj, dtype)?
        } else {
            construct_complex_buffer(obj, dtype)?
        };
        return Ok(Some(buf));
    }
    // Bare Python scalar: NEP 50 "weak", promoted to its kind's numpy
    // DEFAULT dtype -- see this module's doc comment, point 2, for why this
    // (not "cast into self's own dtype") is the measured-correct rule.
    // `PyBool` is checked ahead of `PyInt`/`PyComplex`/`PyFloat` because
    // Python's `bool` is a subclass of `int` (mirrors `classify_scalar` in
    // `ionp-py/src/lib.rs`, which this deliberately duplicates rather than
    // importing -- that one is a private fn of the `lib` module and this
    // ordering rule is simple enough to not be worth threading a `pub(crate)`
    // through for).
    if obj.is_instance_of::<PyBool>() {
        let b: bool = obj.extract()?;
        return Ok(Some(Buffer::Bool(vec![b])));
    }
    if let Ok(c) = obj.cast::<PyComplex>() {
        return Ok(Some(Buffer::C128(vec![C128::new(c.real(), c.imag())])));
    }
    if obj.is_instance_of::<PyFloat>() {
        let v: f64 = obj.extract()?;
        return Ok(Some(Buffer::F64(vec![v])));
    }
    if obj.is_instance_of::<PyInt>() {
        // Residual scope gap, deliberately mirrored from
        // `coerce_operand_for_compare` in `ionp-py/src/lib.rs`: a Python int
        // outside i64's range (e.g. `2**100`) raises `OverflowError` here
        // rather than whatever real numpy's own comparison does for such a
        // value -- not otherwise exercised by the differential corpus.
        let v: i128 = obj.extract()?;
        check_int_bounds(v, DType::I64)?;
        return Ok(Some(Buffer::I64(vec![v as i64])));
    }
    Ok(None)
}

/// Run `op` between `self_buf` (the scalar's own value, at its own dtype)
/// and whatever `other` marshals to via `compare_operand`, returning
/// Python's `NotImplemented` for an unclassifiable `other` and an
/// `anionpy.bool_`/`anionpy.True_`-style scalar (NOT a Python `bool`) otherwise
/// -- numpy returns `np.True_`/`np.False_` from scalar comparisons, not
/// Python `True`/`False` (`type(np.int8(3) == 3) is np.bool_`, verified),
/// so the return TYPE here is as load-bearing as the return VALUE.
///
/// ORDERING-vs-`str` NARROW FIX (2026-08-06, measured against real numpy
/// 2.5.1, not inherited from this module's own older "disclosed, not
/// attempted" note above): that note is right that `==`/`!=` against a
/// `str` already match numpy's actual behavior (numpy 2.x's `np.bool_`
/// scalar type's `__name__` IS `"bool"`, the same plain-identity-fallback
/// value/type this function's `NotImplemented` path already produces --
/// re-measured, not assumed) and that a full string-dtype coercion path is
/// out of reach. What the note did NOT separate out: ordering
/// (`<`/`<=`/`>`/`>=`) against a `str` specifically does NOT fall back to
/// Python's default `NotImplemented` protocol in real numpy at all --
/// numpy scalars always dispatch ordering through a ufunc call, so an
/// unclassifiable comparison type name raises `UFuncTypeError` (a
/// `TypeError` subclass), not CPython's generic `'>' not supported
/// between...` `TypeError`. Verified this is `str`-specific, not
/// "any unclassifiable operand": `np.int8(3) > None` still raises plain
/// `TypeError` (None has no ufunc-dispatchable dtype either, matching
/// this function's existing `NotImplemented` path exactly) -- only `str`
/// diverges. So only a bare Python `str` gets the richer exception here;
/// every other unclassifiable `other` (including `None`, lists, arbitrary
/// objects) keeps the pre-existing `NotImplemented` fallback unchanged.
fn scalar_richcmp<'py>(
    py: Python<'py>,
    op: BinaryOp,
    self_buf: Buffer,
    other: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let other_buf = match compare_operand(other)? {
        Some(b) => b,
        None => {
            let is_ordering = matches!(
                op,
                BinaryOp::Less | BinaryOp::LessEqual | BinaryOp::Greater | BinaryOp::GreaterEqual
            );
            if is_ordering && other.is_instance_of::<pyo3::types::PyString>() {
                // `BinaryOp::numpy_name()` is private to `ionp-core`, not
                // visible across the crate boundary here -- this local
                // table exists solely for the exception MESSAGE (the
                // differential harness only compares exception CLASS, per
                // `_compare_probe` in `scalar_cases.py`, so an approximate
                // but real numpy-style name is enough, not load-bearing).
                let name = match op {
                    BinaryOp::Less => "less",
                    BinaryOp::LessEqual => "less_equal",
                    BinaryOp::Greater => "greater",
                    BinaryOp::GreaterEqual => "greater_equal",
                    _ => unreachable!("is_ordering already restricted op to these four"),
                };
                return Err(no_ufunc_loop_err_dtypes(
                    name,
                    &[Some(self_buf.dtype()), None],
                    "did not contain a loop with signature matching types",
                ));
            }
            return Ok(py.NotImplemented().into_bound(py));
        }
    };
    let a = NdArray::from_buffer(self_buf, vec![], Order::C).map_err(to_py_err)?;
    let b = NdArray::from_buffer(other_buf, vec![], Order::C).map_err(to_py_err)?;
    let out = binary_op(op, &a, &b).map_err(to_py_err)?;
    let result = match out.buffer() {
        Buffer::Bool(v) => v[0],
        other => unreachable!("comparison binary_op must return a Bool buffer, got {other:?}"),
    };
    let scalar = Py::new(py, Bool_::init_chain(result))?;
    Ok(scalar.into_bound(py).into_any())
}

// ---------------------------------------------------------------------
// Scalar arithmetic: `__add__`/`__sub__`/.../`__neg__`/`__abs__`/... for
// all 14 concrete scalar types. Mirrors `coerce_operand`/`binary_op` in
// `ionp-py/src/lib.rs` (the `anionpy.ndarray` dunder implementation) as
// closely as possible so overflow-wrap, floor-division-toward-negative-
// infinity, div/mod-by-zero, shift-count edge cases, NEP 50 weak-scalar
// promotion, and complex-floordiv/mod rejection all come from the SAME
// already-verified `ionp_core::ufunc` engine rather than being
// re-derived here. The one deliberate divergence from `coerce_operand`:
// an unclassifiable operand returns `Ok(None)` here instead of raising
// `PyTypeError`, so the calling dunder can return Python's
// `NotImplemented` (required for scalars, per NEP 50 dunder-interop
// rules -- unlike arrays, this is not itself an in-scope fix for the
// array side).
// ---------------------------------------------------------------------

/// Coerce an arithmetic right-hand-side operand `obj` against a scalar
/// self of dtype `self_dtype` into an `NdArray`, or `Ok(None)` if `obj`
/// is not an arithmetic-compatible operand at all (another `anionpy.ndarray`,
/// a numpy scalar/array, or a bare Python bool/int/float/complex).
fn coerce_scalar_operand(self_dtype: DType, obj: &Bound<'_, PyAny>) -> PyResult<Option<NdArray>> {
    if let Ok(other) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(Some(other.inner.clone()));
    }
    if obj.hasattr("dtype")? {
        let py = obj.py();
        let np = pyo3::types::PyModule::import(py, "numpy")?;
        let as_array = np.getattr("asarray")?.call1((obj,))?;
        return Ok(Some(ndarray_from_numpy(&as_array)?));
    }
    if let Some(kind) = classify_scalar(obj) {
        let target = weak_target_dtype(self_dtype, kind);
        let buf = weak_scalar_buffer(obj, kind, target)?;
        return Ok(Some(NdArray::from_buffer(buf, vec![], Order::C).map_err(to_py_err)?));
    }
    Ok(None)
}

/// `coerce_scalar_operand`'s true-division-specific override: a bare
/// Python `int` against an integer/bool self forces the operand straight
/// to `F64` (matching `binary_out_dtype`'s `Divide` output, which always
/// floats an int/bool input) instead of bounds-checking it against a
/// dtype the division will never actually produce -- exact mirror of
/// `coerce_operand_for_truediv` in `ionp-py/src/lib.rs`.
fn coerce_scalar_operand_for_truediv(
    self_dtype: DType,
    obj: &Bound<'_, PyAny>,
) -> PyResult<Option<NdArray>> {
    if let Some(ScalarKind::Int) = classify_scalar(obj) {
        if self_dtype.is_integer() || self_dtype == DType::Bool {
            let buf = weak_scalar_buffer(obj, ScalarKind::Int, DType::F64)?;
            return Ok(Some(NdArray::from_buffer(buf, vec![], Order::C).map_err(to_py_err)?));
        }
    }
    coerce_scalar_operand(self_dtype, obj)
}

/// Wrap a computed arithmetic result back into a Python object: a 0-d
/// result (both operands were scalar-shaped) becomes the matching
/// concrete `anionpy` scalar via `scalar_from_buffer`; a non-0-d result
/// (`other` was a multi-element anionpy/numpy array) becomes a `PyArray`,
/// matching real numpy (`anionpy.int8(3) + anionpy.array([1,2,3])` is an
/// array, not a scalar).
fn wrap_arith_result(py: Python<'_>, out: NdArray) -> PyResult<Py<PyAny>> {
    if out.shape().is_empty() {
        scalar_from_buffer(py, out.buffer())
    } else {
        Ok(Py::new(py, PyArray { inner: out })?.into_any())
    }
}

/// Shared binary-arithmetic dispatch for every scalar leaf type's
/// `__add__`/`__sub__`/`__and__`/`__lshift__`/etc. (the ops that go
/// through `ionp_core::ufunc::binary_op`, i.e. everything except
/// `__truediv__` (needs `coerce_scalar_operand_for_truediv`), `__mod__`/
/// `__pow__` (go through `math_binary_op` instead), and `__divmod__`
/// (a tuple of two other ops). `reflected` swaps operand order for the
/// `__r*__` forms (`self` is always the "self" object either way; the
/// caller passes `reflected: true` from e.g. `__radd__`/`__rsub__` so
/// `other OP self` is computed instead of `self OP other`).
fn scalar_binop(
    py: Python<'_>,
    op: BinaryOp,
    self_buf: &Buffer,
    self_dtype: DType,
    other: &Bound<'_, PyAny>,
    reflected: bool,
) -> PyResult<Py<PyAny>> {
    let other_arr = match coerce_scalar_operand(self_dtype, other)? {
        Some(a) => a,
        None => return Ok(py.NotImplemented()),
    };
    let self_arr = NdArray::from_buffer(self_buf.clone(), vec![], Order::C).map_err(to_py_err)?;
    let out = if reflected {
        binary_op(op, &other_arr, &self_arr).map_err(to_py_err)?
    } else {
        binary_op(op, &self_arr, &other_arr).map_err(to_py_err)?
    };
    wrap_arith_result(py, out)
}

/// Same shape as `scalar_binop` but for `__mod__`/`__pow__`, which go
/// through `ionp_core::ufunc::math_binary_op` (`Remainder`/`Power`)
/// instead of `binary_op` (`np.mod is np.remainder`, verified against
/// real numpy 2.5.1 -- not a separate loop).
fn scalar_math_binop(
    py: Python<'_>,
    op: MathBinaryOp,
    self_buf: &Buffer,
    self_dtype: DType,
    other: &Bound<'_, PyAny>,
    reflected: bool,
) -> PyResult<Py<PyAny>> {
    let other_arr = match coerce_scalar_operand(self_dtype, other)? {
        Some(a) => a,
        None => return Ok(py.NotImplemented()),
    };
    let self_arr = NdArray::from_buffer(self_buf.clone(), vec![], Order::C).map_err(to_py_err)?;
    let out = if reflected {
        math_binary_op(op, &other_arr, &self_arr).map_err(to_py_err)?
    } else {
        math_binary_op(op, &self_arr, &other_arr).map_err(to_py_err)?
    };
    wrap_arith_result(py, out)
}

/// `__truediv__`/`__rtruediv__` dispatch: identical shape to
/// `scalar_binop` but coerces `other` through
/// `coerce_scalar_operand_for_truediv` instead.
fn scalar_truediv(
    py: Python<'_>,
    self_buf: &Buffer,
    self_dtype: DType,
    other: &Bound<'_, PyAny>,
    reflected: bool,
) -> PyResult<Py<PyAny>> {
    let other_arr = match coerce_scalar_operand_for_truediv(self_dtype, other)? {
        Some(a) => a,
        None => return Ok(py.NotImplemented()),
    };
    let self_arr = NdArray::from_buffer(self_buf.clone(), vec![], Order::C).map_err(to_py_err)?;
    let out = if reflected {
        binary_op(BinaryOp::Divide, &other_arr, &self_arr).map_err(to_py_err)?
    } else {
        binary_op(BinaryOp::Divide, &self_arr, &other_arr).map_err(to_py_err)?
    };
    wrap_arith_result(py, out)
}

/// `divmod(self, other)` / `divmod(other, self)`: numpy's own tuple of
/// `(a // b, a % b)`, not a single combined ufunc call (verified:
/// `type(divmod(np.int8(7), np.int8(2))) is tuple`) -- mirrors
/// `PyArray::__divmod__` exactly, including `retag_divmod_name` renaming
/// the underlying loop's name in the one place it's externally
/// observable (the "unsupported input types" message).
fn scalar_divmod(
    py: Python<'_>,
    self_buf: &Buffer,
    self_dtype: DType,
    other: &Bound<'_, PyAny>,
    reflected: bool,
) -> PyResult<Py<PyAny>> {
    let q = scalar_binop(py, BinaryOp::FloorDivide, self_buf, self_dtype, other, reflected)
        .map_err(|e| retag_divmod_name(e, py))?;
    if q.is_none(py) {
        // NotImplemented sentinel from an unclassifiable operand -- q is
        // literally `py.NotImplemented()` here (a `Py<PyAny>`, not a
        // `PyResult` short-circuit), so surface it as-is rather than
        // building a tuple around it.
        return Ok(q);
    }
    let r = scalar_math_binop(py, MathBinaryOp::Remainder, self_buf, self_dtype, other, reflected)
        .map_err(|e| retag_divmod_name(e, py))?;
    let tup = PyTuple::new(py, [q, r])?;
    Ok(tup.into_any().unbind())
}

/// CPython's `_Py_HashPointer` (`pyhash.c`, stable since 3.10): the hash
/// CPython's OWN float/complex hash algorithm falls back to for a NaN
/// component, derived from the containing PyObject's OWN address rather
/// than its (meaningless, since NaN != NaN) bit pattern -- which is why
/// `hash(float('nan'))` differs across separately-constructed NaN objects
/// but is stable across repeated `hash()` calls on the SAME object
/// (verified live: `a = float('nan'); hash(a) == hash(a)` but a second `b =
/// float('nan')` gives a different value). Reproduced here (not called into
/// numpy or Python for it) using OUR scalar's own pointer, which gives the
/// identical qualitative property for anionpy's own NaN scalars: stable
/// per-instance, differs across distinct instances, and does NOT try to
/// match numpy's own NaN hash bit-for-bit (impossible in principle -- that
/// would require matching CPython's memory allocator address, which is not
/// a value anionpy's Rust core has any business trying to reproduce).
fn hash_pointer(ptr: usize) -> isize {
    let bits = (std::mem::size_of::<usize>() * 8) as u32;
    let y = (ptr >> 4) | (ptr << (bits - 4));
    let x = y as isize;
    if x == -1 {
        -2
    } else {
        x
    }
}

/// CPython's `_Py_HashDouble`, minus the finite/infinite general case (that
/// half is just CPython's own float hash, reused as-is by constructing a
/// real Python `float` and calling its `__hash__` via PyO3's `.hash()` --
/// verified this reproduces numpy's own scalar float hash exactly:
/// `hash(np.float64(1.5)) == hash(1.5)`, `hash(np.float32(0.1)) ==
/// hash(float(np.float32(0.1)))`, `hash(np.uint64(2**64-1)) ==
/// hash(2**64-1)`, all measured live). Only the NaN branch needs its own
/// code, via `hash_pointer` above.
fn py_hash_double(py: Python<'_>, self_ptr: usize, v: f64) -> PyResult<isize> {
    if v.is_nan() {
        Ok(hash_pointer(self_ptr))
    } else {
        v.into_pyobject(py)?.hash()
    }
}

/// CPython's `complex_hash` (`complexobject.c`): `hashreal +
/// _PyHASH_IMAG * hashimag`, wrapping as an unsigned 64-bit add/multiply
/// (NOT Rust's default panicking-on-overflow arithmetic), with the
/// resulting `Py_uhash_t` reinterpreted as signed and `-1` remapped to
/// `-2` exactly like every other hash path here. `_PyHASH_IMAG` is
/// `1000003` (`sys.hash_info.imag`, verified). When `im == 0.0` this
/// collapses to exactly `hashreal` (`hashimag` is `hash(0.0) == 0`), which
/// is why `hash(anionpy.complex128(3+0j)) == hash(anionpy.int8(3)) ==
/// hash(3)` -- numbers of different types that compare equal must hash
/// equal, and this is the mechanism that makes it true, verified against
/// real numpy: `hash(np.complex128(3+0j)) == hash(3)`.
fn combine_complex_hash(hashreal: isize, hashimag: isize) -> isize {
    const PYHASH_IMAG: u64 = 1_000_003;
    let combined = (hashreal as i64 as u64).wrapping_add(PYHASH_IMAG.wrapping_mul(hashimag as i64 as u64));
    let x = combined as i64;
    if x == -1 {
        -2
    } else {
        x as isize
    }
}

/// Construct an integer-kind `target` `Buffer` from `obj`, per the
/// weak/strong rule documented at module level.
fn construct_int_buffer(obj: &Bound<'_, PyAny>, target: DType) -> PyResult<Buffer> {
    let py = obj.py();
    let is_strong = obj.hasattr("dtype")?;
    if is_strong {
        if let Ok(dt) = strong_source_dtype(obj) {
            if dt.is_floating() {
                let f: f64 = py
                    .import("builtins")?
                    .call_method1("float", (obj,))?
                    .extract()?;
                return Ok(int_buffer_from_f64(f, target));
            }
        }
    }
    // Weak source, or a strong bool/int source (a strong complex source
    // naturally raises TypeError from `int()` below, matching numpy).
    let as_int = py.import("builtins")?.call_method1("int", (obj,))?;
    let v: i128 = as_int.extract()?;
    if !is_strong {
        weak_int_overflow_check(v, target)?;
    }
    Ok(int_buffer_from_i128(v, target))
}

/// numpy's own derived two-stage overflow-message rule for a weak (bare
/// Python `int`) source that doesn't fit in the target dtype: first try a
/// signed `i64` conversion; on failure, retry an unsigned `u64` conversion
/// ONLY for unsigned targets whose itemsize is >= 4 bytes (`uint32`/
/// `uint64` -- NOT `uint8`/`uint16`, which stay on the "too large" path
/// even for a value that would fit a `u64`; verified live:
/// `np.uint8(2**63)`/`np.uint16(2**63)` -> "Python int too large to convert
/// to C long", but `np.uint32(2**63)` -> "Python integer ... out of bounds
/// for uint32"). If both stages fail (or the u64 retry doesn't apply), the
/// message is uniformly "Python int too large to convert to C long"
/// (verified: this is also numpy's message for `np.uint64` with a value
/// that doesn't fit in `u64` either, and for signed targets no matter how
/// large the value, since signed targets never get the u64 retry --
/// confirmed via `np.int64(2**63)` -> "too large" even though `2**63` fits
/// a `u64`). If either applicable conversion succeeds, fall through to the
/// ordinary bounds check against the target's real range, producing
/// "Python integer {v} out of bounds for {target}" on failure.
/// SEV-1 dedup (2026-08-02): was private to this module; widened to
/// `pub(crate)` so `lib.rs` can call this SAME implementation instead of
/// hand-maintaining a byte-for-byte duplicate (`list_int_overflow_check`,
/// now removed -- see its former call site's comment) or, worse, skipping
/// the check family entirely (`weak_scalar_buffer`'s `ScalarKind::Int`
/// arm, also fixed alongside this -- see its own comment). No behavior
/// change to this function itself.
pub(crate) fn weak_int_overflow_check(v: i128, target: DType) -> PyResult<()> {
    let fits_i64 = i64::try_from(v).is_ok();
    if !fits_i64 {
        let fits_u64 = target.is_unsigned_integer()
            && target.itemsize() >= 4
            && u64::try_from(v).is_ok();
        if !fits_u64 {
            return Err(PyOverflowError::new_err(
                "Python int too large to convert to C long",
            ));
        }
    }
    check_int_bounds(v, target)
}

/// Construct a float-kind `target` `Buffer` from `obj`. Narrowing never
/// raises, so no weak/strong distinction is needed -- CPython's own
/// `float(obj)` already handles bool/int/float/str/numpy-or-ionp-scalar
/// sources correctly (numpy scalars implement `__float__`) and raises the
/// right `TypeError` for a complex source (verified: `np.float32(3+4j)`
/// raises `TypeError`, and Python's own `float()` on a `complex` raises the
/// identical error for the identical reason -- no `__float__` on `complex`).
fn construct_float_buffer(obj: &Bound<'_, PyAny>, target: DType) -> PyResult<Buffer> {
    if obj.is_none() {
        // numpy coerces `None` to NaN for float-kind scalar constructors
        // (verified: `np.float16(None)`, `np.float32(None)`,
        // `np.float64(None)` all -> nan). NOTE: this rule does NOT extend
        // to integer/bool types, which still raise `TypeError` for `None`
        // -- deliberately not touched here.
        return Ok(float_buffer_from_f64(f64::NAN, target));
    }
    let py = obj.py();
    let f: f64 = py
        .import("builtins")?
        .call_method1("float", (obj,))?
        .extract()?;
    Ok(float_buffer_from_f64(f, target))
}

/// Construct a complex-kind `target` `Buffer` from `obj` via CPython's own
/// `complex(obj)` -- covers bool/int/float/complex/str/numpy-or-ionp-scalar
/// sources uniformly (numpy scalars implement `__complex__`; string parsing
/// verified against numpy's own accepted/rejected forms, e.g. `"1+2j"` and
/// `"(1+2j)"` both parse, `"abc"` raises `ValueError`).
fn construct_complex_buffer(obj: &Bound<'_, PyAny>, target: DType) -> PyResult<Buffer> {
    if obj.is_none() {
        // numpy coerces `None` to `nan+nanj` for complex-kind scalar
        // constructors (verified: `np.complex64(None)`, `np.complex128(None)`
        // both -> `nan+nanj`), matching the float-kind rule above.
        return Ok(complex_buffer_from_f64_pair(f64::NAN, f64::NAN, target));
    }
    let py = obj.py();
    // numpy's `bytes` handling for complex scalars decodes UTF-8 to `str`
    // first, then defers to `complex(str)` -- the "malformed string"
    // `ValueError` text is `complex(str)`'s own pre-existing message for
    // garbage input (verified identical: `complex('abc')` raises the same
    // text), not something numpy invents. Invalid UTF-8 bytes raise
    // `UnicodeDecodeError`, matching numpy's own decode-first behavior.
    let target_obj: Bound<'_, PyAny> = if obj.is_instance_of::<PyBytes>() {
        obj.call_method1("decode", ("utf-8",))?
    } else {
        obj.clone()
    };
    let c = py
        .import("builtins")?
        .call_method1("complex", (target_obj,))?;
    let c = c
        .cast::<pyo3::types::PyComplex>()
        .map_err(|_| PyTypeError::new_err("complex() did not return a complex object"))?;
    Ok(complex_buffer_from_f64_pair(c.real(), c.imag(), target))
}

/// `bool_`'s constructor is Python's own generic truthiness protocol for
/// any non-sequence input (verified against every numpy `np.bool_(x)`
/// result observed: `None` -> `False`, `""` -> `False`, non-empty
/// str/dict/`object()` -> `True`, numeric values via their own `__bool__`).
fn construct_bool_value(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    reject_sequence(obj, "bool_")?;
    obj.is_truthy()
}

// ---------------------------------------------------------------------
// Array-like conversion: `anionpy.<dtype>(obj)` for list/tuple/ndarray `obj`
// builds an *array*, exactly like `np.<dtype>(obj)` does (verified:
// `type(np.int8([1, 2]))` is `numpy.ndarray`, not `numpy.int8`). This is
// wired up via a post-hoc `__new__` monkeypatch in `register()` below (see
// `make_new_override`) rather than the `#[new]` functions above, which stay
// unmodified and are reused directly for the scalar branch.
// ---------------------------------------------------------------------

/// True for exactly the inputs numpy's scalar constructors treat as
/// "array-like" (build an array instead of a scalar): a Python list/tuple,
/// or any object with `.ndim >= 1` (a real numpy ndarray, an anionpy ndarray,
/// or anything else array-like exposing `.ndim`). A 0-d array, bare Python
/// scalar, numpy/anionpy scalar, or string is NOT array-like -- those still
/// produce a scalar (verified: `np.int8(np.array(3))` is a scalar, not a
/// shape-`()` array; `np.int8("3")` is a scalar).
fn is_array_like(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        return Ok(true);
    }
    if obj.is_instance_of::<pyo3::types::PyString>() {
        return Ok(false);
    }
    if let Ok(nd_obj) = obj.getattr("ndim") {
        if let Ok(n) = nd_obj.extract::<i64>() {
            return Ok(n >= 1);
        }
    }
    Ok(false)
}

/// Recursively flatten a (possibly nested, possibly ragged-checked-elsewhere)
/// list/tuple of bool-ish leaves via plain Python truthiness -- matches
/// `np.bool_([...])`'s elementwise behavior exactly (verified: NaN -> `True`,
/// `""` -> `False`, non-empty str/obj -> `True`).
fn flatten_weak_bool(obj: &Bound<'_, PyAny>, out: &mut Vec<bool>) -> PyResult<()> {
    if let Ok(list) = obj.cast::<PyList>() {
        for item in list.iter() {
            flatten_weak_bool(&item, out)?;
        }
        return Ok(());
    }
    if let Ok(tup) = obj.cast::<PyTuple>() {
        for item in tup.iter() {
            flatten_weak_bool(&item, out)?;
        }
        return Ok(());
    }
    out.push(obj.is_truthy()?);
    Ok(())
}

/// Recursively flatten a list/tuple of int-ish leaves for an integer-kind
/// `target`, bounds-checking EVERY leaf via the weak-scalar rule
/// (`check_int_bounds`) regardless of whether the leaf itself is a "strong"
/// numpy/anionpy scalar -- verified this is uniform in list context, unlike
/// bare scalar construction (`np.int8([np.int32(300)])` still raises
/// `OverflowError`, where `np.int8(np.int32(300))` alone wraps to `44`).
/// Python's own `int(x)` on a float leaf truncates toward zero first (then
/// the truncated value is bounds-checked), exactly matching numpy's observed
/// list-conversion behavior (`np.int8([127.9])` -> `[127]`,
/// `np.int8([128.0])` -> `OverflowError`, `np.int8([nan])` -> `ValueError:
/// cannot convert float NaN to integer` -- CPython's own `int()` message,
/// which is exactly what numpy raises here too).
fn flatten_weak_int(obj: &Bound<'_, PyAny>, out: &mut Vec<i128>, target: DType) -> PyResult<()> {
    if let Ok(list) = obj.cast::<PyList>() {
        for item in list.iter() {
            flatten_weak_int(&item, out, target)?;
        }
        return Ok(());
    }
    if let Ok(tup) = obj.cast::<PyTuple>() {
        for item in tup.iter() {
            flatten_weak_int(&item, out, target)?;
        }
        return Ok(());
    }
    let py = obj.py();
    let as_int = py.import("builtins")?.call_method1("int", (obj,))?;
    let v: i128 = as_int.extract()?;
    check_int_bounds(v, target)?;
    out.push(v);
    Ok(())
}

/// Recursively flatten a list/tuple of float-ish leaves via CPython's own
/// `float(x)` -- narrowing to a float target never raises, so no weak/strong
/// distinction is needed here (mirrors `construct_float_buffer`'s reasoning).
fn flatten_weak_float(obj: &Bound<'_, PyAny>, out: &mut Vec<f64>) -> PyResult<()> {
    if let Ok(list) = obj.cast::<PyList>() {
        for item in list.iter() {
            flatten_weak_float(&item, out)?;
        }
        return Ok(());
    }
    if let Ok(tup) = obj.cast::<PyTuple>() {
        for item in tup.iter() {
            flatten_weak_float(&item, out)?;
        }
        return Ok(());
    }
    let py = obj.py();
    let f: f64 = py.import("builtins")?.call_method1("float", (obj,))?.extract()?;
    out.push(f);
    Ok(())
}

/// Recursively flatten a list/tuple of complex-ish leaves via CPython's own
/// `complex(x)` -- mirrors `construct_complex_buffer`'s reasoning.
fn flatten_weak_complex(obj: &Bound<'_, PyAny>, out: &mut Vec<(f64, f64)>) -> PyResult<()> {
    if let Ok(list) = obj.cast::<PyList>() {
        for item in list.iter() {
            flatten_weak_complex(&item, out)?;
        }
        return Ok(());
    }
    if let Ok(tup) = obj.cast::<PyTuple>() {
        for item in tup.iter() {
            flatten_weak_complex(&item, out)?;
        }
        return Ok(());
    }
    let py = obj.py();
    let c = py.import("builtins")?.call_method1("complex", (obj,))?;
    let c = c
        .cast::<PyComplex>()
        .map_err(|_| PyTypeError::new_err("complex() did not return a complex object"))?;
    out.push((c.real(), c.imag()));
    Ok(())
}

fn int_vec_to_buffer(vals: Vec<i128>, target: DType) -> Buffer {
    match target {
        DType::Bool => Buffer::Bool(vals.iter().map(|&v| v != 0).collect()),
        DType::I8 => Buffer::I8(vals.iter().map(|&v| v as i8).collect()),
        DType::I16 => Buffer::I16(vals.iter().map(|&v| v as i16).collect()),
        DType::I32 => Buffer::I32(vals.iter().map(|&v| v as i32).collect()),
        DType::I64 => Buffer::I64(vals.iter().map(|&v| v as i64).collect()),
        DType::U8 => Buffer::U8(vals.iter().map(|&v| v as u8).collect()),
        DType::U16 => Buffer::U16(vals.iter().map(|&v| v as u16).collect()),
        DType::U32 => Buffer::U32(vals.iter().map(|&v| v as u32).collect()),
        DType::U64 => Buffer::U64(vals.iter().map(|&v| v as u64).collect()),
        other => unreachable!("int_vec_to_buffer called with non-integer target {other}"),
    }
}

fn float_vec_to_buffer(vals: Vec<f64>, target: DType) -> Buffer {
    match target {
        DType::F16 => Buffer::F16(vals.iter().map(|&v| half::f16::from_f64(v)).collect()),
        DType::F32 => Buffer::F32(vals.iter().map(|&v| v as f32).collect()),
        DType::F64 => Buffer::F64(vals),
        other => unreachable!("float_vec_to_buffer called with non-float target {other}"),
    }
}

fn complex_vec_to_buffer(vals: Vec<(f64, f64)>, target: DType) -> Buffer {
    match target {
        DType::C64 => Buffer::C64(vals.iter().map(|&(re, im)| C64::new(re as f32, im as f32)).collect()),
        DType::C128 => Buffer::C128(vals.iter().map(|&(re, im)| C128::new(re, im)).collect()),
        other => unreachable!("complex_vec_to_buffer called with non-complex target {other}"),
    }
}

/// Dispatch a list/tuple `obj` to the right `flatten_weak_*` helper for
/// `target`'s kind and pack the result into a `Buffer`.
fn flatten_weak(obj: &Bound<'_, PyAny>, target: DType) -> PyResult<Buffer> {
    if target.is_bool() {
        let mut out = Vec::new();
        flatten_weak_bool(obj, &mut out)?;
        return Ok(Buffer::Bool(out));
    }
    if target.is_floating() {
        let mut out = Vec::new();
        flatten_weak_float(obj, &mut out)?;
        return Ok(float_vec_to_buffer(out, target));
    }
    if target.is_complex() {
        let mut out = Vec::new();
        flatten_weak_complex(obj, &mut out)?;
        return Ok(complex_vec_to_buffer(out, target));
    }
    let mut out = Vec::new();
    flatten_weak_int(obj, &mut out, target)?;
    Ok(int_vec_to_buffer(out, target))
}

/// Build the `NdArray` that `anionpy.<dtype>(obj)` returns for array-like
/// `obj` (list/tuple, or `.ndim >= 1`) -- exactly `np.array(obj,
/// dtype='<dtype>')`'s behavior. List/tuple sources are bounds-checked per
/// leaf (the weak-scalar rule); ndarray sources (anionpy's own, or a real
/// numpy.ndarray) are unsafe-cast via `NdArray::cast_to` (numpy's own
/// `astype`-style wrap, verified: `np.int8(np.array([300], dtype=np.int32))`
/// -> `array([44], dtype=int8)`).
fn build_array_for_dtype(obj: &Bound<'_, PyAny>, target: DType) -> PyResult<NdArray> {
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        if let Err(msg) = check_ragged(obj)? {
            return Err(PyValueError::new_err(msg));
        }
        let shape = shape_of_nested(obj)?;
        let buffer = flatten_weak(obj, target)?;
        return NdArray::from_buffer(buffer, shape, Order::C).map_err(to_py_err);
    }
    if let Ok(arr) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(arr.inner.cast_to(target));
    }
    let src = ndarray_from_numpy(obj)?;
    Ok(src.cast_to(target))
}

/// Build the `Py<PyAny>`-returning `__new__` replacement for one concrete
/// scalar leaf class: array-like input goes through `build_array_for_dtype`
/// (returning an `anionpy.ndarray`); everything else falls through to the
/// leaf's own unmodified `scalar_new` closure (its original `#[new]`
/// associated function), preserving scalar-in/scalar-out exactly.
fn make_new_override<'py, F>(
    py: Python<'py>,
    name: &'static str,
    target: DType,
    scalar_new: F,
) -> PyResult<Bound<'py, PyCFunction>>
where
    F: Fn(Python<'_>, Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> + Send + Sync + 'static,
{
    // Leaked once per (of only 14) leaf class at module-import time --
    // negligible, and `PyCFunction::new_closure` requires `&'static CStr`.
    let cname: &'static std::ffi::CStr =
        Box::leak(std::ffi::CString::new(name).unwrap().into_boxed_c_str());
    PyCFunction::new_closure(
        py,
        Some(cname),
        None,
        move |args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>| -> PyResult<Py<PyAny>> {
            let py = args.py();
            // args[0] is `cls` (this is a `__new__` slot); the actual value
            // (if any) is args[1] or the `value=` keyword.
            let value: Option<Bound<'_, PyAny>> = if args.len() > 1 {
                Some(args.get_item(1)?)
            } else if let Some(kw) = kwargs {
                kw.get_item("value")?
            } else {
                None
            };
            match &value {
                Some(v) if is_array_like(v)? => {
                    let arr = build_array_for_dtype(v, target)?;
                    Ok(Py::new(py, PyArray { inner: arr })?.into_any())
                }
                _ => scalar_new(py, value.as_ref()),
            }
        },
    )
}

// ---------------------------------------------------------------------
// Abstract, non-instantiable base classes. Each `#[new]` always errors, so
// the `PyClassInitializer<Self>` it type-checks against is never actually
// built on that path -- only used by concrete leaf classes below, which
// chain real values through `PyClassInitializer::from(Generic).add_subclass(..)`.
// ---------------------------------------------------------------------

#[pyclass(name = "generic", module = "anionpy", subclass)]
pub struct Generic;
#[pymethods]
impl Generic {
    #[new]
    fn new() -> PyResult<PyClassInitializer<Generic>> {
        Err(PyTypeError::new_err("cannot create 'anionpy.generic' instances"))
    }
}

#[pyclass(name = "number", module = "anionpy", extends = Generic, subclass)]
pub struct Number;
#[pymethods]
impl Number {
    #[new]
    fn new() -> PyResult<PyClassInitializer<Number>> {
        Err(PyTypeError::new_err("cannot create 'anionpy.number' instances"))
    }
}

#[pyclass(name = "integer", module = "anionpy", extends = Number, subclass)]
pub struct Integer;
#[pymethods]
impl Integer {
    #[new]
    fn new() -> PyResult<PyClassInitializer<Integer>> {
        Err(PyTypeError::new_err("cannot create 'anionpy.integer' instances"))
    }
}

#[pyclass(name = "signedinteger", module = "anionpy", extends = Integer, subclass)]
pub struct SignedInteger;
#[pymethods]
impl SignedInteger {
    #[new]
    fn new() -> PyResult<PyClassInitializer<SignedInteger>> {
        Err(PyTypeError::new_err(
            "cannot create 'anionpy.signedinteger' instances",
        ))
    }
}

#[pyclass(name = "unsignedinteger", module = "anionpy", extends = Integer, subclass)]
pub struct UnsignedInteger;
#[pymethods]
impl UnsignedInteger {
    #[new]
    fn new() -> PyResult<PyClassInitializer<UnsignedInteger>> {
        Err(PyTypeError::new_err(
            "cannot create 'anionpy.unsignedinteger' instances",
        ))
    }
}

#[pyclass(name = "inexact", module = "anionpy", extends = Number, subclass)]
pub struct Inexact;
#[pymethods]
impl Inexact {
    #[new]
    fn new() -> PyResult<PyClassInitializer<Inexact>> {
        Err(PyTypeError::new_err("cannot create 'anionpy.inexact' instances"))
    }
}

#[pyclass(name = "floating", module = "anionpy", extends = Inexact, subclass)]
pub struct Floating;
#[pymethods]
impl Floating {
    #[new]
    fn new() -> PyResult<PyClassInitializer<Floating>> {
        Err(PyTypeError::new_err("cannot create 'anionpy.floating' instances"))
    }
}

#[pyclass(name = "complexfloating", module = "anionpy", extends = Inexact, subclass)]
pub struct ComplexFloating;
#[pymethods]
impl ComplexFloating {
    #[new]
    fn new() -> PyResult<PyClassInitializer<ComplexFloating>> {
        Err(PyTypeError::new_err(
            "cannot create 'anionpy.complexfloating' instances",
        ))
    }
}

#[pyclass(name = "flexible", module = "anionpy", extends = Generic, subclass)]
pub struct Flexible;
#[pymethods]
impl Flexible {
    #[new]
    fn new() -> PyResult<PyClassInitializer<Flexible>> {
        Err(PyTypeError::new_err("cannot create 'anionpy.flexible' instances"))
    }
}

#[pyclass(name = "character", module = "anionpy", extends = Flexible, subclass)]
pub struct Character;
#[pymethods]
impl Character {
    #[new]
    fn new() -> PyResult<PyClassInitializer<Character>> {
        Err(PyTypeError::new_err("cannot create 'anionpy.character' instances"))
    }
}

// ---------------------------------------------------------------------
// bool_ -- extends `generic` directly (verified MRO: bool -> generic ->
// object, NOT via integer -- numpy deliberately does not treat bool_ as a
// number-family type).
// ---------------------------------------------------------------------

// numpy 2.0+ renamed the underlying scalar-bool class's `__name__` to
// `"bool"` (verified live: `type(np.True_).__name__ == 'bool'`, and
// `np.bool is np.bool_` -- same object exposed under two module attribute
// names). anionpy mirrors this: the pyclass itself is named "bool" so
// `type(anionpy.True_).__name__ == 'bool'` matches, and `register()` below
// exposes it under BOTH `anionpy.bool` and `anionpy.bool_` (same object, like
// numpy).
#[pyclass(name = "bool", module = "anionpy", extends = Generic)]
pub struct Bool_ {
    pub(crate) value: bool,
}

impl Bool_ {
    pub(crate) fn init_chain(value: bool) -> PyClassInitializer<Bool_> {
        PyClassInitializer::from(Generic).add_subclass(Bool_ { value })
    }
}

#[pymethods]
impl Bool_ {
    #[new]
    #[pyo3(signature = (value=None))]
    fn new(value: Option<&Bound<'_, PyAny>>) -> PyResult<PyClassInitializer<Bool_>> {
        let v = match value {
            None => false,
            Some(obj) => construct_bool_value(obj)?,
        };
        Ok(Bool_::init_chain(v))
    }

    #[classattr]
    fn __ionp_dtype_name__() -> &'static str {
        "bool"
    }

    #[getter]
    fn dtype(&self) -> PyDType {
        PyDType { inner: DType::Bool, spelling: None }
    }
    #[getter]
    fn itemsize(&self) -> usize {
        DType::Bool.itemsize()
    }
    fn __repr__(&self) -> String {
        // numpy special-cases bool_'s repr to the bare `np.True_`/`np.False_`
        // spelling rather than `np.bool_(True)` -- verified. anionpy mirrors the
        // same special-casing under its own module name.
        if self.value {
            "anionpy.True_".to_string()
        } else {
            "anionpy.False_".to_string()
        }
    }
    fn __str__(&self) -> String {
        if self.value {
            "True".to_string()
        } else {
            "False".to_string()
        }
    }
    fn __bool__(&self) -> bool {
        self.value
    }
    // These four numeric-conversion dunders are what let a "strong" anionpy
    // scalar act as a valid input to ANOTHER anionpy scalar constructor (e.g.
    // `anionpy.int8(anionpy.bool_(True))`), exactly the way real numpy's scalars
    // implement them for the same reason -- `construct_int_buffer`/
    // `construct_float_buffer`/`construct_complex_buffer` above call
    // Python's own `int()`/`float()`/`complex()` builtins on the source
    // object, which only succeed if the source implements the matching
    // dunder.
    fn __int__(&self) -> i64 {
        self.value as i64
    }
    fn __index__(&self) -> i64 {
        self.value as i64
    }
    fn __float__(&self) -> f64 {
        if self.value {
            1.0
        } else {
            0.0
        }
    }
    fn __complex__<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyComplex> {
        let v = if self.value { 1.0 } else { 0.0 };
        pyo3::types::PyComplex::from_doubles(py, v, 0.0)
    }

    fn __eq__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        scalar_richcmp(py, BinaryOp::Equal, Buffer::Bool(vec![self.value]), other)
    }
    fn __ne__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        scalar_richcmp(py, BinaryOp::NotEqual, Buffer::Bool(vec![self.value]), other)
    }
    fn __lt__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        scalar_richcmp(py, BinaryOp::Less, Buffer::Bool(vec![self.value]), other)
    }
    fn __le__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        scalar_richcmp(py, BinaryOp::LessEqual, Buffer::Bool(vec![self.value]), other)
    }
    fn __gt__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        scalar_richcmp(py, BinaryOp::Greater, Buffer::Bool(vec![self.value]), other)
    }
    fn __ge__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        scalar_richcmp(py, BinaryOp::GreaterEqual, Buffer::Bool(vec![self.value]), other)
    }
    fn __hash__(&self) -> isize {
        // `hash(False) == 0`, `hash(True) == 1` in CPython, and numpy's
        // `bool_` matches exactly (verified: `hash(np.True_) == hash(True)
        // == 1`); no NaN case exists for bool.
        self.value as isize
    }

    // ---- Arithmetic -------------------------------------------------
    // All binary ops go through the shared `ionp_core::ufunc` engine via
    // `scalar_binop`/`scalar_math_binop`/`scalar_truediv`/`scalar_divmod`
    // (see their doc comments above) so overflow-wrap, NEP 50 promotion,
    // div/mod-by-zero, and error text all come from the SAME
    // already-verified engine `anionpy.ndarray` uses. `bool_ - bool_` and
    // `-bool_(True)` raise numpy's own exact messages ("numpy boolean
    // subtract..."/"The numpy boolean negative...") for free, since
    // `unary_op`/`binary_op` already special-case `DType::Bool` for
    // those two ops identically for arrays and scalars alike.
    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::Add, &Buffer::Bool(vec![self.value]), DType::Bool, other, false)
    }
    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::Add, &Buffer::Bool(vec![self.value]), DType::Bool, other, true)
    }
    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::Subtract, &Buffer::Bool(vec![self.value]), DType::Bool, other, false)
    }
    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::Subtract, &Buffer::Bool(vec![self.value]), DType::Bool, other, true)
    }
    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::Multiply, &Buffer::Bool(vec![self.value]), DType::Bool, other, false)
    }
    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::Multiply, &Buffer::Bool(vec![self.value]), DType::Bool, other, true)
    }
    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_truediv(py, &Buffer::Bool(vec![self.value]), DType::Bool, other, false)
    }
    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_truediv(py, &Buffer::Bool(vec![self.value]), DType::Bool, other, true)
    }
    fn __floordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::FloorDivide, &Buffer::Bool(vec![self.value]), DType::Bool, other, false)
    }
    fn __rfloordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::FloorDivide, &Buffer::Bool(vec![self.value]), DType::Bool, other, true)
    }
    fn __mod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_math_binop(py, MathBinaryOp::Remainder, &Buffer::Bool(vec![self.value]), DType::Bool, other, false)
    }
    fn __rmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_math_binop(py, MathBinaryOp::Remainder, &Buffer::Bool(vec![self.value]), DType::Bool, other, true)
    }
    fn __divmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_divmod(py, &Buffer::Bool(vec![self.value]), DType::Bool, other, false)
    }
    fn __rdivmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_divmod(py, &Buffer::Bool(vec![self.value]), DType::Bool, other, true)
    }
    fn __pow__(&self, py: Python<'_>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        if modulo.is_some() {
            return Ok(py.NotImplemented());
        }
        scalar_math_binop(py, MathBinaryOp::Power, &Buffer::Bool(vec![self.value]), DType::Bool, other, false)
    }
    fn __rpow__(&self, py: Python<'_>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        if modulo.is_some() {
            return Ok(py.NotImplemented());
        }
        scalar_math_binop(py, MathBinaryOp::Power, &Buffer::Bool(vec![self.value]), DType::Bool, other, true)
    }
    fn __lshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::LeftShift, &Buffer::Bool(vec![self.value]), DType::Bool, other, false)
    }
    fn __rlshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::LeftShift, &Buffer::Bool(vec![self.value]), DType::Bool, other, true)
    }
    fn __rshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::RightShift, &Buffer::Bool(vec![self.value]), DType::Bool, other, false)
    }
    fn __rrshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::RightShift, &Buffer::Bool(vec![self.value]), DType::Bool, other, true)
    }
    fn __and__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::BitwiseAnd, &Buffer::Bool(vec![self.value]), DType::Bool, other, false)
    }
    fn __rand__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::BitwiseAnd, &Buffer::Bool(vec![self.value]), DType::Bool, other, true)
    }
    fn __or__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::BitwiseOr, &Buffer::Bool(vec![self.value]), DType::Bool, other, false)
    }
    fn __ror__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::BitwiseOr, &Buffer::Bool(vec![self.value]), DType::Bool, other, true)
    }
    fn __xor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::BitwiseXor, &Buffer::Bool(vec![self.value]), DType::Bool, other, false)
    }
    fn __rxor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        scalar_binop(py, BinaryOp::BitwiseXor, &Buffer::Bool(vec![self.value]), DType::Bool, other, true)
    }
    fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let a = NdArray::from_buffer(Buffer::Bool(vec![self.value]), vec![], Order::C).map_err(to_py_err)?;
        let out = ionp_core::ufunc::unary_op(ionp_core::ufunc::UnaryOp::Negative, &a).map_err(to_py_err)?;
        wrap_arith_result(py, out)
    }
    fn __pos__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // numpy's `positive` ufunc has no `bool` loop at all (verified
        // against real numpy 2.5.1: `+np.bool_(True)` raises
        // `UFuncTypeError`) -- see the identical rejection in
        // `PyArray::__pos__` for the exact message text this mirrors.
        let _ = py;
        Err(no_ufunc_loop_err_dtypes(
            "positive",
            &[Some(DType::Bool), None],
            "ufunc 'positive' did not contain a loop with signature matching types \
             <class 'numpy.dtypes.BoolDType'> -> None",
        ))
    }
    fn __abs__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let a = NdArray::from_buffer(Buffer::Bool(vec![self.value]), vec![], Order::C).map_err(to_py_err)?;
        let out = ionp_core::ufunc::unary_op(ionp_core::ufunc::UnaryOp::Absolute, &a).map_err(to_py_err)?;
        wrap_arith_result(py, out)
    }
    fn __invert__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let a = NdArray::from_buffer(Buffer::Bool(vec![self.value]), vec![], Order::C).map_err(to_py_err)?;
        let out = ionp_core::ufunc::unary_op(ionp_core::ufunc::UnaryOp::Invert, &a).map_err(to_py_err)?;
        wrap_arith_result(py, out)
    }

    // ---- PART 1: 0-d array surface ---------------------------------
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::empty(py)
    }
    #[getter]
    fn ndim(&self) -> usize {
        0
    }
    #[getter]
    fn size(&self) -> usize {
        1
    }
    #[getter]
    fn nbytes(&self) -> usize {
        DType::Bool.itemsize()
    }
    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::empty(py)
    }
    #[getter]
    fn base(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }
    #[getter]
    fn flags(&self) -> ScalarFlags {
        ScalarFlags
    }
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let bytes = PyBytes::new(py, &[self.value as u8]);
        py.import("builtins")?.call_method1("memoryview", (bytes,))
    }
    #[getter]
    fn T(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
        Py::new(py, Bool_::init_chain(self.value))
    }
    #[getter]
    fn real(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
        Py::new(py, Bool_::init_chain(self.value))
    }
    #[getter]
    fn imag(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
        Py::new(py, Bool_::init_chain(false))
    }
    fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.value.into_pyobject(py)?.to_owned().into_any().unbind())
    }
    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.item(py)
    }
    fn copy(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
        Py::new(py, Bool_::init_chain(self.value))
    }
    fn conjugate(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
        Py::new(py, Bool_::init_chain(self.value))
    }
    fn fill(&self, obj: &Bound<'_, PyAny>) -> PyResult<()> {
        construct_bool_value(obj)?;
        Ok(())
    }
    fn all(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
        Py::new(py, Bool_::init_chain(self.value))
    }
    fn any(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
        self.all(py)
    }
    fn min(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
        Py::new(py, Bool_::init_chain(self.value))
    }
    fn max(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
        Py::new(py, Bool_::init_chain(self.value))
    }
    // numpy: `bool_.sum()` widens to `int64` (verified:
    // `type(np.True_.sum()) is np.int64`), unlike `.all()`/`.any()` which
    // stay `bool_`.
    fn sum(&self, py: Python<'_>) -> PyResult<Py<Int64>> {
        Py::new(py, Int64::init_chain(self.value as i64))
    }
    fn prod(&self, py: Python<'_>) -> PyResult<Py<Int64>> {
        self.sum(py)
    }
    fn mean(&self, py: Python<'_>) -> PyResult<Py<Float64>> {
        Py::new(py, Float64::init_chain(if self.value { 1.0 } else { 0.0 }))
    }
    // numpy's `bool_.round()` is a genuinely bizarre special case (verified
    // live): called with NO args, it upcasts to `np.float16(1.0 or 0.0)`;
    // called with an EXPLICIT `ndigits` (even `0`), it raises numpy's own
    // `UFuncTypeError` (a `TypeError` subclass) with message "Cannot cast
    // ufunc 'multiply' output from dtype('float64') to dtype('bool') with
    // casting rule 'same_kind'". A concurrent agent's ufunc-casting work
    // landed a matching `UFuncTypeError` exception class in `lib.rs`
    // (`pyo3::create_exception!(_anionpy, UFuncTypeError, PyTypeError)`,
    // registered into the module); reused here instead of the previous
    // plain-`TypeError` compromise, so `type(e).__name__` now matches
    // numpy's exactly.
    #[pyo3(signature = (ndigits=None))]
    fn round(&self, py: Python<'_>, ndigits: Option<i32>) -> PyResult<Py<PyAny>> {
        match ndigits {
            None => {
                let v = if self.value { 1.0 } else { 0.0 };
                Ok(Py::new(py, Float16::init_chain(half::f16::from_f64(v)))?.into_any())
            }
            Some(_) => Err(UFuncTypeError::new_err(
                "Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('bool') with casting rule 'same_kind'",
            )),
        }
    }
    #[pyo3(signature = (dtype, order=None, casting=None, subok=None, copy=None))]
    fn astype(
        &self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        order: Option<&str>,
        casting: Option<&str>,
        subok: Option<&Bound<'_, PyAny>>,
        copy: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if let Some(v) = subok.as_ref() {
            scalar_index_like_check(v)?;
        }
        if let Some(v) = copy.as_ref() {
            scalar_index_like_check(v)?;
        }
        let _ = (order, casting, subok, copy);
        // A scalar source is never S/U (no scalar wrapper type exists for
        // it, see `scalar_from_buffer`'s own guard) -- decline an S/U TARGET
        // here too, before `cast_to` panics on it.
        let target = crate::dtype_from_pyobj_no_su(dtype)?;
        let buf = Buffer::Bool(vec![self.value]).cast_to(target);
        scalar_from_buffer(py, &buf)
    }
    #[pyo3(signature = (*shape, order=None))]
    fn reshape(
        &self,
        py: Python<'_>,
        shape: &Bound<'_, PyTuple>,
        order: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        scalar_reshape(py, shape, order, Buffer::Bool(vec![self.value]))
    }
    // Delegates to `PyArray::__array___no_cast` (made `pub(crate)` for this
    // reuse) instead of returning anionpy's own `PyArray` directly: numpy's
    // `np.asarray()`/`np.array()` call `__array__()` expecting a REAL
    // `numpy.ndarray` back, and reject anything else with `ValueError:
    // object __array__ method not producing an array` -- confirmed live,
    // this was failing on all 14 scalar types before this fix. `PyArray`'s
    // own `__array__` already builds a genuine, K-order/strides-correct
    // numpy array via the `numpy` crate; reused rather than duplicated.
    fn __array__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let arr = NdArray::from_buffer(Buffer::Bool(vec![self.value]), vec![], Order::C)
            .map_err(to_py_err)?;
        PyArray { inner: arr }.__array___no_cast(py)
    }
    // PART 5 (coordinator audit) + axis-argument follow-up: squeeze/
    // transpose are the 0-d identity (a new same-value/same-type scalar;
    // live-verified NOT the same object, `np.int8(5).squeeze() is not
    // np.int8(5)`, but `==`-equal) for every axis spelling numpy itself
    // accepts. See the `squeeze_axis_ok`/`transpose_axes_ok`/
    // `transpose_no_kwargs_err` doc comments (top of this file) for the
    // full, live-verified, deliberately-inconsistent axis rule set this
    // reproduces exactly.
    #[pyo3(signature = (*args, **kwargs))]
    fn squeeze(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<Bool_>> {
        let axis = resolve_squeeze_axis(args, kwargs)?;
        squeeze_axis_ok(axis.as_ref())?;
        Py::new(py, Bool_::init_chain(self.value))
    }
    #[pyo3(signature = (*axes, **kwargs))]
    fn transpose(
        &self,
        py: Python<'_>,
        axes: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<Bool_>> {
        if kwargs.is_some_and(|d| !d.is_empty()) {
            return Err(transpose_no_kwargs_err());
        }
        transpose_axes_ok(axes)?;
        Py::new(py, Bool_::init_chain(self.value))
    }
    // A single byte has no byte-order to swap; live-verified
    // `np.bool_(True).byteswap() == np.bool_(True)` (no-op). `inplace=True`
    // raises numpy's exact message (live-verified on every dtype:
    // `ValueError: cannot byteswap a scalar in-place`).
    #[pyo3(signature = (inplace=None))]
    fn byteswap(&self, py: Python<'_>, inplace: Option<&Bound<'_, PyAny>>) -> PyResult<Py<Bool_>> {
        if inplace.map(|v| v.is_truthy()).transpose()?.unwrap_or(false) {
            return Err(PyValueError::new_err("cannot byteswap a scalar in-place"));
        }
        Py::new(py, Bool_::init_chain(self.value))
    }
    // ravel/flatten: numpy always returns a shape-`(1,)` array (NOT another
    // 0-d scalar) for a scalar's `.ravel()`/`.flatten()` -- live-verified
    // `np.int8(5).ravel()` -> `array([5], dtype=int8)`. `order=` is accepted
    // and ignored: a single-element buffer is order-invariant.
    #[pyo3(signature = (order=None))]
    fn ravel(&self, py: Python<'_>, order: Option<&str>) -> PyResult<Py<PyArray>> {
        let _ = order;
        let arr = NdArray::from_buffer(Buffer::Bool(vec![self.value]), vec![1], Order::C)
            .map_err(to_py_err)?;
        Py::new(py, PyArray { inner: arr })
    }
    #[pyo3(signature = (order=None))]
    fn flatten(&self, py: Python<'_>, order: Option<&str>) -> PyResult<Py<PyArray>> {
        self.ravel(py, order)
    }
    #[getter]
    fn flat(&self, py: Python<'_>) -> PyResult<Py<ScalarFlatIter>> {
        let v = Py::new(py, Bool_::init_chain(self.value))?.into_any();
        Py::new(py, ScalarFlatIter::new(v))
    }
    // numpy's 0-d `.nonzero()` always raises; `bool_` alone gets a LONGER
    // message than every other dtype -- live-verified difference, not
    // assumed uniform (see module-level `NONZERO_MSG_SHORT`/`_LONG`).
    fn nonzero(&self) -> PyResult<()> {
        Err(PyValueError::new_err(NONZERO_MSG_LONG))
    }
}

// ---------------------------------------------------------------------
// Concrete signed/unsigned integer leaves.
// ---------------------------------------------------------------------

macro_rules! int_leaf {
    ($rust:ident, $pyname:literal, $prim:ty, $parent:ty, $dtype:expr, $buffer_variant:ident, $dtypename:literal) => {
        #[pyclass(name = $pyname, module = "anionpy", extends = $parent)]
        pub struct $rust {
            pub(crate) value: $prim,
        }

        impl $rust {
            pub(crate) fn init_chain(value: $prim) -> PyClassInitializer<$rust> {
                PyClassInitializer::from(Generic)
                    .add_subclass(Number)
                    .add_subclass(Integer)
                    .add_subclass(<$parent>::new_marker())
                    .add_subclass($rust { value })
            }
        }

        #[pymethods]
        impl $rust {
            #[new]
            #[pyo3(signature = (value=None))]
            fn new(value: Option<&Bound<'_, PyAny>>) -> PyResult<PyClassInitializer<$rust>> {
                let v: $prim = match value {
                    None => 0 as $prim,
                    Some(obj) => {
                        reject_sequence(obj, $dtypename)?;
                        match construct_int_buffer(obj, $dtype)? {
                            Buffer::$buffer_variant(vals) => vals[0],
                            _ => unreachable!("construct_int_buffer returned the wrong Buffer variant"),
                        }
                    }
                };
                Ok($rust::init_chain(v))
            }

            #[classattr]
            fn __ionp_dtype_name__() -> &'static str {
                $dtypename
            }

            #[getter]
            fn dtype(&self) -> PyDType {
                PyDType { inner: $dtype, spelling: None }
            }
            #[getter]
            fn itemsize(&self) -> usize {
                $dtype.itemsize()
            }
            fn __repr__(&self) -> String {
                format!(concat!("anionpy.", $pyname, "({})"), self.value)
            }
            fn __str__(&self) -> String {
                format!("{}", self.value)
            }
            // See the matching comment on Bool_ above: these let this type
            // act as a "strong" source for ANOTHER anionpy scalar constructor's
            // `builtins.int()`/`builtins.float()`/`builtins.complex()` call.
            // `__index__` alone is enough for CPython's own `complex()`
            // builtin to succeed (it falls back to `__index__` when
            // `__complex__`/`__float__` are absent), so no separate
            // `__complex__` is needed here.
            // `i128`, NOT `i64`. This macro is instantiated for BOTH signed
            // and unsigned widths, so an `as i64` cast silently REINTERPRETS
            // every u64 above i64::MAX as negative:
            //   int(anionpy.uint64(2**64 - 2))  ->  -2        (numpy: 18446744073709551614)
            //   hex(anionpy.uint64(2**64 - 1))  ->  '-0x1'    (numpy: '0xffffffffffffffff')
            // i128 holds every anionpy integer width (u64 max and i64 min alike)
            // exactly, so the conversion is lossless for all 8 instantiations.
            // NOTE this is invisible to any check that compares raw BYTES --
            // the stored value was always correct; only the Python-int
            // CONVERSION was wrong. It takes a probe that actually calls
            // int()/hex()/__index__ to see it.
            fn __int__(&self) -> i128 {
                self.value as i128
            }
            fn __index__(&self) -> i128 {
                self.value as i128
            }
            fn __float__(&self) -> f64 {
                self.value as f64
            }

            fn __eq__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::Equal, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __ne__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::NotEqual, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __lt__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::Less, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __le__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::LessEqual, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __gt__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::Greater, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __ge__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::GreaterEqual, Buffer::$buffer_variant(vec![self.value]), other)
            }
            // Integers have no NaN case, so this is just "build the exact
            // same Python int real numpy's own `int()` on this value would
            // build, and delegate to CPython's own int hash" -- verified
            // bit-exact against `hash(np.<dtype>(v)) == hash(int(v))` across
            // all eight int/uint widths, including values that need CPython's
            // arbitrary-precision int hash reduction (`hash(np.uint64(2**64-1))
            // == hash(2**64-1)`).
            fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
                self.value.into_pyobject(py)?.hash()
            }

            // ---- Arithmetic ---------------------------------------------
            // See the identical block on `Bool_` above for the shared-engine
            // rationale (overflow-wrap, floor-div-toward-negative-infinity,
            // div/mod-by-zero, NEP 50 promotion all come from
            // `ionp_core::ufunc` for free).
            fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Add, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Add, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Subtract, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Subtract, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Multiply, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Multiply, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_truediv(py, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_truediv(py, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __floordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::FloorDivide, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rfloordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::FloorDivide, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __mod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_math_binop(py, MathBinaryOp::Remainder, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_math_binop(py, MathBinaryOp::Remainder, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __divmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_divmod(py, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rdivmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_divmod(py, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __pow__(&self, py: Python<'_>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
                if modulo.is_some() {
                    return Ok(py.NotImplemented());
                }
                scalar_math_binop(py, MathBinaryOp::Power, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rpow__(&self, py: Python<'_>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
                if modulo.is_some() {
                    return Ok(py.NotImplemented());
                }
                scalar_math_binop(py, MathBinaryOp::Power, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __lshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::LeftShift, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rlshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::LeftShift, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __rshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::RightShift, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rrshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::RightShift, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __and__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::BitwiseAnd, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rand__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::BitwiseAnd, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __or__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::BitwiseOr, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __ror__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::BitwiseOr, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __xor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::BitwiseXor, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rxor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::BitwiseXor, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let a = NdArray::from_buffer(Buffer::$buffer_variant(vec![self.value]), vec![], Order::C).map_err(to_py_err)?;
                let out = ionp_core::ufunc::unary_op(ionp_core::ufunc::UnaryOp::Negative, &a).map_err(to_py_err)?;
                wrap_arith_result(py, out)
            }
            fn __pos__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                Ok(Py::new(py, $rust::init_chain(self.value))?.into_any())
            }
            fn __abs__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let a = NdArray::from_buffer(Buffer::$buffer_variant(vec![self.value]), vec![], Order::C).map_err(to_py_err)?;
                let out = ionp_core::ufunc::unary_op(ionp_core::ufunc::UnaryOp::Absolute, &a).map_err(to_py_err)?;
                wrap_arith_result(py, out)
            }
            fn __invert__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let a = NdArray::from_buffer(Buffer::$buffer_variant(vec![self.value]), vec![], Order::C).map_err(to_py_err)?;
                let out = ionp_core::ufunc::unary_op(ionp_core::ufunc::UnaryOp::Invert, &a).map_err(to_py_err)?;
                wrap_arith_result(py, out)
            }

            // ---- PART 1: 0-d array surface -----------------------------
            #[getter]
            fn shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
                PyTuple::empty(py)
            }
            #[getter]
            fn ndim(&self) -> usize {
                0
            }
            #[getter]
            fn size(&self) -> usize {
                1
            }
            #[getter]
            fn nbytes(&self) -> usize {
                $dtype.itemsize()
            }
            #[getter]
            fn strides<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
                PyTuple::empty(py)
            }
            #[getter]
            fn base(&self, py: Python<'_>) -> Py<PyAny> {
                py.None()
            }
            #[getter]
            fn flags(&self) -> ScalarFlags {
                ScalarFlags
            }
            #[getter]
            fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let bytes = PyBytes::new(py, &self.value.to_ne_bytes());
                py.import("builtins")?.call_method1("memoryview", (bytes,))
            }
            #[getter]
            fn T(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            #[getter]
            fn real(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            #[getter]
            fn imag(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(0 as $prim))
            }
            // numpy: int8/16/32/64 `.item()` promotes via a signed `i64`;
            // uint8/16/32/64 promotes via an unsigned `u64` (needed for a
            // lossless round-trip of `2**64-1`; verified:
            // `np.uint64(2**64-1).item() == 2**64-1`).
            fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                if $dtype.is_unsigned_integer() {
                    Ok((self.value as u64).into_pyobject(py)?.into_any().unbind())
                } else {
                    Ok((self.value as i64).into_pyobject(py)?.into_any().unbind())
                }
            }
            fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                self.item(py)
            }
            fn copy(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            fn conjugate(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            fn fill(&self, obj: &Bound<'_, PyAny>) -> PyResult<()> {
                match construct_int_buffer(obj, $dtype)? {
                    Buffer::$buffer_variant(_) => Ok(()),
                    _ => unreachable!("construct_int_buffer returned the wrong Buffer variant"),
                }
            }
            fn all(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
                Py::new(py, Bool_::init_chain(self.value != 0 as $prim))
            }
            fn any(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
                self.all(py)
            }
            fn min(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            fn max(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            // numpy: int8/16/32 and uint8/16/32 `.sum()`/`.prod()` widen to
            // int64/uint64 respectively (matching own signedness); int64/
            // uint64 stay at the same dtype (verified live: `type(np.int8(3
            // ).sum()) is np.int64`, `type(np.int64(3).sum()) is np.int64`).
            fn sum(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                if $dtype.is_unsigned_integer() {
                    Py::new(py, UInt64::init_chain(self.value as u64)).map(|p| p.into_any())
                } else {
                    Py::new(py, Int64::init_chain(self.value as i64)).map(|p| p.into_any())
                }
            }
            fn prod(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                self.sum(py)
            }
            // numpy: int `.mean()` -> `float64` always (verified:
            // `type(np.int8(3).mean()) is np.float64`).
            fn mean(&self, py: Python<'_>) -> PyResult<Py<Float64>> {
                Py::new(py, Float64::init_chain(self.value as f64))
            }
            // numpy: int `.round(ndigits)` is a no-op, same dtype, regardless
            // of `ndigits` (verified: `np.int8(5).round(2)` -> `5`, still
            // `int8`).
            #[pyo3(signature = (ndigits=None))]
            fn round(&self, py: Python<'_>, ndigits: Option<i32>) -> PyResult<Py<$rust>> {
                let _ = ndigits;
                Py::new(py, $rust::init_chain(self.value))
            }
            #[pyo3(signature = (dtype, order=None, casting=None, subok=None, copy=None))]
            fn astype(
                &self,
                py: Python<'_>,
                dtype: &Bound<'_, PyAny>,
                order: Option<&str>,
                casting: Option<&str>,
                subok: Option<&Bound<'_, PyAny>>,
                copy: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                // Both accepted-and-discarded, but must ACCEPT/REJECT the
                // same values real numpy does -- see `scalar_index_like_check`.
                if let Some(v) = subok.as_ref() {
                    scalar_index_like_check(v)?;
                }
                if let Some(v) = copy.as_ref() {
                    scalar_index_like_check(v)?;
                }
                let _ = (order, casting, subok, copy);
                // A scalar source is never S/U (no scalar wrapper type exists for
        // it, see `scalar_from_buffer`'s own guard) -- decline an S/U TARGET
        // here too, before `cast_to` panics on it.
        let target = crate::dtype_from_pyobj_no_su(dtype)?;
                let buf = Buffer::$buffer_variant(vec![self.value]).cast_to(target);
                scalar_from_buffer(py, &buf)
            }
            #[pyo3(signature = (*shape, order=None))]
            fn reshape(
                &self,
                py: Python<'_>,
                shape: &Bound<'_, PyTuple>,
                order: Option<&str>,
            ) -> PyResult<Py<PyAny>> {
                scalar_reshape(py, shape, order, Buffer::$buffer_variant(vec![self.value]))
            }
            // See `Bool_::__array__`'s comment: delegates to `PyArray`'s
            // own, already-correct `__array__` rather than returning
            // anionpy's own array type, which numpy's `np.asarray()` rejects.
            fn __array__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                let arr = NdArray::from_buffer(
                    Buffer::$buffer_variant(vec![self.value]),
                    vec![],
                    Order::C,
                )
                .map_err(to_py_err)?;
                PyArray { inner: arr }.__array___no_cast(py)
            }
            // ---- PART 5 (coordinator audit) + axis-argument follow-up:
            // squeeze/transpose/byteswap/ravel/flatten/flat/nonzero -- see
            // Bool_'s matching methods for the live-verified reference
            // behavior; squeeze/transpose now accept the full axis-argument
            // surface via the shared `squeeze_axis_ok`/`transpose_axes_ok`
            // helpers (top of this file).
            #[pyo3(signature = (*args, **kwargs))]
            fn squeeze(
                &self,
                py: Python<'_>,
                args: &Bound<'_, PyTuple>,
                kwargs: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<Py<$rust>> {
                let axis = resolve_squeeze_axis(args, kwargs)?;
                squeeze_axis_ok(axis.as_ref())?;
                Py::new(py, $rust::init_chain(self.value))
            }
            #[pyo3(signature = (*axes, **kwargs))]
            fn transpose(
                &self,
                py: Python<'_>,
                axes: &Bound<'_, PyTuple>,
                kwargs: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<Py<$rust>> {
                if kwargs.is_some_and(|d| !d.is_empty()) {
                    return Err(transpose_no_kwargs_err());
                }
                transpose_axes_ok(axes)?;
                Py::new(py, $rust::init_chain(self.value))
            }
            // `$prim::swap_bytes()` is Rust's exact reverse-byte-order
            // primitive -- for `i8`/`u8` (single byte) it is a no-op,
            // matching numpy's own single-byte no-op live-verified above.
            #[pyo3(signature = (inplace=None))]
            fn byteswap(&self, py: Python<'_>, inplace: Option<&Bound<'_, PyAny>>) -> PyResult<Py<$rust>> {
                if inplace.map(|v| v.is_truthy()).transpose()?.unwrap_or(false) {
                    return Err(PyValueError::new_err("cannot byteswap a scalar in-place"));
                }
                Py::new(py, $rust::init_chain(self.value.swap_bytes()))
            }
            #[pyo3(signature = (order=None))]
            fn ravel(&self, py: Python<'_>, order: Option<&str>) -> PyResult<Py<PyArray>> {
                let _ = order;
                let arr =
                    NdArray::from_buffer(Buffer::$buffer_variant(vec![self.value]), vec![1], Order::C)
                        .map_err(to_py_err)?;
                Py::new(py, PyArray { inner: arr })
            }
            #[pyo3(signature = (order=None))]
            fn flatten(&self, py: Python<'_>, order: Option<&str>) -> PyResult<Py<PyArray>> {
                self.ravel(py, order)
            }
            #[getter]
            fn flat(&self, py: Python<'_>) -> PyResult<Py<ScalarFlatIter>> {
                let v = Py::new(py, $rust::init_chain(self.value))?.into_any();
                Py::new(py, ScalarFlatIter::new(v))
            }
            fn nonzero(&self) -> PyResult<()> {
                Err(PyValueError::new_err(NONZERO_MSG_SHORT))
            }
        }
    };
}

// A tiny trait so the macro above can build the SignedInteger/UnsignedInteger
// marker level generically without a second macro parameter per call site.
trait MarkerNew {
    fn new_marker() -> Self;
}
impl MarkerNew for SignedInteger {
    fn new_marker() -> Self {
        SignedInteger
    }
}
impl MarkerNew for UnsignedInteger {
    fn new_marker() -> Self {
        UnsignedInteger
    }
}

int_leaf!(Int8, "int8", i8, SignedInteger, DType::I8, I8, "int8");
int_leaf!(Int16, "int16", i16, SignedInteger, DType::I16, I16, "int16");
int_leaf!(Int32, "int32", i32, SignedInteger, DType::I32, I32, "int32");
int_leaf!(Int64, "int64", i64, SignedInteger, DType::I64, I64, "int64");
int_leaf!(UInt8, "uint8", u8, UnsignedInteger, DType::U8, U8, "uint8");
int_leaf!(UInt16, "uint16", u16, UnsignedInteger, DType::U16, U16, "uint16");
int_leaf!(UInt32, "uint32", u32, UnsignedInteger, DType::U32, U32, "uint32");
int_leaf!(UInt64, "uint64", u64, UnsignedInteger, DType::U64, U64, "uint64");

// ---------------------------------------------------------------------
// Concrete floating leaves.
// ---------------------------------------------------------------------

/// Python's `str(float)`/`repr(float)` always show a decimal point (`5.0`,
/// not `5`), and lowercase `nan`/`inf`/`-inf` -- unlike Rust's own `Display`
/// for floats, which prints whole numbers with NO trailing `.0` (`5`) and
/// capitalizes NaN as `NaN`. Verified against real numpy 2.5.1 that its
/// float scalars follow Python's convention exactly (`str(np.float64(5.0))
/// == '5.0'`, `str(np.float64(float('nan'))) == 'nan'`). Implemented once,
/// generically, over the one trait method each of f16/f32/f64 needs beyond
/// `Display` (their own native shortest-round-trip formatting is reused
/// as-is -- this only patches the two conventions above, it does not
/// reimplement float-to-string).
trait PyFloatFmt: Copy + std::fmt::Display {
    fn ionp_is_nan(&self) -> bool;
    fn ionp_is_infinite(&self) -> bool;
    fn ionp_is_sign_negative(&self) -> bool;
}
impl PyFloatFmt for f32 {
    fn ionp_is_nan(&self) -> bool {
        f32::is_nan(*self)
    }
    fn ionp_is_infinite(&self) -> bool {
        f32::is_infinite(*self)
    }
    fn ionp_is_sign_negative(&self) -> bool {
        f32::is_sign_negative(*self)
    }
}
impl PyFloatFmt for f64 {
    fn ionp_is_nan(&self) -> bool {
        f64::is_nan(*self)
    }
    fn ionp_is_infinite(&self) -> bool {
        f64::is_infinite(*self)
    }
    fn ionp_is_sign_negative(&self) -> bool {
        f64::is_sign_negative(*self)
    }
}
impl PyFloatFmt for half::f16 {
    fn ionp_is_nan(&self) -> bool {
        half::f16::is_nan(*self)
    }
    fn ionp_is_infinite(&self) -> bool {
        half::f16::is_infinite(*self)
    }
    fn ionp_is_sign_negative(&self) -> bool {
        half::f16::is_sign_negative(*self)
    }
}
// Python/numpy switch a float's str/repr from fixed-point to scientific
// notation once the decimal-point position ("decpt" = the base-10 exponent
// of the leading digit, plus one) falls outside a dtype-specific window.
// This window is NOT the same for every dtype -- verified live against
// numpy 2.5.1 (`repr(np.float16/32/64(10**e))` swept over e) rather than
// assumed from Python's own builtin-float threshold:
//   float16: fixed iff -4 <  decpt <= 3   (e.g. 1e3 -> "1e+03", but
//            0.0001 stays fixed)
//   float32: fixed iff -3 <  decpt <= 6   (tighter on BOTH ends than
//            float16/float64 -- 0.0001 is already "1e-04" for float32,
//            while it is fixed for float16 and float64)
//   float64: fixed iff -4 <  decpt <= 16
// `high_decpt_incl`/`low_decpt_excl` below are exactly those two bounds,
// supplied per dtype by each `float_leaf!` invocation.
fn fmt_pyfloat<T: PyFloatFmt + std::fmt::LowerExp>(v: T, high_decpt_incl: i32, low_decpt_excl: i32) -> String {
    if v.ionp_is_nan() {
        return "nan".to_string();
    }
    if v.ionp_is_infinite() {
        return if v.ionp_is_sign_negative() {
            "-inf".to_string()
        } else {
            "inf".to_string()
        };
    }
    // `{:e}` always normalizes to a single leading digit before the decimal
    // point (e.g. "1.5e-5", "-1e30", "0e0"), so its exponent directly gives
    // us decpt = exp + 1 without reimplementing shortest-round-trip digit
    // generation -- we reuse Rust's own (f16 delegates to f32's).
    let sci = format!("{v:e}");
    let (mantissa, exp_str) = sci
        .split_once('e')
        .expect("LowerExp output always contains 'e'");
    let exp: i32 = exp_str
        .parse()
        .expect("LowerExp exponent is always a valid base-10 integer");
    let decpt = exp + 1;
    if decpt > low_decpt_excl && decpt <= high_decpt_incl {
        let s = format!("{v}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    } else {
        let sign = if exp < 0 { "-" } else { "+" };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    }
}

macro_rules! float_leaf {
    ($rust:ident, $pyname:literal, $prim:ty, $dtype:expr, $buffer_variant:ident, $dtypename:literal, $fromf64:expr, $tof64:expr, $high_decpt:literal, $low_decpt:literal) => {
        float_leaf!($rust, $pyname, $prim, $dtype, $buffer_variant, $dtypename, $fromf64, $tof64, $high_decpt, $low_decpt, None, false, false);
    };
    ($rust:ident, $pyname:literal, $prim:ty, $dtype:expr, $buffer_variant:ident, $dtypename:literal, $fromf64:expr, $tof64:expr, $high_decpt:literal, $low_decpt:literal, $spelling:expr, $quote_repr:literal) => {
        float_leaf!($rust, $pyname, $prim, $dtype, $buffer_variant, $dtypename, $fromf64, $tof64, $high_decpt, $low_decpt, $spelling, $quote_repr, false);
    };
    ($rust:ident, $pyname:literal, $prim:ty, $dtype:expr, $buffer_variant:ident, $dtypename:literal, $fromf64:expr, $tof64:expr, $high_decpt:literal, $low_decpt:literal, $spelling:expr, $quote_repr:literal, $strict_str_parse:literal) => {
        #[pyclass(name = $pyname, module = "anionpy", extends = Floating)]
        pub struct $rust {
            pub(crate) value: $prim,
        }

        impl $rust {
            pub(crate) fn init_chain(value: $prim) -> PyClassInitializer<$rust> {
                PyClassInitializer::from(Generic)
                    .add_subclass(Number)
                    .add_subclass(Inexact)
                    .add_subclass(Floating)
                    .add_subclass($rust { value })
            }
        }

        #[pymethods]
        impl $rust {
            #[new]
            #[pyo3(signature = (value=None))]
            fn new(value: Option<&Bound<'_, PyAny>>) -> PyResult<PyClassInitializer<$rust>> {
                let v: $prim = match value {
                    None => ($fromf64)(0.0),
                    Some(obj) => {
                        reject_sequence(obj, $dtypename)?;
                        if $strict_str_parse {
                            reject_trailing_whitespace_str(obj, "long double")?;
                        }
                        match construct_float_buffer(obj, $dtype)? {
                            Buffer::$buffer_variant(vals) => vals[0],
                            _ => unreachable!("construct_float_buffer returned the wrong Buffer variant"),
                        }
                    }
                };
                Ok($rust::init_chain(v))
            }

            #[classattr]
            fn __ionp_dtype_name__() -> &'static str {
                $dtypename
            }

            #[getter]
            fn dtype(&self) -> PyDType {
                PyDType { inner: $dtype, spelling: $spelling }
            }
            #[getter]
            fn itemsize(&self) -> usize {
                $dtype.itemsize()
            }
            fn __repr__(&self) -> String {
                let s = fmt_pyfloat(self.value, $high_decpt, $low_decpt);
                if $quote_repr {
                    format!(concat!("anionpy.", $pyname, "('{}')"), s)
                } else {
                    format!(concat!("anionpy.", $pyname, "({})"), s)
                }
            }
            fn __str__(&self) -> String {
                fmt_pyfloat(self.value, $high_decpt, $low_decpt)
            }
            // See the matching comment on Bool_ above. `__float__` alone is
            // enough for CPython's `complex()` builtin to succeed on this
            // type as a source (falls back to `__float__` when `__complex__`
            // is absent), so no separate `__complex__` is needed here.
            // `__int__` matches numpy's float scalars, which truncate toward
            // zero on `int()` (verified: `int(np.float64(3.9)) == 3`).
            fn __float__(&self) -> f64 {
                ($tof64)(self.value)
            }
            fn __int__(&self) -> i64 {
                ($tof64)(self.value) as i64
            }

            fn __eq__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::Equal, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __ne__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::NotEqual, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __lt__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::Less, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __le__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::LessEqual, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __gt__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::Greater, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __ge__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::GreaterEqual, Buffer::$buffer_variant(vec![self.value]), other)
            }
            // `slf.as_ptr()` (not a temporary object's address) is what makes
            // this stable across repeated `hash()` calls on the SAME anionpy
            // scalar instance despite NaN having no well-defined value hash
            // -- see `py_hash_double`'s doc comment above for the full
            // reasoning and the live measurement that ruled out the naive
            // "build a fresh float and hash it" approach.
            fn __hash__(slf: &Bound<'_, Self>) -> PyResult<isize> {
                let py = slf.py();
                let ptr = slf.as_ptr() as usize;
                let v: f64 = ($tof64)(slf.borrow().value);
                py_hash_double(py, ptr, v)
            }

            // ---- Arithmetic ---------------------------------------------
            // See the identical block on `Bool_` above for the shared-engine
            // rationale. Float arithmetic (including float16) computes
            // through `ionp_core::ufunc`'s own per-width loops, not f32/f64
            // then truncated -- see that engine's own half-precision loop
            // for the rounding guarantee.
            fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Add, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Add, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Subtract, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Subtract, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Multiply, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Multiply, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_truediv(py, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_truediv(py, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __floordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::FloorDivide, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rfloordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::FloorDivide, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __mod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_math_binop(py, MathBinaryOp::Remainder, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_math_binop(py, MathBinaryOp::Remainder, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __divmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_divmod(py, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rdivmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_divmod(py, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __pow__(&self, py: Python<'_>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
                if modulo.is_some() {
                    return Ok(py.NotImplemented());
                }
                scalar_math_binop(py, MathBinaryOp::Power, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rpow__(&self, py: Python<'_>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
                if modulo.is_some() {
                    return Ok(py.NotImplemented());
                }
                scalar_math_binop(py, MathBinaryOp::Power, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let a = NdArray::from_buffer(Buffer::$buffer_variant(vec![self.value]), vec![], Order::C).map_err(to_py_err)?;
                let out = ionp_core::ufunc::unary_op(ionp_core::ufunc::UnaryOp::Negative, &a).map_err(to_py_err)?;
                wrap_arith_result(py, out)
            }
            fn __pos__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                Ok(Py::new(py, $rust::init_chain(self.value))?.into_any())
            }
            fn __abs__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let a = NdArray::from_buffer(Buffer::$buffer_variant(vec![self.value]), vec![], Order::C).map_err(to_py_err)?;
                let out = ionp_core::ufunc::unary_op(ionp_core::ufunc::UnaryOp::Absolute, &a).map_err(to_py_err)?;
                wrap_arith_result(py, out)
            }

            // ---- PART 1: 0-d array surface -----------------------------
            #[getter]
            fn shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
                PyTuple::empty(py)
            }
            #[getter]
            fn ndim(&self) -> usize {
                0
            }
            #[getter]
            fn size(&self) -> usize {
                1
            }
            #[getter]
            fn nbytes(&self) -> usize {
                $dtype.itemsize()
            }
            #[getter]
            fn strides<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
                PyTuple::empty(py)
            }
            #[getter]
            fn base(&self, py: Python<'_>) -> Py<PyAny> {
                py.None()
            }
            #[getter]
            fn flags(&self) -> ScalarFlags {
                ScalarFlags
            }
            #[getter]
            fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let bytes = PyBytes::new(py, &self.value.to_ne_bytes());
                py.import("builtins")?.call_method1("memoryview", (bytes,))
            }
            #[getter]
            fn T(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            #[getter]
            fn real(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            #[getter]
            fn imag(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(($fromf64)(0.0)))
            }
            fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                Ok((($tof64)(self.value)).into_pyobject(py)?.into_any().unbind())
            }
            fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                self.item(py)
            }
            fn copy(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            fn conjugate(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            fn fill(&self, obj: &Bound<'_, PyAny>) -> PyResult<()> {
                match construct_float_buffer(obj, $dtype)? {
                    Buffer::$buffer_variant(_) => Ok(()),
                    _ => unreachable!("construct_float_buffer returned the wrong Buffer variant"),
                }
            }
            fn all(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
                Py::new(py, Bool_::init_chain(($tof64)(self.value) != 0.0))
            }
            fn any(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
                self.all(py)
            }
            fn min(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            fn max(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            // numpy: float `.sum()`/`.prod()`/`.mean()` all stay at the
            // SAME dtype (verified: `type(np.float16(2.5).sum()) is
            // np.float16`, `type(np.float16(5.5).mean()) is np.float16` --
            // unlike integer reductions, float scalars are never widened).
            fn sum(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            fn prod(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            fn mean(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            // numpy: float `.round(ndigits)` uses round-half-to-even at the
            // requested decimal place (default `ndigits=0`), staying at the
            // same dtype (verified: `np.float64(5.5).round()` -> `6.0`,
            // `np.float64(5.567).round(2)` -> `5.57`). Implemented via a
            // multiply/round-ties-even/divide round-trip through `f64` --
            // this matches numpy for the ordinary cases exercised by this
            // module's tests, but is NOT a byte-exact reimplementation of
            // numpy's correctly-rounded decimal algorithm for extreme
            // `ndigits`/magnitude combinations; disclosed as a known
            // precision-edge gap in the final report rather than silently
            // assumed exact.
            #[pyo3(signature = (ndigits=None))]
            fn round(&self, py: Python<'_>, ndigits: Option<i32>) -> PyResult<Py<$rust>> {
                let n = ndigits.unwrap_or(0);
                let v64 = ($tof64)(self.value);
                let r = if v64.is_finite() {
                    let scale = 10f64.powi(n);
                    (v64 * scale).round_ties_even() / scale
                } else {
                    v64
                };
                Py::new(py, $rust::init_chain(($fromf64)(r)))
            }
            #[pyo3(signature = (dtype, order=None, casting=None, subok=None, copy=None))]
            fn astype(
                &self,
                py: Python<'_>,
                dtype: &Bound<'_, PyAny>,
                order: Option<&str>,
                casting: Option<&str>,
                subok: Option<&Bound<'_, PyAny>>,
                copy: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                // Both accepted-and-discarded, but must ACCEPT/REJECT the
                // same values real numpy does -- see `scalar_index_like_check`.
                if let Some(v) = subok.as_ref() {
                    scalar_index_like_check(v)?;
                }
                if let Some(v) = copy.as_ref() {
                    scalar_index_like_check(v)?;
                }
                let _ = (order, casting, subok, copy);
                // A scalar source is never S/U (no scalar wrapper type exists for
        // it, see `scalar_from_buffer`'s own guard) -- decline an S/U TARGET
        // here too, before `cast_to` panics on it.
        let target = crate::dtype_from_pyobj_no_su(dtype)?;
                let buf = Buffer::$buffer_variant(vec![self.value]).cast_to(target);
                scalar_from_buffer(py, &buf)
            }
            #[pyo3(signature = (*shape, order=None))]
            fn reshape(&self, py: Python<'_>, shape: &Bound<'_, PyTuple>, order: Option<&str>) -> PyResult<Py<PyAny>> {
                scalar_reshape(py, shape, order, Buffer::$buffer_variant(vec![self.value]))
            }
            // See `Bool_::__array__`'s comment: delegates to `PyArray`'s
            // own, already-correct `__array__` rather than returning
            // anionpy's own array type, which numpy's `np.asarray()` rejects.
            fn __array__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                let arr = NdArray::from_buffer(Buffer::$buffer_variant(vec![self.value]), vec![], Order::C)
                    .map_err(to_py_err)?;
                PyArray { inner: arr }.__array___no_cast(py)
            }
            // ---- PART 5 (coordinator audit) + axis-argument follow-up:
            // see Bool_'s matching methods for the live-verified reference
            // behavior; squeeze/transpose now accept the full axis-argument
            // surface via the shared `squeeze_axis_ok`/`transpose_axes_ok`
            // helpers (top of this file).
            #[pyo3(signature = (*args, **kwargs))]
            fn squeeze(
                &self,
                py: Python<'_>,
                args: &Bound<'_, PyTuple>,
                kwargs: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<Py<$rust>> {
                let axis = resolve_squeeze_axis(args, kwargs)?;
                squeeze_axis_ok(axis.as_ref())?;
                Py::new(py, $rust::init_chain(self.value))
            }
            #[pyo3(signature = (*axes, **kwargs))]
            fn transpose(
                &self,
                py: Python<'_>,
                axes: &Bound<'_, PyTuple>,
                kwargs: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<Py<$rust>> {
                if kwargs.is_some_and(|d| !d.is_empty()) {
                    return Err(transpose_no_kwargs_err());
                }
                transpose_axes_ok(axes)?;
                Py::new(py, $rust::init_chain(self.value))
            }
            // `$prim::to_bits()/from_bits()` + `swap_bytes()` on the
            // resulting unsigned integer is a real reverse-byte-order
            // reinterpretation of the float's own IEEE-754 bit pattern --
            // NOT a numeric byte-swap of the decimal value (live-verified
            // against numpy: `np.float32(2.5).byteswap()` produces a
            // different, essentially-garbage float, not `2.5` reversed).
            #[pyo3(signature = (inplace=None))]
            fn byteswap(&self, py: Python<'_>, inplace: Option<&Bound<'_, PyAny>>) -> PyResult<Py<$rust>> {
                if inplace.map(|v| v.is_truthy()).transpose()?.unwrap_or(false) {
                    return Err(PyValueError::new_err("cannot byteswap a scalar in-place"));
                }
                let swapped = <$prim>::from_bits(self.value.to_bits().swap_bytes());
                Py::new(py, $rust::init_chain(swapped))
            }
            #[pyo3(signature = (order=None))]
            fn ravel(&self, py: Python<'_>, order: Option<&str>) -> PyResult<Py<PyArray>> {
                let _ = order;
                let arr =
                    NdArray::from_buffer(Buffer::$buffer_variant(vec![self.value]), vec![1], Order::C)
                        .map_err(to_py_err)?;
                Py::new(py, PyArray { inner: arr })
            }
            #[pyo3(signature = (order=None))]
            fn flatten(&self, py: Python<'_>, order: Option<&str>) -> PyResult<Py<PyArray>> {
                self.ravel(py, order)
            }
            #[getter]
            fn flat(&self, py: Python<'_>) -> PyResult<Py<ScalarFlatIter>> {
                let v = Py::new(py, $rust::init_chain(self.value))?.into_any();
                Py::new(py, ScalarFlatIter::new(v))
            }
            fn nonzero(&self) -> PyResult<()> {
                Err(PyValueError::new_err(NONZERO_MSG_SHORT))
            }
        }
    };
}

float_leaf!(
    Float16,
    "float16",
    half::f16,
    DType::F16,
    F16,
    "float16",
    half::f16::from_f64,
    half::f16::to_f64,
    3,
    -4
);
float_leaf!(
    Float32,
    "float32",
    f32,
    DType::F32,
    F32,
    "float32",
    |v: f64| v as f32,
    |v: f32| v as f64,
    6,
    -3
);
float_leaf!(
    Float64,
    "float64",
    f64,
    DType::F64,
    F64,
    "float64",
    |v: f64| v,
    |v: f64| v,
    16,
    -4
);

/// TICKET #90a: `anionpy.longdouble`. On THIS platform (macOS arm64, C `long
/// double` == `double`, 8 bytes) real numpy's `np.longdouble` is storage-
/// identical to `np.float64` -- verified live: `np.dtype(np.longdouble) ==
/// np.dtype('d')`, same itemsize, same bits -- but it is NOT the same
/// class: `np.longdouble is np.float64` is `False` (unlike `np.double`,
/// which genuinely IS `np.float64` by `is`-identity, registered above via
/// `m.add("double", &float64_cls)`). `np.longdouble` is its own distinct
/// leaf class with its own `.dtype` (`np.dtype('g')`, `char='g'`, `.name`
/// still `'float64'`) and its own quoted `repr()` (verified live:
/// `repr(np.longdouble(3.5)) == "np.longdouble('3.5')"`, WITH quotes,
/// unlike every other float leaf's unquoted `np.float64(3.5)` -- numpy
/// formats `longdouble`/`clongdouble` through its extended-precision
/// string path regardless of whether this platform's extended precision
/// happens to collapse to plain `double`). So this is a real second class,
/// not `m.add("longdouble", &float64_cls)` -- that alias would be
/// `is`-identical to `float64`, which is factually wrong here (this is the
/// SAME reasoning `tests/differential/scalar_cases.py`'s
/// `_ALIAS_PAIRS`-doc-comment already gives for excluding `longdouble` from
/// that list). Reuses `float_leaf!` for byte-identical f64 storage/
/// arithmetic (there is no other precision available on this platform, and
/// none is being invented here), but with `spelling = Some('g')` (so
/// `.dtype` carries the `'g'` char via the SAME `PyDType::spelling` tag
/// ticket #78 introduced for `'q'`/`'Q'`/`'g'`/`'G'`, not a new mechanism)
/// and `quote_repr = true` for the quoted repr format.
///
/// NOT implemented here (disclosed gap, matching this file's existing
/// `_ALIAS_PAIRS` comment which already called this out as
/// "deprioritized given the time budget" before this ticket): `type(x) is
/// LongDoubleDType` for `x = anionpy.dtype('g')` -- `anionpy.dtype` itself has
/// no per-dtype `__class__` subclass system at all yet (confirmed: EVERY
/// `anionpy.dtypes.*DType` name is absent, not just this one), so
/// `anionpy.dtype('g').__class__` stays the single `anionpy.dtype` class,
/// same as every other spelling. `anionpy.dtypes.LongDoubleDType` (below,
/// this ticket) is a genuine `PyDType` subclass and its own INSTANCES are
/// correctly typed, but constructing a dtype the OTHER way (via
/// `anionpy.dtype('g')`) does not retroactively become that subclass --
/// exactly mirroring how none of the other 13 dtypes' `dtype(...)` calls
/// return their `numpy.dtypes.*DType` subclass either.
///
/// COMPILE-TIME GATE: `#[cfg(all(target_arch = "aarch64", target_vendor =
/// "apple"))]` below is the ticket's hard constraint made real -- on any
/// target where C `long double` is genuinely wider than `double` (glibc/
/// aarch64, x86-64, both 16 bytes), this whole macro invocation (and thus
/// the `LongDouble` type, `anionpy.longdouble`) is never compiled in, full
/// stop -- not a runtime branch. See `dtypes_module.rs`'s matching gate and
/// `anionpy/__init__.py`'s `try/except ImportError` that turns "symbol
/// missing from `_anionpy`" into "name absent from `anionpy`" on such a
/// target.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
float_leaf!(
    LongDouble,
    "longdouble",
    f64,
    DType::F64,
    F64,
    "float64",
    |v: f64| v,
    |v: f64| v,
    16,
    -4,
    Some('g'),
    true,
    true
);

// ---------------------------------------------------------------------
// Concrete complex leaves.
// ---------------------------------------------------------------------

macro_rules! complex_leaf {
    ($rust:ident, $pyname:literal, $ctype:ty, $dtype:expr, $buffer_variant:ident, $dtypename:literal, $zero:expr, $fmt:expr, $realrust:ident, $realprim:ty) => {
        complex_leaf!($rust, $pyname, $ctype, $dtype, $buffer_variant, $dtypename, $zero, $fmt, $realrust, $realprim, None, false);
    };
    ($rust:ident, $pyname:literal, $ctype:ty, $dtype:expr, $buffer_variant:ident, $dtypename:literal, $zero:expr, $fmt:expr, $realrust:ident, $realprim:ty, $spelling:expr, $quote_repr:literal) => {
        #[pyclass(name = $pyname, module = "anionpy", extends = ComplexFloating)]
        pub struct $rust {
            pub(crate) value: $ctype,
        }

        impl $rust {
            pub(crate) fn init_chain(value: $ctype) -> PyClassInitializer<$rust> {
                PyClassInitializer::from(Generic)
                    .add_subclass(Number)
                    .add_subclass(Inexact)
                    .add_subclass(ComplexFloating)
                    .add_subclass($rust { value })
            }
        }

        #[pymethods]
        impl $rust {
            #[new]
            #[pyo3(signature = (value=None))]
            fn new(value: Option<&Bound<'_, PyAny>>) -> PyResult<PyClassInitializer<$rust>> {
                let v: $ctype = match value {
                    None => $zero,
                    Some(obj) => {
                        reject_sequence(obj, $dtypename)?;
                        match construct_complex_buffer(obj, $dtype)? {
                            Buffer::$buffer_variant(vals) => vals[0],
                            _ => unreachable!("construct_complex_buffer returned the wrong Buffer variant"),
                        }
                    }
                };
                Ok($rust::init_chain(v))
            }

            #[classattr]
            fn __ionp_dtype_name__() -> &'static str {
                $dtypename
            }

            #[getter]
            fn dtype(&self) -> PyDType {
                PyDType { inner: $dtype, spelling: $spelling }
            }
            #[getter]
            fn itemsize(&self) -> usize {
                $dtype.itemsize()
            }
            fn __repr__(&self) -> String {
                let s = $fmt(self.value);
                if $quote_repr {
                    format!(concat!("anionpy.", $pyname, "('{}')"), s)
                } else {
                    format!(concat!("anionpy.", $pyname, "({})"), s)
                }
            }
            fn __str__(&self) -> String {
                fmt_complex_str(self.value.re as f64, $fmt(self.value))
            }
            // See the matching comment on Bool_ above. Deliberately no
            // `__int__`/`__float__`/`__index__`: real numpy raises
            // `TypeError: can't convert complex to <T>` when a complex
            // scalar is narrowed to int/float, which is exactly CPython's
            // OWN generic error for an object missing those dunders, so
            // omitting them here reproduces numpy's error path for free.
            fn __complex__<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyComplex> {
                pyo3::types::PyComplex::from_doubles(py, self.value.re as f64, self.value.im as f64)
            }

            // See this module's top-of-section doc comment for why these do
            // NOT raise `TypeError` on ordering: real numpy 2.5.1 measured
            // live does not either, it lexicographically orders
            // real-then-imaginary via the exact rule
            // `ionp_core::ufunc::cmp_complex` already implements.
            fn __eq__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::Equal, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __ne__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::NotEqual, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __lt__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::Less, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __le__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::LessEqual, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __gt__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::Greater, Buffer::$buffer_variant(vec![self.value]), other)
            }
            fn __ge__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                scalar_richcmp(py, BinaryOp::GreaterEqual, Buffer::$buffer_variant(vec![self.value]), other)
            }
            // CPython's `complex_hash`: combine the real and imaginary
            // components' own hashes via `combine_complex_hash`
            // (`_PyHASH_IMAG`-weighted sum). Both components share the SAME
            // object pointer for their NaN fallback (mirrors CPython, which
            // hashes a complex's real/imag parts against the complex
            // object's OWN identity, not two independent ephemeral floats).
            // When `im == 0.0` this collapses to exactly `hashreal`, which is
            // the mechanism behind `hash(anionpy.complex128(3+0j)) ==
            // hash(3)` -- verified against real numpy.
            fn __hash__(slf: &Bound<'_, Self>) -> PyResult<isize> {
                let py = slf.py();
                let ptr = slf.as_ptr() as usize;
                let v = slf.borrow().value;
                let hashreal = py_hash_double(py, ptr, v.re as f64)?;
                let hashimag = py_hash_double(py, ptr, v.im as f64)?;
                Ok(combine_complex_hash(hashreal, hashimag))
            }

            // ---- Arithmetic ---------------------------------------------
            // Deliberately NO `__floordiv__`/`__rfloordiv__`/`__mod__`/
            // `__rmod__`/`__divmod__`/`__rdivmod__`/`__lshift__`/`__and__`/
            // etc. here: real numpy's complex scalar types genuinely do not
            // implement any of these (verified live against numpy 2.5.1:
            // `np.complex64(3+4j) // np.complex64(1+1j)` raises
            // `TypeError: unsupported operand type(s) for //: 'numpy.complex64'
            // and 'numpy.complex64'` -- CPython's own generic fallback
            // message for a MISSING dunder, not a raised-from-inside-the-
            // dunder error; `np.complex64 // np.complex64` differs from
            // ARRAY `complex_array // complex_array`, which DOES go through
            // the ufunc engine and gets the richer "ufunc not supported..."
            // message -- scalar and array complex floordiv/mod are
            // genuinely different code paths in real numpy, not a typo).
            // Leaving these dunders entirely undefined here reproduces that
            // exact scalar-path behavior for free, with anionpy's own type name
            // substituted for numpy's (CPython builds that message, not us).
            fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Add, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Add, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Subtract, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Subtract, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_binop(py, BinaryOp::Multiply, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                // Same "true operand order, not just delegate to __mul__"
                // requirement as `PyArray::__rmul__` -- complex multiply is
                // not bit-reproducible under operand swap.
                scalar_binop(py, BinaryOp::Multiply, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_truediv(py, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                scalar_truediv(py, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __pow__(&self, py: Python<'_>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
                if modulo.is_some() {
                    return Ok(py.NotImplemented());
                }
                scalar_math_binop(py, MathBinaryOp::Power, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, false)
            }
            fn __rpow__(&self, py: Python<'_>, other: &Bound<'_, PyAny>, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
                if modulo.is_some() {
                    return Ok(py.NotImplemented());
                }
                scalar_math_binop(py, MathBinaryOp::Power, &Buffer::$buffer_variant(vec![self.value]), $dtype, other, true)
            }
            fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let a = NdArray::from_buffer(Buffer::$buffer_variant(vec![self.value]), vec![], Order::C).map_err(to_py_err)?;
                let out = ionp_core::ufunc::unary_op(ionp_core::ufunc::UnaryOp::Negative, &a).map_err(to_py_err)?;
                wrap_arith_result(py, out)
            }
            fn __pos__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                Ok(Py::new(py, $rust::init_chain(self.value))?.into_any())
            }
            fn __abs__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let a = NdArray::from_buffer(Buffer::$buffer_variant(vec![self.value]), vec![], Order::C).map_err(to_py_err)?;
                let out = ionp_core::ufunc::unary_op(ionp_core::ufunc::UnaryOp::Absolute, &a).map_err(to_py_err)?;
                wrap_arith_result(py, out)
            }

            // ---- PART 1: 0-d array surface -----------------------------
            #[getter]
            fn shape<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
                PyTuple::empty(py)
            }
            #[getter]
            fn ndim(&self) -> usize {
                0
            }
            #[getter]
            fn size(&self) -> usize {
                1
            }
            #[getter]
            fn nbytes(&self) -> usize {
                $dtype.itemsize()
            }
            #[getter]
            fn strides<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
                PyTuple::empty(py)
            }
            #[getter]
            fn base(&self, py: Python<'_>) -> Py<PyAny> {
                py.None()
            }
            #[getter]
            fn flags(&self) -> ScalarFlags {
                ScalarFlags
            }
            #[getter]
            fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let mut raw = self.value.re.to_ne_bytes().to_vec();
                raw.extend_from_slice(&self.value.im.to_ne_bytes());
                let bytes = PyBytes::new(py, &raw);
                py.import("builtins")?.call_method1("memoryview", (bytes,))
            }
            #[getter]
            fn T(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            // numpy: `.real`/`.imag` on a complex scalar return the
            // MATCHING-PRECISION float scalar type (verified:
            // `type(np.complex64(3+4j).real) is np.float32`,
            // `type(np.complex128(3+4j).real) is np.float64` -- never
            // widened/narrowed to a fixed width).
            #[getter]
            fn real(&self, py: Python<'_>) -> PyResult<Py<$realrust>> {
                Py::new(py, $realrust::init_chain(self.value.re))
            }
            #[getter]
            fn imag(&self, py: Python<'_>) -> PyResult<Py<$realrust>> {
                Py::new(py, $realrust::init_chain(self.value.im))
            }
            fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                Ok(pyo3::types::PyComplex::from_doubles(py, self.value.re as f64, self.value.im as f64)
                    .into_any()
                    .unbind())
            }
            fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                self.item(py)
            }
            fn copy(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            fn conjugate(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(<$ctype>::new(self.value.re, -self.value.im)))
            }
            fn fill(&self, obj: &Bound<'_, PyAny>) -> PyResult<()> {
                match construct_complex_buffer(obj, $dtype)? {
                    Buffer::$buffer_variant(_) => Ok(()),
                    _ => unreachable!("construct_complex_buffer returned the wrong Buffer variant"),
                }
            }
            fn all(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
                Py::new(py, Bool_::init_chain(self.value.re != 0.0 || self.value.im != 0.0))
            }
            fn any(&self, py: Python<'_>) -> PyResult<Py<Bool_>> {
                self.all(py)
            }
            fn min(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            fn max(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            // numpy: complex `.sum()`/`.prod()`/`.mean()` all stay at the
            // same dtype (verified: `type(np.complex64(2+3j).sum()) is
            // np.complex64`), same "single-element reduction is the
            // identity" reasoning as the float leaves above.
            fn sum(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            fn prod(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            fn mean(&self, py: Python<'_>) -> PyResult<Py<$rust>> {
                Py::new(py, $rust::init_chain(self.value))
            }
            // numpy rounds a complex scalar's real and imaginary components
            // independently (same round-half-to-even rule as the float
            // leaves' `.round()` -- see that doc comment for the same
            // precision-edge disclosure, which applies here too).
            #[pyo3(signature = (ndigits=None))]
            fn round(&self, py: Python<'_>, ndigits: Option<i32>) -> PyResult<Py<$rust>> {
                let n = ndigits.unwrap_or(0);
                let scale = 10f64.powi(n);
                let round_component = |c: f64| -> f64 {
                    if c.is_finite() {
                        (c * scale).round_ties_even() / scale
                    } else {
                        c
                    }
                };
                let new_re = round_component(self.value.re as f64) as _;
                let new_im = round_component(self.value.im as f64) as _;
                Py::new(py, $rust::init_chain(<$ctype>::new(new_re, new_im)))
            }
            #[pyo3(signature = (dtype, order=None, casting=None, subok=None, copy=None))]
            fn astype(
                &self,
                py: Python<'_>,
                dtype: &Bound<'_, PyAny>,
                order: Option<&str>,
                casting: Option<&str>,
                subok: Option<&Bound<'_, PyAny>>,
                copy: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                // Both accepted-and-discarded, but must ACCEPT/REJECT the
                // same values real numpy does -- see `scalar_index_like_check`.
                if let Some(v) = subok.as_ref() {
                    scalar_index_like_check(v)?;
                }
                if let Some(v) = copy.as_ref() {
                    scalar_index_like_check(v)?;
                }
                let _ = (order, casting, subok, copy);
                // A scalar source is never S/U (no scalar wrapper type exists for
        // it, see `scalar_from_buffer`'s own guard) -- decline an S/U TARGET
        // here too, before `cast_to` panics on it.
        let target = crate::dtype_from_pyobj_no_su(dtype)?;
                let buf = Buffer::$buffer_variant(vec![self.value]).cast_to(target);
                scalar_from_buffer(py, &buf)
            }
            #[pyo3(signature = (*shape, order=None))]
            fn reshape(&self, py: Python<'_>, shape: &Bound<'_, PyTuple>, order: Option<&str>) -> PyResult<Py<PyAny>> {
                scalar_reshape(py, shape, order, Buffer::$buffer_variant(vec![self.value]))
            }
            // See `Bool_::__array__`'s comment: delegates to `PyArray`'s
            // own, already-correct `__array__` rather than returning
            // anionpy's own array type, which numpy's `np.asarray()` rejects.
            fn __array__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                let arr = NdArray::from_buffer(Buffer::$buffer_variant(vec![self.value]), vec![], Order::C)
                    .map_err(to_py_err)?;
                PyArray { inner: arr }.__array___no_cast(py)
            }
            // ---- PART 5 (coordinator audit) + axis-argument follow-up:
            // see Bool_'s matching methods for the live-verified reference
            // behavior; squeeze/transpose now accept the full axis-argument
            // surface via the shared `squeeze_axis_ok`/`transpose_axes_ok`
            // helpers (top of this file).
            #[pyo3(signature = (*args, **kwargs))]
            fn squeeze(
                &self,
                py: Python<'_>,
                args: &Bound<'_, PyTuple>,
                kwargs: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<Py<$rust>> {
                let axis = resolve_squeeze_axis(args, kwargs)?;
                squeeze_axis_ok(axis.as_ref())?;
                Py::new(py, $rust::init_chain(self.value))
            }
            #[pyo3(signature = (*axes, **kwargs))]
            fn transpose(
                &self,
                py: Python<'_>,
                axes: &Bound<'_, PyTuple>,
                kwargs: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<Py<$rust>> {
                if kwargs.is_some_and(|d| !d.is_empty()) {
                    return Err(transpose_no_kwargs_err());
                }
                transpose_axes_ok(axes)?;
                Py::new(py, $rust::init_chain(self.value))
            }
            // Complex byteswap reverses EACH component's own byte order
            // independently (re and im each treated as their own
            // `$realprim`-width IEEE-754 bit pattern), NOT the whole 8-/16-byte
            // value as one contiguous block -- live-verified against real
            // numpy via a raw-hex-byte probe: `np.complex64(2.5+1.5j)`'s
            // bytes `000020400000c03f` become `402000003fc00000` after
            // `.byteswap()`, i.e. the first 4 bytes (re) and last 4 bytes
            // (im) are EACH independently reversed, not the 8-byte block as
            // a whole. Same pattern confirmed for complex128 (8-byte
            // components).
            #[pyo3(signature = (inplace=None))]
            fn byteswap(&self, py: Python<'_>, inplace: Option<&Bound<'_, PyAny>>) -> PyResult<Py<$rust>> {
                if inplace.map(|v| v.is_truthy()).transpose()?.unwrap_or(false) {
                    return Err(PyValueError::new_err("cannot byteswap a scalar in-place"));
                }
                let re = <$realprim>::from_bits(self.value.re.to_bits().swap_bytes());
                let im = <$realprim>::from_bits(self.value.im.to_bits().swap_bytes());
                Py::new(py, $rust::init_chain(<$ctype>::new(re, im)))
            }
            #[pyo3(signature = (order=None))]
            fn ravel(&self, py: Python<'_>, order: Option<&str>) -> PyResult<Py<PyArray>> {
                let _ = order;
                let arr =
                    NdArray::from_buffer(Buffer::$buffer_variant(vec![self.value]), vec![1], Order::C)
                        .map_err(to_py_err)?;
                Py::new(py, PyArray { inner: arr })
            }
            #[pyo3(signature = (order=None))]
            fn flatten(&self, py: Python<'_>, order: Option<&str>) -> PyResult<Py<PyArray>> {
                self.ravel(py, order)
            }
            #[getter]
            fn flat(&self, py: Python<'_>) -> PyResult<Py<ScalarFlatIter>> {
                let v = Py::new(py, $rust::init_chain(self.value))?.into_any();
                Py::new(py, ScalarFlatIter::new(v))
            }
            fn nonzero(&self) -> PyResult<()> {
                Err(PyValueError::new_err(NONZERO_MSG_SHORT))
            }
        }
    };
}

fn fmt_c64(v: C64) -> String {
    fmt_complex_core(v.re as f64, v.im as f64)
}
fn fmt_c128(v: C128) -> String {
    fmt_complex_core(v.re, v.im)
}
/// A plain Python-`float`-style number, no forced trailing `.0` (complex
/// components genuinely print WITHOUT it in real numpy -- `str(np.complex128(5+0j))
/// == '(5+0j)'`, not `'(5.0+0j)'`, verified live -- unlike bare float scalars,
/// which DO force it; see `fmt_pyfloat_f64` below for that separate rule) and
/// with NaN lowercased to match Python (Rust's own `Display` already prints
/// `inf`/`-inf` correctly, but `NaN` where Python/numpy print `nan`).
fn fmt_num(v: f64) -> String {
    if v.is_nan() {
        "nan".to_string()
    } else {
        format!("{v}")
    }
}
/// The unwrapped "content" numpy's complex-scalar repr/str both build from,
/// e.g. `5+0j`, `-3j`, `nan+1j` -- verified against real numpy 2.5.1: the
/// real part is dropped ENTIRELY (not shown as `0+`) whenever it is exactly
/// `0.0`, and the imaginary part always carries an explicit sign with no
/// space, e.g. `1+2j`, `1-2j`.
fn fmt_complex_core(re: f64, im: f64) -> String {
    if re == 0.0 {
        format!("{}j", fmt_num(im))
    } else if im.is_sign_negative() && !im.is_nan() {
        format!("{}-{}j", fmt_num(re), fmt_num(-im))
    } else {
        format!("{}+{}j", fmt_num(re), fmt_num(im))
    }
}
/// `str()` wraps the core in parentheses UNLESS the real part was dropped
/// (verified: `str(np.complex128(0j)) == '0j'`, no parens, but
/// `str(np.complex128(5+0j)) == '(5+0j)'`, parens); `repr()` (the `$fmt`
/// call sites above) never wraps -- the class-call syntax `complex128(...)`
/// already supplies the parens, and wrapping again would double them, which
/// is exactly the bug this function's introduction fixed (verified against
/// real numpy: `repr(np.complex128(5+0j)) == 'np.complex128(5+0j)'`, not
/// `'np.complex128((5+0j))'`).
fn fmt_complex_str(re: f64, core: String) -> String {
    if re == 0.0 {
        core
    } else {
        format!("({core})")
    }
}

complex_leaf!(
    Complex64,
    "complex64",
    C64,
    DType::C64,
    C64,
    "complex64",
    C64::new(0.0, 0.0),
    fmt_c64,
    Float32,
    f32
);
complex_leaf!(
    Complex128,
    "complex128",
    C128,
    DType::C128,
    C128,
    "complex128",
    C128::new(0.0, 0.0),
    fmt_c128,
    Float64,
    f64
);

/// TICKET #90a: `anionpy.clongdouble` -- `complex128`'s counterpart to
/// `LongDouble` above: same reasoning (genuinely distinct class from
/// `complex128` in real numpy despite identical 16-byte storage on this
/// platform, verified live `np.clongdouble is not np.complex128`, quoted
/// repr `"np.clongdouble('1+2j')"`), `spelling = Some('G')`, `quote_repr =
/// true`. `$realrust = LongDouble` (not `Float64`): real numpy's
/// `np.clongdouble(1+2j).real` is itself a `np.longdouble` instance, not a
/// plain `np.float64` (verified live) -- this project's `real`/`imag`
/// getters already parameterize that choice per complex leaf via
/// `$realrust`, so this is just supplying the correct one rather than a
/// new mechanism.
///
/// COMPILE-TIME GATE: same `#[cfg(...)]` as `LongDouble` above, for the
/// same reason -- `$realrust = LongDouble` also means this invocation can
/// only compile where `LongDouble` itself exists, i.e. the same gate is
/// required here regardless.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
complex_leaf!(
    CLongDouble,
    "clongdouble",
    C128,
    DType::C128,
    C128,
    "complex128",
    C128::new(0.0, 0.0),
    fmt_c128,
    LongDouble,
    f64,
    Some('G'),
    true
);

/// Registers every abstract-base and concrete scalar-type class, plus the
/// same-object type aliases (`intp`/`uintp`/`byte`/`half`/... -- assigned to
/// the exact same class object numpy uses, verified `is`-identical on this
/// platform: `np.intp is np.int64`, `np.uintp is np.uint64`, etc.) and the
/// module-level constants (`nan`/`inf`/`pi`/`e`/`euler_gamma`/`newaxis`/
/// `little_endian`/`True_`/`False_`).
pub(crate) fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ScalarFlags>()?;
    m.add_class::<ScalarFlatIter>()?;
    m.add_class::<Generic>()?;
    m.add_class::<Number>()?;
    m.add_class::<Integer>()?;
    m.add_class::<SignedInteger>()?;
    m.add_class::<UnsignedInteger>()?;
    m.add_class::<Inexact>()?;
    m.add_class::<Floating>()?;
    m.add_class::<ComplexFloating>()?;
    m.add_class::<Flexible>()?;
    m.add_class::<Character>()?;

    m.add_class::<Bool_>()?;
    // Same object under both names, exactly like numpy's `np.bool is
    // np.bool_` (the pyclass's real `name=` is "bool" -- see the comment on
    // the struct definition above).
    let bool_cls = m.getattr("bool")?;
    m.add("bool_", &bool_cls)?;
    m.add_class::<Int8>()?;
    m.add_class::<Int16>()?;
    m.add_class::<Int32>()?;
    m.add_class::<Int64>()?;
    m.add_class::<UInt8>()?;
    m.add_class::<UInt16>()?;
    m.add_class::<UInt32>()?;
    m.add_class::<UInt64>()?;
    m.add_class::<Float16>()?;
    m.add_class::<Float32>()?;
    m.add_class::<Float64>()?;
    m.add_class::<Complex64>()?;
    m.add_class::<Complex128>()?;
    // TICKET #90a: genuinely distinct classes, NOT same-object aliases --
    // see `LongDouble`'s/`CLongDouble`'s doc comments above. Gated: these
    // two registration calls only compile where the types themselves do
    // (see the `#[cfg(...)]` on the `float_leaf!`/`complex_leaf!`
    // invocations above) -- on a non-Apple-aarch64 target neither the
    // types nor these two lines exist in the compiled extension.
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    {
        m.add_class::<LongDouble>()?;
        m.add_class::<CLongDouble>()?;
    }

    // Same-object aliases: verified `np.<alias> is np.<canonical>` for every
    // pair below on this platform (arm64 Darwin, LP64 -- `intp`/`long`/etc.
    // alias the 64-bit types here; that mapping is itself platform-
    // dependent in real numpy, not hardcoded blindly).
    let int64_cls = m.getattr("int64")?;
    let uint64_cls = m.getattr("uint64")?;
    let int32_cls = m.getattr("int32")?;
    let uint32_cls = m.getattr("uint32")?;
    let int16_cls = m.getattr("int16")?;
    let uint16_cls = m.getattr("uint16")?;
    let int8_cls = m.getattr("int8")?;
    let uint8_cls = m.getattr("uint8")?;
    let float16_cls = m.getattr("float16")?;
    let float32_cls = m.getattr("float32")?;
    let float64_cls = m.getattr("float64")?;
    let complex64_cls = m.getattr("complex64")?;
    let complex128_cls = m.getattr("complex128")?;

    m.add("intp", &int64_cls)?;
    m.add("uintp", &uint64_cls)?;
    m.add("int_", &int64_cls)?;
    m.add("long", &int64_cls)?;
    m.add("uint", &uint64_cls)?;
    m.add("ulong", &uint64_cls)?;
    m.add("intc", &int32_cls)?;
    m.add("uintc", &uint32_cls)?;
    m.add("short", &int16_cls)?;
    m.add("ushort", &uint16_cls)?;
    m.add("byte", &int8_cls)?;
    m.add("ubyte", &uint8_cls)?;
    m.add("half", &float16_cls)?;
    m.add("single", &float32_cls)?;
    m.add("double", &float64_cls)?;
    m.add("csingle", &complex64_cls)?;
    m.add("cdouble", &complex128_cls)?;

    // Module-level float constants -- plain Python `float` literals, exactly
    // as real numpy 2.5.1 exposes them (`type(np.pi) is float`, verified;
    // not numpy scalar instances). No computation: every value is a
    // hardcoded literal matching numpy's own bit pattern (`euler_gamma`
    // cross-checked via `np.euler_gamma.hex()` ==
    // `0x1.2788cfc6fb619p-1`).
    m.add("nan", f64::NAN)?;
    m.add("inf", f64::INFINITY)?;
    m.add("pi", std::f64::consts::PI)?;
    m.add("e", std::f64::consts::E)?;
    m.add("euler_gamma", 0.5772156649015328606065120900824024f64)?;
    m.add("newaxis", py.None())?;
    // `bool` -- native Rust cfg, not a numpy runtime call. This platform is
    // little-endian (arm64 Darwin); `cfg!` bakes in the build target's real
    // endianness rather than hardcoding a boolean literal.
    m.add("little_endian", cfg!(target_endian = "little"))?;

    let true_ = Py::new(py, Bool_::init_chain(true))?;
    let false_ = Py::new(py, Bool_::init_chain(false))?;
    m.add("True_", true_)?;
    m.add("False_", false_)?;

    install_new_overrides(py, m)?;

    Ok(())
}

/// Monkeypatch `__new__` on each of the 14 concrete scalar leaf classes so
/// array-like input (list/tuple, or `.ndim >= 1`) builds an `anionpy.ndarray`
/// instead of raising -- exactly `np.<dtype>(obj)`'s own behavior. This is
/// sound, sanctioned CPython behavior, not UB: PyO3 pyclasses are heap
/// types, whose `__new__` slot is a plain mutable class attribute, and
/// CPython's `type_call()` never requires `tp_new`'s return value to be an
/// instance of `cls` -- it only skips calling `__init__` when it isn't
/// (verified: `Object.__new__ = staticmethod(lambda cls: [])` style
/// monkeypatching is a standard, documented CPython capability, not an
/// exploit of undefined behavior). `#[new]` itself is structurally locked to
/// `PyClassInitializer<Self>` and can never do this directly -- this
/// override sits ON TOP of it, calling the original `#[new]`-generated
/// associated function (`$rust::new`) unmodified for the scalar branch.
fn install_new_overrides(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! install {
        ($pyname:literal, $dtype:expr, $rust:ty) => {
            let f = make_new_override(py, $pyname, $dtype, |py, value| {
                let init = <$rust>::new(value)?;
                Ok(Py::new(py, init)?.into_any())
            })?;
            m.getattr($pyname)?.setattr("__new__", f)?;
        };
    }
    install!("bool", DType::Bool, Bool_);
    install!("int8", DType::I8, Int8);
    install!("int16", DType::I16, Int16);
    install!("int32", DType::I32, Int32);
    install!("int64", DType::I64, Int64);
    install!("uint8", DType::U8, UInt8);
    install!("uint16", DType::U16, UInt16);
    install!("uint32", DType::U32, UInt32);
    install!("uint64", DType::U64, UInt64);
    install!("float16", DType::F16, Float16);
    install!("float32", DType::F32, Float32);
    install!("float64", DType::F64, Float64);
    install!("complex64", DType::C64, Complex64);
    install!("complex128", DType::C128, Complex128);
    // TICKET #90a: `longdouble`/`clongdouble` need the exact same
    // array-like-input override every other concrete leaf gets above --
    // missed on the first pass (caught by the differential suite: without
    // this, `anionpy.longdouble([1, 2])` raised `TypeError` while real
    // `np.longdouble([1, 2])` builds an array, a real behavioral gap, not
    // the disclosed "permanent" list-rejection gap this file's module
    // docstring describes for the general case -- that gap was already
    // closed for the other 14 leaves by this exact mechanism). Gated:
    // these two only compile where `LongDouble`/`CLongDouble` themselves
    // do (see the `#[cfg(...)]` on their `float_leaf!`/`complex_leaf!`
    // invocations above).
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    {
        install!("longdouble", DType::F64, LongDouble);
        install!("clongdouble", DType::C128, CLongDouble);
    }
    Ok(())
}
