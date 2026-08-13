//! Top-level (`anionpy.sum(x)`, not `x.sum()`) reduction/statistics functions
//! -- PATH-TO-100.md item 3. Every one of these is a thin marshaling
//! wrapper: the actual arithmetic lives in `ionp_core::ufunc` (`reduce_axis`
//! for the plain-value family, the two new kernels this task added --
//! `accumulate_axis` for `cumsum`/`cumprod`, `argext` for `argmin`/`argmax`
//! -- for the two families that genuinely needed something new) or is
//! composed from those plus `binary_op` (`ptp`, `count_nonzero`), never a
//! numeric loop written in Python.
//!
//! `do_reduce_axis` below is a deliberate near-duplicate of the
//! same-named private driver in `ndarray_attrs.rs` (which backs the
//! `ndarray.sum`/`.prod`/`.all`/`.any`/`.min`/`.max` METHODS) rather than a
//! shared import: that file is a high-churn/shared-ownership file per
//! PATH-TO-100.md's file fence, and `ndarray_attrs.rs` itself already notes
//! (see its module doc) that `squeeze`/`swapaxes` duplicate rather than
//! import from `creation.rs` for the identical reason -- this file follows
//! the same established precedent instead of introducing a new one.
//!
//! `nanmin`/`nanmax`/`nansum`/`nanprod`/`nanmean`/`var`/`std`/`nanvar`/
//! `nanstd`/`average` were added in a later pass, built on `isnan_array`/
//! `nan_fill`/`overwrite_nan_where`/`overwrite_scalar_where`/`any_true`
//! (`ufunc.rs`) layered on top of the SAME `reduce_axis`/`binary_op`
//! kernels the rest of this file already uses -- no new Rust arithmetic
//! primitive besides those five small NaN-handling helpers.
//!
//! Still explicitly OUT OF SCOPE: `median`/`percentile`/`quantile` and
//! their `nan*` counterparts (need a new sort/partition/selection kernel
//! -- none exists anywhere in the crate, confirmed via grep; PATH-TO-100.md
//! places this as a later, separate roadmap item).
//!
//! complex64/complex128 input to `var`/`std`/`nanvar`/`nanstd` (added
//! 2026-08-02, statistics-grid DEFECT 1): real numpy's complex variance
//! takes the `_complex_to_float` fast path in `_core/_methods.py::_var` --
//! `|deviation|^2 = re^2 + im^2`, landing on the companion REAL dtype
//! (`complex64 -> float32`, `complex128 -> float64`) by default, or
//! whatever `dtype=` was explicitly requested (including a REAL `dtype=`,
//! which reproduces numpy's own mean-loses-the-imaginary-part quirk since
//! the mean sum is computed with that real accumulator dtype BEFORE the
//! deviation step -- see `do_var`'s own inline comments for the full
//! trace). `complex_abs_sq` below is the one new primitive this needed;
//! everything else reuses the existing `reduce_axis`/`binary_op` pipeline
//! unchanged.
//!
//! `nanvar`/`nanstd` deviation-truncation (added 2026-08-02, statistics-grid
//! DEFECT 2): real numpy's `_nanfunctions_impl.py::nanvar` computes the
//! mean-subtracted deviation via `np.subtract(arr, avg, out=arr, ...)`
//! where `arr` is the nan-filled COPY of the original input -- so `out=arr`
//! is a genuine in-place write that truncates the (possibly wider,
//! `dtype=`-driven) subtraction result back down to the ORIGINAL input
//! dtype (e.g. float16) before it is ever squared. Only the final
//! `np.sum(sqr, dtype=dtype)` accumulates at the requested/target dtype.
//! Plain `var`/`std` (`_core/_methods.py::_var`) has no such truncation --
//! its own `out=` is literally `Ellipsis`, i.e. a fresh full-precision
//! array. `do_var` below replicates this bug-for-bug: the nan-aware path
//! casts `x` back to the (nan-filled) input's own dtype right after the
//! subtract, the plain path does not.

use pyo3::exceptions::{PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use ionp_core::sort::{
    argsort_axis, argwhere as core_argwhere, extract as core_extract, flatnonzero as core_flatnonzero,
    lexsort as core_lexsort,
    nanargext, nonzero as core_nonzero, normalize_single_axis, searchsorted as core_searchsorted, sort_axis,
    sort_complex as core_sort_complex, where_select, NanExtremeOp,
};
use ionp_core::ufunc::{
    accumulate_axis, any_true, argext, binary_op, isnan_array, math_unary_op, nan_fill,
    normalize_reduce_axes, overwrite_nan_where, reduce_axis, BinaryOp, ExtremeOp, MathUnaryOp,
};
use ionp_core::{creation, promote_dtype, Buffer, DType, IonpError, NdArray, Order};

use crate::{
    axis_error, extract_array_like, to_py_err,
    write_into_out, PyArray,
};
use crate::fpstate;

// ---------------------------------------------------------------------------
// Shared argument-parsing helpers (deliberately duplicated from
// ndarray_attrs.rs -- see module doc above for why).
// ---------------------------------------------------------------------------

/// `std`/`nanstd`'s own ufunc-output-casting `TypeError`, local copy of the
/// same message-shape `lib.rs::ufunc_output_casting_err` / `fft.rs`'s
/// `fft_output_casting_err` already build (see those doc comments for why
/// this project keeps one small local copy per file rather than importing
/// across the file-ownership fence): real numpy's `_std` (`_core/_methods.py`)
/// runs `um.sqrt(ret, out=ret)` -- an explicit `out=` with the DEFAULT
/// `casting='same_kind'` -- whenever `ret` is a genuine `ndarray` instance,
/// which raises `numpy._core._exceptions._UFuncOutputCastingError` (a
/// private `TypeError` subclass) for any non-inexact target dtype, since
/// `sqrt`'s natural output is always float64-or-wider and float64->int is
/// never `same_kind`. Verified against real numpy 2.5.1: `np.std(a, dtype=np.int64)`
/// with NO axis/keepdims (a genuine full reduction, which numpy's own
/// `_var`/`_std` internals return as a bare numpy SCALAR, not an ndarray)
/// succeeds and truncates (`ret.dtype.type(um.sqrt(ret))`, no casting-rule
/// check at all on that branch) -- but the exact same call WITH `axis=`
/// specified (even a full-covering axis like `axis=0` on a 1-D array) or
/// with `keepdims=True` returns a genuine ndarray and raises. anionpy's own
/// `wrap()` always returns a `PyArray`, never a bare Python/numpy scalar, so
/// this raise is gated here on the same "would numpy's own internals still
/// call it a scalar" rule -- full reduction (`axes.len() == ndim`) AND
/// `!keepdims` -- rather than on anionpy's own (always-array) result shape.
fn std_sqrt_output_casting_err(to: DType) -> PyErr {
    // BUG FOUND AND FIXED 2026-08-02: plain `PyTypeError` -> real numpy's
    // class here is its private `_UFuncOutputCastingError`, which displays as
    // `"UFuncTypeError"` -- same systemic class bug as
    // `lib.rs::ufunc_output_casting_err`/`fft.rs::fft_output_casting_err`
    // (see those doc comments), fixed the same way: raise the crate's own
    // `UFuncTypeError` (a real `TypeError` subclass) instead of a plain one.
    //
    // Statistics-grid task (2026-08-02): this used to hardcode the "from"
    // side of the message as the literal string `'float64'` -- correct only
    // by coincidence for the one `to` dtype this call site was ever reached
    // with (an `int32`/`int64`-family `dtype=` override, whose minimal safe
    // sqrt loop really is float64). Generalized to `std_sqrt_casting_err`
    // below (which computes the correct "from" dtype for ANY `to` via
    // `sqrt_input_promoted_dtype`) so the new `out=`-driven call site (which
    // legitimately hits non-float64-"from" cases, e.g. a `bool` `out=`
    // promotes from `float16`, verified against real numpy 2.5.1:
    // `np.sqrt(np.ones((), dtype='int8'), out=np.zeros((), dtype='bool'))`
    // raises `"...from dtype('float16') to dtype('bool')..."`) can share the
    // same helper instead of duplicating a second hardcoded-'float64'
    // mistake. This wrapper is kept as a thin compatibility shim for the
    // existing `dtype=`-driven call site below, which only ever passes
    // dtypes whose promoted-from IS float64 anyway.
    std_sqrt_casting_err(sqrt_input_promoted_dtype(to), to)
}

/// numpy's own minimal "safe" sqrt ufunc loop selection for a given INPUT
/// dtype -- i.e. the dtype real numpy's `np.sqrt` actually computes in
/// (and so the "from" side of any `Cannot cast ufunc 'sqrt' output from
/// dtype(...) to dtype(...)` message) when asked to operate on/into a
/// buffer of dtype `d`. Verified directly against real numpy 2.5.1 via
/// `np.sqrt(np.ones((), dtype=X), out=np.zeros((), dtype='bool'))` for
/// every `X` anionpy supports (bool always fails to receive any sqrt result,
/// so the "from" side of the raised message reveals exactly this table):
/// `bool/int8/uint8 -> float16`, `int16/uint16 -> float32`,
/// `int32/uint32/int64/uint64 -> float64`, and every already-inexact dtype
/// maps to itself (sqrt has a native loop for each of float16/32/64 and
/// complex64/128, no promotion needed).
fn sqrt_input_promoted_dtype(d: DType) -> DType {
    match d {
        DType::Bool | DType::I8 | DType::U8 => DType::F16,
        DType::I16 | DType::U16 => DType::F32,
        DType::I32 | DType::U32 | DType::I64 | DType::U64 => DType::F64,
        DType::F16 => DType::F16,
        DType::F32 => DType::F32,
        DType::F64 => DType::F64,
        DType::C64 => DType::C64,
        DType::C128 => DType::C128,
        DType::S(_) | DType::U(_) => unreachable!(
            "sqrt is rejected for string dtypes before this promotion table is consulted"
        ),
    }
}

/// Full two-dtype form of `std_sqrt_output_casting_err`, for call sites
/// that already know both the "from" (sqrt's own promoted input/output
/// dtype) and "to" (the `out=`/`ret` buffer sqrt must same-kind-cast into)
/// sides explicitly rather than needing them derived from a single `to`.
fn std_sqrt_casting_err(from: DType, to: DType) -> PyErr {
    crate::UFuncTypeError::new_err(format!(
        "Cannot cast ufunc 'sqrt' output from dtype('{}') to dtype('{}') with casting rule 'same_kind'",
        from.name(),
        to.name()
    ))
}

fn axes_list_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(seq) = obj.extract::<Vec<isize>>() {
        Ok(seq)
    } else {
        Ok(vec![obj.extract::<isize>()?])
    }
}

/// `axis=` marshalling for the REDUCTION family specifically, where a 0-d
/// operand has two boundaries that `axes_list_from_pyobj` cannot see because
/// it has already flattened `axis=0` and `axis=(0,)` into the same `[0]`.
///
/// BUG FOUND + FIXED 2026-08-03 (Monday). Measured against real numpy 2.5.1
/// on a 0-d operand (`/tmp/mg_0d_table.py`, 29 functions), NOT derived from
/// any rule -- because there is no rule. numpy is genuinely inconsistent
/// here and this wrapper's job is to REPRODUCE the inconsistency, not to
/// tidy it up:
///
///   1. THE SCALAR/SEQUENCE SPLIT. `np.sum(np.array(3.0), axis=0)` succeeds
///      (returns the scalar) but `np.sum(np.array(3.0), axis=(0,))` raises
///      `AxisError` -- the SAME axis value, accepted as a bare int and
///      rejected as a one-element tuple. `axis=()` succeeds (it reduces over
///      nothing). So the 0-d courtesy belongs to the scalar form only, and
///      any non-empty sequence is bounds-checked strictly.
///
///   2. THE NAN SPLIT. `mean`/`var`/`std`/`median`/`average`/`size` raise
///      `AxisError` on a 0-d operand even in the scalar form, while their
///      own `nan*` siblings -- `nanmean`/`nanvar`/`nanstd` -- ACCEPT it.
///      Same operand, same kwarg, opposite answers, differing only by the
///      nan-awareness of the caller. Hence `zero_d_courtesy`.
///
///   3. THE NAN SPLIT IS ALSO A DTYPE SPLIT (measured 2026-08-03, Monday).
///      Rule 2 is only half the rule. The courtesy holds for INEXACT
///      operands (float16/32/64, complex64/128) and is WITHDRAWN for bool
///      and every integer width, where `nanmean(np.array(3), axis=0)`
///      raises `AxisError` exactly like plain `mean`. The mechanism is
///      visible in numpy's own source: `_replace_nan` returns a null mask
///      for non-inexact dtypes, so `nanmean`/`nanvar`/`nanstd` delegate
///      verbatim to `mean`/`var`/`std` and inherit their strict axis
///      validation; only the inexact path reaches the nan-aware code that
///      tolerates the axis. Measured across float16/32/64, complex64/128,
///      int32/64, uint8 and bool -- the split is clean, with no dtype
///      landing between the two behaviours.
///
///      So `zero_d_courtesy` is `nan_aware && dtype.is_inexact()`, never
///      `nan_aware` alone. Callers must AND in the operand dtype; passing
///      the bare flag re-opens the integer hole.
///
/// This is deliberately a SEPARATE function rather than a widened
/// `axes_list_from_pyobj`: that one is also used by `squeeze` (see
/// `ndarray_attrs.rs`) and by `moveaxis`/`expand_dims` (`creation.rs`),
/// which normalize through `manip::normalize_axis` and must NOT inherit
/// reduction strictness.
///
/// numpy names the FIRST offending axis in the message and reports the
/// operand's real dimension (0), verified for `(0,)`, `(-1,)`, `(0, 0)`,
/// `(0, -1)` and `(1,)`.
/// A 0-d array used as an `axis=`: `Some(...)` when `obj` IS a 0-d array
/// (numpy's or anionpy's), carrying either the integer axis it denotes or the
/// TypeError numpy raises for a non-integer one; `None` when `obj` is not
/// a 0-d array at all.
///
/// Needed because `obj.extract::<Vec<isize>>()` SUCCEEDS on a 0-d anionpy
/// array by way of the sequence protocol and yields an EMPTY vector, so
/// `nanmedian(a, axis=anionpy.array(0))` silently became `axis=()` and
/// returned the operand unreduced where numpy reduces (measured
/// 2026-08-04: `np.nanmedian(np.zeros((0, 3)), axis=np.array(0)).shape`
/// is `(3,)`). The non-integer case is numpy's, same measurement:
/// `axis=np.array(0.0)` -> TypeError: only integer scalar arrays can be
/// converted to a scalar index.
fn zero_d_array_axis(obj: &Bound<'_, PyAny>) -> Option<PyResult<isize>> {
    if crate::stats::axis_array_ndim(obj) != Some(0) {
        return None;
    }
    Some((|| {
        let item = obj.call_method0("item")?;
        // `is_instance_of::<PyInt>()` is true for `bool` as well, and a 0-d
        // BOOL array is not an integer array for indexing purposes.
        if item.is_instance_of::<pyo3::types::PyInt>()
            && !item.is_instance_of::<pyo3::types::PyBool>()
        {
            item.extract::<isize>()
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "only integer scalar arrays can be converted to a scalar index",
            ))
        }
    })())
}

/// Which of numpy's two sequence policies a caller's axis converter uses.
///
/// numpy does not have one axis converter. The reduce/mean family goes
/// through `PyArray_ConvertMultiAxis`, which accepts a TUPLE and nothing
/// else; `average` goes through `normalize_axis_tuple`, which iterates any
/// sequence. The difference is observable and wide. Measured 2026-08-04
/// against numpy 2.5.1 on `np.zeros((2, 3))`:
///     np.sum(a,     axis=[0])      -> TypeError: 'list' object cannot be
///                                     interpreted as an integer
///     np.sum(a,     axis=range(1)) -> TypeError: 'range' object ...
///     np.sum(a,     axis=b"\x00")  -> TypeError: 'bytes' object ...
///     np.average(a, axis=[0])      -> OK, shape (3,)
/// anionpy accepted ALL of the rejected forms and reduced over axis 0 -- a
/// silent wrong ANSWER, not a wrong message, and it reached the `ndarray`
/// method forms (`a.sum(axis=[0])`) identically.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum AxisSeq {
    /// `PyArray_ConvertMultiAxis`: a tuple is the only sequence accepted.
    TupleOnly,
    /// `normalize_axis_tuple`: any sequence is iterated. `average` only.
    AnySeq,
}

pub(crate) fn reduce_axes_from_pyobj(
    obj: &Bound<'_, PyAny>,
    ndim: usize,
    zero_d_courtesy: bool,
    seq: AxisSeq,
) -> PyResult<Vec<isize>> {
    // BEFORE the `Vec<isize>` attempt, not after: extracting a vector from a
    // 0-d array SUCCEEDS via the sequence protocol and yields an EMPTY one,
    // which reads downstream as "reduce over no axes" -- so the failure was
    // a silently unreduced result, not an error. `zero_d_array_axis` has the
    // measurement.
    // A bool is NOT an axis index on this converter, and unlike the
    // `mean` family's converter there is no range check in front of it --
    // the rejection is unconditional. Measured 2026-08-04, numpy 2.5.1:
    //     np.sum(np.zeros((5,)),   axis=True)  -> TypeError: an integer is required
    //     np.sum(np.zeros((2, 3)), axis=True)  -> TypeError: an integer is required
    //     np.mean(np.zeros((5,)),  axis=True)  -> AxisError: axis 1 is out of bounds
    // (`mean`/`var`/`std` reach their own `mean_family_axis_check` before
    // this function and never fall through to here with a bool.)
    // PyO3's `extract::<Vec<isize>>`/`extract::<isize>()` happily converts
    // `True` to 1, so without this the axis was silently ACCEPTED.
    // Gated to the strict family: `normalize_axis_tuple` (hence `average`)
    // ACCEPTS a Python bool, because it reaches `operator.index(True) == 1`.
    // Measured 2026-08-04: np.average(np.ones((2, 3)), axis=True) -> shape
    // (2,), i.e. reduced over axis 1.
    if seq == AxisSeq::TupleOnly && is_bool_axis(obj) {
        return Err(pyo3::exceptions::PyTypeError::new_err("an integer is required"));
    }
    // An ndim>=1 array is rejected outright. This one was returning a
    // WRONG ANSWER, not a wrong message: `np.sum(a, axis=np.array([0]))`
    // raises in numpy, while anionpy's `Vec<isize>` extraction read the
    // array as the sequence `[0]` and reduced over axis 0.
    if let Some(n) = crate::stats::axis_array_ndim(obj) {
        if n >= 1 {
            return Err(scalar_index_axis_err());
        }
    }
    if let Some(res) = zero_d_array_axis(obj) {
        let ax = res?;
        return reduce_axes_from_pyobj_tail(vec![ax], false, ndim, zero_d_courtesy);
    }
    // A tuple/list ALWAYS takes the per-element path below, never PyO3's
    // whole-container `Vec<isize>` extraction. That extraction succeeds on
    // `(True,)` -- Rust reads the bool as 1 -- so a container-first branch
    // silently accepted an axis numpy rejects, and no element check placed
    // after it could ever run. Going element-first also keeps numpy's
    // LEFT-TO-RIGHT blame order: `np.sum(a, axis=(0.0, True))` names the
    // float, `(True, 0.0)` names the bool, because numpy's own
    // `tuple([operator.index(x) for x in axis])` stops at the first
    // failure. Measured 2026-08-04.
    //
    // On `AxisSeq::TupleOnly` a NON-tuple never reaches a sequence path at
    // all -- not the element loop, and not PyO3's `Vec<isize>` container
    // extraction either. That extraction is what silently accepted
    // `axis=[0]` / `range(1)` / `b"\x00"` / any `__getitem__` object and
    // reduced over axis 0 where numpy raises by container type name; see
    // `AxisSeq`'s doc comment for the measurements. Dropping the whole
    // `Vec<isize>` attempt (rather than filtering after it) is deliberate:
    // there is no input the strict converter accepts that only that
    // extraction can see, now that the 0-d array case is hoisted above.
    let is_tuple = obj.is_instance_of::<pyo3::types::PyTuple>();
    let is_seq = is_tuple
        || (seq == AxisSeq::AnySeq && obj.is_instance_of::<pyo3::types::PyList>());
    let try_container = match seq {
        AxisSeq::TupleOnly => false,
        AxisSeq::AnySeq => !is_seq,
    };
    let (list, was_sequence) = match if try_container { obj.extract::<Vec<isize>>().map_err(|_| ()) } else { Err(()) } {
        Ok(seq) => (seq, true),
        Err(_) => {
            // A SEQUENCE that failed to convert must be blamed on its
            // offending ELEMENT, not on the sequence itself. numpy's
            // `normalize_axis_tuple` tries `operator.index(axis)` first and,
            // when that raises, falls through to
            // `tuple([operator.index(a) for a in axis])` -- so the TypeError
            // that escapes names the type of the bad element. Measured live
            // against numpy 2.5.1 on 2026-08-04:
            //     np.nanmedian(a, axis=(0.0,)) -> 'float' object cannot be
            //                                     interpreted as an integer
            //     np.nanmedian(a, axis=('x',)) -> 'str'   object ...
            // This file used to let PyO3's own failed `extract::<isize>()`
            // on the CONTAINER produce the message, which said 'tuple' in
            // both cases -- right shape, wrong subject, and 588 sweep cases
            // wide.
            if is_seq {
                let mut vals: Vec<isize> = Vec::new();
                for item in obj.try_iter()? {
                    let item = item?;
                    // A 0-d array ELEMENT is a scalar axis, not a nested
                    // empty sequence -- same defect as the container case
                    // handled by `zero_d_array_axis`, one level down.
                    // `(True,)` / `(np.True_,)`: same unconditional bool
                    // rejection as the bare form, one level down.
                    // Measured: np.sum(a, axis=(True,)) -> TypeError:
                    // an integer is required.
                    // Strict family only -- see the gate on the bare form
                    // above. `average`'s converter takes `(True,)` to axis 1.
                    if seq == AxisSeq::TupleOnly && is_bool_axis(&item) {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "an integer is required",
                        ));
                    }
                    // An ndim>=1 array ELEMENT gets the scalar-index
                    // message, not the generic type-name one -- same rule
                    // as the bare form above, one level down. Measured:
                    //     np.sum(a, axis=(np.array([0]),)) -> TypeError:
                    //       only integer scalar arrays can be converted
                    //       to a scalar index
                    // anionpy said "'numpy.ndarray' object cannot be
                    // interpreted as an integer" (130 sweep cases, every
                    // reduce-family item).
                    if let Some(n) = crate::stats::axis_array_ndim(&item) {
                        if n >= 1 {
                            return Err(scalar_index_axis_err());
                        }
                    }
                    if let Some(res) = zero_d_array_axis(&item) {
                        vals.push(res?);
                        continue;
                    }
                    match item.extract::<isize>() {
                        Ok(v) => vals.push(v),
                        Err(_) => {
                        // FULLY QUALIFIED (`numpy.float32`, not `float32`):
                        // numpy formats `tp_name`, which carries the module
                        // for every extension type. Measured 2026-08-04.
                        let tn = item
                            .get_type()
                            .fully_qualified_name()
                            .map(|n| n.to_string())
                            .unwrap_or_else(|_| "object".to_string());
                        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                            "'{tn}' object cannot be interpreted as an integer"
                        )));
                        }
                    }
                    // numpy raises the 0-d "sequence axis" AxisError as
                    // soon as it has ONE converted element -- it does not
                    // finish converting the rest first. The order is
                    // observable whenever a later element is itself bad.
                    // Measured 2026-08-04 on `np.zeros(())`:
                    //     np.sum(a, axis=(0, 0.0)) -> AxisError: axis 0 is
                    //         out of bounds for array of dimension 0
                    // anionpy converted the whole tuple first and so reported
                    // the float's TypeError instead (32 sweep cases, every
                    // reduce-family item). The same rule lives in
                    // `reduce_axes_from_pyobj_tail` for the sequences that
                    // convert cleanly; this is that rule, hoisted to
                    // numpy's actual firing point.
                    if ndim == 0 {
                        if let Some(&first) = vals.first() {
                            return Err(to_py_err(IonpError::AxisError {
                                axis: first,
                                ndim: Some(0),
                            }));
                        }
                    }
                }
                // Every element converted (or the tuple was empty) -- this really is a sequence of
                // axes, it just contained something PyO3's `Vec<isize>`
                // extraction could not see through (a 0-d array).
                (vals, true)
            } else if let Some(res) = zero_d_array_axis(obj) {
                (vec![res?], false)
            } else {
                (vec![obj.extract::<isize>()?], false)
            }
        }
    };
    reduce_axes_from_pyobj_tail(list, was_sequence, ndim, zero_d_courtesy)
}

/// The shared tail of `reduce_axes_from_pyobj`: the 0-d-operand courtesy
/// rule, factored out so the early 0-d-ARRAY-axis path above reaches it
/// too instead of bypassing it.
fn reduce_axes_from_pyobj_tail(
    list: Vec<isize>,
    was_sequence: bool,
    ndim: usize,
    zero_d_courtesy: bool,
) -> PyResult<Vec<isize>> {
    if ndim == 0 {
        if let Some(&first) = list.first() {
            if was_sequence || !zero_d_courtesy {
                return Err(to_py_err(IonpError::AxisError { axis: first, ndim: Some(0) }));
            }
            // BUG FOUND + FIXED 2026-08-03 (Monday). The courtesy branch used
            // to return `list` unvalidated, so ANY scalar axis was accepted on
            // a 0-d operand -- `axis=1`, `axis=-99`, anything. That went
            // unnoticed because every other caller passes the axes on to the
            // core, which range-checks them itself. `nanmean` is the one
            // caller that uses this function purely as a VALIDATOR and then
            // delegates with `axis=None`, so for `nanmean` alone the missing
            // check was load-bearing: `nanmean(0d, axis=1)` returned a value
            // where real numpy raises AxisError (measured,
            // /tmp/mg_cumsum_msg.py).
            //
            // numpy's courtesy on a 0-d operand extends to exactly the two
            // axes that denote "the (nonexistent) first axis" -- 0 and -1 --
            // and nothing else. A validator that accepts what it is asked to
            // reject is worse than no validator, because callers trust it.
            if first != 0 && first != -1 {
                return Err(to_py_err(IonpError::AxisError { axis: first, ndim: Some(0) }));
            }
        }
    }
    Ok(list)
}

/// numpy's real `argmin`/`argmax` `axis=` only ever accepts a single int
/// or `None` -- never a tuple/list (verified against real numpy 2.5.1:
/// `np.argmin(a, axis=(0, 1))` raises `TypeError: 'tuple' object cannot be
/// interpreted as an integer`). Rejected here at the Python-marshaling
/// boundary rather than left to `normalize_reduce_axes` (which is itself
/// tuple-permissive, shared with the multi-axis reduce family) so the
/// wrapper's accepted signature actually matches numpy's, not a superset
/// of it.
///
/// BUG FOUND + FIXED (2026-08-01, reductions/ndarray_attrs/sort task):
/// this used to raise a fixed, generic `"an integer is required (axis must
/// be an int or None)"` message regardless of what was actually passed.
/// Real numpy's message names the ACTUAL offending type
/// (`'tuple' object cannot be interpreted as an integer`,
/// `'list' object cannot be interpreted as an integer`, etc. -- verified
/// directly against real numpy 2.5.1 for tuple/list/str/float axis
/// arguments), mirroring the same `'{type}' object cannot be interpreted
/// as an integer` pattern `creation.rs`'s `shape_from_pyobj` already uses
/// for its own not-an-integer case. 234-failing-case sweep only ever
/// exercises the tuple form (`axis=(0, 1)`), but the fix is general rather
/// than hardcoded to `'tuple'` specifically, since the wrong fixed string
/// was itself the bug.
/// `mean`/`std`/`var` do NOT reach their `axis=` argument through the same
/// C converter as `sum`/`prod`/`min`/`max`/`any`/`nansum`/... do, and the
/// difference is visible in the error text. Measured live against numpy
/// 2.5.1 on 2026-08-04, on `np.zeros((2, 3))`:
///
///     np.sum (a, axis=0.0)     -> 'float' object cannot be interpreted ...
///     np.mean(a, axis=0.0)     -> integer argument expected, got float
///     np.sum (a, axis=(0.0,))  -> 'float' object cannot be interpreted ...
///     np.mean(a, axis=(0.0,))  -> integer argument expected, got float
///
/// and, decisively, `np.mean(a, axis=np.float64(0))` ALSO says "integer
/// argument expected, got float" while `np.mean(a, axis=np.float32(0))`
/// says "'numpy.float32' object cannot be interpreted as an integer". The
/// split is `isinstance(x, float)` -- `np.float64` subclasses Python's
/// `float`, `np.float32` does not -- so this is `PyArray_PyIntAsInt`'s
/// explicit float branch, not a duck-typed `__index__` call.
///
/// The scan is IN ORDER and does the full per-element check, not just the
/// float test, because precedence is observable: `axis=('x', 0.0)` must
/// report the `str`, which is the element numpy trips on first.
/// The `nan*` family's own axis-element rule, measured against numpy 2.5.1
/// on 2026-08-04 -- deliberately NOT the same as `mean_family_axis_check`'s,
/// because numpy's two converters genuinely disagree:
///
///     axis form        np.mean                              np.nanmean
///     -------------    -----------------------------------  ----------------------
///     0.0              "integer argument expected, got      "'float' object cannot
///                       float"                               be interpreted as an
///                                                            integer"
///     np.float64(0)    "integer argument expected, got      "'numpy.float64' object
///                       float"                               cannot be ..."
///     np.True_         "'numpy.bool' object cannot be       "an integer is required"
///                       interpreted as an integer"
///     True             "an integer is required"             "an integer is required"
///     np.array(0.0)    "only integer scalar arrays can be   same
///                       converted to a scalar index"
///     np.float32(0)    "'numpy.float32' object cannot ..."  same
///     'x'              "'str' object cannot be ..."         same
///
/// (operand `np.zeros((0, 3))`, i.e. the inexact branch where `nanmean`
/// does NOT delegate to `mean`.) This path is also what the `nan*` stats
/// items reach on an empty inexact operand, which is how 6 corpus cases
/// each on `nanmedian`/`nanquantile`/`nanpercentile` found it.
fn nan_family_axis_check(obj: &Bound<'_, PyAny>) -> PyResult<()> {
    fn one(item: &Bound<'_, PyAny>) -> PyResult<()> {
        // Bools FIRST: `bool` is an `int` subclass and `numpy.bool` is not,
        // yet numpy rejects BOTH here with the same C-arg-parser message,
        // so neither may be allowed to fall through to the index attempt.
        let type_name = item
            .get_type()
            .fully_qualified_name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "object".to_string());
        if item.is_instance_of::<pyo3::types::PyBool>() || type_name == "numpy.bool" {
            return Err(pyo3::exceptions::PyTypeError::new_err("an integer is required"));
        }
        match crate::stats::axis_array_ndim(item) {
            Some(0) => {
                zero_d_array_axis(item).expect("0-d array")?;
                return Ok(());
            }
            Some(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "only integer scalar arrays can be converted to a scalar index",
                ));
            }
            None => {}
        }
        if item.extract::<isize>().is_err() {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "'{type_name}' object cannot be interpreted as an integer"
            )));
        }
        Ok(())
    }
    if obj.is_none() {
        return Ok(());
    }
    // A 0-d array is checked as a SCALAR here, not iterated: numpy reports
    // "only integer scalar arrays can be converted to a scalar index" for
    // `axis=np.array(0.0)` on this path, not the "iteration over a 0-d
    // array" that `median`'s converter produces for the same object.
    // TUPLE ONLY -- deliberately NOT `|| is_instance_of::<PyList>()`, which
    // is what this said until 2026-08-04. The family behind this check is
    // `PyArray_ConvertMultiAxis`, for which a tuple is the only sequence
    // that exists; numpy never INSPECTS a list's contents here, it rejects
    // the container by type and stops. Measured 2026-08-04, numpy 2.5.1,
    // `np.zeros((2, 3))` -- for mean/var/std AND nanmean/nanvar/nanstd,
    // every one of these gives the identical message "'list' object cannot
    // be interpreted as an integer":
    //     [0]   [0.0]   [0, 1]   [True]   [0, 'x']
    // (`nanmedian` is NOT in this family -- it goes through
    // `normalize_axis_tuple`, which iterates lists happily: `axis=[0, 1]`
    // -> shape (). Do not unify them.)
    // Iterating a list here was harmless only while these loops had no
    // check that could fire before the container-type rejection downstream
    // in `reduce_axes_from_pyobj`. Adding the element range check gave the
    // mean one such a check, and `np.mean(a, axis=[0, 1])` immediately
    // started answering "axis 1 is out of bounds". Caught by sweep.
    if obj.is_instance_of::<pyo3::types::PyTuple>() {
        for item in obj.try_iter()? {
            one(&item?)?;
        }
        return Ok(());
    }
    one(obj)
}

fn mean_family_axis_check(obj: &Bound<'_, PyAny>, ndim: usize) -> PyResult<()> {
    fn one(item: &Bound<'_, PyAny>, ndim: usize) -> PyResult<()> {
        if item.is_instance_of::<pyo3::types::PyFloat>() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "integer argument expected, got float",
            ));
        }
        // A PYTHON bool only. `np.mean(a, axis=True)` is "an integer is
        // required" while `np.mean(a, axis=np.True_)` is the generic
        // "'numpy.bool' object cannot be interpreted as an integer" -- the
        // two are NOT interchangeable on this converter, and the `nan*`
        // one collapses them (see `nan_family_axis_check`). Measured
        // 2026-08-04.
        // ...and the numpy bool must be REJECTED HERE, explicitly, rather
        // than left to fall through. It used to reach PyO3's own failed
        // extraction downstream and pick up the generic message by
        // accident; once `reduce_axes_from_pyobj` grew an unconditional
        // bool rejection of its own (correct for `sum`, wrong for `mean`)
        // that accident stopped happening and `mean`/`var`/`std` started
        // answering "an integer is required" where numpy says
        // "'numpy.bool' object cannot be interpreted as an integer".
        if !item.is_instance_of::<pyo3::types::PyBool>() && is_bool_axis(item) {
            let tn = item
                .get_type()
                .fully_qualified_name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "object".to_string());
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "'{tn}' object cannot be interpreted as an integer"
            )));
        }
        if item.is_instance_of::<pyo3::types::PyBool>() {
            // ORDER MATTERS, and it is not the obvious one: numpy converts
            // the bool to an axis index and RANGE-CHECKS it first, only
            // then re-parses it with an integer-format converter that
            // rejects bools. Measured 2026-08-04 with `np.mean`:
            //     np.zeros((0, 3)), axis=True -> TypeError: an integer is required
            //     np.zeros((5,)),   axis=True -> AxisError: axis 1 is out of
            //                                    bounds for array of dimension 1
            // Raising the TypeError eagerly gets the 1-d case wrong, which
            // is exactly the regression an earlier draft of this check
            // introduced on `mean`/`var`/`std`/`nanmean`.
            let raw = if item.extract::<bool>().unwrap_or(false) { 1 } else { 0 };
            if raw >= ndim as isize {
                return Err(to_py_err(IonpError::AxisError { axis: raw, ndim: Some(ndim) }));
            }
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "an integer is required",
            ));
        }
        // An array of ndim >= 1 is not a scalar index and this converter
        // does not iterate it (unlike `median`'s, which accepts
        // `axis=np.array([0])` happily). Measured 2026-08-04:
        // `np.mean(a, axis=np.array([0]))` -> TypeError: only integer
        // scalar arrays can be converted to a scalar index.
        match crate::stats::axis_array_ndim(item) {
            Some(0) => {
                if let Some(res) = zero_d_array_axis(item) {
                    res?;
                }
            }
            Some(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "only integer scalar arrays can be converted to a scalar index",
                ));
            }
            None => {}
        }
        Ok(())
    }
    if obj.is_none() {
        return Ok(());
    }
    // TUPLE ONLY -- deliberately NOT `|| is_instance_of::<PyList>()`, which
    // is what this said until 2026-08-04. The family behind this check is
    // `PyArray_ConvertMultiAxis`, for which a tuple is the only sequence
    // that exists; numpy never INSPECTS a list's contents here, it rejects
    // the container by type and stops. Measured 2026-08-04, numpy 2.5.1,
    // `np.zeros((2, 3))` -- for mean/var/std AND nanmean/nanvar/nanstd,
    // every one of these gives the identical message "'list' object cannot
    // be interpreted as an integer":
    //     [0]   [0.0]   [0, 1]   [True]   [0, 'x']
    // (`nanmedian` is NOT in this family -- it goes through
    // `normalize_axis_tuple`, which iterates lists happily: `axis=[0, 1]`
    // -> shape (). Do not unify them.)
    // Iterating a list here was harmless only while these loops had no
    // check that could fire before the container-type rejection downstream
    // in `reduce_axes_from_pyobj`. Adding the element range check gave the
    // mean one such a check, and `np.mean(a, axis=[0, 1])` immediately
    // started answering "axis 1 is out of bounds". Caught by sweep.
    if obj.is_instance_of::<pyo3::types::PyTuple>() {
        for item in obj.try_iter()? {
            let item = item?;
            one(&item, ndim)?;
            match item.extract::<isize>() {
                Err(_) => {
                    let tn = item
                        .get_type()
                        .fully_qualified_name()
                        .map(|n| n.to_string())
                        .unwrap_or_else(|_| "object".to_string());
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "'{tn}' object cannot be interpreted as an integer"
                    )));
                }
                Ok(raw) => {
                    // BUG FOUND + FIXED (2026-08-04, axis-form sweep).
                    // numpy RANGE-CHECKS each element the moment it
                    // converts it -- it does not convert the whole
                    // sequence first. The order is observable whenever an
                    // out-of-range element precedes a bad-TYPE one.
                    // Measured 2026-08-04, numpy 2.5.1, `np.mean`:
                    //     shape ()      axis=(0, 0.0)   -> AxisError axis 0
                    //     shape (5,)    axis=(0, 0.0)   -> TypeError float
                    //     shape (5,)    axis=(5, 0.0)   -> AxisError axis 5
                    //     shape (5,)    axis=(0,1,0.0)  -> AxisError axis 1
                    //     shape (2, 3)  axis=(1, 0.0)   -> TypeError float
                    // anionpy converted the whole tuple before validating any
                    // of it and so always reported the float's TypeError.
                    // The 0-d rows are NOT a special case -- they are this
                    // same rule where every index is out of bounds, which
                    // is why this is a range check and not an `ndim == 0`
                    // guard. (`reduce_axes_from_pyobj` carries the same
                    // rule for the strict family; this is the mean
                    // family's copy, which it needs because its element
                    // TYPE checks differ and it never reaches that code.)
                    let n = ndim as isize;
                    let norm = if raw < 0 { raw + n } else { raw };
                    if norm < 0 || norm >= n {
                        return Err(to_py_err(IonpError::AxisError {
                            axis: raw,
                            ndim: Some(ndim),
                        }));
                    }
                }
            }
        }
        return Ok(());
    }
    one(obj, ndim)
}

/// True for a value numpy's axis converters treat as a BOOL rather than as
/// an integer -- Python `bool` and `numpy.bool` alike. `numpy.bool` is not
/// a subclass of Python's `bool` and does not define `__index__`, so
/// neither `is_instance_of::<PyBool>()` nor an `extract::<isize>()` sees
/// it; the type name is the only thing the two forms share.
fn is_bool_axis(obj: &Bound<'_, PyAny>) -> bool {
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return true;
    }
    obj.get_type()
        .fully_qualified_name()
        .map(|n| n.to_string())
        .map(|n| n == "numpy.bool" || n == "anionpy.bool")
        .unwrap_or(false)
}

fn scalar_index_axis_err() -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(
        "only integer scalar arrays can be converted to a scalar index",
    )
}

/// The SINGLE-axis converter, used by `argmin`/`argmax`/`cumsum` and
/// friends. This is NOT the same converter as `reduce_axes_from_pyobj`'s,
/// and the difference is not cosmetic -- numpy ships at least three of
/// them with genuinely different messages for the same input. Measured
/// live against numpy 2.5.1 on 2026-08-04, `np.zeros((2, 3))`:
///
///     axis=            np.argmin / np.cumsum            np.sum
///     ---------------  -------------------------------  ----------------------
///     True             "an integer is required for the   "an integer is required"
///                       axis"
///     np.True_         same as above                     "an integer is required"
///     np.array(0.)     "only integer scalar arrays can   (accepted -> AxisError
///                       be converted to a scalar index"   path, not a TypeError)
///     np.array([0])    same as above                     "only integer scalar
///                                                         arrays ..."
///     np.array(0)      ACCEPTED (axis 0)                 ACCEPTED
///     np.float64(0)    "'numpy.float64' object cannot be interpreted as an integer"
///     Decimal(0)       "'decimal.Decimal' object cannot be interpreted as an integer"
///
/// Note the last two: numpy formats the C-level `tp_name`, which carries
/// the MODULE. This function used `.name()` (bare `float64`, `Decimal`)
/// until 2026-08-04 -- right sentence, wrong noun, on every numpy scalar
/// and every extension type.
pub(crate) fn single_axis_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<isize> {
    // 0-d array: an integer one IS a valid axis, a float one gets numpy's
    // array-specific message. `zero_d_array_axis` carries both arms.
    if let Some(res) = zero_d_array_axis(obj) {
        return res;
    }
    // ndim >= 1 array: never an axis here, whatever it contains.
    if crate::stats::axis_array_ndim(obj).is_some() {
        return Err(scalar_index_axis_err());
    }
    if is_bool_axis(obj) {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "an integer is required for the axis",
        ));
    }
    obj.extract::<isize>().map_err(|_| {
        // FULLY QUALIFIED, see the table above.
        let tn = obj
            .get_type()
            .fully_qualified_name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "object".to_string());
        pyo3::exceptions::PyTypeError::new_err(format!("'{tn}' object cannot be interpreted as an integer"))
    })
}

#[allow(clippy::too_many_arguments)]
/// `keepdims=` on the value-returning reduction family (`sum`/`prod`/
/// `all`/`any`/`min`/`max`/`amin`/`amax`/`mean`/`ptp`/`count_nonzero`/
/// `nansum`/`nanprod`/`nanmin`/`nanmax`/`nanmean`/`var`/`std`/`nanvar`/
/// `nanstd` -- NOT `argmin`/`argmax`/`nanargmin`/`nanargmax`, see below)
/// is parsed by real numpy's C argument machinery using integer-format
/// (`"i"`-style) semantics, NOT plain Python truthiness -- a genuinely
/// different defect shape than `svd`'s `hermitian`/`full_matrices`/
/// `compute_uv` (2026-08-02 bool-axis sweep, measured live against numpy
/// 2.5.1, not assumed uniform across the module: `np.sum`/`np.any`/
/// `np.mean`/`np.ptp`/`np.count_nonzero`/`np.var`/`np.std` and every
/// `nan*` sibling in this file ALL reject `keepdims=1.5` with the
/// C-arg-parser's specific `TypeError: "integer argument expected, got
/// float"` -- distinct from the generic `operator.index()` message this
/// module's own `single_axis_from_pyobj` uses for `axis=` -- and ALSO
/// reject `keepdims=np.bool_(True)` with `TypeError: "'numpy.bool' object
/// cannot be interpreted as an integer"` because `numpy.bool` does not
/// itself define `__index__`, even though it's the falsy/truthy-est type
/// imaginable. Plain Python `int`/`bool` and numpy INTEGER scalars
/// (`np.int32` etc, which do define `__index__`) all succeed, non-zero
/// meaning `True`. An object whose `__index__` itself raises propagates
/// that exception class unchanged (verified live with a raising
/// `__index__`), exactly like `operator.index()` would -- only the "no
/// `__index__` at all" and "is exactly a float" cases get numpy's own
/// synthesized messages here. `argmin`/`argmax`/`nanargmin`/`nanargmax`
/// were measured SEPARATELY (not assumed to share this) and found to be
/// plain truthy instead (accept `1.5`, accept a raising-`__bool__`
/// object's truthiness via a `__bool__` propagating exception, no
/// `__index__` requirement) -- they get the `is_truthy()` treatment at
/// their own call sites instead, same as `svd`.
fn opt_truthy(obj: Option<&Bound<'_, PyAny>>, default: bool) -> PyResult<bool> {
    match obj {
        None => Ok(default),
        Some(v) => v.is_truthy(),
    }
}
fn opt_index_like(obj: Option<&Bound<'_, PyAny>>, default: bool) -> PyResult<bool> {
    match obj {
        None => Ok(default),
        Some(v) => keepdims_index_like(v),
    }
}
fn keepdims_index_like(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if obj.is_instance_of::<pyo3::types::PyFloat>() {
        return Err(pyo3::exceptions::PyTypeError::new_err("integer argument expected, got float"));
    }
    match obj.getattr("__index__") {
        Err(_) => {
            let tn = obj
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "object".to_string());
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "'{tn}' object cannot be interpreted as an integer"
            )))
        }
        Ok(idx_fn) => {
            let v = idx_fn.call0()?;
            let n: i64 = v.extract()?;
            Ok(n != 0)
        }
    }
}

fn do_reduce_axis(
    py: Python<'_>,
    op: BinaryOp,
    a: &NdArray,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let ndim = a.ndim();
    let axes_raw: Option<Vec<isize>> = match axis {
        None => None,
        Some(ax) if ax.is_none() => None,
        // 0-d courtesy: do_reduce_axis: sum/prod/min/max/any/all/amax/amin/nansum/nanprod
        Some(ax) => Some(reduce_axes_from_pyobj(ax, ndim, true, AxisSeq::TupleOnly)?),
    };
    let axes = normalize_reduce_axes(axes_raw.as_deref(), ndim).map_err(to_py_err)?;

    let dtype_override: Option<DType> = match dtype {
        None => None,
        Some(d) if d.is_none() => None,
        // Numeric reduction target: no S/U arm downstream, decline cleanly.
        Some(d) => Some(crate::dtype_from_pyobj_no_su(d)?),
    };

    let initial_arr: Option<NdArray> = match initial {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => Some(extract_array_like(v)?),
    };

    let mask_arr: Option<NdArray> = match r#where {
        None => None,
        Some(w) => match w.extract::<bool>() {
            Ok(true) => None,
            Ok(false) => Some(extract_array_like(w)?),
            Err(_) => Some(extract_array_like(w)?),
        },
    };

    let result = reduce_axis(op, a, &axes, keepdims, dtype_override, initial_arr.as_ref(), mask_arr.as_ref())
        .map_err(to_py_err)?;

    wrap_reduction(py, result, out)
}

fn wrap(py: Python<'_>, inner: NdArray, out: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    if let Some(out_obj) = out {
        return write_into_out(out_obj, &inner, None);
    }
    use pyo3::IntoPyObjectExt;
    Py::new(py, PyArray { inner })?.into_py_any(py)
}

/// Same as `wrap` above, EXCEPT for the one case `wrap` gets wrong for a
/// genuine reduction: when the result has fully collapsed to 0 dimensions
/// (`inner.ndim() == 0`) and there is no `out=` target, real numpy returns a
/// numpy SCALAR (`np.float64`, `np.int64`, `np.bool_`, ...), never a 0-d
/// `ndarray` -- verified directly (`np.sum(np.array([1,2,3]))` is
/// `numpy.int64`, not `numpy.ndarray`), and true regardless of *how* the
/// result got to 0-d: `axis=None`, an explicit axis tuple covering every
/// axis, or an already-0-d input, all produce a scalar; `keepdims=True`
/// only prevents this when it actually keeps at least one (size-1)
/// dimension, which is exactly the `inner.ndim() == 0` test below -- a 0-d
/// INPUT with `keepdims=True` still has no dimension left to keep, and
/// still reduces to a scalar (verified: `np.sum(np.array(5),
/// keepdims=True)` is also `numpy.int64`, shape already `()`).
///
/// See docs/scalar-return-type-defect.md -- this is the fix for the 22
/// items documented there (`sum`, `mean`, `max`, `min`, `amax`, `amin`,
/// `ptp`, `argmax`, `argmin`, `std`, `var`, `nansum`, `nanmean`, `nanmax`,
/// `nanmin`, `count_nonzero`, `any`, `all`, `prod`, `trace`,
/// `matmul`(1-d,1-d), `vecdot`), applied at every reduction call site in
/// this file that can legitimately fully collapse. Deliberately NOT applied
/// to `wrap`'s other callers in this file (`cumsum`/`cumprod`,
/// `sort`/`sort_complex`, `lexsort`, `nonzero`/`flatnonzero`/`argwhere`,
/// `where`, `average`'s side outputs) -- those are not
/// reductions in this sense; numpy itself never scalarizes their output
/// (see each function's own real-numpy behavior), so routing them through
/// this scalarizing path would be a NEW, unmeasured defect, not a fix.
///
/// CORRECTION (2026-08-03): `searchsorted` was previously listed here too;
/// that claim was WRONG, not merely unmeasured -- verified live against
/// real numpy 2.5.1, `np.searchsorted(a, 2.0)` (a scalar `v` against a 1-D
/// `a`) returns a bare `numpy.int64`, not a 0-d `numpy.ndarray`. `wrap`
/// itself is still not the right fix (shared with genuinely
/// non-scalarizing callers, listed above); `searchsorted`'s own call site
/// now checks `result.ndim() == 0` and routes through `crate::
/// numpy_scalar_from_0d` directly instead of through `wrap` -- the same fix
/// pattern this doc describes for the reduction family, applied ad hoc at
/// that one call site rather than by broadening this function.
///
/// The scalar is built by extracting the single buffer element as a native
/// Python value via `elem_to_py` (same, already-verified element-extraction
/// path `.item()` uses) and handing it to the matching REAL `numpy` scalar
/// constructor (`numpy.float64`, `numpy.int64`, ...), looked up by
/// `DType::name()` (numpy's own short dtype name, e.g. "float64" -- and,
/// for `Bool`, numpy 2.x's own `numpy.bool` is the live scalar-type name,
/// identical object to the legacy `numpy.bool_` alias). This constructs the
/// wrapper type for a value anionpy itself already computed -- it does not ask
/// numpy to compute or validate the VALUE, only to mint the correctly-typed
/// container for it, which is the one thing a pure-Rust pyclass hierarchy
/// cannot forge: real numpy scalar types are numpy's own C-level classes,
/// and the differential harness's return-type check (harness.py,
/// 2026-08-02) requires `type(ionp_out) is type(np_out)` exactly, not
/// merely a same-named lookalike.
fn wrap_reduction(py: Python<'_>, inner: NdArray, out: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    if let Some(out_obj) = out {
        return write_into_out(out_obj, &inner, None);
    }
    if inner.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &inner);
    }
    use pyo3::IntoPyObjectExt;
    Py::new(py, PyArray { inner })?.into_py_any(py)
}

// ---------------------------------------------------------------------------
// Tier 1: thin wrappers on the existing `reduce_axis` kernel -- identical
// signature/behavior to the `ndarray` METHODS in `ndarray_attrs.rs`, just
// reachable as `anionpy.sum(x, ...)` instead of `x.sum(...)`. `amin`/`amax`
// are numpy's legacy top-level aliases for `min`/`max` (same signature,
// same function under the hood -- verified against real numpy 2.5.1:
// `np.amin is np.min` is actually False in real numpy, they're separate
// wrapper functions, but behaviorally identical including the same
// `initial=`/`where=` kwargs).
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, keepdims=None, initial=None, r#where=None))]
#[allow(clippy::too_many_arguments)]
fn sum(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    do_reduce_axis(py, BinaryOp::Add, &arr, axis, dtype, out, keepdims, initial, r#where)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, keepdims=None, initial=None, r#where=None))]
#[allow(clippy::too_many_arguments)]
fn prod(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    do_reduce_axis(py, BinaryOp::Multiply, &arr, axis, dtype, out, keepdims, initial, r#where)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=None, r#where=None))]
fn all(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    do_reduce_axis(py, BinaryOp::LogicalAnd, &arr, axis, None, out, keepdims, None, r#where)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=None, r#where=None))]
fn any(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    do_reduce_axis(py, BinaryOp::LogicalOr, &arr, axis, None, out, keepdims, None, r#where)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=None, initial=None, r#where=None))]
fn min(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    do_reduce_axis(py, BinaryOp::Minimum, &arr, axis, None, out, keepdims, initial, r#where)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=None, initial=None, r#where=None))]
fn max(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    do_reduce_axis(py, BinaryOp::Maximum, &arr, axis, None, out, keepdims, initial, r#where)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=None, initial=None, r#where=None))]
fn amin(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    do_reduce_axis(py, BinaryOp::Minimum, &arr, axis, None, out, keepdims, initial, r#where)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=None, initial=None, r#where=None))]
fn amax(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    do_reduce_axis(py, BinaryOp::Maximum, &arr, axis, None, out, keepdims, initial, r#where)
}

// ---------------------------------------------------------------------------
// Tier 2: `argmin`/`argmax` -- new index-returning kernel (`ufunc::argext`).
// ---------------------------------------------------------------------------

// `argmin`/`argmax`/`nanargmin`/`nanargmax`'s `keepdims=` -- measured
// SEPARATELY from the value-returning family above (2026-08-02 sweep):
// these four accept `keepdims=1.5` and a raising-`__bool__` object's
// truthiness with no `__index__` requirement at all (verified live
// against numpy 2.5.1), i.e. plain Python truthiness like `svd`'s
// `hermitian`, NOT the C `"i"`-format `keepdims_index_like` rule the
// `sum`/`mean`/`var`/etc. family above uses. Do not fold these into that
// helper.
#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=None))]
fn argmin(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    do_argext(py, ExtremeOp::ArgMin, a, axis, out, opt_truthy(keepdims, false)?)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=None))]
fn argmax(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    do_argext(py, ExtremeOp::ArgMax, a, axis, out, opt_truthy(keepdims, false)?)
}

fn do_argext(
    py: Python<'_>,
    op: ExtremeOp,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    let arr = extract_array_like(a)?;
    let ndim = arr.ndim();
    // `axis=None`: numpy flattens first (argmin/argmax of the WHOLE array,
    // a single scalar index into the flattened array) -- matches
    // `cumsum`/`cumprod`'s own `axis=None` handling below, ravel then
    // reduce axis 0 of the flattened 1-D result.
    let flattened = matches!(axis, None) || matches!(axis, Some(ax) if ax.is_none());
    let (target, axes): (NdArray, Vec<usize>) = match axis {
        None => {
            let flat = arr.ravel_order("C").map_err(to_py_err)?;
            (flat, vec![0])
        }
        Some(ax) if ax.is_none() => {
            let flat = arr.ravel_order("C").map_err(to_py_err)?;
            (flat, vec![0])
        }
        Some(ax) => {
            let raw = single_axis_from_pyobj(ax)?;
            let n = ndim as isize;
            // THIRD BUG (2026-08-01, coordinator fuzzing): a negative axis
            // must be offset against the courtesy effective ndim
            // (`n.max(1)`), not the real `n` -- `axis=-1` on a 0-d array
            // (`n == 0`) used to normalize to `-1 + 0 == -1`, which then
            // failed the `norm < 0` check below even though real numpy
            // accepts it (`np.argmax(np.array(3.0), axis=-1) == 0`, same
            // courtesy as `axis=0`). Same root cause as
            // `normalize_reduce_axes`'s sibling bug in ufunc.rs, fixed
            // there too.
            let norm = if raw < 0 { raw + n.max(1) } else { raw };
            // SECOND BUG (pre-existing, fixed here since argmin/argmax are
            // directly in scope for the sort/search block): this branch
            // used to raise a plain `pyo3::exceptions::PyIndexError` for
            // an out-of-range axis. Real numpy raises its own
            // `numpy.exceptions.AxisError` here (a `ValueError`+
            // `IndexError` multiple-inheritance subclass) -- verified
            // against real numpy 2.5.1: `np.argmax(a, axis=5)` raises
            // `numpy.exceptions.AxisError`, not builtins.IndexError. This
            // was the ONLY reason `argmin`/`argmax` were absent from the
            // ledger despite being otherwise fully implemented (see
            // `anionpy/_state/toplevel.py`'s documentation of this gap).
            //
            // FOURTH BUG (2026-08-01, this task): the `AxisError`'s own
            // `ndim` field must be the same "courtesy effective ndim"
            // (`n.max(1)`) as the bounds check just above, NOT the real
            // `ndim` -- probed directly against real numpy 2.5.1 across
            // 0d/1d/2d/3d inputs with several out-of-range axes (positive,
            // negative, `axis=100`): a 0-d array's `AxisError` message
            // always reports "for array of dimension 1", never "dimension
            // 0" (e.g. `np.argmax(np.array(3.0), axis=100)` ->
            // `'axis 100 is out of bounds for array of dimension 1'`), and
            // every ndim>=1 case reports its own real `ndim` unchanged.
            // So the rule is exactly `ndim.max(1)`, the same courtesy
            // already used to compute `n`/`norm` above -- not a special
            // case for 0-d specifically. `axis_error()` (`lib.rs`)
            // constructs numpy's REAL `AxisError` class and lets numpy
            // itself render the message from the `ndim` argument passed
            // in, so passing the corrected ndim here is sufficient; no
            // string formatting to fix.
            if norm < 0 || norm >= n.max(1) {
                return Err(to_py_err(IonpError::AxisError { axis: raw, ndim: Some(ndim.max(1)) }));
            }
            (arr, vec![norm as usize])
        }
    };
    let mut result = argext(op, &target, &axes, keepdims).map_err(to_py_err)?;
    // `axis=None` flattens `arr` to 1-D before reducing, so `argext`'s own
    // keepdims logic (ufunc.rs) only ever sees that 1-D intermediate and
    // produces shape `[1]` -- it has no way to know the ORIGINAL array's
    // ndim. numpy's `keepdims=True` with `axis=None` keeps the original
    // rank, all dims collapsed to size 1 (e.g. (4,6) -> (1,1), not (1,)).
    // Relabel the shape here, at the one call site that still knows both
    // "we flattened" and the true original ndim -- the underlying buffer
    // is a single scalar index either way, so this is a shape-only fix,
    // no data movement.
    //
    // THIRD BUG (found this task, fixed here): the guard used to read
    // `ndim > 1`, which also skips the ndim==0 case -- but a 0-d input
    // flattens to shape (1,) same as any other input, and `ndim > 1` never
    // reshapes it back down to 0-d. Verified against real numpy 2.5.1:
    // `np.argmin(np.array(5), keepdims=True)` returns a 0-d/scalar result
    // (`np.int64(0)`), not shape `(1,)`; anionpy was returning shape `(1,)`.
    // `ndim == 1` is the only case that needs no relabeling (flattening a
    // 1-D array to 1-D is already a no-op), so the condition is `ndim != 1`.
    //
    // Kept as a guard rather than an unconditional reshape purely as a
    // micro-optimisation: `argext`'s raw flattened-and-keepdims result is
    // always exactly `[1]` regardless of the original ndim (per the comment
    // above), so reshaping `[1]` -> `[1]` for `ndim == 1` would be a no-op
    // anyway. Contrast with `do_nanargext` below, whose raw flattened result
    // is NOT uniformly `[1]` across ndim (empirically: it's a true 0-d
    // scalar even when the original array was 1-D) -- that function reshapes
    // unconditionally instead of relying on this same skip.
    if flattened && keepdims && ndim != 1 {
        result = result.reshape(&vec![1usize; ndim]).map_err(to_py_err)?;
    }
    wrap_reduction(py, result, out)
}

// ---------------------------------------------------------------------------
// Tier 3: `cumsum`/`cumprod` -- new running-accumulate kernel
// (`ufunc::accumulate_axis`). `axis=None` flattens first (numpy: matches
// `argmin`/`argmax`'s own `axis=None` flattening above).
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None))]
fn cumsum(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    do_accumulate_axis(py, BinaryOp::Add, a, axis, dtype, out, None)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None))]
fn cumprod(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    do_accumulate_axis(py, BinaryOp::Multiply, a, axis, dtype, out, None)
}

/// `nancumsum`/`nancumprod` -- `nan_fill` the input with the accumulate
/// op's IDENTITY (0 for `Add`, 1 for `Multiply`) and then delegate to the
/// very same `do_accumulate_axis` that `cumsum`/`cumprod` use, exactly as
/// `nansum`/`nanprod` delegate to `do_reduce_axis`. Everything downstream
/// -- axis normalisation, the 0-d courtesy `(1,)` shape, the `dtype=`
/// result-narrowing fix, `out=` handling and every error message -- is
/// therefore inherited, not reimplemented, so the two families cannot
/// drift apart.
///
/// The fill runs BEFORE the `dtype=` cast, matching real numpy: its
/// `nancumsum` is literally `a, mask = _replace_nan(a, 0)` followed by
/// `np.cumsum(a, axis=axis, dtype=dtype, out=out)`. Order is observable --
/// filling after a cast to an integer dtype would have nothing left to
/// fill, since the NaN would already have become a garbage integer.
///
/// `nan_fill` is a total function over every dtype and returns a COPY: it
/// clones bool/int operands untouched (so an integer `nancumsum` is
/// bit-identical to `cumsum`, which is what numpy's `_replace_nan`
/// no-op-on-non-inexact branch produces), and for complex it substitutes
/// `fill + 0j`, NOT `fill + fill*j` -- the imaginary part is zeroed
/// because numpy's `copyto(a, fill, where=mask)` broadcasts a real scalar
/// into a complex array. That distinction is invisible for `nancumsum`
/// (fill 0) and load-bearing for `nancumprod` (fill 1); it is already
/// guarded in `nan_fill` itself, which `nanprod` proved by sweep.
#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None))]
fn nancumsum(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    do_accumulate_axis(py, BinaryOp::Add, a, axis, dtype, out, Some(0.0))
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None))]
fn nancumprod(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    do_accumulate_axis(py, BinaryOp::Multiply, a, axis, dtype, out, Some(1.0))
}

fn do_accumulate_axis(
    py: Python<'_>,
    op: BinaryOp,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    nan_fill_value: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let arr = extract_array_like(a)?;
    // See `nancumsum` above for why this precedes the `dtype=` cast.
    let arr = match nan_fill_value {
        Some(v) => nan_fill(&arr, v),
        None => arr,
    };
    let dtype_override: Option<DType> = match dtype {
        None => None,
        Some(d) if d.is_none() => None,
        // Numeric reduction target: no S/U arm downstream, decline cleanly.
        Some(d) => Some(crate::dtype_from_pyobj_no_su(d)?),
    };
    let arr = match dtype_override {
        Some(dt) => arr.cast_to(dt),
        None => arr,
    };
    let ndim = arr.ndim();
    let (target, ax): (NdArray, usize) = match axis {
        None => (arr.ravel_order("C").map_err(to_py_err)?, 0),
        Some(a2) if a2.is_none() => (arr.ravel_order("C").map_err(to_py_err)?, 0),
        Some(a2) => {
            let raw = single_axis_from_pyobj(a2)?;
            let n = ndim as isize;
            // BUG FOUND + FIXED 2026-08-03 (Monday), 0-d operand, two
            // defects in these four lines, both measured against real
            // numpy 2.5.1:
            //
            //   `axis=0`  gave shape ()   -- numpy gives shape (1,)
            //   `axis=-1` raised AxisError -- numpy gives shape (1,)
            //
            // The negative case is the same offset bug already fixed in
            // `ufunc::normalize_reduce_axes`: `raw + n` with `n == 0`
            // leaves -1 as -1 and trips the `norm < 0` guard, so the
            // offset has to use the courtesy effective ndim `n.max(1)`.
            //
            // The shape case is why 0-d then routes to `ravel` rather than
            // to `accumulate_axis(arr, 0)`: a cumulative reduction over a
            // scalar yields a ONE-element 1-d result, not a scalar, and
            // `axis=None` on a 0-d operand already produces exactly that
            // via this same `ravel` branch (`np.cumsum(np.array(3.0))` is
            // `array([3.])`). So the courtesy axis and `axis=None` really
            // do coincide here -- this is not a shape fudge.
            let norm = if raw < 0 { raw + n.max(1) } else { raw };
            // The REPORTED ndim is the courtesy effective ndim, not the
            // operand's own. numpy's cumulative ops promote a 0-d operand to
            // shape (1,) BEFORE validating the axis, so on a 0-d operand it
            // says "out of bounds for array of dimension 1", not "...0"
            // (measured, /tmp/mg_cumsum_msg.py -- this is the same
            // ravel-first mechanism that makes the RESULT shape (1,) rather
            // than (), so reporting 0 here contradicted the shape this very
            // function returns two lines below).
            //
            // Found only because a differential case compared the exception
            // MESSAGE; the earlier sweep compared exception TYPE alone and
            // both sides raise AxisError, so it read as agreement.
            let reported_ndim = ndim.max(1);
            if norm < 0 || norm >= n.max(1) {
                return Err(axis_error(raw, Some(reported_ndim), &format!(
                    "axis {raw} is out of bounds for array of dimension {reported_ndim}"
                )));
            }
            if ndim == 0 {
                (arr.ravel_order("C").map_err(to_py_err)?, 0)
            } else {
                (arr, norm as usize)
            }
        }
    };
    let result = accumulate_axis(op, &target, ax).map_err(to_py_err)?;
    // BUG FOUND + FIXED 2026-08-03 (Monday). An explicit `dtype=` was
    // honoured for the ACCUMULATOR (the `cast_to` above) but thrown away
    // for the RESULT: `accumulate_axis` applies numpy's default
    // sum/prod promotion unconditionally, so every integer narrower than
    // the platform int -- and `bool` -- came back as int64/uint64.
    // Measured against real numpy 2.5.1:
    //     np.cumsum(int8_arr, dtype=np.int8).dtype   -> int8    (anionpy: int64)
    //     np.cumsum(int8_arr, dtype=np.uint8).dtype  -> uint8   (anionpy: uint64)
    //     np.cumsum(int8_arr, dtype=bool).dtype      -> bool    (anionpy: int64)
    // float32/float64/int64 were already correct, which is why this
    // survived: the dtype= cases in `cumsum`/`cumprod`'s own (declared
    // EXACT) case sets never used a narrow integer. `cumulative_sum`'s new
    // cases did, on their first run.
    //
    // Casting the RESULT down is value-identical to accumulating in the
    // narrow type, not merely close: two's-complement truncation is a ring
    // homomorphism for both `+` and `*`, so wrapping at each step and
    // wrapping once at the end agree exactly (checked on the int8
    // [100,100,100] overflow case, which numpy gives as [100,-56,44]).
    let result = match dtype_override {
        Some(dt) => result.cast_to(dt),
        None => result,
    };
    wrap(py, result, out)
}

// ---------------------------------------------------------------------------
// Tier 4: `mean`/`ptp`/`count_nonzero` -- composed from `reduce_axis`/
// `binary_op`, no new kernel.
// ---------------------------------------------------------------------------

/// numpy's `mean` dtype rule (verified against real numpy 2.5.1, and
/// DIFFERENT from `sum`'s own promotion rule): bool/integer input always
/// produces a `float64` output (`np.mean(np.array([1, 2], dtype=np.int8)).dtype
/// == float64`); float/complex input keeps its OWN dtype as the output
/// dtype, no upcast (`np.mean(np.array([1, 2], dtype=np.float32)).dtype ==
/// float32`, unlike `np.sum` on int8 which upcasts to int64 but on float32
/// stays float32 too -- the two rules happen to agree on the float/complex
/// side, only bool/int differs).
///
/// float16's mean does NOT follow `sum`'s memory-layout-dependent
/// wide/narrow rule (see `reduce_axis_f16_narrow_wide` in `ufunc.rs`) --
/// verified against real numpy 2.5.1 with a 3000-sample sweep: `a.mean(axis=0)`
/// on a case where `a.sum(axis=0)` is narrow (per-step float16 rounding)
/// matches neither `sum(axis=0)` divided narrowly NOR that same narrow sum
/// simply cast up and divided in float32 (1766/3000 mismatches) -- it
/// matches ONLY `a.astype(f32).sum(axis=0) / n` computed and rounded once
/// in float32 (0/3000 mismatches). So `mean` on float16 always widens to
/// float32 for the sum step regardless of axis contiguity, divides by the
/// count in float32, and rounds to float16 exactly once at the very end;
/// it is NOT built on top of `sum`'s own (now bit-exact) narrow/wide
/// result for this dtype.
#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, keepdims=None, r#where=None))]
#[allow(clippy::too_many_arguments)]
fn mean(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    // `mean` is a MEAN-FAMILY caller: its axis errors are not `sum`'s.
    // See `mean_family_axis_check`. `nanmean` delegates here for integer
    // and bool operands (numpy does too), which is how a `nanmedian` on an
    // empty integer array ends up reporting this message -- 160 sweep
    // cases traced back to exactly that chain on 2026-08-04.
    if let Some(ax) = axis {
        mean_family_axis_check(ax, arr.ndim())?;
    }
    let ndim = arr.ndim();
    let axes_raw: Option<Vec<isize>> = match axis {
        None => None,
        Some(ax) if ax.is_none() => None,
        // 0-d courtesy: mean -- numpy RAISES AxisError on 0-d scalar axis
        Some(ax) => Some(reduce_axes_from_pyobj(ax, ndim, false, AxisSeq::TupleOnly)?),
    };
    let axes = normalize_reduce_axes(axes_raw.as_deref(), ndim).map_err(to_py_err)?;

    let dtype_override: Option<DType> = match dtype {
        None => None,
        Some(d) if d.is_none() => None,
        // Numeric reduction target: no S/U arm downstream, decline cleanly.
        Some(d) => Some(crate::dtype_from_pyobj_no_su(d)?),
    };
    let mean_dtype_default = match dtype_override {
        Some(dt) => Some(dt),
        None if arr.dtype() == DType::Bool || arr.dtype().is_integer() => Some(DType::F64),
        None => None,
    };

    let mask_arr: Option<NdArray> = match r#where {
        None => None,
        Some(w) => match w.extract::<bool>() {
            Ok(true) => None,
            Ok(false) => Some(extract_array_like(w)?),
            Err(_) => Some(extract_array_like(w)?),
        },
    };

    // float16 output always widens to float32 for the sum + divide (see
    // doc comment above) regardless of `sum`'s own memory-layout-dependent
    // narrow/wide rule -- force the reduce's compute dtype to F32 in that
    // one case and cast the final quotient back down to F16 once, after
    // dividing, rather than reusing `sum`'s own (bit-exact for `sum`, but
    // NOT what `mean` needs) float16 result.
    let effective_out_dtype = mean_dtype_default.unwrap_or(arr.dtype());
    let f16_widen = effective_out_dtype == DType::F16;
    let sum_compute_override = if f16_widen { Some(DType::F32) } else { mean_dtype_default };

    // Compose at keepdims=false (the natural, retained-axes-only shape --
    // see the big comment below, just before the final `relayout_for_
    // reduction` call, for why this -- NOT a keepdims=True composition
    // through `k_order_relayout_composed` -- is the correct mechanism for
    // `mean` specifically, unlike `ptp`).
    let sum_result = reduce_axis(BinaryOp::Add, &arr, &axes, false, sum_compute_override, None, mask_arr.as_ref())
        .map_err(to_py_err)?;

    // Element count of the reduced axes -- with a `where=` mask, numpy
    // divides by the mask's own per-slice popcount, not the raw axis
    // length (verified: `np.mean([1., 2., 3.], where=[True, False, True])
    // == 2.0`, i.e. (1+3)/2, not /3). Build that count via the same
    // `count_nonzero`-style `reduce_axis(Add, mask_as_i64, ...)` path used
    // below, rather than a second bespoke counting loop.
    let out_dtype = sum_result.dtype();

    // numpy's real `_mean` (`numpy/_core/_methods.py`) divides via
    // `um.true_divide(ret, rcount, out=ret, casting='unsafe')` where
    // `rcount` is a `numpy.int64`, NOT a Python int. Under NEP-50 type
    // promotion, a numpy scalar (as opposed to a weakly-promoted Python
    // int) is "strong": `<dtype> / int64` promotes to whichever double-
    // precision type covers `dtype`, for any dtype narrower than double
    // (float32->float64, complex64->complex128) -- the `out=ret` argument
    // then casts that wider result back down to `ret`'s own dtype in one
    // final rounding step, `casting='unsafe'`. Verified empirically
    // against real numpy 2.5.1 with 20000+ samples each for float32
    // (scalar AND per-axis array-output cases) and complex64 (array-output
    // case): dividing at native precision (or with Python-int weak
    // promotion) does NOT reproduce numpy's actual output, but widening to
    // double precision for the divide step and rounding down exactly once
    // matches bit-for-bit with 0 mismatches. The same widen-to-double rule
    // was also confirmed consistent (0 mismatches either way, since it's a
    // strict superset) for float16's already-established float32-widened
    // sum. float64 and complex128 are unaffected: `result_type(float64,
    // int64) == float64` and `result_type(complex128, int64) ==
    // complex128`, so no further promotion happens for those.
    let divide_widen_dtype = match out_dtype {
        DType::F16 | DType::F32 => Some(DType::F64),
        DType::C64 => Some(DType::C128),
        _ => None,
    };
    let divide_dtype = divide_widen_dtype.unwrap_or(out_dtype);

    let count_natural: NdArray = match &mask_arr {
        Some(mask) => {
            let mask_i64 = mask.cast_to(DType::I64);
            let natural = reduce_axis(BinaryOp::Add, &mask_i64, &axes, false, None, None, None).map_err(to_py_err)?;
            // Same layout hazard as `reduce_count`/`count_nonzero`:
            // `mask.cast_to(I64)` forces a C-contiguous copy on the (real)
            // Bool->I64 dtype change, so `reduce_axis`'s internal layout
            // decision (driven by its own `a` parameter, `mask_i64` here)
            // would otherwise always yield a C-contiguous count regardless
            // of `arr`'s real layout. Re-derive from `arr`'s own
            // shape/strides instead, directly at the caller's `keepdims`
            // (same reasoning as `sum_result` above).
            natural.relayout_for_reduction(arr.shape(), arr.strides(), &axes, false)
        }
        None => {
            // numpy divides by a genuine SCALAR here (`rcount`, a bare
            // `numpy.int64`, not an array) whenever there is no `where=`
            // mask -- represented as a 0-d `NdArray` so it broadcasts
            // against `sum_result` (any shape, including a genuinely empty
            // one) without needing the total-element-count special case a
            // full-shaped buffer used to require.
            let n: usize = axes.iter().map(|&ax| arr.shape()[ax]).product();
            NdArray::from_buffer(Buffer::I64(vec![n as i64]), vec![], Order::C).map_err(to_py_err)?
        }
    };
    let count_result = count_natural.cast_to(divide_dtype);

    let sum_for_divide = if divide_widen_dtype.is_some() { sum_result.cast_to(divide_dtype) } else { sum_result.clone() };

    // Measured (spec doc section 3): plain `mean` over an empty slice fires
    // TWO warnings, not one -- `"Mean of empty slice"` (numpy's own
    // Python-level `_methods.py` check, unconditional on count == 0) AND
    // separately `"invalid value encountered in divide"` (from the 0/0
    // division itself, same as any other 0/0 divide -- `nanmean` below
    // already established the "Mean of empty slice" half of this pattern
    // for its own empty-after-NaN-removal case; this is the analogous
    // plain-`mean` site, given the same treatment plus the divide-side
    // half nanmean is not asked to reproduce).
    let isbad = binary_op(BinaryOp::Equal, &count_result, &scalar_f64(0.0)?.cast_to(count_result.dtype())).map_err(to_py_err)?;
    if any_true(&isbad) {
        warn_runtime(py, c"Mean of empty slice")?;
    }
    // This is `mean`'s internal `true_divide(sum, rcount)`, not
    // `floor_divide`/`divmod` -- `divide_dtype` (== `sum_for_divide`'s and
    // `count_result`'s own dtype) is always floating or complex (mean's
    // accumulator/divide dtype is never integer; even an explicit
    // `dtype=np.int64` only affects the FINAL cast, not this internal
    // divide -- see the cast-back comment above), so `out_is_floating` is
    // unconditionally `true` here, preserving this site's pre-existing
    // (correct) unconditional-`Invalid`-at-0/0 behavior.
    if let Some(cat) = ionp_core::fpe::detect_divide_family(&sum_for_divide, &count_result, true) {
        // Measured (spec doc section 3 vs section 7): numpy's OWN internal
        // divide-by-`rcount` reports as `"scalar divide"`, not plain
        // `"divide"`, specifically when the quotient is itself a genuine
        // 0-d scalar (whole-array `mean()`, no axis=) -- an empty-AXIS
        // reduction (`mean(a, axis=0)`, quotient shape `(3,)`, still an
        // array) keeps the ordinary `"divide"` wording. `sum_for_divide`'s
        // ndim (== the quotient's ndim, since `binary_op` never changes
        // rank) is exactly numpy's own "is this a 0-d ufunc call" test.
        let name = if sum_for_divide.ndim() == 0 { "scalar divide" } else { BinaryOp::Divide.numpy_name() };
        fpstate::signal(py, cat, name, 2)?;
    }
    let raw_result = binary_op(BinaryOp::Divide, &sum_for_divide, &count_result).map_err(to_py_err)?;
    // `binary_op` always emits plain C-contiguous output at this natural
    // (keepdims=false) shape, which is exactly the shape/contiguity
    // contract `relayout_for_reduction` expects as its `self` input (the
    // same contract `sum_result` itself satisfied before `reduce_axis`
    // relaid it out). Do ALL remaining dtype casts on this natural-shape
    // result FIRST (a real `cast_to` dtype change always force-flattens to
    // plain C -- see `to_contiguous` -- so casting must happen before, never
    // after, the final layout-determining relayout), THEN relayout once at
    // the very end.
    //
    // This -- NOT `k_order_relayout_composed` (correct for `ptp`, see its
    // doc comment) -- is the right mechanism for `mean`/`nanmean`
    // specifically: real numpy's actual `_mean` implementation
    // (`numpy/_core/_methods.py`) computes
    // `um.true_divide(ret, rcount, out=ret, casting='unsafe')` -- the
    // explicit `out=ret` argument writes the quotient's VALUES directly
    // into `ret`'s (the sum's) own pre-existing buffer/layout, bypassing
    // any order='K' output-layout inference entirely. So `mean`'s result
    // strides are simply, unconditionally, `sum`'s own strides -- confirmed
    // via direct real-numpy comparison, 15/15 matches across every layout
    // (C/F/fulltranspose/partialswap/permuted) x every axis, plus masked
    // (`where=`) C and F cases, all keepdims=True (`/tmp/probe_div_mech2.py`):
    // `np.sum(...).strides == np.mean(...).strides` EXACTLY, every time.
    // Reusing `relayout_for_reduction` -- the exact function `sum_result`
    // itself is built with -- on this divide's natural-shape result
    // guarantees that structural identity by construction, rather than by
    // a separate (and, for `mean`, wrong) K-order vote.
    let mut result = raw_result;
    if divide_widen_dtype.is_some() {
        result = result.cast_to(out_dtype);
    }
    if f16_widen {
        result = result.cast_to(DType::F16);
    } else if result.dtype() != effective_out_dtype {
        // Bug found via this task's `dtype=` probe (withdrawn in fdcdfb6):
        // an explicit `dtype=` override that isn't F16/F32/C64 (so
        // `divide_widen_dtype` above is `None`) never reached ANY cast
        // back to the caller's actually-requested `effective_out_dtype` --
        // `binary_op(Divide, ...)` auto-promotes int/int division to
        // float64 under anionpy's own promotion rules regardless of what
        // dtype was asked for, and that leaked straight through
        // (`np.mean(int_array, dtype=np.int64)` -> real numpy's int64 7,
        // anionpy's float64 7.5). numpy's own `_mean` does this via
        // `um.true_divide(ret, rcount, out=ret, casting='unsafe')` --the
        // `out=ret` argument performs exactly this final unsafe cast-down
        // unconditionally, not just for the three dtypes that already had
        // an explicit widen-then-narrow round trip above. This mirrors
        // that: whatever `result`'s dtype ended up being, cast it back to
        // the target one final time if it isn't already there.
        result = result.cast_to(effective_out_dtype);
    }
    // Final step, after ALL casting is done: relayout onto sum's own
    // (keepdims-aware) layout -- see the big comment above.
    result = result.relayout_for_reduction(arr.shape(), arr.strides(), &axes, keepdims);
    wrap_reduction(py, result, out)
}

/// `np.ptp` (peak-to-peak): `max(axis) - min(axis)`, composed directly from
/// two `reduce_axis` calls (`Maximum`/`Minimum`, the exact same kernel
/// `.max()`/`.min()` use) plus `binary_op(Subtract, ...)` -- no new kernel,
/// no dtype override needed since `Maximum`/`Minimum`'s reduce-family
/// compute dtype never upcasts (unlike `Add`/`Multiply`), so both operands
/// of the final subtract already share the input's own dtype.
#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=None))]
fn ptp(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    let ndim = arr.ndim();
    let axes_raw: Option<Vec<isize>> = match axis {
        None => None,
        Some(ax) if ax.is_none() => None,
        // 0-d courtesy: ptp -- numpy accepts
        Some(ax) => Some(reduce_axes_from_pyobj(ax, ndim, true, AxisSeq::TupleOnly)?),
    };
    let axes = normalize_reduce_axes(axes_raw.as_deref(), ndim).map_err(to_py_err)?;
    // Compose directly at the CALLER's requested `keepdims` -- real numpy's
    // `ptp` genuinely IS `max(axis, keepdims) - min(axis, keepdims)` under
    // the hood, both sub-reductions already correctly laid out by
    // `reduce_axis`'s own fix above (verified: `sum`/`amax`/`amin`/etc.
    // keepdims=True cases are 0/1080 mismatches). The two-operand K-order
    // composition step needs `NdArray::k_order_relayout_composed`, NOT
    // plain `multi_sorted_stride_perm` -- see that function's doc comment:
    // a `keepdims=True` operand's synthetic size-1-axis stride is
    // GENERALIZED-contiguous (in the same sense `is_c_contiguous`/
    // `is_f_contiguous` already define), and real numpy's ufunc order='K'
    // takes a fast path off that contiguity flag BEFORE ever running the
    // generic per-axis-pair vote, discarding the size-1 axis's original
    // stride value in favor of a freshly recomputed one.
    let max_arr = reduce_axis(BinaryOp::Maximum, &arr, &axes, keepdims, None, None, None).map_err(to_py_err)?;
    let min_arr = reduce_axis(BinaryOp::Minimum, &arr, &axes, keepdims, None, None, None).map_err(to_py_err)?;
    let raw = binary_op(BinaryOp::Subtract, &max_arr, &min_arr).map_err(to_py_err)?;
    let result = NdArray::k_order_relayout_composed(&raw, &[&max_arr, &min_arr]);
    wrap_reduction(py, result, out)
}

/// `np.count_nonzero`: count of elements `!= 0` (`!= 0+0j` for complex,
/// `!= False` for bool) along `axis`. Composed as `mask = (a != 0)`
/// (`binary_op(NotEqual, a, zero_scalar)`, a `Bool` array) then
/// `reduce_axis(Add, mask, ...)` -- the default reduce-family compute
/// dtype for `Add` on `Bool` input is ALREADY `I64` (see
/// `reduce_compute_dtype` in `ufunc.rs`), matching numpy's own
/// `count_nonzero` return dtype (`intp`/int64) with no explicit override
/// needed. numpy's real signature also allows `axis=None` (count over the
/// WHOLE array, a scalar) which `normalize_reduce_axes(None, ndim)`
/// already handles by reducing every axis.
#[pyfunction]
#[pyo3(signature = (a, axis=None, keepdims=None))]
fn count_nonzero(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    let ndim = arr.ndim();
    let axes_raw: Option<Vec<isize>> = match axis {
        None => None,
        Some(ax) if ax.is_none() => None,
        // 0-d courtesy: count_nonzero -- numpy accepts
        Some(ax) => Some(reduce_axes_from_pyobj(ax, ndim, true, AxisSeq::TupleOnly)?),
    };
    let axes = normalize_reduce_axes(axes_raw.as_deref(), ndim).map_err(to_py_err)?;
    let zero = NdArray::from_buffer(creation::zeros_buffer(arr.dtype(), 1), vec![], Order::C).map_err(to_py_err)?;
    // `mask = (arr != 0)` is a freshly computed `binary_op` result, which
    // (per `binary_op`'s own doc/implementation) is ALWAYS C-contiguous --
    // it has no memory layout of its own worth propagating. Letting
    // `reduce_axis` derive the output layout from `mask`'s shape/strides
    // (as it would from its own `a` parameter) is therefore a no-op and
    // always yields a C-contiguous result. `count_nonzero`'s real output
    // layout follows `arr`'s (the ORIGINAL input) own memory order instead
    // -- reduce with `keepdims=false` (the natural retained-axes-only
    // C-contiguous buffer) and drive the real layout decision from `arr`'s
    // own shape/strides via the same public helper `reduce_axis` uses
    // internally.
    let mask = binary_op(BinaryOp::NotEqual, &arr, &zero).map_err(to_py_err)?;
    let natural = reduce_axis(BinaryOp::Add, &mask, &axes, false, None, None, None).map_err(to_py_err)?;
    let result = natural.relayout_for_reduction(arr.shape(), arr.strides(), &axes, keepdims);
    wrap_reduction(py, result, None)
}

// ---------------------------------------------------------------------------
// Tier 5: NaN-aware reductions (`nansum`/`nanprod`/`nanmin`/`nanmax`/
// `nanmean`) -- `nan_fill` the input, then delegate to the SAME
// `reduce_axis`/`do_reduce_axis` machinery `sum`/`min`/`max`/`mean` already
// use, so bit-exactness (and its documented exceptions, e.g. masked
// `where=` not being pairwise) is inherited rather than reimplemented.
// ---------------------------------------------------------------------------

fn warn_runtime(py: Python<'_>, msg: &::std::ffi::CStr) -> PyResult<()> {
    let cat = py.get_type::<pyo3::exceptions::PyRuntimeWarning>();
    PyErr::warn(py, &cat, msg, 2)
}

fn scalar_bool(v: bool) -> PyResult<NdArray> {
    NdArray::from_buffer(Buffer::Bool(vec![v]), vec![], Order::C).map_err(to_py_err)
}

fn scalar_f64(v: f64) -> PyResult<NdArray> {
    NdArray::from_buffer(Buffer::F64(vec![v]), vec![], Order::C).map_err(to_py_err)
}

/// Element count along `axes`, honoring a `where=` mask the same way
/// `mean` already does (mask popcount, not raw axis length). Built as
/// `reduce_axis(Add, ones_i64, axes, ..., where_mask)` so a masked count
/// with a zero-length reduced axis, or an all-`False` mask slice, both
/// naturally bottom out at `Add`'s identity (0) via the exact same path
/// `reduce_axis` already uses for every other reduction -- no separate
/// shape-only special case needed.
fn reduce_count(a: &NdArray, axes: &[usize], keepdims: bool, mask: Option<&NdArray>) -> PyResult<NdArray> {
    let ones = NdArray::from_buffer(Buffer::I64(vec![1i64; a.size()]), a.shape().to_vec(), Order::C).map_err(to_py_err)?;
    // `ones` is a brand-new, always-C-contiguous buffer -- it has no
    // meaningful memory layout of its own to propagate, so letting
    // `reduce_axis` derive the output layout from `ones`'s shape/strides
    // (as it normally would from its own `a` parameter) is a no-op and
    // always yields a C-contiguous result. `count_nonzero`'s real output
    // layout must instead follow the layout of the array actually being
    // counted (`a`, this function's own parameter, not `ones`) -- reduce
    // with `keepdims=false` (the natural retained-axes-only C-contiguous
    // buffer `reduce_axis`'s dispatch always builds pre-relayout) and then
    // drive the real layout decision from `a`'s own shape/strides via the
    // same public helper `reduce_axis` itself uses internally.
    let natural = reduce_axis(BinaryOp::Add, &ones, axes, false, None, None, mask).map_err(to_py_err)?;
    Ok(natural.relayout_for_reduction(a.shape(), a.strides(), axes, keepdims))
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, keepdims=None, initial=None, r#where=None))]
#[allow(clippy::too_many_arguments)]
fn nansum(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    let filled = nan_fill(&arr, 0.0);
    do_reduce_axis(py, BinaryOp::Add, &filled, axis, dtype, out, keepdims, initial, r#where)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, keepdims=None, initial=None, r#where=None))]
#[allow(clippy::too_many_arguments)]
fn nanprod(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    let filled = nan_fill(&arr, 1.0);
    do_reduce_axis(py, BinaryOp::Multiply, &filled, axis, dtype, out, keepdims, initial, r#where)
}

/// `nanmin`/`nanmax`: fill NaN with +-infinity (so a NaN never wins the
/// fold when a real value is present in the same reduced slice) and
/// delegate to the ordinary `Minimum`/`Maximum` `reduce_axis` path. Then,
/// ONLY when the caller did not supply `initial=` (matching real numpy,
/// which lets `initial=` rescue an all-NaN slice with no warning --
/// verified: `np.nanmin([np.nan, np.nan], initial=5.0) == 5.0`, silent),
/// separately detect any reduced slice that was "all excluded" (every
/// element NaN, AND-ed with `where=False` positions if `where=` was also
/// given) and force those output slots back to NaN plus emit the same
/// `RuntimeWarning("All-NaN slice encountered")` real numpy does.
/// `nanmax`/`nanmin` on an empty array (via the underlying
/// `Maximum`/`Minimum` `reduce_axis` call in `do_nan_extreme` below) raises
/// numpy's "zero-size array to reduction operation ... which has no
/// identity" `ValueError` -- but real numpy names it `fmax`/`fmin` here,
/// NOT `maximum`/`minimum` (verified against real numpy 2.5.1:
/// `np.nanmax(np.array([]))` -> `'zero-size array to reduction operation
/// fmax which has no identity'`, `np.nanmin` -> `'... fmin ...'`, regardless
/// of dtype -- `nanmax`/`nanmin` are actually implemented via
/// `np.fmax.reduce`/`np.fmin.reduce` internally in real numpy, which is
/// where the name comes from). `reduce_axis` (`ufunc.rs`) has no way to
/// take a name override -- it always renders this message via its own
/// private `op.numpy_name()` (`BinaryOp::Maximum`/`Minimum` ->
/// `"maximum"`/`"minimum"`), and `reduce_axis_generic`/`reduce_axis_masked`
/// (the actual message-construction call sites) aren't `pub`, so there is
/// no lower-level entry point in `ufunc.rs` this file can call instead.
/// Since `nanmin`/`nanmax` are the ONLY two callers of `do_nan_extreme`
/// (both fixed `op`, `BinaryOp::Minimum`/`Maximum` respectively -- see
/// their own `#[pyfunction]` bodies below), the exact wrong substring is
/// known statically at each call site, so this rewrites the specific
/// message shape after the fact rather than needing an upstream API
/// change. Any OTHER `IonpError` from `reduce_axis` (shape errors, dtype
/// errors, etc.) passes through completely unchanged.
/// Two distinct message SHAPES share this `maximum`/`minimum` ->
/// `fmax`/`fmin` renaming need (both from `reduce_axis`'s internal
/// `op.numpy_name()`, neither overridable -- see this function's own doc
/// above): the empty-reduction one already documented there, PLUS
/// `reduce_axis`'s `where=`-mask-without-`initial=` message (verified
/// against real numpy 2.5.1: `np.nanmax([1.0], where=False)` ->
/// `"reduction operation 'fmax' does not have an identity, so to use a
/// where mask one has to specify 'initial'"`, quotes around the name
/// included). Matched as two exact full-string forms (not a loose
/// substring replace) so an unrelated message that happens to contain the
/// word "maximum"/"minimum" is never silently mangled.
fn rename_no_identity_op(e: IonpError, op: BinaryOp) -> IonpError {
    let (wrong, right) = match op {
        BinaryOp::Maximum => ("maximum", "fmax"),
        BinaryOp::Minimum => ("minimum", "fmin"),
        _ => return e,
    };
    match e {
        IonpError::Value(msg) => {
            let empty_wrong = format!("zero-size array to reduction operation {wrong} which has no identity");
            let empty_right = format!("zero-size array to reduction operation {right} which has no identity");
            let mask_wrong = format!(
                "reduction operation '{wrong}' does not have an identity, so to use a where mask one has to specify 'initial'"
            );
            let mask_right = format!(
                "reduction operation '{right}' does not have an identity, so to use a where mask one has to specify 'initial'"
            );
            if msg == empty_wrong {
                IonpError::Value(empty_right)
            } else if msg == mask_wrong {
                IonpError::Value(mask_right)
            } else {
                IonpError::Value(msg)
            }
        }
        other => other,
    }
}

#[allow(clippy::too_many_arguments)]
fn do_nan_extreme(
    py: Python<'_>,
    op: BinaryOp,
    fill: f64,
    warn_msg: &'static ::std::ffi::CStr,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let arr = extract_array_like(a)?;
    let ndim = arr.ndim();
    let axes_raw: Option<Vec<isize>> = match axis {
        None => None,
        Some(ax) if ax.is_none() => None,
        // 0-d courtesy: do_nan_extreme: nanmin/nanmax/nanarg* -- numpy accepts
        Some(ax) => Some(reduce_axes_from_pyobj(ax, ndim, true, AxisSeq::TupleOnly)?),
    };
    let axes = normalize_reduce_axes(axes_raw.as_deref(), ndim).map_err(to_py_err)?;

    let initial_arr: Option<NdArray> = match initial {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => Some(extract_array_like(v)?),
    };
    let mask_arr: Option<NdArray> = match r#where {
        None => None,
        Some(w) => match w.extract::<bool>() {
            Ok(true) => None,
            Ok(false) => Some(extract_array_like(w)?),
            Err(_) => Some(extract_array_like(w)?),
        },
    };

    // numpy's own `nanmax`/`nanmin` only take the fast `fmax`/`fmin.reduce`
    // path (and thus only ever surface an "fmax"/"fmin" identity-error
    // message) when `type(a) is np.ndarray` EXACTLY -- verified via
    // `inspect.getsource(np.nanmax)`: `if (type(a) is np.ndarray or type(a)
    // is np.memmap) and a.dtype != np.object_: res = np.fmax.reduce(...)`.
    // Any other array-like (a bare Python list/tuple, verified: `np.nanmax([])`
    // raises "...reduction operation maximum which has no identity", NOT
    // "fmax") falls through to the slow `_replace_nan` + `np.amax` path,
    // whose error message keeps the plain "maximum"/"minimum" wording. So
    // the fmax/fmin rename below must be gated on the SAME bare-ndarray
    // check, not applied unconditionally.
    //
    // NOTE: this can't be `a.extract::<PyRef<'_, PyArray>>().is_ok()` (the
    // check `trim_zeros` uses) -- `nanmax`/`nanmin` are `kind="method"` in
    // the differential registry, which (unlike `kind="custom"`, trim_zeros'
    // kind) does NOT route array arguments through
    // `make_ionp_array_converter` before the call, so the base (non-form)
    // corpus cases hand `a` to `anionpy.nanmax` as a genuine, UNCONVERTED
    // `numpy.ndarray` -- real numpy's own reference object, not an
    // `anionpy.ndarray`. That numpy.ndarray is exactly the case real numpy's
    // `type(a) is np.ndarray` check says YES to, so it must count as
    // "bare array" here too. What actually distinguishes numpy's fast path
    // from its slow path is "a genuine array object" vs. "a Python list/
    // tuple/range/scalar container" (the form-axis's `form_list`/
    // `form_tuple`/`form_range`/... cases) -- checked here via `.dtype`
    // presence, which both `numpy.ndarray` and `anionpy.ndarray` carry and no
    // bare list/tuple/range/scalar does.
    let is_bare_array = a.hasattr("dtype").unwrap_or(false);
    let filled = nan_fill(&arr, fill);
    let result = reduce_axis(op, &filled, &axes, keepdims, None, initial_arr.as_ref(), mask_arr.as_ref())
        .map_err(|e| if is_bare_array { rename_no_identity_op(e, op) } else { e })
        .map_err(to_py_err)?;

    if initial_arr.is_none() {
        let isnan_mask = isnan_array(&arr).map_err(to_py_err)?;
        let excluded_or_nan = match &mask_arr {
            Some(m) => {
                let not_m = binary_op(BinaryOp::NotEqual, m, &scalar_bool(true)?).map_err(to_py_err)?;
                binary_op(BinaryOp::LogicalOr, &isnan_mask, &not_m).map_err(to_py_err)?
            }
            None => isnan_mask,
        };
        let all_bad = reduce_axis(BinaryOp::LogicalAnd, &excluded_or_nan, &axes, keepdims, None, None, None).map_err(to_py_err)?;
        if any_true(&all_bad) {
            warn_runtime(py, warn_msg)?;
            let result = overwrite_nan_where(&result, &all_bad);
            return wrap_reduction(py, result, out);
        }
    }
    wrap_reduction(py, result, out)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=None, initial=None, r#where=None))]
#[allow(clippy::too_many_arguments)]
fn nanmin(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    do_nan_extreme(
        py,
        BinaryOp::Minimum,
        f64::INFINITY,
        c"All-NaN slice encountered",
        a,
        axis,
        out,
        keepdims,
        initial,
        r#where,
    )
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=None, initial=None, r#where=None))]
#[allow(clippy::too_many_arguments)]
fn nanmax(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    initial: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    do_nan_extreme(
        py,
        BinaryOp::Maximum,
        f64::NEG_INFINITY,
        c"All-NaN slice encountered",
        a,
        axis,
        out,
        keepdims,
        initial,
        r#where,
    )
}

/// `nanmean`: real numpy's `_replace_nan` short-circuits to plain `mean`
/// whenever no NaN is present ANYWHERE in the input (not per-slice) --
/// mirrored here by checking `any_true(isnan_array(&arr))` up front and,
/// if false, delegating straight to this file's own `mean` function so
/// the NaN-free case is bit-for-bit `mean`'s own already-verified result,
/// not a second reimplementation of it. When NaNs ARE present: fill with
/// 0, sum, and divide by the per-slice COUNT OF NON-NAN elements (not the
/// raw axis length). NOTE (2026-08-01, corrected): unlike plain `mean`,
/// which forces its OWN sum to `dtype='f4'` for float16 input, real
/// numpy's `nanmean` forwards the CALLER'S raw `dtype=` (None by default)
/// straight to its internal `np.sum` call -- so a float16 `nanmean` with
/// no explicit `dtype=` sums via `sum`'s own narrow/wide layout-dependent
/// rule, NOT an always-float32 widen. Verified empirically after an
/// out-of-corpus fuzz probe caught the previous (wrong) always-widen
/// implementation diverging on EVERY float16 case, including plain
/// C-contiguous ones. See the sum-step comment inside the function body.
#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, keepdims=None, r#where=None))]
#[allow(clippy::too_many_arguments)]
fn nanmean(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims_resolved = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;

    // This guard used to live below the no-NaN short-circuit (see that
    // branch just below), which made it unreachable for exactly the
    // declared bug case: `np.nanmean([1.,4.,9.,16.], dtype=np.int64)` has
    // no actual NaNs, so it took the short-circuit straight to `mean()`
    // before this check ever ran -- silently returning a float64 result
    // instead of raising `TypeError: If a is inexact, then dtype must be
    // inexact` like real numpy does (verified: real `nanmean` runs this
    // check itself, upfront, in `_replace_nan`/`_divide_by_count`'s caller
    // in `_nanfunctions_impl.py`, before it ever looks at whether the
    // input actually contains a NaN). Moved here so it always runs,
    // independent of the short-circuit decision below.
    let early_dtype_override: Option<DType> = match dtype {
        None => None,
        Some(d) if d.is_none() => None,
        Some(d) => Some(crate::dtype_from_pyobj_no_su(d)?),
    };
    if (arr.dtype().is_floating() || arr.dtype().is_complex())
        && early_dtype_override.is_some_and(|dt| !(dt.is_floating() || dt.is_complex()))
    {
        return Err(pyo3::exceptions::PyTypeError::new_err("If a is inexact, then dtype must be inexact"));
    }

    let isnan_mask = isnan_array(&arr).map_err(to_py_err)?;
    // The no-NaN-present short-circuit to plain `mean` is only safe when
    // `mean`'s own sum step computes identically to `nanmean`'s -- true for
    // every dtype EXCEPT float16 (2026-08-01, found investigating a
    // real-numpy-vs-ionp divergence in this exact short-circuit): plain
    // `mean` forces its internal sum to a widened float32 accumulator for
    // float16 input regardless of memory layout (see `mean`'s own doc
    // comment above), while real `nanmean` calls `np.sum(arr, dtype=dtype)`
    // with the caller's raw `dtype=` (None here), which uses `sum`'s own
    // memory-layout-dependent narrow/wide float16 rule instead -- verified
    // directly against real numpy 2.5.1: `np.mean` and `np.nanmean` give
    // DIFFERENT float16 results on the identical, NaN-free input
    // (`np.mean(a, axis=0) == [-8.016, ...]` vs `np.nanmean(a, axis=0) ==
    // [-8.02, ...]` for one concrete example), so numpy's own `nanmean`
    // does NOT special-case the no-NaN case onto `mean`'s algorithm either.
    // Skipping the short-circuit for float16 (falling through to this
    // function's own sum-based path below, which already matches `sum`'s
    // bit-exact narrow/wide rule) fixes this; every other dtype keeps the
    // fast path since it's provably equivalent there.
    if !any_true(&isnan_mask) && arr.dtype() != DType::F16 {
        // BUG FOUND + FIXED 2026-08-03 (Monday): this short-circuit is
        // sound for the ARITHMETIC (that is what the long comment above
        // establishes) but it also silently inherited `mean`'s AXIS
        // VALIDATION, which numpy deliberately does not share between the
        // two. On a 0-d operand real numpy accepts `nanmean(x, axis=0)`
        // and REJECTS `mean(x, axis=0)` with `AxisError` -- measured, see
        // `reduce_axes_from_pyobj`. Delegating verbatim made `nanmean`
        // raise, i.e. a composition that is wrong even though every
        // primitive in it is right.
        //
        // Validate here under `nanmean`'s OWN rule (courtesy granted), then
        // delegate with `axis=None`: on a 0-d operand the courtesy axis
        // reduces over nothing, which is precisely what `axis=None` already
        // means there, so this changes no arithmetic. A tuple axis or an
        // out-of-range axis still raises from the call below, before any
        // delegation happens.
        if arr.ndim() == 0 {
            if let Some(ax) = axis.filter(|ax| !ax.is_none()) {
                // Courtesy is `nan_aware AND inexact` -- see rule 3 in
                // `reduce_axes_from_pyobj`. On an integer or bool operand
                // numpy's `nanmean` delegates to `mean` and inherits its
                // strict validation, so the courtesy must be withdrawn
                // here or `nanmean(np.array(3), axis=0)` wrongly succeeds.
                // BUG FOUND + FIXED (2026-08-04, axis-form sweep). On an
                // EXACT operand this must not merely withdraw the
                // courtesy -- it must hand the whole call to `mean`,
                // RAW AXIS AND ALL. numpy's `nanmean` is `mean` there:
                // `_replace_nan` returns `mask is None` for an exact
                // dtype and the function `return`s `np.mean(...)`
                // immediately, so `mean`'s converter is the one that
                // produces the message. Running the strict converter
                // here instead produced the reduce family's messages,
                // which are a different family again. Measured
                // 2026-08-04, `np.nanmean(np.zeros((), int), axis=...)`:
                //     0.0       np: integer argument expected, got float
                //               anionpy: 'float' object cannot be interp...
                //     True      np: AxisError: axis 1 is out of bounds
                //               anionpy: an integer is required
                //     np.True_  np: 'numpy.bool' object cannot be int...
                //               anionpy: an integer is required
                // Verified dtype-gated, NOT shape-gated: for int64 and
                // bool, `nanmean`'s message equals `mean`'s at every
                // shape probed ((), (5,), (2, 3)); for float64 it
                // differs at all three. The 0-d rows were simply where
                // the two checks first disagreed observably.
                // No arithmetic risk in forwarding the axis: there is no
                // axis a 0-d exact operand ACCEPTS here -- `mean` raises
                // `AxisError` for every one -- so this branch can only
                // ever turn one error message into the right one.
                if !arr.dtype().is_inexact() {
                    return mean(py, a, axis, dtype, out, keepdims, r#where);
                }
                reduce_axes_from_pyobj(ax, 0, true, AxisSeq::TupleOnly)?;
                return mean(py, a, None, dtype, out, keepdims, r#where);
            }
        }
        // Delegating to `mean` would also inherit `mean`'s axis ERROR
        // MESSAGES, which are not `nanmean`'s on an inexact operand:
        // `np.nanmean(f64, axis=0.0)` says "'float' object cannot be
        // interpreted as an integer" where `np.mean` says "integer argument
        // expected, got float" (measured 2026-08-04). Validate here first,
        // under this function's own rule, so the delegation stays an
        // ARITHMETIC shortcut and never a semantic one. On an exact operand
        // numpy really does delegate, message and all, so leave that alone.
        if arr.dtype().is_inexact() {
            if let Some(ax) = axis.filter(|ax| !ax.is_none()) {
                nan_family_axis_check(ax)?;
                reduce_axes_from_pyobj(ax, arr.ndim(), true, AxisSeq::TupleOnly)?;
            }
        }
        return mean(py, a, axis, dtype, out, keepdims, r#where);
    }

    let ndim = arr.ndim();
    let axes_raw: Option<Vec<isize>> = match axis {
        None => None,
        Some(ax) if ax.is_none() => None,
        // 0-d courtesy: nanmean -- numpy accepts (unlike plain mean!), but
        // only for an INEXACT operand; see rule 3 in reduce_axes_from_pyobj.
        Some(ax) => Some(reduce_axes_from_pyobj(ax, ndim, arr.dtype().is_inexact(), AxisSeq::TupleOnly)?),
    };
    let axes = normalize_reduce_axes(axes_raw.as_deref(), ndim).map_err(to_py_err)?;

    let dtype_override = early_dtype_override;

    let mask_arr: Option<NdArray> = match r#where {
        None => None,
        Some(w) => match w.extract::<bool>() {
            Ok(true) => None,
            Ok(false) => Some(extract_array_like(w)?),
            Err(_) => Some(extract_array_like(w)?),
        },
    };

    // NaN-exclusion happens at the VALUE level (fill-with-0 for the sum,
    // not-nan-as-1/0 for the count), NOT by folding into the reduce
    // mask -- matching real numpy's `nanmean`, which passes the caller's
    // OWN `where=where` straight through to `np.sum` unchanged (see
    // `_nanfunctions_impl.py`). Folding not-nan into the mask too would
    // force every NaN-containing reduction onto `reduce_axis`'s masked/
    // sequential path even when the caller passed no `where=` at all
    // (the common case), losing pairwise bit-exactness for no reason --
    // caught and corrected before this shipped, not after a failing test.
    let not_nan_i64 = {
        let not_nan = binary_op(BinaryOp::NotEqual, &isnan_mask, &scalar_bool(true)?).map_err(to_py_err)?;
        not_nan.cast_to(DType::I64)
    };

    // Unlike plain `mean` (which forces its OWN `umr_sum` call to `dtype=
    // 'f4'` for float16 input -- see `mean`'s doc comment above), real
    // numpy's `nanmean` (`_nanfunctions_impl.py`) does `tot = np.sum(arr,
    // axis=axis, dtype=dtype, ...)` with `dtype` the CALLER'S OWN raw
    // dtype= argument (None by default) -- NOT internally forced to
    // float32 for a float16 input. So `tot` for a float16 array with no
    // explicit `dtype=` stays float16-typed, computed via `sum`'s own
    // memory-layout-dependent narrow/wide accumulator rule (finding 3
    // above), exactly like calling `sum` directly -- verified empirically:
    // `np.sum(filled, axis=0)` (dtype=None, output float16) then dividing
    // by the int64 count reproduces real `nanmean`'s float16 output
    // bit-exactly, where forcing an f4 sum first (mean's own rule) does
    // NOT. Passing `dtype_override` straight through (instead of a forced
    // F32) matches this.
    // Compose at keepdims=false (the natural, retained-axes-only shape --
    // see `mean`'s matching comment for the full mechanism: real numpy's
    // `nanmean` divide is also `out=`-based, so it must reuse
    // `relayout_for_reduction` -- the exact function `sum_result` itself is
    // built with -- rather than `k_order_relayout_composed` (which is only
    // correct for `ptp`'s genuine fresh order='K' composition)).
    let filled = nan_fill(&arr, 0.0);
    let sum_result = reduce_axis(BinaryOp::Add, &filled, &axes, false, dtype_override, None, mask_arr.as_ref())
        .map_err(to_py_err)?;

    // Same double-precision-widened divide as plain `mean` above (see its
    // doc comment): `nanmean`'s divide is `um.true_divide(ret, rcount,
    // out=ret, ...)` with `rcount` an `numpy.int64` too, so the same
    // NEP-50 strong-promotion widening applies here verbatim.
    let out_dtype = sum_result.dtype();
    let divide_widen_dtype = match out_dtype {
        DType::F16 | DType::F32 => Some(DType::F64),
        DType::C64 => Some(DType::C128),
        _ => None,
    };
    let divide_dtype = divide_widen_dtype.unwrap_or(out_dtype);

    // Same composition bug as plain `mean`'s masked branch: `not_nan_i64` is
    // a `binary_op`+`cast_to` intermediate (always C-contiguous per
    // `binary_op`'s own policy), so feeding it straight into `reduce_axis`
    // makes the internal layout decision key off ITS fresh C-contig layout,
    // not `arr`'s real one. Compute the reduction natural (keepdims=false,
    // so `reduce_axis`'s own relayout-by-`not_nan_i64` step is a no-op on a
    // fully-collapsed shape) and then re-derive the true layout from `arr`'s
    // real shape/strides via `relayout_for_reduction`.
    let count_natural = reduce_axis(BinaryOp::Add, &not_nan_i64, &axes, false, None, None, mask_arr.as_ref())
        .map_err(to_py_err)?;
    let count_natural = count_natural.relayout_for_reduction(arr.shape(), arr.strides(), &axes, false);
    let count_result = count_natural.cast_to(divide_dtype);
    let sum_for_divide = if divide_widen_dtype.is_some() { sum_result.cast_to(divide_dtype) } else { sum_result.clone() };

    let raw_result = binary_op(BinaryOp::Divide, &sum_for_divide, &count_result).map_err(to_py_err)?;
    // Same natural-shape-then-relayout-once mechanism as `mean` (see its
    // doc comment above the equivalent line for the full real-numpy
    // `out=ret` justification): do ALL casting on this plain C-contiguous,
    // keepdims=false result first, then relayout once at the very end.
    let mut result = raw_result;
    if divide_widen_dtype.is_some() {
        result = result.cast_to(out_dtype);
    }
    // Same general cast-back-to-target-dtype fix as plain `mean` above (see
    // its doc comment for the full mechanism): once the inexact-dtype
    // guard runs unconditionally (fixed above), the only remaining gap is
    // this final cast, needed e.g. for `np.nanmean(complex_arr,
    // dtype=np.complex64)` where `divide_widen_dtype` widens to C128 but
    // the caller asked for C64.
    let effective_out_dtype = dtype_override.unwrap_or(arr.dtype());
    if result.dtype() != effective_out_dtype {
        result = result.cast_to(effective_out_dtype);
    }
    // Final step, after ALL casting is done: relayout onto sum's own
    // (keepdims-aware) layout.
    result = result.relayout_for_reduction(arr.shape(), arr.strides(), &axes, keepdims_resolved);

    let isbad = binary_op(BinaryOp::Equal, &count_result, &scalar_f64(0.0)?.cast_to(count_result.dtype())).map_err(to_py_err)?;
    if any_true(&isbad) {
        warn_runtime(py, c"Mean of empty slice")?;
    }

    wrap_reduction(py, result, out)
}

// ---------------------------------------------------------------------------
// Tier 6: `var`/`std`/`nanvar`/`nanstd` -- numpy's two-pass algorithm
// (`_core/_methods.py`'s `_var`/`_std`), composed from `reduce_axis`/
// `binary_op`/`math_unary_op(Sqrt)` -- no new arithmetic kernel. Complex
// input is explicitly rejected (`NotImplementedError`, not silently wrong)
// -- see module doc for why it's scoped out this pass.
// ---------------------------------------------------------------------------

fn resolve_ddof(ddof: f64, correction: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    match correction {
        None => Ok(ddof),
        Some(c) if c.is_none() => Ok(ddof),
        Some(c) => {
            let corr: f64 = c.extract()?;
            if ddof != 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "ddof and correction can't be provided simultaneously.",
                ));
            }
            Ok(corr)
        }
    }
}

/// `_core/_methods.py::_var`'s `elif (_float_dtype := _complex_to_float.get(x.dtype))
/// is not None:` branch: `|deviation|^2 = re^2 + im^2`, computed directly
/// (NOT via a separate magnitude-then-square step, which would round-trip
/// through a sqrt and lose precision numpy's own direct form doesn't),
/// landing on the companion REAL dtype of `x`'s complex dtype
/// (`complex64 -> float32`, `complex128 -> float64` -- the ONLY two complex
/// dtypes this crate has, so no lookup table is needed).
fn complex_abs_sq(x: &NdArray) -> NdArray {
    let contig = x.to_contiguous();
    let out_buffer = match contig.buffer() {
        Buffer::C64(v) => Buffer::F32(v.iter().map(|c| c.re * c.re + c.im * c.im).collect()),
        Buffer::C128(v) => Buffer::F64(v.iter().map(|c| c.re * c.re + c.im * c.im).collect()),
        other => other.clone(),
    };
    NdArray::from_buffer(out_buffer, contig.shape().to_vec(), Order::C)
        .expect("complex_abs_sq: buffer/shape already validated by to_contiguous()'s own construction")
}

/// `var`/`std`/`nanvar`/`nanstd`'s no-`dtype=`-override output dtype:
/// "the same as the array type" for every real dtype (matching real
/// numpy's own default), but for complex input the variance/stddev of
/// complex data is always REAL (companion float: `complex64 -> float32`,
/// `complex128 -> float64`) -- verified against real numpy 2.5.1:
/// `np.var(np.array([1+2j, 3+4j], dtype=np.complex64))` returns a plain
/// `float32`, not `complex64`.
fn default_var_out_dtype(arr_dtype: DType) -> DType {
    match arr_dtype {
        DType::C64 => DType::F32,
        DType::C128 => DType::F64,
        other => other,
    }
}

#[allow(clippy::too_many_arguments)]
fn do_var(
    py: Python<'_>,
    arr: &NdArray,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    ddof: f64,
    keepdims: bool,
    r#where: Option<&Bound<'_, PyAny>>,
    mean_obj: Option<&Bound<'_, PyAny>>,
    correction: Option<&Bound<'_, PyAny>>,
    take_sqrt: bool,
    nan_aware: bool,
) -> PyResult<Py<PyAny>> {
    let ndim = arr.ndim();
    // MEAN-FAMILY axis errors, but only for the calls that actually land in
    // `var`/`std`. `nanvar`/`nanstd` on an INEXACT operand run their own
    // Python-level path and report `sum`-style messages instead; on an exact
    // one `_replace_nan` hands the array straight to `var`/`std` and the
    // mean-family message is what surfaces. Measured live against numpy
    // 2.5.1 on 2026-08-04 across {var, std, nanvar, nanstd} x {float64,
    // int64} x {0.0, (0.0,), np.float64(0)} -- the split is exactly this
    // predicate, with no exceptions in that grid.
    if let Some(ax) = axis {
        if !nan_aware || !arr.dtype().is_inexact() {
            mean_family_axis_check(ax, arr.ndim())?;
        }
    }
    let axes_raw: Option<Vec<isize>> = match axis {
        None => None,
        Some(ax) if ax.is_none() => None,
        // 0-d courtesy: do_var: var/std RAISE, nanvar/nanstd accept -- but
        // nanvar/nanstd accept only for an INEXACT operand, since numpy's
        // `_replace_nan` hands an exact one straight to var/std. Rule 3 in
        // reduce_axes_from_pyobj. `nan_aware` alone re-opens the int hole.
        Some(ax) => {
            Some(reduce_axes_from_pyobj(ax, ndim, nan_aware && arr.dtype().is_inexact(), AxisSeq::TupleOnly)?)
        }
    };
    let axes = normalize_reduce_axes(axes_raw.as_deref(), ndim).map_err(to_py_err)?;
    let ddof = resolve_ddof(ddof, correction)?;

    let dtype_override: Option<DType> = match dtype {
        None => None,
        Some(d) if d.is_none() => None,
        // Numeric reduction target: no S/U arm downstream, decline cleanly.
        Some(d) => Some(crate::dtype_from_pyobj_no_su(d)?),
    };
    // Bug found via this task's `dtype=` probe (withdrawn in fdcdfb6): this
    // guard used to apply unconditionally to BOTH `var`/`std` and
    // `nanvar`/`nanstd`, but real numpy only raises `TypeError: If a is
    // inexact, then dtype must be inexact` for the nan-aware pair --
    // verified directly: `np.var(np.array([1.,4.,9.,16.]), dtype=np.int64)`
    // succeeds and returns int64 `32` in real numpy 2.5.1, it does NOT
    // raise. Only `nanvar`/`nanstd` carry this restriction (inherited from
    // `_nanfunctions_impl.py`'s own explicit check, which plain `_methods.py`
    // has no equivalent of). Gated on `nan_aware` to match.
    if nan_aware
        && (arr.dtype().is_floating() || arr.dtype().is_complex())
        && dtype_override.is_some_and(|dt| !(dt.is_floating() || dt.is_complex()))
    {
        return Err(pyo3::exceptions::PyTypeError::new_err("If a is inexact, then dtype must be inexact"));
    }
    let compute_dtype = match dtype_override {
        Some(dt) => Some(dt),
        None if arr.dtype() == DType::Bool || arr.dtype().is_integer() => Some(DType::F64),
        None => None,
    };

    let user_mask: Option<NdArray> = match r#where {
        None => None,
        Some(w) => match w.extract::<bool>() {
            Ok(true) => None,
            Ok(false) => Some(extract_array_like(w)?),
            Err(_) => Some(extract_array_like(w)?),
        },
    };

    // `nan_aware`: NaN-exclusion happens at the VALUE level ONLY -- a
    // not-nan indicator (0/1) summed for the count, an explicit
    // zero-fill/zero-out for the sum/squared-deviation steps (matching
    // real `nanvar`'s `_copyto(sqr, 0, mask)`) -- never by folding
    // not-nan into the reduce mask itself. The mask handed to every
    // `reduce_axis` call below is ALWAYS just `user_mask` (the caller's
    // own `where=`, or `None`), exactly like plain `var`/`std` above, so
    // an ordinary call with no `where=` still takes the full/pairwise
    // path over the (already NaN-zeroed) array rather than being forced
    // onto the masked/sequential path for no reason -- the same fix
    // applied to `nanmean` above, applied here too.
    let isnan_mask: Option<NdArray> = if nan_aware { Some(isnan_array(arr).map_err(to_py_err)?) } else { None };
    // Real `nanvar`/`nanstd` (`_nanfunctions_impl.py`) short-circuit through
    // `_replace_nan`: `mask is None` -- and the function falls straight
    // through to plain `np.var`/`np.std`, no NaN-overwrite-on-dof<=0 step
    // at all -- ONLY when the array's dtype is NOT inexact (bool/int,
    // where NaN is not even representable). CORRECTED 2026-08-02 (see the
    // `x.cast_to` truncation fix a few lines up, same root cause): this
    // does NOT depend on whether the data actually CONTAINS a NaN --
    // `_replace_nan` computes `mask = np.isnan(a)` for every inexact
    // (float/complex) array unconditionally, all-False mask included, so
    // nanvar's own dof<=0-overwrite dance runs for EVERY float/complex
    // input, verified directly: `np.nanvar(np.array([1+2j],
    // dtype=complex64), ddof=1)` (a single-element array, no NaN
    // anywhere) is `nan`, where `np.var` on the same data is `inf` -- the
    // gate below is `arr.dtype()` being inexact, not `has_nan`. Caught via
    // empirical sweep: `np.nanvar([1,2,3,4,5], ddof=5)` (an int8 array, so
    // no NaN is even possible -- and mask really is None here) returns
    // `inf`, matching plain `np.var`, NOT `nan`, confirming the dtype-based
    // gate rather than a data-presence-based one. (No `has_nan` variable
    // needed any more -- both gates that used to read it now read
    // `arr.dtype()` directly instead, see above/below.)
    let count_value: NdArray = match &isnan_mask {
        Some(nm) => {
            let not_nan = binary_op(BinaryOp::NotEqual, nm, &scalar_bool(true)?).map_err(to_py_err)?;
            not_nan.cast_to(DType::I64)
        }
        None => NdArray::from_buffer(Buffer::I64(vec![1i64; arr.size()]), arr.shape().to_vec(), Order::C).map_err(to_py_err)?,
    };

    let compute_arr = if nan_aware { nan_fill(arr, 0.0) } else { arr.clone() };

    let arrmean: NdArray = match mean_obj {
        Some(m) if !m.is_none() => extract_array_like(m)?,
        _ => {
            let rcount_kt = reduce_axis(BinaryOp::Add, &count_value, &axes, true, None, None, user_mask.as_ref()).map_err(to_py_err)?;
            let sum_kt = reduce_axis(BinaryOp::Add, &compute_arr, &axes, true, compute_dtype, None, user_mask.as_ref()).map_err(to_py_err)?;
            let rcount_kt_cast = rcount_kt.cast_to(sum_kt.dtype());
            let divided = binary_op(BinaryOp::Divide, &sum_kt, &rcount_kt_cast).map_err(to_py_err)?;
            // Bug found while measuring the dtype= cast-back fix's own
            // differential coverage (2026-08-02, out-of-corpus probe):
            // real numpy's `_var` computes `arrmean` via
            // `um.true_divide(arrmean, div, out=arrmean, casting='unsafe')`
            // -- the `out=arrmean` forces the quotient BACK into the sum's
            // own accumulator dtype (e.g. int64) BEFORE it is ever used to
            // form the `x = arr - arrmean` deviations, unconditionally
            // (not just when `dtype=` is explicit -- `out=arrmean` runs
            // every time). `binary_op(Divide, ..)` here is a genuine
            // true-divide with no such `out=` cast-back, so an integer
            // accumulator dtype (default `compute_dtype` for bool/int
            // input, or an explicit `dtype=int64`) silently produced a
            // float64 `arrmean`, corrupting every downstream deviation --
            // verified against real numpy: `np.var(np.array([1.5, 2.5,
            // 3.5, 4.5], dtype=np.float32), dtype=np.int64)` is int64 `2`
            // (mean truncates 10/4=2.5 -> int64 2 FIRST, deviations become
            // [-0.5, 0.5, 1.5, 2.5] -> squares [0.25, 0.25, 2.25, 6.25] ->
            // summed truncating per-add into int64 -> 8 -> /4 -> 2), not
            // the `1` an un-cast-back float arrmean of 2.5 (deviations
            // [-1, 0, 1, 2] -> squares [1, 0, 1, 4] -> sum 6 -> /4 -> 1.5
            // -> truncate once at the very end -> 1) would give.
            if divided.dtype() != sum_kt.dtype() {
                divided.cast_to(sum_kt.dtype())
            } else {
                divided
            }
        }
    };

    // `mean=` corpus gap (2026-08-02, coordinator signature-diff task): a
    // genuinely non-broadcastable `mean=` for the nan-aware pair raises a
    // DIFFERENT broadcast message than the plain pair -- real numpy's
    // `nanvar`/`nanstd` compute this deviation via a single 4-operand
    // ufunc call, `np.subtract(arr, avg, out=arr, casting='unsafe',
    // where=where)` (see the un-gated doc comment just below), so ITS OWN
    // broadcast-mismatch error lists all 4 operand shapes -- input `arr`,
    // input `avg`(`mean=`), the `out=arr` output (== `arr`'s own shape
    // again), and `where` (the user's own `where=` shape if given, else
    // `()`, the internal default-True scalar) -- verified against real
    // numpy 2.5.1: `np.nanvar(np.ones((2,3)), mean=np.array([1.,2.]))`
    // raises `"operands could not be broadcast together with shapes (2,3)
    // (2,) (2,3) () "`, NOT the 2-operand `"...shapes (2,3) (2,) "`
    // `binary_op`'s own ordinary broadcast check produces (which plain
    // `var`/`std`'s un-`out=`d, un-`where=`d subtract genuinely only has
    // 2 operands for, and DOES match). Checked here, before the ordinary
    // `binary_op` call, so plain `var`/`std` (this whole block is gated on
    // `nan_aware`) keep the existing/correct 2-operand path untouched.
    if nan_aware {
        let where_shape: Vec<usize> = match &user_mask {
            Some(m) => m.shape().to_vec(),
            None => Vec::new(),
        };
        let bc_ok = ionp_core::shape::broadcast_shapes(compute_arr.shape(), arrmean.shape())
            .is_ok_and(|s| s == compute_arr.shape());
        if !bc_ok {
            fn fmt_shape_tuple_style(shape: &[usize]) -> String {
                let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                if dims.len() == 1 {
                    format!("({},)", dims[0])
                } else {
                    format!("({})", dims.join(","))
                }
            }
            let joined = [compute_arr.shape(), arrmean.shape(), compute_arr.shape(), &where_shape]
                .iter()
                .map(|s| fmt_shape_tuple_style(s))
                .collect::<Vec<_>>()
                .join(" ");
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "operands could not be broadcast together with shapes {joined} "
            )));
        }
    }

    let x = binary_op(BinaryOp::Subtract, &compute_arr, &arrmean).map_err(to_py_err)?;
    // Bug found via this task's DEFECT 2 probe (2026-08-02): real numpy's
    // `nanvar`/`nanstd` (`_nanfunctions_impl.py`) compute this deviation via
    // `np.subtract(arr, avg, out=arr, casting='unsafe', where=where)` --
    // `out=arr` is a genuine in-place write into the nan-filled COPY of the
    // original input, so the subtraction result is unsafe-cast DOWN to that
    // copy's dtype (== the original input's own dtype, e.g. float16) before
    // it is ever squared, regardless of how wide `avg`/`dtype=` made the
    // subtraction result. Plain `var`/`std` (`_core/_methods.py::_var`) has
    // no such truncation -- its own `out=` is literally `Ellipsis`, i.e. a
    // fresh full-precision array. Verified against real numpy 2.5.1:
    // `np.nanstd(float16_data, dtype='float64')` != a naive full-precision
    // recomputation; matches only when the deviation is truncated back to
    // float16 first. Gated on `nan_aware` so plain `var`/`std` keep their
    // existing full-precision behavior.
    //
    // SECOND bug found verifying this task's own new differential coverage
    // (2026-08-02, `sweep/1d/bool` case): the gate above was `nan_aware &&
    // x.dtype() != compute_arr.dtype()` with NO check that the ORIGINAL
    // input was itself inexact -- for a bool/int input, real numpy's
    // `_replace_nan` (`_nanfunctions_impl.py`) takes the `else: mask = None`
    // branch (only `issubclass(a.dtype.type, np.inexact)` gets the
    // NaN-scan/truncation treatment at all) and `nanvar`/`nanstd` then
    // short-circuit ENTIRELY to plain `np.var`/`np.std` on the untouched
    // int/bool array the moment `mask is None` -- none of nanvar's own
    // subtract/copyto/truncate dance ever runs for a non-inexact input.
    // Truncating `x` down to `compute_arr.dtype()` (bool, for a bool input)
    // silently collapsed every nonzero deviation to `true`/`1.0` and every
    // zero deviation to `false`/`0.0` before squaring -- verified against
    // real numpy: `np.nanvar([True, False, True, False, True, False,
    // True])` is `0.24489795918367346`, anionpy's un-gated truncation gave
    // `1.0`. Fixed by requiring `arr.dtype()` (the caller's ORIGINAL input,
    // not the accumulator) be floating or complex before the truncation
    // applies, matching `_replace_nan`'s own `issubclass(..., np.inexact)`
    // gate exactly.
    let x = if nan_aware
        && (arr.dtype().is_floating() || arr.dtype().is_complex())
        && x.dtype() != compute_arr.dtype()
    {
        x.cast_to(compute_arr.dtype())
    } else {
        x
    };
    // DEFECT 1 fix: complex `x` (the mean-subtracted deviation) squares to
    // a REAL magnitude via `complex_abs_sq` (`|x|^2 = re^2 + im^2`),
    // matching real numpy's `_complex_to_float` branch (`_var`) / the
    // `multiply(arr, arr.conj(), out=arr).real` branch (`nanvar`) -- see
    // `complex_abs_sq`'s own doc comment. Every other dtype keeps the
    // ordinary same-dtype `x * x` square.
    let mut x2 = if x.dtype().is_complex() {
        complex_abs_sq(&x)
    } else {
        binary_op(BinaryOp::Multiply, &x, &x).map_err(to_py_err)?
    };
    if let Some(nm) = &isnan_mask {
        x2 = ionp_core::ufunc::overwrite_scalar_where(&x2, nm, 0.0);
    }

    let ssq = reduce_axis(BinaryOp::Add, &x2, &axes, keepdims, compute_dtype, None, user_mask.as_ref()).map_err(to_py_err)?;

    let rcount_final = reduce_axis(BinaryOp::Add, &count_value, &axes, keepdims, None, None, user_mask.as_ref()).map_err(to_py_err)?;
    let rc_f64 = rcount_final.cast_to(DType::F64);
    let diff = binary_op(BinaryOp::Subtract, &rc_f64, &scalar_f64(ddof)?).map_err(to_py_err)?;
    let dof = binary_op(BinaryOp::Maximum, &diff, &scalar_f64(0.0)?).map_err(to_py_err)?;
    let isbad = binary_op(BinaryOp::LessEqual, &dof, &scalar_f64(0.0)?).map_err(to_py_err)?;

    let dof_cast = dof.cast_to(ssq.dtype());
    let mut result = binary_op(BinaryOp::Divide, &ssq, &dof_cast).map_err(to_py_err)?;

    // Plain `var`/`std` (`_core/_methods.py::_var`) only WARNS on
    // dof<=0 and lets the natural division stand (positive-ssq/0 -> +inf,
    // 0/0 -> NaN already) -- there is NO explicit NaN-overwrite step.
    // `nanvar`/`nanstd` (`_nanfunctions_impl.py::nanvar`), by contrast,
    // DOES explicitly `_copyto(var, np.nan, isbad)` after warning, AND
    // uses a warning message with a trailing period ("...for slice.")
    // where plain `var`'s does not ("...for slice") -- verified against
    // real numpy 2.5.1 source for both message text and the presence/
    // absence of the overwrite; conflating the two was an early draft
    // bug caught by empirical smoke-testing (`np.var([1.,3.,5.], ddof=3)
    // == inf`, NOT `nan`) before any sweep/declaration.
    //
    // Gate is `arr.dtype()` inexact, NOT `has_nan` (fixed 2026-08-02, see
    // `has_nan`'s doc comment above for the empirical proof: a
    // zero-actual-NaN complex64 array still gets the overwrite in real
    // numpy, because `nanvar`'s own dof<=0 dance runs for every
    // float/complex input regardless of whether any element is actually
    // NaN -- only a non-inexact bool/int array skips it entirely by
    // delegating straight to plain `np.var`).
    if any_true(&isbad) {
        if nan_aware && (arr.dtype().is_floating() || arr.dtype().is_complex()) {
            warn_runtime(py, c"Degrees of freedom <= 0 for slice.")?;
            result = overwrite_nan_where(&result, &isbad);
        } else {
            warn_runtime(py, c"Degrees of freedom <= 0 for slice")?;
        }
    }

    // Statistics-grid DEFECT 1/2 fix (2026-08-02): when `out=` is given,
    // real numpy's own control flow diverges from the no-`out=` path this
    // whole function was originally written against, so the two are now
    // handled as genuinely separate branches rather than threading `out`
    // through the existing `would_be_scalar`/`effective_out_dtype` logic
    // below (which stays exactly as-is for the `out is None` case).
    //
    // Real numpy's `std`/`nanstd` (`_core/_methods.py::_std`) are a thin
    // wrapper: `ret = _var(a, ..., out=out, ...)` (the variance, written
    // DIRECTLY into `out` -- an unconditional/`casting='unsafe'` write, see
    // `_var`'s own `um.true_divide(..., out=arrmean)`-style casts) followed
    // by `ret = um.sqrt(ret, out=ret)` as a SEPARATE, genuine in-place ufunc
    // call operating on `out`'s own buffer/dtype. So for `std`/`nanstd` with
    // `out=`, the relevant sqrt-casting dtype is `out`'s OWN dtype -- NOT
    // the `dtype=`/`compute_dtype`-derived `sqrt_target_dtype` the no-`out=`
    // branch below uses, and NOT gated on `would_be_scalar` (verified
    // against real numpy 2.5.1: `np.std(np.array([1.,2.,3.,4.]),
    // out=np.zeros((), dtype='int32'))` -- a FULL reduction, the exact
    // shape `would_be_scalar` would call scalar for the no-`out=` case --
    // still raises the `UFuncTypeError`, confirming `out=` bypasses that
    // suppression entirely).
    if let Some(out_obj) = out {
        let out_ref = out_obj
            .cast::<PyArray>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("out= must be an anionpy.ndarray"))?;
        let (out_ndim, out_dtype) = {
            let b = out_ref.borrow();
            (b.inner.ndim(), b.inner.dtype())
        };

        // DEFECT 2, nanvar/nanstd's own extra gate: `_nanfunctions_impl.py`
        // checks `if (np.issubdtype(a.dtype, np.inexact) and out is not None
        // and not np.issubdtype(out.dtype, np.inexact))` and raises BEFORE
        // ever delegating to `_var`/`_std` -- verified against real numpy
        // 2.5.1: `np.nanvar(np.array([1.,2.,np.nan,4.]), out=np.zeros((),
        // dtype='int32'))` raises `TypeError: If a is inexact, then out
        // must be inexact`, and the same call with `np.nanstd` raises the
        // IDENTICAL message (not the sqrt-casting error), confirming this
        // check runs first and takes priority for both. Plain `var`/`std`
        // (non-`nan_aware`) have no such restriction at all -- inexact
        // input into a non-inexact `out=` is always accepted (verified:
        // `np.var(np.array([1.,2.,3.,4.]), out=np.zeros((),
        // dtype='int32'))` succeeds), and non-inexact-input `nanvar`/
        // `nanstd` (bool/int `a`) skip this gate entirely and behave exactly
        // like plain `var`/`std` (verified: `np.nanvar(np.array([1,2,3,4],
        // dtype='int32'), out=np.zeros((), dtype='int32'))` succeeds).
        //
        // ORDERING (statistics-grid FINDING 1 follow-up, 2026-08-02): this
        // gate must run BEFORE either of the two shape checks below, not
        // after -- verified against real numpy 2.5.1: `np.nanvar(a, ...,
        // out=np.zeros((2,), dtype='int32'))` where `a` is inexact and the
        // natural result is a 0-d scalar (an ndim MISMATCH, the case the
        // block below used to catch first) still raises `TypeError: If a is
        // inexact, then out must be inexact`, NOT the ndim-mismatch
        // ValueError -- because real `nanvar` is a thin Python wrapper that
        // checks this BEFORE ever calling into `_var` (where the ufunc
        // reduce that produces both shape-mismatch messages actually lives).
        // Same result for the new same-ndim-different-size check just below
        // it. This function used to check shape first; reordered here.
        if nan_aware
            && (arr.dtype().is_floating() || arr.dtype().is_complex())
            && !(out_dtype.is_floating() || out_dtype.is_complex())
        {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "If a is inexact, then out must be inexact",
            ));
        }

        // DEFECT 1: `out=`'s NUMBER OF DIMENSIONS is checked against the
        // reduction's own natural (pre-`out=`) result shape -- an EXACT
        // match, not the ordinary ufunc-`out=` broadcast-compatible rule
        // `ionp_core::ufunc::write_out` applies for non-reduction calls --
        // verified against real numpy 2.5.1:
        // `np.var(np.arange(6).reshape(2,3), axis=0, out=np.zeros((3,2)))`
        // raises `ValueError: output parameter for reduction operation add
        // has the wrong number of dimensions: Found 2 but expected 1`, even
        // though `(3,)` IS broadcastable into `(3,2)`'s trailing dims by the
        // ordinary rule. The underlying reduce op is always "add" for this
        // whole family (var/std/nanvar/nanstd all reduce a sum of squared
        // deviations).
        let out_shape: Vec<usize> = {
            let b = out_ref.borrow();
            b.inner.shape().to_vec()
        };
        if out_ndim != result.ndim() {
            // `keepdims=True` (2026-08-02, found by this task's own grid
            // rerun): real numpy appends a trailing
            // `" (must match the operand's when keepdims=True)"` clause to
            // this exact message whenever the reduction call itself was
            // made with `keepdims=True` -- verified against real numpy
            // 2.5.1: `np.var(np.ones((2,3)), axis=0, keepdims=True,
            // out=np.zeros((3,2)))` raises `"...Found 2 but expected 2
            // (must match the operand's when keepdims=True)"`, where the
            // identical call with `keepdims=False` raises the same prefix
            // with NO trailing clause at all.
            let suffix = if keepdims {
                " (must match the operand's when keepdims=True)"
            } else {
                ""
            };
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "output parameter for reduction operation add has the wrong number of dimensions: Found {out_ndim} but expected {}{suffix}",
                result.ndim()
            )));
        }

        // Statistics-grid FINDING 1 fix (2026-08-02, statgrid.py's new
        // `wrongsize` axis: SAME ndim as the natural result, but a
        // DIFFERENT extent along at least one dim -- the branch the
        // original grid's `wrongshape` construction could never reach,
        // since it always appended an extra trailing dim, guaranteeing an
        // ndim mismatch caught by DEFECT 1 above instead). Real numpy uses
        // a THIRD, differently-formatted message here, produced by the
        // underlying nditer/ufunc-reduce broadcast machinery, NOT the
        // reduction-specific ndim check above:
        //
        //   ValueError: operands could not be broadcast together with
        //   remapped shapes [original->remapped]: {out}->{remapped} {in}[ {where}]
        //
        // (a trailing space always follows the LAST shape shown, `where`
        // included). Both `{out}` and `{in}` are formatted the ordinary
        // numpy-tuple-minus-space way (`(4,)` for a 1-elem shape, WITH the
        // trailing comma; `(2,3)` for multi-elem, no space). `{remapped}`
        // differs by `keepdims`:
        //
        //  - `keepdims=True`: `{remapped}` == `{out}`'s own shape,
        //    formatted the SAME tuple-minus-space way (verified: a genuine
        //    1-elem case still shows the trailing comma on BOTH sides,
        //    `(2,)->(2,)`). No `newaxis` tokens ever appear when
        //    `keepdims=True` -- `out` already has one real (if
        //    wrong-sized) dim per original axis.
        //
        //  - `keepdims=False`: `{remapped}` is built by walking the
        //    original array's axes 0..ndim in order and, for each axis,
        //    emitting: NOTHING if the axis is part of the LEADING
        //    CONTIGUOUS PREFIX of reduced axes (i.e. axes 0..k-1 are ALL
        //    reduced, for the largest such k -- these are silently
        //    dropped, matching ordinary right-aligned broadcasting, no
        //    placeholder needed); the literal token `newaxis` if the axis
        //    is reduced but NOT part of that leading prefix (i.e. at least
        //    one earlier axis survived); otherwise the next real dim
        //    pulled off `out`'s own shape, left to right. The tokens are
        //    then joined with `,` (no trailing comma even for a single
        //    token) and wrapped in parens. Exhaustively verified against
        //    real numpy 2.5.1 across a 3-d `(2,3,4)` input crossing every
        //    single-axis and every pairwise-axis reduction, both
        //    `keepdims` states: e.g. `axis=0` (a leading-prefix reduce)
        //    gives `(4,)->(4) (2,3,4)` (no `newaxis`, no trailing comma on
        //    the remapped single dim); `axis=2` (last axis, but NOT a
        //    leading prefix since axes 0,1 survive first) gives
        //    `(3,4)->(3,4,newaxis) (2,3,4)`; `axis=(0,2)` gives
        //    `(4,)->(4,newaxis) (2,3,4)` (axis 0 dropped as the leading
        //    prefix, axis 1 survives as the real dim, axis 2 gets
        //    `newaxis`); `axis=(1,2)` gives
        //    `(3,)->(3,newaxis,newaxis) (2,3,4)` (axis 0 survives so there
        //    is NO leading prefix to drop at all, both reduced axes get
        //    their own `newaxis`).
        //
        // `where=`: `where=None` (absent) and the explicit `where=True`
        // produce IDENTICAL messages with no trailing shape at all --
        // real numpy's fast path elides the mask entirely as "no
        // filtering" and it never becomes an nditer operand. Any OTHER
        // `where` value (verified: scalar `False`, and an actual boolean
        // array) DOES become a real operand and its own shape is appended,
        // space-separated, after the input operand's shape -- `where=False`
        // (a 0-d array) appends `()`, and `where=`&lt;array of shape (2,3)&gt;
        // appends `(2,3)`. This reuses `user_mask` (computed far above,
        // `None` for both absent-`where` and explicit `where=True`, `Some`
        // otherwise) rather than re-deriving it.
        if out_shape != result.shape() {
            fn fmt_shape_tuple_style(shape: &[usize]) -> String {
                let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                if dims.len() == 1 {
                    format!("({},)", dims[0])
                } else {
                    format!("({})", dims.join(","))
                }
            }
            let out_str = fmt_shape_tuple_style(&out_shape);
            let in_str = fmt_shape_tuple_style(arr.shape());
            let remapped_str = if keepdims {
                fmt_shape_tuple_style(&out_shape)
            } else {
                // Leading contiguous prefix of reduced axes (`axes` is
                // sorted/deduped by `normalize_reduce_axes`).
                let mut prefix_len = 0usize;
                while prefix_len < ndim && axes.contains(&prefix_len) {
                    prefix_len += 1;
                }
                let mut tokens: Vec<String> = Vec::new();
                let mut out_idx = 0usize;
                for ax in 0..ndim {
                    if ax < prefix_len {
                        continue;
                    }
                    if axes.contains(&ax) {
                        tokens.push("newaxis".to_string());
                    } else {
                        tokens.push(out_shape.get(out_idx).map(|d| d.to_string()).unwrap_or_default());
                        out_idx += 1;
                    }
                }
                format!("({})", tokens.join(","))
            };
            let where_suffix = match &user_mask {
                Some(m) => format!(" {}", fmt_shape_tuple_style(m.shape())),
                None => String::new(),
            };
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "operands could not be broadcast together with remapped shapes [original->remapped]: {out_str}->{remapped_str} {in_str}{where_suffix} "
            )));
        }

        // The `_var`-stage write: unconditional/unsafe cast of the
        // (pre-sqrt, for std/nanstd) variance into `out`'s own dtype --
        // verified via the `dtype=`-precision-preserving probe (`np.var`
        // with `dtype='int8'` vs `dtype='float64'`, both written into the
        // SAME `out=int32`, produce genuinely different values, 16 vs
        // 10000, confirming `dtype=` still controls internal accumulation
        // precision and only the FINAL write targets `out`'s dtype
        // unconditionally).
        //
        // Statistics-grid FINDING 2 fix (2026-08-02, found via statgrid.py's
        // `wrongsize` rerun -- adjacent to, but NOT the same bug as, the
        // `wrongsize`/Finding-1 shape defect; isolated separately per the
        // coordinator's "adjacency is not evidence" instruction). Real
        // numpy's `_var` (`_core/_methods.py`) does NOT compute the natural
        // `ssq / dof` quotient once and cast it into `out` as a single fused
        // step -- it is genuinely two ufunc calls in sequence, both of which
        // target `out` directly: `ret = umr_sum(x2, axis, dtype, out=out,
        // keepdims=keepdims)` (the SUM of squared deviations is itself
        // unsafe-cast into `out`'s dtype FIRST), then `ret =
        // um.true_divide(ret, rcount, out=ret, casting='unsafe')` (a
        // SEPARATE true-divide of that already-cast sum, unsafe-cast back
        // into `out` again). These two orderings only diverge when the
        // compute-dtype accumulator saturates/overflows before the natural
        // quotient would -- verified against real numpy 2.5.1 (statgrid.py
        // tuple: `np.var(uint8_arr, dtype='float16', out=int32_zeros)` on a
        // noncontiguous `(2,3)` slice of `[0,0,255,0,0,255]`, `axis=None`,
        // `ddof=0`): the float16 sum-of-squared-deviations (~86700) exceeds
        // float16's ~65504 max and saturates the ACCUMULATOR to `inf`
        // BEFORE any division happens; numpy's sum-first-into-`out` step
        // unsafe-casts that `inf` to `2147483647` (int32 max) immediately,
        // and the SEPARATE divide-by-6-with-unsafe-cast-back step then
        // floors `2147483647 / 6` to `357913941` -- NOT `2147483647`, which
        // is what dividing `inf / 6 == inf` first (in float16/naturally)
        // and casting the still-infinite quotient only once at the very end
        // (anionpy's prior single-step behavior) produces instead. The
        // dof<=0 NaN-overwrite path above is left on the natural
        // (pre-`out`-cast) `result` exactly as before -- this two-stage
        // recompute only replaces the NORMAL (non-`isbad`) case, since real
        // numpy's own dof<=0 handling operates on values already written
        // into `out`, a narrower edge case this fix does not attempt to
        // additionally chase.
        if !any_true(&isbad) {
            let ssq_out = ssq.cast_to(out_dtype);
            let dof_out = dof.cast_to(ssq_out.dtype());
            let divided_out = binary_op(BinaryOp::Divide, &ssq_out, &dof_out).map_err(to_py_err)?;
            result = if divided_out.dtype() != out_dtype {
                divided_out.cast_to(out_dtype)
            } else {
                divided_out
            };
        } else if result.dtype() != out_dtype {
            result = result.cast_to(out_dtype);
        }

        if take_sqrt {
            if !(out_dtype.is_floating() || out_dtype.is_complex()) {
                return Err(std_sqrt_casting_err(sqrt_input_promoted_dtype(out_dtype), out_dtype));
            }
            result = math_unary_op(MathUnaryOp::Sqrt, &result).map_err(to_py_err)?;
            if result.dtype() != out_dtype {
                result = result.cast_to(out_dtype);
            }
        }

        return wrap_reduction(py, result, out);
    }

    if take_sqrt {
        let would_be_scalar = axes.len() == ndim && !keepdims;
        let sqrt_target_dtype = compute_dtype.unwrap_or_else(|| default_var_out_dtype(arr.dtype()));
        if !would_be_scalar
            && !(sqrt_target_dtype.is_floating() || sqrt_target_dtype.is_complex())
        {
            return Err(std_sqrt_output_casting_err(sqrt_target_dtype));
        }
        result = math_unary_op(MathUnaryOp::Sqrt, &result).map_err(to_py_err)?;
    }

    // Same general `dtype=`-override cast-back gap as `mean`/`nanmean`
    // above (see `mean`'s doc comment for the full mechanism): an explicit
    // `dtype=` that isn't already `result`'s own dtype never got cast back
    // to it -- `np.var(int_array, dtype=np.int64)` -> real numpy's int64
    // `32`, anionpy's float64 `32.0`. numpy's real `_var`/`_std` do this via
    // `ret = ret.dtype.type(ret / rcount)` / `um.sqrt(ret, out=ret)` with
    // `ret` an accumulator array of the REQUESTED dtype, i.e. an
    // unconditional final cast, not a special case for a handful of
    // dtypes. `take_sqrt`'s own sqrt (computed above) can itself widen an
    // integer accumulator to float internally, so this cast must run
    // AFTER it, not before, to reproduce `std`'s int-dtype truncation too.
    let effective_out_dtype = compute_dtype.unwrap_or_else(|| default_var_out_dtype(arr.dtype()));
    if result.dtype() != effective_out_dtype {
        result = result.cast_to(effective_out_dtype);
    }

    wrap_reduction(py, result, out)
}

// ---------------------------------------------------------------------------
// Tier 5b: sorting/searching -- PATH-TO-100.md's sort/search block. Thin
// marshaling wrappers over `ionp_core::sort`; see that module's own doc
// comment for the scope decision (argsort/partition/argpartition/msort
// deliberately absent) and the empirically-verified tie-break rules these
// wrappers must not disturb.
// ---------------------------------------------------------------------------

/// `kind=` string -> `stable` bool, matching `sort_axis`'s own `stable`
/// parameter meaning (true selects the no-signbit-tiebreak / natural
/// stable-sort rule; false selects the signbit-tiebreak rule). Verified
/// against real numpy 2.5.1's exact accepted spellings and exact error
/// message (`np.sort(a, kind='bogus')` raises `ValueError: sort kind must
/// be one of 'quick', 'heap', or 'stable' (got 'bogus')`).
fn kind_to_stable(kind: &str) -> PyResult<bool> {
    match kind.to_ascii_lowercase().as_str() {
        "quicksort" | "quick" | "heapsort" | "heap" => Ok(false),
        "mergesort" | "merge" | "stable" => Ok(true),
        _ => Err(PyValueError::new_err(format!(
            "sort kind must be one of 'quick', 'heap', or 'stable' (got '{kind}')"
        ))),
    }
}

/// `np.sort`. `axis` defaults to `-1` (NOT `None` -- unlike `argmin`'s own
/// `axis=None` default -- verified: `inspect.signature(np.sort)` shows
/// `axis=-1`); an explicit `axis=None` flattens first, matching
/// `ionp_core::sort::normalize_single_axis`'s own doc contract. `kind=` and
/// `stable=`/`descending=` are mutually exclusive (real numpy: `ValueError:
/// \`kind\` and keyword parameters can't be provided at the same time.`).
/// `order=` is accepted only as `None` (anionpy has no structured dtypes;
/// real numpy: `ValueError: Cannot specify order when the array has no
/// fields.` for any non-None value on a plain dtype array).
#[pyfunction]
#[pyo3(signature = (a, axis=-1, kind=None, order=None, *, stable=None, descending=None))]
#[allow(clippy::too_many_arguments)]
fn sort(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<isize>,
    kind: Option<&Bound<'_, PyAny>>,
    order: Option<&Bound<'_, PyAny>>,
    stable: Option<&Bound<'_, PyAny>>,
    descending: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    if let Some(o) = order {
        if !o.is_none() {
            return Err(PyValueError::new_err("Cannot specify order when the array has no fields."));
        }
    }
    if kind.is_some() && (stable.is_some() || descending.is_some()) {
        return Err(PyValueError::new_err(
            "`kind` and keyword parameters can't be provided at the same time. Use only one of them.",
        ));
    }
    // `kind=` accepts `str` (any subclass, including `np.str_`) and `bytes`
    // (ASCII-decoded) but not `bytearray` -- same str/bytes/bytearray shape
    // as `order=`/`casting=`/`side=` elsewhere in this project, verified
    // live against real numpy 2.5.1 (`np.sort(a, kind=b'quicksort')`
    // succeeds, `np.sort(a, kind=bytearray(b'quicksort'))` raises
    // `TypeError: sort kind must be str, not bytearray` matching the same
    // "must be str, not {type}" wording every other kwarg here uses; a
    // non-str/bytes/bytearray value like `1.5`/`None`/`object()` raises
    // that same shape, `{type}` naming ITS OWN type name).
    let kind: Option<String> = match kind {
        None => None,
        Some(v) => Some(crate::extract_str_or_ascii_bytes(v, "sort kind")?),
    };
    let kind = kind.as_deref();
    // `stable=`/`descending=` are plain Python truthiness in real numpy
    // (measured 2026-08-02, same as `svd`'s flags -- `np.sort(a,
    // stable=1.5)`/`stable="x"`/etc all succeed, truthiness-dispatched),
    // NOT the `keepdims_index_like` rule the reduction family above uses.
    let stable_flag: Option<bool> = match stable {
        None => None,
        Some(v) => Some(v.is_truthy()?),
    };
    let descending_flag: Option<bool> = match descending {
        None => None,
        Some(v) => Some(v.is_truthy()?),
    };
    let is_stable = match kind {
        Some(k) => kind_to_stable(k)?,
        None => stable_flag.unwrap_or(false),
    };
    let desc = descending_flag.unwrap_or(false);
    let arr = extract_array_like(a)?;
    let ndim = arr.ndim();
    let (target, ax): (NdArray, usize) = match axis {
        None => (arr.ravel_order("C").map_err(to_py_err)?, 0),
        Some(raw) => {
            let norm = normalize_single_axis(raw, ndim).map_err(to_py_err)?;
            (arr, norm)
        }
    };
    let result = sort_axis(&target, ax, is_stable, desc).map_err(to_py_err)?;
    wrap(py, result, None)
}

/// `np.argsort` -- the indices that would sort `a`, as `int64`.
///
/// Kwarg handling is `sort`'s, deliberately shared down to the error
/// strings (all measured against real numpy 2.5.1 on `argsort` itself, not
/// assumed from `sort`): the same `sort kind must be one of 'quick',
/// 'heap', or 'stable' (got '...')`, the same `Cannot specify order when
/// the array has no fields.`, the same `` `kind` and keyword parameters
/// can't be provided at the same time. ``, and the same str/bytes-yes,
/// bytearray-no rule for `kind=`.
///
/// ONE behaviour genuinely differs from `sort`, and it is not a detail
/// that could be inherited: 0-d input. `np.sort(np.array(5))` raises
/// `AxisError: axis -1 is out of bounds for array of dimension 0`, but
/// `np.argsort(np.array(5))` succeeds and returns `array([0])`. numpy
/// promotes the 0-d array to shape `(1,)` FIRST -- provable from the error
/// text on the failing case, which names the promoted rank, not the
/// original one:
///
///     np.argsort(np.array(5), axis=1)
///         -> AxisError: axis 1 is out of bounds for array of dimension 1
///
/// So the promotion happens before axis validation, which is why it is
/// done here as a `ravel` rather than as a special case inside the axis
/// check.
#[pyfunction]
#[pyo3(signature = (a, axis=-1, kind=None, order=None, *, stable=None, descending=None))]
#[allow(clippy::too_many_arguments)]
fn argsort(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<isize>,
    kind: Option<&Bound<'_, PyAny>>,
    order: Option<&Bound<'_, PyAny>>,
    stable: Option<&Bound<'_, PyAny>>,
    descending: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    if let Some(o) = order {
        if !o.is_none() {
            return Err(PyValueError::new_err("Cannot specify order when the array has no fields."));
        }
    }
    if kind.is_some() && (stable.is_some() || descending.is_some()) {
        return Err(PyValueError::new_err(
            "`kind` and keyword parameters can't be provided at the same time. Use only one of them.",
        ));
    }
    let kind: Option<String> = match kind {
        None => None,
        Some(v) => Some(crate::extract_str_or_ascii_bytes(v, "sort kind")?),
    };
    let kind = kind.as_deref();
    let stable_flag: Option<bool> = match stable {
        None => None,
        Some(v) => Some(v.is_truthy()?),
    };
    let descending_flag: Option<bool> = match descending {
        None => None,
        Some(v) => Some(v.is_truthy()?),
    };
    let is_stable = match kind {
        Some(k) => kind_to_stable(k)?,
        None => stable_flag.unwrap_or(false),
    };
    let desc = descending_flag.unwrap_or(false);
    let arr = extract_array_like(a)?;
    // 0-d -> (1,) before anything else; see the doc note above.
    let arr = if arr.ndim() == 0 { arr.ravel_order("C").map_err(to_py_err)? } else { arr };
    let ndim = arr.ndim();
    let (target, ax): (NdArray, usize) = match axis {
        None => (arr.ravel_order("C").map_err(to_py_err)?, 0),
        Some(raw) => {
            let norm = normalize_single_axis(raw, ndim).map_err(to_py_err)?;
            (arr, norm)
        }
    };
    let result = argsort_axis(&target, ax, is_stable, desc).map_err(to_py_err)?;
    wrap(py, result, None)
}

/// `np.sort_complex` -- no `axis=` parameter at all in real numpy, always
/// operates along the last axis (see `ionp_core::sort::sort_complex`'s own
/// doc comment for the dtype-promotion rule).
#[pyfunction]
fn sort_complex(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = extract_array_like(a)?;
    let result = core_sort_complex(&arr).map_err(to_py_err)?;
    wrap(py, result, None)
}

/// `np.lexsort(keys, axis=-1)`. `keys` is either a sequence of arrays
/// (each of the SAME shape, of any dimensionality) or a single array one
/// rank higher whose leading axis enumerates the keys (numpy: each ROW of
/// a `(nkeys, ...)`-shaped `keys` array is treated as one key, identical
/// to passing a tuple of its rows -- verified: `np.lexsort(np.array([a,
/// b]))` matches `np.lexsort((a, b))` exactly); Python's default
/// iteration protocol (used here via `try_iter`) already yields rows for
/// both a numpy array and an `anionpy.ndarray` (whose own `__getitem__`
/// supports integer row indexing), so the same iteration loop naturally
/// covers both call shapes without a special case.
///
/// N-d key bug found+fixed (coordinator's harder probe, 2026-08-01): this
/// used to (a) reject ANY `axis != -1` including `axis=0` with a generic
/// `ValueError`, based on the wrong assumption that keys are ALWAYS 1-D
/// regardless of how they were shaped, and (b) reject any actually-N-d
/// key outright in the core function. Verified against real numpy 2.5.1:
/// `np.lexsort((a, b), axis=0)` for `(2, 3)`/`(1, 4)`/`(4, 1)`-shaped
/// `a`/`b` genuinely works, sorting each line along `axis` independently
/// and returning a result of the SAME shape as the keys (not flattened).
/// `axis` is now validated against the keys' REAL ndim (via the core
/// function's own `normalize_single_axis`, same `AxisError` contract as
/// `sort`), not a hardcoded ndim=1.
#[pyfunction]
#[pyo3(signature = (keys, axis=-1))]
fn lexsort(py: Python<'_>, keys: &Bound<'_, PyAny>, axis: isize) -> PyResult<Py<PyAny>> {
    let mut arrs: Vec<NdArray> = Vec::new();
    for item in keys.try_iter()? {
        arrs.push(extract_array_like(&item?)?);
    }
    let result = core_lexsort(&arrs, axis).map_err(to_py_err)?;
    wrap(py, result, None)
}

/// `np.nonzero` -- a tuple of one 1-D `I64` index array per dimension.
#[pyfunction]
fn nonzero(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = extract_array_like(a)?;
    let cols = core_nonzero(&arr).map_err(to_py_err)?;
    let mut items: Vec<Py<PyAny>> = Vec::with_capacity(cols.len());
    for c in cols {
        items.push(Py::new(py, PyArray { inner: c })?.into_any());
    }
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

/// `np.flatnonzero` -- `nonzero` of the raveled array, single 1-D result.
#[pyfunction]
fn flatnonzero(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = extract_array_like(a)?;
    let result = core_flatnonzero(&arr).map_err(to_py_err)?;
    wrap(py, result, None)
}

/// `np.argwhere` -- shape `(count, ndim)` coordinate array.
#[pyfunction]
fn argwhere(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = extract_array_like(a)?;
    let result = core_argwhere(&arr).map_err(to_py_err)?;
    wrap(py, result, None)
}

/// `np.extract(condition, arr)` -- `arr`'s elements selected where
/// `condition` is truthy, broadcasting the two together first.
#[pyfunction]
fn extract(py: Python<'_>, condition: &Bound<'_, PyAny>, arr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let cond = extract_array_like(condition)?;
    let a = extract_array_like(arr)?;
    let result = core_extract(&cond, &a).map_err(to_py_err)?;
    wrap(py, result, None)
}

/// `np.where` -- the 1-arg form (`x`/`y` both omitted) is an alias for
/// `nonzero`; the 3-arg form is an elementwise ternary select. Passing
/// exactly one of `x`/`y` is a `ValueError` in real numpy (verified:
/// "either both or neither of x and y should be given").
#[pyfunction]
#[pyo3(name = "where")]
#[pyo3(signature = (condition, x=None, y=None))]
fn r#where(
    py: Python<'_>,
    condition: &Bound<'_, PyAny>,
    x: Option<&Bound<'_, PyAny>>,
    y: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let cond = extract_array_like(condition)?;
    match (x, y) {
        (None, None) => {
            let cols = core_nonzero(&cond).map_err(to_py_err)?;
            let mut items: Vec<Py<PyAny>> = Vec::with_capacity(cols.len());
            for c in cols {
                items.push(Py::new(py, PyArray { inner: c })?.into_any());
            }
            Ok(PyTuple::new(py, items)?.into_any().unbind())
        }
        (Some(xx), Some(yy)) => {
            // BUG FOUND + FIXED (2026-08-04): this was two INDEPENDENT
            // `extract_array_like` calls, which give a bare Python scalar a
            // fixed STRONG dtype (int64/float64/complex128) with nothing to
            // promote it against. numpy's `where` is an ordinary NEP 50
            // three-operand ufunc: `x`/`y` promote against EACH OTHER, and a
            // bare Python `bool`/`int`/`float`/`complex` is WEAK there.
            // Measured 2026-08-04, numpy 2.5.1, `c = np.array([True, False])`:
            //     np.where(c, 1,    int8_arr)      -> int8      (anionpy: int64)
            //     np.where(c, 1.0,  float32_arr)   -> float32   (anionpy: float64)
            //     np.where(c, 1e-20, complex64_arr)-> complex64 (anionpy: complex128)
            //     np.where(c, 1,    complex64_arr) -> complex64 (anionpy: complex128)
            // 14 of 45 cells of a {9 dtypes} x {5 scalar kinds} grid diverged
            // on DTYPE. Values agreed, which is exactly why the corpus (which
            // only ever passed array operands to `where`) never caught it --
            // `where` was declared "exact" while carrying this.
            //
            // Overflow behaviour was measured too, and it is the STRICT
            // (`relaxed = false`) variant, not the compare-op relaxed one:
            //     np.where(c, -7,  uint8_arr) -> OverflowError: Python integer
            //                                    -7 out of bounds for uint8
            //     np.where(c, 300, uint8_arr) -> OverflowError (same shape)
            //     np.where(c, 1e20, float16_arr) -> float16 [inf, 2.] (no raise)
            // and it is symmetric in the two operand positions. `forces_float`
            // is false: `np.where(c, 1.5, uint8_arr)` is float64 via ordinary
            // `weak_target_dtype(U8, Float)`, not via any divide-style forced
            // promotion.
            //
            // Delegating to the SAME `extract_binary_pair` every other binary
            // op already uses rather than re-deriving the rule here.
            let (xa, ya) = crate::extract_binary_pair(xx, yy, false, false)?;
            let result = where_select(&cond, &xa, &ya).map_err(to_py_err)?;
            wrap(py, result, None)
        }
        _ => Err(PyValueError::new_err("either both or neither of x and y should be given")),
    }
}

/// Shared `(a, v)` marshaling for BOTH `anionpy.searchsorted` and
/// `ndarray.searchsorted`. Returns the two operands at a single common
/// dtype, which is what `ionp_core::sort::searchsorted` requires and what
/// numpy's own `result_type(a, v)` produces.
///
/// See `searchsorted`'s doc comment below for the measured contract this
/// implements and for the phantom it closes. The one thing worth repeating
/// here, because it is the part a reader is most likely to "simplify" back
/// into a bug: the weak-scalar extraction is the RELAXED variant. `where`
/// and `clip` raise `OverflowError` for a bare Python int that does not fit
/// the other operand's dtype; `searchsorted` does not -- it compares by
/// VALUE at a width that holds it, so `np.searchsorted(np.array([1,2,3],
/// np.uint8), -7)` is `0`, not the `3` that narrowing `-7` to `249` gives.
pub(crate) fn searchsorted_operands(
    base: &NdArray,
    v: &Bound<'_, PyAny>,
) -> PyResult<(NdArray, NdArray)> {
    // Strong-operand gate FIRST, for the reason spelled out at length in
    // `ndarray_attrs.rs::is_strong_operand`: `classify_scalar` matches by
    // `is_instance_of`, and `np.float64`/`np.int64`/`np.bool_` are genuine
    // subclasses of Python `float`/`int`/`bool`, so a numpy scalar reaching
    // `scalar_against` ungated would be treated as WEAK and narrowed to the
    // array's dtype instead of promoting it. Same shape as
    // `lib.rs::extract_binary_pair_tiered`'s own gate.
    let v_is_strong = v.extract::<PyRef<PyArray>>().is_ok() || v.hasattr("dtype")?;
    let values = if v_is_strong {
        extract_array_like(v)?
    } else {
        match crate::scalar_against(base, v, true, false, false)? {
            Some(w) => w,
            None => extract_array_like(v)?,
        }
    };
    if base.dtype() == values.dtype() {
        return Ok((base.clone(), values));
    }
    let common = promote_dtype(base.dtype(), values.dtype());
    Ok((base.cast_to(common), values.cast_to(common)))
}

/// `np.searchsorted(a, v, side='left', sorter=None)`. `side` must be
/// exactly `'left'` or `'right'` (real numpy's exact error text: `search
/// side must be 'left' or 'right' (got '...')`).
///
/// `side` accepts `str`/`bytes` (ASCII-decoded, not `bytearray`, same
/// shape as `order=`/`casting=`/`kind=` elsewhere in this project) --
/// EXCEPT that unlike `kind=`, an explicitly-passed `side=None` is NOT
/// treated the same as an omitted `side=`: verified live against real
/// numpy 2.5.1, `np.searchsorted(a, v, side=None)` raises `TypeError:
/// search side must be str, not NoneType` while an omitted `side=` uses
/// the `'left'` default. `crate::OptionalArg` (see its own doc comment in
/// `lib.rs`) is what makes the two distinguishable here, the same
/// mechanism used for `casting=`.
///
/// `result`'s scalar-return-type case (a 0-d/rank-0 result, produced
/// whenever `v` is a Python scalar or a genuinely 0-d array rather than an
/// array-valued `v`) is routed through the shared `crate::
/// numpy_scalar_from_0d` helper -- verified live: `np.searchsorted(a, 2.0)`
/// returns a bare `numpy.int64` scalar, not a 0-d `numpy.ndarray` -- rather
/// than the generic `wrap` used for every genuinely array-shaped result.
/// This mirrors ~15 other call sites across this crate that already do the
/// same thing for their own scalar-producing results (see that helper's own
/// doc comment) -- `searchsorted` was the one CONFIRMED MISSED site (this
/// task's brief: commit `5c1b959` wired the helper into other call sites
/// but missed this one), not a second, newly-invented scalar constructor.
#[pyfunction]
#[pyo3(signature = (a, v, side=crate::OptionalArg::Omitted, sorter=None))]
fn searchsorted(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    v: &Bound<'_, PyAny>,
    side: crate::OptionalArg<'_>,
    sorter: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // PHANTOM FOUND + FIXED (2026-08-04): this was two INDEPENDENT
    // `extract_array_like` calls, so `a` and `v` arrived at
    // `core_searchsorted` with unrelated dtypes and the kernel's final match
    // arm (`ionp-core/src/sort.rs:1462`) rejected the pair outright:
    //     TypeError: searchsorted: value dtype mismatch
    // A 192-cell grid (12 operand dtypes x 8 `v` spellings x {left, right})
    // measured 154/192 diverging, EVERY one of them that same raise against a
    // numpy that simply answers. Minimal repros, numpy 2.5.1:
    //     np.searchsorted(np.array([False, True]), 1)          -> 1
    //     np.searchsorted(np.array([1, 3, 5], np.int8), 2.5)    -> 1
    // numpy promotes `a` and `v` to their common `result_type` and searches
    // there; there is no "must match" rule at all.
    //
    // The weak-scalar variant here is the RELAXED one, and that is a measured
    // distinction, not a guess. `where` and `clip` both raise `OverflowError`
    // for an out-of-range weak int; `searchsorted` does not -- it compares by
    // VALUE at a width that can hold it:
    //     np.searchsorted(np.array([1,2,3], np.int8),   300) -> 3   (not 44)
    //     np.searchsorted(np.array([1,2,3], np.int8),   256) -> 3   (not 0)
    //     np.searchsorted(np.array([1,2,3], np.int8),  -200) -> 0
    //     np.searchsorted(np.array([1,2,3], np.uint8),  -7)  -> 0   (not 249)
    //     np.searchsorted(np.array([1,2,3], np.int64), 2**63)-> 3
    // Narrowing (the strict path) would give the parenthesised wrong answers.
    //
    // `extract_binary_pair` marshals the weak side but does NOT equalise two
    // strong dtypes (nor the widened-scalar-vs-narrow-array case it can
    // legitimately produce), so the explicit `promote_dtype` + `cast_to`
    // below is still required -- it is what stands in for numpy's
    // `result_type(a, v)`. Confirmed against numpy for the mixed-strong case
    // too: `np.searchsorted(np.array([1,2,3], np.uint64), np.int64(-5))` is
    // `0`, i.e. both sides go to float64, exactly `promote_dtype(U64, I64)`.
    // ...which is why the rule lives in ONE place, `searchsorted_operands`
    // below: `ndarray.searchsorted` is a second, independent call site (it
    // re-implements rather than delegates) and fixing only this one left
    // 578/984 method-form cases failing.
    let (sorted, values) = searchsorted_operands(&extract_array_like(a)?, v)?;
    let side_str: String = match &side {
        crate::OptionalArg::Omitted => "left".to_string(),
        crate::OptionalArg::Given(v) => {
            if v.is_none() {
                return Err(PyTypeError::new_err("search side must be str, not NoneType"));
            }
            crate::extract_str_or_ascii_bytes(v, "search side")?
        }
    };
    let right = match side_str.as_str() {
        "left" => false,
        "right" => true,
        other => return Err(crate::search_side_value_error(other)),
    };
    let sorter_arr: Option<NdArray> = match sorter {
        None => None,
        Some(s) if s.is_none() => None,
        Some(s) => Some(extract_array_like(s)?),
    };
    let result = core_searchsorted(&sorted, &values, right, sorter_arr.as_ref()).map_err(to_py_err)?;
    if result.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &result);
    }
    wrap(py, result, None)
}

/// `np.nanargmax`/`np.nanargmin`. `axis=None` flattens first (matches
/// `argmin`/`argmax`'s own convention); `keepdims=True` re-inserts a
/// size-1 dim at the reduced axis position (`nanargext` itself always
/// drops it, same as `argext`'s own non-keepdims shape -- see
/// `do_argext`'s own comment above for why this reshape has to happen at
/// the wrapper, not the kernel, layer).
fn do_nanargext(
    py: Python<'_>,
    op: NanExtremeOp,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    let arr = extract_array_like(a)?;
    let ndim = arr.ndim();
    let flattened = matches!(axis, None) || matches!(axis, Some(ax) if ax.is_none());
    let norm_axis: Option<usize> = if flattened {
        None
    } else {
        // `nanargmin`/`nanargmax` do not have a converter of their own --
        // they are the COMPOSITION of two. numpy's
        // `_nanfunctions_impl.nanarg{min,max}` runs
        //     a, mask = _replace_nan(a, ...)
        //     if mask is not None: mask = np.all(mask, axis=axis)   # REDUCE
        //     res = np.arg{min,max}(a, axis=axis, ...)              # SINGLE
        // so an inexact operand is validated by the REDUCE converter first
        // and only then by the single-axis one, while `_replace_nan`
        // returns `mask = None` for an exact dtype and the pre-check is
        // skipped entirely. That is the whole explanation for a table that
        // otherwise looks arbitrary. Measured 2026-08-04, numpy 2.5.1,
        // `np.zeros((2, 3), dtype=...)`:
        //                 float64                          int64
        //   True          "an integer is required"         "... required for the axis"
        //   (0, 0)        ValueError: duplicate value      "'tuple' object ..."
        //   (0, 5)        AxisError: axis 5 out of bounds  "'tuple' object ..."
        //   (0.0,)        "'float' object ..."             "'tuple' object ..."
        //   (0, 1)        "'tuple' object ..."             "'tuple' object ..."
        // The `(0, 1)` row is the one that proves the ORDER: the reduce
        // converter accepts it, and the 'tuple' rejection that escapes
        // comes from `argmin` afterwards.
        //
        // This accounted for all 80 residual diverging cases in the
        // 3,900-case axis-form sweep after the converter work in cc5523f.
        // Gate is `inexact AND size != 0`, both measured, neither guessed.
        // numpy's `nanarg{min,max}` raises its own "attempt to get
        // arg{min,max} of an empty sequence" for a size-0 operand BEFORE
        // the `np.all(mask, axis=axis)` line, so an empty float array
        // behaves exactly like an exact dtype here. A 0-d array has size 1
        // and DOES take the pre-check. Probe (axis=True; the "for the
        // axis" suffix is present iff the pre-check was skipped), 2026-08-04:
        //     float64/float16/complex128 (0,) (0,3) (3,0) (0,0) -> skipped
        //     float64/float16/complex128 (1,) (2,3) ()          -> taken
        //     int64/bool                 every shape           -> skipped
        if arr.dtype().is_inexact() && arr.size() != 0 {
            // `normalize_reduce_axes` too, not just the converter: the
            // pre-check numpy runs is a whole `np.all(mask, axis=axis)`
            // call, which RANGE-checks and DUPLICATE-checks as well as
            // type-checks. Converting only was measurably not enough --
            // anionpy reported argmin's 'tuple' message where numpy reports
            // the reduce family's. Measured 2026-08-04, float64 (5,):
            //     axis=(0, 0) -> ValueError: duplicate value in 'axis'
            //     axis=(0, 5) -> AxisError: axis 5 is out of bounds ...
            //     axis=(0, 1) -> AxisError: axis 1 is out of bounds ...
            let pre = reduce_axes_from_pyobj(axis.unwrap(), ndim, true, AxisSeq::TupleOnly)?;
            normalize_reduce_axes(Some(&pre), ndim).map_err(to_py_err)?;
        }
        let raw = single_axis_from_pyobj(axis.unwrap())?;
        // BUG FOUND + FIXED (2026-08-01, coordinator fuzzing): this used to
        // call `normalize_single_axis`, which is deliberately STRICT about
        // `ndim == 0` (correct for `sort`'s own axis family -- verified
        // `np.sort(np.array(3.0), axis=0)` really does raise `AxisError`).
        // But `nanargmin`/`nanargmax` are in the `argmin`/`argmax` REDUCE
        // family, not the sort family, and that family grants a 0-d array
        // a courtesy no-op for `axis=0`/`axis=-1` (verified against real
        // numpy 2.5.1: `np.nanargmin(np.array(3.0), axis=0) == 0`, same as
        // plain `np.argmin`). Mirror `do_argext`'s own inline
        // normalization above -- which had the identical negative-axis
        // 0-d bug (`raw + n` instead of `raw + n.max(1)`, fixed alongside
        // this one) -- rather than the sort-family helper.
        let n = ndim as isize;
        let norm = if raw < 0 { raw + n.max(1) } else { raw };
        // BUG FOUND + FIXED (2026-08-01, this task): UNLIKE plain
        // `argmin`/`argmax` (which always report the `ndim.max(1)`
        // courtesy dimension for an out-of-range axis on a 0-d array, see
        // `do_argext`'s own comment), `nanargmin`/`nanargmax`'s
        // out-of-range `AxisError` dimension depends on the input's
        // dtype -- probed directly against real numpy 2.5.1
        // (float16/32/64, complex64/128, int8/uint8, bool, all 0-d,
        // `axis=100`): float/complex dtypes report the REAL ndim (0 for a
        // 0-d array), while int/bool dtypes report the SAME `ndim.max(1)`
        // courtesy as plain argmax. This isn't arbitrary: real numpy's
        // `nanargmax` (`numpy/lib/_nanfunctions_impl.py`) replaces NaNs and
        // builds a `mask = np.isnan(a)` for float/complex dtypes only, then
        // calls `np.all(mask, axis=axis)` to check for all-NaN slices
        // BEFORE ever calling `np.argmax` -- so for those dtypes the
        // `AxisError` for `axis=100` actually comes from the plain-reduce
        // family's `np.all` (real ndim, no courtesy, verified separately:
        // `np.all(np.array(True), axis=100)` also reports "dimension 0"),
        // not from argmax's own axis handling. int/bool dtypes can never
        // contain NaN, so numpy skips the mask/`np.all` step entirely and
        // calls `np.argmax` directly, inheriting ITS `ndim.max(1)`
        // courtesy instead.
        let report_ndim = if matches!(
            arr.dtype(),
            DType::F16 | DType::F32 | DType::F64 | DType::C64 | DType::C128
        ) {
            ndim
        } else {
            ndim.max(1)
        };
        if norm < 0 || norm >= n.max(1) {
            return Err(to_py_err(IonpError::AxisError { axis: raw, ndim: Some(report_ndim) }));
        }
        Some(norm as usize)
    };
    let mut result = nanargext(op, &arr, norm_axis).map_err(to_py_err)?;
    if keepdims {
        if flattened {
            // Same bug class as `do_argext`/`do_method_argext` (found+fixed
            // earlier this task): `ndim > 1` incorrectly also skipped the 0-d
            // case, where real numpy still returns a 0-d/scalar result, not
            // shape (1,).
            //
            // UNLIKE `argext`, whose raw flattened+keepdims result is always
            // shape `[1]` regardless of the original ndim (so `ndim == 1`
            // could safely skip a now-provably-no-op reshape),
            // `nanargext`'s raw flattened result is a genuine 0-d scalar
            // even when the original array was 1-D (verified empirically:
            // `nanargmin` on a 1-D array with `keepdims=True` regressed to
            // shape `()` instead of `(1,)` when this used the same
            // `ndim != 1` guard as `do_argext`). So this reshape must be
            // unconditional whenever `flattened && keepdims`, for every
            // ndim including 0 and 1 -- `vec![1usize; ndim]` already
            // degrades correctly to `[]` (0-d) at ndim==0.
            result = result.reshape(&vec![1usize; ndim]).map_err(to_py_err)?;
        } else if ndim != 0 {
            // BUG FOUND + FIXED (2026-08-01, own extended sweep after the
            // coordinator's report): when `ndim == 0`, `norm_axis` can only
            // be the 0-d courtesy phantom axis (`Some(0)`), which is not a
            // real dimension to keep -- real numpy's `keepdims=True` is a
            // no-op for a 0-d array's courtesy axis (verified:
            // `np.nanargmin(np.array(3.0), axis=0, keepdims=True).shape ==
            // ()`, not `(1,)`). Same fix as `argext`'s own sibling bug in
            // ufunc.rs.
            let ax = norm_axis.unwrap();
            let mut shape = result.shape().to_vec();
            shape.insert(ax, 1);
            result = result.reshape(&shape).map_err(to_py_err)?;
        }
    }
    wrap_reduction(py, result, out)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, ddof=0.0, keepdims=None, r#where=None, mean=None, correction=None))]
#[allow(clippy::too_many_arguments)]
fn var(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    ddof: f64,
    keepdims: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
    mean: Option<&Bound<'_, PyAny>>,
    correction: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    do_var(py, &arr, axis, dtype, out, ddof, keepdims, r#where, mean, correction, false, false)
}

// Named `std_fn` at the Rust level (plain `std` collides with the `std`
// crate root inside `wrap_pyfunction!`'s own macro expansion -- E0659
// ambiguity); `#[pyo3(name = "std")]` still exposes it as `anionpy.std` to
// Python, which is all that matters for the public surface.
#[pyfunction]
#[pyo3(name = "std")]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, ddof=0.0, keepdims=None, r#where=None, mean=None, correction=None))]
#[allow(clippy::too_many_arguments)]
fn std_fn(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    ddof: f64,
    keepdims: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
    mean: Option<&Bound<'_, PyAny>>,
    correction: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    do_var(py, &arr, axis, dtype, out, ddof, keepdims, r#where, mean, correction, true, false)
}

// `mean=`/`correction=` (2026-08-02, coordinator signature-diff task):
// real `nanvar`/`nanstd` accept BOTH keywords -- verified against real
// numpy 2.5.1's actual signature, `(a, axis=None, dtype=None, out=None,
// ddof=0, keepdims=<no value>, *, where=<no value>, mean=<no value>,
// correction=<no value>)`, identical in shape to plain `var`/`std`'s own
// signature just above. `do_var` already threads `mean_obj`/`correction`
// through for BOTH the `nan_aware` and non-`nan_aware` paths (the same
// function body plain `var`/`std` use above) -- these two `#[pyfunction]`
// wrappers used to just hardcode `None, None` here instead of exposing
// the params at all, which is why real numpy accepted `correction=` and
// anionpy raised `TypeError: nanvar() got an unexpected keyword argument
// 'correction'` for EVERY non-absent value (verified: 399/420 mismatches
// in the coordinator's `/tmp/meancorr.py` grid, all but the `ABSENT`
// row). No separate/duplicated ddof-vs-correction or mean logic is
// written here -- both wrappers call the exact same shared `do_var`
// plain `var`/`std` already use, so there is only ONE place this rule is
// implemented, not two copies that could drift apart again.
#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, ddof=0.0, keepdims=None, r#where=None, mean=None, correction=None))]
#[allow(clippy::too_many_arguments)]
fn nanvar(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    ddof: f64,
    keepdims: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
    mean: Option<&Bound<'_, PyAny>>,
    correction: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    do_var(py, &arr, axis, dtype, out, ddof, keepdims, r#where, mean, correction, false, true)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, dtype=None, out=None, ddof=0.0, keepdims=None, r#where=None, mean=None, correction=None))]
#[allow(clippy::too_many_arguments)]
fn nanstd(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    ddof: f64,
    keepdims: Option<&Bound<'_, PyAny>>,
    r#where: Option<&Bound<'_, PyAny>>,
    mean: Option<&Bound<'_, PyAny>>,
    correction: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let keepdims = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    do_var(py, &arr, axis, dtype, out, ddof, keepdims, r#where, mean, correction, true, true)
}

// ---------------------------------------------------------------------------
// Tier 7: `average` -- weighted mean. Supports `weights=None` (identical
// to `mean`), `weights` with the SAME shape as `a`, and 1-D `weights`
// broadcast along a single integer `axis` (numpy's own three supported
// forms -- verified against real numpy 2.5.1's `average` source, which
// itself only special-cases exactly these three; anything else numpy
// rejects too). `axis=None` with 1-D weights requires `a` to ALSO be 1-D
// (matching numpy: "Axis must be specified when shapes of a and weights
// differ").
// ---------------------------------------------------------------------------

/// `average` does NOT validate its axis through the C reduce path that
/// `sum`/`mean`/etc use. Real numpy routes it through
/// `normalize_axis_tuple(axis, a.ndim, argname="axis")`, and that argname
/// changes BOTH error renderings (measured 2026-08-03, Monday, at ranks
/// 0/1/2, weighted and unweighted -- 162 divergent cells):
///
///   * bounds  -- `"axis: axis 1 is out of bounds for array of dimension 1"`
///                (the `axis: ` prefix is `AxisError`'s third ctor arg),
///                where the reduce path says `"axis 1 is out of ..."`.
///   * repeat  -- ``"repeated axis in `axis` argument"``, where the reduce
///                path says `"duplicate value in 'axis'"`.
///
/// This is average-LOCAL and must not be pushed down into the shared
/// helpers: `np.sum(a, axis=(0, 0))` renders `"duplicate value in 'axis'"`
/// and anionpy already matches it there, so widening this would BREAK `sum`
/// to fix `average`.
///
/// Check ORDER is numpy's, not ours: `normalize_axis_tuple` normalizes
/// every entry first (so an out-of-bounds axis wins) and only then tests
/// for repeats. Hence `axis=(0, 0)` is an `AxisError` on a 0-d operand but
/// a `ValueError` at rank 1 -- both measured, both reproduced here.
///
/// Runs BEFORE the shared helpers purely as a message-shaping pre-pass;
/// anything it lets through is validated again downstream, so it can only
/// ever change which message you get, never whether a call succeeds.
/// `average`'s axis conversion, in numpy's own two steps.
///
/// `_average` runs `axis = normalize_axis_tuple(axis, a.ndim,
/// argname="axis")` -- `operator.index` FIRST, and only when that raises
/// does the object get iterated. So the integer-like forms (int, bool,
/// `np.int64`, a 0-d INTEGER array) keep the existing scalar validation,
/// and everything else -- floats, `Decimal`, `np.float32`, `np.bool`, a
/// 0-d FLOAT array, an ndim>=1 array, str, bytes, tuples, lists, ranges --
/// goes to the shared sequence converter in `stats.rs`, which is the same
/// one `median`/`quantile` use.
///
/// BUG FOUND + FIXED (2026-08-04, axis-form sweep): the non-integer forms
/// used to go through `reduce_axes_from_pyobj`, whose messages belong to
/// `PyArray_ConvertMultiAxis`. That is a different converter, and every
/// one of those forms said the wrong thing -- 118 of the 432 cases in a
/// dedicated `average` sweep. Two examples of the ten distinct pairs:
///     np.average(a, axis=0.0)         np: 'float' object is not iterable
///                                   anionpy: 'float' object cannot be
///                                         interpreted as an integer
///     np.average(a, axis=np.array([0]))  np: OK, shape (3,)
///                                      anionpy: only integer scalar arrays
///                                            can be converted ...
/// The second is the one that matters most: numpy ACCEPTS it.
fn average_axes(ax: &Bound<'_, PyAny>, ndim: usize) -> PyResult<Option<Vec<isize>>> {
    if let Ok(scalar) = ax.extract::<isize>() {
        // Range-checked HERE rather than in `average_validate_axes`, which
        // this replaced. That helper opened with
        // `axis.extract::<Vec<isize>>()`, and PyO3 answers that with an
        // EMPTY VEC for a 0-d array -- so the loop that follows had nothing
        // to check and the axis went unvalidated, picking up the
        // unprefixed `AxisError` from the shared path downstream.
        // Measured 2026-08-04, `np.average(np.zeros(()), axis=np.array(0))`:
        //     np  : axis: axis 0 is out of bounds for array of dimension 0
        //     anionpy: axis 0 is out of bounds for array of dimension 0
        // Found by the `axistype/*` suite guards added in this same
        // change, NOT by the sweep -- the sweep passes `np.array(0)` on a
        // 0-d operand too, but only the guard compares the full message.
        // A single scalar cannot repeat, so no duplicate check is needed.
        let n = ndim as isize;
        let norm = if scalar < 0 { scalar + n } else { scalar };
        if norm < 0 || norm >= n {
            return Err(crate::axis_error_prefixed(scalar, ndim, "axis"));
        }
        return Ok(Some(vec![norm]));
    }
    let norm = crate::stats::normalize_axis_tuple_seq(ax, ndim, Some("axis"))?;
    Ok(Some(norm.into_iter().map(|u| u as isize).collect()))
}

// `average_validate_axes` REMOVED 2026-08-04. It opened with
// `axis.extract::<Vec<isize>>()`, which PyO3 satisfies with an empty vec
// for a 0-d array and with `[1]` for `(True,)`, so it silently validated
// nothing in the first case. Its two jobs now live in exactly one place
// each: the scalar range check in `average_axes` above, and the sequence
// range/duplicate checks in `stats::normalize_axis_tuple_seq`, which
// `median`/`quantile` already share.

#[pyfunction]
#[pyo3(signature = (a, axis=None, weights=None, returned=None, keepdims=None))]
fn average(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    weights: Option<&Bound<'_, PyAny>>,
    returned: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Measured separately (2026-08-02 sweep): `returned=` is plain Python
    // truthiness in real numpy (accepts `1.5`/a raising `__bool__` object
    // exactly like `svd`'s flags), while `keepdims=` here follows the
    // SAME C `"i"`-format rule as `sum`/`mean`/etc above (rejects
    // `keepdims=1.5` with "integer argument expected, got float") --
    // verified live, not assumed from either sibling.
    let returned = opt_truthy(returned, false)?;
    let keepdims_resolved = opt_index_like(keepdims, false)?;
    let arr = extract_array_like(a)?;
    let ndim = arr.ndim();

    let weights_arr: Option<NdArray> = match weights {
        None => None,
        Some(w) if w.is_none() => None,
        Some(w) => Some(extract_array_like(w)?),
    };

    let Some(w) = weights_arr else {
        let axes_raw: Option<Vec<isize>> = match axis {
            None => None,
            Some(ax) if ax.is_none() => None,
            // 0-d courtesy: average (unweighted) -- numpy RAISES
            Some(ax) => {
                average_axes(ax, ndim)?
            }
        };
        let axes = normalize_reduce_axes(axes_raw.as_deref(), ndim).map_err(to_py_err)?;

        // NOTE: real numpy's `average(a, weights=None, ...)` unconditionally
        // computes `scl = avg_as_array.dtype.type(a.size / avg_as_array.size)`
        // BEFORE checking `returned` -- so a `ZeroDivisionError` fires even
        // when `returned=False`, whenever the reduced output has zero
        // elements (e.g. `np.average(np.zeros((0,5)), axis=1)`). This is
        // NOT the documented "weights sum to zero" case (no weights were
        // passed at all) -- verified live: `np.average(np.zeros((0,5)),
        // axis=1)` raises `ZeroDivisionError: division by zero` even with
        // `returned=False`, while `np.average(np.zeros((0,5)))` (axis=None,
        // scalar output, size 1) returns `nan` cleanly with no error.
        // `mean()` itself never raises here (it returns an empty/NaN array),
        // so `average` must special-case this before delegating to `mean`.
        let out_size: usize = arr
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, &d)| d)
            .product();
        if out_size == 0 {
            return Err(PyZeroDivisionError::new_err("division by zero"));
        }

        // Delegate with the NORMALIZED axes as a TUPLE, never with the
        // caller's raw `axis` object. Real numpy's `_average` does exactly
        // this -- `axis = normalize_axis_tuple(axis, a.ndim, argname="axis")`
        // runs BEFORE `a.mean(axis, ...)` -- and the distinction is
        // observable, because the two functions use DIFFERENT converters
        // (see `AxisSeq`). Measured 2026-08-04, numpy 2.5.1:
        //     np.average(np.zeros((2, 3)), axis=[0])  -> OK, shape (3,)
        //     np.mean(np.zeros((2, 3)),    axis=[0])  -> TypeError: 'list'
        //                                                object cannot be ...
        // Forwarding the raw object made `average` silently inherit
        // `mean`'s strictness and reject every non-tuple sequence numpy
        // accepts here. Caught by sweep, not by review.
        let axis_arg = match axes_raw.as_deref() {
            None => None,
            Some(ax) => Some(
                pyo3::types::PyTuple::new(py, ax.iter().copied())?
                    .into_any(),
            ),
        };
        let avg = mean(py, a, axis_arg.as_ref(), None, None, keepdims, None)?;
        if !returned {
            return Ok(avg);
        }
        let n = reduce_count(&arr, &axes, keepdims_resolved, None)?.cast_to(DType::F64);
        let n_obj = wrap(py, n, None)?;
        return Ok((avg, n_obj).into_pyobject(py)?.into_any().unbind());
    };

    let axes_raw: Option<Vec<isize>> = match axis {
        None => None,
        Some(ax) if ax.is_none() => None,
        // 0-d courtesy: average (weighted) -- numpy RAISES
        Some(ax) => {
            average_axes(ax, ndim)?
        }
    };
    let axes = normalize_reduce_axes(axes_raw.as_deref(), ndim).map_err(to_py_err)?;

    // Statistics-grid DEFECT 3 fix (2026-08-02): real numpy's
    // `_weights_are_valid` (`_function_base_impl.py`) selects between its
    // TWO distinct error messages purely on whether `axis` itself was
    // `None`, NOT on whether `weights` happens to be 1-D:
    //
    //   if a.shape != wgt.shape:
    //       if axis is None:
    //           raise TypeError("Axis must be specified when shapes of a
    //               and weights differ.")
    //       if wgt.shape != tuple(a.shape[ax] for ax in axis):
    //           raise ValueError("Shape of weights must be consistent
    //               with shape of a along specified axis.")
    //       wgt = wgt.transpose(np.argsort(axis))
    //       wgt = wgt.reshape(tuple((s if ax in axis else 1)
    //                                for ax, s in enumerate(a.shape)))
    //
    // There is no dimensionality restriction on `weights` at all once
    // `axis` is given -- an N-D `weights` whose shape equals the tuple of
    // `a`'s sizes along the (possibly multi-axis, possibly out-of-order,
    // e.g. `axis=(1, 0)`) `axis` argument is perfectly valid, verified
    // live against numpy 2.5.1: `np.average(np.ones((2,3)), axis=(0,1),
    // weights=np.ones((2,3)))` succeeds. anionpy's old code instead branched
    // on `w.ndim() == 1` -- wrong on two counts: (1) it raised the
    // TypeError whenever `axis` was ALSO absent, conflating "shapes
    // differ" detection with "is weights 1-D", so a 1-D `axis` + N-D
    // `weights` case fell through to an ionp-only "1D weights expected"
    // message numpy never produces at all; (2) it flatly rejected a tuple
    // `axis` via `single_axis_from_pyobj`, which real numpy fully
    // supports here. Replaced with the same shape-tuple-equality-then-
    // transpose-then-reshape sequence numpy's own source performs, using
    // `axes` (already axis-order-preserving, negative-normalized) as the
    // `axis` tuple stand-in.
    let w_broadcast: NdArray = if w.shape() == arr.shape() {
        w.clone()
    } else if axes_raw.is_none() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Axis must be specified when shapes of a and weights differ.",
        ));
    } else {
        let expected_shape: Vec<usize> = axes.iter().map(|&ax| arr.shape()[ax]).collect();
        if w.shape() != expected_shape.as_slice() {
            // Message matches numpy's `_weights_are_valid` verbatim (it
            // uses this same text for the 1-D-weights-length-mismatch case,
            // not a length-specific message) -- verified live against
            // numpy 2.5.1: `np.average(np.ones(1), axis=0, weights=[1,2])`
            // raises `ValueError: Shape of weights must be consistent with
            // shape of a along specified axis.`
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Shape of weights must be consistent with shape of a along specified axis.",
            ));
        }
        // `np.argsort(axis)`: the permutation that sorts `axes` ascending
        // -- e.g. `axes = [1, 0]` (from `axis=(1, 0)`) sorts to `[1, 0]`
        // (position 1 holds the smaller value 0). `sort_by_key` is stable,
        // matching `np.argsort`'s default stable behavior for the (rare,
        // but real-numpy-permitted) case of a repeated axis value.
        let mut perm: Vec<usize> = (0..axes.len()).collect();
        perm.sort_by_key(|&i| axes[i]);
        let perm_isize: Vec<isize> = perm.iter().map(|&i| i as isize).collect();
        let transposed = w.transpose_axes(&perm_isize).map_err(to_py_err)?;
        // After the transpose, `transposed`'s own dims are in ASCENDING
        // real-axis order (that's what the argsort permutation achieves),
        // so zipping them against `axes` sorted ascending assigns each
        // transposed dim to its correct real-array axis position; every
        // other position of the full-`ndim` shape gets numpy's `1`
        // (broadcast placeholder).
        let mut sorted_axes = axes.clone();
        sorted_axes.sort_unstable();
        let mut wshape = vec![1usize; ndim];
        for (i, &ax) in sorted_axes.iter().enumerate() {
            wshape[ax] = transposed.shape()[i];
        }
        transposed.reshape(&wshape).map_err(to_py_err)?
    };

    let num = binary_op(BinaryOp::Multiply, &arr, &w_broadcast).map_err(to_py_err)?;
    let num_sum = reduce_axis(BinaryOp::Add, &num, &axes, keepdims_resolved, None, None, None).map_err(to_py_err)?;
    let w_full = if w_broadcast.shape() == arr.shape() {
        w_broadcast
    } else {
        binary_op(BinaryOp::Multiply, &w_broadcast, &creation_ones_like(&arr)?).map_err(to_py_err)?
    };
    let w_sum = reduce_axis(BinaryOp::Add, &w_full, &axes, keepdims_resolved, None, None, None).map_err(to_py_err)?;

    // Real numpy: `if np.any(scl == 0.0): raise ZeroDivisionError("Weights
    // sum to zero, can't be normalized")` -- fires whenever ANY position of
    // the (possibly per-axis) weight-sum result is exactly zero, not just
    // for a whole-array reduction. Verified live against numpy 2.5.1.
    let w_sum_is_zero = binary_op(BinaryOp::Equal, &w_sum, &scalar_f64(0.0)?.cast_to(w_sum.dtype())).map_err(to_py_err)?;
    if any_true(&w_sum_is_zero) {
        return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
            "Weights sum to zero, can't be normalized",
        ));
    }

    let avg = binary_op(BinaryOp::Divide, &num_sum, &w_sum).map_err(to_py_err)?;
    // Same scalar-return-type fix as the no-weights path above (which
    // already goes through `mean`'s own `wrap_reduction`): a weighted
    // `average` that fully collapses to 0-d (e.g. both `a` and `weights`
    // 0-d, or `axis=None` over an N-D array) also returns a numpy SCALAR in
    // real numpy, not a 0-d array -- verified live:
    // `np.average(np.array(4.0), weights=np.array(2.0))` is `numpy.float64`.
    // `wrap` (unconditional `PyArray`) was wrong here; `wrap_reduction`
    // (same helper `do_reduce_axis` above uses) is the correct one.
    let avg_obj = wrap_reduction(py, avg, None)?;
    if !returned {
        return Ok(avg_obj);
    }
    let wsum_obj = wrap_reduction(py, w_sum, None)?;
    Ok((avg_obj, wsum_obj).into_pyobject(py)?.into_any().unbind())
}

fn creation_ones_like(a: &NdArray) -> PyResult<NdArray> {
    NdArray::from_buffer(creation::ones_buffer(a.dtype(), a.size().max(1)), a.shape().to_vec(), Order::C).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// Tier 8: `nanargmax`/`nanargmin` thin wrappers over `do_nanargext` (defined
// in the Tier 5b sort/search block above, alongside `NanExtremeOp`).
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=None))]
fn nanargmax(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Plain truthy, same as `argmin`/`argmax` above (measured separately,
    // not assumed) -- not the `keepdims_index_like` rule.
    do_nanargext(py, NanExtremeOp::Max, a, axis, out, opt_truthy(keepdims, false)?)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, keepdims=None))]
fn nanargmin(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    do_nanargext(py, NanExtremeOp::Min, a, axis, out, opt_truthy(keepdims, false)?)
}

// ---------------------------------------------------------------------------

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(prod, m)?)?;
    m.add_function(wrap_pyfunction!(all, m)?)?;
    m.add_function(wrap_pyfunction!(any, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(amin, m)?)?;
    m.add_function(wrap_pyfunction!(amax, m)?)?;
    m.add_function(wrap_pyfunction!(argmin, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(cumprod, m)?)?;
    m.add_function(wrap_pyfunction!(nancumsum, m)?)?;
    m.add_function(wrap_pyfunction!(nancumprod, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(ptp, m)?)?;
    m.add_function(wrap_pyfunction!(count_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(nansum, m)?)?;
    m.add_function(wrap_pyfunction!(nanprod, m)?)?;
    m.add_function(wrap_pyfunction!(nanmin, m)?)?;
    m.add_function(wrap_pyfunction!(nanmax, m)?)?;
    m.add_function(wrap_pyfunction!(nanmean, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(std_fn, m)?)?;
    m.add_function(wrap_pyfunction!(nanvar, m)?)?;
    m.add_function(wrap_pyfunction!(nanstd, m)?)?;
    m.add_function(wrap_pyfunction!(average, m)?)?;
    m.add_function(wrap_pyfunction!(sort, m)?)?;
    m.add_function(wrap_pyfunction!(argsort, m)?)?;
    m.add_function(wrap_pyfunction!(sort_complex, m)?)?;
    m.add_function(wrap_pyfunction!(lexsort, m)?)?;
    m.add_function(wrap_pyfunction!(nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(flatnonzero, m)?)?;
    m.add_function(wrap_pyfunction!(argwhere, m)?)?;
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    m.add_function(wrap_pyfunction!(r#where, m)?)?;
    m.add_function(wrap_pyfunction!(searchsorted, m)?)?;
    m.add_function(wrap_pyfunction!(nanargmax, m)?)?;
    m.add_function(wrap_pyfunction!(nanargmin, m)?)?;
    Ok(())
}
