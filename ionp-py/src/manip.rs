//! PyO3 bindings for the array-manipulation block: concatenate/stack and
//! the hstack/vstack/dstack/column_stack/row_stack wrappers, flip/fliplr/
//! flipud, roll, tile, repeat, broadcast_shapes/broadcast_arrays,
//! atleast_1d/2d/3d, and the diagonal family (diag/diagonal/diagflat/
//! tril/triu/trace).
//!
//! This file only parses Python arguments (arrays, axis ints, shape
//! tuples) and marshals them into `ionp_core::manip`'s pure-Rust
//! implementations -- every actual numeric loop lives there, per the
//! crate's "Rust does arithmetic, Python does dispatch" rule (mirrors
//! `creation.rs`'s own module doc). Every function accepts either a real
//! `anionpy.ndarray` OR a foreign `numpy.ndarray`/list/tuple, via the same
//! `extract_or_ingest_ndarray` pattern `creation.rs` established (this
//! module keeps its own private copy since `creation.rs`'s is not `pub`
//! and this task does not own that file -- see this task's file-ownership
//! rules).

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyList, PySequence, PySlice, PyTuple};

use ionp_core::{manip, Buffer, DType, NdArray};

use crate::{dtype_from_pyobj, to_py_err, wrap_shape_view, PyArray};

// ---------------------------------------------------------------------------
// Shared argument-parsing helpers, local to this module (mirrors
// creation.rs's own local helpers, not shared, per this codebase's
// established per-module-copy convention for small parsing utilities).
// ---------------------------------------------------------------------------

fn extract_or_ingest_ndarray(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(pyref.inner.clone());
    }
    crate::array_impl(obj, None)
}

/// Ingest a `take`-style *index* argument the way numpy does.
///
/// numpy does not infer a dtype for these and then cast: it calls
/// `PyArray_FromAny` with `intp` *requested*, so an untyped Python sequence is
/// built directly at the index dtype, coercing each element through Python's
/// own `int()`. An argument that is ALREADY an array keeps its own dtype and
/// is then subject to the ordinary `same_kind` cast check. The two paths give
/// visibly different answers, so which one you take is not cosmetic --
/// measured on numpy 2.5.1 against `np.arange(4, dtype=np.int8)`:
///
///     a.take([])                -> shape (0,)     (an untyped empty sequence
///                                                  is intp, never float64)
///     a.take(())                -> shape (0,)
///     a.take([[], []])          -> shape (2, 0)
///     a.take([0.5])             -> shape (1,)     (int(0.5) == 0, truncated)
///     a.take(np.array([]))      -> TypeError: Cannot cast array data from
///                                  dtype('float64') ... rule 'same_kind'
///
/// anionpy inferred float64 for every one of the untyped forms and so raised the
/// cast error on all of them -- 81 of 1806 cells in the 2026-08-04 take/
/// compress grid. Note the last line: the SAME empty input spelled as a real
/// float64 array must still raise, which is why this cannot be "just be
/// lenient about empties".
/// True when `obj` is a *scalar* of an array library (numpy's `np.float64`,
/// anionpy's own equivalent) rather than an array. Detected by walking the type's
/// `__mro__` for a `generic` base -- the same mechanism `dtypeinfo.rs` uses,
/// and deliberately a structural test rather than a hardcoded name list, so an
/// anionpy scalar and a numpy scalar are classified by the same rule.
fn is_array_scalar(obj: &Bound<'_, PyAny>) -> bool {
    let Ok(mro) = obj.get_type().getattr("__mro__") else {
        return false;
    };
    let Ok(iter) = mro.try_iter() else { return false };
    for base in iter {
        let Ok(base) = base else { return false };
        if let Ok(n) = base.getattr("__name__").and_then(|n| n.extract::<String>()) {
            if n == "generic" {
                return true;
            }
        }
    }
    false
}

fn ingest_index_ndarray(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    // A numpy SCALAR is not an array, and numpy treats the two differently
    // here -- the scalar is coerced through `int()` like any other Python
    // object, while a 0-d array takes the cast path and is rejected. Measured
    // on numpy 2.5.1 against `np.arange(4, dtype=np.int8)`:
    //     a.take(np.float64(0.5)) -> 0-d result, value 0   (scalar: coerced)
    //     a.take(np.array(0.5))   -> TypeError: Cannot cast scalar from
    //                                dtype('float64') ... rule 'same_kind'
    // Gating on `hasattr("dtype")` alone gets this backwards, because a numpy
    // scalar has a `.dtype` too.
    let is_array =
        obj.extract::<PyRef<'_, PyArray>>().is_ok() || (obj.hasattr("dtype")? && !is_array_scalar(obj));
    if is_array {
        return extract_or_ingest_ndarray(obj);
    }
    // `intp` on every platform anionpy targets; built as a real `anionpy.dtype`
    // object because `array_impl` takes the Python-level dtype argument, the
    // same object a caller would have passed by hand.
    let intp = pyo3::Py::new(
        obj.py(),
        crate::PyDType {
            inner: ionp_core::dtype::DType::I64,
            spelling: None,
        },
    )?
    .into_bound(obj.py())
    .into_any();
    crate::array_impl(obj, Some(&intp))
}

// ---------------------------------------------------------------------------
// split/hsplit/dsplit/rollaxis/vsplit's numpy-duck-typing gap (2026-08-02,
// CLASS B closure task): unlike every other function in this module, these
// five do NOT go through `np.asarray()` internally in real numpy -- their
// pure-Python implementations touch `ary.ndim`/`ary.shape` as bare
// attribute access (sometimes with a lenient `np.ndim()`-style fallback to
// `asarray(ary).ndim` first, sometimes not -- verified live against numpy
// 2.5.1's `lib/_shape_base_impl.py`/`core/numeric.py`, NOT hardcoded from
// that reading: the two helpers below reproduce the *mechanism* -- real
// attribute lookups that surface Python's own generic AttributeError --
// rather than embedding numpy's message text, so the exact wording is an
// emergent property of doing the same dispatch, not a copied string).
// A raw Python list/tuple/range has neither attribute, so real numpy
// genuinely rejects those forms here, unlike every other manip function in
// this file. anionpy must reproduce that rejection, not paper over it by
// unconditionally coercing first.

/// Mirrors `numpy.ndim(a)`: `a.ndim` if present, else falls back to
/// `asarray(a).ndim` (this is the LENIENT check `hsplit`/`dsplit`/`vsplit`
/// each perform before ever touching `.shape`/`.ndim` strictly).
fn ndim_lenient(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    match obj.getattr("ndim") {
        Ok(nd) => nd.extract::<usize>(),
        Err(_) => Ok(extract_or_ingest_ndarray(obj)?.ndim()),
    }
}

/// Strict `.ndim` attribute access, no fallback -- what `rollaxis` (from
/// its very first line) and `hsplit` (after its own lenient pre-check)
/// both do. Raises Python's own bare `AttributeError` on a raw
/// list/tuple/range, exactly like real numpy.
fn ndim_strict(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    obj.getattr("ndim")?.extract::<usize>()
}

/// Strict `.shape` attribute presence gate, no fallback -- what `split`'s
/// own `N = ary.shape[axis]` line does for a scalar `indices_or_sections`
/// (unlike `array_split`, which wraps that same access in a
/// `try/except AttributeError: len(ary)` and so tolerates a raw list).
fn require_shape_attr(obj: &Bound<'_, PyAny>) -> PyResult<()> {
    obj.getattr("shape")?;
    Ok(())
}

/// Accepts any Python sequence (list/tuple) of array-likes and ingests
/// every element via `extract_or_ingest_ndarray`.
fn extract_ndarray_seq(obj: &Bound<'_, PyAny>) -> PyResult<Vec<NdArray>> {
    let items: Vec<Bound<'_, PyAny>> = obj.try_iter()?.collect::<PyResult<_>>()?;
    items.iter().map(extract_or_ingest_ndarray).collect()
}

/// PHANTOM-ADJACENT BUG FOUND + FIXED (2026-08-04): the n-ary form of
/// `extract_ndarray_seq` for `concatenate`, which is the only member of
/// the stack/concat family that sees a bare Python scalar as a scalar.
///
/// `extract_ndarray_seq` ingests every element independently, so a bare
/// `300` alongside an `int8` array became its own strong `int64` 0-d array
/// and dragged the result to int64. Real numpy 2.5.1 applies NEP 50 here:
/// promote over the STRONG operands only, then wrapping-cast each weak
/// scalar into that dtype. Measured, `a8 = np.array([1,0,2], np.int8)`:
///
///   np.concatenate((a8, 300),           axis=None) -> int8    [1,0,2,44]
///   np.concatenate((a8, [300]),         axis=None) -> int64   [1,0,2,300]
///   np.concatenate((a8, np.int64(300)), axis=None) -> int64   [1,0,2,300]
///   np.concatenate((300, 400),          axis=None) -> int64   [300,400]
///   np.concatenate((a8, a16, 300),      axis=None) -> int16   [...,300]
///   np.concatenate((f16, 10**100),      axis=None) -> float16 [..., inf]
///   np.concatenate((a8,  10**100),      axis=None) -> OverflowError
///   np.concatenate((bool_arr, 300),     axis=None) -> int64   [1,300]
///
/// Three orderings in there are easy to get wrong and are load-bearing:
///   1. a plain Python LIST element is STRONG (it goes through `asarray`),
///      only a BARE scalar is weak -- hence the `[300]` row differing from
///      the `300` row;
///   2. all-weak falls back to the default dtype rather than to some
///      "smallest that fits" -- hence the `(300, 400)` row, which is why
///      the no-strong-operand case defers to `extract_ndarray_seq`;
///   3. overflow is INT-TARGET-SPECIFIC. Against a float/complex target
///      the huge int saturates to `inf` with no error at all, which is
///      already what `weak_scalar_wrapping` does via `weak_scalar_buffer`'s
///      f64 path -- do not "helpfully" hoist the OverflowError out of the
///      integer branch.
///
/// `hstack`/`vstack`/`stack`/`append` are deliberately NOT routed through
/// here: verified live, `hstack`'s `atleast_1d` promotes the bare scalar
/// to a strong 1-d array BEFORE concatenation (so `np.hstack((a8, 300))`
/// really is int64), and `vstack`/`stack` raise on shape before dtype ever
/// matters. Sending them here would be a regression, not a fix.
fn extract_concat_seq(obj: &Bound<'_, PyAny>) -> PyResult<Vec<NdArray>> {
    let items: Vec<Bound<'_, PyAny>> = obj.try_iter()?.collect::<PyResult<_>>()?;
    // The `hasattr("dtype")` gate MUST run before `classify_scalar`, which
    // matches by `is_instance_of` and therefore also matches `np.float64`/
    // `np.int64`/`np.bool_` (subclasses of `float`/`int`/`bool`).
    let mut weak = Vec::with_capacity(items.len());
    for it in &items {
        let strong = it.extract::<PyRef<'_, PyArray>>().is_ok() || it.hasattr("dtype")?;
        weak.push(!strong && crate::classify_scalar(it).is_some());
    }
    if !weak.iter().any(|w| *w) || weak.iter().all(|w| *w) {
        return items.iter().map(extract_or_ingest_ndarray).collect();
    }
    let mut out: Vec<Option<NdArray>> = Vec::with_capacity(items.len());
    let mut target: Option<DType> = None;
    for (it, &w) in items.iter().zip(weak.iter()) {
        if w {
            out.push(None);
        } else {
            let a = extract_or_ingest_ndarray(it)?;
            target = Some(match target {
                Some(t) => ionp_core::dtype::promote_dtype(t, a.dtype()),
                None => a.dtype(),
            });
            out.push(Some(a));
        }
    }
    let target = target.expect("at least one strong operand: the all-weak case returned above");
    let mut result = Vec::with_capacity(items.len());
    for (it, slot) in items.iter().zip(out.into_iter()) {
        match slot {
            Some(a) => result.push(a),
            None => result.push(
                crate::weak_scalar_wrapping(target, it)?
                    .expect("classified weak above, so classify_scalar matches"),
            ),
        }
    }
    Ok(result)
}

/// A single int or a sequence of ints -- the shape numpy accepts for
/// `axis=`/`shift=`/`reps=`-style arguments across this whole block.
fn isize_list_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    // Same ionp-own-array-type gap as `usize_list_from_pyobj` above --
    // `roll`'s `shift=`/`axis=` accept an anionpy array just like numpy's do.
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        let ints = pyref.inner.cast_to(DType::I64);
        let vals = match ints.buffer() {
            ionp_core::Buffer::I64(v) => v.clone(),
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

fn usize_list_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    // anionpy's own array type (e.g. `anionpy.repeat(m, anionpy.asarray([1,2,3,0]),
    // axis=0)`) never satisfies `Vec<i64>`/`i64` PyO3 extraction -- only a
    // real numpy array (via the buffer/sequence protocol) or a plain
    // list/tuple/scalar does. Check for it explicitly first so anionpy's own
    // array type is accepted here exactly like numpy's is (bug: the
    // previous version silently rejected its own array type while
    // accepting a foreign numpy one for this exact argument).
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        let ints = pyref.inner.cast_to(DType::I64);
        let vals = match ints.buffer() {
            ionp_core::Buffer::I64(v) => v.clone(),
            _ => unreachable!("cast_to(DType::I64) always yields Buffer::I64"),
        };
        return vals
            .into_iter()
            .map(|v| {
                if v < 0 {
                    Err(PyValueError::new_err("negative dimensions are not allowed"))
                } else {
                    Ok(v as usize)
                }
            })
            .collect();
    }
    if let Ok(seq) = obj.extract::<Vec<i64>>() {
        seq.into_iter()
            .map(|v| {
                if v < 0 {
                    Err(PyValueError::new_err("negative dimensions are not allowed"))
                } else {
                    Ok(v as usize)
                }
            })
            .collect()
    } else {
        let n: i64 = obj.extract()?;
        if n < 0 {
            return Err(PyValueError::new_err("negative dimensions are not allowed"));
        }
        Ok(vec![n as usize])
    }
}

/// `np.concatenate`'s own message for a non-sequence `arrays` argument
/// (a generator, a `dict`, a bare iterator -- anything lacking the real
/// `PySequence` protocol; `dict` fails this despite being iterable, and a
/// real ndarray or list/tuple passes it, both verified live against numpy
/// 2.5.1 -- same predicate `creation.rs`'s `shape_from_pyobj` already
/// relies on for the identical reason).
const CONCATENATE_NOT_A_SEQUENCE_MSG: &str = "The first input argument needs to be a sequence";

/// `np.stack`/`np.hstack`/`np.vstack` (and `row_stack`, its alias) share
/// this SECOND, differently-worded message for the same non-sequence
/// input -- confirmed byte-for-byte against numpy 2.5.1, including the
/// trailing period and the double-quoted `"sequence"` (distinct from
/// `CONCATENATE_NOT_A_SEQUENCE_MSG` above, which has neither).
const STACK_NOT_A_SEQUENCE_MSG: &str = "arrays to stack must be passed as a \"sequence\" type such as list or tuple.";

/// Reject a generator/dict/bare-iterator `arrays`/`tup` argument the same
/// way real numpy does: `PySequence` (list/tuple/ndarray/str/...) passes,
/// anything relying only on the iterator protocol does not. Must run
/// BEFORE `extract_ndarray_seq` (which uses `try_iter` and would happily
/// drain a generator numpy itself refuses).
fn require_sequence_arg(obj: &Bound<'_, PyAny>, msg: &'static str) -> PyResult<()> {
    if obj.cast::<PySequence>().is_err() {
        return Err(PyTypeError::new_err(msg));
    }
    Ok(())
}

/// Shared by `concatenate`/`stack`/`hstack`/`vstack` (`row_stack` aliases
/// `vstack`): numpy's `casting=` kwarg (default `'same_kind'`) gates
/// whichever dtype the output actually lands on -- the explicit `dtype=`
/// target when given, else the ordinary promoted-common-dtype of every
/// input, the SAME value `ionp_core::manip::concatenate` computes
/// internally via `promote_dtype` -- against EVERY input array's own
/// dtype, in input order. Mirrors that same fold exactly (not exported by
/// core, so recomputed here; dtype-only, unaffected by the `atleast_1d`/
/// `atleast_2d`/`expand_dims` shape promotion `hstack`/`vstack`/`stack`
/// apply before calling core `concatenate`, so folding over the ORIGINAL
/// input arrays here is equivalent).
fn concat_target_dtype(arrays: &[NdArray], explicit: Option<DType>) -> DType {
    match explicit {
        Some(d) => d,
        None => arrays
            .iter()
            .skip(1)
            .fold(arrays[0].dtype(), |acc, a| ionp_core::dtype::promote_dtype(acc, a.dtype())),
    }
}

/// The actual per-input casting-rule check itself, split out from
/// `concat_target_dtype` so callers can run it only AFTER
/// `ionp_core::manip`'s own structural validation (ndim/shape/axis/empty)
/// has already succeeded -- verified live against numpy 2.5.1 that those
/// structural errors take priority over any casting complaint (e.g. an
/// out-of-bounds `axis=` or a non-axis shape mismatch raises its own
/// `ValueError`/`AxisError` even when the requested cast would ALSO have
/// failed). Reports the FIRST offending input in order (not always index
/// 0 -- confirmed live: `np.concatenate([int8, int16], dtype='int8',
/// casting='safe')` blames the int16 array; reversing the input order
/// blames whichever comes first instead), with byte-identical `TypeError`
/// text to `ndarray.astype`'s own casting error (always the "array data"
/// noun here: `ionp_core::manip`'s structural check already rejects any
/// 0-d input before this ever runs, and `hstack`/`stack`/`vstack` promote
/// 0-d to >=1-d before reaching here too, both confirmed live).
fn check_concat_casting(arrays: &[NdArray], target: DType, rule: &str) -> PyResult<()> {
    for a in arrays {
        if !ionp_core::dtype::can_cast(a.dtype(), target, rule) {
            return Err(crate::astype_casting_type_error(a.dtype(), target, rule, a.ndim().max(1)));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// concatenate / stack
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (arrays, axis=0, *, out=None, dtype=None, casting=crate::OptionalArg::Omitted))]
fn concatenate(
    arrays: &Bound<'_, PyAny>,
    axis: Option<isize>,
    out: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    casting: crate::OptionalArg<'_>,
) -> PyResult<PyArray> {
    if out.is_some() {
        return Err(PyValueError::new_err("anionpy.concatenate: out= is not supported"));
    }
    // Order confirmed live against numpy 2.5.1: `concatenate` validates
    // the `casting=` kwarg's own value BEFORE it even looks at whether
    // `arrays` is a sequence (`np.concatenate(some_generator,
    // casting='bogus')` raises the casting `ValueError`, not the sequence
    // `TypeError`) -- the reverse of `stack`/`hstack`/`vstack` below.
    let rule = crate::check_casting_kwarg(&casting)?.unwrap_or_else(|| "same_kind".to_string());
    require_sequence_arg(arrays, CONCATENATE_NOT_A_SEQUENCE_MSG)?;
    let ax: Option<isize> = axis;
    let arrs = extract_concat_seq(arrays)?;
    let explicit_dtype = dtype.map(dtype_from_pyobj).transpose()?;
    let refs: Vec<&NdArray> = arrs.iter().collect();
    let inner = manip::concatenate(&refs, ax).map_err(to_py_err)?;
    // Structural validation (ndim/shape/axis/empty, all raised by
    // `manip::concatenate` above) takes priority over any casting
    // complaint -- confirmed live -- so the per-input casting-rule check
    // only runs once that has already succeeded.
    if !arrs.is_empty() {
        let target = concat_target_dtype(&arrs, explicit_dtype);
        check_concat_casting(&arrs, target, &rule)?;
    }
    let inner = match explicit_dtype {
        Some(d) => inner.cast_to(d),
        None => inner,
    };
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (arrays, axis=0, *, out=None, dtype=None, casting=crate::OptionalArg::Omitted))]
fn stack(
    arrays: &Bound<'_, PyAny>,
    axis: isize,
    out: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    casting: crate::OptionalArg<'_>,
) -> PyResult<PyArray> {
    if out.is_some() {
        return Err(PyValueError::new_err("anionpy.stack: out= is not supported"));
    }
    // `stack` checks the sequence-ness of `arrays` BEFORE validating
    // `casting=`'s own value -- the reverse of `concatenate` above,
    // confirmed live.
    require_sequence_arg(arrays, STACK_NOT_A_SEQUENCE_MSG)?;
    let rule = crate::check_casting_kwarg(&casting)?.unwrap_or_else(|| "same_kind".to_string());
    let arrs = extract_ndarray_seq(arrays)?;
    let explicit_dtype = dtype.map(dtype_from_pyobj).transpose()?;
    let refs: Vec<&NdArray> = arrs.iter().collect();
    let inner = manip::stack(&refs, axis).map_err(to_py_err)?;
    if !arrs.is_empty() {
        let target = concat_target_dtype(&arrs, explicit_dtype);
        check_concat_casting(&arrs, target, &rule)?;
    }
    let inner = match explicit_dtype {
        Some(d) => inner.cast_to(d),
        None => inner,
    };
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (tup, *, dtype=None, casting=crate::OptionalArg::Omitted))]
fn hstack(tup: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>, casting: crate::OptionalArg<'_>) -> PyResult<PyArray> {
    require_sequence_arg(tup, STACK_NOT_A_SEQUENCE_MSG)?;
    let rule = crate::check_casting_kwarg(&casting)?.unwrap_or_else(|| "same_kind".to_string());
    let arrs = extract_ndarray_seq(tup)?;
    let explicit_dtype = dtype.map(dtype_from_pyobj).transpose()?;
    let refs: Vec<&NdArray> = arrs.iter().collect();
    let inner = manip::hstack(&refs).map_err(to_py_err)?;
    if !arrs.is_empty() {
        let target = concat_target_dtype(&arrs, explicit_dtype);
        check_concat_casting(&arrs, target, &rule)?;
    }
    let inner = match explicit_dtype {
        Some(d) => inner.cast_to(d),
        None => inner,
    };
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (tup, *, dtype=None, casting=crate::OptionalArg::Omitted))]
fn vstack(tup: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>, casting: crate::OptionalArg<'_>) -> PyResult<PyArray> {
    require_sequence_arg(tup, STACK_NOT_A_SEQUENCE_MSG)?;
    let rule = crate::check_casting_kwarg(&casting)?.unwrap_or_else(|| "same_kind".to_string());
    let arrs = extract_ndarray_seq(tup)?;
    let explicit_dtype = dtype.map(dtype_from_pyobj).transpose()?;
    let refs: Vec<&NdArray> = arrs.iter().collect();
    let inner = manip::vstack(&refs).map_err(to_py_err)?;
    if !arrs.is_empty() {
        let target = concat_target_dtype(&arrs, explicit_dtype);
        check_concat_casting(&arrs, target, &rule)?;
    }
    let inner = match explicit_dtype {
        Some(d) => inner.cast_to(d),
        None => inner,
    };
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (tup, *, dtype=None, casting=crate::OptionalArg::Omitted))]
fn row_stack(tup: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>, casting: crate::OptionalArg<'_>) -> PyResult<PyArray> {
    vstack(tup, dtype, casting)
}

#[pyfunction]
fn dstack(tup: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let arrs = extract_ndarray_seq(tup)?;
    let refs: Vec<&NdArray> = arrs.iter().collect();
    let inner = manip::dstack(&refs).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
fn column_stack(tup: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let arrs = extract_ndarray_seq(tup)?;
    let refs: Vec<&NdArray> = arrs.iter().collect();
    let inner = manip::column_stack(&refs).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

// ---------------------------------------------------------------------------
// flip / fliplr / flipud
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (m, axis=None))]
fn flip(m: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let arr = extract_or_ingest_ndarray(m)?;
    let ndim = arr.ndim();
    let axes: Option<Vec<usize>> = match axis {
        None => None,
        Some(a) if a.is_none() => None,
        Some(a) => {
            let raw = isize_list_from_pyobj(a)?;
            let mut seen = std::collections::HashSet::new();
            let mut out = Vec::with_capacity(raw.len());
            for r in raw {
                let n = manip::normalize_axis(r, ndim).map_err(to_py_err)?;
                if !seen.insert(n) {
                    return Err(PyValueError::new_err("repeated axis"));
                }
                out.push(n);
            }
            Some(out)
        }
    };
    let inner = manip::flip(&arr, axes.as_deref());
    // Root-cause fix (same defect class as `trace` above and
    // `crate::numpy_scalar_from_0d`'s own doc comment): `flip` never changes
    // shape, so this only triggers when the INPUT was already 0-d (there are
    // no axes to reverse), and real numpy returns a numpy SCALAR there too
    // (verified live against numpy 2.5.1: `np.flip(np.array(5))` is
    // `numpy.int64`, not `numpy.ndarray`). A 0-d result can't share a base
    // with a parent array the way a view normally would, so this bypasses
    // `wrap_shape_view` entirely rather than trying to make that helper's
    // `Py<PyArray>` return type accommodate a non-array result.
    if inner.ndim() == 0 {
        return crate::numpy_scalar_from_0d(m.py(), &inner);
    }
    Ok(wrap_shape_view(m.py(), m, &arr, inner)?.into_any())
}

#[pyfunction]
fn fliplr(m: &Bound<'_, PyAny>) -> PyResult<Py<PyArray>> {
    let arr = extract_or_ingest_ndarray(m)?;
    let inner = manip::fliplr(&arr).map_err(to_py_err)?;
    wrap_shape_view(m.py(), m, &arr, inner)
}

#[pyfunction]
fn flipud(m: &Bound<'_, PyAny>) -> PyResult<Py<PyArray>> {
    let arr = extract_or_ingest_ndarray(m)?;
    let inner = manip::flipud(&arr).map_err(to_py_err)?;
    wrap_shape_view(m.py(), m, &arr, inner)
}

// ---------------------------------------------------------------------------
// roll
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (a, shift, axis=None))]
fn roll(a: &Bound<'_, PyAny>, shift: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let arr = extract_or_ingest_ndarray(a)?;
    let shifts = isize_list_from_pyobj(shift)?;
    let axes: Option<Vec<isize>> = match axis {
        None => None,
        Some(a) if a.is_none() => None,
        Some(a) => Some(isize_list_from_pyobj(a)?),
    };
    // numpy allows a tuple `shift` with `axis=None` (it sums the shifts
    // onto the single flattened axis) and a scalar `shift` with a tuple
    // `axis` (it broadcasts the scalar across every named axis) -- both
    // confirmed against real numpy 2.5.1, so there is no length
    // constraint to enforce here; `manip::roll` implements the actual
    // numpy broadcasting rule.
    let inner = manip::roll(&arr, &shifts, axes.as_deref()).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

// ---------------------------------------------------------------------------
// tile / repeat
// ---------------------------------------------------------------------------

/// True for a real `numpy.ndarray` (any subclass, `np.matrix` included) or
/// for anionpy's own array type -- and, critically, FALSE for a numpy SCALAR.
///
/// `np.float64`/`np.int8`/`np.bool_` all carry `.shape`, `.strides` and
/// `.dtype`, so every "duck-typed array" predicate you would reach for
/// first also matches them. `__array_finalize__` is the one attribute the
/// scalar types do not have (measured, numpy 2.5.1: ndarray True, matrix
/// True, 0-d ndarray True, float64/int8/bool_ False). The distinction is
/// load-bearing, not cosmetic -- see `tile` below, where `isinstance(A,
/// ndarray)` gates numpy's own all-ones fast path and therefore decides
/// whether a float `reps` raises or is silently accepted.
fn is_ndarray_instance(o: &Bound<'_, PyAny>) -> bool {
    o.extract::<PyRef<'_, PyArray>>().is_ok() || o.hasattr("__array_finalize__").unwrap_or(false)
}

/// Reproduce the ERROR behaviour of `ndarray.repeat`'s count coercion for
/// one `reps` element. The returned value is deliberately discarded: in
/// `tile` the count that actually shapes the output comes from
/// `shape_out`, and this call exists only so that the exceptions numpy
/// raises inside its repeat loop fire at the right moment and with the
/// right class and text. Measured against numpy 2.5.1 (`a` = int8 `[1,0,2]`,
/// `a.repeat(v, 0)`):
///
///   2.5 -> ok (6,)        2.9  -> ok (6,)      -0.5 -> ok (0,)   [truncates]
///   True -> (3,)          False -> (0,)        "2" -> ok (6,)    b"2" -> ok
///   np.float32(2.0) -> ok        np.uint64(2) -> ok   np.array(2, i8) -> ok
///   np.array(2.0)  -> TypeError: Cannot cast scalar from dtype('float64')
///                     to dtype('int64') according to the rule 'safe'
///   1j             -> TypeError: int() argument must be a string, a
///                     bytes-like object or a real number, not 'complex'
///   "x"            -> ValueError: invalid literal for int() with base 10: 'x'
///   nan            -> ValueError: cannot convert float NaN to integer
///   inf            -> OverflowError: cannot convert float infinity to integer
///   -1, -1.0, -(2**63) -> ValueError: negative dimensions are not allowed
///   2**63, 1e100, -1e100 -> OverflowError: Python int too large to convert
///                           to C long
///
/// Two orderings in there are easy to get backwards and both are measured,
/// not inferred. (1) `-(2**63)` is LLONG_MIN: it converts successfully and
/// then trips the negative check, so it is a `ValueError`, while `2**63`
/// does not convert at all and is an `OverflowError`. Range check first,
/// sign check second. (2) `-0.5` is NOT negative as far as numpy is
/// concerned -- it truncates toward zero to 0 first -- so the sign test has
/// to run on the TRUNCATED value, not the original.
///
/// `rows` is the length of the axis being repeated (`c.reshape(-1, n)`'s
/// first dimension). It exists because numpy's "negative dimensions"
/// test is NOT `count < 0` -- it is a test on the WRAPPING `intp` product
/// `rows * count`. For every ordinary count the two agree, since `rows` is
/// positive, but at the edge they diverge and numpy's answer is the
/// surprising one (all measured):
///
///   np.ones((3,), i8).repeat(-2**63, 0) -> ValueError negative dimensions
///   np.ones((6,), i8).repeat(-2**63, 0) -> ok, shape (0,)
///   np.ones((2,3),i8).repeat(-2**63, 0) -> ok, shape (0, 3)
///   np.ones((1,3),i8).repeat(-2**63, 0) -> ValueError negative dimensions
///
/// 6 * -2**63 and 2 * -2**63 both wrap to exactly 0, which is a legal
/// dimension; 3 * -2**63 and 1 * -2**63 stay negative. Returns the
/// truncated count so the caller can carry the running size forward.
fn tile_repeat_coercion_check(nrep: &Bound<'_, PyAny>, rows: i64) -> PyResult<i64> {
    const NEGATIVE: &str = "negative dimensions are not allowed";
    const TOO_LARGE: &str = "Python int too large to convert to C long";
    let signed_ok = |v: i64| -> PyResult<i64> {
        if rows.wrapping_mul(v) < 0 {
            return Err(PyValueError::new_err(NEGATIVE));
        }
        Ok(v)
    };

    // Python `complex` only. A numpy complex SCALAR is accepted by numpy's
    // repeat (with a ComplexWarning) -- `np.complex64(2+0j)` gives shape
    // (6,) -- so this must not be a general "is it complex" test.
    if nrep.is_instance_of::<pyo3::types::PyComplex>() {
        return Err(PyTypeError::new_err(
            "int() argument must be a string, a bytes-like object or a real number, not 'complex'",
        ));
    }
    // A 0-d ARRAY of non-integer dtype fails inside numpy's safe-cast of
    // the count, with a message that names both dtypes. The equivalent
    // numpy SCALAR (`np.float64(2.0)`) is accepted, which is why
    // `is_ndarray_instance` and not a `.dtype` probe decides this.
    if is_ndarray_instance(nrep) {
        if let Ok(dt) = nrep.getattr("dtype") {
            let name = dt.getattr("name").and_then(|n| n.extract::<String>()).unwrap_or_default();
            let is_int = name.starts_with("int") || name.starts_with("uint") || name == "bool";
            let ndim = nrep.getattr("ndim").and_then(|n| n.extract::<usize>()).unwrap_or(1);
            if ndim == 0 && !is_int {
                return Err(PyTypeError::new_err(format!(
                    "Cannot cast scalar from dtype('{name}') to dtype('int64') according to the rule 'safe'"
                )));
            }
        }
    }
    // Integers (Python int, numpy integer scalars, bool) go through the
    // exact-width path so that `2**63 - 1` is not pushed over the edge by
    // an f64 round-trip.
    if let Ok(v) = nrep.extract::<i128>() {
        if v > i64::MAX as i128 || v < i64::MIN as i128 {
            return Err(pyo3::exceptions::PyOverflowError::new_err(TOO_LARGE));
        }
        return signed_ok(v as i64);
    }
    if let Ok(f) = nrep.extract::<f64>() {
        if f.is_nan() {
            return Err(PyValueError::new_err("cannot convert float NaN to integer"));
        }
        if f.is_infinite() {
            return Err(pyo3::exceptions::PyOverflowError::new_err(
                "cannot convert float infinity to integer",
            ));
        }
        let t = f.trunc();
        if t > i64::MAX as f64 || t < i64::MIN as f64 {
            return Err(pyo3::exceptions::PyOverflowError::new_err(TOO_LARGE));
        }
        return signed_ok(t as i64);
    }
    if let Ok(s) = nrep.extract::<String>() {
        match s.trim().parse::<i64>() {
            Ok(v) => return signed_ok(v),
            Err(_) => {
                return Err(PyValueError::new_err(format!(
                    "invalid literal for int() with base 10: '{s}'"
                )))
            }
        }
    }
    // Anything else (a list, an arbitrary object) is left alone rather
    // than assigned an invented error: it survived the `s * t` multiply
    // above, and if it is not a valid dimension the final `reshape` below
    // reports it with CPython's own `__index__` message. Inventing a
    // message here would be a guess, and a guess that fires before the
    // real one is worse than no check at all.
    Ok(1)
}

/// CPython's `PyNumber_Index`, used verbatim so that the "'float' object
/// cannot be interpreted as an integer" family of messages -- including
/// the fully-qualified `'numpy.float64'` spelling, which comes from the
/// type's `tp_name` and cannot be reconstructed reliably from `__name__`
/// -- is produced by the interpreter rather than approximated here.
fn index_as_i64(o: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = o.py();
    let obj = unsafe {
        let p = pyo3::ffi::PyNumber_Index(o.as_ptr());
        if p.is_null() {
            return Err(PyErr::fetch(py));
        }
        Bound::from_owned_ptr(py, p)
    };
    // An `__index__` that does not fit `intp` is a DIMENSION error, not an
    // integer-conversion one: `np.ones(3, np.int8).reshape((2**63,))` and
    // `.reshape((3 * 2**63,))` both raise `ValueError: Maximum allowed
    // dimension exceeded`, while `.reshape((-2**63,))` -- which does fit,
    // exactly -- is accepted and read as an "infer this axis" marker.
    // Boundary measured, not assumed.
    obj.extract::<i64>()
        .map_err(|_| PyValueError::new_err("Maximum allowed dimension exceeded"))
}

/// PHANTOM FOUND + FIXED 2026-08-04 (Monday).
///
/// `tile` was declared "exact" while 11 of 36 `reps` spellings diverged.
/// The old body was `usize_list_from_pyobj(reps)` -- a single "give me a
/// list of non-negative ints" coercion -- and numpy's `tile` is nothing
/// like that. It is a small Python function whose observable behaviour is
/// almost entirely a by-product of WHICH primitive fails FIRST:
///
///   tup = tuple(reps) except TypeError -> (reps,)
///   d = len(tup)
///   if all(x == 1 for x in tup) and isinstance(A, ndarray):
///       return array(A, copy=True, ndmin=d)          # <- no int coercion
///   c = array(A, copy=None, subok=True, ndmin=d)
///   if d < c.ndim: tup = (1,)*(c.ndim-d) + tup
///   shape_out = tuple(s*t for s, t in zip(c.shape, tup))   # <- Python `*`
///   n = c.size
///   if n > 0:
///       for dim_in, nrep in zip(c.shape, tup):
///           if nrep != 1: c = c.reshape(-1, n).repeat(nrep, 0)
///           n //= dim_in
///   return c.reshape(shape_out)                       # <- __index__ here
///
/// Three consequences, each of which was a live bug:
///
/// 1. THE ALL-ONES FAST PATH NEVER COERCES. `np.tile(int8_arr, 1.0)`
///    succeeds and returns the array unchanged, because `1.0 == 1` is True
///    and the function returns before any integer conversion happens. Same
///    for `True`, `np.True_`, `np.float64(1.0)`, `(1.0, 1.0)`, `[]`,
///    `np.array([])`, and `{1: 2}` (a dict tuple-izes to its KEYS). anionpy
///    raised `TypeError: 'float' object cannot be interpreted as an
///    integer` for every one of them. The `isinstance(A, ndarray)` half of
///    the guard is equally real: `np.tile([1,2,3], 1.0)` -- same reps, a
///    LIST for `A` -- does NOT take the fast path and DOES raise.
///
/// 2. THE 'float' MESSAGE COMES FROM THE FINAL RESHAPE, NOT FROM REPEAT.
///    `ndarray.repeat` happily accepts `2.0`, `2.5`, `"2"` and `b"2"`. What
///    rejects them is `c.reshape(shape_out)`, where `shape_out` holds
///    `3 * 2.0 == 6.0`. That is why the offending element is reported with
///    the dtype of the PRODUCT: `np.tile(a, np.array([2., 1.]))` says
///    `'numpy.float64'`, not `'float'`, and a `(1.0, 2.0)` tuple reports
///    `'float'` for its SECOND element while the first is skipped by
///    `nrep != 1`. Getting this backwards produces messages that are
///    plausible, uniformly wrong, and invisible to a type-only check.
///
/// 3. SOME ERRORS DO FIRE INSIDE THE LOOP, AND THEY WIN. `-1.0` is a
///    `ValueError: negative dimensions are not allowed` (from repeat), not
///    the reshape `TypeError`; `1j` is `int() argument must be ...`;
///    `np.array(2.0)` is the safe-cast `TypeError`; `1e100` and `2**63`
///    are `OverflowError`. And the loop is guarded by `n > 0`, so on an
///    EMPTY operand none of them fire at all and the reshape message wins
///    instead (`np.tile(np.array([], np.int8), 2.5)` -> `TypeError:
///    'float' ...`, not `ValueError`). Ordering is the contract.
///
/// So the sequence below is: tuple-ize, all-ones fast path, `s * t` via
/// Python's own multiply, repeat-coercion checks (only when `nrep != 1`
/// and only when the operand is non-empty), then `__index__` on the
/// products. Only after all four gates pass does any arithmetic happen,
/// and it happens where it belongs, in `ionp_core::manip::tile`.
///
/// KNOWN REMAINING DIVERGENCE, deliberately not papered over: a negative
/// rep on an EMPTY operand skips the repeat loop and reaches numpy's
/// reshape, which reads the negative as a `-1` "infer this axis" marker
/// and reports `ValueError: cannot reshape array of size 0 into shape
/// (0,newaxis)`. Reproducing that means reproducing `reshape`'s
/// newaxis-inference diagnostics, which belong to `reshape`, not here.
/// `np.tile(np.zeros((0,3)), (2,-1))` is the repro.
#[pyfunction]
fn tile(a: &Bound<'_, PyAny>, reps: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let py = reps.py();

    // `tup = tuple(reps)`, falling back to a 1-tuple on TypeError. Note
    // this runs BEFORE `array(A)`, matching numpy's own statement order,
    // and that it accepts any iterable: a dict (yielding its keys), a bare
    // iterator, a `range`, an ndarray.
    let tup: Vec<Bound<'_, PyAny>> = match reps.try_iter() {
        Ok(it) => it.collect::<PyResult<Vec<_>>>()?,
        Err(e) if e.is_instance_of::<PyTypeError>(py) => vec![reps.clone()],
        Err(e) => return Err(e),
    };
    let d = tup.len();

    let arr = extract_or_ingest_ndarray(a)?;

    // `all(x == 1 for x in tup)` -- short-circuiting, because numpy's
    // generator does too and a later element that raises on comparison is
    // never reached once an earlier one is False.
    let mut all_ones = true;
    for x in &tup {
        if !x.eq(1i32)? {
            all_ones = false;
            break;
        }
    }
    if all_ones && is_ndarray_instance(a) {
        let inner = manip::tile(&arr, &vec![1usize; d]).map_err(to_py_err)?;
        return Ok(PyArray { inner });
    }

    // `c = array(A, ndmin=d)` then, if `d < c.ndim`, left-pad `tup` with
    // literal 1s. `manip::tile` performs the same padding internally, so
    // only the shapes are materialised here.
    let ndim_out = arr.ndim().max(d);
    let mut padded_shape = vec![1usize; ndim_out - arr.ndim()];
    padded_shape.extend_from_slice(arr.shape());
    let one = PyInt::new(py, 1i32).into_any();
    let mut padded_tup: Vec<Bound<'_, PyAny>> = vec![one; ndim_out - d];
    padded_tup.extend(tup.iter().cloned());

    // `shape_out = tuple(s * t ...)` -- Python's multiply, so that e.g.
    // `None` reports "unsupported operand type(s) for *: 'int' and
    // 'NoneType'" exactly as numpy does, and a `str` rep silently becomes
    // a repeated string that only the `__index__` below rejects.
    let mut shape_out = Vec::with_capacity(ndim_out);
    for (s, t) in padded_shape.iter().zip(padded_tup.iter()) {
        shape_out.push(PyInt::new(py, *s as i64).into_any().mul(t)?);
    }

    // The `n > 0` guard is numpy's, and it is why an empty operand never
    // reports a repeat-loop error.
    // The loop mirrors numpy's exactly, including the running `n` and the
    // `c.reshape(-1, n)` row count that `tile_repeat_coercion_check` needs
    // for its wrapping-product sign test. `size` tracks `c.size` as the
    // repeats accumulate; both are carried in wrapping `i64`, which is the
    // width numpy does this arithmetic in.
    if arr.size() > 0 {
        let mut n = arr.size() as i64;
        let mut size = arr.size() as i64;
        for (i, t) in padded_tup.iter().enumerate() {
            if !t.eq(1i32)? {
                let rows = size / n;
                let cnt = tile_repeat_coercion_check(t, rows)?;
                size = size.wrapping_mul(cnt);
            }
            n /= padded_shape[i] as i64;
        }
    }

    let mut dims = Vec::with_capacity(ndim_out);
    for so in &shape_out {
        dims.push(index_as_i64(so)?);
    }

    // Recover the per-axis repetition counts from the products numpy
    // itself would have reshaped to. Dividing back out (rather than
    // re-coercing `tup`) keeps this consistent with `shape_out` by
    // construction; a zero-length axis contributes nothing to the output
    // either way, so its count is irrelevant and taken as 0.
    // A NEGATIVE entry in `shape_out` is not an error in itself -- it
    // reaches `reshape`, which reads any negative as an "infer this axis"
    // marker. It can only get this far on an EMPTY operand (a positive
    // axis times a negative rep would have been caught by the repeat loop
    // above, which the `n > 0` guard skips when the operand is empty), so
    // the inferred axis is always being solved against a total of 0.
    // Measured, numpy 2.5.1, `z = np.zeros((0,3), np.int8)`:
    //   np.tile(z, (-1, 2))  -> ok, shape (0, 6)   [0 * -1 == 0, no marker]
    //   np.tile(z, (2, -1))  -> ValueError: cannot reshape array of size 0
    //                           into shape (0,newaxis)
    //   np.tile(np.zeros((0,3,2)), (2,-3,1))
    //                        -> ... into shape (0,newaxis,2)
    //   np.tile(np.zeros((0,3,2)), (-2,-3,-4))
    //                        -> ValueError: can only specify one unknown
    //                           dimension
    // The rendering is the raw dimension list with every negative printed
    // as `newaxis`, comma-separated, no spaces.
    let negatives = dims.iter().filter(|&&d| d < 0).count();
    if negatives > 1 {
        return Err(PyValueError::new_err("can only specify one unknown dimension"));
    }
    if negatives == 1 {
        let known: i64 = dims.iter().filter(|&&d| d >= 0).product();
        // `c.size` at the point numpy calls `reshape`. It is 0 by
        // construction: a marker can only survive the repeat loop when
        // that loop was skipped, and it is skipped exactly when the
        // operand is empty.
        let total: i64 = 0;
        if known == 0 {
            let rendered = dims
                .iter()
                .map(|&d| if d < 0 { "newaxis".to_string() } else { d.to_string() })
                .collect::<Vec<_>>()
                .join(",");
            return Err(PyValueError::new_err(format!(
                "cannot reshape array of size {total} into shape ({rendered})"
            )));
        }
    }
    let mut counts = Vec::with_capacity(ndim_out);
    for (i, &s) in padded_shape.iter().enumerate() {
        if s == 0 {
            counts.push(0usize);
        } else if dims[i] < 0 {
            // Single unknown against a total of 0 and a non-zero known
            // product: the inferred extent is 0.
            counts.push(0usize);
        } else {
            counts.push((dims[i] / s as i64) as usize);
        }
    }
    let inner = manip::tile(&arr, &counts).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (a, repeats, axis=None))]
fn repeat(a: &Bound<'_, PyAny>, repeats: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<PyArray> {
    let arr = extract_or_ingest_ndarray(a)?;
    // BUG FOUND + FIXED 2026-08-03 (Monday): 0-d operand + explicit axis.
    // numpy's `repeat` promotes a 0-dimensional operand to shape (1,)
    // BEFORE it validates `axis`, so `np.repeat(np.array(3.0), 2, axis=0)`
    // succeeds with shape (2,) and an out-of-range axis on a 0-d operand
    // reports "array of dimension 1", never "dimension 0" (both measured
    // against real numpy 2.5.1, not inferred from the docs).
    //
    // This is the INVERSE of the 0-d defect fixed across the reduction
    // family the same day: there anionpy wrongly ACCEPTED an axis numpy
    // rejects; here anionpy wrongly REJECTED one numpy accepts. Sharing a
    // shape does not make them the same bug, and neither fix implies the
    // other -- numpy's 0-d rules are per-function and inconsistent by
    // design; reproducing the inconsistency IS the contract.
    let arr = if arr.ndim() == 0 {
        arr.ravel_order("C").map_err(to_py_err)?
    } else {
        arr
    };
    let target_len = match axis {
        Some(ax) => {
            let n = manip::normalize_axis(ax, arr.ndim()).map_err(to_py_err)?;
            arr.shape()[n]
        }
        None => arr.size(),
    };
    let raw = usize_list_from_pyobj(repeats)?;
    let resolved: Vec<usize> = if raw.len() == 1 {
        vec![raw[0]; target_len]
    } else {
        raw
    };
    let ax = match axis {
        Some(a) => Some(manip::normalize_axis(a, arr.ndim()).map_err(to_py_err)?),
        None => None,
    };
    let inner = manip::repeat(&arr, &resolved, ax).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

// ---------------------------------------------------------------------------
// broadcast_shapes / broadcast_arrays
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (*args))]
fn broadcast_shapes(py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let shapes: Vec<Vec<usize>> = args.iter().map(|a| usize_list_from_pyobj(&a)).collect::<PyResult<_>>()?;
    let refs: Vec<&[usize]> = shapes.iter().map(|s| s.as_slice()).collect();
    let result = manip::broadcast_shapes(&refs).map_err(to_py_err)?;
    Ok(PyTuple::new(py, result)?.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (*args, subok=None))]
fn broadcast_arrays(py: Python<'_>, args: &Bound<'_, PyTuple>, subok: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    crate::check_subok(subok);
    let srcs: Vec<Bound<'_, PyAny>> = args.iter().collect();
    let arrs: Vec<NdArray> = srcs.iter().map(extract_or_ingest_ndarray).collect::<PyResult<_>>()?;
    let out = manip::broadcast_arrays(&arrs).map_err(to_py_err)?;
    // Each output forwards `.base`/`OWNDATA` from its OWN positional input
    // (same `wrap_shape_view` free-function-shape-op plumbing as
    // `broadcast_to`, which this is the multi-array sibling of).
    //
    // NOT marked read-only, unlike `broadcast_to`: verified live against
    // real numpy 2.5.1 that `np.broadcast_arrays(...)`'s outputs are
    // `WRITEABLE=True` (numpy only emits a `FutureWarning` that a later
    // version will flip this default -- current numpy still returns
    // writeable arrays here), so matching TODAY's numpy means leaving these
    // writeable, not read-only.
    let objs: Vec<Py<PyArray>> = out
        .into_iter()
        .zip(arrs.iter().zip(srcs.iter()))
        // `identity_if_unchanged`, not `wrap_shape_view`: when an input is
        // already at the broadcast shape numpy hands the SAME OBJECT back
        // (`np.broadcast_arrays(a)[0] is a` -> True, verified on 2.5.1), so
        // its `.base` stays None and `OWNDATA` stays True.
        .map(|(inner, (before, src))| crate::identity_if_unchanged(py, src, before, inner))
        .collect::<PyResult<_>>()?;
    Ok(PyTuple::new(py, objs)?.into_any().unbind())
}

// ---------------------------------------------------------------------------
// atleast_1d / atleast_2d / atleast_3d
// ---------------------------------------------------------------------------

macro_rules! atleast_fn {
    ($name:ident, $core:path) => {
        #[pyfunction]
        #[pyo3(signature = (*arys))]
        fn $name(py: Python<'_>, arys: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
            if arys.is_empty() {
                return Err(PyTypeError::new_err(concat!(
                    stringify!($name),
                    "() missing at least one required positional argument"
                )));
            }
            let srcs: Vec<Bound<'_, PyAny>> = arys.iter().collect();
            let arrs: Vec<NdArray> = srcs.iter().map(extract_or_ingest_ndarray).collect::<PyResult<_>>()?;
            // numpy returns the INPUT OBJECT itself when it already has
            // enough dimensions (`np.atleast_1d(a) is a` -> True for
            // ndim >= 1), and a VIEW -- not a copy -- when it has to add
            // one (`np.atleast_1d(np.array(5.0)).base is not None`).
            // Returning a fresh owning array in either case is visible
            // through `.base`, `OWNDATA`, and write-through aliasing.
            if arrs.len() == 1 {
                let inner = $core(&arrs[0]);
                return Ok(crate::identity_if_unchanged(py, &srcs[0], &arrs[0], inner)?.into_any());
            }
            let objs: Vec<Py<PyArray>> = arrs
                .iter()
                .zip(srcs.iter())
                .map(|(a, src)| crate::identity_if_unchanged(py, src, a, $core(a)))
                .collect::<PyResult<_>>()?;
            // Real numpy returns a plain tuple for the multi-array call
            // form (confirmed via direct introspection: `type(np.atleast_1d(a, b))`
            // is `tuple`, not `list`) -- matched here for API fidelity even
            // though the differential harness (this task) only exercises
            // the single-array form (see manip_cases.py's module docstring
            // for why the multi-array tuple form isn't wired through the
            // harness's `multi_output` mechanism).
            Ok(PyTuple::new(py, objs)?.into_any().unbind())
        }
    };
}

atleast_fn!(atleast_1d, manip::atleast_1d);
atleast_fn!(atleast_2d, manip::atleast_2d);
atleast_fn!(atleast_3d, manip::atleast_3d);

// ---------------------------------------------------------------------------
// diag / diagonal / diagflat / tril / triu / trace
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (v, k=0))]
fn diag(v: &Bound<'_, PyAny>, k: isize) -> PyResult<PyArray> {
    let arr = extract_or_ingest_ndarray(v)?;
    let inner = manip::diag(&arr, k).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (v, k=0))]
fn diagflat(v: &Bound<'_, PyAny>, k: isize) -> PyResult<PyArray> {
    let arr = extract_or_ingest_ndarray(v)?;
    let inner = manip::diagflat(&arr, k).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (a, offset=0, axis1=0, axis2=1))]
fn diagonal(a: &Bound<'_, PyAny>, offset: isize, axis1: isize, axis2: isize) -> PyResult<PyArray> {
    let arr = extract_or_ingest_ndarray(a)?;
    let inner = manip::diagonal(&arr, offset, axis1, axis2).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (m, k=0))]
fn tril(m: &Bound<'_, PyAny>, k: isize) -> PyResult<PyArray> {
    let arr = extract_or_ingest_ndarray(m)?;
    let inner = manip::tril(&arr, k).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (m, k=0))]
fn triu(m: &Bound<'_, PyAny>, k: isize) -> PyResult<PyArray> {
    let arr = extract_or_ingest_ndarray(m)?;
    let inner = manip::triu(&arr, k).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (a, offset=0, axis1=0, axis2=1, dtype=None, out=None))]
fn trace(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    offset: isize,
    axis1: isize,
    axis2: isize,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    if out.is_some() {
        return Err(PyValueError::new_err("anionpy.trace: out= is not supported"));
    }
    let arr = extract_or_ingest_ndarray(a)?;
    let dt: Option<DType> = match dtype {
        // `manip::trace` unconditionally `cast_to`s the diagonal into this
        // target (see its own body); S/U has no numeric-cast arm, so decline
        // cleanly instead of panicking.
        Some(d) => Some(crate::dtype_from_pyobj_no_su(d)?),
        None => None,
    };
    let inner = manip::trace(&arr, offset, axis1, axis2, dt).map_err(to_py_err)?;
    // Root-cause fix (docs/scalar-return-type-defect.md): `np.trace` on a
    // plain 2-D input (`axis1`/`axis2` covering the whole array, the
    // overwhelmingly common case) collapses to 0 dimensions, and real numpy
    // returns a numpy SCALAR there (verified: `np.trace(np.eye(2))` is
    // `numpy.float64`, not `numpy.ndarray`) -- same defect class, same fix,
    // as the reduction family in `reductions.rs` (see
    // `crate::numpy_scalar_from_0d`'s own doc comment for the shared
    // mechanism). Higher-rank input (`axis1`/`axis2` leave batch dimensions
    // behind) legitimately stays an array; that path is untouched.
    if inner.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &inner);
    }
    use pyo3::IntoPyObjectExt;
    Py::new(py, PyArray { inner })?.into_py_any(py)
}

// ---------------------------------------------------------------------------
// split / array_split / hsplit / vsplit / dsplit
// ---------------------------------------------------------------------------

/// `indices_or_sections`: numpy dispatches on `len(indices_or_sections)`
/// (a bare `TypeError` from a missing `__len__` sends it down the
/// scalar-int path) -- anionpy's own array type is checked first (matches
/// `isize_list_from_pyobj`'s established pattern above: it never
/// satisfies a plain `Vec<isize>`/`isize` PyO3 extraction), then a
/// Python-level list/tuple/numpy-array (`Vec<isize>`), falling back to a
/// scalar `isize`.
fn parse_indices_or_sections(obj: &Bound<'_, PyAny>) -> PyResult<manip::SplitArg> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        if pyref.inner.ndim() == 0 {
            let ints = pyref.inner.cast_to(DType::I64);
            let v = match ints.buffer() {
                Buffer::I64(v) => v[0],
                _ => unreachable!("cast_to(DType::I64) always yields Buffer::I64"),
            };
            return Ok(manip::SplitArg::Sections(v as isize));
        }
        let ints = pyref.inner.cast_to(DType::I64);
        let vals = match ints.to_contiguous().buffer() {
            Buffer::I64(v) => v.clone(),
            _ => unreachable!("cast_to(DType::I64) always yields Buffer::I64"),
        };
        return Ok(manip::SplitArg::Indices(vals.into_iter().map(|v| v as isize).collect()));
    }
    if let Ok(seq) = obj.extract::<Vec<isize>>() {
        Ok(manip::SplitArg::Indices(seq))
    } else {
        Ok(manip::SplitArg::Sections(obj.extract::<isize>()?))
    }
}

/// `np.split`'s own equal-division pre-check evaluates `N % sections` in
/// plain Python -- for `sections == 0` that expression itself raises
/// `ZeroDivisionError: division by zero` (verified live against numpy
/// 2.5.1) BEFORE `array_split`'s own "number sections must be larger than
/// 0." `ValueError` is ever reached. `ionp_core::manip` cannot itself
/// raise a `ZeroDivisionError` (it only owns `IonpError`, which has no
/// such variant, and this task does not own `error.rs` to add one), so
/// this one Python-visible exception type is raised here directly instead
/// of being threaded through `manip::split`.
fn split_dispatch(a: &NdArray, arg: &manip::SplitArg, axis: isize, strict: bool) -> PyResult<Vec<NdArray>> {
    if strict {
        if let manip::SplitArg::Sections(0) = arg {
            return Err(PyZeroDivisionError::new_err("division by zero"));
        }
        manip::split(a, arg, axis).map_err(to_py_err)
    } else {
        manip::array_split(a, arg, axis).map_err(to_py_err)
    }
}

fn wrap_parts(py: Python<'_>, parts: Vec<NdArray>) -> PyResult<Py<PyAny>> {
    let objs: Vec<Py<PyArray>> = parts.into_iter().map(|inner| Py::new(py, PyArray { inner })).collect::<PyResult<_>>()?;
    Ok(PyList::new(py, objs)?.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections, axis=0))]
fn split(py: Python<'_>, ary: &Bound<'_, PyAny>, indices_or_sections: &Bound<'_, PyAny>, axis: isize) -> PyResult<Py<PyAny>> {
    let arg = parse_indices_or_sections(indices_or_sections)?;
    // `split()`'s own scalar-sections precheck touches `ary.shape[axis]`
    // with no fallback (unlike `array_split`) -- see the block comment on
    // `require_shape_attr` above.
    if matches!(arg, manip::SplitArg::Sections(_)) {
        require_shape_attr(ary)?;
    }
    let arr = extract_or_ingest_ndarray(ary)?;
    let parts = split_dispatch(&arr, &arg, axis, true)?;
    wrap_parts(py, parts)
}

#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections, axis=0))]
fn array_split(py: Python<'_>, ary: &Bound<'_, PyAny>, indices_or_sections: &Bound<'_, PyAny>, axis: isize) -> PyResult<Py<PyAny>> {
    let arr = extract_or_ingest_ndarray(ary)?;
    let arg = parse_indices_or_sections(indices_or_sections)?;
    let parts = split_dispatch(&arr, &arg, axis, false)?;
    wrap_parts(py, parts)
}

#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections))]
fn hsplit(py: Python<'_>, ary: &Bound<'_, PyAny>, indices_or_sections: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // numpy's own `hsplit`: `_nx.ndim(ary) == 0` (lenient, asarray-fallback)
    // first, THEN an unconditional strict `ary.ndim` right after -- both
    // branches of its own `if ary.ndim > 1: ... else: ...` touch it before
    // ever reaching `split()`.
    let ndim = ndim_lenient(ary)?;
    if ndim < 1 {
        return Err(PyValueError::new_err("hsplit only works on arrays of 1 or more dimensions"));
    }
    ndim_strict(ary)?;
    let arg = parse_indices_or_sections(indices_or_sections)?;
    let axis = if ndim == 1 { 0isize } else { 1isize };
    let arr = extract_or_ingest_ndarray(ary)?;
    let parts = split_dispatch(&arr, &arg, axis, true)?;
    wrap_parts(py, parts)
}

#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections))]
fn vsplit(py: Python<'_>, ary: &Bound<'_, PyAny>, indices_or_sections: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // numpy's own `vsplit`: lenient `_nx.ndim(ary) < 2` first (a raw 1-D
    // list/range genuinely fails HERE with ValueError, before ever
    // touching `.shape`), then delegates to `split(ary, ..., 0)`, whose own
    // strict `ary.shape[axis]` is what rejects a raw >=2-D list/tuple. This
    // is why vsplit's error CLASS genuinely varies by input ndim, unlike
    // hsplit/rollaxis/dsplit's uniform-across-forms AttributeError.
    let ndim = ndim_lenient(ary)?;
    if ndim < 2 {
        return Err(PyValueError::new_err("vsplit only works on arrays of 2 or more dimensions"));
    }
    require_shape_attr(ary)?;
    let arg = parse_indices_or_sections(indices_or_sections)?;
    let arr = extract_or_ingest_ndarray(ary)?;
    let parts = split_dispatch(&arr, &arg, 0, true)?;
    wrap_parts(py, parts)
}

#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections))]
fn dsplit(py: Python<'_>, ary: &Bound<'_, PyAny>, indices_or_sections: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Same lenient-then-strict shape as vsplit, but dsplit's own low-ndim
    // ValueError threshold (<3) means a raw list3d (ndim 3, passes the
    // lenient check) still falls through to split()'s strict `.shape`
    // touch and genuinely raises AttributeError under real numpy.
    let ndim = ndim_lenient(ary)?;
    if ndim < 3 {
        return Err(PyValueError::new_err("dsplit only works on arrays of 3 or more dimensions"));
    }
    require_shape_attr(ary)?;
    let arg = parse_indices_or_sections(indices_or_sections)?;
    let arr = extract_or_ingest_ndarray(ary)?;
    let parts = split_dispatch(&arr, &arg, 2, true)?;
    wrap_parts(py, parts)
}

// ---------------------------------------------------------------------------
// append / resize
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (arr, values, axis=None))]
fn append(arr: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<PyArray> {
    let a = extract_or_ingest_ndarray(arr)?;
    let v = extract_or_ingest_ndarray(values)?;
    let inner = manip::append(&a, &v, axis).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

/// `np.resize`'s own negative-shape-element message is a bespoke string
/// ("all elements of `new_shape` must be non-negative", verified live
/// against numpy 2.5.1) -- distinct from the generic "negative dimensions
/// are not allowed" that `usize_list_from_pyobj` raises for every other
/// shape-consuming binding in this module, so `resize` needs its own
/// small parser rather than reusing that shared helper.
fn resize_shape_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let raw: Vec<i64> = if let Ok(seq) = obj.extract::<Vec<i64>>() {
        seq
    } else {
        vec![obj.extract::<i64>()?]
    };
    raw.into_iter()
        .map(|v| {
            if v < 0 {
                Err(PyValueError::new_err("all elements of `new_shape` must be non-negative"))
            } else {
                Ok(v as usize)
            }
        })
        .collect()
}

#[pyfunction]
fn resize(a: &Bound<'_, PyAny>, new_shape: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let arr = extract_or_ingest_ndarray(a)?;
    let shape = resize_shape_from_pyobj(new_shape)?;
    let inner = manip::resize(&arr, &shape).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

// ---------------------------------------------------------------------------
// insert / delete
// ---------------------------------------------------------------------------

fn is_plain_int(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PyInt>() && !obj.is_instance_of::<PyBool>()
}

/// Saturating-cast defect #4 fix (`insert`/`delete`'s `obj` argument):
/// mirrors `array_impl`'s own real-ndarray-vs-list detection (this file's
/// header explains why that detection is duplicated locally rather than
/// shared -- `creation.rs`'s copy is not `pub`). Both `np.insert` and
/// `np.delete` special-case an EMPTY `obj` that is NOT already a real
/// ndarray (a Python list/tuple/range) by force-casting it to `intp`
/// regardless of its apparent dtype (`obj.astype(intp)` on the empty
/// array numpy's own `_function_base_impl.py` builds from it) -- so
/// `insert(a, [], 99)` and `delete(a, [])` both succeed as a true no-op,
/// while `insert(a, np.array([]), 99)` and `delete(a, np.array([]))` (an
/// ALREADY-ndarray empty float64 default) both raise `IndexError: arrays
/// used as indices must be of integer (or boolean) type` -- measured live
/// against numpy 2.5.1, `.venv/bin/python`, fresh-process scripts.
fn obj_is_ndarray_instance(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if obj.extract::<PyRef<'_, PyArray>>().is_ok() {
        return Ok(true);
    }
    if obj.hasattr("__array_interface__")? {
        return Ok(true);
    }
    Ok(obj.get_type().name()?.to_string() == "ndarray")
}

/// The exact message text numpy's own fancy-indexing machinery raises
/// (`keep[obj,] = False` / the multi-index `old_mask[indices]` assignment)
/// when `obj`'s dtype is neither integer nor boolean -- shared verbatim by
/// both `insert`'s multi-element/size-0-already-ndarray path and EVERY
/// `delete` non-integer-non-bool path (measured identical text on both
/// call sites, live, numpy 2.5.1). This is a literal string constant, not
/// a helper that embeds either caller's differing CONTROL FLOW -- sharing
/// the string is safe precisely because both callers only reach it after
/// their own independent, non-shared bounds/shape logic has already
/// decided to raise it.
const FLOAT_INDEX_MSG: &str = "arrays used as indices must be of integer (or boolean) type";

/// Reads a floating-dtype 0-d/1-element `NdArray` back out as a plain
/// `f64`, exactly (F16/F32 -> F64 is always a lossless upcast) -- used
/// only after `arr.dtype().is_floating()` has already been checked, so
/// this never goes through `cast_to(DType::I64)`'s saturating float->int
/// cast that caused defect #4 in the first place.
fn f64_scalar_value(arr: &NdArray) -> f64 {
    let f = arr.cast_to(DType::F64);
    match f.to_contiguous().buffer() {
        Buffer::F64(v) => v[0],
        _ => unreachable!("cast_to(DType::F64) always yields Buffer::F64"),
    }
}

/// `np.insert`'s own axis normalization (`_function_base_impl.py`,
/// `insert`): `axis=None` ravels `arr` first, so the effective axis
/// reported in an out-of-bounds message is always `0` and `N` is the
/// array's total element count; `axis=Some(ax)` normalizes `ax` against
/// `a.ndim()` the same way `ionp_core::manip::insert` itself does (kept
/// in lock-step with that function's own `match axis` block so the N this
/// parses against and the N the core function bounds-checks against can
/// never diverge). Returns `(effective_axis, N)`.
fn insert_axis_len(a: &NdArray, axis: Option<isize>) -> PyResult<(usize, usize)> {
    match axis {
        None => Ok((0, a.size())),
        Some(ax) => {
            let n = manip::normalize_axis(ax, a.ndim()).map_err(to_py_err)?;
            Ok((n, a.shape()[n]))
        }
    }
}

fn slice_bounds(s: &Bound<'_, PySlice>) -> PyResult<(Option<isize>, Option<isize>, Option<isize>)> {
    let start: Option<isize> = s.getattr("start")?.extract()?;
    let stop: Option<isize> = s.getattr("stop")?.extract()?;
    let step: Option<isize> = s.getattr("step")?.extract()?;
    Ok((start, stop, step))
}

/// `np.insert`'s `obj` argument: slice / scalar (Python int or 0-d
/// int array) / 1-d boolean array (converted to `flatnonzero` positions,
/// matching numpy's own conversion) / 1-d integer array-or-list.
/// Translated directly from numpy's own `insert` source (see
/// `ionp-core/src/manip.rs::insert`'s doc comment for the exact source
/// excerpt this mirrors).
///
/// SATURATING-CAST DEFECT #4 FIX (2026-08-02): a floating-dtype `obj`
/// (NaN/inf/huge-magnitude/ordinary-fractional/exact-integral-valued --
/// numpy does NOT special-case 2.0 as accepted, measured) must never
/// reach the old unconditional `arr.cast_to(DType::I64)` below, because
/// that cast SATURATES (NaN->0, out-of-range->MIN/MAX) where real numpy
/// raises. `np.insert`'s own source (`_function_base_impl.py`) treats a
/// float `obj` as a SLICE bound, not a fancy index -- this is why
/// `insert`'s contract differs from `delete`'s (see that function's own
/// doc comment) and must be handled independently:
///   - `indices.size == 1` (a bare scalar, OR a single-element array/list
///     -- ndim doesn't matter, only element count): numpy computes
///     `index = indices.item()` (the raw, uncast float) and FIRST does
///     `if index < -N or index > N: raise IndexError(f"index {obj} is
///     out of bounds for axis {axis} with size {N}")` against the RAW
///     float, using the ORIGINAL `obj`'s `str()` in the message (not the
///     coerced index) -- this is why `inf`/`-inf`/magnitudes >= N always
///     hit IndexError (even NaN's comparisons are both False under IEEE,
///     so NaN never raises here) while every IN-BOUNDS float (including
///     exact integral values like `2.0`, and `nan`) falls through to
///     `slobj[axis] = slice(None, index)` immediately after, which is
///     where Python's own slicing machinery raises `TypeError: slice
///     indices must be integers or None or have an __index__ method` --
///     numpy never even attempts an int conversion for this path.
///   - `indices.size == 0`: numpy's own `elif indices.size == 0 and not
///     isinstance(obj, np.ndarray): indices = indices.astype(intp)` only
///     forces an int dtype (unconditional success) when the ORIGINAL
///     `obj` was NOT already an ndarray (a Python list/tuple) -- an
///     empty `np.array([])` (float64 by default) skips that cast and
///     falls all the way to the general fancy-index path below, which
///     dtype-checks and raises even though it touches zero indices
///     (measured: `insert(a, np.array([]), 99)` raises `IndexError:
///     arrays used as indices must be of integer (or boolean) type`,
///     while `insert(a, [], 99)` succeeds as a no-op).
///   - every other `indices.size` (>= 2, or == 0 but already an ndarray):
///     numpy's general multi-index path does `indices[indices < 0] += N`
///     (fine for any numeric dtype) but then `old_mask[indices] = False`
///     -- a fancy-index ASSIGNMENT, which numpy's core indexing rejects
///     for any non-integer/non-boolean dtype UNCONDITIONALLY, regardless
///     of the actual values (even an all-integral-valued float array like
///     `[1.0, 2.0]` is rejected) -- same `FLOAT_INDEX_MSG` text `delete`
///     also raises (see `FLOAT_INDEX_MSG`'s own doc comment for why
///     sharing that literal string, not control flow, is safe here).
fn parse_insert_obj(obj: &Bound<'_, PyAny>, a: &NdArray, axis: Option<isize>) -> PyResult<manip::InsertObj> {
    if let Ok(s) = obj.cast::<PySlice>() {
        let (start, stop, step) = slice_bounds(s)?;
        return Ok(manip::InsertObj::Slice { start, stop, step });
    }
    if is_plain_int(obj) {
        return Ok(manip::InsertObj::Scalar(obj.extract::<isize>()?));
    }
    let arr = extract_or_ingest_ndarray(obj)?;
    if arr.dtype() == DType::Bool {
        if arr.ndim() != 1 {
            return Err(PyValueError::new_err("boolean array argument obj to insert must be one dimensional"));
        }
        let mask = match arr.to_contiguous().buffer() {
            Buffer::Bool(v) => v.clone(),
            _ => unreachable!("dtype() == Bool"),
        };
        let idx: Vec<isize> = mask.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i as isize).collect();
        return Ok(manip::InsertObj::Indices(idx));
    }
    if arr.ndim() > 1 {
        return Err(PyValueError::new_err(
            "index array argument obj to insert must be one dimensional or scalar",
        ));
    }
    if arr.dtype().is_floating() {
        let size = arr.size();
        if size == 0 {
            if obj_is_ndarray_instance(obj)? {
                return Err(PyIndexError::new_err(FLOAT_INDEX_MSG));
            }
            return Ok(manip::InsertObj::Indices(Vec::new()));
        }
        if size == 1 {
            let f = f64_scalar_value(&arr);
            let (eff_axis, n) = insert_axis_len(a, axis)?;
            let nf = n as f64;
            if f < -nf || f > nf {
                let repr = obj.str()?.to_string();
                return Err(PyIndexError::new_err(format!(
                    "index {repr} is out of bounds for axis {eff_axis} with size {n}"
                )));
            }
            return Err(PyTypeError::new_err(
                "slice indices must be integers or None or have an __index__ method",
            ));
        }
        return Err(PyIndexError::new_err(FLOAT_INDEX_MSG));
    }
    if arr.ndim() == 0 {
        let ints = arr.cast_to(DType::I64);
        let v = match ints.buffer() {
            Buffer::I64(v) => v[0],
            _ => unreachable!("cast_to(DType::I64) always yields Buffer::I64"),
        };
        return Ok(manip::InsertObj::Scalar(v as isize));
    }
    let ints = arr.cast_to(DType::I64);
    let vals = match ints.to_contiguous().buffer() {
        Buffer::I64(v) => v.clone(),
        _ => unreachable!("cast_to(DType::I64) always yields Buffer::I64"),
    };
    Ok(manip::InsertObj::Indices(vals.into_iter().map(|v| v as isize).collect()))
}

/// `np.delete`'s `obj` argument: slice / scalar (Python int) / boolean
/// array (any ndim -- numpy compares `obj.shape != (N,)`, which the core
/// `delete`'s own length check against the flattened mask already
/// reproduces byte-for-byte for both a wrong-ndim AND a wrong-length
/// 1-d mask) / integer array-or-list (flattened row-major; numpy's own
/// `keep[obj,] = False` fancy-index assignment for a >1-d `obj` is not
/// reproduced exactly here -- a documented, narrow gap, since delete's
/// corpus/probe coverage overwhelmingly exercises scalar/1-d obj).
///
/// SATURATING-CAST DEFECT #4 FIX (2026-08-02): unlike `insert` (which
/// treats a scalar `obj` as a SLICE bound, see that function's own doc
/// comment), `np.delete`'s own source (`_function_base_impl.py`) treats
/// EVERY non-int, non-bool `obj` -- scalar or array, ANY ndim, ANY size
/// -- as a fancy INDEX ARRAY: `keep = ones(N, dtype=bool); keep[obj,] =
/// False`. Real numpy's fancy-index-assignment machinery dtype-checks
/// `obj` and rejects anything that isn't integer/boolean UNCONDITIONALLY
/// -- not by value, not by magnitude, not even by whether `obj` is
/// actually empty (an empty `np.array([])` is float64 by default and
/// still gets rejected touching zero elements). This makes `delete`
/// dramatically simpler than `insert`: there is no bounds check to race
/// against, no TypeError-vs-IndexError split, no per-magnitude tier --
/// EVERY floating `obj` (ordinary fractional, negative, zero, NaN, +-inf,
/// any magnitude, exact-integral-valued like `2.0` -- measured, none of
/// these are special-cased as accepted) raises the exact same
/// `FLOAT_INDEX_MSG` text, live-verified identical for `delete(a, 2.7)`
/// through `delete(a, [1.5, 2.5])` through `delete(a, np.float64(nan))`
/// through `delete(a2, 2.7, axis=0)` (the message never even mentions
/// axis/size, unlike `insert`'s). The ONE carve-out (shared with
/// `insert`, see `obj_is_ndarray_instance`'s doc comment): a truly empty
/// Python list/tuple `obj` (not already an ndarray) is force-cast to an
/// int dtype by numpy's own `astype(intp)` before this check ever runs,
/// so `delete(a, [])` is a genuine no-op while `delete(a, np.array([]))`
/// still raises.
fn parse_delete_obj(obj: &Bound<'_, PyAny>) -> PyResult<manip::DeleteObj> {
    if let Ok(s) = obj.cast::<PySlice>() {
        let (start, stop, step) = slice_bounds(s)?;
        return Ok(manip::DeleteObj::Slice { start, stop, step });
    }
    if is_plain_int(obj) {
        return Ok(manip::DeleteObj::Single(obj.extract::<isize>()?));
    }
    let arr = extract_or_ingest_ndarray(obj)?;
    if arr.dtype().is_floating() {
        if arr.size() == 0 && !obj_is_ndarray_instance(obj)? {
            return Ok(manip::DeleteObj::Indices(Vec::new()));
        }
        return Err(PyIndexError::new_err(FLOAT_INDEX_MSG));
    }
    if arr.dtype() == DType::Bool {
        let mask = match arr.to_contiguous().buffer() {
            Buffer::Bool(v) => v.clone(),
            _ => unreachable!("dtype() == Bool"),
        };
        return Ok(manip::DeleteObj::Bool(mask));
    }
    if arr.ndim() == 0 {
        let ints = arr.cast_to(DType::I64);
        let v = match ints.buffer() {
            Buffer::I64(v) => v[0],
            _ => unreachable!("cast_to(DType::I64) always yields Buffer::I64"),
        };
        return Ok(manip::DeleteObj::Single(v as isize));
    }
    let ints = arr.cast_to(DType::I64);
    let vals = match ints.to_contiguous().buffer() {
        Buffer::I64(v) => v.clone(),
        _ => unreachable!("cast_to(DType::I64) always yields Buffer::I64"),
    };
    Ok(manip::DeleteObj::Indices(vals.into_iter().map(|v| v as isize).collect()))
}

#[pyfunction]
#[pyo3(signature = (arr, obj, values, axis=None))]
fn insert(arr: &Bound<'_, PyAny>, obj: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<PyArray> {
    let a = extract_or_ingest_ndarray(arr)?;
    let iobj = parse_insert_obj(obj, &a, axis)?;
    let v = extract_or_ingest_ndarray(values)?;
    let inner = manip::insert(&a, &iobj, &v, axis).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (arr, obj, axis=None))]
fn delete(arr: &Bound<'_, PyAny>, obj: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<PyArray> {
    let a = extract_or_ingest_ndarray(arr)?;
    let dobj = parse_delete_obj(obj)?;
    let inner = manip::delete(&a, &dobj, axis).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

// ---------------------------------------------------------------------------
// rot90 / rollaxis
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (m, k=1, axes=(0, 1)))]
fn rot90(m: &Bound<'_, PyAny>, k: isize, axes: (isize, isize)) -> PyResult<PyArray> {
    let arr = extract_or_ingest_ndarray(m)?;
    let inner = manip::rot90(&arr, k, axes).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (a, axis, start=0))]
fn rollaxis(a: &Bound<'_, PyAny>, axis: isize, start: isize) -> PyResult<PyArray> {
    // numpy's own `rollaxis` starts with an unconditional strict `n =
    // a.ndim` -- no lenient asarray fallback at all, unlike
    // hsplit/dsplit/vsplit's own preliminary lenient check. A raw
    // list/tuple/range genuinely fails HERE under real numpy, for every
    // ndim uniformly (see require_shape_attr's block comment above).
    ndim_strict(a)?;
    let arr = extract_or_ingest_ndarray(a)?;
    let inner = manip::rollaxis(&arr, axis, start).map_err(|e| match e {
        manip::RollaxisError::Core(err) => to_py_err(err),
        manip::RollaxisError::BadStart(msg) => {
            Python::attach(|py| crate::errors::raise_axis_error_custom_msg(py, &msg))
                .unwrap_or_else(|_| PyValueError::new_err(msg.clone()))
        }
    })?;
    Ok(PyArray { inner })
}

// ---------------------------------------------------------------------------
// grid / index construction: diag_indices(_from), tril_indices,
// triu_indices, ix_, indices, meshgrid, ravel_multi_index, unravel_index,
// fill_diagonal
// ---------------------------------------------------------------------------

fn wrap_tuple(py: Python<'_>, parts: Vec<NdArray>) -> PyResult<Py<PyAny>> {
    let objs: Vec<Py<PyArray>> = parts.into_iter().map(|inner| Py::new(py, PyArray { inner })).collect::<PyResult<_>>()?;
    Ok(PyTuple::new(py, objs)?.into_any().unbind())
}

/// Same tuple-of-arrays wrapping as `wrap_tuple`, except each element is
/// individually demoted to a genuine numpy scalar when it collapsed to 0-d
/// -- for `unravel_index` only, whose per-element shape tracks its own
/// `indices` input (a scalar `int` index produces a tuple of numpy
/// scalars, e.g. `np.unravel_index(22, (7,6))` -> `(np.int64(3),
/// np.int64(4))`; an array `indices` input produces a tuple of arrays,
/// same shape as the input). This is exactly the Cluster A full-reduction
/// scalar contract (`numpy_scalar_from_0d`), applied per-tuple-element
/// instead of to a single top-level result. `diag_indices`/`tril_indices`/
/// `triu_indices`/`indices`/`mask_indices`/etc, this function's other
/// callers, never produce a 0-d part (always at least 1-d index arrays by
/// construction) so `wrap_tuple` itself is left untouched for them.
fn wrap_tuple_scalarize(py: Python<'_>, parts: Vec<NdArray>) -> PyResult<Py<PyAny>> {
    let objs: Vec<Py<PyAny>> = parts
        .into_iter()
        .map(|inner| {
            if inner.ndim() == 0 {
                crate::numpy_scalar_from_0d(py, &inner)
            } else {
                Ok(Py::new(py, PyArray { inner })?.into_any())
            }
        })
        .collect::<PyResult<_>>()?;
    Ok(PyTuple::new(py, objs)?.into_any().unbind())
}

/// `n`/`m` for `diag_indices`/`tril_indices`/`triu_indices`: real numpy's
/// own implementation never validates these as integers up front -- it
/// just feeds them straight into `np.arange`/`np.tri`, both of which
/// happily accept a bare Python float (with a `DeprecationWarning` numpy
/// itself emits, not an error; verified live against numpy 2.5.1:
/// `np.diag_indices(5.5)` succeeds, returning FLOAT64 index arrays
/// `[0., 1., 2., 3., 4., 5.]`, i.e. `arange(5.5)`'s own length/dtype).
/// `tril_indices`/`triu_indices` differ in dtype only because `np.tri`
/// picks an explicit integer dtype for its internal `arange` calls via
/// `_min_int` (always integer, regardless of whether `N`/`M` themselves
/// were float) -- so for THOSE two, a float `n`/`m` still bottoms out at
/// integer-valued, integer-dtyped output, with the same effective count
/// (`ceil` of the float bound, `arange`'s own length rule for a
/// start=0/step=1 range) as if `ceil(n)` had been passed as a plain int.
enum SizeArg {
    Int(isize),
    Float(f64),
}

fn parse_size_arg(obj: &Bound<'_, PyAny>) -> PyResult<SizeArg> {
    if let Ok(v) = obj.extract::<isize>() {
        return Ok(SizeArg::Int(v));
    }
    // Extracting isize first is deliberate: an `anionpy`/numpy 0-d integer
    // scalar or a Python `bool` must stay on the integer path (`bool`
    // extracts as isize via PyO3's own int coercion), only a genuine
    // Python `float` (or a float-valued numpy scalar) falls through here.
    let v: f64 = obj.extract()?;
    Ok(SizeArg::Float(v))
}

/// `arange(bound)`'s own length for a `start=0, step=1` range: `ceil(bound)`
/// clamped to non-negative, matching `np.arange`'s general
/// `ceil((stop-start)/step)` length rule at `start=0, step=1`.
///
/// Measured against real numpy 2.5.1 (`np.arange(bound)`, and transitively
/// `np.diag_indices`/`np.tril_indices`/`np.triu_indices`, which all bottom
/// out in `arange`'s own length computation -- confirmed by reading
/// `numpy/_core/numeric.py`'s `tri`, which calls `arange(N, ...)` on the raw
/// float `N` with no upfront range check of its own): a bound that is NOT
/// finite, or whose magnitude does not fit in `isize` (Rust's analogue of
/// numpy's `npy_intp`/C `Py_ssize_t`), NEVER silently produces a value --
/// numpy always raises. Previously this function used `bound.ceil() as
/// usize`, a SATURATING cast: `NaN` silently became `0` (wrong answer, no
/// error) and any out-of-range magnitude silently saturated to
/// `usize::MAX`/`0`, which downstream (`diag_indices`'s float branch) could
/// even panic via an `NdArray::from_buffer(...).unwrap()` shape/length
/// mismatch. This now returns `Err` instead, with numpy's own exact
/// message text on the two axes that are actually deterministic
/// (independent of available system memory):
///   - `NaN` -> `"arange: cannot compute length"`.
///   - `|bound|` beyond `isize` range (`bound >= 2**63` or
///     `bound < -(2**63)`) -> `"Maximum allowed size exceeded"`, EXCEPT the
///     single exact float64 value `2**63` itself, which numpy reports as
///     `"array is too big; \`arr.size * arr.dtype.itemsize\` is larger than
///     the maximum possible size."` (an artifact of `isize::MAX == 2**63-1`
///     not being exactly representable as `f64`, so numpy's own raw-double
///     range check and its separate byte-size check land on the same
///     double from opposite sides).
/// A THIRD numpy error tier exists for large-but-`isize`-representable
/// finite bounds (e.g. `1e18`): numpy attempts a real allocation and raises
/// either its own `"array is too big"` `ValueError` (size*itemsize
/// overflow) or a system-memory-dependent `MemoryError`. That tier is NOT
/// reproduced here -- it is not part of the saturating-cast defect (the
/// cast itself is exact for those bounds; only the downstream allocation is
/// enormous), and unlike the two axes above it is not deterministic across
/// machines/memory states, so there is no single "numpy value" to match.
fn arange_len_from_zero(bound: f64) -> PyResult<usize> {
    const TWO_POW_63: f64 = 9223372036854775808.0; // 2**63, exact in f64
    if bound.is_nan() {
        return Err(PyValueError::new_err("arange: cannot compute length"));
    }
    if bound >= TWO_POW_63 {
        return Err(PyValueError::new_err(if bound == TWO_POW_63 {
            "array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."
        } else {
            "Maximum allowed size exceeded"
        }));
    }
    if bound < -TWO_POW_63 {
        return Err(PyValueError::new_err("Maximum allowed size exceeded"));
    }
    Ok(if bound <= 0.0 { 0 } else { bound.ceil() as usize })
}

#[pyfunction]
#[pyo3(signature = (n, ndim=2))]
fn diag_indices(py: Python<'_>, n: &Bound<'_, PyAny>, ndim: usize) -> PyResult<Py<PyAny>> {
    match parse_size_arg(n)? {
        SizeArg::Int(n) => wrap_tuple(py, manip::diag_indices(n, ndim)),
        SizeArg::Float(n) => {
            let count = arange_len_from_zero(n)?;
            let idx: Vec<f64> = (0..count as i64).map(|v| v as f64).collect();
            let parts: Vec<NdArray> = (0..ndim)
                .map(|_| NdArray::from_buffer(Buffer::F64(idx.clone()), vec![count], ionp_core::Order::C).unwrap())
                .collect();
            wrap_tuple(py, parts)
        }
    }
}

#[pyfunction]
#[pyo3(signature = (arr))]
fn diag_indices_from(py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let a = extract_or_ingest_ndarray(arr)?;
    let parts = manip::diag_indices_from(&a).map_err(to_py_err)?;
    wrap_tuple(py, parts)
}

/// A possibly-float `n`/`m` reduced to the isize `ceil` `tri_indices`'s
/// existing integer-only core needs -- see `SizeArg`'s doc comment above for
/// why this is exact (not an approximation) for `tril_indices`/
/// `triu_indices` specifically.
fn size_arg_to_ceil_isize(obj: &Bound<'_, PyAny>) -> PyResult<isize> {
    match parse_size_arg(obj)? {
        SizeArg::Int(v) => Ok(v),
        SizeArg::Float(v) => Ok(arange_len_from_zero(v)? as isize),
    }
}

#[pyfunction]
#[pyo3(signature = (n, k=0, m=None))]
fn tril_indices(py: Python<'_>, n: &Bound<'_, PyAny>, k: isize, m: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let n = size_arg_to_ceil_isize(n)?;
    let m = m.map(size_arg_to_ceil_isize).transpose()?;
    let (r, c) = manip::tril_indices(n, k, m);
    wrap_tuple(py, vec![r, c])
}

#[pyfunction]
#[pyo3(signature = (n, k=0, m=None))]
fn triu_indices(py: Python<'_>, n: &Bound<'_, PyAny>, k: isize, m: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let n = size_arg_to_ceil_isize(n)?;
    let m = m.map(size_arg_to_ceil_isize).transpose()?;
    let (r, c) = manip::triu_indices(n, k, m);
    wrap_tuple(py, vec![r, c])
}

#[pyfunction]
#[pyo3(signature = (*args))]
fn ix_(py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let arrays: Vec<NdArray> =
        args.iter().map(|obj| extract_or_ingest_ndarray(&obj)).collect::<PyResult<_>>()?;
    let parts = manip::ix_(&arrays).map_err(to_py_err)?;
    wrap_tuple(py, parts)
}

#[pyfunction]
#[pyo3(signature = (dimensions, dtype=None, sparse=None))]
fn indices(
    py: Python<'_>,
    dimensions: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    sparse: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Plain Python truthiness (measured 2026-08-02, same as `svd`).
    let sparse = match sparse {
        None => false,
        Some(v) => v.is_truthy()?,
    };
    let dims = usize_list_from_pyobj(dimensions)?;
    let dt = match dtype {
        // `manip::indices` unconditionally `cast_to`s an I64 coordinate
        // buffer into this target -- no S/U arm, decline cleanly.
        Some(d) => crate::dtype_from_pyobj_no_su(d)?,
        None => DType::I64,
    };
    let parts = manip::indices(&dims, dt, sparse).map_err(to_py_err)?;
    if sparse {
        wrap_tuple(py, parts)
    } else {
        Ok(Py::new(py, PyArray { inner: parts.into_iter().next().unwrap() })?.into_any())
    }
}

#[pyfunction]
#[pyo3(signature = (*xi, copy=None, sparse=None, indexing="xy".to_string()))]
fn meshgrid(
    py: Python<'_>,
    xi: &Bound<'_, PyTuple>,
    copy: Option<&Bound<'_, PyAny>>,
    sparse: Option<&Bound<'_, PyAny>>,
    indexing: String,
) -> PyResult<Py<PyAny>> {
    // Plain Python truthiness for both (measured 2026-08-02). `copy`
    // still has no observable value-level effect (see doc reference
    // below) but must still ACCEPT the same values numpy accepts and
    // propagate a raising `__bool__` the same way.
    let copy = match copy {
        None => true,
        Some(v) => v.is_truthy()?,
    };
    let sparse = match sparse {
        None => false,
        Some(v) => v.is_truthy()?,
    };
    let _ = copy; // accepted, no observable value-level effect -- see manip::meshgrid docs
    let arrays: Vec<NdArray> =
        xi.iter().map(|obj| extract_or_ingest_ndarray(&obj)).collect::<PyResult<_>>()?;
    let refs: Vec<&NdArray> = arrays.iter().collect();
    let parts = manip::meshgrid(&refs, &indexing, sparse).map_err(to_py_err)?;
    wrap_tuple(py, parts)
}

fn ravel_mode_from_pyobj(obj: &Bound<'_, PyAny>, ndim: usize) -> PyResult<Vec<manip::RavelMode>> {
    fn one(s: &str) -> PyResult<manip::RavelMode> {
        match s {
            "raise" => Ok(manip::RavelMode::Raise),
            "wrap" => Ok(manip::RavelMode::Wrap),
            "clip" => Ok(manip::RavelMode::Clip),
            other => Err(PyTypeError::new_err(format!("Unrecognized mode: {other}"))),
        }
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(vec![one(&s)?; ndim]);
    }
    let strs: Vec<String> = obj.extract()?;
    strs.iter().map(|s| one(s)).collect()
}

#[pyfunction]
#[pyo3(signature = (multi_index, dims, mode=None, order="C".to_string()))]
fn ravel_multi_index(
    multi_index: &Bound<'_, PyAny>,
    dims: &Bound<'_, PyAny>,
    mode: Option<&Bound<'_, PyAny>>,
    order: String,
) -> PyResult<PyArray> {
    let dim_vec = usize_list_from_pyobj(dims)?;
    let entries = extract_ndarray_seq(multi_index)?;
    let modes = match mode {
        Some(m) => ravel_mode_from_pyobj(m, dim_vec.len())?,
        None => vec![manip::RavelMode::Raise; dim_vec.len()],
    };
    let ord = match order.as_str() {
        "C" => ionp_core::Order::C,
        "F" => ionp_core::Order::F,
        other => return Err(PyValueError::new_err(format!("order not understood: {other}"))),
    };
    let inner = manip::ravel_multi_index(&entries, &dim_vec, &modes, ord).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (indices, shape, order="C".to_string()))]
fn unravel_index(py: Python<'_>, indices: &Bound<'_, PyAny>, shape: &Bound<'_, PyAny>, order: String) -> PyResult<Py<PyAny>> {
    let dim_vec = usize_list_from_pyobj(shape)?;
    let arr = extract_or_ingest_ndarray(indices)?;
    let ord = match order.as_str() {
        "C" => ionp_core::Order::C,
        "F" => ionp_core::Order::F,
        other => return Err(PyValueError::new_err(format!("order not understood: {other}"))),
    };
    let parts = manip::unravel_index(&arr, &dim_vec, ord).map_err(to_py_err)?;
    wrap_tuple_scalarize(py, parts)
}

#[pyfunction]
#[pyo3(signature = (a, val, wrap=None))]
fn fill_diagonal(a: &Bound<'_, PyAny>, val: &Bound<'_, PyAny>, wrap: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    // Plain Python truthiness (measured 2026-08-02, same as `svd`).
    let wrap = match wrap {
        None => false,
        Some(v) => v.is_truthy()?,
    };
    // numpy's own `fill_diagonal` mutates its first argument's storage in
    // place and returns `None` -- a non-`anionpy.ndarray` first argument (a
    // foreign numpy array, a list, ...) does not share storage with
    // anything anionpy can hand back to the caller, so (mirroring `.at()`'s
    // own `cast::<PyArray>()` gate above) only anionpy's own array type is
    // accepted here.
    let target = a
        .cast::<PyArray>()
        .map_err(|_| PyTypeError::new_err("first argument must be an anionpy.ndarray"))?;
    let val_arr = extract_or_ingest_ndarray(val)?;
    let mut target_ref = target.borrow_mut();
    manip::fill_diagonal(&mut target_ref.inner, &val_arr, wrap).map_err(to_py_err)
}

// ---------------------------------------------------------------------------
// take / put / take_along_axis / put_along_axis / compress
//
// `take`/`take_along_axis`/`compress` are pure (accept any array-like via
// `extract_or_ingest_ndarray`, return a fresh `anionpy.ndarray`); `put`/
// `put_along_axis` mutate their first argument's storage in place and
// return `None`, so -- exactly like `fill_diagonal` above -- only anionpy's
// own array type is accepted there (a foreign numpy array/list has no
// storage anionpy could mutate through).
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (a, indices, axis=None, out=None, mode="raise".to_string()))]
fn take(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    axis: Option<isize>,
    out: Option<&Bound<'_, PyAny>>,
    mode: String,
) -> PyResult<Py<PyAny>> {
    let arr = extract_or_ingest_ndarray(a)?;
    let idx = ingest_index_ndarray(indices)?;
    let clip_mode = manip::ClipMode::parse(&mode).map_err(to_py_err)?;
    // numpy's real `take` promotes a 0-d array to shape `(1,)` before
    // resolving `axis` -- verified live, see `manip::take`'s own doc
    // comment on its 0-d branch for the exact behavior this mirrors.
    let axis_ndim = if arr.ndim() == 0 { 1 } else { arr.ndim() };
    let ax = match axis {
        Some(v) => Some(manip::normalize_axis(v, axis_ndim).map_err(to_py_err)?),
        None => None,
    };
    let result = manip::take(&arr, &idx, ax, clip_mode).map_err(to_py_err)?;
    if let Some(out_obj) = out {
        let out_bound = out_obj
            .cast::<PyArray>()
            .map_err(|_| PyTypeError::new_err("output must be an array"))?;
        {
            let mut out_ref = out_bound.borrow_mut();
            manip::write_exact_shape_out(&mut out_ref.inner, &result).map_err(to_py_err)?;
        }
        return Ok(out_obj.clone().unbind());
    }
    // A 0-d result collapses to a numpy SCALAR, exactly as everywhere else in
    // this library (see `numpy_scalar_from_0d`'s doc comment for why the
    // wrapper type has to be numpy's own). This is reachable whenever the
    // index argument is itself a scalar and the base is 1-d -- verified live
    // on numpy 2.5.1:
    //     np.arange(4, dtype=np.int8).take(1)            -> np.int8
    //     np.arange(4, dtype=np.int8).take(np.array(1))  -> np.int8
    //     np.arange(4, dtype=np.int8).take([1])          -> ndarray, shape (1,)
    //     np.arange(24).reshape(2,3,4).take(1, axis=0)   -> ndarray (3, 4)
    // The `out=` path above deliberately returns BEFORE this: with `out=`
    // given, numpy hands back the `out` ARRAY even when it is 0-d
    // (`a.take(1, out=np.zeros((), np.int8))` is an ndarray, not a scalar).
    if result.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &result);
    }
    Ok(Py::new(py, PyArray { inner: result })?.into_any())
}

#[pyfunction]
#[pyo3(signature = (a, ind, v, mode="raise".to_string()))]
fn put(a: &Bound<'_, PyAny>, ind: &Bound<'_, PyAny>, v: &Bound<'_, PyAny>, mode: String) -> PyResult<()> {
    let target = a
        .cast::<PyArray>()
        .map_err(|_| PyTypeError::new_err("first argument must be an anionpy.ndarray"))?;
    let idx = extract_or_ingest_ndarray(ind)?;
    let vals = extract_or_ingest_ndarray(v)?;
    let clip_mode = manip::ClipMode::parse(&mode).map_err(to_py_err)?;
    let mut target_ref = target.borrow_mut();
    manip::put(&mut target_ref.inner, &idx, &vals, clip_mode).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (arr, indices, axis=-1))]
fn take_along_axis(arr: &Bound<'_, PyAny>, indices: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<PyArray> {
    let a = extract_or_ingest_ndarray(arr)?;
    let idx = extract_or_ingest_ndarray(indices)?;
    let ax = match axis {
        Some(v) => Some(manip::normalize_axis(v, a.ndim()).map_err(to_py_err)?),
        None => None,
    };
    let inner = manip::take_along_axis(&a, &idx, ax).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (arr, indices, values, axis))]
fn put_along_axis(
    arr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    axis: Option<isize>,
) -> PyResult<()> {
    let target = arr
        .cast::<PyArray>()
        .map_err(|_| PyTypeError::new_err("first argument must be an anionpy.ndarray"))?;
    let idx = extract_or_ingest_ndarray(indices)?;
    let vals = extract_or_ingest_ndarray(values)?;
    let ndim = target.borrow().inner.ndim();
    let ax = match axis {
        Some(v) => Some(manip::normalize_axis(v, ndim).map_err(to_py_err)?),
        None => None,
    };
    let mut target_ref = target.borrow_mut();
    manip::put_along_axis(&mut target_ref.inner, &idx, &vals, ax).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(signature = (condition, a, axis=None, out=None))]
fn compress(
    py: Python<'_>,
    condition: &Bound<'_, PyAny>,
    a: &Bound<'_, PyAny>,
    axis: Option<isize>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let cond = extract_or_ingest_ndarray(condition)?;
    let arr = extract_or_ingest_ndarray(a)?;
    // `compress` is implemented as a `take` under the hood in real numpy
    // (see `manip::compress`'s doc comment) and inherits the same 0-d ->
    // shape-`(1,)` axis-promotion courtesy -- verified live.
    let axis_ndim = if arr.ndim() == 0 { 1 } else { arr.ndim() };
    let ax = match axis {
        Some(v) => Some(manip::normalize_axis(v, axis_ndim).map_err(to_py_err)?),
        None => None,
    };
    let result = manip::compress(&cond, &arr, ax).map_err(to_py_err)?;
    if let Some(out_obj) = out {
        let out_bound = out_obj
            .cast::<PyArray>()
            .map_err(|_| PyTypeError::new_err("output must be an array"))?;
        {
            let mut out_ref = out_bound.borrow_mut();
            manip::write_exact_shape_out(&mut out_ref.inner, &result).map_err(to_py_err)?;
        }
        return Ok(out_obj.clone().unbind());
    }
    Ok(Py::new(py, PyArray { inner: result })?.into_any())
}

// ---------------------------------------------------------------------------
// Registration: one call from lib.rs's #[pymodule] fn, mirroring
// creation::register's existing pattern.
// ---------------------------------------------------------------------------

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(concatenate, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(hstack, m)?)?;
    m.add_function(wrap_pyfunction!(vstack, m)?)?;
    m.add_function(wrap_pyfunction!(row_stack, m)?)?;
    m.add_function(wrap_pyfunction!(dstack, m)?)?;
    m.add_function(wrap_pyfunction!(column_stack, m)?)?;
    m.add_function(wrap_pyfunction!(flip, m)?)?;
    m.add_function(wrap_pyfunction!(fliplr, m)?)?;
    m.add_function(wrap_pyfunction!(flipud, m)?)?;
    m.add_function(wrap_pyfunction!(roll, m)?)?;
    m.add_function(wrap_pyfunction!(tile, m)?)?;
    m.add_function(wrap_pyfunction!(repeat, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast_shapes, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(atleast_1d, m)?)?;
    m.add_function(wrap_pyfunction!(atleast_2d, m)?)?;
    m.add_function(wrap_pyfunction!(atleast_3d, m)?)?;
    m.add_function(wrap_pyfunction!(diag, m)?)?;
    m.add_function(wrap_pyfunction!(diagflat, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(tril, m)?)?;
    m.add_function(wrap_pyfunction!(triu, m)?)?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(array_split, m)?)?;
    m.add_function(wrap_pyfunction!(hsplit, m)?)?;
    m.add_function(wrap_pyfunction!(vsplit, m)?)?;
    m.add_function(wrap_pyfunction!(dsplit, m)?)?;
    m.add_function(wrap_pyfunction!(append, m)?)?;
    m.add_function(wrap_pyfunction!(resize, m)?)?;
    m.add_function(wrap_pyfunction!(insert, m)?)?;
    m.add_function(wrap_pyfunction!(delete, m)?)?;
    m.add_function(wrap_pyfunction!(rot90, m)?)?;
    m.add_function(wrap_pyfunction!(rollaxis, m)?)?;
    m.add_function(wrap_pyfunction!(diag_indices, m)?)?;
    m.add_function(wrap_pyfunction!(diag_indices_from, m)?)?;
    m.add_function(wrap_pyfunction!(tril_indices, m)?)?;
    m.add_function(wrap_pyfunction!(triu_indices, m)?)?;
    m.add_function(wrap_pyfunction!(ix_, m)?)?;
    m.add_function(wrap_pyfunction!(indices, m)?)?;
    m.add_function(wrap_pyfunction!(meshgrid, m)?)?;
    m.add_function(wrap_pyfunction!(ravel_multi_index, m)?)?;
    m.add_function(wrap_pyfunction!(unravel_index, m)?)?;
    m.add_function(wrap_pyfunction!(fill_diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(take, m)?)?;
    m.add_function(wrap_pyfunction!(put, m)?)?;
    m.add_function(wrap_pyfunction!(take_along_axis, m)?)?;
    m.add_function(wrap_pyfunction!(put_along_axis, m)?)?;
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    Ok(())
}
