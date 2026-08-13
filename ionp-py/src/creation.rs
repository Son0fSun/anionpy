//! Top-level array-creation and shape-manipulation functions
//! (`anionpy.zeros`/`ones`/`empty`/`full` + `_like` variants, `arange`,
//! `linspace`, `eye`, `identity`, `asarray`/`copy`/`ascontiguousarray`, and
//! the top-level view wrappers `reshape`/`ravel`/`transpose`/`swapaxes`/
//! `moveaxis`/`squeeze`/`expand_dims`/`broadcast_to`).
//!
//! This file only parses Python arguments (shape tuples, dtype objects,
//! scalar fill values) and dispatches into `ionp_core::creation` and
//! `ionp_core::array::NdArray`'s existing view methods -- every actual
//! numeric loop (buffer allocation, arange's `start + i*step`, linspace's
//! formula, eye's diagonal, view stride math) lives in `ionp-core`, per
//! the crate's "Rust does arithmetic, Python does dispatch" rule (see
//! `ionp_core::creation`'s module docs for where each computation lives).
//!
//! Reuses `crate::{dtype_from_pyobj, shape_args_to_isize_vec,
//! classify_scalar, weak_scalar_buffer, to_py_err, PyArray}` from the
//! crate root (`lib.rs`) rather than duplicating any of that parsing/
//! marshaling logic -- those items are private (not `pub`) but visible
//! here because this module is a child of the crate root, which is
//! ordinary Rust module-privacy, not a visibility change to `lib.rs`.

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList, PySequence, PyTuple};

use ionp_core::{array::Order, creation, Buffer, DType, IonpError, NdArray, ScalarKind};

use crate::{
    axis_error_prefixed, check_device, check_int_bounds, check_like, check_subok, classify_scalar,
    default_int_list_dtype, dtype_from_pyobj, ndarray_from_numpy, ndarray_from_pylist_typed, to_py_err,
    weak_scalar_buffer, wrap_shape_view, PyArray,
};

// ---------------------------------------------------------------------------
// Argument-parsing helpers, local to this module.
// ---------------------------------------------------------------------------

/// numpy accepts a bare non-negative int (`zeros(5)`) or a sequence of
/// non-negative ints (`zeros((2, 3))`, `zeros([2, 3])`) as a `shape`
/// argument, and raises `ValueError: negative dimensions are not allowed`
/// for a negative entry.
///
/// BUG FOUND AND FIXED 2026-08-01 (harness now compares exception
/// MESSAGES, not just types -- f69aceb -- which caught this): a plain
/// non-sequence, non-int object used to always get the generic `TypeError:
/// expected a sequence of integers or a single integer, got '...'`
/// message, but real numpy's algorithm is closer to CPython's own
/// `operator.index()` machinery and gives three DIFFERENT messages
/// depending on exactly what's wrong (verified against numpy 2.5.1):
///   - `None` -> `TypeError: Use () not None as shape arguments` (a
///     hand-written special case, not from `operator.index`).
///   - the object itself, or any element of it if it IS a genuine sequence
///     (real `PySequence` protocol -- a `dict` does NOT qualify even
///     though it's iterable, confirmed: `np.zeros({})` gives the generic
///     message below, not a per-element one), fails `operator.index()`
///     (not an int, and `bool` is explicitly excluded even though it's an
///     int subclass -- `np.zeros(True)` also gives the generic message) ->
///     `TypeError: '{type name}' object cannot be interpreted as an
///     integer` (CPython's own `operator.index()` wording, naming
///     whichever type -- the whole argument's own type for a bad scalar,
///     or the failing ELEMENT's type for a bad entry inside an otherwise
///     valid sequence, e.g. `np.zeros([1, 'a'])` names `str`, not `list`).
///   - anything else not covered above (a bad scalar that also isn't a
///     sequence, e.g. `5.5`, `True`, `np.float64(5.5)`, `{}`, an arbitrary
///     object) -> the original generic `expected a sequence of integers or
///     a single integer, got '...'` message.
fn shape_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    fn is_valid_index_int(item: &Bound<'_, PyAny>) -> Option<i64> {
        if item.is_instance_of::<PyBool>() {
            return None;
        }
        item.extract::<i64>().ok()
    }
    fn not_an_integer(item: &Bound<'_, PyAny>) -> PyErr {
        let tn = item
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "object".to_string());
        PyTypeError::new_err(format!("'{tn}' object cannot be interpreted as an integer"))
    }
    let dims: Vec<i64> = if obj.is_none() {
        return Err(PyTypeError::new_err("Use () not None as shape arguments"));
    } else if let Some(n) = is_valid_index_int(obj) {
        vec![n]
    } else if let Ok(seq) = obj.cast::<PySequence>() {
        // A genuine sequence (list/tuple/str/ndarray/... -- NOT a dict,
        // which fails this downcast despite being iterable): each element
        // must itself satisfy `operator.index()`.
        let mut out: Vec<i64> = Vec::with_capacity(seq.len().unwrap_or(0));
        for i in 0..seq.len().map_err(|_| not_an_integer(obj))? {
            let item = seq.get_item(i)?;
            match is_valid_index_int(&item) {
                Some(v) => out.push(v),
                None => return Err(not_an_integer(&item)),
            }
        }
        out
    } else {
        return Err(PyTypeError::new_err(format!(
            "expected a sequence of integers or a single integer, got '{}'",
            obj.repr().map(|r| r.to_string()).unwrap_or_default()
        )));
    };
    for &d in &dims {
        if d < 0 {
            return Err(PyValueError::new_err("negative dimensions are not allowed"));
        }
    }
    Ok(dims.into_iter().map(|d| d as usize).collect())
}

/// `zeros`/`ones`/`empty`/`full`'s own `order=` -- unlike the `_like`
/// family (`like_strides` above) or `ravel`/`flatten`/`copy`/`astype`
/// (`array.rs`'s `axis_perm_for_order`), there is no prototype array here
/// to make `'A'`/`'K'` mean anything, so numpy refuses them outright
/// (verified against real numpy 2.5.1: `np.zeros((2,3), order='A')`
/// raises `ValueError: only 'C' or 'F' order is permitted`, and a letter
/// outside `'C'`/`'F'`/`'A'`/`'K'` gets the general four-letter message
/// instead).
fn order_from_pyobj(order: Option<&str>) -> PyResult<Order> {
    match order {
        None | Some("C") => Ok(Order::C),
        Some("F") => Ok(Order::F),
        Some("A") | Some("K") => Err(PyValueError::new_err(
            "only 'C' or 'F' order is permitted",
        )),
        Some(other) => Err(PyValueError::new_err(format!(
            "order must be one of 'C', 'F', 'A', or 'K' (got '{other}')"
        ))),
    }
}

fn extract_pyarray<'py>(obj: &Bound<'py, PyAny>) -> PyResult<PyRef<'py, PyArray>> {
    Ok(obj.extract::<PyRef<'py, PyArray>>()?)
}

/// Like `extract_pyarray`, but also accepts a genuine external
/// `numpy.ndarray` (or a Python list/tuple), ingesting it via the same
/// `array_impl` path `array()`/`asarray()` use, instead of requiring an
/// already-`anionpy.ndarray` input.
///
/// Added 2026-08-01, fixing a real divergence found by out-of-corpus
/// probing (not by the differential corpus, which always hands these
/// functions an already-ingested `anionpy.ndarray` via
/// `make_ionp_array_converter`, so the gap never showed up there):
/// `zeros_like`/`ones_like`/`empty_like`/`full_like`/`copy`/
/// `ascontiguousarray`/`broadcast_to` (and, discovered while fixing this,
/// every other top-level view wrapper in this file --
/// `reshape`/`ravel`/`transpose`/`swapaxes`/`moveaxis`/`squeeze`/
/// `expand_dims`) previously called `extract_pyarray`, which raises
/// PyO3's own "'ndarray' object is not an instance of 'ndarray'" TypeError
/// for ANY raw `numpy.ndarray` argument -- regardless of `subok`'s value,
/// which is a red herring `check_subok` never touches (`check_subok` is
/// already an intentional no-op). Real numpy accepts a plain
/// `numpy.ndarray` for every one of these; this closes that gap for the
/// three the coordinator explicitly flagged as probed
/// (`zeros_like`/`copy`/`broadcast_to`). The remaining functions in this
/// list have the identical defect (confirmed directly) but are reported,
/// not silently fixed here -- out of the scope the coordinator authorized
/// for this pass (see this task's final report).
fn extract_or_ingest_ndarray(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(pyref.inner.clone());
    }
    crate::array_impl(obj, None)
}

/// The `_like` family's default dtype: the prototype array's own dtype,
/// overridable by an explicit `dtype=` kwarg -- matches
/// `zeros_like`/`ones_like`/`empty_like`/`full_like` exactly.
fn resolve_dtype(explicit: Option<&Bound<'_, PyAny>>, default: DType) -> PyResult<DType> {
    match explicit {
        Some(d) => dtype_from_pyobj(d),
        None => Ok(default),
    }
}

/// A bare Python scalar's default numpy dtype when no `dtype=` kwarg is
/// given (`np.array(5).dtype == int64`, `np.array(5.0).dtype == float64`,
/// `np.array(True).dtype == bool`, `np.array(5j).dtype == complex128`,
/// each verified against real numpy 2.5.1) -- used by `full`'s no-dtype
/// path.
///
/// SEV-1 fix (2026-08-02): the `Int` arm used to unconditionally return
/// `DType::I64`, so `full(1, 2**64-1)` (a value that fits `u64` but not
/// `i64`) raised a spurious `OverflowError: Python integer ... out of
/// bounds for int64` where real numpy correctly infers `dtype=uint64`
/// (verified: `np.full(1, 2**64-1).dtype == np.uint64`). Fixed by reusing
/// `crate::default_int_list_dtype` -- the SAME int64-else-uint64-else-raise
/// rule `array()`'s own untyped-list inference already implements at full
/// `i128` precision -- fed the single scalar as a one-element slice, rather
/// than writing a third copy of that rule here.
fn default_dtype_for_scalar(kind: ScalarKind, int_value: Option<i128>) -> PyResult<DType> {
    Ok(match kind {
        ScalarKind::Bool => DType::Bool,
        ScalarKind::Int => default_int_list_dtype(&[int_value.expect(
            "ScalarKind::Int must carry its i128 value",
        )])?
        .expect("a single-element slice can never hit the mixed i64/u64 float64 fallback"),
        ScalarKind::Float => DType::F64,
        ScalarKind::Complex => DType::C128,
    })
}

/// `fill_value` also accepts a numpy scalar (`np.int64(300)`) or a 0-d
/// `numpy.ndarray`/`anionpy.ndarray` -- not just a bare Python bool/int/float/
/// complex -- per real numpy (`np.full((2,2), np.int64(300), dtype=np.uint8)`
/// -> wraps to `44`, `np.full((3,), np.float64(2.5))` -> `[2.5, 2.5, 2.5]`,
/// both verified against numpy 2.5.1). This used to go straight to
/// `classify_scalar`, which only recognizes bare Python scalar types (by
/// design -- see `coerce_operand`'s doc comment: numpy scalars are NEP-50
/// "strong", weak Python scalars are not, and conflating them would corrupt
/// binary-op promotion), and so raised `TypeError: full()'s fill_value must
/// be a bool/int/float/complex scalar` for any numpy scalar input -- found
/// via this task's foreign-numpy-input probe. Fixed the same way
/// `coerce_operand` distinguishes the two: anything with a `.dtype`
/// attribute (a numpy scalar, 0-d numpy array, or another `anionpy.ndarray`) is
/// routed through the array path and cast directly to `target`; only a bare
/// Python scalar falls through to the weak-scalar path.
/// A `full`/`full_like` fill_value, resolved to `target` dtype, in one of
/// two forms depending on how it was recognized:
///
/// - `ScalarLiteral`: a bare Python scalar (`5`, `2.5`, `True`, ...), gone
///   through `weak_scalar_buffer`'s NEP-50-weak narrowing/saturation rules
///   exactly as before this enum existed -- that logic is delicate
///   (verified complex-to-non-complex and narrow-int-cast-saturation
///   behavior) and this variant's ONLY consumer is the original
///   `repeat_scalar_buffer` fast path, byte-for-byte unchanged.
/// - `Array`: anything else that produced an actual `NdArray` -- a numpy
///   scalar/array, an `anionpy.ndarray`, or (new) a Python list/tuple/nested
///   sequence ingested via `crate::array_impl`. May have any number of
///   elements, including 0 or 1; the caller is responsible for
///   broadcasting it to the output shape (see `broadcast_or_err`) rather
///   than assuming a single repeated value.
enum FillMaterial {
    ScalarLiteral(Buffer),
    Array(NdArray),
}

/// Bug found via this task's array-like-fill_value probe: `np.full((2,2),
/// [[2.,1.],[0.,3.]])` and `np.full((2,2), [7.,8.])` both broadcast the
/// fill_value against `shape` in real numpy, but the old `fill_buffer`
/// (see git history) had no code path for a bare Python list/tuple at
/// all -- it fell straight to `classify_scalar`, which only recognizes
/// scalar types, so any list/tuple fill_value raised `full()'s fill_value
/// must be a bool/int/float/complex scalar` even where numpy broadcasts
/// happily. Fixed by adding a final fallback through `crate::array_impl`
/// (the same ingestion `anionpy.array`/`anionpy.asarray` use for arbitrary
/// Python sequences) before giving up with the original TypeError -- this
/// keeps the PyArray/numpy-scalar/bare-Python-scalar branches completely
/// unchanged (avoiding any regression risk to the already-verified
/// complex/narrow-int scalar-cast rules in `weak_scalar_buffer`) and only
/// adds a NEW path for inputs none of those three recognized.
fn fill_material(fill_value: &Bound<'_, PyAny>, target: DType) -> PyResult<FillMaterial> {
    if let Ok(other) = fill_value.extract::<PyRef<'_, PyArray>>() {
        return Ok(FillMaterial::Array(other.inner.cast_to(target)));
    }
    if fill_value.hasattr("dtype")? {
        let py = fill_value.py();
        let np = PyModule::import(py, "numpy")?;
        let as_array = np.getattr("asarray")?.call1((fill_value,))?;
        let nd = ndarray_from_numpy(&as_array)?;
        return Ok(FillMaterial::Array(nd.cast_to(target)));
    }
    if let Some(kind) = classify_scalar(fill_value) {
        return Ok(FillMaterial::ScalarLiteral(weak_scalar_buffer(
            fill_value, kind, target,
        )?));
    }
    // SEV-1-adjacent fix (2026-08-02): found via this task's own out-of-
    // corpus `full` sweep, NOT one of the three declared defects, but the
    // identical root cause. Verified against real numpy 2.5.1 first:
    // `np.full((2,), [256, 256], dtype=np.uint8)` -> `[0, 0]` (silent
    // wraparound, NOT `OverflowError`) -- unlike `array()`/`asarray()`,
    // `full`'s array-like fill_value is never construction-time
    // bounds-checked against an explicit integer `dtype=`, so the
    // unconditional `array_impl(fill_value, None)` + unchecked `cast_to`
    // below is already CORRECT for every integer target and must stay
    // exactly as-is for one (verified: this task's fix must not turn that
    // existing, numpy-matching wraparound into a spurious raise). The one
    // real bug: for a FLOAT/COMPLEX target, that same untyped
    // `array_impl(fill_value, None)` first tries to fit every element into
    // `i64`/`u64` (numpy's own untyped-list default) and fails for a huge
    // Python int that fits the float/complex target fine but neither of
    // those (`np.full((2,), [2**64, 2**64], dtype=np.float64)` ->
    // `[1.8446744e+19, 1.8446744e+19]` in real numpy; anionpy raised the
    // wrong, misleading `TypeError('full()'s fill_value must be a
    // bool/int/float/complex scalar')`, having swallowed the real
    // `OverflowError`). Fixed the same way `array_impl`'s own
    // `ndarray_from_pylist_typed` bypasses that lane for a float/complex
    // target (see its doc comment in lib.rs): call
    // `ndarray_from_pylist_typed` DIRECTLY with `Some(target)` -- not
    // `crate::array_impl`, which would also turn on the (here, wrong)
    // integer-target bounds-check path -- only when `target` is itself
    // floating/complex, since that is the one case where threading the
    // dtype through changes anything (an integer/bool target's untyped
    // `i64`/`u64` inference already agrees with `cast_to`'s own
    // wraparound whenever it doesn't outright fail, and outright failing
    // for an int target here matches real numpy's own `dtype=None`
    // fallback-to-`object` limitation exactly, unchanged).
    let built = if target.is_floating() || target.is_complex() {
        ndarray_from_pylist_typed(fill_value, Some(target))
    } else {
        crate::array_impl(fill_value, None)
    };
    match built {
        Ok(nd) => Ok(FillMaterial::Array(nd.cast_to(target))),
        // SEV-1-adjacent fix (2026-08-02), found by the same `full` sweep as
        // the float/complex fix just above: a Python-int list fill_value
        // with an integer `dtype=` that doesn't fit `i64`/`u64` at all
        // (e.g. `full((2,), [2**64, 2**64], dtype=np.uint8)`) legitimately
        // raises in real numpy too -- verified against 2.5.1:
        // `OverflowError('Python int too large to convert to C long')`,
        // the EXACT text `default_int_list_dtype` (lib.rs) already
        // produces for this case. The old code blanket-converted every
        // error from the fallback build (including this one) into the
        // generic `TypeError` below, so a genuine, already-correctly-
        // worded `OverflowError` was being swallowed and replaced with a
        // misleading "must be a scalar" message. Fixed by letting a
        // `PyOverflowError` specifically propagate unchanged; every other
        // error (a genuinely unsupported fill_value type, e.g. a bare
        // string or set -- real numpy accepts those via `dtype=object`,
        // which anionpy has no equivalent for, an accepted, pre-existing
        // limitation, NOT part of this task's scope) still falls through
        // to the original generic `TypeError`, unchanged.
        Err(e) if e.is_instance_of::<PyOverflowError>(fill_value.py()) => Err(e),
        Err(_) => Err(PyTypeError::new_err(
            "full()'s fill_value must be a bool/int/float/complex scalar",
        )),
    }
}

/// numpy's own broadcast-input-array error for `full`/`full_like`
/// (`could not broadcast input array from shape X into shape Y`) --
/// deliberately NOT `ionp_core::creation::broadcast_to`'s own
/// `IonpError::Broadcast` message text, which is worded for elementwise-op
/// broadcasting (`operands could not be broadcast together with remapped
/// shapes...`) and was verified against real numpy 2.5.1 to be a
/// genuinely different message at this call site
/// (`np.broadcast_to([1,2,3], (2,2))` vs `np.full((2,2), [1,2,3])` raise
/// different text for the same underlying shape mismatch).
fn broadcast_or_err(arr: &NdArray, dims: &[usize]) -> PyResult<NdArray> {
    creation::broadcast_to(arr, dims).map_err(|_| {
        PyValueError::new_err(format!(
            "could not broadcast input array from shape {} into shape {}",
            fmt_dims(arr.shape()),
            fmt_dims(dims)
        ))
    })
}

/// numpy-style shape tuple formatting (`(N,)` for one element, `(a,b,c)`
/// with no spaces for more) -- matches `ionp-core`'s own private
/// `error::fmt_shape` convention, duplicated here in a `pub`-free helper
/// since that formatter isn't exposed outside `ionp-core::error`.
fn fmt_dims(shape: &[usize]) -> String {
    if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

/// Same numpy-scalar/`anionpy.ndarray` recognition as `fill_buffer`, for
/// `full`'s dtype=None default-dtype-from-fill_value path (`np.full((2,2),
/// np.int64(300))` -> `dtype=int64`, taken from the fill value's own dtype,
/// not from `classify_scalar`'s bare-Python-scalar table).
fn default_dtype_for_fill_value(fill_value: &Bound<'_, PyAny>) -> PyResult<DType> {
    if let Ok(other) = fill_value.extract::<PyRef<'_, PyArray>>() {
        return Ok(other.inner.dtype());
    }
    if fill_value.hasattr("dtype")? {
        let py = fill_value.py();
        let np = PyModule::import(py, "numpy")?;
        let as_array = np.getattr("asarray")?.call1((fill_value,))?;
        let nd = ndarray_from_numpy(&as_array)?;
        return Ok(nd.dtype());
    }
    if let Some(kind) = classify_scalar(fill_value) {
        let int_value = if matches!(kind, ScalarKind::Int) {
            Some(fill_value.extract::<i128>()?)
        } else {
            None
        };
        return default_dtype_for_scalar(kind, int_value);
    }
    // Same list/tuple fallback as `fill_material` (see its doc comment):
    // `np.full((2,2), [7, 8])` with no explicit `dtype=` infers int64 from
    // the list's own elements, exactly like `np.array([7, 8]).dtype`.
    crate::array_impl(fill_value, None).map(|nd| nd.dtype()).map_err(|_| {
        PyTypeError::new_err("full()'s fill_value must be a bool/int/float/complex scalar")
    })
}

// ---------------------------------------------------------------------------
// zeros / ones / empty / full
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (shape, dtype=None, order=None, *, device=None, like=None))]
fn zeros(
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: Option<&str>,
    device: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    check_device(device)?;
    check_like(like)?;
    let dims = shape_from_pyobj(shape)?;
    let dt = resolve_dtype(dtype, DType::F64)?;
    let ord = order_from_pyobj(order)?;
    let n: usize = dims.iter().product();
    let inner = NdArray::from_buffer(creation::zeros_buffer(dt, n), dims, ord).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None, order=None, *, device=None, like=None))]
fn ones(
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: Option<&str>,
    device: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    check_device(device)?;
    check_like(like)?;
    let dims = shape_from_pyobj(shape)?;
    let dt = resolve_dtype(dtype, DType::F64)?;
    let ord = order_from_pyobj(order)?;
    let n: usize = dims.iter().product();
    let inner = NdArray::from_buffer(creation::ones_buffer(dt, n), dims, ord).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None, order=None, *, device=None, like=None))]
fn empty(
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: Option<&str>,
    device: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    check_device(device)?;
    check_like(like)?;
    let dims = shape_from_pyobj(shape)?;
    let dt = resolve_dtype(dtype, DType::F64)?;
    let ord = order_from_pyobj(order)?;
    let n: usize = dims.iter().product();
    let inner = NdArray::from_buffer(creation::empty_buffer(dt, n), dims, ord).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype=None, order=None, *, device=None, like=None))]
fn full(
    shape: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: Option<&str>,
    device: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    check_device(device)?;
    check_like(like)?;
    let dims = shape_from_pyobj(shape)?;
    let dt = match dtype {
        // `full`'s fill-value path (`fill_material` below) is a numeric
        // scalar/array broadcaster; it was never extended in this phase
        // to build S/U buffers from a fill value, so an S/U target here
        // must be declined cleanly rather than reaching whatever numeric
        // `cast_to`/dispatch panics deepest in `fill_material`.
        Some(d) => crate::dtype_from_pyobj_no_su(d)?,
        None => default_dtype_for_fill_value(fill_value)?,
    };
    let ord = order_from_pyobj(order)?;
    let n: usize = dims.iter().product();
    let inner = match fill_material(fill_value, dt)? {
        FillMaterial::ScalarLiteral(one) => {
            NdArray::from_buffer(creation::repeat_scalar_buffer(&one, n), dims, ord)
                .map_err(to_py_err)?
        }
        FillMaterial::Array(arr) => {
            let bview = broadcast_or_err(&arr, &dims)?;
            let letter = match ord {
                Order::C => "C",
                Order::F => "F",
            };
            bview.to_contiguous_order(letter).map_err(to_py_err)?
        }
    };
    Ok(PyArray { inner })
}

/// The `_like` family's `order=` resolution (default `'K'`, per numpy:
/// `zeros_like(a, dtype=None, order='K', subok=True, shape=None)` and
/// siblings). When the (possibly `shape=`-overridden) output `dims` has
/// the SAME rank as the prototype, this reuses `proto`'s own
/// `axis_perm_for_order` exactly -- the prototype's stride-magnitude
/// permutation, relabeled onto the new dims' sizes -- which matches real
/// numpy 2.5.1 even when `dims != proto.shape()` (verified: an F-contig
/// (2,3) prototype's `zeros_like(order='K', shape=(3,2))` has strides
/// `(8,24)`, exactly what this formula produces). When the rank differs
/// (numpy's `shape=` can change ndim too), there is no per-axis
/// correspondence left to preserve, so this falls back to the coarser
/// "F if the prototype is F-contiguous-and-not-C-contiguous, else C" rule,
/// which is what real numpy reduces to in that case (verified: same
/// prototype, `shape=(6,)` -> strides `(8,)`, i.e. plain C/F-equivalent
/// for a 1-D result).
/// The axis permutation `like_strides` derives its strides from, factored
/// out so a second caller (`full_like`'s array-fill_value broadcast path,
/// which needs to relayout gathered VALUES the same way `like_strides`
/// lays out empty/zero/one-filled buffers) can reuse the identical
/// prototype-aware order resolution instead of re-deriving its own
/// (potentially divergent) permutation. Pure refactor of the logic that
/// used to live inline in `like_strides` -- same two branches, same
/// results, `like_strides` below is now just "turn this perm into
/// strides."
fn like_perm(proto: &NdArray, dims: &[usize], order: Option<&str>) -> PyResult<Vec<usize>> {
    let ord = order.unwrap_or("K");
    if dims.len() == proto.ndim() {
        proto.axis_perm_for_order(ord).map_err(to_py_err)
    } else {
        // Rank changed: fall back through `order_from_pyobj` (raises on a
        // genuinely invalid letter, same as everywhere else) and pick C or
        // F strides for the new rank based on the prototype's contiguity.
        let use_f = matches!(ord, "F") || ((ord == "A" || ord == "K") && proto.is_f_contiguous() && !proto.is_c_contiguous());
        if use_f {
            Ok((0..dims.len()).rev().collect())
        } else if matches!(ord, "C" | "A" | "K") {
            Ok((0..dims.len()).collect())
        } else {
            Err(to_py_err(IonpError::Value(format!(
                "order must be one of 'C', 'F', 'A', or 'K' (got '{ord}')"
            ))))
        }
    }
}

fn like_strides(proto: &NdArray, dims: &[usize], order: Option<&str>) -> PyResult<Vec<isize>> {
    // Real numpy gives a freshly-allocated size-0 array all-zero strides
    // regardless of requested order/layout (verified directly against
    // numpy 2.5.1 for shapes (0,), (0,3), (3,0), (2,0,4), (0,0), both
    // C- and F-order) -- `empty_like`/`zeros_like`/`ones_like`/
    // `full_like`'s scalar-fill branches all go through this function to
    // build a brand-new buffer of `dims`, so that rule applies here
    // uniformly, before the order-permutation logic below (which
    // implements the *nonempty* "reshape formula" and does not zero).
    if dims.iter().product::<usize>() == 0 {
        return Ok(vec![0isize; dims.len()]);
    }
    let perm = like_perm(proto, dims, order)?;
    let permuted_shape: Vec<usize> = perm.iter().map(|&a| dims[a]).collect();
    let permuted_strides = ionp_core::shape::c_strides(&permuted_shape);
    let mut strides = vec![0isize; dims.len()];
    for (i, &axis) in perm.iter().enumerate() {
        strides[axis] = permuted_strides[i];
    }
    Ok(strides)
}

#[pyfunction]
#[pyo3(signature = (a, dtype=None, order=None, subok=None, shape=None, *, device=None))]
fn zeros_like(
    a: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: Option<&str>,
    subok: Option<&Bound<'_, PyAny>>,
    shape: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    check_subok(subok);
    check_device(device)?;
    let proto = extract_or_ingest_ndarray(a)?;
    let dt = resolve_dtype(dtype, proto.dtype())?;
    let dims = match shape {
        Some(s) => shape_from_pyobj(s)?,
        None => proto.shape().to_vec(),
    };
    let n: usize = dims.iter().product();
    let strides = like_strides(&proto, &dims, order)?;
    let inner = NdArray::from_buffer_with_strides(creation::zeros_buffer(dt, n), dims, strides).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (a, dtype=None, order=None, subok=None, shape=None, *, device=None))]
fn ones_like(
    a: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: Option<&str>,
    subok: Option<&Bound<'_, PyAny>>,
    shape: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    check_subok(subok);
    check_device(device)?;
    let proto = extract_or_ingest_ndarray(a)?;
    let dt = resolve_dtype(dtype, proto.dtype())?;
    let dims = match shape {
        Some(s) => shape_from_pyobj(s)?,
        None => proto.shape().to_vec(),
    };
    let n: usize = dims.iter().product();
    let strides = like_strides(&proto, &dims, order)?;
    let inner = NdArray::from_buffer_with_strides(creation::ones_buffer(dt, n), dims, strides).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (a, dtype=None, order=None, subok=None, shape=None, *, device=None))]
fn empty_like(
    a: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: Option<&str>,
    subok: Option<&Bound<'_, PyAny>>,
    shape: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    check_subok(subok);
    check_device(device)?;
    let proto = extract_or_ingest_ndarray(a)?;
    let dt = resolve_dtype(dtype, proto.dtype())?;
    let dims = match shape {
        Some(s) => shape_from_pyobj(s)?,
        None => proto.shape().to_vec(),
    };
    let n: usize = dims.iter().product();
    let strides = like_strides(&proto, &dims, order)?;
    let inner = NdArray::from_buffer_with_strides(creation::empty_buffer(dt, n), dims, strides).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (a, fill_value, dtype=None, order=None, subok=None, shape=None, *, device=None))]
fn full_like(
    a: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: Option<&str>,
    subok: Option<&Bound<'_, PyAny>>,
    shape: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    check_subok(subok);
    check_device(device)?;
    let proto = extract_or_ingest_ndarray(a)?;
    let dt = resolve_dtype(dtype, proto.dtype())?;
    let dims = match shape {
        Some(s) => shape_from_pyobj(s)?,
        None => proto.shape().to_vec(),
    };
    let n: usize = dims.iter().product();
    let inner = match fill_material(fill_value, dt)? {
        FillMaterial::ScalarLiteral(one) => {
            let strides = like_strides(&proto, &dims, order)?;
            NdArray::from_buffer_with_strides(
                creation::repeat_scalar_buffer(&one, n),
                dims,
                strides,
            )
            .map_err(to_py_err)?
        }
        FillMaterial::Array(arr) => {
            let bview = broadcast_or_err(&arr, &dims)?;
            // `bview.shape() == dims` (broadcast_to always returns exactly
            // the target shape), so `perm` -- computed the same
            // prototype-aware way `like_strides` computes it for the
            // empty/zero/one-filled paths -- indexes it validly.
            let perm = like_perm(&proto, &dims, order)?;
            bview.relayout_by_perm(&perm)
        }
    };
    Ok(PyArray { inner })
}

// ---------------------------------------------------------------------------
// arange / linspace / eye / identity
// ---------------------------------------------------------------------------

/// One resolved `arange` endpoint.
///
/// `dtype` is `Some` only for a STRONG endpoint -- a numpy/anionpy 0-d scalar
/// or 0-d array, which carries its own width -- and `None` for a bare Python
/// `bool`/`int`/`float`, which is weak and contributes only its kind.
struct Endpoint {
    value: f64,
    kind: ScalarKind,
    dtype: Option<DType>,
}

/// Resolve one `arange` endpoint.
///
/// `classify_scalar` covers only *bare* Python `bool`/`int`/`float`, and it
/// must stay that way: it is the PyO3 half of the NEP 50 weak-promotion
/// rule, under which a numpy scalar is emphatically NOT weak. So numpy
/// scalars are accepted HERE instead, carrying their dtype with them.
///
/// Accepted: `ndim == 0` only. Measured, not guessed -- a 0-d array works in
/// numpy (`np.arange(np.array(3))` -> `int64 [0,1,2]`) while ndim >= 1
/// raises, so `.ndim` is the predicate, not "has a dtype".
///
/// NOT accepted: complex, of either spelling. `arange` on a complex operand
/// is an existing, separate divergence -- anionpy already raises on a bare
/// `3+0j` where numpy returns an empty complex128 array, and numpy's own
/// complex behaviour there is erratic (`np.arange(np.complex64(3))` has
/// three elements, `np.arange(np.complex128(3))` has none). Letting complex
/// numpy scalars in would trade one honest raise for a new wrong answer.
/// Complex `arange` is its own item, with its own probe.
///
/// A NOTE ON HOW THIS WAS GOT WRONG FIRST (2026-08-03, Monday), because the
/// wrong version is the tempting one: this shipped briefly as "numpy scalars
/// are weak here, only the kind matters", on the strength of a probe over
/// {int8,uint8,int64,float16,float32,float64} x {stop, start+stop, step}
/// that showed 0/18 disagreements with the bare-Python spelling. The probe
/// was clean and the conclusion was false. It contained no `start > stop`
/// case and no `uint64`, which are exactly the two places the width leaks
/// out -- see `arange_result_dtype` and `arange_count` below. A sample that
/// varies the KWARGS while holding the DATA in one shape can agree
/// perfectly and still be measuring nothing.
fn extract_arange_endpoint(obj: &Bound<'_, PyAny>) -> PyResult<Endpoint> {
    if let Some(kind) = classify_scalar(obj) {
        return Ok(Endpoint { value: obj.extract::<f64>()?, kind, dtype: None });
    }
    if let Some(dt) = zero_dim_scalar_dtype(obj)? {
        let kind = if dt.is_bool() {
            ScalarKind::Bool
        } else if dt.is_integer() {
            ScalarKind::Int
        } else {
            ScalarKind::Float
        };
        return Ok(Endpoint { value: obj.extract::<f64>()?, kind, dtype: Some(dt) });
    }
    Err(PyTypeError::new_err("expected a bool/int/float scalar"))
}

/// `Some(dtype)` when `obj` is a 0-d numpy/anionpy scalar or array of
/// bool/int/float dtype; `None` for anything else (complex included, ndim
/// >= 1 included). See `extract_arange_endpoint`.
fn zero_dim_scalar_dtype(obj: &Bound<'_, PyAny>) -> PyResult<Option<DType>> {
    if !obj.hasattr("dtype")? {
        return Ok(None);
    }
    match obj.getattr("ndim").and_then(|n| n.extract::<usize>()) {
        Ok(0) => {}
        _ => return Ok(None),
    }
    let dt = match dtype_from_pyobj(&obj.getattr("dtype")?) {
        Ok(dt) => dt,
        // An unresolvable dtype spelling is not an arange endpoint. Fall
        // through to the ordinary TypeError rather than surfacing a
        // dtype-parsing message from a call that never mentioned a dtype.
        Err(_) => return Ok(None),
    };
    Ok((dt.is_bool() || dt.is_integer() || dt.is_floating()).then_some(dt))
}

/// numpy's default `arange` result dtype: `result_type(start, stop, step,
/// <default int>)`.
///
/// Verified against real numpy 2.5.1 over all 144 ordered dtype pairs drawn
/// from {bool, int8..int64, uint8..uint64, float16/32/64}: 0/144 disagree.
/// The `int64` seed is what makes the two surprising cases fall out rather
/// than needing to be special-cased:
///
///     int8    -> promote(int64, int8)    = int64    (NOT int8)
///     float16 -> promote(int64, float16) = float64  (NOT float16)
///     uint64  -> promote(int64, uint64)  = float64  (!)
///
/// The `uint64` line is the one that falsified the earlier "width doesn't
/// matter" model outright: no weak-scalar reading of `np.arange(np.uint64(0),
/// np.uint64(3))` produces `float64`, and numpy does.
/// Reduce `v` into `dt`'s value range with two's-complement wraparound --
/// what a C store into that width does, and what numpy's integer `arange`
/// does at every element (`np.arange(0, 300, 100, dtype='int8')` ->
/// `[0, 100, -56]`, not a clamp to `127` and not an error).
///
/// Every modulus here is a power of two, so doing the surrounding
/// arithmetic with `wrapping_*` on `i128` and reducing afterwards is
/// exact: `2^bits` divides `2^128`, so a wrap at 128 bits never disturbs
/// the low `bits`. That is what lets the caller multiply a `u64`-width
/// delta by a large index without an overflow check.
fn wrap_to_int_dtype(v: i128, dt: DType) -> i128 {
    let (bits, signed) = match dt {
        DType::I8 => (8u32, true),
        DType::I16 => (16, true),
        DType::I32 => (32, true),
        DType::I64 => (64, true),
        DType::U8 => (8, false),
        DType::U16 => (16, false),
        DType::U32 => (32, false),
        DType::U64 => (64, false),
        _ => return v,
    };
    let modulus = 1i128 << bits;
    let mut r = v.rem_euclid(modulus);
    if signed && r >= modulus >> 1 {
        r -= modulus;
    }
    r
}

/// numpy's integer-dtype `arange` fill, measured rather than assumed.
///
/// numpy does NOT compute `start + i*step` in f64 and truncate each
/// element, and it does NOT truncate `start` and `step` separately and
/// then walk. It writes exactly TWO seed values into the output buffer,
/// each an independent cast of a full-precision f64:
///
///     buf[0] = trunc(start)
///     buf[1] = trunc(start + step)
///
/// and then fills the rest by repeatedly adding the buffer's own
/// difference, `delta = buf[1] - buf[0]`, computed IN THE TARGET WIDTH:
///
///     buf[i] = buf[1] + (i-1) * delta
///
/// The step is therefore never rounded on its own -- it is only ever
/// rounded jointly with `start`, and the effective integer step is a
/// DIFFERENCE OF TWO TRUNCATIONS. That single distinction is the whole
/// behavior, and it is why a fractional `start` changes the step:
///
///     np.arange(-0.5, 3.5, 1, dtype='int8')  -> [0, 0, 0, 0]
///         trunc(-0.5) = 0, trunc(0.5) = 0  => delta 0
///     np.arange( 0.5, 3.5, 1, dtype='int8')  -> [0, 1, 2, 3]
///         trunc( 0.5) = 0, trunc(1.5) = 1  => delta 1
///
/// Same nominal step of 1, different sequences, no rounding rule applied
/// to the step in either. Likewise `np.arange(-1.5, 3.5, 2, dtype='int8')`
/// -> `[-1, 0, 1]`: an effective step of 1 from a nominal step of 2.
///
/// The LENGTH is not computed here -- it comes from the caller's f64
/// formula, unchanged. Length and values genuinely come from different
/// arithmetic, which is why the count can exceed what the integer step
/// would justify and produce a constant array.
///
/// Verified against real numpy 2.5.1 over a 6,079-case cross-product
/// (12 starts x 7 stops x 10 steps x all 8 integer dtypes, spanning
/// fractional/negative/large starts, negative and sub-unit steps, and
/// deliberate width overflow): 0 mismatches. Deliberately scoped to
/// integer dtypes only -- the float path was measured at the same time
/// and does NOT follow this rule, so it is left exactly as it was.
fn arange_int_values(start: f64, step: f64, n: usize, dt: DType) -> Buffer {
    let mut out: Vec<i128> = Vec::with_capacity(n);
    if n > 0 {
        let b0 = wrap_to_int_dtype(start.trunc() as i128, dt);
        out.push(b0);
        if n > 1 {
            let b1 = wrap_to_int_dtype((start + step).trunc() as i128, dt);
            let delta = wrap_to_int_dtype(b1.wrapping_sub(b0), dt);
            for i in 1..n {
                let v = b1.wrapping_add((i as i128 - 1).wrapping_mul(delta));
                out.push(wrap_to_int_dtype(v, dt));
            }
        }
    }
    match dt {
        DType::I8 => Buffer::I8(out.into_iter().map(|v| v as i8).collect()),
        DType::I16 => Buffer::I16(out.into_iter().map(|v| v as i16).collect()),
        DType::I32 => Buffer::I32(out.into_iter().map(|v| v as i32).collect()),
        DType::I64 => Buffer::I64(out.into_iter().map(|v| v as i64).collect()),
        DType::U8 => Buffer::U8(out.into_iter().map(|v| v as u8).collect()),
        DType::U16 => Buffer::U16(out.into_iter().map(|v| v as u16).collect()),
        DType::U32 => Buffer::U32(out.into_iter().map(|v| v as u32).collect()),
        DType::U64 => Buffer::U64(out.into_iter().map(|v| v as u64).collect()),
        _ => unreachable!("arange_int_values is only called for integer dtypes"),
    }
}

fn arange_result_dtype(endpoints: &[&Endpoint]) -> DType {
    let any_float = endpoints
        .iter()
        .any(|e| matches!(e.kind, ScalarKind::Float | ScalarKind::Complex));
    let mut acc = if any_float { DType::F64 } else { DType::I64 };
    for e in endpoints {
        if let Some(dt) = e.dtype {
            acc = ionp_core::dtype::promote_dtype(acc, dt);
        }
    }
    acc
}

/// The number of elements numpy will produce.
///
/// numpy computes `stop - start` in the COMMON DTYPE OF THE OPERANDS -- not
/// in the result dtype, and not in f64 -- so when every operand is either an
/// unsigned numpy scalar or a weak integer, the subtraction WRAPS:
///
///     np.arange(np.uint8(1),  np.uint8(0)) -> 255 elements (1,2,...,255)
///     np.arange(np.uint8(1),  0)           -> 255 elements   (weak int, still unsigned)
///     np.arange(np.uint8(5),  np.uint8(2)) -> 253 elements
///     np.arange(np.uint8(1),  np.int8(0))  ->   0 elements   (common int16, signed)
///     np.arange(np.uint8(1),  0.0)         ->   0 elements   (common float64)
///     np.arange(np.uint64(1), np.uint64(0)) -> ValueError: Maximum allowed size exceeded
///
/// A weak integer does NOT widen the common dtype (row 2 still wraps); one
/// signed or float operand is enough to stop it (rows 4 and 5). All six rows
/// measured against real numpy 2.5.1.
fn arange_count(start: f64, stop: f64, step: f64, operand_common: Option<DType>) -> PyResult<usize> {
    let uint_bits = match operand_common {
        Some(DType::U8) => 8u32,
        Some(DType::U16) => 16,
        Some(DType::U32) => 32,
        Some(DType::U64) => 64,
        _ => return Ok(creation::arange_len(start, stop, step)),
    };
    // A NEGATIVE endpoint against an unsigned common dtype wraps to an
    // enormous unsigned value, and the length computation overflows before
    // any array is built. numpy reports that as a size error, not as an
    // OverflowError on the value:
    //
    //     np.arange(-2, np.uint8(0))   -> ValueError: Maximum allowed size exceeded
    //     np.arange(-1, np.uint8(3))   -> ValueError
    //     np.arange(np.uint8(3), -2)   -> ValueError   (either order)
    //     np.arange(-2, np.uint64(3))  -> ValueError   (every width)
    //     np.arange(-0.5, np.uint8(3)) -> float64 [-0.5, 0.5, 1.5, 2.5]
    //
    // 7 of 7 measured combinations raise; the last row does NOT, and does
    // not contradict the rule -- a weak FLOAT endpoint pulls the common
    // dtype to float64, so it never reaches this branch at all.
    if start < 0.0 || stop < 0.0 {
        return Err(PyValueError::new_err("Maximum allowed size exceeded"));
    }
    let modulus = 2f64.powi(uint_bits as i32);
    let delta = stop.trunc() - start.trunc();
    let wrapped = delta.rem_euclid(modulus);
    let n = (wrapped / step).ceil();
    if !(n > 0.0) {
        return Ok(0);
    }
    // numpy refuses the allocation rather than attempting it; `np.arange(
    // np.uint64(1), np.uint64(0))` asks for 2**64 - 1 elements and raises
    // this exact ValueError.
    // `>=`, not `>`: `isize::MAX as f64` rounds UP to 2^63 exactly, so a
    // requested count of exactly 2^63 (which `np.arange(np.uint64(1),
    // np.uint64(0), 2)` asks for) compared `>` false, sailed past this
    // guard, and aborted the interpreter with a PyO3 PanicException on the
    // allocation instead of raising. Caught by the out-of-corpus sweep, not
    // by reasoning about the bound.
    if n >= isize::MAX as f64 {
        return Err(PyValueError::new_err("Maximum allowed size exceeded"));
    }
    Ok(n as usize)
}

#[pyfunction]
#[pyo3(signature = (start_or_stop, stop=None, step=None, *, dtype=None, device=None, like=None))]
fn arange(
    start_or_stop: &Bound<'_, PyAny>,
    stop: Option<&Bound<'_, PyAny>>,
    step: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    check_device(device)?;
    check_like(like)?;
    let (start_ep, stop_ep) = match stop {
        Some(s) => {
            let a = extract_arange_endpoint(start_or_stop)?;
            let b = extract_arange_endpoint(s)?;
            (Some(a), b)
        }
        None => (None, extract_arange_endpoint(start_or_stop)?),
    };
    let step_ep = match step {
        Some(s) => Some(extract_arange_endpoint(s)?),
        None => None,
    };
    let start_v = start_ep.as_ref().map_or(0.0, |e| e.value);
    let start_kind = start_ep.as_ref().map(|e| e.kind);
    let stop_v = stop_ep.value;
    let step_v = step_ep.as_ref().map_or(1.0, |e| e.value);
    if step_v == 0.0 {
        return Err(PyZeroDivisionError::new_err("division by zero"));
    }
    let endpoints: Vec<&Endpoint> = start_ep
        .iter()
        .chain(std::iter::once(&stop_ep))
        .chain(step_ep.iter())
        .collect();

    // The dtype the LENGTH is computed in, which is NOT the result dtype:
    // it is the common dtype of START and STOP alone -- no default-int
    // seed, and NO STEP. A weak integer contributes nothing to it (so it
    // stays unsigned and wraps); a weak float pulls it to f64 (so it does
    // not).
    //
    // Excluding `step` is measured, and an earlier draft that included it
    // was wrong on 376 cases:
    //
    //     np.arange(1, np.uint8(0), 0.5) -> float64, 510 elements
    //
    // The float step makes the RESULT float64, but the length is still
    // computed in uint8 -- (0 - 1) wraps to 255, and 255/0.5 = 510. Fold
    // the step in and the common dtype becomes float64, the wrap
    // disappears, and the answer collapses to an empty array.
    let mut start_stop_common: Option<DType> = None;
    for e in start_ep.iter().chain(std::iter::once(&stop_ep)) {
        let contrib = match (e.dtype, e.kind) {
            (Some(d), _) => Some(d),
            (None, ScalarKind::Float | ScalarKind::Complex) => Some(DType::F64),
            (None, _) => None,
        };
        if let Some(c) = contrib {
            start_stop_common =
                Some(start_stop_common.map_or(c, |acc| ionp_core::dtype::promote_dtype(acc, c)));
        }
    }

    // numpy computes the length as `stop - start` in the operands' own
    // dtype (see `arange_count`), and numpy has no boolean `-`. So a
    // start/stop pair that are BOTH strong bools raises before any array
    // exists. The trigger is genuinely the pair, not the promoted dtype:
    //
    //     np.arange(np.bool_(0), np.bool_(1)) -> TypeError (boolean subtract)
    //     np.arange(np.bool_(0), 3)           -> int64 [0,1,2]   (weak stop, fine)
    //     np.arange(0, np.bool_(1))           -> int64 [0]       (weak start, fine)
    //     np.arange(np.bool_(1))              -> int64 [0]       (no start at all)
    //
    // all four measured against real numpy 2.5.1. An earlier draft keyed
    // this off "common operand dtype is Bool", which row 2 falsifies -- a
    // weak operand contributes nothing to that common dtype, so row 2 would
    // have raised.
    if matches!(
        (start_ep.as_ref().and_then(|e| e.dtype), stop_ep.dtype),
        (Some(DType::Bool), Some(DType::Bool))
    ) {
        return Err(PyTypeError::new_err(
            "numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, \
             the `^` operator, or the logical_xor function instead.",
        ));
    }
    // THE UNESTABLISHED CORNER -- deliberately REJECTED rather than
    // answered. Everything above is a rule I measured and could state; what
    // follows is a region where I could not, so anionpy declines it with the
    // same `TypeError` it raised before numpy scalars were accepted at all.
    // That keeps this change a strict improvement: no input that worked
    // today changes its answer, and no input gets a fabricated one.
    //
    // The region: a STRONG UNSIGNED operand together with any negative
    // value, or a STRONG BOOL step. numpy's behaviour there is erratic and
    // position-dependent, and FOUR successive models of it were falsified,
    // each by data the previous model had not seen:
    //
    //     np.arange(np.uint8(3), 0.5, -1) -> ValueError: Maximum allowed size exceeded
    //     np.arange(7, np.uint8(4),  -2)  -> int64 []          (same shape, no error)
    //     np.arange(np.uint8(3), -0.5, -1)   -> ValueError
    //     np.arange(np.uint8(3), -0.5, -1.0) -> float64 [3.0, 2.0, 1.0, 0.0]
    //     np.arange(-3, 7, np.uint8(3))   -> ValueError
    //     np.arange(-0.5, np.uint8(3))    -> float64 [-0.5, 0.5, 1.5, 2.5]
    //     np.arange(np.bool_(1), 7, np.bool_(1)) -> int64 [1,1,1,1,1,1]  (not 1..6)
    //
    // Whether it raises depends on which POSITION the unsigned scalar
    // occupies, on whether the negative operand is spelled `-1` or `-1.0`,
    // and for bool on the step accumulating in bool arithmetic. A fitted
    // predicate covering the first six rows with zero counterexamples over
    // 25,014 combinations was then wrong on 416 cases of a 55,004-case
    // sweep using fresh values. That is the whole reason this is a
    // rejection and not a fifth guess: a predicate that reproduces only the
    // data it was fitted to has not been verified, it has been memorised.
    //
    // The ONE part of this region that IS established keeps its correct
    // answer and is checked first, below, in `arange_count`: a negative
    // START or STOP against an unsigned start/stop common dtype raises
    // numpy's real `ValueError`, 7 of 7 measured combinations. The
    // rejection here covers only what is left after that.
    let touches_strong_unsigned = endpoints
        .iter()
        .any(|e| matches!(e.dtype, Some(DType::U8 | DType::U16 | DType::U32 | DType::U64)));
    let any_negative = endpoints.iter().any(|e| e.value < 0.0);
    // Both start AND step strong bools. Narrowed by measurement -- a strong
    // bool step alone is perfectly ordinary, and rejecting it cost 4,050
    // working cases in a fresh sweep before this was tightened:
    //
    //     np.arange(np.bool_(1), 7, np.bool_(1)) -> [1, 1, 1, 1, 1, 1]  (!)
    //     np.arange(1,           7, np.bool_(1)) -> [1, 2, 3, 4, 5, 6]
    //     np.arange(np.int8(1),  7, np.bool_(1)) -> [1, 2, 3, 4, 5, 6]
    //     np.arange(np.bool_(1), 7, 1)           -> [1, 2, 3, 4, 5, 6]
    //     np.arange(np.bool_(0), 7, np.bool_(1)) -> [0, 1, 2, 3, 4, 5, 6]
    //
    // Only the first row is anomalous, and only its exact shape is
    // rejected. Note row 5 shows even that shape behaves normally when
    // `start` is False, so this over-rejects one working case -- one that
    // raised this same TypeError before today regardless.
    let strong_bool_step = step_ep.as_ref().is_some_and(|e| e.dtype == Some(DType::Bool))
        && start_ep.as_ref().and_then(|e| e.dtype) == Some(DType::Bool);
    let established_negative_unsigned = matches!(
        start_stop_common,
        Some(DType::U8 | DType::U16 | DType::U32 | DType::U64)
    ) && (start_v < 0.0 || stop_v < 0.0);
    if touches_strong_unsigned && any_negative && !established_negative_unsigned {
        return Err(PyTypeError::new_err("expected a bool/int/float scalar"));
    }

    let any_float = endpoints
        .iter()
        .any(|e| matches!(e.kind, ScalarKind::Float | ScalarKind::Complex));
    // `resolve_dtype` itself stays permissive (it's shared with the
    // `_like` family, which DOES have real S/U support via
    // `zeros_buffer`/`ones_buffer`/`empty_buffer`'s `bump_zero_width`).
    // `arange`'s own generation path (`arange_int_values`/float-range
    // construction below) builds a numeric buffer and `cast_to`s it,
    // which has no S/U arm -- so `arange`'s explicit `dtype=` must be
    // pre-declined here, before `resolve_dtype` ever returns an S/U value
    // to this specific caller.
    if let Some(d) = dtype {
        crate::dtype_from_pyobj_no_su(d)?;
    }
    let dt = resolve_dtype(dtype, arange_result_dtype(&endpoints))?;

    let n = arange_count(start_v, stop_v, step_v, start_stop_common)?;

    // The strong-bool start+step anomaly is a defect in the VALUES, so it
    // cannot manifest in an array that has none. Gating on `n > 0` rather
    // than rejecting the shape outright recovers 2,398 fresh-sweep cases
    // where numpy simply returns an empty array (`np.arange(np.bool_(1),
    // 0.25, np.bool_(1))` -> `[]`), and those are answered exactly.
    if strong_bool_step && n > 0 {
        return Err(PyTypeError::new_err("expected a bool/int/float scalar"));
    }

    // `dtype=bool` is only legal when the result fits in 2 elements (numpy
    // 2.5.1: `np.arange(0, 3, dtype=bool)` -> `TypeError: arange() is only
    // supported for booleans when the result has at most length 2.`,
    // verified interactively; `np.arange(0, 2, dtype=bool)` -> `[False
    // True]` succeeds). Checked before generation, not after, since numpy
    // raises without ever materializing the (potentially huge) array.
    if dt == DType::Bool && n > 2 {
        return Err(PyTypeError::new_err(
            "arange() is only supported for booleans when the result has at most length 2.",
        ));
    }

    // BUG FOUND AND FIXED 2026-08-01: an out-of-range `start` with an
    // explicit narrow-integer `dtype=` used to be silently wrapped (e.g.
    // `anionpy.arange(-3, 3, 2, dtype='uint8')` -> `[253, 255, 1]`) instead of
    // raising, because `arange` generates the whole sequence in `f64` and
    // only narrows via `Buffer::cast_to` at the very end -- there was no
    // up-front validation at all. Real numpy DOES validate: it range-checks
    // only `start` (truncated toward zero, matching a plain int/float
    // scalar's own narrowing) against the target dtype and raises numpy's
    // exact `OverflowError` if it doesn't fit -- `np.arange(-3, 3, 2,
    // dtype='uint8')` -> `OverflowError: Python integer -3 out of bounds
    // for uint8`, verified against real numpy 2.5.1. Established by direct
    // probing that this check applies ONLY to an explicit `start` (the
    // two/three-argument call form, i.e. `start_kind.is_some()` here --
    // `np.arange(300, dtype='uint8')`, a single-arg call where `300` is
    // `stop` and `start` defaults to `0`, does NOT raise and silently
    // wraps), ONLY for genuinely-integer targets (NOT `dtype=bool`: `np.
    // arange(-3, -1, 1, dtype=bool)` -> `[True, True]`, no error, since
    // `check_int_bounds` itself is not meaningful for a 2-valued target),
    // and does NOT extend to `stop`/`step` (`np.arange(0, 300, 2,
    // dtype='uint8')` and `np.arange(0, 3, 300, dtype='uint8')` both wrap
    // silently rather than raising). `start_v` is truncated toward zero
    // (not floored) before the range check, matching numpy's own
    // fractional-`start` behavior (`np.arange(-300.5, 3, 2, dtype='int8')`
    // -> `OverflowError: Python integer -300 out of bounds for int8`, i.e.
    // `-300.5` truncates to `-300`, not `-301`).
    // The range check applies to the TWO SEED VALUES the fill below writes
    // (`trunc(start)` and `trunc(start+step)`) -- and to each one only if
    // it is actually written. Both halves of that are observable:
    //
    //   - GATED ON LENGTH. `np.arange(-3.5, -2.0, -1, dtype='uint8')` is
    //     empty and does NOT raise, even though `-3` is nowhere near
    //     `uint8`. Nothing is stored, so nothing is checked. anionpy used to
    //     raise `OverflowError` here (357 cases of an 8,736-case sweep).
    //   - IT IS NOT JUST `start`. `np.arange(-0.5, -2.0, -1, dtype='uint8')`
    //     RAISES, though `trunc(-0.5) == 0` is a perfectly good `uint8`:
    //     the second seed is `trunc(-1.5) == -1`, and that is what fails.
    //     anionpy used to return `[0, 255]` here (the remaining 29).
    //
    // It is emphatically NOT a check on every element -- past the two
    // seeds numpy is doing raw C arithmetic in the target width, which
    // wraps silently (`np.arange(0, 300, 100, dtype='int8')` ->
    // `[0, 100, -56]`). Checked seeds, unchecked fill.
    //
    // Measured over a 3,024-case sweep against real numpy 2.5.1 (starts
    // straddling both ends of int8/uint8/int16/uint16, negative and
    // sub-unit steps, empty ranges): 0 mismatches on raise-vs-not.
    if dt.is_integer() {
        if n >= 1 {
            check_int_bounds(start_v.trunc() as i128, dt)?;
        }
        if n >= 2 {
            check_int_bounds((start_v + step_v).trunc() as i128, dt)?;
        }
    }

    // The two gaps that stood documented here as OPEN and "not derivable
    // from any per-element float->int cast rule this task could
    // reverse-engineer" are BOTH FIXED as of 2026-08-03, by the
    // `arange_int_values` two-seed rule below. They were never two bugs
    // and never a "genuine internal numpy C-level quirk" -- they were one
    // wrong model of how numpy fills an integer buffer, showing its edges
    // in two different places:
    //   (a) `anionpy.arange(200.9, 3, -2, dtype='uint32')` repeated `200`
    //       instead of counting down, blamed on `cast_from_float!`
    //       saturating a negative float to `0` for an unsigned target.
    //       That saturation is real, but it was only ever reached because
    //       the old code cast the STEP through it as a standalone value.
    //       numpy never casts a step. Nothing crate-wide needed changing.
    //   (b) `np.arange(-1, 1, 0.3, dtype=np.int8)` -> `[-1,0,1,2,3,4,5]`
    //       (effective integer step 1) against `np.arange(0, 1, 0.3,
    //       dtype=np.int8)` -> `[0,0,0]` (effective step 0) from the SAME
    //       nominal `step=0.3`. The note recorded that floor/trunc/ceil
    //       hypotheses "of `step`" were each falsified by a further data
    //       point. They were -- because every one of them was a rounding
    //       rule applied to the step, and the step is not what is rounded.
    //       Both sequences fall straight out of the rule below: the
    //       effective step is `trunc(start+step) - trunc(start)`, which is
    //       `0-(-1) = 1` in the first and `0-0 = 0` in the second.
    // Recorded rather than deleted, because the lesson is the reusable
    // part: three rounding-mode families were enumerated and falsified
    // one data point at a time, and the conclusion drawn was "numpy is
    // quirky here" -- when the actual fault was that all three shared an
    // unexamined premise. A hypothesis space that keeps failing is worth
    // suspecting at its ROOT, not just at its leaves.

    // numpy generates an integer/bool-dtype `arange` by first truncating
    // `start`/`step` INTO the target dtype, THEN doing the `start + i*step`
    // walk in that narrowed representation -- not by generating in f64 and
    // truncating each element afterward. The difference is only visible
    // when `start`/`step` are non-integer floats with an explicit
    // integer/bool `dtype=`: `np.arange(0.0, 5.0, 0.5, dtype=np.int8)` ->
    // `[0]*10` (length 10 from the float length formula, but `step` casts
    // to `int8(0.5) == 0` FIRST, so every element is `0 + i*0 == 0`), NOT
    // `[0,0,1,1,2,2,3,3,4,4]` (truncating each of `0.0,0.5,1.0,...`
    // separately) -- verified against real numpy 2.5.1. Found via this
    // task's own out-of-corpus probe (byte-exact, not caught by the
    // existing corpus, which never crosses float-start/int-dtype). Only
    // applies when the target is integer/bool AND at least one of
    // start/step/stop was float-kind in the first place -- when everything
    // was already integer-kind, `start_v`/`step_v` are exact in f64 for any
    // magnitude this generates (no observable difference), so the cast is
    // skipped to avoid an unnecessary extra narrowing round-trip.
    let (gen_start, gen_step) = if any_float && (dt.is_integer() || dt == DType::Bool) {
        let cast_start = Buffer::F64(vec![start_v]).cast_to(dt).cast_to(DType::F64);
        let cast_step = Buffer::F64(vec![step_v]).cast_to(dt).cast_to(DType::F64);
        let s = match cast_start {
            Buffer::F64(v) => v[0],
            _ => unreachable!("cast_to(DType::F64) must produce Buffer::F64"),
        };
        let st = match cast_step {
            Buffer::F64(v) => v[0],
            _ => unreachable!("cast_to(DType::F64) must produce Buffer::F64"),
        };
        (s, st)
    } else {
        (start_v, step_v)
    };

    // BUG FOUND AND FIXED 2026-08-01: a narrow float target (`F16`/`F32`)
    // used to always compute the whole sequence in `f64` (`start + i*step`
    // per `arange_values`'s own already-correct non-accumulating formula)
    // and only narrow via `Buffer::cast_to` at the very end. Real numpy does
    // NOT do this -- it casts `start`/`step` DOWN to the target dtype FIRST,
    // then performs the `start + i*step` walk using THAT narrower type's own
    // arithmetic (which rounds at every multiply/add, not just once at the
    // end). The difference is only visible when the target is narrower than
    // f64 and the true mathematical result isn't exactly representable at
    // that width: `np.arange(0, 1, 0.3, dtype=np.float16)`'s last element is
    // `0.9004` (`float16(0.3) == 0.300048828125` exactly, and `float16(3) *
    // float16(0.300048828125)` rounds to `0.9004` when computed in float16),
    // but computing `3 * 0.3` in f64 first (`0.8999999999999999`) and only
    // then narrowing to float16 gives `0.9` -- a different, wrong bit
    // pattern (`0x38cd` vs numpy's `0x3b34`) -- verified against real numpy
    // 2.5.1 (found via this task's own out-of-corpus probe; the existing
    // corpus never varied `dtype=` against a `step` whose f64 arithmetic and
    // narrowed-arithmetic results diverge). `half::f16`'s and `f32`'s own
    // `Add`/`Mul` operators are used directly here (rather than hand-rolled
    // float64 emulation) specifically because they reproduce this same
    // round-at-every-operation behavior; `f64` targets are exact matches to
    // the existing `arange_values` path (`f64` narrowed to `f64` is a no-op),
    // so only `F16`/`F32` need this separate branch.
    if dt.is_integer() {
        let buf = arange_int_values(start_v, step_v, n, dt);
        let inner = NdArray::from_buffer(buf, vec![n], Order::C).map_err(to_py_err)?;
        return Ok(PyArray { inner });
    }

    let buf = match dt {
        DType::F16 => {
            let start16 = half::f16::from_f64(gen_start);
            let step16 = half::f16::from_f64(gen_step);
            let values: Vec<half::f16> = (0..n)
                .map(|i| start16 + half::f16::from_f64(i as f64) * step16)
                .collect();
            Buffer::F16(values)
        }
        DType::F32 => {
            let start32 = gen_start as f32;
            let step32 = gen_step as f32;
            let values: Vec<f32> = (0..n).map(|i| start32 + (i as f32) * step32).collect();
            Buffer::F32(values)
        }
        _ => {
            let values = creation::arange_values(gen_start, gen_step, n);
            Buffer::F64(values).cast_to(dt)
        }
    };
    let inner = NdArray::from_buffer(buf, vec![n], Order::C).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

/// Accepts a bare bool/int/float scalar, an `anionpy.ndarray`, or a nested
/// Python list/tuple (via `array_impl`, the same coercion `asarray` uses)
/// and returns a 0-d-or-more `f64` `NdArray` -- `linspace`'s own `start`/
/// `stop` coercion, since neither has to be scalar in real numpy (`np.
/// linspace([0,10],[1,20])` is legal and array-valued, verified against
/// real numpy 2.5.1).
fn to_ndarray_f64(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(pyarr) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(pyarr.inner.cast_to(DType::F64));
    }
    if let Ok(v) = obj.extract::<f64>() {
        return NdArray::from_buffer(Buffer::F64(vec![v]), vec![], Order::C).map_err(to_py_err);
    }
    let arr = crate::array_impl(obj, None)?;
    Ok(arr.cast_to(DType::F64))
}

/// Extract a contiguous `f64` `NdArray`'s data as a flat `Vec<f64>` in C
/// order. `a` must already be C-contiguous AND `f64` (callers here always
/// pass the output of `.to_contiguous()` on an already-`cast_to(F64)`
/// array), so this is a plain pattern match, not a general gather.
fn f64_vec_of(a: &NdArray) -> Vec<f64> {
    match a.buffer() {
        Buffer::F64(v) => v.clone(),
        _ => unreachable!("f64_vec_of called on a non-f64 buffer"),
    }
}

/// numpy's `np.linspace(start, stop, num=50, endpoint=True, retstep=False,
/// dtype=None, axis=0, *, device=None)`. `start`/`stop` are broadcast to a
/// common shape (`ionp_core::shape::broadcast_shapes`, the same primitive
/// `broadcast_to`/binary ufuncs already use) BEFORE the elementwise
/// linspace formula runs -- `ionp_core::creation::linspace_values_nd`
/// computes the interpolation itself (per-element, in Rust) and always
/// lays the new `num` axis out as the OUTERMOST axis; `axis=` is then
/// satisfied by `moveaxis`ing that axis to the caller's requested
/// position, reusing the existing view primitive rather than adding new
/// axis-placement math here. Verified against real numpy 2.5.1 across
/// scalar/scalar (old behavior, unchanged), array/scalar, scalar/array,
/// array/array (matching and broadcastable shapes), `axis=0` (default),
/// positive and negative non-zero `axis=`, and `retstep=True` with an
/// array-shaped step output.
#[pyfunction]
#[pyo3(signature = (start, stop, num=50, endpoint=None, retstep=None, dtype=None, axis=0, *, device=None))]
fn linspace<'py>(
    py: Python<'py>,
    start: &Bound<'_, PyAny>,
    stop: &Bound<'_, PyAny>,
    num: i64,
    endpoint: Option<&Bound<'_, PyAny>>,
    retstep: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    axis: isize,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    // Plain Python truthiness for both (measured 2026-08-02, same as
    // `svd`).
    let endpoint = match endpoint {
        None => true,
        Some(v) => v.is_truthy()?,
    };
    let retstep = match retstep {
        None => false,
        Some(v) => v.is_truthy()?,
    };
    check_device(device)?;
    if num < 0 {
        return Err(PyValueError::new_err(format!(
            "Number of samples, {num}, must be non-negative."
        )));
    }
    let start_nd = to_ndarray_f64(start)?;
    let stop_nd = to_ndarray_f64(stop)?;
    let bshape = ionp_core::shape::broadcast_shapes(start_nd.shape(), stop_nd.shape()).map_err(to_py_err)?;
    let start_flat = f64_vec_of(&creation::broadcast_to(&start_nd, &bshape).map_err(to_py_err)?.to_contiguous());
    let stop_flat = f64_vec_of(&creation::broadcast_to(&stop_nd, &bshape).map_err(to_py_err)?.to_contiguous());
    let dt = resolve_dtype(dtype, DType::F64)?;
    let (mut flat, steps) = creation::linspace_values_nd(&start_flat, &stop_flat, num as usize, endpoint);
    // numpy's `linspace` with an explicit integer/bool `dtype=` does NOT
    // compute the float64 sequence and then truncate-toward-zero (the
    // ordinary `Buffer::cast_to` float->int narrowing this crate uses
    // everywhere else) -- it FLOORS first. Verified against real numpy
    // 2.5.1: `np.linspace(-5, 5, 11, endpoint=False, dtype=np.int8)` ->
    // `[-5,-5,-4,-3,-2,-1,0,1,2,3,4]`, but the same values computed as
    // float64 and truncated give `[-5,-4,-3,-2,-1,0,0,1,2,3,4]` (differs at
    // every negative fractional element, since floor(-4.09) == -5 but
    // trunc(-4.09) == -4). Found via this task's own out-of-corpus probe
    // (the existing corpus never crossed a non-integer step with an
    // explicit narrower/integer dtype). Only affects genuinely-narrowing
    // integer targets; float/complex targets are unaffected (`cast_to`
    // there is not a truncation at all).
    //
    // BUG FOUND AND FIXED 2026-08-01: this used to also floor for `dtype=
    // bool` (`dt.is_integer() || dt == DType::Bool`), which is wrong --
    // numpy's `linspace(..., dtype=bool)` casts each element by the
    // NONZERO-ness of the un-floored float value (exactly like every other
    // float->bool cast in this crate, e.g. `weak_scalar_buffer`'s `v !=
    // 0.0`), not by flooring first and then checking nonzero-ness of the
    // floored integer. `np.linspace(0, 1, 5, dtype=bool)` -> `[False, True,
    // True, True, True]` (`0, 0.25, 0.5, 0.75, 1` are each individually
    // nonzero except the first), but flooring first collapses `0.25/0.5/
    // 0.75` all down to `0` before the bool cast, giving the wrong `[False,
    // False, False, False, True]` (found via this task's own out-of-corpus
    // probe; verified against real numpy 2.5.1). `Buffer::cast_to`'s own
    // F64->Bool path already does the correct `x != 0.0` -- the fix is
    // simply to stop pre-flooring for a `Bool` target and let `cast_to`
    // (called below, unconditionally, on `flat`) do that cast directly on
    // the original float value.
    if dt.is_integer() {
        for v in flat.iter_mut() {
            if v.is_finite() {
                *v = v.floor();
            }
        }
    }
    let mut out_shape = vec![num as usize];
    out_shape.extend_from_slice(&bshape);
    let inner0 = NdArray::from_buffer(Buffer::F64(flat), out_shape, Order::C).map_err(to_py_err)?;
    let out_ndim = inner0.ndim();
    // 2026-08-01: numpy's own out-of-range message here is the
    // `msg_prefix="destination"`-labeled AxisError (verified directly:
    // `np.linspace(a, b, 3, axis=5)` -> 'destination: axis 5 is out of
    // bounds...'), since this axis placement is implemented by reusing
    // moveaxis's own "destination" labeling internally -- see
    // normalize_axis_labeled.
    let ax = normalize_axis_labeled(axis, out_ndim, "destination")?;
    let moved = if ax == 0 {
        inner0
    } else {
        creation::moveaxis(&inner0, &[0], &[ax]).map_err(to_py_err)?
    };
    let inner = moved.cast_to(dt);
    let arr = Py::new(py, PyArray { inner })?.into_bound(py).into_any();
    if retstep {
        let step_obj: Bound<'py, PyAny> = if bshape.is_empty() {
            steps[0].into_pyobject(py)?.into_any()
        } else {
            let step_arr = NdArray::from_buffer(Buffer::F64(steps), bshape.clone(), Order::C).map_err(to_py_err)?;
            Py::new(py, PyArray { inner: step_arr })?.into_bound(py).into_any()
        };
        let tup = PyTuple::new(py, [arr, step_obj])?;
        Ok(tup.into_any())
    } else {
        Ok(arr)
    }
}

// BUG FOUND AND FIXED 2026-08-01: real numpy's `eye` signature is
// `(N, M=None, k=0, dtype=<class 'float'>, order='C', *, device=None,
// like=None)` -- `N`/`M` are POSITIONAL_OR_KEYWORD (no `/` before them),
// so `np.eye(N=3, M=5)` genuinely works on real numpy. The previous
// binding used lowercase `n=`/`m=`, which raised `TypeError: eye() got
// an unexpected keyword argument 'N'` for the exact call real numpy
// accepts -- found via this task's audit re-run (inspect.signature
// diffing every declared-passing item against real numpy), not part of
// the originally assigned gap list. Verified fix: `anionpy.eye(N=3, M=5)`
// now succeeds and matches `np.eye(N=3, M=5)`.
#[allow(non_snake_case)]
#[pyfunction]
#[pyo3(signature = (N, M=None, k=0, dtype=None, order=None, *, device=None, like=None))]
fn eye(
    N: i64,
    M: Option<i64>,
    k: isize,
    dtype: Option<&Bound<'_, PyAny>>,
    order: Option<&str>,
    device: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    check_device(device)?;
    check_like(like)?;
    let n = N;
    let m = M;
    if n < 0 || m.unwrap_or(0) < 0 {
        return Err(PyValueError::new_err("negative dimensions are not allowed"));
    }
    let n_rows = n as usize;
    let n_cols = m.unwrap_or(n) as usize;
    let dt = resolve_dtype(dtype, DType::F64)?;
    let ord = order_from_pyobj(order)?;
    let values = creation::eye_values(n_rows, n_cols, k);
    let buf = Buffer::F64(values).cast_to(dt);
    // BUG FOUND AND FIXED 2026-08-01: `eye_values` always produces its flat
    // buffer in row-major (C) LOGICAL order, but `NdArray::from_buffer`
    // (array.rs) does not physically rearrange data for `Order::F` -- it
    // just tags the SAME buffer with F-strides, i.e. it assumes the buffer
    // is ALREADY laid out column-major. Passing `ord=Order::F` straight
    // into `from_buffer` here therefore reinterpreted C-order identity
    // data as if it were F-order data, producing both wrong VALUES (not
    // merely a layout/strides mismatch) and a result that was actually
    // C-contiguous, not F-contiguous as `order='F'` requires -- caught by
    // the `eye/order_F` differential case (`(3, 5)`), which showed
    // `anionpy.eye(3, 5, order='F')` returning `[[1,0,1,0,1],[0,0,0,0,0],
    // [0,0,0,0,0]]` instead of the correct identity pattern, and
    // `.flags.F_CONTIGUOUS == False`. Fix: always build the buffer as a
    // real `Order::C` array first (matching `eye_values`'s actual layout),
    // then use the SAME `to_contiguous_order` primitive `asarray`/`copy`/
    // `ascontiguousarray` already rely on to physically transpose into a
    // genuine F-contiguous layout when requested -- verified after the fix
    // that `anionpy.eye(3, 5, order='F')` matches numpy's values exactly and
    // reports `F_CONTIGUOUS=True, C_CONTIGUOUS=False`, `OWNDATA=True`.
    let c_inner = NdArray::from_buffer(buf, vec![n_rows, n_cols], Order::C).map_err(to_py_err)?;
    let inner = match ord {
        Order::C => c_inner,
        Order::F => c_inner.to_contiguous_order("F").map_err(to_py_err)?,
    };
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (n, dtype=None, *, like=None))]
fn identity(n: i64, dtype: Option<&Bound<'_, PyAny>>, like: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    check_like(like)?;
    eye(n, None, 0, dtype, None, None, None)
}

// ---------------------------------------------------------------------------
// asarray / copy / ascontiguousarray -- thin reuses of existing machinery.
// ---------------------------------------------------------------------------

/// numpy's tri-state `copy=` (`None`/`True`/`False`, shared by `asarray`
/// and `reshape`) is resolved through real numpy's own `PyArray_BoolConverter`
/// -- NOT a plain `isinstance(copy, bool)` check, and NOT plain-truthy
/// either. Measured live against numpy 2.5.1 (2026-08-02): `copy=1`,
/// `copy=1.5`, `copy=[1]`, `copy={1:2}`, a nonzero-`__len__` object, and
/// `np.bool_(True)` are all accepted and dispatch exactly like `True`
/// (truthy); `copy=0`, `copy=0.0`, `copy=[]`, `copy={}`, a zero-`__len__`
/// object, and `np.bool_(False)` all dispatch like `False` (falsy); a
/// raising-`__bool__` object propagates its exception unchanged. The ONE
/// carve-out: a `str` (empty or not) is explicitly rejected with its own
/// `ValueError` regardless of truthiness -- `"strings are not allowed for
/// 'copy' keyword. Use True/False/None instead."` (exact wording verified
/// live) -- str is otherwise falsy/truthy-capable in Python but numpy's
/// converter special-cases it before the truthiness check ever runs.
/// `None` is left as `None` (means "copy only if needed", handled
/// separately by each caller).
fn resolve_copy_flag(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<bool>> {
    let obj = match obj {
        None => return Ok(None),
        Some(o) => o,
    };
    if obj.is_none() {
        return Ok(None);
    }
    if obj.is_instance_of::<pyo3::types::PyString>() {
        return Err(PyValueError::new_err(
            "strings are not allowed for 'copy' keyword. Use True/False/None instead.",
        ));
    }
    Ok(Some(obj.is_truthy()?))
}

/// numpy's `np.asarray(a, dtype=None, order=None, *, device=None,
/// copy=None, like=None)`. `order` here is a MUCH looser check than
/// `zeros`'s own `order_from_pyobj` -- `asarray` has an existing array to
/// fall back on for `'A'`/`'K'`'s meaning (unlike `zeros`, which has
/// nothing to key them off and refuses both outright, verified against
/// real numpy 2.5.1), so this reuses `NdArray::to_contiguous_order`, the
/// same primitive `copy`/`ndarray.copy` already use for exactly that
/// prototype-relative resolution. `copy` is numpy 2.x's tri-state: `None`
/// (default, copy only if needed for dtype/layout), `False` (never copy;
/// raise if a copy would be unavoidable), `True` (always copy) --
/// verified against real numpy 2.5.1's `np.asarray(a, copy=False)` on an
/// already-matching array (no copy needed, no error) vs. one needing an
/// order change it cannot honor without copying (raises `ValueError:
/// Unable to avoid copy while creating an array as requested.`, exact
/// wording confirmed against 2.5.1, distinct from `reshape`'s copy=False
/// message).
#[pyfunction]
#[pyo3(signature = (a, dtype=None, order=None, *, device=None, copy=None, like=None))]
fn asarray(
    a: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    order: Option<&Bound<'_, PyAny>>,
    device: Option<&Bound<'_, PyAny>>,
    copy: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    let copy = resolve_copy_flag(copy)?;
    check_device(device)?;
    check_like(like)?;
    // BUG FOUND AND FIXED 2026-08-03 (Monday): `order` used to be typed
    // `Option<&str>` and read directly (`order.unwrap_or("K")`) with no
    // validation at all -- any string, valid or not ("Z", "", "corder"),
    // silently reached the `matches!(target_order, "C" | "F")` checks
    // below, which just fall through to "no layout change" for anything
    // that isn't literally "C" or "F". Real numpy raises `ValueError`
    // (`crate::check_ufunc_order_kwarg`'s exact wording/case-insensitive/
    // bytes-accepting contract, already used by `copy`/`ndarray.copy`
    // above and `reshape`/`ravel`/`flatten` elsewhere) for anything outside
    // {C, F, A, K}. Routed through the same validator so `asarray` stops
    // silently accepting garbage `order=` numpy rejects.
    let order_letter = crate::check_ufunc_order_kwarg(order)?;
    let target_order_owned = order_letter.map(|c| c.to_string()).unwrap_or_else(|| "K".to_string());
    let target_order = target_order_owned.as_str();
    // BUG FOUND AND FIXED 2026-08-01: `crate::array_impl` (see its own
    // doc/`ndarray_from_numpy` in lib.rs) ALWAYS physically copies a real
    // external `numpy.ndarray` source into a fresh ionp-owned buffer --
    // and that ingestion only re-lays-out the genuinely-Fortran case,
    // silently normalizing anything that is neither C- nor F-contiguous
    // (e.g. a strided middle-axis slice view) into a fresh C-contiguous
    // buffer. Computing `needs_reorder` from the POST-conversion `arr`
    // (the previous code, now replaced) therefore always sees a
    // C-contiguous result and can never observe that the ORIGINAL numpy
    // source required a copy to satisfy `order='C'`/`'F'` -- so
    // `asarray(non_contiguous_view, order='C', copy=False)` silently
    // returned a copy instead of raising `ValueError`, exactly like real
    // numpy does. Verified via a differential case
    // (`view/3d_middle_slice/copy_false_order_C`) that failed before this
    // fix and passes after it. Fix: when the source is a genuine external
    // numpy array (not already an anionpy `PyArray`, which never loses
    // layout since no lossy ingestion happens for it), read ITS OWN
    // `c_contiguous`/`f_contiguous` flags -- the ground truth numpy
    // itself uses -- before any conversion happens, instead of asking the
    // already-normalized post-conversion buffer.
    let needs_reorder = if extract_pyarray(a).is_ok() {
        // Already an anionpy array: ingestion is a real view, no layout loss.
        None // resolved below from `arr` once built.
    } else if let Ok(flags) = a.getattr("flags") {
        let c = flags.getattr("c_contiguous").and_then(|v| v.extract::<bool>()).unwrap_or(true);
        let f = flags.getattr("f_contiguous").and_then(|v| v.extract::<bool>()).unwrap_or(true);
        Some(matches!(target_order, "C") && !c || matches!(target_order, "F") && !f)
    } else {
        // Freshly built from a Python list/tuple/scalar: always
        // C-contiguous immediately after construction, no information
        // lost by checking the post-conversion buffer for those sources.
        None
    };
    // BUG FOUND AND FIXED 2026-08-01: `copy=False` combined with a `dtype=`
    // that actually differs from the source's own dtype must raise
    // `ValueError` -- a dtype change is never copy-free -- but the old code
    // called `crate::array_impl(a, dtype)` (which performs the cast
    // internally) BEFORE any copy=False check existed for dtype at all, so
    // a genuine dtype-widening/narrowing request with `copy=False` silently
    // succeeded instead of raising. Verified against real numpy 2.5.1:
    // `np.asarray(np.array(True), dtype=np.float64, copy=False)` raises
    // `ValueError('Unable to avoid copy while creating an array as
    // requested.\n...')` -- the SAME long message (with the "replace it
    // with np.asarray(obj)" migration-guide text) `zeros_like`'s sibling
    // layout-only case below does NOT use. Also verified: when only the
    // LAYOUT (not dtype) needs to change, real numpy instead raises the
    // SHORTER, DIFFERENT message `ValueError('Unable to avoid copy while
    // creating a new array.')` -- no migration-guide text at all. These are
    // two genuinely distinct wordings for two distinct causes (confirmed by
    // probing every combination of {dtype change, layout change, both}
    // against real numpy 2.5.1), not a single message with cosmetic
    // variation, so `dtype_changes` must be checked and reported with its
    // own distinct message before falling through to the layout-only case.
    // SEV-1 fix (2026-08-02): this used to always call `crate::array_impl(a,
    // None)` (untyped construction) and, when `dtype=` was given, cast the
    // result afterward through an UNCHECKED `cast_to` -- exactly the same
    // defect `array()` itself had before its own `array_impl`/
    // `ndarray_from_pylist_typed` fix (see that function's doc comment in
    // lib.rs). Two symptoms, both from the one cause: (1) a Python-int list
    // headed for a narrower integer dtype silently wrapped instead of
    // raising numpy's own `OverflowError`
    // (`asarray([256], dtype='uint8')` -> `array([0], uint8)` where real
    // numpy raises `OverflowError('Python integer 256 out of bounds for
    // uint8')`), and (2) a huge Python int headed for a FLOAT/complex dtype
    // was rejected during the untyped construction step -- which tries to
    // fit it into `i64`/`u64` first -- before the float cast that would
    // have carried it through ever ran (`asarray([2**64], dtype='float64')`
    // -> spurious `OverflowError('Python int too large to convert to C
    // long')` where real numpy returns `array([1.84467441e+19])`).
    //
    // Fixed by threading `dtype` directly into `array_impl` for the actual
    // build below, reusing the exact bounds-checked/float-bypass machinery
    // `array()` already goes through via `ndarray_from_pylist_typed` --
    // not a parallel mechanism. `dtype_changes` (needed only for the
    // `copy=False` diagnostic just below) is answered by a SEPARATE,
    // untyped `array_impl(a, None)` probe -- the same "what would `dtype=
    // None` have produced" question the old code answered by construction
    // as a side effect. That probe can itself legitimately hit the same
    // untyped-construction overflow described above (a value that fits the
    // requested float target but fits neither `i64` nor `u64`); when it
    // does, the requested concrete dtype can only differ from the
    // (unrepresentable-without-`dtype=object`) natural one, so that failure
    // is treated as `dtype_changes = true` rather than propagated -- it
    // must never leak out and mask the successful typed build above it.
    let arr = crate::array_impl(a, dtype)?;
    let dtype_changes = match dtype {
        Some(d) => {
            let target = dtype_from_pyobj(d)?;
            match crate::array_impl(a, None) {
                Ok(natural) => target != natural.dtype(),
                Err(_) => true,
            }
        }
        None => false,
    };
    if dtype_changes && copy == Some(false) {
        return Err(PyValueError::new_err(
            "Unable to avoid copy while creating an array as requested.\n\
             If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` to allow a copy when needed (no behavior change in NumPy 1.x).\n\
             For more details, see https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.",
        ));
    }
    let needs_reorder = needs_reorder.unwrap_or_else(|| {
        matches!(target_order, "C") && !arr.is_c_contiguous()
            || matches!(target_order, "F") && !arr.is_f_contiguous()
    });
    let inner = if needs_reorder {
        if copy == Some(false) {
            // BUG FOUND AND FIXED 2026-08-01: which of the two "can't avoid
            // a copy" messages real numpy 2.5.1 raises for a LAYOUT-only
            // conflict (dtype unchanged) depends, bizarrely but verifiably,
            // on whether `dtype=` was passed EXPLICITLY at all -- not on
            // whether it changes anything. Probed directly: given the same
            // non-contiguous source and `order='C', copy=False`,
            // `np.asarray(a, dtype=None, ...)` (the default) and
            // `np.asarray(a, ...)` (kwarg omitted entirely) both raise the
            // LONG message (with the "replace it with np.asarray(obj)"
            // migration-guide text) -- while `np.asarray(a, dtype=a.dtype,
            // ...)`, an EXPLICIT dtype that is byte-for-byte the array's own
            // dtype and therefore changes nothing, raises the SHORT message
            // `'Unable to avoid copy while creating a new array.'` instead.
            // So: an explicitly-passed dtype that doesn't actually change
            // anything still flips which message fires. Matched here by
            // keying off `dtype.is_some()` (explicit-and-unchanged, since
            // `dtype_changes` is already false in this branch) rather than
            // any property of the resulting array.
            if dtype.is_some() {
                return Err(PyValueError::new_err(
                    "Unable to avoid copy while creating a new array.",
                ));
            }
            return Err(PyValueError::new_err(
                "Unable to avoid copy while creating an array as requested.\n\
                 If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` to allow a copy when needed (no behavior change in NumPy 1.x).\n\
                 For more details, see https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.",
            ));
        }
        arr.to_contiguous_order(target_order).map_err(to_py_err)?
    } else if copy == Some(true) {
        arr.to_contiguous_order(target_order).map_err(to_py_err)?
    } else {
        arr
    };
    Ok(PyArray { inner })
}

/// numpy's top-level `np.copy(a, order='K', subok=False)`. Same `order`
/// semantics as `ndarray.copy()` (`lib.rs`) -- default `'K'`, matching the
/// input's own layout as closely as possible -- via the same
/// `to_contiguous_order` primitive (`array.rs`); `subok` isn't meaningful
/// here (anionpy has no subclassing) and isn't accepted as a call form numpy
/// itself would reject either way.
#[pyfunction]
#[pyo3(signature = (a, order=None, subok=None))]
fn copy(a: &Bound<'_, PyAny>, order: Option<&Bound<'_, PyAny>>, subok: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    check_subok(subok);
    let arr = extract_or_ingest_ndarray(a)?;
    let order_letter = crate::check_ufunc_order_kwarg(order)?;
    let ord_string = order_letter.map(|c| c.to_string()).unwrap_or_else(|| "K".to_string());
    let inner = arr.to_contiguous_order(ord_string.as_str()).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

#[pyfunction]
#[pyo3(signature = (a, dtype=None, *, like=None))]
fn ascontiguousarray(
    a: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    like: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    check_like(like)?;
    // SEV-1 fix (2026-08-02): this had the exact same defect as `asarray`
    // (see that function's own doc comment) -- `extract_or_ingest_ndarray`
    // builds via untyped `array_impl(a, None)`, so a list/tuple source with
    // `dtype=` was narrowed afterward through an unchecked `cast_to`
    // (silent wraparound: `ascontiguousarray([256], dtype='uint8')` ->
    // `array([0], uint8)` instead of numpy's `OverflowError`) or had a huge
    // int headed for a float target spuriously rejected during the untyped
    // construction step (`ascontiguousarray([2**64], dtype='float64')`
    // raised where real numpy returns `array([1.84467441e+19])`). Fixed the
    // same way: reuse `crate::array_impl(a, dtype)` directly, which is a
    // strict superset of `extract_or_ingest_ndarray` (that helper's own
    // fallback arm is just `array_impl(obj, None)`) and gives list/tuple
    // sources the bounds-checked/float-bypass construction `array()`
    // already uses via `ndarray_from_pylist_typed`, while still doing the
    // correct unchecked `.astype()`-style wraparound cast for genuine
    // ndarray/`__array__` sources.
    let inner = crate::array_impl(a, dtype)?.to_contiguous();
    // `NdArray::to_contiguous` is a generic "gather into a fresh
    // C-contiguous buffer" primitive shared by dozens of call sites
    // across `ionp-core` (sort/setops/manip/...), several of which
    // deliberately want its size-0 output to KEEP the plain
    // `shape::c_strides` formula's computed strides rather than zero
    // them (e.g. `roll`'s `axis=None` flatten+reshape composition,
    // verified against real numpy 2.5.1: `np.roll(np.zeros((0,3)),
    // 1).strides == (24, 8)`, not all-zero) -- so that zeroing can't be
    // done inside `to_contiguous` itself without breaking those. This
    // function's own contract is different: real numpy's
    // `ascontiguousarray`, even though it always documents/behaves as
    // "at least 1-d", genuinely IS a freshly-allocated array whenever a
    // copy actually happens, and freshly-allocated size-0 arrays get
    // all-zero strides regardless of dtype or requested shape (verified
    // directly: `np.ascontiguousarray(np.zeros((0,3))).strides ==
    // (0, 0)`, `np.zeros((3,0))` -> `(0, 0)`, `np.zeros((2,0,4))` ->
    // `(0, 0, 0)`), so the zeroing belongs HERE, applied only to this
    // function's own result.
    let inner = if inner.size() == 0 {
        inner.zero_strides()
    } else {
        inner
    };
    // numpy's `ascontiguousarray` documents (and, verified against 2.5.1,
    // actually does) "at least 1-d" -- a 0-d input comes back shape (1,),
    // unlike every other view/copy function in this file, which all
    // preserve 0-d shape faithfully. Found via this task's differential
    // corpus (every `sweep/0d/*` case failed a shape-mismatch before this
    // fix). `reshape_with_order` can't fail here: a size-1 buffer always
    // reshapes to `[1]`.
    let inner = if inner.ndim() == 0 {
        inner.reshape_with_order(&[1], "C").map_err(to_py_err)?
    } else {
        inner
    };
    Ok(PyArray { inner })
}

// ---------------------------------------------------------------------------
// Top-level view wrappers. Each is a thin dispatcher onto either an
// existing `NdArray` method (reshape/transpose already prove view-vs-copy
// correctness in array.rs) or the new `ionp_core::creation` view functions
// (squeeze/expand_dims/swapaxes/moveaxis/broadcast_to).
// ---------------------------------------------------------------------------

/// numpy 2.x's `copy=` tri-state, shared by both this top-level `reshape`
/// and `ndarray.reshape` (`lib.rs`): `None` (default, copy only if the
/// requested shape/order can't be satisfied by a view), `True` (always
/// return a fresh-buffer copy even if a view would do), `False` (never
/// copy; raise `ValueError: Unable to avoid creating a copy while
/// reshaping.` if a view genuinely can't satisfy the request) -- exact
/// message verified against real numpy 2.5.1, and distinct from
/// `asarray`'s own copy=False message. `reshape_with_order` already
/// returns a real view when one exists (see array.rs); `shares_buffer_with`
/// is how this tells the two cases apart from outside `ionp-core` without
/// widening the `buffer` field's own visibility.
pub(crate) fn apply_reshape_copy(orig: &NdArray, reshaped: NdArray, copy: Option<bool>) -> PyResult<NdArray> {
    let is_view = orig.shares_buffer_with(&reshaped);
    match copy {
        Some(false) if !is_view => Err(PyValueError::new_err(
            "Unable to avoid creating a copy while reshaping.",
        )),
        Some(true) if is_view => Ok(reshaped.to_contiguous()),
        _ => Ok(reshaped),
    }
}

#[pyfunction]
#[pyo3(signature = (a, shape, order=None, *, copy=None))]
fn reshape(
    a: &Bound<'_, PyAny>,
    shape: &Bound<'_, PyAny>,
    order: Option<&Bound<'_, PyAny>>,
    copy: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyArray>> {
    let copy = resolve_copy_flag(copy)?;
    let order_letter = crate::check_ufunc_order_kwarg(order)?;
    let order_string = order_letter.map(|c| c.to_string()).unwrap_or_else(|| "C".to_string());
    let order: &str = order_string.as_str();
    // BUG FOUND AND FIXED 2026-08-01: this used `extract_pyarray(a)?`,
    // which requires `a` to ALREADY be an `anionpy.ndarray` -- a real
    // `numpy.ndarray` fed to `anionpy.reshape()` failed PyO3's raw `extract`
    // with the generic, wrong-for-this-case error
    // `TypeError("'numpy.ndarray' object is not an instance of
    // 'anionpy.ndarray'")` instead of being ingested like every sibling
    // function (`ravel`/`transpose`/`broadcast_to`, all just above/below,
    // already correctly use `extract_or_ingest_ndarray`). Found via
    // independent foreign-numpy-input probing: `anionpy.reshape(np.zeros(4),
    // (2, 2))` crashed with that TypeError where real
    // `np.reshape(np.zeros(4), (2, 2))` succeeds. Fixed by switching to
    // the same `extract_or_ingest_ndarray` every other shape-op here uses.
    let arr = extract_or_ingest_ndarray(a)?;
    let dims = if let Ok(seq) = shape.extract::<Vec<isize>>() {
        seq
    } else {
        vec![shape.extract::<isize>()?]
    };
    // See `PyArray::reshape` in `lib.rs` for why a raw negative
    // ("infer this axis") dim must skip `NdArray`'s §3a identity
    // short-circuit -- numpy's own identity check runs on the unresolved
    // requested dims, and a `-1` never equals a concrete old dimension
    // there even when it numerically resolves to one.
    let resolved_dims = dims_to_usize_placeholder(&dims, &arr)?;
    let reshaped = if dims.iter().any(|&d| d < 0) {
        arr.reshape_with_order_no_identity_shortcut(&resolved_dims, order).map_err(to_py_err)?
    } else {
        arr.reshape_with_order(&resolved_dims, order).map_err(to_py_err)?
    };
    let inner = apply_reshape_copy(&arr, reshaped, copy)?;
    crate::wrap_reshape_result(a.py(), a, &arr, inner, order)
}

/// `NdArray::reshape_with_order` takes an already-resolved `&[usize]` (no
/// `-1` inference) -- resolve a single `-1` (or leave sizes as-is) here,
/// matching `reshape`'s existing `-1`-anywhere-in-tuple inference rule
/// (mirrors `ndarray.reshape`'s own resolution, see lib.rs's
/// `PyArray::reshape` method just above this module's registration point).
fn dims_to_usize_placeholder(dims: &[isize], arr: &NdArray) -> PyResult<Vec<usize>> {
    let total = arr.size();
    let neg_count = dims.iter().filter(|&&d| d < 0).count();
    if neg_count > 1 {
        return Err(PyValueError::new_err("can only specify one unknown dimension"));
    }
    if neg_count == 0 {
        return dims
            .iter()
            .map(|&d| {
                if d < 0 {
                    unreachable!()
                } else {
                    Ok(d as usize)
                }
            })
            .collect();
    }
    let known: usize = dims.iter().filter(|&&d| d >= 0).map(|&d| d as usize).product();
    let inferred = if known == 0 { 0 } else { total / known };
    Ok(dims
        .iter()
        .map(|&d| if d < 0 { inferred } else { d as usize })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (a, order=None))]
fn ravel(a: &Bound<'_, PyAny>, order: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyArray>> {
    let arr = extract_or_ingest_ndarray(a)?;
    let order_letter = crate::check_ufunc_order_kwarg(order)?;
    let ord_string = order_letter.map(|c| c.to_string()).unwrap_or_else(|| "C".to_string());
    let inner = crate::ravel_view_or_copy(&arr, &ord_string)?;
    wrap_shape_view(a.py(), a, &arr, inner)
}

#[pyfunction]
#[pyo3(signature = (a, axes=None))]
fn transpose(a: &Bound<'_, PyAny>, axes: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyArray>> {
    let arr = extract_or_ingest_ndarray(a)?;
    let inner = match axes {
        None => arr.transpose(),
        Some(ax) => {
            let dims: Vec<isize> = if let Ok(seq) = ax.extract::<Vec<isize>>() {
                seq
            } else {
                vec![ax.extract::<isize>()?]
            };
            // 2026-08-03: an out-of-range axis here used to surface as a
            // plain `IndexError` from `transpose_axes`, where real numpy
            // raises `numpy.exceptions.AxisError` -- the same defect
            // `squeeze`/`swapaxes`/`moveaxis`/`expand_dims` were fixed for
            // on 2026-08-01 and `transpose` was missed by. It went
            // unnoticed because `transpose` is declared exact and its own
            // case set never passed an out-of-range axis; `permute_dims`'s
            // new cases found it. AxisError subclasses both ValueError and
            // IndexError, so no `except` clause that used to catch this
            // stops catching it -- but a `type()` check (which this
            // project's harness does) sees the difference.
            //
            // Order matters and is numpy's, verified directly:
            // `np.transpose(a3, (0,1,2,5))` is `ValueError: axes don't
            // match array` (LENGTH is checked first) while
            // `np.transpose(a3, (0,1,5))` is the AxisError. So the length
            // check is left to `transpose_axes` and only same-length
            // axis tuples are range-normalized here.
            if dims.len() == arr.ndim() {
                let ndim = arr.ndim();
                let normalized: Vec<isize> = dims
                    .iter()
                    .map(|&ax| normalize_axis(ax, ndim).map(|n| n as isize))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(to_py_err)?;
                arr.transpose_axes(&normalized).map_err(to_py_err)?
            } else {
                arr.transpose_axes(&dims).map_err(to_py_err)?
            }
        }
    };
    wrap_shape_view(a.py(), a, &arr, inner)
}

/// numpy's own axis normalization for every axis-taking function in this
/// file: negative-wraps, then raises the REAL `numpy.exceptions.AxisError`
/// (via `IonpError::AxisError` -> `to_py_err`, the same machinery the six
/// `ndarray` reduction methods already use -- see `error.rs`) for anything
/// still out of `[-ndim, ndim)`, rather than a plain `IndexError`. AxisError
/// subclasses BOTH `ValueError` and `IndexError`, so a caller's `except
/// ValueError:` must catch it same as anionpy's old plain `IndexError` did, but
/// a caller's `except numpy.exceptions.AxisError:` (or a `type()` identity
/// check, which is what this project's own differential harness uses) would
/// not -- verified against real numpy 2.5.1 that `squeeze`/`swapaxes`/
/// `moveaxis`/`expand_dims` all raise this exact type for an out-of-range
/// axis, not a plain `IndexError`.
fn normalize_axis(ax: isize, ndim: usize) -> Result<usize, IonpError> {
    let n = ndim as isize;
    let norm = if ax < 0 { ax + n } else { ax };
    if norm < 0 || norm >= n {
        Err(IonpError::AxisError { axis: ax, ndim: Some(ndim) })
    } else {
        Ok(norm as usize)
    }
}

/// `normalize_axis`'s sibling for the two call sites (`moveaxis`'s own
/// `source`/`destination` params, and `linspace(..., axis=...)`'s internal
/// placement) where numpy's out-of-range message carries a `msg_prefix`
/// label (`"source: axis ..."` / `"destination: axis ..."`) instead of the
/// unprefixed form every other axis-taking function raises. Returns a
/// `PyErr` directly (not `IonpError`) since `axis_error_prefixed` already
/// builds numpy's real `AxisError` instance -- there's no unprefixed
/// `IonpError::AxisError` step to go through.
fn normalize_axis_labeled(ax: isize, ndim: usize, label: &str) -> Result<usize, PyErr> {
    let n = ndim as isize;
    let norm = if ax < 0 { ax + n } else { ax };
    if norm < 0 || norm >= n {
        Err(axis_error_prefixed(ax, ndim, label))
    } else {
        Ok(norm as usize)
    }
}

#[pyfunction]
fn swapaxes(a: &Bound<'_, PyAny>, axis1: isize, axis2: isize) -> PyResult<Py<PyArray>> {
    let arr = extract_or_ingest_ndarray(a)?;
    let ndim = arr.ndim();
    // 2026-08-01: numpy labels an out-of-range axis1/axis2 with its own
    // param name (`AxisError(axis, ndim, 'axis1')` / `('axis2')`), same
    // fix as `moveaxis`/`linspace`/`ndarray.swapaxes` above -- verified
    // directly against `np.swapaxes(np.zeros((2,3)), 5, 0)` vs `(...,0,5)`.
    let ax1 = normalize_axis_labeled(axis1, ndim, "axis1")?;
    let ax2 = normalize_axis_labeled(axis2, ndim, "axis2")?;
    let inner = creation::swapaxes(&arr, ax1, ax2).map_err(to_py_err)?;
    wrap_shape_view(a.py(), a, &arr, inner)
}

fn axes_list_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(seq) = obj.extract::<Vec<isize>>() {
        Ok(seq)
    } else {
        Ok(vec![obj.extract::<isize>()?])
    }
}

#[pyfunction]
fn moveaxis(a: &Bound<'_, PyAny>, source: &Bound<'_, PyAny>, destination: &Bound<'_, PyAny>) -> PyResult<Py<PyArray>> {
    let arr = extract_or_ingest_ndarray(a)?;
    let ndim = arr.ndim();
    let src_raw = axes_list_from_pyobj(source)?;
    let dst_raw = axes_list_from_pyobj(destination)?;
    if src_raw.len() != dst_raw.len() {
        return Err(PyValueError::new_err(
            "`source` and `destination` arguments must have the same number of elements",
        ));
    }
    // 2026-08-01: `source`/`destination` out-of-range axes get numpy's
    // real "source: ..."/"destination: ..." msg_prefix-labeled AxisError
    // (verified against `np.moveaxis(np.zeros((2,3)), 5, 0)` vs `(...,0,5)`
    // -- the plain unprefixed form the generic `normalize_axis` raises was
    // wrong here and failed under the harness's exact-message comparison).
    let src: Vec<usize> = src_raw
        .into_iter()
        .map(|a| normalize_axis_labeled(a, ndim, "source"))
        .collect::<Result<_, PyErr>>()?;
    let dst: Vec<usize> = dst_raw
        .into_iter()
        .map(|a| normalize_axis_labeled(a, ndim, "destination"))
        .collect::<Result<_, PyErr>>()?;
    let inner = creation::moveaxis(&arr, &src, &dst).map_err(to_py_err)?;
    wrap_shape_view(a.py(), a, &arr, inner)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn squeeze(a: &Bound<'_, PyAny>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyArray>> {
    let arr = extract_or_ingest_ndarray(a)?;
    let ndim = arr.ndim();
    // numpy special-cases a BARE (non-tuple) `axis=0`/`axis=-1` on a 0-d
    // array as an always-valid no-op, NOT covered by the general
    // `normalize_axis_tuple` out-of-range rule the rest of numpy's
    // axis-taking functions (and squeeze's own tuple-`axis` form) use --
    // verified against real numpy 2.5.1: `np.squeeze(np.array(5), axis=0)`
    // returns the array unchanged, while `np.squeeze(np.array(5),
    // axis=(0,))` (tuple form, same logical axis) DOES raise AxisError, and
    // `np.squeeze(np.array(5), axis=1)` (bare int, wrong axis) also raises.
    if ndim == 0 {
        if let Some(ax) = axis {
            if let Ok(single) = ax.extract::<isize>() {
                if single == 0 || single == -1 {
                    let inner = creation::squeeze(&arr, Some(&[])).map_err(to_py_err)?;
                    return crate::identity_if_unchanged(a.py(), a, &arr, inner);
                }
            }
        }
    }
    let axes: Option<Vec<usize>> = match axis {
        None => None,
        Some(ax) => Some(
            axes_list_from_pyobj(ax)?
                .into_iter()
                .map(|a| normalize_axis(a, ndim))
                .collect::<Result<_, IonpError>>()
                .map_err(to_py_err)?,
        ),
    };
    let inner = creation::squeeze(&arr, axes.as_deref()).map_err(to_py_err)?;
    // numpy returns `a` ITSELF when there is no length-1 axis to drop.
    crate::identity_if_unchanged(a.py(), a, &arr, inner)
}

#[pyfunction]
fn expand_dims(a: &Bound<'_, PyAny>, axis: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let arr = extract_or_ingest_ndarray(a)?;
    let old_ndim = arr.ndim();
    let new_ndim = old_ndim + axes_list_from_pyobj(axis)?.len();
    let raw = axes_list_from_pyobj(axis)?;
    let axes: Vec<usize> = raw
        .into_iter()
        .map(|a| normalize_axis(a, new_ndim))
        .collect::<Result<_, IonpError>>()
        .map_err(to_py_err)?;
    let inner = creation::expand_dims(&arr, &axes).map_err(to_py_err)?;
    Ok(PyArray { inner })
}

/// Shape rendering for `np.broadcast_to`'s own error message: `(3,)` for a
/// single dimension (trailing comma, like Python tuple repr) but `(2,3)`
/// -- comma with NO space -- for multiple dimensions, which is NOT how
/// Python's own tuple `repr()` formats it (that would be `(2, 3)`).
/// Verified char-for-char against real numpy 2.5.1's own `broadcast_to`
/// message text (a bespoke C-level formatter, not a `repr()` call) via
/// `np.broadcast_to(np.zeros((3,2)), (3,4))` -> `...requested shape (3,4)`
/// with no space -- caught only because the differential probe compared
/// exact message strings and found the space-joined version this function
/// used to produce (`(3, 4)`) did not match.
fn shape_repr(shape: &[usize]) -> String {
    if shape.is_empty() {
        return "()".to_string();
    }
    if shape.len() == 1 {
        return format!("({},)", shape[0]);
    }
    let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("({})", parts.join(","))
}

#[pyfunction]
#[pyo3(signature = (array, shape, subok=None))]
fn broadcast_to(array: &Bound<'_, PyAny>, shape: &Bound<'_, PyAny>, subok: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyArray>> {
    check_subok(subok);
    let arr = extract_or_ingest_ndarray(array)?;
    let dims = shape_from_pyobj(shape)?;
    // BUG FOUND AND FIXED 2026-08-01: `np.broadcast_to` raises its OWN
    // distinct error wording on a shape mismatch -- `'operands could not
    // be broadcast together with remapped shapes [original->remapped]:
    // SRC  and requested shape DST'` (note the double space before "and",
    // verified char-for-char against real numpy 2.5.1) when the source has
    // no more dims than the target but the trailing dims don't align/
    // aren't 1, or `'input operand has more dimensions than allowed by the
    // axis remapping'` when the source has MORE dims than the target --
    // which is a DIFFERENT message from the generic elementwise-broadcast
    // wording (`'operands could not be broadcast together with shapes A B
    // '`) that `ionp_core::creation::broadcast_to` reuses from the shared
    // binary-op broadcasting error path (`error.rs`, owned by ufunc.rs's
    // sibling agent, not touched here). Reproduced the check locally
    // instead of reformatting the shared error, to avoid touching
    // off-limits shared broadcasting machinery.
    if arr.shape().len() > dims.len() {
        return Err(PyValueError::new_err(
            "input operand has more dimensions than allowed by the axis remapping",
        ));
    }
    let src = arr.shape();
    let offset = dims.len() - src.len();
    let compatible = src
        .iter()
        .enumerate()
        .all(|(i, &d)| d == 1 || d == dims[offset + i]);
    if !compatible {
        return Err(PyValueError::new_err(format!(
            "operands could not be broadcast together with remapped shapes [original->remapped]: {}  and requested shape {}",
            shape_repr(src),
            shape_repr(&dims),
        )));
    }
    let inner = creation::broadcast_to(&arr, &dims).map_err(to_py_err)?;
    let child = wrap_shape_view(array.py(), array, &arr, inner)?;
    // numpy's broadcast view is read-only (`WRITEABLE=False`) -- independent
    // of the OWNDATA/base labelling fixed above by `wrap_shape_view`, and
    // still wrong after it: `crate::mark_readonly` flips the `flags`
    // getter's `WRITEABLE` bit for this instance (see its doc comment in
    // `lib.rs` for why anionpy otherwise defaults every array to writeable).
    crate::mark_readonly(child.bind(array.py()))?;
    Ok(child)
}

// ---------------------------------------------------------------------------
// Registration: one call from lib.rs's #[pymodule] fn, mirroring
// linalg::register's existing pattern.
// ---------------------------------------------------------------------------

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(empty_like, m)?)?;
    m.add_function(wrap_pyfunction!(full_like, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(identity, m)?)?;
    m.add_function(wrap_pyfunction!(asarray, m)?)?;
    m.add_function(wrap_pyfunction!(copy, m)?)?;
    m.add_function(wrap_pyfunction!(ascontiguousarray, m)?)?;
    m.add_function(wrap_pyfunction!(reshape, m)?)?;
    m.add_function(wrap_pyfunction!(ravel, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(swapaxes, m)?)?;
    m.add_function(wrap_pyfunction!(moveaxis, m)?)?;
    m.add_function(wrap_pyfunction!(squeeze, m)?)?;
    m.add_function(wrap_pyfunction!(expand_dims, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast_to, m)?)?;
    let _ = PyList::empty(m.py()); // keep PyList import used even if unused elsewhere
    Ok(())
}
