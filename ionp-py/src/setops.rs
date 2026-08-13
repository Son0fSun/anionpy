//! PyO3 bindings for `ionp_core::setops` -- see that module's doc for the
//! full scope note (unique family + intersect1d/union1d/setdiff1d/
//! setxor1d/isin + diff/ediff1d/trim_zeros). Every function here is a thin
//! marshaling wrapper: argument coercion (`crate::extract_array`, reused
//! from the crate root rather than duplicated -- it already handles
//! `PyArray`/real-numpy-array/python-scalar uniformly) plus a direct call
//! into `ionp_core::setops`, never any arithmetic or index loop of its own.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyTuple};

use ionp_core::setops as core_setops;
use ionp_core::{IonpError, NdArray};

use crate::{extract_array_like, to_py_err, PyArray};

fn wrap(inner: NdArray) -> PyArray {
    PyArray { inner }
}

// `extract_array_like` (array-like list/tuple coercion for `diff`'s
// `prepend=`/`append=` and `ediff1d`'s `to_begin=`/`to_end=`, originally
// written here in `38bc751`) now lives at the crate root as
// `crate::extract_array_like`, shared with `Ufunc::__call__`'s ufunc-input
// coercion (same bug, same fix, same precedent) -- see its doc comment in
// `lib.rs` for the full rationale.

// PyO3 can't combine a bare `&Bound<PyAny>` param with a literal `bool`
// default in `#[pyo3(signature=...)]` (the macro-generated default-value
// code can't convert a `bool` literal into a `&Bound<PyAny>`), so every
// plain-truthy bool-typed kwarg below is `Option<&Bound<PyAny>>` with a
// `None` pyo3 default, defaulted by hand via this helper.
fn opt_truthy_setops(obj: Option<&Bound<'_, PyAny>>, default: bool) -> PyResult<bool> {
    match obj {
        None => Ok(default),
        Some(v) => v.is_truthy(),
    }
}

fn extract_opt_array(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<NdArray>> {
    match obj {
        None => Ok(None),
        Some(o) => {
            if o.is_none() {
                Ok(None)
            } else {
                Ok(Some(extract_array_like(o)?))
            }
        }
    }
}

#[pyfunction]
fn unique_values(a: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let arr = extract_array_like(a)?;
    let out = core_setops::unique_values(&arr).map_err(to_py_err)?;
    Ok(wrap(out))
}

#[pyfunction]
fn unique_counts(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = extract_array_like(a)?;
    let (values, counts) = core_setops::unique_counts(&arr).map_err(to_py_err)?;
    let items = vec![
        Py::new(py, wrap(values))?.into_any(),
        Py::new(py, wrap(counts))?.into_any(),
    ];
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

#[pyfunction]
fn unique_inverse(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = extract_array_like(a)?;
    let (values, inverse) = core_setops::unique_inverse(&arr).map_err(to_py_err)?;
    let items = vec![
        Py::new(py, wrap(values))?.into_any(),
        Py::new(py, wrap(inverse))?.into_any(),
    ];
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

#[pyfunction]
fn unique_all(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr = extract_array_like(a)?;
    let (values, indices, inverse, counts) = core_setops::unique_all(&arr).map_err(to_py_err)?;
    let items = vec![
        Py::new(py, wrap(values))?.into_any(),
        Py::new(py, wrap(indices))?.into_any(),
        Py::new(py, wrap(inverse))?.into_any(),
        Py::new(py, wrap(counts))?.into_any(),
    ];
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

/// `np.unique(ar, return_index=False, return_inverse=False,
/// return_counts=False, axis=None, *, equal_nan=True, sorted=True)`.
///
/// `axis=` dispatches to `core_setops::unique` (which routes `ndim<=1`
/// straight through the flattened `unique_general` path -- real numpy's own
/// dispatcher hardcodes `axis=None` internally whenever `ar.ndim <= 1`, so
/// an explicit `axis=0`/`axis=-1` on a 1-D array is validated but otherwise
/// byte-identical to `axis=None`, `equal_nan` included -- and for `ndim>=2`
/// takes the genuinely different moveaxis + structured/void-row dedup path,
/// which NEVER collapses NaN regardless of `equal_nan`'s value since a void
/// dtype never qualifies for numpy's `aux.dtype.kind in "cfmM"`
/// equal_nan-special-case gate; verified empirically against live numpy
/// 2.5.1 both ways).
///
/// `sorted=` (numpy 2.x keyword) is accepted-and-ignored here, for the same
/// reason `unique_values` doesn't expose a hash-fast-path at all: this
/// build always takes the sort-based path, which is verified equivalent to
/// `sorted=True`'s behavior (bare `unique()`'s own default). We do NOT
/// attempt to reproduce `sorted=False`'s genuinely different (and, for
/// int/uint/complex dtypes with no explicit return flags, UNSPECIFIED
/// hash-iteration-order) output -- same landmine as `unique_values`, which
/// stays undeclared for exactly this reason. For the `axis=` (ndim>=2)
/// path, `sorted=` has been verified empirically to have zero observable
/// effect at all (the structured/void row dtype is never hash-eligible, so
/// real numpy's own hash fast path is never taken there either) -- so
/// `axis=`+`sorted=False` together are byte-exact regardless.
#[pyfunction]
#[pyo3(signature = (ar, return_index=None, return_inverse=None, return_counts=None, axis=None, equal_nan=None, sorted=None))]
fn unique(
    py: Python<'_>,
    ar: &Bound<'_, PyAny>,
    return_index: Option<&Bound<'_, PyAny>>,
    return_inverse: Option<&Bound<'_, PyAny>>,
    return_counts: Option<&Bound<'_, PyAny>>,
    axis: Option<isize>,
    equal_nan: Option<&Bound<'_, PyAny>>,
    sorted: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // All five bool-shaped kwargs are plain Python truthiness in real
    // numpy (measured 2026-08-02: `unique(a, return_index=1.5)`,
    // `equal_nan="x"`, `sorted=[1]` etc all dispatch on `if x:`), matching
    // `svd`'s original fix, not `isinstance(x, bool)`.
    let return_index = opt_truthy_setops(return_index, false)?;
    let return_inverse = opt_truthy_setops(return_inverse, false)?;
    let return_counts = opt_truthy_setops(return_counts, false)?;
    let equal_nan = opt_truthy_setops(equal_nan, true)?;
    let sorted = opt_truthy_setops(sorted, true)?;
    let _ = sorted; // accepted-and-ignored, see doc comment above
    let arr = extract_array_like(ar)?;
    let (values, indices, inverse, counts) =
        core_setops::unique(&arr, return_index, return_inverse, return_counts, axis, equal_nan)
            .map_err(to_py_err)?;

    if !return_index && !return_inverse && !return_counts {
        return Ok(Py::new(py, wrap(values))?.into_any());
    }
    let mut items: Vec<Py<PyAny>> = vec![Py::new(py, wrap(values))?.into_any()];
    if return_index {
        items.push(Py::new(py, wrap(indices.expect("return_index requested")))?.into_any());
    }
    if return_inverse {
        items.push(Py::new(py, wrap(inverse.expect("return_inverse requested")))?.into_any());
    }
    if return_counts {
        items.push(Py::new(py, wrap(counts.expect("return_counts requested")))?.into_any());
    }
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

/// `np.diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue)`.
///
/// Deliberately takes raw `*args`/`**kwargs` instead of pyo3's usual typed
/// `#[pyo3(signature=...)]` parameters -- the SAME technique
/// `Ufunc::__call__`'s `out=` keyword uses in `ionp-py/src/lib.rs` (see
/// its own doc comment), for the identical underlying reason: pyo3's
/// `Option<T>: FromPyObject` blanket impl maps a Python `None` ARGUMENT to
/// Rust `None` identically to an OMITTED argument, which is structurally
/// indistinguishable through a typed named parameter. That distinction is
/// load-bearing for THREE of this function's keywords, not just one:
///
/// - `n`'s real default is the concrete int `1`; real numpy's `n == 0`/
///   `n < 0` checks run against whatever object was actually passed, so
///   `n=None` explicitly raises `TypeError: '<' not supported between
///   instances of 'NoneType' and 'int'` (verified live) where an omitted
///   `n` is simply `1`. A typed `n: usize = 1` parameter can't observe
///   this at all (pyo3 either extracts a real int or raises its OWN
///   generic extraction error for `None`, which is a different message).
/// - `prepend`/`append`'s real default is a private `np._NoValue`
///   sentinel, and explicit `None` is a normal, distinct value that
///   crashes downstream (see `core_setops::DiffOperand`'s own doc for the
///   full derivation and exact message text per dtype).
///
/// Before this fix (2026-08-02) all three were typed parameters with a
/// literal `None`/`1` Rust-level default, so `anionpy.diff(a, prepend=None)`
/// (and `n=None`, and `append=None`) silently behaved identically to
/// omitting the keyword entirely -- a real, confirmed divergence from
/// live numpy 2.5.1 (which raises in all three cases). Fixed by resolving
/// `a`/`n`/`axis`/`prepend`/`append` manually from the raw positional
/// tuple plus raw keyword dict (mirroring `Ufunc::__call__`'s
/// `out_kw_present`/`out_kw_value` bookkeeping), so "key/slot absent" and
/// "key/slot present holding `None`" are visibly different states all the
/// way through to `core_setops::diff`'s `DiffOperand` enum.
#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn diff(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyArray> {
    const NAMES: [&str; 5] = ["a", "n", "axis", "prepend", "append"];
    if args.len() > NAMES.len() {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "diff() takes from 1 to {} positional arguments but {} were given",
            NAMES.len(),
            args.len()
        )));
    }
    let mut slots: [Option<Bound<'_, PyAny>>; 5] = [None, None, None, None, None];
    for (i, item) in args.iter().enumerate() {
        slots[i] = Some(item);
    }
    if let Some(kw) = kwargs {
        for (idx, name) in NAMES.iter().enumerate() {
            if let Some(v) = kw.get_item(name)? {
                if slots[idx].is_some() {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "diff() got multiple values for argument '{name}'"
                    )));
                }
                slots[idx] = Some(v);
            }
        }
        for key in kw.keys().iter() {
            let key_str: String = key.extract()?;
            if !NAMES.contains(&key_str.as_str()) {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "diff() got an unexpected keyword argument '{key_str}'"
                )));
            }
        }
    }
    let [a_slot, n_slot, axis_slot, prepend_slot, append_slot] = slots;

    let a_obj = a_slot.ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err(
            "diff() missing 1 required positional argument: 'a'",
        )
    })?;
    let arr = extract_array_like(&a_obj)?;

    // Real numpy: `if n == 0: return a` / `if n < 0: raise ValueError(...)`
    // are plain Python comparisons against whatever `n` object was passed
    // -- `None == 0` is `False` (falls through) and `None < 0` raises.
    let n_val: i64 = match &n_slot {
        None => 1,
        Some(v) => {
            if v.is_none() {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "'<' not supported between instances of 'NoneType' and 'int'",
                ));
            }
            v.extract::<i64>()?
        }
    };
    if n_val < 0 {
        // Real numpy: `"order must be non-negative but got " + repr(n)`;
        // `repr()` of a plain Python int is just its decimal digits.
        return Err(PyValueError::new_err(format!(
            "order must be non-negative but got {n_val}"
        )));
    }
    let n = n_val as usize;

    // Real numpy's `if n == 0: return a` is the FIRST thing `diff` does --
    // before axis is normalized, and before prepend/append are looked at
    // AT ALL (not even coerced to an array; see `core_setops::diff`'s own
    // doc for the live-verified evidence, e.g. `prepend=np.zeros((2,2))`
    // being silently ignored). `core_setops::diff` itself already gets
    // this ordering right internally -- but it can't help here, because
    // THIS wrapper extracted `axis`/`prepend`/`append` into concrete Rust
    // types before ever calling into core, which is too early: e.g.
    // `anionpy.diff(a, n=0, axis=None)` used to fail extracting `axis` as an
    // `isize` (raising `TypeError: 'NoneType' object cannot be
    // interpreted as an integer`) even though real numpy returns `a`
    // unchanged, never even glancing at `axis`. Short-circuiting here,
    // before any of `axis`/`prepend`/`append` are touched, matches real
    // numpy's precedence exactly regardless of what garbage those three
    // hold.
    if n == 0 {
        return Ok(wrap(arr.clone()));
    }

    let axis: isize = match &axis_slot {
        None => -1,
        Some(v) => v.extract::<isize>()?,
    };

    // `prepend`/`append` each need their extracted `NdArray` (when a real
    // array-like was given) to outlive the `core_setops::diff` call below,
    // so they're materialized here as owned locals rather than through the
    // helper stub above.
    let prepend_owned: Option<NdArray> = match &prepend_slot {
        Some(v) if !v.is_none() => Some(extract_array_like(v)?),
        _ => None,
    };
    let append_owned: Option<NdArray> = match &append_slot {
        Some(v) if !v.is_none() => Some(extract_array_like(v)?),
        _ => None,
    };
    let prepend_op = match &prepend_slot {
        None => core_setops::DiffOperand::Omitted,
        Some(v) if v.is_none() => core_setops::DiffOperand::ExplicitNone,
        Some(_) => core_setops::DiffOperand::Value(prepend_owned.as_ref().unwrap()),
    };
    let append_op = match &append_slot {
        None => core_setops::DiffOperand::Omitted,
        Some(v) if v.is_none() => core_setops::DiffOperand::ExplicitNone,
        Some(_) => core_setops::DiffOperand::Value(append_owned.as_ref().unwrap()),
    };

    let out = core_setops::diff(&arr, n, axis, prepend_op, append_op).map_err(to_py_err)?;
    Ok(wrap(out))
}

#[pyfunction]
#[pyo3(signature = (ary, to_end=None, to_begin=None))]
fn ediff1d(
    ary: &Bound<'_, PyAny>,
    to_end: Option<&Bound<'_, PyAny>>,
    to_begin: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    let arr = extract_array_like(ary)?;
    let to_end_arr = extract_opt_array(to_end)?;
    let to_begin_arr = extract_opt_array(to_begin)?;
    let out = core_setops::ediff1d(&arr, to_end_arr.as_ref(), to_begin_arr.as_ref())
        .map_err(to_py_err)?;
    Ok(wrap(out))
}

/// Mirrors the `axes_list_from_pyobj`-style int-or-sequence parsing pattern
/// used by `ionp-py/src/reductions.rs` (each binding file keeps its own
/// private copy rather than sharing one, per that file's established
/// convention) -- real numpy's `normalize_axis_tuple` (which `trim_zeros`'s
/// `axis=` argument is threaded through) accepts either a bare int or any
/// sequence of ints.
fn axis_tuple_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(v) = obj.extract::<isize>() {
        Ok(vec![v])
    } else {
        obj.extract::<Vec<isize>>()
    }
}

/// `np.trim_zeros(filt, trim='fb', axis=None)`.
///
/// `axis=None` trims over every axis (the pre-existing 1-D-shaped
/// behavior, now generalized to N-D via `core_setops::trim_zeros`'s
/// `axis_tuple = 0..ndim` default). An explicit `axis=` raises numpy's
/// real (PREFIXED) `AxisError` -- `'axis: axis N is out of bounds for
/// array of dimension M'` -- via `axis_error_prefixed`, which is why the
/// `IonpError::AxisError` case is intercepted here rather than falling
/// through to the generic `to_py_err` (which raises the UNPREFIXED form
/// `unique(axis=)` uses instead; the two functions genuinely disagree on
/// this, verified against live numpy 2.5.1: `trim_zeros` re-raises through
/// numpy's own `AxisError(axis, ndim, "axis")`, `unique`'s dispatcher does
/// not).
///
/// Return TYPE preservation (verified against real numpy 2.5.1's actual
/// source, `_function_base_impl.py::trim_zeros`, not guessed at): real
/// numpy never builds a fresh array for its return value except through
/// slicing/identity of the ORIGINAL `filt` object --
///   - identity case (`axis_tuple` empty, i.e. `axis=()`, or any input
///     whose ndim is 0): `return filt` -- the exact same object, unchanged,
///     whatever its type (list, tuple, ndarray, bare scalar...).
///   - 1-D case (`len(sl) == 1`, i.e. `filt`'s ndim is 1): `return
///     filt[sl[0]]` -- Python's own `__getitem__`/slicing dispatch on the
///     ORIGINAL object, not on `np.asarray(filt)`. For a list/tuple/range
///     input this means list-in-list-out, tuple-in-tuple-out, etc; for an
///     ndarray (or `PyArray`) input it's an ndarray view either way, same
///     type either path.
///   - N-D case: `return filt[sl]` -- same object, multi-dim slice; for a
///     bare Python list of lists this would already have failed inside
///     `np.asarray` or slicing long before reaching here in real numpy, so
///     this path is only ever reached with an actual array here.
/// This is reproduced by only special-casing the 1-D non-`PyArray` input:
/// compute the (start, stop) bounding box via `core_setops::
/// trim_zeros_bounds` (identical math to `core_setops::trim_zeros`, just
/// stopping short of the slice), then apply it as Python's OWN slice
/// mechanism (`filt.get_item(PySlice::new(...))`) on the original `filt`
/// object -- this reproduces numpy's MECHANISM (real `__getitem__`
/// dispatch), not a hardcoded type map, so it covers any sliceable
/// sequence the same way numpy's own `filt[sl[0]]` does. `PyArray` input
/// (already an ndarray) and non-1-D input both fall through to the
/// existing ndarray-returning path, matching numpy exactly since ndarray
/// stays ndarray on both branches.
#[pyfunction]
#[pyo3(signature = (filt, trim="fb", axis=None))]
fn trim_zeros<'py>(
    filt: &Bound<'py, PyAny>,
    trim: &str,
    axis: Option<&Bound<'py, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let py = filt.py();
    let arr = extract_array_like(filt)?;
    let axis_vec = match axis {
        None => None,
        Some(o) => Some(axis_tuple_from_pyobj(o)?),
    };
    let ndim = arr.ndim();
    let is_bare_array = filt.extract::<PyRef<'_, PyArray>>().is_ok();

    let bounds =
        core_setops::trim_zeros_bounds(&arr, trim, axis_vec.as_deref()).map_err(|e| match e {
            IonpError::AxisError { axis, ndim: Some(nd) } => crate::axis_error_prefixed(axis, nd, "axis"),
            other => to_py_err(other),
        })?;
    let (starts, ends) = match bounds {
        // Identity case: numpy returns `filt` itself, verbatim -- same
        // object, same type, not even wrapped through ndarray.
        None => return Ok(filt.clone().unbind()),
        Some(b) => b,
    };

    if !is_bare_array {
        // Real numpy: `sl = tuple(slice(start[ax], stop[ax]) ...)`, then
        // `if len(sl) == 1: return filt[sl[0]]` (bare slice, so a plain
        // list/tuple/range slices itself and stays that type) `else:
        // return filt[sl]` (a TUPLE of slices -- which a bare Python
        // list/tuple does NOT support indexing with: `[1,2,3][(slice(0,1),
        // slice(0,1))]` raises `TypeError: list indices must be integers or
        // slices, not tuple`, verified against real numpy 2.5.1 for ndim>=2
        // list/tuple input). Reproducing this generally (not just the
        // ndim==1 case) by always dispatching through the ORIGINAL
        // object's own `__getitem__` -- via a bare `PySlice` for ndim==1,
        // a `PyTuple` of `PySlice`s otherwise -- means Python's own
        // indexing protocol decides pass/raise exactly the way numpy's
        // `filt[sl]` does, for any sliceable-or-not object, not just the
        // list/tuple/range forms this corpus happens to exercise.
        if ndim == 1 {
            let slice = pyo3::types::PySlice::new(py, starts[0] as isize, ends[0] as isize, 1);
            return Ok(filt.get_item(slice)?.unbind());
        }
        let slices: Vec<Bound<'py, PyAny>> = (0..ndim)
            .map(|ax| {
                pyo3::types::PySlice::new(py, starts[ax] as isize, ends[ax] as isize, 1)
                    .into_any()
            })
            .collect();
        let sl_tuple = PyTuple::new(py, slices)?;
        return Ok(filt.get_item(sl_tuple)?.unbind());
    }

    let flat = arr.to_contiguous();
    let items: Vec<ionp_core::array::SliceItem> = (0..ndim)
        .map(|ax| ionp_core::array::SliceItem::Slice {
            start: Some(starts[ax] as isize),
            stop: Some(ends[ax] as isize),
            step: Some(1),
        })
        .collect();
    let out = flat.get_view(&items).map_err(to_py_err)?.to_contiguous();
    Ok(Py::new(py, wrap(out))?.into_any())
}

/// `np.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)`.
/// `return_indices=True` returns a 3-tuple `(int1d, comm1, comm2)` via
/// `core_setops::intersect1d_return_indices`, mirroring the
/// `unique_all`/`unique_counts` tuple-building pattern above.
#[pyfunction]
#[pyo3(signature = (ar1, ar2, assume_unique=None, return_indices=None))]
fn intersect1d(
    py: Python<'_>,
    ar1: &Bound<'_, PyAny>,
    ar2: &Bound<'_, PyAny>,
    assume_unique: Option<&Bound<'_, PyAny>>,
    return_indices: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Plain Python truthiness, measured 2026-08-02 (same as `unique`
    // above).
    let assume_unique = opt_truthy_setops(assume_unique, false)?;
    let return_indices = opt_truthy_setops(return_indices, false)?;
    let a = extract_array_like(ar1)?;
    let b = extract_array_like(ar2)?;
    if return_indices {
        let (int1d, comm1, comm2) =
            core_setops::intersect1d_return_indices(&a, &b, assume_unique).map_err(to_py_err)?;
        let items = vec![
            Py::new(py, wrap(int1d))?.into_any(),
            Py::new(py, wrap(comm1))?.into_any(),
            Py::new(py, wrap(comm2))?.into_any(),
        ];
        Ok(PyTuple::new(py, items)?.into_any().unbind())
    } else {
        let out = core_setops::intersect1d(&a, &b, assume_unique).map_err(to_py_err)?;
        Ok(Py::new(py, wrap(out))?.into_any())
    }
}

#[pyfunction]
fn union1d(ar1: &Bound<'_, PyAny>, ar2: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    // PHANTOM FOUND + FIXED (2026-08-04): this was two INDEPENDENT
    // `extract_array_like` calls, which give a bare Python scalar a fixed
    // STRONG dtype (int64/float64/complex128). numpy's `union1d` is
    // literally `unique(np.concatenate((ar1, ar2), axis=None))` (its own
    // `_arraysetops_impl.py`), and `concatenate` applies NEP 50 weak
    // promotion, so `np.union1d(np.array([1,0,2], np.int8), 1)` is int8
    // where anionpy produced int64. 56 of a 168-cell (dtype x scalar-kind x
    // position) sweep diverged, all on DTYPE, which is why the corpus --
    // which only ever passes two arrays -- never saw it.
    //
    // Its three sibling set ops are NOT affected and were measured clean on
    // the same sweep, for a real reason rather than luck: `intersect1d`,
    // `setdiff1d` and `setxor1d` each call `np.asanyarray` on their inputs
    // FIRST, which makes a bare Python int a strong int64 before any
    // promotion happens. `union1d` is the only one that hands the raw
    // operand straight to `concatenate`. Fixing all four the same way would
    // have broken the other three.
    //
    // `extract_concat_pair` (lib.rs) rather than `extract_binary_pair`
    // because concatenation's out-of-range behaviour is a WRAPPING cast,
    // not the OverflowError of plain arithmetic nor the value-widening of
    // the comparison ops -- see that function's own doc comment for the
    // measured table.
    let (a, b) = crate::extract_concat_pair(ar1, ar2)?;
    let out = core_setops::union1d(&a, &b).map_err(to_py_err)?;
    Ok(wrap(out))
}

#[pyfunction]
#[pyo3(signature = (ar1, ar2, assume_unique=None))]
fn setdiff1d(ar1: &Bound<'_, PyAny>, ar2: &Bound<'_, PyAny>, assume_unique: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let assume_unique = opt_truthy_setops(assume_unique, false)?;
    let a = extract_array_like(ar1)?;
    let b = extract_array_like(ar2)?;
    let out = core_setops::setdiff1d(&a, &b, assume_unique).map_err(to_py_err)?;
    Ok(wrap(out))
}

#[pyfunction]
#[pyo3(signature = (ar1, ar2, assume_unique=None))]
fn setxor1d(ar1: &Bound<'_, PyAny>, ar2: &Bound<'_, PyAny>, assume_unique: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
    let assume_unique = opt_truthy_setops(assume_unique, false)?;
    let a = extract_array_like(ar1)?;
    let b = extract_array_like(ar2)?;
    let out = core_setops::setxor1d(&a, &b, assume_unique).map_err(to_py_err)?;
    Ok(wrap(out))
}

#[pyfunction]
#[pyo3(signature = (element, test_elements, assume_unique=None, invert=None, kind=None))]
fn isin(
    element: &Bound<'_, PyAny>,
    test_elements: &Bound<'_, PyAny>,
    assume_unique: Option<&Bound<'_, PyAny>>,
    invert: Option<&Bound<'_, PyAny>>,
    kind: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyArray> {
    // Plain Python truthiness (measured 2026-08-02), same as `unique`/
    // `intersect1d` above. NOTE: a raising-`__bool__` object passed as
    // `assume_unique` was observed to sometimes NOT raise in real numpy
    // (small-array fast path short-circuits before ever evaluating
    // `bool(assume_unique)`, an internal-algorithm-selection quirk, not a
    // documented contract) -- `invert`'s raising case DID propagate.
    // `is_truthy()` always evaluates (and so always propagates a raising
    // `__bool__`) for both, which is a strict superset of numpy's
    // guaranteed behavior and never LESS permissive than numpy's own
    // worst case (numpy sometimes doesn't call `__bool__` at all; anionpy
    // always does, so anionpy raises in every case numpy might, plus a few
    // numpy's internal fast path happens to skip) -- not chased further,
    // flagged here per the brief's "measure, don't assume" instruction.
    let assume_unique = opt_truthy_setops(assume_unique, false)?;
    let invert = opt_truthy_setops(invert, false)?;
    // `core_setops::isin` (forbidden territory -- setops.rs in ionp-core is
    // another agent's assignment) has no `kind` parameter at all: its
    // single hash/mask algorithm already produces numpy-identical RESULTS
    // for both 'sort' and 'table' (verified -- these are just two
    // internal-algorithm choices numpy offers for its own performance
    // tuning, not behavior-affecting), so `kind` genuinely never needs to
    // reach it. What was missing here was numpy's own upfront validation:
    // `np.isin(a, b, kind='bogus')` raises `ValueError: Invalid kind:
    // 'bogus'. Please use None, 'sort' or 'table'.` before doing any work
    // at all; anionpy used to accept-and-ignore any string silently. Fixed by
    // validating against the same three accepted values numpy documents,
    // matching its exact message text, before calling into core.
    //
    // 2026-08-02 follow-up: the original fix typed `kind` as `Option<&str>`,
    // so PyO3's own arg coercion rejected any non-string, non-None `kind`
    // (e.g. `kind=0`, `kind=True`) with a generic TypeError *before* this
    // function's body ever ran -- diverging from real numpy, which does
    // `kind not in {None, "sort", "table"}` (a genuine `in`-on-a-set
    // membership test against the raw object, no string coercion) and only
    // then formats the ValueError via `str(kind)`. Measured against numpy
    // 2.5.1: `kind=0`/`kind=True`/`kind=1.5`/`kind=object()` all raise
    // `ValueError: Invalid kind: '<str(kind)>'. Please use None, 'sort' or
    // 'table'.`; `kind=[1, 2]` (unhashable) raises `TypeError: cannot use
    // 'list' as a set element (unhashable type: 'list')` -- CPython's own
    // set-membership error, not anything numpy-authored. Fixed by taking
    // `kind` as an arbitrary object and doing the actual `in {None, "sort",
    // "table"}` check via a real Python frozenset, so an unhashable `kind`
    // raises CPython's own TypeError verbatim (no numpy call involved --
    // this is Python's own set implementation, same as `x in {1, 2}`
    // anywhere else in this crate), and only a hashable-but-invalid `kind`
    // falls through to the ValueError, formatted with `str(kind)` to match
    // numpy's f-string (not `repr(kind)` -- `str(True)` is `"True"` with no
    // quotes, matching numpy's observed message, whereas `repr(True)` would
    // also be `"True"` but `repr("x")` would add an extra quote layer that
    // numpy's message does not have for string kinds).
    if let Some(k) = kind {
        let py = k.py();
        let allowed = PyFrozenSet::new(
            py,
            &[py.None().into_bound(py), "sort".into_pyobject(py)?.into_any(), "table".into_pyobject(py)?.into_any()],
        )?;
        if !allowed.contains(k)? {
            let kind_str = k.str()?.to_string();
            return Err(PyValueError::new_err(format!(
                "Invalid kind: '{kind_str}'. Please use None, 'sort' or 'table'."
            )));
        }
    }
    let e = extract_array_like(element)?;
    let t = extract_array_like(test_elements)?;
    let out = core_setops::isin(&e, &t, assume_unique, invert).map_err(to_py_err)?;
    Ok(wrap(out))
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(unique_values, m)?)?;
    m.add_function(wrap_pyfunction!(unique_counts, m)?)?;
    m.add_function(wrap_pyfunction!(unique_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(unique_all, m)?)?;
    m.add_function(wrap_pyfunction!(unique, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(ediff1d, m)?)?;
    m.add_function(wrap_pyfunction!(trim_zeros, m)?)?;
    m.add_function(wrap_pyfunction!(intersect1d, m)?)?;
    m.add_function(wrap_pyfunction!(union1d, m)?)?;
    m.add_function(wrap_pyfunction!(setdiff1d, m)?)?;
    m.add_function(wrap_pyfunction!(setxor1d, m)?)?;
    m.add_function(wrap_pyfunction!(isin, m)?)?;
    Ok(())
}
