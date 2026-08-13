//! `median` / `percentile` / `quantile` / `nanmedian` / `nanpercentile` /
//! `nanquantile` -- thin marshaling wrappers over `ionp_core::stats`.
//!
//! `axis` here only ever accepts a single int or `None` (never a tuple) --
//! matches this module's own exposed signature, not a limitation snuck in
//! silently: `ionp_core::stats::quantile_axis`/`median_axis` themselves
//! only take one `usize` axis. `overwrite_input` is accepted but is a
//! deliberate no-op: real numpy's own contract for it explicitly leaves
//! the input array's post-call contents "undefined" (only the RETURN
//! VALUE is documented/guaranteed), and this implementation never mutates
//! its input regardless of the flag, so the observable (return-value)
//! behavior matches either way.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use ionp_core::stats::{median_axis, quantile_axis, QuantileArgs, QuantileMethod};
use ionp_core::{Buffer, DType, IonpError, NdArray};

use crate::{extract_array_like, ndarray_from_pylist, to_py_err, write_into_out, PyArray};

/// Returns the (possibly-flattened) array to reduce over, the axis to pass
/// to the core `_axis` functions, and the ORIGINAL ndim -- callers need the
/// latter because `axis=None` flattens to a 1-d array with a single axis
/// 0 internally, so a naive `keepdims` insertion on the core result would
/// only add back ONE dimension of size 1, not one per original axis (which
/// is what real numpy's `axis=None, keepdims=True` actually produces).
/// The TypeError `normalize_axis_index` raises for an axis entry that is not
/// an integer. Three distinct messages, all measured live against numpy
/// 2.5.1 on 2026-08-04 with `np.median(np.zeros((2, 3)), axis=(x,))`:
///
/// * any instance of Python's `float` -- which includes `np.float64`, since
///   that subclasses it, and any user subclass --
///   "integer argument expected, got float". This is the C helper
///   `PyArray_PyIntAsInt`'s own float branch, NOT `operator.index`'s
///   message; `np.float32` is not a `float` subclass and so takes the
///   generic branch instead ("'numpy.float32' object cannot be
///   interpreted as an integer"). That split is the tell that this is a
///   type CHECK, not a duck-typed `__index__` call.
/// * an ndarray -- "only integer scalar arrays can be converted to a
///   scalar index" (numpy's own `__index__`; a 0-d INTEGER array converts
///   fine and never reaches here).
/// * anything else -- "'{fully.qualified.name}' object cannot be
///   interpreted as an integer".
///
/// The name must be FULLY QUALIFIED (`numpy.float32`, `decimal.Decimal`),
/// which is `tp_name`, not `__name__`; `PyType::name()` returns the latter
/// and produced a bare "float32" here until 2026-08-04.
/// `(is_array, ndim)` for an axis operand that might be an array, covering
/// BOTH a real numpy array and an anionpy one.
///
/// The operand-parity rule means an anionpy caller passes `anionpy.array(0.0)`
/// where a numpy caller passes `np.array(0.0)`, so a check written against
/// `numpy::PyUntypedArray` alone silently misses exactly the case the
/// corpus exercises -- which is how both defects below survived until the
/// 2026-08-04 regression block caught them.
pub(crate) fn axis_array_ndim(item: &Bound<'_, PyAny>) -> Option<usize> {
    // Guarded by `numpy_available` FIRST: `.cast::<numpy::PyUntypedArray>()`
    // does not fail gracefully when numpy's C API capsule can't be loaded
    // at all -- it panics (see `errors::numpy_available`'s doc comment).
    // With numpy present this changes nothing (the branch always ran
    // before); with numpy absent it is skipped entirely so an ordinary
    // `axis=<int>` no longer crashes the interpreter on its way to falling
    // through to `None`.
    if crate::errors::numpy_available(item.py()) {
        if let Ok(a) = item.cast::<numpy::PyUntypedArray>() {
            use numpy::PyUntypedArrayMethods;
            return Some(a.ndim());
        }
    }
    if let Ok(a) = item.cast::<PyArray>() {
        return Some(a.borrow().inner.ndim());
    }
    None
}

fn axis_index_type_error(item: &Bound<'_, PyAny>) -> PyErr {
    if item.is_instance_of::<pyo3::types::PyFloat>() {
        return PyTypeError::new_err("integer argument expected, got float");
    }
    // Any array that failed `operator.index` -- 0-d float, 1-d, numpy or
    // anionpy alike -- gets numpy's array-specific wording, NOT the generic
    // "object cannot be interpreted as an integer". Measured:
    // `np.median(a, axis=(np.array(0.0),))` -> TypeError: only integer
    // scalar arrays can be converted to a scalar index.
    if axis_array_ndim(item).is_some() {
        return PyTypeError::new_err("only integer scalar arrays can be converted to a scalar index");
    }
    let tn = item
        .get_type()
        .fully_qualified_name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "object".to_string());
    PyTypeError::new_err(format!("'{tn}' object cannot be interpreted as an integer"))
}

/// numpy's `normalize_axis_tuple`, the SEQUENCE half -- shared, not copied.
///
/// This is a FOURTH axis converter, distinct from the three in
/// `reductions.rs` (`PyArray_ConvertMultiAxis`, the mean-family pre-check,
/// and the single-axis one). Its callers are `median`/`quantile`/
/// `percentile` and their `nan*` siblings via `resolve_axis`, and
/// `average`. It iterates ANY sequence, accepts bools through
/// `operator.index`, and has its own message set at both the container and
/// the element level. Measured against numpy 2.5.1 on 2026-08-04,
/// `np.average(np.zeros((2, 3)), axis=...)`:
///
///     axis=              bare                        inside a tuple
///     -----------------  --------------------------  -----------------------
///     0.0                'float' object is not       integer argument
///                        iterable                    expected, got float
///     np.float64(0)      'numpy.float64' ... not     integer argument
///                        iterable                    expected, got float
///     np.float32(0)      'numpy.float32' ... not     'numpy.float32' object
///                        iterable                    cannot be interpreted...
///     Decimal(0)         'decimal.Decimal' ... not   'decimal.Decimal' object
///                        iterable                    cannot be interpreted...
///     np.True_           'numpy.bool' ... not        'numpy.bool' object
///                        iterable                    cannot be interpreted...
///     np.array(0.0)      iteration over a 0-d array  only integer scalar
///                                                    arrays ...
///     np.array([0])      OK, shape (3,)              only integer scalar
///                                                    arrays ...
///     'x'                'str' object cannot be      (same)
///                        interpreted as an integer
///     True / np.int64(1) OK, shape (2,)              OK, shape (2,)
///     b"\x00"            OK, shape (3,)              'bytes' object cannot
///                                                    be interpreted ...
///
/// Note the two rows that look like exceptions and are not: `'x'` iterates
/// (into the CHARACTER `'x'`, which then fails the element rule) and
/// `b"\x00"` iterates into the INTEGER 0. Both fall straight out of
/// "`operator.index` first, then iterate" without a special case.
///
/// Extracted from `resolve_axis` on 2026-08-04 so `average` could share it
/// instead of growing a copy. The copies are what have broken this library
/// twice now (see `ndarray_attrs.rs`'s converter and cab03ab).
///
/// `argname` mirrors numpy's own parameter of the same name and is NOT
/// cosmetic -- it selects between two message sets. `_ureduce` (median,
/// quantile, percentile) calls this with NO argname; `_average` calls it
/// with `argname="axis"`. Measured 2026-08-04, numpy 2.5.1, 3-d operand:
///     np.median (a, axis=(0, -3)) -> ValueError: repeated axis
///     np.average(a, axis=(0, -3)) -> ValueError: repeated axis in `axis`
///                                    argument
///     np.median (a, axis=(0, 9))  -> AxisError: axis 9 is out of bounds...
///     np.average(a, axis=(0, 9))  -> AxisError: axis: axis 9 is out of...
pub(crate) fn normalize_axis_tuple_seq(
    ax: &Bound<'_, PyAny>,
    ndim: usize,
    argname: Option<&str>,
) -> PyResult<Vec<usize>> {
    if axis_array_ndim(ax) == Some(0) {
        return Err(PyTypeError::new_err("iteration over a 0-d array"));
    }
    let mut norm: Vec<usize> = Vec::new();
    for item in ax.try_iter()? {
        let item = item?;
        let raw: isize = item.extract().map_err(|_| axis_index_type_error(&item))?;
        let n = ndim as isize;
        let normed = if raw < 0 { raw + n } else { raw };
        if normed < 0 || normed >= n {
            return Err(match argname {
                None => to_py_err(IonpError::AxisError { axis: raw, ndim: Some(ndim) }),
                Some(name) => crate::axis_error_prefixed(raw, ndim, name),
            });
        }
        let normed = normed as usize;
        // numpy normalizes EVERY entry first and only then rejects
        // duplicates -- `median(a, axis=(0, 5))` on a 3-d array is an
        // AxisError, while `axis=(0, -3)` is `ValueError: repeated axis`.
        // Checking inside the loop preserves that precedence because the
        // out-of-bounds test above already ran for this element.
        if norm.contains(&normed) {
            // "repeated axis", NOT "duplicate value in 'axis'":
            // `_ureduce` calls `normalize_axis_tuple(axis, nd)` with NO
            // `argname`, and that is the message the no-argname branch
            // emits. The argname'd variant IS reachable from these six
            // functions, but only via the size-0 `nan*` delegation to
            // `nanmean` (which passes `argname="axis"`), and that path
            // never reaches this line. Both measured live against numpy
            // 2.5.1 on 2026-08-04; a first attempt to "fix" this to the
            // argname'd text regressed 3,800 sweep cases.
            return Err(PyValueError::new_err(match argname {
                None => "repeated axis".to_string(),
                Some(name) => format!("repeated axis in `{name}` argument"),
            }));
        }
        norm.push(normed);
    }
    Ok(norm)
}

fn resolve_axis(
    arr: &NdArray,
    axis: Option<&Bound<'_, PyAny>>,
    preserve_f_layout: bool,
) -> PyResult<(NdArray, usize, usize, Option<Vec<usize>>)> {
    let ndim = arr.ndim();
    let is_none = matches!(axis, None) || matches!(axis, Some(a) if a.is_none());
    if is_none {
        let flat = arr.ravel_order("C").map_err(to_py_err)?;
        return Ok((flat, 0, ndim, None));
    }
    let ax = axis.unwrap();
    // TUPLE / LIST axis. Added 2026-08-04; the module header previously
    // recorded "tuple axes are not supported" as a deliberate scope cut,
    // which is now closed. This reproduces `numpy.lib._function_base_impl.
    // _ureduce`'s own multi-axis strategy rather than inventing one: move
    // every KEPT axis to the front (in increasing original order), collapse
    // all REDUCED axes into a single trailing axis, and reduce over that.
    // For median/quantile the ORDER of elements inside the collapsed axis is
    // irrelevant (both sort it first), so the C-order flattening this uses is
    // free to differ from numpy's swapaxes-then-reshape element order without
    // any observable consequence.
    // numpy's `normalize_axis_tuple` tries `operator.index(axis)` FIRST and
    // only iterates when that raises, so what makes something "a sequence of
    // axes" is failing to be an integer -- not being a tuple or a list. The
    // difference is observable: `np.median(a, axis=np.float64(0))` reports
    // "'numpy.float64' object is not iterable" (the ITERATION failing), not
    // an index error, while `axis='x'` iterates happily and then fails on the
    // CHARACTER. Restricting this branch to tuple/list, as this file did
    // until 2026-08-04, got both of those wrong. `try_iter`'s error is
    // propagated verbatim because CPython's and numpy's own messages
    // ("'float' object is not iterable", "iteration over a 0-d array") are
    // already exactly what numpy surfaces here. All measured live against
    // numpy 2.5.1 on 2026-08-04.
    if ax.extract::<isize>().is_err() {
        // A 0-d array reaches here (it is not an integer) and must fail the
        // ITERATION, not silently yield nothing. anionpy's `ndarray` has no
        // `__iter__`, so Python falls back to the sequence protocol and
        // `list(iter(anionpy.array(0.0)))` was returning `[]` -- which made
        // `median(a, axis=anionpy.array(0.0))` behave like `axis=()` and
        // return the operand unreduced, where numpy raises. numpy's own
        // 0-d array raises this from `try_iter` below and is left alone;
        // this only supplies the message anionpy cannot produce yet.
        // Measured: `np.median(a, axis=np.array(0.0))` -> TypeError:
        // iteration over a 0-d array.
        let norm = normalize_axis_tuple_seq(ax, ndim, None)?;
        // `axis=()` reduces over NOTHING: numpy returns the array itself
        // (dtype-promoted), same shape, for every ndim including 0-d. Model
        // it as a reduction over a freshly appended LENGTH-1 axis -- the
        // median/quantile of a one-element slice is that element, promoted,
        // which is exactly the identity numpy produces, and it needs no
        // special case anywhere downstream.
        if norm.is_empty() {
            // ... EXCEPT on a size-0 operand. numpy does not special-case
            // `axis=()`; it runs `_ureduce`'s generic reshape
            // `a.reshape(a.shape[:nkeep] + (-1,))`, and an unknown `-1`
            // dimension cannot be inferred when the known dimensions
            // already multiply to zero. So numpy RAISES there, e.g.
            // `np.median(np.zeros((0, 3)), axis=())` ->
            // "cannot reshape array of size 0 into shape (0,3,newaxis)".
            // Measured live against numpy 2.5.1 on 2026-08-04. The `nan*`
            // siblings never reach this line -- they short-circuit any
            // size-0 operand to `nanmean` before `resolve_axis` runs.
            if arr.size() == 0 {
                let dims: Vec<String> = arr.shape().iter().map(|d| d.to_string()).collect();
                return Err(PyValueError::new_err(format!(
                    "cannot reshape array of size 0 into shape ({},newaxis)",
                    dims.join(",")
                )));
            }
            // LAYOUT, and only for `median`/`nanmedian`. `axis=()` reduces
            // nothing, so numpy's result is `mean()` of a length-1 axis of
            // the ORIGINAL array and inherits its memory order: on an
            // F-contiguous (5,6) float32 input `np.median(a, axis=())` has
            // strides (4,20), not (24,4). `quantile`/`percentile` do NOT do
            // this -- same input, same `axis=()`, they come back C-order --
            // which is why this is a caller-chosen flag rather than a
            // property of `axis=()`. Both measured live against numpy 2.5.1
            // on 2026-08-04; every OTHER axis form (0, 1, 2, tuples, None,
            // keepdims) already matched on F input before this change, so
            // the C-forcing reshape below was the only leak.
            let order = if preserve_f_layout && !arr.is_c_contiguous() && arr.is_f_contiguous() { "F" } else { "C" };
            let mut shape = arr.shape().to_vec();
            shape.push(1);
            let target = arr
                .to_contiguous_order(order)
                .map_err(to_py_err)?
                .reshape_with_order(&shape, order)
                .map_err(to_py_err)?;
            return Ok((target, ndim, ndim, Some(norm)));
        }
        if norm.len() == 1 {
            // A 1-tuple is exactly the scalar-axis case; hand it to the
            // existing single-axis path so it inherits that path's already-
            // verified output layout and keepdims stride convention.
            return Ok((arr.clone(), norm[0], ndim, None));
        }
        let mut keep: Vec<usize> = (0..ndim).filter(|i| !norm.contains(i)).collect();
        let nkeep = keep.len();
        let mut order: Vec<usize> = Vec::with_capacity(ndim);
        order.append(&mut keep);
        let mut reduced_sorted = norm.clone();
        reduced_sorted.sort_unstable();
        order.extend(reduced_sorted.iter().copied());
        let permuted = ionp_core::creation::moveaxis(arr, &order, &(0..ndim).collect::<Vec<_>>()).map_err(to_py_err)?;
        let mut shape: Vec<usize> = permuted.shape()[..nkeep].to_vec();
        shape.push(permuted.shape()[nkeep..].iter().product());
        let target = permuted.to_contiguous_order("C").map_err(to_py_err)?.reshape(&shape).map_err(to_py_err)?;
        return Ok((target, nkeep, ndim, Some(norm)));
    }
    // Anything that is not an integer took the iterate-as-a-sequence branch
    // above, so this extract cannot fail; the error path is kept rather than
    // unwrapped because that is a property of the branch above, not of this
    // line, and a future edit there should not turn into a panic here.
    let raw: isize = ax.extract().map_err(|_| axis_index_type_error(ax))?;
    let n = ndim as isize;
    // BUG FOUND + FIXED 2026-08-03 (Monday). The `n.max(1)` courtesy below
    // let a 0-d operand through with `axis=0`/`axis=-1`, returning the
    // scalar. That courtesy is real in numpy's REDUCTION family
    // (`np.sum(np.array(3.0), axis=0)` succeeds) but this module is not
    // that family: measured against real numpy 2.5.1
    // (/tmp/mg_0d_stats.py), ALL SIX functions reachable through this
    // helper -- `median`, `nanmedian`, `quantile`, `nanquantile`,
    // `percentile`, `nanpercentile` -- raise `AxisError` on a 0-d operand
    // for both `axis=0` and `axis=-1`. numpy is simply inconsistent
    // between the two families (`nanmean` grants it, `mean` does not), so
    // the courtesy has to be decided per-family rather than centrally.
    //
    // Safe to reject here for every caller: `resolve_axis` has exactly two
    // call sites, `do_median` and `do_quantile`, which are precisely those
    // six functions. `axis=None` returned above and is unaffected.
    if ndim == 0 {
        return Err(to_py_err(IonpError::AxisError { axis: raw, ndim: Some(0) }));
    }
    let norm = if raw < 0 { raw + n.max(1) } else { raw };
    if norm < 0 || norm >= n.max(1) {
        return Err(to_py_err(IonpError::AxisError { axis: raw, ndim: Some(ndim) }));
    }
    Ok((arr.clone(), norm as usize, ndim, None))
}

/// Post-processes a core result for the `axis=None, keepdims=True` case:
/// the core call is made with `keepdims=false` (since the core function
/// only knows about the flattened single axis), producing either a 0-d
/// result (median) or a `[nq]` result (quantile array `q`); this reshapes
/// it to append `orig_ndim` trailing size-1 axes, matching what real numpy
/// produces when the ORIGINAL (pre-flatten) array had `orig_ndim` axes.
fn expand_keepdims_none(result: NdArray, orig_ndim: usize) -> Result<NdArray, IonpError> {
    if orig_ndim == 0 {
        return Ok(result);
    }
    // A plain `reshape` here (the previous implementation) relabels onto
    // freshly-computed C-contiguous strides for the padded shape -- but
    // real numpy's `axis=None, keepdims=True` result carries a genuine
    // BROADCAST stride of 0 on every one of these inserted axes instead
    // (same convention `median_axis`'s own keepdims=True case uses --
    // see its doc comment in `ionp-core/src/stats.rs` -- confirmed
    // empirically for `median`/`nanmedian` via `/tmp/probe_median_mech2.py`
    // and `/tmp/probe_reduce_verify2.py`: e.g. `np.median(a, axis=None,
    // keepdims=True).strides == (0, 0, 0)` for a (2,3,4) input, never a
    // nonzero C-contiguous stride). `creation::insert_newaxis` implements
    // exactly this stride-0 insert convention -- NOT `creation::
    // expand_dims`, which (since Task #newaxis-split, 2026-08-08)
    // implements the DIFFERENT `a.reshape(...)`-based numpy function of
    // that name and would compute a genuine nonzero stride here instead.
    let axes: Vec<usize> = (result.ndim()..result.ndim() + orig_ndim).collect();
    ionp_core::creation::insert_newaxis(&result, &axes)
}

/// Restores the reduced axes for the TUPLE-axis path, which always runs the
/// core call with `keepdims=false` (the core only knows about the single
/// collapsed axis, so it cannot place `len(axes)` size-1 axes itself).
///
/// numpy's `_ureduce` finishes a keepdims multi-axis reduction with
/// `r[(Ellipsis,) + tuple(newaxis if i in axis else slice(None) ...)]` --
/// leading `Ellipsis`, so the inserts land on the TRAILING `nd` dimensions
/// and any leading `q` axis is skipped, hence the `q_ndim` offset. A
/// `newaxis` index is a genuine BROADCAST insert (stride 0), which is what
/// `creation::insert_newaxis` implements and what real numpy reports:
/// `np.median(a, axis=(0, 2), keepdims=True).strides` is `(0, 8, 0)` for a
/// C-contiguous (2,3,4) input, measured 2026-08-04. NOT `creation::
/// expand_dims` -- see `expand_keepdims_none` above for why that function
/// no longer means "stride-0 insert" as of Task #newaxis-split.
///
/// With `keepdims=false` there is nothing to insert and the core result is
/// already the right shape.
fn expand_keepdims_multi(result: NdArray, axes: &[usize], q_ndim: usize, keepdims: bool) -> Result<NdArray, IonpError> {
    if !keepdims || axes.is_empty() {
        return Ok(result);
    }
    let mut positions: Vec<usize> = axes.iter().map(|&a| a + q_ndim).collect();
    positions.sort_unstable();
    ionp_core::creation::insert_newaxis(&result, &positions)
}

/// numpy's `weak_q`, transcribed literally from `_function_base_impl.py`:
///
///     weak_q = type(q) in (int, float)  # use weak promotion for final result type
///
/// `type(q) in (...)` is an EXACT type test, so `bool` (an `int` subclass),
/// `np.float64`, `np.float32`, a list, a tuple and an ndarray are all
/// non-weak -- each measured live against numpy 2.5.1 on 2026-08-04:
/// `np.percentile(float32_arr, True)` is float64 while
/// `np.percentile(float32_arr, 50)` is float32.
///
/// `is_exact_instance_of` is PyO3's `type(o) is T`, which likewise excludes
/// `bool` from the `PyInt` test.
fn q_is_weak(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_exact_instance_of::<pyo3::types::PyInt>() || obj.is_exact_instance_of::<pyo3::types::PyFloat>()
}

/// Extracts `q` as (values, was_scalar). Accepts a bare Python
/// int/float/bool, or an array-like (`anionpy.ndarray`, numpy array, list,
/// tuple) of 0 or 1 dimensions -- matching `q must be a scalar or 1d`.
/// Whether `np.asanyarray(q)` would land on an INTEGER or BOOL dtype.
/// Mirrors `q_from_pyobj`'s own branching exactly so the two can never
/// disagree about what `q` is. A `float` (or anything containing one) is
/// not integral; a numpy float scalar is not integral; `True` is.
fn q_is_integral(obj: &Bound<'_, PyAny>) -> bool {
    // PyBool subclasses PyInt, so the bool test has to come first to be
    // meaningful -- both answer `true` here, but the ordering documents
    // that `bool` is deliberately included (numpy's gate is on the
    // resulting virtual-index dtype, and `bool * int` is an int).
    if obj.is_instance_of::<pyo3::types::PyBool>() || obj.is_instance_of::<pyo3::types::PyInt>() {
        return true;
    }
    if obj.is_instance_of::<pyo3::types::PyFloat>() {
        return false;
    }
    let arr = if obj.is_instance_of::<pyo3::types::PyList>() || obj.is_instance_of::<pyo3::types::PyTuple>() {
        ndarray_from_pylist(obj)
    } else {
        extract_array_like(obj)
    };
    match arr {
        Ok(a) => a.dtype() == DType::Bool || a.dtype().is_integer(),
        // A `q` this can't ingest will raise from `q_from_pyobj` a moment
        // later; answering `false` here just declines the shortcut.
        Err(_) => false,
    }
}

fn q_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, bool)> {
    if let Ok(v) = obj.extract::<f64>() {
        return Ok((vec![v], true));
    }
    let arr = if obj.is_instance_of::<pyo3::types::PyList>() || obj.is_instance_of::<pyo3::types::PyTuple>() {
        ndarray_from_pylist(obj)?
    } else {
        extract_array_like(obj)?
    };
    if arr.ndim() == 0 {
        let f64arr = arr.cast_to(DType::F64);
        let v = match f64arr.buffer() {
            Buffer::F64(v) => v[0],
            _ => unreachable!(),
        };
        return Ok((vec![v], true));
    }
    if arr.ndim() != 1 {
        return Err(PyValueError::new_err("q must be a scalar or 1d"));
    }
    let f64arr = arr.cast_to(DType::F64);
    let v = match f64arr.buffer() {
        Buffer::F64(v) => v.clone(),
        _ => unreachable!(),
    };
    Ok((v, false))
}

fn wrap(py: Python<'_>, inner: NdArray, out: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    if let Some(out_obj) = out {
        return write_into_out(out_obj, &inner, None);
    }
    // Root-cause fix (same defect class as the ufunc-reduce/trace/matmul
    // family, see `crate::numpy_scalar_from_0d`'s own doc comment): a full
    // `median`/`quantile`/`percentile` (scalar `q`, `axis=None`,
    // `keepdims=False` -- by far the common case) collapses to 0
    // dimensions, and real numpy returns a numpy SCALAR there too (verified
    // live against numpy 2.5.1: `np.median(np.arange(5))` is
    // `numpy.float64`, not `numpy.ndarray`; likewise `np.percentile(a, 50)`).
    // A non-scalar `q` always keeps at least a leading `len(q)` axis (never
    // 0-d) and an explicit `out=` always returns above before this point,
    // so neither is affected.
    if inner.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &inner);
    }
    use pyo3::IntoPyObjectExt;
    Py::new(py, PyArray { inner })?.into_py_any(py)
}

/// numpy's own escape hatch for an EMPTY operand in the `nan*` family.
///
/// `_nanmedian` / `_nanquantile_unchecked` both open with, verbatim:
///
///     # apply_along_axis in _nanmedian doesn't handle empty arrays well,
///     # so deal them upfront
///     if a.size == 0:
///         return np.nanmean(a, axis, out=out, keepdims=keepdims)
///
/// -- so a size-0 `nanmedian`/`nanquantile`/`nanpercentile` is literally a
/// `nanmean` call, and every consequence follows from that and not from any
/// quantile logic: `q` is DROPPED entirely (no leading `len(q)` axis in the
/// result, even for a list `q`), `method=` is never validated, and the
/// result dtype/shape/warning behaviour are `nanmean`'s. Measured live
/// against numpy 2.5.1 on 2026-08-04:
/// `np.nanpercentile(np.zeros((0,3)), [10,90], axis=(0,1), keepdims=True)`
/// is `array([[nan]])`, shape `(1, 1)` -- not `(2, 1, 1)`.
///
/// Dispatches through the public `anionpy` module rather than reaching into
/// `crate::reductions`, so it inherits `nanmean`'s own already-verified
/// axis validation, `out=` contract and keepdims layout with no second
/// implementation to keep in step. This calls IONP's `nanmean`, never
/// numpy's.
fn empty_nan_delegate_to_nanmean(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    let kwargs = pyo3::types::PyDict::new(py);
    match axis {
        Some(ax) => kwargs.set_item("axis", ax)?,
        None => kwargs.set_item("axis", py.None())?,
    }
    if let Some(o) = out {
        kwargs.set_item("out", o)?;
    }
    kwargs.set_item("keepdims", keepdims)?;
    use pyo3::IntoPyObjectExt;
    py.import("anionpy")?.getattr("nanmean")?.call((a,), Some(&kwargs))?.into_py_any(py)
}

#[allow(clippy::too_many_arguments)]
fn do_median(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
    skip_nan: bool,
) -> PyResult<Py<PyAny>> {
    let arr = extract_array_like(a)?;
    if skip_nan && arr.size() == 0 {
        return empty_nan_delegate_to_nanmean(py, a, axis, out, keepdims);
    }
    let axis_is_none = matches!(axis, None) || matches!(axis, Some(a) if a.is_none());
    // `true`: median alone inherits the operand's memory order for
    // `axis=()` -- see `resolve_axis`.
    let (target, ax, orig_ndim, multi) = resolve_axis(&arr, axis, true)?;
    let core_keepdims = keepdims && !axis_is_none && multi.is_none();
    let mut result = median_axis(&target, ax, core_keepdims, skip_nan).map_err(to_py_err)?;
    if axis_is_none && keepdims {
        result = expand_keepdims_none(result, orig_ndim).map_err(to_py_err)?;
    }
    if let Some(axes) = multi {
        result = expand_keepdims_multi(result, &axes, 0, keepdims).map_err(to_py_err)?;
    }
    wrap(py, result, out)
}

#[allow(clippy::too_many_arguments)]
fn do_quantile(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    method: &str,
    keepdims: bool,
    skip_nan: bool,
    as_percentile: bool,
) -> PyResult<Py<PyAny>> {
    let arr = extract_array_like(a)?;
    // Real numpy's `_quantile_unchecked`/`_quantile_ureduce_func` rejects a
    // complex-dtype `a` (`TypeError: a must be an array of real numbers`)
    // BEFORE it ever validates the `q` range -- confirmed live: with a
    // complex `a` AND an out-of-range `q` (e.g. percentile 5000.0), real
    // numpy still raises the complex-dtype TypeError, not the range
    // ValueError. `quantile_axis` below re-checks this (it has to, for
    // callers that reach it directly), but that check happens AFTER our
    // own `q` range validation a few lines down, so it must be duplicated
    // here, ahead of the range check, to match numpy's error-precedence.
    if matches!(arr.dtype(), DType::C64 | DType::C128) {
        return Err(pyo3::exceptions::PyTypeError::new_err("a must be an array of real numbers"));
    }
    let weak_q = q_is_weak(q);
    // `percentile` divides `q` by 100 before it ever reaches the virtual-
    // index computation, so its `q` array is ALWAYS float -- the integral
    // shortcut is structurally unreachable there and must not be claimed.
    let q_integral = !as_percentile && q_is_integral(q);
    let (mut q_vals, q_is_scalar) = q_from_pyobj(q)?;
    if as_percentile {
        for v in &q_vals {
            if !(0.0..=100.0).contains(v) {
                return Err(PyValueError::new_err("Percentiles must be in the range [0, 100]"));
            }
        }
        for v in &mut q_vals {
            *v /= 100.0;
        }
    } else {
        for v in &q_vals {
            if !(0.0..=1.0).contains(v) {
                return Err(PyValueError::new_err("Quantiles must be in the range [0, 1]"));
            }
        }
    }
    // AFTER the `q` range check (numpy validates `q` in the public
    // `nanquantile`/`nanpercentile`, above `_nanquantile_unchecked`'s
    // size-0 branch) and BEFORE the method check (numpy only resolves
    // `method` inside `_quantile`, which a size-0 operand never reaches --
    // so `np.nanquantile(np.zeros((0,)), 0.5, method="nope")` does NOT
    // raise). Measured live on numpy 2.5.1, 2026-08-04.
    if skip_nan && arr.size() == 0 {
        return empty_nan_delegate_to_nanmean(py, a, axis, out, keepdims);
    }
    let qmethod = QuantileMethod::from_str(method).map_err(to_py_err)?;
    let axis_is_none = matches!(axis, None) || matches!(axis, Some(a) if a.is_none());
    // `false`: quantile/percentile return C-order for `axis=()` even on an
    // F-contiguous operand, unlike median -- see `resolve_axis`.
    let (target, ax, orig_ndim, multi) = resolve_axis(&arr, axis, false)?;
    let core_keepdims = keepdims && !axis_is_none && multi.is_none();
    let args = QuantileArgs {
        q: &q_vals,
        q_is_scalar,
        method: qmethod,
        keepdims: core_keepdims,
        skip_nan,
        weak_q,
        q_is_integral: q_integral,
        out_is_none: out.is_none(),
    };
    let mut result = quantile_axis(&target, ax, &args).map_err(to_py_err)?;
    if axis_is_none && keepdims {
        result = expand_keepdims_none(result, orig_ndim).map_err(to_py_err)?;
    }
    if let Some(axes) = multi {
        let q_ndim = if q_is_scalar { 0 } else { 1 };
        result = expand_keepdims_multi(result, &axes, q_ndim, keepdims).map_err(to_py_err)?;
    }
    wrap(py, result, out)
}

// PyO3 can't combine a bare `&Bound<PyAny>` param with a literal `bool`
// default in `#[pyo3(signature=...)]` (the macro-generated default-value
// code can't convert a `bool` literal into a `&Bound<PyAny>`), so every
// plain-truthy bool-typed kwarg below is `Option<&Bound<PyAny>>` with a
// `None` pyo3 default, defaulted by hand via this helper.
fn opt_truthy_stats(obj: Option<&Bound<'_, PyAny>>, default: bool) -> PyResult<bool> {
    match obj {
        None => Ok(default),
        Some(v) => v.is_truthy(),
    }
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, overwrite_input=None, keepdims=None))]
fn median(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    overwrite_input: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Both `overwrite_input=` and `keepdims=` are plain Python truthiness
    // in real numpy (measured 2026-08-02, e.g. `median(a, keepdims=1.5)`
    // and `median(a, overwrite_input="x")` both succeed against real
    // numpy). `overwrite_input`'s truthiness doesn't change this
    // implementation's behavior (see module doc comment: it's already a
    // deliberate no-op), but a bad-typed value must still be ACCEPTED
    // (not TypeError'd) exactly where numpy accepts it -- so it still
    // needs the `is_truthy()` call, purely to reject the same things
    // numpy rejects (a raising `__bool__`) and accept the same things
    // numpy accepts.
    let overwrite_input = opt_truthy_stats(overwrite_input, false)?;
    let _ = overwrite_input;
    let keepdims = opt_truthy_stats(keepdims, false)?;
    do_median(py, a, axis, out, keepdims, false)
}

#[pyfunction]
#[pyo3(signature = (a, axis=None, out=None, overwrite_input=None, keepdims=None))]
fn nanmedian(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    overwrite_input: Option<&Bound<'_, PyAny>>,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let overwrite_input = opt_truthy_stats(overwrite_input, false)?;
    let _ = overwrite_input;
    let keepdims = opt_truthy_stats(keepdims, false)?;
    do_median(py, a, axis, out, keepdims, true)
}

#[pyfunction]
#[pyo3(signature = (a, q, axis=None, out=None, overwrite_input=None, method="linear", keepdims=None))]
#[allow(clippy::too_many_arguments)]
fn percentile(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    overwrite_input: Option<&Bound<'_, PyAny>>,
    method: &str,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // See `median` above: plain Python truthiness for both flags,
    // measured 2026-08-02.
    let overwrite_input = opt_truthy_stats(overwrite_input, false)?;
    let _ = overwrite_input;
    let keepdims = opt_truthy_stats(keepdims, false)?;
    do_quantile(py, a, q, axis, out, method, keepdims, false, true)
}

#[pyfunction]
#[pyo3(signature = (a, q, axis=None, out=None, overwrite_input=None, method="linear", keepdims=None))]
#[allow(clippy::too_many_arguments)]
fn quantile(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    overwrite_input: Option<&Bound<'_, PyAny>>,
    method: &str,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let overwrite_input = opt_truthy_stats(overwrite_input, false)?;
    let _ = overwrite_input;
    let keepdims = opt_truthy_stats(keepdims, false)?;
    do_quantile(py, a, q, axis, out, method, keepdims, false, false)
}

#[pyfunction]
#[pyo3(signature = (a, q, axis=None, out=None, overwrite_input=None, method="linear", keepdims=None))]
#[allow(clippy::too_many_arguments)]
fn nanpercentile(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    overwrite_input: Option<&Bound<'_, PyAny>>,
    method: &str,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let overwrite_input = opt_truthy_stats(overwrite_input, false)?;
    let _ = overwrite_input;
    let keepdims = opt_truthy_stats(keepdims, false)?;
    do_quantile(py, a, q, axis, out, method, keepdims, true, true)
}

#[pyfunction]
#[pyo3(signature = (a, q, axis=None, out=None, overwrite_input=None, method="linear", keepdims=None))]
#[allow(clippy::too_many_arguments)]
fn nanquantile(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    q: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    overwrite_input: Option<&Bound<'_, PyAny>>,
    method: &str,
    keepdims: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let overwrite_input = opt_truthy_stats(overwrite_input, false)?;
    let _ = overwrite_input;
    let keepdims = opt_truthy_stats(keepdims, false)?;
    do_quantile(py, a, q, axis, out, method, keepdims, true, false)
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(median, m)?)?;
    m.add_function(wrap_pyfunction!(nanmedian, m)?)?;
    m.add_function(wrap_pyfunction!(percentile, m)?)?;
    m.add_function(wrap_pyfunction!(quantile, m)?)?;
    m.add_function(wrap_pyfunction!(nanpercentile, m)?)?;
    m.add_function(wrap_pyfunction!(nanquantile, m)?)?;
    Ok(())
}
