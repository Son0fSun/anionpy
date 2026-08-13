//! `ndarray` attribute/method block -- metadata getters (`itemsize`,
//! `nbytes`, `mT`) and methods that reuse machinery already proven
//! elsewhere: `ravel`/`flatten` reuse `NdArray::reshape_with_order` (the
//! same primitive `PyArray::reshape` in `lib.rs` and the top-level `ravel`
//! wrapper in `creation.rs` already use); `squeeze`/`swapaxes` reuse
//! `ionp_core::creation::{squeeze, swapaxes}` (the top-level view functions
//! `creation.rs` already wraps -- this file mirrors that wrapper's
//! axis-normalization logic rather than duplicating the core functions);
//! `sum`/`prod`/`all`/`any`/`min`/`max` reuse `ionp_core::ufunc::reduce_binary`
//! (the same reduce kernel `PyUfunc::reduce` in `lib.rs` already exposes as
//! `anionpy.add.reduce` etc). `item`/`tolist`/`conj`/`conjugate` are new,
//! self-contained element-walk code, kept in this file specifically so it
//! never needs to touch `ufunc.rs`/`repr.rs` (shared, high-collision files
//! per this task's file-ownership fence).
//!
//! This is a SEPARATE `#[pymethods] impl PyArray` block from the one in
//! `lib.rs`, made possible by pyo3's `multiple-pymethods` Cargo feature
//! (enabled in `ionp-py/Cargo.toml`) -- deliberately, so this file's
//! churn never touches `lib.rs` beyond the one `mod ndarray_attrs;` line.

use pyo3::exceptions::{PyIndexError, PyKeyError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyComplex, PyDict, PyList, PyTuple};
use pyo3::IntoPyObjectExt;

use ionp_core::ufunc::{accumulate_axis, binary_op, normalize_reduce_axes, reduce_axis, BinaryOp, ExtremeOp};
use ionp_core::{creation, manip, promote_dtype, Buffer, DType, IonpError, NdArray, Order, ScalarKind};

use crate::{
    apply_ufunc_order, axis_error, axis_error_prefixed, classify_scalar, dtype_from_pyobj,
    extract_array, scalar_against, to_py_err, weak_scalar_buffer, write_into_out, PyArray,
};

// ---------------------------------------------------------------------------
// Element extraction: buffer value -> native Python scalar. Mirrors numpy's
// own `.item()`/`.tolist()` promotion -- every dtype narrower than the
// native Python type (int8..int32 -> int, float16/float32 -> float) is
// upcast to the native type, exactly as real numpy's `.item()` does (never
// a numpy scalar type, a bare Python int/float/bool/complex -- verified
// against real numpy 2.5.1). `flat_offset` is a raw buffer index (i.e.
// already includes the array's own `.offset()` -- see `nth_offset` below),
// not a logical (shape-relative) index.
// ---------------------------------------------------------------------------

pub(crate) fn elem_to_py(py: Python<'_>, arr: &NdArray, flat_offset: isize) -> PyResult<Py<PyAny>> {
    let i = flat_offset as usize;
    match arr.buffer() {
        Buffer::Bool(v) => v[i].into_py_any(py),
        Buffer::I8(v) => (v[i] as i64).into_py_any(py),
        Buffer::I16(v) => (v[i] as i64).into_py_any(py),
        Buffer::I32(v) => (v[i] as i64).into_py_any(py),
        Buffer::I64(v) => v[i].into_py_any(py),
        Buffer::U8(v) => (v[i] as u64).into_py_any(py),
        Buffer::U16(v) => (v[i] as u64).into_py_any(py),
        Buffer::U32(v) => (v[i] as u64).into_py_any(py),
        Buffer::U64(v) => v[i].into_py_any(py),
        Buffer::F16(v) => v[i].to_f64().into_py_any(py),
        Buffer::F32(v) => (v[i] as f64).into_py_any(py),
        Buffer::F64(v) => v[i].into_py_any(py),
        Buffer::C64(v) => {
            let c = v[i];
            PyComplex::from_doubles(py, c.re as f64, c.im as f64).into_py_any(py)
        }
        Buffer::C128(v) => {
            let c = v[i];
            PyComplex::from_doubles(py, c.re, c.im).into_py_any(py)
        }
        // `S`/`U` scalar extraction: mirrors numpy's `.item()`/`.tolist()`
        // for string dtypes, which strip only the TRAILING run of NUL
        // padding (an embedded NUL mid-string survives) and hand back a
        // native Python `bytes`/`str`, never a numpy scalar type --
        // verified against real numpy 2.5.1 (`np.array([b'ab\x00cd\x00\x00'],
        // dtype='S8')[0] == b'ab\x00cd'`).
        Buffer::S(_, v) => {
            let elem = &v[i];
            let end = elem.iter().rposition(|&b| b != 0).map(|p| p + 1).unwrap_or(0);
            PyBytes::new(py, &elem[..end]).into_py_any(py)
        }
        Buffer::U(_, v) => {
            let elem = &v[i];
            let end = elem.iter().rposition(|&c| c != 0).map(|p| p + 1).unwrap_or(0);
            let s: String = elem[..end]
                .iter()
                .map(|&c| char::from_u32(c).unwrap_or('\u{FFFD}'))
                .collect();
            s.into_py_any(py)
        }
    }
}

/// The raw buffer index of the `n`-th element in this array's logical
/// (C-order) walk -- `NdArray::iter_offsets()` yields offsets relative to
/// the array's own `.offset()`, matching the convention already
/// established by `NdArray::to_contiguous()` (`buf[(self.offset + off) as
/// usize]`, see array.rs).
pub(crate) fn nth_offset(arr: &NdArray, n: usize) -> Option<isize> {
    arr.iter_offsets().nth(n).map(|off| arr.offset() + off)
}

fn all_offsets(arr: &NdArray) -> Vec<isize> {
    arr.iter_offsets().map(|off| arr.offset() + off).collect()
}

/// Recursively nest a flat, C-order list of already-converted Python
/// scalars into `.tolist()`'s shape-matching nested-list structure. A 0-d
/// shape returns the bare scalar itself (numpy: `np.array(3).tolist() ==
/// 3`, NOT `[3]`) -- handled by the caller, not here, since this function
/// assumes at least one dimension.
fn nest_list(py: Python<'_>, values: &[Py<PyAny>], shape: &[usize]) -> PyResult<Py<PyAny>> {
    if shape.len() == 1 {
        let list = PyList::new(py, values.iter().map(|v| v.clone_ref(py)))?;
        return Ok(list.into_any().unbind());
    }
    let (head, rest) = (shape[0], &shape[1..]);
    let chunk = rest.iter().product::<usize>();
    let mut rows = Vec::with_capacity(head);
    for i in 0..head {
        rows.push(nest_list(py, &values[i * chunk..(i + 1) * chunk], rest)?);
    }
    let list = PyList::new(py, rows)?;
    Ok(list.into_any().unbind())
}

fn normalize_axis(ax: isize, ndim: usize) -> PyResult<usize> {
    let n = ndim as isize;
    let norm = if ax < 0 { ax + n } else { ax };
    if norm < 0 || norm >= n {
        // numpy.exceptions.AxisError (subclasses both ValueError and
        // IndexError), not a plain IndexError -- matches the real
        // squeeze(axis=<oob>)/swapaxes(<oob>, ...) exception type. See
        // `crate::axis_error` (used identically by the axis-reduction path
        // in this same file via `normalize_reduce_axes`) -- this was
        // previously a plain PyIndexError, a documented type gap that kept
        // the out-of-range-axis CallForms scoped out of
        // tests/differential/ndarray_attrs_cases.py.
        Err(axis_error(
            ax,
            Some(ndim),
            &format!("axis {ax} is out of bounds for array of dimension {ndim}"),
        ))
    } else {
        Ok(norm as usize)
    }
}

/// `normalize_axis`'s sibling for `ndarray.swapaxes`, which (like the
/// top-level `swapaxes` in creation.rs) numpy labels with a `msg_prefix` of
/// which parameter was out of range: `AxisError(axis, ndim, 'axis1')` /
/// `('axis2')`. Verified directly: `np.zeros((2,3)).swapaxes(5, 0)` ->
/// `'axis1: axis 5 is out of bounds for array of dimension 2'`. Found
/// 2026-08-01 fixing the same class of bug as top-level `swapaxes`/
/// `moveaxis`/`linspace(axis=...)` under the harness's new exact-message
/// comparison (f69aceb).
fn normalize_axis_labeled(ax: isize, ndim: usize, label: &str) -> PyResult<usize> {
    let n = ndim as isize;
    let norm = if ax < 0 { ax + n } else { ax };
    if norm < 0 || norm >= n {
        Err(axis_error_prefixed(ax, ndim, label))
    } else {
        Ok(norm as usize)
    }
}

fn axes_list_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(seq) = obj.extract::<Vec<isize>>() {
        Ok(seq)
    } else {
        Ok(vec![obj.extract::<isize>()?])
    }
}

/// Method-form axis converter -- now a THIN DELEGATION to
/// `reductions.rs`'s `reduce_axes_from_pyobj`, not a copy of it.
///
/// It used to be a local copy, "per this file's established convention of
/// small per-module copies rather than a cross-file import". That
/// convention is what broke it. The copy carried none of the free-function
/// side's axis guards, so every fix made there stopped at the module
/// boundary and the METHOD forms stayed wrong. Measured 2026-08-04, 13
/// methods x 5 operand shapes x 31 axis forms: 678 diverging cases,
/// including silent wrong ANSWERS --
///     np.zeros((2, 3)).sum(axis=[0])          -> TypeError: 'list' object
///                                                cannot be interpreted ...
///     anionpy.zeros((2, 3)).sum(axis=[0])        -> reduced over axis 0
/// A duplicated converter that drifts from the original is worse than the
/// import it was avoiding.
///
/// Deliberately still NOT folded into this file's `axes_list_from_pyobj`
/// above: that one also serves `squeeze`, which normalizes through
/// `manip::normalize_axis` and must not inherit reduction strictness.
///
/// Every reduction reachable as an `ndarray` METHOD (`a.sum(axis=0)`,
/// `a.min(axis=0)`, ...) is in the courtesy-granting group, hence the
/// hard-coded `true`: `a.mean(axis=0)` on a 0-d operand routes through
/// `reductions.rs`, not here.
fn reduce_axes_from_pyobj(obj: &Bound<'_, PyAny>, ndim: usize) -> PyResult<Vec<isize>> {
    crate::reductions::reduce_axes_from_pyobj(
        obj,
        ndim,
        true,
        crate::reductions::AxisSeq::TupleOnly,
    )
}

/// Shared driver for `sum`/`prod`/`all`/`any`/`min`/`max`'s axis-reduction
/// signatures. `dtype`/`initial` are `None` for `all`/`any` (numpy's real
/// signatures don't have those parameters at all -- callers pass `None`
/// unconditionally at the call site, never surfacing them to Python).
///
/// `where=True` (the literal Python bool singleton, numpy's own default)
/// is deliberately normalized to "no mask" rather than routed through
/// `reduce_axis`'s masked path: real numpy short-circuits that case
/// internally too, and taking the masked branch here would silently lose
/// pairwise-summation bit-exactness on `sum` for no semantic reason (see
/// `reduce_axis`'s doc comment on why the masked path is sequential-only).
/// Any other `where=` value (an array, or the literal `False`) DOES take
/// the masked path, honestly reported as value-correct/not bit-exact.
#[allow(clippy::too_many_arguments)]
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
        // 0-d: scalar axis is numpy's courtesy no-op, any non-empty
        // sequence raises AxisError. See `reduce_axes_from_pyobj` above.
        Some(ax) => Some(reduce_axes_from_pyobj(ax, ndim)?),
    };
    let axes = normalize_reduce_axes(axes_raw.as_deref(), ndim).map_err(to_py_err)?;

    let dtype_override: Option<DType> = match dtype {
        None => None,
        Some(d) if d.is_none() => None,
        Some(d) => Some(crate::dtype_from_pyobj_no_su(d)?),
    };

    let initial_arr: Option<NdArray> = match initial {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => Some(extract_array(v)?),
    };

    let mask_arr: Option<NdArray> = match r#where {
        None => None,
        Some(w) => match w.extract::<bool>() {
            Ok(true) => None,
            Ok(false) => Some(extract_array(w)?),
            Err(_) => Some(extract_array(w)?),
        },
    };

    let result = reduce_axis(
        op,
        a,
        &axes,
        keepdims,
        dtype_override,
        initial_arr.as_ref(),
        mask_arr.as_ref(),
    )
    .map_err(to_py_err)?;

    if let Some(out_obj) = out {
        return write_into_out(out_obj, &result, None);
    }
    if result.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &result);
    }
    Ok(Py::new(py, PyArray { inner: result })?.into_any())
}

#[pymethods]
impl PyArray {
    // -- metadata getters, plain arithmetic on shape/dtype already held --
    #[getter]
    fn itemsize(&self) -> usize {
        self.inner.dtype().itemsize()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.size() * self.inner.dtype().itemsize()
    }

    /// `ndarray.base`. numpy: `None` if the array owns its data, else the
    /// chain-flattened ultimate parent (`a[1:][1:].base is a`, never an
    /// intermediate view -- verified against real numpy 2.5.1). Backed by
    /// the `_base` attribute the view-producing methods (`__getitem__`,
    /// `.T`/`.transpose()` so far -- see `lib.rs::attach_base`) set on this
    /// instance's real Python `__dict__` (`#[pyclass(..., dict)]` on
    /// `PyArray`), not a Rust struct field -- see that pyclass's doc
    /// comment for why a field was rejected (~230 unrelated construction
    /// sites would all need touching).
    #[getter]
    fn base<'py>(slf: &Bound<'py, PyArray>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match slf.getattr("_base") {
            Ok(v) if !v.is_none() => Ok(Some(v)),
            _ => Ok(None),
        }
    }

    /// `ndarray.flags`. Exposes exactly the four keys this task's brief
    /// names as the minimum bar (`C_CONTIGUOUS`, `F_CONTIGUOUS`, `OWNDATA`,
    /// `WRITEABLE`) -- real numpy's `flags` object has several more
    /// (`ALIGNED`, `WRITEBACKIFCOPY`, ...) not attempted here. `OWNDATA` is
    /// the negation of "has a `.base`" (numpy: a view's OWNDATA is always
    /// False; verified live, not merely assumed plausible). `WRITEABLE` is
    /// unconditionally `True`: anionpy has no read-only-array concept and no
    /// `__setitem__` at all yet (see KNOWN-DIFFERENCES.md / this task's
    /// report) -- reporting `True` matches numpy's OWN default for every
    /// array this constructor family can produce (numpy only sets
    /// `WRITEABLE=False` for a few special cases -- e.g. some `.base`
    /// chains through non-writeable buffers -- none of which anionpy can
    /// construct), but the actual write path this flag nominally describes
    /// does not exist in anionpy yet.
    #[getter]
    fn flags(slf: &Bound<'_, PyArray>) -> PyResult<PyFlags> {
        let has_base = matches!(slf.getattr("_base"), Ok(v) if !v.is_none());
        // `_readonly` mirrors `_base`'s Python-`__dict__` attribute trick
        // (see `lib.rs::mark_readonly`'s doc comment): unset on every array
        // except a `broadcast_to`/`broadcast_arrays` result, where numpy's
        // real `WRITEABLE=False` is a genuine, independent-of-OWNDATA
        // property (a broadcast view repeats elements via a zero stride, so
        // a write through it would silently write more than one logical
        // position).
        let readonly = matches!(
            slf.getattr("_readonly"),
            Ok(v) if v.extract::<bool>().unwrap_or(false)
        );
        let inner = &slf.borrow().inner;
        Ok(PyFlags {
            c_contiguous: inner.is_c_contiguous(),
            f_contiguous: inner.is_f_contiguous(),
            owndata: !has_base,
            writeable: !readonly,
        })
    }

    /// `ndarray.real`. numpy: for a complex dtype, returns the real
    /// component as the matching real dtype (complex64 -> float32,
    /// complex128 -> float64); for every non-complex dtype (bool/int/
    /// uint/float, including float16), `.real` is the array itself,
    /// UNCHANGED dtype and values (verified against real numpy 2.5.1
    /// across all 8 non-complex dtypes -- `.real` is not a no-op cast to
    /// float, it is a genuine identity on those dtypes). Returns a COPY,
    /// not numpy's zero-copy view (anionpy has no view mechanism at all, see
    /// `KNOWN-DIFFERENCES.md` / this task's brief -- values and dtype are
    /// exact, the aliasing guarantee is not attempted or claimed).
    #[getter]
    fn real(&self) -> PyArray {
        let contig = self.inner.to_contiguous();
        let out_buffer = match contig.buffer() {
            Buffer::C64(v) => Buffer::F32(v.iter().map(|c| c.re).collect()),
            Buffer::C128(v) => Buffer::F64(v.iter().map(|c| c.re).collect()),
            other => other.clone(),
        };
        let inner = NdArray::from_buffer(out_buffer, contig.shape().to_vec(), ionp_core::Order::C)
            .expect("real: buffer/shape already validated by to_contiguous()'s own construction");
        PyArray { inner }
    }

    /// `ndarray.imag`. numpy: for a complex dtype, the imaginary component
    /// as the matching real dtype; for every non-complex dtype, an array
    /// of the SAME dtype and shape filled with the type's zero (`False`
    /// for bool, `0` for every int/uint width, `0.0` for every float
    /// width including float16 -- verified against real numpy 2.5.1).
    /// Same copy-not-view caveat as `.real` above.
    #[getter]
    fn imag(&self) -> PyArray {
        let contig = self.inner.to_contiguous();
        let out_buffer = match contig.buffer() {
            Buffer::C64(v) => Buffer::F32(v.iter().map(|c| c.im).collect()),
            Buffer::C128(v) => Buffer::F64(v.iter().map(|c| c.im).collect()),
            other => creation::zeros_buffer(other.dtype(), other.len()),
        };
        let inner = NdArray::from_buffer(out_buffer, contig.shape().to_vec(), ionp_core::Order::C)
            .expect("imag: buffer/shape already validated by to_contiguous()'s own construction");
        PyArray { inner }
    }

    /// `ndarray.__copy__()` -- the `copy.copy()` protocol hook. numpy:
    /// `a.__copy__()` takes no arguments and returns a fresh array with
    /// the SAME memory order as `a` (verified against real numpy 2.5.1:
    /// `np.asfortranarray(a).__copy__()` stays F-contiguous, matching
    /// `a.copy(order='K')`, not `a.copy()`'s own default of `'C'` --
    /// `ndarray.copy`'s default arg is `'C'`, `__copy__`'s effective order
    /// is `'K'`. Reuses the same `to_contiguous_order` primitive
    /// `PyArray::copy` (lib.rs) already uses and is declared exact
    /// against, just with a different hardcoded order string).
    fn __copy__(&self) -> PyResult<PyArray> {
        let inner = self.inner.to_contiguous_order("K").map_err(to_py_err)?;
        Ok(PyArray { inner })
    }

    /// `ndarray.__deepcopy__(memo)` -- the `copy.deepcopy()` protocol
    /// hook. numpy requires exactly one positional argument (a memo dict,
    /// verified live: `a.__deepcopy__()` raises `TypeError: __deepcopy__()
    /// takes exactly 1 argument (0 given)`, `a.__deepcopy__(1, 2)` raises
    /// the 2-given form) -- pyo3 enforces the same arity by construction
    /// (`memo` is a required positional parameter with no `signature`
    /// override), raising its own `TypeError` for a missing/extra
    /// argument. An `ndarray` has no nested Python object references (a
    /// numeric buffer, not a container of arbitrary objects), so a "deep"
    /// copy is byte-identical to a shallow one -- verified against real
    /// numpy, which does exactly this internally (`__deepcopy__` never
    /// even reads `memo`). Same order semantics as `__copy__` above.
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> PyResult<PyArray> {
        let inner = self.inner.to_contiguous_order("K").map_err(to_py_err)?;
        Ok(PyArray { inner })
    }

    /// `ndarray.mT` (NEP 51 / numpy >= 2.0): transpose of the last two
    /// axes only, leaving any leading batch axes alone. Requires ndim >=
    /// 2 -- numpy raises `ValueError: matrix transpose with ndim < 2 is
    /// undefined` for a 0-d or 1-d array (verified against real numpy
    /// 2.5.1), matched here rather than silently no-op'ing like bare `.T`
    /// does for those ranks.
    ///
    /// Takes `slf: &Bound<PyArray>` rather than `&self` so the result can
    /// carry a real `.base`/`OWNDATA=False` through `wrap_shape_view`: this
    /// IS a view (`transpose_axes` only permutes strides), and reporting it
    /// as data-owning was a measured divergence from numpy, not a cosmetic
    /// one -- `.base` and `.flags` are directly queryable.
    #[getter(mT)]
    fn matrix_transpose(slf: &Bound<'_, PyArray>) -> PyResult<Py<PyArray>> {
        let ndim = slf.borrow().inner.ndim();
        if ndim < 2 {
            return Err(PyValueError::new_err(
                "matrix transpose with ndim < 2 is undefined",
            ));
        }
        let mut axes: Vec<isize> = (0..(ndim as isize - 2)).collect();
        axes.push(ndim as isize - 1);
        axes.push(ndim as isize - 2);
        let before = slf.borrow().inner.clone();
        let out = before.transpose_axes(&axes).map_err(to_py_err)?;
        crate::wrap_shape_view(slf.py(), slf.as_any(), &before, out)
    }

    // -- view/copy methods, composed from already-proven primitives --

    /// Same order semantics as `PyArray::reshape`/top-level `ravel`
    /// (`creation.rs`), plus `'K'` ("read the elements in the order they
    /// occur in memory") which `reshape_with_order` deliberately refuses
    /// (numpy itself refuses 'K' for reshaping specifically) but ravel and
    /// flatten accept -- `NdArray::ravel_order` (`array.rs`) is the shared
    /// primitive for all four letters here.
    ///
    /// `wrap_shape_view` attaches a base only when the result actually
    /// shares the source buffer, which is exactly numpy's own rule for
    /// `ravel` (a view when the requested order can be satisfied by
    /// reinterpreting the existing bytes, a copy otherwise) -- so the
    /// copy path keeps reporting `OWNDATA=True`/`base=None` without a
    /// special case here.
    #[pyo3(signature = (order=None))]
    fn ravel(slf: &Bound<'_, PyArray>, order: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyArray>> {
        let order_letter = crate::check_ufunc_order_kwarg(order)?;
        let ord_string = order_letter.map(|c| c.to_string()).unwrap_or_else(|| "C".to_string());
        let before = slf.borrow().inner.clone();
        let out = crate::ravel_view_or_copy(&before, &ord_string)?;
        crate::wrap_shape_view(slf.py(), slf.as_any(), &before, out)
    }

    /// Same as `ravel` above but always an independent copy (numpy:
    /// `.flatten()` never returns a view, `.ravel()` does when it can) --
    /// `ravel_order` already gathers into a freshly allocated buffer, so
    /// there is nothing further to force here (unlike the old
    /// `reshape_with_order` + `to_contiguous()` composition, which needed
    /// `to_contiguous()` because `reshape_with_order` could hand back a
    /// view).
    #[pyo3(signature = (order=None))]
    fn flatten(&self, order: Option<&Bound<'_, PyAny>>) -> PyResult<PyArray> {
        let order_letter = crate::check_ufunc_order_kwarg(order)?;
        let ord_string = order_letter.map(|c| c.to_string()).unwrap_or_else(|| "C".to_string());
        let out = self.inner.ravel_order(&ord_string).map_err(to_py_err)?;
        Ok(PyArray { inner: out })
    }

    /// Mirrors `creation::squeeze`'s top-level pyfunction wrapper
    /// (`creation.rs`) exactly (axis normalization then delegate to
    /// `ionp_core::creation::squeeze`) -- duplicated here rather than
    /// imported because that wrapper is a private `#[pyfunction]`, not a
    /// reusable `fn`, and this task's file-ownership fence keeps
    /// `creation.rs` off limits to edit (e.g. to make it `pub`).
    #[pyo3(signature = (axis=None))]
    fn squeeze(slf: &Bound<'_, PyArray>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyArray>> {
        let this = slf.borrow().inner.clone();
        let ndim = this.ndim();
        // numpy special-case: `arr.squeeze(axis=0)`/`arr.squeeze(axis=-1)`
        // on a 0-d array is a no-op (returns the same shape), NOT an
        // AxisError, even though axis 0 doesn't literally exist on a 0-d
        // array -- verified against real numpy 2.5.1
        // (`np.array(5.0).squeeze(axis=0)` -> `array(5.0)`, shape `()`).
        // Only a BARE int axis gets this treatment; `axis=(0,)` still goes
        // through the normal AxisError path below (not exercised by the
        // differential corpus, kept narrow to what's verified).
        if ndim == 0 {
            if let Some(ax) = axis {
                if let Ok(single) = ax.extract::<isize>() {
                    if single == 0 || single == -1 {
                        let out = creation::squeeze(&this, Some(&[])).map_err(to_py_err)?;
                        return crate::identity_if_unchanged(slf.py(), slf.as_any(), &this, out);
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
                    .collect::<PyResult<_>>()?,
            ),
        };
        let out = creation::squeeze(&this, axes.as_deref()).map_err(to_py_err)?;
        crate::identity_if_unchanged(slf.py(), slf.as_any(), &this, out)
    }

    /// Mirrors `creation.rs`'s top-level `swapaxes` pyfunction, same
    /// reasoning as `squeeze` above.
    fn swapaxes(slf: &Bound<'_, PyArray>, axis1: isize, axis2: isize) -> PyResult<Py<PyArray>> {
        let before = slf.borrow().inner.clone();
        let ndim = before.ndim();
        let ax1 = normalize_axis_labeled(axis1, ndim, "axis1")?;
        let ax2 = normalize_axis_labeled(axis2, ndim, "axis2")?;
        let out = creation::swapaxes(&before, ax1, ax2).map_err(to_py_err)?;
        crate::wrap_shape_view(slf.py(), slf.as_any(), &before, out)
    }

    // -- scalar/list extraction --

    /// numpy `ndarray.item()` (no args, or a single empty tuple: requires
    /// size == 1, raises `ValueError` otherwise); `ndarray.item(i)` /
    /// `ndarray.item((i,))` (a SINGLE index, given either bare or wrapped
    /// in a 1-tuple: always a flat C-order index into the whole array,
    /// regardless of `ndim` -- this is the one case numpy accepts
    /// independent of dimensionality); and `ndarray.item(i0, i1, ...)` /
    /// `ndarray.item((i0, i1, ...))` (exactly `ndim` per-axis indices,
    /// wrapped or not).
    ///
    /// BUG FOUND AND FIXED 2026-08-01: this previously only accepted 0 or
    /// 1 positional args (rejecting the N-arg/tuple per-axis form outright
    /// with a made-up `TypeError`, and using generic non-numpy wording for
    /// the single-flat-index out-of-bounds case). Verified against real
    /// numpy 2.5.1 that:
    ///   - a single index (bare int OR a length-1 tuple) is ALWAYS a flat
    ///     C-order index over the whole array, for ANY `ndim` -- e.g.
    ///     `np.arange(6).reshape(2,3).item(5) == 5`. Its out-of-bounds
    ///     message depends on `ndim`: for a 1-d array it uses the same
    ///     per-axis wording as the N-arg form below (`'index {i} is out of
    ///     bounds for axis 0 with size {n}'`, since flat and per-axis-0
    ///     coincide); for any other `ndim` it uses whole-array wording
    ///     (`'index {i} is out of bounds for size {n}'`, no "axis" text).
    ///   - exactly `ndim` indices (bare varargs OR one `ndim`-tuple) are
    ///     per-axis indices, each independently bounds-checked against its
    ///     own axis with numpy's `AxisError`-flavored wording (`'index {i}
    ///     is out of bounds for axis {k} with size {dim_k}'`), and the
    ///     ORIGINAL (not axis-normalized) index value is what's reported.
    ///   - any other argument count raises `ValueError('incorrect number
    ///     of indices for array')` -- e.g. `np.arange(6).reshape(2,3)
    ///     .item(0, 1, 0)`.
    ///   - `item(())` (an explicit empty tuple) is identical to no args at
    ///     all -- the size==1 check.
    /// Always returns a native Python scalar (int/float/bool/complex),
    /// never a numpy or anionpy scalar/0-d-array type -- see `elem_to_py`'s
    /// docstring.
    #[pyo3(signature = (*args))]
    fn item(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
        let n = self.inner.size();
        let ndim = self.inner.ndim();
        let shape = self.inner.shape();

        // Collapse a single tuple-shaped argument into its own contents --
        // `item((0, 1))` behaves identically to `item(0, 1)`.
        let indices: Vec<isize> = if args.len() == 1 {
            if let Ok(inner_tuple) = args.get_item(0)?.cast::<PyTuple>() {
                inner_tuple.iter().map(|v| v.extract::<isize>()).collect::<PyResult<_>>()?
            } else {
                vec![args.get_item(0)?.extract::<isize>()?]
            }
        } else {
            args.iter().map(|v| v.extract::<isize>()).collect::<PyResult<_>>()?
        };

        if indices.is_empty() {
            if n != 1 {
                return Err(PyValueError::new_err(
                    "can only convert an array of size 1 to a Python scalar",
                ));
            }
            let off = nth_offset(&self.inner, 0).expect("size == 1 implies at least one element");
            return elem_to_py(py, &self.inner, off);
        }

        if indices.len() == 1 {
            let raw = indices[0];
            if ndim == 1 {
                let dim0 = shape[0];
                let adjusted = if raw < 0 { raw + dim0 as isize } else { raw };
                if adjusted < 0 || adjusted as usize >= dim0 {
                    return Err(PyIndexError::new_err(format!(
                        "index {raw} is out of bounds for axis 0 with size {dim0}"
                    )));
                }
                let off = nth_offset(&self.inner, adjusted as usize).expect("bounds already checked above");
                return elem_to_py(py, &self.inner, off);
            }
            let adjusted = if raw < 0 { raw + n as isize } else { raw };
            if adjusted < 0 || adjusted as usize >= n {
                return Err(PyIndexError::new_err(format!("index {raw} is out of bounds for size {n}")));
            }
            let off = nth_offset(&self.inner, adjusted as usize).expect("bounds already checked above");
            return elem_to_py(py, &self.inner, off);
        }

        if indices.len() != ndim {
            return Err(PyValueError::new_err("incorrect number of indices for array"));
        }
        let mut flat = 0usize;
        for (axis, &raw) in indices.iter().enumerate() {
            let dim = shape[axis];
            let adjusted = if raw < 0 { raw + dim as isize } else { raw };
            if adjusted < 0 || adjusted as usize >= dim {
                return Err(PyIndexError::new_err(format!(
                    "index {raw} is out of bounds for axis {axis} with size {dim}"
                )));
            }
            let stride_extent: usize = shape[axis + 1..].iter().product();
            flat += adjusted as usize * stride_extent;
        }
        let off = nth_offset(&self.inner, flat).expect("bounds already checked above");
        elem_to_py(py, &self.inner, off)
    }

    /// numpy `ndarray.tolist()`: nested Python list of native scalars, or
    /// (for a 0-d array) the bare scalar itself -- `np.array(3).tolist()
    /// == 3`, not `[3]` (verified against real numpy 2.5.1).
    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let offsets = all_offsets(&self.inner);
        let mut values = Vec::with_capacity(offsets.len());
        for off in offsets {
            values.push(elem_to_py(py, &self.inner, off)?);
        }
        let shape = self.inner.shape();
        if shape.is_empty() {
            return Ok(values.into_iter().next().expect("0-d array has exactly one element"));
        }
        nest_list(py, &values, shape)
    }

    // -- axis reductions: real signatures matching numpy's own (verified
    //    against real numpy 2.5.1 `help()` text, which is NOT uniform
    //    across these six -- `sum`/`prod` alone carry `dtype=`/`initial=`
    //    defaulting to the op's identity (0/1); `min`/`max` carry
    //    `initial=` but NOT `dtype=`; `all`/`any` carry NEITHER `dtype=`
    //    nor `initial=`. Accepting a parameter numpy doesn't have on a
    //    given method would make a signature MORE permissive than numpy's,
    //    which is its own divergence -- see PATH-TO-100.md's "test the
    //    SIGNATURE, not just the values" note this task was written to
    //    fix.). All six now route through the shared `reduce_axis` kernel
    //    (`ufunc.rs`) rather than the old always-`full=true`
    //    `reduce_binary` call.

    #[pyo3(signature = (axis=None, dtype=None, out=None, keepdims=false, initial=None, r#where=None))]
    #[allow(clippy::too_many_arguments)]
    fn sum(
        &self,
        py: Python<'_>,
        axis: Option<&Bound<'_, PyAny>>,
        dtype: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
        keepdims: bool,
        initial: Option<&Bound<'_, PyAny>>,
        r#where: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        do_reduce_axis(py, BinaryOp::Add, &self.inner, axis, dtype, out, keepdims, initial, r#where)
    }

    #[pyo3(signature = (axis=None, dtype=None, out=None, keepdims=false, initial=None, r#where=None))]
    #[allow(clippy::too_many_arguments)]
    fn prod(
        &self,
        py: Python<'_>,
        axis: Option<&Bound<'_, PyAny>>,
        dtype: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
        keepdims: bool,
        initial: Option<&Bound<'_, PyAny>>,
        r#where: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        do_reduce_axis(py, BinaryOp::Multiply, &self.inner, axis, dtype, out, keepdims, initial, r#where)
    }

    #[pyo3(signature = (axis=None, out=None, keepdims=false, r#where=None))]
    fn all(
        &self,
        py: Python<'_>,
        axis: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
        keepdims: bool,
        r#where: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        do_reduce_axis(py, BinaryOp::LogicalAnd, &self.inner, axis, None, out, keepdims, None, r#where)
    }

    #[pyo3(signature = (axis=None, out=None, keepdims=false, r#where=None))]
    fn any(
        &self,
        py: Python<'_>,
        axis: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
        keepdims: bool,
        r#where: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        do_reduce_axis(py, BinaryOp::LogicalOr, &self.inner, axis, None, out, keepdims, None, r#where)
    }

    #[pyo3(signature = (axis=None, out=None, keepdims=false, initial=None, r#where=None))]
    fn min(
        &self,
        py: Python<'_>,
        axis: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
        keepdims: bool,
        initial: Option<&Bound<'_, PyAny>>,
        r#where: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        do_reduce_axis(py, BinaryOp::Minimum, &self.inner, axis, None, out, keepdims, initial, r#where)
    }

    #[pyo3(signature = (axis=None, out=None, keepdims=false, initial=None, r#where=None))]
    fn max(
        &self,
        py: Python<'_>,
        axis: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
        keepdims: bool,
        initial: Option<&Bound<'_, PyAny>>,
        r#where: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        do_reduce_axis(py, BinaryOp::Maximum, &self.inner, axis, None, out, keepdims, initial, r#where)
    }

    // -- complex conjugate. No `ionp_core::ufunc` op for this exists yet;
    //    implemented here as a direct element walk over the array's own
    //    contiguous copy rather than adding a new op to the shared
    //    ufunc.rs (this task's file-ownership fence keeps that file off
    //    limits).
    //
    //    CORRECTED 2026-08-04. The note that stood here said real-dtype
    //    inputs are "returned as a plain contiguous copy, dtype unchanged,
    //    matching numpy exactly". The dtype half was right and the rest was
    //    not, and the error was in assuming the METHOD behaves like the
    //    ufunc. It does not -- measured live on numpy 2.5.1:
    //
    //      dtype        a.conj()                     np.conj(a)   (0-d in)
    //      bool         `a` ITSELF (is a -> True)    int8 scalar
    //      int32        `a` ITSELF                   int32 scalar
    //      float64      `a` ITSELF                   float64 scalar
    //      complex64    complex64 SCALAR             complex64 scalar
    //      complex128   complex128 SCALAR            complex128 scalar
    //
    //    Two rules fall out, and neither is "0-d collapses to a scalar":
    //
    //    1. For a NON-complex dtype the method is a total no-op that returns
    //       the receiver object itself -- not a copy, not even a view.
    //       `np.arange(3).conj()[0] = 99` mutates the original. Note this
    //       also explains the bool row: the method leaves bool alone while
    //       the ufunc promotes it to int8, so routing the method through the
    //       ufunc would be wrong for reasons beyond the return type.
    //    2. Only the complex path builds a new array, and only THERE does a
    //       0-d result collapse to a numpy scalar -- because only there does
    //       the method actually call the ufunc.
    //
    //    anionpy previously returned a fresh contiguous copy in every case,
    //    which was wrong twice over: it broke the complex 0-d return type
    //    (2 of 166 differential cases) and it silently dropped the aliasing
    //    on the real path, where a caller mutating the result expects the
    //    original to change. Returning `slf` restores both, and costs less.
    //
    //    `out` is POSITIONAL-ONLY, which is not a style choice: numpy's
    //    method rejects the keyword spelling outright (`a.conj(out=o)` ->
    //    TypeError "ndarray.conj() takes no keyword arguments") while
    //    accepting the positional one (`a.conj(o)`), so the `/` below is
    //    load-bearing.
    #[pyo3(signature = (*args, **kwargs))]
    fn conj(
        slf: &Bound<'_, PyArray>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        conjugate_dispatch(slf, args, kwargs, "conj")
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn conjugate(
        slf: &Bound<'_, PyArray>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        conjugate_dispatch(slf, args, kwargs, "conjugate")
    }

    // -- sort/search family (PATH-TO-100.md's sort/search block). Thin
    //    wrappers over `ionp_core::sort`, same kernels the top-level
    //    `anionpy.sort`/`anionpy.argmin`/`anionpy.nonzero`/`anionpy.searchsorted`
    //    functions in `reductions.rs` use -- see that file's own doc
    //    comments for the exact numpy-verified kwarg/error-message
    //    contracts (`kind=`/`stable=`/`descending=` mutual exclusivity,
    //    `order=` rejection, `side=` validation), duplicated here rather
    //    than shared since this is a `#[pymethods]` receiver (`&mut self`/
    //    `&self`) not a free function taking an explicit array argument.

    /// `ndarray.sort` -- TRUE in-place mutation, returns `None` (numpy:
    /// `arr.sort()` never returns the array itself, unlike `np.sort(arr)`).
    #[pyo3(signature = (axis=-1, kind=None, order=None, *, stable=None, descending=None))]
    fn sort(
        &mut self,
        axis: isize,
        kind: Option<&Bound<'_, PyAny>>,
        order: Option<&Bound<'_, PyAny>>,
        stable: Option<bool>,
        descending: Option<bool>,
    ) -> PyResult<()> {
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
        // Same str/bytes/bytearray shape as top-level `anionpy.sort`'s own
        // `kind=` (see `reductions.rs::sort`'s doc comment for the
        // measured message wording, identical here).
        let kind_str: Option<String> = match kind {
            None => None,
            Some(v) => Some(crate::extract_str_or_ascii_bytes(v, "sort kind")?),
        };
        let is_stable = match kind_str.as_deref() {
            Some(k) => sort_kind_to_stable(k)?,
            None => stable.unwrap_or(false),
        };
        let desc = descending.unwrap_or(false);
        let ndim = self.inner.ndim();
        let norm = normalize_axis(axis, ndim)?;
        ionp_core::sort::sort_in_place(&mut self.inner, norm, is_stable, desc).map_err(to_py_err)
    }

    /// `ndarray.argsort` -- NOT in-place and does NOT return `None`; it
    /// returns a fresh `int64` index array, the same contract as top-level
    /// `anionpy.argsort` (see `reductions.rs::argsort` for the measured
    /// kwarg/error surface, and `ionp-core/src/sort.rs` for why the
    /// permutation has to reproduce numpy's introsort exactly).
    ///
    /// Two divergences from the sibling `sort` method above, both measured
    /// on `ndarray.argsort` itself rather than assumed from it:
    ///   * `axis` is `Option<isize>`, because `arr.argsort(axis=None)` is
    ///     legal and flattens (`sort`'s in-place form has no such case).
    ///   * `stable=`/`descending=` are plain Python truthiness, not
    ///     `bool` extraction -- `a.argsort(stable=1.5)` succeeds in real
    ///     numpy. (The `sort` method above extracts `Option<bool>` and so
    ///     would reject that; left alone here as it is a separate item's
    ///     surface, but noted so the difference is not read as an
    ///     oversight in this one.)
    /// 0-d input is accepted and yields `array([0])`, as for the
    /// top-level function.
    #[pyo3(signature = (axis=-1, kind=None, order=None, *, stable=None, descending=None))]
    fn argsort(
        &self,
        py: Python<'_>,
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
        let kind_str: Option<String> = match kind {
            None => None,
            Some(v) => Some(crate::extract_str_or_ascii_bytes(v, "sort kind")?),
        };
        let is_stable = match kind_str.as_deref() {
            Some(k) => sort_kind_to_stable(k)?,
            None => match stable {
                None => false,
                Some(v) => v.is_truthy()?,
            },
        };
        let desc = match descending {
            None => false,
            Some(v) => v.is_truthy()?,
        };
        // 0-d -> (1,) before axis validation; see reductions.rs::argsort.
        let base = if self.inner.ndim() == 0 {
            self.inner.ravel_order("C").map_err(to_py_err)?
        } else {
            self.inner.clone()
        };
        let (target, ax) = match axis {
            None => (base.ravel_order("C").map_err(to_py_err)?, 0usize),
            Some(raw) => {
                let norm = normalize_axis(raw, base.ndim())?;
                (base, norm)
            }
        };
        let result = ionp_core::sort::argsort_axis(&target, ax, is_stable, desc).map_err(to_py_err)?;
        Ok(Py::new(py, PyArray { inner: result })?.into_any())
    }

    #[pyo3(signature = (axis=None, out=None, keepdims=false))]
    fn argmax(
        &self,
        py: Python<'_>,
        axis: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
        keepdims: bool,
    ) -> PyResult<Py<PyAny>> {
        do_method_argext(py, ExtremeOp::ArgMax, &self.inner, axis, out, keepdims)
    }

    #[pyo3(signature = (axis=None, out=None, keepdims=false))]
    fn argmin(
        &self,
        py: Python<'_>,
        axis: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
        keepdims: bool,
    ) -> PyResult<Py<PyAny>> {
        do_method_argext(py, ExtremeOp::ArgMin, &self.inner, axis, out, keepdims)
    }

    /// `ndarray.nonzero()` -- identical contract to top-level `anionpy.nonzero`
    /// (0-d input rejected with the same `ValueError`).
    fn nonzero(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cols = ionp_core::sort::nonzero(&self.inner).map_err(to_py_err)?;
        let mut items: Vec<Py<PyAny>> = Vec::with_capacity(cols.len());
        for c in cols {
            items.push(Py::new(py, PyArray { inner: c })?.into_any());
        }
        Ok(PyTuple::new(py, items)?.into_any().unbind())
    }

    /// `ndarray.searchsorted(v, side='left', sorter=None)` -- identical
    /// contract to top-level `anionpy.searchsorted(self, v, ...)`.
    #[pyo3(signature = (v, side=crate::OptionalArg::Omitted, sorter=None))]
    fn searchsorted(
        &self,
        py: Python<'_>,
        v: &Bound<'_, PyAny>,
        side: crate::OptionalArg<'_>,
        sorter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // PHANTOM FOUND + FIXED (2026-08-04): this was a bare
        // `extract_array(v)?`, so `self` and `v` reached the kernel at
        // unrelated dtypes and it raised `TypeError: searchsorted: value
        // dtype mismatch`. Sharing the top-level's rule rather than
        // re-deriving it here -- the two call sites had already drifted
        // once, which is exactly how this survived: fixing only
        // `reductions.rs::searchsorted` left 578/984 METHOD-form cases
        // still failing while the top-level read 984/984 clean.
        let (base, values) = crate::reductions::searchsorted_operands(&self.inner, v)?;
        // Same `str`/`bytes`/explicit-`None`-rejects shape as top-level
        // `anionpy.searchsorted`'s own `side=` -- see `reductions.rs::
        // searchsorted`'s doc comment for the measured contract, identical
        // here.
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
            Some(s) => Some(extract_array(s)?),
        };
        let result =
            ionp_core::sort::searchsorted(&base, &values, right, sorter_arr.as_ref()).map_err(to_py_err)?;
        if result.ndim() == 0 {
            return crate::numpy_scalar_from_0d(py, &result);
        }
        Ok(Py::new(py, PyArray { inner: result })?.into_any())
    }

    // -- diagonal/trace/repeat: thin wraps over `ionp_core::manip`'s
    //    already-differentially-proven kernels (top-level `anionpy.diagonal`/
    //    `anionpy.trace`/`anionpy.repeat` in `manip.rs`, a file this task does
    //    NOT own, wrap the exact same core functions and all three pass
    //    their full differential suites -- see this task's report). Only
    //    `trace` needs new logic here at all: numpy's `ndarray.trace` carries
    //    a real `out=` parameter that the top-level `anionpy.trace` wrapper
    //    explicitly rejects (`"anionpy.trace: out= is not supported"`), so this
    //    method adds `write_into_out` support rather than mirroring that
    //    rejection.

    /// `ndarray.diagonal(offset=0, axis1=0, axis2=1)` -- numpy's own
    /// signature carries no `out=` (verified via `inspect.signature` against
    /// real numpy 2.5.1: `(self, /, offset=0, axis1=0, axis2=1)`).
    #[pyo3(signature = (offset=0, axis1=0, axis2=1))]
    fn diagonal(&self, offset: isize, axis1: isize, axis2: isize) -> PyResult<PyArray> {
        let inner = manip::diagonal(&self.inner, offset, axis1, axis2).map_err(to_py_err)?;
        Ok(PyArray { inner })
    }

    /// `ndarray.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)`.
    #[pyo3(signature = (offset=0, axis1=0, axis2=1, dtype=None, out=None))]
    fn trace(
        &self,
        py: Python<'_>,
        offset: isize,
        axis1: isize,
        axis2: isize,
        dtype: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let dt: Option<DType> = match dtype {
            None => None,
            Some(d) if d.is_none() => None,
            Some(d) => Some(crate::dtype_from_pyobj_no_su(d)?),
        };
        let result = manip::trace(&self.inner, offset, axis1, axis2, dt).map_err(to_py_err)?;
        if let Some(out_obj) = out {
            return write_into_out(out_obj, &result, None);
        }
        if result.ndim() == 0 {
            return crate::numpy_scalar_from_0d(py, &result);
        }
        Ok(Py::new(py, PyArray { inner: result })?.into_any())
    }

    /// `ndarray.repeat(repeats, axis=None)` -- `repeats` is either a single
    /// int (broadcast to every element along `axis`, or every element of
    /// the flattened array when `axis=None`) or a per-element sequence of
    /// ints, exactly mirroring top-level `anionpy.repeat`'s own
    /// `usize_list_from_pyobj` resolution (duplicated here, not imported --
    /// `manip.rs` is a file this task does not own, see module doc).
    ///
    /// Axis bound-checking is done HERE, not delegated to
    /// `manip::normalize_axis`, for one narrow, measured reason: real
    /// numpy's `AxisError` message for an out-of-range axis on a 0-d
    /// array reports `"...dimension 1"` for `.repeat()`/`.cumsum()` but
    /// `"...dimension 0"` for `.sum()`/`.trace()` -- verified directly
    /// against numpy 2.5.1, an actual inconsistency in numpy's own
    /// messages, not a single fixed contract. `manip::normalize_axis`
    /// (unowned, `manip.rs`) deliberately implements the `.sum()`/
    /// `.trace()` convention (documented at its own definition), which is
    /// wrong for THIS caller -- so `repeat` pre-validates against the
    /// `.repeat()`-family convention itself and only calls
    /// `manip::normalize_axis` once the axis is already known in-range
    /// (where both conventions agree and it can never raise).
    #[pyo3(signature = (repeats, axis=None))]
    fn repeat(&self, repeats: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<PyArray> {
        let ndim = self.inner.ndim();
        if let Some(raw_axis) = axis {
            let reported_ndim = ndim.max(1);
            let n = reported_ndim as isize;
            let norm = if raw_axis < 0 { raw_axis + n } else { raw_axis };
            if norm < 0 || norm >= n {
                return Err(axis_error(
                    raw_axis,
                    Some(reported_ndim),
                    &format!("axis {raw_axis} is out of bounds for array of dimension {reported_ndim}"),
                ));
            }
        }
        // 0-d operand: numpy promotes to shape (1,) BEFORE validating the
        // axis (see the free-function `repeat` in manip.rs for the measured
        // rule). Only the ERROR MESSAGE half of that was implemented here
        // (`reported_ndim` above), so a 0-d operand produced numpy's exact
        // wording for an out-of-range axis and then still raised on
        // `axis=0`, which numpy accepts. That is the same failure shape as
        // `do_cumulative` earlier today: half a mechanism, and the half you
        // can see in an error string rather than the half you can see in a
        // result. This file duplicating the free function is what let one
        // copy be fixed while the other stayed wrong -- again.
        let base = if ndim == 0 {
            self.inner.ravel_order("C").map_err(to_py_err)?
        } else {
            self.inner.clone()
        };
        let eff_ndim = base.ndim();
        let target_len = match axis {
            Some(ax) => {
                let n = manip::normalize_axis(ax, eff_ndim).map_err(to_py_err)?;
                base.shape()[n]
            }
            None => base.size(),
        };
        let raw = usize_list_from_pyobj(repeats)?;
        let resolved: Vec<usize> = if raw.len() == 1 { vec![raw[0]; target_len] } else { raw };
        let ax = match axis {
            Some(a) => Some(manip::normalize_axis(a, eff_ndim).map_err(to_py_err)?),
            None => None,
        };
        let inner = manip::repeat(&base, &resolved, ax).map_err(to_py_err)?;
        Ok(PyArray { inner })
    }

    // -- cumsum/cumprod: shared `ionp_core::ufunc::accumulate_axis` kernel
    //    (the same one top-level `anionpy.cumsum`/`anionpy.cumprod` in
    //    `reductions.rs` use -- both pass their full differential suites,
    //    see this task's report), `axis=None` flattens first like every
    //    other axis-reduction method in this file.

    #[pyo3(signature = (axis=None, dtype=None, out=None))]
    fn cumsum(
        &self,
        py: Python<'_>,
        axis: Option<&Bound<'_, PyAny>>,
        dtype: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        do_cumulative(py, BinaryOp::Add, &self.inner, axis, dtype, out)
    }

    #[pyo3(signature = (axis=None, dtype=None, out=None))]
    fn cumprod(
        &self,
        py: Python<'_>,
        axis: Option<&Bound<'_, PyAny>>,
        dtype: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        do_cumulative(py, BinaryOp::Multiply, &self.inner, axis, dtype, out)
    }

    /// `ndarray.fill(value)` -- TRUE in-place mutation (numpy: returns
    /// `None`, existing Python references to this same object observe the
    /// new values). `value` is marshaled into this array's OWN dtype via
    /// `weak_scalar_buffer` with `target` set to `self.inner.dtype()`
    /// directly (NOT NEP-50-promoted against it) -- this is the same
    /// scalar->fixed-dtype marshaling `np.full(shape, value, dtype=target)`
    /// needs, which is exactly numpy's own `fill` cast contract: an
    /// out-of-range int raises `OverflowError` (verified:
    /// `np.zeros(3,dtype=np.uint8).fill(300)` ->
    /// `OverflowError: Python integer 300 out of bounds for uint8`), a
    /// float truncates toward zero into an integer target, matching
    /// `weak_scalar_buffer`'s existing narrowing rules exactly (already
    /// proven by every `anionpy.full`/`anionpy.array` differential case that
    /// exercises them).
    #[pyo3(signature = (value))]
    fn fill(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let dtype = self.inner.dtype();
        let kind = classify_scalar(value)
            .ok_or_else(|| PyTypeError::new_err("fill() argument must be a scalar"))?;
        // numpy's real `.fill()` marshals its `value` argument through a C
        // integer conversion FIRST, before any target-dtype bounds check --
        // so a Python int outside that conversion's own range raises
        // `OverflowError('Python int too large to convert to C long')`
        // regardless of target dtype, and only a value that SURVIVES that
        // conversion gets checked against the target dtype's own narrower
        // range (yielding the ordinary "Python integer N out of bounds for
        // <dtype>" message instead). This applies to EVERY integer target,
        // not just the 64-bit ones -- verified directly against real numpy
        // 2.5.1 by sweeping int8/16/32/64 and uint8/16/32/64 across the
        // i64/u64 boundary values (2**63, 2**63-1, -2**63-1, -2**63,
        // 2**64-1, 2**64, plus in-band 300/-300).
        //
        // The C-integer conversion is NOT uniformly "i64" though: it is
        // `PyLong_AsLong` (signed C long, i.e. i64::MIN..=i64::MAX) for
        // int8/int16/int32/int64/uint8/uint16, but `uint32` and `uint64`
        // SPECIFICALLY go through the wider unsigned conversion
        // (i64::MIN..=u64::MAX -- yes, the lower bound is still the signed
        // i64::MIN, not 0; a negative value down to -2**63 still gets the
        // dtype-bounds message, not the C-long one). Measured directly:
        // `np.zeros(2,dtype='uint16').fill(2**63)` raises the C-long
        // message, but the byte-identical call on `uint32` raises
        // `OverflowError('Python integer 9223372036854775808 out of bounds
        // for uint32')` instead -- uint32 is NOT treated like uint16 despite
        // both being narrower than 64 bits. `uint32`/`uint64` accept the
        // full `[0, 2**64-1]` (or fail with the dtype message inside that
        // band) before ever hitting the C-long path; every other integer
        // target dtype uses the plain signed i64 range.
        // GATE: this conversion only exists on the way into an INTEGER
        // target -- a float/complex target never goes through a C-integer
        // marshal at all (`np.zeros(2,dtype='float16').fill(2**63)` is
        // fine, converts straight to `inf`). A prior version of this
        // pre-check gated on the VALUE's kind only (`ScalarKind::Int`),
        // which was too wide: it fired for every target dtype whenever the
        // *value* was a Python int, silently breaking `float16`/`float32`/
        // `float64`/`complex64`/`complex128` targets with a spurious
        // `OverflowError` on the exact values this pre-check exists to
        // handle for integer targets. Found via the coordinator's
        // out-of-corpus probe; both the value AND the target must be
        // integer-kind for this branch to apply.
        if matches!(kind, ScalarKind::Int) && dtype.is_integer() {
            if let Ok(v) = value.extract::<i128>() {
                let extended_range = matches!(dtype, DType::U32 | DType::U64);
                let hi = if extended_range { u64::MAX as i128 } else { i64::MAX as i128 };
                let in_range = (i64::MIN as i128..=hi).contains(&v);
                if !in_range {
                    return Err(PyOverflowError::new_err(
                        "Python int too large to convert to C long",
                    ));
                }
            }
        }
        // A non-finite float `value` into an integer TARGET is a second,
        // previously-uncaught divergence: real numpy 2.5.1 raises rather
        // than silently narrowing --
        //   inf/-inf -> OverflowError('cannot convert float infinity to integer')
        //   nan      -> ValueError('cannot convert float NaN to integer')
        // (both verified directly, all 8 integer dtypes). Without this,
        // anionpy's downstream narrowing (`f64 as <int>`, via
        // `weak_scalar_buffer`/`int_buffer_from_f64`) silently SATURATES
        // inf/-inf to the dtype's max/min and writes 0 for nan -- a wrong
        // buffer with no exception at all, not merely a wrong message.
        if matches!(kind, ScalarKind::Float) && dtype.is_integer() {
            if let Ok(v) = value.extract::<f64>() {
                if v.is_infinite() {
                    return Err(PyOverflowError::new_err(
                        "cannot convert float infinity to integer",
                    ));
                }
                if v.is_nan() {
                    return Err(PyValueError::new_err(
                        "cannot convert float NaN to integer",
                    ));
                }
            }
        }
        let one = weak_scalar_buffer(value, kind, dtype)?;
        let filled = creation::repeat_scalar_buffer(&one, self.inner.size());
        let shape = self.inner.shape().to_vec();
        self.inner = NdArray::from_buffer(filled, shape, Order::C).map_err(to_py_err)?;
        Ok(())
    }

    /// `ndarray.tobytes(order='C')` -- raw little-endian element bytes
    /// (this machine, like every platform real numpy 2.5.1 ships wheels
    /// for, is little-endian; anionpy's `Buffer` vectors already store native
    /// Rust primitives in that same byte layout, so this is a direct
    /// per-element `to_le_bytes()` walk, not a reinterpret-cast). `order`
    /// is passed straight to `NdArray::ravel_order` -- the same primitive
    /// `ravel`/`flatten` above use -- which flattens in the requested
    /// order before the byte walk. `'A'` is deliberately NOT exercised by
    /// this file's differential cases (see `_ravel_flatten_forms`'s doc
    /// comment above: `ravel_order`'s `'A'` branch has a known, reported,
    /// NOT-yet-fixed bug for Fortran-order/transposed sources -- out of
    /// this task's file-ownership fence to fix in `array.rs`). `'C'`/`'F'`
    /// are unconditional (not data-dependent) and fully covered.
    #[pyo3(signature = (order="C"))]
    fn tobytes(&self, py: Python<'_>, order: &str) -> PyResult<Py<PyAny>> {
        let flat = self.inner.ravel_order(order).map_err(to_py_err)?;
        let bytes = buffer_to_bytes(flat.buffer());
        Ok(PyBytes::new(py, &bytes).into_any().unbind())
    }

    /// `ndarray.clip(min=None, max=None, out=None)` -- composed as
    /// `minimum(maximum(a, min), max)` via the shared `binary_op` kernel
    /// (the same kernel `anionpy.maximum`/`anionpy.minimum` use, both already
    /// differentially proven), which gets broadcasting, dtype promotion
    /// (including NEP-50 weak-scalar promotion for a bare Python
    /// int/float `min`/`max` via `scalar_against`), and NaN propagation
    /// for free rather than reimplementing any of it here. Verified
    /// against real numpy 2.5.1 that this compose order matches even the
    /// `min > max` case (`a.clip(3, 0)` clips everything to `0`, i.e.
    /// `maximum` applied BEFORE `minimum`, not the other way around) and
    /// that omitting both `min` and `max` is not an error (`a.clip()`
    /// returns an equal-valued copy, `is` `a` is `False`) -- both probed
    /// directly, not assumed. numpy's real signature also carries a
    /// `**kwargs` catch-all (forwarded to the underlying `um.clip` ufunc
    /// call, e.g. `where=`/`casting=`) that this method does not accept;
    /// undeclared/untested here, see this task's report.
    #[pyo3(signature = (min=None, max=None, out=None))]
    fn clip(
        &self,
        py: Python<'_>,
        min: Option<&Bound<'_, PyAny>>,
        max: Option<&Bound<'_, PyAny>>,
        out: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // numpy's real `.clip()` is NOT one fixed algorithm across its
        // three `(min, max)` presence combinations -- verified directly
        // against real numpy 2.5.1's actual signed-zero handling, which
        // differs case by case:
        //   - min only (max=None): byte-identical to `np.maximum(a, min)`
        //     -- ties normalize a signed zero to `+0.0`
        //     (`np.array([-0.0]).clip(0, None)` -> `[0.]`).
        //   - max only (min=None): byte-identical to `np.minimum(a, max)`
        //     -- ties PRESERVE the original sign
        //     (`np.array([-0.0]).clip(None, 0)` -> `[-0.]`, matching
        //     `np.minimum(-0.0, 0.0)` -> `-0.0`).
        //   - both given: neither of the above -- a genuinely different
        //     ternary-select code path that also preserves the original
        //     element's sign on a tie
        //     (`np.array([-0.0]).clip(0, 5)` -> `[-0.]`, NOT the `[0.]` a
        //     `maximum(a,min)` then `minimum(...,max)` composition would
        //     give since `maximum` alone already normalizes the tie away
        //     before `minimum` ever runs). Modeled here as a
        //     strict-inequality `where_select` (replace only elements
        //     strictly outside a bound, otherwise keep the original
        //     element's own bits untouched), which reproduces this
        //     exactly for every corpus case tested (float32/float64/
        //     complex64/complex128, NaN/inf/signed-zero included).
        let mut result = self.inner.clone();
        // Operands that participate in the `order='K'` output-layout vote
        // (see `apply_ufunc_order` in ionp-py/src/lib.rs for the general
        // mechanism and its derivation): `self` always votes, plus whichever
        // of `min`/`max` turned out to be array-valued (not a bare Python
        // scalar) -- confirmed empirically that `clip` does full
        // multi-operand K-order voting, not just self-alone, e.g.
        // `a.T.clip(b.T, None)` takes `b.T`'s layout into account too.
        let mut order_operands: Vec<NdArray> = vec![self.inner.clone()];
        let mn_present = min.is_some_and(|m| !m.is_none());
        let mx_present = max.is_some_and(|m| !m.is_none());
        if mn_present && mx_present {
            // BUG FOUND + FIXED 2026-08-03 (Monday). The strict-inequality
            // select above reproduces numpy's signed-zero tie behaviour, but
            // it silently dropped NaN BOUNDS: `a < NaN` is false everywhere,
            // so nothing was replaced and `np.clip(a, nan, 1.0)` -- which real
            // numpy 2.5.1 returns as all-NaN -- came back as the unclipped
            // values. Measured across float16/32/64 x {None,nan,0.0,-1.0}^2
            // and array-valued bounds with NaN inside (/tmp/mg_clipnan.py):
            // a NaN BOUND propagates elementwise from either side, while a
            // NaN in the OPERAND is preserved rather than overwritten. That
            // is exactly maximum/minimum NaN semantics, which is why the
            // min-only and max-only branches below were already correct and
            // only this ternary branch was wrong.
            //
            // `x != x` is used as isnan rather than a dedicated unary: it is
            // dtype-agnostic (constant-false for integer bounds, so integer
            // clipping is untouched) and it matches numpy's COMPLEX isnan,
            // which is true when EITHER component is NaN -- complex `!=`
            // already has that component-wise-or semantics. Note the asymmetry
            // is load-bearing: the NaN test is on the BOUND only, never on
            // `result`, because clipping must not overwrite an operand NaN.
            // SECOND defect in this same branch, found and fixed 2026-08-03
            // while verifying the NaN fix above (/tmp/mg_cliptie.py). The
            // strict-select tie rule documented at the top of this method is
            // real, but it is the FLOAT rule only. Real numpy 2.5.1 uses the
            // opposite tie rule for COMPLEX operands with both bounds given:
            // the bound wins, so the result carries the BOUND's signed zero,
            // not the operand's. Measured over a x mn x mx in {+0.0,-0.0}^3
            // with the other bound moved away from the tie to isolate each
            // stage -- every one of the 8 complex rows takes the bound's
            // sign, and every float row keeps the operand's:
            //   complex: clip(-0.0, -0.0, +0.0) -> +0j   (bound wins)
            //   float:   clip(-0.0, -0.0, +0.0) -> -0.0  (operand wins)
            //
            // THIRD refinement, same session: the float rule is not
            // "operand wins" either. It depends on whether the bound is a
            // WEAK operand -- a Python scalar or a 0-d array -- or a real
            // array of ndim >= 1. Same a, same mn, three spellings, two
            // different answers (verified against numpy 2.5.1):
            //   np.clip(f64([-0.0]), 0.0,            1.0) -> -0.0
            //   np.clip(f64([-0.0]), np.float64(0.0),1.0) -> -0.0   (0-d)
            //   np.clip(f64([-0.0]), np.array([0.0]),1.0) -> +0.0   (1-d)
            // That is numpy's weak-promotion split showing through: the
            // scalar/0-d spellings take a different inner loop than the
            // array-array one. Complex is bound-wins in ALL THREE
            // spellings, so the complex rule is unconditional and the
            // float rule is conditioned on weakness.
            //
            // The weakness is a property of the CALL, not of each bound
            // separately. That distinction was not guessed -- a per-bound
            // model was implemented first and measured, and it got 50 of
            // 720 tie-spelling cases wrong. The counterexample that killed
            // it: `np.clip(f64([0.0]), -0.0, np.array([1.0]*4))` is -0.0,
            // i.e. the MIN stage used the bound-wins rule even though the
            // min bound is a plain Python float. A single array anywhere
            // in the call promotes the whole operation to the array-array
            // loop, and both stages follow it. Hence one `strict` flag
            // computed from both bounds before either stage runs.
            //
            // None of this changes any VALUE: at any tie on a non-zero
            // number the bound and the operand are bit-identical, so
            // replacing or not is a no-op. It is only ever the sign of a
            // zero. It is still worth getting right -- a signed zero is a
            // real bit pattern that propagates (1/-0.0 is -inf), and the
            // whole point of this project is that "close enough" is not
            // the standard.
            let is_complex = result.dtype().is_complex();
            let (mn_arr, mn_weak) = clip_bound(&result, min.unwrap())?;
            let (mx_arr, mx_weak) = clip_bound(&result, max.unwrap())?;
            // float16 is strict UNCONDITIONALLY -- the array-array loop does
            // not flip it the way f32/f64 do. That is the same first-operand-
            // wins tie quirk float16 `maximum`/`minimum` already have (see
            // `extreme_f16_tie_first` in ufunc.rs, and the two f16 branches
            // further down this method), surfacing here too. Found by
            // measurement, not by analogy: after the call-level `strict` fix
            // below, the 30 cases still diverging out of 720 were ALL f16.
            let strict = if is_complex {
                false
            } else if result.dtype() == DType::F16 {
                true
            } else {
                mn_weak && mx_weak
            };
            let (below, above) = if strict {
                (BinaryOp::Less, BinaryOp::Greater)
            } else {
                (BinaryOp::LessEqual, BinaryOp::GreaterEqual)
            };
            order_operands.push(mn_arr.clone());
            let mn_nan = binary_op(BinaryOp::NotEqual, &mn_arr, &mn_arr).map_err(to_py_err)?;
            let mask = binary_op(below, &result, &mn_arr).map_err(to_py_err)?;
            let mask = binary_op(BinaryOp::LogicalOr, &mask, &mn_nan).map_err(to_py_err)?;
            result = ionp_core::sort::where_select(&mask, &mn_arr, &result).map_err(to_py_err)?;
            order_operands.push(mx_arr.clone());
            let mx_nan = binary_op(BinaryOp::NotEqual, &mx_arr, &mx_arr).map_err(to_py_err)?;
            let mask = binary_op(above, &result, &mx_arr).map_err(to_py_err)?;
            let mask = binary_op(BinaryOp::LogicalOr, &mask, &mx_nan).map_err(to_py_err)?;
            result = ionp_core::sort::where_select(&mask, &mx_arr, &result).map_err(to_py_err)?;
        } else if mn_present {
            let mn_arr = scalar_or_array_against(&result, min.unwrap())?;
            order_operands.push(mn_arr.clone());
            // float16 `maximum` has a different (first-operand-wins) tie
            // rule than f32/f64/complex in real numpy -- see
            // `extreme_f16_tie_first`'s doc comment in ufunc.rs.
            //
            // BUG FOUND + FIXED 2026-08-04 (Monday): this branch was gated on
            // `result.dtype() == F16` ALONE, i.e. on the OPERAND's dtype, and
            // `extreme_f16_tie_first` computes at f16 and returns f16. So a
            // WIDER bound was silently narrowed to the operand instead of
            // promoting the output, which for a complex bound DISCARDED THE
            // IMAGINARY PART -- a wrong value, not merely a wrong dtype.
            // Measured 2026-08-04, numpy 2.5.1:
            //   np.clip(f16([1,0,2]), 1+2j, None)
            //       np  : complex64 [1+2j, 1+2j, 2+0j]
            //       anionpy: float16   [1.,   1.,   2.  ]
            //   np.clip(f16([1,0,2]), np.float32(1), None) -> float32 (anionpy f16)
            //   np.clip(f16([1,0,2]), np.array([1.]*3, f32), None) -> float32
            // 12 of a 360-cell (12 operand dtypes x 10 bound spellings x
            // {min-only, max-only, both}) grid diverged, ALL of them f16
            // operand with a wider bound, and ALL of them in these two
            // single-bound branches -- the both-bounds branch above routes
            // through `where_select`, which promotes correctly already.
            // `clip` was declared "exact" while carrying this.
            //
            // Gating on the PROMOTED dtype instead: the f16 tie quirk is a
            // property of the f16 LOOP, so it applies exactly when the f16
            // loop is the one numpy would pick, which is when the promotion
            // of both operands is still f16. A narrower bound (int8, bool,
            // or a weak Python scalar, all of which promote back to f16)
            // therefore still takes it, as before.
            result = if promote_dtype(result.dtype(), mn_arr.dtype()) == DType::F16 {
                ionp_core::ufunc::extreme_f16_tie_first(BinaryOp::Maximum, &result, &mn_arr).map_err(to_py_err)?
            } else {
                binary_op(BinaryOp::Maximum, &result, &mn_arr).map_err(to_py_err)?
            };
        } else if mx_present {
            let mx_arr = scalar_or_array_against(&result, max.unwrap())?;
            order_operands.push(mx_arr.clone());
            // Same fix as the min-only branch directly above -- see its
            // comment for the measured table. `max`-only is the
            // `np.clip(f16_arr, None, wider)` half of those 12 rows.
            result = if promote_dtype(result.dtype(), mx_arr.dtype()) == DType::F16 {
                ionp_core::ufunc::extreme_f16_tie_first(BinaryOp::Minimum, &result, &mx_arr).map_err(to_py_err)?
            } else {
                binary_op(BinaryOp::Minimum, &result, &mx_arr).map_err(to_py_err)?
            };
        }
        let operand_refs: Vec<&NdArray> = order_operands.iter().collect();
        result = apply_ufunc_order(result, 'K', &operand_refs);
        if let Some(out_obj) = out {
            return write_into_out(out_obj, &result, None);
        }
        if result.ndim() == 0 {
            return crate::numpy_scalar_from_0d(py, &result);
        }
        Ok(Py::new(py, PyArray { inner: result })?.into_any())
    }
}

/// Same accept-ionp-array/list/scalar resolution as `manip.rs`'s own
/// private `usize_list_from_pyobj` (deliberately duplicated -- see this
/// file's own module doc on why `ndarray_attrs.rs` prefers duplication
/// over a cross-file import for a private helper in a sibling module this
/// task does not own).
fn usize_list_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        let ints = pyref.inner.cast_to(DType::I64);
        let vals = match ints.buffer() {
            Buffer::I64(v) => v.clone(),
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

/// Resolves a `clip(min=..., max=...)` bound: either a bare Python scalar
/// (NEP-50 weak promotion against `base`, via the same `scalar_against`
/// machinery `lib.rs`'s binary-op dispatch uses) or a full array-like
/// (broadcast-compatible, ingested via `extract_array`).
/// A `clip` bound, plus whether it is a WEAK operand in numpy's sense: a
/// Python scalar or a 0-d array, as opposed to an array of ndim >= 1.
/// `clip`'s signed-zero tie rule differs between the two (see the long
/// comment in `clip`), and `scalar_against` is already exactly the
/// "did this participate as a scalar" predicate the rest of this file
/// uses, so the split costs nothing extra to compute.
/// BUG FOUND + FIXED 2026-08-04 (Monday): both helpers below called
/// `scalar_against` FIRST, with no strong-operand gate in front of it.
/// `classify_scalar` recognizes bare Python `bool`/`int`/`float`/`complex`
/// by `is_instance_of`, and several numpy scalar types are genuine
/// SUBCLASSES of those (`np.float64` of `float`, `np.int64`/`np.bool_` of
/// `int`/`bool`) -- `classify_scalar`'s own doc comment says as much and
/// says the `hasattr(obj, "dtype")` check "must run first". `lib.rs`'s
/// `extract_binary_pair_tiered` does exactly that. These two did not, so a
/// numpy scalar was treated as WEAK and narrowed to the operand's dtype
/// instead of promoting it. Measured 2026-08-04, numpy 2.5.1:
///     np.clip(f16([1,0,2]),       np.float64(1), None) -> float64
///     np.clip(f32([1,0,2]),       np.float64(1), None) -> float64
///     np.clip(complex64([1,0,2]), np.float64(1), None) -> complex128
///     anionpy, all three                                  -> operand's own dtype
/// The narrower numpy-scalar spellings (`np.float32`, `np.complex64`) were
/// never affected: they are not Python-scalar subclasses, so they already
/// fell through to `extract_array`. That asymmetry is why an earlier,
/// narrower probe passed -- it had no `np.float64` bound in it.
///
/// The `weak` flag `clip_bound` returns is deliberately NOT changed by
/// this: a numpy scalar now takes the `extract_array` path, which reports
/// `weak = arr.ndim() == 0`, i.e. still weak. That matches the measured
/// signed-zero tie rule recorded in `clip` (`np.clip(f64([-0.0]),
/// np.float64(0.0), 1.0)` is `-0.0`, same as the bare Python spelling and
/// unlike the 1-d array spelling).
fn is_strong_operand(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(obj.extract::<PyRef<'_, PyArray>>().is_ok() || obj.hasattr("dtype")?)
}

fn clip_bound(base: &NdArray, obj: &Bound<'_, PyAny>) -> PyResult<(NdArray, bool)> {
    if !is_strong_operand(obj)? {
        if let Some(w) = scalar_against(base, obj, false, false, false)? {
            return Ok((w, true));
        }
    }
    let arr = extract_array(obj)?;
    let weak = arr.ndim() == 0;
    Ok((arr, weak))
}

fn scalar_or_array_against(base: &NdArray, obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if !is_strong_operand(obj)? {
        if let Some(w) = scalar_against(base, obj, false, false, false)? {
            return Ok(w);
        }
    }
    extract_array(obj)
}

/// Per-element native-endian (== little-endian on every platform this
/// project ships for, see the `tobytes` method doc) byte serialization,
/// one arm per `Buffer` variant.
fn buffer_to_bytes(buf: &Buffer) -> Vec<u8> {
    match buf {
        Buffer::Bool(v) => v.iter().map(|&b| b as u8).collect(),
        Buffer::I8(v) => v.iter().map(|&x| x as u8).collect(),
        Buffer::I16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::U8(v) => v.clone(),
        Buffer::U16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::U64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::F16(v) => v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
        Buffer::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Buffer::C64(v) => v
            .iter()
            .flat_map(|c| {
                let mut b = c.re.to_le_bytes().to_vec();
                b.extend_from_slice(&c.im.to_le_bytes());
                b
            })
            .collect(),
        Buffer::C128(v) => v
            .iter()
            .flat_map(|c| {
                let mut b = c.re.to_le_bytes().to_vec();
                b.extend_from_slice(&c.im.to_le_bytes());
                b
            })
            .collect(),
        // `S`/`U` are already stored as fixed-width, zero-padded
        // per-element buffers (see `ionp-core/src/buffer.rs`'s `Buffer::S`/
        // `Buffer::U` doc) -- `S` is already raw bytes (1 byte/char, numpy's
        // own on-disk layout), `U` is UCS-4 (4 bytes/char, little-endian on
        // every platform this project ships for, same as every other
        // multi-byte arm above), so this is a direct flatten with no
        // NUL-stripping (`tobytes()` includes the padding, unlike
        // `.item()`/`.tolist()` -- verified against real numpy 2.5.1:
        // `np.array([b'ab'], dtype='S5').tobytes() == b'ab\x00\x00\x00'`).
        Buffer::S(_, v) => v.iter().flat_map(|e| e.iter().copied()).collect(),
        Buffer::U(_, v) => v.iter().flat_map(|e| e.iter().flat_map(|c| c.to_le_bytes())).collect(),
    }
}

/// Shared `cumsum`/`cumprod` driver -- mirrors `reductions.rs`'s own
/// private `do_accumulate_axis` exactly (dtype-override-then-cast,
/// `axis=None` flattens via `ravel_order("C")` first, same
/// `axis_error`-raising bounds check), duplicated rather than imported
/// since `reductions.rs` is a file this task does not own and the
/// function is private there.
fn do_cumulative(
    py: Python<'_>,
    op: BinaryOp,
    arr: &NdArray,
    axis: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let dtype_override: Option<DType> = match dtype {
        None => None,
        Some(d) if d.is_none() => None,
        Some(d) => Some(crate::dtype_from_pyobj_no_su(d)?),
    };
    let arr2 = match dtype_override {
        Some(dt) => arr.cast_to(dt),
        None => arr.clone(),
    };
    let ndim = arr2.ndim();
    let (target, ax): (NdArray, usize) = match axis {
        None => (arr2.ravel_order("C").map_err(to_py_err)?, 0),
        Some(a2) if a2.is_none() => (arr2.ravel_order("C").map_err(to_py_err)?, 0),
        Some(a2) => {
            // Was a bare `a2.extract()?`, i.e. PyO3's OWN message, which
            // matches numpy for nothing at all. Same delegation and same
            // measurements as `do_method_argext` above -- `.cumsum(axis=X)`
            // shares numpy's single-axis converter with `argmin`/`argmax`.
            let raw: isize = crate::reductions::single_axis_from_pyobj(a2)?;
            // `.cumsum()`/`.cumprod()` report a 0-d array's out-of-range
            // AxisError as `"...dimension 1"`, not `"...dimension 0"` --
            // verified against real numpy 2.5.1, the same
            // `.repeat()`-family convention documented at length on
            // `repeat`'s own method above (numpy is not internally
            // consistent about this across its own methods).
            let reported_ndim = ndim.max(1);
            let n = reported_ndim as isize;
            let norm = if raw < 0 { raw + n } else { raw };
            if norm < 0 || norm >= n {
                return Err(axis_error(
                    raw,
                    Some(reported_ndim),
                    &format!("axis {raw} is out of bounds for array of dimension {reported_ndim}"),
                ));
            }
            // BUG FOUND + FIXED 2026-08-03 (Monday). `reported_ndim` above
            // already encodes the fact that numpy promotes a 0-d operand to
            // shape (1,) BEFORE validating the axis -- but this driver then
            // accumulated over the UNPROMOTED 0-d array, so the message said
            // "dimension 1" while the result came back shape () instead of
            // numpy's (1,). Half the mechanism was implemented: the half you
            // can see in an error string, not the half you can see in a
            // result.
            //
            // `ndarray.cumsum`/`.cumprod` were DECLARED "exact" while this
            // was live (re-declared 2026-08-02), because the cases covering
            // them never crossed a 0-d operand with an explicit axis. The
            // top-level `anionpy.cumsum` was fixed earlier today; this file
            // keeps its own copy of the driver (see the doc comment above on
            // the file-ownership fence), and the copy did not receive the
            // fix. That is the standing cost of duplication: a fix applied
            // to one copy silently leaves the other wrong.
            if ndim == 0 {
                (arr2.ravel_order("C").map_err(to_py_err)?, 0)
            } else {
                (arr2, norm as usize)
            }
        }
    };
    let result = accumulate_axis(op, &target, ax).map_err(to_py_err)?;
    if let Some(out_obj) = out {
        return write_into_out(out_obj, &result, None);
    }
    Ok(Py::new(py, PyArray { inner: result })?.into_any())
}

/// Same accepted spellings/error text as `reductions.rs`'s own
/// `kind_to_stable` (deliberately duplicated -- see this file's own module
/// doc on why `ndarray_attrs.rs` prefers duplication over a cross-file
/// import for this kind of small shared helper).
fn sort_kind_to_stable(kind: &str) -> PyResult<bool> {
    match kind.to_ascii_lowercase().as_str() {
        "quicksort" | "quick" | "heapsort" | "heap" => Ok(false),
        "mergesort" | "merge" | "stable" => Ok(true),
        _ => Err(PyValueError::new_err(format!(
            "sort kind must be one of 'quick', 'heap', or 'stable' (got '{kind}')"
        ))),
    }
}

/// `ndarray.argmax`/`.argmin` driver -- mirrors `reductions.rs`'s own
/// `do_argext` exactly (including the `axis=None` flatten-first convention,
/// the `AxisError` fix for out-of-range explicit axis, and the
/// `keepdims=True` + `axis=None` original-ndim reshape), duplicated per
/// this file's established file-ownership-fence precedent (see module doc).
fn do_method_argext(
    py: Python<'_>,
    op: ExtremeOp,
    a: &NdArray,
    axis: Option<&Bound<'_, PyAny>>,
    out: Option<&Bound<'_, PyAny>>,
    keepdims: bool,
) -> PyResult<Py<PyAny>> {
    let ndim = a.ndim();
    let flattened = matches!(axis, None) || matches!(axis, Some(ax) if ax.is_none());
    let (target, axes): (NdArray, Vec<usize>) = match axis {
        None => (a.ravel_order("C").map_err(to_py_err)?, vec![0]),
        Some(ax) if ax.is_none() => (a.ravel_order("C").map_err(to_py_err)?, vec![0]),
        Some(ax) => {
            // BUG FOUND + FIXED (2026-08-01, argmax/argmin/nonzero task):
            // this used to raise a fixed, generic `"an integer is required
            // (axis must be an int or None)"` message regardless of the
            // actual offending type. Real numpy names the ACTUAL type
            // (`'tuple' object cannot be interpreted as an integer`,
            // verified against real numpy 2.5.1 for `arr.argmax(axis=(0,
            // 1))`), same fix and same reasoning as `reductions.rs`'s
            // `single_axis_from_pyobj` (this file's established
            // duplication precedent -- see module doc).
            // BUG FOUND + FIXED (2026-08-04, axis-form sweep): the local
            // copy above was a THIRD copy of the single-axis converter and
            // had drifted from `reductions.rs`'s `single_axis_from_pyobj`
            // in three measured ways. It used `.name()` (the BARE type
            // name) where numpy prints the FULLY QUALIFIED one, it had no
            // bool arm, and it had no ndim>=1-array arm. Measured
            // 2026-08-04, numpy 2.5.1, `np.zeros(5)`:
            //     .argmin(axis=np.float64(0.0))
            //         np  : 'numpy.float64' object cannot be interpreted...
            //         anionpy: 'float64' object cannot be interpreted...
            //     .argmin(axis=Decimal(0))
            //         np  : 'decimal.Decimal' ...   anionpy: 'Decimal' ...
            //     .argmin(axis=True)
            //         np  : TypeError: an integer is required for the axis
            //         anionpy: AxisError: axis 1 is out of bounds ...
            // 160 diverging cases across argmin/argmax/cumsum method forms.
            // Delegating, not re-copying: the "small per-module copy"
            // convention is precisely what let this drift, exactly as it
            // did for the reduce-family converter in this same file.
            let raw: isize = crate::reductions::single_axis_from_pyobj(ax)?;
            // BUG FOUND + FIXED (2026-08-01, coordinator fuzzing): this
            // used to call the shared `normalize_axis` helper above, which
            // is deliberately strict for `ndim == 0` (correct for
            // `squeeze`/`swapaxes`, its other callers, which do NOT get an
            // axis courtesy on a 0-d array) -- but `argmax`/`argmin` DO get
            // numpy's `axis=0`/`axis=-1` 0-d courtesy no-op (verified:
            // `np.array(3.0).argmax(axis=0) == 0`,
            // `np.array(3.0).argmin(axis=-1) == 0`). Same root bug already
            // found+fixed in `reductions.rs`'s `do_argext`/`do_nanargext`;
            // mirror their inline `n.max(1)` normalization here instead of
            // the shared strict helper.
            let n = ndim as isize;
            let norm = if raw < 0 { raw + n.max(1) } else { raw };
            // BUG FOUND + FIXED (2026-08-01, this task): the `AxisError`'s
            // `ndim` field must be the SAME courtesy `n.max(1)` as the
            // bounds check, not the real `ndim` -- probed directly against
            // real numpy 2.5.1 (0d/1d/2d/3d, several out-of-range axes):
            // a 0-d array's `AxisError` always reports "for array of
            // dimension 1", never "dimension 0". See `reductions.rs`'s
            // `do_argext` for the full probe this mirrors.
            if norm < 0 || norm >= n.max(1) {
                return Err(to_py_err(IonpError::AxisError { axis: raw, ndim: Some(ndim.max(1)) }));
            }
            (a.clone(), vec![norm as usize])
        }
    };
    let mut result = ionp_core::ufunc::argext(op, &target, &axes, keepdims).map_err(to_py_err)?;
    // Same fix as `reductions.rs`'s `do_argext` (this file's established
    // duplication precedent) -- `ndim > 1` skipped the 0-d + keepdims=True
    // case, leaving a spurious shape `(1,)` where real numpy returns a
    // 0-d/scalar result. `ndim != 1` is correct: ndim==1 flattening to 1-D
    // is already a no-op, every other ndim (including 0) needs relabeling.
    if flattened && keepdims && ndim != 1 {
        result = result.reshape(&vec![1usize; ndim]).map_err(to_py_err)?;
    }
    if let Some(out_obj) = out {
        return write_into_out(out_obj, &result, None);
    }
    if result.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &result);
    }
    Ok(Py::new(py, PyArray { inner: result })?.into_any())
}

/// `ndarray.flags` return type. numpy's real `flags` object also supports
/// dict-style `flags['C_CONTIGUOUS']` and `repr()`; both are mirrored here
/// since real code (including numpy's own test helpers) commonly indexes
/// rather than attribute-accesses.
#[pyclass(name = "flagsobj", module = "anionpy")]
pub struct PyFlags {
    c_contiguous: bool,
    f_contiguous: bool,
    owndata: bool,
    writeable: bool,
}

/// Alignment and writeback-if-copy are CONSTANTS for every array anionpy can
/// construct, and are named here rather than stored so that the reason is
/// attached to the value.
///
/// `ALIGNED`: numpy reports `False` only for a buffer whose data pointer is
/// not a multiple of the itemsize -- reachable there exclusively through
/// `.view()` on a byte buffer at an odd offset or `as_strided`. anionpy has
/// neither constructor: every buffer is allocated by `Buffer` itself and is
/// therefore natively aligned. Verified against numpy 2.5.1 that a
/// deliberately unaligned array (`np.zeros(400,uint8)[1:].view(float64)`)
/// is the ONLY way to get `ALIGNED=False`, and that anionpy cannot express it.
const IONP_ALIGNED: bool = true;
/// `WRITEBACKIFCOPY`: numpy sets this only on the temporary produced by the
/// deprecated `PyArray_ResolveWritebackIfCopy` C-API path. There is no
/// Python-level constructor for it in numpy 2.5.1 and no analogue in anionpy.
const IONP_WRITEBACKIFCOPY: bool = false;

#[pymethods]
impl PyFlags {
    // -- lowercase attributes -------------------------------------------
    //
    // These, NOT the uppercase spellings, are real numpy's attribute API:
    // `np.arange(3).flags.C_CONTIGUOUS` raises AttributeError in numpy 2.5.1
    // while `.c_contiguous` works. anionpy had the mapping exactly INVERTED --
    // it published four uppercase getters numpy does not have and none of
    // the twelve lowercase ones it does. Measured, not assumed: `dir()` of
    // both objects differed by 12 names in one direction and 4 in the other.
    //
    // The uppercase getters are kept as an ionp-only superset rather than
    // removed, because the differential harness and several call sites in
    // this crate already read them, and an extra attribute is a smaller
    // divergence than a missing one -- but they are NOT numpy-parity and
    // must not be cited as such. Dict-style `flags['C_CONTIGUOUS']` is the
    // spelling that agrees with numpy.
    #[getter(c_contiguous)]
    fn c_contiguous_lower(&self) -> bool {
        self.c_contiguous
    }
    #[getter(f_contiguous)]
    fn f_contiguous_lower(&self) -> bool {
        self.f_contiguous
    }
    #[getter(owndata)]
    fn owndata_lower(&self) -> bool {
        self.owndata
    }
    #[getter(writeable)]
    fn writeable_lower(&self) -> bool {
        self.writeable
    }
    #[getter(aligned)]
    fn aligned_lower(&self) -> bool {
        IONP_ALIGNED
    }
    #[getter(writebackifcopy)]
    fn writebackifcopy_lower(&self) -> bool {
        IONP_WRITEBACKIFCOPY
    }
    /// `flags.contiguous` / `flags.fortran` -- numpy's legacy aliases for
    /// `c_contiguous` / `f_contiguous` (confirmed identical across an
    /// 18-array sweep against numpy 2.5.1, never divergent).
    #[getter(contiguous)]
    fn contiguous_lower(&self) -> bool {
        self.c_contiguous
    }
    #[getter(fortran)]
    fn fortran_lower(&self) -> bool {
        self.f_contiguous
    }
    /// `flags.behaved` == ALIGNED and WRITEABLE. Measured: the only case in
    /// an 18-array numpy sweep where this went `False` with ALIGNED true was
    /// a broadcast view (`WRITEABLE=False`) -- which anionpy CAN construct, so
    /// this is exercised, not a dead branch.
    #[getter(behaved)]
    fn behaved_lower(&self) -> bool {
        IONP_ALIGNED && self.writeable
    }
    /// `flags.carray` == C_CONTIGUOUS and ALIGNED and WRITEABLE (measured:
    /// a read-only C-contiguous array reports `carray=False`).
    #[getter(carray)]
    fn carray_lower(&self) -> bool {
        self.c_contiguous && IONP_ALIGNED && self.writeable
    }
    /// `flags.farray` -- numpy's value here is NOT the symmetric partner of
    /// `carray`. Measured across 18 numpy 2.5.1 arrays spanning every
    /// combination of C/F contiguity, alignment and writeability, `farray`
    /// tracked `not C_CONTIGUOUS` exactly, including cases where
    /// F_CONTIGUOUS was False (a negative-stride 2-D view: C=0, F=0,
    /// farray=1) and cases where WRITEABLE was False (a read-only Fortran
    /// array: farray=1). The obvious formula `F && ALIGNED && WRITEABLE`
    /// is contradicted by both of those, so it is not used.
    ///
    /// The discriminating case (C=0, F=0, ALIGNED=1, WRITEABLE=1) is
    /// reachable in anionpy via a negative-stride view, so this is validated
    /// inside anionpy's own state space rather than extrapolated into it. What
    /// is NOT validated is the ALIGNED=False corner, which anionpy cannot
    /// construct at all.
    #[getter(farray)]
    fn farray_lower(&self) -> bool {
        !self.c_contiguous
    }
    /// `flags.num` -- numpy's raw flag bitmask. Bit values were read off
    /// numpy 2.5.1 by solving the observed `num` against the observed
    /// booleans over 18 arrays (each bit isolated by at least one pair
    /// differing in only that flag), not copied from a header: C=0x001,
    /// F=0x002, OWNDATA=0x004, ALIGNED=0x100, WRITEABLE=0x400.
    /// WRITEBACKIFCOPY's bit is NOT determined by that sweep -- it was
    /// False in all 18 -- so it contributes 0 here for the same reason it
    /// reports False, and this getter is exact only while that holds.
    #[getter(num)]
    fn num_lower(&self) -> i64 {
        let mut n = 0i64;
        if self.c_contiguous {
            n |= 0x001;
        }
        if self.f_contiguous {
            n |= 0x002;
        }
        if self.owndata {
            n |= 0x004;
        }
        if IONP_ALIGNED {
            n |= 0x100;
        }
        if self.writeable {
            n |= 0x400;
        }
        n
    }

    // -- ionp-only uppercase attributes (see note above) ------------------
    #[getter(C_CONTIGUOUS)]
    fn c_contiguous(&self) -> bool {
        self.c_contiguous
    }
    #[getter(F_CONTIGUOUS)]
    fn f_contiguous(&self) -> bool {
        self.f_contiguous
    }
    #[getter(OWNDATA)]
    fn owndata(&self) -> bool {
        self.owndata
    }
    #[getter(WRITEABLE)]
    fn writeable(&self) -> bool {
        self.writeable
    }
    /// `flags.fnc` -- real numpy's derived "Fortran, Not C" boolean
    /// (`F_CONTIGUOUS and not C_CONTIGUOUS`), backing `np.isfortran(a)`
    /// (`_core/numeric.py`: `return a.flags.fnc`, verified against numpy
    /// 2.5.1 -- no `asarray` fallback, no try/except, a bare attribute
    /// chain). Added for the comparison/predicate/introspection cluster's
    /// `anionpy.isfortran`, 2026-08-02.
    #[getter(fnc)]
    fn fnc(&self) -> bool {
        self.f_contiguous && !self.c_contiguous
    }
    /// `flags.forc` -- real numpy's derived "Fortran or C" boolean
    /// (`F_CONTIGUOUS or C_CONTIGUOUS`). Not currently consumed by any
    /// declared item in this task, added alongside `fnc` because both are
    /// the same two-line addition to the same struct and `flagsobj` is
    /// otherwise missing this documented numpy attribute entirely.
    #[getter(forc)]
    fn forc(&self) -> bool {
        self.f_contiguous || self.c_contiguous
    }
    /// Dict-style access. The accepted key set was enumerated against numpy
    /// 2.5.1 by probing candidates rather than transcribed: all 22 keys
    /// below were accepted and `"UPDATEIFCOPY"` was rejected, so the
    /// long-removed legacy spelling is deliberately absent here too.
    ///
    /// numpy raises `KeyError` for an unknown key (measured -- probing
    /// `"UPDATEIFCOPY"` produced `KeyError`, not `ValueError`). anionpy
    /// previously raised `ValueError` for every key outside its four, which
    /// meant `try: f['ALIGNED'] except KeyError` -- the idiom numpy's own
    /// docs imply -- would not catch. Both halves of that are fixed.
    fn __getitem__(&self, key: &str) -> PyResult<bool> {
        match key {
            "C_CONTIGUOUS" | "C" | "CONTIGUOUS" => Ok(self.c_contiguous),
            "F_CONTIGUOUS" | "F" | "FORTRAN" => Ok(self.f_contiguous),
            "OWNDATA" | "O" => Ok(self.owndata),
            "WRITEABLE" | "W" => Ok(self.writeable),
            "ALIGNED" | "A" => Ok(IONP_ALIGNED),
            "WRITEBACKIFCOPY" | "X" => Ok(IONP_WRITEBACKIFCOPY),
            "B" | "BEHAVED" => Ok(self.behaved_lower()),
            "CA" | "CARRAY" => Ok(self.carray_lower()),
            "FA" | "FARRAY" => Ok(self.farray_lower()),
            "FNC" => Ok(self.fnc()),
            "FORC" => Ok(self.forc()),
            // numpy's message is the bare string "Unknown flag" -- it does NOT
            // name the offending key (measured on two different bad keys,
            // both `KeyError('Unknown flag',)`). Matched literally.
            _ => Err(PyKeyError::new_err("Unknown flag")),
        }
    }
    /// numpy's `repr` is six rows with Python-spelled booleans. anionpy printed
    /// four rows with Rust's `Display` for `bool` -- lowercase `true`/
    /// `false` -- so the repr was distinguishable from numpy's at a glance,
    /// which is the one thing a repr must not be.
    fn __repr__(&self) -> String {
        fn b(v: bool) -> &'static str {
            if v {
                "True"
            } else {
                "False"
            }
        }
        format!(
            "  C_CONTIGUOUS : {}\n  F_CONTIGUOUS : {}\n  OWNDATA : {}\n  WRITEABLE : {}\n  \
             ALIGNED : {}\n  WRITEBACKIFCOPY : {}\n",
            b(self.c_contiguous),
            b(self.f_contiguous),
            b(self.owndata),
            b(self.writeable),
            b(IONP_ALIGNED),
            b(IONP_WRITEBACKIFCOPY),
        )
    }
}

/// `anionpy.shares_memory(a, b)`. Bounds-based (not exact-lattice) overlap
/// test -- see `NdArray::may_share_memory_with`'s doc for the precise
/// caveat (two disjoint-but-bounds-overlapping strided views, e.g.
/// `a[0::2]`/`a[1::2]`, would be reported as sharing memory here though
/// real numpy's exact `shares_memory` says `False` for that specific
/// pattern). Declared/verified only for the simpler patterns this task's
/// probes actually exercise (contiguous and single-step slices, whole-view
/// aliasing) -- not a general claim of numpy's exact semantics.
#[pyfunction]
fn shares_memory(a: &PyArray, b: &PyArray) -> bool {
    a.inner.may_share_memory_with(&b.inner)
}

/// `anionpy.may_share_memory(a, b)`. Numpy's own conservative/bounds-based
/// check -- `NdArray::may_share_memory_with` matches its contract directly
/// (over-report allowed, under-report is not).
#[pyfunction]
fn may_share_memory(a: &PyArray, b: &PyArray) -> bool {
    a.inner.may_share_memory_with(&b.inner)
}

pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFlags>()?;
    m.add_function(wrap_pyfunction!(shares_memory, m)?)?;
    m.add_function(wrap_pyfunction!(may_share_memory, m)?)?;
    Ok(())
}

/// Shared body of `ndarray.conj`/`ndarray.conjugate` -- see the measured
/// numpy table above the two methods for why this is not a single code
/// path. Takes the receiver as a `&Bound` rather than `&self` precisely so
/// the non-complex branch can hand BACK that same Python object, which is
/// what numpy's method does and what a `&self` signature cannot express.
/// Argument shim for `conj`/`conjugate`. Hand-parses `*args`/`**kwargs`
/// instead of declaring `(out=None, /)` because pyo3's generated messages
/// for arity and keyword misuse do not match numpy's, and the messages are
/// part of the contract. Measured on numpy 2.5.1:
///
/// ```text
///   a.conj(o, o)      TypeError: conjugate() takes at most 1 argument (2 given)
///   a.conjugate(o, o) TypeError: conjugate() takes at most 1 argument (2 given)
///   a.conj(out=o)     TypeError: ndarray.conj() takes no keyword arguments
///   a.conjugate(zz=1) TypeError: ndarray.conjugate() takes no keyword arguments
/// ```
///
/// Note the ASYMMETRY, which is numpy's and is reproduced deliberately: the
/// arity message says "conjugate()" for BOTH spellings (there is one
/// underlying method and `conj` is its alias), while the keyword message
/// uses the spelling actually called, qualified with "ndarray.".
///
/// CAVEAT on the keyword message, recorded so nobody promotes it into the
/// corpus later: its text is generated by CPython, not numpy, and it varies
/// with CALL SYNTAX rather than with anything numpy does --
///
/// ```text
///   a.conj(out=o)              -> "ndarray.conj() takes no keyword arguments"
///   getattr(a, "conj")(out=o)  -> "conj() takes no keyword arguments"
/// ```
///
/// Same object, same numpy, two different strings. We match the direct
/// attribute spelling because that is what a user writes. A differential
/// case for this belongs nowhere near the corpus: the corpus dispatches by
/// NAME (i.e. through the `getattr` form), so such a case would be
/// asserting a property of the interpreter's calling convention rather than
/// of numpy, and would "fail" against a perfectly correct implementation.
fn conjugate_dispatch(
    slf: &Bound<'_, PyArray>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    method_name: &str,
) -> PyResult<Py<PyAny>> {
    if kwargs.is_some_and(|d| !d.is_empty()) {
        return Err(PyTypeError::new_err(format!("ndarray.{method_name}() takes no keyword arguments")));
    }
    if args.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "conjugate() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    let out = args.get_item(0).ok();
    conjugate_method(slf, out.as_ref())
}

fn conjugate_method(slf: &Bound<'_, PyArray>, out: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    // An explicit `None` is not an `out` -- numpy accepts `a.conj(None)` and
    // treats it as the plain no-output call, so it must fall through to the
    // method semantics below, not into the ufunc path.
    let out = out.filter(|o| !o.is_none());
    if let Some(out_obj) = out {
        // THE OUT PATH IS A DIFFERENT OPERATION, not the no-out path with a
        // destination bolted on -- and it is TWO different operations, split
        // on whether the source is complex. Measured on numpy 2.5.1 over a
        // 5 x 8 dtype grid (/tmp/cj12.py), 40 of 40 cells agreeing:
        //
        //   source dtype   computed dtype   casting error on a bad `out`
        //   ------------   --------------   ----------------------------
        //   non-complex    the SOURCE's     TypeError, "Cannot cast array
        //                  (bool stays      data|scalar from dtype('src')
        //                  bool)            to dtype('out') according to
        //                                   the rule 'same_kind'"
        //   complex        conjugated,      UFuncTypeError, "Cannot cast
        //                  same dtype       ufunc 'conjugate' output from
        //                                   ... with casting rule
        //                                   'same_kind'"
        //
        // The non-complex row is the surprising one, and it is why this
        // branch does NOT call `ufunc::conj_array`: the real ufunc promotes
        // bool to int8, so `np.ones(2, bool).conj(uint8_out)` would wrongly
        // raise "from dtype('int8')" when numpy accepts it outright. With an
        // output the method is a plain COPY of the source cast into `out` --
        // exactly `np.copyto`'s contract, which is why it routes through
        // `round::copy_into` (the same function `np.round`'s `out=` uses).
        //
        // Note also that the identity shortcut in rule 1 above does NOT
        // apply here: `np.arange(3, dtype='int32').conj(o)` returns `o`,
        // and `r is a` is False.
        //
        // `out` is validated HERE rather than downstream, because the method
        // and the ufunc disagree about the wording and anionpy must not inherit
        // the wrong one -- measured:
        //
        //   a.conj([0,0,0])         TypeError "output must be an array"
        //   np.conj(a, out=[0,0,0]) TypeError "return arrays must be of
        //                           ArrayType"
        //   a.conj((o,))            TypeError "output must be an array"
        //   np.conj(a, out=(o,))    ACCEPTED (ufuncs take an out tuple; the
        //                           method does not)
        //
        // `cast::<PyArray>` rejects the tuple and every non-array for free.
        if out_obj.cast::<PyArray>().is_err() {
            return Err(PyTypeError::new_err("output must be an array"));
        }
        let src_is_complex = matches!(slf.borrow().inner.dtype(), DType::C64 | DType::C128);
        if src_is_complex {
            let computed = conjugated_copy(&slf.borrow().inner);
            // "conjugate", never "conj": the two spellings share one ufunc
            // object, so that is the only name numpy ever reports here.
            return crate::write_into_out_ufunc(out_obj, &computed, None, "conjugate", "same_kind", None);
        }
        // Taken as an OWNED contiguous copy before `out` is borrowed
        // mutably, so that `a.conj(a)` -- legal in numpy, and a no-op --
        // cannot trip a RefCell double-borrow panic.
        let src = slf.borrow().inner.to_contiguous();
        let out_bound = out_obj.cast::<PyArray>()?;
        {
            let mut out_ref = out_bound.borrow_mut();
            ionp_core::round::copy_into(&mut out_ref.inner, &src).map_err(to_py_err)?;
        }
        return Ok(out_obj.clone().unbind());
    }
    // NO-OUTPUT PATH. Rule 1 from the table above: for a non-complex dtype
    // the method is a total no-op returning the RECEIVER ITSELF -- not a
    // copy, not a view. Dropping this line does not merely cost a little
    // aliasing; it also breaks the 0-d return type in the opposite
    // direction from the complex case, because the collapse below would
    // then turn `np.ones(()).conj()` into a scalar when numpy returns a
    // 0-d ndarray. (Measured: removing it fails 456 of 532 cells in the
    // no-output grid /tmp/cj5.py, so this line is load-bearing twice over.)
    if !matches!(slf.borrow().inner.dtype(), DType::C64 | DType::C128) {
        return Ok(slf.clone().into_any().unbind());
    }
    let result = conjugated_copy(&slf.borrow().inner);
    // Complex-only, per rule 2 above. `numpy_scalar_from_0d` is the same
    // helper `take` uses (lib.rs) -- anionpy computes the value, numpy is asked
    // only to mint the wrapper TYPE, which a pyclass cannot forge.
    if result.ndim() == 0 {
        return crate::numpy_scalar_from_0d(py, &result);
    }
    Ok(Py::new(py, PyArray { inner: result })?.into_any())
}

/// Elementwise complex conjugate as a fresh contiguous array, dtype
/// unchanged. Non-complex buffers are copied verbatim -- deliberately NOT
/// promoted the way `ufunc::conj_array` promotes bool to int8, because both
/// callers here model the METHOD, whose computed dtype is always the
/// source's (see the measured grid in `conjugate_method`).
fn conjugated_copy(a: &NdArray) -> NdArray {
    let contig = a.to_contiguous();
    let buffer = match contig.buffer().clone() {
        Buffer::C64(v) => Buffer::C64(v.into_iter().map(|c| c.conj()).collect()),
        Buffer::C128(v) => Buffer::C128(v.into_iter().map(|c| c.conj()).collect()),
        // Total rather than `unreachable!()`: a future dtype added to
        // `Buffer` degrades to a plain copy, which is also the correct
        // answer for every non-complex dtype.
        other => other,
    };
    let raw = NdArray::from_buffer(buffer, contig.shape().to_vec(), ionp_core::Order::C)
        .expect("conjugate: buffer/shape already validated by to_contiguous()'s own construction");
    // ORDER='K'. numpy's conjugate ufunc preserves the INPUT's memory layout;
    // it does not force C. Measured 2026-08-04 on a transposed complex input
    // of shape (6,4):
    //     complex64   numpy strides (8,48)    anionpy (pre-fix) (32,8)
    //     complex128  numpy strides (16,96)   anionpy (pre-fix) (64,16)
    // i.e. an F-ordered input must yield an F-ordered result. VALUES were
    // identical in every one of those cells, so all 1300 corpus cases passed
    // both before and after this fix -- layout is invisible to a value-only
    // comparison, which is exactly why this needed an explicit strides probe
    // (48 checks, /tmp/cj_strides.py) and why the corpus alone was not
    // sufficient evidence to declare the item.
    //
    // Vote on `a`, the ORIGINAL operand, NOT on `contig`: `to_contiguous()`
    // always produces C, so voting on it would hardcode the very bug this
    // fixes. Same reasoning, and same single-operand convention, as
    // `emath.rs::apply_k_order`.
    let perm = NdArray::multi_sorted_stride_perm(raw.ndim(), &[(a.shape(), a.strides())]);
    raw.relayout_by_perm(&perm)
}
