//! anionpy's OWN `LinAlgError` and `AxisError` classes.
//!
//! `import anionpy` must succeed with no numpy installed at all (GOAL-ionp.md).
//! Before this file existed, `linalg.rs`'s module-init line unconditionally
//! did `py.import("numpy.linalg")?.getattr("LinAlgError")?` with a bare `?`
//! -- the ONE runtime numpy borrow in the whole crate with no
//! numpy-absent fallback, which alone made `import anionpy` require numpy.
//!
//! The fix builds anionpy's own class, once, lazily, the first time either
//! error is needed (which happens no later than `linalg::register`, itself
//! called from `_anionpy()`'s module init, so the class exists in time to be
//! published as `anionpy.linalg.LinAlgError` same as before):
//!   - numpy importable  -> anionpy's class SUBCLASSES numpy's real class, so
//!     `except numpy.linalg.LinAlgError:` / `except numpy.exceptions.AxisError:`
//!     still catch errors anionpy raises (drop-in compatibility in the
//!     direction that matters: user code written against numpy).
//!   - numpy absent      -> `LinAlgError` subclasses `ValueError`;
//!     `AxisError` subclasses BOTH `ValueError` and `IndexError` (numpy's
//!     own `AxisError` does the same dual inheritance, verified against
//!     numpy 2.5.1's `numpy/exceptions.py`, so callers relying on either
//!     `except` clause keep working even without numpy on the machine).
//!
//! Neither path ever uses a bare `?` on the numpy import -- an absent
//! numpy falls back, it never aborts module init.
//!
//! The AxisError fallback's `__init__`/`__str__` are plain Python functions
//! compiled once via `PyModule::from_code` (NOT constructed by calling into
//! numpy -- this is our own hand-written reimplementation of numpy's
//! `AxisError.__init__`/`__str__` algorithm, verified line-for-line against
//! `numpy/exceptions.py` on numpy 2.5.1, see the source string below).
//! Plain Python functions are used (rather than Rust-implemented `#[pyclass]`
//! methods) so the two fallback bases can be genuine builtin `ValueError`/
//! `IndexError` with zero added instance layout -- multiple-inheriting two
//! *different* C-extension types that each add native struct fields risks
//! CPython's "instance lay-out conflict", which pure-Python functions on a
//! `type()`-built class never trigger (this is exactly how numpy's own
//! `class AxisError(ValueError, IndexError)` is implemented: pure Python,
//! no extra C slots).

use std::ffi::CString;

use pyo3::exceptions::{PyIndexError, PyRuntimeWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyModule, PyTuple};

static LINALG_ERROR_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static AXIS_ERROR_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static COMPLEX_WARNING_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static NUMPY_AVAILABLE: PyOnceLock<bool> = PyOnceLock::new();
static IONP_MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

/// Whether `numpy` is importable in this process, cached after the first
/// check (an import that already failed once will not spontaneously start
/// succeeding later in the same interpreter).
///
/// Exists because several `numpy`-crate type-check entry points --
/// `.cast::<numpy::PyUntypedArray>()`, and anything else that goes through
/// `numpy::npyffi::get_type_object` -- do NOT return a `PyResult` on
/// failure to reach numpy's C API capsule. They `.expect(...)` internally
/// and PANIC the whole thread, even when merely probing "is this object a
/// numpy array" on something that plainly isn't one (verified live: calling
/// any `.sum(axis=<int>)` with numpy completely absent panicked inside
/// `stats::axis_array_ndim`'s unconditional `.cast::<numpy::PyUntypedArray>()`
/// probe, despite `axis` being an ordinary Python `int` -- rust-numpy's
/// `is_type_of` fetches the numpy type object unconditionally before it can
/// even compare it against `int`'s type). Any call site that might run with
/// numpy absent must check `numpy_available` FIRST and skip the numpy
/// cast/type-check branch entirely when `false`, rather than let
/// rust-numpy's own machinery decide -- this is the guard, not a general
/// "is this specific object a numpy array" answer.
pub(crate) fn numpy_available(py: Python<'_>) -> bool {
    *NUMPY_AVAILABLE.get_or_init(py, || py.import("numpy").is_ok())
}

/// Handle to anionpy's own already-imported COMPILED extension module,
/// `anionpy._anionpy` -- NOT the `anionpy` package itself. `#[pymodule] fn _anionpy`
/// (this file's crate root, `lib.rs`) is the module `scalars::register`
/// installs every scalar pyclass onto directly; `anionpy/__init__.py` is a
/// hand-written Python wrapper that re-exports only a curated subset of
/// those names (e.g. it exports `bool_` but deliberately withholds `bool`,
/// which would shadow the builtin) -- verified live: `hasattr(anionpy, "bool")`
/// is `False` while `hasattr(anionpy._anionpy, "bool")` is `True`. Looking up
/// `dtype().name()` (which yields the SHORT numpy-style names like
/// `"bool"`/`"int64"`) against the wrong module would `AttributeError` on
/// every dtype the wrapper renames or omits, `"bool"` included.
///
/// Cached after the first lookup (same `PyOnceLock` idiom as
/// `numpy_available`/the error class caches above). By the time any call
/// site needs this, `import anionpy` (which imports `anionpy._anionpy` as a side
/// effect) has already run to completion -- these are runtime calls made
/// FROM Python code that already holds a reference to the package -- so
/// `py.import("anionpy._anionpy")` here just re-fetches the live module object
/// from `sys.modules` rather than re-running module init.
pub(crate) fn ionp_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    IONP_MODULE
        .get_or_try_init(py, || py.import("anionpy._anionpy").map(|m| m.unbind()))
        .map(|m| m.bind(py).clone())
}

/// Hand-written reimplementation of `numpy.exceptions.AxisError`'s
/// `__init__`/`__str__`, verified line-for-line against numpy 2.5.1's
/// `numpy/exceptions.py`:
///
/// ```python
/// class AxisError(ValueError, IndexError):
///     def __init__(self, axis, ndim=None, msg_prefix=None):
///         if ndim is msg_prefix is None:
///             self._msg = axis
///             self.axis = None
///             self.ndim = None
///         else:
///             self._msg = msg_prefix
///             self.axis = axis
///             self.ndim = ndim
///     def __str__(self):
///         axis, ndim = self.axis, self.ndim
///         if axis is ndim is None:
///             return self._msg
///         msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
///         if self._msg is not None:
///             msg = f"{self._msg}: {msg}"
///         return msg
/// ```
/// Only used when numpy itself is not importable (see `axis_error_class`
/// below); when numpy IS present, anionpy's class subclasses numpy's real
/// `AxisError` and inherits ITS `__init__`/`__str__` instead, so this
/// source only has to match numpy's behavior for the numpy-absent case,
/// where there is no live numpy to diverge from anyway.
const AXIS_ERROR_FALLBACK_SRC: &str = "\
def __init__(self, axis, ndim=None, msg_prefix=None):
    if ndim is None and msg_prefix is None:
        self._msg = axis
        self.axis = None
        self.ndim = None
    else:
        self._msg = msg_prefix
        self.axis = axis
        self.ndim = ndim

def __str__(self):
    axis = self.axis
    ndim = self.ndim
    if axis is None and ndim is None:
        return self._msg
    msg = f\"axis {axis} is out of bounds for array of dimension {ndim}\"
    if self._msg is not None:
        msg = f\"{self._msg}: {msg}\"
    return msg
";

fn axis_error_fallback_namespace<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let code = CString::new(AXIS_ERROR_FALLBACK_SRC).expect("no interior NUL");
    let module = PyModule::from_code(
        py,
        code.as_c_str(),
        c"_ionp_axis_error_fallback.py",
        c"_ionp_axis_error_fallback",
    )?;
    let ns = PyDict::new(py);
    ns.set_item("__init__", module.getattr("__init__")?)?;
    ns.set_item("__str__", module.getattr("__str__")?)?;
    Ok(ns)
}

/// Builds `type(name, bases, namespace)`, first stamping `__module__` and
/// `__qualname__` into `namespace`.
///
/// Without this, `type()`'s 3-arg form takes `__module__` from the calling
/// Python frame's `__name__` global -- and since the call here originates
/// from Rust via PyO3 rather than from a `def`/`class` statement in some
/// `anionpy` source file, CPython falls back to whatever frame happens to be
/// on top of the interpreter stack at the time (observed live: `__main__`
/// when the class is built lazily from a user's top-level script, i.e.
/// `repr(type(exc))` prints `<class '__main__.AxisError'>` -- plainly wrong
/// regardless of the exact-type-vs-isinstance harness question, since these
/// classes are unambiguously part of `anionpy`, not the caller's module).
/// Setting `__module__`/`__qualname__` directly in the namespace dict
/// handed to `type()` is the constructor-carried fix `type()` itself reads
/// from (rather than assigning `cls.__module__ = ...` after the fact,
/// which works but fights the constructor instead of using it).
fn build_class<'py>(
    py: Python<'py>,
    name: &str,
    bases: Vec<Bound<'py, PyAny>>,
    namespace: Bound<'py, PyDict>,
) -> PyResult<Py<PyAny>> {
    namespace.set_item("__module__", "anionpy")?;
    namespace.set_item("__qualname__", name)?;
    let bases_tuple = PyTuple::new(py, bases)?;
    let type_builtin = py.import("builtins")?.getattr("type")?;
    let cls = type_builtin.call1((name, bases_tuple, namespace))?;
    Ok(cls.unbind())
}

fn build_linalg_error_class(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let bases: Vec<Bound<'_, PyAny>> = match py
        .import("numpy.linalg")
        .and_then(|m| m.getattr("LinAlgError"))
    {
        // numpy present: subclass its real, public `LinAlgError` so
        // `except numpy.linalg.LinAlgError:` still catches anionpy's own.
        // `LinAlgError` itself has no custom `__init__` (inherits
        // `BaseException`'s), so no namespace overrides are needed here.
        Ok(np_cls) => vec![np_cls],
        // numpy absent: same shape numpy's own class has --
        // `class LinAlgError(ValueError): pass`.
        Err(_) => vec![py.get_type::<PyValueError>().into_any()],
    };
    build_class(py, "LinAlgError", bases, PyDict::new(py))
}

fn build_axis_error_class(py: Python<'_>) -> PyResult<Py<PyAny>> {
    match py
        .import("numpy.exceptions")
        .and_then(|m| m.getattr("AxisError"))
    {
        // numpy present: subclass its real `AxisError`, inheriting its
        // genuine `__init__`/`__str__` -- no reimplementation needed or
        // used on this path.
        Ok(np_cls) => build_class(py, "AxisError", vec![np_cls], PyDict::new(py)),
        // numpy absent: `class AxisError(ValueError, IndexError)` with
        // anionpy's own hand-written `__init__`/`__str__` (see
        // `AXIS_ERROR_FALLBACK_SRC`'s doc comment above).
        Err(_) => {
            let bases = vec![
                py.get_type::<PyValueError>().into_any(),
                py.get_type::<PyIndexError>().into_any(),
            ];
            let ns = axis_error_fallback_namespace(py)?;
            build_class(py, "AxisError", bases, ns)
        }
    }
}

/// Returns anionpy's `LinAlgError` class, building (and caching) it on first
/// use. Never lets a numpy-import failure abort the caller -- the numpy
/// probe inside `build_linalg_error_class` always falls back rather than
/// propagating.
pub(crate) fn linalg_error_class(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    LINALG_ERROR_CLASS
        .get_or_try_init(py, || build_linalg_error_class(py))
        .map(|c| c.bind(py).clone())
}

/// `linalg_error_class`'s `AxisError` sibling.
pub(crate) fn axis_error_class(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    AXIS_ERROR_CLASS
        .get_or_try_init(py, || build_axis_error_class(py))
        .map(|c| c.bind(py).clone())
}

/// Raise anionpy's `LinAlgError` with `msg` as its sole constructor argument
/// -- same call shape `linalg.rs`'s `linalg_err`/`linalg_err_raw` already
/// used when constructing the real numpy class directly.
pub(crate) fn raise_linalg_error(py: Python<'_>, msg: &str) -> PyResult<PyErr> {
    let cls = linalg_error_class(py)?;
    let instance = cls.call1((msg,))?;
    Ok(PyErr::from_value(instance))
}

/// Raise anionpy's `AxisError(axis, ndim)` / `AxisError(axis)` (single-argument
/// custom-message form when `ndim` is `None`) -- same call shape `lib.rs`'s
/// `axis_error` and `manip.rs`'s `rollaxis` already used when constructing
/// the real numpy class directly.
pub(crate) fn raise_axis_error(py: Python<'_>, axis: isize, ndim: Option<usize>) -> PyResult<PyErr> {
    let cls = axis_error_class(py)?;
    let instance = match ndim {
        Some(nd) => cls.call1((axis, nd))?,
        None => cls.call1((axis,))?,
    };
    Ok(PyErr::from_value(instance))
}

/// `raise_axis_error`'s `msg_prefix`-carrying sibling, for `moveaxis`/
/// `linspace(..., axis=...)` -- see `lib.rs`'s `axis_error_prefixed` doc
/// comment for why this third-argument form exists.
pub(crate) fn raise_axis_error_prefixed(
    py: Python<'_>,
    axis: isize,
    ndim: usize,
    prefix: &str,
) -> PyResult<PyErr> {
    let cls = axis_error_class(py)?;
    let instance = cls.call1((axis, ndim, prefix))?;
    Ok(PyErr::from_value(instance))
}

/// `manip.rs`'s `rollaxis` raises `AxisError` with a single, already-fully-
/// formatted custom message (the "single-argument form" per numpy's own
/// `AxisError.__init__`, where `axis` positionally holds the message
/// string, not a numeric axis). Kept separate from `raise_axis_error`
/// (whose `axis` parameter is `isize`, incompatible with a `String`).
pub(crate) fn raise_axis_error_custom_msg(py: Python<'_>, msg: &str) -> PyResult<PyErr> {
    let cls = axis_error_class(py)?;
    let instance = cls.call1((msg,))?;
    Ok(PyErr::from_value(instance))
}

/// `ComplexWarning`'s sibling to `build_linalg_error_class`/
/// `build_axis_error_class` above, SAME identity policy:
///   - numpy importable  -> subclass numpy's real, public
///     `numpy.exceptions.ComplexWarning` (there is no top-level
///     `numpy.ComplexWarning` alias in numpy 2.5.1 -- measured live,
///     `numpy.exceptions` is the sole public home), so
///     `except numpy.exceptions.ComplexWarning:` (and, since Python
///     warning classes are matched by `issubclass`, `except
///     RuntimeWarning:`) still catch what anionpy raises.
///   - numpy absent      -> subclass the builtin numpy's own class itself
///     derives from. Verified live (not assumed), numpy 2.5.1:
///     `numpy.exceptions.ComplexWarning.__mro__` is `(ComplexWarning,
///     RuntimeWarning, Warning, Exception, BaseException, object)` -- a
///     plain `class ComplexWarning(RuntimeWarning): pass`, no custom
///     `__init__`/`__str__` of its own (inherits `Exception`'s), so unlike
///     `AxisError` there is no hand-written fallback body to reimplement
///     here -- an empty namespace is the whole class.
fn build_complex_warning_class(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let bases: Vec<Bound<'_, PyAny>> = match py
        .import("numpy.exceptions")
        .and_then(|m| m.getattr("ComplexWarning"))
    {
        Ok(np_cls) => vec![np_cls],
        Err(_) => vec![py.get_type::<PyRuntimeWarning>().into_any()],
    };
    build_class(py, "ComplexWarning", bases, PyDict::new(py))
}

/// `linalg_error_class`/`axis_error_class`'s `ComplexWarning` sibling.
pub(crate) fn complex_warning_class(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    COMPLEX_WARNING_CLASS
        .get_or_try_init(py, || build_complex_warning_class(py))
        .map(|c| c.bind(py).clone())
}

/// numpy's exact `ComplexWarning` text, verified live against numpy 2.5.1
/// (`x.__array__(dtype=float64)` / `np.asarray(x, dtype=float64)` on a
/// complex128 array): `"Casting complex values to real discards the
/// imaginary part"`. Shared by every cast call site below so the wording
/// can never drift between them.
const COMPLEX_CAST_WARNING_MSG: &::std::ffi::CStr =
    c"Casting complex values to real discards the imaginary part";

/// Emits anionpy's `ComplexWarning` with numpy's own message text, IFF `from`
/// is a complex dtype and `to` is not -- the single condition numpy's own
/// cast-time warning fires on (verified live: narrowing within complex,
/// e.g. complex128 -> complex64, and any real->real/int->float cast, both
/// stay silent on numpy's side too).
///
/// `stacklevel=2`: matches the existing `warn_runtime` idiom in
/// `reductions.rs` (the all-NaN-slice `RuntimeWarning`, anionpy's one other
/// shipped warning site) and was verified live to reproduce numpy's own
/// attribution -- `x.astype('float64')` on numpy 2.5.1 reports the warning
/// at the CALLER's line, not inside numpy's C cast routine, i.e. one frame
/// up from the `#[pymethod]`/`#[pyfunction]` body that calls this helper,
/// which is exactly what `stacklevel=2` produces from here.
///
/// Never called with numpy imported for the VALUE or the message text --
/// both are anionpy's own (the class per `complex_warning_class`'s doc, the
/// text a literal constant copied from a one-time live measurement) -- only
/// `complex_warning_class`'s numpy PROBE (identity, not computation) may
/// touch numpy, and only when numpy happens to be installed.
/// MEASURED 2026-08-06 (differential corpus false-negative caught in
/// review, not assumed): real numpy does NOT emit `ComplexWarning` when the
/// destination dtype is `bool`. `astype('bool')`/`array(..., dtype=bool)`
/// on a complex source treats the cast as a truthiness test (`x != 0`), not
/// a numeric real-part-discarding downcast, and that specific path never
/// goes through the warning machinery -- confirmed live against numpy
/// 2.5.1 across all four fixed sites, zero warnings on all of them, for
/// both zero and nonzero imaginary parts. Excluding `to.is_bool()` here is
/// therefore not a guess, it is matching a measured negative.
pub(crate) fn warn_complex_cast(py: Python<'_>, from: ionp_core::DType, to: ionp_core::DType) -> PyResult<()> {
    if from.is_complex() && !to.is_complex() && !to.is_bool() {
        let cat = complex_warning_class(py)?;
        PyErr::warn(py, &cat, COMPLEX_CAST_WARNING_MSG, 2)?;
    }
    Ok(())
}
