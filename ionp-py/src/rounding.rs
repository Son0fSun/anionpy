//! PyO3 bindings for `ionp_core::round`'s three private primitives.
//!
//! These are the *only* parts of `np.round` that cannot be expressed as a
//! sequence of whole-array `anionpy` calls -- see `ionp-core/src/round.rs`'s
//! module doc for why each one exists and `anionpy/_round_compose.py` for the
//! driver that uses them. All three are underscore-prefixed and are NOT
//! re-exported from `anionpy/__init__.py`: they are implementation detail of
//! `round`/`around`/`ndarray.round`, not new public numpy surface, and in
//! particular `_copy_into` is NOT `np.copyto` (whose `casting=`/`where=`
//! surface has not been measured).
//!
//! Like `manip.rs`, this module only parses Python arguments and marshals
//! them into `ionp-core`; every loop lives there.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use ionp_core::NdArray;

use crate::{to_py_err, PyArray};

/// Accept an `anionpy.ndarray` directly, or ingest any other array-like
/// (list/tuple/foreign `numpy.ndarray`) the same way `manip.rs` does.
/// Only the SOURCE operand goes through this -- the destination must be a
/// real `anionpy.ndarray`, since a list has no storage to mutate.
fn source_array(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(pyref.inner.clone());
    }
    crate::array_impl(obj, None)
}

/// The destination gate. `"output must be an array"` is numpy's own
/// wording for a non-array `out=` argument to `np.round` (measured live,
/// `/tmp/mg_rnd2.py`: `np.round(np.array([1.5]), 0, out=[0.0])` ->
/// `TypeError: output must be an array`), so raising it here means the
/// compose layer does not have to duplicate the check.
fn dest_array<'py>(obj: &'py Bound<'py, PyAny>) -> PyResult<&'py Bound<'py, PyArray>> {
    obj.cast::<PyArray>()
        .map_err(|_| PyTypeError::new_err("output must be an array"))
}

/// numpy's `power_of_ten` (see the core doc). Exposed so the Python
/// driver never computes a scale factor itself -- `10.0 ** n` in Python
/// is both arithmetic in the wrong language and, for `n >= 23`, a
/// different value than numpy's repeated-multiplication loop produces.
#[pyfunction]
fn _power_of_ten(n: i64) -> f64 {
    ionp_core::round::power_of_ten(n)
}

/// `PyArray_CopyInto(dst, src)` with `casting='same_kind'`. Mutates `dst`
/// in place and returns `None`, mirroring numpy's C-level call.
#[pyfunction]
fn _copy_into(dst: &Bound<'_, PyAny>, src: &Bound<'_, PyAny>) -> PyResult<()> {
    let target = dest_array(dst)?;
    let source = source_array(src)?;
    let mut target_ref = target.borrow_mut();
    ionp_core::round::copy_into(&mut target_ref.inner, &source).map_err(to_py_err)
}

/// `dst.real = src`.
#[pyfunction]
fn _set_real(dst: &Bound<'_, PyAny>, src: &Bound<'_, PyAny>) -> PyResult<()> {
    let target = dest_array(dst)?;
    let source = source_array(src)?;
    let mut target_ref = target.borrow_mut();
    ionp_core::round::set_part(&mut target_ref.inner, &source, false).map_err(to_py_err)
}

/// `dst.imag = src`.
#[pyfunction]
fn _set_imag(dst: &Bound<'_, PyAny>, src: &Bound<'_, PyAny>) -> PyResult<()> {
    let target = dest_array(dst)?;
    let source = source_array(src)?;
    let mut target_ref = target.borrow_mut();
    ionp_core::round::set_part(&mut target_ref.inner, &source, true).map_err(to_py_err)
}

pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_power_of_ten, m)?)?;
    m.add_function(wrap_pyfunction!(_copy_into, m)?)?;
    m.add_function(wrap_pyfunction!(_set_real, m)?)?;
    m.add_function(wrap_pyfunction!(_set_imag, m)?)?;
    Ok(())
}
