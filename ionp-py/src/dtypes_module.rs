//! TICKET #90a: `anionpy.dtypes` -- a genuine dotted submodule (same
//! `PyModule::new` + `add_submodule` pattern `emath.rs`/`random.rs`/
//! `linalg.rs`/`fft.rs` already use), mirroring real numpy's
//! `numpy.dtypes` module.
//!
//! SCOPE. Real numpy's `numpy.dtypes` exposes ~20 per-dtype classes
//! (`Float64DType`, `Int32DType`, `BoolDType`, ...) -- confirmed live
//! (`tools/coverage.py --list absent`) that every single one is currently
//! absent from anionpy, not just the two this ticket targets. This file
//! registers ONLY `LongDoubleDType`/`CLongDoubleDType`, the two ticket
//! #90a asks for. It deliberately does NOT stub in placeholder classes for
//! the other ~18 (`Float64DType`, `Int32DType`, `BoolDType`, ...) --
//! fabricating unimplemented classes just to make the module "feel
//! complete" would be exactly the "fabricate class-`__dict__` entries"
//! anti-pattern this project's rules forbid, and those 18 remain honestly
//! `absent`, tracked as separate, larger, un-ticketed work (a real
//! `dtype.type`/per-dtype-`__class__` architecture across all 14 concrete
//! `DType` variants, not a 2-class add-on).
//!
//! Each class is a REAL `PyDType` subclass (`extends = PyDType`, which
//! required adding `subclass` to `PyDType`'s own `#[pyclass(...)]`
//! attribute in `lib.rs` -- see that change's comment), constructible with
//! no arguments exactly like real numpy's own per-dtype classes
//! (`np.dtypes.LongDoubleDType()` returns a genuine `dtype('g')`
//! instance): `anionpy.dtypes.LongDoubleDType()` returns a `LongDoubleDType`
//! instance that IS a `dtype` (inherits `.name`/`.char`/`.itemsize`/
//! `__eq__`/`__repr__`/... from `PyDType` for free via normal Python
//! subclassing -- no re-implementation here), tagged with the SAME
//! `spelling = Some('g')`/`Some('G')` ticket #78 introduced for the
//! `'g'`/`'G'` duplicate-dtype spellings, so `.char`/`.num` come out
//! correct without a second mechanism.
//!
//! NOT implemented (disclosed gap, same one `scalars.rs`'s
//! `LongDouble`/`CLongDouble` doc comments disclose): `anionpy.dtype('g')`
//! (constructed the OTHER way, via the string spelling rather than this
//! class) is NOT retroactively an instance of `LongDoubleDType` --
//! `type(anionpy.dtype('g'))` stays plain `anionpy.dtype`, unlike real
//! numpy where `type(np.dtype('g')) is np.dtypes.LongDoubleDType` always.
//! Wiring dtype-string resolution back into a per-spelling `__class__`
//! choice would mean giving EVERY dtype (not just these two) its own
//! `PyDType` subclass and routing `dtype_name_to_dtype`'s return type
//! through a class table instead of a bare enum -- the same larger,
//! out-of-scope architecture change the module doc above declines.

//! COMPILE-TIME GATE. Everything below (`LongDoubleDType`, `CLongDoubleDType`,
//! and this module's contribution to `register()`) is behind
//! `#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]` -- the
//! same condition under which C `long double` is 8 bytes (== `double`) on
//! every target this project's `Cargo.toml`/CI actually builds for. On any
//! OTHER target (glibc/aarch64, x86-64, ...) where `long double` is
//! genuinely 16 bytes, these two classes and even the `anionpy.dtypes`
//! submodule itself do not exist in the compiled `_anionpy` extension at
//! all -- not a runtime Python `if`, a Rust item that is never compiled in.
//! `anionpy/__init__.py`'s `try: from anionpy._anionpy import ... dtypes
//! except ImportError: pass` is what turns "symbol doesn't exist in the
//! compiled module" into "name absent from `anionpy`" on such a target.
//! This is necessarily reasoned about rather than cross-compiled and run
//! (no non-Apple-aarch64 target is available to actually execute in this
//! environment) -- see the ticket report for exactly what was and wasn't
//! verified.

use pyo3::prelude::*;

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
use ionp_core::DType;

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
use crate::PyDType;

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[pyclass(name = "LongDoubleDType", module = "anionpy.dtypes", extends = PyDType)]
pub struct LongDoubleDType;

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[pymethods]
impl LongDoubleDType {
    #[new]
    fn new() -> PyClassInitializer<LongDoubleDType> {
        PyClassInitializer::from(PyDType { inner: DType::F64, spelling: Some('g') })
            .add_subclass(LongDoubleDType)
    }
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[pyclass(name = "CLongDoubleDType", module = "anionpy.dtypes", extends = PyDType)]
pub struct CLongDoubleDType;

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[pymethods]
impl CLongDoubleDType {
    #[new]
    fn new() -> PyClassInitializer<CLongDoubleDType> {
        PyClassInitializer::from(PyDType { inner: DType::C128, spelling: Some('G') })
            .add_subclass(CLongDoubleDType)
    }
}

/// On the gated target, builds `anionpy.dtypes` with both classes and
/// attaches it as a real submodule. On any other target, this is a no-op --
/// no `anionpy.dtypes` submodule is created at all, matching
/// `anionpy/__init__.py`'s `except ImportError: pass` fallback (there is
/// nothing named `dtypes` in `_anionpy` for that import to find).
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    {
        let m = PyModule::new(py, "dtypes")?;
        m.add_class::<LongDoubleDType>()?;
        m.add_class::<CLongDoubleDType>()?;
        parent.add_submodule(&m)?;
    }
    #[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
    {
        let _ = (py, parent);
    }
    Ok(())
}
