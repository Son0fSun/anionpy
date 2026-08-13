//! Array-protocol dunders for `anionpy.ndarray`: `__array_ufunc__`,
//! `__array_function__`, `__array_finalize__`, `__array_wrap__`,
//! `__array_priority__`, `__array_namespace__`, `__array_interface__`,
//! `__array_struct__`, `__dlpack_device__`.
//!
//! SEPARATE `#[pymethods] impl PyArray` block, same pattern as
//! `ndarray_attrs.rs` (see that file's module doc for why: pyo3's
//! `multiple-pymethods` Cargo feature, already enabled in
//! `ionp-py/Cargo.toml`, lets this whole file's churn stay off `lib.rs`
//! beyond the one `mod array_protocol;` line).
//!
//! SCOPE, measured against real numpy 2.5.1 (see
//! `/private/tmp/probe_protocol.py`-style probes, not shipped): numpy 2.x
//! removed `__array_prepare__` from `ndarray` entirely (`hasattr(np.ndarray,
//! "__array_prepare__")` is `False`) -- it is correctly absent from
//! `tools/numpy_surface.json`'s `ndarray` dunder list and is NOT
//! implemented here; there is nothing to implement. `__buffer__` (PEP 688,
//! real `tp_as_buffer` C-level wiring) and the full `__dlpack__` capsule
//! (a `DLManagedTensor` with a producer-side deleter contract) are NOT
//! implemented here either -- see this task's report for the exact reasons
//! and repros; `__dlpack_device__` (the paired, much smaller device-tuple
//! query) IS implemented below since it carries none of that risk.
//!
//! `__array_interface__`/`__array_struct__` expose a REAL pointer into this
//! array's own live buffer (via `Buffer`'s per-variant `Vec<T>::as_ptr()`),
//! not a copy -- except for `S`/`U` string dtypes, whose `Buffer::S`/`U`
//! storage is `Vec<Vec<u8>>`/`Vec<Vec<u32>>` (one heap allocation per
//! element, see `ionp-core/src/buffer.rs`'s own doc comment), which has no
//! single flat address to report at all. Both getters raise
//! `NotImplementedError` for `S`/`U` rather than fabricate one; declared/
//! tested only for the 12 non-string dtypes.

use core::ffi::{c_char, c_int, c_void};

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyTuple};

use ionp_core::{Buffer, DType};

use crate::{numpy_scalar_from_0d, PyArray, PyDType};

/// The raw `PyArrayInterface` C struct (`numpy/ndarraytypes.h`) this
/// module's `__array_struct__` getter builds and leaks into a capsule.
/// Module-level (not nested in the getter) so `unsafe impl Send` can be
/// written for it: the struct is plain, self-contained POD handed to a
/// consumer exactly like real numpy's own C code hands it out (no thread
/// affinity, no interior mutability), so `Send`ing the raw pointers it
/// carries is sound by the same reasoning numpy's C implementation relies
/// on implicitly.
#[repr(C)]
struct RawArrayInterface {
    two: c_int,
    nd: c_int,
    typekind: c_char,
    itemsize: c_int,
    flags: c_int,
    shape: *mut isize,
    strides: *mut isize,
    data: *mut c_void,
    descr: *mut c_void,
}

// SAFETY: see the struct's doc comment above.
unsafe impl Send for RawArrayInterface {}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// The array-protocol `typestr`: identical string `PyDType::str` (lib.rs)
/// already computes for `dtype.str`. Reconstructed via a bare `PyDType`
/// value (both its fields are `pub(crate)`, visible from this module)
/// rather than duplicating the byte-order-marker/itemsize logic a second
/// time.
fn typestr(dtype: DType) -> String {
    PyDType { inner: dtype, spelling: None }.str()
}

/// Raw base pointer (element 0 of the BUFFER, i.e. before this array's own
/// `.offset()` is applied) into a `Buffer`'s live storage, as a plain
/// address. `None` for `S`/`U` (see module doc: no single flat allocation
/// exists to point at).
fn buffer_base_ptr(buf: &Buffer) -> Option<usize> {
    Some(match buf {
        Buffer::Bool(v) => v.as_ptr() as usize,
        Buffer::I8(v) => v.as_ptr() as usize,
        Buffer::I16(v) => v.as_ptr() as usize,
        Buffer::I32(v) => v.as_ptr() as usize,
        Buffer::I64(v) => v.as_ptr() as usize,
        Buffer::U8(v) => v.as_ptr() as usize,
        Buffer::U16(v) => v.as_ptr() as usize,
        Buffer::U32(v) => v.as_ptr() as usize,
        Buffer::U64(v) => v.as_ptr() as usize,
        Buffer::F16(v) => v.as_ptr() as usize,
        Buffer::F32(v) => v.as_ptr() as usize,
        Buffer::F64(v) => v.as_ptr() as usize,
        Buffer::C64(v) => v.as_ptr() as usize,
        Buffer::C128(v) => v.as_ptr() as usize,
        Buffer::S(..) | Buffer::U(..) => return None,
    })
}

/// `(address, readonly)` of element `arr.offset()` in `arr`'s buffer, or
/// `None` for an `S`/`U` array. `readonly` mirrors the same `_readonly`
/// Python-`__dict__` marker `.flags`'s `WRITEABLE` getter reads
/// (`ndarray_attrs.rs::flags`, set by `lib.rs::mark_readonly` for
/// `broadcast_to`/`broadcast_arrays` results) -- the one case anionpy
/// itself considers non-writeable.
fn data_ptr_and_readonly(slf: &Bound<'_, PyArray>) -> PyResult<Option<(usize, bool)>> {
    let arr = &slf.borrow().inner;
    let Some(base) = buffer_base_ptr(arr.buffer()) else {
        return Ok(None);
    };
    let itemsize = arr.dtype().itemsize() as isize;
    let addr = (base as isize + arr.offset() * itemsize) as usize;
    let readonly = matches!(
        slf.getattr("_readonly"),
        Ok(v) if v.extract::<bool>().unwrap_or(false)
    );
    Ok(Some((addr, readonly)))
}

fn no_flat_buffer_err(dtype: DType) -> PyErr {
    PyNotImplementedError::new_err(format!(
        "anionpy: {dtype} has no single flat buffer to expose through the array \
         protocol (S/U storage is one heap allocation per element, not a \
         contiguous run) -- unlike every other anionpy dtype, which shares \
         numpy's flat-buffer layout exactly",
    ))
}

// ---------------------------------------------------------------------------
// The dunders
// ---------------------------------------------------------------------------

#[pymethods]
impl PyArray {
    /// `ndarray.__array_finalize__(obj)`. Verified against real numpy
    /// 2.5.1: calling it directly on a base `ndarray` instance (`obj=None`
    /// or `obj=self`) always returns `None` and has no observable side
    /// effect -- it exists on `ndarray` purely as the hook subclasses
    /// override, and anionpy's `ndarray` cannot itself be subclassed from
    /// Python (`class Foo(anionpy.ndarray)` raises `TypeError`, see
    /// `anionpy/matrix.py`'s module doc), so this is the entire contract:
    /// present, callable, a no-op.
    #[pyo3(signature = (_obj))]
    fn __array_finalize__(&self, _obj: &Bound<'_, PyAny>) {}

    /// `ndarray.__array_wrap__(obj, context=None, return_scalar=False)`.
    /// Verified against real numpy 2.5.1 (base `ndarray`, not a
    /// subclass): with `return_scalar=False` (the historical default)
    /// `obj` comes back completely unchanged, same object, `is`-identical
    /// -- including when `obj` is itself a 0-d array (it stays a 0-d
    /// array, not a scalar). With `return_scalar=True` AND `obj.ndim ==
    /// 0`, the 0-d array collapses to a scalar instead
    /// (`a.__array_wrap__(np.array(5.0), None, True)` -> `numpy.float64(5.0)`,
    /// not a 0-d array) -- any other ndim is returned unchanged regardless
    /// of `return_scalar`. `context` is accepted (numpy passes a
    /// `(ufunc, args, out_index)` tuple internally) and deliberately
    /// unused here: it exists for a subclass overriding `__array_wrap__`
    /// to inspect, and base `ndarray`'s own implementation ignores it too
    /// (verified: passing an arbitrary `context` value changes nothing
    /// about the return).
    #[pyo3(signature = (obj, context=None, return_scalar=false))]
    fn __array_wrap__(
        &self,
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        context: Option<&Bound<'_, PyAny>>,
        return_scalar: bool,
    ) -> PyResult<Py<PyAny>> {
        let _ = context;
        if return_scalar {
            if let Ok(arr) = obj.extract::<PyRef<'_, PyArray>>() {
                if arr.inner.ndim() == 0 {
                    return numpy_scalar_from_0d(py, &arr.inner);
                }
            }
        }
        Ok(obj.clone().unbind())
    }

    /// `ndarray.__array_priority__`. Verified against real numpy 2.5.1:
    /// `numpy.ndarray`'s own default is the plain float `0.0` (a bare
    /// class-level float, not a descriptor doing any computation) --
    /// `matrix`/`memmap`/`MaskedArray` each override it to a different
    /// constant on THEIR OWN class in anionpy's composition layer (see
    /// those modules), which is the correct place for that override since
    /// they do not inherit from this pyclass (see this task's report).
    #[getter]
    fn __array_priority__(&self) -> f64 {
        0.0
    }

    /// `ndarray.__array_namespace__(*, api_version=None)`. Verified
    /// against real numpy 2.5.1: called with no arguments, it returns the
    /// `numpy` PACKAGE object itself (`a.__array_namespace__() is numpy`).
    /// The anionpy equivalent is the `anionpy` package (not the compiled
    /// `_anionpy` extension module `errors::ionp_module` warns against
    /// conflating -- see that function's doc comment). `api_version` is
    /// accepted and, like real numpy for the one version string it
    /// recognizes, not otherwise validated here.
    #[pyo3(signature = (*, api_version=None))]
    fn __array_namespace__(&self, py: Python<'_>, api_version: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let _ = api_version;
        Ok(py.import("anionpy")?.unbind().into_any())
    }

    /// `ndarray.__array_function__(func, types, args, kwargs)`. Real
    /// numpy's own base `ndarray.__array_function__` is "just call
    /// `func(*args, **kwargs)`" because for real numpy, `func` (e.g.
    /// `np.sum`) recursing into that same base `ndarray` implementation a
    /// second time is a no-op fixed point -- there is no OTHER, more
    /// specific `__array_function__` for it to bounce to. That fixed point
    /// does NOT hold for us: `func` here is frequently a real numpy public
    /// dispatcher (`np.sum`, `np.add.reduce`'s caller, etc.) reached
    /// because one of `args` is `self` (an `anionpy.ndarray`, which numpy
    /// cannot handle natively), so `func(*args, **kwargs)` re-invokes
    /// numpy's `__array_function__` protocol dispatch on the very same
    /// arguments, which calls back into THIS method -- infinite recursion
    /// (confirmed live: `np.sum(anionpy_array)` -> `RecursionError: stack
    /// overflow`, `/private/tmp/probe_array_function_recursion.py`).
    /// Fix: never call `func` itself. Resolve the equivalent callable from
    /// anionpy's OWN namespace by name (`func.__name__`, e.g. `"sum"` ->
    /// `anionpy.sum`) and call THAT instead -- anionpy's top-level
    /// functions/`Ufunc`s never re-enter numpy's Python-level dispatcher,
    /// so this terminates. Per the `__array_function__` protocol's own
    /// convention, an unresolvable name returns `NotImplemented` (numpy
    /// then either tries another operand's override or raises
    /// `TypeError` itself -- this method must never raise for "I don't
    /// know this function", only for a genuine error once actually
    /// executing).
    #[pyo3(signature = (func, types, args, kwargs))]
    fn __array_function__<'py>(
        &self,
        py: Python<'py>,
        func: &Bound<'py, PyAny>,
        types: &Bound<'py, PyAny>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = types;
        let name: String = match func.getattr("__name__") {
            Ok(n) => n.extract()?,
            Err(_) => return Ok(py.NotImplemented()),
        };
        let anionpy = py.import("anionpy")?;
        let resolved = match anionpy.getattr(name.as_str()) {
            Ok(r) => r,
            Err(_) => return Ok(py.NotImplemented()),
        };
        Ok(resolved.call(args, kwargs)?.unbind())
    }

    /// `ndarray.__array_ufunc__(ufunc, method, *inputs, **kwargs)`. Same
    /// re-entrancy hazard as `__array_function__` above, and for the same
    /// reason: `ufunc` is frequently a REAL numpy ufunc (`np.add`, etc.)
    /// reached because one operand is `self`, and numpy cannot handle a
    /// foreign operand natively, so `numpy`'s ufunc machinery defers to
    /// this method. Blindly forwarding to `ufunc`/`getattr(ufunc,
    /// method)` re-invokes that same numpy ufunc's `__call__`, which
    /// re-inspects all operands, finds `self` again, and calls this method
    /// again -- infinite recursion (confirmed live: `np.add(numpy_arr,
    /// anionpy_arr)` and `np.add(anionpy_arr, numpy_arr)` both ->
    /// `RecursionError: stack overflow`,
    /// `/private/tmp/probe_mixed_ufunc.py`). Fix: resolve the equivalent
    /// `Ufunc` from anionpy's own namespace by name (`ufunc.__name__`,
    /// e.g. `"add"` -> `anionpy.add`) and dispatch method/`__call__` on
    /// THAT object -- anionpy's own `Ufunc` pyclass (`lib.rs`) never
    /// consults any operand's `__array_ufunc__`, so it is a genuine
    /// terminal step, not a re-entry. Any non-`anionpy.ndarray` input
    /// (e.g. a real numpy operand or a Python scalar) is coerced via
    /// `anionpy.array(...)` first so the resolved anionpy ufunc never
    /// receives an object it doesn't understand; an unresolvable name (or
    /// a method anionpy's `Ufunc` doesn't expose) returns `NotImplemented`
    /// per protocol convention rather than raising.
    #[pyo3(signature = (ufunc, method, *inputs, **kwargs))]
    fn __array_ufunc__<'py>(
        &self,
        py: Python<'py>,
        ufunc: &Bound<'py, PyAny>,
        method: &str,
        inputs: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let name: String = match ufunc.getattr("__name__") {
            Ok(n) => n.extract()?,
            Err(_) => return Ok(py.NotImplemented()),
        };
        let anionpy = py.import("anionpy")?;
        let resolved_ufunc = match anionpy.getattr(name.as_str()) {
            Ok(r) => r,
            Err(_) => return Ok(py.NotImplemented()),
        };
        let callee = if method == "__call__" {
            resolved_ufunc.clone()
        } else {
            match resolved_ufunc.getattr(method) {
                Ok(m) => m,
                Err(_) => return Ok(py.NotImplemented()),
            }
        };
        let array_ctor = anionpy.getattr("array")?;
        let mut coerced: Vec<Py<PyAny>> = Vec::with_capacity(inputs.len());
        for item in inputs.iter() {
            if item.is_instance_of::<PyArray>() {
                coerced.push(item.unbind());
            } else {
                match array_ctor.call1((item.clone(),)) {
                    Ok(converted) => coerced.push(converted.unbind()),
                    // Not everything numpy hands us here is array-like
                    // (e.g. a bare Python scalar in a mixed op) -- pass it
                    // through unconverted and let the resolved anionpy
                    // ufunc's own argument handling decide.
                    Err(_) => coerced.push(item.unbind()),
                }
            }
        }
        let coerced_tuple = pyo3::types::PyTuple::new(py, coerced)?;
        Ok(callee.call(&coerced_tuple, kwargs)?.unbind())
    }

    /// `ndarray.__dlpack_device__()`. Verified against real numpy 2.5.1:
    /// a fixed `(1, 0)` for any array on this machine -- `1` is
    /// `DLDeviceType.kDLCPU` (DLPack's device-type enum), `0` is the
    /// device index, always `0` for CPU. anionpy is CPU-only (no GPU
    /// buffer variant exists anywhere in `ionp-core::Buffer`), so this is
    /// unconditional, not data-dependent.
    fn __dlpack_device__(&self) -> (i32, i32) {
        (1, 0)
    }

    /// `ndarray.__array_interface__`. Verified against real numpy 2.5.1
    /// for the dict shape (`data`, `strides`, `descr`, `typestr`, `shape`,
    /// `version`) and for two structural rules: `strides` is `None`
    /// (never an explicit tuple) whenever the array is C-contiguous, and
    /// is an explicit BYTE-stride tuple otherwise (probed live: a reversed
    /// view and an F-order array both report explicit `strides`, a fresh
    /// C-order array reports `None`) -- `version` is always `3`. `data`'s
    /// second element is the same `_readonly` marker `.flags`'s
    /// `WRITEABLE` getter reads (see `data_ptr_and_readonly`'s doc).
    /// Raises `NotImplementedError` for `S`/`U` (see module doc): declared
    /// and tested only for the other 12 dtypes.
    #[getter]
    fn __array_interface__<'py>(slf: &Bound<'py, PyArray>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let dtype = slf.borrow().inner.dtype();
        let Some((addr, readonly)) = data_ptr_and_readonly(slf)? else {
            return Err(no_flat_buffer_err(dtype));
        };
        let inner_shape: Vec<usize> = slf.borrow().inner.shape().to_vec();
        let is_c_contig = slf.borrow().inner.is_c_contiguous();
        let itemsize = dtype.itemsize() as isize;

        let d = PyDict::new(py);
        d.set_item("data", (addr, readonly))?;
        if is_c_contig {
            d.set_item("strides", py.None())?;
        } else {
            let byte_strides: Vec<isize> =
                slf.borrow().inner.strides().iter().map(|s| s * itemsize).collect();
            d.set_item("strides", PyTuple::new(py, byte_strides)?)?;
        }
        let ts = typestr(dtype);
        d.set_item("typestr", &ts)?;
        d.set_item("descr", vec![("".to_string(), ts)])?;
        d.set_item("shape", PyTuple::new(py, inner_shape)?)?;
        d.set_item("version", 3)?;
        Ok(d)
    }

    /// `ndarray.__array_struct__`. Verified against real numpy 2.5.1:
    /// `type(a.__array_struct__) is PyCapsule`, unnamed (`name=NULL`),
    /// whose `PyCapsule_GetPointer(cap, NULL)` resolves to a
    /// `PyArrayInterface` C struct (`numpy/ndarraytypes.h`) carrying the
    /// same information as `__array_interface__` above in the ABI
    /// consumers of the OLD (pre-`__array_interface__`) protocol expect.
    /// `shape`/`strides` are heap-allocated here (leaked into the struct,
    /// `Box<[isize]>` sized to `nd`) and freed by the capsule's own
    /// destructor when the capsule itself is garbage-collected -- their
    /// lifetime is the capsule's, not `self`'s, exactly as the real
    /// protocol requires (a consumer may outlive the producing array's
    /// current Python-visible instance, though not its buffer -- see
    /// `data_ptr_and_readonly`). Raises `NotImplementedError` for `S`/`U`
    /// (see module doc); declared/tested only for the other 12 dtypes.
    #[getter]
    fn __array_struct__<'py>(slf: &Bound<'py, PyArray>) -> PyResult<Bound<'py, PyCapsule>> {
        let py = slf.py();
        let dtype = slf.borrow().inner.dtype();
        let Some((addr, readonly)) = data_ptr_and_readonly(slf)? else {
            return Err(no_flat_buffer_err(dtype));
        };
        let (shape_vec, strides_vec, nd, is_c, is_f) = {
            let arr = &slf.borrow().inner;
            let itemsize = dtype.itemsize() as isize;
            let shape: Vec<isize> = arr.shape().iter().map(|&s| s as isize).collect();
            let strides: Vec<isize> = arr.strides().iter().map(|s| s * itemsize).collect();
            (shape, strides, arr.ndim(), arr.is_c_contiguous(), arr.is_f_contiguous())
        };

        // NPY_ARRAY_* flag bits (numpy/ndarraytypes.h): CONTIGUOUS=0x1,
        // FORTRAN=0x2, ALIGNED=0x100, NOTSWAPPED=0x200, WRITEABLE=0x400.
        // anionpy has no unaligned/byteswapped representation at all, so
        // ALIGNED and NOTSWAPPED are unconditional.
        let mut flags: c_int = 0x100 | 0x200;
        if is_c {
            flags |= 0x1;
        }
        if is_f {
            flags |= 0x2;
        }
        if !readonly {
            flags |= 0x400;
        }

        // nd==0 (a 0-d array) is a special case: real numpy leaves `shape`
        // and `strides` as NULL in this struct rather than pointing at a
        // zero-length allocation (measured directly -- probing real
        // numpy's own `__array_struct__` on `np.array(7.0)` gives a NULL
        // `strides` pointer, which ctypes surfaces as Python `None` when
        // read back). A `Box<[isize]>` of length 0 in Rust is NOT
        // guaranteed to lower to a null pointer (it's a valid dangling
        // *non-null* pointer per Rust's own allocator contract), so without
        // this branch anionpy's capsule differed from numpy's here even
        // though both correctly describe "no dimensions" -- caught by this
        // task's own differential corpus on the `0d` case, not asserted
        // from first principles.
        let (shape_ptr, strides_ptr): (*mut isize, *mut isize) = if nd == 0 {
            (std::ptr::null_mut(), std::ptr::null_mut())
        } else {
            let shape_box: Box<[isize]> = shape_vec.into_boxed_slice();
            let strides_box: Box<[isize]> = strides_vec.into_boxed_slice();
            (
                Box::into_raw(shape_box) as *mut isize,
                Box::into_raw(strides_box) as *mut isize,
            )
        };
        // Captured as `usize` (not the raw pointer types), purely so the
        // destructor closure below is trivially `Send` -- raw pointers
        // themselves are not `Send`, `usize` is. Cast back to `*mut isize`
        // only inside the destructor, right before freeing.
        let shape_addr = shape_ptr as usize;
        let strides_addr = strides_ptr as usize;

        let value = RawArrayInterface {
            two: 2,
            nd: nd as c_int,
            typekind: dtype.kind_char() as c_char,
            itemsize: dtype.itemsize() as c_int,
            flags,
            shape: shape_ptr,
            strides: strides_ptr,
            data: addr as *mut c_void,
            descr: std::ptr::null_mut(),
        };

        // SAFETY: `nd` is exactly the length both `shape_box`/`strides_box`
        // were allocated with (`Vec::into_boxed_slice` never changes
        // length), so reconstructing each `Box<[isize]>` from
        // `(ptr, nd)` and dropping it here is the exact inverse of the
        // `Box::into_raw` calls above -- no other code ever reads or frees
        // these pointers except through this destructor. When `nd == 0`
        // both pointers are NULL (see above) and nothing was ever
        // allocated, so the null checks below skip the free entirely
        // rather than reconstructing a `Box` from a null pointer.
        let destructor = move |_val: RawArrayInterface, _ctx: *mut c_void| unsafe {
            let shape_ptr = shape_addr as *mut isize;
            let strides_ptr = strides_addr as *mut isize;
            if !shape_ptr.is_null() {
                let _ = Box::from_raw(std::slice::from_raw_parts_mut(shape_ptr, nd) as *mut [isize]);
            }
            if !strides_ptr.is_null() {
                let _ = Box::from_raw(std::slice::from_raw_parts_mut(strides_ptr, nd) as *mut [isize]);
            }
        };

        #[allow(deprecated)]
        let cap = PyCapsule::new_with_destructor(py, value, None, destructor)?;
        Ok(cap)
    }
}
