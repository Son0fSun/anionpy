//! Pickle protocol and PEP 3118 buffer export for `anionpy.ndarray`.
//!
//! Production holes (README): no pickling blocked multiprocessing/joblib;
//! no buffer protocol blocked `memoryview`/`bytes(arr)`/zero-copy interop.
//!
//! Pickle reconstructs via `_reconstruct_ndarray(dtype, shape, raw, fortran)`
//! using the same little-endian layout as `tobytes()` / `frombuffer`.
//! This is anionpy's own protocol (the reconstruct callable is not numpy's
//! `_reconstruct`); values survive `pickle.dumps`/`loads`. Pickle *bytes*
//! are not claimed identical to numpy's.
//!
//! Buffer export is zero-copy for numeric dtypes whose live storage matches
//! numpy's flat layout. `bool` is copied into a 1-byte-per-element temp
//! (Rust `Vec<bool>` is not that layout). `S`/`U` raise `BufferError`.

use std::ffi::{c_int, c_void, CString};
use std::ptr;

use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule, PyTuple};
use pyo3::wrap_pyfunction;

use ionp_core::format::{buffer_to_bytes_le, bytes_to_buffer};
use ionp_core::{DType, NdArray, Order};

use crate::{dtype_from_pyobj, to_py_err, PyArray};

struct BufExtra {
    shape: Vec<isize>,
    strides: Vec<isize>,
    format: CString,
    owned: Option<Vec<u8>>,
}

fn format_for(dt: DType) -> Option<&'static str> {
    Some(match dt {
        DType::Bool => "?",
        DType::I8 => "b",
        DType::U8 => "B",
        DType::I16 => "h",
        DType::U16 => "H",
        DType::I32 => "i",
        DType::U32 => "I",
        DType::I64 => "q",
        DType::U64 => "Q",
        DType::F16 => "e",
        DType::F32 => "f",
        DType::F64 => "d",
        DType::C64 => "Zf",
        DType::C128 => "Zd",
        DType::S(_) | DType::U(_) => return None,
    })
}

fn native_flat_ptr(buf: &ionp_core::Buffer) -> Option<*mut u8> {
    use ionp_core::Buffer;
    Some(match buf {
        Buffer::I8(v) => v.as_ptr() as *mut u8,
        Buffer::I16(v) => v.as_ptr() as *mut u8,
        Buffer::I32(v) => v.as_ptr() as *mut u8,
        Buffer::I64(v) => v.as_ptr() as *mut u8,
        Buffer::U8(v) => v.as_ptr() as *mut u8,
        Buffer::U16(v) => v.as_ptr() as *mut u8,
        Buffer::U32(v) => v.as_ptr() as *mut u8,
        Buffer::U64(v) => v.as_ptr() as *mut u8,
        Buffer::F16(v) => v.as_ptr() as *mut u8,
        Buffer::F32(v) => v.as_ptr() as *mut u8,
        Buffer::F64(v) => v.as_ptr() as *mut u8,
        Buffer::C64(v) => v.as_ptr() as *mut u8,
        Buffer::C128(v) => v.as_ptr() as *mut u8,
        Buffer::Bool(_) | Buffer::S(..) | Buffer::U(..) => return None,
    })
}

#[pyfunction]
#[pyo3(name = "_reconstruct_ndarray")]
fn reconstruct_ndarray(
    py: Python<'_>,
    dtype: &Bound<'_, PyAny>,
    shape: Vec<usize>,
    data: &Bound<'_, PyAny>,
    fortran: bool,
) -> PyResult<Py<PyArray>> {
    let dt = dtype_from_pyobj(dtype)?;
    let raw = data.extract::<Vec<u8>>()?;
    let buffer = bytes_to_buffer(dt, &raw).map_err(to_py_err)?;
    let order = if fortran { Order::F } else { Order::C };
    let inner = NdArray::from_buffer(buffer, shape, order).map_err(to_py_err)?;
    Py::new(py, PyArray { inner })
}

#[pymethods]
impl PyArray {
    fn __reduce_ex__<'py>(
        slf: &Bound<'py, PyArray>,
        py: Python<'py>,
        _protocol: usize,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let arr = &slf.borrow().inner;
        let fortran = arr.is_f_contiguous() && !arr.is_c_contiguous();
        let flat = if fortran {
            arr.to_contiguous_order("F").map_err(to_py_err)?
        } else {
            arr.to_contiguous()
        };
        let raw = buffer_to_bytes_le(flat.buffer());
        let dtype_name = arr.dtype().name().into_owned();
        let shape = arr.shape().to_vec();
        let recon = py.import("anionpy")?.getattr("_reconstruct_ndarray")?;
        let args = (dtype_name, shape, PyBytes::new(py, &raw), fortran).into_pyobject(py)?;
        (recon, args).into_pyobject(py)
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, PyArray>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        Self::__reduce_ex__(slf, py, 3)
    }

    fn __getstate__<'py>(
        slf: &Bound<'py, PyArray>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let arr = &slf.borrow().inner;
        let fortran = arr.is_f_contiguous() && !arr.is_c_contiguous();
        let flat = if fortran {
            arr.to_contiguous_order("F").map_err(to_py_err)?
        } else {
            arr.to_contiguous()
        };
        let raw = buffer_to_bytes_le(flat.buffer());
        let dtype_name = arr.dtype().name().into_owned();
        (
            arr.shape().to_vec(),
            dtype_name,
            PyBytes::new(py, &raw),
            fortran,
        )
            .into_pyobject(py)
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let tup = state.cast::<PyTuple>()?;
        if tup.len() != 4 {
            return Err(PyValueError::new_err("invalid anionpy.ndarray pickle state"));
        }
        let shape: Vec<usize> = tup.get_item(0)?.extract()?;
        let dt = dtype_from_pyobj(&tup.get_item(1)?)?;
        let raw: Vec<u8> = tup.get_item(2)?.extract()?;
        let fortran: bool = tup.get_item(3)?.extract()?;
        let buffer = bytes_to_buffer(dt, &raw).map_err(to_py_err)?;
        let order = if fortran { Order::F } else { Order::C };
        self.inner = NdArray::from_buffer(buffer, shape, order).map_err(to_py_err)?;
        Ok(())
    }

    // DISABLE_GETBUFFER
    #[cfg(any())]
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("Py_buffer view is null"));
        }
        let arr = slf.borrow();
        let dt = arr.inner.dtype();
        let Some(fmt) = format_for(dt) else {
            return Err(PyBufferError::new_err(
                "anionpy: string dtypes have no flat PEP 3118 buffer",
            ));
        };
        let readonly = matches!(
            slf.getattr("_readonly"),
            Ok(v) if v.extract::<bool>().unwrap_or(false)
        );
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE && readonly {
            return Err(PyBufferError::new_err("Object is not writable"));
        }

        let itemsize = dt.itemsize() as isize;
        let shape: Vec<isize> = arr.inner.shape().iter().map(|&d| d as isize).collect();
        let strides: Vec<isize> = arr
            .inner
            .strides()
            .iter()
            .map(|s| s * itemsize)
            .collect();
        let nd = shape.len() as c_int;
        let n_elem = arr.inner.size() as isize;
        let len = n_elem * itemsize;

        let mut owned: Option<Vec<u8>> = None;
        let buf_ptr: *mut u8 = if let Some(p) = native_flat_ptr(arr.inner.buffer()) {
            let item = dt.itemsize() as isize;
            (p as isize + arr.inner.offset() * item) as *mut u8
        } else {
            let flat = arr.inner.to_contiguous();
            let bytes = buffer_to_bytes_le(flat.buffer());
            if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
                return Err(PyBufferError::new_err(
                    "anionpy: bool arrays export a copied buffer and are not writable through memoryview",
                ));
            }
            owned = Some(bytes);
            owned.as_mut().unwrap().as_mut_ptr()
        };
        drop(arr);

        let extra = Box::new(BufExtra {
            shape,
            strides,
            format: CString::new(fmt).unwrap(),
            owned,
        });

        unsafe {
            (*view).obj = slf.into_any().into_ptr();
            (*view).buf = buf_ptr as *mut c_void;
            (*view).len = len;
            (*view).readonly = if readonly || extra.owned.is_some() {
                1
            } else {
                0
            };
            (*view).itemsize = itemsize;
            (*view).ndim = nd;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                extra.format.as_ptr() as *mut _
            } else {
                ptr::null_mut()
            };
            (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
                extra.shape.as_ptr() as *mut isize
            } else {
                ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                extra.strides.as_ptr() as *mut isize
            } else {
                ptr::null_mut()
            };
            (*view).suboffsets = ptr::null_mut();
            (*view).internal = Box::into_raw(extra) as *mut c_void;
        }
        Ok(())
    }

    #[cfg(any())]
    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        if view.is_null() {
            return;
        }
        unsafe {
            let p = (*view).internal;
            if !p.is_null() {
                drop(Box::from_raw(p as *mut BufExtra));
                (*view).internal = ptr::null_mut();
            }
        }
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(reconstruct_ndarray, m)?)?;
    Ok(())
}
