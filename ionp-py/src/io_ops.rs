//! PyO3 bindings for the `.npy`/`.npz` file-I/O toplevel functions
//! (`save`, `load`, `savez`, `savez_compressed`). All the actual byte-level
//! codec work (header parsing/emission, ZIP container, DEFLATE) lives in
//! `ionp_core::format` -- this file only resolves the Python-level `file`
//! argument (a path, `os.PathLike`, or an already-open file-like object)
//! into bytes in and out, and marshals `PyArray`s across the boundary. No
//! numeric loop of any kind belongs here, and none is present.

use std::fs;
use std::path::PathBuf;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use ionp_core::format as fmt;
use ionp_core::NdArray;

use crate::{to_py_err, PyArray};

/// Extracts an `NdArray` from either a real `anionpy.ndarray` or a
/// foreign array-like (numpy array / nested sequence), matching the
/// `extract_or_ingest_ndarray` convention every other binding module in
/// this crate independently keeps a copy of (see `manip.rs`'s own copy
/// and its doc comment for why this is not shared).
fn extract_or_ingest_ndarray(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(pyref.inner.clone());
    }
    crate::array_impl(obj, None)
}

/// Whether `obj` looks like an already-open binary file object (has a
/// `write` method) rather than a path.
fn is_file_like(obj: &Bound<'_, PyAny>) -> bool {
    obj.hasattr("write").unwrap_or(false) || obj.hasattr("read").unwrap_or(false)
}

/// numpy's own behaviour (`np.lib.format` / `np.save`): when `file` is a
/// path (str or `os.PathLike`, not a file object) and its name does not
/// already end with `ext`, numpy appends it before opening.
fn resolve_save_path(file: &Bound<'_, PyAny>, ext: &str) -> PyResult<PathBuf> {
    let raw: PathBuf = file
        .extract()
        .map_err(|_| PyTypeError::new_err("expected a path-like or file-like object"))?;
    let s = raw.to_string_lossy();
    if s.ends_with(ext) {
        Ok(raw)
    } else {
        Ok(PathBuf::from(format!("{}{}", s, ext)))
    }
}

fn write_bytes_to_target(py: Python<'_>, file: &Bound<'_, PyAny>, ext: &str, bytes: &[u8]) -> PyResult<()> {
    if is_file_like(file) {
        let pybytes = PyBytes::new(py, bytes);
        file.call_method1("write", (pybytes,))?;
        Ok(())
    } else {
        let path = resolve_save_path(file, ext)?;
        fs::write(&path, bytes).map_err(|e| {
            PyValueError::new_err(format!("could not write {}: {}", path.display(), e))
        })
    }
}

fn read_bytes_from_source(py: Python<'_>, file: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if is_file_like(file) {
        let data = file.call_method0("read")?;
        let bytes: Vec<u8> = data.extract().map_err(|_| {
            PyTypeError::new_err("file-like object's read() must return bytes for binary .npy/.npz data")
        })?;
        Ok(bytes)
    } else {
        let path: PathBuf = file
            .extract()
            .map_err(|_| PyTypeError::new_err("expected a path-like or file-like object"))?;
        // numpy's np.load, given a bare path with no matching extension,
        // tries the path as given first (it does not auto-append) --
        // unlike save, load never guesses an extension.
        let _ = py; // silence unused warning on some feature configs
        fs::read(&path).map_err(|e| PyValueError::new_err(format!("could not read {}: {}", path.display(), e)))
    }
}

/// `np.save(file, arr, allow_pickle=True, fix_imports=True)`. `allow_pickle`/
/// `fix_imports` are accepted for signature compatibility and otherwise
/// unused: this crate has no object dtype, so the pickle path they gate in
/// real numpy never applies here.
#[pyfunction]
#[pyo3(signature = (file, arr, allow_pickle=true, fix_imports=true))]
fn save(py: Python<'_>, file: &Bound<'_, PyAny>, arr: &Bound<'_, PyAny>, allow_pickle: bool, fix_imports: bool) -> PyResult<()> {
    let _ = (allow_pickle, fix_imports);
    let inner = extract_or_ingest_ndarray(arr)?;
    let bytes = fmt::write_array(&inner);
    write_bytes_to_target(py, file, ".npy", &bytes)
}

/// `np.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True,
/// encoding='ASCII', max_header_size=10000)`. Returns a single `ndarray`
/// for a `.npy` source, or an `NpzFile`-like `dict` (numpy itself makes
/// `NpzFile` lazily-loading and dict-like; this returns an eagerly-loaded
/// plain `dict` of `ndarray`s, which is a strict behavioural subset --
/// every `npz[key]` / `for k in npz` / `npz.files`-style access a plain
/// dict already satisfies works identically) for a `.npz` source.
/// `mmap_mode` is accepted but ignored (no memory-mapping backend in this
/// crate); `allow_pickle`/`fix_imports`/`encoding`/`max_header_size` are
/// likewise accepted for signature compatibility only.
#[pyfunction]
#[pyo3(signature = (file, mmap_mode=None, allow_pickle=false, fix_imports=true, encoding="ASCII".to_string(), max_header_size=10000))]
fn load(
    py: Python<'_>,
    file: &Bound<'_, PyAny>,
    mmap_mode: Option<String>,
    allow_pickle: bool,
    fix_imports: bool,
    encoding: String,
    max_header_size: usize,
) -> PyResult<Py<PyAny>> {
    let _ = (mmap_mode, allow_pickle, fix_imports, encoding, max_header_size);
    let bytes = read_bytes_from_source(py, file)?;
    if bytes.len() >= fmt::MAGIC_LEN && &bytes[..6] == fmt::MAGIC_PREFIX {
        let arr = fmt::read_array(&bytes).map_err(to_py_err)?;
        use pyo3::IntoPyObjectExt;
        return Py::new(py, PyArray { inner: arr })?.into_py_any(py);
    }
    // Not a bare .npy: try it as a .npz (ZIP) container.
    match fmt::read_npz(&bytes) {
        Ok(entries) => {
            let dict = PyDict::new(py);
            for (name, arr) in entries {
                let pyarr = Py::new(py, PyArray { inner: arr })?;
                dict.set_item(name, pyarr)?;
            }
            use pyo3::IntoPyObjectExt;
            dict.into_py_any(py)
        }
        Err(_) => Err(PyValueError::new_err(
            "Failed to interpret file as a pickle or as a valid .npy/.npz archive",
        )),
    }
}

/// Shared implementation for `savez`/`savez_compressed`: positional args
/// are named `arr_0`, `arr_1`, ... (numpy's own convention), keyword args
/// use their keyword as the stored name.
fn savez_impl(
    py: Python<'_>,
    file: &Bound<'_, PyAny>,
    args: Vec<Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
    compressed: bool,
) -> PyResult<()> {
    let mut owned: Vec<(String, NdArray)> = Vec::new();
    for (i, a) in args.iter().enumerate() {
        owned.push((format!("arr_{}", i), extract_or_ingest_ndarray(a)?));
    }
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            let key: String = k.extract()?;
            owned.push((key, extract_or_ingest_ndarray(&v)?));
        }
    }
    let entries: Vec<fmt::NpzEntry> = owned
        .iter()
        .map(|(name, arr)| fmt::NpzEntry { name: name.clone(), array: arr })
        .collect();
    let bytes = if compressed { fmt::write_npz_compressed(&entries) } else { fmt::write_npz(&entries) };
    write_bytes_to_target(py, file, ".npz", &bytes)
}

#[pyfunction]
#[pyo3(signature = (file, *args, **kwargs))]
fn savez(py: Python<'_>, file: &Bound<'_, PyAny>, args: Vec<Bound<'_, PyAny>>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
    savez_impl(py, file, args, kwargs, false)
}

/// `np.savez_compressed`: values round-trip exactly (both `anionpy.load`
/// and real `np.load` read the archive back byte-for-byte correct), but
/// the DEFLATE encoder backing this (`ionp_core::format::deflate`) is
/// hand-rolled and, per its own module doc comment, emits RFC-1951
/// "stored" (uncompressed) blocks only -- a valid DEFLATE stream that
/// achieves essentially no size reduction (measured: a 100-element
/// float64 array's compressed entry is 933 bytes here vs 255 bytes from
/// real numpy's genuine Huffman/LZ77 encoder). The single thing this
/// function's name promises -- that the output is smaller than the
/// uncompressed equivalent -- does not currently hold, so every call
/// warns rather than leaving that gap discoverable only by inspecting
/// file sizes after the fact or reading a ledger comment nobody sees at
/// the call site.
#[pyfunction]
#[pyo3(signature = (file, *args, **kwargs))]
fn savez_compressed(
    py: Python<'_>,
    file: &Bound<'_, PyAny>,
    args: Vec<Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let msg = std::ffi::CString::new(
        "anionpy.savez_compressed: values are exact, but the DEFLATE encoder \
         backing this call emits uncompressed ('stored') blocks only -- the \
         output is a valid .npz archive but is NOT actually compressed \
         (typically larger than savez's plain output for the same arrays). \
         See docs/TICKET-deflate-real-compressor.md."
    ).expect("no interior NUL");
    let warn_cat = py.get_type::<pyo3::exceptions::PyUserWarning>();
    PyErr::warn(py, &warn_cat, msg.as_c_str(), 1)?;
    savez_impl(py, file, args, kwargs, true)
}

/// `np.frombuffer(buffer, dtype=float, count=-1, offset=0)`: reinterprets
/// an existing bytes-like object's raw memory as an array (a VIEW when the
/// source buffer is writable and long-lived; here it always copies, since
/// there is no borrowed-buffer NdArray variant in this crate -- see
/// `ndarray_attrs.rs`'s own `tobytes`/`frombuffer`-adjacent notes for the
/// same constraint elsewhere in this codebase).
#[pyfunction]
#[pyo3(signature = (buffer, dtype=None, count=-1, offset=0))]
fn frombuffer(
    py: Python<'_>,
    buffer: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    count: isize,
    offset: usize,
) -> PyResult<Py<PyArray>> {
    let dt = match dtype {
        Some(d) => crate::dtype_from_pyobj_no_su(d)?,
        None => ionp_core::DType::F64,
    };
    let buf = buffer.extract::<pyo3::buffer::PyBuffer<u8>>()?;
    let all_bytes: Vec<u8> = buf.to_vec(py)?;
    if offset > all_bytes.len() {
        return Err(PyValueError::new_err("offset must be non-negative and no greater than buffer length"));
    }
    let slice = &all_bytes[offset..];
    let itemsize = dt.itemsize();
    if !slice.len().is_multiple_of(itemsize) {
        return Err(PyValueError::new_err("buffer size must be a multiple of element size"));
    }
    let avail = slice.len() / itemsize;
    let n = if count < 0 { avail } else { count as usize };
    if n > avail {
        return Err(PyValueError::new_err("buffer is smaller than requested size"));
    }
    let take = &slice[..n * itemsize];
    let buffer_val = fmt::bytes_to_buffer(dt, take).map_err(to_py_err)?;
    let inner = NdArray::from_buffer(buffer_val, vec![n], ionp_core::Order::C).map_err(to_py_err)?;
    Py::new(py, PyArray { inner })
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(savez, m)?)?;
    m.add_function(wrap_pyfunction!(savez_compressed, m)?)?;
    m.add_function(wrap_pyfunction!(frombuffer, m)?)?;
    Ok(())
}
