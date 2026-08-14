//! `anionpy.random`: numpy-bit-exact random number generation, PyO3 binding
//! layer over `ionp_core::random`.
//!
//! Scope (see `ionp_core::random`'s module docs for the algorithm-level
//! detail): `SeedSequence`, `PCG64`, `PCG64DXSM`, `Generator`,
//! `default_rng`. Every actual draw happens in `ionp_core::random`
//! (`Pcg64`/`Pcg64Dxsm`/`bounded::*`) -- this file only parses Python
//! arguments (seed coercion, `size=`/`dtype=`/`endpoint=`) and marshals
//! `ionp_core` output into `PyArray`/Python scalars. No numeric loop
//! lives here.
//!
//! IMPORTANT, empirically corrected vs the task's original assumption
//! (see `ionp_core::random::bounded`'s test module docs for the full
//! derivation): `np.random.default_rng(seed)` in the installed numpy
//! 2.5.1 instantiates plain `PCG64` (XSL-RR), NOT `PCG64DXSM`. So
//! `default_rng` here binds to `Pcg64`, not `Pcg64Dxsm`.
//!
//! `SFC64` (2026-08-13, this session) is wired the same way as `PCG64`/
//! `PCG64DXSM`: its own `#[pyclass]`, its own `BitGenKind::Sfc64` variant,
//! accepted by `Generator.__init__`. `MT19937`/`Philox` remain NOT
//! implemented -- measured this session (numpy's own C source for both is
//! locally cached and reads as tractable, same `BitGen64` trait shape) but
//! not attempted, a scope/time deferral, not a bit-exactness decline --
//! see the task's final report. Legacy `RandomState` (~45 methods) is also
//! not implemented.
//! `SeedSequence`'s `entropy=` argument only accepts a single Python int
//! (or `None`, which falls back to non-reproducible OS-ish entropy) --
//! numpy's own richer `_coerce_to_uint32_array` also accepts strings and
//! (possibly nested) sequences of ints/strings; that is a documented,
//! deliberate gap, not a silent one.

use std::time::{SystemTime, UNIX_EPOCH};

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyFloat;
use pyo3::IntoPyObjectExt;

use ionp_core::random::{bounded, discrete, distributions, BitGen64, Pcg64, Pcg64Dxsm, SeedSequence, Sfc64};
use ionp_core::{manip, Buffer, DType, NdArray, Order};

use crate::{dtype_from_pyobj, to_py_err, PyArray};

/// Same pattern as `creation.rs`/`manip.rs`/etc.'s own private
/// `extract_or_ingest_ndarray` (duplicated per-file rather than shared,
/// matching this codebase's existing convention): accept either an
/// existing `PyArray` (cloned, cheap -- `NdArray` clone is a buffer `Arc`
/// bump) or ingest a Python sequence/scalar through `array_impl`.
fn extract_or_ingest_ndarray(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(pyref.inner.clone());
    }
    crate::array_impl(obj, None)
}

/// In-place Fisher-Yates shuffle of a `Buffer`'s elements along its outer
/// (only, for the 1-D case this is scoped to) axis, dispatching to
/// `discrete::shuffle_masked` for every concrete element type `Buffer`
/// can hold. `shuffle_masked<T>` has no `Copy`/`Clone` bound (it uses
/// `<[T]>::swap`, not element duplication), so this covers the fixed-width
/// string variants (`S`/`U`, each `Vec<Vec<u8|u32>>`) for free alongside
/// every numeric variant -- no dtype is out of scope for `shuffle` itself
/// (unlike `choice`, which only returns/reindexes, or the continuous
/// distributions, which are numeric-only by nature).
fn shuffle_buffer_masked(bg: &mut dyn BitGen64, buf: &mut Buffer) {
    match buf {
        Buffer::Bool(v) => discrete::shuffle_masked(bg, v),
        Buffer::I8(v) => discrete::shuffle_masked(bg, v),
        Buffer::I16(v) => discrete::shuffle_masked(bg, v),
        Buffer::I32(v) => discrete::shuffle_masked(bg, v),
        Buffer::I64(v) => discrete::shuffle_masked(bg, v),
        Buffer::U8(v) => discrete::shuffle_masked(bg, v),
        Buffer::U16(v) => discrete::shuffle_masked(bg, v),
        Buffer::U32(v) => discrete::shuffle_masked(bg, v),
        Buffer::U64(v) => discrete::shuffle_masked(bg, v),
        Buffer::F16(v) => discrete::shuffle_masked(bg, v),
        Buffer::F32(v) => discrete::shuffle_masked(bg, v),
        Buffer::F64(v) => discrete::shuffle_masked(bg, v),
        Buffer::C64(v) => discrete::shuffle_masked(bg, v),
        Buffer::C128(v) => discrete::shuffle_masked(bg, v),
        Buffer::S(_, v) => discrete::shuffle_masked(bg, v),
        Buffer::U(_, v) => discrete::shuffle_masked(bg, v),
    }
}

// ---------------------------------------------------------------------------
// Seed / entropy coercion.
// ---------------------------------------------------------------------------

/// Non-reproducible entropy for `seed=None`. numpy pulls real OS entropy
/// (`secrets.randbits`) here; since a `None` seed can never be compared
/// byte-for-byte against a separate numpy process anyway (both sides draw
/// different OS entropy), this only needs to produce *some* well-mixed
/// 32-bit words, not match numpy's specific entropy source.
fn os_entropy_words(n: usize) -> Vec<u32> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let mut words = Vec::with_capacity(n);
    for i in 0..n {
        let mut h = RandomState::new().build_hasher();
        h.write_u128(nanos);
        h.write_usize(i);
        words.push(h.finish() as u32);
    }
    words
}

/// Exact port of `_int_to_uint32_array` (`bit_generator.pyx`): splits a
/// non-negative Python int into little-endian 32-bit words, lowest bits
/// first. `n == 0` -> `[0]` (one word), matching numpy's explicit
/// `if n == 0: arr.append(uint32(0))` special case. Uses Python's own
/// arbitrary-precision `int.bit_length()`/`int.to_bytes()` rather than
/// capping at `u128`, so this supports the SAME unbounded seed range
/// numpy does (no scope limitation here, unlike an earlier draft of this
/// task).
fn int_to_uint32_words(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    let repr: String = obj.str()?.extract()?;
    if repr.starts_with('-') {
        return Err(PyValueError::new_err("expected non-negative integer"));
    }
    let bit_length: u64 = obj.call_method0("bit_length")?.extract()?;
    let nbytes = ((bit_length + 7) / 8).max(1) as usize;
    let word_bytes = ((nbytes + 3) / 4) * 4;
    let bytes: Vec<u8> = obj.call_method1("to_bytes", (word_bytes, "little"))?.extract()?;
    let mut words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if words.is_empty() {
        words.push(0);
    }
    Ok(words)
}

/// `SeedSequence(entropy)`/`PCG64(seed)`/`PCG64DXSM(seed)`/`default_rng(seed)`
/// all funnel through here. `None` -> OS-ish entropy (non-reproducible,
/// see `os_entropy_words`).
///
/// CORRECTED against real numpy 2.5.1 (`.venv`, this task's own
/// verification -- an earlier draft of this file assumed the message came
/// from `_coerce_to_uint32_array`'s `'seed must be integer'`, which is
/// WRONG for this call path): `SeedSequence.__init__` has its own,
/// earlier type check before `_coerce_to_uint32_array` is ever reached,
/// and BOTH a float seed and any other non-int, non-sequence object hit
/// it with the exact same message shape:
///   `TypeError: SeedSequence expects int or sequence of ints for entropy
///   not {str(value)}` (verified: `not 2.0`, `not 5.5`, `not abc` for a
///   bad string -- `str()`, not `repr()`, of the value).
/// Anything else that isn't a Python `int` also raises this (numpy
/// additionally accepts strings/sequences here -- documented gap, see
/// module docs).
fn entropy_words_for_seed(seed: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<u32>> {
    match seed {
        None => Ok(os_entropy_words(4)),
        Some(obj) if obj.is_none() => Ok(os_entropy_words(4)),
        Some(obj) => {
            let bad_type = |obj: &Bound<'_, PyAny>| -> PyErr {
                let s = obj.str().map(|s| s.to_string()).unwrap_or_else(|_| "?".to_string());
                PyTypeError::new_err(format!(
                    "SeedSequence expects int or sequence of ints for entropy not {s}"
                ))
            };
            if obj.cast::<PyFloat>().is_ok() {
                return Err(bad_type(obj));
            }
            // `bool` is a Python `int` subclass, same as real numpy.
            if obj.call_method0("__index__").is_err() {
                return Err(bad_type(obj));
            }
            int_to_uint32_words(obj)
        }
    }
}

fn pcg64_from_seed(seed: Option<&Bound<'_, PyAny>>) -> PyResult<Pcg64> {
    let words = entropy_words_for_seed(seed)?;
    let seq = SeedSequence::new(&words, &[], 4);
    let w = seq.generate_state_u64(4);
    Ok(Pcg64::from_seed_words([w[0], w[1], w[2], w[3]]))
}

fn pcg64dxsm_from_seed(seed: Option<&Bound<'_, PyAny>>) -> PyResult<Pcg64Dxsm> {
    let words = entropy_words_for_seed(seed)?;
    let seq = SeedSequence::new(&words, &[], 4);
    let w = seq.generate_state_u64(4);
    Ok(Pcg64Dxsm::from_seed_words([w[0], w[1], w[2], w[3]]))
}

/// `SFC64.__init__`: `self._seed_seq.generate_state(3, np.uint64)`, unlike
/// `PCG64`/`PCG64DXSM`'s 4 words -- confirmed from `_sfc64.pyx` directly
/// (see `ionp_core::random::sfc64`'s module docs), not assumed by analogy.
fn sfc64_from_seed(seed: Option<&Bound<'_, PyAny>>) -> PyResult<Sfc64> {
    let words = entropy_words_for_seed(seed)?;
    let seq = SeedSequence::new(&words, &[], 4);
    let w = seq.generate_state_u64(3);
    Ok(Sfc64::from_seed_words([w[0], w[1], w[2]]))
}

// ---------------------------------------------------------------------------
// `size=` parsing, shared by `SeedSequence.generate_state`/`PCG64.random_raw`/
// `Generator.random`/`Generator.integers`.
// ---------------------------------------------------------------------------

/// `size` accepts a bare non-negative int or a tuple of non-negative
/// ints (`()` is a valid, 0-d shape). Negative entries raise numpy's own
/// `ValueError: negative dimensions are not allowed`.
fn shape_from_size_arg(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let dims: Vec<i64> = if let Ok(n) = obj.extract::<i64>() {
        vec![n]
    } else if let Ok(seq) = obj.extract::<Vec<i64>>() {
        seq
    } else {
        return Err(PyTypeError::new_err("'size' must be an int or tuple of ints"));
    };
    for &d in &dims {
        if d < 0 {
            return Err(PyValueError::new_err("negative dimensions are not allowed"));
        }
    }
    Ok(dims.into_iter().map(|d| d as usize).collect())
}

// ---------------------------------------------------------------------------
// `BitGenKind`: the one concrete backing store `Generator` holds, so a
// `Generator` can be built from either `PCG64` or `PCG64DXSM`.
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum BitGenKind {
    Pcg64(Pcg64),
    Pcg64Dxsm(Pcg64Dxsm),
    Sfc64(Sfc64),
}

impl BitGen64 for BitGenKind {
    fn next_u64(&mut self) -> u64 {
        match self {
            BitGenKind::Pcg64(bg) => bg.next_u64(),
            BitGenKind::Pcg64Dxsm(bg) => bg.next_u64(),
            BitGenKind::Sfc64(bg) => bg.next_u64(),
        }
    }
    fn next_u32(&mut self) -> u32 {
        match self {
            BitGenKind::Pcg64(bg) => bg.next_u32(),
            BitGenKind::Pcg64Dxsm(bg) => bg.next_u32(),
            BitGenKind::Sfc64(bg) => bg.next_u32(),
        }
    }
}

// ---------------------------------------------------------------------------
// `SeedSequence`
// ---------------------------------------------------------------------------

#[pyclass(name = "SeedSequence", module = "anionpy.random")]
struct PySeedSequence {
    inner: SeedSequence,
}

#[pymethods]
impl PySeedSequence {
    #[new]
    #[pyo3(signature = (entropy=None, *, spawn_key=vec![], pool_size=4))]
    fn new(entropy: Option<&Bound<'_, PyAny>>, spawn_key: Vec<u32>, pool_size: usize) -> PyResult<Self> {
        let words = entropy_words_for_seed(entropy)?;
        Ok(Self { inner: SeedSequence::new(&words, &spawn_key, pool_size) })
    }

    #[pyo3(signature = (n_words=1, dtype=None))]
    fn generate_state(
        &self,
        py: Python<'_>,
        n_words: usize,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let dt = match dtype {
            None => DType::U32,
            Some(d) => dtype_from_pyobj(d)?,
        };
        let inner = match dt {
            DType::U32 => {
                NdArray::from_buffer(Buffer::U32(self.inner.generate_state_u32(n_words)), vec![n_words], Order::C)
                    .map_err(to_py_err)?
            }
            DType::U64 => {
                NdArray::from_buffer(Buffer::U64(self.inner.generate_state_u64(n_words)), vec![n_words], Order::C)
                    .map_err(to_py_err)?
            }
            _ => return Err(PyValueError::new_err("dtype must be uint32 or uint64")),
        };
        Py::new(py, PyArray { inner })?.into_py_any(py)
    }
}

// ---------------------------------------------------------------------------
// `PCG64` / `PCG64DXSM`
// ---------------------------------------------------------------------------

#[pyclass(name = "PCG64", module = "anionpy.random", skip_from_py_object)]
#[derive(Clone)]
struct PyPCG64 {
    inner: Pcg64,
}

#[pymethods]
impl PyPCG64 {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self { inner: pcg64_from_seed(seed)? })
    }

    #[pyo3(signature = (size=None))]
    fn random_raw(&mut self, py: Python<'_>, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        match size {
            None => self.inner.next_u64().into_py_any(py),
            Some(obj) => {
                let shape = shape_from_size_arg(obj)?;
                let n: usize = shape.iter().product();
                let data: Vec<u64> = (0..n).map(|_| self.inner.next_u64()).collect();
                let inner = NdArray::from_buffer(Buffer::U64(data), shape, Order::C).map_err(to_py_err)?;
                Py::new(py, PyArray { inner })?.into_py_any(py)
            }
        }
    }
}

#[pyclass(name = "PCG64DXSM", module = "anionpy.random", skip_from_py_object)]
#[derive(Clone)]
struct PyPCG64DXSM {
    inner: Pcg64Dxsm,
}

#[pymethods]
impl PyPCG64DXSM {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self { inner: pcg64dxsm_from_seed(seed)? })
    }

    #[pyo3(signature = (size=None))]
    fn random_raw(&mut self, py: Python<'_>, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        match size {
            None => self.inner.next_u64().into_py_any(py),
            Some(obj) => {
                let shape = shape_from_size_arg(obj)?;
                let n: usize = shape.iter().product();
                let data: Vec<u64> = (0..n).map(|_| self.inner.next_u64()).collect();
                let inner = NdArray::from_buffer(Buffer::U64(data), shape, Order::C).map_err(to_py_err)?;
                Py::new(py, PyArray { inner })?.into_py_any(py)
            }
        }
    }
}

#[pyclass(name = "SFC64", module = "anionpy.random", skip_from_py_object)]
#[derive(Clone)]
struct PySFC64 {
    inner: Sfc64,
}

#[pymethods]
impl PySFC64 {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(Self { inner: sfc64_from_seed(seed)? })
    }

    #[pyo3(signature = (size=None))]
    fn random_raw(&mut self, py: Python<'_>, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        match size {
            None => self.inner.next_u64().into_py_any(py),
            Some(obj) => {
                let shape = shape_from_size_arg(obj)?;
                let n: usize = shape.iter().product();
                let data: Vec<u64> = (0..n).map(|_| self.inner.next_u64()).collect();
                let inner = NdArray::from_buffer(Buffer::U64(data), shape, Order::C).map_err(to_py_err)?;
                Py::new(py, PyArray { inner })?.into_py_any(py)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// `Generator.integers`: bounds-checking + error messages transcribed from
// `numpy/random/_bounded_integers.pyx.in`'s scalar (`_rand_{nptype}`)
// path -- `low`/`high` are plain Python ints, not array_like (array
// low/high is a documented, out-of-scope gap: numpy broadcasts per-
// element bounds via a wholly separate `_rand_{nptype}_broadcast`
// function this port does not implement).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum IntDtype {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl IntDtype {
    fn from_dtype(dt: DType) -> PyResult<Self> {
        Ok(match dt {
            DType::I8 => IntDtype::I8,
            DType::I16 => IntDtype::I16,
            DType::I32 => IntDtype::I32,
            DType::I64 => IntDtype::I64,
            DType::U8 => IntDtype::U8,
            DType::U16 => IntDtype::U16,
            DType::U32 => IntDtype::U32,
            DType::U64 => IntDtype::U64,
            DType::Bool => IntDtype::Bool,
            _ => return Err(PyTypeError::new_err("integers() dtype must be an integer or bool type")),
        })
    }

    /// Inclusive `[lb, ub]`, from `_bounded_integers.pyx.in`'s `type_info`
    /// table (the scalar-path variant, where `ub` is the dtype's own
    /// inclusive maximum, e.g. `0x7FFFFFFF` for `int32` -- NOT the
    /// separate exclusive-style table used by the array-broadcast path).
    fn bounds(self) -> (i128, i128) {
        match self {
            IntDtype::I8 => (-0x80, 0x7F),
            IntDtype::I16 => (-0x8000, 0x7FFF),
            IntDtype::I32 => (-0x8000_0000, 0x7FFF_FFFF),
            IntDtype::I64 => (i64::MIN as i128, i64::MAX as i128),
            IntDtype::U8 => (0, 0xFF),
            IntDtype::U16 => (0, 0xFFFF),
            IntDtype::U32 => (0, 0xFFFF_FFFF),
            IntDtype::U64 => (0, u64::MAX as i128),
            IntDtype::Bool => (0, 1),
        }
    }

    fn name(self) -> &'static str {
        match self {
            IntDtype::I8 => "int8",
            IntDtype::I16 => "int16",
            IntDtype::I32 => "int32",
            IntDtype::I64 => "int64",
            IntDtype::U8 => "uint8",
            IntDtype::U16 => "uint16",
            IntDtype::U32 => "uint32",
            IntDtype::U64 => "uint64",
            IntDtype::Bool => "bool",
        }
    }
}

/// `format_bounds_error(closed, low)` (`_bounded_integers.pyx.in`):
/// `closed` is `endpoint`. Verified message text against real numpy
/// 2.5.1 for the default (`endpoint=False`, `low != 0`) case: exactly
/// `"low >= high"`.
fn format_bounds_error(endpoint: bool, low: i128) -> &'static str {
    if low == 0 {
        if endpoint {
            "high < 0"
        } else {
            "high <= 0"
        }
    } else if endpoint {
        "low > high"
    } else {
        "low >= high"
    }
}

/// Bounds-checks and returns `(low, high_inclusive)` after `endpoint`
/// adjustment, exactly mirroring `_rand_{nptype}`'s scalar-path checks.
fn check_integer_bounds(dt: IntDtype, low: i128, high: i128, endpoint: bool) -> PyResult<(i128, i128)> {
    let (lb, ub) = dt.bounds();
    let high = if endpoint { high } else { high - 1 };
    if low < lb {
        return Err(PyValueError::new_err(format!("low is out of bounds for {}", dt.name())));
    }
    if high > ub {
        return Err(PyValueError::new_err(format!("high is out of bounds for {}", dt.name())));
    }
    if low > high {
        return Err(PyValueError::new_err(format_bounds_error(endpoint, low)));
    }
    Ok((low, high))
}

/// One bounded draw at the given width, dispatching to
/// `ionp_core::random::bounded`'s per-width Lemire functions. `off`/`rng`
/// are already validated in-range for `dt`. Returns the drawn value
/// SIGN-EXTENDED to `i128` (bit-reinterpreted per dtype, matching the
/// unsigned-domain-draw + reinterpret-as-signed shape of numpy's own
/// `<utype>`-then-view-as-`<nptype>` fill loops).
fn draw_one(bg: &mut dyn BitGen64, dt: IntDtype, off: i128, rng: i128, bcnt: &mut i32, buf: &mut u32) -> i128 {
    match dt {
        IntDtype::U64 => bounded::bounded_u64(bg, off as u64, rng as u64) as i128,
        IntDtype::I64 => (bounded::bounded_u64(bg, off as u64, rng as u64) as i64) as i128,
        IntDtype::U32 => bounded::bounded_u32(bg, off as u32, rng as u32) as i128,
        IntDtype::I32 => (bounded::bounded_u32(bg, off as u32, rng as u32) as i32) as i128,
        IntDtype::U16 => bounded::bounded_u16(bg, off as u16, rng as u16, bcnt, buf) as i128,
        IntDtype::I16 => (bounded::bounded_u16(bg, off as u16, rng as u16, bcnt, buf) as i16) as i128,
        IntDtype::U8 => bounded::bounded_u8(bg, off as u8, rng as u8, bcnt, buf) as i128,
        IntDtype::I8 => (bounded::bounded_u8(bg, off as u8, rng as u8, bcnt, buf) as i8) as i128,
        IntDtype::Bool => bounded::bounded_bool(bg, off != 0, rng != 0, bcnt, buf) as i128,
    }
}

fn int_buffer_from_i128s(dt: IntDtype, values: &[i128]) -> Buffer {
    match dt {
        IntDtype::I8 => Buffer::I8(values.iter().map(|&v| v as i8).collect()),
        IntDtype::I16 => Buffer::I16(values.iter().map(|&v| v as i16).collect()),
        IntDtype::I32 => Buffer::I32(values.iter().map(|&v| v as i32).collect()),
        IntDtype::I64 => Buffer::I64(values.iter().map(|&v| v as i64).collect()),
        IntDtype::U8 => Buffer::U8(values.iter().map(|&v| v as u8).collect()),
        IntDtype::U16 => Buffer::U16(values.iter().map(|&v| v as u16).collect()),
        IntDtype::U32 => Buffer::U32(values.iter().map(|&v| v as u32).collect()),
        IntDtype::U64 => Buffer::U64(values.iter().map(|&v| v as u64).collect()),
        IntDtype::Bool => Buffer::Bool(values.iter().map(|&v| v != 0).collect()),
    }
}

/// `Generator.integers(..., size=None)`'s scalar return. numpy's own
/// `_bounded_integers.pyx` `_rand_<dtype>` scalar path returns a REAL,
/// width-typed numpy scalar (`numpy.int8`/`numpy.uint16`/`numpy.int64`/
/// ...) for every dtype EXCEPT `bool`, where it returns a plain Python
/// `bool` -- confirmed directly: `type(np.random.default_rng(5).integers(
/// 0, 100, dtype=np.int8))` is `numpy.int8`, but `type(...,
/// dtype=bool))` is `bool`, not `numpy.bool_`. An earlier version of this
/// function returned a bare Python `int`/`bool` for every dtype, which
/// silently loses width information the differential harness's
/// `np.asarray(x).dtype` check CAN catch (a bare Python int's `asarray`
/// dtype is always `int64`/`uint64`, which only coincidentally matches
/// numpy's own default `dtype=int64` case -- every explicit narrower
/// dtype, e.g. `int16`/`uint32`, diverged: found by an out-of-corpus
/// probe sweeping `dtype=` at `size=None`, a call-form combination the
/// existing corpus never exercised together). Fixed by routing through
/// the same `numpy_scalar_from_0d` mechanism `lib.rs` already uses for
/// exact-type-matching scalar returns elsewhere in the crate: build a
/// real 1-element, 0-d `NdArray` of the target dtype and mint the
/// correctly-typed scalar wrapper from it (real numpy scalar constructor
/// when numpy is importable, anionpy's own lookalike otherwise -- see
/// that function's own doc comment). This does not change what VALUE is
/// computed, only the container type the already-computed value is
/// wrapped in.
fn int_scalar_to_py(py: Python<'_>, dt: IntDtype, v: i128) -> PyResult<Py<PyAny>> {
    if dt == IntDtype::Bool {
        return (v != 0).into_py_any(py);
    }
    let buf = int_buffer_from_i128s(dt, &[v]);
    let arr = NdArray::from_buffer(buf, vec![], Order::C).map_err(to_py_err)?;
    crate::numpy_scalar_from_0d(py, &arr)
}

// ---------------------------------------------------------------------------
// Continuous-distribution argument validation, transcribed from
// `numpy/random/_common.pyx`'s `check_constraint` (see this session's
// research): `CONS_NON_NEGATIVE` raises `f"{name} < 0"` for any non-NaN
// value with the sign bit set (so `-0.0` DOES raise, matching
// `signbit(val)` -- not `val < 0.0`); `CONS_POSITIVE` raises
// `f"{name} <= 0"` for `val <= 0` (NaN compares false, so NaN passes
// through uncaught, matching Cython's `val <= 0` with no separate NaN
// check for plain `CONS_POSITIVE` as opposed to `CONS_POSITIVE_NOT_NAN`).
// ---------------------------------------------------------------------------

fn check_non_negative(name: &str, val: f64) -> PyResult<()> {
    if !val.is_nan() && val.is_sign_negative() {
        return Err(PyValueError::new_err(format!("{name} < 0")));
    }
    Ok(())
}

fn check_positive(name: &str, val: f64) -> PyResult<()> {
    if val <= 0.0 {
        return Err(PyValueError::new_err(format!("{name} <= 0")));
    }
    Ok(())
}

/// Shared scalar-or-`size=`-shaped-array emission for every float64
/// continuous distribution below -- mirrors `random()`'s own `size=None`
/// vs `size=<shape>` branch, just parameterized over the per-draw
/// closure so each distribution method body is a single `emit_f64` call.
fn emit_f64(
    py: Python<'_>,
    bg: &mut dyn BitGen64,
    size: Option<&Bound<'_, PyAny>>,
    mut draw: impl FnMut(&mut dyn BitGen64) -> f64,
) -> PyResult<Py<PyAny>> {
    match size {
        None => draw(bg).into_py_any(py),
        Some(obj) => {
            let shape = shape_from_size_arg(obj)?;
            let n: usize = shape.iter().product();
            let data: Vec<f64> = (0..n).map(|_| draw(bg)).collect();
            let inner = NdArray::from_buffer(Buffer::F64(data), shape, Order::C).map_err(to_py_err)?;
            Py::new(py, PyArray { inner })?.into_py_any(py)
        }
    }
}

/// Shared scalar-or-`size=`-shaped-array emission for every int64-output
/// discrete distribution below (`poisson`/`binomial`/`negative_binomial`/
/// `geometric`/`zipf`/`logseries`/`hypergeometric`). numpy's own `disc()`
/// dispatcher (`_generator.pyx`) always allocates `np.int64` for these,
/// scalar path included -- so the scalar return routes through
/// `int_scalar_to_py(IntDtype::I64, ...)` for exact-type-matching
/// (`numpy.int64`, not a bare Python `int`), same rationale as
/// `int_scalar_to_py`'s own doc comment.
fn emit_i64(
    py: Python<'_>,
    bg: &mut dyn BitGen64,
    size: Option<&Bound<'_, PyAny>>,
    mut draw: impl FnMut(&mut dyn BitGen64) -> i64,
) -> PyResult<Py<PyAny>> {
    match size {
        None => int_scalar_to_py(py, IntDtype::I64, draw(bg) as i128),
        Some(obj) => {
            let shape = shape_from_size_arg(obj)?;
            let n: usize = shape.iter().product();
            let data: Vec<i64> = (0..n).map(|_| draw(bg)).collect();
            let inner = NdArray::from_buffer(Buffer::I64(data), shape, Order::C).map_err(to_py_err)?;
            Py::new(py, PyArray { inner })?.into_py_any(py)
        }
    }
}

// ---------------------------------------------------------------------------
// Discrete-distribution argument validation, transcribed from
// `numpy/random/_common.pyx`'s `check_constraint` for the constraint
// kinds the continuous-only `check_non_negative`/`check_positive` above
// don't cover.
// ---------------------------------------------------------------------------

/// `CONS_BOUNDED_0_1`: `f"{name} < 0, {name} > 1 or {name} is NaN"`.
fn check_bounded_0_1(name: &str, val: f64) -> PyResult<()> {
    if !(val >= 0.0) || !(val <= 1.0) {
        return Err(PyValueError::new_err(format!("{name} < 0, {name} > 1 or {name} is NaN")));
    }
    Ok(())
}

/// `CONS_BOUNDED_GT_0_1`: `f"{name} <= 0, {name} > 1 or {name} contains NaNs"`.
fn check_bounded_gt_0_1(name: &str, val: f64) -> PyResult<()> {
    if !(val > 0.0) || !(val <= 1.0) {
        return Err(PyValueError::new_err(format!("{name} <= 0, {name} > 1 or {name} contains NaNs")));
    }
    Ok(())
}

/// `CONS_BOUNDED_LT_0_1`: `f"{name} < 0, {name} >= 1 or {name} is NaN"`.
fn check_bounded_lt_0_1(name: &str, val: f64) -> PyResult<()> {
    if !(val >= 0.0) || !(val < 1.0) {
        return Err(PyValueError::new_err(format!("{name} < 0, {name} >= 1 or {name} is NaN")));
    }
    Ok(())
}

/// `CONS_GT_1`: `f"{name} <= 1 or {name} is NaN"`.
fn check_gt_1(name: &str, val: f64) -> PyResult<()> {
    if !(val > 1.0) {
        return Err(PyValueError::new_err(format!("{name} <= 1 or {name} is NaN")));
    }
    Ok(())
}

/// `CONS_POSITIVE_NOT_NAN`: NaN raises `f"{name} must not be NaN"`
/// (checked FIRST, so this differs from plain `CONS_POSITIVE` which lets
/// NaN pass through); otherwise `val <= 0` raises `f"{name} <= 0"`.
fn check_positive_not_nan(name: &str, val: f64) -> PyResult<()> {
    if val.is_nan() {
        return Err(PyValueError::new_err(format!("{name} must not be NaN")));
    }
    if val <= 0.0 {
        return Err(PyValueError::new_err(format!("{name} <= 0")));
    }
    Ok(())
}

/// `CONS_POISSON` (`_common.pyx`): `val < 0` (or NaN) raises
/// `f"{name} < 0 or {name} is NaN"`; otherwise `val > POISSON_LAM_MAX`
/// raises `f"{name} value too large"`. `POISSON_LAM_MAX` (`distributions.h`,
/// non-legacy `Generator` bound) is `int64::MAX - sqrt(int64::MAX) * 10`.
fn poisson_lam_max() -> f64 {
    (i64::MAX as f64) - (i64::MAX as f64).sqrt() * 10.0
}

fn check_poisson(name: &str, val: f64) -> PyResult<()> {
    if !(val >= 0.0) {
        return Err(PyValueError::new_err(format!("{name} < 0 or {name} is NaN")));
    }
    if val > poisson_lam_max() {
        return Err(PyValueError::new_err(format!("{name} value too large")));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `Generator`
// ---------------------------------------------------------------------------

#[pyclass(name = "Generator", module = "anionpy.random")]
struct PyGenerator {
    bg: BitGenKind,
}

#[pymethods]
impl PyGenerator {
    /// numpy's `Generator.__init__(bit_generator)` shares the SAME
    /// BitGenerator object between the two (mutating one advances the
    /// other's state too). This binding CLONES the bit generator's state
    /// instead -- a documented gap: `g = anionpy.random.Generator(bg); g2 =
    /// anionpy.random.Generator(bg)` will NOT observe each other's draws the
    /// way real numpy's would. `default_rng()` (below) does not go
    /// through this aliasing path at all and is unaffected.
    #[new]
    fn new(bit_generator: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(bg) = bit_generator.extract::<PyRef<'_, PyPCG64>>() {
            return Ok(Self { bg: BitGenKind::Pcg64(bg.inner.clone()) });
        }
        if let Ok(bg) = bit_generator.extract::<PyRef<'_, PyPCG64DXSM>>() {
            return Ok(Self { bg: BitGenKind::Pcg64Dxsm(bg.inner.clone()) });
        }
        if let Ok(bg) = bit_generator.extract::<PyRef<'_, PySFC64>>() {
            return Ok(Self { bg: BitGenKind::Sfc64(bg.inner.clone()) });
        }
        Err(PyTypeError::new_err("bit_generator must be a PCG64, PCG64DXSM, or SFC64 instance"))
    }

    /// `Generator.random(size=None, dtype=np.float64, out=None)`. `out=`
    /// is a documented, untested gap (not wired up).
    #[pyo3(signature = (size=None, dtype=None))]
    fn random(&mut self, py: Python<'_>, size: Option<&Bound<'_, PyAny>>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let dt = match dtype {
            None => DType::F64,
            Some(d) => dtype_from_pyobj(d)?,
        };
        if !matches!(dt, DType::F64 | DType::F32) {
            return Err(PyTypeError::new_err("random() dtype must be float64 or float32"));
        }
        match size {
            None => match dt {
                DType::F64 => (self.bg.next_f64()).into_py_any(py),
                _ => (self.bg.next_f32()).into_py_any(py),
            },
            Some(obj) => {
                let shape = shape_from_size_arg(obj)?;
                let n: usize = shape.iter().product();
                let buffer = match dt {
                    DType::F64 => Buffer::F64((0..n).map(|_| self.bg.next_f64()).collect()),
                    _ => Buffer::F32((0..n).map(|_| self.bg.next_f32()).collect()),
                };
                let inner = NdArray::from_buffer(buffer, shape, Order::C).map_err(to_py_err)?;
                Py::new(py, PyArray { inner })?.into_py_any(py)
            }
        }
    }

    /// `Generator.integers(low, high=None, size=None, dtype=np.int64,
    /// endpoint=False)`. `low`/`high` must be plain Python ints (array_like
    /// bounds are a documented, out-of-scope gap -- see this section's
    /// module-level doc comment).
    #[pyo3(signature = (low, high=None, size=None, dtype=None, endpoint=false))]
    #[allow(clippy::too_many_arguments)]
    fn integers(
        &mut self,
        py: Python<'_>,
        low: i128,
        high: Option<i128>,
        size: Option<&Bound<'_, PyAny>>,
        dtype: Option<&Bound<'_, PyAny>>,
        endpoint: bool,
    ) -> PyResult<Py<PyAny>> {
        let dt = match dtype {
            None => IntDtype::I64,
            Some(d) => IntDtype::from_dtype(dtype_from_pyobj(d)?)?,
        };
        // `high=None` means `low` was actually the (exclusive) high bound
        // and the true low is 0, matching numpy's own call-form.
        let (lo, hi) = match high {
            None => (0i128, low),
            Some(h) => (low, h),
        };
        // Resolve `size=` to a concrete shape BEFORE validating `low`/`high`
        // against `dt`'s range: numpy's own `_bounded_integers.pyx`
        // `_rand_<dtype>` fill functions early-return on `cnt == 0` before
        // ever touching the range/bounds computation, so a `size=0` (or any
        // shape with a zero dimension) draw is UNCONDITIONALLY exempt from
        // bounds/`low>=high` validation -- confirmed directly:
        // `np.random.default_rng(1).integers(500, 300, size=0,
        // dtype=np.int8)` returns `array([], dtype=int8)`, no error, despite
        // `low > high` AND both being out of `int8`'s range. `size=None`
        // (the scalar path) always draws exactly one value and so is never
        // exempt. Found by an out-of-corpus probe explicitly sweeping
        // `size=0` against out-of-range `high` values, a combination the
        // existing corpus's `size=0` cases (all `dtype=None`, in-range
        // bounds) never crossed with an out-of-range explicit dtype.
        let shape = match size {
            None => None,
            Some(obj) => Some(shape_from_size_arg(obj)?),
        };
        let n_if_sized = shape.as_ref().map(|s| s.iter().product::<usize>());
        if n_if_sized != Some(0) {
            let (lo2, hi_incl2) = check_integer_bounds(dt, lo, hi, endpoint)?;
            let off = lo2;
            let rng = hi_incl2 - lo2;
            let mut bcnt = 0i32;
            let mut buf = 0u32;
            match shape {
                None => {
                    let v = draw_one(&mut self.bg, dt, off, rng, &mut bcnt, &mut buf);
                    return int_scalar_to_py(py, dt, v);
                }
                Some(shape) => {
                    let n: usize = shape.iter().product();
                    let values: Vec<i128> = (0..n)
                        .map(|_| draw_one(&mut self.bg, dt, off, rng, &mut bcnt, &mut buf))
                        .collect();
                    let inner = NdArray::from_buffer(int_buffer_from_i128s(dt, &values), shape, Order::C)
                        .map_err(to_py_err)?;
                    return Py::new(py, PyArray { inner })?.into_py_any(py);
                }
            }
        }
        // `n_if_sized == Some(0)`: bounds/`low>=high` deliberately NOT
        // validated (see comment above) -- emit the empty array directly.
        let shape = shape.expect("n_if_sized is Some only when shape is Some");
        let inner = NdArray::from_buffer(int_buffer_from_i128s(dt, &[]), shape, Order::C).map_err(to_py_err)?;
        Py::new(py, PyArray { inner })?.into_py_any(py)
    }

    /// `Generator.bytes(length)`: numpy implements this in terms of
    /// `self.integers(0, 2**32, size=ceil(length/4), dtype=uint32)`,
    /// viewed little-endian and truncated to `length` bytes
    /// (`_generator.pyx`'s `Generator.bytes`).
    fn bytes(&mut self, py: Python<'_>, length: usize) -> PyResult<Py<PyAny>> {
        let n_words = length.div_ceil(4);
        let mut out = Vec::with_capacity(n_words * 4);
        for _ in 0..n_words {
            // `rng == 0xFFFFFFFF` (2**32 - 1, i.e. `integers(0, 2**32)`'s
            // exclusive-high-1 inclusive range) bypasses Lemire entirely
            // in `bounded_u32` and draws a raw `next_uint32` -- exactly
            // `random_bounded_uint32_fill`'s own fast path.
            let word = bounded::bounded_u32(&mut self.bg, 0, 0xFFFF_FFFF);
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.truncate(length);
        pyo3::types::PyBytes::new(py, &out).into_py_any(py)
    }

    // -----------------------------------------------------------------
    // Continuous distributions layered on `ionp_core::random::distributions`.
    // `dtype=`/`out=`/array_like-parameter broadcasting are documented,
    // out-of-scope gaps (same shape as `random()`'s own `out=` gap
    // above) -- only scalar-float parameters + scalar-or-`size=` output
    // are wired up. `method='zig'` is the only ziggurat variant
    // implemented (numpy's legacy inverse-CDF `method='inv'` path for
    // `standard_exponential`/`standard_normal` is NOT ported).
    // -----------------------------------------------------------------

    #[pyo3(signature = (size=None))]
    fn standard_normal(&mut self, py: Python<'_>, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        emit_f64(py, &mut self.bg, size, distributions::standard_normal)
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn normal(&mut self, py: Python<'_>, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_non_negative("scale", scale)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::normal(bg, loc, scale))
    }

    #[pyo3(signature = (size=None))]
    fn standard_exponential(&mut self, py: Python<'_>, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        emit_f64(py, &mut self.bg, size, distributions::standard_exponential)
    }

    #[pyo3(signature = (scale=1.0, size=None))]
    fn exponential(&mut self, py: Python<'_>, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_non_negative("scale", scale)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::exponential(bg, scale))
    }

    #[pyo3(signature = (shape, size=None))]
    fn standard_gamma(&mut self, py: Python<'_>, shape: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_non_negative("shape", shape)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::standard_gamma(bg, shape))
    }

    #[pyo3(signature = (shape, scale=1.0, size=None))]
    fn gamma(
        &mut self,
        py: Python<'_>,
        shape: f64,
        scale: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        check_non_negative("shape", shape)?;
        check_non_negative("scale", scale)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::gamma(bg, shape, scale))
    }

    #[pyo3(signature = (a, b, size=None))]
    fn beta(&mut self, py: Python<'_>, a: f64, b: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_positive("a", a)?;
        check_positive("b", b)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::beta(bg, a, b))
    }

    #[pyo3(signature = (df, size=None))]
    fn chisquare(&mut self, py: Python<'_>, df: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_positive("df", df)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::chisquare(bg, df))
    }

    #[pyo3(signature = (dfnum, dfden, size=None))]
    fn f(&mut self, py: Python<'_>, dfnum: f64, dfden: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_positive("dfnum", dfnum)?;
        check_positive("dfden", dfden)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::f(bg, dfnum, dfden))
    }

    /// `Generator.uniform(low=0.0, high=1.0, size=None)`. The
    /// `OverflowError('high - low range exceeds valid bounds')` path
    /// (non-finite `high - low`) IS ported since it's a plain scalar
    /// check; the array_like `low`/`high` broadcasting path is not.
    #[pyo3(signature = (low=0.0, high=1.0, size=None))]
    fn uniform(&mut self, py: Python<'_>, low: f64, high: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        let range = high - low;
        if !range.is_finite() {
            return Err(pyo3::exceptions::PyOverflowError::new_err("high - low range exceeds valid bounds"));
        }
        check_non_negative("high - low", range)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::uniform(bg, low, range))
    }

    #[pyo3(signature = (size=None))]
    fn standard_cauchy(&mut self, py: Python<'_>, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        emit_f64(py, &mut self.bg, size, distributions::standard_cauchy)
    }

    #[pyo3(signature = (df, size=None))]
    fn standard_t(&mut self, py: Python<'_>, df: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_positive("df", df)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::standard_t(bg, df))
    }

    #[pyo3(signature = (a, size=None))]
    fn pareto(&mut self, py: Python<'_>, a: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_positive("a", a)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::pareto(bg, a))
    }

    #[pyo3(signature = (a, size=None))]
    fn weibull(&mut self, py: Python<'_>, a: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_non_negative("a", a)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::weibull(bg, a))
    }

    #[pyo3(signature = (a, size=None))]
    fn power(&mut self, py: Python<'_>, a: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_positive("a", a)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::power(bg, a))
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn laplace(&mut self, py: Python<'_>, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_non_negative("scale", scale)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::laplace(bg, loc, scale))
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn gumbel(&mut self, py: Python<'_>, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_non_negative("scale", scale)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::gumbel(bg, loc, scale))
    }

    #[pyo3(signature = (loc=0.0, scale=1.0, size=None))]
    fn logistic(&mut self, py: Python<'_>, loc: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_non_negative("scale", scale)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::logistic(bg, loc, scale))
    }

    #[pyo3(signature = (mean=0.0, sigma=1.0, size=None))]
    fn lognormal(&mut self, py: Python<'_>, mean: f64, sigma: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_non_negative("sigma", sigma)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::lognormal(bg, mean, sigma))
    }

    #[pyo3(signature = (scale=1.0, size=None))]
    fn rayleigh(&mut self, py: Python<'_>, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_non_negative("scale", scale)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::rayleigh(bg, scale))
    }

    #[pyo3(signature = (mean, scale, size=None))]
    fn wald(&mut self, py: Python<'_>, mean: f64, scale: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_positive("mean", mean)?;
        check_positive("scale", scale)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::wald(bg, mean, scale))
    }

    #[pyo3(signature = (mu, kappa, size=None))]
    fn vonmises(&mut self, py: Python<'_>, mu: f64, kappa: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_non_negative("kappa", kappa)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::vonmises(bg, mu, kappa))
    }

    /// `Generator.triangular(left, mode, right, size=None)`: order of
    /// checks matters (`left > mode` before `mode > right` before
    /// `left == right`), matching `_generator.pyx`'s scalar path exactly.
    #[pyo3(signature = (left, mode, right, size=None))]
    fn triangular(
        &mut self,
        py: Python<'_>,
        left: f64,
        mode: f64,
        right: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if left > mode {
            return Err(PyValueError::new_err("left > mode"));
        }
        if mode > right {
            return Err(PyValueError::new_err("mode > right"));
        }
        if left == right {
            return Err(PyValueError::new_err("left == right"));
        }
        emit_f64(py, &mut self.bg, size, |bg| distributions::triangular(bg, left, mode, right))
    }

    #[pyo3(signature = (df, nonc, size=None))]
    fn noncentral_chisquare(
        &mut self,
        py: Python<'_>,
        df: f64,
        nonc: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        check_positive("df", df)?;
        check_non_negative("nonc", nonc)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::noncentral_chisquare(bg, df, nonc))
    }

    #[pyo3(signature = (dfnum, dfden, nonc, size=None))]
    fn noncentral_f(
        &mut self,
        py: Python<'_>,
        dfnum: f64,
        dfden: f64,
        nonc: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        check_positive("dfnum", dfnum)?;
        check_positive("dfden", dfden)?;
        check_non_negative("nonc", nonc)?;
        emit_f64(py, &mut self.bg, size, |bg| distributions::noncentral_f(bg, dfnum, dfden, nonc))
    }

    #[pyo3(signature = (lam=1.0, size=None))]
    fn poisson(&mut self, py: Python<'_>, lam: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_poisson("lam", lam)?;
        emit_i64(py, &mut self.bg, size, |bg| discrete::poisson(bg, lam))
    }

    /// `Generator.binomial(n, p, size=None)`. `n` is a Python float/int
    /// truncated to `int64` via a C-style `<int64_t>n` cast (Rust's `as`
    /// truncation toward zero matches this for finite inputs); `p` is
    /// checked BEFORE `n`, matching `_generator.pyx`'s own order.
    #[pyo3(signature = (n, p, size=None))]
    fn binomial(&mut self, py: Python<'_>, n: f64, p: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_bounded_0_1("p", p)?;
        let n_i = n as i64;
        check_non_negative("n", n_i as f64)?;
        emit_i64(py, &mut self.bg, size, |bg| discrete::binomial(bg, p, n_i))
    }

    /// `Generator.negative_binomial(n, p, size=None)`: after the base
    /// `CONS_POSITIVE_NOT_NAN`/`CONS_BOUNDED_GT_0_1` checks, numpy also
    /// rejects `(n, p)` combinations whose derived `poisson` intermediate
    /// `lam` (mean + 10 sigma of the internal gamma draw) would exceed
    /// `POISSON_LAM_MAX`, since `negative_binomial` is implemented as a
    /// gamma-then-poisson mixture.
    #[pyo3(signature = (n, p, size=None))]
    fn negative_binomial(&mut self, py: Python<'_>, n: f64, p: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_positive_not_nan("n", n)?;
        check_bounded_gt_0_1("p", p)?;
        let max_lam = (1.0 - p) / p * (n + 10.0 * n.sqrt());
        if max_lam > poisson_lam_max() {
            return Err(PyValueError::new_err(
                "n too large or p too small, see Generator.negative_binomial Notes",
            ));
        }
        emit_i64(py, &mut self.bg, size, |bg| discrete::negative_binomial(bg, n, p))
    }

    #[pyo3(signature = (p, size=None))]
    fn geometric(&mut self, py: Python<'_>, p: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_bounded_gt_0_1("p", p)?;
        emit_i64(py, &mut self.bg, size, |bg| discrete::geometric(bg, p))
    }

    #[pyo3(signature = (a, size=None))]
    fn zipf(&mut self, py: Python<'_>, a: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_gt_1("a", a)?;
        emit_i64(py, &mut self.bg, size, |bg| discrete::zipf(bg, a))
    }

    #[pyo3(signature = (p, size=None))]
    fn logseries(&mut self, py: Python<'_>, p: f64, size: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_bounded_lt_0_1("p", p)?;
        emit_i64(py, &mut self.bg, size, |bg| discrete::logseries(bg, p))
    }

    /// `Generator.hypergeometric(ngood, nbad, nsample, size=None)`: the
    /// `HYPERGEOM_MAX = 10**9` and `ngood + nbad < nsample` checks run
    /// BEFORE the per-arg `CONS_NON_NEGATIVE` checks (matching
    /// `_generator.pyx`'s scalar path, which computes `lngood`/`lnbad`/
    /// `lnsample` via C-style truncating casts first).
    #[pyo3(signature = (ngood, nbad, nsample, size=None))]
    fn hypergeometric(
        &mut self,
        py: Python<'_>,
        ngood: f64,
        nbad: f64,
        nsample: f64,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        const HYPERGEOM_MAX: i64 = 1_000_000_000;
        let lngood = ngood as i64;
        let lnbad = nbad as i64;
        let lnsample = nsample as i64;
        if lngood >= HYPERGEOM_MAX || lnbad >= HYPERGEOM_MAX {
            return Err(PyValueError::new_err(format!(
                "both ngood and nbad must be less than {HYPERGEOM_MAX}"
            )));
        }
        if lngood + lnbad < lnsample {
            return Err(PyValueError::new_err("ngood + nbad < nsample"));
        }
        check_non_negative("ngood", lngood as f64)?;
        check_non_negative("nbad", lnbad as f64)?;
        check_non_negative("nsample", lnsample as f64)?;
        emit_i64(py, &mut self.bg, size, |bg| discrete::hypergeometric(bg, lngood, lnbad, lnsample))
    }

    /// `Generator.dirichlet(alpha, size=None)`, scoped to the common
    /// non-broadcast call form: `alpha` a 1-D sequence of floats, `size`
    /// `None`/int/tuple-of-int (matches `linalg.svd`'s already-accepted
    /// precedent of declaring a non-batched subset -- see
    /// `anionpy/_state/linalg.py`). Output shape `(k,)` if `size is None`
    /// else `size + (k,)` (as a tuple), matching `_generator.pyx` exactly.
    ///
    /// Constraint: `alpha < 0` anywhere raises `ValueError('alpha < 0')`
    /// (transcribed verbatim -- this is NOT `CONS_POSITIVE`; zero entries
    /// are allowed, unlike `beta`'s own scalar `a`/`b`).
    ///
    /// Algorithm selection (`alpha.iter().max() < 0.1`) is computed ONCE
    /// per call, matching numpy's own control flow (see
    /// `distributions::dirichlet_sample`'s doc comment) -- NOT per draw.
    #[pyo3(signature = (alpha, size=None))]
    fn dirichlet(
        &mut self,
        py: Python<'_>,
        alpha: Vec<f64>,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let k = alpha.len();
        if alpha.iter().any(|&a| a < 0.0) {
            return Err(PyValueError::new_err("alpha < 0"));
        }
        let use_small_alpha = k > 0
            && alpha.iter().cloned().fold(f64::MIN, f64::max) < distributions::DIRICHLET_SMALL_ALPHA_THRESHOLD;

        let mut out_shape: Vec<usize> = match size {
            None => vec![],
            Some(obj) => shape_from_size_arg(obj)?,
        };
        out_shape.push(k);
        let total_draws: usize = out_shape[..out_shape.len() - 1].iter().product();

        let mut data = vec![0.0f64; total_draws * k];
        // k == 0 (empty `alpha`): numpy returns an all-empty array of shape
        // `size + (0,)` without drawing anything -- `chunks_mut(0)` panics
        // ("chunk size must be non-zero"), so skip the draw loop entirely
        // rather than attempt zero-width chunking.
        if k > 0 {
            for chunk in data.chunks_mut(k) {
                distributions::dirichlet_sample(&mut self.bg, &alpha, use_small_alpha, chunk);
            }
        }
        let inner = NdArray::from_buffer(Buffer::F64(data), out_shape, Order::C).map_err(to_py_err)?;
        Py::new(py, PyArray { inner })?.into_py_any(py)
    }

    /// `Generator.multinomial(n, pvals, size=None)`, scoped to the common
    /// non-broadcast call form: `n` a scalar int, `pvals` a 1-D sequence
    /// of floats, `size` `None`/int/tuple-of-int (`n` as array-like /
    /// `pvals` with ndim > 1 broadcasting is the documented-but-
    /// unsupported form, same scoping precedent as `dirichlet` above).
    /// Output shape `(d,)` if `size is None` else `size + (d,)`.
    ///
    /// Constraint order (transcribed from `_generator.pyx`): (1) `pvals`
    /// elementwise `CONS_BOUNDED_0_1` (`"pvals < 0, pvals > 1 or pvals is
    /// NaN"` per-element, `_check_array_cons_bounded_0_1`'s scalar-style
    /// message, not the array-style `CONS_NON_NEGATIVE` message), (2)
    /// `kahan_sum(pvals[:-1]) > 1.0 + 1e-12` -> `ValueError("sum(pvals[:-1])
    /// > 1.0")`, (3) `n` `CONS_NON_NEGATIVE` (`"n < 0"`). The loop trip
    /// count is `total_output_size / d` and is data-dependent (`size`) --
    /// see `discrete::multinomial`'s own doc comment for the per-draw
    /// algorithm, unchanged here.
    #[pyo3(signature = (n, pvals, size=None))]
    fn multinomial(
        &mut self,
        py: Python<'_>,
        n: i64,
        pvals: Vec<f64>,
        size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let d = pvals.len();
        if d == 0 {
            return Err(PyValueError::new_err(
                "pvals must have at least 1 dimension and the last dimension of pvals must be greater than 0.",
            ));
        }
        for &p in &pvals {
            if !(p >= 0.0) || !(p <= 1.0) {
                // Array-path CONS_BOUNDED_0_1 message text differs from the
                // scalar path's ("is NaN" vs "contains NaNs") --
                // `_check_array_cons_bounded_0_1` in `_common.pyx`.
                return Err(PyValueError::new_err("pvals < 0, pvals > 1 or pvals contains NaNs"));
            }
        }
        // Kahan-compensated sum of pvals[..d-1], matching `_common.pyx`'s
        // `kahan_sum` bit-for-bit (see `distributions.rs`'s FMA-contraction
        // lesson -- this loop has none: every step is already a separate,
        // non-fusable C expression in the original).
        if d > 1 {
            let mut sum = pvals[0];
            let mut c = 0.0f64;
            for &x in &pvals[1..d - 1] {
                let y = x - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            if sum > 1.0 + 1e-12 {
                return Err(PyValueError::new_err("sum(pvals[:-1]) > 1.0"));
            }
        }
        check_non_negative("n", n as f64)?;

        let mut out_shape: Vec<usize> = match size {
            None => vec![],
            Some(obj) => shape_from_size_arg(obj)?,
        };
        out_shape.push(d);
        let total_draws: usize = out_shape[..out_shape.len() - 1].iter().product();

        let mut data = vec![0i64; total_draws * d];
        for chunk in data.chunks_mut(d) {
            discrete::multinomial(&mut self.bg, n, &pvals, chunk);
        }
        let inner = NdArray::from_buffer(Buffer::I64(data), out_shape, Order::C).map_err(to_py_err)?;
        Py::new(py, PyArray { inner })?.into_py_any(py)
    }

    /// `Generator.multivariate_hypergeometric(colors, nsample, size=None,
    /// method='marginals')`, scoped to `method='marginals'` (numpy's
    /// DEFAULT) ONLY -- `method='count'` is a real, numpy-successful,
    /// STATISTICALLY DIFFERENT algorithm (temp array of size
    /// `sum(colors)`, distinct draw order) that this binding does not
    /// implement; rather than silently return marginals-shaped output for
    /// a caller who explicitly asked for 'count' (a silent wrong-answer,
    /// the exact failure mode this whole task exists to prevent), an
    /// explicit `method='count'` raises `NotImplementedError` loudly. A
    /// genuinely invalid method string still raises numpy's own exact
    /// `ValueError` message, since that check is free and matches
    /// real numpy's own validation ORDER (method checked before nsample,
    /// before colors -- transcribed from `_generator.pyx`).
    ///
    /// Constraint order (transcribed from `_generator.pyx`): (1) `method`
    /// membership, (2) `nsample` `CONS_NON_NEGATIVE`
    /// (`"nsample must be nonnegative."`), (3) `colors` 1-D nonnegative
    /// `int64`-range check, (4) `sum(colors)` overflow check, (5)
    /// `method == 'marginals'` -> `sum(colors) < 1_000_000_000` check
    /// (the ONLY total-size ceiling that applies to this binding's
    /// scoped-in method), (6) `nsample > sum(colors)` check. See
    /// `discrete::multivariate_hypergeometric_marginals`'s own doc
    /// comment for the per-draw algorithm (built on the existing
    /// `discrete::hypergeometric`, already bit-exact-verified).
    #[pyo3(signature = (colors, nsample, size=None, method=None))]
    fn multivariate_hypergeometric(
        &mut self,
        py: Python<'_>,
        colors: Vec<i64>,
        nsample: i64,
        size: Option<&Bound<'_, PyAny>>,
        method: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        match method {
            None | Some("marginals") => {}
            Some("count") => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "multivariate_hypergeometric(method='count') is not implemented; only the default method='marginals' is scoped in",
                ));
            }
            Some(_) => {
                return Err(PyValueError::new_err("method must be \"count\" or \"marginals\"."));
            }
        }
        if nsample < 0 {
            return Err(PyValueError::new_err("nsample must be nonnegative."));
        }

        let num_colors = colors.len();
        if colors.iter().any(|&c| c < 0) {
            return Err(PyValueError::new_err(
                "colors must be a one-dimensional sequence of nonnegative integers not exceeding 9223372036854775807.",
            ));
        }
        let mut total: i64 = 0;
        for &c in &colors {
            total = total.checked_add(c).ok_or_else(|| {
                PyValueError::new_err(
                    "sum(colors) must not exceed the maximum value of a 64 bit signed integer (9223372036854775807)",
                )
            })?;
        }
        if total >= 1_000_000_000 {
            return Err(PyValueError::new_err(
                "When method is \"marginals\", sum(colors) must be less than 1000000000.",
            ));
        }
        if nsample > total {
            return Err(PyValueError::new_err("nsample > sum(colors)"));
        }

        let mut out_shape: Vec<usize> = match size {
            None => vec![],
            Some(obj) => shape_from_size_arg(obj)?,
        };
        out_shape.push(num_colors);
        let total_draws: usize = out_shape[..out_shape.len() - 1].iter().product();

        let mut data = vec![0i64; total_draws * num_colors];
        // num_colors == 0: numpy returns an all-empty array of shape
        // `size + (0,)` without drawing anything -- same empty-chunk
        // panic hazard `dirichlet`'s k=0 fix already addressed, guarded
        // the same way here.
        if num_colors > 0 {
            for chunk in data.chunks_mut(num_colors) {
                discrete::multivariate_hypergeometric_marginals(&mut self.bg, total, &colors, nsample, chunk);
            }
        }
        let inner = NdArray::from_buffer(Buffer::I64(data), out_shape, Order::C).map_err(to_py_err)?;
        Py::new(py, PyArray { inner })?.into_py_any(py)
    }

    // -----------------------------------------------------------------
    // `shuffle`/`permutation`/`permuted`/`choice`: the index/sequence
    // family built on `discrete::shuffle_masked`/`shuffle_lemire`/
    // `choice_no_replace_no_p` (see those functions' own doc comments in
    // `ionp-core/src/random/discrete.rs` for the numpy-source-derived
    // algorithm detail, and `bounded.rs`'s module doc for why
    // `shuffle`/`permutation`/`permuted` draw via the MASKED primitive
    // while `choice`'s own no-replace path draws via Lemire -- two
    // genuinely distinct bit-stream consumers, not a stylistic choice).
    //
    // Scoped IN this pass: 1-D `anionpy.ndarray`/int input for all four;
    // `choice`'s `replace=True` (both `p=None` and `p` given) and
    // `replace=False, p=None` branches.
    //
    // Scoped OUT, loudly (`NotImplementedError`, never a silent
    // wrong answer): any ndim != 1 input to `shuffle`/`permutation`
    // (numpy shuffles/permutes along `axis` by swapping whole
    // cross-sections there, a different algorithm this pass didn't
    // transcribe); `permuted(axis=...)` and `permuted(out=...)` (only
    // `axis=None`, no `out`, is scoped in -- flatten, shuffle the flat
    // buffer via a fresh C-contiguous copy, keep the original shape
    // metadata); `choice`'s `replace=False, p=...` branch (numpy's own
    // algorithm there is an iterative `np.unique`+`searchsorted`
    // rejection loop whose bit-exactness this pass did not have time to
    // verify). Declining these is a deliberate, documented decision, not
    // an oversight -- see the coordinator's own framing: a decline is a
    // success, not a shortfall.
    // -----------------------------------------------------------------

    /// `Generator.shuffle(x, axis=0)`, in-place. `x` must be an
    /// `anionpy.ndarray` (a raw Python list/foreign array has no storage
    /// this could mutate through -- same established precedent as
    /// `put`/`put_along_axis` in `manip.rs`).
    #[pyo3(signature = (x, axis=0))]
    fn shuffle(&mut self, x: &Bound<'_, PyAny>, axis: isize) -> PyResult<()> {
        let bound = x
            .cast::<PyArray>()
            .map_err(|_| PyTypeError::new_err("shuffle() requires an anionpy.ndarray as its in-place mutation target"))?;
        let mut pyref = bound.borrow_mut();
        if pyref.inner.ndim() != 1 {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Generator.shuffle() on an array with ndim != 1 is not implemented; only whole 1-D shuffling is scoped in",
            ));
        }
        // 1-D only, so the only legal `axis` values are 0/-1 -- this both
        // validates that and matches numpy's own `axis` bounds-check
        // message shape for a bad value.
        manip::normalize_axis(axis, 1).map_err(to_py_err)?;
        shuffle_buffer_masked(&mut self.bg, pyref.inner.buffer_mut());
        Ok(())
    }

    /// `Generator.permutation(x, axis=0)`. `x` as a non-negative Python
    /// int draws `shuffle(arange(x))` (matching numpy's own int branch:
    /// `arr = arange(x); self.shuffle(arr); return arr`, so it uses this
    /// SAME masked primitive, not `choice`'s Lemire one). `x` as a 1-D
    /// `anionpy.ndarray`/sequence returns a shuffled COPY -- the input is
    /// left untouched, unlike `shuffle`.
    #[pyo3(signature = (x, axis=0))]
    fn permutation(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, axis: isize) -> PyResult<Py<PyAny>> {
        if let Ok(n) = x.extract::<i64>() {
            if n < 0 {
                return Err(PyValueError::new_err("x must be a non-negative integer"));
            }
            let mut idx: Vec<i64> = (0..n).collect();
            discrete::shuffle_masked(&mut self.bg, &mut idx);
            let inner = NdArray::from_buffer(Buffer::I64(idx), vec![n as usize], Order::C).map_err(to_py_err)?;
            return Py::new(py, PyArray { inner })?.into_py_any(py);
        }
        let arr = extract_or_ingest_ndarray(x)?;
        if arr.ndim() != 1 {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Generator.permutation() on an array with ndim != 1 is not implemented; only 1-D permutation is scoped in",
            ));
        }
        manip::normalize_axis(axis, 1).map_err(to_py_err)?;
        // `to_contiguous_order` always gathers into a FRESH, uniquely-owned
        // buffer (`array.rs`'s `gather_by_perm`), never aliasing `arr`'s own
        // storage -- exactly the "copy, then shuffle in place" numpy does.
        let mut copy = arr.to_contiguous_order("C").map_err(to_py_err)?;
        shuffle_buffer_masked(&mut self.bg, copy.buffer_mut());
        Py::new(py, PyArray { inner: copy })?.into_py_any(py)
    }

    /// `Generator.permuted(x, axis=None, out=None)`, scoped to
    /// `axis=None` (numpy flattens `x`, shuffles the flat sequence, and
    /// reshapes back to `x`'s original shape) with no `out=` support.
    /// `axis=k` independently shuffles every slice perpendicular to
    /// `axis` -- a real, distinct algorithm this pass didn't have time to
    /// transcribe and verify, so it is declined loudly.
    #[pyo3(signature = (x, axis=None, out=None))]
    fn permuted(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        axis: Option<isize>,
        out: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if axis.is_some() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Generator.permuted(axis=...) is not implemented; only axis=None (flatten + shuffle + reshape) is scoped in",
            ));
        }
        if out.is_some() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Generator.permuted(out=...) is not implemented",
            ));
        }
        let arr = extract_or_ingest_ndarray(x)?;
        // Fresh, uniquely-owned C-contiguous buffer (same `gather_by_perm`
        // guarantee `permutation` relies on above) with `arr`'s ORIGINAL
        // shape preserved -- shuffling this buffer's raw storage in place
        // is exactly "flatten, shuffle the flat sequence, reshape back",
        // since the flat storage IS the flattened sequence and the shape
        // metadata is untouched.
        let mut flat = arr.to_contiguous_order("C").map_err(to_py_err)?;
        shuffle_buffer_masked(&mut self.bg, flat.buffer_mut());
        Py::new(py, PyArray { inner: flat })?.into_py_any(py)
    }

    /// `Generator.choice(a, size=None, replace=True, p=None, axis=0,
    /// shuffle=True)`. `a` as a non-negative int or a 1-D
    /// `anionpy.ndarray`/sequence; N-D `a` is out of scope (see the
    /// family doc comment above `shuffle`). `replace=False, p=...` is
    /// out of scope and raises loudly -- see the family doc comment.
    ///
    /// `p`'s validation (1-D length match, Kahan-summed
    /// NaN/non-negative/sums-to-1-within-`atol` checks) mirrors
    /// `multinomial`'s own Kahan-sum block above, with one real
    /// difference: `multinomial` sums only `pvals[..d-1]` (the last
    /// probability there is implied), while `choice`'s `kahan_sum` sums
    /// ALL `d` probabilities (`_generator.pyx`'s `p_sum = kahan_sum(pix,
    /// d)`) -- transcribed faithfully, not copy-pasted. One documented
    /// simplification: numpy widens `atol` when `p`'s ORIGINAL array
    /// dtype is a lower-precision float (`max(atol,
    /// sqrt(finfo(p.dtype).eps))`); since `p` is ingested here as a plain
    /// `Vec<f64>` with no source-dtype tracking, that widening is NOT
    /// applied -- `atol` is always `sqrt(f64::EPSILON)`. A caller who
    /// passes a float32 `p` deliberately right at the wider-but-not-the-
    /// narrower tolerance boundary would see a stricter accept/reject
    /// than real numpy; this is a real, narrow, documented gap, not
    /// silent handling.
    #[pyo3(signature = (a, size=None, replace=true, p=None, axis=0, shuffle=true))]
    #[allow(clippy::too_many_arguments)]
    fn choice(
        &mut self,
        py: Python<'_>,
        a: &Bound<'_, PyAny>,
        size: Option<&Bound<'_, PyAny>>,
        replace: bool,
        p: Option<&Bound<'_, PyAny>>,
        axis: isize,
        shuffle: bool,
    ) -> PyResult<Py<PyAny>> {
        let is_scalar = size.is_none();
        let out_shape: Vec<usize> = match size {
            None => vec![],
            Some(obj) => shape_from_size_arg(obj)?,
        };
        let total_size: i64 = if is_scalar { 1 } else { out_shape.iter().product::<usize>() as i64 };

        let (pop_size, a_is_int, a_arr): (i64, bool, Option<NdArray>) = if let Ok(v) = a.extract::<i64>() {
            if v <= 0 && total_size != 0 {
                return Err(PyValueError::new_err("a must be a positive integer unless no samples are taken"));
            }
            (v, true, None)
        } else {
            let arr = extract_or_ingest_ndarray(a)?;
            if arr.ndim() != 1 {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Generator.choice() with an 'a' of ndim != 1 is not implemented; only int or 1-D 'a' is scoped in",
                ));
            }
            let ps = arr.shape()[0] as i64;
            if ps == 0 && total_size != 0 {
                return Err(PyValueError::new_err("a cannot be empty unless no samples are taken"));
            }
            (ps, false, Some(arr))
        };

        let p_vec: Option<Vec<f64>> = match p {
            None => None,
            Some(obj) if obj.is_none() => None,
            Some(obj) => Some(
                obj.extract::<Vec<f64>>()
                    .map_err(|_| PyTypeError::new_err("p must be a 1-dimensional sequence of floats"))?,
            ),
        };
        if let Some(ref pv) = p_vec {
            if pv.len() as i64 != pop_size {
                return Err(PyValueError::new_err("a and p must have same size"));
            }
            let atol = f64::EPSILON.sqrt();
            // `kahan_sum` (`_common.pyx`): `sum = darr[0]; c = 0.0; for i in
            // 1..n: y = darr[i]-c; t = sum+y; c = (t-sum)-y; sum = t` --
            // over ALL `d` entries (unlike `multinomial`'s truncated sum
            // above), returning `0.0` for `d == 0`.
            let mut sum = pv.first().copied().unwrap_or(0.0);
            let mut c = 0.0f64;
            for &x in pv.iter().skip(1) {
                let y = x - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            let p_sum = sum;
            if p_sum.is_nan() {
                return Err(PyValueError::new_err("Probabilities contain NaN"));
            }
            if pv.iter().any(|&x| x < 0.0) {
                return Err(PyValueError::new_err("Probabilities are not non-negative"));
            }
            if (p_sum - 1.0).abs() > atol {
                return Err(PyValueError::new_err(
                    "Probabilities do not sum to 1. See Notes section of docstring for more information.",
                ));
            }
        }

        let n_draws: usize = if is_scalar { 1 } else { total_size as usize };
        let idx: Vec<i64> = if replace {
            match &p_vec {
                Some(pv) => {
                    // Plain (non-Kahan) running cumsum, matching numpy's
                    // own `p.cumsum()` -- only the initial validation sum
                    // above is Kahan-compensated.
                    let mut cdf: Vec<f64> = Vec::with_capacity(pv.len());
                    let mut running = 0.0f64;
                    for &x in pv {
                        running += x;
                        cdf.push(running);
                    }
                    let last = *cdf.last().unwrap_or(&1.0);
                    for v in cdf.iter_mut() {
                        *v /= last;
                    }
                    (0..n_draws)
                        .map(|_| {
                            let u = self.bg.next_f64();
                            // `searchsorted(..., side='right')`: first index
                            // where `cdf[i] > u`.
                            cdf.partition_point(|&c| c <= u) as i64
                        })
                        .collect()
                }
                None => (0..n_draws)
                    .map(|_| bounded::bounded_u64(&mut self.bg, 0, (pop_size - 1) as u64) as i64)
                    .collect(),
            }
        } else {
            if total_size > pop_size {
                return Err(PyValueError::new_err(
                    "Cannot take a larger sample than population when replace is False",
                ));
            }
            if p_vec.is_some() {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Generator.choice(replace=False, p=...) is not implemented; see the family doc comment above 'shuffle' for why this path is deliberately declined",
                ));
            }
            discrete::choice_no_replace_no_p(&mut self.bg, pop_size, total_size, shuffle)
        };

        let idx_shape: Vec<usize> = if is_scalar { vec![] } else { out_shape.clone() };
        let idx_arr = NdArray::from_buffer(Buffer::I64(idx.clone()), idx_shape, Order::C).map_err(to_py_err)?;

        if a_is_int {
            if is_scalar {
                // numpy: "When passing a as an integer type and size is not
                // specified, the return type is a native Python int" --
                // NOT a numpy scalar, unlike every other branch here.
                return idx[0].into_py_any(py);
            }
            return Py::new(py, PyArray { inner: idx_arr })?.into_py_any(py);
        }

        let arr = a_arr.expect("a_arr is Some whenever a_is_int is false");
        let ax = manip::normalize_axis(axis, arr.ndim()).map_err(to_py_err)?;
        let taken = manip::take(&arr, &idx_arr, Some(ax), manip::ClipMode::Raise).map_err(to_py_err)?;
        if is_scalar {
            return crate::numpy_scalar_from_0d(py, &taken);
        }
        Py::new(py, PyArray { inner: taken })?.into_py_any(py)
    }
}

#[pyfunction]
#[pyo3(signature = (seed=None))]
fn default_rng(seed: Option<&Bound<'_, PyAny>>) -> PyResult<PyGenerator> {
    Ok(PyGenerator { bg: BitGenKind::Pcg64(pcg64_from_seed(seed)?) })
}

/// `anionpy.random` as a genuine dotted submodule (`PyModule::new(py,
/// "random")` + `add_submodule`, the same pattern `linalg.rs`/`fft.rs`
/// use), not a flat re-export -- `random.SeedSequence`/`random.PCG64`/
/// `random.PCG64DXSM`/`random.Generator`/`random.default_rng` are already
/// naturally dotted in numpy's own namespace.
pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "random")?;
    m.add_class::<PySeedSequence>()?;
    m.add_class::<PyPCG64>()?;
    m.add_class::<PyPCG64DXSM>()?;
    m.add_class::<PySFC64>()?;
    m.add_class::<PyGenerator>()?;
    m.add_function(wrap_pyfunction!(default_rng, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
