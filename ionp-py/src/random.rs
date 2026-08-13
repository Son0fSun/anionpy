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
//! NOT implemented (out of scope for this pass, see the task's final
//! report): every distribution beyond the uniform-double/bounded-integer
//! paths (`.random`/`.integers`/`.bytes`), `.choice`/`.permutation`/
//! `.shuffle`/`.permuted`, MT19937/Philox/SFC64, legacy `RandomState`.
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

use ionp_core::random::{bounded, discrete, distributions, BitGen64, Pcg64, Pcg64Dxsm, SeedSequence};
use ionp_core::{Buffer, DType, NdArray, Order};

use crate::{dtype_from_pyobj, to_py_err, PyArray};

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
}

impl BitGen64 for BitGenKind {
    fn next_u64(&mut self) -> u64 {
        match self {
            BitGenKind::Pcg64(bg) => bg.next_u64(),
            BitGenKind::Pcg64Dxsm(bg) => bg.next_u64(),
        }
    }
    fn next_u32(&mut self) -> u32 {
        match self {
            BitGenKind::Pcg64(bg) => bg.next_u32(),
            BitGenKind::Pcg64Dxsm(bg) => bg.next_u32(),
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
        Err(PyTypeError::new_err("bit_generator must be a PCG64 or PCG64DXSM instance"))
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
    m.add_class::<PyGenerator>()?;
    m.add_function(wrap_pyfunction!(default_rng, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
