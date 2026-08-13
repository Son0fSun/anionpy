//! PyO3 bindings for numpy's dtype introspection/promotion toplevel
//! functions: `can_cast`, `promote_types`, `result_type`, `min_scalar_type`,
//! `finfo`(partial -- see module doc below), `iinfo`, `typename`,
//! `mintypecode`. `issubdtype`/`isdtype`/`common_type` are NOT registered
//! here at all -- see the bottom of this file for why (scalar-type-class
//! dependency owned by a concurrently-running agent, per this task's
//! mission).
//!
//! `promote_types`/`result_type` are NOT the same function (mission's own
//! framing, confirmed against real numpy 2.5.1): `promote_types(a, b)` is
//! exactly two dtypes, value-independent, pure `ionp_core::dtype::
//! promote_dtype`. `result_type(*args)` mixes arrays/dtypes ("strong") with
//! bare Python scalars ("weak", NEP 50) in one variadic call -- weak
//! scalars fold in via `weak_target_dtype` AFTER all strong operands have
//! been combined via `promote_dtype`, never before, which is what makes
//! `result_type(int8_array, 300)` stay `int8` (300's magnitude is
//! irrelevant to a *weak* scalar) while `min_scalar_type(300)` (a
//! standalone, explicitly *value-based* function) is `int16`.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use ionp_core::dtype::{
    can_cast as core_can_cast, finfo_for, iinfo_for, min_scalar_type_complex,
    min_scalar_type_float, min_scalar_type_signed, min_scalar_type_unsigned, mintypecode_select,
    promote_dtype, typename_for,
};
use ionp_core::{weak_target_dtype, Buffer, DType, ScalarKind};

use crate::{classify_scalar, dtype_from_pyobj, dtype_name_to_dtype, extract_array_like, PyArray, PyDType};

/// Resolve any dtype-LIKE argument (never a bare weak scalar -- callers
/// that need to distinguish weak scalars check `classify_scalar` first)
/// into a `DType`: an anionpy `ndarray` (its own `.dtype()`, no Python
/// round-trip needed), anything with a `.dtype` attribute (numpy arrays,
/// numpy scalars, anionpy's own `PyDType` via its `.dtype` numpy-interop
/// getter -- all three expose `.dtype.name` once you're inside this
/// branch), a bare name string, or -- last resort -- real numpy's own
/// `np.dtype(obj).name` coercion (covers numpy scalar TYPE objects like
/// `np.int8`, and bare `np.dtype(...)` instances, which have no `.dtype`
/// attribute themselves).
fn coerce_dtype_like(obj: &Bound<'_, PyAny>) -> PyResult<DType> {
    // NOTE on `None`: real numpy treats a bare `None` differently in
    // different top-level functions -- `np.result_type(None)` silently
    // defaults to float64 (numpy's historical "no dtype given" sentinel,
    // `np.dtype(None) == np.dtype('float64')`), but `np.promote_types(None,
    // 'int8')` explicitly REJECTS it (`TypeError: did not understand one
    // of the types`). This is a genuine, function-specific divergence
    // discovered via differential probing (a first-pass fix here rejected
    // `None` unconditionally, which was correct for `promote_types` but
    // WRONG for `result_type` -- it would have turned a real numpy success
    // into a spurious anionpy error). Deliberately NOT special-cased in this
    // shared helper for that reason; `promote_types`/`result_type` each
    // handle their own `None` semantics before ever calling this function.
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        return Ok(pyref.inner.dtype());
    }
    if let Ok(s) = obj.extract::<String>() {
        return dtype_name_to_dtype(&s);
    }
    // `bytes` is dtype-name-like too, real numpy decodes it exactly like a
    // `str` before the name lookup (`np.dtype(b'int8') == np.dtype('int8')`,
    // live-verified against numpy 2.5.1) -- distinct from `dtype_from_pyobj`'s
    // own last-resort numpy round-trip, which does not have this case at all
    // (that function is shared by `anionpy.array(..., dtype=...)` and other
    // call sites this file does not own; the `bytes`-as-name coercion is
    // handled here, narrowly, for the dtype-introspection functions only).
    if obj.is_instance_of::<pyo3::types::PyBytes>() {
        if let Ok(b) = obj.extract::<Vec<u8>>() {
            let s = String::from_utf8_lossy(&b).into_owned();
            return dtype_name_to_dtype(&s);
        }
    }
    if obj.hasattr("dtype")? {
        let d = obj.getattr("dtype")?;
        if let Ok(name) = d.getattr("name").and_then(|n| n.extract::<String>()) {
            return dtype_name_to_dtype(&name);
        }
    }
    dtype_from_pyobj(obj)
}

// ---------------------------------------------------------------------------
// can_cast
// ---------------------------------------------------------------------------

const CAN_CAST_SCALAR_MSG: &str = "can_cast() does not support Python ints, floats, and complex because the result used to depend on the value.\nThis change was part of adopting NEP 50, we may explicitly allow them again in the future.";
const DTYPE_NOT_UNDERSTOOD_MSG: &str =
    "did not understand one of the types; 'None' not accepted";

/// `can_cast`'s own dtype-like coercion is stricter than the shared
/// `coerce_dtype_like` above: verified against real numpy 2.5.1, `can_cast`
/// rejects `None` and any bare Python `bool`/`int`/`float`/`complex` value
/// with two DIFFERENT, fixed exception messages (the NEP 50 scalar message
/// for scalars; a generic "did not understand one of the types" message --
/// literally that fixed string, verbatim, for EVERY other unrecognized
/// input including `None` itself, an unknown string, or a garbage object --
/// this is real numpy's actual message text, not a paraphrase; it always
/// says "'None'" even when the offending value was `'xyz'` or `[1, 2]`,
/// confirmed by direct probing, not assumed).
fn coerce_can_cast_dtype(obj: &Bound<'_, PyAny>) -> PyResult<DType> {
    if obj.is_none() {
        return Err(PyTypeError::new_err(DTYPE_NOT_UNDERSTOOD_MSG));
    }
    if classify_scalar(obj).is_some() {
        return Err(PyTypeError::new_err(CAN_CAST_SCALAR_MSG));
    }
    coerce_dtype_like(obj).map_err(|_| PyTypeError::new_err(DTYPE_NOT_UNDERSTOOD_MSG))
}

/// numpy's `PyArray_DescrConverter` recognizes an empty `list`/`dict` as a
/// zero-field structured/void dtype spec (`np.dtype([]) == np.dtype({}) ==
/// dtype([])`, kind `'V'`, itemsize 0) -- anionpy's `DType` enum has no
/// void/structured variant at all (an architectural absence of the same
/// KIND as `min_scalar_type`'s missing object dtype, see that function's
/// own doc comment; adding one would ripple into every exhaustive match
/// over `DType` across the crate, not a targeted fix), so this narrow
/// helper recognizes exactly the two void-dtype SPELLINGS numpy accepts
/// without ever constructing/representing the void dtype itself --
/// `can_cast` only needs a yes/no answer about it, never the value.
/// Deliberately does NOT recognize a non-empty list of field tuples (a
/// REAL structured dtype with fields) -- that is genuine structured-dtype
/// parsing, a much larger feature anionpy does not implement, and
/// misrecognizing it here would silently claim support this function does
/// not have.
fn is_void_dtype_spec(obj: &Bound<'_, PyAny>) -> bool {
    if let Ok(list) = obj.cast::<pyo3::types::PyList>() {
        return list.is_empty();
    }
    if let Ok(dict) = obj.cast::<pyo3::types::PyDict>() {
        return dict.is_empty();
    }
    false
}

#[pyfunction]
#[pyo3(signature = (from_, to, casting="safe"))]
fn can_cast(from_: &Bound<'_, PyAny>, to: &Bound<'_, PyAny>, casting: &str) -> PyResult<bool> {
    if !matches!(casting, "no" | "equiv" | "safe" | "same_kind" | "unsafe") {
        return Err(PyValueError::new_err(format!(
            "casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '{casting}')"
        )));
    }
    let from_is_void = is_void_dtype_spec(from_);
    let to_is_void = is_void_dtype_spec(to);
    // Verified against real numpy 2.5.1 across all five `casting=` values
    // and both operand orders (`np.can_cast(dtype, [], casting=r)` /
    // `np.can_cast([], dtype, casting=r)` for every concrete dtype and
    // every rule): void<->void (both spellings, either order) is always
    // castable (identical dtype); a concrete dtype casting INTO void is
    // castable ONLY under `'unsafe'`; void casting OUT to a concrete dtype
    // is never castable under any rule (there is no way to interpret a
    // zero-field struct's -- nonexistent -- bytes as a scalar of any real
    // dtype, not even losslessly-anything's-allowed `'unsafe'`).
    if from_is_void && to_is_void {
        return Ok(true);
    }
    if from_is_void {
        return Ok(false);
    }
    if to_is_void {
        // Still validate `from_` the normal way first (an invalid `from_`
        // must raise its own error before casting-rule logic ever runs --
        // verified above: `can_cast(None, [], 'unsafe')` raises the same
        // "did not understand" error a bad `from_` would against any
        // concrete `to`, not a silent `False`).
        coerce_can_cast_dtype(from_)?;
        return Ok(casting == "unsafe");
    }
    let f = coerce_can_cast_dtype(from_)?;
    let t = coerce_can_cast_dtype(to)?;
    Ok(core_can_cast(f, t, casting))
}

// ---------------------------------------------------------------------------
// promote_types / result_type
// ---------------------------------------------------------------------------

fn cannot_interpret_err(obj: &Bound<'_, PyAny>) -> PyErr {
    let text = obj.str().map(|s| s.to_string()).unwrap_or_else(|_| "?".to_string());
    PyTypeError::new_err(format!("Cannot interpret '{text}' as a data type"))
}

/// The error to raise for a `coerce_dtype_like` failure that is NOT the
/// `None` case (each caller below handles `None` itself, before ever
/// calling `coerce_dtype_like`, since real numpy's `None` handling is
/// function-specific -- see `coerce_dtype_like`'s doc comment). A bad
/// dtype-NAME string gets its own distinct numpy message ("data type 'x'
/// not understood") separate from the generic non-string-non-dtype-like
/// message ("Cannot interpret 'x' as a data type") -- verified live
/// against real numpy 2.5.1 for both `promote_types` and `result_type`,
/// which share this exact wording.
fn dtype_like_err(obj: &Bound<'_, PyAny>) -> PyErr {
    if let Ok(s) = obj.extract::<String>() {
        return PyTypeError::new_err(format!("data type '{s}' not understood"));
    }
    // `bytes`: real numpy decodes it exactly like a `str` before the
    // name-not-understood message too (`np.dtype(b'x')` ->
    // `"data type 'x' not understood"`, same wording as the `str` case
    // above, live-verified against numpy 2.5.1) -- this arm only fires when
    // `coerce_dtype_like`'s own bytes-decoding attempt above already failed
    // to resolve a real dtype name, so `s` here is definitely not valid.
    if obj.is_instance_of::<pyo3::types::PyBytes>() {
        if let Ok(b) = obj.extract::<Vec<u8>>() {
            let s = String::from_utf8_lossy(&b).into_owned();
            return PyTypeError::new_err(format!("data type '{s}' not understood"));
        }
    }
    // A bare tuple (not a dtype-like object, not already handled above):
    // real numpy's dtype coercion treats ANY 2-tuple as an attempted
    // `(base_dtype, shape_or_field)` sub-array/field spec, so a tuple of
    // any OTHER length fails a length check with its own distinct message
    // before ever reaching the generic "Cannot interpret" fallback --
    // live-verified against numpy 2.5.1: `np.dtype(())` ->
    // `"Tuple must have size 2, but has size 0"`; a genuine 2-tuple (e.g.
    // `('int8', (2, 2))`) is a real, still-unimplemented sub-array dtype
    // spec, out of scope here, and correctly falls through to the generic
    // message below instead of being misreported as a length error.
    if let Ok(tup) = obj.extract::<Vec<Bound<'_, PyAny>>>() {
        if obj.is_instance_of::<PyTuple>() && tup.len() != 2 {
            return PyTypeError::new_err(format!(
                "Tuple must have size 2, but has size {}",
                tup.len()
            ));
        }
    }
    cannot_interpret_err(obj)
}

#[pyfunction]
fn promote_types(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyDType> {
    // `np.promote_types(None, ...)` explicitly rejects `None` (unlike
    // `result_type`, which treats it as the float64 default -- see
    // `result_type` below and `coerce_dtype_like`'s doc comment).
    if a.is_none() || b.is_none() {
        return Err(PyTypeError::new_err("did not understand one of the types"));
    }
    let da = coerce_dtype_like(a).map_err(|_| dtype_like_err(a))?;
    let db = coerce_dtype_like(b).map_err(|_| dtype_like_err(b))?;
    Ok(PyDType { inner: promote_dtype(da, db), spelling: None })
}

#[pyfunction]
#[pyo3(signature = (*args))]
fn result_type(args: &Bound<'_, PyTuple>) -> PyResult<PyDType> {
    if args.is_empty() {
        return Err(PyValueError::new_err("at least one array or dtype is required"));
    }
    let mut strong: Vec<DType> = Vec::new();
    let mut weak: Vec<ScalarKind> = Vec::new();
    for a in args.iter() {
        // `np.result_type(None, ...)` treats a bare `None` as `np.dtype(None)`
        // == float64 (numpy's historical "no dtype given" sentinel) --
        // confirmed live, and deliberately different from `promote_types`
        // above, which rejects `None` outright. Handled here directly
        // (never via a live numpy call) so the float64 default is anionpy's
        // own decision, not a side effect of falling through to
        // `dtype_from_pyobj`'s numpy-coercion last resort.
        if a.is_none() {
            strong.push(DType::F64);
            continue;
        }
        if let Some(kind) = classify_scalar(&a) {
            weak.push(kind);
            continue;
        }
        strong.push(coerce_dtype_like(&a).map_err(|_| dtype_like_err(&a))?);
    }
    let mut base = match strong.split_first() {
        Some((first, rest)) => rest.iter().fold(*first, |acc, d| promote_dtype(acc, *d)),
        None => DType::Bool,
    };
    for kind in weak {
        base = weak_target_dtype(base, kind);
    }
    Ok(PyDType { inner: base, spelling: None })
}

// ---------------------------------------------------------------------------
// min_scalar_type
// ---------------------------------------------------------------------------

/// Read a 0-d `NdArray`'s single element out as an owned `Buffer`-typed
/// Rust value tuple `(is_bool, is_int, is_unsigned, i128 or 0, f64 or 0,
/// f64_imag or 0, is_complex)` -- deliberately not a `PyObject` round-trip
/// (no scalar-type classes to hand back exist yet, see module doc), just
/// enough structure for `min_scalar_type`'s own value-based branching.
enum ScalarValue {
    #[allow(dead_code)]
    Bool(bool),
    UInt(u128),
    Int(i128),
    Float(f64),
    Complex(f64, f64),
}

fn read_0d_scalar(arr: &ionp_core::NdArray) -> ScalarValue {
    let off = arr.offset() as usize;
    match arr.buffer() {
        Buffer::Bool(v) => ScalarValue::Bool(v[off]),
        Buffer::I8(v) => ScalarValue::Int(v[off] as i128),
        Buffer::I16(v) => ScalarValue::Int(v[off] as i128),
        Buffer::I32(v) => ScalarValue::Int(v[off] as i128),
        Buffer::I64(v) => ScalarValue::Int(v[off] as i128),
        Buffer::U8(v) => ScalarValue::UInt(v[off] as u128),
        Buffer::U16(v) => ScalarValue::UInt(v[off] as u128),
        Buffer::U32(v) => ScalarValue::UInt(v[off] as u128),
        Buffer::U64(v) => ScalarValue::UInt(v[off] as u128),
        Buffer::F16(v) => ScalarValue::Float(v[off].to_f64()),
        Buffer::F32(v) => ScalarValue::Float(v[off] as f64),
        Buffer::F64(v) => ScalarValue::Float(v[off]),
        Buffer::C64(v) => ScalarValue::Complex(v[off].re as f64, v[off].im as f64),
        Buffer::C128(v) => ScalarValue::Complex(v[off].re, v[off].im),
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("dtypeinfo.rs: min_scalar_type has no meaning for string dtypes (phase 2 declines this operation on S/U)"),
    }
}

/// `None` means "real numpy would return `dtype('O')` (object dtype) here"
/// -- a value whose magnitude exceeds what any concrete anionpy integer dtype
/// (up to 64-bit) can hold. anionpy has NO object dtype at all (architectural
/// absence, not a bug/oversight), so this is a genuine, permanent
/// divergence from numpy for these inputs -- callers MUST surface this as
/// an error, never silently widen to `int64`/`uint64` (an earlier version
/// of this code did exactly that: `.unwrap_or(DType::U64)` etc., which
/// silently produced a WRONG dtype instead of a loud, honest failure --
/// caught by a differential probe against real numpy on
/// `-9223372036854775809` [`i64::MIN - 1`], where numpy returns `object`
/// and the buggy code returned `int64`). This gap is why `min_scalar_type`
/// is NOT declared as fully matching numpy in `toplevel.py` -- see report.
fn min_scalar_type_of_value(v: &ScalarValue) -> Option<DType> {
    Some(match v {
        ScalarValue::Bool(_) => DType::Bool,
        ScalarValue::UInt(u) => min_scalar_type_unsigned(*u)?,
        ScalarValue::Int(i) => {
            if *i >= 0 {
                min_scalar_type_unsigned(*i as u128)?
            } else {
                min_scalar_type_signed(*i)?
            }
        }
        ScalarValue::Float(f) => min_scalar_type_float(*f),
        ScalarValue::Complex(re, im) => min_scalar_type_complex(*re, *im),
    })
}

// TICKET #38: no longer called from `min_scalar_type` (the object-dtype
// marker now covers that overflow path, see the call site's comment) --
// kept, not deleted, since `dtypeinfo.rs` is not this ticket's file to
// prune and another agent may still rely on this helper elsewhere.
#[allow(dead_code)]
fn no_object_dtype_err(display: impl std::fmt::Display) -> PyErr {
    // Documented, deliberate divergence from numpy (see
    // `min_scalar_type_of_value`'s doc comment): real numpy returns
    // `dtype('O')` for these inputs, which anionpy has no representation for
    // at all. `OverflowError` chosen over a generic `ValueError` since the
    // trigger is specifically "magnitude too large for any fixed-width
    // dtype anionpy supports" -- not a malformed-input error.
    PyErr::new::<pyo3::exceptions::PyOverflowError, _>(format!(
        "anionpy has no object dtype: {display} exceeds the range of every concrete integer/float dtype anionpy supports (numpy would return dtype('O') here)"
    ))
}

#[pyfunction]
fn min_scalar_type(obj: &Bound<'_, PyAny>) -> PyResult<PyDType> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyArray>>() {
        if pyref.inner.ndim() == 0 {
            let v = read_0d_scalar(&pyref.inner);
            // An array element is always already within its own stored
            // dtype's representable range, so this can never hit the
            // object-dtype gap above -- `.expect` is safe, not a punt.
            let d = min_scalar_type_of_value(&v)
                .expect("0-d array element is always within a representable dtype's range");
            return Ok(PyDType { inner: d, spelling: None });
        }
        return Ok(PyDType { inner: pyref.inner.dtype(), spelling: None });
    }
    // `bool` MUST be checked before `int`/`complex`/`float`: Python `bool`
    // is an `int` subclass, so `obj.extract::<i128>()` on a bare `True`
    // would silently succeed as `1` and misclassify it -- verified against
    // real numpy: `np.min_scalar_type(True) == np.dtype('bool')`, not uint8.
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(PyDType { inner: if b { DType::Bool } else { DType::Bool }, spelling: None });
    }
    // `PyComplex`-specific check only -- a plain Python `int`/`float` is
    // never an instance of `complex`, so this ordering relative to the
    // int/float branches below is safe.
    if let Ok(pc) = obj.cast::<pyo3::types::PyComplex>() {
        use pyo3::types::PyComplexMethods;
        return Ok(PyDType {
            inner: min_scalar_type_complex(pc.real(), pc.imag()),
            spelling: None,
        });
    }
    if let Ok(i) = obj.extract::<i128>() {
        let v = if i >= 0 { ScalarValue::UInt(i as u128) } else { ScalarValue::Int(i) };
        // TICKET #38: a magnitude no fixed-width anionpy dtype can hold is
        // exactly the case real numpy answers with `dtype('O')` (object
        // dtype) -- now representable via the marker-only object dtype
        // (`PyDType { spelling: Some('O'), .. }`, see `lib.rs`'s
        // `PyDType::new` doc comment for the full scope of what this marker
        // does/doesn't cover). Previously this branch raised
        // `no_object_dtype_err` unconditionally for every out-of-range
        // magnitude; now it returns the same value real numpy does.
        return Ok(min_scalar_type_of_value(&v)
            .map(|inner| PyDType { inner, spelling: None })
            .unwrap_or(PyDType { inner: DType::Bool, spelling: Some('O') }));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(PyDType { inner: min_scalar_type_float(f), spelling: None });
    }
    Err(PyTypeError::new_err(format!(
        "data type not understood: {}",
        obj.str().map(|s| s.to_string()).unwrap_or_default()
    )))
}

// ---------------------------------------------------------------------------
// iinfo / finfo
// ---------------------------------------------------------------------------

/// numpy's `dtype.kind` single-char code -- NOT the same as `np.dtype(x).char`
/// (which is per-itemsize, e.g. `'l'` for int64 on this platform). Verified
/// by direct probe against real numpy 2.5.1 that `iinfo`'s error message
/// uses `.kind` (bool -> `'b'`, float32 AND float64 both -> `'f'`, complex64
/// AND complex128 both -> `'c'`) -- this is why bool's error text is
/// `"Invalid integer data type 'b'."` even though bool's own `.char` is
/// `'?'`: `'b'` here is the *kind* code, which bool happens to share with
/// int8's kind grouping in numpy's scheme (`'b'` = "signed integer kind").
fn dtype_kind_char(d: DType) -> char {
    match d {
        DType::Bool => 'b',
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => 'i',
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => 'u',
        DType::F16 | DType::F32 | DType::F64 => 'f',
        DType::C64 | DType::C128 => 'c',
        // Verified against real numpy 2.5.1: `np.dtype('S5').kind == 'S'`,
        // `np.dtype('U5').kind == 'U'` -- same as `DType::kind_char`, no
        // separate "kind grouping" quirk like bool/int share for the
        // numeric dtypes above. Not reachable via `iinfo` (which rejects
        // non-integer dtypes before this point), kept exhaustive anyway.
        DType::S(_) => 'S',
        DType::U(_) => 'U',
    }
}

fn dtype_repr(d: DType) -> String {
    // Mirrors `PyDType::__repr__` in `lib.rs` exactly (kept as two call
    // sites, not merged, because `lib.rs`'s version is a `#[pymethods]`
    // instance method and this one takes a bare `DType` for error-message
    // formatting) -- see that function's doc comment for the verified
    // S/U asymmetry (`S` has no `|` marker in repr, `U` keeps its `<`).
    // Zero-width special case (#52): see `PyDType::__repr__` in `lib.rs`
    // for the verified real-numpy behavior this mirrors (`dtype('S0')` ->
    // `"dtype('S')"`, `dtype('U0')` -> `"dtype('<U')"`).
    match d {
        DType::S(0) => "dtype('S')".to_string(),
        DType::S(n) => format!("dtype('S{n}')"),
        DType::U(_) if d.char_count() == 0 => "dtype('<U')".to_string(),
        DType::U(_) => format!("dtype('<U{}')", d.char_count()),
        _ => format!("dtype('{}')", d.name()),
    }
}

/// numpy's per-dtype single-character `.char` typecode, as measured
/// directly (`np.dtype(name).char` for all 14 dtypes on this platform:
/// macOS/Linux 64-bit, where `int64`/`uint64` are `'l'`/`'L'` -- NOT the
/// Windows-only `'q'`/`'Q'` longlong aliases). Used by `mintypecode` when
/// an element of its input is a dtype-like object rather than a bare
/// typecode character.
pub(crate) fn dtype_typecode(d: DType) -> char {
    match d {
        DType::Bool => '?',
        DType::I8 => 'b',
        DType::I16 => 'h',
        DType::I32 => 'i',
        DType::I64 => 'l',
        DType::U8 => 'B',
        DType::U16 => 'H',
        DType::U32 => 'I',
        DType::U64 => 'L',
        DType::F16 => 'e',
        DType::F32 => 'f',
        DType::F64 => 'd',
        DType::C64 => 'F',
        DType::C128 => 'D',
        // Verified against real numpy 2.5.1: `np.dtype('S5').char == 'S'`,
        // `np.dtype('U5').char == 'U'` -- the bare kind letter, not a
        // width-suffixed spelling (unlike `.str`, which does include the
        // width).
        DType::S(_) => 'S',
        DType::U(_) => 'U',
    }
}

/// Real numpy's `np.dtype(obj)` resolution -- try to interpret `obj`
/// itself as a dtype spec; if that raises, fall back to asking about its
/// TYPE instead (`np.dtype(type(obj))`). The fallback step never raises on
/// its own -- an unrecognized type resolves to numpy's generic `object`
/// dtype (`kind == 'O'`), which is why `np.iinfo([1, 2])` and
/// `np.iinfo(np.array([1, 2]))` both fail with kind `'O'` rather than a
/// resolution error: list/ndarray are not valid integer dtypes, but they
/// ARE valid (object) dtype specs by the second step, so resolution itself
/// succeeds and it is `iinfo`'s own `'iu'` kind check that does the
/// rejecting -- there is no dedicated "reject an array" branch anywhere in
/// this function, deliberately: that would be the wrong shape and would
/// miss `[1, 2]`/`object()`, which reach the identical `'O'` kind via the
/// exact same fallback with no array involved at all.
///
/// SHARED by `iinfo` AND `finfo` (measured live 2026-08-06 against numpy
/// 2.5.1's actual `finfo.__new__`, which does exactly this same two-step
/// `numeric.dtype(dtype)` / `numeric.dtype(type(dtype))` resolution before
/// its own inexact-kind check -- not a guess, read from numpy's own
/// `getlimits.py` source). `finfo`'s `None` handling is its own special
/// case, function-specific and handled by the caller BEFORE this function
/// is ever reached (see `finfo`'s own doc comment) -- this function's
/// `None` resolution (`Some(F64)`, matching `iinfo`'s behaviour) is never
/// exercised on the `finfo` path for that reason, not because it's wrong
/// there, but because `finfo` never gets this far with a `None`.
///
/// Returns `Ok(Some(d))` for one of anionpy's 14 known dtypes, `Ok(None)` for
/// numpy's generic `object` dtype (kind `'O'`) -- the only "other kind"
/// either caller needs to name, since both `iinfo` and `finfo` reject
/// every non-matching kind with the identical message shape regardless of
/// which one it is (a real, documented, permanent gap for the handful of
/// inputs where numpy's actual fallback dtype is `'<U'`/`'V'`/etc. rather
/// than `'O'` -- e.g. `np.dtype(str) == dtype('<U0')`, not object; anionpy has
/// no string/void dtype to represent that with, so it reports `'O'`
/// uniformly instead, same as `iinfo` already did before this function had
/// a second caller).
fn resolve_dtype_or_object(obj: &Bound<'_, PyAny>) -> PyResult<Option<DType>> {
    if let Some(resolved) = try_dtype_of_value(obj)? {
        return Ok(resolved);
    }
    type_to_dtype_kind(&obj.get_type())
}

/// Step 1 of `resolve_dtype_or_object`: interpret `obj` itself as a dtype
/// spec, exactly as real numpy's `np.dtype(obj)` would. The OUTER
/// `Option` means step 1 succeeded (`Some`) vs. failed/raised (`None`,
/// caller falls back to step 2); the INNER `Option<DType>` distinguishes
/// a known dtype from numpy's object dtype (e.g. `obj` is itself an
/// unrecognized TYPE, `np.dtype(list) == object`, which is a step-1
/// SUCCESS, not a failure).
fn try_dtype_of_value(obj: &Bound<'_, PyAny>) -> PyResult<Option<Option<DType>>> {
    let py = obj.py();
    // `obj` is itself a class/type object -- `np.dtype(cls)` resolves by
    // TYPE identity and never raises (an unrecognized type falls through
    // to `type_to_dtype_kind`'s own object-dtype default).
    if obj.is_instance_of::<pyo3::types::PyType>() {
        return Ok(Some(type_to_dtype_kind(obj)?));
    }
    if obj.is_none() {
        return Ok(Some(Some(DType::F64)));
    }
    // anionpy's own dtype instance.
    if let Ok(pd) = obj.extract::<PyRef<'_, PyDType>>() {
        return Ok(Some(Some(pd.inner)));
    }
    // Both isinstance checks below identify a REAL numpy object (a genuine
    // `np.dtype` instance, or a foreign `np.ndarray`) -- narrow,
    // numpy-specific identity checks per this project's settled policy
    // (see `dtype_from_pyobj`/`axis_error` and the module-level idiom in
    // `errors::numpy_available`). When numpy is not importable at all,
    // neither check can be true -- an anionpy process with no numpy installed
    // cannot be holding a real numpy object -- so this degrades to "not a
    // numpy dtype/ndarray" and falls through to the remaining, numpy-free
    // steps below, instead of propagating the `ImportError` (that
    // propagation was a regression: `iinfo`/`finfo` must keep working with
    // no numpy at all, same as the rest of anionpy -- see `errors::
    // numpy_available`'s callers for the established pattern).
    if crate::errors::numpy_available(py) {
        let np = PyModule::import(py, "numpy")?;
        // A real `np.dtype` instance -- verified live that numpy 2's own
        // per-dtype classes (e.g. `Int32DType`) all subclass the legacy
        // monolithic `np.dtype`, so this isinstance check is reliable across
        // numpy's dtype hierarchy, not just old-style `dtype` objects.
        if obj.is_instance(&np.getattr("dtype")?)? {
            if let Ok(name) = obj.getattr("name").and_then(|n| n.extract::<String>()) {
                return Ok(Some(dtype_name_to_dtype(&name).ok()));
            }
        }
        // An ndarray -- anionpy's own or a foreign numpy one -- is explicitly
        // REJECTED by real numpy's `np.dtype()` at this step
        // (`TypeError: Cannot construct a dtype from an array`, live-verified),
        // even though it has a `.dtype` attribute the way a scalar does. This
        // is the crux of defect 2: `coerce_dtype_like` (used elsewhere in this
        // file) reads `.dtype` off ANY object that has one, arrays included --
        // correct for its own callers, wrong for `np.dtype()`'s actual rule --
        // so `iinfo` gets its own resolution here instead of reusing it.
        // Checked BEFORE the generic scalar `hasattr("dtype")` branch below
        // for exactly this reason.
        if obj.is_instance(&np.getattr("ndarray")?)? {
            return Ok(None);
        }
    }
    if obj.extract::<PyRef<'_, PyArray>>().is_ok() {
        return Ok(None);
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(match dtype_name_to_dtype(&s) {
            Ok(d) => Some(Some(d)),
            Err(_) => None,
        });
    }
    if obj.is_instance_of::<pyo3::types::PyBytes>() {
        if let Ok(b) = obj.extract::<Vec<u8>>() {
            let s = String::from_utf8_lossy(&b).into_owned();
            return Ok(match dtype_name_to_dtype(&s) {
                Ok(d) => Some(Some(d)),
                Err(_) => None,
            });
        }
    }
    // A numpy (or anionpy) scalar VALUE, e.g. `np.int8(3)`: real numpy's
    // `np.dtype()` accepts these directly, resolving to their own dtype --
    // distinct from the ndarray rejection just above.
    if obj.hasattr("dtype")? {
        if let Ok(name) =
            obj.getattr("dtype")?.getattr("name").and_then(|n| n.extract::<String>())
        {
            return Ok(Some(dtype_name_to_dtype(&name).ok()));
        }
    }
    // Everything else -- a bare Python `int`/`float`/`complex`/`bool`
    // VALUE, a list, a plain `object()` -- fails step 1 (`np.dtype(3)`,
    // `np.dtype(3.5)`, `np.dtype([1, 2])`, `np.dtype(object())` all raise,
    // live-verified). Falls through to step 2 in the caller.
    Ok(None)
}

/// Step 2 of `resolve_dtype_or_object`: `np.dtype(type(obj))`. Never fails --
/// an unrecognized type resolves to numpy's generic object dtype
/// (`Ok(None)`), live-verified for `list`, `dict`, `tuple`,
/// `numpy.ndarray`, and a plain user-defined class, all of which
/// `np.dtype(...)` maps to `dtype('O')`.
fn type_to_dtype_kind(ty: &Bound<'_, PyAny>) -> PyResult<Option<DType>> {
    let py = ty.py();
    if ty.is(&py.get_type::<pyo3::types::PyInt>()) {
        return Ok(Some(DType::I64));
    }
    if ty.is(&py.get_type::<pyo3::types::PyBool>()) {
        return Ok(Some(DType::Bool));
    }
    if ty.is(&py.get_type::<pyo3::types::PyFloat>()) {
        return Ok(Some(DType::F64));
    }
    if ty.is(&py.get_type::<pyo3::types::PyComplex>()) {
        return Ok(Some(DType::C128));
    }
    if let Ok(name) = ty.getattr("__ionp_dtype_name__").and_then(|n| n.extract::<String>()) {
        return Ok(dtype_name_to_dtype(&name).ok());
    }
    if let Ok(name) = ty.getattr("__name__").and_then(|n| n.extract::<String>()) {
        let alt = if name == "bool_" { "bool".to_string() } else { name };
        return Ok(dtype_name_to_dtype(&alt).ok());
    }
    Ok(None)
}

#[pyfunction]
fn iinfo(dtype: &Bound<'_, PyAny>) -> PyResult<(i128, i128, i32, String)> {
    // Returned as a plain tuple `(min, max, bits, dtype_name)` rather than a
    // custom `iinfo`-mimicking class -- Python-side wrapper in `anionpy/`
    // turns this into the user-facing object; see report for why this
    // split exists (keeps this file numpy-value-derivation-only).
    match resolve_dtype_or_object(dtype)? {
        Some(d) => match iinfo_for(d) {
            Some(info) => Ok((info.min, info.max, info.bits, d.name().to_string())),
            None => Err(PyValueError::new_err(format!(
                "Invalid integer data type {:?}.",
                dtype_kind_char(d)
            ))),
        },
        // numpy's generic object dtype -- resolution itself succeeded
        // (step 2 never raises), it is the `'iu'` kind check that rejects
        // it, same as any other non-integer kind above.
        None => Err(PyValueError::new_err("Invalid integer data type 'O'.".to_string())),
    }
}

#[pyfunction]
fn finfo(py: Python<'_>, dtype: &Bound<'_, PyAny>) -> PyResult<Py<PyTuple>> {
    // Same plain-tuple contract as `iinfo` above, EXCEPT returned as a
    // manually-built `PyTuple` rather than a native Rust tuple: pyo3 0.29's
    // `IntoPyObject` is only implemented for tuples up to a fixed arity,
    // and this field list (18 elements) exceeds it. Field order:
    // (eps, epsneg, max, min, tiny, smallest_normal, smallest_subnormal,
    //  resolution, precision, bits, iexp, nexp, nmant, machep, negep,
    //  minexp, maxexp, dtype_name).
    // NOTE, CORRECTED 2026-08-06 (Monday): this comment used to say the
    // function was left undeclared because "anionpy has no scalar type classes
    // yet to return". That reason is STALE and was stale for some time
    // before anyone re-checked it. anionpy does have a scalar hierarchy, and
    // the Python wrapper (`anionpy/_dtypeinfo.py::_finfo_scalar`) now mints
    // properly dtype-typed scalars through the established
    // `numpy_scalar_from_0d` 0-d-array contract. The raw tuple this
    // function returns is still plain values by design -- the typing
    // happens one layer up, which is the right seam.
    //
    // Verified by direct probe: `np.finfo(complex64)`/`np.finfo(complex128)`
    // do NOT error -- they silently return the COMPONENT float's finfo
    // (`finfo(complex64).dtype == dtype('float32')`, not complex64 at all),
    // so complex inputs are routed to their component float dtype for both
    // the lookup AND the reported `dtype_name` before ever reaching
    // `finfo_for`. Non-float, non-complex inputs error with
    // `"data type dtype('{name}') not compatible with finfo"` (uses the
    // dtype's OWN repr, not the component/kind char pattern `iinfo` uses).
    // `None` is special-cased BEFORE the generic coercion because numpy
    // special-cases it too, and only in `finfo`. MEASURED vs numpy 2.5.1:
    //
    //   np.finfo(None) -> TypeError: dtype must not be None
    //   np.iinfo(None) -> ValueError: Invalid integer data type 'f'.
    //
    // `iinfo` has no such guard -- there, `None` coerces to float64 and
    // then fails the integer check, and anionpy already reproduced that exact
    // ValueError. So this guard must NOT be lifted into the shared
    // `resolve_dtype_or_object` helper: doing so would "fix" finfo by
    // breaking iinfo, which currently matches.
    //
    // Found by out-of-corpus probing AFTER the four finfo.* differential
    // items had gone green. The corpus never passes None, so the suite was
    // and would have stayed silent about this.
    if dtype.is_none() {
        return Err(PyTypeError::new_err("dtype must not be None"));
    }
    // RE-MEASURED 2026-08-06 (task #28): this used to call the narrower
    // `coerce_iinfo_finfo_dtype` (int/float/bool TYPE special-cases, then
    // the SAME `coerce_dtype_like` `iinfo`/`can_cast`/`promote_types` all
    // share), which reads `.dtype` off ANY object that has one -- anionpy
    // ARRAYS included. Real numpy's `finfo` never does that: its actual
    // `__new__` (`numpy/_core/getlimits.py`, read directly, not guessed)
    // is `try: dtype = numeric.dtype(dtype); except TypeError: dtype =
    // numeric.dtype(type(dtype))` -- i.e. exactly `iinfo`'s own two-step
    // `np.dtype(obj)` / `np.dtype(type(obj))` resolution, which already
    // rejects arrays (`np.dtype(some_array)` raises "Cannot construct a
    // dtype from an array", live-verified) and falls through to the
    // object-dtype kind instead. Using the narrower coercion made
    // `anionpy.finfo(anionpy.array(1.5, dtype='float32'))` SUCCEED where numpy
    // raises `ValueError: data type dtype('O') not compatible with finfo`
    // -- the dangerous direction, anionpy more permissive than numpy -- and
    // separately made every input `coerce_dtype_like` couldn't resolve at
    // all (a bare `list`/`dict`/`object()`, an unresolvable string, a bare
    // Python `int`/`bool` VALUE) raise the wrong exception CLASS
    // (`TypeError`) with an invented message instead of numpy's actual
    // `ValueError` naming the type-fallback dtype it resolved to. Sharing
    // `resolve_dtype_or_object` with `iinfo` (see that function's doc
    // comment) fixes both at once, by the same shape `iinfo` was already
    // fixed by in task #12 -- not a second idiom.
    let resolved = resolve_dtype_or_object(dtype)?;
    let d_in = match resolved {
        Some(d) => d,
        // numpy's generic object dtype via the type-fallback step -- same
        // "resolution succeeds, the kind check rejects it" shape as
        // `iinfo`'s identical `None` arm below, and the same known,
        // documented, permanent imprecision: numpy's REAL fallback dtype
        // for some inputs (a plain `str` -> `dtype('<U0')`, an empty
        // `dict` -> a 0-field structured dtype) is not literally `'O'`,
        // but anionpy has no string/void dtype to name those with, so this
        // reports the object-dtype message uniformly instead, same as
        // `iinfo` already does for the identical inputs.
        None => {
            return Err(PyValueError::new_err(
                "data type dtype('O') not compatible with finfo".to_string(),
            ))
        }
    };
    let d = match d_in {
        DType::C64 => DType::F32,
        DType::C128 => DType::F64,
        other => other,
    };
    match finfo_for(d) {
        Some(f) => {
            let items: Vec<Py<PyAny>> = vec![
                f.eps.into_pyobject(py)?.into_any().unbind(),
                f.epsneg.into_pyobject(py)?.into_any().unbind(),
                f.max.into_pyobject(py)?.into_any().unbind(),
                f.min.into_pyobject(py)?.into_any().unbind(),
                f.tiny.into_pyobject(py)?.into_any().unbind(),
                f.smallest_normal.into_pyobject(py)?.into_any().unbind(),
                f.smallest_subnormal.into_pyobject(py)?.into_any().unbind(),
                f.resolution.into_pyobject(py)?.into_any().unbind(),
                f.precision.into_pyobject(py)?.into_any().unbind(),
                f.bits.into_pyobject(py)?.into_any().unbind(),
                f.iexp.into_pyobject(py)?.into_any().unbind(),
                f.nexp.into_pyobject(py)?.into_any().unbind(),
                f.nmant.into_pyobject(py)?.into_any().unbind(),
                f.machep.into_pyobject(py)?.into_any().unbind(),
                f.negep.into_pyobject(py)?.into_any().unbind(),
                f.minexp.into_pyobject(py)?.into_any().unbind(),
                f.maxexp.into_pyobject(py)?.into_any().unbind(),
                d.name().into_pyobject(py)?.into_any().unbind(),
            ];
            Ok(PyTuple::new(py, items)?.unbind())
        }
        None => Err(PyValueError::new_err(format!(
            "data type {} not compatible with finfo",
            dtype_repr(d_in)
        ))),
    }
}

// ---------------------------------------------------------------------------
// typename / mintypecode
// ---------------------------------------------------------------------------

#[pyfunction]
fn typename(char: &Bound<'_, PyAny>) -> PyResult<String> {
    // Real numpy's `typename` is a one-line `typechars[char]` dict lookup
    // (`numpy/_core/numerictypes.py`), not a string-typed function -- it
    // accepts ANY hashable object as `char` and raises exactly what a
    // plain dict subscript would: `KeyError(char)` (with `char`'s own
    // repr/value, not a string coercion of it) for a hashable miss, or
    // whatever `TypeError` Python's own hashing machinery raises for an
    // unhashable one (e.g. "unhashable type: 'list'"). A first draft here
    // only accepted `&str` and pyo3 rejected everything else with its own
    // generic type-error text before ever reaching this function -- caught
    // by a differential probe with `None`/`123`/`[]`/`{}`/`()` as `char`,
    // all of which real numpy accepts as dict keys. Mirrored here with a
    // real Python `.hash()` call (plain CPython dict-key semantics, not a
    // numpy call) so the unhashable-type message is Python's own text,
    // not an invented one.
    if let Ok(s) = char.extract::<String>() {
        return typename_for(&s)
            .map(|v| v.to_string())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(s));
    }
    char.hash()?;
    Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
        char.clone().unbind(),
    ))
}

#[pyfunction]
#[pyo3(signature = (typechars, typeset="GDFgdf", default="d"))]
fn mintypecode(typechars: &Bound<'_, PyAny>, typeset: &str, default: &str) -> PyResult<String> {
    let codes: Vec<String> = if let Ok(s) = typechars.extract::<String>() {
        s.chars().map(|c| c.to_string()).collect()
    } else {
        typechars
            .try_iter()?
            .map(|item| -> PyResult<String> {
                let item = item?;
                if let Ok(s) = item.extract::<String>() {
                    Ok(s)
                } else {
                    let arr = extract_array_like(&item)?;
                    Ok(dtype_typecode(arr.dtype()).to_string())
                }
            })
            .collect::<PyResult<Vec<String>>>()?
    };
    let refs: Vec<&str> = codes.iter().map(|s: &String| s.as_str()).collect();
    Ok(mintypecode_select(&refs, typeset, default))
}

// ---------------------------------------------------------------------------
// isdtype (string/tuple-of-string kind argument ONLY -- see module doc)
// ---------------------------------------------------------------------------

fn dtype_matches_kind_string(d: DType, kind: &str) -> PyResult<bool> {
    use DType::*;
    Ok(match kind {
        "bool" => d == Bool,
        "signed integer" => matches!(d, I8 | I16 | I32 | I64),
        "unsigned integer" => matches!(d, U8 | U16 | U32 | U64),
        "integral" => matches!(d, I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64),
        "real floating" => matches!(d, F16 | F32 | F64),
        "complex floating" => matches!(d, C64 | C128),
        "numeric" => d != Bool,
        _ => {
            return Err(PyValueError::new_err(format!(
                "kind argument is a string, but '{kind}' is not a known kind name."
            )))
        }
    })
}

/// The 34 scalar-type names numpy's `allTypes` exposes, i.e. exactly the
/// set `numpy._core.numerictypes._preprocess_dtype` will accept. Anything
/// else -- a Python `int`/`float`/`bool`/`str`/`object`, an `ndarray`, a
/// list, `None`, a bare number -- is a `TypeError`. Transcribed from a live
/// `sorted({t.__name__ for t in allTypes.values()})` against numpy 2.5.1;
/// it deliberately includes the ABSTRACT names (`integer`, `floating`,
/// `generic`, ...) and the names anionpy has no dtype for (`longdouble`,
/// `longlong`, `object_`, `str_`, `void`, `datetime64`, ...), because
/// numpy ACCEPTS those as a `kind` and simply reports no match -- it does
/// not raise. Dropping them would turn numpy's `False` into a `TypeError`.
const NUMPY_SCALAR_TYPE_NAMES: [&str; 34] = [
    "bool", "bytes_", "character", "clongdouble", "complex128", "complex64",
    "complexfloating", "datetime64", "flexible", "float16", "float32", "float64",
    "floating", "generic", "inexact", "int16", "int32", "int64", "int8", "integer",
    "longdouble", "longlong", "number", "object_", "signedinteger", "str_",
    "timedelta64", "uint16", "uint32", "uint64", "uint8", "ulonglong",
    "unsignedinteger", "void",
];

/// Resolve an `isdtype` argument to numpy's own notion of a dtype identity:
/// the NAME of the scalar type. numpy's `_preprocess_dtype` maps a `dtype`
/// instance to its `.type` scalar class and then requires membership in
/// `allTypes.values()`, and `isdtype` finishes with a plain `in` test over
/// those classes -- so the whole function is scalar-type IDENTITY matching,
/// which a name comparison reproduces exactly for every type anionpy can
/// name. Returns `None` for anything numpy would reject.
///
/// Recognising a numpy scalar class WITHOUT importing numpy: every one of
/// them has `generic` in its MRO (verified: `np.bool_.__mro__` is
/// `(numpy.bool, numpy.generic, object)`), and anionpy mirrors that hierarchy
/// exactly (`anionpy.int8.__mro__` is `(anionpy.int8, anionpy.signedinteger,
/// anionpy.integer, anionpy.number, anionpy.generic, object)`), so the same test
/// accepts anionpy's own scalar classes -- which is required, since anionpy must
/// answer this question about its own types, not only about numpy's.
///
/// KNOWN ALIAS CAVEAT, recorded rather than papered over: numpy compares
/// CLASSES, and on some platforms two distinct classes share a width
/// (`np.longlong` is not `np.int64` here even though both are 8 bytes).
/// Name comparison gets those rows right for every dtype anionpy HAS -- an
/// anionpy int64 vs a `longlong` kind is `False` on both sides -- but anionpy
/// has no `longlong`/`longdouble` dtype to ask the mirror-image question
/// about, so the divergence is unreachable rather than fixed.
fn scalar_type_name(obj: &Bound<'_, PyAny>) -> Option<String> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyDType>>() {
        return Some(pyref.inner.name().to_string());
    }
    let ty = obj.cast::<pyo3::types::PyType>().ok()?;
    let mro = ty.getattr("__mro__").ok()?;
    let mut is_generic = false;
    for base in mro.try_iter().ok()? {
        let base: Bound<'_, PyAny> = base.ok()?;
        let n: String = base.getattr("__name__").ok()?.extract().ok()?;
        if n == "generic" {
            is_generic = true;
            break;
        }
    }
    if !is_generic {
        return None;
    }
    let name: String = ty.getattr("__name__").ok()?.extract().ok()?;
    if NUMPY_SCALAR_TYPE_NAMES.contains(&name.as_str()) {
        Some(name)
    } else {
        None
    }
}

/// The concrete scalar-type names a `kind` STRING expands to, mirroring
/// numpy's `sctypes[...]` unions. Names, not `DType`s, so the result can be
/// compared against an abstract or ionp-less scalar name without needing a
/// `DType` to exist for it.
fn kind_string_names(kind: &str) -> PyResult<&'static [&'static str]> {
    const SIGNED: &[&str] = &["int8", "int16", "int32", "int64"];
    const UNSIGNED: &[&str] = &["uint8", "uint16", "uint32", "uint64"];
    const INTEGRAL: &[&str] = &[
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ];
    const REAL: &[&str] = &["float16", "float32", "float64"];
    const COMPLEX: &[&str] = &["complex64", "complex128"];
    const NUMERIC: &[&str] = &[
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64", "complex64", "complex128",
    ];
    Ok(match kind {
        "bool" => &["bool"],
        "signed integer" => SIGNED,
        "unsigned integer" => UNSIGNED,
        "integral" => INTEGRAL,
        "real floating" => REAL,
        "complex floating" => COMPLEX,
        "numeric" => NUMERIC,
        _ => {
            return Err(PyValueError::new_err(format!(
                "kind argument is a string, but '{kind}' is not a known kind name."
            )))
        }
    })
}

/// BUG FOUND + FIXED (2026-08-04): `isdtype`'s `kind` argument used to
/// accept ONLY a string or a homogeneous tuple-of-strings, so THREE real
/// numpy forms raised `TypeError: kind argument must be a string, a dtype,
/// or a tuple of strings and dtypes` where numpy answers:
///   1. a scalar TYPE class:   `np.isdtype(np.dtype('int8'), np.integer)`
///      -> False (accepted, no match -- abstract classes are legal kinds
///      and simply never equal a concrete one);
///      `np.isdtype(np.dtype('int8'), np.int8)` -> True;
///   2. a MIXED tuple:         `np.isdtype(np.dtype('int8'),
///      ('integral', np.integer))` -> True. The old `extract::<Vec<String>>`
///      fails on the whole tuple the moment one element is not a string, so
///      even the STRING half of a mixed tuple was being thrown away;
///   3. a scalar TYPE class as the `dtype` argument, which numpy's own
///      docstring uses: `np.isdtype(np.float32, "real floating")` -> True.
/// All three measured live against numpy 2.5.1. The `kind`-string and
/// bare-string-`dtype`-rejection paths are unchanged.
#[pyfunction]
fn isdtype(dtype: &Bound<'_, PyAny>, kind: &Bound<'_, PyAny>) -> PyResult<bool> {
    // `dtype` must be an ACTUAL dtype object or a scalar TYPE class --
    // real numpy rejects everything else, including a bare dtype-NAME
    // STRING and even a bare ndarray (which has its own `.dtype` but is
    // not itself one). A first-pass implementation here reused the
    // permissive `coerce_dtype_like` (which accepts strings/arrays-via-
    // `.dtype`/anything `dtype_from_pyobj` can coerce), so
    // `anionpy.isdtype('int8', 'signed integer')` silently returned `True`
    // where real numpy raises `TypeError: dtype argument must be a NumPy
    // dtype, but it is a <class 'str'>.` -- caught by a differential
    // probe, not anticipated. The exact numpy message (including the
    // trailing period after the type repr) is reproduced via the
    // argument's own type `repr()`, which is byte-identical to what
    // CPython would print for the same object -- not a numpy call.
    let dtype_name = scalar_type_name(dtype).ok_or_else(|| {
        let type_repr = dtype
            .get_type()
            .repr()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "?".to_string());
        PyTypeError::new_err(format!(
            "dtype argument must be a NumPy dtype, but it is a {type_repr}."
        ))
    })?;

    // numpy iterates the kind tuple IN ORDER and raises at the first bad
    // element, so a `ValueError` from an unknown kind string can precede a
    // `TypeError` from a later non-dtype element. Matching that ordering
    // means resolving elements one at a time rather than pre-validating.
    let elems: Vec<Bound<'_, PyAny>> = match kind.cast::<pyo3::types::PyTuple>() {
        Ok(t) => t.iter().collect::<Vec<Bound<'_, PyAny>>>(),
        Err(_) => vec![kind.clone()],
    };
    let mut matched = false;
    for elem in &elems {
        if let Ok(s) = elem.extract::<String>() {
            if kind_string_names(&s)?.contains(&dtype_name.as_str()) {
                matched = true;
            }
            continue;
        }
        match scalar_type_name(elem) {
            Some(n) => {
                if n == dtype_name {
                    matched = true;
                }
            }
            None => {
                let type_repr = elem
                    .get_type()
                    .repr()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|_| "?".to_string());
                return Err(PyTypeError::new_err(format!(
                    "kind argument must be comprised of NumPy dtypes or strings only, \
                     but is a {type_repr}."
                )));
            }
        }
    }
    Ok(matched)
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(can_cast, m)?)?;
    m.add_function(wrap_pyfunction!(promote_types, m)?)?;
    m.add_function(wrap_pyfunction!(result_type, m)?)?;
    m.add_function(wrap_pyfunction!(min_scalar_type, m)?)?;
    m.add_function(wrap_pyfunction!(iinfo, m)?)?;
    m.add_function(wrap_pyfunction!(finfo, m)?)?;
    m.add_function(wrap_pyfunction!(typename, m)?)?;
    m.add_function(wrap_pyfunction!(mintypecode, m)?)?;
    m.add_function(wrap_pyfunction!(isdtype, m)?)?;
    Ok(())
}
