//! `Buffer`: the owned, typed storage backing an `NdArray`. One variant per
//! `DType`, each holding a flat `Vec<T>` in element (not byte) units.
//! Multiple `NdArray` views (slices, transposes, reshapes-as-view) share a
//! `Buffer` through `Arc`; the buffer itself is written once at
//! construction/cast time and read thereafter, which keeps aliasing rules
//! trivial.

use half::f16;
use num_complex::Complex;
use std::sync::Arc;

use crate::dtype::DType;

pub type C64 = Complex<f32>;
pub type C128 = Complex<f64>;

#[derive(Debug, Clone)]
pub enum Buffer {
    Bool(Vec<bool>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    // `half::f16` -- ionp has no hardware/stable-Rust f16 primitive to
    // store natively (see `ufunc.rs` module docs for how arithmetic on it
    // is computed: always by round-tripping through f32, matching numpy's
    // own float16 ufunc loops exactly).
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    C64(Vec<C64>),
    C128(Vec<C128>),
    // Fixed-width, zero(NUL)-padded string storage, matching numpy's own
    // in-memory layout for `S`/`U` byte-for-byte: one `Vec<u8>`/`Vec<u32>`
    // per array element, each ALWAYS exactly `width` bytes / `width / 4`
    // UCS-4 code units long (padded with trailing 0 on construction --
    // never trimmed here; trimming trailing NULs happens only when an
    // element is read back out as a Python `bytes`/`str`, same as numpy).
    // The `u32` alongside each variant is the dtype's `itemsize()` (bytes
    // for `S`, bytes -- NOT char count -- for `U`, matching `DType::S(n)`/
    // `DType::U(n)`'s own convention), kept here so every consumer of a
    // bare `Buffer::S`/`Buffer::U` value (no `NdArray`/`DType` in scope)
    // can still answer "how wide is one element" without guessing from
    // the first element (which would break on a 0-length array).
    //
    // WHY NOT `ionp_core::strings::StrElem`: `StrElem` (see `strings.rs`)
    // is a RAGGED, variable-length `Vec<u32>` + an `is_bytes` flag, built
    // for the numpy-interop `char`/`strings` binding layer, where numpy
    // itself already owns the fixed-width padded buffer and ionp only
    // ever sees one already-unpadded element at a time coming in over
    // FFI. It has no notion of a shared per-array width and nothing to
    // enforce padding, so reusing it here would (a) lose the fixed-width
    // invariant this task explicitly requires bit-for-bit (padding bytes
    // must compare equal to numpy's), and (b) conflate two different
    // concerns -- "one already-extracted element for a predicate/transform"
    // vs "the array's own owned, zero-padded, fixed-stride storage" --
    // under one type. `StrElem` is left untouched and still backs the
    // existing `char`/`strings` numpy-interop path exactly as before.
    S(u32, Vec<Vec<u8>>),
    U(u32, Vec<Vec<u32>>),
}

/// Generates the numeric-only match arms (bool and complex handled by hand
/// in `cast_buffer`, see below) that cast a concrete-typed source slice
/// into every destination `DType`. Invoked once per source primitive type
/// with that type substituted textually, so `0 as $srcty` and `x as $dst`
/// are always casts between two concrete, known types — this is what lets
/// one macro stand in for the 10x13 numeric cast matrix instead of hand
/// writing each arm.
///
/// Integer source only: Rust's `as` between two integer types is always a
/// plain bit-truncating/wraparound cast (never saturates), which already
/// matches numpy's integer downcast semantics exactly, so no special
/// handling is needed here. Float sources need different treatment — see
/// `cast_from_float!` below — which is why that's a separate macro instead
/// of one shared by both.
macro_rules! cast_from_int {
    ($v:expr, $dst:expr, $srcty:ty) => {{
        let v: &[$srcty] = $v;
        match $dst {
            DType::Bool => Buffer::Bool(v.iter().map(|&x| x != 0 as $srcty).collect()),
            DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
            DType::I8 => Buffer::I8(v.iter().map(|&x| x as i8).collect()),
            DType::I16 => Buffer::I16(v.iter().map(|&x| x as i16).collect()),
            DType::I32 => Buffer::I32(v.iter().map(|&x| x as i32).collect()),
            DType::I64 => Buffer::I64(v.iter().map(|&x| x as i64).collect()),
            DType::U8 => Buffer::U8(v.iter().map(|&x| x as u8).collect()),
            DType::U16 => Buffer::U16(v.iter().map(|&x| x as u16).collect()),
            DType::U32 => Buffer::U32(v.iter().map(|&x| x as u32).collect()),
            DType::U64 => Buffer::U64(v.iter().map(|&x| x as u64).collect()),
            DType::F16 => Buffer::F16(v.iter().map(|&x| f16::from_f64(x as f64)).collect()),
            DType::F32 => Buffer::F32(v.iter().map(|&x| x as f32).collect()),
            DType::F64 => Buffer::F64(v.iter().map(|&x| x as f64).collect()),
            DType::C64 => Buffer::C64(v.iter().map(|&x| C64::new(x as f32, 0.0)).collect()),
            DType::C128 => Buffer::C128(v.iter().map(|&x| C128::new(x as f64, 0.0)).collect()),
        }
    }};
}

/// Float source cast, covering f32/f64 (f16 has its own hand-written
/// `cast_from_f16` below since it needs `.to_f32()`/`.to_f64()`, not a bare
/// `as`, but mirrors this same I8/I16/U8/U16 fix — as do the complex ->
/// narrow-int arms in `cast_from_complex64`/`cast_from_complex128`, which
/// cast the same way off `c.re`).
///
/// Rust's `x as i8` for a float `x` is a DIRECT saturating cast to i8's own
/// range (-128..=127). numpy's float -> 8/16-bit-int cast is NOT that: it
/// first saturates to `i32` range (numpy's astype-to-int32 rule: NaN -> 0,
/// +inf/overflow -> i32::MAX, -inf/underflow -> i32::MIN — which happens to
/// be exactly what Rust's `x as i32` already does) and only THEN
/// truncates/wraps that i32 down to the narrow destination width by taking
/// its low bits, two's-complement, regardless of the destination's
/// signedness. Verified empirically against numpy 2.5.1 (`.tobytes()`
/// comparison, not `==`):
///   200.0f64  -> i8:   -56 (wraps, like `(200i32) as i8`)   -- NOT 127
///   130.0f64  -> i8:  -126 (wraps)                          -- NOT 127
///  -129.0f64  -> i8:   127 (wraps)                          -- NOT -128
///   -1.5f64   -> u8:   255 (i32 -1's low byte)               -- NOT 0
///     (rules out a `u32`-saturating intermediate for unsigned
///      destinations: a negative float saturating straight to u32 would
///      give 0, not 255 — the intermediate is signed i32 either way)
///  1e300/inf  -> i8:    -1 (i32::MAX's low byte)             -- NOT 127
/// -1e300/-inf -> i8:     0 (i32::MIN's low byte)             -- NOT -128
///        nan  -> i8:     0 (i32-cast-of-nan's low byte)
/// I32/I64/U32/U64 destinations are NOT affected: those saturate directly
/// to their own full range (e.g. `-1.5f64 as u32` numpy-and-Rust both give
/// 0, not the 4294967295 a via-i32 route would produce), matching Rust's
/// native `as` already — only destinations narrower than i32 (I8/I16/U8/
/// U16) go through this two-step pipeline in numpy, which is what this
/// macro now reproduces bit-for-bit.
macro_rules! cast_from_float {
    ($v:expr, $dst:expr, $srcty:ty) => {{
        let v: &[$srcty] = $v;
        match $dst {
            DType::Bool => Buffer::Bool(v.iter().map(|&x| x != 0 as $srcty).collect()),
            DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
            DType::I8 => Buffer::I8(v.iter().map(|&x| (x as i32) as i8).collect()),
            DType::I16 => Buffer::I16(v.iter().map(|&x| (x as i32) as i16).collect()),
            DType::I32 => Buffer::I32(v.iter().map(|&x| x as i32).collect()),
            DType::I64 => Buffer::I64(v.iter().map(|&x| x as i64).collect()),
            DType::U8 => Buffer::U8(v.iter().map(|&x| (x as i32) as u8).collect()),
            DType::U16 => Buffer::U16(v.iter().map(|&x| (x as i32) as u16).collect()),
            DType::U32 => Buffer::U32(v.iter().map(|&x| x as u32).collect()),
            DType::U64 => Buffer::U64(v.iter().map(|&x| x as u64).collect()),
            DType::F16 => Buffer::F16(v.iter().map(|&x| f16::from_f64(x as f64)).collect()),
            DType::F32 => Buffer::F32(v.iter().map(|&x| x as f32).collect()),
            DType::F64 => Buffer::F64(v.iter().map(|&x| x as f64).collect()),
            DType::C64 => Buffer::C64(v.iter().map(|&x| C64::new(x as f32, 0.0)).collect()),
            DType::C128 => Buffer::C128(v.iter().map(|&x| C128::new(x as f64, 0.0)).collect()),
        }
    }};
}

impl Buffer {
    pub fn dtype(&self) -> DType {
        match self {
            Buffer::Bool(_) => DType::Bool,
            Buffer::I8(_) => DType::I8,
            Buffer::I16(_) => DType::I16,
            Buffer::I32(_) => DType::I32,
            Buffer::I64(_) => DType::I64,
            Buffer::U8(_) => DType::U8,
            Buffer::U16(_) => DType::U16,
            Buffer::U32(_) => DType::U32,
            Buffer::U64(_) => DType::U64,
            Buffer::F16(_) => DType::F16,
            Buffer::F32(_) => DType::F32,
            Buffer::F64(_) => DType::F64,
            Buffer::C64(_) => DType::C64,
            Buffer::C128(_) => DType::C128,
            Buffer::S(n, _) => DType::S(*n),
            Buffer::U(n, _) => DType::U(*n),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Buffer::Bool(v) => v.len(),
            Buffer::I8(v) => v.len(),
            Buffer::I16(v) => v.len(),
            Buffer::I32(v) => v.len(),
            Buffer::I64(v) => v.len(),
            Buffer::U8(v) => v.len(),
            Buffer::U16(v) => v.len(),
            Buffer::U32(v) => v.len(),
            Buffer::U64(v) => v.len(),
            Buffer::F16(v) => v.len(),
            Buffer::F32(v) => v.len(),
            Buffer::F64(v) => v.len(),
            Buffer::C64(v) => v.len(),
            Buffer::C128(v) => v.len(),
            Buffer::S(_, v) => v.len(),
            Buffer::U(_, v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Elementwise cast to `dst`, matching numpy's `astype`/ufunc-result
    /// casting semantics: C-style truncating casts between numeric types,
    /// nonzero test for numeric->bool, real-part extraction for
    /// complex->real (numpy does the same and additionally raises
    /// `ComplexWarning`; ionp does not warn — see KNOWN-DIFFERENCES.md).
    pub fn cast_to(&self, dst: DType) -> Buffer {
        if self.dtype() == dst {
            return self.clone();
        }
        match self {
            Buffer::Bool(v) => cast_from_bool(v, dst),
            Buffer::I8(v) => cast_from_int!(v, dst, i8),
            Buffer::I16(v) => cast_from_int!(v, dst, i16),
            Buffer::I32(v) => cast_from_int!(v, dst, i32),
            Buffer::I64(v) => cast_from_int!(v, dst, i64),
            Buffer::U8(v) => cast_from_int!(v, dst, u8),
            Buffer::U16(v) => cast_from_int!(v, dst, u16),
            Buffer::U32(v) => cast_from_int!(v, dst, u32),
            Buffer::U64(v) => cast_from_int!(v, dst, u64),
            Buffer::F16(v) => cast_from_f16(v, dst),
            Buffer::F32(v) => cast_from_float!(v, dst, f32),
            Buffer::F64(v) => cast_from_float!(v, dst, f64),
            Buffer::C64(v) => cast_from_complex64(v, dst),
            Buffer::C128(v) => cast_from_complex128(v, dst),
            // `self.dtype() == dst` (same width, same kind) is already
            // handled by the early return above; every other S/U
            // transition (width change, S<->U, or to/from a numeric
            // dtype) is astype's `DType::S`/`DType::U` half -- explicitly
            // OUT OF SCOPE for this phase (see `ionp-py/src/lib.rs`'s
            // `astype` binding, which now rejects those combinations with
            // a real `PyResult` error BEFORE ever calling this, so this
            // arm stays genuinely unreachable from Python rather than a
            // silent trap). Left as `unreachable!` rather than invented
            // behavior, matching the same-shaped placeholder this crate
            // already carries in the mirror (numeric-source) direction.
            Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("ionp-core: S/U <-> S/U (width change) and S/U <-> numeric astype are not implemented yet (phase 2 defers this; ionp-py's astype binding must reject before reaching here)"),
        }
    }
}

/// `f16` can't go through the `cast_from_float!` macro (it needs `.to_f32()`/
/// `.to_f64()` conversions, not a bare `as` cast -- Rust has no native f16
/// primitive). Same semantics as every other numeric source: truncating
/// numeric casts, nonzero test for ->bool, zero-imaginary for ->complex.
fn cast_from_f16(v: &[f16], dst: DType) -> Buffer {
    match dst {
        DType::Bool => Buffer::Bool(v.iter().map(|&x| x != f16::ZERO).collect()),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        // See `cast_from_float!`'s doc comment: narrower-than-i32
        // destinations go through an i32-saturate-then-truncate pipeline in
        // numpy, not a direct saturating cast to the narrow type.
        DType::I8 => Buffer::I8(v.iter().map(|&x| (x.to_f32() as i32) as i8).collect()),
        DType::I16 => Buffer::I16(v.iter().map(|&x| (x.to_f32() as i32) as i16).collect()),
        DType::I32 => Buffer::I32(v.iter().map(|&x| x.to_f32() as i32).collect()),
        DType::I64 => Buffer::I64(v.iter().map(|&x| x.to_f64() as i64).collect()),
        DType::U8 => Buffer::U8(v.iter().map(|&x| (x.to_f32() as i32) as u8).collect()),
        DType::U16 => Buffer::U16(v.iter().map(|&x| (x.to_f32() as i32) as u16).collect()),
        DType::U32 => Buffer::U32(v.iter().map(|&x| x.to_f32() as u32).collect()),
        DType::U64 => Buffer::U64(v.iter().map(|&x| x.to_f64() as u64).collect()),
        DType::F16 => Buffer::F16(v.to_vec()),
        DType::F32 => Buffer::F32(v.iter().map(|&x| x.to_f32()).collect()),
        DType::F64 => Buffer::F64(v.iter().map(|&x| x.to_f64()).collect()),
        DType::C64 => Buffer::C64(v.iter().map(|&x| C64::new(x.to_f32(), 0.0)).collect()),
        DType::C128 => Buffer::C128(v.iter().map(|&x| C128::new(x.to_f64(), 0.0)).collect()),
    }
}

fn cast_from_bool(v: &[bool], dst: DType) -> Buffer {
    match dst {
        DType::Bool => Buffer::Bool(v.to_vec()),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => Buffer::I8(v.iter().map(|&x| x as i8).collect()),
        DType::I16 => Buffer::I16(v.iter().map(|&x| x as i16).collect()),
        DType::I32 => Buffer::I32(v.iter().map(|&x| x as i32).collect()),
        DType::I64 => Buffer::I64(v.iter().map(|&x| x as i64).collect()),
        DType::U8 => Buffer::U8(v.iter().map(|&x| x as u8).collect()),
        DType::U16 => Buffer::U16(v.iter().map(|&x| x as u16).collect()),
        DType::U32 => Buffer::U32(v.iter().map(|&x| x as u32).collect()),
        DType::U64 => Buffer::U64(v.iter().map(|&x| x as u64).collect()),
        DType::F16 => Buffer::F16(v.iter().map(|&x| if x { f16::from_f32(1.0) } else { f16::ZERO }).collect()),
        DType::F32 => Buffer::F32(v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect()),
        DType::F64 => Buffer::F64(v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect()),
        DType::C64 => Buffer::C64(v.iter().map(|&x| C64::new(if x { 1.0 } else { 0.0 }, 0.0)).collect()),
        DType::C128 => Buffer::C128(v.iter().map(|&x| C128::new(if x { 1.0 } else { 0.0 }, 0.0)).collect()),
    }
}

fn cast_from_complex64(v: &[C64], dst: DType) -> Buffer {
    match dst {
        DType::Bool => Buffer::Bool(v.iter().map(|c| c.re != 0.0 || c.im != 0.0).collect()),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        // See `cast_from_float!`'s doc comment (same i32-saturate-then-
        // truncate pipeline applies to the real part here).
        DType::I8 => Buffer::I8(v.iter().map(|c| (c.re as i32) as i8).collect()),
        DType::I16 => Buffer::I16(v.iter().map(|c| (c.re as i32) as i16).collect()),
        DType::I32 => Buffer::I32(v.iter().map(|c| c.re as i32).collect()),
        DType::I64 => Buffer::I64(v.iter().map(|c| c.re as i64).collect()),
        DType::U8 => Buffer::U8(v.iter().map(|c| (c.re as i32) as u8).collect()),
        DType::U16 => Buffer::U16(v.iter().map(|c| (c.re as i32) as u16).collect()),
        DType::U32 => Buffer::U32(v.iter().map(|c| c.re as u32).collect()),
        DType::U64 => Buffer::U64(v.iter().map(|c| c.re as u64).collect()),
        DType::F16 => Buffer::F16(v.iter().map(|c| f16::from_f32(c.re)).collect()),
        DType::F32 => Buffer::F32(v.iter().map(|c| c.re).collect()),
        DType::F64 => Buffer::F64(v.iter().map(|c| c.re as f64).collect()),
        DType::C64 => Buffer::C64(v.to_vec()),
        DType::C128 => Buffer::C128(v.iter().map(|c| C128::new(c.re as f64, c.im as f64)).collect()),
    }
}

fn cast_from_complex128(v: &[C128], dst: DType) -> Buffer {
    match dst {
        DType::Bool => Buffer::Bool(v.iter().map(|c| c.re != 0.0 || c.im != 0.0).collect()),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        // See `cast_from_float!`'s doc comment (same i32-saturate-then-
        // truncate pipeline applies to the real part here).
        DType::I8 => Buffer::I8(v.iter().map(|c| (c.re as i32) as i8).collect()),
        DType::I16 => Buffer::I16(v.iter().map(|c| (c.re as i32) as i16).collect()),
        DType::I32 => Buffer::I32(v.iter().map(|c| c.re as i32).collect()),
        DType::I64 => Buffer::I64(v.iter().map(|c| c.re as i64).collect()),
        DType::U8 => Buffer::U8(v.iter().map(|c| (c.re as i32) as u8).collect()),
        DType::U16 => Buffer::U16(v.iter().map(|c| (c.re as i32) as u16).collect()),
        DType::U32 => Buffer::U32(v.iter().map(|c| c.re as u32).collect()),
        DType::U64 => Buffer::U64(v.iter().map(|c| c.re as u64).collect()),
        DType::F16 => Buffer::F16(v.iter().map(|c| f16::from_f64(c.re)).collect()),
        DType::F32 => Buffer::F32(v.iter().map(|c| c.re as f32).collect()),
        DType::F64 => Buffer::F64(v.iter().map(|c| c.re).collect()),
        DType::C64 => Buffer::C64(v.iter().map(|c| C64::new(c.re as f32, c.im as f32)).collect()),
        DType::C128 => Buffer::C128(v.to_vec()),
    }
}

pub type BufferHandle = Arc<Buffer>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cast_int_to_float16() {
        let b = Buffer::I8(vec![1, -2, 127]);
        match b.cast_to(DType::F16) {
            Buffer::F16(v) => {
                assert_eq!(v, vec![f16::from_f32(1.0), f16::from_f32(-2.0), f16::from_f32(127.0)]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn cast_float16_to_float32_and_back() {
        let b = Buffer::F16(vec![f16::from_f32(1.5), f16::from_f32(-0.25)]);
        match b.cast_to(DType::F32) {
            Buffer::F32(v) => assert_eq!(v, vec![1.5f32, -0.25f32]),
            _ => panic!("wrong variant"),
        }
        let b64 = Buffer::F64(vec![1.5, -0.25]);
        match b64.cast_to(DType::F16) {
            Buffer::F16(v) => assert_eq!(v, vec![f16::from_f32(1.5), f16::from_f32(-0.25)]),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn cast_int_to_float() {
        let b = Buffer::I32(vec![1, 2, 3]);
        let c = b.cast_to(DType::F64);
        assert_eq!(c.dtype(), DType::F64);
        match c {
            Buffer::F64(v) => assert_eq!(v, vec![1.0, 2.0, 3.0]),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn cast_numeric_to_bool_is_nonzero_test() {
        let b = Buffer::I32(vec![0, 1, -3, 0]);
        match b.cast_to(DType::Bool) {
            Buffer::Bool(v) => assert_eq!(v, vec![false, true, true, false]),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn cast_bool_to_numeric() {
        let b = Buffer::Bool(vec![true, false, true]);
        match b.cast_to(DType::I64) {
            Buffer::I64(v) => assert_eq!(v, vec![1, 0, 1]),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn cast_real_to_complex_sets_zero_imag() {
        let b = Buffer::F64(vec![2.0, -1.5]);
        match b.cast_to(DType::C128) {
            Buffer::C128(v) => {
                assert_eq!(v[0], C128::new(2.0, 0.0));
                assert_eq!(v[1], C128::new(-1.5, 0.0));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn cast_complex_to_real_takes_real_part() {
        let b = Buffer::C128(vec![C128::new(3.0, 4.0)]);
        match b.cast_to(DType::F64) {
            Buffer::F64(v) => assert_eq!(v, vec![3.0]),
            _ => panic!("wrong variant"),
        }
    }
}
