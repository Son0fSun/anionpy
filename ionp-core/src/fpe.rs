//! Floating-point-error FLAG DETECTION -- the "which of numpy's four IEEE
//! category flags fired" half of the `seterr`/`errstate` subsystem (see
//! `docs/FPE-SUBSYSTEM-SPEC-2026-08-06.md`). This module is pure Rust, zero
//! Python dependency, by the same rule the rest of `ionp-core` follows
//! (`lib.rs`'s module doc: "Only `ionp-py` ever imports `pyo3` or `numpy`").
//!
//! `ionp-py` owns the OTHER half -- the per-thread mode table
//! (`seterr`/`geterr`/`errstate`) and what to DO once a flag is known to
//! have fired (warn/raise/print/call/log) -- because only it can touch
//! `warnings.warn`, raise a Python exception, or call back into a Python
//! object. This module's job stops at "did this call trigger `Divide`/
//! `Over`/`Under`/`Invalid`, yes or no" (real numpy answers the same
//! question via a hardware `fenv.h` flag register; there is no portable
//! stable-Rust equivalent, so this derives the same four-way answer from
//! the operands/result directly -- semantically equivalent for every op
//! this module is scoped to, see each function's doc for the exact rule
//! and what it was measured against).
//!
//! SCOPE: this is deliberately NOT a generic per-ufunc flag engine for
//! every one of ionp's ~130 ufuncs. It covers exactly the ops the
//! differential corpus (`tests/differential/fperr_cases.py`) exercises --
//! `divide`/`true_divide`, `floor_divide`, `remainder`/`mod`, `multiply`,
//! `sqrt`, `log`, and cast (`astype`) -- per this task's scope-control
//! directive ("a correct partial implementation, honestly scoped, beats a
//! sprawling one"). Extending to the remaining ufuncs (`power`, `arcsin`,
//! `exp`, etc. -- all individually measured in the spec doc's section 2
//! table but none exercised by the executable corpus) is future work.

use crate::array::NdArray;
use crate::buffer::Buffer;
use crate::dtype::DType;

/// One of numpy's four IEEE-754-exception categories (`FE_DIVBYZERO`,
/// `FE_OVERFLOW`, `FE_UNDERFLOW`, `FE_INVALID`), i.e. exactly the four keys
/// of `numpy.geterr()`'s dict. Bit values match numpy's own
/// `UFUNC_FPE_*` C constants (`umath/npy_math_common.h`) -- confirmed live
/// against real numpy 2.5.1's `'call'`-mode callback argument (see
/// `docs/FPE-SUBSYSTEM-SPEC-2026-08-06.md` and this task's own live probe:
/// divide->1, over->2, under->4, invalid->8) -- so `as u8` on this enum is
/// directly usable as the bitmask the `'call'` mode handler expects, no
/// separate lookup table needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FpCategory {
    Divide = 1,
    Over = 2,
    Under = 4,
    Invalid = 8,
}

impl FpCategory {
    /// The `seterr`/`geterr` dict key / kwarg name for this category.
    pub fn key(self) -> &'static str {
        match self {
            FpCategory::Divide => "divide",
            FpCategory::Over => "over",
            FpCategory::Under => "under",
            FpCategory::Invalid => "invalid",
        }
    }
    /// The SHORT phrase used by `'call'`/`'log'` mode and the `NameError`
    /// text for an unregistered `'log'` -- measured live against real
    /// numpy 2.5.1 (this task's own probe, not carried over from a prior
    /// measurement pass): `"divide by zero"`, `"invalid value"`,
    /// `"overflow"`, `"underflow"`.
    pub fn short_phrase(self) -> &'static str {
        match self {
            FpCategory::Divide => "divide by zero",
            FpCategory::Over => "overflow",
            FpCategory::Under => "underflow",
            FpCategory::Invalid => "invalid value",
        }
    }
    /// The bitmask numpy's `'call'` mode hands the registered callback as
    /// its second (`flag: int`) argument.
    pub fn bit(self) -> u8 {
        self as u8
    }
}

/// Every one of the four categories, in the fixed order `geterr()`'s dict
/// uses (`{'divide', 'over', 'under', 'invalid'}` -- measured order in the
/// spec doc's section 4/6).
pub const ALL_CATEGORIES: [FpCategory; 4] =
    [FpCategory::Divide, FpCategory::Over, FpCategory::Under, FpCategory::Invalid];

/// Convert one array's elements to `f64` for flag-detection purposes only
/// (never for producing an actual arithmetic VALUE -- `binary_op`/
/// `math_unary_op` already computed the real, correctly-typed result
/// before this function is ever called; this is a second, throwaway,
/// coarser view used only to answer "is this input/output zero, negative,
/// infinite, or NaN"). Complex buffers are intentionally excluded (real
/// component only would be a silent wrong answer for a complex-specific
/// question) -- callers needing complex detection use `is_complex_zero`/
/// `is_complex_finite_nonzero` below instead.
fn to_f64_lossy(v: &NdArray) -> Vec<f64> {
    let c = v.to_contiguous();
    match c.buffer() {
        Buffer::Bool(b) => b.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect(),
        Buffer::I8(b) => b.iter().map(|&x| x as f64).collect(),
        Buffer::I16(b) => b.iter().map(|&x| x as f64).collect(),
        Buffer::I32(b) => b.iter().map(|&x| x as f64).collect(),
        Buffer::I64(b) => b.iter().map(|&x| x as f64).collect(),
        Buffer::U8(b) => b.iter().map(|&x| x as f64).collect(),
        Buffer::U16(b) => b.iter().map(|&x| x as f64).collect(),
        Buffer::U32(b) => b.iter().map(|&x| x as f64).collect(),
        Buffer::U64(b) => b.iter().map(|&x| x as f64).collect(),
        Buffer::F16(b) => b.iter().map(|&x| x.to_f64()).collect(),
        Buffer::F32(b) => b.iter().map(|&x| x as f64).collect(),
        Buffer::F64(b) => b.clone(),
        Buffer::C64(_) | Buffer::C128(_) => {
            unreachable!("to_f64_lossy: complex buffers use the dedicated complex helpers")
        }
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!(
            "to_f64_lossy: fp-error-flag detection has no meaning for string dtypes; \
             callers must not reach this on S/U (numpy has no fp-exception concept for strings either)"
        ),
    }
}

/// (real, imaginary) parts as `f64`, for EITHER a complex array (imaginary
/// part read straight off the buffer) or a real one (imaginary part is
/// implicitly zero -- matches numpy's own real/complex mixed-dtype divide,
/// which promotes the real side to complex before dividing). Two operands
/// of a mixed-dtype call (`complex128_array / float32_array`) legitimately
/// reach this function with only ONE of the two actually complex --
/// `complex_parts` used to `unreachable!()` on the real side in exactly
/// that shape (caught live via the differential suite's
/// `divide`/`true_divide`/`__truediv__`/`__rtruediv__` mixed-dtype cases,
/// e.g. `complex128_array / float32_array`, panicking instead of detecting
/// -- this function existing at all is the fix).
fn complex_parts(v: &NdArray) -> Vec<(f64, f64)> {
    let c = v.to_contiguous();
    match c.buffer() {
        Buffer::C64(b) => b.iter().map(|z| (z.re as f64, z.im as f64)).collect(),
        Buffer::C128(b) => b.iter().map(|z| (z.re, z.im)).collect(),
        _ => to_f64_lossy(v).into_iter().map(|re| (re, 0.0)).collect(),
    }
}

/// `divide`/`floor_divide`/`divmod` share one detection rule -- NOT
/// `remainder`/`mod`, which is dtype-gated differently, see
/// `detect_remainder` below for the split and the live measurement that
/// forced it (task #56, 2026-08-07: `numpy 2.5.1` real probe, not carried
/// forward from an earlier pass). Elementwise, a zero RHS is either
/// `Divide` (the common case: `1.0/0.0`, int `1//0`, `int8(1)` divmod
/// `int8(0)`) or, ONLY when the numerator is ALSO zero, either `Invalid`
/// (true_divide always; floor_divide/divmod when the promoted OUTPUT
/// dtype is floating) or `Divide` again (floor_divide/divmod when the
/// promoted output dtype is integer -- measured live: `np.floor_divide(
/// np.array([0]), np.array([0]))` and `np.divmod(np.array([0]),
/// np.array([0]))` both warn `"divide by zero"`, NOT `"invalid value"`,
/// while the float-dtype 0/0 shape of the exact same two ops warns
/// `"invalid value"`; `true_divide`'s own 0/0 is `"invalid value"`
/// unconditionally because true_divide's output is always floating
/// regardless of input dtype). `out_is_floating` is the caller-supplied
/// answer to "is the promoted/loop OUTPUT dtype floating" -- MUST be the
/// output dtype, not either input's raw dtype (measured live: a mixed
/// `int32 // float64` 0/0 promotes to float and follows the float rule,
/// so passing an input dtype here would silently mis-classify every mixed
/// -dtype call). Complex operands keep the OLD unconditional-`Invalid`-at-
/// 0/0 rule below, unaffected by `out_is_floating` (complex `floor_divide`/
/// `divmod` do not exist as loops in real numpy, so the only complex
/// caller of this function is `true_divide`, whose 0/0 is unconditionally
/// `Invalid` at every dtype anyway -- `out_is_floating` would be `true`
/// for every real caller here regardless). At most one category is
/// reported per call (numpy itself only ever warns once per API call
/// regardless of how many elements in a broadcast trigger it) -- the
/// 0/0 shape (`Invalid` or integer-`Divide` per the rule above) wins if
/// ANY element hits it, plain `Divide` otherwise if ANY element hits the
/// x/0 shape (checked in that order so a mixed array with both shapes
/// reports the more specific one, matching real numpy's own
/// per-element-then-OR'd flag register semantics).
pub fn detect_divide_family(a: &NdArray, b: &NdArray, out_is_floating: bool) -> Option<FpCategory> {
    // An empty operand means the broadcast result has zero elements -- no
    // scalar division ever actually happens, so no flag can fire (matches
    // real numpy: an empty-array divide never warns). Guard this BEFORE
    // the `i % av.len()`-style indexing below, which would otherwise panic
    // (`remainder with a divisor of zero`) the moment either side is a
    // genuinely empty (0-length, not 0-d) array -- caught live via
    // `np.unwrap` on a length-1 axis, whose internal `diff` produces an
    // empty array that then gets divided/modded against a scalar.
    if a.size() == 0 || b.size() == 0 {
        return None;
    }
    if a.dtype().is_complex() || b.dtype().is_complex() {
        let av = complex_parts(a);
        let bv = complex_parts(b);
        let n = av.len().max(bv.len());
        let mut saw_divide = false;
        for i in 0..n {
            let (are, aim) = av[if av.len() == 1 { 0 } else { i % av.len() }];
            let (bre, bim) = bv[if bv.len() == 1 { 0 } else { i % bv.len() }];
            if bre == 0.0 && bim == 0.0 {
                if are == 0.0 && aim == 0.0 {
                    return Some(FpCategory::Invalid);
                }
                saw_divide = true;
            }
        }
        return if saw_divide { Some(FpCategory::Divide) } else { None };
    }
    let av = to_f64_lossy(a);
    let bv = to_f64_lossy(b);
    let n = av.len().max(bv.len());
    let mut saw_divide = false;
    for i in 0..n {
        let x = av[if av.len() == 1 { 0 } else { i % av.len() }];
        let y = bv[if bv.len() == 1 { 0 } else { i % bv.len() }];
        if y == 0.0 {
            if x == 0.0 && out_is_floating {
                return Some(FpCategory::Invalid);
            }
            saw_divide = true;
        }
    }
    if saw_divide { Some(FpCategory::Divide) } else { None }
}

/// `remainder`/`mod`'s OWN detection rule -- deliberately NOT a call to
/// `detect_divide_family` above, because the two diverge on FLOAT operands
/// in a way `out_is_floating` alone cannot express: `detect_divide_family`
/// still reports `Divide` for a floating x/0 (matching `floor_divide`/
/// `divmod`'s real behavior at that same shape), but real numpy's
/// `remainder`/`mod` is COMPLETELY SILENT for every floating-dtype input,
/// at every numerator value, including 0/0 -- measured live against numpy
/// 2.5.1 (task #56's original defect measurement, re-confirmed by this
/// task's own sweep across int8/16/32/64, uint8/16/32/64, float16/32/64,
/// scalar-vs-array, and `out=`): `np.remainder(array([1.,2.]),
/// array([0.,0.]))` and `np.remainder(array([0.]), array([0.]))` both
/// warn NOTHING, while the identical shapes at any INTEGER dtype (signed
/// or unsigned, any width) warn `"divide by zero encountered in
/// remainder"` -- never `"invalid value"`, even at 0/0 (measured:
/// `np.remainder(array([0]), array([0]))` is `"divide by zero"`, not
/// `"invalid value"` -- integer remainder-by-zero has no `Invalid` case at
/// all, unlike `floor_divide`/`divmod`'s dtype-gated 0/0 rule above).
/// `out_is_floating` MUST be the promoted/loop OUTPUT dtype, same
/// requirement and same reasoning as `detect_divide_family`'s own doc
/// comment (a mixed int/float `remainder` promotes to float and is
/// therefore silent, matching the float rule, not the int one).
pub fn detect_remainder(a: &NdArray, b: &NdArray, out_is_floating: bool) -> Option<FpCategory> {
    if out_is_floating {
        return None;
    }
    // Same empty-operand guard as `detect_divide_family` -- see its doc.
    if a.size() == 0 || b.size() == 0 {
        return None;
    }
    let av = to_f64_lossy(a);
    let bv = to_f64_lossy(b);
    let n = av.len().max(bv.len());
    for i in 0..n {
        let y = bv[if bv.len() == 1 { 0 } else { i % bv.len() }];
        if y == 0.0 {
            return Some(FpCategory::Divide);
        }
    }
    None
}

/// `multiply`'s `Over`/`Under` detection: elementwise, both operands
/// finite and nonzero, but the (already-computed, real, correctly-typed)
/// output is non-finite (`Over`) or exactly zero / subnormal (`Under`).
/// Checked against the OUTPUT buffer's own native type (not a lossy f64
/// round-trip of it) so a float32 overflow/underflow that is invisible at
/// f64 width (e.g. `3e38f32 * 10f32` is very much finite as an f64
/// product) is still caught correctly -- only the two INPUT operands go
/// through the lossy f64 finite/nonzero check, which is precision-
/// insensitive (`== 0.0` / `is_finite()` survive widening losslessly for
/// this purpose).
pub fn detect_multiply_over_under(a: &NdArray, b: &NdArray, out: &NdArray) -> Option<FpCategory> {
    // Same empty-operand guard as `detect_divide_family` -- see its doc
    // comment for why this must run before any `i % len()`-style indexing.
    if a.size() == 0 || b.size() == 0 || out.size() == 0 {
        return None;
    }
    if a.dtype().is_complex() || b.dtype().is_complex() || out.dtype().is_complex() {
        return detect_complex_multiply_over(a, b, out);
    }
    let av = to_f64_lossy(a);
    let bv = to_f64_lossy(b);
    let n = av.len().max(bv.len()).max(out_len(out));
    let mut saw_over = false;
    let mut saw_under = false;
    let out_finite_zero = out_finite_and_zero_or_subnormal(out);
    let out_infinite = out_is_infinite(out);
    for i in 0..n {
        let x = av[if av.len() == 1 { 0 } else { i % av.len() }];
        let y = bv[if bv.len() == 1 { 0 } else { i % bv.len() }];
        if !x.is_finite() || !y.is_finite() || x == 0.0 || y == 0.0 {
            continue;
        }
        let oi = if out_len(out) == 1 { 0 } else { i % out_len(out) };
        if out_infinite[oi] {
            saw_over = true;
        } else if out_finite_zero[oi] {
            saw_under = true;
        }
    }
    if saw_over {
        Some(FpCategory::Over)
    } else if saw_under {
        Some(FpCategory::Under)
    } else {
        None
    }
}

fn out_len(out: &NdArray) -> usize {
    out.size().max(1)
}

fn out_is_infinite(out: &NdArray) -> Vec<bool> {
    let c = out.to_contiguous();
    match c.buffer() {
        Buffer::F16(b) => b.iter().map(|x| x.is_infinite()).collect(),
        Buffer::F32(b) => b.iter().map(|x| x.is_infinite()).collect(),
        Buffer::F64(b) => b.iter().map(|x| x.is_infinite()).collect(),
        _ => vec![false; out.size()],
    }
}

/// `true` for an element that is finite, nonzero-input-derived, but landed
/// on exactly zero or a subnormal (denormalized, i.e. `!is_normal()` while
/// still nonzero) magnitude -- numpy's `under` flag condition. Scoped to
/// what the corpus needs (`1e-38f32 * 1e-38f32` flushes clean to `0.0`);
/// the subnormal (nonzero-but-denormal) half of this check is included on
/// the strength of matching numpy's documented `under` semantics but was
/// not separately re-verified live this pass (the corpus's own
/// `underflow_multiply_f32` case is a default-mode negative control, never
/// exercised through a non-default `under` setting -- see the corpus
/// module doc and `docs/FPE-SUBSYSTEM-SPEC-2026-08-06.md` section 5's
/// explicit flag to that effect).
fn out_finite_and_zero_or_subnormal(out: &NdArray) -> Vec<bool> {
    let c = out.to_contiguous();
    match c.buffer() {
        Buffer::F16(b) => b.iter().map(|x| x.is_finite() && (*x == half::f16::ZERO || !x.is_normal())).collect(),
        Buffer::F32(b) => b.iter().map(|x| x.is_finite() && (*x == 0.0 || !x.is_normal())).collect(),
        Buffer::F64(b) => b.iter().map(|x| x.is_finite() && (*x == 0.0 || !x.is_normal())).collect(),
        _ => vec![false; out.size()],
    }
}

fn detect_complex_multiply_over(a: &NdArray, b: &NdArray, out: &NdArray) -> Option<FpCategory> {
    let oc = out.to_contiguous();
    let out_infinite: Vec<bool> = match oc.buffer() {
        Buffer::C64(v) => v.iter().map(|z| !z.re.is_finite() || !z.im.is_finite()).collect(),
        Buffer::C128(v) => v.iter().map(|z| !z.re.is_finite() || !z.im.is_finite()).collect(),
        _ => vec![false; out.size()],
    };
    let _ = (a, b);
    if out_infinite.iter().any(|&x| x) { Some(FpCategory::Over) } else { None }
}

/// `sqrt`'s domain check: real dtypes only -- `Invalid` iff any element is
/// strictly negative (and not itself already NaN, which propagates
/// silently with no domain warning, matching real numpy). Complex `sqrt`
/// has a defined branch value everywhere (spec doc section 5) and this
/// function is never called for a complex operand.
pub fn detect_sqrt(a: &NdArray) -> Option<FpCategory> {
    if a.dtype().is_complex() {
        return None;
    }
    let av = to_f64_lossy(a);
    if av.iter().any(|&x| x < 0.0) { Some(FpCategory::Invalid) } else { None }
}

/// `log`'s two-way domain check, per the spec doc's measured table:
/// exactly-zero input is filed under `Divide` ("`log` at zero is filed
/// under `divide`, not `invalid`"), strictly-negative input under
/// `Invalid`. `Invalid` is checked first only for determinism when BOTH
/// shapes appear in the same call (not the corpus's own single-element
/// cases, which never mix them) -- unmeasured tie-break, flagged here
/// rather than silently assumed.
pub fn detect_log(a: &NdArray) -> Option<FpCategory> {
    if a.dtype().is_complex() {
        return None;
    }
    let av = to_f64_lossy(a);
    if av.iter().any(|&x| x < 0.0) {
        return Some(FpCategory::Invalid);
    }
    if av.iter().any(|&x| x == 0.0) {
        return Some(FpCategory::Divide);
    }
    None
}

/// Cast (`astype`) overflow: `Invalid` iff `to` is an integer dtype and any
/// finite `from` element is out of `to`'s representable range, OR is
/// NaN/infinite (numpy: `RuntimeWarning: invalid value encountered in
/// cast`, spec doc section 2's `cast_overflow_i32` row -- "cast overflow is
/// a distinct message shape (`in cast`, not `in <ufunc-name>`)"). Only
/// float-source -> integer-dest is checked (the corpus's own shape,
/// `float64 -> int32`); integer -> integer narrowing wraparound is
/// deliberately NOT flagged here (spec doc section 5: integer overflow is
/// never flagged by numpy at any setting -- extending this check to
/// int->int would be a regression, not a feature).
pub fn detect_cast_overflow(from: &NdArray, to: DType) -> Option<FpCategory> {
    if !to.is_integer() || !from.dtype().is_floating() {
        return None;
    }
    let (lo, hi) = int_dtype_f64_bounds(to);
    let av = to_f64_lossy(from);
    if av.iter().any(|&x| !x.is_finite() || x < lo || x > hi) {
        Some(FpCategory::Invalid)
    } else {
        None
    }
}

fn int_dtype_f64_bounds(dt: DType) -> (f64, f64) {
    match dt {
        DType::I8 => (i8::MIN as f64, i8::MAX as f64),
        DType::I16 => (i16::MIN as f64, i16::MAX as f64),
        DType::I32 => (i32::MIN as f64, i32::MAX as f64),
        DType::I64 => (i64::MIN as f64, i64::MAX as f64),
        DType::U8 => (u8::MIN as f64, u8::MAX as f64),
        DType::U16 => (u16::MIN as f64, u16::MAX as f64),
        DType::U32 => (u32::MIN as f64, u32::MAX as f64),
        DType::U64 => (u64::MIN as f64, u64::MAX as f64),
        _ => (f64::MIN, f64::MAX),
    }
}
