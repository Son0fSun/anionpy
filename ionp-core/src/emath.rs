//! `numpy.emath` (a.k.a. `numpy.lib.scimath`) -- the branch-cut-aware
//! variants of `sqrt`/`log`/`log2`/`log10`/`logn`/`power`/`arccos`/
//! `arcsin`/`arctanh` that promote to complex instead of returning NaN
//! (`np.emath.sqrt(-1) == 1j`, where `np.sqrt(-1) == nan`).
//!
//! This is deliberately NOT a reimplementation of the transcendental math:
//! `ufunc.rs`'s `math_unary_op`/`math_binary_op` already compute correct
//! complex results for every op this module needs (`Sqrt`/`Log`/`Log2`/
//! `Log10`/`Arcsin`/`Arccos`/`Arctanh`/`Power` all have real complex loops
//! there, verified live: `math_unary_has_complex_loop` returns `true` for
//! all seven, and `math_binary_out_dtype` special-cases `Power` as the one
//! `MathBinaryOp` with a complex loop). The entire job of this module is
//! numpy's own preprocessing step: decide whether the input needs
//! promoting to complex BEFORE calling the existing ufunc, exactly the way
//! `numpy/lib/_scimath_impl.py`'s `_fix_real_lt_zero` / `_fix_int_lt_zero`
//! / `_fix_real_abs_gt_1` / `_tocomplex` do (read directly from numpy
//! 2.5.1's installed source, not guessed):
//!
//!   - `_fix_real_lt_zero(x)`: if `x`'s dtype is real (not already complex)
//!     AND any element is negative, cast the WHOLE array to complex (never
//!     a per-element promotion); otherwise return `x` unchanged (dtype and
//!     all).
//!   - `_fix_int_lt_zero(x)`: same trigger, but casts to float64 instead of
//!     complex (used for `power`'s exponent operand, so `int ** negative`
//!     doesn't hit ionp's existing "Integers to negative integer powers are
//!     not allowed" `ValueError` -- verified live to already match numpy's
//!     own message). A float dtype input is left as its own width
//!     (`np.array([-1,2], dtype=float32) * 1.0` stays float32 under NEP 50
//!     weak-scalar promotion, verified live); only integer dtypes actually
//!     change width, to float64 always (also verified live, independent of
//!     the integer's own width). Bool can never trigger this (no bool
//!     value is negative), so no bool-specific branch is needed.
//!   - `_fix_real_abs_gt_1(x)`: same shape as `_fix_real_lt_zero`, but the
//!     trigger is `abs(x) > 1` (used by `arccos`/`arcsin`, whose real
//!     domain is `[-1, 1]`; `arctanh`'s real domain is `(-1, 1)` but numpy
//!     uses the SAME `>1` trigger function for it too -- verified directly
//!     from the installed source above, not an ionp guess -- so `x == 1`
//!     or `x == -1` for `arctanh` is NOT promoted here and instead reaches
//!     the real `+-inf` ionp already produces, matching numpy).
//!   - `_tocomplex(arr)`: picks complex64 for byte/ubyte/short/ushort/
//!     single (ionp: I8/U8/I16/U16/F32) inputs, complex128 for everything
//!     else (verified live across every ionp-representable dtype: I32,
//!     I64, F16, F64, and Bool -- though bool can never actually reach
//!     this function, see above -- all map to complex128; U32/U64 have no
//!     numpy `_tocomplex` counterpart to check against directly since numpy
//!     itself has no uint32/64 `_fix_*` test coverage in scimath, but they
//!     are the same "not in the narrow list" case as I32/I64 by the same
//!     rule, so complex128 is the correct read of the rule, not a guess by
//!     omission).
//!
//! `logn(n, x) = log(x) / log(n)` (numpy's own definition, read from
//! source) applies `_fix_real_lt_zero` to BOTH operands independently, then
//! divides -- `ufunc.rs`'s existing `binary_op(Divide, ..)` already handles
//! every real/complex dtype combination the two independent promotions can
//! produce (complex64/complex128/f16/f32/f64, mixed), so no new promotion
//! logic is needed there either.
//!
//! An input already complex-dtyped is passed straight through *unchanged*
//! here (no re-inspection of its elements): numpy's own `isreal(x)` is an
//! ELEMENTWISE test (true only for elements whose imaginary part is
//! exactly 0) that can, in principle, still fire on a negative-real-valued
//! element of an already-complex array, feeding it back through
//! `_tocomplex`, but `_tocomplex` on an already-complex array is a same-
//! dtype identity cast (complex64 -> complex64, complex128 -> complex128,
//! matching the mapping above) -- so the net effect on an already-complex
//! input is always a no-op, and skipping the re-inspection entirely is
//! behavior-preserving, not an approximation.

use crate::array::{NdArray, Order};
use crate::buffer::Buffer;
use crate::dtype::DType;
use crate::error::IonpError;
use crate::ufunc::{self, BinaryOp, MathBinaryOp, MathUnaryOp, UnaryOp};

/// `_tocomplex`'s dtype table, transcribed from `numpy/lib/_scimath_impl.py`
/// (see this module's doc comment for the exact verification).
fn tocomplex_target(d: DType) -> DType {
    match d {
        DType::I8 | DType::U8 | DType::I16 | DType::U16 | DType::F32 => DType::C64,
        _ => DType::C128,
    }
}

/// A 0-d real scalar array, used only as the broadcast operand of a
/// comparison (`x < 0`, `abs(x) > 1`) -- never returned to a caller, so its
/// own dtype (`F64`) is irrelevant beyond "comparable to anything via the
/// existing `promote_dtype` machinery `binary_op` already applies", which
/// is exactly the same promotion every other mixed-dtype comparison in
/// this crate already relies on.
fn scalar_f64(v: f64) -> NdArray {
    NdArray::from_buffer(Buffer::F64(vec![v]), vec![], Order::C)
        .expect("0-d F64 buffer construction cannot fail")
}

/// `any(isreal(x) & (x < 0))`, restricted to real (non-complex) `x` --
/// complex `x` is handled by the caller before this is ever invoked (see
/// module doc comment on why re-inspecting an already-complex array is
/// unnecessary).
fn any_negative(a: &NdArray) -> Result<bool, IonpError> {
    let mask = ufunc::binary_op(BinaryOp::Less, a, &scalar_f64(0.0))?;
    Ok(ufunc::any_true(&mask))
}

/// `any(isreal(x) & (abs(x) > 1))`, same real-only precondition as
/// `any_negative`.
fn any_abs_gt_one(a: &NdArray) -> Result<bool, IonpError> {
    let abs = ufunc::unary_op(UnaryOp::Absolute, a)?;
    let mask = ufunc::binary_op(BinaryOp::Greater, &abs, &scalar_f64(1.0))?;
    Ok(ufunc::any_true(&mask))
}

/// `_fix_real_lt_zero`: promote the whole array to complex if it is real
/// and has any negative element; otherwise return it unchanged.
fn fix_real_lt_zero(a: &NdArray) -> Result<NdArray, IonpError> {
    if a.dtype().is_complex() {
        return Ok(a.clone());
    }
    if any_negative(a)? {
        Ok(a.cast_to(tocomplex_target(a.dtype())))
    } else {
        Ok(a.clone())
    }
}

/// `_fix_real_abs_gt_1`: same shape as `fix_real_lt_zero`, `abs(x) > 1`
/// trigger (used by `arccos`/`arcsin`/`arctanh`).
fn fix_real_abs_gt_1(a: &NdArray) -> Result<NdArray, IonpError> {
    if a.dtype().is_complex() {
        return Ok(a.clone());
    }
    if any_abs_gt_one(a)? {
        Ok(a.cast_to(tocomplex_target(a.dtype())))
    } else {
        Ok(a.clone())
    }
}

/// `_fix_int_lt_zero`: promote an integer array to float64 if it has any
/// negative element (bool can never trigger this -- no bool value is
/// negative -- so it always passes through unchanged, matching numpy's
/// `_fix_int_lt_zero(np.array([True, False]))` returning `bool` unchanged,
/// verified live); a float array is returned unchanged regardless (NEP 50
/// weak-scalar `x * 1.0` never changes a float array's width, verified
/// live for f16/f32/f64); a complex array is returned unchanged (same
/// no-op argument as `fix_real_lt_zero`'s complex short-circuit).
fn fix_int_lt_zero(a: &NdArray) -> Result<NdArray, IonpError> {
    if a.dtype().is_complex() || a.dtype().is_floating() || a.dtype() == DType::Bool {
        return Ok(a.clone());
    }
    if any_negative(a)? {
        Ok(a.cast_to(DType::F64))
    } else {
        Ok(a.clone())
    }
}

/// Re-apply `order='K'` output layout, voted from the ORIGINAL (pre-`_fix_*`)
/// operand(s), not the post-promotion ones. This matters because `cast_to`
/// (invoked by `fix_real_lt_zero`/`fix_int_lt_zero`/`fix_real_abs_gt_1` only
/// when promotion actually triggers) always forces C-contiguous when it does
/// a real copy -- if we voted on the post-cast operand instead, a promoted
/// call would silently lose the caller's original layout even though real
/// numpy's own `astype(complex)` internally defaults to `order='K'` and so
/// preserves it. Voting on the pre-fix operand(s) sidesteps that asymmetry
/// entirely and matches numpy's observed behavior in both the promoted and
/// non-promoted cases.
fn apply_k_order(computed: NdArray, originals: &[&NdArray]) -> NdArray {
    let ndim = computed.shape().len();
    let pairs: Vec<(&[usize], &[isize])> = originals.iter().map(|a| (a.shape(), a.strides())).collect();
    let perm = NdArray::multi_sorted_stride_perm(ndim, &pairs);
    computed.relayout_by_perm(&perm)
}

pub fn emath_sqrt(a: &NdArray) -> Result<NdArray, IonpError> {
    let out = ufunc::math_unary_op(MathUnaryOp::Sqrt, &fix_real_lt_zero(a)?)?;
    Ok(apply_k_order(out, &[a]))
}

pub fn emath_log(a: &NdArray) -> Result<NdArray, IonpError> {
    let out = ufunc::math_unary_op(MathUnaryOp::Log, &fix_real_lt_zero(a)?)?;
    Ok(apply_k_order(out, &[a]))
}

pub fn emath_log2(a: &NdArray) -> Result<NdArray, IonpError> {
    let out = ufunc::math_unary_op(MathUnaryOp::Log2, &fix_real_lt_zero(a)?)?;
    Ok(apply_k_order(out, &[a]))
}

pub fn emath_log10(a: &NdArray) -> Result<NdArray, IonpError> {
    let out = ufunc::math_unary_op(MathUnaryOp::Log10, &fix_real_lt_zero(a)?)?;
    Ok(apply_k_order(out, &[a]))
}

/// `logn(n, x) = log(x) / log(n)`, each operand independently fixed --
/// matches `numpy/lib/_scimath_impl.py`'s `logn`, argument order and all
/// (`n` is fixed too, not just `x`: `np.emath.logn(-2, 4)` is a real numpy
/// call that promotes the BASE to complex).
pub fn emath_logn(n: &NdArray, x: &NdArray) -> Result<NdArray, IonpError> {
    let x_fixed = fix_real_lt_zero(x)?;
    let n_fixed = fix_real_lt_zero(n)?;
    let log_x = ufunc::math_unary_op(MathUnaryOp::Log, &x_fixed)?;
    let log_n = ufunc::math_unary_op(MathUnaryOp::Log, &n_fixed)?;
    let out = ufunc::binary_op(BinaryOp::Divide, &log_x, &log_n)?;
    Ok(apply_k_order(out, &[n, x]))
}

/// `power(x, p) = x ** p`, with `x` fixed via `fix_real_lt_zero` and `p`
/// fixed via `fix_int_lt_zero` (numpy's own asymmetric treatment of base
/// vs. exponent -- read directly from source, not an ionp simplification).
pub fn emath_power(x: &NdArray, p: &NdArray) -> Result<NdArray, IonpError> {
    let x_fixed = fix_real_lt_zero(x)?;
    let p_fixed = fix_int_lt_zero(p)?;
    let out = ufunc::math_binary_op(MathBinaryOp::Power, &x_fixed, &p_fixed)?;
    Ok(apply_k_order(out, &[x, p]))
}

pub fn emath_arccos(a: &NdArray) -> Result<NdArray, IonpError> {
    let out = ufunc::math_unary_op(MathUnaryOp::Arccos, &fix_real_abs_gt_1(a)?)?;
    Ok(apply_k_order(out, &[a]))
}

pub fn emath_arcsin(a: &NdArray) -> Result<NdArray, IonpError> {
    let out = ufunc::math_unary_op(MathUnaryOp::Arcsin, &fix_real_abs_gt_1(a)?)?;
    Ok(apply_k_order(out, &[a]))
}

pub fn emath_arctanh(a: &NdArray) -> Result<NdArray, IonpError> {
    let out = ufunc::math_unary_op(MathUnaryOp::Arctanh, &fix_real_abs_gt_1(a)?)?;
    Ok(apply_k_order(out, &[a]))
}
