//! The three in-place storage primitives real numpy's `PyArray_Round`
//! (`numpy/_core/src/multiarray/calculation.c`) reaches for that ionp did
//! not previously expose, plus numpy's own `power_of_ten` table.
//!
//! WHY THESE THREE AND NOTHING MORE
//! --------------------------------
//! `np.round` is not a ufunc. It is a small C driver that CALLS ufuncs
//! (`multiply`/`divide`/`rint`) and, on two of its four branches, does
//! something no ufunc can express:
//!
//!   * the integer-with-non-negative-`decimals` branch is a plain
//!     `PyArray_CopyInto(out, a)` -- a CAST, not arithmetic, with
//!     `'same_kind'` enforcement and its own distinct error wording
//!     (`"Cannot cast array data from ..."`, NOT the ufunc spelling
//!     `"Cannot cast ufunc 'x' output from ..."`); and
//!   * the complex branch rounds the real and imaginary parts separately
//!     and reassembles them by ATTRIBUTE ASSIGNMENT (`arr.real = ...`,
//!     `arr.imag = ...`), which is where the observable
//!     `"Cannot set imaginary part of non-complex array."` TypeError comes
//!     from when a complex input is given a non-complex `out=`.
//!
//! Everything else `round` needs -- the `multiply`/`rint`/`divide`
//! sequence, the `out=` dtype/shape enforcement, the UFuncTypeError
//! wording -- ionp's existing ufunc objects already produce byte-for-byte
//! (verified live, `/tmp/mg_rnd3.py`: all three of numpy's
//! `Cannot cast ufunc 'multiply'/'divide'/'rint' output from dtype('X') to
//! dtype('Y') with casting rule 'same_kind'` messages came back identical
//! from `ionp.multiply`/`ionp.divide`/`ionp.rint` with no work at all).
//! So the driver itself lives in `ionp/_round_compose.py` as a
//! whole-array-call-only transcription of numpy's C flow, and this module
//! supplies only the parts that layer genuinely cannot reach. Nothing here
//! loops over elements on the Python side; every loop below is Rust.
//!
//! DELIBERATELY NOT `copyto`
//! -------------------------
//! `copy_into` below is `np.copyto(dst, src)` with `casting='same_kind'`
//! and no `where=`. It is NOT exported to Python as `copyto`, because
//! `np.copyto`'s real surface (a `casting=` string with five legal values,
//! a broadcastable `where=` mask, and its own array-likeness rules for
//! `dst`) has not been measured, and shipping a same-named function that
//! implements a subset of a numpy function is exactly the kind of
//! near-miss this project treats as worse than an absence. `copyto` stays
//! an undeclared, absent item; this is its private half.

use crate::buffer::{C128, C64};
use crate::{creation, dtype, ufunc, Buffer, DType, IonpError, NdArray, Order};

/// numpy's `power_of_ten` (static to `calculation.c`), transcribed
/// including its two-regime structure:
///
/// ```c
/// static double power_of_ten(int n) {
///     static const double p10[] = {1e0,1e1,...,1e8};
///     double ret;
///     if (n < 9) ret = p10[n];
///     else { ret = 1e8; while (n-- > 8) ret *= 10.; }
///     return ret;
/// }
/// ```
///
/// The `n >= 9` branch is REPEATED MULTIPLICATION, not `10.0f64.powi(n)`,
/// and the two are not interchangeable in general: `powi`/`powf` are
/// correctly-rounded-ish single operations while the loop accumulates one
/// rounding per step. They happen to agree for every `n` where `10^n` is
/// exactly representable (n <= 22), which is why a naive `10f64.powi(n)`
/// would pass a shallow test and then drift on large `decimals`. Kept
/// literal so it cannot drift.
///
/// `n` is always `>= 1` on every path that reaches here (`decimals == 0`
/// short-circuits to `rint` before any scale factor is computed, and
/// negative `decimals` are negated by the caller first), but `n == 0` is
/// still handled by the table for total-function safety.
///
/// WHY `i64` AND NOT `i32`
/// ----------------------
/// numpy's C parameter is `int`, and its caller negates in `int` too, so
/// `decimals == INT_MIN` negates to itself and indexes `p10[INT_MIN]` --
/// undefined behaviour that is not worth transcribing. What numpy
/// OBSERVABLY does at both C-int extremes was measured directly instead
/// (`/tmp/mg_rnd10.py`, five repeats each, stable every time):
///
/// ```text
/// np.round([1.5,2.0,123.456,0.0,-7.0], -2**31)   -> [nan nan nan nan nan]
/// np.round(..., 2**31-1)                          -> [nan nan nan nan nan]
/// np.round(..., -300)                             -> [ 0.  0.  0.  0. -0.]
/// np.round(..., 300)                              -> unchanged
/// ```
///
/// Taking `|decimals|` in a WIDE type reproduces all four rows: `2**31`
/// and `2**31-1` both drive the loop to +inf (`x/inf -> 0`, `rint`,
/// `0*inf -> nan`, for EVERY element including `0.0`, which is what makes
/// the all-nan row diagnostic); `300` lands on a finite `1e300` and so is
/// a round-trip; `-300` divides into the subnormal floor, rints to
/// +-0.0, and multiplies back to +-0.0 with the sign kept. A `pow(10,
/// decimals)`-based model was considered and FALSIFIED by the `-300` row:
/// it would give `1e-300`, `1.5/1e-300 = 1.5e300`, and hand back `1.5`
/// unchanged instead of `0.0`.
pub fn power_of_ten(n: i64) -> f64 {
    const P10: [f64; 9] = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8];
    if n < 9 {
        // Negative `n` cannot occur (see doc); treat as 1e0 rather than
        // panicking on a negative index.
        if n <= 0 {
            return 1.0;
        }
        return P10[n as usize];
    }
    let mut ret: f64 = 1e8;
    let mut k = n;
    while {
        let cur = k;
        k -= 1;
        cur > 8
    } {
        ret *= 10.0;
        // EARLY EXIT, value-identical, not an approximation: once `ret`
        // is +inf every remaining `ret *= 10.0` is a no-op (inf * 10 ==
        // inf), so stopping here returns exactly what the full loop
        // would. Without it `decimals = 2**31 - 1` -- which numpy
        // ACCEPTS, returning nan -- would spin two billion times. numpy's
        // C loop really does run them; ionp returns the same answer
        // without the wait.
        if ret.is_infinite() {
            break;
        }
    }
    ret
}

/// `PyArray_CopyInto(dst, src)`: cast `src` into `dst`'s dtype and write
/// it through `dst`'s existing storage, broadcasting `src` up to `dst`'s
/// shape. `dst` keeps its identity -- this mutates, it does not allocate.
///
/// The casting rule is `'same_kind'` and the rejection wording is the
/// ARRAY-DATA spelling, not the ufunc spelling. Both halves were measured
/// against numpy 2.5.1 rather than assumed (`/tmp/mg_rnd5.py`):
///
/// ```text
/// np.round(int64([1,200,3]), 2, out=int8)    -> array([1, -56, 3])
/// np.round(int64([1,200,3]), 2, out=uint8)   -> TypeError: Cannot cast array data
///                                               from dtype('int64') to dtype('uint8')
///                                               according to the rule 'same_kind'
/// np.round(int64([1,200,3]), 2, out=bool)    -> same TypeError, dtype('bool')
/// ```
///
/// Note the FIRST line: `int64 -> int8` IS `'same_kind'` (both integer
/// kind), so 200 silently wraps to -56. That is not a bug being copied by
/// accident -- `same_kind` is a KIND check, not a range check, and the
/// wrap is the observable numpy behaviour.
pub fn copy_into(dst: &mut NdArray, src: &NdArray) -> Result<(), IonpError> {
    if !dtype::can_cast(src.dtype(), dst.dtype(), "same_kind") {
        return Err(IonpError::Type(format!(
            // BUG FIXED 2026-08-04: this noun was hardcoded to "array data",
            // which is right for the doc examples above (all 1-d) and wrong
            // for a 0-d source. numpy picks the noun from the SOURCE's ndim
            // -- not the destination's -- the same rule `manip.rs`'s
            // `cast_indices` documents for take/put/choose/repeat. Measured
            // on numpy 2.5.1 via `np.copyto`, whose casting check this
            // function reproduces, and via `ndarray.conj(out)`, which routes
            // its real-dtype path straight through here:
            //
            //   src ()   dst ()    -> "Cannot cast scalar from ..."
            //   src ()   dst (2,)  -> "Cannot cast scalar from ..."   <-- dst
            //                         is 1-d and the noun is STILL "scalar",
            //                         which is what proves it is the source
            //                         that decides.
            //   src (2,) dst (2,)  -> "Cannot cast array data from ..."
            "Cannot cast {} from dtype('{}') to dtype('{}') according to the rule 'same_kind'",
            if src.ndim() == 0 { "scalar" } else { "array data" },
            src.dtype().name(),
            dst.dtype().name()
        )));
    }
    let bc = broadcast_for_assignment(src, dst.shape())?;
    let casted = bc.to_contiguous().cast_to(dst.dtype());
    ufunc::write_out(dst, &casted, None)
}

/// Shape repr in numpy's OWN assignment-error spelling: comma-separated
/// with NO space after the comma, and a trailing comma on a 1-tuple --
/// `(2,3)`, `(6,)`, `()`. This is NOT Python's `repr(tuple)` (which would
/// give `(2, 3)`), and it is NOT the spelling the ufunc broadcast error
/// uses either. Both were read off real numpy output rather than assumed.
fn fmt_shape(shape: &[usize]) -> String {
    let mut s = String::from("(");
    for (i, d) in shape.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&d.to_string());
    }
    if shape.len() == 1 {
        s.push(',');
    }
    s.push(')');
    s
}

/// Broadcast for an ASSIGNMENT (`PyArray_CopyInto` / the `.real=`/`.imag=`
/// setters), which fails with a different message than a ufunc's operand
/// broadcast does. Measured side by side on the same `out=` mistake:
///
/// ```text
/// np.round(arange(6.).reshape(2,3), 0, empty(6))       # ufunc path
///     ValueError: operands could not be broadcast together with shapes (2,3) (6,)
/// np.round(arange(6).reshape(2,3),  2, empty(6,'i8'))  # assignment path
///     ValueError: could not broadcast input array from shape (2,3) into shape (6,)
/// ```
///
/// Same input shapes, same `out=`, two different sentences -- so the
/// wording is part of the observable contract of WHICH branch ran, not
/// incidental. Rewriting the underlying broadcast error rather than
/// pre-checking the shapes keeps the two in sync if `broadcast_to`'s own
/// rules ever change.
fn broadcast_for_assignment(src: &NdArray, shape: &[usize]) -> Result<NdArray, IonpError> {
    // The leading-1 trim below is NOT cosmetic. numpy accepts an assignment
    // source with MORE dimensions than the destination as long as the extra
    // leading axes are all length 1 -- `np.copyto(np.zeros(8), np.ones((1,1)))`
    // succeeds -- whereas a plain `broadcast_to((1,1) -> (8,))` rejects it on
    // rank alone. Trimming first makes ionp accept exactly what numpy
    // accepts, and makes the error below report exactly what numpy reports.
    // (Discovered because reporting alone left 1 of 24 measured cells wrong,
    // in the accept/reject direction rather than the wording one.)
    let reported = reported_src_shape(src.shape(), shape.len());
    let trimmed;
    let effective: &NdArray = if reported.len() == src.ndim() {
        src
    } else {
        // Dropping leading length-1 axes preserves both element count and
        // element order, so this reshape can never itself fail.
        trimmed = src.reshape(&reported)?;
        &trimmed
    };
    creation::broadcast_to(effective, shape).map_err(|_| {
        IonpError::Value(format!(
            "could not broadcast input array from shape {} into shape {}",
            fmt_shape(&reported),
            fmt_shape(shape)
        ))
    })
}

/// The source shape AS NUMPY REPORTS IT in the assignment-broadcast error,
/// which is not always the source's actual shape: numpy drops LEADING
/// length-1 axes until the source has no more dimensions than the
/// destination, then stops. Found 2026-08-04 by a new `ndarray.conj(out)`
/// differential case (`view/2d_offset_contig_one_row/out_bigger_shape`),
/// which is the only reason this was ever visible -- the shape it reports
/// is otherwise invisible to a caller who already knows what they passed.
///
/// Measured on numpy 2.5.1 via `np.copyto` (/tmp/bc.py, 24 cells):
///
/// ```text
///   src            dst      numpy reports
///   (1,5)          (8,)     (5,)      leading 1 dropped
///   (1,1,5)        (8,)     (5,)      both dropped
///   (1,1,1,7)      (8,)     (7,)      all three dropped
///   (1,1,1,7)      (2,4)    (1,7)     stops at dst.ndim == 2, NOT (7,)
///   (5,1)          (8,)     (5,1)     leading axis is 5, nothing to drop
///   (1,5,1)        (8,)     (5,1)     drops the leading 1, then stops
///   (3,)           (2,4)    (3,)      already shorter than dst; no padding
/// ```
///
/// The `(1,1,1,7)` pair is the one that pins the rule down: the same source
/// reports two different shapes depending on the destination's ndim, so the
/// stripping is bounded by `dst_ndim` rather than being "strip all leading
/// ones".
fn reported_src_shape(src_shape: &[usize], dst_ndim: usize) -> Vec<usize> {
    let mut start = 0;
    while src_shape.len() - start > dst_ndim && src_shape[start] == 1 {
        start += 1;
    }
    src_shape[start..].to_vec()
}

/// `arr.real = src` / `arr.imag = src`, the reassembly step of
/// `PyArray_Round`'s complex branch.
///
/// Three behaviours, all measured, none guessed:
///
///  1. `dst` complex -> the named component is overwritten and the OTHER
///     component is preserved bit-for-bit. This is why the implementation
///     reads `dst`'s current contents back before writing: there is no
///     "component view" in ionp: a complex buffer stores whole values, and
///     no stride/offset pair addresses only their real (or only their
///     imaginary) halves. That is a property of the CORE representation and
///     is unaffected by the Python-surface view machinery. (This
///     parenthetical previously read "no `__setitem__`, no true views" --
///     both of which arrived on 2026-08-02. It is corrected rather than
///     deleted because the conclusion still holds for the reason above, and
///     a stale reason invites re-deriving the wrong thing from it.) So the
///     only way to write half of a complex array is to rebuild the whole
///     value and push it through `write_out`.
///  2. `dst` non-complex and `imag == false` -> numpy's `.real` setter on
///     a real array is a plain `PyArray_CopyObject`, i.e. an UNSAFE cast,
///     NOT `same_kind`. So this branch deliberately does not go through
///     `copy_into` (which would enforce `same_kind` and reject e.g.
///     `float64 -> bool`).
///  3. `dst` non-complex and `imag == true` -> the observable TypeError,
///     verbatim including its trailing period:
///     `Cannot set imaginary part of non-complex array.`
///
/// Reached in `round`'s flow only for a complex input; case 2 and case 3
/// are the two halves of `np.round(complex_array, d, out=float_array)`,
/// which numpy leaves PARTIALLY MUTATED (the real part IS written before
/// the imaginary-part assignment raises). That partial write is
/// reproduced, not papered over, because the compose layer performs the
/// same two assignments in the same order.
pub fn set_part(dst: &mut NdArray, src: &NdArray, imag: bool) -> Result<(), IonpError> {
    let (re_dtype, is_complex) = match dst.dtype() {
        DType::C64 => (DType::F32, true),
        DType::C128 => (DType::F64, true),
        _ => (dst.dtype(), false),
    };
    if !is_complex {
        if imag {
            return Err(IonpError::Type(
                "Cannot set imaginary part of non-complex array.".to_string(),
            ));
        }
        // Case 2: unsafe cast, no `same_kind` gate (see doc).
        let bc = broadcast_for_assignment(src, dst.shape())?;
        let casted = bc.to_contiguous().cast_to(dst.dtype());
        return ufunc::write_out(dst, &casted, None);
    }

    let bc = broadcast_for_assignment(src, dst.shape())?
        .to_contiguous()
        .cast_to(re_dtype);
    let cur = dst.to_contiguous();
    let shape = dst.shape().to_vec();
    let combined = match (cur.buffer(), bc.buffer()) {
        (Buffer::C64(cv), Buffer::F32(nv)) => {
            let out: Vec<C64> = cv
                .iter()
                .zip(nv.iter())
                .map(|(c, &x)| if imag { C64::new(c.re, x) } else { C64::new(x, c.im) })
                .collect();
            NdArray::from_buffer(Buffer::C64(out), shape, Order::C)?
        }
        (Buffer::C128(cv), Buffer::F64(nv)) => {
            let out: Vec<C128> = cv
                .iter()
                .zip(nv.iter())
                .map(|(c, &x)| if imag { C128::new(c.re, x) } else { C128::new(x, c.im) })
                .collect();
            NdArray::from_buffer(Buffer::C128(out), shape, Order::C)?
        }
        // Unreachable: `re_dtype` is derived from `dst.dtype()` two lines
        // above and `cast_to` is total, so the pairing above is exhaustive
        // for every complex `dst`. Returned as an error rather than
        // `unreachable!()` so a future dtype addition surfaces as a Python
        // exception instead of a process-killing panic.
        _ => {
            return Err(IonpError::Type(
                "internal: complex component assignment reached a non-complex buffer".to_string(),
            ))
        }
    };
    ufunc::write_out(dst, &combined, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_of_ten_matches_the_table_regime() {
        for n in 1..9i64 {
            assert_eq!(power_of_ten(n), 10f64.powi(n as i32));
        }
    }

    #[test]
    fn power_of_ten_loop_regime_is_repeated_multiplication() {
        // 1e9..=1e22 are exactly representable, so the loop and `powi`
        // must agree there; this pins the loop's starting point (1e8) and
        // its trip count.
        for n in 9..=22i64 {
            assert_eq!(power_of_ten(n), 10f64.powi(n as i32));
        }
        // Far past the representable range the loop saturates to inf,
        // which is what makes `np.round(x, 400)` return nan (inf scale,
        // then inf/inf).
        assert!(power_of_ten(400).is_infinite());
        // The C-int extreme numpy reaches by negating INT_MIN.
        assert!(power_of_ten(2i64.pow(31)).is_infinite());
    }

    #[test]
    fn copy_into_rejects_cross_kind_by_same_kind_rule() {
        let src = NdArray::from_buffer(Buffer::I64(vec![1, 2]), vec![2], Order::C).unwrap();
        let mut dst = NdArray::from_buffer(Buffer::Bool(vec![false, false]), vec![2], Order::C).unwrap();
        let err = copy_into(&mut dst, &src).unwrap_err();
        match err {
            IonpError::Type(m) => assert_eq!(
                m,
                "Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'same_kind'"
            ),
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn set_part_preserves_the_other_component() {
        let mut dst = NdArray::from_buffer(
            Buffer::C128(vec![C128::new(1.0, 2.0), C128::new(3.0, 4.0)]),
            vec![2],
            Order::C,
        )
        .unwrap();
        let src = NdArray::from_buffer(Buffer::F64(vec![9.0, 8.0]), vec![2], Order::C).unwrap();
        set_part(&mut dst, &src, true).unwrap();
        match dst.buffer() {
            Buffer::C128(v) => {
                assert_eq!(v[0], C128::new(1.0, 9.0));
                assert_eq!(v[1], C128::new(3.0, 8.0));
            }
            _ => panic!("dtype changed"),
        }
    }

    #[test]
    fn set_imag_on_real_array_is_the_documented_type_error() {
        let mut dst = NdArray::from_buffer(Buffer::F64(vec![1.0]), vec![1], Order::C).unwrap();
        let src = NdArray::from_buffer(Buffer::F64(vec![2.0]), vec![1], Order::C).unwrap();
        match set_part(&mut dst, &src, true).unwrap_err() {
            IonpError::Type(m) => assert_eq!(m, "Cannot set imaginary part of non-complex array."),
            other => panic!("wrong error: {other:?}"),
        }
    }
}
