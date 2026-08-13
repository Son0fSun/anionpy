//! `numpy.matmul` family: `matmul` (the `@` operator), `vecdot`, `matvec`,
//! `vecmat`. All four are gufuncs with the same shape: two "core"
//! dimensions per operand consumed/produced according to a fixed
//! signature, arbitrary "batch" dimensions to their left broadcast against
//! each other numpy-style.
//!
//! Every error string in this module is built to match real numpy 2.5.1
//! byte-for-byte (verified empirically against the canonical venv
//! interpreter — see the differential sweep in
//! `ionp-py`/`tests/differential` for the harness that checks this). All
//! four raise plain `ValueError` in real numpy (confirmed via
//! `type(e).__module__ + '.' + type(e).__name__`), so callers at the PyO3
//! boundary can map every `Err(String)` from this module straight onto
//! `PyValueError::new_err`.
//!
//! ## Dtype handling
//! numpy resolves the two operands' dtypes via its ordinary array-array
//! promotion rule before dispatch (confirmed empirically: `int32 @ int64
//! -> int64`, `float32 @ float64 -> float64`, `int32 @ float64 ->
//! float64`) -- not a same-dtype-only restriction, despite `np.matmul.
//! types` listing only same-type loops (that list is numpy's *registered
//! ufunc loops*; the ufunc's own type-resolution step still promotes
//! mismatched inputs to the smallest loop both can safely cast to, same
//! as any other ufunc). `ionp_core::dtype::promote_dtype` already
//! implements that promotion table, so it is reused as-is here rather
//! than re-derived.
//!
//! ## Numeric kernels
//! - float32/float64/complex64/complex128: Accelerate BLAS
//!   (`sgemm`/`dgemm`/`cgemm`/`zgemm`) via `baseline::dense_matmul*`, one
//!   call per batch element. This is the asymptotically dominant case and
//!   the one the ULP sweep and "never slower than numpy" gate care about.
//! - integer dtypes: exact-by-construction wraparound arithmetic. Each
//!   product/sum is accumulated in a native-width `i64`/`u64` accumulator
//!   via `wrapping_mul`/`wrapping_add`, then truncated to the target
//!   width with a single `as` cast at the end. This reproduces numpy's
//!   per-element two's-complement overflow exactly: modular arithmetic
//!   mod 2^N is associative and commutative, so truncating once at the
//!   end gives the identical bit pattern to truncating after every
//!   operation, for every source width up to 64 bits (verified against
//!   real numpy's actual int8/uint8 overflow output, see
//!   `ionp-ion/tests` and the report's anti-tautology section).
//! - bool: logical AND (elementwise) / OR (accumulate), matching numpy's
//!   actual bool matmul semantics (NOT integer 0/1 arithmetic — verified
//!   empirically: `[[True]] @ [[True]]` under naive int arithmetic would
//!   give integer `1`, but real numpy's bool matmul loop returns `bool`
//!   `True`, and mixed True/True/False cases confirm AND/OR, not sum).
//!
//! ## What this module does NOT cover
//! `np.matmul.types` also lists `ee->e` (float16), `gg->g` (longdouble),
//! and `OO->O` (object) loops. `ionp_core::DType` has no float16,
//! longdouble, or object variant at all — this is a whole-project
//! limitation (every ufunc in ionp is missing these three), not something
//! specific to matmul, so it is reported as a pre-existing scope gap
//! rather than solved ad hoc here.

use ionp_core::array::{NdArray, Order};
use ionp_core::buffer::{Buffer, C128, C64};
use ionp_core::dtype::{promote_dtype, DType};
use ionp_core::shape::{broadcast_shapes, broadcast_strides_to, c_strides, NdIter};

use crate::baseline::{dense_matmul, dense_matmul_c128, dense_matmul_c64, dense_matmul_f32};
use crate::bridge;
use crate::compose;
use crate::dense::Dense;
use crate::gate;
use crate::operator::Operator;
use crate::product::Product;
use crate::rngutil::SplitMix64;
use num_complex::Complex64;

// ---------------------------------------------------------------------
// structured-operator path (ticket #101): route real f64 2-D matmul
// through ionp-ion's detect/compose/gate machinery instead of only ever
// calling the dense BLAS kernel. See `ionp_ion::bridge` for the
// NdArray-shaped-data <-> Operator conversion this depends on.
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// EXACT primary fast paths (2026-08-13 follow-up): the general
// detect-both-sides-then-certify-then-byte-compare path below
// (`try_structured_matmul_f64`) is a *certified no-op* on this codebase's
// actual corpus — it always computes `dense_matmul` too and only ships
// the structured answer when it happens to match bit-for-bit, which
// empirically (`dtype_f64_matmul_end_to_end_circulant_detects_but_falls_back_safely`)
// never happens for the interesting structures (Circulant, LowRank,
// Toeplitz collapse to a *different but equal* summation order, not the
// *same* one). Deleting it changes nothing about anionpy's output or
// timing, which is precisely the defect ticket #101 was filed against —
// see `ionp-ion/STRUCTURED-VS-BITEXACT.md` for the full accounting of
// which compositions are exact-in-value vs exact-in-bits.
//
// The three functions below are different in kind, not degree: each is
// backed by a hand-provable "this term is added to the sum as an exact
// zero" argument (see each function's doc comment), so they run
// INSTEAD of `dense_matmul`, not alongside it — no redundant O(n^3) work,
// no byte-compare, no fallback-by-default. `kernel_2d`'s `DType::F64` arm
// tries these FIRST and only calls `dense_matmul` when none apply.
// `cfg(debug_assertions)` still cross-checks every result against
// `dense_matmul` as a standing regression guard (cheap in debug, free in
// release), per the "certify, or a debug assertion at minimum" bar this
// path is held to.
// ---------------------------------------------------------------------

/// `diag(d) @ B`, `d` is an exact `n x n` diagonal (`is_exact_diagonal_square_f64`
/// already checked by the caller), `B` is `n x m` and fully finite
/// (`all_finite_f64` already checked). `O(n*m)` instead of `dense_matmul`'s
/// `O(n^2*m)`.
///
/// **Why this is bit-exact, not just numerically close:** `dense_matmul`
/// computes `out[i,j] = sum_{t=0}^{n-1} a[i,t] * b[t,j]`. Since `a` is
/// exactly diagonal, `a[i,t] == 0.0` for every `t != i` (bit-exact, not
/// approximately), and `b[t,j]` is finite (checked), so `a[i,t]*b[t,j] ==
/// 0.0` for every `t != i` (a zero times any finite value is exactly
/// zero, never `NaN`/`Inf` contamination, unlike `0.0 * Inf` or `0.0 *
/// NaN`). Summing zeros into an accumulator is a no-op in IEEE 754
/// round-to-nearest regardless of insertion order (`x + 0.0 == x` and `x +
/// (-0.0) == x` for any finite `x`), so the only term that survives is
/// `t == i`: `out[i,j] == a[i,i] * b[i,j]`, computed directly here with no
/// summation loop at all — literally the same single multiply
/// `dense_matmul` would land on after adding `n-1` zeros to it.
fn diagonal_scale_left_f64(a: &[f64], n: usize, b: &[f64], m: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n * m);
    let mut out = vec![0.0f64; n * m];
    for i in 0..n {
        let di = a[i * n + i];
        let brow = &b[i * m..(i + 1) * m];
        let orow = &mut out[i * m..(i + 1) * m];
        for j in 0..m {
            orow[j] = di * brow[j];
        }
    }
    out
}

/// `A @ diag(d)`, `d` is an exact `k x k` diagonal, `A` is `n x k` and
/// fully finite. `O(n*k)` instead of `dense_matmul`'s `O(n*k^2)`.
///
/// Mirror of [`diagonal_scale_left_f64`]'s argument with the roles of the
/// zero-supplying operand swapped: `out[i,j] = sum_t a[i,t]*b[t,j]`, `b`
/// diagonal means `b[t,j] == 0.0` for `t != j` (bit-exact), `a[i,t]`
/// finite (checked), so every `t != j` term is exactly `0.0` and the sum
/// collapses to the single surviving term `t == j`: `out[i,j] == a[i,j] *
/// b[j,j]`.
fn diagonal_scale_right_f64(a: &[f64], n: usize, k: usize, b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), n * k);
    debug_assert_eq!(b.len(), k * k);
    let diag: Vec<f64> = (0..k).map(|j| b[j * k + j]).collect();
    let mut out = vec![0.0f64; n * k];
    for i in 0..n {
        let arow = &a[i * k..(i + 1) * k];
        let orow = &mut out[i * k..(i + 1) * k];
        for j in 0..k {
            orow[j] = arow[j] * diag[j];
        }
    }
    out
}

/// `diag(d1) @ diag(d2)`, both exact `n x n` diagonals. `O(n)` instead of
/// `dense_matmul`'s `O(n^3)` — every off-diagonal output entry is exactly
/// `0.0` (both operands contribute an exact-zero factor there, so the
/// zero-multiply argument applies with no finiteness precondition needed
/// at all: even `0.0 * NaN` cases can't arise off-diagonal, because at
/// least one of the two factors in every off-diagonal term of the FULL
/// `n x n x n` triple-loop is `a[i,t]` or `b[t,j]` with `t` ranging over
/// *both* structures' zero patterns — concretely: for output `(i,j)` with
/// `i != j`, every `t` has either `t != i` (so `a[i,t] == 0.0` exactly,
/// finite) or `t == i != j` (so `b[t,j] == b[i,j] == 0.0` exactly, since
/// `i != j`); either way the term is `(finite or 0) * 0.0 == 0.0`
/// generically UNLESS the nonzero factor itself is non-finite. To keep the
/// same conservative finiteness bar as the other two functions (and avoid
/// re-deriving a term-by-term finiteness proof here), the caller still
/// requires both diagonals finite before calling this).
fn diagonal_times_diagonal_f64(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n * n);
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        out[i * n + i] = a[i * n + i] * b[i * n + i];
    }
    out
}

/// Try the exact zero-cost-detection primary fast paths, in cheapest/most
/// specific order. Returns `None` when neither operand is an exact
/// diagonal (or the finiteness precondition on the non-diagonal side
/// fails) — caller falls back to `dense_matmul`, unconditionally exact by
/// construction, same as always.
fn try_exact_diagonal_fast_path_f64(a: &[f64], n: usize, k: usize, b: &[f64], m: usize) -> Option<Vec<f64>> {
    let a_diag = n == k && bridge::is_exact_diagonal_square_f64(a, n);
    let b_diag = k == m && bridge::is_exact_diagonal_square_f64(b, k);

    let result = if a_diag && b_diag {
        diagonal_times_diagonal_f64(a, b, n)
    } else if a_diag && bridge::all_finite_f64(b) {
        diagonal_scale_left_f64(a, n, b, m)
    } else if b_diag && bridge::all_finite_f64(a) {
        diagonal_scale_right_f64(a, n, k, b)
    } else {
        return None;
    };

    // Standing regression guard: cross-check against the triple-loop/BLAS
    // ground truth in debug builds only (this is exactly the "or a debug
    // assertion at minimum" bar; paying for a second O(n^2*m)+ dense
    // compute on every release-mode call would defeat the entire point of
    // this fast path existing).
    #[cfg(debug_assertions)]
    {
        let reference = dense_matmul(a, n, k, b, m);
        debug_assert!(
            bit_exact(&result, &reference),
            "exact diagonal fast path produced a result that is NOT bit-identical to dense_matmul \
             -- the zero-multiply proof in this function's doc comment has a hole, or the \
             is_exact_diagonal_square_f64/all_finite_f64 preconditions were checked wrong"
        );
    }

    Some(result)
}

/// Attempt the general structured-operator path for one real `f64` 2-D
/// matmul block: detect structure on both operands, compose, and certify
/// against a dense reference via `gate::certify` (gate.rs). Returns
/// `Some(candidate)` only when detection found exploitable structure in
/// *both* operands and the composed operator passed the tolerance-based
/// gate.
///
/// **NOT wired into `kernel_2d`'s hot path** (see the module-level comment
/// above `diagonal_scale_left_f64`): on this codebase's structures, the
/// only compositions that survive `gate::certify`'s tolerance check are
/// exactly the ones `try_exact_diagonal_fast_path_f64` already handles
/// for free (no dense compute at all), and the rest (Circulant, LowRank,
/// Toeplitz collapses) are mathematically correct but a *different*
/// summation order than `dense_matmul` — see
/// `dtype_f64_matmul_end_to_end_circulant_detects_but_falls_back_safely`
/// below and `STRUCTURED-VS-BITEXACT.md`. Calling this from the hot path
/// would only ever recompute `dense_matmul` a second time for nothing.
/// Retained (not deleted) because it is real, tested capability that
/// proves detect/compose/gate/bridge all function end-to-end for
/// structures beyond plain diagonal, and because `STRUCTURED-VS-BITEXACT.md`
/// documents it as the mechanism behind the opt-in-tolerance option that
/// document lays out for a human to decide on.
#[allow(dead_code)]
fn try_structured_matmul_f64(a: &[f64], n: usize, k: usize, b: &[f64], m: usize) -> Option<Vec<f64>> {
    // Detection is O(n^2) and the composition/materialization round-trip
    // has its own constant overhead; below this size dense BLAS already
    // wins outright, so don't pay for a detection pass that can't help.
    if n < 8 || k < 8 || m < 8 {
        return None;
    }

    let op_a = bridge::detect_f64(a, n, k, 1e-9, true, 0x494F4E_A)?;
    let op_b = bridge::detect_f64(b, k, m, 1e-9, true, 0x494F4E_B)?;
    if !op_a.is_real() || !op_b.is_real() {
        return None;
    }

    // Dense reference for the correctness gate, built from the *same*
    // input bytes (not re-derived from the detected structure), so
    // gate::certify is checking the composed operator against the actual
    // operands, not against detection's own idea of what they are.
    let dense_a = Dense::new(n, k, bridge::f64_rowmajor_to_c64_colmajor(a, n, k), true);
    let dense_b = Dense::new(k, m, bridge::f64_rowmajor_to_c64_colmajor(b, k, m), true);
    let dense_ref = Product::new(Box::new(dense_a), Box::new(dense_b));

    let composed = compose::matmul(op_a, op_b);
    if !composed.is_real() {
        return None;
    }
    debug_assert_eq!(composed.shape(), dense_ref.shape());

    // gate.rs's correctness gate: certify() is the only way to obtain a
    // `Certified` handle, and it is impossible to reach the `Ok` arm
    // below without `apply`/`apply_mat` having matched the dense
    // reference to < 1e-9 on a real probe vector/matrix.
    let (_rows, cols) = composed.shape();
    let mut rng = SplitMix64::new(0x6761_7465_0000 ^ ((n as u64) << 32) ^ (k as u64) ^ ((m as u64) << 16));
    let x: Vec<Complex64> = (0..cols).map(|_| Complex64::new(rng.next_gaussian(), 0.0)).collect();
    let probe_cols = 3.min(cols).max(1);
    let xmat: Vec<Complex64> = (0..cols * probe_cols).map(|_| Complex64::new(rng.next_gaussian(), 0.0)).collect();

    match gate::certify("structured_matmul_f64", composed.as_ref(), &dense_ref, &x, &xmat, probe_cols, 1e-9) {
        Ok(_certified) => Some(bridge::materialize_f64(composed.as_ref())),
        Err(_gate_failure) => None,
    }
}

/// Bit-for-bit comparison, treating each `f64` by its raw bit pattern
/// (not `==`, which considers `-0.0 == 0.0` and would let a sign-of-zero
/// drift through undetected, and which makes `NaN != NaN` in a way that
/// would incorrectly *reject* two results that are both, say, the exact
/// same NaN payload numpy would also produce). `dense_matmul`'s own
/// output is the ground truth this suite already declares bit-exact
/// against numpy, so equality here is the only bar the structured path
/// is allowed to clear.
fn bit_exact(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

// ---------------------------------------------------------------------
// numpy-exact error message builders
// ---------------------------------------------------------------------

// BUG FOUND AND FIXED 2026-08-01: this hardcoded "requires 1" regardless of
// the actual per-operand, per-signature core-dims requirement. Real numpy's
// message reports the TRUE required core-dims count for that operand under
// that gufunc signature -- e.g. matvec's `(m,n),(n)->(m)` requires 2 core
// dims for operand 0 but only 1 for operand 1; vecmat's `(n),(n,m)->(m)`
// requires 1 for operand 0 but 2 for operand 1. Verified char-for-char
// against real numpy 2.5.1, e.g. `np.matvec(np.arange(4), np.arange(20).reshape(4,5))`
// raises `'matvec: Input operand 0 does not have enough dimensions (has 1,
// gufunc core with signature (m,n),(n)->(m) requires 2)'`. Found via
// independent out-of-corpus exception-message probing of the matmul family
// (agent_sweep3.py's sweep_matmul_family), 12/37026 mismatches all of this
// exact shape. Fixed by making the required count an explicit parameter,
// supplied correctly at each of the 8 call sites below.
fn not_enough_dims(name: &str, sig: &str, operand: usize, ndim: usize, required: usize) -> String {
    format!(
        "{name}: Input operand {operand} does not have enough dimensions \
         (has {ndim}, gufunc core with signature {sig} requires {required})"
    )
}

fn core_mismatch(name: &str, sig: &str, operand: usize, axis: usize, expected: usize, got: usize) -> String {
    format!(
        "{name}: Input operand {operand} has a mismatch in its core dimension {axis}, \
         with gufunc signature {sig} (size {got} is different from {expected})"
    )
}

/// numpy's shape formatter for this specific error family: comma-joined,
/// *no* space after the comma (unlike the plain-broadcast error's `", "`
/// join in `ionp_core::error::IonpError::Broadcast`), a trailing comma
/// for a single-element tuple, and `()` for empty — verified against real
/// numpy's `(2,2,3)->(2,newaxis,newaxis)` / `(2,4)->(2,)` output.
fn fmt_compact(tokens: &[String]) -> String {
    match tokens.len() {
        0 => "()".to_string(),
        1 => format!("({},)", tokens[0]),
        _ => format!("({})", tokens.join(",")),
    }
}

/// numpy's formatter for the trailing "requested shape (...)" — same
/// comma convention as `fmt_compact` but *no* trailing comma on a
/// singleton (`(3)` not `(3,)`), verified against real numpy.
fn fmt_requested(dims: &[usize]) -> String {
    if dims.is_empty() {
        "()".to_string()
    } else {
        format!("({})", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","))
    }
}

fn remapped_broadcast_err(
    name: &str,
    a_orig: &[usize],
    a_batch: &[usize],
    b_orig: &[usize],
    b_batch: &[usize],
    out_core_ndim: usize,
    requested: &[usize],
) -> String {
    let orig_tokens = |s: &[usize]| -> Vec<String> { s.iter().map(|d| d.to_string()).collect() };
    let mut a_remap = orig_tokens(a_batch);
    a_remap.extend(std::iter::repeat("newaxis".to_string()).take(out_core_ndim));
    let mut b_remap = orig_tokens(b_batch);
    b_remap.extend(std::iter::repeat("newaxis".to_string()).take(out_core_ndim));
    let _ = name; // numpy's message has no function-name prefix for this one (verified)
    format!(
        "operands could not be broadcast together with remapped shapes [original->remapped]: \
         {}->{} {}->{}  and requested shape {}",
        fmt_compact(&orig_tokens(a_orig)),
        fmt_compact(&a_remap),
        fmt_compact(&orig_tokens(b_orig)),
        fmt_compact(&b_remap),
        fmt_requested(requested)
    )
}

fn broadcast_batch_or_err(
    name: &str,
    a_orig: &[usize],
    a_batch: &[usize],
    b_orig: &[usize],
    b_batch: &[usize],
    out_core_ndim: usize,
    requested: &[usize],
) -> Result<Vec<usize>, String> {
    broadcast_shapes(a_batch, b_batch)
        .map_err(|_| remapped_broadcast_err(name, a_orig, a_batch, b_orig, b_batch, out_core_ndim, requested))
}

// ---------------------------------------------------------------------
// per-dtype 2-D kernels: C[n,m] = A[n,k] @ B[k,m], all row-major
// ---------------------------------------------------------------------

macro_rules! signed_int_kernel {
    ($fname:ident, $ty:ty) => {
        fn $fname(a: &[$ty], n: usize, k: usize, b: &[$ty], m: usize) -> Vec<$ty> {
            let mut out = vec![0 as $ty; n * m];
            for i in 0..n {
                for j in 0..m {
                    let mut acc: i64 = 0;
                    for kk in 0..k {
                        let av = a[i * k + kk] as i64;
                        let bv = b[kk * m + j] as i64;
                        acc = acc.wrapping_add(av.wrapping_mul(bv));
                    }
                    out[i * m + j] = acc as $ty;
                }
            }
            out
        }
    };
}
signed_int_kernel!(kernel_i8, i8);
signed_int_kernel!(kernel_i16, i16);
signed_int_kernel!(kernel_i32, i32);
signed_int_kernel!(kernel_i64, i64);

macro_rules! unsigned_int_kernel {
    ($fname:ident, $ty:ty) => {
        fn $fname(a: &[$ty], n: usize, k: usize, b: &[$ty], m: usize) -> Vec<$ty> {
            let mut out = vec![0 as $ty; n * m];
            for i in 0..n {
                for j in 0..m {
                    let mut acc: u64 = 0;
                    for kk in 0..k {
                        let av = a[i * k + kk] as u64;
                        let bv = b[kk * m + j] as u64;
                        acc = acc.wrapping_add(av.wrapping_mul(bv));
                    }
                    out[i * m + j] = acc as $ty;
                }
            }
            out
        }
    };
}
unsigned_int_kernel!(kernel_u8, u8);
unsigned_int_kernel!(kernel_u16, u16);
unsigned_int_kernel!(kernel_u32, u32);
unsigned_int_kernel!(kernel_u64, u64);

/// bool matmul is logical AND/OR, *not* integer 0/1 arithmetic (verified
/// against real numpy).
fn kernel_bool(a: &[bool], n: usize, k: usize, b: &[bool], m: usize) -> Vec<bool> {
    let mut out = vec![false; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut acc = false;
            for kk in 0..k {
                acc = acc || (a[i * k + kk] && b[kk * m + j]);
            }
            out[i * m + j] = acc;
        }
    }
    out
}

macro_rules! as_slice {
    ($buf:expr, $variant:ident) => {
        match $buf {
            Buffer::$variant(v) => v.as_slice(),
            other => unreachable!("matmul kernel dtype mismatch: expected {}, got {:?}", stringify!($variant), other.dtype()),
        }
    };
}

/// float16 matmul: numpy's own `ee->e` loop accumulates through `float32`
/// (there is no native binary16 BLAS path anywhere), so we widen both
/// operands to `f32`, reuse the existing f32 dense kernel verbatim, and
/// narrow the result back to `f16` -- never hand-rolled binary16 arithmetic.
fn kernel_f16(a: &[half::f16], n: usize, k: usize, b: &[half::f16], m: usize) -> Vec<half::f16> {
    let a32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
    let b32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();
    let out32 = dense_matmul_f32(&a32, n, k, &b32, m);
    out32.iter().map(|&x| half::f16::from_f32(x)).collect()
}

fn kernel_2d(dtype: DType, a: &Buffer, n: usize, k: usize, b: &Buffer, m: usize) -> Buffer {
    match dtype {
        DType::S(_) | DType::U(_) => unreachable!("ionp-ion: matmul has no semantics for string dtypes (phase 2 declines this operation on S/U; ionp-py bindings must reject before reaching here)"),
        DType::Bool => Buffer::Bool(kernel_bool(as_slice!(a, Bool), n, k, as_slice!(b, Bool), m)),
        DType::I8 => Buffer::I8(kernel_i8(as_slice!(a, I8), n, k, as_slice!(b, I8), m)),
        DType::I16 => Buffer::I16(kernel_i16(as_slice!(a, I16), n, k, as_slice!(b, I16), m)),
        DType::I32 => Buffer::I32(kernel_i32(as_slice!(a, I32), n, k, as_slice!(b, I32), m)),
        DType::I64 => Buffer::I64(kernel_i64(as_slice!(a, I64), n, k, as_slice!(b, I64), m)),
        DType::U8 => Buffer::U8(kernel_u8(as_slice!(a, U8), n, k, as_slice!(b, U8), m)),
        DType::U16 => Buffer::U16(kernel_u16(as_slice!(a, U16), n, k, as_slice!(b, U16), m)),
        DType::U32 => Buffer::U32(kernel_u32(as_slice!(a, U32), n, k, as_slice!(b, U32), m)),
        DType::U64 => Buffer::U64(kernel_u64(as_slice!(a, U64), n, k, as_slice!(b, U64), m)),
        DType::F16 => Buffer::F16(kernel_f16(as_slice!(a, F16), n, k, as_slice!(b, F16), m)),
        DType::F32 => Buffer::F32(dense_matmul_f32(as_slice!(a, F32), n, k, as_slice!(b, F32), m)),
        DType::F64 => {
            let av = as_slice!(a, F64);
            let bv = as_slice!(b, F64);
            // Exact fast path FIRST, and — when it fires — INSTEAD of
            // dense_matmul, not alongside it: diagonal@diagonal,
            // diagonal@dense, dense@diagonal are hand-proven bit-exact
            // (see try_exact_diagonal_fast_path_f64's doc comment and the
            // three functions it calls), so there is nothing to compare
            // against at runtime and no reason to pay for the O(n^3)
            // dense_matmul this arm used to compute unconditionally.
            // Anything that doesn't match one of those three shapes falls
            // through to the same unconditional dense_matmul this suite's
            // ULP sweep already declares bit-exact against numpy.
            let out = match try_exact_diagonal_fast_path_f64(av, n, k, bv, m) {
                Some(candidate) => candidate,
                None => dense_matmul(av, n, k, bv, m),
            };
            Buffer::F64(out)
        }
        DType::C64 => Buffer::C64(dense_matmul_c64(as_slice!(a, C64), n, k, as_slice!(b, C64), m)),
        DType::C128 => Buffer::C128(dense_matmul_c128(as_slice!(a, C128), n, k, as_slice!(b, C128), m)),
    }
}

fn conj_buffer(buf: &Buffer) -> Buffer {
    match buf {
        Buffer::C64(v) => Buffer::C64(v.iter().map(|c| c.conj()).collect::<Vec<C64>>()),
        Buffer::C128(v) => Buffer::C128(v.iter().map(|c| c.conj()).collect::<Vec<C128>>()),
        other => other.clone(),
    }
}

fn slice_buffer(buf: &Buffer, start: usize, len: usize) -> Buffer {
    macro_rules! sl {
        ($variant:ident) => {
            match buf {
                Buffer::$variant(v) => Buffer::$variant(v[start..start + len].to_vec()),
                _ => unreachable!(),
            }
        };
    }
    match buf.dtype() {
        DType::S(_) | DType::U(_) => unreachable!("ionp-ion: matmul has no semantics for string dtypes (phase 2 declines this operation on S/U; ionp-py bindings must reject before reaching here)"),
        DType::Bool => sl!(Bool),
        DType::I8 => sl!(I8),
        DType::I16 => sl!(I16),
        DType::I32 => sl!(I32),
        DType::I64 => sl!(I64),
        DType::U8 => sl!(U8),
        DType::U16 => sl!(U16),
        DType::U32 => sl!(U32),
        DType::U64 => sl!(U64),
        DType::F16 => sl!(F16),
        DType::F32 => sl!(F32),
        DType::F64 => sl!(F64),
        DType::C64 => sl!(C64),
        DType::C128 => sl!(C128),
    }
}

fn empty_buffer(dtype: DType, cap: usize) -> Buffer {
    match dtype {
        DType::S(_) | DType::U(_) => unreachable!("ionp-ion: matmul has no semantics for string dtypes (phase 2 declines this operation on S/U; ionp-py bindings must reject before reaching here)"),
        DType::Bool => Buffer::Bool(Vec::with_capacity(cap)),
        DType::I8 => Buffer::I8(Vec::with_capacity(cap)),
        DType::I16 => Buffer::I16(Vec::with_capacity(cap)),
        DType::I32 => Buffer::I32(Vec::with_capacity(cap)),
        DType::I64 => Buffer::I64(Vec::with_capacity(cap)),
        DType::U8 => Buffer::U8(Vec::with_capacity(cap)),
        DType::U16 => Buffer::U16(Vec::with_capacity(cap)),
        DType::U32 => Buffer::U32(Vec::with_capacity(cap)),
        DType::U64 => Buffer::U64(Vec::with_capacity(cap)),
        DType::F16 => Buffer::F16(Vec::with_capacity(cap)),
        DType::F32 => Buffer::F32(Vec::with_capacity(cap)),
        DType::F64 => Buffer::F64(Vec::with_capacity(cap)),
        DType::C64 => Buffer::C64(Vec::with_capacity(cap)),
        DType::C128 => Buffer::C128(Vec::with_capacity(cap)),
    }
}

fn extend_buffer(dst: &mut Buffer, src: Buffer) {
    macro_rules! ext {
        ($variant:ident) => {
            match (dst, src) {
                (Buffer::$variant(d), Buffer::$variant(s)) => d.extend(s),
                _ => unreachable!("matmul: buffer dtype drift during batch accumulation"),
            }
        };
    }
    match dst.dtype() {
        DType::S(_) | DType::U(_) => unreachable!("ionp-ion: matmul has no semantics for string dtypes (phase 2 declines this operation on S/U; ionp-py bindings must reject before reaching here)"),
        DType::Bool => ext!(Bool),
        DType::I8 => ext!(I8),
        DType::I16 => ext!(I16),
        DType::I32 => ext!(I32),
        DType::I64 => ext!(I64),
        DType::U8 => ext!(U8),
        DType::U16 => ext!(U16),
        DType::U32 => ext!(U32),
        DType::U64 => ext!(U64),
        DType::F16 => ext!(F16),
        DType::F32 => ext!(F32),
        DType::F64 => ext!(F64),
        DType::C64 => ext!(C64),
        DType::C128 => ext!(C128),
    }
}

// ---------------------------------------------------------------------
// shared batch engine
// ---------------------------------------------------------------------

/// Runs the batched 2-D kernel over every broadcast batch position and
/// assembles the result `NdArray`. `a_c`/`b_c` must already be
/// C-contiguous and share dtype `dtype`. `out_core_dims` is the trailing
/// shape to append after the (already-broadcast) `batch_shape` — empty
/// for `vecdot`, one dim for `matvec`/`vecmat`, zero-or-one-or-two for
/// `matmul` depending on which side(s) were 1-D.
#[allow(clippy::too_many_arguments)]
fn run_batched(
    dtype: DType,
    a_buf: &Buffer,
    a_batch_shape: &[usize],
    n: usize,
    k: usize,
    b_buf: &Buffer,
    b_batch_shape: &[usize],
    m: usize,
    batch_shape: &[usize],
    out_core_dims: &[usize],
) -> NdArray {
    // `batch_shape` is always `broadcast_batch_or_err`'s (broadcast_shapes-
    // backed) result of `a_batch_shape`/`b_batch_shape`, computed by every
    // caller before `run_batched` runs -- both operands are guaranteed
    // broadcastable to it already.
    let a_batch_strides = broadcast_strides_to(a_batch_shape, &c_strides(a_batch_shape), batch_shape)
        .expect("batch_shape already validated broadcastable against a_batch_shape by the caller");
    let b_batch_strides = broadcast_strides_to(b_batch_shape, &c_strides(b_batch_shape), batch_shape)
        .expect("batch_shape already validated broadcastable against b_batch_shape by the caller");
    let a_core_size = n * k;
    let b_core_size = k * m;
    let batch_size: usize = batch_shape.iter().product();

    let mut out_buf = empty_buffer(dtype, batch_size * n * m);
    let a_iter = NdIter::new(batch_shape, &a_batch_strides);
    let b_iter = NdIter::new(batch_shape, &b_batch_strides);
    for (off_a, off_b) in a_iter.zip(b_iter) {
        let a_block = slice_buffer(a_buf, off_a as usize * a_core_size, a_core_size);
        let b_block = slice_buffer(b_buf, off_b as usize * b_core_size, b_core_size);
        let c_block = kernel_2d(dtype, &a_block, n, k, &b_block, m);
        extend_buffer(&mut out_buf, c_block);
    }

    let mut out_shape = batch_shape.to_vec();
    out_shape.extend_from_slice(out_core_dims);
    NdArray::from_buffer(out_buf, out_shape, Order::C)
        .expect("matmul: output buffer length always matches computed output shape by construction")
}

/// Promote both operands to their common dtype and materialize each as a
/// fresh C-contiguous buffer (handles non-contiguous inputs uniformly —
/// `NdArray::cast_to` always materializes contiguous, even when the dtype
/// is unchanged).
fn prepare(a: &NdArray, b: &NdArray) -> (DType, NdArray, NdArray) {
    let dtype = promote_dtype(a.dtype(), b.dtype());
    (dtype, a.cast_to(dtype), b.cast_to(dtype))
}

// ---------------------------------------------------------------------
// matmul: (n?,k),(k,m?)->(n?,m?)
// ---------------------------------------------------------------------

const MATMUL_SIG: &str = "(n?,k),(k,m?)->(n?,m?)";

pub fn matmul(a: &NdArray, b: &NdArray) -> Result<NdArray, String> {
    const NAME: &str = "matmul";
    if a.ndim() == 0 {
        return Err(not_enough_dims(NAME, MATMUL_SIG, 0, 0, 1));
    }
    if b.ndim() == 0 {
        return Err(not_enough_dims(NAME, MATMUL_SIG, 1, 0, 1));
    }
    let a_1d = a.ndim() == 1;
    let b_1d = b.ndim() == 1;

    let k_a = if a_1d { a.shape()[0] } else { a.shape()[a.ndim() - 1] };
    let k_b = if b_1d { b.shape()[0] } else { b.shape()[b.ndim() - 2] };
    if k_a != k_b {
        return Err(core_mismatch(NAME, MATMUL_SIG, 1, 0, k_a, k_b));
    }

    let n_dim = if a_1d { None } else { Some(a.shape()[a.ndim() - 2]) };
    let m_dim = if b_1d { None } else { Some(b.shape()[b.ndim() - 1]) };

    let a_core_ndim = if a_1d { 1 } else { 2 };
    let b_core_ndim = if b_1d { 1 } else { 2 };
    let a_batch = &a.shape()[..a.ndim() - a_core_ndim];
    let b_batch = &b.shape()[..b.ndim() - b_core_ndim];

    let mut out_core_dims = Vec::new();
    if let Some(n) = n_dim {
        out_core_dims.push(n);
    }
    if let Some(m) = m_dim {
        out_core_dims.push(m);
    }
    let batch_shape = broadcast_batch_or_err(NAME, a.shape(), a_batch, b.shape(), b_batch, 2, &out_core_dims)?;

    let (dtype, a_c, b_c) = prepare(a, b);
    let n = n_dim.unwrap_or(1);
    let m = m_dim.unwrap_or(1);
    let k = k_a;

    Ok(run_batched(dtype, a_c.buffer(), a_batch, n, k, b_c.buffer(), b_batch, m, &batch_shape, &out_core_dims))
}

// ---------------------------------------------------------------------
// vecdot: (n),(n)->()  (conjugates the first operand for complex dtypes)
// ---------------------------------------------------------------------

const VECDOT_SIG: &str = "(n),(n)->()";

pub fn vecdot(a: &NdArray, b: &NdArray) -> Result<NdArray, String> {
    const NAME: &str = "vecdot";
    if a.ndim() == 0 {
        return Err(not_enough_dims(NAME, VECDOT_SIG, 0, 0, 1));
    }
    if b.ndim() == 0 {
        return Err(not_enough_dims(NAME, VECDOT_SIG, 1, 0, 1));
    }
    let n_a = a.shape()[a.ndim() - 1];
    let n_b = b.shape()[b.ndim() - 1];
    if n_a != n_b {
        return Err(core_mismatch(NAME, VECDOT_SIG, 1, 0, n_a, n_b));
    }
    let a_batch = &a.shape()[..a.ndim() - 1];
    let b_batch = &b.shape()[..b.ndim() - 1];
    let out_core_dims: [usize; 0] = [];
    let batch_shape = broadcast_batch_or_err(NAME, a.shape(), a_batch, b.shape(), b_batch, 0, &out_core_dims)?;

    let (dtype, a_c, b_c) = prepare(a, b);
    let a_buf = conj_buffer(a_c.buffer());
    Ok(run_batched(dtype, &a_buf, a_batch, 1, n_a, b_c.buffer(), b_batch, 1, &batch_shape, &out_core_dims))
}

// ---------------------------------------------------------------------
// matvec: (m,n),(n)->(m)  (no conjugation)
// ---------------------------------------------------------------------

const MATVEC_SIG: &str = "(m,n),(n)->(m)";

pub fn matvec(a: &NdArray, b: &NdArray) -> Result<NdArray, String> {
    const NAME: &str = "matvec";
    if a.ndim() < 2 {
        return Err(not_enough_dims(NAME, MATVEC_SIG, 0, a.ndim(), 2));
    }
    if b.ndim() == 0 {
        return Err(not_enough_dims(NAME, MATVEC_SIG, 1, 0, 1));
    }
    let n_a = a.shape()[a.ndim() - 1];
    let n_b = b.shape()[b.ndim() - 1];
    if n_a != n_b {
        return Err(core_mismatch(NAME, MATVEC_SIG, 1, 0, n_a, n_b));
    }
    let m_rows = a.shape()[a.ndim() - 2];
    let a_batch = &a.shape()[..a.ndim() - 2];
    let b_batch = &b.shape()[..b.ndim() - 1];
    let out_core_dims = [m_rows];
    let batch_shape = broadcast_batch_or_err(NAME, a.shape(), a_batch, b.shape(), b_batch, 1, &out_core_dims)?;

    let (dtype, a_c, b_c) = prepare(a, b);
    Ok(run_batched(dtype, a_c.buffer(), a_batch, m_rows, n_a, b_c.buffer(), b_batch, 1, &batch_shape, &out_core_dims))
}

// ---------------------------------------------------------------------
// vecmat: (n),(n,m)->(m)  (conjugates the vector operand, `a`)
// ---------------------------------------------------------------------

const VECMAT_SIG: &str = "(n),(n,m)->(m)";

pub fn vecmat(a: &NdArray, b: &NdArray) -> Result<NdArray, String> {
    const NAME: &str = "vecmat";
    if a.ndim() == 0 {
        return Err(not_enough_dims(NAME, VECMAT_SIG, 0, 0, 1));
    }
    if b.ndim() < 2 {
        return Err(not_enough_dims(NAME, VECMAT_SIG, 1, b.ndim(), 2));
    }
    let n_a = a.shape()[a.ndim() - 1];
    let n_b = b.shape()[b.ndim() - 2];
    if n_a != n_b {
        return Err(core_mismatch(NAME, VECMAT_SIG, 1, 0, n_a, n_b));
    }
    let m_cols = b.shape()[b.ndim() - 1];
    let a_batch = &a.shape()[..a.ndim() - 1];
    let b_batch = &b.shape()[..b.ndim() - 2];
    let out_core_dims = [m_cols];
    let batch_shape = broadcast_batch_or_err(NAME, a.shape(), a_batch, b.shape(), b_batch, 1, &out_core_dims)?;

    let (dtype, a_c, b_c) = prepare(a, b);
    let a_buf = conj_buffer(a_c.buffer());
    Ok(run_batched(dtype, &a_buf, a_batch, 1, n_a, b_c.buffer(), b_batch, m_cols, &batch_shape, &out_core_dims))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ionp_core::array::Order as Ord2;

    fn arr_f64(data: Vec<f64>, shape: Vec<usize>) -> NdArray {
        NdArray::from_buffer(Buffer::F64(data), shape, Ord2::C).unwrap()
    }
    fn arr_i32(data: Vec<i32>, shape: Vec<usize>) -> NdArray {
        NdArray::from_buffer(Buffer::I32(data), shape, Ord2::C).unwrap()
    }

    // ── structured-operator path (ticket #101) ──────────────────────────
    // These call `try_structured_matmul_f64` directly (module-private, so
    // only reachable from within this crate) to prove the structured path
    // actually fires -- not just that the public `matmul()` entry point
    // returns the right numbers, which the dense fallback would also do.

    #[test]
    fn structured_path_fires_for_diagonal_diagonal_and_is_bit_exact() {
        let n = 16;
        let mut a = vec![0.0f64; n * n];
        let mut b = vec![0.0f64; n * n];
        for i in 0..n {
            a[i * n + i] = (i + 1) as f64 * 1.5;
            b[i * n + i] = (i + 1) as f64 * 0.25;
        }
        let candidate = try_structured_matmul_f64(&a, n, n, &b, n).expect("diagonal@diagonal must detect structure");
        let dense_ref = dense_matmul(&a, n, n, &b, n);
        assert!(bit_exact(&candidate, &dense_ref), "structured diagonal@diagonal result must be bit-identical to dense_matmul");
    }

    #[test]
    fn structured_path_declines_when_only_one_side_is_structured() {
        // `try_structured_matmul_f64` requires BOTH operands to detect
        // structure (`compose.rs`'s closed-form collapses are all
        // same-structure pairs; a lone structured side still has real
        // value via `compose::matmul`'s lazy `Product` fallback, but this
        // implementation only attempts the structured path when detection
        // succeeds on both operands -- see its doc comment). A diagonal
        // left side paired with a genuinely full-rank random right side
        // must decline, not silently degrade to something unverified.
        let n = 16;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            a[i * n + i] = (i + 1) as f64;
        }
        let mut rng = SplitMix64::new(7);
        let b: Vec<f64> = (0..n * n).map(|_| rng.next_gaussian()).collect();
        assert!(try_structured_matmul_f64(&a, n, n, &b, n).is_none());
    }

    #[test]
    fn structured_path_declines_for_unstructured_input() {
        let n = 16;
        let mut rng = SplitMix64::new(11);
        let a: Vec<f64> = (0..n * n).map(|_| rng.next_gaussian()).collect();
        let b: Vec<f64> = (0..n * n).map(|_| rng.next_gaussian()).collect();
        assert!(
            try_structured_matmul_f64(&a, n, n, &b, n).is_none(),
            "two genuinely unstructured full-rank matrices must not detect structure"
        );
    }

    #[test]
    fn structured_path_declines_below_size_threshold() {
        // n=4 is below the n>=8 floor `try_structured_matmul_f64` bails out
        // on before ever calling `detect`, even for an exactly-diagonal
        // input that WOULD detect above the floor.
        let n = 4;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            a[i * n + i] = 2.0;
        }
        let b = a.clone();
        assert!(try_structured_matmul_f64(&a, n, n, &b, n).is_none());
    }

    #[test]
    fn dtype_f64_matmul_end_to_end_circulant_detects_but_falls_back_safely() {
        // Circulant@Circulant is exactly the case this task's ticket
        // flagged in advance: `compose.rs` collapses it to a tighter
        // `Circulant` via an FFT-based eigenvalue product
        // (`Circulant::compose_circulant`), which is mathematically exact
        // but NOT bit-identical to `dense_matmul`'s BLAS/triple-loop
        // summation order. `try_structured_matmul_f64` DOES detect
        // structure on both operands and DOES produce a candidate (proving
        // detect/compose/bridge/gate all fired), but that candidate must
        // NOT be bit-exact against dense -- and the public `matmul()`
        // entry point must still return the dense-exact answer, because
        // `kernel_2d`'s F64 arm only ships the structured candidate when
        // `bit_exact` holds.
        let n = 16;
        let mut rng = SplitMix64::new(99);
        let c: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
        let c2: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
        let circ = |first_col: &[f64]| -> Vec<f64> {
            let mut m = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    m[i * n + j] = first_col[(i + n - j) % n];
                }
            }
            m
        };
        let a_data = circ(&c);
        let b_data = circ(&c2);
        let dense_ref = dense_matmul(&a_data, n, n, &b_data, n);

        // The structured path fires and produces a candidate...
        let candidate = try_structured_matmul_f64(&a_data, n, n, &b_data, n).expect("circulant@circulant must detect structure on both operands");
        // ...but it is a DIFFERENT summation order (FFT diagonalization vs
        // dense triple-loop/BLAS), so it must NOT be bit-exact -- this is
        // the exact non-bit-exactness the ticket predicted in advance.
        assert!(!bit_exact(&candidate, &dense_ref), "circulant collapse is expected to differ from dense_matmul at the bit level -- if this now passes, the FFT path became bit-exact and the fallback below should be revisited");
        // it must still be numerically very close (a real answer, just
        // different rounding), not garbage.
        let max_rel_err = candidate
            .iter()
            .zip(&dense_ref)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1e-300))
            .fold(0.0, f64::max);
        assert!(max_rel_err < 1e-6, "structured candidate should still be numerically correct, just not byte-identical (max_rel_err={max_rel_err})");

        // And the public, ionp-py-facing entry point must return the
        // dense-exact answer regardless, because the byte-exact gate in
        // `kernel_2d` rejects the mismatching candidate and falls back.
        let a = arr_f64(a_data.clone(), vec![n, n]);
        let b = arr_f64(b_data.clone(), vec![n, n]);
        let out = matmul(&a, &b).unwrap();
        match out.buffer() {
            Buffer::F64(v) => assert!(bit_exact(v, &dense_ref), "matmul() must still return the dense-exact answer when the structured candidate doesn't match"),
            _ => panic!("expected F64 output"),
        }
    }

    // ── exact primary fast paths (2026-08-13 follow-up) ─────────────────
    // Unlike `try_structured_matmul_f64` above, these are meant to run
    // INSTEAD of `dense_matmul`, so the tests below prove two separate
    // things: (1) the result is bit-exact against an independently
    // computed `dense_matmul` reference, for shapes and sizes the
    // detect-based path's n>=8 floor would even decline, and (2) the
    // dispatcher (`try_exact_diagonal_fast_path_f64` / `kernel_2d`)
    // actually reaches these functions for the three licensed shapes and
    // declines for everything else -- including through the public,
    // ionp-py-facing `matmul()` entry point, not just the private helper.

    #[test]
    fn exact_fast_path_diagonal_times_diagonal_matches_dense_at_small_and_large_n() {
        for n in [1usize, 2, 5, 33, 129] {
            let mut a = vec![0.0f64; n * n];
            let mut b = vec![0.0f64; n * n];
            for i in 0..n {
                a[i * n + i] = (i + 1) as f64 * 1.5 - 0.5;
                b[i * n + i] = (i + 1) as f64 * 0.25 + 0.1;
            }
            let dense_ref = dense_matmul(&a, n, n, &b, n);
            let candidate = try_exact_diagonal_fast_path_f64(&a, n, n, &b, n).expect("diagonal@diagonal must take the exact fast path");
            assert!(bit_exact(&candidate, &dense_ref), "n={n}: diagonal@diagonal fast path must be bit-identical to dense_matmul");
        }
    }

    #[test]
    fn exact_fast_path_diagonal_scale_left_matches_dense() {
        let n = 6;
        let m = 4;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            a[i * n + i] = (i as f64 - 2.5) * 3.0;
        }
        let mut rng = SplitMix64::new(21);
        let b: Vec<f64> = (0..n * m).map(|_| rng.next_gaussian()).collect();
        let dense_ref = dense_matmul(&a, n, n, &b, m);
        let candidate = try_exact_diagonal_fast_path_f64(&a, n, n, &b, m).expect("diagonal@dense must take the exact fast path");
        assert!(bit_exact(&candidate, &dense_ref), "diagonal@dense fast path must be bit-identical to dense_matmul");
    }

    #[test]
    fn exact_fast_path_diagonal_scale_right_matches_dense() {
        let n = 5;
        let k = 7;
        let mut b = vec![0.0f64; k * k];
        for i in 0..k {
            b[i * k + i] = (i as f64 + 1.0) * -0.75;
        }
        let mut rng = SplitMix64::new(22);
        let a: Vec<f64> = (0..n * k).map(|_| rng.next_gaussian()).collect();
        let dense_ref = dense_matmul(&a, n, k, &b, k);
        let candidate = try_exact_diagonal_fast_path_f64(&a, n, k, &b, k).expect("dense@diagonal must take the exact fast path");
        assert!(bit_exact(&candidate, &dense_ref), "dense@diagonal fast path must be bit-identical to dense_matmul");
    }

    #[test]
    fn exact_fast_path_declines_for_non_diagonal_operands() {
        let n = 6;
        let mut rng = SplitMix64::new(23);
        let a: Vec<f64> = (0..n * n).map(|_| rng.next_gaussian()).collect();
        let b: Vec<f64> = (0..n * n).map(|_| rng.next_gaussian()).collect();
        assert!(try_exact_diagonal_fast_path_f64(&a, n, n, &b, n).is_none());
    }

    #[test]
    fn exact_fast_path_declines_when_near_zero_offdiagonal_present() {
        // A diagonal-*ish* matrix (an off-diagonal entry that a
        // tolerance-based detector would call zero) must NOT take this
        // fast path -- only an exact zero licenses dropping the term.
        let n = 4;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            a[i * n + i] = 2.0;
        }
        a[1] = 1e-14; // technically nonzero
        let mut rng = SplitMix64::new(24);
        let b: Vec<f64> = (0..n * n).map(|_| rng.next_gaussian()).collect();
        assert!(try_exact_diagonal_fast_path_f64(&a, n, n, &b, n).is_none());
    }

    #[test]
    fn exact_fast_path_declines_when_non_diagonal_side_has_nan() {
        // Diagonal @ B where B contains a NaN must decline the fast path
        // -- `0.0 * NaN` is NaN, so the zero-multiply proof does not hold,
        // and dense_matmul's own NaN-propagation behaviour (whatever it
        // is) must be preserved unconditionally.
        let n = 4;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            a[i * n + i] = 2.0;
        }
        let mut b = vec![1.0f64; n * n];
        b[5] = f64::NAN;
        assert!(try_exact_diagonal_fast_path_f64(&a, n, n, &b, n).is_none());
    }

    #[test]
    fn matmul_public_entry_point_takes_exact_fast_path_for_diagonal_diagonal() {
        let n = 10;
        let mut a_data = vec![0.0f64; n * n];
        let mut b_data = vec![0.0f64; n * n];
        for i in 0..n {
            a_data[i * n + i] = (i + 1) as f64;
            b_data[i * n + i] = (i + 2) as f64 * 0.5;
        }
        let dense_ref = dense_matmul(&a_data, n, n, &b_data, n);
        let a = arr_f64(a_data, vec![n, n]);
        let b = arr_f64(b_data, vec![n, n]);
        let out = matmul(&a, &b).unwrap();
        match out.buffer() {
            Buffer::F64(v) => assert!(bit_exact(v, &dense_ref), "matmul() on diagonal@diagonal must equal dense_matmul bit-for-bit"),
            _ => panic!("expected F64 output"),
        }
    }

    #[test]
    fn matmul_2d_2d_matches_naive() {
        let a = arr_f64(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = arr_f64(vec![7., 8., 9., 10., 11., 12.], vec![3, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        match c.buffer() {
            Buffer::F64(v) => assert_eq!(v, &[58., 64., 139., 154.]),
            _ => panic!(),
        }
    }

    #[test]
    fn matmul_1d_1d_is_scalar() {
        let a = arr_f64(vec![1., 2., 3.], vec![3]);
        let b = arr_f64(vec![4., 5., 6.], vec![3]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[] as &[usize]);
        match c.buffer() {
            Buffer::F64(v) => assert_eq!(v, &[32.]),
            _ => panic!(),
        }
    }

    #[test]
    fn matmul_batched_broadcast() {
        let a = arr_f64((0..5 * 1 * 2 * 3).map(|i| i as f64).collect(), vec![5, 1, 2, 3]);
        let b = arr_f64((0..4 * 3 * 6).map(|i| i as f64).collect(), vec![4, 3, 6]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[5, 4, 2, 6]);
        // Value check (not just shape) so a "collapsed batch loop" bug
        // (every batch position silently reading block 0) cannot pass this
        // test -- caught this exact gap during the anti-tautology sabotage
        // proof, where dropping the batch walk left shapes correct but
        // values wrong; strengthened here so unit tests alone catch it too.
        let block_a1 = arr_f64((3 * 2..3 * 2 + 6).map(|i| i as f64).collect(), vec![2, 3]);
        let block_b1 = arr_f64((3 * 6..3 * 6 + 18).map(|i| i as f64).collect(), vec![3, 6]);
        let expected_block = matmul(&block_a1, &block_b1).unwrap();
        match (c.buffer(), expected_block.buffer()) {
            (Buffer::F64(full), Buffer::F64(exp)) => {
                // batch index (a=1,b=1) -> flat batch position 1*4+1=5 in
                // the [5,4] broadcast batch grid, each block 2*6=12 elems.
                let start = 5 * 12;
                assert_eq!(&full[start..start + 12], exp.as_slice());
            }
            _ => panic!(),
        }
    }

    #[test]
    fn matmul_int_wraparound_matches_numpy_measured() {
        // 100*100 + 100*100 = 20000, truncated to i8 == 32 (verified vs
        // real numpy 2.5.1 in the differential probe this module's docs
        // describe).
        let a = arr_i32(vec![100, 100, 100, 100], vec![2, 2]).cast_to(DType::I8);
        let b = a.clone();
        let c = matmul(&a, &b).unwrap();
        match c.buffer() {
            Buffer::I8(v) => assert_eq!(v, &[32, 32, 32, 32]),
            _ => panic!(),
        }
    }

    #[test]
    fn matmul_scalar_rejected() {
        let a = arr_f64(vec![2.0], vec![]);
        let b = arr_f64(vec![1., 2., 3., 4.], vec![2, 2]);
        let err = matmul(&a, &b).unwrap_err();
        assert!(err.contains("does not have enough dimensions"), "{err}");
        assert!(err.contains("Input operand 0"), "{err}");
    }

    #[test]
    fn matmul_core_mismatch_message() {
        let a = arr_f64(vec![0.; 6], vec![2, 3]);
        let b = arr_f64(vec![0.; 20], vec![4, 5]);
        let err = matmul(&a, &b).unwrap_err();
        assert_eq!(
            err,
            "matmul: Input operand 1 has a mismatch in its core dimension 0, \
             with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is different from 3)"
        );
    }

    #[test]
    fn matmul_batch_broadcast_mismatch_message() {
        let a = arr_f64(vec![0.; 2 * 2 * 3], vec![2, 2, 3]);
        let b = arr_f64(vec![0.; 5 * 3 * 6], vec![5, 3, 6]);
        let err = matmul(&a, &b).unwrap_err();
        assert_eq!(
            err,
            "operands could not be broadcast together with remapped shapes \
             [original->remapped]: (2,2,3)->(2,newaxis,newaxis) (5,3,6)->(5,newaxis,newaxis)  \
             and requested shape (2,6)"
        );
    }

    #[test]
    fn vecdot_conjugates_complex_first_operand() {
        let a = NdArray::from_buffer(Buffer::C128(vec![C128::new(1., 2.), C128::new(3., 4.)]), vec![2], Ord2::C).unwrap();
        let b = NdArray::from_buffer(Buffer::C128(vec![C128::new(5., 6.), C128::new(7., 8.)]), vec![2], Ord2::C).unwrap();
        let r = vecdot(&a, &b).unwrap();
        match r.buffer() {
            Buffer::C128(v) => assert_eq!(v[0], C128::new(70., -8.)),
            _ => panic!(),
        }
    }

    #[test]
    fn matvec_shape_and_values() {
        let a = arr_f64(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = arr_f64(vec![1., 1., 1.], vec![3]);
        let r = matvec(&a, &b).unwrap();
        assert_eq!(r.shape(), &[2]);
        match r.buffer() {
            Buffer::F64(v) => assert_eq!(v, &[6., 15.]),
            _ => panic!(),
        }
    }

    #[test]
    fn vecmat_shape_and_values() {
        let a = arr_f64(vec![1., 1., 1.], vec![3]);
        let b = arr_f64(vec![1., 2., 3., 4., 5., 6., 7., 8., 9.], vec![3, 3]);
        let r = vecmat(&a, &b).unwrap();
        assert_eq!(r.shape(), &[3]);
        match r.buffer() {
            Buffer::F64(v) => assert_eq!(v, &[12., 15., 18.]),
            _ => panic!(),
        }
    }

    #[test]
    fn matmul_empty_dims() {
        let a = arr_f64(vec![], vec![0, 3]);
        let b = arr_f64(vec![0.; 12], vec![3, 4]);
        let r = matmul(&a, &b).unwrap();
        assert_eq!(r.shape(), &[0, 4]);

        let a2 = arr_f64(vec![], vec![2, 0]);
        let b2 = arr_f64(vec![], vec![0, 4]);
        let r2 = matmul(&a2, &b2).unwrap();
        assert_eq!(r2.shape(), &[2, 4]);
        match r2.buffer() {
            Buffer::F64(v) => assert_eq!(v, &[0.; 8]),
            _ => panic!(),
        }
    }

    #[test]
    fn matmul_noncontiguous_input() {
        // shape (4,6) sliced [:, ::2] -> (4,3), stride 2 on the last axis,
        // built via the public `get_view` slicing API (this crate has no
        // access to `NdArray`'s private fields).
        use ionp_core::array::SliceItem;
        let base = arr_f64((0..24).map(|i| i as f64).collect(), vec![4, 6]);
        let sliced = base
            .get_view(&[
                SliceItem::Slice { start: None, stop: None, step: None },
                SliceItem::Slice { start: None, stop: None, step: Some(2) },
            ])
            .unwrap();
        assert_eq!(sliced.shape(), &[4, 3]);
        assert!(!sliced.is_c_contiguous());
        let b = arr_f64(vec![1.; 6], vec![3, 2]);
        let r = matmul(&sliced, &b).unwrap();
        match r.buffer() {
            Buffer::F64(v) => assert_eq!(v, &[6., 6., 24., 24., 42., 42., 60., 60.]),
            _ => panic!(),
        }
    }

    // ── honest speed measurement (coordinator's ask, 2026-08-13) ────────
    // `#[ignore]`d because wall-clock timing on shared CI/dev hardware is
    // not a pass/fail correctness check and does not belong in the normal
    // suite -- run explicitly:
    //   cargo test -p ionp-ion --release exact_fast_path_is_faster_than_dense_matmul -- --ignored --nocapture
    // Prints diagonal_scale_left's wall-clock ratio against `dense_matmul`
    // at n=64,256,1024 (m fixed at 64, i.e. `diag(n,n) @ dense(n,64)`, the
    // shape this fast path actually changes the asymptotic complexity for:
    // O(n*m) instead of dense_matmul's O(n^2*m)). No assertion on the
    // ratio -- the coordinator asked to report the number "either way,"
    // not to gate on it.
    #[test]
    #[ignore]
    fn exact_fast_path_is_faster_than_dense_matmul() {
        use std::time::Instant;

        fn bench_one(n: usize, m: usize) {
            let mut a = vec![0.0f64; n * n];
            for i in 0..n {
                a[i * n + i] = (i as f64 + 1.0) * 0.5;
            }
            let mut rng = SplitMix64::new(1234 + (n * 1000 + m) as u64);
            let b: Vec<f64> = (0..n * m).map(|_| rng.next_gaussian()).collect();

            let reps = if n <= 64 { 2000 } else if n <= 256 { 200 } else { 20 };

            // Warm up both paths once so allocator/cache state doesn't
            // bias whichever runs first.
            let _ = dense_matmul(&a, n, n, &b, m);
            let _ = try_exact_diagonal_fast_path_f64(&a, n, n, &b, m);

            let t0 = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(dense_matmul(std::hint::black_box(&a), n, n, std::hint::black_box(&b), m));
            }
            let dense_elapsed = t0.elapsed();

            let t1 = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(try_exact_diagonal_fast_path_f64(std::hint::black_box(&a), n, n, std::hint::black_box(&b), m));
            }
            let fast_elapsed = t1.elapsed();

            let dense_per_call = dense_elapsed.as_secs_f64() / reps as f64;
            let fast_per_call = fast_elapsed.as_secs_f64() / reps as f64;
            println!(
                "n={n:5} m={m:5}  dense_matmul={:>12.3}us  diagonal_fast_path={:>12.3}us  speedup={:.2}x",
                dense_per_call * 1e6,
                fast_per_call * 1e6,
                dense_per_call / fast_per_call
            );
        }

        println!("-- diag(n,n) @ dense(n,64), m fixed small --");
        for n in [64usize, 256, 1024] {
            bench_one(n, 64);
        }
        println!("-- diag(n,n) @ dense(n,n), m == n (square) --");
        for n in [64usize, 256, 1024] {
            bench_one(n, n);
        }
    }
}
