//! The missing link between the numpy surface (`ionp_core::NdArray`, row-
//! major C-order by default, real dtypes like `f64`/`f32`) and the
//! structured-operator world (`Box<dyn Operator>`, column-major
//! `Complex64` buffers throughout — see `operator.rs`'s layout doc comment).
//!
//! Before this file, `ionp-py` never imported `detect`/`compose`/`gate`/any
//! concrete `Operator` — the numpy-facing crate and the structured-operator
//! crate were two islands with no bridge between their data
//! representations. This module is that bridge, in both directions:
//!
//! - [`f64_rowmajor_to_c64_colmajor`] / [`f32_rowmajor_to_c64_colmajor`]:
//!   real row-major -> complex column-major, for handing a numpy block to
//!   [`crate::detect::detect`].
//! - [`c64_colmajor_to_f64_rowmajor`] / [`c64_colmajor_to_f32_rowmajor`]:
//!   the inverse, extracting the real part — callers must have already
//!   confirmed `Operator::is_real()` (this module does not check it; the
//!   caller owns "was this operation supposed to stay real").
//! - [`detect_f64`]: convenience wrapper chaining the row-major conversion
//!   straight into `detect::detect`.
//!
//! ## The layout trap
//! `Operator` buffers are **column-major**: `data[col * n_rows + row] ==
//! A[row, col]`. `NdArray`'s default (C) order is **row-major**:
//! `data[row * n_cols + col] == A[row, col]`. Transposing the index math
//! (using `row * n_cols + col` on one side and `col * n_rows + row` on the
//! other) silently produces a transposed matrix rather than an error —
//! every function below is exercised by `tests` specifically for
//! non-square shapes, where a row/col mixup changes the *shape* of the
//! result and panics immediately instead of hiding, but also for the
//! square case where it would hide.

use crate::detect;
use crate::operator::Operator;
use num_complex::Complex64;

/// Real row-major `rows x cols` `f64` -> complex column-major `Complex64`.
/// `a[i * cols + j] == A[i,j]` in, `out[j * rows + i] == A[i,j]` out.
pub fn f64_rowmajor_to_c64_colmajor(a: &[f64], rows: usize, cols: usize) -> Vec<Complex64> {
    assert_eq!(a.len(), rows * cols, "f64_rowmajor_to_c64_colmajor: length mismatch");
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = Complex64::new(a[i * cols + j], 0.0);
        }
    }
    out
}

/// `f32` sibling of [`f64_rowmajor_to_c64_colmajor`] (widening cast — exact,
/// every `f32` value is exactly representable in `f64`).
pub fn f32_rowmajor_to_c64_colmajor(a: &[f32], rows: usize, cols: usize) -> Vec<Complex64> {
    assert_eq!(a.len(), rows * cols, "f32_rowmajor_to_c64_colmajor: length mismatch");
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = Complex64::new(a[i * cols + j] as f64, 0.0);
        }
    }
    out
}

/// Complex column-major `Complex64` -> real row-major `f64`, taking the
/// real part of every entry. Caller must already know the source operator
/// is real-valued (`Operator::is_real()`); this function does not check —
/// silently discarding a real matmul's imaginary part would be correct
/// (numpy also never carries imaginary noise for a real @ real product),
/// but silently discarding a *complex* operator's imaginary part would be
/// data loss, so the real/complex decision belongs to the caller, which
/// has the dtype context this module intentionally does not.
pub fn c64_colmajor_to_f64_rowmajor(a: &[Complex64], rows: usize, cols: usize) -> Vec<f64> {
    assert_eq!(a.len(), rows * cols, "c64_colmajor_to_f64_rowmajor: length mismatch");
    let mut out = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = a[j * rows + i].re;
        }
    }
    out
}

/// `f32` sibling of [`c64_colmajor_to_f64_rowmajor`] (narrowing cast).
pub fn c64_colmajor_to_f32_rowmajor(a: &[Complex64], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(a.len(), rows * cols, "c64_colmajor_to_f32_rowmajor: length mismatch");
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = a[j * rows + i].re as f32;
        }
    }
    out
}

/// Detect structure in a real row-major `f64` block, handing back the
/// `Operator` `detect::detect` proves for it, or `None` when the block is
/// genuinely unstructured (caller falls back to dense).
pub fn detect_f64(a: &[f64], rows: usize, cols: usize, rtol: f64, try_lowrank: bool, seed: u64) -> Option<Box<dyn Operator>> {
    let colmajor = f64_rowmajor_to_c64_colmajor(a, rows, cols);
    detect::detect(&colmajor, rows, cols, rtol, try_lowrank, seed)
}

/// Materialise an `Operator` known to be real-valued back into row-major
/// `f64` — the exact inverse of handing a numpy `f64` block to `detect_f64`.
/// Debug-asserts `op.is_real()` since every call site in this crate only
/// reaches here after confirming that; a release build trusts the caller
/// (same contract as `c64_colmajor_to_f64_rowmajor`).
pub fn materialize_f64(op: &dyn Operator) -> Vec<f64> {
    debug_assert!(op.is_real(), "materialize_f64: operator is not real-valued");
    let (rows, cols) = op.shape();
    let dense = op.to_dense();
    c64_colmajor_to_f64_rowmajor(&dense, rows, cols)
}

/// **Exact** (not tolerance-based) diagonal test on a row-major `n x n`
/// `f64` block: every off-diagonal entry must be `0.0` bit-for-bit (either
/// sign — `x == 0.0` is true for both `+0.0` and `-0.0`, and false for
/// anything else including NaN).
///
/// This is deliberately a *different, stricter* predicate than
/// `detect::is_diagonal`, which is tolerance-based (`rtol`, default
/// `1e-9`) — right for detecting *approximate* structure worth exploiting
/// via `apply`/`apply_mat`, wrong for deciding whether a fast path may
/// skip computing the dense ground truth entirely. An off-diagonal entry
/// that is merely "small relative to rtol" is still a nonzero term that a
/// triple-loop sum would add; only a term that is `0.0` on the nose can be
/// dropped from the sum with *zero* effect on the accumulated bit pattern
/// (`acc + 0.0 == acc` for any finite `acc`, `acc + (-0.0) == acc` under
/// round-to-nearest — see `matmul.rs`'s `diagonal_scale_left_f64` /
/// `diagonal_scale_right_f64` / `diagonal_times_diagonal_f64` doc comments
/// for the full zero-multiply argument these functions exist to license).
pub fn is_exact_diagonal_square_f64(a: &[f64], n: usize) -> bool {
    if a.len() != n * n {
        return false;
    }
    for i in 0..n {
        for j in 0..n {
            if i != j && a[i * n + j] != 0.0 {
                return false;
            }
        }
    }
    true
}

/// True iff every entry is finite (no NaN, no +/-Infinity). Required
/// alongside [`is_exact_diagonal_square_f64`] before a zero-multiply
/// shortcut is safe: `0.0 * NaN` and `0.0 * Infinity` are both `NaN`, so a
/// "provably zero" term only stays zero when multiplied against a finite
/// value.
pub fn all_finite_f64(a: &[f64]) -> bool {
    a.iter().all(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagonal::Diagonal;

    #[test]
    fn roundtrip_is_identity_nonsquare() {
        // 2 x 3, deliberately non-square so a row/col transpose bug changes
        // the shape (and would panic on the length assert) rather than
        // silently reordering entries.
        let rows = 2;
        let cols = 3;
        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // row-major: [[1,2,3],[4,5,6]]
        let colmajor = f64_rowmajor_to_c64_colmajor(&a, rows, cols);
        // column-major layout must be [1,4, 2,5, 3,6]
        assert_eq!(
            colmajor,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(6.0, 0.0),
            ]
        );
        let back = c64_colmajor_to_f64_rowmajor(&colmajor, rows, cols);
        assert_eq!(back, a);
    }

    #[test]
    fn detect_and_materialize_diagonal_roundtrips() {
        let n = 5;
        let d = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let diag = Diagonal::from_real(d.clone());
        let dense_colmajor = diag.to_dense();
        let dense_rowmajor = c64_colmajor_to_f64_rowmajor(&dense_colmajor, n, n);

        let found = detect_f64(&dense_rowmajor, n, n, 1e-9, true, 1).expect("should detect diagonal");
        assert!(found.as_any().downcast_ref::<Diagonal>().is_some());
        assert!(found.is_real());
        let back = materialize_f64(found.as_ref());
        assert_eq!(back, dense_rowmajor);
    }

    #[test]
    fn exact_diagonal_accepts_true_zero_rejects_near_zero() {
        let n = 3;
        let exact = vec![2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 7.0];
        assert!(is_exact_diagonal_square_f64(&exact, n));

        // detect::is_diagonal (tolerance-based) would accept this; the
        // exact predicate must not.
        let near = vec![2.0, 1e-12, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 7.0];
        assert!(!is_exact_diagonal_square_f64(&near, n));

        // negative zero off-diagonal is still exactly zero
        let neg_zero = vec![2.0, -0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 7.0];
        assert!(is_exact_diagonal_square_f64(&neg_zero, n));
    }

    #[test]
    fn all_finite_rejects_nan_and_inf() {
        assert!(all_finite_f64(&[1.0, -2.0, 0.0]));
        assert!(!all_finite_f64(&[1.0, f64::NAN, 0.0]));
        assert!(!all_finite_f64(&[1.0, f64::INFINITY, 0.0]));
        assert!(!all_finite_f64(&[1.0, f64::NEG_INFINITY, 0.0]));
    }
}
