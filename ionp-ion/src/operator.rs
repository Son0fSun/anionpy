//! The `Operator` trait — a structured linear map carried in whatever basis
//! makes its action cheap. Mirrors the structured-operator design in the
//! project's Python reference implementation (`wave/operators.py`) but is a Rust port of the
//! *algorithm*, not a translation of the Python: storage is `Complex64`
//! throughout (matching the reference's internal `astype(complex128)`
//! upconversion), and `is_real()` tracks whether an operator's coefficients
//! are real-valued so callers can cast outputs back to `f64` when both the
//! operator and the input are real.
//!
//! Layout convention: matrices passed to/from `apply_mat`/`to_dense` are
//! **column-major** `Complex64` buffers — `data[col * n_rows + row] ==
//! A[row, col]` — chosen so a matmat is naturally a sequence of contiguous
//! matvec-shaped column slices.

use num_complex::Complex64;
use std::any::Any;

pub trait Operator: Send + Sync + 'static {
    /// (rows, cols) — `apply` takes a length-`cols` vector and returns a
    /// length-`rows` vector.
    fn shape(&self) -> (usize, usize);

    /// True iff every coefficient of this operator has zero imaginary part.
    /// Purely a display/casting hint; all arithmetic happens in Complex64
    /// regardless, exactly as the Python reference always stores
    /// `complex128` internally and casts back to real only cosmetically.
    fn is_real(&self) -> bool;

    /// Matrix-vector product `A x`. Cost is O(whatever the concrete type
    /// promises) — see each impl's doc comment for its asymptotic cost and
    /// the dense cost it beats.
    fn apply(&self, x: &[Complex64]) -> Vec<Complex64>;

    /// Matrix-matrix product `A X`, `X` column-major with `ncols` columns of
    /// length `self.shape().1`. Default falls back to per-column `apply`;
    /// Circulant/Toeplitz/LowRank override this with a batched transform
    /// (one FFT plan reused across columns) for the real matmat win.
    fn apply_mat(&self, x: &[Complex64], ncols: usize) -> Vec<Complex64> {
        let (n_out, n_in) = self.shape();
        let mut out = vec![Complex64::new(0.0, 0.0); n_out * ncols];
        for j in 0..ncols {
            let col = &x[j * n_in..(j + 1) * n_in];
            let y = self.apply(col);
            out[j * n_out..(j + 1) * n_out].copy_from_slice(&y);
        }
        out
    }

    /// Materialise the dense matrix, column-major. Verification only — never
    /// call this on a hot path; it is the O(n^2) (or worse, for lazy
    /// products) ground truth the correctness gate checks `apply` against.
    fn to_dense(&self) -> Vec<Complex64>;

    /// Enables downcasting through `&dyn Operator` for the composition
    /// collapse rules in `compose.rs` (Circulant@Circulant,
    /// LowRank@LowRank, Diagonal@Diagonal). Every impl below is the same one
    /// line: `fn as_any(&self) -> &dyn Any { self }`.
    fn as_any(&self) -> &dyn Any;
}
