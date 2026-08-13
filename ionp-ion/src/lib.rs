//! ionp-ion: structured operators, detection cascade, composition algebra,
//! FFT. Rust port of the algorithms proven in the project's Python reference
//! implementation — `wave/operators.py` and `detect.py` — per
//! `GOAL-ionp.md` step 3. **Ported algorithms, not translated Python**: this
//! crate has zero Python in it and never will (`ionp-py` is the only crate
//! in the workspace allowed to import `pyo3`).
//!
//! ## What's here
//! - [`operator::Operator`] — the trait every structured operator implements
//!   (`apply` = matvec, `apply_mat` = matmat, `to_dense` for verification).
//! - [`diagonal::Diagonal`], [`circulant::Circulant`], [`toeplitz::Toeplitz`],
//!   [`lowrank::LowRank`] — the four structured classes, plus [`dense::Dense`]
//!   as the honest fallback when nothing structured is found.
//! - [`detect::detect`] — the detection cascade (diagonal -> circulant ->
//!   toeplitz -> randomized low-rank), proof-carrying and O(n^2) or better in
//!   every branch (see `detect.rs` docs — this is a hard constraint, not a
//!   preference; regressing to O(n^3) detection was the failure that sank an
//!   earlier attempt at this codebase).
//! - [`compose::matmul`] — lazy composition; closed-form collapse for
//!   `Circulant@Circulant`, `LowRank@LowRank`, `Diagonal@Diagonal` (see
//!   `compose.rs` for which pairs collapse, which don't, and why).
//! - [`gate::certify`] — the correctness gate. No timing number may be
//!   produced for an operator that hasn't first matched a dense reference to
//!   `< 1e-9`; `Certified` is the only handle the bench binary's timing
//!   functions accept, so this is structural, not a convention.
//! - [`baseline`] — the real (rayon-parallel, cache-friendly) dense GEMV/GEMM
//!   the bench binary compares against, with its own GFLOP/s sanity-checked
//!   before any speedup is trusted.
//! - [`numpy_ref`] — shells out to the canonical numpy/BLAS reference at
//!   bench time so a speedup claim can be checked against a competitive
//!   baseline, not just a plausible one (see
//!   `reports/ion-gate-baseline-audit-2026-08-01.md`).

pub mod baseline;
pub mod bridge;
pub mod circulant;
pub mod compose;
pub mod dense;
pub mod dense_linalg;
pub mod detect;
pub mod diagonal;
pub mod gate;
pub mod linalg;
pub mod lowrank;
pub mod matmul;
pub mod numpy_ref;
pub mod operator;
pub mod product;
mod rngutil;
pub mod toeplitz;

pub use circulant::Circulant;
pub use compose::matmul;
pub use dense::Dense;
pub use dense_linalg::LinAlgError;
pub use detect::detect;
pub use diagonal::Diagonal;
pub use gate::{certify, Certified, GateFailure};
pub use lowrank::LowRank;
pub use operator::Operator;
pub use product::Product;
pub use rngutil::SplitMix64;
pub use toeplitz::Toeplitz;

pub type C64 = num_complex::Complex64;

/// Forces linking Apple's Accelerate framework (via `blas-src`'s
/// `accelerate` feature) so `baseline::dense_matmul`/`dense_matvec` can call
/// `cblas_dgemm`/`cblas_dgemv` and be a fair, competitive comparison against
/// numpy's BLAS (numpy on this machine also dispatches to Accelerate — see
/// `baseline.rs` docs and `reports/ion-gate-baseline-audit-2026-08-01.md`).
extern crate blas_src;

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    /// Deterministic pseudo-random real vector for tests — reuses the
    /// dependency-free SplitMix64 so tests need no external RNG crate.
    fn rand_real(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = SplitMix64::new(seed);
        (0..n).map(|_| rng.next_gaussian()).collect()
    }

    fn to_c64(x: &[f64]) -> Vec<Complex64> {
        x.iter().map(|&v| Complex64::new(v, 0.0)).collect()
    }

    fn dense_from_op(op: &dyn Operator) -> Dense {
        let (n, m) = op.shape();
        Dense::new(n, m, op.to_dense(), op.is_real())
    }

    fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max)
    }

    // ── per-operator correctness: apply() must match to_dense() @ x ────────

    #[test]
    fn diagonal_matches_dense() {
        let n = 64;
        let d = rand_real(1, n);
        let op = Diagonal::from_real(d);
        let dense = dense_from_op(&op);
        let x = to_c64(&rand_real(2, n));
        let xm = to_c64(&rand_real(3, n * 3));
        assert!(max_err(&op.apply(&x), &dense.apply(&x)) < 1e-12);
        assert!(max_err(&op.apply_mat(&xm, 3), &dense.apply_mat(&xm, 3)) < 1e-12);
    }

    #[test]
    fn circulant_matches_dense() {
        let n = 128;
        let c = to_c64(&rand_real(4, n));
        let op = Circulant::new(c, true);
        let dense = dense_from_op(&op);
        let x = to_c64(&rand_real(5, n));
        let xm = to_c64(&rand_real(6, n * 3));
        assert!(max_err(&op.apply(&x), &dense.apply(&x)) < 1e-9);
        assert!(max_err(&op.apply_mat(&xm, 3), &dense.apply_mat(&xm, 3)) < 1e-9);
    }

    #[test]
    fn toeplitz_matches_dense() {
        let n = 96;
        let col = to_c64(&rand_real(7, n));
        let row = to_c64(&rand_real(8, n));
        // row[0] must equal col[0] to be a valid Toeplitz spec
        let mut row = row;
        row[0] = col[0];
        let op = Toeplitz::new(col, row, true);
        let dense = dense_from_op(&op);
        let x = to_c64(&rand_real(9, n));
        let xm = to_c64(&rand_real(10, n * 3));
        assert!(max_err(&op.apply(&x), &dense.apply(&x)) < 1e-9);
        assert!(max_err(&op.apply_mat(&xm, 3), &dense.apply_mat(&xm, 3)) < 1e-9);
    }

    #[test]
    fn lowrank_matches_dense() {
        let n = 80;
        let m = 60;
        let r = 5;
        let u = to_c64(&rand_real(11, n * r));
        let v = to_c64(&rand_real(12, m * r));
        let op = LowRank::new(u, v, r, n, m, true);
        let dense = dense_from_op(&op);
        let x = to_c64(&rand_real(13, m));
        let xm = to_c64(&rand_real(14, m * 3));
        assert!(max_err(&op.apply(&x), &dense.apply(&x)) < 1e-9);
        assert!(max_err(&op.apply_mat(&xm, 3), &dense.apply_mat(&xm, 3)) < 1e-9);
    }

    // ── composition collapse correctness: collapsed op must match dense@dense ─

    #[test]
    fn circulant_product_matches_dense() {
        let n = 64;
        let c1 = to_c64(&rand_real(20, n));
        let c2 = to_c64(&rand_real(21, n));
        let op1 = Circulant::new(c1, true);
        let op2 = Circulant::new(c2, true);
        let d1 = dense_from_op(&op1);
        let d2 = dense_from_op(&op2);

        let collapsed = op1.compose_circulant(&op2);
        assert_eq!(collapsed.shape(), (n, n));

        let x = to_c64(&rand_real(22, n));
        let via_collapsed = collapsed.apply(&x);
        // dense@dense@x ground truth via the generic lazy Product's to_dense
        let lazy = Product::new(Box::new(dense_from_op(&op1)), Box::new(dense_from_op(&op2)));
        let dense_dense = lazy.to_dense();
        let dense_wrapped = Dense::new(n, n, dense_dense, true);
        let via_dense = dense_wrapped.apply(&x);

        assert!(max_err(&via_collapsed, &via_dense) < 1e-9);
        let _ = (d1, d2);
    }

    #[test]
    fn lowrank_product_matches_dense() {
        let n = 40;
        let k = 30;
        let m = 25;
        let r1 = 4;
        let r2 = 3;
        let u1 = to_c64(&rand_real(30, n * r1));
        let v1 = to_c64(&rand_real(31, k * r1));
        let u2 = to_c64(&rand_real(32, k * r2));
        let v2 = to_c64(&rand_real(33, m * r2));
        let op1 = LowRank::new(u1, v1, r1, n, k, true);
        let op2 = LowRank::new(u2, v2, r2, k, m, true);

        let collapsed = op1.compose_lowrank(&op2);
        assert_eq!(collapsed.shape(), (n, m));

        let lazy = Product::new(Box::new(dense_from_op(&op1)), Box::new(dense_from_op(&op2)));
        let dense_dense = Dense::new(n, m, lazy.to_dense(), true);

        let x = to_c64(&rand_real(34, m));
        assert!(max_err(&collapsed.apply(&x), &dense_dense.apply(&x)) < 1e-9);
    }

    #[test]
    fn diagonal_product_matches_dense() {
        let n = 30;
        let d1 = Diagonal::from_real(rand_real(40, n));
        let d2 = Diagonal::from_real(rand_real(41, n));
        let collapsed = d1.compose_diagonal(&d2);
        let dense = Dense::new(n, n, Product::new(Box::new(dense_from_op(&d1)), Box::new(dense_from_op(&d2))).to_dense(), true);
        let x = to_c64(&rand_real(42, n));
        assert!(max_err(&collapsed.apply(&x), &dense.apply(&x)) < 1e-12);
    }

    /// Documents (and locks in, via test) the disproved collapses: a general
    /// two-sided Toeplitz@Toeplitz is not Toeplitz, and Diagonal@Circulant is
    /// neither circulant nor Toeplitz. `matmul()` correctly leaves both as
    /// lazy `Product` rather than a wrong closed-form guess; this test checks
    /// that the lazy path is still numerically correct (the honest fallback
    /// works) and that the dense product genuinely fails the structural
    /// predicate it was tempting to claim.
    #[test]
    fn toeplitz_product_and_diag_circulant_stay_lazy_and_correct() {
        let n = 24;
        let col1 = to_c64(&rand_real(50, n));
        let mut row1 = to_c64(&rand_real(51, n));
        row1[0] = col1[0];
        let t1 = Toeplitz::new(col1, row1, true);

        let col2 = to_c64(&rand_real(52, n));
        let mut row2 = to_c64(&rand_real(53, n));
        row2[0] = col2[0];
        let t2 = Toeplitz::new(col2, row2, true);

        let prod: Box<dyn Operator> = matmul(Box::new(t1), Box::new(t2));
        // matmul() must NOT have collapsed this to a Toeplitz — verify by
        // checking the dense product genuinely fails is_toeplitz.
        let dense_prod = prod.to_dense();
        assert!(!detect::is_toeplitz(&dense_prod, n, n, 1e-9), "general Toeplitz@Toeplitz should not itself be Toeplitz");

        // but the lazy Product must still be numerically correct
        let x = to_c64(&rand_real(54, n));
        let via_lazy = prod.apply(&x);
        let via_dense = Dense::new(n, n, dense_prod, true).apply(&x);
        assert!(max_err(&via_lazy, &via_dense) < 1e-9);

        // Diagonal @ Circulant: also stays lazy, also correct
        let d = Diagonal::from_real(rand_real(55, n));
        let c = Circulant::new(to_c64(&rand_real(56, n)), true);
        let dc: Box<dyn Operator> = matmul(Box::new(d), Box::new(c));
        let dc_dense = dc.to_dense();
        assert!(!detect::is_circulant(&dc_dense, n, 1e-9));
        let via_lazy2 = dc.apply(&x);
        let via_dense2 = Dense::new(n, n, dc_dense, true).apply(&x);
        assert!(max_err(&via_lazy2, &via_dense2) < 1e-9);
    }

    // ── detection cascade ────────────────────────────────────────────────

    #[test]
    fn detect_finds_diagonal() {
        let n = 20;
        let d = Diagonal::from_real(rand_real(60, n));
        let dense = dense_from_op(&d);
        let found = detect::detect(&dense.to_dense(), n, n, 1e-9, true, 1).expect("should detect diagonal");
        assert_eq!(found.as_any().downcast_ref::<Diagonal>().is_some(), true);
    }

    #[test]
    fn detect_finds_circulant() {
        let n = 20;
        let c = Circulant::new(to_c64(&rand_real(61, n)), true);
        let dense = dense_from_op(&c);
        let found = detect::detect(&dense.to_dense(), n, n, 1e-9, true, 1).expect("should detect circulant");
        assert!(found.as_any().downcast_ref::<Circulant>().is_some());
    }

    #[test]
    fn detect_finds_toeplitz() {
        let n = 20;
        let col = to_c64(&rand_real(62, n));
        let mut row = to_c64(&rand_real(63, n));
        row[0] = col[0];
        let t = Toeplitz::new(col, row, true);
        let dense = dense_from_op(&t);
        let found = detect::detect(&dense.to_dense(), n, n, 1e-9, true, 1).expect("should detect toeplitz (not circulant)");
        assert!(found.as_any().downcast_ref::<Toeplitz>().is_some());
    }

    #[test]
    fn detect_finds_lowrank() {
        let n = 64;
        let m = 64;
        let r = 4;
        let u = to_c64(&rand_real(64, n * r));
        let v = to_c64(&rand_real(65, m * r));
        let lr = LowRank::new(u, v, r, n, m, true);
        let dense = dense_from_op(&lr);
        let found = detect::detect(&dense.to_dense(), n, m, 1e-9, true, 1).expect("should detect low rank");
        let found_lr = found.as_any().downcast_ref::<LowRank>().expect("should be LowRank");
        assert!(found_lr.rank() <= 2 * r, "detected rank {} should be close to true rank {}", found_lr.rank(), r);
    }

    #[test]
    fn detect_returns_none_for_unstructured() {
        let n = 24;
        let data = to_c64(&rand_real(66, n * n));
        let found = detect::detect(&data, n, n, 1e-9, true, 1);
        assert!(found.is_none(), "a full-rank random matrix should not be claimed as structured");
    }
}

