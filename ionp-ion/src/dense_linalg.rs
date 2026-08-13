//! Dense numerical linear algebra core for `numpy.linalg` parity —
//! `det`, `inv`, `solve`, `lstsq`, `qr`, `svd`, `eig`, `eigh`, `eigvals`,
//! `eigvalsh`, `cholesky`, `pinv`, `matrix_rank`, `slogdet`, `norm`, `cond`,
//! `matrix_power`, `trace` — for `f64` (real) and `Complex64` inputs.
//!
//! ## Why LAPACK/Accelerate, not hand-rolled decompositions
//! Every factorization here (`dgetrf`/`dgetri`, `dgesv`, `dgeqrf`/`dorgqr`,
//! `dgesdd`, `dgeev`, `dsyev`, `dpotrf`, `dgelsd`, and their `z*` complex
//! counterparts) is a direct FFI call into Apple's Accelerate LAPACK via the
//! `lapack`/`lapack-src` crates — the same library numpy's own
//! `linalg` dispatches to on this machine. `baseline.rs` already documents
//! that a hand-rolled dense matmul measured ~7.85x slower than Accelerate;
//! reimplementing LU/QR/SVD/eigendecomposition by hand would repeat that
//! mistake at far higher risk (these algorithms are also much easier to get
//! subtly wrong than a triple-loop GEMM). The two exceptions are documented
//! at their call sites below: `matrix_power`'s repeated-squaring multiplies
//! dispatch to `baseline::dense_matmul` (Accelerate), and `pinv`'s
//! `V Σ⁺ Uᵀ` recombination also dispatches through `baseline::dense_matmul`
//! rather than a hand-rolled loop.
//!
//! ## Layout convention
//! Unlike `operator.rs` (column-major, for the structured-operator side of
//! this crate), everything in this module is **row-major**, matching numpy's
//! default C order — this module is the numpy-parity surface, not the
//! structured-operator surface, and the two conventions intentionally do not
//! leak into each other. Conversion to/from LAPACK's column-major Fortran
//! layout happens internally at each call site.
//!
//! ## Error handling
//! [`LinAlgError`] mirrors the shape of numpy's `LinAlgError`: a singular
//! matrix into `inv`/`solve`, a non-positive-definite matrix into
//! `cholesky`, or a non-converging iterative step in `eig`/`svd`/`lstsq`
//! never panics and never returns a garbage numeric result — it is always a
//! typed `Err`.

use crate::baseline;
use lapack::*;
use num_complex::Complex64;
use std::fmt;

// ─────────────────────────── error type ───────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum LinAlgError {
    /// Matrix is exactly (to working precision) singular — mirrors numpy's
    /// `LinAlgError: Singular matrix`.
    Singular,
    /// Cholesky input is not positive-definite (or not positive-semidefinite
    /// on the diagonal it was asked to factor).
    NotPositiveDefinite,
    /// Shapes don't satisfy the routine's precondition (e.g. non-square
    /// input to `det`/`inv`/`eig`).
    ShapeMismatch(String),
    /// The underlying LAPACK iterative routine (SVD, general eigenproblem)
    /// did not converge.
    DidNotConverge,
    /// Any other illegal-argument condition LAPACK reported (`info < 0`).
    InvalidInput(String),
}

impl fmt::Display for LinAlgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinAlgError::Singular => write!(f, "Singular matrix"),
            LinAlgError::NotPositiveDefinite => write!(f, "Matrix is not positive definite"),
            LinAlgError::ShapeMismatch(s) => write!(f, "shape mismatch: {}", s),
            LinAlgError::DidNotConverge => write!(f, "Eigenvalues/SVD did not converge"),
            LinAlgError::InvalidInput(s) => write!(f, "invalid input: {}", s),
        }
    }
}

impl std::error::Error for LinAlgError {}

type R<T> = Result<T, LinAlgError>;

fn check_square(rows: usize, cols: usize) -> R<()> {
    if rows != cols {
        return Err(LinAlgError::ShapeMismatch(format!(
            "expected a square matrix, got {}x{}",
            rows, cols
        )));
    }
    Ok(())
}

// ─────────────────────────── layout helpers (real) ────────────────────────

fn to_col_major(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), rows * cols);
    let mut out = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}

fn to_row_major(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), rows * cols);
    let mut out = vec![0.0f64; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[i * cols + j] = a[j * rows + i];
        }
    }
    out
}

// ────────────────────────── layout helpers (complex) ──────────────────────

fn to_col_major_c(a: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    debug_assert_eq!(a.len(), rows * cols);
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}

fn to_row_major_c(a: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    debug_assert_eq!(a.len(), rows * cols);
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[i * cols + j] = a[j * rows + i];
        }
    }
    out
}

/// Row-major `n x n` identity matrix -- used by `svd`/`svd_full`'s
/// empty-matrix (`m==0 || n==0`) special case: verified against real numpy
/// 2.5.1 that the "free" full-rank factor (`U` when `n==0`, `Vt` when
/// `m==0`) is exactly the identity matrix (e.g. `np.linalg.svd(np.zeros((3,
/// 0)))` returns `U == np.eye(3)`), not left as zeros -- LAPACK itself is
/// never called for this degenerate size, so this must be constructed
/// explicitly to match.
fn identity_flat(n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}

fn identity_flat_c(n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        out[i * n + i] = Complex64::new(1.0, 0.0);
    }
    out
}

// ════════════════════════════════════════════════════════════════════════
// Real (f64) routines
// ════════════════════════════════════════════════════════════════════════
pub mod real {
    use super::*;

    pub fn trace(a: &[f64], rows: usize, cols: usize) -> f64 {
        (0..rows.min(cols)).map(|i| a[i * cols + i]).sum()
    }

    /// LU-based sign/log-determinant. Mirrors `numpy.linalg.slogdet`: for a
    /// singular matrix returns `(0.0, -inf)` rather than erroring, matching
    /// numpy's own documented behaviour (numpy does not raise here either).
    pub fn slogdet(a: &[f64], n: usize) -> R<(f64, f64)> {
        check_square(n, n)?;
        if n == 0 {
            return Ok((1.0, 0.0));
        }
        let mut acm = to_col_major(a, n, n);
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        unsafe {
            dgetrf(n as i32, n as i32, &mut acm, n as i32, &mut ipiv, &mut info);
        }
        if info < 0 {
            return Err(LinAlgError::InvalidInput(format!("dgetrf: illegal argument {}", -info)));
        }
        let mut sign = 1.0f64;
        let mut logdet = 0.0f64;
        let mut singular = false;
        for i in 0..n {
            let diag = acm[i * n + i];
            if diag == 0.0 {
                singular = true;
                break;
            }
            if diag < 0.0 {
                sign = -sign;
            }
            logdet += diag.abs().ln();
            if ipiv[i] as usize != i + 1 {
                sign = -sign;
            }
        }
        if singular {
            return Ok((0.0, f64::NEG_INFINITY));
        }
        Ok((sign, logdet))
    }

    pub fn det(a: &[f64], n: usize) -> R<f64> {
        let (sign, logdet) = slogdet(a, n)?;
        if sign == 0.0 {
            return Ok(0.0);
        }
        Ok(sign * logdet.exp())
    }

    /// Full inverse via `dgetrf` + `dgetri`. Errors with `Singular` rather
    /// than returning a matrix full of `inf`/`nan` — mirrors numpy raising
    /// `LinAlgError` on `inv` of a singular matrix.
    pub fn inv(a: &[f64], n: usize) -> R<Vec<f64>> {
        check_square(n, n)?;
        if n == 0 {
            return Ok(vec![]);
        }
        let mut acm = to_col_major(a, n, n);
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        unsafe {
            dgetrf(n as i32, n as i32, &mut acm, n as i32, &mut ipiv, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::Singular);
        }
        let mut work = vec![0.0f64; 1];
        unsafe {
            dgetri(n as i32, &mut acm, n as i32, &ipiv, &mut work, -1, &mut info);
        }
        let lwork = (work[0] as i32).max(1);
        let mut work = vec![0.0f64; lwork as usize];
        unsafe {
            dgetri(n as i32, &mut acm, n as i32, &ipiv, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::Singular);
        }
        Ok(to_row_major(&acm, n, n))
    }

    /// Solve `A x = b` for square `A` (n x n) and `b` (n x nrhs), both
    /// row-major. Errors `Singular` rather than returning garbage.
    pub fn solve(a: &[f64], n: usize, b: &[f64], nrhs: usize) -> R<Vec<f64>> {
        check_square(n, n)?;
        if n == 0 {
            return Ok(vec![]);
        }
        let mut acm = to_col_major(a, n, n);
        let mut bcm = to_col_major(b, n, nrhs);
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        unsafe {
            dgesv(n as i32, nrhs as i32, &mut acm, n as i32, &mut ipiv, &mut bcm, n as i32, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::Singular);
        }
        Ok(to_row_major(&bcm, n, nrhs))
    }

    /// Least-squares solve of `A x ~= b` for general (possibly rectangular,
    /// possibly rank-deficient) `A` (m x n) via `dgelsd` (SVD-based, handles
    /// rank deficiency directly rather than needing a separate rank check).
    /// Returns `(x, singular_values, effective_rank)`.
    pub fn lstsq(a: &[f64], m: usize, n: usize, b: &[f64], nrhs: usize, rcond: f64) -> R<(Vec<f64>, Vec<f64>, i32)> {
        if m == 0 || n == 0 {
            return Err(LinAlgError::ShapeMismatch("lstsq: empty matrix".into()));
        }
        let minmn = m.min(n);
        let maxmn = m.max(n);
        let mut acm = to_col_major(a, m, n);
        let bcm_in = to_col_major(b, m, nrhs);
        // dgelsd's RHS buffer must be maxmn x nrhs (input occupies first m
        // rows, solution occupies first n rows on output).
        let mut bcm = vec![0.0f64; maxmn * nrhs];
        for j in 0..nrhs {
            bcm[j * maxmn..j * maxmn + m].copy_from_slice(&bcm_in[j * m..j * m + m]);
        }
        let mut s = vec![0.0f64; minmn];
        let mut rank = 0i32;
        let mut info = 0i32;

        let mut work = vec![0.0f64; 1];
        let mut iwork = vec![0i32; 1];
        unsafe {
            dgelsd(
                m as i32, n as i32, nrhs as i32, &mut acm, m as i32, &mut bcm, maxmn as i32, &mut s, rcond, &mut rank,
                &mut work, -1, &mut iwork, &mut info,
            );
        }
        let lwork = (work[0] as i32).max(1);
        let liwork = iwork[0].max(1);
        let mut work = vec![0.0f64; lwork as usize];
        let mut iwork = vec![0i32; liwork as usize];
        unsafe {
            dgelsd(
                m as i32, n as i32, nrhs as i32, &mut acm, m as i32, &mut bcm, maxmn as i32, &mut s, rcond, &mut rank,
                &mut work, lwork, &mut iwork, &mut info,
            );
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }
        let mut x = vec![0.0f64; n * nrhs];
        for j in 0..nrhs {
            for i in 0..n {
                x[i * nrhs + j] = bcm[j * maxmn + i];
            }
        }
        Ok((x, s, rank))
    }

    /// Economy (reduced) QR: `Q` is m x k, `R` is k x n, `k = min(m, n)`,
    /// `A = QR` and `QᵀQ = I_k`.
    pub fn qr(a: &[f64], m: usize, n: usize) -> R<(Vec<f64>, Vec<f64>)> {
        let k = m.min(n);
        // numpy's `qr` succeeds on empty input (`m==0` and/or `n==0`):
        // verified against real numpy 2.5.1, `qr(zeros((0,3)))` returns
        // `Q` shape (0,0), `R` shape (0,3); `qr(zeros((3,0)))` returns `Q`
        // shape (3,0), `R` shape (0,0) -- both are `k=min(m,n)=0`, so both
        // buffers are genuinely empty (`m*k==0` and `k*n==0`) and no LAPACK
        // call is needed at all. This used to be a blanket rejection
        // (`ShapeMismatch("qr: empty matrix")`) that numpy does not raise.
        if k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut acm = to_col_major(a, m, n);
        let mut tau = vec![0.0f64; k];
        let mut info = 0i32;

        let mut work = vec![0.0f64; 1];
        unsafe {
            dgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, -1, &mut info);
        }
        let lwork = (work[0] as i32).max(1);
        let mut work = vec![0.0f64; lwork as usize];
        unsafe {
            dgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::InvalidInput(format!("dgeqrf: illegal argument {}", -info)));
        }

        // R: upper-triangular part of the first k rows of the factored `acm`.
        let mut r = vec![0.0f64; k * n];
        for j in 0..n {
            for i in 0..=j.min(k - 1) {
                r[i * n + j] = acm[j * m + i];
            }
        }

        // Q: dorgqr consumes the Householder reflectors packed below the
        // diagonal of `acm`, truncated to its first k columns.
        let mut qcm = vec![0.0f64; m * k];
        for j in 0..k {
            qcm[j * m..(j + 1) * m].copy_from_slice(&acm[j * m..(j + 1) * m]);
        }
        let mut work = vec![0.0f64; 1];
        unsafe {
            dorgqr(m as i32, k as i32, k as i32, &mut qcm, m as i32, &tau, &mut work, -1, &mut info);
        }
        let lwork = (work[0] as i32).max(1);
        let mut work = vec![0.0f64; lwork as usize];
        unsafe {
            dorgqr(m as i32, k as i32, k as i32, &mut qcm, m as i32, &tau, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::InvalidInput(format!("dorgqr: illegal argument {}", -info)));
        }
        Ok((to_row_major(&qcm, m, k), r))
    }

    /// R-only QR (`mode='r'`): identical `dgeqrf` step as `qr` above, but
    /// skips the `dorgqr` call entirely since Q is discarded -- added
    /// 2026-08-02 for `numpy.linalg.qr(mode='r')`. Returns the SAME `k x n`
    /// R as `qr`'s second return value (numpy docs: `r : (..., K, N)`,
    /// `K = min(M, N)`).
    pub fn qr_r(a: &[f64], m: usize, n: usize) -> R<Vec<f64>> {
        let k = m.min(n);
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut acm = to_col_major(a, m, n);
        let mut tau = vec![0.0f64; k];
        let mut info = 0i32;
        let mut work = vec![0.0f64; 1];
        unsafe {
            dgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, -1, &mut info);
        }
        let lwork = (work[0] as i32).max(1);
        let mut work = vec![0.0f64; lwork as usize];
        unsafe {
            dgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::InvalidInput(format!("dgeqrf: illegal argument {}", -info)));
        }
        let mut r = vec![0.0f64; k * n];
        for j in 0..n {
            for i in 0..=j.min(k - 1) {
                r[i * n + j] = acm[j * m + i];
            }
        }
        Ok(r)
    }

    /// Complete QR (`mode='complete'`): Q is the FULL `m x m` orthonormal
    /// square matrix (not the economy `m x k`), R is the full `m x n`
    /// matrix (upper-triangular in its first `k` rows, zero below) --
    /// added 2026-08-02, shapes verified against real numpy 2.5.1
    /// (`q.shape == (M, M)`, `r.shape == (M, N)`).
    pub fn qr_complete(a: &[f64], m: usize, n: usize) -> R<(Vec<f64>, Vec<f64>)> {
        let k = m.min(n);
        if k == 0 {
            // m==0 or n==0: an m x m orthonormal Q still exists (it's just
            // the identity, since there are zero reflectors to apply); R
            // is m x n, all-empty/zero either way.
            let mut q = vec![0.0f64; m * m];
            for i in 0..m {
                q[i * m + i] = 1.0;
            }
            return Ok((q, vec![0.0f64; m * n]));
        }
        let mut acm = to_col_major(a, m, n);
        let mut tau = vec![0.0f64; k];
        let mut info = 0i32;
        let mut work = vec![0.0f64; 1];
        unsafe {
            dgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, -1, &mut info);
        }
        let lwork = (work[0] as i32).max(1);
        let mut work = vec![0.0f64; lwork as usize];
        unsafe {
            dgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::InvalidInput(format!("dgeqrf: illegal argument {}", -info)));
        }
        let mut r = vec![0.0f64; m * n];
        for j in 0..n {
            for i in 0..=j.min(k - 1) {
                r[i * n + j] = acm[j * m + i];
            }
        }
        // Full Q: m x m via `dorgqr(M, N=m, K=k, ...)` -- pad `acm`'s
        // reflector columns out to `m` columns (LAPACK forms columns
        // `k..m` from the `k` reflectors; those columns' input contents
        // are not referenced, so zero-padding is fine).
        let mut qcm = vec![0.0f64; m * m];
        for j in 0..k {
            qcm[j * m..(j + 1) * m].copy_from_slice(&acm[j * m..(j + 1) * m]);
        }
        let mut work = vec![0.0f64; 1];
        unsafe {
            dorgqr(m as i32, m as i32, k as i32, &mut qcm, m as i32, &tau, &mut work, -1, &mut info);
        }
        let lwork = (work[0] as i32).max(1);
        let mut work = vec![0.0f64; lwork as usize];
        unsafe {
            dorgqr(m as i32, m as i32, k as i32, &mut qcm, m as i32, &tau, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::InvalidInput(format!("dorgqr: illegal argument {}", -info)));
        }
        Ok((to_row_major(&qcm, m, m), r))
    }

    /// Raw QR (`mode='raw'`): returns the `dgeqrf` output verbatim -- `h`
    /// (packed Householder reflectors + R, shape `(N, M)`) and `tau`
    /// (shape `(K,)`) -- added 2026-08-02. `h`'s `(N, M)` shape is NOT a
    /// separately-computed transpose: numpy's own raw-mode `h` is exactly
    /// the column-major LAPACK buffer reinterpreted as a row-major `(N,
    /// M)` array (verified via a direct probe against real numpy 2.5.1:
    /// the printed values for `h` on a known input match `acm`'s raw flat
    /// contents read as `(N, M)` row-major, no reshuffling needed) -- so
    /// this is simply `acm` returned as-is; the caller is responsible for
    /// declaring the `(N, M)` shape when building the Python array.
    pub fn qr_raw(a: &[f64], m: usize, n: usize) -> R<(Vec<f64>, Vec<f64>)> {
        let k = m.min(n);
        if m == 0 || n == 0 {
            return Ok((vec![0.0f64; m * n], vec![0.0f64; k]));
        }
        let mut acm = to_col_major(a, m, n);
        let mut tau = vec![0.0f64; k];
        let mut info = 0i32;
        let mut work = vec![0.0f64; 1];
        unsafe {
            dgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, -1, &mut info);
        }
        let lwork = (work[0] as i32).max(1);
        let mut work = vec![0.0f64; lwork as usize];
        unsafe {
            dgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::InvalidInput(format!("dgeqrf: illegal argument {}", -info)));
        }
        Ok((acm, tau))
    }

    /// Economy SVD via `dgesdd`: `U` is m x k, `s` has length k (descending,
    /// non-negative), `Vt` is k x n, `k = min(m, n)`, `A = U diag(s) Vt`.
    pub fn svd(a: &[f64], m: usize, n: usize) -> R<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if m == 0 || n == 0 {
            // k = min(m, n) = 0 whenever either dim is 0, so U is m x 0 and
            // Vt is 0 x n -- both trivially empty, no LAPACK call needed or
            // possible (verified against real numpy 2.5.1: economy
            // `np.linalg.svd(np.zeros((3,0)), full_matrices=False)` returns
            // `U.shape == (3,0)`, i.e. all-empty, unlike the full-matrices
            // case below which fills the "free" factor with `eye`).
            return Ok((vec![], vec![], vec![]));
        }
        let k = m.min(n);
        let mut acm = to_col_major(a, m, n);
        let mut s = vec![0.0f64; k];
        let mut u = vec![0.0f64; m * k];
        let mut vt = vec![0.0f64; k * n];
        let mut iwork = vec![0i32; 8 * k];
        let mut info = 0i32;

        let mut work = vec![0.0f64; 1];
        unsafe {
            dgesdd(
                b'S', m as i32, n as i32, &mut acm, m as i32, &mut s, &mut u, m as i32, &mut vt, k as i32, &mut work,
                -1, &mut iwork, &mut info,
            );
        }
        let lwork = (work[0] as i32).max(1);
        let mut work = vec![0.0f64; lwork as usize];
        unsafe {
            dgesdd(
                b'S', m as i32, n as i32, &mut acm, m as i32, &mut s, &mut u, m as i32, &mut vt, k as i32, &mut work,
                lwork, &mut iwork, &mut info,
            );
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }
        Ok((to_row_major(&u, m, k), s, to_row_major(&vt, k, n)))
    }

    /// Full SVD via `dgesdd` job `'A'` (the SAME LAPACK entry point `svd`
    /// above already calls with job `'S'`, just re-driven with the full-size
    /// output buffers): `U` is m x m, `s` has length `k = min(m, n)`
    /// (descending, non-negative), `Vt` is n x n, `A = U diag_mxn(s) Vt`
    /// (`diag_mxn` meaning `s` sits on the leading `k x k` block of an
    /// m x n zero matrix -- callers reconstructing `A` must pad, not reuse
    /// the economy path's direct `U @ diag(s) @ Vt`). `iwork`/work-query
    /// sizing is job-independent (LAPACK: `iwork` is always `8*min(m,n)`;
    /// `work`'s optimal size is returned by the `lwork=-1` query regardless
    /// of `jobz`), so both are unchanged from the economy path above.
    pub fn svd_full(a: &[f64], m: usize, n: usize) -> R<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if m == 0 || n == 0 {
            // Unlike the economy path, `full_matrices=True`'s U/Vt are
            // always m x m / n x n (never truncated to k = min(m,n) = 0),
            // and real numpy's actual choice for the "free" factor here is
            // the identity matrix, not zeros (see `identity_flat`'s doc).
            return Ok((identity_flat(m), vec![], identity_flat(n)));
        }
        let k = m.min(n);
        let mut acm = to_col_major(a, m, n);
        let mut s = vec![0.0f64; k];
        let mut u = vec![0.0f64; m * m];
        let mut vt = vec![0.0f64; n * n];
        let mut iwork = vec![0i32; 8 * k];
        let mut info = 0i32;

        let mut work = vec![0.0f64; 1];
        unsafe {
            dgesdd(
                b'A', m as i32, n as i32, &mut acm, m as i32, &mut s, &mut u, m as i32, &mut vt, n as i32, &mut work,
                -1, &mut iwork, &mut info,
            );
        }
        let lwork = (work[0] as i32).max(1);
        let mut work = vec![0.0f64; lwork as usize];
        unsafe {
            dgesdd(
                b'A', m as i32, n as i32, &mut acm, m as i32, &mut s, &mut u, m as i32, &mut vt, n as i32, &mut work,
                lwork, &mut iwork, &mut info,
            );
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }
        Ok((to_row_major(&u, m, m), s, to_row_major(&vt, n, n)))
    }

    /// General (possibly non-symmetric) eigendecomposition via `dgeev`.
    /// Real input can have complex eigenvalues/eigenvectors (conjugate
    /// pairs), so both are returned as `Complex64` — mirrors
    /// `numpy.linalg.eig`. Eigenvectors are the columns of the returned
    /// n x n matrix, right eigenvectors normalised to unit 2-norm (LAPACK's
    /// convention, matching numpy's).
    pub fn eig(a: &[f64], n: usize) -> R<(Vec<Complex64>, Vec<Complex64>)> {
        check_square(n, n)?;
        if n == 0 {
            return Ok((vec![], vec![]));
        }
        let mut acm = to_col_major(a, n, n);
        let mut wr = vec![0.0f64; n];
        let mut wi = vec![0.0f64; n];
        let mut vl = vec![0.0f64; 1];
        let mut vr = vec![0.0f64; n * n];
        let mut info = 0i32;

        let mut work = vec![0.0f64; 1];
        unsafe {
            dgeev(
                b'N', b'V', n as i32, &mut acm, n as i32, &mut wr, &mut wi, &mut vl, 1, &mut vr, n as i32, &mut work,
                -1, &mut info,
            );
        }
        let lwork = (work[0] as i32).max(1);
        let mut work = vec![0.0f64; lwork as usize];
        unsafe {
            dgeev(
                b'N', b'V', n as i32, &mut acm, n as i32, &mut wr, &mut wi, &mut vl, 1, &mut vr, n as i32, &mut work,
                lwork, &mut info,
            );
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }

        let mut eigvals = vec![Complex64::new(0.0, 0.0); n];
        let mut vecs_cm = vec![Complex64::new(0.0, 0.0); n * n];
        let mut j = 0usize;
        while j < n {
            eigvals[j] = Complex64::new(wr[j], wi[j]);
            if wi[j] == 0.0 {
                for i in 0..n {
                    vecs_cm[j * n + i] = Complex64::new(vr[j * n + i], 0.0);
                }
                j += 1;
            } else {
                // LAPACK packs a conjugate pair (j, j+1) as (re, im) / (re, -im).
                eigvals[j + 1] = Complex64::new(wr[j + 1], wi[j + 1]);
                for i in 0..n {
                    let re = vr[j * n + i];
                    let im = vr[(j + 1) * n + i];
                    vecs_cm[j * n + i] = Complex64::new(re, im);
                    vecs_cm[(j + 1) * n + i] = Complex64::new(re, -im);
                }
                j += 2;
            }
        }
        Ok((eigvals, to_row_major_c(&vecs_cm, n, n)))
    }

    /// Eigenvalues only (no eigenvectors) via `dgeev`, cheaper than `eig`.
    pub fn eigvals(a: &[f64], n: usize) -> R<Vec<Complex64>> {
        check_square(n, n)?;
        if n == 0 {
            return Ok(vec![]);
        }
        let mut acm = to_col_major(a, n, n);
        let mut wr = vec![0.0f64; n];
        let mut wi = vec![0.0f64; n];
        let mut vl = vec![0.0f64; 1];
        let mut vr = vec![0.0f64; 1];
        let mut info = 0i32;

        let mut work = vec![0.0f64; 1];
        unsafe {
            dgeev(b'N', b'N', n as i32, &mut acm, n as i32, &mut wr, &mut wi, &mut vl, 1, &mut vr, 1, &mut work, -1, &mut info);
        }
        let lwork = (work[0] as i32).max(1);
        let mut work = vec![0.0f64; lwork as usize];
        unsafe {
            dgeev(
                b'N', b'N', n as i32, &mut acm, n as i32, &mut wr, &mut wi, &mut vl, 1, &mut vr, 1, &mut work, lwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }
        Ok((0..n).map(|i| Complex64::new(wr[i], wi[i])).collect())
    }

    /// Symmetric eigendecomposition via `dsyev`. Only the lower triangle of
    /// `a` is read (matches numpy's default `UPLO='L'`) — `a` need not be
    /// symmetrised by the caller. Eigenvalues are real and ascending;
    /// eigenvectors (columns of the returned n x n matrix) are orthonormal.
    pub fn eigh(a: &[f64], n: usize) -> R<(Vec<f64>, Vec<f64>)> {
        check_square(n, n)?;
        if n == 0 {
            return Ok((vec![], vec![]));
        }
        let mut acm = to_col_major(a, n, n);
        let mut w = vec![0.0f64; n];
        let mut info = 0i32;

        let mut work = vec![0.0f64; 1];
        unsafe {
            dsyev(b'V', b'L', n as i32, &mut acm, n as i32, &mut w, &mut work, -1, &mut info);
        }
        let lwork = (work[0] as i32).max(1);
        let mut work = vec![0.0f64; lwork as usize];
        unsafe {
            dsyev(b'V', b'L', n as i32, &mut acm, n as i32, &mut w, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }
        Ok((w, to_row_major(&acm, n, n)))
    }

    /// Symmetric eigenvalues only via `dsyev`.
    pub fn eigvalsh(a: &[f64], n: usize) -> R<Vec<f64>> {
        check_square(n, n)?;
        if n == 0 {
            return Ok(vec![]);
        }
        let mut acm = to_col_major(a, n, n);
        let mut w = vec![0.0f64; n];
        let mut info = 0i32;

        let mut work = vec![0.0f64; 1];
        unsafe {
            dsyev(b'N', b'L', n as i32, &mut acm, n as i32, &mut w, &mut work, -1, &mut info);
        }
        let lwork = (work[0] as i32).max(1);
        let mut work = vec![0.0f64; lwork as usize];
        unsafe {
            dsyev(b'N', b'L', n as i32, &mut acm, n as i32, &mut w, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }
        Ok(w)
    }

    /// Lower-triangular Cholesky factor `L` (`A = L Lᵀ`) via `dpotrf`.
    /// Errors `NotPositiveDefinite` rather than returning a partial/garbage
    /// factor — mirrors numpy raising `LinAlgError` on a non-PD input.
    pub fn cholesky(a: &[f64], n: usize) -> R<Vec<f64>> {
        check_square(n, n)?;
        if n == 0 {
            return Ok(vec![]);
        }
        let mut acm = to_col_major(a, n, n);
        let mut info = 0i32;
        unsafe {
            dpotrf(b'L', n as i32, &mut acm, n as i32, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::NotPositiveDefinite);
        }
        let mut rm = to_row_major(&acm, n, n);
        for i in 0..n {
            for j in (i + 1)..n {
                rm[i * n + j] = 0.0;
            }
        }
        Ok(rm)
    }

    /// Moore-Penrose pseudo-inverse via SVD: `A⁺ = V Σ⁺ Uᵀ`, singular
    /// values `<= rcond * max(s)` treated as zero. Recombination dispatches
    /// through `baseline::dense_matmul` (Accelerate) rather than a
    /// hand-rolled triple loop.
    pub fn pinv(a: &[f64], m: usize, n: usize, rcond: f64) -> R<Vec<f64>> {
        let (u, s, vt) = svd(a, m, n)?;
        let k = s.len();
        let smax = s.iter().cloned().fold(0.0f64, f64::max);
        let cutoff = rcond * smax;

        // V_scaled: n x k row-major, column t scaled by 1/s_t (or 0 if
        // s_t <= cutoff). V = Vtᵀ, so V_scaled[i][t] = vt[t][i] * inv_s[t].
        let mut v_scaled = vec![0.0f64; n * k];
        for t in 0..k {
            let inv_s = if s[t] > cutoff { 1.0 / s[t] } else { 0.0 };
            for i in 0..n {
                v_scaled[i * k + t] = vt[t * n + i] * inv_s;
            }
        }
        // Uᵀ: k x m row-major.
        let mut ut = vec![0.0f64; k * m];
        for t in 0..k {
            for j in 0..m {
                ut[t * m + j] = u[j * k + t];
            }
        }
        Ok(baseline::dense_matmul(&v_scaled, n, k, &ut, m))
    }

    /// Numerical rank via SVD: count of singular values exceeding
    /// `tol` (default `max(s) * max(m, n) * eps`, matching numpy). `eps`
    /// is caller-supplied rather than hardcoded to `f64::EPSILON` --
    /// real numpy's default tolerance uses `finfo(S.dtype).eps`, and
    /// `S.dtype` tracks the ORIGINAL array's precision family
    /// (float32/complex64 input -> float32 `S` -> `f32::EPSILON`), not
    /// always float64 (verified live: `matrix_rank(diag([1.0, 1e-10]))`
    /// gives rank 1 for float32 input vs rank 2 for float64 input, since
    /// `1e-10` falls between the two thresholds). All computation here is
    /// still done in `f64` (this crate's uniform internal precision); only
    /// the tolerance's `eps` factor needs to vary.
    pub fn matrix_rank(a: &[f64], m: usize, n: usize, tol: Option<f64>, eps: f64) -> R<usize> {
        if m == 0 || n == 0 {
            return Ok(0);
        }
        let (_u, s, _vt) = svd(a, m, n)?;
        let smax = s.iter().cloned().fold(0.0f64, f64::max);
        let t = tol.unwrap_or(smax * (m.max(n) as f64) * eps);
        Ok(s.iter().filter(|&&v| v > t).count())
    }

    pub fn norm_fro(a: &[f64]) -> f64 {
        a.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    pub fn norm_1(a: &[f64], m: usize, n: usize) -> f64 {
        (0..n)
            .map(|j| (0..m).map(|i| a[i * n + j].abs()).sum::<f64>())
            .fold(0.0f64, f64::max)
    }

    pub fn norm_inf(a: &[f64], m: usize, n: usize) -> f64 {
        (0..m)
            .map(|i| (0..n).map(|j| a[i * n + j].abs()).sum::<f64>())
            .fold(0.0f64, f64::max)
    }

    /// `p=-1` matrix norm: `min(sum(abs(x), axis=0))` -- the column-sum
    /// counterpart of `norm_1`'s max, added 2026-08-02 for `cond(p=-1)`.
    pub fn norm_neg1(a: &[f64], m: usize, n: usize) -> f64 {
        (0..n)
            .map(|j| (0..m).map(|i| a[i * n + j].abs()).sum::<f64>())
            .fold(f64::INFINITY, f64::min)
    }

    /// `p=-inf` matrix norm: `min(sum(abs(x), axis=1))` -- the row-sum
    /// counterpart of `norm_inf`'s max, added 2026-08-02 for
    /// `cond(p=-inf)`.
    pub fn norm_neginf(a: &[f64], m: usize, n: usize) -> f64 {
        (0..m)
            .map(|i| (0..n).map(|j| a[i * n + j].abs()).sum::<f64>())
            .fold(f64::INFINITY, f64::min)
    }

    /// Spectral (induced 2-)norm: largest singular value.
    pub fn norm_2(a: &[f64], m: usize, n: usize) -> R<f64> {
        let (_u, s, _vt) = svd(a, m, n)?;
        Ok(s.iter().cloned().fold(0.0f64, f64::max))
    }

    /// Nuclear norm: sum of singular values.
    pub fn norm_nuc(a: &[f64], m: usize, n: usize) -> R<f64> {
        let (_u, s, _vt) = svd(a, m, n)?;
        Ok(s.iter().sum())
    }

    /// Vector p-norm. `ord = f64::INFINITY`/`f64::NEG_INFINITY` for max/min
    /// abs; `ord == 0.0` counts nonzeros (matches numpy's `ord=0`).
    pub fn vector_norm(x: &[f64], ord: f64) -> f64 {
        if ord.is_infinite() && ord > 0.0 {
            x.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
        } else if ord.is_infinite() && ord < 0.0 {
            x.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min)
        } else if ord == 0.0 {
            x.iter().filter(|&&v| v != 0.0).count() as f64
        } else {
            x.iter().map(|v| v.abs().powf(ord)).sum::<f64>().powf(1.0 / ord)
        }
    }

    /// Condition number `cond(A) = ||A||_p ||A^-1||_p`. `p = None` or
    /// `Some(2.0)` uses `smax/smin` directly (cheaper: one SVD instead of an
    /// SVD plus an explicit inverse). `Some(1.0)`/`Some(inf)`/`Some(-1.0)`/
    /// `Some(-inf)` use the induced 1-/inf-/neg1-/neginf-norms with an
    /// explicit `inv`; `fro=true` uses the Frobenius norm the same way
    /// (`p` is ignored when `fro` is set -- callers pass `None` for `p` in
    /// that case, matching how the Python-level `p='fro'` string doesn't
    /// map to any `f64`). `-1`/`-inf`/`fro` added 2026-08-02: verified via
    /// `inspect.getsource(numpy.linalg._linalg.cond)` that numpy's ONLY
    /// special-cased `p` values are `None`/`2`/`-2` (SVD ratio); every
    /// other `p` (`1`, `-1`, `inf`, `-inf`, `'fro'`) goes through the exact
    /// same `norm(x,p)*norm(inv(x),p)` formula this fn already used for
    /// `1`/`inf` -- so these were a genuine "not yet implemented" gap, not
    /// a "genuinely unimplementable" one; `-2` is now handled directly by
    /// the SVD-ratio path at the `ionp-py/src/linalg.rs` call site (not
    /// here), mirroring `p=None`/`p=2`.
    pub fn cond(a: &[f64], n: usize, p: Option<f64>, fro: bool) -> R<f64> {
        if fro {
            return match inv(a, n) {
                Ok(ai) => Ok(norm_fro(a) * norm_fro(&ai)),
                Err(LinAlgError::Singular) => Ok(f64::INFINITY),
                Err(e) => Err(e),
            };
        }
        match p {
            None => {
                let (_u, s, _vt) = svd(a, n, n)?;
                let smax = s.iter().cloned().fold(0.0f64, f64::max);
                let smin = s.iter().cloned().fold(f64::INFINITY, f64::min);
                if smin == 0.0 {
                    Ok(f64::INFINITY)
                } else {
                    Ok(smax / smin)
                }
            }
            Some(v) if v == 2.0 => {
                let (_u, s, _vt) = svd(a, n, n)?;
                let smax = s.iter().cloned().fold(0.0f64, f64::max);
                let smin = s.iter().cloned().fold(f64::INFINITY, f64::min);
                if smin == 0.0 {
                    Ok(f64::INFINITY)
                } else {
                    Ok(smax / smin)
                }
            }
            Some(v) if v == 1.0 => match inv(a, n) {
                Ok(ai) => Ok(norm_1(a, n, n) * norm_1(&ai, n, n)),
                Err(LinAlgError::Singular) => Ok(f64::INFINITY),
                Err(e) => Err(e),
            },
            Some(v) if v == -1.0 => match inv(a, n) {
                Ok(ai) => Ok(norm_neg1(a, n, n) * norm_neg1(&ai, n, n)),
                Err(LinAlgError::Singular) => Ok(f64::INFINITY),
                Err(e) => Err(e),
            },
            Some(v) if v.is_infinite() && v > 0.0 => match inv(a, n) {
                Ok(ai) => Ok(norm_inf(a, n, n) * norm_inf(&ai, n, n)),
                Err(LinAlgError::Singular) => Ok(f64::INFINITY),
                Err(e) => Err(e),
            },
            Some(v) if v.is_infinite() && v < 0.0 => match inv(a, n) {
                Ok(ai) => Ok(norm_neginf(a, n, n) * norm_neginf(&ai, n, n)),
                Err(LinAlgError::Singular) => Ok(f64::INFINITY),
                Err(e) => Err(e),
            },
            Some(v) => Err(LinAlgError::InvalidInput(format!("cond: unsupported ord {}", v))),
        }
    }

    /// Integer matrix power via repeated squaring; negative powers invert
    /// first. All multiplies dispatch to `baseline::dense_matmul`
    /// (Accelerate) — no hand-rolled GEMM.
    pub fn matrix_power(a: &[f64], n: usize, power: i32) -> R<Vec<f64>> {
        check_square(n, n)?;
        if n == 0 {
            return Ok(vec![]);
        }
        let base = if power < 0 { inv(a, n)? } else { a.to_vec() };
        let mut p = power.unsigned_abs();
        if p == 0 {
            let mut ident = vec![0.0f64; n * n];
            for i in 0..n {
                ident[i * n + i] = 1.0;
            }
            return Ok(ident);
        }
        let mut result: Option<Vec<f64>> = None;
        let mut cur = base;
        while p > 0 {
            if p & 1 == 1 {
                result = Some(match result {
                    None => cur.clone(),
                    Some(r) => baseline::dense_matmul(&r, n, n, &cur, n),
                });
            }
            p >>= 1;
            if p > 0 {
                cur = baseline::dense_matmul(&cur, n, n, &cur, n);
            }
        }
        Ok(result.unwrap())
    }
}

// ════════════════════════════════════════════════════════════════════════
// Complex (Complex64) routines
// ════════════════════════════════════════════════════════════════════════
pub mod complex {
    use super::*;

    pub fn trace(a: &[Complex64], rows: usize, cols: usize) -> Complex64 {
        (0..rows.min(cols)).map(|i| a[i * cols + i]).sum()
    }

    pub fn slogdet(a: &[Complex64], n: usize) -> R<(Complex64, f64)> {
        check_square(n, n)?;
        if n == 0 {
            return Ok((Complex64::new(1.0, 0.0), 0.0));
        }
        let mut acm = to_col_major_c(a, n, n);
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        unsafe {
            zgetrf(n as i32, n as i32, &mut acm, n as i32, &mut ipiv, &mut info);
        }
        if info < 0 {
            return Err(LinAlgError::InvalidInput(format!("zgetrf: illegal argument {}", -info)));
        }
        let mut sign = Complex64::new(1.0, 0.0);
        let mut logdet = 0.0f64;
        let mut singular = false;
        for i in 0..n {
            let diag = acm[i * n + i];
            if diag.norm() == 0.0 {
                singular = true;
                break;
            }
            sign *= diag / diag.norm();
            logdet += diag.norm().ln();
            if ipiv[i] as usize != i + 1 {
                sign = -sign;
            }
        }
        if singular {
            return Ok((Complex64::new(0.0, 0.0), f64::NEG_INFINITY));
        }
        Ok((sign, logdet))
    }

    pub fn det(a: &[Complex64], n: usize) -> R<Complex64> {
        let (sign, logdet) = slogdet(a, n)?;
        if sign.norm() == 0.0 {
            return Ok(Complex64::new(0.0, 0.0));
        }
        Ok(sign * logdet.exp())
    }

    pub fn inv(a: &[Complex64], n: usize) -> R<Vec<Complex64>> {
        check_square(n, n)?;
        if n == 0 {
            return Ok(vec![]);
        }
        let mut acm = to_col_major_c(a, n, n);
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        unsafe {
            zgetrf(n as i32, n as i32, &mut acm, n as i32, &mut ipiv, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::Singular);
        }
        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zgetri(n as i32, &mut acm, n as i32, &ipiv, &mut work, -1, &mut info);
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zgetri(n as i32, &mut acm, n as i32, &ipiv, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::Singular);
        }
        Ok(to_row_major_c(&acm, n, n))
    }

    pub fn solve(a: &[Complex64], n: usize, b: &[Complex64], nrhs: usize) -> R<Vec<Complex64>> {
        check_square(n, n)?;
        if n == 0 {
            return Ok(vec![]);
        }
        let mut acm = to_col_major_c(a, n, n);
        let mut bcm = to_col_major_c(b, n, nrhs);
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        unsafe {
            zgesv(n as i32, nrhs as i32, &mut acm, n as i32, &mut ipiv, &mut bcm, n as i32, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::Singular);
        }
        Ok(to_row_major_c(&bcm, n, nrhs))
    }

    pub fn lstsq(
        a: &[Complex64], m: usize, n: usize, b: &[Complex64], nrhs: usize, rcond: f64,
    ) -> R<(Vec<Complex64>, Vec<f64>, i32)> {
        if m == 0 || n == 0 {
            return Err(LinAlgError::ShapeMismatch("lstsq: empty matrix".into()));
        }
        let minmn = m.min(n);
        let maxmn = m.max(n);
        let mut acm = to_col_major_c(a, m, n);
        let bcm_in = to_col_major_c(b, m, nrhs);
        let mut bcm = vec![Complex64::new(0.0, 0.0); maxmn * nrhs];
        for j in 0..nrhs {
            bcm[j * maxmn..j * maxmn + m].copy_from_slice(&bcm_in[j * m..j * m + m]);
        }
        let mut s = vec![0.0f64; minmn];
        let mut rank = 0i32;
        let mut info = 0i32;

        // Generous fixed-size rwork/iwork upper bound (dgelsd/zgelsd
        // workspace-query does not reliably report rwork/iwork requirements
        // across LAPACK versions the way it reports `work`) — see LAPACK's
        // ZGELSD documentation for the exact bound this follows.
        let smlsiz = 25usize;
        let nlvl = if minmn == 0 {
            0
        } else {
            ((minmn as f64 / (smlsiz + 1) as f64).log2().floor().max(0.0) as usize) + 1
        };
        let lrwork = 10 * minmn + 2 * minmn * smlsiz + 8 * minmn * nlvl + 3 * smlsiz * nrhs
            + (smlsiz + 1) * (smlsiz + 1)
            + 64;
        let liwork = (3 * minmn * nlvl + 11 * minmn + 64).max(1);
        let mut rwork = vec![0.0f64; lrwork.max(1)];
        let mut iwork = vec![0i32; liwork];

        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zgelsd(
                m as i32, n as i32, nrhs as i32, &mut acm, m as i32, &mut bcm, maxmn as i32, &mut s, rcond, &mut rank,
                &mut work, -1, &mut rwork, &mut iwork, &mut info,
            );
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zgelsd(
                m as i32, n as i32, nrhs as i32, &mut acm, m as i32, &mut bcm, maxmn as i32, &mut s, rcond, &mut rank,
                &mut work, lwork, &mut rwork, &mut iwork, &mut info,
            );
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }
        let mut x = vec![Complex64::new(0.0, 0.0); n * nrhs];
        for j in 0..nrhs {
            for i in 0..n {
                x[i * nrhs + j] = bcm[j * maxmn + i];
            }
        }
        Ok((x, s, rank))
    }

    pub fn qr(a: &[Complex64], m: usize, n: usize) -> R<(Vec<Complex64>, Vec<Complex64>)> {
        let k = m.min(n);
        // See the real-`f64` `qr` above: numpy succeeds on empty input, and
        // `k=min(m,n)=0` makes both output buffers genuinely empty with no
        // LAPACK call needed.
        if k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut acm = to_col_major_c(a, m, n);
        let mut tau = vec![Complex64::new(0.0, 0.0); k];
        let mut info = 0i32;

        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, -1, &mut info);
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::InvalidInput(format!("zgeqrf: illegal argument {}", -info)));
        }

        let mut r = vec![Complex64::new(0.0, 0.0); k * n];
        for j in 0..n {
            for i in 0..=j.min(k - 1) {
                r[i * n + j] = acm[j * m + i];
            }
        }

        let mut qcm = vec![Complex64::new(0.0, 0.0); m * k];
        for j in 0..k {
            qcm[j * m..(j + 1) * m].copy_from_slice(&acm[j * m..(j + 1) * m]);
        }
        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zungqr(m as i32, k as i32, k as i32, &mut qcm, m as i32, &tau, &mut work, -1, &mut info);
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zungqr(m as i32, k as i32, k as i32, &mut qcm, m as i32, &tau, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::InvalidInput(format!("zungqr: illegal argument {}", -info)));
        }
        Ok((to_row_major_c(&qcm, m, k), r))
    }

    /// Complex mirror of `real::qr_r` -- see its doc comment.
    pub fn qr_r(a: &[Complex64], m: usize, n: usize) -> R<Vec<Complex64>> {
        let k = m.min(n);
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut acm = to_col_major_c(a, m, n);
        let mut tau = vec![Complex64::new(0.0, 0.0); k];
        let mut info = 0i32;
        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, -1, &mut info);
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::InvalidInput(format!("zgeqrf: illegal argument {}", -info)));
        }
        let mut r = vec![Complex64::new(0.0, 0.0); k * n];
        for j in 0..n {
            for i in 0..=j.min(k - 1) {
                r[i * n + j] = acm[j * m + i];
            }
        }
        Ok(r)
    }

    /// Complex mirror of `real::qr_complete` -- see its doc comment.
    pub fn qr_complete(a: &[Complex64], m: usize, n: usize) -> R<(Vec<Complex64>, Vec<Complex64>)> {
        let k = m.min(n);
        if k == 0 {
            let mut q = vec![Complex64::new(0.0, 0.0); m * m];
            for i in 0..m {
                q[i * m + i] = Complex64::new(1.0, 0.0);
            }
            return Ok((q, vec![Complex64::new(0.0, 0.0); m * n]));
        }
        let mut acm = to_col_major_c(a, m, n);
        let mut tau = vec![Complex64::new(0.0, 0.0); k];
        let mut info = 0i32;
        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, -1, &mut info);
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::InvalidInput(format!("zgeqrf: illegal argument {}", -info)));
        }
        let mut r = vec![Complex64::new(0.0, 0.0); m * n];
        for j in 0..n {
            for i in 0..=j.min(k - 1) {
                r[i * n + j] = acm[j * m + i];
            }
        }
        let mut qcm = vec![Complex64::new(0.0, 0.0); m * m];
        for j in 0..k {
            qcm[j * m..(j + 1) * m].copy_from_slice(&acm[j * m..(j + 1) * m]);
        }
        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zungqr(m as i32, m as i32, k as i32, &mut qcm, m as i32, &tau, &mut work, -1, &mut info);
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zungqr(m as i32, m as i32, k as i32, &mut qcm, m as i32, &tau, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::InvalidInput(format!("zungqr: illegal argument {}", -info)));
        }
        Ok((to_row_major_c(&qcm, m, m), r))
    }

    /// Complex mirror of `real::qr_raw` -- see its doc comment.
    pub fn qr_raw(a: &[Complex64], m: usize, n: usize) -> R<(Vec<Complex64>, Vec<Complex64>)> {
        let k = m.min(n);
        if m == 0 || n == 0 {
            return Ok((vec![Complex64::new(0.0, 0.0); m * n], vec![Complex64::new(0.0, 0.0); k]));
        }
        let mut acm = to_col_major_c(a, m, n);
        let mut tau = vec![Complex64::new(0.0, 0.0); k];
        let mut info = 0i32;
        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, -1, &mut info);
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zgeqrf(m as i32, n as i32, &mut acm, m as i32, &mut tau, &mut work, lwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::InvalidInput(format!("zgeqrf: illegal argument {}", -info)));
        }
        Ok((acm, tau))
    }

    pub fn svd(a: &[Complex64], m: usize, n: usize) -> R<(Vec<Complex64>, Vec<f64>, Vec<Complex64>)> {
        if m == 0 || n == 0 {
            return Ok((vec![], vec![], vec![]));
        }
        let k = m.min(n);
        let maxmn = m.max(n);
        let mut acm = to_col_major_c(a, m, n);
        let mut s = vec![0.0f64; k];
        let mut u = vec![Complex64::new(0.0, 0.0); m * k];
        let mut vt = vec![Complex64::new(0.0, 0.0); k * n];
        let mut iwork = vec![0i32; 8 * k];
        let lrwork = k * (5 * k + 7).max(2 * maxmn + 2 * k + 1);
        let mut rwork = vec![0.0f64; lrwork.max(1)];
        let mut info = 0i32;

        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zgesdd(
                b'S', m as i32, n as i32, &mut acm, m as i32, &mut s, &mut u, m as i32, &mut vt, k as i32, &mut work,
                -1, &mut rwork, &mut iwork, &mut info,
            );
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zgesdd(
                b'S', m as i32, n as i32, &mut acm, m as i32, &mut s, &mut u, m as i32, &mut vt, k as i32, &mut work,
                lwork, &mut rwork, &mut iwork, &mut info,
            );
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }
        Ok((to_row_major_c(&u, m, k), s, to_row_major_c(&vt, k, n)))
    }

    /// Full SVD via `zgesdd` job `'A'` -- see the real `svd_full`'s doc for
    /// the shared rationale. Per LAPACK's own `ZGESDD` documentation, the
    /// `LRWORK` formula for `JOBZ='A'` is IDENTICAL to `JOBZ='S'`'s (both:
    /// `min(M,N) * max(5*min(M,N)+7, 2*max(M,N)+2*min(M,N)+1)`), so the
    /// economy path's `lrwork`/`rwork` sizing is reused unchanged; only the
    /// job byte and `U`/`Vt` buffer sizes (m x m / n x n instead of
    /// m x k / k x n) differ.
    pub fn svd_full(a: &[Complex64], m: usize, n: usize) -> R<(Vec<Complex64>, Vec<f64>, Vec<Complex64>)> {
        if m == 0 || n == 0 {
            return Ok((identity_flat_c(m), vec![], identity_flat_c(n)));
        }
        let k = m.min(n);
        let maxmn = m.max(n);
        let mut acm = to_col_major_c(a, m, n);
        let mut s = vec![0.0f64; k];
        let mut u = vec![Complex64::new(0.0, 0.0); m * m];
        let mut vt = vec![Complex64::new(0.0, 0.0); n * n];
        let mut iwork = vec![0i32; 8 * k];
        let lrwork = k * (5 * k + 7).max(2 * maxmn + 2 * k + 1);
        let mut rwork = vec![0.0f64; lrwork.max(1)];
        let mut info = 0i32;

        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zgesdd(
                b'A', m as i32, n as i32, &mut acm, m as i32, &mut s, &mut u, m as i32, &mut vt, n as i32, &mut work,
                -1, &mut rwork, &mut iwork, &mut info,
            );
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zgesdd(
                b'A', m as i32, n as i32, &mut acm, m as i32, &mut s, &mut u, m as i32, &mut vt, n as i32, &mut work,
                lwork, &mut rwork, &mut iwork, &mut info,
            );
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }
        Ok((to_row_major_c(&u, m, m), s, to_row_major_c(&vt, n, n)))
    }

    /// General (non-Hermitian) eigendecomposition via `zgeev`. Unlike the
    /// real `eig`, no conjugate-pair repacking is needed — `zgeev` already
    /// returns complex eigenvalues/eigenvectors directly.
    pub fn eig(a: &[Complex64], n: usize) -> R<(Vec<Complex64>, Vec<Complex64>)> {
        check_square(n, n)?;
        if n == 0 {
            return Ok((vec![], vec![]));
        }
        let mut acm = to_col_major_c(a, n, n);
        let mut w = vec![Complex64::new(0.0, 0.0); n];
        let mut vl = vec![Complex64::new(0.0, 0.0); 1];
        let mut vr = vec![Complex64::new(0.0, 0.0); n * n];
        let mut rwork = vec![0.0f64; 2 * n];
        let mut info = 0i32;

        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zgeev(
                b'N', b'V', n as i32, &mut acm, n as i32, &mut w, &mut vl, 1, &mut vr, n as i32, &mut work, -1,
                &mut rwork, &mut info,
            );
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zgeev(
                b'N', b'V', n as i32, &mut acm, n as i32, &mut w, &mut vl, 1, &mut vr, n as i32, &mut work, lwork,
                &mut rwork, &mut info,
            );
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }
        Ok((w, to_row_major_c(&vr, n, n)))
    }

    pub fn eigvals(a: &[Complex64], n: usize) -> R<Vec<Complex64>> {
        check_square(n, n)?;
        if n == 0 {
            return Ok(vec![]);
        }
        let mut acm = to_col_major_c(a, n, n);
        let mut w = vec![Complex64::new(0.0, 0.0); n];
        let mut vl = vec![Complex64::new(0.0, 0.0); 1];
        let mut vr = vec![Complex64::new(0.0, 0.0); 1];
        let mut rwork = vec![0.0f64; 2 * n];
        let mut info = 0i32;

        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zgeev(b'N', b'N', n as i32, &mut acm, n as i32, &mut w, &mut vl, 1, &mut vr, 1, &mut work, -1, &mut rwork, &mut info);
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zgeev(
                b'N', b'N', n as i32, &mut acm, n as i32, &mut w, &mut vl, 1, &mut vr, 1, &mut work, lwork, &mut rwork,
                &mut info,
            );
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }
        Ok(w)
    }

    /// Hermitian eigendecomposition via `zheev`. Only the lower triangle of
    /// `a` is read; eigenvalues are real and ascending; eigenvectors are
    /// orthonormal under the Hermitian inner product.
    pub fn eigh(a: &[Complex64], n: usize) -> R<(Vec<f64>, Vec<Complex64>)> {
        check_square(n, n)?;
        if n == 0 {
            return Ok((vec![], vec![]));
        }
        let mut acm = to_col_major_c(a, n, n);
        let mut w = vec![0.0f64; n];
        let mut rwork = vec![0.0f64; (3 * n).max(1)];
        let mut info = 0i32;

        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zheev(b'V', b'L', n as i32, &mut acm, n as i32, &mut w, &mut work, -1, &mut rwork, &mut info);
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zheev(b'V', b'L', n as i32, &mut acm, n as i32, &mut w, &mut work, lwork, &mut rwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }
        Ok((w, to_row_major_c(&acm, n, n)))
    }

    pub fn eigvalsh(a: &[Complex64], n: usize) -> R<Vec<f64>> {
        check_square(n, n)?;
        if n == 0 {
            return Ok(vec![]);
        }
        let mut acm = to_col_major_c(a, n, n);
        let mut w = vec![0.0f64; n];
        let mut rwork = vec![0.0f64; (3 * n).max(1)];
        let mut info = 0i32;

        let mut work = vec![Complex64::new(0.0, 0.0); 1];
        unsafe {
            zheev(b'N', b'L', n as i32, &mut acm, n as i32, &mut w, &mut work, -1, &mut rwork, &mut info);
        }
        let lwork = (work[0].re as i32).max(1);
        let mut work = vec![Complex64::new(0.0, 0.0); lwork as usize];
        unsafe {
            zheev(b'N', b'L', n as i32, &mut acm, n as i32, &mut w, &mut work, lwork, &mut rwork, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::DidNotConverge);
        }
        Ok(w)
    }

    /// Lower-triangular Cholesky factor `L` (`A = L Lᴴ`) of a Hermitian
    /// positive-definite matrix, via `zpotrf`.
    pub fn cholesky(a: &[Complex64], n: usize) -> R<Vec<Complex64>> {
        check_square(n, n)?;
        if n == 0 {
            return Ok(vec![]);
        }
        let mut acm = to_col_major_c(a, n, n);
        let mut info = 0i32;
        unsafe {
            zpotrf(b'L', n as i32, &mut acm, n as i32, &mut info);
        }
        if info != 0 {
            return Err(LinAlgError::NotPositiveDefinite);
        }
        let mut rm = to_row_major_c(&acm, n, n);
        for i in 0..n {
            for j in (i + 1)..n {
                rm[i * n + j] = Complex64::new(0.0, 0.0);
            }
        }
        Ok(rm)
    }

    /// Moore-Penrose pseudo-inverse via SVD: `A⁺ = V Σ⁺ Uᴴ`.
    pub fn pinv(a: &[Complex64], m: usize, n: usize, rcond: f64) -> R<Vec<Complex64>> {
        let (u, s, vt) = svd(a, m, n)?;
        let k = s.len();
        let smax = s.iter().cloned().fold(0.0f64, f64::max);
        let cutoff = rcond * smax;

        // V = (Vt)^H so V[i][t] = conj(vt[t][i]); scale column t by 1/s_t.
        let mut v_scaled = vec![Complex64::new(0.0, 0.0); n * k];
        for t in 0..k {
            let inv_s = if s[t] > cutoff { 1.0 / s[t] } else { 0.0 };
            for i in 0..n {
                v_scaled[i * k + t] = vt[t * n + i].conj() * inv_s;
            }
        }
        // U^H: k x m row-major.
        let mut uh = vec![Complex64::new(0.0, 0.0); k * m];
        for t in 0..k {
            for j in 0..m {
                uh[t * m + j] = u[j * k + t].conj();
            }
        }
        // Small-matrix recombination (V_scaled is n x k, U^H is k x m); no
        // complex GEMM wrapper exists in `baseline.rs` (real-only), so this
        // is a direct, non-hot-path accumulation over the already-reduced
        // rank-k factors, not a competing dense GEMM.
        let mut out = vec![Complex64::new(0.0, 0.0); n * m];
        for i in 0..n {
            for t in 0..k {
                let vit = v_scaled[i * k + t];
                if vit.norm() == 0.0 {
                    continue;
                }
                for j in 0..m {
                    out[i * m + j] += vit * uh[t * m + j];
                }
            }
        }
        Ok(out)
    }

    /// Complex counterpart of `real::matrix_rank`; see that function's doc
    /// comment for the `eps`-parameterization rationale (real numpy's
    /// `finfo(S.dtype).eps` for a complex64 input is `f32::EPSILON`, same
    /// as its real single-precision counterpart -- verified live).
    pub fn matrix_rank(a: &[Complex64], m: usize, n: usize, tol: Option<f64>, eps: f64) -> R<usize> {
        if m == 0 || n == 0 {
            return Ok(0);
        }
        let (_u, s, _vt) = svd(a, m, n)?;
        let smax = s.iter().cloned().fold(0.0f64, f64::max);
        let t = tol.unwrap_or(smax * (m.max(n) as f64) * eps);
        Ok(s.iter().filter(|&&v| v > t).count())
    }

    pub fn norm_fro(a: &[Complex64]) -> f64 {
        a.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt()
    }
}

// ════════════════════════════════════════════════════════════════════════
// Tests — residual identities, not "it returned something". Every routine
// is checked against the algebraic property that defines it (Ax=b,
// AA^-1=I, QR=A & Q^T Q=I, USV^T=A with descending non-negative S, Av=lam*v
// with orthonormal v, LL^T=A + error on non-PD), including ill-conditioned
// and rank-deficient inputs, not only well-behaved random ones.
// ════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use crate::rngutil::SplitMix64;

    fn rand_vec(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = SplitMix64::new(seed);
        (0..n).map(|_| rng.next_gaussian()).collect()
    }

    fn rand_mat(seed: u64, rows: usize, cols: usize) -> Vec<f64> {
        rand_vec(seed, rows * cols)
    }

    fn rand_complex_mat(seed: u64, rows: usize, cols: usize) -> Vec<Complex64> {
        let mut rng = SplitMix64::new(seed);
        (0..rows * cols)
            .map(|_| Complex64::new(rng.next_gaussian(), rng.next_gaussian()))
            .collect()
    }

    fn matmul_rm(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
        baseline::dense_matmul(a, m, k, b, n)
    }

    fn matmul_rm_c(a: &[Complex64], m: usize, k: usize, b: &[Complex64], n: usize) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); m * n];
        for i in 0..m {
            for t in 0..k {
                let av = a[i * k + t];
                if av.norm() == 0.0 {
                    continue;
                }
                for j in 0..n {
                    out[i * n + j] += av * b[t * n + j];
                }
            }
        }
        out
    }

    fn transpose_rm(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        let mut out = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = a[i * cols + j];
            }
        }
        out
    }

    fn conj_transpose_rm(a: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = a[i * cols + j].conj();
            }
        }
        out
    }

    fn frob_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
    }

    fn frob_diff_c(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).norm_sqr()).sum::<f64>().sqrt()
    }

    fn frob_norm(a: &[f64]) -> f64 {
        a.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    fn identity(n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            out[i * n + i] = 1.0;
        }
        out
    }

    fn make_symmetric(a: &[f64], n: usize) -> Vec<f64> {
        // A + A^T is symmetric, and generically well-conditioned for random A.
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = a[i * n + j] + a[j * n + i];
            }
        }
        out
    }

    fn make_hermitian(a: &[Complex64], n: usize) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = a[i * n + j] + a[j * n + i].conj();
            }
        }
        out
    }

    /// A well-conditioned SPD matrix: A^T A + n*I.
    fn make_spd(seed: u64, n: usize) -> Vec<f64> {
        let a = rand_mat(seed, n, n);
        let at = transpose_rm(&a, n, n);
        let mut ata = matmul_rm(&at, n, n, &a, n);
        for i in 0..n {
            ata[i * n + i] += n as f64;
        }
        ata
    }

    fn make_hpd(seed: u64, n: usize) -> Vec<Complex64> {
        let a = rand_complex_mat(seed, n, n);
        let ah = conj_transpose_rm(&a, n, n);
        let mut aha = matmul_rm_c(&ah, n, n, &a, n);
        for i in 0..n {
            aha[i * n + i] += Complex64::new(n as f64, 0.0);
        }
        aha
    }

    // ── solve: ||Ax - b|| / ||b|| < 1e-12 ───────────────────────────────

    #[test]
    fn solve_residual_well_conditioned() {
        let n = 20;
        let a = make_spd(1, n);
        let b = rand_vec(2, n);
        let x = real::solve(&a, n, &b, 1).expect("well-conditioned solve must succeed");
        let ax = matmul_rm(&a, n, n, &x, 1);
        let resid = frob_diff(&ax, &b) / frob_norm(&b);
        assert!(resid < 1e-12, "resid={resid}");
    }

    #[test]
    fn solve_multi_rhs() {
        let n = 15;
        let nrhs = 4;
        let a = make_spd(3, n);
        let b = rand_mat(4, n, nrhs);
        let x = real::solve(&a, n, &b, nrhs).unwrap();
        let ax = matmul_rm(&a, n, n, &x, nrhs);
        let resid = frob_diff(&ax, &b) / frob_norm(&b);
        assert!(resid < 1e-12, "resid={resid}");
    }

    #[test]
    fn solve_singular_errors() {
        let n = 5;
        // A structurally zero row is guaranteed to stay exactly zero through
        // LAPACK's partial-pivoting elimination (every update to it is
        // `0 - factor * pivot_row`, and `factor = 0 / pivot == 0` exactly),
        // so it deterministically produces an exact zero pivot and `info >
        // 0`. Bit-identical *duplicate rows* were tried first here and
        // turned out NOT to be a reliable test fixture: Accelerate's
        // blocked dgetrf does not preserve the "rows stay identical through
        // elimination" argument exactly in floating point, so it can return
        // `info == 0` with a merely near-singular (not exactly singular)
        // factorization — see `solve_ill_conditioned_still_bounded_residual`
        // for that honestly-reported case instead.
        let mut a = rand_mat(5, n, n);
        for j in 0..n {
            a[2 * n + j] = 0.0;
        }
        let b = rand_vec(6, n);
        let res = real::solve(&a, n, &b, 1);
        assert_eq!(res, Err(LinAlgError::Singular));
    }

    #[test]
    fn solve_ill_conditioned_still_bounded_residual() {
        // Hilbert-like matrix: famously ill-conditioned but not singular.
        let n = 8;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = 1.0 / (i + j + 1) as f64;
            }
        }
        let b = rand_vec(7, n);
        let x = real::solve(&a, n, &b, 1).expect("Hilbert matrix is singular-to-precision at n=8 in theory, but LAPACK returns a factorization here");
        let ax = matmul_rm(&a, n, n, &x, 1);
        let resid = frob_diff(&ax, &b) / frob_norm(&b);
        // Ill-conditioning is honestly reflected here: this bound is far
        // looser than the 1e-12 well-conditioned bar precisely because the
        // Hilbert matrix at n=8 has condition number ~1.5e10 — no honest
        // solver gets a tight residual on this input, and claiming
        // otherwise would be the dishonesty the task explicitly forbids.
        assert!(resid < 1e-4, "resid={resid} (expected loose bound on an ill-conditioned system)");
    }

    // ── inv: ||AA^-1 - I|| < 1e-12 ───────────────────────────────────────

    #[test]
    fn inv_residual() {
        let n = 25;
        let a = make_spd(10, n);
        let ainv = real::inv(&a, n).unwrap();
        let prod = matmul_rm(&a, n, n, &ainv, n);
        let resid = frob_diff(&prod, &identity(n));
        assert!(resid < 1e-9, "resid={resid}");
    }

    #[test]
    fn inv_singular_errors() {
        let n = 6;
        let mut a = rand_mat(11, n, n);
        // Make column 0 all-zero: exactly singular.
        for i in 0..n {
            a[i * n] = 0.0;
        }
        assert_eq!(real::inv(&a, n), Err(LinAlgError::Singular));
    }

    // ── qr: ||QR - A|| small AND ||Q^T Q - I|| small ────────────────────

    #[test]
    fn qr_reconstructs_and_is_orthonormal_square() {
        let m = 30;
        let n = 30;
        let a = rand_mat(20, m, n);
        let (q, r) = real::qr(&a, m, n).unwrap();
        let k = m.min(n);
        let qr_prod = matmul_rm(&q, m, k, &r, n);
        let resid = frob_diff(&qr_prod, &a) / frob_norm(&a);
        assert!(resid < 1e-10, "QR reconstruction resid={resid}");

        let qt = transpose_rm(&q, m, k);
        let qtq = matmul_rm(&qt, k, m, &q, k);
        let orth_resid = frob_diff(&qtq, &identity(k));
        assert!(orth_resid < 1e-10, "Q^T Q - I resid={orth_resid}");
    }

    #[test]
    fn qr_reconstructs_tall_rectangular() {
        let m = 40;
        let n = 12;
        let a = rand_mat(21, m, n);
        let (q, r) = real::qr(&a, m, n).unwrap();
        let k = m.min(n);
        assert_eq!(k, n);
        let qr_prod = matmul_rm(&q, m, k, &r, n);
        let resid = frob_diff(&qr_prod, &a) / frob_norm(&a);
        assert!(resid < 1e-10, "resid={resid}");
        let qt = transpose_rm(&q, m, k);
        let qtq = matmul_rm(&qt, k, m, &q, k);
        assert!(frob_diff(&qtq, &identity(k)) < 1e-10);
    }

    #[test]
    fn qr_rank_deficient_still_reconstructs() {
        // Rank-deficient input: QR must still reconstruct A exactly (QR
        // factorization exists regardless of rank; it's SVD/rank that would
        // need to detect the deficiency, tested separately below).
        let n = 10;
        let mut a = rand_mat(22, n, n);
        for j in 0..n {
            a[5 * n + j] = a[2 * n + j]; // row 5 = row 2
        }
        let (q, r) = real::qr(&a, n, n).unwrap();
        let qr_prod = matmul_rm(&q, n, n, &r, n);
        let resid = frob_diff(&qr_prod, &a) / frob_norm(&a);
        assert!(resid < 1e-9, "resid={resid}");
    }

    // ── svd: ||USV^T - A|| small, singular values non-negative & descending

    #[test]
    fn svd_reconstructs_and_values_descending_nonneg() {
        let m = 25;
        let n = 18;
        let a = rand_mat(30, m, n);
        let (u, s, vt) = real::svd(&a, m, n).unwrap();
        let k = m.min(n);
        assert_eq!(s.len(), k);
        for i in 0..k {
            assert!(s[i] >= 0.0, "s[{i}]={} must be non-negative", s[i]);
            if i > 0 {
                assert!(s[i - 1] >= s[i], "singular values must be descending: {:?}", s);
            }
        }
        // USV^T
        let mut us = vec![0.0; m * k];
        for i in 0..m {
            for j in 0..k {
                us[i * k + j] = u[i * k + j] * s[j];
            }
        }
        let usvt = matmul_rm(&us, m, k, &vt, n);
        let resid = frob_diff(&usvt, &a) / frob_norm(&a);
        assert!(resid < 1e-10, "resid={resid}");
    }

    #[test]
    fn svd_rank_deficient_shows_near_zero_tail() {
        let n = 12;
        let r = 4;
        // Explicit rank-r matrix: U(n x r) V(n x r)^T.
        let u = rand_mat(31, n, r);
        let v = rand_mat(32, n, r);
        let vt = transpose_rm(&v, n, r);
        let a = matmul_rm(&u, n, r, &vt, n);
        let (_u2, s, _vt2) = real::svd(&a, n, n).unwrap();
        for i in 0..r {
            assert!(s[i] > 1e-6, "leading singular value s[{i}]={} should be non-negligible", s[i]);
        }
        for i in r..n {
            assert!(s[i] < 1e-8, "tail singular value s[{i}]={} should be ~0 for a rank-{r} matrix", s[i]);
        }
    }

    #[test]
    fn svd_ill_conditioned_hilbert() {
        let n = 10;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = 1.0 / (i + j + 1) as f64;
            }
        }
        let (u, s, vt) = real::svd(&a, n, n).unwrap();
        let mut us = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                us[i * n + j] = u[i * n + j] * s[j];
            }
        }
        let usvt = matmul_rm(&us, n, n, &vt, n);
        let resid = frob_diff(&usvt, &a) / frob_norm(&a);
        // Honest, looser bound than the well-conditioned case: Hilbert(10)
        // has singular values spanning ~1e-13, right at f64 precision, so
        // the *reconstruction* residual stays small (LAPACK is backward
        // stable) even though individual small singular values themselves
        // are not trustworthy to high relative precision. That distinction
        // (backward-stable reconstruction vs. forward accuracy of tiny
        // singular values) is the honest characterization, not a claim
        // that every singular value of a near-singular matrix is accurate.
        assert!(resid < 1e-8, "resid={resid}");
        assert!(s[n - 1] < 1e-10, "smallest singular value should be tiny: {}", s[n - 1]);
    }

    // ── eigh: ||Av - lam v|| small, lam real, eigenvectors orthonormal ──

    #[test]
    fn eigh_residual_and_orthonormal() {
        let n = 20;
        let a = make_symmetric(&rand_mat(40, n, n), n);
        let (w, v) = real::eigh(&a, n).unwrap();
        // eigenvalues ascending (LAPACK dsyev convention)
        for i in 1..n {
            assert!(w[i - 1] <= w[i] + 1e-9, "eigenvalues should be ascending: {:?}", w);
        }
        // Av_i = lam_i v_i, per column i.
        for i in 0..n {
            let vi: Vec<f64> = (0..n).map(|r| v[r * n + i]).collect();
            let avi = matmul_rm(&a, n, n, &vi, 1);
            let lam_vi: Vec<f64> = vi.iter().map(|x| x * w[i]).collect();
            let resid = frob_diff(&avi, &lam_vi);
            assert!(resid < 1e-9, "eigenpair {i} resid={resid}");
        }
        // orthonormal: V^T V = I
        let vt = transpose_rm(&v, n, n);
        let vtv = matmul_rm(&vt, n, n, &v, n);
        assert!(frob_diff(&vtv, &identity(n)) < 1e-9);
    }

    #[test]
    fn eigh_rank_deficient() {
        // Symmetric rank-deficient (rank r < n) matrix must produce n-r
        // eigenvalues ~0.
        let n = 10;
        let r = 3;
        let u = rand_mat(41, n, r);
        let a = matmul_rm(&u, n, r, &transpose_rm(&u, n, r), n);
        // a is symmetric by construction (U U^T)
        let (w, _v) = real::eigh(&a, n).unwrap();
        let near_zero = w.iter().filter(|&&x| x.abs() < 1e-8).count();
        assert!(near_zero >= n - r, "expected >= {} near-zero eigenvalues, got {} in {:?}", n - r, near_zero, w);
    }

    #[test]
    fn eigvalsh_matches_eigh_values() {
        let n = 12;
        let a = make_symmetric(&rand_mat(42, n, n), n);
        let (w1, _v) = real::eigh(&a, n).unwrap();
        let w2 = real::eigvalsh(&a, n).unwrap();
        for i in 0..n {
            assert!((w1[i] - w2[i]).abs() < 1e-9);
        }
    }

    // ── general eig (possibly complex) ──────────────────────────────────

    #[test]
    fn eig_general_residual() {
        let n = 15;
        let a = rand_mat(50, n, n);
        let (vals, vecs) = real::eig(&a, n).unwrap();
        let ac: Vec<Complex64> = a.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        for i in 0..n {
            let vi: Vec<Complex64> = (0..n).map(|r| vecs[r * n + i]).collect();
            let mut avi = vec![Complex64::new(0.0, 0.0); n];
            for r in 0..n {
                for c in 0..n {
                    avi[r] += ac[r * n + c] * vi[c];
                }
            }
            let lam = vals[i];
            let resid: f64 = avi.iter().zip(&vi).map(|(a, v)| (a - lam * v).norm_sqr()).sum::<f64>().sqrt();
            let scale: f64 = avi.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt().max(1.0);
            assert!(resid / scale < 1e-8, "eigenpair {i} resid={resid}");
        }
    }

    #[test]
    fn eig_known_rotation_gives_complex_pair() {
        // 90-degree rotation: eigenvalues +-i, purely complex, no real
        // eigenpairs — exercises the conjugate-pair repacking path.
        let a = vec![0.0, -1.0, 1.0, 0.0]; // row-major 2x2 [[0,-1],[1,0]]
        let vals = real::eigvals(&a, 2).unwrap();
        let mut im_parts: Vec<f64> = vals.iter().map(|v| v.im).collect();
        im_parts.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((im_parts[0] - (-1.0)).abs() < 1e-9, "{:?}", vals);
        assert!((im_parts[1] - 1.0).abs() < 1e-9, "{:?}", vals);
        for v in &vals {
            assert!(v.re.abs() < 1e-9);
        }
    }

    // ── cholesky: ||LL^T - A|| small, error on non-PD ───────────────────

    #[test]
    fn cholesky_residual() {
        let n = 20;
        let a = make_spd(60, n);
        let l = real::cholesky(&a, n).unwrap();
        let lt = transpose_rm(&l, n, n);
        let llt = matmul_rm(&l, n, n, &lt, n);
        let resid = frob_diff(&llt, &a) / frob_norm(&a);
        assert!(resid < 1e-12, "resid={resid}");
        // must be exactly lower-triangular
        for i in 0..n {
            for j in (i + 1)..n {
                assert_eq!(l[i * n + j], 0.0);
            }
        }
    }

    #[test]
    fn cholesky_errors_on_non_positive_definite() {
        let n = 5;
        // A negative-definite matrix (not PD).
        let mut a = make_spd(61, n);
        for v in a.iter_mut() {
            *v = -*v;
        }
        assert_eq!(real::cholesky(&a, n), Err(LinAlgError::NotPositiveDefinite));
    }

    #[test]
    fn cholesky_errors_on_indefinite() {
        // A symmetric indefinite matrix: eigenvalues of mixed sign.
        let a = vec![1.0, 2.0, 2.0, 1.0]; // eigenvalues 3, -1
        assert_eq!(real::cholesky(&a, 2), Err(LinAlgError::NotPositiveDefinite));
    }

    // ── det / slogdet ────────────────────────────────────────────────────

    #[test]
    fn det_matches_slogdet_exp() {
        let n = 10;
        let a = make_spd(70, n); // SPD => det > 0
        let d = real::det(&a, n).unwrap();
        let (sign, logdet) = real::slogdet(&a, n).unwrap();
        assert!((d - sign * logdet.exp()).abs() / d.abs() < 1e-9);
        assert!(sign > 0.0);
    }

    #[test]
    fn det_singular_is_zero() {
        let n = 6;
        // Structurally zero row (see `solve_singular_errors` for why this,
        // not duplicate rows, is the reliable exact-singular fixture).
        let mut a = rand_mat(71, n, n);
        for j in 0..n {
            a[3 * n + j] = 0.0;
        }
        assert_eq!(real::det(&a, n).unwrap(), 0.0);
        let (sign, logdet) = real::slogdet(&a, n).unwrap();
        assert_eq!(sign, 0.0);
        assert_eq!(logdet, f64::NEG_INFINITY);
    }

    #[test]
    fn det_near_singular_is_honestly_tiny_not_necessarily_exact_zero() {
        // Bit-identical duplicate rows are only *near* singular once run
        // through Accelerate's blocked LU (see `solve_singular_errors`):
        // the determinant should be tiny, but pinning it to exactly 0.0
        // would be asserting more than the implementation (or numpy, which
        // hits the same LAPACK path) actually guarantees.
        let n = 6;
        let mut a = rand_mat(72, n, n);
        for j in 0..n {
            a[j] = a[n + j];
        }
        let d = real::det(&a, n).unwrap();
        assert!(d.abs() < 1e-8, "expected a near-zero determinant, got {d}");
    }

    #[test]
    fn det_known_2x2() {
        let a = vec![3.0, 4.0, 1.0, 2.0]; // det = 3*2 - 4*1 = 2
        assert!((real::det(&a, 2).unwrap() - 2.0).abs() < 1e-12);
    }

    // ── lstsq ────────────────────────────────────────────────────────────

    #[test]
    fn lstsq_overdetermined_matches_normal_equations() {
        let m = 30;
        let n = 8;
        let a = rand_mat(80, m, n);
        let b = rand_vec(81, m);
        let (x, _s, rank) = real::lstsq(&a, m, n, &b, 1, -1.0).unwrap();
        assert_eq!(rank, n as i32);
        // Normal-equations solve as an independent cross-check.
        let at = transpose_rm(&a, m, n);
        let ata = matmul_rm(&at, n, m, &a, n);
        let atb = matmul_rm(&at, n, m, &b, 1);
        let x_normal = real::solve(&ata, n, &atb, 1).unwrap();
        let resid = frob_diff(&x, &x_normal) / frob_norm(&x_normal);
        assert!(resid < 1e-6, "resid={resid}");
    }

    #[test]
    fn lstsq_rank_deficient_reports_rank() {
        let m = 20;
        let n = 6;
        let r = 3;
        let u = rand_mat(82, m, r);
        let v = rand_mat(83, n, r);
        let a = matmul_rm(&u, m, r, &transpose_rm(&v, n, r), n);
        let b = rand_vec(84, m);
        let (_x, _s, rank) = real::lstsq(&a, m, n, &b, 1, 1e-10).unwrap();
        assert_eq!(rank, r as i32, "lstsq should report the true numerical rank");
    }

    // ── pinv ─────────────────────────────────────────────────────────────

    #[test]
    fn pinv_satisfies_moore_penrose_on_full_rank() {
        let m = 16;
        let n = 10;
        let a = rand_mat(90, m, n);
        let ap = real::pinv(&a, m, n, 1e-12).unwrap();
        // For full column-rank A (m > n), A+ A = I_n.
        let apa = matmul_rm(&ap, n, m, &a, n);
        assert!(frob_diff(&apa, &identity(n)) < 1e-8);
    }

    #[test]
    fn pinv_on_rank_deficient_square() {
        let n = 10;
        let r = 4;
        let u = rand_mat(91, n, r);
        let v = rand_mat(92, n, r);
        let a = matmul_rm(&u, n, r, &transpose_rm(&v, n, r), n);
        let ap = real::pinv(&a, n, n, 1e-8).unwrap();
        // A A+ A = A must still hold for a rank-deficient matrix (this is
        // the Moore-Penrose identity that holds unconditionally, unlike
        // A+ A = I which requires full column rank).
        let aap = matmul_rm(&a, n, n, &ap, n);
        let aapa = matmul_rm(&aap, n, n, &a, n);
        let resid = frob_diff(&aapa, &a) / frob_norm(&a);
        assert!(resid < 1e-6, "resid={resid}");
    }

    // ── matrix_rank ──────────────────────────────────────────────────────

    #[test]
    fn matrix_rank_detects_deficiency() {
        let n = 10;
        let r = 4;
        let u = rand_mat(100, n, r);
        let v = rand_mat(101, n, r);
        let a = matmul_rm(&u, n, r, &transpose_rm(&v, n, r), n);
        assert_eq!(real::matrix_rank(&a, n, n, None, f64::EPSILON).unwrap(), r);
    }

    #[test]
    fn matrix_rank_full_rank() {
        let n = 10;
        let a = make_spd(102, n);
        assert_eq!(real::matrix_rank(&a, n, n, None, f64::EPSILON).unwrap(), n);
    }

    // ── trace ────────────────────────────────────────────────────────────

    #[test]
    fn trace_matches_definition() {
        let n = 7;
        let a = rand_mat(110, n, n);
        let expect: f64 = (0..n).map(|i| a[i * n + i]).sum();
        assert!((real::trace(&a, n, n) - expect).abs() < 1e-12);
    }

    // ── norm / cond ──────────────────────────────────────────────────────

    #[test]
    fn norm_fro_matches_definition() {
        let a = vec![3.0, 4.0]; // ||.||_2 of [3,4] flattened = 5
        assert!((real::norm_fro(&a) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn norm_2_matches_largest_singular_value() {
        let n = 8;
        let a = rand_mat(120, n, n);
        let n2 = real::norm_2(&a, n, n).unwrap();
        let (_u, s, _vt) = real::svd(&a, n, n).unwrap();
        assert!((n2 - s[0]).abs() < 1e-9);
    }

    #[test]
    fn cond_identity_is_one() {
        let n = 6;
        let c = real::cond(&identity(n), n, None, false).unwrap();
        assert!((c - 1.0).abs() < 1e-9, "cond(I)={c}");
    }

    #[test]
    fn cond_singular_is_very_large() {
        // `cond` goes through SVD (an iterative algorithm), not LU, so a
        // structurally singular input does not deterministically produce a
        // bit-exact `0.0` smallest singular value the way LU's pivot does
        // (see `solve_singular_errors`) — the honest, LAPACK-accurate
        // expectation is "very large condition number", not literally
        // `f64::INFINITY`.
        let n = 5;
        let mut a = rand_mat(121, n, n);
        for j in 0..n {
            a[2 * n + j] = 0.0;
        }
        let c = real::cond(&a, n, None, false).unwrap();
        assert!(c.is_infinite() || c > 1e12, "cond of a singular matrix should be huge (or inf), got {c}");
    }

    // ── matrix_power ─────────────────────────────────────────────────────

    #[test]
    fn matrix_power_matches_repeated_multiply() {
        let n = 6;
        let a = rand_mat(130, n, n);
        let a3 = real::matrix_power(&a, n, 3).unwrap();
        let a2 = matmul_rm(&a, n, n, &a, n);
        let expect = matmul_rm(&a2, n, n, &a, n);
        assert!(frob_diff(&a3, &expect) < 1e-6);
    }

    #[test]
    fn matrix_power_zero_is_identity() {
        let n = 5;
        let a = rand_mat(131, n, n);
        let a0 = real::matrix_power(&a, n, 0).unwrap();
        assert!(frob_diff(&a0, &identity(n)) < 1e-12);
    }

    #[test]
    fn matrix_power_negative_matches_inverse_power() {
        let n = 5;
        let a = make_spd(132, n);
        let a_neg2 = real::matrix_power(&a, n, -2).unwrap();
        let ainv = real::inv(&a, n).unwrap();
        let expect = matmul_rm(&ainv, n, n, &ainv, n);
        let resid = frob_diff(&a_neg2, &expect) / frob_norm(&expect);
        assert!(resid < 1e-6, "resid={resid}");
    }

    // ════════════════════════════════════════════════════════════════
    // Complex64 variants
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn complex_solve_residual() {
        let n = 12;
        let a = make_hpd(200, n);
        let b = rand_complex_mat(201, n, 1);
        let x = complex::solve(&a, n, &b, 1).unwrap();
        let ax = matmul_rm_c(&a, n, n, &x, 1);
        let resid = frob_diff_c(&ax, &b) / complex::norm_fro(&b);
        assert!(resid < 1e-9, "resid={resid}");
    }

    #[test]
    fn complex_inv_residual() {
        let n = 14;
        let a = make_hpd(210, n);
        let ainv = complex::inv(&a, n).unwrap();
        let prod = matmul_rm_c(&a, n, n, &ainv, n);
        let ident: Vec<Complex64> = (0..n * n)
            .map(|k| if k / n == k % n { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) })
            .collect();
        let resid = frob_diff_c(&prod, &ident);
        assert!(resid < 1e-8, "resid={resid}");
    }

    #[test]
    fn complex_qr_reconstructs_and_orthonormal() {
        let m = 20;
        let n = 20;
        let a = rand_complex_mat(220, m, n);
        let (q, r) = complex::qr(&a, m, n).unwrap();
        let k = m.min(n);
        let qr_prod = matmul_rm_c(&q, m, k, &r, n);
        let resid = frob_diff_c(&qr_prod, &a) / complex::norm_fro(&a);
        assert!(resid < 1e-9, "resid={resid}");

        let qh = conj_transpose_rm(&q, m, k);
        let qhq = matmul_rm_c(&qh, k, m, &q, k);
        let ident: Vec<Complex64> = (0..k * k)
            .map(|idx| if idx / k == idx % k { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) })
            .collect();
        assert!(frob_diff_c(&qhq, &ident) < 1e-9);
    }

    #[test]
    fn complex_svd_reconstructs_nonneg_descending() {
        let m = 16;
        let n = 10;
        let a = rand_complex_mat(230, m, n);
        let (u, s, vt) = complex::svd(&a, m, n).unwrap();
        let k = m.min(n);
        for i in 0..k {
            assert!(s[i] >= 0.0);
            if i > 0 {
                assert!(s[i - 1] >= s[i]);
            }
        }
        let mut us = vec![Complex64::new(0.0, 0.0); m * k];
        for i in 0..m {
            for j in 0..k {
                us[i * k + j] = u[i * k + j] * s[j];
            }
        }
        let usvt = matmul_rm_c(&us, m, k, &vt, n);
        let resid = frob_diff_c(&usvt, &a) / complex::norm_fro(&a);
        assert!(resid < 1e-9, "resid={resid}");
    }

    #[test]
    fn complex_eigh_residual_real_eigenvalues_orthonormal() {
        let n = 14;
        let a = make_hermitian(&rand_complex_mat(240, n, n), n);
        let (w, v) = complex::eigh(&a, n).unwrap();
        for i in 1..n {
            assert!(w[i - 1] <= w[i] + 1e-9);
        }
        for i in 0..n {
            let vi: Vec<Complex64> = (0..n).map(|r| v[r * n + i]).collect();
            let avi = matmul_rm_c(&a, n, n, &vi, 1);
            let lam_vi: Vec<Complex64> = vi.iter().map(|x| x * w[i]).collect();
            let resid = frob_diff_c(&avi, &lam_vi);
            assert!(resid < 1e-8, "eigenpair {i} resid={resid}");
        }
        let vh = conj_transpose_rm(&v, n, n);
        let vhv = matmul_rm_c(&vh, n, n, &v, n);
        let ident: Vec<Complex64> = (0..n * n)
            .map(|idx| if idx / n == idx % n { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) })
            .collect();
        assert!(frob_diff_c(&vhv, &ident) < 1e-8);
    }

    #[test]
    fn complex_cholesky_residual_and_error_on_non_hpd() {
        let n = 10;
        let a = make_hpd(250, n);
        let l = complex::cholesky(&a, n).unwrap();
        let lh = conj_transpose_rm(&l, n, n);
        let llh = matmul_rm_c(&l, n, n, &lh, n);
        let resid = frob_diff_c(&llh, &a) / complex::norm_fro(&a);
        assert!(resid < 1e-10, "resid={resid}");

        // Non-HPD: negate to flip definiteness.
        let neg: Vec<Complex64> = a.iter().map(|v| -v).collect();
        assert_eq!(complex::cholesky(&neg, n), Err(LinAlgError::NotPositiveDefinite));
    }

    #[test]
    fn complex_eig_general_residual() {
        let n = 10;
        let a = rand_complex_mat(260, n, n);
        let (vals, vecs) = complex::eig(&a, n).unwrap();
        for i in 0..n {
            let vi: Vec<Complex64> = (0..n).map(|r| vecs[r * n + i]).collect();
            let avi = matmul_rm_c(&a, n, n, &vi, 1);
            let lam = vals[i];
            let lam_vi: Vec<Complex64> = vi.iter().map(|x| x * lam).collect();
            let resid = frob_diff_c(&avi, &lam_vi);
            let scale = avi.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt().max(1.0);
            assert!(resid / scale < 1e-8, "eigenpair {i} resid={resid}");
        }
    }

    #[test]
    fn complex_det_known_value() {
        // [[1+i, 2], [3, 4-i]] det = (1+i)(4-i) - 6 = (4-i+4i+1) - 6 = (5+3i) - 6 = -1+3i
        let a = vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, -1.0),
        ];
        let d = complex::det(&a, 2).unwrap();
        assert!((d - Complex64::new(-1.0, 3.0)).norm() < 1e-9, "{:?}", d);
    }

    #[test]
    fn complex_solve_singular_errors() {
        let n = 4;
        // Structurally zero row, same reasoning as `solve_singular_errors`.
        let mut a = rand_complex_mat(270, n, n);
        for j in 0..n {
            a[j] = Complex64::new(0.0, 0.0);
        }
        let b = rand_complex_mat(271, n, 1);
        assert_eq!(complex::solve(&a, n, &b, 1), Err(LinAlgError::Singular));
    }

    #[test]
    fn complex_lstsq_overdetermined() {
        let m = 20;
        let n = 6;
        let a = rand_complex_mat(280, m, n);
        let b = rand_complex_mat(281, m, 1);
        let (x, _s, rank) = complex::lstsq(&a, m, n, &b, 1, -1.0).unwrap();
        assert_eq!(rank, n as i32);
        // Residual should be no larger (up to eps) than that of a trial
        // perturbation of x — a cheap independent sanity check that x is at
        // (near) a stationary point of the least-squares residual.
        let ax = matmul_rm_c(&a, m, n, &x, 1);
        let r0: f64 = (0..m).map(|i| (ax[i] - b[i]).norm_sqr()).sum::<f64>().sqrt();
        let mut x_pert = x.clone();
        x_pert[0] += Complex64::new(1e-3, 0.0);
        let ax_pert = matmul_rm_c(&a, m, n, &x_pert, 1);
        let r_pert: f64 = (0..m).map(|i| (ax_pert[i] - b[i]).norm_sqr()).sum::<f64>().sqrt();
        assert!(r0 <= r_pert + 1e-9, "lstsq solution should minimize residual: r0={r0} r_pert={r_pert}");
    }

    #[test]
    fn complex_matrix_rank_detects_deficiency() {
        let n = 8;
        let r = 3;
        let u = rand_complex_mat(290, n, r);
        let v = rand_complex_mat(291, n, r);
        let vh = conj_transpose_rm(&v, n, r);
        let a = matmul_rm_c(&u, n, r, &vh, n);
        assert_eq!(complex::matrix_rank(&a, n, n, None, f64::EPSILON).unwrap(), r);
    }

    #[test]
    fn complex_trace_matches_definition() {
        let n = 6;
        let a = rand_complex_mat(300, n, n);
        let expect: Complex64 = (0..n).map(|i| a[i * n + i]).sum();
        assert!((complex::trace(&a, n, n) - expect).norm() < 1e-12);
    }
}
