//! Small dense complex linear algebra helpers used only by the randomized
//! low-rank detector in `detect.rs`. These operate on the *probe* matrices
//! (n x p, p small — 4..~n/4+8), never on the full n x n operator beyond a
//! constant number of `A @ Omega` products, which is exactly what keeps
//! `detect_lowrank` at O(n^2 r) instead of O(n^3): no full SVD, no full QR
//! of an n x n matrix, only of the skinny n x p sketch.

use num_complex::Complex64;

/// C(n x p) = A(n x m, column-major) @ B(m x p, column-major).
pub fn matmul_dense(a: &[Complex64], n: usize, m: usize, b: &[Complex64], p: usize) -> Vec<Complex64> {
    debug_assert_eq!(a.len(), n * m);
    debug_assert_eq!(b.len(), m * p);
    let mut c = vec![Complex64::new(0.0, 0.0); n * p];
    for j in 0..p {
        for k in 0..m {
            let bkj = b[j * m + k];
            if bkj.norm() == 0.0 {
                continue;
            }
            let acol = &a[k * n..(k + 1) * n];
            let ccol = &mut c[j * n..(j + 1) * n];
            for i in 0..n {
                ccol[i] += acol[i] * bkj;
            }
        }
    }
    c
}

/// C(p x k) = A(n x p, column-major)^H @ B(n x k, column-major).
pub fn matmul_adj_dense(a: &[Complex64], n: usize, p: usize, b: &[Complex64], k: usize) -> Vec<Complex64> {
    debug_assert_eq!(a.len(), n * p);
    debug_assert_eq!(b.len(), n * k);
    let mut c = vec![Complex64::new(0.0, 0.0); p * k];
    for j in 0..k {
        let bcol = &b[j * n..(j + 1) * n];
        for t in 0..p {
            let acol = &a[t * n..(t + 1) * n];
            let mut acc = Complex64::new(0.0, 0.0);
            for i in 0..n {
                acc += acol[i].conj() * bcol[i];
            }
            c[j * p + t] = acc;
        }
    }
    c
}

/// Conjugate-transpose: B(cols x rows) from A(rows x cols), both column-major.
pub fn conj_transpose(a: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut b = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            // A[i,j] -> B[j,i]; B is cols x rows column-major: b[i*cols+j]
            b[i * cols + j] = a[j * rows + i].conj();
        }
    }
    b
}

/// In-place modified Gram-Schmidt: orthonormalise the columns of `y`
/// (n x p, column-major) against each other. O(n p^2) — p is small
/// (rank probe + oversample), so this stays sub-cubic in n.
///
/// When `y`'s true column rank is less than `p` (the common case: the probe
/// width `p` is chosen to *exceed* the guessed rank on purpose, so the
/// range finder can tell "converged" from "needs a bigger probe"), the
/// later columns deflate to pure floating-point noise (~1e-15) after
/// projecting out the earlier, real directions. Normalizing that noise up
/// to a spurious unit-length "orthonormal" column would corrupt `Q` with
/// directions that have nothing to do with `A`'s actual range, silently
/// breaking the projector `Q Q^H` used downstream — so any column whose
/// norm is negligible *relative to the largest column norm seen* is left
/// as an exact zero column instead of being normalized. A fixed absolute
/// threshold (e.g. `1e-300`) is not sufficient here: 1e-15 clears that bar
/// easily while still being pure rounding noise relative to an O(1)-scale
/// input, which is exactly the bug this guards against.
pub fn qr_orthonormal(mut y: Vec<Complex64>, n: usize, p: usize) -> Vec<Complex64> {
    let mut scale = 0.0f64;
    for k in 0..p {
        let mut norm_sq = 0.0f64;
        for i in 0..n {
            norm_sq += y[k * n + i].norm_sqr();
        }
        scale = scale.max(norm_sq.sqrt());
    }
    let zero_tol = scale * 1e-10;

    for k in 0..p {
        let mut norm_sq = 0.0f64;
        for i in 0..n {
            norm_sq += y[k * n + i].norm_sqr();
        }
        let norm = norm_sq.sqrt();
        if norm > zero_tol {
            for i in 0..n {
                y[k * n + i] /= norm;
            }
        } else {
            // Below the true rank: leave this column as exact zero rather
            // than normalizing noise into a fake orthonormal direction.
            for i in 0..n {
                y[k * n + i] = Complex64::new(0.0, 0.0);
            }
            continue;
        }
        for j in (k + 1)..p {
            let mut proj = Complex64::new(0.0, 0.0);
            for i in 0..n {
                proj += y[k * n + i].conj() * y[j * n + i];
            }
            for i in 0..n {
                let qk = y[k * n + i];
                y[j * n + i] -= proj * qk;
            }
        }
    }
    y
}

pub fn frobenius_norm(a: &[Complex64]) -> f64 {
    a.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt()
}

pub fn max_abs_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max)
}
