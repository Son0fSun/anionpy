//! `detect` — find the hidden structure in a dense matrix, cheaply. Ports
//! the Python reference implementation's `detect.py` to Rust.
//!
//! The Ion wave operators win big *when the structure is known a priori*.
//! This module removes that precondition: given a dense array, it probes for
//! exploitable structure at sub-cubic cost and returns the best proven
//! operator, or `None` when the matrix is genuinely unstructured (caller
//! falls back to `Dense`, never slower than a plain matvec).
//!
//! **Every detection test here is O(n^2) or better — this is a hard
//! constraint, not a preference.** Diagonal/Toeplitz/circulant are one
//! direct O(n^2) entry-by-entry scan each. Low-rank uses a randomized range
//! finder (Halko-Martinsson-Tropp 2011) at O(n^2 r): a handful of `A @
//! Omega` products (each O(n^2) for the small probe width p) plus a QR of
//! the resulting skinny n x p sketch (O(n p^2), p << n) — no full n x n QR,
//! no SVD, ever, in this file. If detection regressed to O(n^3) — e.g. a
//! full SVD of the operator — the detection cost would swamp the win it is
//! supposed to unlock: break-even would move from ~8-94 applications
//! (see `bench/reference_gate.json`, `dispatch_gate`) to roughly 1000, and
//! the library would lose its reason to exist. That is the exact failure
//! that sank an earlier attempt at this codebase. Keep it true.
//!
//! Detection is *proof-carrying*: nothing here guesses. Every branch is a
//! numerical check against a tolerance before it is returned.

use crate::circulant::Circulant;
use crate::diagonal::Diagonal;
use crate::linalg::{conj_transpose, frobenius_norm, matmul_adj_dense, matmul_dense, qr_orthonormal};
use crate::lowrank::LowRank;
use crate::operator::Operator;
use crate::rngutil::SplitMix64;
use crate::toeplitz::Toeplitz;
use num_complex::Complex64;

fn rel(a: f64, scale: f64) -> f64 {
    if scale > 0.0 {
        a / scale
    } else {
        a
    }
}

/// `a` is a dense n x n matrix, column-major: `a[j*n+i] == A[i,j]`.
/// O(n^2): one pass over every entry.
pub fn is_diagonal(a: &[Complex64], n: usize, rtol: f64) -> bool {
    let mut diag_scale = 0.0f64;
    for i in 0..n {
        diag_scale = diag_scale.max(a[i * n + i].norm());
    }
    if diag_scale == 0.0 {
        return a.iter().all(|v| v.norm() == 0.0);
    }
    let mut max_off = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            if i != j {
                max_off = max_off.max(a[j * n + i].norm());
            }
        }
    }
    rel(max_off, diag_scale) < rtol
}

/// `a` is rows x cols, column-major. O(rows*cols): compares A to itself
/// shifted by (1,1) — constant along every diagonal.
pub fn is_toeplitz(a: &[Complex64], rows: usize, cols: usize, rtol: f64) -> bool {
    if rows < 2 || cols < 2 {
        return true;
    }
    let mut scale = 0.0f64;
    for v in a {
        scale = scale.max(v.norm());
    }
    if scale == 0.0 {
        return true;
    }
    let mut max_diff = 0.0f64;
    for j in 0..(cols - 1) {
        for i in 0..(rows - 1) {
            let d = (a[j * rows + i] - a[(j + 1) * rows + (i + 1)]).norm();
            max_diff = max_diff.max(d);
        }
    }
    rel(max_diff, scale) < rtol
}

/// `a` is n x n, column-major. Toeplitz + wrap-around check. O(n^2).
pub fn is_circulant(a: &[Complex64], n: usize, rtol: f64) -> bool {
    if !is_toeplitz(a, n, n, rtol) {
        return false;
    }
    let mut scale = 0.0f64;
    for v in a {
        scale = scale.max(v.norm());
    }
    if scale == 0.0 {
        return true;
    }
    // c = A[:,0] = a[0..n]; top = A[0,:] = a[j*n] for j in 0..n
    let mut max_diff = 0.0f64;
    for j in 0..n {
        let top = a[j * n];
        let ref_idx = (n - j) % n;
        let refv = a[ref_idx];
        max_diff = max_diff.max((top - refv).norm());
    }
    rel(max_diff, scale) < rtol
}

/// Randomized range finder (Halko-Martinsson-Tropp): probes A(n x m,
/// column-major) with growing random sketches until the captured range's
/// residual on a fresh probe drops below `rtol`, or gives up at `max_rank`
/// (default min(n,m)/4 — beyond that low-rank saves nothing over dense).
/// O(n^2 r): each doubling costs O(n*m*p) for the sketch + O(n*p^2) for the
/// QR of the sketch, p = O(r). No full n x n decomposition is ever formed.
pub fn detect_lowrank(
    a: &[Complex64],
    n: usize,
    m: usize,
    rtol: f64,
    max_rank: Option<usize>,
    oversample: usize,
    seed: u64,
) -> Option<LowRank> {
    let ceiling = max_rank.unwrap_or_else(|| (n.min(m) / 4).max(4));
    let fro = frobenius_norm(a);
    let a_scale = fro / ((n * m).max(1) as f64).sqrt();

    let real = a.iter().all(|v| v.im == 0.0);

    if a_scale == 0.0 {
        return Some(LowRank::new(
            vec![Complex64::new(0.0, 0.0); n],
            vec![Complex64::new(0.0, 0.0); m],
            1,
            n,
            m,
            real,
        ));
    }

    let mut rng = SplitMix64::new(seed);
    let mut r = 4usize;
    while r <= ceiling {
        let p = (r + oversample).min(m);
        let omega = random_real_gaussian(&mut rng, m, p);
        let y = matmul_dense(a, n, m, &omega, p); // n x p
        let q = qr_orthonormal(y, n, p);

        let g = random_real_gaussian(&mut rng, m, 5.min(m).max(1));
        let g_cols = 5.min(m).max(1);
        let ag = matmul_dense(a, n, m, &g, g_cols); // n x g_cols
        let qhag = matmul_adj_dense(&q, n, p, &ag, g_cols); // p x g_cols
        let qqhag = matmul_dense(&q, n, p, &qhag, g_cols); // n x g_cols

        let mut resid_sq = 0.0f64;
        let mut ag_sq = 0.0f64;
        for idx in 0..(n * g_cols) {
            resid_sq += (ag[idx] - qqhag[idx]).norm_sqr();
            ag_sq += ag[idx].norm_sqr();
        }
        let rel_resid = resid_sq.sqrt() / (ag_sq.sqrt() + 1e-300);

        if rel_resid < rtol {
            // `q` has width `p` (rank probe + oversample), but the true
            // rank may be smaller: `qr_orthonormal` zeroes out any column
            // that carried no new information (see its doc comment). Drop
            // those exact-zero columns so the returned `LowRank`'s rank
            // reflects reality instead of the (deliberately oversized)
            // probe width.
            let mut kept: Vec<usize> = Vec::with_capacity(p);
            for k in 0..p {
                let mut norm_sq = 0.0f64;
                for i in 0..n {
                    norm_sq += q[k * n + i].norm_sqr();
                }
                if norm_sq > 0.5 {
                    // surviving columns are unit-normalized (norm ~= 1);
                    // zeroed columns are exact 0.0, so 0.5 cleanly separates them.
                    kept.push(k);
                }
            }
            let rank = kept.len().max(1);
            let mut q_reduced = vec![Complex64::new(0.0, 0.0); n * rank];
            for (new_k, &old_k) in kept.iter().enumerate() {
                q_reduced[new_k * n..(new_k + 1) * n].copy_from_slice(&q[old_k * n..(old_k + 1) * n]);
            }
            let b = matmul_adj_dense(&q_reduced, n, rank, a, m); // rank x m ; A ~= Q B
            let v = conj_transpose(&b, rank, m); // m x rank ; V = B^H
            return Some(LowRank::new(q_reduced, v, rank, n, m, real));
        }
        r *= 2;
    }
    None
}

fn random_real_gaussian(rng: &mut SplitMix64, rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for v in out.iter_mut() {
        v.re = rng.next_gaussian();
    }
    out
}

/// The unified detector. `a` is a dense n x m matrix, column-major.
/// Order is cheapest-and-most-specific first: diagonal -> circulant ->
/// toeplitz -> randomized low-rank. Every branch is verified before it is
/// returned. Returns `None` if `a` is genuinely unstructured.
pub fn detect(a: &[Complex64], n: usize, m: usize, rtol: f64, try_lowrank: bool, seed: u64) -> Option<Box<dyn Operator>> {
    let real = a.iter().all(|v| v.im == 0.0);

    if n == m && is_diagonal(a, n, rtol) {
        let d: Vec<Complex64> = (0..n).map(|i| a[i * n + i]).collect();
        return Some(Box::new(Diagonal::new(d, real)));
    }

    if n == m && is_circulant(a, n, rtol) {
        let c: Vec<Complex64> = a[0..n].to_vec();
        return Some(Box::new(Circulant::new(c, real)));
    }

    if n == m && is_toeplitz(a, n, n, rtol) {
        let col: Vec<Complex64> = a[0..n].to_vec();
        let row: Vec<Complex64> = (0..n).map(|j| a[j * n]).collect();
        return Some(Box::new(Toeplitz::new(col, row, real)));
    }

    if try_lowrank {
        if let Some(lr) = detect_lowrank(a, n, m, rtol, None, 8, seed) {
            if lr.rank() < n.min(m) {
                return Some(Box::new(lr));
            }
        }
    }

    None
}
