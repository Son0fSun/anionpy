//! A real dense baseline — row-major f64 GEMV/GEMM, dispatched to Apple's
//! Accelerate BLAS (`cblas_dgemm`/`cblas_dgemv`) via the `cblas` +
//! `blas-src/accelerate` crates. This is what the bench binary measures
//! Ion's structured operators against, and its own achieved GFLOP/s is
//! sanity-checked *before* any speedup number is trusted (see the bench
//! binary's `assert_plausible_gflops`).
//!
//! ## Why BLAS, not a hand-rolled loop
//! A prior hand-rolled rayon triple-loop version of this module measured
//! ~7.85x *slower* than numpy's real `dgemm` on the same machine (Apple M4,
//! numpy 2.5.1) — see `reports/ion-gate-baseline-audit-2026-08-01.md`. That
//! is not a fair baseline: every Ion speedup printed against it was
//! inflated by the gap between "not broken" and "competitive." `numpy.
//! show_config()` on this machine's canonical interpreter confirms numpy's
//! blas/lapack are Accelerate, so this module now calls exactly what numpy
//! calls — same library, same silicon, honest comparison. This is not a
//! from-scratch GEMM; it is Accelerate doing the work, same as it does for
//! numpy. The bench's `numpy_ref` module still cross-checks against the
//! live Python/numpy reference so this claim is verified, not assumed.

use cblas::{Layout, Transpose};
use num_complex::Complex;

pub type C64 = Complex<f32>;
pub type C128 = Complex<f64>;

/// y = A x. A is row-major n_out x n_in.
pub fn dense_matvec(a: &[f64], n_out: usize, n_in: usize, x: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), n_out * n_in);
    assert_eq!(x.len(), n_in);
    let mut y = vec![0.0f64; n_out];
    if n_out == 0 || n_in == 0 {
        return y;
    }
    unsafe {
        cblas::dgemv(
            Layout::RowMajor,
            Transpose::None,
            n_out as i32,
            n_in as i32,
            1.0,
            a,
            n_in as i32,
            x,
            1,
            0.0,
            &mut y,
            1,
        );
    }
    y
}

/// C = A B. A row-major n x k, B row-major k x m, C row-major n x m.
pub fn dense_matmul(a: &[f64], n: usize, k: usize, b: &[f64], m: usize) -> Vec<f64> {
    assert_eq!(a.len(), n * k);
    assert_eq!(b.len(), k * m);
    let mut c = vec![0.0f64; n * m];
    if n == 0 || k == 0 || m == 0 {
        return c;
    }
    unsafe {
        cblas::dgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            a,
            k as i32,
            b,
            m as i32,
            0.0,
            &mut c,
            m as i32,
        );
    }
    c
}

/// f32 sibling of `dense_matmul`, added for `numpy.matmul` dtype coverage
/// (float32 loop `ff->f`). Same row-major/Accelerate-`sgemm` shape.
pub fn dense_matmul_f32(a: &[f32], n: usize, k: usize, b: &[f32], m: usize) -> Vec<f32> {
    assert_eq!(a.len(), n * k);
    assert_eq!(b.len(), k * m);
    let mut c = vec![0.0f32; n * m];
    if n == 0 || k == 0 || m == 0 {
        return c;
    }
    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            n as i32,
            m as i32,
            k as i32,
            1.0,
            a,
            k as i32,
            b,
            m as i32,
            0.0,
            &mut c,
            m as i32,
        );
    }
    c
}

/// Complex64 (`Complex<f32>`) sibling of `dense_matmul`, via Accelerate's
/// `cgemm`. `cblas::c32` is a re-export of the exact same `num_complex::
/// Complex<f32>` this crate already uses as `C64` (see `cblas`'s own
/// `extern crate num_complex as num;`), so no per-element conversion is
/// needed at the call boundary.
pub fn dense_matmul_c64(a: &[C64], n: usize, k: usize, b: &[C64], m: usize) -> Vec<C64> {
    assert_eq!(a.len(), n * k);
    assert_eq!(b.len(), k * m);
    let mut c = vec![C64::new(0.0, 0.0); n * m];
    if n == 0 || k == 0 || m == 0 {
        return c;
    }
    unsafe {
        cblas::cgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            n as i32,
            m as i32,
            k as i32,
            C64::new(1.0, 0.0),
            a,
            k as i32,
            b,
            m as i32,
            C64::new(0.0, 0.0),
            &mut c,
            m as i32,
        );
    }
    c
}

/// Complex128 (`Complex<f64>`) sibling of `dense_matmul`, via Accelerate's
/// `zgemm`.
pub fn dense_matmul_c128(a: &[C128], n: usize, k: usize, b: &[C128], m: usize) -> Vec<C128> {
    assert_eq!(a.len(), n * k);
    assert_eq!(b.len(), k * m);
    let mut c = vec![C128::new(0.0, 0.0); n * m];
    if n == 0 || k == 0 || m == 0 {
        return c;
    }
    unsafe {
        cblas::zgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            n as i32,
            m as i32,
            k as i32,
            C128::new(1.0, 0.0),
            a,
            k as i32,
            b,
            m as i32,
            C128::new(0.0, 0.0),
            &mut c,
            m as i32,
        );
    }
    c
}

pub fn matvec_gflops(n_out: usize, n_in: usize, seconds: f64) -> f64 {
    let flops = 2.0 * n_out as f64 * n_in as f64;
    flops / seconds / 1e9
}

pub fn matmul_gflops(n: usize, k: usize, m: usize, seconds: f64) -> f64 {
    let flops = 2.0 * n as f64 * k as f64 * m as f64;
    flops / seconds / 1e9
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_matches_naive() {
        let n = 37;
        let m = 41;
        let a: Vec<f64> = (0..n * m).map(|i| (i as f64 * 0.013).sin()).collect();
        let x: Vec<f64> = (0..m).map(|i| (i as f64 * 0.7).cos()).collect();
        let y = dense_matvec(&a, n, m, &x);
        for i in 0..n {
            let mut expect = 0.0;
            for j in 0..m {
                expect += a[i * m + j] * x[j];
            }
            assert!((y[i] - expect).abs() < 1e-9);
        }
    }

    #[test]
    fn matmul_matches_naive() {
        let n = 13;
        let k = 17;
        let m = 11;
        let a: Vec<f64> = (0..n * k).map(|i| (i as f64 * 0.021).sin()).collect();
        let b: Vec<f64> = (0..k * m).map(|i| (i as f64 * 0.031).cos()).collect();
        let c = dense_matmul(&a, n, k, &b, m);
        for i in 0..n {
            for j in 0..m {
                let mut expect = 0.0;
                for kk in 0..k {
                    expect += a[i * k + kk] * b[kk * m + j];
                }
                assert!((c[i * m + j] - expect).abs() < 1e-9);
            }
        }
    }
}
