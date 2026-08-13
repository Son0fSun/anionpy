//! `Toeplitz` — constant-diagonal operator via circulant embedding.
//!
//! ```text
//! T[i,j] = col[i-j]  for i>=j,  row[j-i]  for j>i   (row[0] ignored)
//! ```
//!
//! Not circulant, but embeds into a circulant of size m >= 2n-1 whose action
//! on a zero-padded vector reproduces T x in its first n entries — so matvec
//! is O(n log n) via FFT, dense is O(n^2). Mirrors `ToeplitzOperator` in the
//! Python reference.
//!
//! NOTE on composition: `Toeplitz @ Toeplitz` is **not implemented as a
//! collapse**. A general two-sided Toeplitz matrix is not closed under
//! multiplication (verified numerically: T1@T2 for random two-sided
//! Toeplitz T1,T2 is not itself Toeplitz — only the one-sided/causal case is,
//! via polynomial-multiplication, which this operator's general col+row
//! representation does not restrict to). Toeplitz@Toeplitz therefore falls
//! through `compose::matmul` to a lazy `Product`, which still gets O(n log n)
//! per-factor apply cost — just not a single tighter closed-form type. See
//! `compose.rs` for the pairs that do collapse and why.

use crate::operator::Operator;
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::any::Any;
use std::sync::Arc;

pub struct Toeplitz {
    n: usize,
    m: usize,
    circ_eig: Vec<Complex64>,
    col: Vec<Complex64>,
    row: Vec<Complex64>,
    real: bool,
    fwd: Arc<dyn Fft<f64>>,
    inv: Arc<dyn Fft<f64>>,
}

impl Toeplitz {
    pub fn new(col: Vec<Complex64>, row: Vec<Complex64>, real: bool) -> Self {
        let n = col.len();
        assert_eq!(row.len(), n, "Toeplitz: col and row must have equal length");
        let mut m = 1usize;
        while m < 2 * n - 1 {
            m <<= 1;
        }
        let mut planner = FftPlanner::new();
        let fwd = planner.plan_fft_forward(m);
        let inv = planner.plan_fft_inverse(m);

        let mut emb = vec![Complex64::new(0.0, 0.0); m];
        emb[..n].copy_from_slice(&col);
        // emb[m-(n-1)+i] = row[(n-1)-i] for i in 0..n-1  (row[1:][::-1])
        if n > 1 {
            for i in 0..(n - 1) {
                emb[m - (n - 1) + i] = row[(n - 1) - i];
            }
        }
        let mut circ_eig = emb.clone();
        fwd.process(&mut circ_eig);

        Self { n, m, circ_eig, col, row, real, fwd, inv }
    }

    pub fn from_real(col: Vec<f64>, row: Option<Vec<f64>>) -> Self {
        let row = row.unwrap_or_else(|| col.clone());
        Self::new(
            col.into_iter().map(|v| Complex64::new(v, 0.0)).collect(),
            row.into_iter().map(|v| Complex64::new(v, 0.0)).collect(),
            true,
        )
    }

    pub fn symmetric_from_real(col: Vec<f64>) -> Self {
        Self::from_real(col, None)
    }

    fn apply_padded(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut buf = vec![Complex64::new(0.0, 0.0); self.m];
        buf[..self.n].copy_from_slice(x);
        self.fwd.process(&mut buf);
        for (b, e) in buf.iter_mut().zip(&self.circ_eig) {
            *b *= e;
        }
        self.inv.process(&mut buf);
        let m = self.m as f64;
        for v in buf.iter_mut() {
            *v /= m;
        }
        buf.truncate(self.n);
        buf
    }

    pub fn adjoint(&self) -> Toeplitz {
        // (T^H)[i,j] = conj(T[j,i]); T^H is Toeplitz with col <-> conj(row).
        let col: Vec<Complex64> = self.row.iter().map(|v| v.conj()).collect();
        let row: Vec<Complex64> = self.col.iter().map(|v| v.conj()).collect();
        Toeplitz::new(col, row, self.real)
    }
}

impl Operator for Toeplitz {
    fn shape(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn is_real(&self) -> bool {
        self.real
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(x.len(), self.n, "Toeplitz::apply: dimension mismatch");
        self.apply_padded(x)
    }

    fn apply_mat(&self, x: &[Complex64], ncols: usize) -> Vec<Complex64> {
        let n = self.n;
        let mut out = vec![Complex64::new(0.0, 0.0); n * ncols];
        for j in 0..ncols {
            let col = &x[j * n..(j + 1) * n];
            let y = self.apply_padded(col);
            out[j * n..(j + 1) * n].copy_from_slice(&y);
        }
        out
    }

    fn to_dense(&self) -> Vec<Complex64> {
        let n = self.n;
        let mut mtx = vec![Complex64::new(0.0, 0.0); n * n];
        for j in 0..n {
            for i in 0..n {
                mtx[j * n + i] = if i >= j { self.col[i - j] } else { self.row[j - i] };
            }
        }
        mtx
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
