//! `Circulant` — diagonal in the DFT basis. The headline structural win.
//!
//! ```text
//! C[i, j] = c[(i - j) mod n]
//! C x   = IFFT(FFT(c) .* FFT(x))          O(n log n)   [dense O(n^2)]
//! C1 C2 = circulant(IFFT(FFT(c1).*FFT(c2))) O(n log n) [dense O(n^3)]
//! ```
//!
//! Mirrors `CirculantOperator` in the Python reference. The FFT plans are
//! computed once at construction and reused for every `apply`/`apply_mat`
//! call and for composition — this is what makes the *product* case (the
//! 441-444x showcase) O(n log n) instead of O(n^3): the dense product is
//! never formed, spectra are just multiplied elementwise.

use crate::operator::Operator;
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::any::Any;
use std::sync::Arc;

pub struct Circulant {
    n: usize,
    /// eigenvalues lambda = FFT(first_column)
    eig: Vec<Complex64>,
    first_col: Vec<Complex64>,
    real: bool,
    fwd: Arc<dyn Fft<f64>>,
    inv: Arc<dyn Fft<f64>>,
}

impl Circulant {
    pub fn new(c: Vec<Complex64>, real: bool) -> Self {
        let n = c.len();
        let mut planner = FftPlanner::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let mut eig = c.clone();
        fwd.process(&mut eig);
        Self { n, eig, first_col: c, real, fwd, inv }
    }

    pub fn from_real(c: Vec<f64>) -> Self {
        Self::new(c.into_iter().map(|v| Complex64::new(v, 0.0)).collect(), true)
    }

    pub fn eigenvalues(&self) -> &[Complex64] {
        &self.eig
    }

    pub fn first_column(&self) -> &[Complex64] {
        &self.first_col
    }

    fn ifft_normalized(&self, buf: &mut [Complex64]) {
        self.inv.process(buf);
        let n = buf.len() as f64;
        for v in buf.iter_mut() {
            *v /= n;
        }
    }

    pub fn adjoint(&self) -> Circulant {
        // C^H is circulant with conjugated spectrum; its first column is the
        // conjugate-reversed original: col'[0] = conj(c[0]), col'[k] = conj(c[n-k]).
        let n = self.n;
        let mut col = vec![Complex64::new(0.0, 0.0); n];
        col[0] = self.first_col[0].conj();
        for k in 1..n {
            col[k] = self.first_col[n - k].conj();
        }
        Circulant::new(col, self.real)
    }

    /// C1 @ C2 (same shape) is circulant with spectrum lambda1 .* lambda2 —
    /// O(n log n), never forming the O(n^3) dense product. This is the
    /// showcase collapse the acceptance gate is built around.
    pub fn compose_circulant(&self, other: &Circulant) -> Circulant {
        assert_eq!(self.n, other.n, "circulant composition shape mismatch");
        let mut prod_eig: Vec<Complex64> = self.eig.iter().zip(&other.eig).map(|(a, b)| a * b).collect();
        // one ifft to recover the first column of the product circulant
        let mut planner = FftPlanner::new();
        let inv = planner.plan_fft_inverse(self.n);
        inv.process(&mut prod_eig);
        let n = self.n as f64;
        for v in prod_eig.iter_mut() {
            *v /= n;
        }
        Circulant::new(prod_eig, self.real && other.real)
    }
}

impl Operator for Circulant {
    fn shape(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn is_real(&self) -> bool {
        self.real
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(x.len(), self.n, "Circulant::apply: dimension mismatch");
        let mut buf = x.to_vec();
        self.fwd.process(&mut buf);
        for (b, e) in buf.iter_mut().zip(&self.eig) {
            *b *= e;
        }
        self.ifft_normalized(&mut buf);
        buf
    }

    fn apply_mat(&self, x: &[Complex64], ncols: usize) -> Vec<Complex64> {
        // Batched: the same forward/inverse FFT plan (built once at
        // construction) is reused per column, so this is k * O(n log n) with
        // the planning cost paid exactly once, matching the reference's
        // `np.fft.fft(X, axis=0)` batching intent.
        let n = self.n;
        let mut out = vec![Complex64::new(0.0, 0.0); n * ncols];
        for j in 0..ncols {
            let col = &x[j * n..(j + 1) * n];
            let y = self.apply(col);
            out[j * n..(j + 1) * n].copy_from_slice(&y);
        }
        out
    }

    fn to_dense(&self) -> Vec<Complex64> {
        let n = self.n;
        let mut m = vec![Complex64::new(0.0, 0.0); n * n];
        for j in 0..n {
            for i in 0..n {
                let idx = (i as isize - j as isize).rem_euclid(n as isize) as usize;
                m[j * n + i] = self.first_col[idx];
            }
        }
        m
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
