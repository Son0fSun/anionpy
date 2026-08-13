//! `Diagonal` — trivial O(n) baseline / composition unit. Storage is n
//! numbers, not n^2. Mirrors `DiagonalOperator` in the Python reference.

use crate::operator::Operator;
use num_complex::Complex64;
use std::any::Any;

pub struct Diagonal {
    d: Vec<Complex64>,
    real: bool,
}

impl Diagonal {
    pub fn new(d: Vec<Complex64>, real: bool) -> Self {
        Self { d, real }
    }

    pub fn from_real(d: Vec<f64>) -> Self {
        Self {
            d: d.into_iter().map(|v| Complex64::new(v, 0.0)).collect(),
            real: true,
        }
    }

    pub fn diag(&self) -> &[Complex64] {
        &self.d
    }

    pub fn adjoint(&self) -> Diagonal {
        Diagonal::new(self.d.iter().map(|v| v.conj()).collect(), self.real)
    }

    /// diag(d1) @ diag(d2) = diag(d1 .* d2). Elementwise product, O(n).
    /// Exact — tested against the dense product in `tests`.
    pub fn compose_diagonal(&self, other: &Diagonal) -> Diagonal {
        assert_eq!(self.d.len(), other.d.len(), "diagonal composition shape mismatch");
        let d: Vec<Complex64> = self.d.iter().zip(&other.d).map(|(a, b)| a * b).collect();
        Diagonal::new(d, self.real && other.real)
    }
}

impl Operator for Diagonal {
    fn shape(&self) -> (usize, usize) {
        (self.d.len(), self.d.len())
    }

    fn is_real(&self) -> bool {
        self.real
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(x.len(), self.d.len(), "Diagonal::apply: dimension mismatch");
        self.d.iter().zip(x).map(|(a, b)| a * b).collect()
    }

    fn apply_mat(&self, x: &[Complex64], ncols: usize) -> Vec<Complex64> {
        let n = self.d.len();
        let mut out = vec![Complex64::new(0.0, 0.0); n * ncols];
        for j in 0..ncols {
            for i in 0..n {
                out[j * n + i] = self.d[i] * x[j * n + i];
            }
        }
        out
    }

    fn to_dense(&self) -> Vec<Complex64> {
        let n = self.d.len();
        let mut m = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            m[i * n + i] = self.d[i];
        }
        m
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
