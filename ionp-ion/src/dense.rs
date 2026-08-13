//! `Dense` — the fallback operator for when detection finds no exploitable
//! structure (`detect()` returns `None`). O(n*m) matvec, O(n*m*k) matmat: no
//! asymptotic win claimed, this is what `smart()` hands back so the caller
//! is never slower than dense, never faster either. Also doubles as the
//! ground-truth operand every correctness gate check compares a structured
//! operator's `apply`/`apply_mat` against.

use crate::operator::Operator;
use num_complex::Complex64;
use std::any::Any;

pub struct Dense {
    rows: usize,
    cols: usize,
    /// column-major: data[j * rows + i] == A[i, j]
    data: Vec<Complex64>,
    real: bool,
}

impl Dense {
    pub fn new(rows: usize, cols: usize, data: Vec<Complex64>, real: bool) -> Self {
        assert_eq!(data.len(), rows * cols, "Dense::new: data length mismatch");
        Self { rows, cols, data, real }
    }

    /// Build from a real-valued closure `f(i, j) -> A[i,j]`.
    pub fn from_real_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64) -> Self {
        let mut data = vec![Complex64::new(0.0, 0.0); rows * cols];
        for j in 0..cols {
            for i in 0..rows {
                data[j * rows + i] = Complex64::new(f(i, j), 0.0);
            }
        }
        Self { rows, cols, data, real: true }
    }

    pub fn from_real_col_major(rows: usize, cols: usize, data: &[f64]) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self {
            rows,
            cols,
            data: data.iter().map(|&v| Complex64::new(v, 0.0)).collect(),
            real: true,
        }
    }

    pub fn get(&self, i: usize, j: usize) -> Complex64 {
        self.data[j * self.rows + i]
    }

    pub fn data(&self) -> &[Complex64] {
        &self.data
    }
}

impl Operator for Dense {
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn is_real(&self) -> bool {
        self.real
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(x.len(), self.cols, "Dense::apply: dimension mismatch");
        let mut y = vec![Complex64::new(0.0, 0.0); self.rows];
        for j in 0..self.cols {
            let xj = x[j];
            if xj.norm() == 0.0 {
                continue;
            }
            let col = &self.data[j * self.rows..(j + 1) * self.rows];
            for i in 0..self.rows {
                y[i] += col[i] * xj;
            }
        }
        y
    }

    fn apply_mat(&self, x: &[Complex64], ncols: usize) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); self.rows * ncols];
        for c in 0..ncols {
            let xin = &x[c * self.cols..(c + 1) * self.cols];
            let y = self.apply(xin);
            out[c * self.rows..(c + 1) * self.rows].copy_from_slice(&y);
        }
        out
    }

    fn to_dense(&self) -> Vec<Complex64> {
        self.data.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
