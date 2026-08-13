//! `Product` — lazy composition `A @ B`: applies `B` then `A`, never forming
//! the dense product. Cost is the sum of the factor costs, so two structured
//! factors give a structured-cost matvec even when their dense product would
//! be full. Mirrors `ProductOperator` in the Python reference; this is what
//! every operator pair falls back to in `compose::matmul` when no tighter
//! closed-form collapse is provably correct (see that module for the list).

use crate::operator::Operator;
use num_complex::Complex64;
use std::any::Any;

pub struct Product {
    a: Box<dyn Operator>,
    b: Box<dyn Operator>,
}

impl Product {
    pub fn new(a: Box<dyn Operator>, b: Box<dyn Operator>) -> Self {
        assert_eq!(
            a.shape().1,
            b.shape().0,
            "Product: shape mismatch, {:?} . {:?}",
            a.shape(),
            b.shape()
        );
        Self { a, b }
    }
}

impl Operator for Product {
    fn shape(&self) -> (usize, usize) {
        (self.a.shape().0, self.b.shape().1)
    }

    fn is_real(&self) -> bool {
        self.a.is_real() && self.b.is_real()
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        self.a.apply(&self.b.apply(x))
    }

    fn apply_mat(&self, x: &[Complex64], ncols: usize) -> Vec<Complex64> {
        let mid = self.b.apply_mat(x, ncols);
        self.a.apply_mat(&mid, ncols)
    }

    fn to_dense(&self) -> Vec<Complex64> {
        // Verification-only: materialise both factors and multiply densely.
        let (n, _) = self.shape();
        let (k, m) = self.b.shape();
        let ad = self.a.to_dense(); // n x k, column-major
        let bd = self.b.to_dense(); // k x m, column-major
        let mut out = vec![Complex64::new(0.0, 0.0); n * m];
        for j in 0..m {
            for t in 0..k {
                let bval = bd[j * k + t];
                if bval.norm() == 0.0 {
                    continue;
                }
                for i in 0..n {
                    out[j * n + i] += ad[t * n + i] * bval;
                }
            }
        }
        out
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
