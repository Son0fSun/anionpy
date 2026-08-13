//! `LowRank` — A = U V^H, rank r << min(n,m).
//!
//! ```text
//! A x = U (V^H x)     O((n+m) r)   [dense O(n m)]
//! A1 @ A2 (both low-rank) stays low-rank, O(r1 r2 max(n,m)) middle
//!     contraction instead of forming either dense factor.
//! ```
//!
//! Mirrors `LowRankOperator` in the Python reference. This is the operator
//! class attention maps and similar inference structures are built from.

use crate::operator::Operator;
use num_complex::Complex64;
use std::any::Any;

pub struct LowRank {
    n: usize,
    m: usize,
    r: usize,
    /// column-major n x r
    u: Vec<Complex64>,
    /// column-major m x r  (V, not V^H — V^H is computed on the fly as
    /// conj-transpose so adjoint() is a free U/V swap)
    v: Vec<Complex64>,
    real: bool,
}

impl LowRank {
    pub fn new(u: Vec<Complex64>, v: Vec<Complex64>, r: usize, n: usize, m: usize, real: bool) -> Self {
        assert_eq!(u.len(), n * r, "LowRank: U shape mismatch");
        assert_eq!(v.len(), m * r, "LowRank: V shape mismatch");
        Self { n, m, r, u, v, real }
    }

    pub fn from_real(u: Vec<f64>, v: Vec<f64>, n: usize, m: usize, r: usize) -> Self {
        Self {
            n,
            m,
            r,
            u: u.into_iter().map(|x| Complex64::new(x, 0.0)).collect(),
            v: v.into_iter().map(|x| Complex64::new(x, 0.0)).collect(),
            real: true,
        }
    }

    pub fn rank(&self) -> usize {
        self.r
    }

    pub fn u(&self) -> &[Complex64] {
        &self.u
    }
    pub fn v(&self) -> &[Complex64] {
        &self.v
    }

    /// V^H x : (r x m) @ (m) -> r.  O(m r).
    fn vh_apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut w = vec![Complex64::new(0.0, 0.0); self.r];
        for t in 0..self.r {
            let mut acc = Complex64::new(0.0, 0.0);
            for i in 0..self.m {
                acc += self.v[t * self.m + i].conj() * x[i];
            }
            w[t] = acc;
        }
        w
    }

    /// U w : (n x r) @ (r) -> n.  O(n r).
    fn u_apply(&self, w: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.n];
        for t in 0..self.r {
            let wt = w[t];
            if wt.norm() == 0.0 {
                continue;
            }
            for i in 0..self.n {
                y[i] += self.u[t * self.n + i] * wt;
            }
        }
        y
    }

    pub fn adjoint(&self) -> LowRank {
        // A^H = V U^H : swap U and V.
        LowRank::new(self.v.clone(), self.u.clone(), self.r, self.m, self.n, self.real)
    }

    /// (U1 V1^H)(U2 V2^H) = U1 (V1^H U2) V2^H = U' V2^H, U' = U1 (V1^H U2).
    /// Middle contraction is (r1 x n_mid) @ (n_mid x r2) = r1 x r2, then
    /// U1(n x r1) @ middle(r1 x r2) -> (n x r2). Total O(n_mid*r1*r2 + n*r1*r2),
    /// never forming either dense factor. O((n+m)r) style win, generalised.
    pub fn compose_lowrank(&self, other: &LowRank) -> LowRank {
        assert_eq!(self.m, other.n, "low-rank composition shape mismatch");
        let n_mid = self.m; // == other.n
        let r1 = self.r;
        let r2 = other.r;

        // middle[t1, t2] = sum_i conj(V1[i,t1]) * U2[i,t2]   (r1 x r2)
        let mut middle = vec![Complex64::new(0.0, 0.0); r1 * r2];
        for t2 in 0..r2 {
            for i in 0..n_mid {
                let u2v = other.u[t2 * n_mid + i];
                if u2v.norm() == 0.0 {
                    continue;
                }
                for t1 in 0..r1 {
                    middle[t2 * r1 + t1] += self.v[t1 * n_mid + i].conj() * u2v;
                }
            }
        }

        // u_new[:, t2] = U1 @ middle[:, t2]   (n x r2)
        let mut u_new = vec![Complex64::new(0.0, 0.0); self.n * r2];
        for t2 in 0..r2 {
            for t1 in 0..r1 {
                let mv = middle[t2 * r1 + t1];
                if mv.norm() == 0.0 {
                    continue;
                }
                for i in 0..self.n {
                    u_new[t2 * self.n + i] += self.u[t1 * self.n + i] * mv;
                }
            }
        }

        LowRank::new(u_new, other.v.clone(), r2, self.n, other.m, self.real && other.real)
    }
}

impl Operator for LowRank {
    fn shape(&self) -> (usize, usize) {
        (self.n, self.m)
    }

    fn is_real(&self) -> bool {
        self.real
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        assert_eq!(x.len(), self.m, "LowRank::apply: dimension mismatch");
        self.u_apply(&self.vh_apply(x))
    }

    fn apply_mat(&self, x: &[Complex64], ncols: usize) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); self.n * ncols];
        for c in 0..ncols {
            let xin = &x[c * self.m..(c + 1) * self.m];
            let y = self.apply(xin);
            out[c * self.n..(c + 1) * self.n].copy_from_slice(&y);
        }
        out
    }

    fn to_dense(&self) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); self.n * self.m];
        for j in 0..self.m {
            for t in 0..self.r {
                let vjt = self.v[t * self.m + j].conj();
                if vjt.norm() == 0.0 {
                    continue;
                }
                for i in 0..self.n {
                    out[j * self.n + i] += self.u[t * self.n + i] * vjt;
                }
            }
        }
        out
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
