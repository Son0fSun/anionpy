//! The correctness gate — adopted verbatim from the Python reference
//! implementation (`wave/operators.py` docstring / `bench/matmul_vs_numpy.py`):
//! *"A speed number attached to a wrong answer is a lie."*
//!
//! `certify()` is the **only** way to obtain a `Certified` handle, and the
//! bench binary's timing functions accept nothing but a `Certified` — there
//! is no code path that lets a timing number escape without first passing
//! this check to `< 1e-9` against a dense reference. That is what "wired in
//! structurally" means: it is not a convention the bench happens to follow,
//! it is a type the compiler enforces.

use crate::operator::Operator;
use num_complex::Complex64;
use std::fmt;

pub struct GateFailure {
    pub name: String,
    pub matvec_err: f64,
    pub matmat_err: f64,
    pub tol: f64,
}

impl fmt::Display for GateFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "correctness gate FAILED for '{}': matvec_err={:.3e} matmat_err={:.3e} (tol={:.1e}) \
             — refusing to report a timing number",
            self.name, self.matvec_err, self.matmat_err, self.tol
        )
    }
}

/// Proof that `op` matched a dense reference to `< tol` on both a matvec and
/// a matmat probe. The only way to construct one is `certify()` succeeding.
pub struct Certified<'a> {
    op: &'a dyn Operator,
    pub matvec_err: f64,
    pub matmat_err: f64,
}

impl<'a> Certified<'a> {
    pub fn op(&self) -> &dyn Operator {
        self.op
    }
}

/// Check `op.apply`/`op.apply_mat` against `dense_expected` on the supplied
/// probe vector/matrix. `tol` should be `1e-9` (the gate file's threshold)
/// for anything claiming to be production-representative.
pub fn certify<'a>(
    name: &str,
    op: &'a dyn Operator,
    dense_expected: &dyn Operator,
    x: &[Complex64],
    xmat: &[Complex64],
    ncols: usize,
    tol: f64,
) -> Result<Certified<'a>, GateFailure> {
    assert_eq!(op.shape(), dense_expected.shape(), "certify: shape mismatch between op and reference");

    let y_op = op.apply(x);
    let y_dense = dense_expected.apply(x);
    let matvec_err = crate::linalg::max_abs_diff(&y_op, &y_dense);

    let ym_op = op.apply_mat(xmat, ncols);
    let ym_dense = dense_expected.apply_mat(xmat, ncols);
    let matmat_err = crate::linalg::max_abs_diff(&ym_op, &ym_dense);

    if matvec_err < tol && matmat_err < tol {
        Ok(Certified { op, matvec_err, matmat_err })
    } else {
        Err(GateFailure { name: name.to_string(), matvec_err, matmat_err, tol })
    }
}
