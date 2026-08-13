//! `matmul` — the composition dispatcher. `A @ B` returns an `Operator`,
//! never materialising the dense product.
//!
//! ## Which pairs collapse to a tighter closed form, and why
//!
//! | pair                        | collapses to | proof |
//! |------------------------------|---------------|-------|
//! | `Circulant @ Circulant`      | `Circulant`   | eigenvalues multiply elementwise (DFT diagonalises both); `tests::circulant_product_matches_dense` |
//! | `LowRank @ LowRank`          | `LowRank`     | `(U1 V1^H)(U2 V2^H) = U1(V1^H U2)V2^H`, algebraic identity; `tests::lowrank_product_matches_dense` |
//! | `Diagonal @ Diagonal`        | `Diagonal`    | elementwise product of the two diagonals; `tests::diagonal_product_matches_dense` |
//!
//! ## Pairs that were checked and do **not** collapse (stay lazy `Product`)
//!
//! - **`Toeplitz @ Toeplitz`** — a *general* two-sided Toeplitz matrix is not
//!   closed under multiplication. Verified numerically (n=5, random col+row):
//!   the product of two general Toeplitz matrices is not itself Toeplitz.
//!   (Only the causal/one-sided special case — lower-triangular Toeplitz,
//!   equivalent to polynomial multiplication — is closed, and this crate's
//!   `Toeplitz` type does not track that restriction, so it is not claimed.)
//! - **`Diagonal @ Circulant`** / **`Circulant @ Diagonal`** — row/column
//!   scaling a circulant breaks both the circulant wrap-around relation and
//!   the constant-diagonal relation; the product is neither diagonal,
//!   circulant, nor Toeplitz in general (verified numerically, n=5).
//!
//! Both of the above still get the real win from lazy `Product`: applying
//! the composed operator costs `cost(A) + cost(B)`, e.g. O(n log n) for a
//! `Diagonal @ Circulant` product — the dense O(n^2)/O(n^3) form is still
//! never built. What is missing is only a *single, tighter* representation
//! (fewer stored numbers, one shared FFT plan) — a real but strictly smaller
//! gap than "no structure exploited at all".
//!
//! Everything else (`Dense` on either side, `Product` on either side, mixed
//! pairs not listed above) falls through to lazy `Product`.

use crate::circulant::Circulant;
use crate::diagonal::Diagonal;
use crate::lowrank::LowRank;
use crate::operator::Operator;
use crate::product::Product;

/// `A @ B`. Consumes both operands (ownership makes chaining natural:
/// `matmul(matmul(a, b), c)`), inspects them for a provable closed-form
/// collapse via `downcast_ref`, and falls back to a lazy `Product`
/// otherwise. Never materialises a dense product.
pub fn matmul(a: Box<dyn Operator>, b: Box<dyn Operator>) -> Box<dyn Operator> {
    assert_eq!(
        a.shape().1,
        b.shape().0,
        "matmul: shape mismatch, {:?} . {:?}",
        a.shape(),
        b.shape()
    );

    if let (Some(ca), Some(cb)) = (a.as_any().downcast_ref::<Circulant>(), b.as_any().downcast_ref::<Circulant>()) {
        if ca.shape() == cb.shape() {
            return Box::new(ca.compose_circulant(cb));
        }
    }

    if let (Some(la), Some(lb)) = (a.as_any().downcast_ref::<LowRank>(), b.as_any().downcast_ref::<LowRank>()) {
        return Box::new(la.compose_lowrank(lb));
    }

    if let (Some(da), Some(db)) = (a.as_any().downcast_ref::<Diagonal>(), b.as_any().downcast_ref::<Diagonal>()) {
        return Box::new(da.compose_diagonal(db));
    }

    Box::new(Product::new(a, b))
}
