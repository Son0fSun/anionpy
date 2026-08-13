//! Contraction family: `dot`/`vdot`/`inner`/`tensordot`/`cross`.
//!
//! Deliberately composed ENTIRELY out of already-verified primitives --
//! `ufunc::binary_op(Multiply, ..)` (broadcasting elementwise product) and
//! `ufunc::reduce_axis(Add, ..)` (the same summation machinery `np.sum`
//! itself routes through, already differential-tested elsewhere in this
//! crate) -- rather than a hand-rolled accumulation loop. Two reasons:
//!
//!  1. `matmul.rs`/`linalg.rs` (owned by another task) may already be
//!     solving the hard "match numpy's BLAS accumulation order bit-for-bit"
//!     problem; duplicating that from scratch here risks silently
//!     producing a `dot` that LOOKS right but is not bit-exact for float
//!     dtypes on any shape where summation order matters, which is worse
//!     than not shipping it. Composing with `reduce_axis` reuses whatever
//!     algorithm `np.sum`/`np.add.reduce` already use, so this file wins
//!     or loses precision fidelity together with those already-shipped
//!     functions -- no new unverified arithmetic is introduced.
//!  2. Integer/bool contractions are EXACT under any summation order
//!     (integer addition is associative, no rounding), so this approach is
//!     always correct there regardless of the float-precision caveat.
//!
//! Known, reported (not silently ignored) risk: numpy's actual `dot`/
//! `tensordot` for float32/float64 on 2-D+ inputs typically dispatches to
//! BLAS (`cblas_dgemm`/`sdot`/`ddot`), which may use a different
//! accumulation order than this crate's `reduce_axis` (which mirrors
//! `np.sum`'s pairwise-summation algorithm). For large contraction
//! dimensions with cancellation-prone data this MAY produce a non-bit-exact
//! result vs real numpy even though the math is "correct" -- differential
//! testing (`.tobytes()` comparison) is the actual arbiter here, not this
//! comment's expectation. See the differential suite for the measured
//! pass/fail rate on this cluster.

use crate::array::NdArray;
use crate::dtype::DType;
use crate::error::IonpError;
use crate::ufunc::{self, BinaryOp};

/// numpy's `dot`/`inner`/`tensordot` special-case boolean inputs: rather
/// than promoting to int64 and summing (which is what plain `np.sum`/
/// `np.add.reduce` would do, and what this file's other dtype paths ride
/// on), a boolean contraction stays boolean and accumulates via logical OR
/// -- i.e. `dot(a, b)` on bool arrays is `any(a & b)` per output element,
/// not `sum(a & b)`. Confirmed empirically against real numpy (`np.dot` on
/// two bool 2x2 arrays returns dtype bool, value equal to the OR-reduction,
/// not the int64 popcount `Add` would give) and used here rather than
/// invented: `BinaryOp::Maximum` reduced over bool values IS logical OR
/// (`max(True, False) == True`), and is already an existing, independently
/// verified reduction path (shared with `np.max`/`np.any`-adjacent code),
/// so this is still "compose from verified primitives," not new arithmetic.
fn reduce_op_for(dtype: DType) -> BinaryOp {
    if dtype == DType::Bool {
        BinaryOp::Maximum
    } else {
        BinaryOp::Add
    }
}

/// `np.dot(a, b)` semantics:
///  - both 0-d: ordinary scalar multiply.
///  - either 0-d: elementwise multiply (broadcasts).
///  - both 1-d: inner product (sum of elementwise product), returns 0-d.
///  - both 2-d: matrix product.
///  - `a` N-d, `b` 1-d: sum product over the last axis of `a`.
///  - `a` N-d, `b` M-d (M>=2): sum product over the last axis of `a` and
///    the second-to-last axis of `b`.
pub fn dot(a: &NdArray, b: &NdArray) -> Result<NdArray, IonpError> {
    if a.ndim() == 0 || b.ndim() == 0 {
        return ufunc::binary_op(BinaryOp::Multiply, a, b);
    }
    if b.ndim() == 1 {
        // Contract a's last axis against b (b broadcasts against it).
        return contract_last_axis(a, b);
    }
    // a: (..., K), b: (..., K, N) -- contract a's last axis with b's
    // second-to-last axis. Implemented by moving b's contraction axis to
    // the front, then reshaping the problem into the "contract trailing
    // axis of a against leading axis of b" shape via broadcasting: for
    // every combination of a's leading indices and b's non-contracted
    // indices, out[..a_idx.., ..b_idx..] = sum_k a[..a_idx.., k] * b[..k,
    // ..b_idx..]. We realize this generically (any rank) by:
    //   a2 = a reshaped to (Ra, K)             where Ra = prod(a.shape[:-1])
    //   b2 = b with contraction axis moved to front, reshaped to (K, Rb)
    //   out2[i, j] = sum_k a2[i, k] * b2[k, j]   (plain 2-D matmul)
    //   out = out2 reshaped to a.shape[:-1] + b.shape without the
    //         contraction axis
    let k = *a.shape().last().unwrap();
    let b_contract_axis = b.ndim() - 2;
    if b.shape().get(b_contract_axis).copied() != Some(k) {
        return Err(IonpError::Value(format!(
            "shapes {:?} and {:?} not aligned: {} (dim {}) != {} (dim {})",
            a.shape(),
            b.shape(),
            k,
            a.ndim() - 1,
            b.shape().get(b_contract_axis).copied().unwrap_or(0),
            b_contract_axis
        )));
    }
    let a_lead: Vec<usize> = a.shape()[..a.ndim() - 1].to_vec();
    let mut b_axes: Vec<isize> = (0..b.ndim() as isize).collect();
    let moved = b_axes.remove(b_contract_axis);
    b_axes.insert(0, moved);
    let b_moved = b.transpose_axes(&b_axes)?; // (K, ...b without contraction axis...)
    let b_rest: Vec<usize> = b_moved.shape()[1..].to_vec();

    let ra: usize = a_lead.iter().product();
    let rb: usize = b_rest.iter().product();
    let a2 = a.reshape(&[ra, k])?;
    let b2 = b_moved.reshape(&[k, rb])?;
    let out2 = matmul_2d(&a2, &b2)?;
    let mut out_shape = a_lead;
    out_shape.extend_from_slice(&b_rest);
    out2.reshape(&out_shape)
}

/// Plain 2-D matrix product via broadcasting compose: (M,K) x (K,N) ->
/// (M,N), computed as `sum_axis1( a[:, :, newaxis] * b[newaxis, :, :] )`.
fn matmul_2d(a: &NdArray, b: &NdArray) -> Result<NdArray, IonpError> {
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    let a3 = a.reshape(&[m, k, 1])?;
    let b3 = b.reshape(&[1, k, n])?;
    let prod = ufunc::binary_op(BinaryOp::Multiply, &a3, &b3)?; // (m, k, n)
    let op = reduce_op_for(prod.dtype());
    ufunc::reduce_axis(op, &prod, &[1], false, None, None, None)
}

/// Contract `a`'s last axis against 1-D `b` (numpy's "N-d dot 1-d" and
/// "1-d dot 1-d" case, also the base of `inner`'s no-batch-alignment
/// variant).
fn contract_last_axis(a: &NdArray, b: &NdArray) -> Result<NdArray, IonpError> {
    let k = *a.shape().last().unwrap();
    if b.shape() != [k] {
        return Err(IonpError::Value(format!(
            "shapes {:?} and {:?} not aligned: {} (dim {}) != {} (dim 0)",
            a.shape(),
            b.shape(),
            k,
            a.ndim() - 1,
            b.shape().first().copied().unwrap_or(0)
        )));
    }
    let prod = ufunc::binary_op(BinaryOp::Multiply, a, b)?; // broadcasts b over a's last axis
    let last = a.ndim() - 1;
    let op = reduce_op_for(prod.dtype());
    ufunc::reduce_axis(op, &prod, &[last], false, None, None, None)
}

/// `np.vdot(a, b)`: flattens both to 1-D first, and conjugates `a` when
/// complex (numpy: `vdot(a, b) == dot(a.conj().ravel(), b.ravel())` for
/// complex inputs; identical to `dot` on flattened reals).
pub fn vdot(a: &NdArray, b: &NdArray) -> Result<NdArray, IonpError> {
    let af = a.ravel_order("C")?;
    let bf = b.ravel_order("C")?;
    let af = if af.dtype().is_complex() {
        ufunc::conj_array(&af)?
    } else {
        af
    };
    if af.size() != bf.size() {
        return Err(IonpError::Value(format!(
            "vdot: arrays have inconsistent sizes {} and {}",
            af.size(),
            bf.size()
        )));
    }
    contract_last_axis_flat(&af, &bf)
}

fn contract_last_axis_flat(a: &NdArray, b: &NdArray) -> Result<NdArray, IonpError> {
    let prod = ufunc::binary_op(BinaryOp::Multiply, a, b)?;
    let op = reduce_op_for(prod.dtype());
    ufunc::reduce_axis(op, &prod, &[0], false, None, None, None)
}

/// `np.inner(a, b)`: sum product over the LAST axis of both operands
/// (unlike `dot`, no axis alignment beyond "both last axes", and the
/// non-contracted axes are simply concatenated: a.shape[:-1] +
/// b.shape[:-1]).
pub fn inner(a: &NdArray, b: &NdArray) -> Result<NdArray, IonpError> {
    if a.ndim() == 0 || b.ndim() == 0 {
        return ufunc::binary_op(BinaryOp::Multiply, a, b);
    }
    let k = *a.shape().last().unwrap();
    if *b.shape().last().unwrap() != k {
        return Err(IonpError::Value(format!(
            "shapes {:?} and {:?} not aligned: {} (dim {}) != {} (dim {})",
            a.shape(),
            b.shape(),
            k,
            a.ndim() - 1,
            b.shape().last().copied().unwrap_or(0),
            b.ndim() - 1
        )));
    }
    let a_lead: Vec<usize> = a.shape()[..a.ndim() - 1].to_vec();
    let b_lead: Vec<usize> = b.shape()[..b.ndim() - 1].to_vec();
    let ra: usize = a_lead.iter().product();
    let rb: usize = b_lead.iter().product();
    let a2 = a.reshape(&[ra, k])?;
    let b2 = b.reshape(&[rb, k])?;
    // out2[i, j] = sum_k a2[i, k] * b2[j, k]  ==  matmul_2d(a2, b2^T)
    let b2t = b2.transpose_axes(&[1, 0])?;
    let out2 = matmul_2d(&a2, &b2t)?;
    let mut out_shape = a_lead;
    out_shape.extend_from_slice(&b_lead);
    out2.reshape(&out_shape)
}

/// `np.tensordot(a, b, axes)` with explicit paired axis lists (already
/// normalized to non-negative by the caller).
pub fn tensordot(a: &NdArray, b: &NdArray, axes_a: &[usize], axes_b: &[usize]) -> Result<NdArray, IonpError> {
    if axes_a.len() != axes_b.len() {
        return Err(IonpError::Value(
            "shape mismatch for sum: axes_a and axes_b must have the same length".to_string(),
        ));
    }
    for (&ka, &kb) in axes_a.iter().zip(axes_b.iter()) {
        let da = a.shape().get(ka).copied().unwrap_or(0);
        let db = b.shape().get(kb).copied().unwrap_or(0);
        if da != db {
            return Err(IonpError::Value(format!(
                "shape-mismatch for sum: {} != {}",
                da, db
            )));
        }
    }
    // Move contracted axes of `a` to the end, contracted axes of `b` to
    // the front (in the given pairing order), reshape both to 2-D, matmul,
    // reshape back.
    let a_free: Vec<usize> = (0..a.ndim()).filter(|i| !axes_a.contains(i)).collect();
    let b_free: Vec<usize> = (0..b.ndim()).filter(|i| !axes_b.contains(i)).collect();

    let mut a_perm: Vec<isize> = a_free.iter().map(|&x| x as isize).collect();
    a_perm.extend(axes_a.iter().map(|&x| x as isize));
    let mut b_perm: Vec<isize> = axes_b.iter().map(|&x| x as isize).collect();
    b_perm.extend(b_free.iter().map(|&x| x as isize));

    let a_t = a.transpose_axes(&a_perm)?;
    let b_t = b.transpose_axes(&b_perm)?;

    let a_free_shape: Vec<usize> = a_free.iter().map(|&i| a.shape()[i]).collect();
    let b_free_shape: Vec<usize> = b_free.iter().map(|&i| b.shape()[i]).collect();
    let k: usize = axes_a.iter().map(|&i| a.shape()[i]).product();

    let ra: usize = a_free_shape.iter().product();
    let rb: usize = b_free_shape.iter().product();
    let a2 = a_t.reshape(&[ra, k])?;
    let b2 = b_t.reshape(&[k, rb])?;
    let out2 = matmul_2d(&a2, &b2)?;
    let mut out_shape = a_free_shape;
    out_shape.extend_from_slice(&b_free_shape);
    out2.reshape(&out_shape)
}

/// `np.cross(a, b)` for the 3-component case (the overwhelmingly common
/// one; 2-component `cross` returns the scalar z-component only and is
/// handled by the same formula restricted to that one output component).
/// Broadcasts a and b's leading dimensions (all axes except `axis`, which
/// must have length 2 or 3) the same way numpy does, then computes the
/// determinant-formula cross product componentwise via existing
/// elementwise multiply/subtract ufuncs.
pub fn cross(a: &NdArray, b: &NdArray, axis_a: usize, axis_b: usize, axis_c: usize) -> Result<NdArray, IonpError> {
    let na = a.shape()[axis_a];
    let nb = b.shape()[axis_b];
    if !(2..=3).contains(&na) || !(2..=3).contains(&nb) {
        return Err(IonpError::Value(
            "incompatible dimensions for cross product (dimension must be 2 or 3)".to_string(),
        ));
    }
    let take = |arr: &NdArray, axis: usize, idx: usize| -> Result<NdArray, IonpError> {
        let slices: Vec<crate::array::SliceItem> = (0..arr.ndim())
            .map(|d| {
                if d == axis {
                    crate::array::SliceItem::Index(idx as isize)
                } else {
                    crate::array::SliceItem::Slice { start: None, stop: None, step: None }
                }
            })
            .collect();
        arr.get_view(&slices)
    };
    let a0 = take(a, axis_a, 0)?;
    let a1 = take(a, axis_a, 1)?;
    let a2 = if na == 3 { Some(take(a, axis_a, 2)?) } else { None };
    let b0 = take(b, axis_b, 0)?;
    let b1 = take(b, axis_b, 1)?;
    let b2 = if nb == 3 { Some(take(b, axis_b, 2)?) } else { None };

    let mul = |x: &NdArray, y: &NdArray| ufunc::binary_op(BinaryOp::Multiply, x, y);
    let sub = |x: &NdArray, y: &NdArray| ufunc::binary_op(BinaryOp::Subtract, x, y);
    let neg = |x: &NdArray| ufunc::unary_op(crate::ufunc::UnaryOp::Negative, x);

    let cp0; // x component
    let cp1; // y component
    let cp2; // z component
    match (&a2, &b2) {
        (Some(a2), Some(b2)) => {
            cp0 = sub(&mul(&a1, b2)?, &mul(a2, &b1)?)?;
            cp1 = sub(&mul(a2, &b0)?, &mul(&a0, b2)?)?;
            cp2 = sub(&mul(&a0, &b1)?, &mul(&a1, &b0)?)?;
        }
        (Some(a2), None) => {
            // a is 3-component, b is 2-component (b2 treated as 0).
            cp0 = neg(&mul(a2, &b1)?)?;
            cp1 = mul(a2, &b0)?;
            cp2 = sub(&mul(&a0, &b1)?, &mul(&a1, &b0)?)?;
        }
        (None, Some(b2)) => {
            // a is 2-component (a2 treated as 0), b is 3-component.
            cp0 = mul(&a1, b2)?;
            cp1 = neg(&mul(&a0, b2)?)?;
            cp2 = sub(&mul(&a0, &b1)?, &mul(&a1, &b0)?)?;
        }
        (None, None) => {
            // Both 2-component: numpy returns only the scalar z component.
            return sub(&mul(&a0, &b1)?, &mul(&a1, &b0)?);
        }
    }
    // Stack the three components back along axis_c.
    let parts = [&cp0, &cp1, &cp2];
    let expanded: Vec<NdArray> = parts
        .iter()
        .map(|p| crate::creation::expand_dims(p, &[axis_c.min(p.ndim())]))
        .collect::<Result<_, _>>()?;
    let refs: Vec<&NdArray> = expanded.iter().collect();
    crate::manip::concatenate(&refs, Some(axis_c as isize))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;

    fn arr_f64(data: &[f64], shape: &[usize]) -> NdArray {
        NdArray::from_buffer(Buffer::F64(data.to_vec()), shape.to_vec(), crate::array::Order::C).unwrap()
    }

    #[test]
    fn dot_1d_1d_is_sum_of_products() {
        let a = arr_f64(&[1.0, 2.0, 3.0], &[3]);
        let b = arr_f64(&[4.0, 5.0, 6.0], &[3]);
        let out = dot(&a, &b).unwrap();
        assert_eq!(out.ndim(), 0);
        match out.ravel_order("C").unwrap().buffer() {
            Buffer::F64(v) => assert_eq!(v[0], 32.0),
            _ => panic!("wrong buffer"),
        }
    }

    #[test]
    fn dot_2d_2d_matches_hand_computed() {
        let a = arr_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = arr_f64(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let out = dot(&a, &b).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        let flat = out.ravel_order("C").unwrap();
        match flat.buffer() {
            Buffer::F64(v) => assert_eq!(v, &[19.0, 22.0, 43.0, 50.0]),
            _ => panic!("wrong buffer"),
        }
    }

    #[test]
    fn inner_matches_hand_computed() {
        let a = arr_f64(&[1.0, 2.0, 3.0], &[3]);
        let b = arr_f64(&[0.0, 1.0, 0.0], &[3]);
        let out = inner(&a, &b).unwrap();
        match out.ravel_order("C").unwrap().buffer() {
            Buffer::F64(v) => assert_eq!(v[0], 2.0),
            _ => panic!("wrong buffer"),
        }
    }

    #[test]
    fn tensordot_full_contraction_matches_dot_scalar() {
        let a = arr_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = arr_f64(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let out = tensordot(&a, &b, &[0, 1], &[0, 1]).unwrap();
        assert_eq!(out.ndim(), 0);
        match out.ravel_order("C").unwrap().buffer() {
            Buffer::F64(v) => assert_eq!(v[0], 5.0),
            _ => panic!("wrong buffer"),
        }
    }
}
