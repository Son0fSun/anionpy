//! Array-creation and shape-manipulation primitives shared by ionp-py's
//! top-level `zeros`/`ones`/`empty`/`full`/`arange`/`linspace`/`eye`/
//! `identity` and by the top-level view wrappers (`reshape`/`ravel`/
//! `transpose`/`squeeze`/`expand_dims`/`swapaxes`/`moveaxis`/
//! `broadcast_to`). Lives in `ionp-core` (no PyO3) so every number here is
//! plain Rust, matching the crate's "Rust does arithmetic, Python does
//! dispatch" rule -- `ionp-py/src/creation.rs` only parses Python arguments
//! and calls straight into this module.
//!
//! `NdArray`'s `buffer`/`shape`/`strides`/`offset` fields are `pub(crate)`
//! (see array.rs) -- visible everywhere inside this crate, including this
//! file, without needing new public methods on `NdArray` itself. That is
//! deliberately used below for the view-returning functions
//! (`squeeze`/`expand_dims`/`broadcast_to`/`swapaxes`/`moveaxis`), which
//! construct a new `NdArray` sharing the SAME `Arc<Buffer>` (a real view,
//! per GOAL-ionp.md's "views must be real views" requirement) rather than
//! copying data.

use std::sync::Arc;

use crate::array::NdArray;
use crate::buffer::Buffer;
use crate::dtype::DType;
use crate::error::IonpError;
use crate::shape::{self, Shape, Strides};

/// A bare zero-width flexible dtype (`DType::S(0)`/`DType::U(0)`, i.e.
/// numpy's own `'S0'`/`'S'`/`'bytes'`/`'U0'`/`'U'`/`'str'` spellings) is a
/// real, valid `np.dtype(...)` OBJECT with itemsize 0 -- but ACTUALLY
/// ALLOCATING an array at that dtype always bumps it to the minimum
/// storage width of 1: verified live against real numpy 2.5.1,
/// `np.dtype('S0').itemsize == 0` yet
/// `np.zeros(3, dtype='S0').itemsize == 1` (`np.zeros(3,
/// dtype=str).dtype == '<U1'`, `np.ones`/`np.empty` identical, even at
/// `n == 0` elements -- `np.zeros(0, dtype=str).itemsize == 4`, the width
/// bump is independent of element count). So this bump belongs at the
/// "materialize a buffer" boundary (here), never at `dtype_name_to_dtype`/
/// `dtype_from_pyobj` itself, which must keep reporting the genuine
/// zero-width dtype for plain introspection (`anionpy.dtype('S0').itemsize
/// == 0`, matching `np.dtype`).
pub fn bump_zero_width(dtype: DType) -> DType {
    match dtype {
        DType::S(0) => DType::S(1),
        DType::U(0) => DType::U(4),
        other => other,
    }
}

/// Build a `Buffer` of `n` zero-valued elements of `dtype`. Used by
/// `zeros`/`zeros_like`.
pub fn zeros_buffer(dtype: DType, n: usize) -> Buffer {
    match bump_zero_width(dtype) {
        DType::Bool => Buffer::Bool(vec![false; n]),
        // `S`/`U` "zero" is the all-NUL element (every byte/code-unit 0),
        // matching numpy's own `np.zeros(n, dtype='S5')` (5 NUL bytes
        // each) -- and, on read-back, an all-NUL element decodes to the
        // empty string, matching `np.zeros(3, dtype='S5')[0] == b''`.
        DType::S(w) => Buffer::S(w, vec![vec![0u8; w as usize]; n]),
        DType::U(w) => Buffer::U(w, vec![vec![0u32; (w / 4) as usize]; n]),
        DType::I8 => Buffer::I8(vec![0; n]),
        DType::I16 => Buffer::I16(vec![0; n]),
        DType::I32 => Buffer::I32(vec![0; n]),
        DType::I64 => Buffer::I64(vec![0; n]),
        DType::U8 => Buffer::U8(vec![0; n]),
        DType::U16 => Buffer::U16(vec![0; n]),
        DType::U32 => Buffer::U32(vec![0; n]),
        DType::U64 => Buffer::U64(vec![0; n]),
        DType::F16 => Buffer::F16(vec![half::f16::from_f32(0.0); n]),
        DType::F32 => Buffer::F32(vec![0.0; n]),
        DType::F64 => Buffer::F64(vec![0.0; n]),
        DType::C64 => Buffer::C64(vec![num_complex::Complex::new(0.0, 0.0); n]),
        DType::C128 => Buffer::C128(vec![num_complex::Complex::new(0.0, 0.0); n]),
    }
}

/// Build a `Buffer` of `n` one-valued elements of `dtype`. Used by
/// `ones`/`ones_like`.
pub fn ones_buffer(dtype: DType, n: usize) -> Buffer {
    match bump_zero_width(dtype) {
        DType::Bool => Buffer::Bool(vec![true; n]),
        // numpy's `np.ones(n, dtype='S5')` fills with the STRING "1"
        // (b'1', NUL-padded), not with byte value 1 -- verified live:
        // `np.ones(2, dtype='S5').tolist() == [b'1', b'1']`. Same for `U`.
        DType::S(w) => {
            let mut e = vec![0u8; w as usize];
            e[0] = b'1';
            Buffer::S(w, vec![e; n])
        }
        DType::U(w) => {
            let mut e = vec![0u32; (w / 4) as usize];
            e[0] = b'1' as u32;
            Buffer::U(w, vec![e; n])
        }
        DType::I8 => Buffer::I8(vec![1; n]),
        DType::I16 => Buffer::I16(vec![1; n]),
        DType::I32 => Buffer::I32(vec![1; n]),
        DType::I64 => Buffer::I64(vec![1; n]),
        DType::U8 => Buffer::U8(vec![1; n]),
        DType::U16 => Buffer::U16(vec![1; n]),
        DType::U32 => Buffer::U32(vec![1; n]),
        DType::U64 => Buffer::U64(vec![1; n]),
        DType::F16 => Buffer::F16(vec![half::f16::from_f32(1.0); n]),
        DType::F32 => Buffer::F32(vec![1.0; n]),
        DType::F64 => Buffer::F64(vec![1.0; n]),
        DType::C64 => Buffer::C64(vec![num_complex::Complex::new(1.0, 0.0); n]),
        DType::C128 => Buffer::C128(vec![num_complex::Complex::new(1.0, 0.0); n]),
    }
}

/// `empty`'s contents are documented-garbage in real numpy (uninitialized
/// heap memory) -- the differential test contract for this item is
/// explicitly SHAPE/DTYPE only, never contents (see the task brief).
/// Zero-filling in safe Rust satisfies that contract without reaching for
/// `unsafe`/`MaybeUninit`, and has the added benefit of never producing
/// nondeterministic/UB-adjacent output for an item that already can't be
/// compared by value.
pub fn empty_buffer(dtype: DType, n: usize) -> Buffer {
    zeros_buffer(dtype, n)
}

/// Repeat the single element held by a length-1 `Buffer` `n` times,
/// preserving its dtype. Used by `full`/`full_like`: the PyO3 boundary
/// marshals the Python fill value into a length-1 `Buffer` of the target
/// dtype (reusing `ionp-py`'s existing scalar-to-Buffer helpers), and this
/// function does the actual (Rust-side) repetition.
pub fn repeat_scalar_buffer(src: &Buffer, n: usize) -> Buffer {
    macro_rules! rep {
        ($variant:ident, $v:expr) => {
            Buffer::$variant(vec![$v[0]; n])
        };
    }
    match src {
        Buffer::S(w, v) => Buffer::S(*w, vec![v[0].clone(); n]),
        Buffer::U(w, v) => Buffer::U(*w, vec![v[0].clone(); n]),
        Buffer::Bool(v) => rep!(Bool, v),
        Buffer::I8(v) => rep!(I8, v),
        Buffer::I16(v) => rep!(I16, v),
        Buffer::I32(v) => rep!(I32, v),
        Buffer::I64(v) => rep!(I64, v),
        Buffer::U8(v) => rep!(U8, v),
        Buffer::U16(v) => rep!(U16, v),
        Buffer::U32(v) => rep!(U32, v),
        Buffer::U64(v) => rep!(U64, v),
        Buffer::F16(v) => rep!(F16, v),
        Buffer::F32(v) => rep!(F32, v),
        Buffer::F64(v) => rep!(F64, v),
        Buffer::C64(v) => rep!(C64, v),
        Buffer::C128(v) => rep!(C128, v),
    }
}

/// numpy's exact `arange` length formula for the float path (verified
/// against numpy 2.5.1 source, `_array_fromobject`'s `_calc_length`):
/// `length = ceil((stop - start) / step)`, clamped to 0 when negative (an
/// empty range, e.g. `arange(5, 0, 1)`). This one formula is used for
/// every dtype ionp supports -- numpy's own integer-`arange` path computes
/// the identical quantity, just via integer division that agrees with this
/// ceil-of-float-division formula for every case tested here (see
/// `creation_cases.py`'s `arange_cases` for the corpus that pins this,
/// including negative step and non-integer step).
pub fn arange_len(start: f64, stop: f64, step: f64) -> usize {
    if step == 0.0 {
        return 0; // caller raises ZeroDivisionError before this is reached
    }
    let raw = (stop - start) / step;
    let len = raw.ceil();
    if len <= 0.0 || !len.is_finite() {
        0
    } else {
        len as usize
    }
}

/// Build the `n`-element `arange(start, start+step, ..., dtype)` sequence
/// as an `f64` vector (the caller casts down to the target dtype via
/// `Buffer::cast_to`, matching numpy's own "compute in a wide type, cast
/// once" behavior for `arange`'s non-float dtypes). Each element is
/// `start + i*step`, computed exactly this way (not by repeated addition)
/// to avoid float error accumulation numpy itself does not have (numpy's
/// C loop is also `start + i*step`, not a running sum).
pub fn arange_values(start: f64, step: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| start + (i as f64) * step).collect()
}

/// `linspace(start, stop, num, endpoint)`: numpy's own formula (see
/// `numpy/core/function_base.py`) -- `step = delta / div` where
/// `div = num - 1` if `endpoint` else `num`; `y[i] = start + i*step`; when
/// `endpoint` and `num > 1`, the LAST element is overwritten with `stop`
/// exactly (not `start + (num-1)*step`, which can differ from `stop` by a
/// float ULP or two) -- this exact-last-value behavior is why the
/// overwrite is a separate step here, not folded into the `i*step` formula.
/// Returns `(values, step)`; `step` is `NaN` when `div == 0` (i.e.
/// `num <= 1` and endpoint, or `num == 0` and not endpoint), matching
/// numpy's own `retstep` output for that case.
pub fn linspace_values(start: f64, stop: f64, num: usize, endpoint: bool) -> (Vec<f64>, f64) {
    let delta = stop - start;
    let div = if endpoint {
        num.checked_sub(1).unwrap_or(0)
    } else {
        num
    };
    let (mut values, step) = if div > 0 {
        let step = delta / (div as f64);
        let values: Vec<f64> = (0..num).map(|i| (i as f64) * step + start).collect();
        (values, step)
    } else {
        // numpy: step = NaN; y = arange(num) * delta (i.e. every element is
        // `0 * delta + start` here since arange(num) with div<=0 only
        // happens for num in {0, 1}).
        let values: Vec<f64> = (0..num).map(|i| (i as f64) * delta + start).collect();
        (values, f64::NAN)
    };
    if endpoint && num > 1 {
        let last = values.len() - 1;
        values[last] = stop;
    }
    (values, step)
}

/// numpy's array-valued `np.linspace(start, stop, num, axis=...)`: `start`
/// and `stop` are broadcast to a common shape first (that broadcasting
/// itself happens in `ionp-py`, which owns `NdArray`/view machinery; this
/// function only takes the two already-broadcast flat `f64` slices, both
/// length `k = product(broadcast_shape)`). Returns a flat buffer of length
/// `num * k`, laid out as `[num, ...broadcast_shape]` in C order (i.e. the
/// new `num` axis is the OUTERMOST/slowest axis) -- `ionp-py` moves that
/// axis to the caller's requested `axis=` position afterwards by reusing
/// the existing `moveaxis` view primitive rather than this function
/// knowing anything about axis placement. Also returns the per-element
/// `step` vector (length `k`), matching `linspace_values`'s scalar
/// `retstep` output broadcast over every element.
pub fn linspace_values_nd(start: &[f64], stop: &[f64], num: usize, endpoint: bool) -> (Vec<f64>, Vec<f64>) {
    let k = start.len();
    debug_assert_eq!(stop.len(), k);
    let mut out = vec![0.0f64; num * k];
    let mut steps = vec![0.0f64; k];
    for i in 0..k {
        let (vals, step) = linspace_values(start[i], stop[i], num, endpoint);
        steps[i] = step;
        for j in 0..num {
            out[j * k + i] = vals[j];
        }
    }
    (out, steps)
}

/// `eye(n_rows, n_cols, k)`: 1.0 on the k-th diagonal (k=0 main, k>0 above,
/// k<0 below), 0.0 elsewhere, row-major. Returns an `f64` vector the
/// caller casts to the target dtype, mirroring `arange_values` above.
pub fn eye_values(n_rows: usize, n_cols: usize, k: isize) -> Vec<f64> {
    let mut out = vec![0.0f64; n_rows * n_cols];
    for row in 0..n_rows {
        let col = row as isize + k;
        if col >= 0 && (col as usize) < n_cols {
            out[row * n_cols + col as usize] = 1.0;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// View-returning shape manipulations. Each shares the source's Arc<Buffer>
// -- real views, not copies -- matching numpy exactly for the cases where
// numpy itself returns a view (squeeze/expand_dims/broadcast_to always do;
// numpy's own ravel/reshape fall back to a copy when a view isn't possible,
// which is why those two are NOT reimplemented here: `NdArray::reshape`
// (array.rs) already has that exact copy-or-view logic and is reused
// as-is by ionp-py/src/creation.rs's top-level wrappers instead of being
// duplicated).
// ---------------------------------------------------------------------------

/// `squeeze(a, axis=None)`. `axis=None` drops every size-1 dimension;
/// an explicit `axis` (list of ints, already normalized to non-negative by
/// the caller) drops only those dimensions, raising (numpy: `ValueError`)
/// if any named axis does not have size 1.
pub fn squeeze(a: &NdArray, axis: Option<&[usize]>) -> Result<NdArray, IonpError> {
    let shape = a.shape();
    let strides = a.strides();
    let keep: Vec<bool> = match axis {
        None => shape.iter().map(|&d| d != 1).collect(),
        Some(axes) => {
            for &ax in axes {
                if ax >= shape.len() {
                    return Err(IonpError::Index(format!(
                        "axis {ax} is out of bounds for array of dimension {}",
                        shape.len()
                    )));
                }
                if shape[ax] != 1 {
                    return Err(IonpError::Value(format!(
                        "cannot select an axis to squeeze out which has size not equal to one"
                    )));
                }
            }
            (0..shape.len()).map(|i| !axes.contains(&i)).collect()
        }
    };
    let new_shape: Shape = shape
        .iter()
        .zip(keep.iter())
        .filter(|(_, &k)| k)
        .map(|(&d, _)| d)
        .collect();
    let new_strides: Strides = strides
        .iter()
        .zip(keep.iter())
        .filter(|(_, &k)| k)
        .map(|(&s, _)| s)
        .collect();
    Ok(NdArray {
        buffer: Arc::clone(&a.buffer),
        shape: new_shape,
        strides: new_strides,
        offset: a.offset(),
    })
}

/// Insert size-1 axes using numpy's *indexing* stride convention: the new
/// axis always gets stride 0, and every EXISTING axis keeps its own
/// original stride untouched (verified against real numpy 2.5.1: `a[None]`,
/// `a[..., None]`, `np.atleast_2d`/`np.atleast_3d`'s `ary[newaxis, :]`-
/// style bodies in `_core/shape_base.py` all behave this way). This is
/// NOT what `np.expand_dims` does -- see `expand_dims` below, which is a
/// different numpy function (`a.reshape(new_shape)`) with a genuinely
/// different stride rule, and the two were conflated in this function
/// under the `expand_dims` name until Task #newaxis-split (2026-08-08)
/// found the conflation made 22/30 probed nonempty `expand_dims` cases
/// wrong while it was declared exact.
///
/// `axes` (already normalized to non-negative, against the RESULT ndim, by
/// the caller -- matching numpy's own normalization target) is the set of
/// positions where a new size-1 dimension is inserted.
///
/// No size-0 special case is needed here (unlike `expand_dims`): a fresh
/// empty base array's own strides are already all-zero (every constructor
/// in this crate keeps that invariant), so copying them through for the
/// kept axes and inserting a literal 0 for each new axis reproduces
/// numpy's empty-input newaxis strides exactly on its own, e.g.
/// `np.zeros((0,3))[None].strides == (0, 0, 0)`.
pub fn insert_newaxis(a: &NdArray, axes: &[usize]) -> Result<NdArray, IonpError> {
    let old_shape = a.shape();
    let old_strides = a.strides();
    let new_ndim = old_shape.len() + axes.len();
    for &ax in axes {
        if ax >= new_ndim {
            return Err(IonpError::Index(format!(
                "axis {ax} is out of bounds for array of dimension {new_ndim}"
            )));
        }
    }
    let mut new_shape = Vec::with_capacity(new_ndim);
    let mut new_strides = Vec::with_capacity(new_ndim);
    let mut old_i = 0;
    for i in 0..new_ndim {
        if axes.contains(&i) {
            new_shape.push(1);
            new_strides.push(0);
        } else {
            new_shape.push(old_shape[old_i]);
            new_strides.push(old_strides[old_i]);
            old_i += 1;
        }
    }
    Ok(NdArray {
        buffer: Arc::clone(&a.buffer),
        shape: new_shape,
        strides: new_strides,
        offset: a.offset(),
    })
}

/// `expand_dims(a, axis)`. `axis` (already normalized to non-negative,
/// against the RESULT ndim, by the caller -- matching numpy's own
/// normalization target) is a set of positions where a new size-1
/// dimension is inserted.
///
/// Real numpy's `expand_dims` is literally `a.reshape(new_shape)`
/// (`_core/shape_base.py`), NOT a stride-0-newaxis insertion (that was
/// this function's bug -- see `insert_newaxis` above, which is the
/// correct primitive for numpy's `ary[newaxis, :]`-style callers).
/// Routed through `NdArray::reshape` (`array.rs`) rather than
/// recomputing a stride formula here: that is the SAME function
/// `anionpy.reshape`/`ndarray.reshape` already use, so this needs no new
/// stride math and inherits reshape's existing C-contiguous-fast-path /
/// `attempt_nocopy_reshape` / copy-fallback behavior exactly, matching
/// numpy's own "expand_dims IS reshape" relationship byte for byte
/// (verified: `np.expand_dims(np.zeros((2,3)), 0).strides == (48, 24, 8)`,
/// same as `anionpy.reshape` on the padded shape). No empty-size special
/// case is needed either: `NdArray::reshape`'s own C-contiguous fast path
/// already produces numpy's `(1,1,1)`-elementwise strides for every
/// zero-sized `new_shape` it's asked to build, exactly like real numpy's
/// `expand_dims` on `(3,0)`/`(0,3)`/etc.
pub fn expand_dims(a: &NdArray, axes: &[usize]) -> Result<NdArray, IonpError> {
    let old_shape = a.shape();
    let new_ndim = old_shape.len() + axes.len();
    for &ax in axes {
        if ax >= new_ndim {
            return Err(IonpError::Index(format!(
                "axis {ax} is out of bounds for array of dimension {new_ndim}"
            )));
        }
    }
    let mut new_shape = Vec::with_capacity(new_ndim);
    let mut old_i = 0;
    for i in 0..new_ndim {
        if axes.contains(&i) {
            new_shape.push(1);
        } else {
            new_shape.push(old_shape[old_i]);
            old_i += 1;
        }
    }
    a.reshape(&new_shape)
}

/// `swapaxes(a, axis1, axis2)`: a view with two dimensions' shape/stride
/// entries swapped (already-normalized non-negative indices, caller's
/// responsibility -- same convention as `expand_dims` above).
pub fn swapaxes(a: &NdArray, axis1: usize, axis2: usize) -> Result<NdArray, IonpError> {
    let ndim = a.ndim();
    if axis1 >= ndim || axis2 >= ndim {
        return Err(IonpError::Index(format!(
            "axis is out of bounds for array of dimension {ndim}"
        )));
    }
    let mut new_shape = a.shape().to_vec();
    let mut new_strides = a.strides().to_vec();
    new_shape.swap(axis1, axis2);
    new_strides.swap(axis1, axis2);
    Ok(NdArray {
        buffer: Arc::clone(&a.buffer),
        shape: new_shape,
        strides: new_strides,
        offset: a.offset(),
    })
}

/// `moveaxis(a, source, destination)`: a view with a full axis permutation
/// built from moving each `source[i]` to `destination[i]`, matching numpy's
/// own semantics (build the permutation by removing sources from the
/// natural order, then inserting each at its destination). `source`/
/// `destination` are already-normalized non-negative axis lists of equal
/// length, caller's responsibility.
pub fn moveaxis(a: &NdArray, source: &[usize], destination: &[usize]) -> Result<NdArray, IonpError> {
    let ndim = a.ndim();
    // Defensive only -- `source`/`destination` are documented (see doc
    // comment above) as already-normalized by the caller, and in practice
    // both `ionp-py` callers (the `moveaxis` binding and `linspace`'s
    // internal axis placement) normalize and bounds-check with their own
    // "source: "/"destination: "-prefixed numpy-exact message BEFORE
    // calling this function (see ionp-py/src/creation.rs), so this branch
    // is not reachable from Python input today. Kept numpy-shaped anyway
    // so a direct/future ionp-core caller doesn't get a worse message.
    for &ax in source.iter() {
        if ax >= ndim {
            return Err(IonpError::Index(format!(
                "source: axis {ax} is out of bounds for array of dimension {ndim}"
            )));
        }
    }
    for &ax in destination.iter() {
        if ax >= ndim {
            return Err(IonpError::Index(format!(
                "destination: axis {ax} is out of bounds for array of dimension {ndim}"
            )));
        }
    }
    let mut order: Vec<usize> = (0..ndim).filter(|i| !source.contains(i)).collect();
    // Insert each source axis at its destination position, processing in
    // ascending destination order (matches numpy's `np.core.numeric.moveaxis`
    // reference implementation: `order.insert(dest, src)` for each pair
    // sorted by dest).
    let mut pairs: Vec<(usize, usize)> = destination.iter().copied().zip(source.iter().copied()).collect();
    pairs.sort_by_key(|&(d, _)| d);
    for (d, s) in pairs {
        order.insert(d, s);
    }
    let old_shape = a.shape();
    let old_strides = a.strides();
    let new_shape: Shape = order.iter().map(|&i| old_shape[i]).collect();
    let new_strides: Strides = order.iter().map(|&i| old_strides[i]).collect();
    Ok(NdArray {
        buffer: Arc::clone(&a.buffer),
        shape: new_shape,
        strides: new_strides,
        offset: a.offset(),
    })
}

/// `broadcast_to(a, shape)`: a READ-ONLY view (numpy: writing through a
/// broadcast view is a `ValueError` because a stride-0 axis would alias
/// multiple logical elements to one physical one -- ionp has no mutation
/// API yet at all, so every array is already read-only in that sense; this
/// function does not itself add an enforcement mechanism, it only produces
/// the correct shape/stride-0 view, matching `shape.broadcast_strides_to`'s
/// existing broadcasting-arithmetic logic exactly rather than
/// reimplementing it).
pub fn broadcast_to(a: &NdArray, target_shape: &[usize]) -> Result<NdArray, IonpError> {
    // `shape::broadcast_strides_to` is a pure stride-mechanics helper --
    // it does not itself validate that `a.shape()` can legally broadcast
    // to `target_shape` (that was this function's job and, until this
    // fix, it was skipped entirely: an incompatible target_shape silently
    // produced a view with wrong/aliased strides that could read out of
    // the buffer's bounds, which crashed the whole process with a Rust
    // panic the moment anything -- __array__, repr -- walked it, instead
    // of raising the ValueError numpy raises at the `broadcast_to` call
    // itself). `shape::broadcast_shapes` (already used elsewhere in the
    // crate for elementwise-op broadcasting) computes the mutual
    // broadcast result of two shapes and errors on incompatible ones; for
    // `broadcast_to` specifically, `a` is broadcastable to `target_shape`
    // iff that mutual result equals `target_shape` exactly (numpy also
    // requires target ndim >= a's ndim, which `broadcast_shapes` already
    // enforces since it can only ever grow, never shrink, the shorter
    // shape).
    match shape::broadcast_shapes(a.shape(), target_shape) {
        Ok(ref result) if result.as_slice() == target_shape => {}
        _ => {
            return Err(IonpError::Broadcast {
                shape_a: a.shape().to_vec(),
                shape_b: target_shape.to_vec(),
            })
        }
    }
    // `broadcast_shapes` above already proved `a.shape()` broadcasts to
    // `target_shape` exactly, so this cannot fail -- `expect` documents
    // that invariant rather than threading a second, unreachable error
    // path through this function's `Result`.
    let new_strides = shape::broadcast_strides_to(a.shape(), a.strides(), target_shape)
        .expect("already validated broadcastable via broadcast_shapes above");
    Ok(NdArray {
        buffer: Arc::clone(&a.buffer),
        shape: target_shape.to_vec(),
        strides: new_strides,
        offset: a.offset(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Order;

    #[test]
    fn arange_len_matches_numpy_examples() {
        assert_eq!(arange_len(0.0, 5.0, 1.0), 5);
        assert_eq!(arange_len(0.0, 5.0, 2.0), 3);
        assert_eq!(arange_len(5.0, 0.0, -1.0), 5);
        assert_eq!(arange_len(0.0, 0.0, 1.0), 0);
        assert_eq!(arange_len(5.0, 0.0, 1.0), 0);
        assert_eq!(arange_len(0.0, 1.0, 0.3), 4); // ceil(3.333..) = 4
    }

    #[test]
    fn linspace_endpoint_hits_stop_exactly() {
        let (vals, step) = linspace_values(0.0, 1.0, 5, true);
        assert_eq!(vals, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(step, 0.25);
    }

    #[test]
    fn linspace_no_endpoint_excludes_stop() {
        let (vals, step) = linspace_values(0.0, 1.0, 5, false);
        assert_eq!(vals, vec![0.0, 0.2, 0.4, 0.6000000000000001, 0.8]);
        assert_eq!(step, 0.2);
    }

    #[test]
    fn linspace_num1_is_just_start() {
        let (vals, step) = linspace_values(3.0, 7.0, 1, true);
        assert_eq!(vals, vec![3.0]);
        assert!(step.is_nan());
    }

    #[test]
    fn eye_offdiagonal() {
        let v = eye_values(2, 3, 1);
        assert_eq!(v, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn squeeze_drops_size1_dims_and_shares_buffer() {
        let a = NdArray::from_buffer(Buffer::F64(vec![1.0, 2.0, 3.0]), vec![1, 3, 1], Order::C).unwrap();
        let s = squeeze(&a, None).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert!(Arc::ptr_eq(&a.buffer, &s.buffer));
    }

    #[test]
    fn expand_dims_inserts_size1() {
        let a = NdArray::from_buffer(Buffer::F64(vec![1.0, 2.0, 3.0]), vec![3], Order::C).unwrap();
        let e = expand_dims(&a, &[0]).unwrap();
        assert_eq!(e.shape(), &[1, 3]);
        let e2 = expand_dims(&a, &[1]).unwrap();
        assert_eq!(e2.shape(), &[3, 1]);
    }

    #[test]
    fn broadcast_to_is_view_with_zero_stride() {
        let a = NdArray::from_buffer(Buffer::F64(vec![1.0, 2.0, 3.0]), vec![1, 3], Order::C).unwrap();
        let b = broadcast_to(&a, &[4, 3]).unwrap();
        assert_eq!(b.shape(), &[4, 3]);
        assert!(Arc::ptr_eq(&a.buffer, &b.buffer));
    }

    #[test]
    fn moveaxis_matches_numpy_example() {
        // np.moveaxis(np.zeros((3,4,5)), 0, -1).shape == (4,5,3)
        let a = NdArray::from_buffer(Buffer::F64(vec![0.0; 60]), vec![3, 4, 5], Order::C).unwrap();
        let m = moveaxis(&a, &[0], &[2]).unwrap();
        assert_eq!(m.shape(), &[4, 5, 3]);
    }
}
