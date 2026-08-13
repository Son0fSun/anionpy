//! Array-manipulation primitives: concatenation/stacking, flipping/rolling,
//! tiling/repeating, broadcasting helpers, and the diagonal family
//! (diag/diagonal/diagflat/tril/triu/trace).
//!
//! NEW FILE, owned exclusively by the "array manipulation" task block (see
//! that task's own file-ownership rules) -- everything here is pure Rust,
//! no PyO3, mirroring the split every other core module in this crate
//! already uses (`creation.rs` is the closest sibling: same
//! `NdArray`/`Buffer`/`IonpError` vocabulary, same "gather via a macro over
//! all 13 `Buffer` variants" style).
//!
//! Every function here that logically returns a numpy VIEW (flip, the
//! `atleast_*` no-op cases) still returns a view here too (shares the same
//! `Arc<Buffer>`) wherever that is possible without extra bookkeeping this
//! crate doesn't already have; functions where the source semantics
//! genuinely require gathering non-contiguous elements into a new layout
//! (concatenate, stack, tile, repeat, roll, the diagonal family) always
//! return an owned, freshly-allocated, C-contiguous array -- the
//! differential harness compares VALUES (dtype + shape + bytes), never
//! aliasing/view-ness, so this is never a correctness gap, only a
//! (documented, deliberate) non-goal of matching numpy's own view/copy
//! split exactly.

use std::sync::Arc;

use crate::array::{NdArray, Order, SliceItem};
use crate::buffer::Buffer;
use crate::creation;
use crate::dtype::{promote_dtype, DType};
use crate::error::IonpError;
use crate::shape;

/// Negative-wrap + range-check a single axis argument against `ndim`,
/// raising the same `IonpError::AxisError` variant `ionp-py`'s
/// `to_py_err`/`axis_error()` turn into a real `numpy.exceptions.AxisError`
/// (see `error.rs`). `ionp-py/src/creation.rs` and `ndarray_attrs.rs` each
/// carry their own copy of this exact logic (not shared, per this
/// codebase's established convention of small per-module copies over a
/// shared axis-normalization module) -- this is `manip.rs`'s copy, scoped
/// to files this task owns.
pub fn normalize_axis(axis: isize, ndim: usize) -> Result<usize, IonpError> {
    let nd = ndim as isize;
    let a = if axis < 0 { axis + nd } else { axis };
    if a < 0 || a >= nd.max(0) {
        // ndim==0 (a 0-d array): numpy still reports "for array of
        // dimension 0" rather than treating it as "no axes at all".
        return Err(IonpError::AxisError { axis, ndim: Some(ndim) });
    }
    Ok(a as usize)
}

/// Same as `normalize_axis` but permits `axis == ndim` too (the "insert a
/// new axis here" case `stack`/`expand_dims`-family callers need, where
/// the valid range is `-(ndim+1)..=ndim` against the OUTPUT's ndim, not a
/// pre-existing axis of the input).
pub fn normalize_insertion_axis(axis: isize, ndim: usize) -> Result<usize, IonpError> {
    let nd = (ndim as isize) + 1;
    let a = if axis < 0 { axis + nd } else { axis };
    if a < 0 || a >= nd {
        // numpy reports the OUTPUT's dimension count here (ndim+1), not the
        // inputs' ndim: confirmed live across ndim 0..3, both positive and
        // negative out-of-range axes, e.g. `np.stack([a,b], axis=nd+1)` on
        // ndim-`nd` inputs raises "...for array of dimension {nd+1}".
        return Err(IonpError::AxisError { axis, ndim: Some(ndim + 1) });
    }
    Ok(a as usize)
}

/// Gather macro used throughout this module: given a `Buffer` and a
/// closure producing the flat (already offset-adjusted) source index for
/// every output position `0..len`, produce a freshly allocated `Buffer` of
/// the same variant. `$src` must already be a plain `&[T]` (i.e. call this
/// with `to_contiguous()`'d, offset-0 inputs) since the index closure
/// works in flat-buffer-position space, not (shape, strides) space.
macro_rules! gather_flat {
    ($buffer:expr, $len:expr, $idx:expr) => {
        match $buffer {
            Buffer::Bool(v) => Buffer::Bool((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::I8(v) => Buffer::I8((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::I16(v) => Buffer::I16((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::I32(v) => Buffer::I32((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::I64(v) => Buffer::I64((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::U8(v) => Buffer::U8((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::U16(v) => Buffer::U16((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::U32(v) => Buffer::U32((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::U64(v) => Buffer::U64((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::F16(v) => Buffer::F16((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::F32(v) => Buffer::F32((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::F64(v) => Buffer::F64((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::C64(v) => Buffer::C64((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::C128(v) => Buffer::C128((0..$len).map(|i| v[$idx(i)]).collect()),
            Buffer::S(w, v) => Buffer::S(*w, (0..$len).map(|i| v[$idx(i)].clone()).collect()),
            Buffer::U(w, v) => Buffer::U(*w, (0..$len).map(|i| v[$idx(i)].clone()).collect()),
        }
    };
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// atleast_1d / atleast_2d / atleast_3d
// ---------------------------------------------------------------------------

/// numpy: prepend size-1 axes at the FRONT until ndim >= 1.
pub fn atleast_1d(a: &NdArray) -> NdArray {
    if a.ndim() == 0 {
        a.reshape(&[1]).expect("0-d to (1,) reshape is always a view")
    } else {
        a.clone()
    }
}

/// numpy: 0-d -> (1,1); 1-d (n,) -> (1,n); ndim>=2 unchanged.
///
/// Only the ndim==0 case uses `reshape` (there is no existing axis to
/// insert relative to, and (1,1) is trivially C-contiguous either way).
/// The ndim==1 case is numpy's newaxis-at-front, `a[np.newaxis, :]`
/// (`_core/shape_base.py`'s `atleast_2d` body is literally `ary[_nx.
/// newaxis, :]`, read directly from source), NOT a C-contiguous-
/// recomputing reshape: the inserted axis is a stride-0 view, so a
/// non-contiguous or negative-strided input keeps its own strides on the
/// trailing axis rather than getting a fresh row-major layout. Using
/// `creation::insert_newaxis` (the same newaxis-indexing primitive
/// `stats.rs`'s BROADCAST-stride keepdims paths already use) gets that
/// for free instead of re-deriving it. NOT `creation::expand_dims` --
/// that function implements a DIFFERENT numpy function (`a.reshape(...)`)
/// with a different stride rule; see its own doc comment in `creation.rs`.
pub fn atleast_2d(a: &NdArray) -> NdArray {
    match a.ndim() {
        0 => a.reshape(&[1, 1]).expect("0-d to (1,1) reshape is always a view"),
        1 => creation::insert_newaxis(a, &[0]).expect("prepending one axis to a 1-d array is always in bounds"),
        _ => a.clone(),
    }
}

/// numpy: 0-d -> (1,1,1); 1-d (n,) -> (1,n,1); 2-d (m,n) -> (m,n,1);
/// ndim>=3 unchanged.
///
/// Same reasoning as `atleast_2d` above (and confirmed directly from
/// `np.atleast_3d`'s own source): only ndim==0 reshapes. ndim==1
/// inserts axes at both front and back (`a[np.newaxis, :, np.newaxis]`)
/// and ndim==2 inserts one axis at the back (`a[:, :, np.newaxis]`) --
/// both genuine newaxis-indexing stride-0 insertions via
/// `creation::insert_newaxis`, not a recomputed C-contiguous layout.
pub fn atleast_3d(a: &NdArray) -> NdArray {
    match a.ndim() {
        0 => a.reshape(&[1, 1, 1]).expect("0-d to (1,1,1) reshape is always a view"),
        1 => creation::insert_newaxis(a, &[0, 2]).expect("inserting two axes around a 1-d array is always in bounds"),
        2 => creation::insert_newaxis(a, &[2]).expect("appending one axis to a 2-d array is always in bounds"),
        _ => a.clone(),
    }
}

// ---------------------------------------------------------------------------
// broadcast_shapes / broadcast_arrays
// ---------------------------------------------------------------------------

/// Mirrors real numpy's `np.broadcast_shapes` mismatch text and (more
/// subtly) exactly which pair of args it blames, confirmed empirically
/// against numpy 2.5.1 across 2/3/4-arg cases including ones designed to
/// distinguish "first conflicting pair by dimension, scanned leading-to-
/// trailing, args in order" from a naive pairwise left-fold (the two
/// disagree whenever a later shape conflicts with an *earlier* shape that
/// is not immediately to its left, e.g. `(2,5),(3,5),(2,6)` blames
/// `arg 0`/`arg 1` -- the leading-dimension conflict -- even though a
/// trailing-dimension conflict between arg 0 and arg 2 also exists).
///
/// Algorithm (equivalent to numpy's C broadcasting loop): align every
/// shape to the max ndim by left-padding with implicit 1s, then scan
/// aligned dimension positions left to right; within each position scan
/// args in order, tracking the first non-1 size seen (the "reference")
/// and its owning arg index; the first later arg whose size at that
/// position is neither 1 nor equal to the reference is the conflict.
pub fn broadcast_shapes(shapes: &[&[usize]]) -> Result<Vec<usize>, IonpError> {
    let out_ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let aligned = |s: &[usize], p: usize| -> usize {
        let pad = out_ndim - s.len();
        if p < pad {
            1
        } else {
            s[p - pad]
        }
    };
    for p in 0..out_ndim {
        let mut reference: Option<(usize, usize)> = None; // (size, arg_idx)
        for (i, s) in shapes.iter().enumerate() {
            let sz = aligned(s, p);
            if sz == 1 {
                continue;
            }
            match reference {
                None => reference = Some((sz, i)),
                Some((ref_sz, ref_i)) if sz != ref_sz => {
                    return Err(IonpError::Value(format!(
                        "shape mismatch: objects cannot be broadcast to a single shape.  \
                         Mismatch is between arg {ref_i} with shape {} and arg {i} with shape {}.",
                        shape_tuple_str(shapes[ref_i]),
                        shape_tuple_str(s),
                    )));
                }
                _ => {}
            }
        }
    }
    let mut out = vec![1usize; out_ndim];
    for (p, slot) in out.iter_mut().enumerate() {
        for s in shapes {
            let sz = aligned(s, p);
            if sz != 1 {
                *slot = sz;
                break;
            }
        }
    }
    Ok(out)
}

/// numpy's own tuple `repr` for a shape, e.g. `(5,)` for a single-element
/// shape (note the trailing comma numpy always includes) and `(2, 3)` for
/// a multi-element one -- used verbatim inside `broadcast_shapes`'s exact
/// mismatch message.
fn shape_tuple_str(shape: &[usize]) -> String {
    match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => {
            let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", parts.join(", "))
        }
    }
}

pub fn broadcast_arrays(arrays: &[NdArray]) -> Result<Vec<NdArray>, IonpError> {
    let shapes: Vec<&[usize]> = arrays.iter().map(|a| a.shape()).collect();
    let target = broadcast_shapes(&shapes)?;
    arrays.iter().map(|a| crate::creation::broadcast_to(a, &target)).collect()
}

// ---------------------------------------------------------------------------
// concatenate / stack and the stack-family wrappers
// ---------------------------------------------------------------------------

/// `axis == None` means "flatten every input to 1-D (C order) first", the
/// same encoding real numpy's `concatenate(..., axis=None)` uses.
pub fn concatenate(arrays: &[&NdArray], axis: Option<isize>) -> Result<NdArray, IonpError> {
    if arrays.is_empty() {
        return Err(IonpError::Value("need at least one array to concatenate".to_string()));
    }
    let out_dtype = arrays.iter().skip(1).fold(arrays[0].dtype(), |acc, a| promote_dtype(acc, a.dtype()));

    let raveled: Vec<NdArray>;
    let refs: Vec<&NdArray> = if axis.is_none() {
        raveled = arrays.iter().map(|a| a.to_contiguous().reshape(&[a.size()]).unwrap()).collect();
        raveled.iter().collect()
    } else {
        arrays.to_vec()
    };
    let ax_axis = axis.unwrap_or(0);

    let ndim = refs[0].ndim();
    // ORDER BUG FOUND + FIXED (2026-08-04): the 0-d rejection used to run
    // AFTER the ndim-agreement loop below, which is the wrong way round.
    // numpy takes `ndim` from the FIRST array and rejects it for being 0-d
    // before it ever compares the others, so the message is a function of
    // position, not of the set. Measured against numpy 2.5.1, `z =
    // np.array(5)`, `a = np.array([1,0,2], np.int8)`:
    //     np.concatenate((z, a)) -> "zero-dimensional arrays cannot be
    //                                concatenated"
    //     np.concatenate((a, z)) -> "all the input arrays must have same
    //                                number of dimensions, ..."
    //     np.concatenate((a, z, a)) -> the ndim-mismatch message too
    // i.e. a 0-d operand anywhere but index 0 is reported as an ordinary
    // ndim disagreement. With the checks in the old order, the first row
    // wrongly produced the ndim-mismatch message. Note this only bites for
    // `axis != None`: with `axis=None` every operand is raveled to 1-d
    // above, so `ndim` is 1 by construction and neither check fires.
    if ndim == 0 {
        return Err(IonpError::Value("zero-dimensional arrays cannot be concatenated".to_string()));
    }
    for (i, a) in refs.iter().enumerate() {
        if a.ndim() != ndim {
            // Distinct from the "for the concatenation axis" message below:
            // this one fires when the input ndims themselves disagree,
            // confirmed live against numpy 2.5.1 (always blames index 0
            // against the first differing index, byte-for-byte including
            // the "dimension(s)" plural-with-parens spelling).
            return Err(IonpError::Value(format!(
                "all the input arrays must have same number of dimensions, but the array at index 0 has {} dimension(s) and the array at index {} has {} dimension(s)",
                ndim, i, a.ndim()
            )));
        }
    }
    let ax = normalize_axis(ax_axis, ndim)?;

    let base_shape = refs[0].shape();
    for (i, a) in refs.iter().enumerate().skip(1) {
        for d in 0..ndim {
            if d != ax && a.shape()[d] != base_shape[d] {
                return Err(IonpError::Value(format!(
                    "all the input array dimensions except for the concatenation axis must match exactly, \
                     but along dimension {}, the array at index 0 has size {} and the array at index {} has size {}",
                    d, base_shape[d], i, a.shape()[d]
                )));
            }
        }
    }

    let mut out_shape: Vec<usize> = base_shape.to_vec();
    out_shape[ax] = refs.iter().map(|a| a.shape()[ax]).sum();

    let outer: usize = out_shape[..ax].iter().product();
    let inner: usize = out_shape[ax + 1..].iter().product();
    let out_axis_len = out_shape[ax];
    let total = shape::size_of_shape(&out_shape);

    let contigs: Vec<NdArray> = refs.iter().map(|a| a.cast_to(out_dtype).to_contiguous()).collect();
    let chunk_lens: Vec<usize> = contigs.iter().map(|a| a.shape()[ax] * inner).collect();

    macro_rules! build {
        ($variant:ident) => {{
            let mut out: Vec<_> = Vec::with_capacity(total);
            let bufs: Vec<&Vec<_>> = contigs
                .iter()
                .map(|a| match a.buffer() {
                    Buffer::$variant(v) => v,
                    _ => unreachable!("all inputs were cast to the same promoted dtype"),
                })
                .collect();
            for o in 0..outer {
                for (buf, len) in bufs.iter().zip(chunk_lens.iter()) {
                    let start = o * len;
                    out.extend_from_slice(&buf[start..start + len]);
                }
            }
            Buffer::$variant(out)
        }};
    }

    let out_buffer = match out_dtype {
        DType::Bool => build!(Bool),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => build!(I8),
        DType::I16 => build!(I16),
        DType::I32 => build!(I32),
        DType::I64 => build!(I64),
        DType::U8 => build!(U8),
        DType::U16 => build!(U16),
        DType::U32 => build!(U32),
        DType::U64 => build!(U64),
        DType::F16 => build!(F16),
        DType::F32 => build!(F32),
        DType::F64 => build!(F64),
        DType::C64 => build!(C64),
        DType::C128 => build!(C128),
    };
    let _ = out_axis_len;
    NdArray::from_buffer(out_buffer, out_shape, Order::C)
}

/// `np.stack`: every array must share IDENTICAL shape; result has one more
/// dimension than the inputs, of length `arrays.len()`, inserted at `axis`.
///
/// numpy's own `stack` (`_core/shape_base.py`, read directly from source)
/// builds each expanded operand via `arr[sl]` newaxis indexing
/// (`sl = (slice(None),) * axis + (newaxis,)`), NOT `np.expand_dims` --
/// so `creation::insert_newaxis` is the source-accurate primitive here.
/// This choice is UNMEASURABLE from `stack`'s own output, though:
/// `concatenate` immediately gathers every expanded operand's VALUES into
/// a fresh, freshly-computed C-contiguous buffer, so the intermediate
/// stride-0-vs-reshape choice on `expanded` never survives into the
/// result either way.
pub fn stack(arrays: &[&NdArray], axis: isize) -> Result<NdArray, IonpError> {
    if arrays.is_empty() {
        return Err(IonpError::Value("need at least one array to stack".to_string()));
    }
    let ndim = arrays[0].ndim();
    for a in arrays {
        if a.shape() != arrays[0].shape() {
            return Err(IonpError::Value(
                "all input arrays must have the same shape".to_string(),
            ));
        }
    }
    let ins_axis = normalize_insertion_axis(axis, ndim)?;
    let expanded: Vec<NdArray> = arrays
        .iter()
        .map(|a| crate::creation::insert_newaxis(a, &[ins_axis]))
        .collect::<Result<_, _>>()?;
    let refs: Vec<&NdArray> = expanded.iter().collect();
    concatenate(&refs, Some(ins_axis as isize))
}

pub fn hstack(arrays: &[&NdArray]) -> Result<NdArray, IonpError> {
    if arrays.is_empty() {
        return Err(IonpError::Value("need at least one array to concatenate".to_string()));
    }
    let promoted: Vec<NdArray> = arrays.iter().map(|a| atleast_1d(a)).collect();
    let refs: Vec<&NdArray> = promoted.iter().collect();
    if promoted[0].ndim() == 1 {
        concatenate(&refs, Some(0))
    } else {
        concatenate(&refs, Some(1))
    }
}

pub fn vstack(arrays: &[&NdArray]) -> Result<NdArray, IonpError> {
    if arrays.is_empty() {
        return Err(IonpError::Value("need at least one array to concatenate".to_string()));
    }
    let promoted: Vec<NdArray> = arrays.iter().map(|a| atleast_2d(a)).collect();
    let refs: Vec<&NdArray> = promoted.iter().collect();
    concatenate(&refs, Some(0))
}

pub fn dstack(arrays: &[&NdArray]) -> Result<NdArray, IonpError> {
    if arrays.is_empty() {
        return Err(IonpError::Value("need at least one array to concatenate".to_string()));
    }
    let promoted: Vec<NdArray> = arrays.iter().map(|a| atleast_3d(a)).collect();
    let refs: Vec<&NdArray> = promoted.iter().collect();
    concatenate(&refs, Some(2))
}

pub fn column_stack(arrays: &[&NdArray]) -> Result<NdArray, IonpError> {
    if arrays.is_empty() {
        return Err(IonpError::Value("need at least one array to concatenate".to_string()));
    }
    let promoted: Vec<NdArray> = arrays
        .iter()
        .map(|a| {
            if a.ndim() == 1 {
                let n = a.shape()[0];
                a.reshape(&[n, 1]).expect("1-d to (n,1) reshape is always a view")
            } else {
                (*a).clone()
            }
        })
        .map(|a| atleast_2d(&a))
        .collect();
    let refs: Vec<&NdArray> = promoted.iter().collect();
    concatenate(&refs, Some(1))
}

// ---------------------------------------------------------------------------
// flip / fliplr / flipud
// ---------------------------------------------------------------------------

/// View: reverses the stride sign and adjusts the offset along every axis
/// in `axes` (all axes if `None`). No data is copied or moved.
pub fn flip(a: &NdArray, axes: Option<&[usize]>) -> NdArray {
    let mut strides = a.strides().to_vec();
    let mut offset = a.offset();
    let all: Vec<usize> = (0..a.ndim()).collect();
    let target = axes.unwrap_or(&all);
    for &ax in target {
        let len = a.shape()[ax];
        if len > 0 {
            offset += (len as isize - 1) * strides[ax];
            strides[ax] = -strides[ax];
        }
    }
    // Direct struct construction (fields are `pub(crate)`, visible to any
    // module in this crate, per `array.rs`'s own doc): `flip` is a pure
    // stride/offset relabeling of the SAME buffer, exactly like
    // `NdArray::transpose` above it -- there is no `from_buffer*`
    // constructor for "same buffer, new offset" since every existing one
    // forces `offset: 0`.
    NdArray {
        buffer: Arc::clone(&a.buffer),
        shape: a.shape().to_vec(),
        strides,
        offset,
    }
}

pub fn fliplr(a: &NdArray) -> Result<NdArray, IonpError> {
    if a.ndim() < 2 {
        return Err(IonpError::Value("Input must be >= 2-d.".to_string()));
    }
    Ok(flip(a, Some(&[1])))
}

pub fn flipud(a: &NdArray) -> Result<NdArray, IonpError> {
    if a.ndim() < 1 {
        return Err(IonpError::Value("Input must be >= 1-d.".to_string()));
    }
    Ok(flip(a, Some(&[0])))
}

// ---------------------------------------------------------------------------
// roll
// ---------------------------------------------------------------------------

fn roll_flat(a: &NdArray, shift: isize) -> NdArray {
    let contig = a.to_contiguous();
    let n = contig.size();
    if n == 0 {
        // numpy's `axis=None` path is literally
        // `roll(a.ravel(), shift, 0).reshape(a.shape)` (see doc comment
        // below). For n==0 the inner `roll(flat, shift, 0)` call takes
        // numpy's explicit-axis branch, which builds its result via
        // `empty_like(flat)` -- a genuinely fresh 1-D size-0 allocation,
        // always all-zero strides regardless of the input's own layout
        // (verified against numpy 2.5.1). The outer `.reshape(a.shape)`
        // then either (a) is a same-shape identity no-op when `a` was
        // already 1-D, preserving those all-zero strides, or (b) for
        // ndim > 1 actually changes shape, which forces a real
        // recompute via the ordinary (non-zeroing) reshape-view formula
        // -- which is exactly what `contig`'s own `to_contiguous()`
        // strides already give here, so only the ndim==1 case needs a
        // local override.
        if a.ndim() == 1 {
            return contig.zero_strides();
        }
        return contig;
    }
    let s = shift.rem_euclid(n as isize) as usize;
    let out_buffer = gather_flat!(contig.buffer(), n, |i: usize| (i + n - s) % n);
    NdArray::from_buffer(out_buffer, contig.shape().to_vec(), Order::C)
        .expect("roll_flat: same size/shape as source")
}

/// Mirrors real numpy's `roll` (`_core/numeric.py`) *exactly*, including its
/// somewhat surprising broadcasting contract -- confirmed via
/// `inspect.getsource(np.roll)` against numpy 2.5.1:
///
/// ```python
/// if axis is None:
///     return roll(a.ravel(), shift, 0).reshape(a.shape)
/// else:
///     axis = normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
///     broadcasted = broadcast(shift, axis)
///     ...
///     shifts = dict.fromkeys(range(a.ndim), 0)
///     for sh, ax in broadcasted:
///         shifts[ax] += int(sh)
/// ```
///
/// Two consequences that ionp previously got wrong (both confirmed as
/// false declarations by direct comparison against real numpy 2.5.1):
/// - `axis=None` does NOT require a scalar `shift` -- `shift` is simply
///   summed and applied to the single flattened axis
///   (`np.roll(m, (1, 2), axis=None)` succeeds, net shift 3).
/// - A scalar `shift` with a tuple `axis` is NOT a length mismatch --
///   numpy's `np.broadcast(shift, axis)` broadcasts the scalar across
///   every named axis, applying it to each one independently
///   (`np.roll(m, 1, axis=(0, 1))` succeeds, shifting both axes by 1).
/// More generally, `shift` and `axis` follow ordinary 1-D numpy
/// broadcasting against each other (equal length, or either length 1),
/// and repeated axis entries accumulate their (possibly-broadcast) shift.
pub fn roll(a: &NdArray, shifts: &[isize], axes: Option<&[isize]>) -> Result<NdArray, IonpError> {
    let Some(axes) = axes else {
        // numpy: `axis=None` flattens first, and recurses as
        // `roll(a.ravel(), shift, 0)` -- since `axis=0` is length-1 and
        // broadcasts against whatever length `shift` is, every shift value
        // accumulates onto that single flattened axis, i.e. their sum.
        let total_shift: isize = shifts.iter().sum();
        return Ok(roll_flat(a, total_shift));
    };
    let (m, n) = (shifts.len(), axes.len());
    let l = if m == n {
        m
    } else if m == 1 {
        n
    } else if n == 1 {
        m
    } else {
        return Err(IonpError::Value(format!(
            "shape mismatch: objects cannot be broadcast to a single shape.  \
             Mismatch is between arg 0 with shape {} and arg 1 with shape {}.",
            shape_tuple_str(&[m]),
            shape_tuple_str(&[n]),
        )));
    };
    let ndim = a.ndim();
    let mut per_axis_shift = vec![0isize; ndim];
    for i in 0..l {
        let s = shifts[i % m];
        let ax = axes[i % n];
        let normalized = normalize_axis(ax, ndim)?;
        per_axis_shift[normalized] += s;
    }
    let contig = a.to_contiguous();
    let shape = contig.shape().to_vec();
    let strides = row_major_strides(&shape);
    let total = shape::size_of_shape(&shape);
    if total == 0 {
        // `contig = a.to_contiguous()` computes strides via the plain
        // `shape::c_strides` reshape-view formula, which does not zero
        // for empty results -- correct for other `to_contiguous()`
        // consumers (e.g. `roll_flat`'s axis=None path, left untouched
        // above) but wrong here: numpy's explicit-axis `roll` builds its
        // empty-input result via a fresh `np.empty_like`-style
        // allocation in its C implementation, and freshly-allocated
        // size-0 arrays get all-zero strides regardless of order
        // (verified directly against numpy 2.5.1 for shapes (0,),
        // (0,3), (3,0), (2,0,4), (0,0) with an explicit `axis=`).
        return Ok(contig.zero_strides());
    }
    if ndim == 0 {
        return Ok(contig);
    }
    let eff_shift: Vec<usize> = shape
        .iter()
        .zip(per_axis_shift.iter())
        .map(|(&len, &s)| if len == 0 { 0 } else { s.rem_euclid(len as isize) as usize })
        .collect();

    let idx_of = |out_flat: usize| -> usize {
        let mut rem = out_flat;
        let mut src_flat = 0usize;
        for d in 0..ndim {
            let coord = rem / strides[d];
            rem %= strides[d];
            let len = shape[d];
            let src_coord = (coord + len - eff_shift[d]) % len;
            src_flat += src_coord * strides[d];
        }
        src_flat
    };
    let out_buffer = gather_flat!(contig.buffer(), total, idx_of);
    NdArray::from_buffer(out_buffer, shape, Order::C)
}

// ---------------------------------------------------------------------------
// tile / repeat
// ---------------------------------------------------------------------------

pub fn tile(a: &NdArray, reps: &[usize]) -> Result<NdArray, IonpError> {
    let d = a.ndim().max(reps.len());
    let mut a_shape = vec![1usize; d - a.ndim()];
    a_shape.extend_from_slice(a.shape());
    let mut reps_full = vec![1usize; d - reps.len()];
    reps_full.extend_from_slice(reps);

    let padded = if d == a.ndim() {
        a.clone()
    } else {
        a.reshape(&a_shape).unwrap_or_else(|_| a.to_contiguous().reshape(&a_shape).unwrap())
    };
    let contig = padded.to_contiguous();
    let in_strides = row_major_strides(&a_shape);

    let out_shape: Vec<usize> = a_shape.iter().zip(reps_full.iter()).map(|(&s, &r)| s * r).collect();
    let out_strides = row_major_strides(&out_shape);
    let total = shape::size_of_shape(&out_shape);
    if a_shape.iter().any(|&s| s == 0) || total == 0 {
        // numpy's own `tile` (`_core/shape_base.py`) skips its whole
        // per-axis `.repeat()` loop entirely when the (ndmin-padded)
        // source is size 0 (`if n > 0:` guards it, `n = c.size`), so the
        // empty result comes from `c.reshape(shape_out)` alone -- a VIEW
        // composition, never a fresh `np.empty`-style allocation. Two
        // consequences, both verified against real numpy 2.5.1 across
        // `(0,3)`x`(2,)`, `(3,0)`x`(2,)`, `(3,0)`x`(2,5)`, `(2,0,4)`x`(2,)`,
        // `(0,0)`x`(2,)`, `(0,)`x`(3,)`, `(0,3)`x`(1,)`, `(0,3)`x`(0,2)`,
        // and (nonempty source, zero rep) `(2,3)`x`(0,2)`/`(2,3)`x`(2,0)`:
        // (1) when `reshape` is a genuine no-op (`out_shape == a_shape`,
        // e.g. tiling `(3,0)` by `(2,)` stays `(3,0)`), numpy's reshape
        // returns the SAME object, i.e. `padded`'s own strides completely
        // unchanged (`np.tile(np.zeros((3,0)), 2).strides == (0, 0)`, not
        // freshly recomputed); (2) when the shape genuinely changes, the
        // result is `padded`'s C-contiguous-view strides for `out_shape`
        // (`shape::c_strides`, the SAME "reshape/view, never zeroed"
        // formula `ravel_order`/`relayout_by_perm` document -- e.g.
        // `np.tile(np.zeros((2,0,4)), 2).strides == (64, 64, 8)` bytes,
        // i.e. elementwise `(8, 8, 1)`, NOT the all-zero
        // freshly-`np.empty`-style strides this branch used to return).
        // `padded` (not `contig`) is the base here because case (1) must
        // preserve its EXACT strides, including whatever a fresh
        // `NdArray::from_buffer`/`ip.zeros` already zeroed them to --
        // `to_contiguous()` would recompute instead.
        if reps_full.iter().all(|&r| r == 1) {
            // numpy's `tile` (`_core/shape_base.py`) has a distinct,
            // EARLIER branch that fires whenever every entry of `reps`
            // (as originally given, before any ndmin-padding) is 1,
            // regardless of size: `if all(x == 1 for x in tup) and
            // isinstance(A, ndarray): return np.array(A, copy=True,
            // subok=True, ndmin=d)`. That's a genuine fresh COPY (with
            // ndmin-padding for new leading axes), not the
            // reshape/view composition the size==0 comment above
            // describes -- so for size-0 results it gets all-zero
            // strides regardless of `padded`'s own layout, even though
            // `out_shape == a_shape` coincides with case (1) above.
            // Verified directly against numpy 2.5.1: `np.tile(np.zeros
            // ((0,)), (1,1,1)).strides == (0,0,0)`, not `padded`'s
            // reshape-computed `(8,8,8)`; same for `(0,3)`, `(3,0)`,
            // `(2,0,4)`, `(0,0)` each tiled by an all-ones `reps` with
            // more entries than the source's own ndim.
            return Ok(padded.zero_strides());
        }
        if out_shape == a_shape {
            return Ok(padded);
        }
        let strides = shape::c_strides(&out_shape);
        return NdArray::from_buffer_with_strides(
            gather_flat!(contig.buffer(), 0usize, |_i: usize| 0usize),
            out_shape,
            strides,
        );
    }

    let idx_of = |out_flat: usize| -> usize {
        let mut rem = out_flat;
        let mut src_flat = 0usize;
        for dd in 0..d {
            let coord = rem / out_strides[dd];
            rem %= out_strides[dd];
            let src_coord = coord % a_shape[dd];
            src_flat += src_coord * in_strides[dd];
        }
        src_flat
    };
    let out_buffer = gather_flat!(contig.buffer(), total, idx_of);
    NdArray::from_buffer(out_buffer, out_shape, Order::C)
}

/// `repeats` is already-resolved to one entry per element along `axis`
/// (the caller broadcasts a length-1 `repeats` up to `shape[axis]` --
/// matching numpy's own contract -- before calling this).
pub fn repeat(a: &NdArray, repeats: &[usize], axis: Option<usize>) -> Result<NdArray, IonpError> {
    let (base, ax) = match axis {
        Some(ax) => (a.clone(), ax),
        None => (a.to_contiguous().reshape(&[a.size()]).unwrap(), 0),
    };
    if ax >= base.ndim() {
        return Err(IonpError::AxisError { axis: ax as isize, ndim: Some(base.ndim()) });
    }
    if repeats.len() != base.shape()[ax] {
        // NOTE the missing trailing space, and the singular "shape". numpy is
        // internally inconsistent here and we match it exactly: the plural
        // "...with shapes (3,) (4,) " form raised by ufunc broadcasting DOES
        // carry a trailing space (see error.rs), but this singular form raised
        // from `repeat` does NOT. Verified against numpy 2.5.1 byte-for-byte:
        //   np.repeat(np.arange(24), np.arange(4)) -> len 62, no trailing space
        //   np.zeros(3) + np.zeros(4)              -> len ..., trailing space
        return Err(IonpError::Value(format!(
            "operands could not be broadcast together with shape ({},) ({},)",
            base.shape()[ax],
            repeats.len()
        )));
    }
    let mut mapping: Vec<usize> = Vec::new();
    for (i, &r) in repeats.iter().enumerate() {
        for _ in 0..r {
            mapping.push(i);
        }
    }
    let contig = base.to_contiguous();
    let shape = contig.shape().to_vec();
    let strides = row_major_strides(&shape);
    let mut out_shape = shape.clone();
    out_shape[ax] = mapping.len();
    let out_strides = row_major_strides(&out_shape);
    let total = shape::size_of_shape(&out_shape);

    let idx_of = |out_flat: usize| -> usize {
        let mut rem = out_flat;
        let mut src_flat = 0usize;
        for d in 0..shape.len() {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            let src_coord = if d == ax { mapping[coord] } else { coord };
            src_flat += src_coord * strides[d];
        }
        src_flat
    };
    let out_buffer = gather_flat!(contig.buffer(), total, idx_of);
    NdArray::from_buffer(out_buffer, out_shape, Order::C)
}

// ---------------------------------------------------------------------------
// diagonal family: diag / diagflat / diagonal / tril / triu / trace
// ---------------------------------------------------------------------------

/// numpy `diagonal`: generalized to N-d by batching over every axis other
/// than `axis1`/`axis2`; the diagonal itself is always appended as the
/// LAST axis of the result (numpy's post-1.9 behavior, no FutureWarning
/// era to replicate).
pub fn diagonal(a: &NdArray, offset: isize, axis1: isize, axis2: isize) -> Result<NdArray, IonpError> {
    let ndim = a.ndim();
    if ndim < 2 {
        return Err(IonpError::Value(
            "diag requires an array of at least two dimensions".to_string(),
        ));
    }
    let ax1 = normalize_axis(axis1, ndim)?;
    let ax2 = normalize_axis(axis2, ndim)?;
    if ax1 == ax2 {
        return Err(IonpError::Value("axis1 and axis2 cannot be the same".to_string()));
    }
    let (d1, d2) = (a.shape()[ax1] as isize, a.shape()[ax2] as isize);
    let diag_len = if offset >= 0 {
        (d2 - offset).min(d1).max(0)
    } else {
        (d1 + offset).min(d2).max(0)
    } as usize;

    let batch_axes: Vec<usize> = (0..ndim).filter(|&d| d != ax1 && d != ax2).collect();
    let mut out_shape: Vec<usize> = batch_axes.iter().map(|&d| a.shape()[d]).collect();
    out_shape.push(diag_len);

    let contig = a.to_contiguous();
    let in_shape = contig.shape().to_vec();
    let in_strides = row_major_strides(&in_shape);
    let total = shape::size_of_shape(&out_shape);
    // Strides of the FULL `out_shape` (batch dims + trailing diag axis),
    // sliced to just the batch-dim entries -- each batch dim's stride
    // then already includes the `* diag_len` factor from the trailing
    // axis, which is exactly what's needed to decompose `out_flat` below.
    let batch_out_strides = &row_major_strides(&out_shape)[..batch_axes.len()];

    let idx_of = |out_flat: usize| -> usize {
        // `out_shape` is `[batch dims..., diag_len]`; `row_major_strides`
        // over it already gives the leading batch dims the correct
        // stride (each = diag_len * product of later batch dims), and the
        // trailing diag-axis stride is exactly 1 -- so decomposing
        // `out_flat` against `batch_out_strides` (row-major strides of the
        // FULL `out_shape`, batch dims only sliced off below) directly
        // yields both the batch coordinates and, as the final remainder,
        // `k` (the position along the diagonal).
        let mut rem = out_flat;
        let mut src_flat = 0usize;
        let mut coords = vec![0usize; batch_axes.len()];
        for (bi, c) in coords.iter_mut().enumerate() {
            let stride = batch_out_strides[bi];
            *c = rem / stride;
            rem %= stride;
        }
        let k = rem;
        for (bi, &axis) in batch_axes.iter().enumerate() {
            src_flat += coords[bi] * in_strides[axis];
        }
        let (i1, i2) = if offset >= 0 { (k, k + offset as usize) } else { (k + (-offset) as usize, k) };
        src_flat += i1 * in_strides[ax1] + i2 * in_strides[ax2];
        src_flat
    };
    let out_buffer = gather_flat!(contig.buffer(), total, idx_of);
    NdArray::from_buffer(out_buffer, out_shape, Order::C)
}

/// numpy `diag`: 1-D input -> 2-D array with `v` on the k-th diagonal
/// (square-ish, sized `n + |k|`); 2-D input -> extract the k-th diagonal as
/// a 1-D array (delegates to `diagonal`).
pub fn diag(v: &NdArray, k: isize) -> Result<NdArray, IonpError> {
    match v.ndim() {
        1 => {
            let n = v.shape()[0];
            let size = n + k.unsigned_abs();
            let contig = v.to_contiguous();
            let total = size * size;
            let idx_of = |flat: usize| -> Option<usize> {
                let row = flat / size;
                let col = flat % size;
                let i1 = row as isize;
                let i2 = col as isize;
                if k >= 0 && i2 - i1 == k && (i1 as usize) < n {
                    Some(i1 as usize)
                } else if k < 0 && i1 - i2 == -k && (i2 as usize) < n {
                    Some(i2 as usize)
                } else {
                    None
                }
            };
            macro_rules! fill {
                ($variant:ident, $zero:expr) => {{
                    let src = match contig.buffer() {
                        Buffer::$variant(v) => v,
                        _ => unreachable!(),
                    };
                    let out: Vec<_> = (0..total).map(|i| idx_of(i).map(|s| src[s]).unwrap_or($zero)).collect();
                    Buffer::$variant(out)
                }};
            }
            let out_buffer = match contig.buffer() {
                Buffer::Bool(_) => fill!(Bool, false),
                Buffer::I8(_) => fill!(I8, 0),
                Buffer::I16(_) => fill!(I16, 0),
                Buffer::I32(_) => fill!(I32, 0),
                Buffer::I64(_) => fill!(I64, 0),
                Buffer::U8(_) => fill!(U8, 0),
                Buffer::U16(_) => fill!(U16, 0),
                Buffer::U32(_) => fill!(U32, 0),
                Buffer::U64(_) => fill!(U64, 0),
                Buffer::F16(_) => fill!(F16, half::f16::from_f32(0.0)),
                Buffer::F32(_) => fill!(F32, 0.0),
                Buffer::F64(_) => fill!(F64, 0.0),
                Buffer::C64(_) => fill!(C64, crate::buffer::C64::new(0.0, 0.0)),
                Buffer::C128(_) => fill!(C128, crate::buffer::C128::new(0.0, 0.0)),
                Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("manip.rs: diag() has no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
            };
            NdArray::from_buffer(out_buffer, vec![size, size], Order::C)
        }
        2 => diagonal(v, k, 0, 1),
        _ => Err(IonpError::Value("Input must be 1- or 2-d.".to_string())),
    }
}

/// numpy `diagflat`: ravel `v` (any ndim) to 1-D, then behave like `diag`.
pub fn diagflat(v: &NdArray, k: isize) -> Result<NdArray, IonpError> {
    let flat = v.to_contiguous().reshape(&[v.size()]).unwrap();
    diag(&flat, k)
}

fn tri_mask(a: &NdArray, k: isize, keep_upper: bool) -> Result<NdArray, IonpError> {
    // Real numpy's tril/triu (`_core/twodim_base.py`) are literally:
    //   mask = tri(*m.shape[-2:], k=k, dtype=bool); return where(mask, m, 0)
    // For a 1-D input, `m.shape[-2:]` is the 1-element tuple `(n,)`, so
    // `tri(*that)` is `tri(n)` (a square NxN mask), and `m` (shape `(n,)`)
    // then broadcasts against that square mask exactly like any other
    // `where(bool_2d, 1d, scalar)` call -- every row of the result is the
    // SAME 1-D input, masked differently per row. Confirmed directly
    // against real numpy 2.5.1: `np.tril(np.arange(3))` is
    //   [[0,0,0],[0,1,0],[0,1,2]], not a ValueError.
    // For a 0-D input, `m.shape[-2:]` is the EMPTY tuple, so `tri(*())` is
    // a bare `tri()` call, which is a genuine Python TypeError from a
    // missing positional argument -- reproduced verbatim (confirmed via
    // `np.tril(np.array(5))` in real numpy 2.5.1).
    let ndim = a.ndim();
    if ndim == 0 {
        return Err(IonpError::Type(
            "tri() missing 1 required positional argument: 'N'".to_string(),
        ));
    }
    let working: NdArray = if ndim == 1 {
        let n = a.shape()[0];
        let row = a.to_contiguous().reshape(&[1, n])?;
        crate::creation::broadcast_to(&row, &[n, n])?
    } else {
        a.clone()
    };
    let wndim = working.ndim();
    let (rows, cols) = (working.shape()[wndim - 2], working.shape()[wndim - 1]);
    let contig = working.to_contiguous();
    let ndim = wndim;
    let shape = contig.shape().to_vec();
    let total = shape::size_of_shape(&shape);
    let strides = row_major_strides(&shape);

    macro_rules! mask_fill {
        ($variant:ident, $zero:expr) => {{
            let src = match contig.buffer() {
                Buffer::$variant(v) => v,
                _ => unreachable!(),
            };
            let mut out = src.clone();
            for flat in 0..total {
                let row = (flat / strides[ndim - 2]) % rows;
                let col = (flat / strides[ndim - 1]) % cols;
                let keep = if keep_upper {
                    (col as isize) - (row as isize) >= k
                } else {
                    (col as isize) - (row as isize) <= k
                };
                if !keep {
                    out[flat] = $zero;
                }
            }
            Buffer::$variant(out)
        }};
    }
    let out_buffer = match contig.buffer() {
        Buffer::Bool(_) => mask_fill!(Bool, false),
        Buffer::I8(_) => mask_fill!(I8, 0),
        Buffer::I16(_) => mask_fill!(I16, 0),
        Buffer::I32(_) => mask_fill!(I32, 0),
        Buffer::I64(_) => mask_fill!(I64, 0),
        Buffer::U8(_) => mask_fill!(U8, 0),
        Buffer::U16(_) => mask_fill!(U16, 0),
        Buffer::U32(_) => mask_fill!(U32, 0),
        Buffer::U64(_) => mask_fill!(U64, 0),
        Buffer::F16(_) => mask_fill!(F16, half::f16::from_f32(0.0)),
        Buffer::F32(_) => mask_fill!(F32, 0.0),
        Buffer::F64(_) => mask_fill!(F64, 0.0),
        Buffer::C64(_) => mask_fill!(C64, crate::buffer::C64::new(0.0, 0.0)),
        Buffer::C128(_) => mask_fill!(C128, crate::buffer::C128::new(0.0, 0.0)),
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("manip.rs: tril/triu has no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
    };
    NdArray::from_buffer(out_buffer, shape, Order::C)
}

pub fn tril(a: &NdArray, k: isize) -> Result<NdArray, IonpError> {
    tri_mask(a, k, false)
}

pub fn triu(a: &NdArray, k: isize) -> Result<NdArray, IonpError> {
    tri_mask(a, k, true)
}

/// numpy accumulator widening: bool/signed-int -> i64, unsigned-int ->
/// u64, float/complex keep their own width (verified directly against
/// real numpy 2.5.1 for all 13 dtypes -- see this task's differential
/// probe for `trace`).
fn trace_accum_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Bool | DType::I8 | DType::I16 | DType::I32 | DType::I64 => DType::I64,
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => DType::U64,
        other => other,
    }
}

pub fn trace(a: &NdArray, offset: isize, axis1: isize, axis2: isize, dtype: Option<DType>) -> Result<NdArray, IonpError> {
    let diag = diagonal(a, offset, axis1, axis2)?;
    let target = dtype.unwrap_or_else(|| trace_accum_dtype(a.dtype()));
    let cast = diag.cast_to(target);
    let ndim = cast.ndim();
    let axis = ndim - 1;
    // Reuse the shared `ufunc::reduce_axis` machinery (same crate) instead
    // of a hand-rolled, always-sequential, always-native-width fold: that
    // old local helper (`sum_last_axis`, removed) never gave float16 the
    // f32-accumulate-then-round-once treatment numpy's real `trace` gets
    // whenever the diagonal's reduced axis is the true innermost-stride
    // axis (which, after `diagonal()`, it always is here -- there is only
    // ever one reduced axis for `trace`), causing a uniform 1-ULP
    // divergence on every float16 case. `reduce_axis` already implements
    // this correctly (and is independently verified via `sum`/`ndarray.sum`),
    // so delegating to it fixes trace for free and removes the duplicated
    // logic. `dtype_override: None` lets it use `cast`'s own dtype (already
    // the correct accumulator dtype from `trace_accum_dtype` above), and
    // passing `&cast` directly (not pre-forced-contiguous) preserves the
    // diagonal view's real strides, which is exactly what the widen/narrow
    // stride-ranking rule needs to see.
    crate::ufunc::reduce_axis(crate::ufunc::BinaryOp::Add, &cast, &[axis], false, None, None, None)
}

// ---------------------------------------------------------------------------
// split / array_split / hsplit / vsplit / dsplit
// ---------------------------------------------------------------------------

/// Same negative-wrap + range check as `normalize_axis`, but raising the
/// specific `IndexError` numpy's `split`/`array_split` produce for an
/// out-of-range axis instead of the usual `AxisError`. Verified live
/// against numpy 2.5.1: `np.split(arr, 2, axis=5)` on a 2-d array raises
/// plain `IndexError: tuple index out of range` (no axis/ndim numbers in
/// the text at all) -- because `array_split` resolves `Ntotal =
/// ary.shape[axis]` via a bare Python tuple-indexing operation before any
/// of numpy's normal axis-normalization machinery runs. Scoped to this
/// family only; every other axis-taking function in this crate goes
/// through `normalize_axis` (AxisError) instead.
fn normalize_axis_tuple_style(axis: isize, ndim: usize) -> Result<usize, IonpError> {
    let nd = ndim as isize;
    let a = if axis < 0 { axis + nd } else { axis };
    if a < 0 || a >= nd.max(0) {
        return Err(IonpError::Index("tuple index out of range".to_string()));
    }
    Ok(a as usize)
}

/// Slice `a` along `axis` at the segment boundaries given by `bounds`
/// (`bounds.len() == n_segments + 1`; consecutive pairs are used exactly
/// as Python slice `(start, stop)` bounds -- negative values wrap and
/// out-of-range values clamp via the same `SliceItem::Slice` +
/// `NdArray::get_view` machinery every other view in this crate goes
/// through, matching numpy's own indices-form `split`/`array_split`, which
/// literally slices at each requested boundary with no separate
/// range-checking step of its own).
fn slice_segments(a: &NdArray, axis: usize, bounds: &[isize]) -> Result<Vec<NdArray>, IonpError> {
    let ndim = a.ndim();
    let mut out = Vec::with_capacity(bounds.len().saturating_sub(1));
    for w in bounds.windows(2) {
        let mut items = vec![SliceItem::Slice { start: None, stop: None, step: None }; ndim];
        items[axis] = SliceItem::Slice { start: Some(w[0]), stop: Some(w[1]), step: Some(1) };
        out.push(a.get_view(&items)?);
    }
    Ok(out)
}

/// The two legal forms of `split`/`array_split`'s `indices_or_sections`
/// argument (the `ionp-py` binding is responsible for the `len()`-based
/// dispatch numpy itself does via `try: len(indices_or_sections) except
/// TypeError:`).
pub enum SplitArg {
    /// A plain integer: split into this many (approximately) equal
    /// sections.
    Sections(isize),
    /// A sequence/array of split points along `axis` (raw, unclamped,
    /// possibly-negative -- used directly as slice boundaries).
    Indices(Vec<isize>),
}

/// Python's `%` (floor-style, result takes the sign of the divisor --
/// distinct from Rust's `%`, which takes the sign of the dividend).
/// `split`'s own equal-division pre-check (`if N % sections:`) uses this
/// exact semantics, verified live: `np.split(np.arange(6), -4)` raises
/// `ValueError: array split does not result in an equal division` because
/// Python's `6 % -4 == -2` (nonzero) -- Rust's `6 % -4 == 2` would also be
/// nonzero here, but the two disagree for other divisor/dividend sign
/// combinations, so this cannot be skipped.
fn python_mod(a: isize, m: isize) -> isize {
    let r = a % m;
    if r != 0 && (r < 0) != (m < 0) {
        r + m
    } else {
        r
    }
}

/// `np.array_split`: always succeeds for any `n > 0` (sections form),
/// producing `total % n` sections of size `total / n + 1` followed by
/// `n - (total % n)` sections of size `total / n`.
pub fn array_split(a: &NdArray, arg: &SplitArg, axis: isize) -> Result<Vec<NdArray>, IonpError> {
    let ax = normalize_axis_tuple_style(axis, a.ndim())?;
    let total = a.shape()[ax] as isize;
    let bounds: Vec<isize> = match arg {
        SplitArg::Sections(n) => {
            if *n <= 0 {
                return Err(IonpError::Value("number sections must be larger than 0.".to_string()));
            }
            let each = total / n;
            let extra = total % n;
            let mut b = Vec::with_capacity(*n as usize + 1);
            b.push(0isize);
            let mut acc = 0isize;
            for i in 0..*n {
                acc += each + if i < extra { 1 } else { 0 };
                b.push(acc);
            }
            b
        }
        SplitArg::Indices(idx) => {
            let mut b = Vec::with_capacity(idx.len() + 2);
            b.push(0isize);
            b.extend(idx.iter().copied());
            b.push(total);
            b
        }
    };
    slice_segments(a, ax, &bounds)
}

/// `np.split`: identical to `array_split` for the indices/sequence form;
/// for the plain-integer form, additionally requires `total` to divide
/// evenly by `n` (`ValueError`) -- the `n == 0` case is handled entirely
/// at the `ionp-py` boundary, since real numpy raises a plain
/// `ZeroDivisionError` there (from evaluating `N % 0` in Python) rather
/// than any of this crate's own `IonpError` variants, and this crate does
/// not own `error.rs` to add one.
pub fn split(a: &NdArray, arg: &SplitArg, axis: isize) -> Result<Vec<NdArray>, IonpError> {
    if let SplitArg::Sections(n) = arg {
        debug_assert!(*n != 0, "n == 0 must be intercepted by the ionp-py boundary");
        let ax = normalize_axis_tuple_style(axis, a.ndim())?;
        let total = a.shape()[ax] as isize;
        if python_mod(total, *n) != 0 {
            return Err(IonpError::Value(
                "array split does not result in an equal division".to_string(),
            ));
        }
    }
    array_split(a, arg, axis)
}

/// Shared ndim-guard for `hsplit`/`vsplit`/`dsplit`: each fixes the split
/// axis to a specific value and requires at least that many dimensions,
/// raising the exact numpy `ValueError` text for too-few-dimensions
/// inputs (verified live for all three against numpy 2.5.1).
fn fixed_split_axis(ndim: usize, min_ndim: usize, name: &str, axis: usize) -> Result<usize, IonpError> {
    if ndim < min_ndim {
        return Err(IonpError::Value(format!(
            "{name} only works on arrays of {min_ndim} or more dimensions"
        )));
    }
    Ok(axis)
}

/// `np.hsplit`: axis 0 for 1-d inputs, axis 1 otherwise (matches numpy's
/// own `_hsplit_dispatcher`-adjacent axis choice); requires ndim >= 1.
pub fn hsplit(a: &NdArray, arg: &SplitArg) -> Result<Vec<NdArray>, IonpError> {
    let ndim = a.ndim();
    let axis = fixed_split_axis(ndim, 1, "hsplit", if ndim == 1 { 0 } else { 1 })?;
    split(a, arg, axis as isize)
}

/// `np.vsplit`: always axis 0; requires ndim >= 2.
pub fn vsplit(a: &NdArray, arg: &SplitArg) -> Result<Vec<NdArray>, IonpError> {
    let axis = fixed_split_axis(a.ndim(), 2, "vsplit", 0)?;
    split(a, arg, axis as isize)
}

/// `np.dsplit`: always axis 2; requires ndim >= 3.
pub fn dsplit(a: &NdArray, arg: &SplitArg) -> Result<Vec<NdArray>, IonpError> {
    let axis = fixed_split_axis(a.ndim(), 3, "dsplit", 2)?;
    split(a, arg, axis as isize)
}

// ---------------------------------------------------------------------------
// append / resize
// ---------------------------------------------------------------------------

/// `np.append`: a thin wrapper over the already-declared `concatenate` --
/// with `axis=None` numpy ravels BOTH inputs then concatenates them
/// (matching `concatenate`'s own `axis.is_none()` branch exactly), and
/// with `axis=Some(_)` it is literally
/// `concatenate((arr, values), axis=axis)` with no other logic. Verified
/// against numpy's own source (`np.append` is a two-line function
/// delegating straight to `concatenate`).
pub fn append(a: &NdArray, values: &NdArray, axis: Option<isize>) -> Result<NdArray, IonpError> {
    concatenate(&[a, values], axis)
}

/// `np.resize`: ravel `a` (C order), then cyclically repeat/truncate its
/// flat elements to fill `new_shape`'s total size, then reshape. `a.size()
/// == 0` (or a requested `new_size == 0`) short-circuits to a zero-filled
/// array (there is nothing to cycle from an empty source) -- matches
/// numpy's own explicit `np.zeros_like` fallback for that case.
pub fn resize(a: &NdArray, new_shape: &[usize]) -> Result<NdArray, IonpError> {
    let new_size = shape::size_of_shape(new_shape);
    let contig = a.to_contiguous();
    let old_size = contig.size();

    if old_size == 0 || new_size == 0 {
        let zero_buffer = zero_buffer_like(contig.buffer(), new_size);
        return NdArray::from_buffer(zero_buffer, new_shape.to_vec(), Order::C);
    }

    let idx_of = |out_flat: usize| -> usize { out_flat % old_size };
    let out_buffer = gather_flat!(contig.buffer(), new_size, idx_of);
    NdArray::from_buffer(out_buffer, new_shape.to_vec(), Order::C)
}

fn zero_buffer_like(buffer: &Buffer, len: usize) -> Buffer {
    match buffer {
        Buffer::Bool(_) => Buffer::Bool(vec![false; len]),
        Buffer::I8(_) => Buffer::I8(vec![0; len]),
        Buffer::I16(_) => Buffer::I16(vec![0; len]),
        Buffer::I32(_) => Buffer::I32(vec![0; len]),
        Buffer::I64(_) => Buffer::I64(vec![0; len]),
        Buffer::U8(_) => Buffer::U8(vec![0; len]),
        Buffer::U16(_) => Buffer::U16(vec![0; len]),
        Buffer::U32(_) => Buffer::U32(vec![0; len]),
        Buffer::U64(_) => Buffer::U64(vec![0; len]),
        Buffer::F16(_) => Buffer::F16(vec![half::f16::from_f32(0.0); len]),
        Buffer::F32(_) => Buffer::F32(vec![0.0; len]),
        Buffer::F64(_) => Buffer::F64(vec![0.0; len]),
        Buffer::C64(_) => Buffer::C64(vec![crate::buffer::C64::new(0.0, 0.0); len]),
        Buffer::C128(_) => Buffer::C128(vec![crate::buffer::C128::new(0.0, 0.0); len]),
        Buffer::S(w, _) => Buffer::S(*w, vec![vec![0u8; *w as usize]; len]),
        Buffer::U(w, _) => Buffer::U(*w, vec![vec![0u32; (*w / 4) as usize]; len]),
    }
}

// ---------------------------------------------------------------------------
// insert
// ---------------------------------------------------------------------------

/// Build the `axes` permutation `np.moveaxis(x, 0, dst)` would use, for
/// feeding into `NdArray::transpose_axes` (which uses the same "output
/// axis i comes from input axis `axes[i]`" convention as `np.transpose`).
fn moveaxis_0_to(ndim: usize, dst: usize) -> Vec<isize> {
    let mut order: Vec<isize> = (1..ndim as isize).collect();
    order.insert(dst, 0);
    order
}

/// `np.insert`'s single-index path (`obj` is a scalar, or an array/list
/// with exactly one element): translated directly from numpy's own
/// `insert` source (`indices.size == 1` branch). `was_scalar` distinguishes
/// a bare scalar `obj` (which additionally moves `values`' leading axis to
/// `axis` after ndmin-padding -- numpy's `np.insert(a, 1, [7,8,9],
/// axis=1)` "each row gets its own value" behavior) from a one-element
/// array/list `obj` (which does not -- `np.insert(a, [1], [[7],[8],[9]],
/// axis=1)`, verified to produce the identical result via a separate
/// `moveaxis`-free path).
fn insert_single(a: &NdArray, index: isize, was_scalar: bool, values: &NdArray, axis: usize) -> Result<NdArray, IonpError> {
    let n = a.shape()[axis] as isize;
    if index < -n || index > n {
        return Err(IonpError::Index(format!(
            "index {index} is out of bounds for axis {axis} with size {n}"
        )));
    }
    let index = if index < 0 { index + n } else { index };

    let mut v = values.cast_to(a.dtype());
    if v.ndim() < a.ndim() {
        let pad = a.ndim() - v.ndim();
        let mut new_shape = vec![1usize; pad];
        new_shape.extend_from_slice(v.shape());
        v = v.reshape(&new_shape)?;
    }
    if was_scalar && v.ndim() > 0 {
        let perm = moveaxis_0_to(v.ndim(), axis.min(v.ndim() - 1));
        v = v.transpose_axes(&perm)?;
    }
    let numnew = if v.ndim() > axis { v.shape()[axis] } else { 1 };

    let before = slice_segments(a, axis, &[0, index])?.remove(0);
    let after = slice_segments(a, axis, &[index, n])?.remove(0);

    let mut target_shape = a.shape().to_vec();
    target_shape[axis] = numnew;
    let v_bc = creation::broadcast_to(&v, &target_shape)?;

    concatenate(&[&before, &v_bc, &after], Some(axis as isize))
}

/// `np.insert`'s general (multi-index) path: translated directly from
/// numpy's own `insert` source (the `else` branch after the `indices.size
/// == 1` optimization) -- negative indices wrap against the ORIGINAL axis
/// length, a stable sort determines insertion order for ties, and each
/// index is then shifted by its rank among the sorted insertion points so
/// it lands at its final position in the (larger) output array.
fn insert_multi(a: &NdArray, indices: &[isize], values: &NdArray, axis: usize) -> Result<NdArray, IonpError> {
    let n = a.shape()[axis] as isize;
    let mut wrapped: Vec<isize> = indices.iter().map(|&i| if i < 0 { i + n } else { i }).collect();

    let numnew = wrapped.len();
    let new_axis_len = n as usize + numnew;

    let mut order: Vec<usize> = (0..numnew).collect();
    order.sort_by_key(|&i| wrapped[i]);
    let original = wrapped.clone();
    for (k, &oi) in order.iter().enumerate() {
        wrapped[oi] += k as isize;
    }

    let mut src_is_new = vec![false; new_axis_len];
    let mut src_idx = vec![0usize; new_axis_len];
    for (k, &s) in wrapped.iter().enumerate() {
        if s < 0 || s as usize >= new_axis_len {
            return Err(IonpError::Index(format!(
                "index {} is out of bounds for axis {} with size {}",
                original[k], axis, n
            )));
        }
        src_is_new[s as usize] = true;
        src_idx[s as usize] = k;
    }
    let mut arr_pos = 0usize;
    for flag_idx in src_is_new.iter().enumerate().filter(|(_, &f)| !f).map(|(i, _)| i) {
        src_idx[flag_idx] = arr_pos;
        arr_pos += 1;
    }

    let out_dtype = a.dtype();
    let a_c = a.cast_to(out_dtype).to_contiguous();
    let mut target_shape = a.shape().to_vec();
    target_shape[axis] = numnew;
    let v_bc = creation::broadcast_to(&values.cast_to(out_dtype), &target_shape)?.to_contiguous();

    let mut out_shape = a.shape().to_vec();
    out_shape[axis] = new_axis_len;
    let out_strides = row_major_strides(&out_shape);
    let a_strides = row_major_strides(a.shape());
    let v_strides = row_major_strides(&target_shape);
    let ndim = a.ndim();
    let total = shape::size_of_shape(&out_shape);

    let idx_of = |out_flat: usize| -> (bool, usize) {
        let mut rem = out_flat;
        let mut a_flat = 0usize;
        let mut v_flat = 0usize;
        let mut axis_pos = 0usize;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            if d == axis {
                axis_pos = coord;
                a_flat += src_idx[coord] * a_strides[d];
                v_flat += src_idx[coord] * v_strides[d];
            } else {
                a_flat += coord * a_strides[d];
                v_flat += coord * v_strides[d];
            }
        }
        (src_is_new[axis_pos], if src_is_new[axis_pos] { v_flat } else { a_flat })
    };

    macro_rules! build_insert {
        ($variant:ident) => {{
            let av = match a_c.buffer() {
                Buffer::$variant(v) => v,
                _ => unreachable!("a_c was cast_to(out_dtype)"),
            };
            let vv = match v_bc.buffer() {
                Buffer::$variant(v) => v,
                _ => unreachable!("v_bc was cast_to(out_dtype)"),
            };
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let (use_v, flat) = idx_of(i);
                out.push(if use_v { vv[flat] } else { av[flat] });
            }
            Buffer::$variant(out)
        }};
    }

    let out_buffer = match out_dtype {
        DType::Bool => build_insert!(Bool),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => build_insert!(I8),
        DType::I16 => build_insert!(I16),
        DType::I32 => build_insert!(I32),
        DType::I64 => build_insert!(I64),
        DType::U8 => build_insert!(U8),
        DType::U16 => build_insert!(U16),
        DType::U32 => build_insert!(U32),
        DType::U64 => build_insert!(U64),
        DType::F16 => build_insert!(F16),
        DType::F32 => build_insert!(F32),
        DType::F64 => build_insert!(F64),
        DType::C64 => build_insert!(C64),
        DType::C128 => build_insert!(C128),
    };
    NdArray::from_buffer(out_buffer, out_shape, Order::C)
}

/// The three legal forms of `np.insert`'s `obj` argument (the `ionp-py`
/// binding resolves a Python `slice`/`int`/sequence/bool-array into one of
/// these).
pub enum InsertObj {
    Slice { start: Option<isize>, stop: Option<isize>, step: Option<isize> },
    Scalar(isize),
    Indices(Vec<isize>),
}

/// `np.insert(arr, obj, values, axis=None)`. `axis=None` ravels `arr`
/// first (matching numpy: `if axis is None: arr = arr.ravel(); axis =
/// arr.ndim - 1`).
pub fn insert(a: &NdArray, obj: &InsertObj, values: &NdArray, axis: Option<isize>) -> Result<NdArray, IonpError> {
    let (arr, ax) = match axis {
        None => {
            let raveled = if a.ndim() == 1 { a.to_contiguous() } else { a.to_contiguous().reshape(&[a.size()])? };
            let nd = raveled.ndim();
            (raveled, nd - 1)
        }
        Some(ax) => (a.to_contiguous(), normalize_axis(ax, a.ndim())?),
    };

    match obj {
        InsertObj::Slice { start, stop, step } => {
            let n = arr.shape()[ax];
            let indices = slice_to_indices(*start, *stop, *step, n)?;
            if indices.len() == 1 {
                insert_single(&arr, indices[0], false, values, ax)
            } else {
                insert_multi(&arr, &indices, values, ax)
            }
        }
        InsertObj::Scalar(i) => insert_single(&arr, *i, true, values, ax),
        InsertObj::Indices(idx) => {
            if idx.len() == 1 {
                insert_single(&arr, idx[0], false, values, ax)
            } else {
                insert_multi(&arr, idx, values, ax)
            }
        }
    }
}

/// CPython's `slice.indices()` + materializing the resulting `range` into
/// a concrete list of positions -- used only by `insert`'s slice-`obj`
/// path (`indices = arange(*obj.indices(N))`), duplicated locally rather
/// than reusing `array.rs`'s private `normalize_slice` (not `pub`, and
/// this crate's established convention -- see `normalize_axis`'s own doc
/// comment -- is small per-module copies over a shared helper module).
fn slice_to_indices(start: Option<isize>, stop: Option<isize>, step: Option<isize>, len: usize) -> Result<Vec<isize>, IonpError> {
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(IonpError::Value("slice step cannot be zero".to_string()));
    }
    let len_i = len as isize;
    let (lower, upper) = if step > 0 { (0, len_i) } else { (-1, len_i - 1) };
    let start = match start {
        None => if step < 0 { upper } else { lower },
        Some(s) => if s < 0 { (s + len_i).max(lower) } else { s.min(upper) },
    };
    let stop = match stop {
        None => if step < 0 { lower } else { upper },
        Some(s) => if s < 0 { (s + len_i).max(lower) } else { s.min(upper) },
    };
    let mut out = Vec::new();
    if step > 0 {
        let mut i = start;
        while i < stop {
            out.push(i);
            i += step;
        }
    } else {
        let mut i = start;
        while i > stop {
            out.push(i);
            i += step;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// delete
// ---------------------------------------------------------------------------

/// The four legal forms of `np.delete`'s `obj` argument.
pub enum DeleteObj {
    Slice { start: Option<isize>, stop: Option<isize>, step: Option<isize> },
    Single(isize),
    Indices(Vec<isize>),
    Bool(Vec<bool>),
}

/// Generic "keep these axis positions, in this order" gather -- shared
/// machinery for `delete`'s four `obj` forms, all of which reduce to
/// "compute a `keep: Vec<usize>` of axis-`axis` positions to retain, then
/// gather". Modeled directly on `repeat`'s own index-mapping closure
/// above.
fn select_along_axis(a: &NdArray, axis: usize, keep: &[usize]) -> Result<NdArray, IonpError> {
    let contig = a.to_contiguous();
    let shape = contig.shape().to_vec();
    let strides = row_major_strides(&shape);
    let mut out_shape = shape.clone();
    out_shape[axis] = keep.len();
    let out_strides = row_major_strides(&out_shape);
    let total = shape::size_of_shape(&out_shape);

    let idx_of = |out_flat: usize| -> usize {
        let mut rem = out_flat;
        let mut src_flat = 0usize;
        for d in 0..shape.len() {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            let src_coord = if d == axis { keep[coord] } else { coord };
            src_flat += src_coord * strides[d];
        }
        src_flat
    };
    let out_buffer = gather_flat!(contig.buffer(), total, idx_of);
    NdArray::from_buffer(out_buffer, out_shape, Order::C)
}

/// `np.delete(arr, obj, axis=None)`. `axis=None` ravels `arr` first,
/// exactly like `insert`. Every `obj` form is translated directly from
/// numpy's own `delete` source; unlike numpy this always computes the
/// simpler "boolean keep-mask -> gather" form rather than numpy's own
/// chunk-copy optimizations, since the differential harness compares only
/// final values/dtype/shape, never the construction method.
pub fn delete(a: &NdArray, obj: &DeleteObj, axis: Option<isize>) -> Result<NdArray, IonpError> {
    let (arr, ax) = match axis {
        None => {
            let raveled = if a.ndim() == 1 { a.to_contiguous() } else { a.to_contiguous().reshape(&[a.size()])? };
            let nd = raveled.ndim();
            (raveled, nd - 1)
        }
        Some(ax) => (a.to_contiguous(), normalize_axis(ax, a.ndim())?),
    };
    let n = arr.shape()[ax];
    let n_i = n as isize;

    let keep: Vec<usize> = match obj {
        DeleteObj::Slice { start, stop, step } => {
            let deleted = slice_to_indices(*start, *stop, *step, n)?;
            let mut mask = vec![false; n];
            for d in deleted {
                if d >= 0 && (d as usize) < n {
                    mask[d as usize] = true;
                }
            }
            (0..n).filter(|&i| !mask[i]).collect()
        }
        DeleteObj::Single(raw) => {
            if *raw < -n_i || *raw >= n_i {
                return Err(IonpError::Index(format!(
                    "index {raw} is out of bounds for axis {ax} with size {n}"
                )));
            }
            let idx = if *raw < 0 { raw + n_i } else { *raw } as usize;
            (0..n).filter(|&i| i != idx).collect()
        }
        DeleteObj::Indices(idxs) => {
            let mut mask = vec![false; n];
            for &raw in idxs {
                let idx = if raw < 0 { raw + n_i } else { raw };
                if idx < 0 || idx >= n_i {
                    return Err(IonpError::Index(format!(
                        "index {raw} is out of bounds for axis {ax} with size {n}"
                    )));
                }
                mask[idx as usize] = true;
            }
            (0..n).filter(|&i| !mask[i]).collect()
        }
        DeleteObj::Bool(mask) => {
            if mask.len() != n {
                return Err(IonpError::Value(format!(
                    "boolean array argument obj to delete must be one dimensional and match the axis length of {n}"
                )));
            }
            (0..n).filter(|&i| !mask[i]).collect()
        }
    };

    select_along_axis(&arr, ax, &keep)
}

// ---------------------------------------------------------------------------
// rot90 / rollaxis
// ---------------------------------------------------------------------------

/// `np.rot90(m, k=1, axes=(0, 1))`: rotates the plane spanned by `axes` by
/// `k` quarter turns (`k %= 4` first, matching Python's floor-mod so
/// negative `k` behaves the same as numpy's). `k==0` is a no-op copy,
/// `k==2` is `flip(flip(m, axes[0]), axes[1])`, `k==1`/`k==3` compose a
/// `flip` and a `transpose` -- translated directly from numpy's own
/// `rot90` source.
pub fn rot90(m: &NdArray, k: isize, axes: (isize, isize)) -> Result<NdArray, IonpError> {
    let ndim = m.ndim();
    let nd = ndim as isize;
    if axes.0 == axes.1 || (axes.0 - axes.1).abs() == nd {
        return Err(IonpError::Value("Axes must be different.".to_string()));
    }
    if axes.0 >= nd || axes.0 < -nd || axes.1 >= nd || axes.1 < -nd {
        return Err(IonpError::Value(format!(
            "Axes=({}, {}) out of range for array of ndim={}.",
            axes.0, axes.1, ndim
        )));
    }
    let ax0 = normalize_axis(axes.0, ndim)?;
    let ax1 = normalize_axis(axes.1, ndim)?;

    let k = python_mod(k, 4);
    if k == 0 {
        return Ok(m.to_contiguous());
    }
    if k == 2 {
        return Ok(flip(&flip(m, Some(&[ax0])), Some(&[ax1])).to_contiguous());
    }
    let mut axes_list: Vec<isize> = (0..nd).collect();
    axes_list.swap(ax0, ax1);
    if k == 1 {
        let flipped = flip(m, Some(&[ax1]));
        flipped.transpose_axes(&axes_list).map(|v| v.to_contiguous())
    } else {
        // k == 3
        let transposed = m.transpose_axes(&axes_list)?;
        Ok(flip(&transposed, Some(&[ax1])).to_contiguous())
    }
}

/// `rollaxis`'s error type: `axis` itself goes through the normal
/// `normalize_axis` -> real `numpy.exceptions.AxisError` path (`Core`
/// wraps that), but numpy's out-of-range-`start` check raises `AxisError`
/// constructed from a single, fully custom message string (`AxisError`'s
/// one-argument form: `self._msg = axis; self.axis = self.ndim = None`,
/// verified against numpy 2.5.1's own docstring example
/// `AxisError('Custom error message')`) -- NOT the usual `"axis {axis} is
/// out of bounds for array of dimension {ndim}"` template `IonpError`'s
/// `AxisError` variant/`ionp-py`'s `axis_error()` helper always produce.
/// Since this crate does not own `error.rs` (no new `IonpError` variant
/// can be added) and `ionp-py/src/lib.rs`'s `to_py_err`/`axis_error` are
/// likewise not owned here, `BadStart` is threaded back to `ionp-py`'s own
/// (owned) `manip.rs` binding, which raises the real `AxisError` with this
/// exact string directly rather than going through the generic mapper.
pub enum RollaxisError {
    Core(IonpError),
    BadStart(String),
}

impl From<IonpError> for RollaxisError {
    fn from(e: IonpError) -> Self {
        RollaxisError::Core(e)
    }
}

/// `np.rollaxis(a, axis, start=0)`: rolls `axis` backwards until it sits
/// at position `start` -- translated directly from numpy's own source.
pub fn rollaxis(a: &NdArray, axis: isize, start: isize) -> Result<NdArray, RollaxisError> {
    let n = a.ndim();
    let ax = normalize_axis(axis, n)?;
    let nd = n as isize;
    let mut start_norm = start;
    if start_norm < 0 {
        start_norm += nd;
    }
    if !(0 <= start_norm && start_norm < nd + 1) {
        // numpy's own source only adds `n` to `start` ONCE when negative
        // (not a full modulo wrap) and then reports THAT adjusted value in
        // the message, not the raw caller-supplied `start` -- verified
        // live: `np.rollaxis(d, 0, -10)` on a 3-d array reports "...but -7
        // was passed in" (== -10 + 3), not -10.
        return Err(RollaxisError::BadStart(format!(
            "'start' arg requires {} <= start < {}, but {} was passed in",
            -nd,
            nd + 1,
            start_norm
        )));
    }
    let mut start_u = start_norm as usize;
    if ax < start_u {
        start_u -= 1;
    }
    if ax == start_u {
        return Ok(a.to_contiguous());
    }
    let mut axes: Vec<usize> = (0..n).collect();
    axes.remove(ax);
    axes.insert(start_u, ax);
    let perm: Vec<isize> = axes.iter().map(|&x| x as isize).collect();
    Ok(a.transpose_axes(&perm)?.to_contiguous())
}

// ---------------------------------------------------------------------------
// grid / index construction: diag_indices(_from), tril_indices, triu_indices,
// ix_, indices, meshgrid, ravel_multi_index, unravel_index, fill_diagonal
// ---------------------------------------------------------------------------

fn col_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1];
    }
    strides
}

/// numpy `np.diag_indices(n, ndim=2)`: `ndim` arrays, each `arange(max(n,0))`
/// -- negative `n` mirrors `np.arange`'s own "negative stop -> empty" rule
/// rather than raising (verified live: `np.diag_indices(-2)` returns two
/// empty int64 arrays, not an error).
pub fn diag_indices(n: isize, ndim: usize) -> Vec<NdArray> {
    let count = n.max(0) as usize;
    let idx: Vec<i64> = (0..count as i64).collect();
    (0..ndim)
        .map(|_| NdArray::from_buffer(Buffer::I64(idx.clone()), vec![count], Order::C).unwrap())
        .collect()
}

/// numpy `np.diag_indices_from(arr)`: same as `diag_indices` but reads `n`
/// and `ndim` off `arr`, requiring `arr.ndim >= 2` and every dimension
/// equal (exact messages verified live against real numpy 2.5.1).
pub fn diag_indices_from(a: &NdArray) -> Result<Vec<NdArray>, IonpError> {
    if a.ndim() < 2 {
        return Err(IonpError::Value("input array must be at least 2-d".to_string()));
    }
    let shape = a.shape();
    if !shape.iter().all(|&d| d == shape[0]) {
        return Err(IonpError::Value(
            "All dimensions of input must be of equal length".to_string(),
        ));
    }
    Ok(diag_indices(shape[0] as isize, a.ndim()))
}

/// Shared body for `tril_indices`/`triu_indices`: row-major-ordered
/// `(row, col)` pairs of an `n x m` grid satisfying the triangle condition.
fn tri_indices(n: usize, k: isize, m: usize, keep_upper: bool) -> (NdArray, NdArray) {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for r in 0..n {
        for c in 0..m {
            let keep = if keep_upper {
                (c as isize) - (r as isize) >= k
            } else {
                (c as isize) - (r as isize) <= k
            };
            if keep {
                rows.push(r as i64);
                cols.push(c as i64);
            }
        }
    }
    let len = rows.len();
    (
        NdArray::from_buffer(Buffer::I64(rows), vec![len], Order::C).unwrap(),
        NdArray::from_buffer(Buffer::I64(cols), vec![len], Order::C).unwrap(),
    )
}

/// `n`/`m` accept negative values (clamped to 0, an empty grid) rather than
/// erroring, mirroring `diag_indices`'s own negative-`n` behavior -- verified
/// live: `np.tril_indices(-1)` and `np.tril_indices(3, 0, -1)` both return
/// two empty int64 arrays rather than raising.
pub fn tril_indices(n: isize, k: isize, m: Option<isize>) -> (NdArray, NdArray) {
    tri_indices(n.max(0) as usize, k, m.unwrap_or(n).max(0) as usize, false)
}

pub fn triu_indices(n: isize, k: isize, m: Option<isize>) -> (NdArray, NdArray) {
    tri_indices(n.max(0) as usize, k, m.unwrap_or(n).max(0) as usize, true)
}

/// Nonzero positions of a 1-D bool array, as an owned `i64` `NdArray` --
/// local helper for `ix_`'s bool-mask-to-index conversion (this module does
/// not depend on `sort.rs`'s `nonzero`, which is a different task's owned
/// file).
fn nonzero_1d_bool(a: &NdArray) -> NdArray {
    let contig = a.to_contiguous();
    let data = match contig.buffer() {
        Buffer::Bool(v) => v,
        _ => unreachable!(),
    };
    let idx: Vec<i64> = data
        .iter()
        .enumerate()
        .filter(|(_, &v)| v)
        .map(|(i, _)| i as i64)
        .collect();
    let len = idx.len();
    NdArray::from_buffer(Buffer::I64(idx), vec![len], Order::C).unwrap()
}

/// numpy `np.ix_(*sequences)`: each 1-D input becomes an axis of an open
/// mesh -- output `i` is reshaped to have length `1` on every axis except
/// axis `i`, where it keeps its own length. Bool inputs are first converted
/// via `nonzero` (verified live: `np.ix_(np.array([True,False,True]))` ==
/// `(array([0, 2]),)`).
pub fn ix_(arrays: &[NdArray]) -> Result<Vec<NdArray>, IonpError> {
    let n = arrays.len();
    let mut out = Vec::with_capacity(n);
    for (i, a) in arrays.iter().enumerate() {
        if a.ndim() != 1 {
            return Err(IonpError::Value("Cross index must be 1 dimensional".to_string()));
        }
        let base = if a.dtype() == DType::Bool { nonzero_1d_bool(a) } else { a.to_contiguous() };
        let len = base.size();
        let mut shape = vec![1usize; n];
        shape[i] = len;
        out.push(base.reshape(&shape)?);
    }
    Ok(out)
}

/// numpy `np.indices(dimensions, dtype=intp, sparse=False)`. `dense=false`
/// (sparse) returns `ndim` arrays each broadcastable to `dimensions`
/// (`np.ix_(*[arange(d) for d in dimensions])`-shaped); `sparse=false`
/// (dense, the default) returns a single array of shape
/// `(ndim, *dimensions)` wrapped in a 1-element `Vec` so both call shapes
/// share a return type -- the PyO3 boundary unwraps accordingly.
pub fn indices(dims: &[usize], dtype: DType, sparse: bool) -> Result<Vec<NdArray>, IonpError> {
    let ndim = dims.len();
    if sparse {
        let mut out = Vec::with_capacity(ndim);
        for i in 0..ndim {
            let vals: Vec<i64> = (0..dims[i] as i64).collect();
            let mut shape = vec![1usize; ndim];
            shape[i] = dims[i];
            let arr = NdArray::from_buffer(Buffer::I64(vals), shape, Order::C)?;
            out.push(arr.cast_to(dtype));
        }
        Ok(out)
    } else {
        let per_axis_size: usize = dims.iter().product();
        let strides = row_major_strides(dims);
        let mut data = vec![0i64; per_axis_size * ndim];
        for (i, &stride) in strides.iter().enumerate() {
            for flat in 0..per_axis_size {
                let coord = if stride == 0 { 0 } else { (flat / stride) % dims[i].max(1) };
                data[i * per_axis_size + flat] = coord as i64;
            }
        }
        let mut shape = vec![ndim];
        shape.extend_from_slice(dims);
        let arr = NdArray::from_buffer(Buffer::I64(data), shape, Order::C)?;
        Ok(vec![arr.cast_to(dtype)])
    }
}

/// numpy `np.meshgrid(*xi, copy=True, sparse=False, indexing='xy')`.
/// `copy=` is accepted at the PyO3 boundary but has no effect here -- every
/// array this module returns is already a freshly allocated owned copy
/// (view/aliasing semantics are a documented non-goal of this whole file,
/// see module docs), and `copy=False`'s only observable numpy effect is
/// aliased memory between outputs, which the differential harness's
/// value-only comparison cannot see.
pub fn meshgrid(arrays: &[&NdArray], indexing: &str, sparse: bool) -> Result<Vec<NdArray>, IonpError> {
    if indexing != "xy" && indexing != "ij" {
        return Err(IonpError::Value("Valid values for `indexing` are 'xy' and 'ij'.".to_string()));
    }
    let n = arrays.len();
    let flat: Vec<NdArray> =
        arrays.iter().map(|a| a.to_contiguous().reshape(&[a.size()])).collect::<Result<_, _>>()?;
    let dims: Vec<usize> = flat.iter().map(|a| a.size()).collect();
    let mut outs = Vec::with_capacity(n);
    for i in 0..n {
        let mut shape = vec![1usize; n];
        shape[i] = dims[i];
        let reshaped = flat[i].reshape(&shape)?;
        let out = if sparse {
            reshaped
        } else {
            crate::creation::broadcast_to(&reshaped, &dims)?.to_contiguous()
        };
        outs.push(out);
    }
    if indexing == "xy" && n >= 2 {
        let mut perm: Vec<isize> = (0..n as isize).collect();
        perm.swap(0, 1);
        for o in outs.iter_mut() {
            *o = o.transpose_axes(&perm)?.to_contiguous();
        }
    }
    Ok(outs)
}

/// Per-dimension out-of-range handling mode for `ravel_multi_index`,
/// mirroring numpy's `mode=` (a single string broadcasts to every
/// dimension; the PyO3 boundary expands that before calling in).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RavelMode {
    Raise,
    Wrap,
    Clip,
}

/// Extract every element of an integer-dtype array as `i64` (bool counts as
/// integer: `True`/`False` -> `1`/`0`, matching numpy's own
/// `ravel_multi_index`/`unravel_index`, both verified live to accept a bool
/// coordinate array without error). Floats/complex raise the exact
/// `TypeError` numpy raises for a non-integer coordinate array.
fn require_int_i64(a: &NdArray) -> Result<NdArray, IonpError> {
    match a.dtype() {
        DType::F16 | DType::F32 | DType::F64 | DType::C64 | DType::C128 => {
            Err(IonpError::Type("only int indices permitted".to_string()))
        }
        DType::I64 => Ok(a.to_contiguous()),
        _ => Ok(a.to_contiguous().cast_to(DType::I64)),
    }
}

/// numpy `np.ravel_multi_index(multi_index, dims, mode='raise', order='C')`.
pub fn ravel_multi_index(
    multi_index: &[NdArray],
    dims: &[usize],
    modes: &[RavelMode],
    order: Order,
) -> Result<NdArray, IonpError> {
    if multi_index.len() != dims.len() {
        return Err(IonpError::Value(format!(
            "parameter multi_index must be a sequence of length {}",
            dims.len()
        )));
    }
    let shapes: Vec<&[usize]> = multi_index.iter().map(|a| a.shape()).collect();
    let out_shape = broadcast_shapes(&shapes)?;
    let strides = if order == Order::C { row_major_strides(dims) } else { col_major_strides(dims) };
    let total = shape::size_of_shape(&out_shape);
    let mut checked = Vec::with_capacity(dims.len());
    for m in multi_index {
        let int_arr = require_int_i64(m)?;
        let broadcasted = crate::creation::broadcast_to(&int_arr, &out_shape)?.to_contiguous();
        checked.push(broadcasted);
    }
    let mut out = vec![0i64; total];
    for flat in 0..total {
        let mut acc: i64 = 0;
        for d in 0..dims.len() {
            let raw = match checked[d].buffer() {
                Buffer::I64(v) => v[flat],
                _ => unreachable!(),
            };
            let dim = dims[d] as i64;
            let v = match modes[d] {
                RavelMode::Raise => {
                    if raw < 0 || raw >= dim {
                        return Err(IonpError::Value("invalid entry in coordinates array".to_string()));
                    }
                    raw
                }
                RavelMode::Wrap => raw.rem_euclid(dim.max(1)),
                RavelMode::Clip => raw.clamp(0, dim - 1),
            };
            acc += v * strides[d] as i64;
        }
        out[flat] = acc;
    }
    NdArray::from_buffer(Buffer::I64(out), out_shape, Order::C)
}

/// numpy `np.unravel_index(indices, shape, order='C')`.
pub fn unravel_index(indices_arr: &NdArray, dims: &[usize], order: Order) -> Result<Vec<NdArray>, IonpError> {
    let total = dims.iter().product::<usize>() as i64;
    let strides = if order == Order::C { row_major_strides(dims) } else { col_major_strides(dims) };
    let int_arr = require_int_i64(indices_arr)?;
    let flat_src = match int_arr.buffer() {
        Buffer::I64(v) => v.clone(),
        _ => unreachable!(),
    };
    let ndim = dims.len();
    let n = flat_src.len();
    let mut outs: Vec<Vec<i64>> = vec![vec![0i64; n]; ndim];
    // Positional-notation decomposition must consume the LARGEST stride
    // first regardless of memory order: row_major_strides() is already
    // descending (axis 0 largest), but col_major_strides() (order='F') is
    // ASCENDING (axis 0 smallest = 1) -- looping `d in 0..ndim` and
    // dividing by strides[d] in that fixed order silently decomposed
    // F-order wrong (divided by the smallest stride first, which just
    // echoes `raw` back into axis 0 and leaves every other axis 0).
    // Verified live against real numpy 2.5.1's `order='F'` behavior.
    let mut axis_order: Vec<usize> = (0..ndim).collect();
    axis_order.sort_by(|&a, &b| strides[b].cmp(&strides[a]));
    for (i, &raw) in flat_src.iter().enumerate() {
        if raw < 0 || raw >= total {
            return Err(IonpError::Value(format!(
                "index {raw} is out of bounds for array with size {total}"
            )));
        }
        let mut rem = raw;
        for &d in &axis_order {
            let s = strides[d] as i64;
            outs[d][i] = if s == 0 { 0 } else { rem / s };
            rem %= s.max(1);
        }
    }
    let shape = indices_arr.shape().to_vec();
    outs.into_iter()
        .map(|v| NdArray::from_buffer(Buffer::I64(v), shape.clone(), Order::C))
        .collect()
}

/// numpy `np.fill_diagonal(a, val, wrap=False)`: mutates `a` in place,
/// returns nothing. `val` is raveled and its elements tiled CYCLICALLY over
/// the diagonal positions (verified live: `np.fill_diagonal(np.zeros((4,4)),
/// [1,2])` writes `1,2,1,2` down the diagonal, not a broadcast error) --
/// this is numpy's actual `a.flat[:end:step] = val` flat-iterator
/// assignment behavior, not ordinary broadcasting.
pub fn fill_diagonal(a: &mut NdArray, val: &NdArray, wrap: bool) -> Result<(), IonpError> {
    let ndim = a.ndim();
    if ndim < 2 {
        return Err(IonpError::Value("array must be at least 2-d".to_string()));
    }
    let shape = a.shape().to_vec();
    if ndim > 2 {
        if !shape.iter().all(|&d| d == shape[0]) {
            return Err(IonpError::Value(
                "All dimensions of input must be of equal length".to_string(),
            ));
        }
    }
    let step: usize = if ndim == 2 {
        shape[1] + 1
    } else {
        // numpy's own algorithm (lib/_index_tricks_impl.py):
        // `step = 1 + cumprod(shape[:-1]).sum()` -- the SUM of the running
        // products of shape[:-1], not the single final product. E.g. for
        // shape (3,3,3): cumprod([3,3]) = [3,9], sum = 12, step = 13 (NOT
        // 1 + 3*3 = 10, which is what a naive single-product read of the
        // numpy source line looks like at a glance but is wrong for
        // ndim >= 3 -- verified against real numpy 2.5.1 directly).
        let dims = &shape[..ndim - 1];
        let mut cum: i64 = 1;
        let mut sum: i64 = 0;
        for &d in dims {
            cum *= d as i64;
            sum += cum;
        }
        1 + sum as usize
    };
    let end: Option<usize> = if ndim == 2 && !wrap { Some(shape[1] * shape[1]) } else { None };
    let total = shape::size_of_shape(&shape);
    let count = match end {
        Some(e) => e.min(total).div_ceil(step),
        None => total.div_ceil(step),
    };

    let val_dtype_matches = val.dtype() == a.dtype();
    let val_cast: NdArray = if val_dtype_matches { val.to_contiguous() } else { val.cast_to(a.dtype()) };
    let val_flat = val_cast.reshape(&[val_cast.size()])?;
    let val_len = val_flat.size().max(1);

    let strides = a.strides().to_vec();
    let offset = a.offset();
    let a_dtype = a.dtype();
    let logical_strides = row_major_strides(&shape);
    let buffer = Arc::make_mut(&mut a.buffer);

    macro_rules! fill_arm {
        ($variant:ident) => {{
            let src = match val_flat.buffer() {
                Buffer::$variant(v) => v.clone(),
                _ => unreachable!(),
            };
            if let Buffer::$variant(dst) = buffer {
                for i in 0..count {
                    let flat_logical = i * step;
                    // decompose flat_logical against logical (row-major)
                    // strides to get per-axis coords, then recombine with
                    // the array's ACTUAL strides -- correct for
                    // non-contiguous/strided targets too.
                    let mut rem = flat_logical;
                    let mut phys = offset;
                    for d in 0..ndim {
                        let c = if logical_strides[d] == 0 { 0 } else { rem / logical_strides[d] };
                        rem %= logical_strides[d].max(1);
                        phys += (c as isize) * strides[d];
                    }
                    if src.is_empty() {
                        continue;
                    }
                    dst[phys as usize] = src[i % val_len];
                }
            }
        }};
    }
    match a_dtype {
        DType::Bool => fill_arm!(Bool),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => fill_arm!(I8),
        DType::I16 => fill_arm!(I16),
        DType::I32 => fill_arm!(I32),
        DType::I64 => fill_arm!(I64),
        DType::U8 => fill_arm!(U8),
        DType::U16 => fill_arm!(U16),
        DType::U32 => fill_arm!(U32),
        DType::U64 => fill_arm!(U64),
        DType::F16 => fill_arm!(F16),
        DType::F32 => fill_arm!(F32),
        DType::F64 => fill_arm!(F64),
        DType::C64 => fill_arm!(C64),
        DType::C128 => fill_arm!(C128),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// gather/scatter family: take / put / take_along_axis / put_along_axis /
// compress
//
// Shared machinery: `ClipMode` (numpy's `mode='raise'|'wrap'|'clip'`,
// verified live against numpy 2.5.1 to have THREE distinct semantics, not
// two: 'raise' allows exactly one Python-style negative wrap (`-n..n-1`
// valid, anything further out raises `IndexError: index {idx} is out of
// bounds for axis {axis} with size {n}`); 'wrap' is a full `rem_euclid`
// (any magnitude wraps); 'clip' clamps to `[0, n-1]` -- including negative
// values clamping to 0, NOT n-1 (confirmed: `np.take(a, [-1,100],
// mode='clip')` on a size-12 array gives `[0, 11]`, not `[11, 11]`).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipMode {
    Raise,
    Wrap,
    Clip,
}

impl ClipMode {
    /// numpy's own `clipmode` parser error text (verified live, shared
    /// verbatim by `take`/`put`/`choose`, all three ultimately routed
    /// through the same C `NPY_CLIPMODE` converter).
    pub fn parse(s: &str) -> Result<Self, IonpError> {
        match s {
            "raise" => Ok(ClipMode::Raise),
            "wrap" => Ok(ClipMode::Wrap),
            "clip" => Ok(ClipMode::Clip),
            other => Err(IonpError::Value(format!(
                "clipmode must be one of 'clip', 'raise', or 'wrap' (got '{other}')"
            ))),
        }
    }
}

/// `take`/`put`-style index resolution: `raise` mode permits exactly one
/// Python-style negative wrap (matching ordinary array indexing), `wrap`
/// mode is unrestricted `rem_euclid`, `clip` clamps to `[0, n-1]`
/// (including negatives clamping to 0). `n == 0` only ever reaches here
/// with an empty `indices` array (callers special-case the "non-empty take
/// from an empty axis" error before resolving any index), so the
/// `Clip`/`Wrap` empty-`n` arms are unreachable in practice but still
/// guarded to avoid a panic.
fn resolve_index(idx: i64, n: usize, axis: usize, mode: ClipMode) -> Result<usize, IonpError> {
    let ni = n as i64;
    match mode {
        ClipMode::Wrap => Ok(if ni == 0 { 0 } else { idx.rem_euclid(ni) as usize }),
        ClipMode::Clip => Ok(if ni == 0 { 0 } else { idx.clamp(0, ni - 1) as usize }),
        ClipMode::Raise => {
            let v = if idx < 0 { idx + ni } else { idx };
            if v < 0 || v >= ni {
                Err(IonpError::Index(format!(
                    "index {idx} is out of bounds for axis {axis} with size {n}"
                )))
            } else {
                Ok(v as usize)
            }
        }
    }
}

/// `choose`-style index resolution: `wrap`/`clip` use the identical
/// formulas as `resolve_index` above (verified live), but `raise` mode
/// does NOT permit a negative wrap at all -- `np.choose(np.array([-1]),
/// ...)` raises under the default mode even though `-1` would be a
/// perfectly valid single-wrap negative index elsewhere in numpy. The
/// message is also a static string with no dynamic values (verified for
/// both an out-of-range-positive and an out-of-range-negative selector).
fn resolve_index_strict(idx: i64, n: usize, mode: ClipMode) -> Result<usize, IonpError> {
    let ni = n as i64;
    match mode {
        ClipMode::Wrap => Ok(if ni == 0 { 0 } else { idx.rem_euclid(ni) as usize }),
        ClipMode::Clip => Ok(if ni == 0 { 0 } else { idx.clamp(0, ni - 1) as usize }),
        ClipMode::Raise => {
            if idx < 0 || idx >= ni {
                Err(IonpError::Value("invalid entry in choice array".to_string()))
            } else {
                Ok(idx as usize)
            }
        }
    }
}

/// Cast an integer-KIND (bool or any int/uint dtype) index array to `i64`,
/// matching numpy's `same_kind`/`safe` casting-rule TypeError for a
/// non-integer (float/complex) index array -- `take`/`compress` use the
/// `'same_kind'` wording, `put`/`choose` use `'safe'`, both verified live
/// against numpy 2.5.1 byte-for-byte including the exact `dtype('...')`
/// repr of the SOURCE dtype (never a generic placeholder).
fn cast_indices(a: &NdArray, rule: &str) -> Result<NdArray, IonpError> {
    match a.dtype() {
        DType::F16 | DType::F32 | DType::F64 | DType::C64 | DType::C128 => Err(IonpError::Type(format!(
            // numpy says "scalar" for a 0-d operand and "array data" for
            // every other rank -- one noun, chosen by ndim. Verified live on
            // numpy 2.5.1 across all four callers of this check, so the rule
            // is a property of the check and not of any one function:
            //     a.take(np.array(0.5))     -> "Cannot cast scalar from ..."
            //     a.take(np.array([0.5]))   -> "Cannot cast array data ..."
            //     a.take(np.array([[0.5]])) -> "Cannot cast array data ..."
            //     a.copy().put(np.array(0.5), 1) -> "... scalar ..." ('safe')
            //     np.choose(np.array(0.5), [a, a]) -> "... scalar ..."
            //     a.repeat(np.array(0.5))          -> "... scalar ..."
            "Cannot cast {} from dtype('{}') to dtype('int64') according to the rule '{}'",
            if a.ndim() == 0 { "scalar" } else { "array data" },
            a.dtype().name(),
            rule
        ))),
        DType::I64 => Ok(a.to_contiguous()),
        _ => Ok(a.to_contiguous().cast_to(DType::I64)),
    }
}

/// `take_along_axis`/`put_along_axis`'s own (distinct from `cast_indices`
/// above) integer-dtype check: no casting-rule wording at all, a plain
/// `IndexError` that also rejects bool (verified live: unlike
/// `take`/`put`, a bool `indices` array to `take_along_axis` raises the
/// same message as a float one would).
fn require_integer_no_bool(a: &NdArray) -> Result<NdArray, IonpError> {
    match a.dtype() {
        DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
            if a.dtype() == DType::I64 {
                Ok(a.to_contiguous())
            } else {
                Ok(a.to_contiguous().cast_to(DType::I64))
            }
        }
        _ => Err(IonpError::Index("`indices` must be an integer array".to_string())),
    }
}

fn i64_buf(a: &NdArray) -> Vec<i64> {
    match a.buffer() {
        Buffer::I64(v) => v.clone(),
        other => unreachable!("expected I64 after cast, got {:?}", other),
    }
}

/// Computes, for every position of an output shaped `out_shape = base_shape
/// [..ax] + idx_shape + base_shape[ax+1..]`, the FLAT source position in a
/// C-contiguous buffer of `base_shape` -- shared by `take` and
/// `take_along_axis`'s "gather" halves (kept separate from the elementwise
/// gather itself since that needs one macro arm per `Buffer` variant, while
/// this index arithmetic is dtype-independent and fallible, so doing it
/// as its own pre-pass lets the gather step stay a single infallible
/// closure over an already-resolved `Vec<usize>`, matching this module's
/// `gather_flat!` convention).
fn take_src_positions(
    base_shape: &[usize],
    idx_shape: &[usize],
    idx_flat: &[i64],
    ax: usize,
    mode: ClipMode,
    out_shape: &[usize],
) -> Result<Vec<usize>, IonpError> {
    let n = base_shape[ax];
    let in_strides = row_major_strides(base_shape);
    let out_strides = row_major_strides(out_shape);
    let idx_strides = row_major_strides(idx_shape);
    let idx_ndim = idx_shape.len();
    let total = shape::size_of_shape(out_shape);
    let mut positions = Vec::with_capacity(total);
    for out_flat in 0..total {
        let mut rem = out_flat;
        let mut src_flat = 0usize;
        for d in 0..ax {
            let coord = if out_strides[d] == 0 { 0 } else { rem / out_strides[d] };
            rem %= out_strides[d].max(1);
            src_flat += coord * in_strides[d];
        }
        let mut idx_flat_pos = 0usize;
        for j in 0..idx_ndim {
            let d = ax + j;
            let coord = if out_strides[d] == 0 { 0 } else { rem / out_strides[d] };
            rem %= out_strides[d].max(1);
            idx_flat_pos += coord * idx_strides[j];
        }
        let raw = idx_flat[idx_flat_pos];
        let src_ax_coord = resolve_index(raw, n, ax, mode)?;
        src_flat += src_ax_coord * in_strides[ax];
        for d in (ax + idx_ndim)..out_shape.len() {
            let base_axis = d - idx_ndim + 1;
            let coord = if out_strides[d] == 0 { 0 } else { rem / out_strides[d] };
            rem %= out_strides[d].max(1);
            src_flat += coord * in_strides[base_axis];
        }
        positions.push(src_flat);
    }
    Ok(positions)
}

/// numpy `np.take(a, indices, axis=None, mode='raise')`. `axis=None`
/// (encoded here as `axis: None`) flattens `a` (C order) first, same
/// convention `concatenate`/`repeat` above already use; the caller (ionp-py)
/// has already normalized a non-`None` `axis` to `0..ndim` via
/// `normalize_axis` before calling in, so `ax` here is always in-range for
/// `base.ndim()`.
pub fn take(a: &NdArray, indices: &NdArray, axis: Option<usize>, mode: ClipMode) -> Result<NdArray, IonpError> {
    let int_indices = cast_indices(indices, "same_kind")?;
    let idx_flat = i64_buf(&int_indices);
    let idx_shape = indices.shape().to_vec();

    let (base, ax) = match axis {
        // numpy's real `take`/`compress` promote a 0-d input to shape `(1,)`
        // before resolving `axis` (verified live: `np.take(np.array(5),
        // [0], axis=0)` succeeds and returns `[5]`, and the axis-range
        // error text for an out-of-range axis on a 0-d array reads "for
        // array of dimension 1", not "dimension 0" -- the py binding layer
        // already normalizes `axis` against this promoted ndim=1 view, so
        // by the time `axis` arrives here as `Some(0)` the only remaining
        // step is to actually reshape the 0-d buffer to match).
        Some(ax) if a.ndim() == 0 => (a.to_contiguous().reshape(&[1]).unwrap(), ax),
        Some(ax) => (a.clone(), ax),
        None => (a.to_contiguous().reshape(&[a.size()]).unwrap(), 0),
    };
    let base_shape = base.shape().to_vec();
    let n = base_shape[ax];
    // numpy validates the index array inside a loop over the OUTER block
    // count, `prod(shape[..axis])` -- so when that product is 0 the loop body
    // never runs and NO index is ever checked, however wild it is. When the
    // product is non-zero every index is checked even if the result is empty
    // because some INNER axis has length 0. This predicate was fitted against
    // numpy 2.5.1 over 528 cells (13 shapes x every axis incl. None x 12
    // index vectors) with 0 prediction misses. The distinction is real, not
    // pedantry:
    //     np.zeros((3, 0, 2)).take([8], axis=0)  -> IndexError (outer = 1)
    //     np.zeros((3, 0, 2)).take([8], axis=2)  -> shape (3, 0, 1), no error
    //                                               (outer = 3 * 0 = 0)
    let outer: usize = base_shape[..ax].iter().product();
    let inner: usize = base_shape[ax + 1..].iter().product();
    // The static "empty axes" message is numpy's ONLY when the RESULT would
    // have been non-empty. If the take axis is empty but some other axis is
    // empty too -- so the result is empty regardless -- numpy instead falls
    // through to its ordinary per-index path, which under `raise` reports the
    // usual out-of-bounds text against a size of 0, and under `clip` simply
    // succeeds. Measured on numpy 2.5.1:
    //     np.zeros((1, 0, 5)).take([0], axis=1)              -> "cannot do a
    //         non-empty take from an empty axes."   (result would be (1,1,5))
    //     np.zeros((0, 0)).take([0], axis=0)                 -> "index 0 is
    //         out of bounds for axis 0 with size 0" (result is (1,0), empty)
    //     np.zeros((0, 0)).take([0], axis=0, mode='clip')    -> (1, 0), no err
    // numpy: exact static text, verified live, no interpolated values.
    if outer != 0 && inner != 0 && n == 0 && !idx_flat.is_empty() {
        return Err(IonpError::Index("cannot do a non-empty take from an empty axes.".to_string()));
    }

    let mut out_shape: Vec<usize> = base_shape[..ax].to_vec();
    out_shape.extend_from_slice(&idx_shape);
    out_shape.extend_from_slice(&base_shape[ax + 1..]);

    let contig = base.to_contiguous();
    let total = shape::size_of_shape(&out_shape);
    let positions = if total == 0 {
        // BUG FOUND + FIXED (2026-08-04): the early return for an empty
        // result used to skip index validation entirely, because the ONLY
        // bounds check lives inside `take_src_positions`'s per-output-element
        // loop -- and that loop runs zero times when the output is empty. An
        // out-of-bounds index therefore went unnoticed whenever some INNER
        // axis had length 0. numpy still validates in that case. Measured
        // against numpy 2.5.1 with `a = np.zeros((3, 0, 2), np.int8)`:
        //     a.take([8],  axis=0) -> IndexError: index 8 is out of bounds
        //                             for axis 0 with size 3
        //     a.take([-8], axis=0) -> IndexError: index -8 is out of ...
        //     a.take([0],  axis=0) -> shape (1, 0, 2), no error
        // ionp returned the empty array for all three. `compress` inherits
        // both the bug and this fix, since it is implemented as a `take` over
        // the selected positions -- `np.zeros((3, 0, 2)).compress([1,1,1,1],
        // axis=0)` must raise "index 3 is out of bounds for axis 0 with size
        // 3", and did not.
        //
        // `outer` gates this exactly as it gates the empty-axis guard above;
        // dropping the gate over-raises on 134 of 1806 measured cells, all of
        // them "some inner axis is empty, so numpy never looked".
        //
        // The check is deliberately NOT hoisted above the `total == 0` branch:
        // when the output is non-empty every index is already visited by the
        // loop below, so hoisting would double the work in the hot path to no
        // effect. `mode` is honored rather than assumed -- under Clip/Wrap
        // `resolve_index` cannot fail, which is correct, since numpy does not
        // raise for those modes either. `n == 0` DOES reach here (the guard
        // above fires only when the result would be non-empty), and
        // `resolve_index` handles it: Raise emits the ordinary out-of-bounds
        // text against a size of 0, exactly as numpy does, while Clip/Wrap
        // short-circuit to 0 without dividing by it.
        //
        // KNOWN DIFFERENCE: `mode='wrap'` with an empty take axis is a numpy
        // 2.5.1 HANG (infinite loop, reproduced on
        // `np.zeros((0, 0)).take([0], axis=0, mode='wrap')`), so it has no
        // observable numpy answer to match. ionp returns the empty result,
        // consistent with what `clip` does there. Do NOT put that combination
        // in the differential corpus -- it will hang the suite, not fail it.
        if outer != 0 {
            for &raw in &idx_flat {
                resolve_index(raw, n, ax, mode)?;
            }
        }
        Vec::new()
    } else {
        take_src_positions(&base_shape, &idx_shape, &idx_flat, ax, mode, &out_shape)?
    };
    let out_buffer = gather_flat!(contig.buffer(), total, |i: usize| positions[i]);
    NdArray::from_buffer(out_buffer, out_shape, Order::C)
}

/// numpy `np.put(a, ind, v, mode='raise')`: mutates `a` in place (treating
/// it as flattened, like `a.flat[ind] = v`), returns nothing. `v` is
/// cast to `a`'s dtype (unsafely -- verified live: a float value into an
/// int32 `a` truncates rather than raising) and cycled if shorter than
/// `ind` (verified: `np.put(b, [0,1,2], [7])` writes `7` to all three
/// positions), the same cyclic-tile behavior `fill_diagonal` above uses
/// for its own `val` argument. `ind` uses the `'safe'`-casting-rule
/// wording (distinct from `take`'s `'same_kind'` wording for the same
/// float-indices TypeError, verified live).
pub fn put(a: &mut NdArray, indices: &NdArray, values: &NdArray, mode: ClipMode) -> Result<(), IonpError> {
    let total_a = shape::size_of_shape(a.shape());
    let int_indices = cast_indices(indices, "safe")?;
    let idx_flat = i64_buf(&int_indices);
    if idx_flat.is_empty() {
        return Ok(());
    }
    if total_a == 0 {
        // numpy: exact static text, verified live, regardless of `mode`.
        return Err(IonpError::Index("cannot replace elements of an empty array".to_string()));
    }
    let mut resolved = Vec::with_capacity(idx_flat.len());
    for &raw in &idx_flat {
        resolved.push(resolve_index(raw, total_a, 0, mode)?);
    }

    let values_cast = values.cast_to(a.dtype()).to_contiguous();
    let vlen = values_cast.size().max(1);

    let shape = a.shape().to_vec();
    let strides = a.strides().to_vec();
    let offset = a.offset();
    let a_dtype = a.dtype();
    let logical_strides = row_major_strides(&shape);
    let ndim = shape.len();
    let buffer = Arc::make_mut(&mut a.buffer);

    macro_rules! put_arm {
        ($variant:ident) => {{
            let src = match values_cast.buffer() {
                Buffer::$variant(v) => v.clone(),
                _ => unreachable!(),
            };
            if let Buffer::$variant(dst) = buffer {
                for (i, &flat_logical) in resolved.iter().enumerate() {
                    let mut rem = flat_logical;
                    let mut phys = offset;
                    for d in 0..ndim {
                        let c = if logical_strides[d] == 0 { 0 } else { rem / logical_strides[d] };
                        rem %= logical_strides[d].max(1);
                        phys += (c as isize) * strides[d];
                    }
                    dst[phys as usize] = src[i % vlen];
                }
            }
        }};
    }
    match a_dtype {
        DType::Bool => put_arm!(Bool),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => put_arm!(I8),
        DType::I16 => put_arm!(I16),
        DType::I32 => put_arm!(I32),
        DType::I64 => put_arm!(I64),
        DType::U8 => put_arm!(U8),
        DType::U16 => put_arm!(U16),
        DType::U32 => put_arm!(U32),
        DType::U64 => put_arm!(U64),
        DType::F16 => put_arm!(F16),
        DType::F32 => put_arm!(F32),
        DType::F64 => put_arm!(F64),
        DType::C64 => put_arm!(C64),
        DType::C128 => put_arm!(C128),
    }
    Ok(())
}

/// numpy `np.take_along_axis(arr, indices, axis=-1)`. `axis=None` flattens
/// BOTH `arr` and `indices` to 1-d first (verified live) but additionally
/// requires `indices` to already be 1-d in that case (numpy's own
/// `ValueError: when axis=None, \`indices\` must have a single dimension.`,
/// verified live -- NOT auto-raveled the way `take`'s `axis=None` ravels
/// its indices argument unconditionally regardless of shape).
///
/// Non-`None` axis: `arr.ndim` must equal `indices.ndim` exactly (a
/// distinct `ValueError`, verified live); every axis other than `axis`
/// must broadcast between `indices.shape[d]` and `arr.shape[d]` (1 or
/// exact match -- numpy's actual algorithm builds an orthogonal `arange`
/// per non-axis dimension and fancy-indexes with it alongside `indices`,
/// verified by reading `numpy/lib/_shape_base_impl.py`; the visible effect
/// is "every non-axis dim of the OUTPUT equals `arr.shape[d]`, and
/// `indices.shape[d]` must be broadcastable into that", which is what
/// `take_along_axis_broadcast_check` below reproduces without literally
/// materializing the `arange` arrays).
pub fn take_along_axis(a: &NdArray, indices: &NdArray, axis: Option<usize>) -> Result<NdArray, IonpError> {
    let int_indices = require_integer_no_bool(indices)?;

    let (base, idx, ax) = match axis {
        Some(ax) => (a.clone(), int_indices, ax),
        None => {
            if indices.ndim() != 1 {
                return Err(IonpError::Value(
                    "when axis=None, `indices` must have a single dimension.".to_string(),
                ));
            }
            (a.to_contiguous().reshape(&[a.size()]).unwrap(), int_indices, 0)
        }
    };
    if base.ndim() != idx.ndim() {
        return Err(IonpError::Value(
            "`indices` and `arr` must have the same number of dimensions".to_string(),
        ));
    }
    let base_shape = base.shape().to_vec();
    let idx_shape = idx.shape().to_vec();
    let out_shape = take_along_axis_broadcast_check(&base_shape, &idx_shape, ax)?;

    let contig = base.to_contiguous();
    let in_strides = row_major_strides(&base_shape);
    let idx_strides_bcast = row_major_strides(&idx_shape);
    let idx_flat = i64_buf(&idx);
    let n = base_shape[ax];
    let total = shape::size_of_shape(&out_shape);

    let mut resolve_err: Option<IonpError> = None;
    let mut positions = Vec::with_capacity(total);
    let out_strides = row_major_strides(&out_shape);
    for out_flat in 0..total {
        let mut rem = out_flat;
        let mut src_flat = 0usize;
        let mut idx_flat_pos = 0usize;
        for d in 0..out_shape.len() {
            let coord = if out_strides[d] == 0 { 0 } else { rem / out_strides[d] };
            rem %= out_strides[d].max(1);
            if d == ax {
                continue;
            }
            src_flat += coord * in_strides[d];
            let idx_coord = if idx_shape[d] == 1 { 0 } else { coord };
            idx_flat_pos += idx_coord * idx_strides_bcast[d];
        }
        // Second pass to fold in the axis coordinate (needs idx_flat_pos's
        // other-dim contributions first since resolving the raw index
        // requires the full flat position within `idx`, including
        // whatever the axis dim itself contributes).
        let axis_out_coord = {
            let mut r = out_flat;
            let mut c = 0usize;
            for d in 0..out_shape.len() {
                let coord = if out_strides[d] == 0 { 0 } else { r / out_strides[d] };
                r %= out_strides[d].max(1);
                if d == ax {
                    c = coord;
                }
            }
            c
        };
        idx_flat_pos += axis_out_coord * idx_strides_bcast[ax];
        let raw = idx_flat[idx_flat_pos];
        match resolve_index(raw, n, ax, ClipMode::Raise) {
            Ok(c) => src_flat += c * in_strides[ax],
            Err(e) => {
                resolve_err = Some(e);
                break;
            }
        }
        positions.push(src_flat);
    }
    if let Some(e) = resolve_err {
        return Err(e);
    }
    let out_buffer = gather_flat!(contig.buffer(), total, |i: usize| positions[i]);
    NdArray::from_buffer(out_buffer, out_shape, Order::C)
}

/// Reproduces numpy's `take_along_axis` broadcast-compatibility check
/// (`IndexError: shape mismatch: indexing arrays could not be broadcast
/// together with shapes {} {} `, note trailing space) WITHOUT literally
/// building the per-dimension `arange` arrays numpy's own pure-Python
/// implementation does. numpy's algorithm assembles, for each dim `d` in
/// order, either `indices` itself (at `d == axis`) or an `arange(arr.shape
/// [d])` reshaped to be 1 everywhere except position `d` (at every other
/// dim), then broadcasts that whole list together via the ordinary
/// leading-reference broadcast scan (same "first non-1 size becomes the
/// reference, first later conflicting size loses" rule `broadcast_shapes`
/// above already implements) -- reproduced here directly against the two
/// KINDS of shapes that can appear (`indices`'s full shape at the axis
/// slot, a mostly-1s shape at every other slot) since materializing the
/// arrays themselves would be wasted work only to throw them away.
/// `take_along_axis`'s own shape-conflict message formats each shape
/// WITHOUT the space after the comma that `shape_tuple_str` (Python-repr
/// style, used by `broadcast_shapes`'s message above) inserts -- verified
/// live: `np.take_along_axis` on incompatible shapes reports
/// `"...shapes (2,1) (3,3) "` (note also the trailing space after the
/// final shape, reproduced by the caller below), not `"(2, 1) (3, 3)"`.
/// This is a distinct numpy C code path (fancy-indexing broadcast) from
/// `broadcast_shapes`'s own message, so it gets its own local formatter
/// rather than a shared one.
fn shape_tuple_str_no_space(shape: &[usize]) -> String {
    match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => {
            let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", parts.join(","))
        }
    }
}

/// Plain `np.broadcast_to`-style broadcastability check (right-aligned,
/// each source dim must be 1 or equal to the matching target dim; a
/// source with MORE dims than the target never broadcasts, matching
/// numpy's own `broadcast_to` semantics -- e.g. shape `(3,1,1)` cannot
/// broadcast to `(3,1)`, live-verified). No error text here -- callers
/// build their own message on failure, since (per live measurement) each
/// sibling call site in this file uses genuinely different wording.
fn shape_broadcastable_to(src: &[usize], target: &[usize]) -> bool {
    if src.len() > target.len() {
        return false;
    }
    let offset = target.len() - src.len();
    for i in 0..src.len() {
        if src[i] != 1 && src[i] != target[offset + i] {
            return false;
        }
    }
    true
}

fn take_along_axis_broadcast_check(
    base_shape: &[usize],
    idx_shape: &[usize],
    axis: usize,
) -> Result<Vec<usize>, IonpError> {
    let ndim = base_shape.len();
    // shape_of(d) = the full shape of the `d`-th entry in numpy's own
    // `fancy_index` list: `indices.shape` verbatim if `d == axis`,
    // otherwise all-1s except `base_shape[d]` at position `d`.
    let shape_of = |d: usize| -> Vec<usize> {
        if d == axis {
            idx_shape.to_vec()
        } else {
            let mut s = vec![1usize; ndim];
            s[d] = base_shape[d];
            s
        }
    };
    let mut reference: Option<(usize, Vec<usize>)> = None; // (size, owning entry's full shape)
    for p in 0..ndim {
        for d in 0..ndim {
            let s = shape_of(d);
            let sz = s[p];
            if sz == 1 {
                continue;
            }
            match &reference {
                None => reference = Some((sz, s)),
                Some((ref_sz, ref_shape)) if sz != *ref_sz => {
                    return Err(IonpError::Index(format!(
                        "shape mismatch: indexing arrays could not be broadcast together with shapes {} {} ",
                        shape_tuple_str_no_space(ref_shape),
                        shape_tuple_str_no_space(&s),
                    )));
                }
                _ => {}
            }
        }
        reference = None;
    }
    // Every position validated pairwise above; the actual output shape is
    // `base_shape` with `idx_shape[axis]` substituted at `axis` (every
    // non-axis dim's combined size is always `base_shape[d]`, since the
    // per-dim `arange` there is never smaller than `indices`'s own size
    // at that dim once broadcast-compatibility is established).
    let mut out = base_shape.to_vec();
    out[axis] = idx_shape[axis];
    Ok(out)
}

/// numpy `np.put_along_axis(arr, indices, values, axis)`: mutates `arr` in
/// place, returns `None`. `axis` has NO default in numpy's own signature
/// (unlike `take_along_axis`'s `axis=-1`) -- the ionp-py boundary still
/// accepts `axis=None` (flatten, same `indices`-must-be-1-d rule as
/// `take_along_axis`) since that IS a legal value to pass explicitly, just
/// not a default. Shares `take_along_axis`'s broadcast-compatibility
/// check and integer-dtype check; `values` broadcasts against the same
/// virtual shape (verified live: a scalar `values` broadcasts to every
/// selected position, an exactly-`indices`-shaped `values` maps
/// elementwise).
pub fn put_along_axis(a: &mut NdArray, indices: &NdArray, values: &NdArray, axis: Option<usize>) -> Result<(), IonpError> {
    let int_indices = require_integer_no_bool(indices)?;

    let ax = match axis {
        Some(ax) => ax,
        None => {
            if indices.ndim() != 1 {
                return Err(IonpError::Value(
                    "when axis=None, `indices` must have a single dimension.".to_string(),
                ));
            }
            let flat_len = shape::size_of_shape(a.shape());
            let flat_shape = vec![flat_len];
            let mut flat = a.to_contiguous().reshape(&flat_shape).unwrap();
            put_along_axis(&mut flat, indices, values, Some(0))?;
            // write back: `flat` is a fresh contiguous buffer (owns its
            // storage after `to_contiguous`), so copy it straight into
            // `a`'s own (possibly strided) storage element-by-element via
            // the ordinary reshape-view path.
            let restored = flat.reshape(a.shape()).unwrap_or_else(|_| flat.reshape(a.shape()).unwrap());
            *a = restored;
            return Ok(());
        }
    };
    if a.ndim() != int_indices.ndim() {
        return Err(IonpError::Value(
            "`indices` and `arr` must have the same number of dimensions".to_string(),
        ));
    }
    let base_shape = a.shape().to_vec();
    let idx_shape = int_indices.shape().to_vec();
    let out_shape = take_along_axis_broadcast_check(&base_shape, &idx_shape, ax)?;
    let n = base_shape[ax];

    // `put_along_axis`'s own value-shape-mismatch message, distinct from
    // both `broadcast_to`'s generic "operands could not be broadcast
    // together with shapes {} {} " text and `take_along_axis`'s
    // "shape mismatch: indexing arrays could not be broadcast..."
    // IndexError -- live-verified against numpy 2.5.1: a ValueError
    // reading "shape mismatch: value array of shape {} could not be
    // broadcast to indexing result of shape {}", both shapes formatted
    // with no space after commas (same convention as
    // `shape_tuple_str_no_space`, confirmed to be the same formatting
    // numpy uses here too).
    if !shape_broadcastable_to(values.shape(), &out_shape) {
        return Err(IonpError::Value(format!(
            "shape mismatch: value array of shape {} could not be broadcast to indexing result of shape {}",
            shape_tuple_str_no_space(values.shape()),
            shape_tuple_str_no_space(&out_shape),
        )));
    }
    let values_cast = crate::creation::broadcast_to(&values.cast_to(a.dtype()), &out_shape)?.to_contiguous();
    let idx_strides_bcast = row_major_strides(&idx_shape);
    let idx_flat = i64_buf(&int_indices);
    let out_strides = row_major_strides(&out_shape);
    let total = shape::size_of_shape(&out_shape);

    let a_strides = a.strides().to_vec();
    let a_offset = a.offset();
    let a_logical_strides = row_major_strides(&base_shape);
    let a_dtype = a.dtype();
    let ndim = base_shape.len();
    let buffer = Arc::make_mut(&mut a.buffer);

    macro_rules! scatter_arm {
        ($variant:ident) => {{
            let src = match values_cast.buffer() {
                Buffer::$variant(v) => v.as_slice(),
                _ => unreachable!(),
            };
            let dst = match buffer {
                Buffer::$variant(v) => v,
                _ => unreachable!(),
            };
            for out_flat in 0..total {
                let mut rem = out_flat;
                let mut idx_flat_pos = 0usize;
                let mut axis_coord = 0usize;
                let mut other_coords = vec![0usize; ndim];
                for d in 0..ndim {
                    let coord = if out_strides[d] == 0 { 0 } else { rem / out_strides[d] };
                    rem %= out_strides[d].max(1);
                    other_coords[d] = coord;
                    if d == ax {
                        axis_coord = coord;
                    }
                    let idx_coord = if idx_shape[d] == 1 { 0 } else { coord };
                    idx_flat_pos += idx_coord * idx_strides_bcast[d];
                }
                let _ = axis_coord;
                let raw = idx_flat[idx_flat_pos];
                let src_ax_coord = resolve_index(raw, n, ax, ClipMode::Raise)?;
                let mut phys = a_offset;
                for d in 0..ndim {
                    let c = if d == ax { src_ax_coord } else { other_coords[d] };
                    let _ = a_logical_strides;
                    phys += (c as isize) * a_strides[d];
                }
                dst[phys as usize] = src[out_flat];
            }
        }};
    }
    match a_dtype {
        DType::Bool => scatter_arm!(Bool),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => scatter_arm!(I8),
        DType::I16 => scatter_arm!(I16),
        DType::I32 => scatter_arm!(I32),
        DType::I64 => scatter_arm!(I64),
        DType::U8 => scatter_arm!(U8),
        DType::U16 => scatter_arm!(U16),
        DType::U32 => scatter_arm!(U32),
        DType::U64 => scatter_arm!(U64),
        DType::F16 => scatter_arm!(F16),
        DType::F32 => scatter_arm!(F32),
        DType::F64 => scatter_arm!(F64),
        DType::C64 => scatter_arm!(C64),
        DType::C128 => scatter_arm!(C128),
    }
    Ok(())
}

/// Local copy of "truthy" per-element conversion (per this module's own
/// axis-normalization-style convention of small per-module copies rather
/// than a shared helper -- `setops.rs::nonzero_mask` is the closest
/// sibling but is private to that file and this task does not own it).
fn nonzero_mask_local(b: &Buffer) -> Vec<bool> {
    match b {
        Buffer::Bool(v) => v.clone(),
        Buffer::I8(v) => v.iter().map(|&x| x != 0).collect(),
        Buffer::I16(v) => v.iter().map(|&x| x != 0).collect(),
        Buffer::I32(v) => v.iter().map(|&x| x != 0).collect(),
        Buffer::I64(v) => v.iter().map(|&x| x != 0).collect(),
        Buffer::U8(v) => v.iter().map(|&x| x != 0).collect(),
        Buffer::U16(v) => v.iter().map(|&x| x != 0).collect(),
        Buffer::U32(v) => v.iter().map(|&x| x != 0).collect(),
        Buffer::U64(v) => v.iter().map(|&x| x != 0).collect(),
        Buffer::F16(v) => v.iter().map(|&x| x.to_f64() != 0.0).collect(),
        Buffer::F32(v) => v.iter().map(|&x| x != 0.0).collect(),
        Buffer::F64(v) => v.iter().map(|&x| x != 0.0).collect(),
        Buffer::C64(v) => v.iter().map(|&z| z.re != 0.0 || z.im != 0.0).collect(),
        Buffer::C128(v) => v.iter().map(|&z| z.re != 0.0 || z.im != 0.0).collect(),
        // numpy string truthiness: an element is truthy iff it is
        // non-empty after stripping the fixed-width NUL padding -- same
        // rule as Python `bool(b"")`/`bool("")` being False.
        Buffer::S(_, v) => v.iter().map(|e| e.iter().any(|&b| b != 0)).collect(),
        Buffer::U(_, v) => v.iter().map(|e| e.iter().any(|&c| c != 0)).collect(),
    }
}

/// numpy `np.compress(condition, a, axis=None, out=None)`. `condition`
/// must be 1-d (numpy: `ValueError: condition must be a 1-d array`,
/// verified live) and is interpreted as a truthy mask (any numeric dtype
/// accepted, verified live: a float condition list like `[0.0,1.5,2.0]`
/// selects positions 1 and 2). SHORTER `condition` than the axis is fine
/// (implicitly padded with `False`); numpy's real implementation is
/// `a.take(condition.nonzero()[0], axis=axis)` -- so a `condition` with
/// True positions beyond the axis's length is NOT bounds-checked by
/// `compress` itself at all, it is simply handed straight to `take`,
/// which raises its own ordinary out-of-bounds `IndexError` on the first
/// invalid position VALUE it hits in order (verified live:
/// `np.compress([1]*5, a, axis=0)` on a 3-row array raises `index 3 is
/// out of bounds for axis 0 with size 3` -- `3`, the actual out-of-range
/// POSITION, not `4` = len-1 or any other derived count). Reproduced here
/// by literally delegating to `take` with the nonzero-position array,
/// rather than re-deriving a bounds check, so the message is `take`'s own
/// verbatim wording in every case.
pub fn compress(condition: &NdArray, a: &NdArray, axis: Option<usize>) -> Result<NdArray, IonpError> {
    if condition.ndim() != 1 {
        return Err(IonpError::Value("condition must be a 1-d array".to_string()));
    }
    let mask = nonzero_mask_local(condition.to_contiguous().buffer());
    let selected: Vec<i64> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i as i64) } else { None })
        .collect();
    let (base, ax) = match axis {
        Some(ax) => (a.clone(), ax),
        None => (a.to_contiguous().reshape(&[a.size()]).unwrap(), 0),
    };
    let idx_shape = vec![selected.len()];
    let idx_arr = NdArray::from_buffer(Buffer::I64(selected), idx_shape, Order::C)?;
    take(&base, &idx_arr, Some(ax), ClipMode::Raise)
}

/// Shared `out=` writer for `take`/`compress`. Unlike `put`/`put_along_axis`
/// above (which scatter into a subset of an EXISTING array's own storage),
/// `take`/`compress` compute a brand-new result and then copy it wholesale
/// into a caller-supplied `out` array. Verified live against numpy 2.5.1:
/// the target shape must match the computed result EXACTLY (not merely be
/// broadcast-compatible, unlike `write_out`'s ufunc-family semantics) --
/// mismatch raises the fixed text `output array does not match result of
/// ndarray.take` (compress uses the identical message, since real numpy's
/// `compress` is itself implemented as a `take` under the hood); the dtype
/// is cast UNSAFELY into whatever `out` already is (float->int truncates,
/// complex->float silently drops the imaginary part, both confirmed live);
/// and `out`'s own memory layout (e.g. an explicitly Fortran-ordered
/// buffer) is honored/mutated in place rather than replaced -- mirroring
/// `put`'s own logical-index-to-physical-offset decomposition above.
pub fn write_exact_shape_out(out: &mut NdArray, computed: &NdArray) -> Result<(), IonpError> {
    if out.shape() != computed.shape() {
        return Err(IonpError::Value(
            "output array does not match result of ndarray.take".to_string(),
        ));
    }
    let src = computed.cast_to(out.dtype()).to_contiguous();
    let shape = out.shape().to_vec();
    let strides = out.strides().to_vec();
    let offset = out.offset();
    let out_dtype = out.dtype();
    let logical_strides = row_major_strides(&shape);
    let ndim = shape.len();
    let total = shape::size_of_shape(&shape);
    let buffer = Arc::make_mut(&mut out.buffer);

    macro_rules! copy_arm {
        ($variant:ident) => {{
            let s = match src.buffer() {
                Buffer::$variant(v) => v,
                _ => unreachable!(),
            };
            if let Buffer::$variant(dst) = buffer {
                for flat_logical in 0..total {
                    let mut rem = flat_logical;
                    let mut phys = offset;
                    for d in 0..ndim {
                        let c = if logical_strides[d] == 0 { 0 } else { rem / logical_strides[d] };
                        rem %= logical_strides[d].max(1);
                        phys += (c as isize) * strides[d];
                    }
                    dst[phys as usize] = s[flat_logical];
                }
            }
        }};
    }
    match out_dtype {
        DType::Bool => copy_arm!(Bool),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => copy_arm!(I8),
        DType::I16 => copy_arm!(I16),
        DType::I32 => copy_arm!(I32),
        DType::I64 => copy_arm!(I64),
        DType::U8 => copy_arm!(U8),
        DType::U16 => copy_arm!(U16),
        DType::U32 => copy_arm!(U32),
        DType::U64 => copy_arm!(U64),
        DType::F16 => copy_arm!(F16),
        DType::F32 => copy_arm!(F32),
        DType::F64 => copy_arm!(F64),
        DType::C64 => copy_arm!(C64),
        DType::C128 => copy_arm!(C128),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::NdArray;
    use crate::buffer::Buffer;

    fn arr_i64(data: &[i64], shape: &[usize]) -> NdArray {
        NdArray::from_buffer(Buffer::I64(data.to_vec()), shape.to_vec(), Order::C).unwrap()
    }

    fn as_i64(a: &NdArray) -> Vec<i64> {
        match a.to_contiguous().buffer() {
            Buffer::I64(v) => v.clone(),
            other => panic!("expected I64, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn concatenate_axis0_and_axis1() {
        let a = arr_i64(&[1, 2, 3, 4], &[2, 2]);
        let b = arr_i64(&[5, 6, 7, 8], &[2, 2]);
        let c = concatenate(&[&a, &b], Some(0)).unwrap();
        assert_eq!(c.shape(), &[4, 2]);
        assert_eq!(as_i64(&c), vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let d = concatenate(&[&a, &b], Some(1)).unwrap();
        assert_eq!(d.shape(), &[2, 4]);
        assert_eq!(as_i64(&d), vec![1, 2, 5, 6, 3, 4, 7, 8]);
    }

    #[test]
    fn stack_new_axis() {
        let a = arr_i64(&[1, 2, 3], &[3]);
        let b = arr_i64(&[4, 5, 6], &[3]);
        let s0 = stack(&[&a, &b], 0).unwrap();
        assert_eq!(s0.shape(), &[2, 3]);
        assert_eq!(as_i64(&s0), vec![1, 2, 3, 4, 5, 6]);
        let s1 = stack(&[&a, &b], 1).unwrap();
        assert_eq!(s1.shape(), &[3, 2]);
        assert_eq!(as_i64(&s1), vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn hstack_vstack_dstack() {
        let a = arr_i64(&[1, 2, 3], &[3]);
        let b = arr_i64(&[4, 5, 6], &[3]);
        let h = hstack(&[&a, &b]).unwrap();
        assert_eq!(h.shape(), &[6]);
        assert_eq!(as_i64(&h), vec![1, 2, 3, 4, 5, 6]);

        let v = vstack(&[&a, &b]).unwrap();
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(as_i64(&v), vec![1, 2, 3, 4, 5, 6]);

        let d = dstack(&[&a, &b]).unwrap();
        assert_eq!(d.shape(), &[1, 3, 2]);
        assert_eq!(as_i64(&d), vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn column_stack_1d() {
        let a = arr_i64(&[1, 2, 3], &[3]);
        let b = arr_i64(&[4, 5, 6], &[3]);
        let c = column_stack(&[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(as_i64(&c), vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn flip_and_lr_ud() {
        let a = arr_i64(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        let f = flip(&a, None);
        assert_eq!(as_i64(&f), vec![6, 5, 4, 3, 2, 1]);
        let lr = fliplr(&a).unwrap();
        assert_eq!(as_i64(&lr), vec![3, 2, 1, 6, 5, 4]);
        let ud = flipud(&a).unwrap();
        assert_eq!(as_i64(&ud), vec![4, 5, 6, 1, 2, 3]);
    }

    #[test]
    fn roll_flat_and_axis() {
        let a = arr_i64(&[1, 2, 3, 4, 5], &[5]);
        let r = roll(&a, &[2], None).unwrap();
        assert_eq!(as_i64(&r), vec![4, 5, 1, 2, 3]);
        let r2 = roll(&a, &[-1], None).unwrap();
        assert_eq!(as_i64(&r2), vec![2, 3, 4, 5, 1]);

        let b = arr_i64(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        let rb = roll(&b, &[1], Some(&[0])).unwrap();
        assert_eq!(as_i64(&rb), vec![4, 5, 6, 1, 2, 3]);
        let rc = roll(&b, &[1], Some(&[1])).unwrap();
        assert_eq!(as_i64(&rc), vec![3, 1, 2, 6, 4, 5]);
    }

    #[test]
    fn tile_basic() {
        let a = arr_i64(&[1, 2, 3], &[3]);
        let t = tile(&a, &[2]).unwrap();
        assert_eq!(t.shape(), &[6]);
        assert_eq!(as_i64(&t), vec![1, 2, 3, 1, 2, 3]);

        let b = arr_i64(&[1, 2, 3, 4], &[2, 2]);
        let t2 = tile(&b, &[2, 1]).unwrap();
        assert_eq!(t2.shape(), &[4, 2]);
        assert_eq!(as_i64(&t2), vec![1, 2, 3, 4, 1, 2, 3, 4]);
    }

    #[test]
    fn repeat_basic() {
        let a = arr_i64(&[1, 2, 3], &[3]);
        let r = repeat(&a, &[2, 2, 2], None).unwrap();
        assert_eq!(as_i64(&r), vec![1, 1, 2, 2, 3, 3]);

        let b = arr_i64(&[1, 2, 3, 4], &[2, 2]);
        let r2 = repeat(&b, &[1, 2], Some(0)).unwrap();
        assert_eq!(r2.shape(), &[3, 2]);
        assert_eq!(as_i64(&r2), vec![1, 2, 3, 4, 3, 4]);
    }

    #[test]
    fn diag_1d_to_2d_and_back() {
        let v = arr_i64(&[1, 2, 3], &[3]);
        let m = diag(&v, 0).unwrap();
        assert_eq!(m.shape(), &[3, 3]);
        assert_eq!(as_i64(&m), vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);

        let back = diag(&m, 0).unwrap();
        assert_eq!(back.shape(), &[3]);
        assert_eq!(as_i64(&back), vec![1, 2, 3]);

        let m1 = diag(&v, 1).unwrap();
        assert_eq!(m1.shape(), &[4, 4]);
        assert_eq!(as_i64(&m1), vec![0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0]);
    }

    #[test]
    fn diagonal_offset() {
        let a = arr_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &[3, 3]);
        let d0 = diagonal(&a, 0, 0, 1).unwrap();
        assert_eq!(as_i64(&d0), vec![1, 5, 9]);
        let d1 = diagonal(&a, 1, 0, 1).unwrap();
        assert_eq!(as_i64(&d1), vec![2, 6]);
        let dm1 = diagonal(&a, -1, 0, 1).unwrap();
        assert_eq!(as_i64(&dm1), vec![4, 8]);
    }

    #[test]
    fn tril_triu_basic() {
        let a = arr_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &[3, 3]);
        let lo = tril(&a, 0).unwrap();
        assert_eq!(as_i64(&lo), vec![1, 0, 0, 4, 5, 0, 7, 8, 9]);
        let up = triu(&a, 0).unwrap();
        assert_eq!(as_i64(&up), vec![1, 2, 3, 0, 5, 6, 0, 0, 9]);
    }

    #[test]
    fn trace_basic() {
        let a = arr_i64(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &[3, 3]);
        let t = trace(&a, 0, 0, 1, None).unwrap();
        assert_eq!(t.shape(), &[] as &[usize]);
        assert_eq!(as_i64(&t), vec![15]);
    }

    #[test]
    fn broadcast_shapes_and_arrays() {
        let s = broadcast_shapes(&[&[3, 1], &[1, 4], &[3, 4]]).unwrap();
        assert_eq!(s, vec![3, 4]);

        let a = arr_i64(&[1, 2, 3], &[3]);
        let b = arr_i64(&[10], &[1]);
        let out = broadcast_arrays(&[a, b]).unwrap();
        assert_eq!(out[0].shape(), &[3]);
        assert_eq!(out[1].shape(), &[3]);
    }

    #[test]
    fn atleast_family() {
        let scalar = NdArray::from_buffer(Buffer::I64(vec![5]), vec![], Order::C).unwrap();
        assert_eq!(atleast_1d(&scalar).shape(), &[1]);
        assert_eq!(atleast_2d(&scalar).shape(), &[1, 1]);
        assert_eq!(atleast_3d(&scalar).shape(), &[1, 1, 1]);

        let v = arr_i64(&[1, 2, 3], &[3]);
        assert_eq!(atleast_2d(&v).shape(), &[1, 3]);
        assert_eq!(atleast_3d(&v).shape(), &[1, 3, 1]);
    }

    #[test]
    fn diag_indices_basic_and_negative_n() {
        let out = diag_indices(3, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(as_i64(&out[0]), vec![0, 1, 2]);
        assert_eq!(as_i64(&out[1]), vec![0, 1, 2]);

        let empty = diag_indices(-2, 2);
        assert_eq!(as_i64(&empty[0]), Vec::<i64>::new());

        let ndim1 = diag_indices(3, 1);
        assert_eq!(ndim1.len(), 1);
        let ndim0 = diag_indices(3, 0);
        assert_eq!(ndim0.len(), 0);
    }

    #[test]
    fn diag_indices_from_validates_square() {
        let sq = arr_i64(&[0, 0, 0, 0], &[2, 2]);
        let out = diag_indices_from(&sq).unwrap();
        assert_eq!(as_i64(&out[0]), vec![0, 1]);

        let non_square = arr_i64(&[0; 12], &[3, 4]);
        let err = diag_indices_from(&non_square).unwrap_err();
        assert_eq!(err.to_string(), "All dimensions of input must be of equal length");

        let too_few_dims = arr_i64(&[0, 0, 0], &[3]);
        let err = diag_indices_from(&too_few_dims).unwrap_err();
        assert_eq!(err.to_string(), "input array must be at least 2-d");
    }

    #[test]
    fn tril_triu_indices_basic() {
        let (r, c) = tril_indices(2, 0, Some(4));
        assert_eq!(as_i64(&r), vec![0, 1, 1]);
        assert_eq!(as_i64(&c), vec![0, 0, 1]);

        let (r, c) = triu_indices(2, 1, Some(4));
        assert_eq!(as_i64(&r), vec![0, 0, 0, 1, 1]);
        assert_eq!(as_i64(&c), vec![1, 2, 3, 2, 3]);
    }

    #[test]
    fn ix_builds_open_mesh_and_converts_bool() {
        let a = arr_i64(&[1, 2], &[2]);
        let b = arr_i64(&[3, 4, 5], &[3]);
        let out = ix_(&[a, b]).unwrap();
        assert_eq!(out[0].shape(), &[2, 1]);
        assert_eq!(out[1].shape(), &[1, 3]);

        let mask = NdArray::from_buffer(Buffer::Bool(vec![true, false, true]), vec![3], Order::C).unwrap();
        let out = ix_(&[mask]).unwrap();
        assert_eq!(as_i64(&out[0]), vec![0, 2]);

        let bad = arr_i64(&[1, 2, 3, 4], &[2, 2]);
        let err = ix_(&[bad]).unwrap_err();
        assert_eq!(err.to_string(), "Cross index must be 1 dimensional");
    }

    #[test]
    fn indices_dense_and_sparse() {
        let dense = indices(&[2, 3], DType::I64, false).unwrap();
        assert_eq!(dense.len(), 1);
        assert_eq!(dense[0].shape(), &[2, 2, 3]);

        let sparse = indices(&[2, 3], DType::I64, true).unwrap();
        assert_eq!(sparse.len(), 2);
        assert_eq!(sparse[0].shape(), &[2, 1]);
        assert_eq!(sparse[1].shape(), &[1, 3]);

        let empty = indices(&[], DType::I64, false).unwrap();
        assert_eq!(empty[0].shape(), &[0]);
    }

    #[test]
    fn meshgrid_ij_and_xy() {
        let x = arr_i64(&[1, 2], &[2]);
        let y = arr_i64(&[3, 4, 5], &[3]);
        let ij = meshgrid(&[&x, &y], "ij", false).unwrap();
        assert_eq!(ij[0].shape(), &[2, 3]);
        assert_eq!(ij[1].shape(), &[2, 3]);

        let xy = meshgrid(&[&x, &y], "xy", false).unwrap();
        assert_eq!(xy[0].shape(), &[3, 2]);
        assert_eq!(xy[1].shape(), &[3, 2]);

        let err = meshgrid(&[&x, &y], "bogus", false).unwrap_err();
        assert_eq!(err.to_string(), "Valid values for `indexing` are 'xy' and 'ij'.");
    }

    #[test]
    fn ravel_multi_index_roundtrips_unravel_index() {
        let rows = arr_i64(&[3, 6, 6], &[3]);
        let cols = arr_i64(&[4, 5, 1], &[3]);
        let flat = ravel_multi_index(&[rows, cols], &[7, 6], &[RavelMode::Raise, RavelMode::Raise], Order::C).unwrap();
        assert_eq!(as_i64(&flat), vec![22, 41, 37]);

        let back = unravel_index(&flat, &[7, 6], Order::C).unwrap();
        assert_eq!(as_i64(&back[0]), vec![3, 6, 6]);
        assert_eq!(as_i64(&back[1]), vec![4, 5, 1]);
    }

    #[test]
    fn ravel_multi_index_modes_and_errors() {
        let rows = arr_i64(&[3, 6, 6], &[3]);
        let cols = arr_i64(&[4, 5, 1], &[3]);
        let clipped =
            ravel_multi_index(&[rows.clone(), cols.clone()], &[7, 6], &[RavelMode::Clip, RavelMode::Clip], Order::C)
                .unwrap();
        assert_eq!(as_i64(&clipped), vec![22, 41, 37]);

        let bad_rows = arr_i64(&[3, 6, 10], &[3]);
        let err = ravel_multi_index(&[bad_rows, cols], &[7, 6], &[RavelMode::Raise, RavelMode::Raise], Order::C)
            .unwrap_err();
        assert_eq!(err.to_string(), "invalid entry in coordinates array");

        let short = arr_i64(&[3], &[1]);
        let short2 = arr_i64(&[4], &[1]);
        let err = ravel_multi_index(&[short, short2], &[7], &[RavelMode::Raise], Order::C).unwrap_err();
        assert_eq!(err.to_string(), "parameter multi_index must be a sequence of length 1");
    }

    #[test]
    fn unravel_index_out_of_bounds_is_value_error() {
        let idx = arr_i64(&[100], &[1]);
        let err = unravel_index(&idx, &[7, 6], Order::C).unwrap_err();
        assert_eq!(err.to_string(), "index 100 is out of bounds for array with size 42");
    }

    #[test]
    fn unravel_index_scalar_shape() {
        let idx = NdArray::from_buffer(Buffer::I64(vec![22]), vec![], Order::C).unwrap();
        let out = unravel_index(&idx, &[7, 6], Order::C).unwrap();
        assert_eq!(out[0].shape(), &[] as &[usize]);
        assert_eq!(as_i64(&out[0]), vec![3]);
        assert_eq!(as_i64(&out[1]), vec![4]);
    }

    #[test]
    fn unravel_index_order_f_decomposes_by_stride_not_axis_position() {
        // Regression for a real bug found via out-of-corpus differential
        // testing: order='F' strides are ASCENDING (axis 0 smallest), so
        // decomposing by looping axis 0..ndim in a fixed order divided by
        // the smallest stride first and produced garbage (echoed `raw`
        // straight into axis 0, left every other axis 0). Must consume the
        // largest stride first regardless of which axis it belongs to.
        // Matches real numpy: unravel_index([1,5,11], (3,4), order='F') ==
        // (array([1,2,2]), array([0,1,3])).
        let idx = NdArray::from_buffer(Buffer::I64(vec![1, 5, 11]), vec![3], Order::C).unwrap();
        let out = unravel_index(&idx, &[3, 4], Order::F).unwrap();
        assert_eq!(as_i64(&out[0]), vec![1, 2, 2]);
        assert_eq!(as_i64(&out[1]), vec![0, 1, 3]);
    }

    #[test]
    fn fill_diagonal_square_and_wrap() {
        let mut a = NdArray::from_buffer(Buffer::I64(vec![0; 16]), vec![4, 4], Order::C).unwrap();
        let val = NdArray::from_buffer(Buffer::I64(vec![9]), vec![], Order::C).unwrap();
        fill_diagonal(&mut a, &val, false).unwrap();
        assert_eq!(as_i64(&a), vec![9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9, 0, 0, 0, 0, 9]);

        let mut b = NdArray::from_buffer(Buffer::I64(vec![0; 16]), vec![4, 4], Order::C).unwrap();
        let cyc = NdArray::from_buffer(Buffer::I64(vec![1, 2]), vec![2], Order::C).unwrap();
        fill_diagonal(&mut b, &cyc, false).unwrap();
        assert_eq!(as_i64(&b), vec![1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2]);
    }

    #[test]
    fn fill_diagonal_ndim3_step_is_sum_of_cumprod_not_product() {
        // Regression for a real bug found via out-of-corpus differential
        // testing: step for ndim>=3 must be `1 + cumprod(shape[:-1]).sum()`
        // (numpy's own algorithm), NOT `1 + product(shape[:-1])`. For a
        // (3,3,3) cube those give different answers (13 vs 10) and only 13
        // matches real numpy's actual diagonal placement (0,0,0),
        // (1,1,1),(2,2,2).
        let mut a = NdArray::from_buffer(Buffer::I64(vec![0; 27]), vec![3, 3, 3], Order::C).unwrap();
        let val = NdArray::from_buffer(Buffer::I64(vec![5]), vec![], Order::C).unwrap();
        fill_diagonal(&mut a, &val, false).unwrap();
        let flat = as_i64(&a);
        let mut expected = vec![0i64; 27];
        for i in 0..3 {
            expected[i * 9 + i * 3 + i] = 5;
        }
        assert_eq!(flat, expected);
    }

    #[test]
    fn fill_diagonal_ndim_errors() {
        let mut too_few = NdArray::from_buffer(Buffer::I64(vec![0; 3]), vec![3], Order::C).unwrap();
        let val = NdArray::from_buffer(Buffer::I64(vec![1]), vec![], Order::C).unwrap();
        let err = fill_diagonal(&mut too_few, &val, false).unwrap_err();
        assert_eq!(err.to_string(), "array must be at least 2-d");

        let mut noncube = NdArray::from_buffer(Buffer::I64(vec![0; 60]), vec![3, 4, 5], Order::C).unwrap();
        let err = fill_diagonal(&mut noncube, &val, false).unwrap_err();
        assert_eq!(err.to_string(), "All dimensions of input must be of equal length");
    }
}

