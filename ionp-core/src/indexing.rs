//! Advanced (fancy / boolean) indexing.
//!
//! `NdArray::get_view` (array.rs) covers numpy's BASIC indexing — integers,
//! slices, `Ellipsis`, `newaxis` — and is always a view. This module covers
//! the rest: integer-array indices and boolean masks, alone or mixed with
//! basic items. numpy's own rule is that any key containing an array index
//! produces a COPY, never a view, and that is what this module produces.
//!
//! ## The one-offsets-vector design
//!
//! Everything here funnels into a single representation: a `Plan::Gather`
//! carrying an output shape and a vector of ABSOLUTE buffer offsets, one per
//! output element, in C order over that shape. Reading is
//! `dst[i] = src[offsets[i]]`; writing is `src[offsets[i]] = val[i]`. Get and
//! set are therefore the *same* computation run in opposite directions, which
//! is the entire reason `__setitem__` and fancy `__getitem__` could land
//! together rather than as two independent re-derivations of numpy's indexing
//! rules. It also makes duplicate indices fall out correctly for free
//! (`a[[0, 0]] = [1, 2]` leaves `2`, last write wins, matching numpy).
//!
//! The cost is honest and stated rather than hidden: the offsets vector is
//! `prod(out_shape)` `isize`s, materialised up front. For a gather that is
//! the same order of memory as the result itself. It is NOT a lazy iterator,
//! so a fancy index producing a huge result allocates the index vector as
//! well as the result. That is a deliberate simplicity-for-memory trade, not
//! an oversight.
//!
//! ## Placement of the advanced block (measured, not inferred)
//!
//! numpy puts the broadcast shape of the advanced indices *in place* when the
//! advanced items are consecutive in the key, and *at the front* when they are
//! separated by anything else. Verified live against numpy 2.5.1 on a
//! `(2,3,4,5)` array:
//!
//! ```text
//! a[None, i, i]      -> (1, 2, 4, 5)   consecutive: B stays where it is
//! a[i, i, None]      -> (2, 1, 4, 5)   consecutive
//! a[:, k, k, :]      -> (2, 3, 5)      consecutive
//! a[i, ..., i]       -> (2, 3, 4)      separated by the ellipsis -> B first
//! a[:, k, None, k]   -> (3, 2, 1, 5)   separated by a NEWAXIS -> B first
//! ```
//!
//! That last row is the one worth pinning: a bare `None` between two array
//! indices consumes no axis at all, so "separated" is a statement about the
//! KEY, not about the axes. An implementation that decides contiguity by
//! looking at which axes the advanced items consume gets `a[:, k, None, k]`
//! wrong and every other row above right.
//!
//! ## Boolean masks
//!
//! A boolean mask of rank `k` consumes `k` axes and is exactly equivalent to
//! its own `nonzero()` — `k` integer arrays of length `count_nonzero`, which
//! are consecutive with each other by construction. This is why
//! `a[a > 30]` flattens: the mask has the same rank as the array, so it
//! becomes `ndim` index arrays and the whole result is 1-D.
//!
//! A 0-d boolean mask (`a[True]`, `a[False]`, or a bare Python `True`) is
//! numpy's degenerate case: it consumes no axis and contributes a leading
//! dimension of length 1 or 0. It is handled, not excluded — as an
//! advanced index bound to no axis at all, so it rides the same
//! broadcast-and-offset machinery as everything else rather than getting a
//! parallel code path. See `AdvIndex::axis`.

use std::sync::Arc;

use crate::array::{NdArray, Order, SliceItem};
use crate::buffer::Buffer;
use crate::error::IonpError;
use crate::shape::{self, NdIter};

/// One item of an index key. A superset of `SliceItem`: the first four
/// variants are the basic ones and are forwarded verbatim to `get_view`
/// when no advanced item is present anywhere in the key.
#[derive(Debug, Clone)]
pub enum IndexItem {
    Index(isize),
    Slice {
        start: Option<isize>,
        stop: Option<isize>,
        step: Option<isize>,
    },
    NewAxis,
    Ellipsis,
    /// An integer-dtype array index. Values may be negative (wrapped against
    /// the axis length) and may repeat.
    IntArray(NdArray),
    /// A boolean-dtype mask. Consumes as many axes as it has dimensions.
    BoolArray(NdArray),
}

impl IndexItem {
    /// How many axes of the indexed array this item consumes. `NewAxis` and
    /// `Ellipsis` consume none; a boolean mask consumes one per dimension.
    fn axes_consumed(&self) -> usize {
        match self {
            IndexItem::Index(_) | IndexItem::Slice { .. } | IndexItem::IntArray(_) => 1,
            IndexItem::NewAxis | IndexItem::Ellipsis => 0,
            IndexItem::BoolArray(m) => m.ndim(),
        }
    }

    fn is_advanced(&self) -> bool {
        matches!(self, IndexItem::IntArray(_) | IndexItem::BoolArray(_))
    }

    fn as_slice_item(&self) -> Option<SliceItem> {
        Some(match self {
            IndexItem::Index(i) => SliceItem::Index(*i),
            IndexItem::Slice { start, stop, step } => SliceItem::Slice {
                start: *start,
                stop: *stop,
                step: *step,
            },
            IndexItem::NewAxis => SliceItem::NewAxis,
            IndexItem::Ellipsis => SliceItem::Ellipsis,
            _ => return None,
        })
    }
}

/// The outcome of analysing a key against an array.
pub enum Plan {
    /// The key was purely basic: the result is a real view, no copy.
    View(NdArray),
    /// The key contained at least one array index: the result is a copy,
    /// described by an output shape plus one absolute buffer offset per
    /// output element, in C order over `out_shape`.
    ///
    /// `adv_block_at`/`adv_ndim` locate the broadcast advanced-index block
    /// inside `out_shape`. They exist for `__getitem__`'s memory LAYOUT, not
    /// for its values: see `gather_order` below. `__setitem__` ignores them,
    /// which is why they are extra fields rather than a change to the
    /// meaning of `out_shape`/`offsets` -- those two still describe the
    /// FINAL result in C order, exactly as before.
    Gather {
        out_shape: Vec<usize>,
        offsets: Vec<isize>,
        adv_block_at: usize,
        adv_ndim: usize,
    },
}

/// numpy's advanced-indexing result is not a plain C-contiguous copy of the
/// final shape. Measured on numpy 2.5.1, it gathers into a C-contiguous
/// temporary with the broadcast block LEADING and then transposes that
/// temporary into final axis order, so the visible result is a VIEW of a
/// hidden intermediate:
///
///     a = np.arange(12).reshape(3, 4)          # int64
///     v = a[:, [0, 1]]
///     v.shape   (3, 2)      v.strides   (8, 24)     <- not C-contiguous
///     v.flags['OWNDATA']  False
///     v.base.shape  (2, 3)  v.base.flags['OWNDATA']  True
///
///     b = np.arange(24).reshape(2, 3, 4)
///     w = b[:, [0, 1], :]
///     w.shape (2, 2, 4)  w.strides (32, 64, 8)  w.base.shape (2, 2, 4)
///
/// In both cases the base is exactly the block-first gather and the result
/// is `base.transpose(perm)`. When the block is already leading -- which is
/// every key whose advanced items are at the front, and every key with more
/// than one advanced run (numpy moves the block to the front itself) -- the
/// result IS the C-contiguous copy and has no base, so this returns `None`
/// and the caller keeps the cheap path.
///
/// This is not cosmetic. `.strides`, `.base` and `flags['OWNDATA']` are all
/// public and directly queryable on `ionp.ndarray`, so returning a plain
/// C-contiguous owning array here is an observable divergence -- the same
/// class of defect that had the shape ops revoked for a week.
///
/// Returns `(gather_shape, gather_offsets, perm)` where `gather_shape` is
/// block-first, `gather_offsets` is `offsets` reordered to C order over
/// `gather_shape`, and `perm` is the axis permutation taking the gather
/// result back to `out_shape` (i.e. `temp.transpose_axes(perm)`).
#[allow(clippy::type_complexity)]
pub fn gather_order(
    out_shape: &[usize],
    offsets: &[isize],
    adv_block_at: usize,
    adv_ndim: usize,
) -> Option<(Vec<usize>, Vec<isize>, Vec<usize>)> {
    if adv_block_at == 0 || adv_ndim == 0 {
        return None;
    }
    let ndim = out_shape.len();
    // `perm[i]` = which GATHER axis becomes final axis `i`. Gather axis
    // layout is [block (adv_ndim axes)] ++ [the non-block output axes, in
    // their original relative order].
    let mut perm = vec![0usize; ndim];
    for i in 0..adv_block_at {
        perm[i] = adv_ndim + i;
    }
    for k in 0..adv_ndim {
        perm[adv_block_at + k] = k;
    }
    for i in (adv_block_at + adv_ndim)..ndim {
        perm[i] = i;
    }
    let gather_shape: Vec<usize> = (0..ndim).map(|g| out_shape[inv_at(&perm, g)]).collect();

    // Reorder the offsets. `offsets` is indexed in C order over `out_shape`;
    // the output here is indexed in C order over `gather_shape`. Walk the
    // gather odometer and compute the matching flat index into `offsets`.
    let final_c = c_index_strides(out_shape);
    let mut gather_offsets = vec![0isize; offsets.len()];
    let mut counter = vec![0usize; ndim];
    for slot in gather_offsets.iter_mut() {
        let mut src = 0isize;
        for (fi, &gi) in perm.iter().enumerate() {
            src += final_c[fi] * counter[gi] as isize;
        }
        *slot = offsets[src as usize];
        for ax in (0..ndim).rev() {
            counter[ax] += 1;
            if counter[ax] < gather_shape[ax] {
                break;
            }
            counter[ax] = 0;
        }
    }
    Some((gather_shape, gather_offsets, perm))
}

/// Index of `g` inside `perm` (`perm` is a permutation, so this is its
/// inverse evaluated at `g`).
fn inv_at(perm: &[usize], g: usize) -> usize {
    perm.iter().position(|&x| x == g).expect("perm is a permutation")
}

/// Does this key contain any advanced (array) item?
pub fn has_advanced(items: &[IndexItem]) -> bool {
    items.iter().any(|i| i.is_advanced())
}

/// Read every element of an integer-dtype array in C order, as `i64`.
///
/// Bool is deliberately NOT accepted: a boolean array reaching here would be
/// a caller bug (it should have gone through `expand_bool`), and silently
/// reading it as 0/1 integers is precisely the failure mode that turns a
/// mask into a wrong-but-plausible gather. numpy itself made this mistake
/// historically and deprecated its way out of it.
fn int_index_values(a: &NdArray) -> Result<Vec<i64>, IonpError> {
    let n = shape::size_of_shape(a.shape());
    let iter = NdIter::new(a.shape(), a.strides());
    let off = a.offset();
    macro_rules! arm {
        ($v:expr) => {
            iter.map(|o| $v[(off + o) as usize] as i64).collect()
        };
    }
    let out: Vec<i64> = match a.buffer() {
        Buffer::I8(v) => arm!(v),
        Buffer::I16(v) => arm!(v),
        Buffer::I32(v) => arm!(v),
        Buffer::I64(v) => arm!(v),
        Buffer::U8(v) => arm!(v),
        Buffer::U16(v) => arm!(v),
        Buffer::U32(v) => arm!(v),
        Buffer::U64(v) => arm!(v),
        _ => {
            return Err(IonpError::Index(
                "arrays used as indices must be of integer (or boolean) type".to_string(),
            ))
        }
    };
    debug_assert_eq!(out.len(), n);
    Ok(out)
}

/// Read a boolean array's elements in C order.
fn bool_values(a: &NdArray) -> Result<Vec<bool>, IonpError> {
    let iter = NdIter::new(a.shape(), a.strides());
    let off = a.offset();
    match a.buffer() {
        Buffer::Bool(v) => Ok(iter.map(|o| v[(off + o) as usize]).collect()),
        _ => Err(IonpError::Index(
            "boolean mask expected but a non-boolean buffer was supplied".to_string(),
        )),
    }
}

/// Turn a rank-`k` boolean mask into the `k` integer index arrays numpy's
/// own `nonzero()` would produce, after checking the mask against the `k`
/// axes it covers.
///
/// The shape check is numpy's, message included: it names the FIRST axis
/// that disagrees and reports both sizes. Verified live:
/// `np.arange(60).reshape(3,4,5)[np.array([True, False])]` ->
/// "boolean index did not match indexed array along axis 0; size of axis is
/// 3 but size of corresponding boolean axis is 2".
fn expand_bool(mask: &NdArray, dims: &[usize], first_axis: usize) -> Result<Vec<Vec<i64>>, IonpError> {
    for (k, &m) in mask.shape().iter().enumerate() {
        let axis = first_axis + k;
        let size = dims.get(k).copied().unwrap_or(0);
        if m != size {
            return Err(IonpError::Index(format!(
                "boolean index did not match indexed array along axis {axis}; \
                 size of axis is {size} but size of corresponding boolean axis is {m}"
            )));
        }
    }
    let flags = bool_values(mask)?;
    let k = mask.ndim();
    let mut cols: Vec<Vec<i64>> = vec![Vec::new(); k];
    // C-order walk over the mask's own shape, recording the multi-index of
    // every `true`. This IS `nonzero()`; it is spelled out here rather than
    // routed through `manip::nonzero` so that the mask's own strides (it may
    // itself be a view) are handled by `bool_values` above and this loop only
    // ever deals with C-order logical positions.
    let mut idx = vec![0usize; k];
    for &flag in flags.iter() {
        if flag {
            for (c, &i) in idx.iter().enumerate() {
                cols[c].push(i as i64);
            }
        }
        for ax in (0..k).rev() {
            idx[ax] += 1;
            if idx[ax] < mask.shape()[ax] {
                break;
            }
            idx[ax] = 0;
        }
    }
    Ok(cols)
}

/// A single advanced index, already flattened to `i64` values plus the shape
/// they carry, bound to the axis of the source array it indexes.
struct AdvIndex {
    /// The source axis this index selects along, or `None` for the one case
    /// that selects along nothing: a 0-d boolean mask. numpy's `a[True]`
    /// consumes no axis and prepends a length-1 dimension (`a[False]`, a
    /// length-0 one) — it is a pure shape contribution. Modelling it as an
    /// axis-less advanced index with value `0` folds it into the same
    /// broadcast-and-offset machinery as every other index instead of
    /// bolting on a parallel code path.
    axis: Option<usize>,
    shape: Vec<usize>,
    values: Vec<i64>,
}

/// Either an output dimension contributed by a basic item, or the position
/// where the advanced block belongs.
enum OutSlot {
    /// `(axis_of_source, start_offset_contribution, stride_step, len)` — a
    /// slice's output dimension.
    Slice { stride: isize, start: isize, step: isize, len: usize },
    /// A `newaxis`: one output dimension of length 1 contributing no offset.
    NewAxis,
    /// The advanced block. At most one of these exists.
    AdvBlock,
}

/// Analyse `items` against `a`. Basic-only keys return `Plan::View`.
pub fn plan(a: &NdArray, items: &[IndexItem]) -> Result<Plan, IonpError> {
    if !has_advanced(items) {
        let basic: Vec<SliceItem> = items
            .iter()
            .map(|i| i.as_slice_item().expect("no advanced item present"))
            .collect();
        return Ok(Plan::View(a.get_view(&basic)?));
    }

    // --- expand the ellipsis ------------------------------------------------
    // Same rule as `array.rs::expand_ellipsis`, but the axis accounting has to
    // know that a rank-k boolean mask consumes k axes, not one.
    let ell: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, i)| matches!(i, IndexItem::Ellipsis))
        .map(|(p, _)| p)
        .collect();
    if ell.len() > 1 {
        return Err(IonpError::Index(
            "an index can only have a single ellipsis ('...')".to_string(),
        ));
    }
    let consumed: usize = items.iter().map(|i| i.axes_consumed()).sum();
    let items: Vec<IndexItem> = if let Some(&pos) = ell.first() {
        let fill = a.ndim().saturating_sub(consumed);
        let mut out = Vec::with_capacity(items.len() - 1 + fill);
        out.extend_from_slice(&items[..pos]);
        for _ in 0..fill {
            out.push(IndexItem::Slice { start: None, stop: None, step: None });
        }
        out.extend_from_slice(&items[pos + 1..]);
        out
    } else {
        items.to_vec()
    };

    let consumed: usize = items.iter().map(|i| i.axes_consumed()).sum();
    if consumed > a.ndim() {
        return Err(IonpError::Index(format!(
            "too many indices for array: array is {}-dimensional, but {} were indexed",
            a.ndim(),
            consumed
        )));
    }

    // --- walk the key, collecting output slots and advanced indices ---------
    let mut slots: Vec<OutSlot> = Vec::new();
    let mut advs: Vec<AdvIndex> = Vec::new();
    let mut base_offset = a.offset();
    let mut axis = 0usize;
    // Contiguity is a property of the KEY, not of the axes: see the module
    // docs for `a[:, k, None, k]`, where a newaxis that consumes no axis at
    // all still separates the two array indices.
    let mut adv_runs = 0usize;
    let mut prev_was_adv = false;
    let mut adv_slot_emitted = false;

    for item in &items {
        match item {
            IndexItem::Ellipsis => unreachable!("expanded above"),
            IndexItem::NewAxis => {
                slots.push(OutSlot::NewAxis);
                prev_was_adv = false;
            }
            IndexItem::Index(i) => {
                let dim = a.shape()[axis] as isize;
                let idx = if *i < 0 { *i + dim } else { *i };
                if idx < 0 || idx >= dim {
                    return Err(IonpError::Index(format!(
                        "index {i} is out of bounds for axis {axis} with size {dim}"
                    )));
                }
                base_offset += a.strides()[axis] * idx;
                axis += 1;
                prev_was_adv = false;
            }
            IndexItem::Slice { start, stop, step } => {
                let dim = a.shape()[axis];
                let stride = a.strides()[axis];
                let (nstart, _nstop, nstep, len) =
                    crate::array::normalize_slice_pub(*start, *stop, *step, dim)?;
                slots.push(OutSlot::Slice { stride, start: nstart, step: nstep, len });
                axis += 1;
                prev_was_adv = false;
            }
            IndexItem::IntArray(ix) => {
                if !prev_was_adv {
                    adv_runs += 1;
                }
                if !adv_slot_emitted {
                    slots.push(OutSlot::AdvBlock);
                    adv_slot_emitted = true;
                }
                advs.push(AdvIndex {
                    axis: Some(axis),
                    shape: ix.shape().to_vec(),
                    values: int_index_values(ix)?,
                });
                axis += 1;
                prev_was_adv = true;
            }
            IndexItem::BoolArray(mask) => {
                if !prev_was_adv {
                    adv_runs += 1;
                }
                if !adv_slot_emitted {
                    slots.push(OutSlot::AdvBlock);
                    adv_slot_emitted = true;
                }
                if mask.ndim() == 0 {
                    // 0-d mask: `a[True]` -> one length-1 block position,
                    // `a[False]` -> a length-0 one. No axis consumed, no
                    // offset contributed.
                    let keep = bool_values(mask)?[0];
                    let n = if keep { 1 } else { 0 };
                    advs.push(AdvIndex {
                        axis: None,
                        shape: vec![n],
                        values: vec![0; n],
                    });
                } else {
                    let dims: Vec<usize> = a.shape()[axis..].to_vec();
                    let cols = expand_bool(mask, &dims, axis)?;
                    let n = cols.first().map(|c| c.len()).unwrap_or(0);
                    for (k, values) in cols.into_iter().enumerate() {
                        advs.push(AdvIndex { axis: Some(axis + k), shape: vec![n], values });
                    }
                }
                axis += mask.ndim();
                prev_was_adv = true;
            }
        }
    }
    // Trailing axes not covered by any item stay whole.
    for ax in axis..a.ndim() {
        slots.push(OutSlot::Slice {
            stride: a.strides()[ax],
            start: 0,
            step: 1,
            len: a.shape()[ax],
        });
    }

    // --- broadcast the advanced indices together ---------------------------
    let shapes: Vec<&[usize]> = advs.iter().map(|x| x.shape.as_slice()).collect();
    let bshape = broadcast_index_shapes(&shapes)?;
    let bsize = shape::size_of_shape(&bshape);

    // Resolve each advanced index into one absolute-offset contribution per
    // position of the broadcast shape. Bounds are checked here, against the
    // axis each index is bound to, with numpy's own message.
    let mut adv_offsets = vec![0isize; bsize];
    for adv in &advs {
        let (dim, stride) = match adv.axis {
            Some(ax) => (a.shape()[ax] as i64, a.strides()[ax]),
            // The axis-less 0-d-mask case: every value is 0 and contributes
            // no offset, so a nominal dim of 1 keeps the bounds check a no-op
            // rather than making it a special case.
            None => (1, 0),
        };
        let src_strides = shape::broadcast_strides_to(&adv.shape, &c_index_strides(&adv.shape), &bshape)?;
        for (out_i, o) in NdIter::new(&bshape, &src_strides).enumerate() {
            let raw = adv.values[o as usize];
            let idx = if raw < 0 { raw + dim } else { raw };
            if idx < 0 || idx >= dim {
                return Err(IonpError::Index(format!(
                    "index {raw} is out of bounds for axis {} with size {dim}",
                    adv.axis.expect("axis-less index values are always 0, never out of bounds")
                )));
            }
            adv_offsets[out_i] += stride * idx as isize;
        }
    }

    // --- assemble the output shape -----------------------------------------
    // If the advanced items formed more than one run in the key, numpy moves
    // the broadcast block to the FRONT; otherwise it stays where the run is.
    let block_first = adv_runs > 1;
    let mut out_shape: Vec<usize> = Vec::new();
    if block_first {
        out_shape.extend_from_slice(&bshape);
    }
    for slot in &slots {
        match slot {
            OutSlot::Slice { len, .. } => out_shape.push(*len),
            OutSlot::NewAxis => out_shape.push(1),
            OutSlot::AdvBlock => {
                if !block_first {
                    out_shape.extend_from_slice(&bshape);
                }
            }
        }
    }

    // --- materialise the offsets -------------------------------------------
    // Walk the output in C order. Each output multi-index splits into the
    // advanced block's position (one lookup into `adv_offsets`) and the basic
    // slots' positions (an affine offset each).
    let total = shape::size_of_shape(&out_shape);
    let mut offsets = vec![0isize; total];
    // Where does the advanced block sit in `out_shape`? Hoisted out of the
    // `total > 0` guard because `Plan::Gather` reports it unconditionally --
    // an empty result still has a layout, and `gather_order` still has to
    // agree with it.
    let block_at = if block_first {
        0
    } else {
        let mut n = 0usize;
        for slot in &slots {
            match slot {
                OutSlot::AdvBlock => break,
                _ => n += 1,
            }
        }
        n
    };
    let bndim = bshape.len();
    if total > 0 {
        let mut counter = vec![0usize; out_shape.len()];
        for out_i in 0..total {
            // advanced part
            let mut bpos = 0usize;
            for k in 0..bndim {
                bpos = bpos * bshape[k] + counter[block_at + k];
            }
            let mut off = base_offset + adv_offsets[bpos];
            // Basic part. `d` walks the OUTPUT axes, so it must account for
            // where the advanced block actually landed: when the block was
            // moved to the front it already consumed the first `bndim` output
            // axes and the in-key `AdvBlock` slot then consumes none, whereas
            // when the block stayed in place the slot consumes `bndim` axes
            // exactly where it sits. Adding `bndim` in both places is the bug
            // this comment exists to prevent a second time.
            let mut d = if block_first { bndim } else { 0 };
            for slot in &slots {
                match slot {
                    OutSlot::AdvBlock => {
                        if !block_first {
                            d += bndim;
                        }
                    }
                    OutSlot::NewAxis => d += 1,
                    OutSlot::Slice { stride, start, step, .. } => {
                        off += stride * (start + step * counter[d] as isize);
                        d += 1;
                    }
                }
            }
            offsets[out_i] = off;
            // odometer
            for ax in (0..counter.len()).rev() {
                counter[ax] += 1;
                if counter[ax] < out_shape[ax] {
                    break;
                }
                counter[ax] = 0;
            }
        }
    }

    Ok(Plan::Gather {
        out_shape,
        offsets,
        adv_block_at: block_at,
        adv_ndim: bndim,
    })
}

/// C-order strides for a shape, in ELEMENT units, for indexing a flat
/// C-order value vector (which is what `int_index_values` produces).
fn c_index_strides(shape: &[usize]) -> Vec<isize> {
    let mut s = vec![1isize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1] as isize;
    }
    s
}

/// Broadcast the index arrays' shapes together, with numpy's own message for
/// the failure (verified live: "shape mismatch: indexing arrays could not be
/// broadcast together with shapes (2,) (5,) " — trailing space included,
/// because numpy builds it by appending "{shape} " per operand).
fn broadcast_index_shapes(shapes: &[&[usize]]) -> Result<Vec<usize>, IonpError> {
    let ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut out = vec![1usize; ndim];
    for s in shapes {
        let pad = ndim - s.len();
        for (k, &d) in s.iter().enumerate() {
            let o = &mut out[pad + k];
            if *o == 1 {
                *o = d;
            } else if d != 1 && d != *o {
                let mut msg =
                    "shape mismatch: indexing arrays could not be broadcast together with shapes "
                        .to_string();
                for s2 in shapes {
                    msg.push_str(&format_shape(s2));
                    msg.push(' ');
                }
                return Err(IonpError::Index(msg));
            }
        }
    }
    Ok(out)
}

/// Python's own tuple repr for a shape: `(2,)`, `(2, 3)`, `()`.
fn format_shape(s: &[usize]) -> String {
    match s.len() {
        0 => "()".to_string(),
        1 => format!("({},)", s[0]),
        _ => format!(
            "({})",
            s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// Materialise a gather: a fresh C-contiguous array of `out_shape` holding
/// `a`'s elements at `offsets`.
pub fn gather(a: &NdArray, out_shape: &[usize], offsets: &[isize]) -> Result<NdArray, IonpError> {
    let n = offsets.len();
    macro_rules! arm {
        ($variant:ident, $v:expr) => {
            Buffer::$variant((0..n).map(|i| $v[offsets[i] as usize]).collect())
        };
    }
    macro_rules! arm_str {
        ($variant:ident, $w:expr, $v:expr) => {
            Buffer::$variant(*$w, (0..n).map(|i| $v[offsets[i] as usize].clone()).collect())
        };
    }
    let buf = match a.buffer() {
        Buffer::Bool(v) => arm!(Bool, v),
        Buffer::I8(v) => arm!(I8, v),
        Buffer::I16(v) => arm!(I16, v),
        Buffer::I32(v) => arm!(I32, v),
        Buffer::I64(v) => arm!(I64, v),
        Buffer::U8(v) => arm!(U8, v),
        Buffer::U16(v) => arm!(U16, v),
        Buffer::U32(v) => arm!(U32, v),
        Buffer::U64(v) => arm!(U64, v),
        Buffer::F16(v) => arm!(F16, v),
        Buffer::F32(v) => arm!(F32, v),
        Buffer::F64(v) => arm!(F64, v),
        Buffer::C64(v) => arm!(C64, v),
        Buffer::C128(v) => arm!(C128, v),
        Buffer::S(w, v) => arm_str!(S, w, v),
        Buffer::U(w, v) => arm_str!(U, w, v),
    };
    NdArray::from_buffer(buf, out_shape.to_vec(), Order::C)
}

/// Scatter `value` into `a` at `offsets`.
///
/// `value` is broadcast to `out_shape` first — the same shape the matching
/// gather would have produced — so `a[[0, 2]] = 5` and `a[[0, 2]] = [5, 6]`
/// both work, and a genuinely unbroadcastable value gets numpy's own
/// "could not broadcast input array from shape (X,) into shape (Y,)".
///
/// `value` must ALREADY be in `a`'s dtype: casting is the caller's job (the
/// Python layer, which owns numpy's assignment-casting rules — unsafe casts,
/// `OverflowError` for out-of-range Python ints, `TypeError` for complex into
/// a real array). Doing it here would put half of those rules in Rust and
/// half in Python, and the half in Rust would be the half with no access to
/// the original Python object that the error messages quote.
///
/// SAFETY / ALIASING: writes go through `ufunc::shared_buffer_mut`, which
/// deliberately bypasses `Arc`'s copy-on-write so that a write through one
/// view is visible through every other alias — see that function's own doc
/// for the full argument and its single-threaded (GIL-holding build)
/// precondition. This function inherits that precondition and adds no new
/// unsafety of its own.
pub fn scatter(a: &NdArray, out_shape: &[usize], offsets: &[isize], value: &NdArray) -> Result<(), IonpError> {
    if value.dtype() != a.dtype() {
        return Err(IonpError::Type(
            "scatter value dtype does not match the target array dtype".to_string(),
        ));
    }
    let vstrides = shape::broadcast_strides_to(value.shape(), value.strides(), out_shape)
        .map_err(|_| {
            IonpError::Value(format!(
                "could not broadcast input array from shape {} into shape {}",
                format_shape(value.shape()),
                format_shape(out_shape)
            ))
        })?;
    let voff = value.offset();
    // Source positions are read into a temporary FIRST when the value could
    // overlap the target. numpy buffers the right-hand side for exactly this
    // reason: `a[1:] = a[:-1]` on `arange(5)` gives `[0, 0, 1, 2, 3]`, not the
    // cascading `[0, 0, 0, 0, 0]` an in-place left-to-right copy produces.
    // Detecting "could overlap" precisely would need provenance tracking; the
    // cheap sound test is buffer identity, which is what `Arc::ptr_eq` gives.
    let overlaps = Arc::ptr_eq(a.buffer_arc(), value.buffer_arc());

    macro_rules! arm {
        ($variant:ident, $src:expr) => {{
            let staged: Option<Vec<_>> = if overlaps {
                Some(NdIter::new(out_shape, &vstrides).map(|o| $src[(voff + o) as usize]).collect())
            } else {
                None
            };
            // SAFETY: see this function's doc comment.
            let dst_buf = unsafe { crate::ufunc::shared_buffer_mut_pub(a.buffer_arc()) };
            if let Buffer::$variant(dst) = dst_buf {
                match staged {
                    Some(vals) => {
                        for (i, &val) in vals.iter().enumerate() {
                            dst[offsets[i] as usize] = val;
                        }
                    }
                    None => {
                        for (i, o) in NdIter::new(out_shape, &vstrides).enumerate() {
                            dst[offsets[i] as usize] = $src[(voff + o) as usize];
                        }
                    }
                }
            } else {
                return Err(IonpError::Type(
                    "scatter target buffer variant does not match the value's".to_string(),
                ));
            }
        }};
    }
    // Same as `arm!` above but `.clone()`s instead of relying on `Copy`,
    // for `S`/`U`'s `Vec<u8>`/`Vec<u32>` per-element storage.
    macro_rules! arm_str {
        ($variant:ident, $src:expr) => {{
            let staged: Option<Vec<_>> = if overlaps {
                Some(NdIter::new(out_shape, &vstrides).map(|o| $src[(voff + o) as usize].clone()).collect())
            } else {
                None
            };
            let dst_buf = unsafe { crate::ufunc::shared_buffer_mut_pub(a.buffer_arc()) };
            if let Buffer::$variant(_, dst) = dst_buf {
                match staged {
                    Some(vals) => {
                        for (i, val) in vals.into_iter().enumerate() {
                            dst[offsets[i] as usize] = val;
                        }
                    }
                    None => {
                        for (i, o) in NdIter::new(out_shape, &vstrides).enumerate() {
                            dst[offsets[i] as usize] = $src[(voff + o) as usize].clone();
                        }
                    }
                }
            } else {
                return Err(IonpError::Type(
                    "scatter target buffer variant does not match the value's".to_string(),
                ));
            }
        }};
    }
    match value.buffer() {
        Buffer::Bool(v) => arm!(Bool, v),
        Buffer::I8(v) => arm!(I8, v),
        Buffer::I16(v) => arm!(I16, v),
        Buffer::I32(v) => arm!(I32, v),
        Buffer::I64(v) => arm!(I64, v),
        Buffer::U8(v) => arm!(U8, v),
        Buffer::U16(v) => arm!(U16, v),
        Buffer::U32(v) => arm!(U32, v),
        Buffer::U64(v) => arm!(U64, v),
        Buffer::F16(v) => arm!(F16, v),
        Buffer::F32(v) => arm!(F32, v),
        Buffer::F64(v) => arm!(F64, v),
        Buffer::C64(v) => arm!(C64, v),
        Buffer::C128(v) => arm!(C128, v),
        Buffer::S(_, v) => arm_str!(S, v),
        Buffer::U(_, v) => arm_str!(U, v),
    }
    Ok(())
}
