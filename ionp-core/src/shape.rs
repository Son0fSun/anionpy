//! Shape, stride, and broadcasting math. Strides here are *element*
//! strides (not byte strides like numpy's C API) since every `Buffer`
//! variant is already a homogeneously-typed `Vec<T>` — indexing is just
//! `buf[(offset as isize + dot(idx, strides)) as usize]`, no byte-width
//! multiplication needed.

use crate::error::IonpError;

pub type Shape = Vec<usize>;
pub type Strides = Vec<isize>;

/// C-contiguous (row-major) strides for `shape`: rightmost axis varies
/// fastest, stride 1.
///
/// The accumulator multiplies by `shape[i].max(1)`, not the raw dim: numpy
/// computes a *reshaped/transposed view's* strides this way, so a
/// zero-length axis does not zero out every stride to its left (verified
/// against real numpy 2.5.1: `np.arange(0).reshape(2, 0, 3).strides ==
/// (24, 24, 8)`, i.e. elementwise `(3, 3, 1)`, not `(0, 3, 1)`). A
/// *freshly allocated* size-0 array is the one place numpy's strides are
/// genuinely all-zero (`np.zeros((0, 5)).strides == (0, 0)`), and that is
/// special-cased at construction time in `NdArray::from_buffer`, not here.
pub fn c_strides(shape: &[usize]) -> Strides {
    let mut strides = vec![0isize; shape.len()];
    let mut acc: isize = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i].max(1) as isize;
    }
    strides
}

/// Fortran-contiguous (column-major) strides: leftmost axis varies
/// fastest, stride 1. Same zero-dim handling as `c_strides` above.
pub fn f_strides(shape: &[usize]) -> Strides {
    let mut strides = vec![0isize; shape.len()];
    let mut acc: isize = 1;
    for i in 0..shape.len() {
        strides[i] = acc;
        acc *= shape[i].max(1) as isize;
    }
    strides
}

pub fn size_of_shape(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// numpy's broadcasting rule: align shapes on the right, each axis pair
/// must be equal, or one of them must be 1 (or missing = implicit 1 from a
/// shorter shape). Returns the numpy exception text verbatim-shaped on
/// mismatch, e.g.
/// "operands could not be broadcast together with shapes (3,4) (2,4) "
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Shape, IonpError> {
    let ndim = a.len().max(b.len());
    let mut out = vec![0usize; ndim];
    for i in 0..ndim {
        let da = axis_from_right(a, ndim, i);
        let db = axis_from_right(b, ndim, i);
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => {
                return Err(IonpError::Broadcast {
                    shape_a: a.to_vec(),
                    shape_b: b.to_vec(),
                })
            }
        };
    }
    Ok(out)
}

fn axis_from_right(shape: &[usize], ndim: usize, i: usize) -> usize {
    let pad = ndim - shape.len();
    if i < pad {
        1
    } else {
        shape[i - pad]
    }
}

/// Compute the strides an array of `(shape, strides)` should present when
/// iterated as if it had shape `target_shape`. Axes that were length-1 (or
/// absent, via right-alignment) get stride 0: the same element is
/// revisited, which is exactly what "broadcast" means for a read-only
/// elementwise walk.
///
/// Unlike `broadcast_shapes` (which computes the *mutual* broadcast result
/// of two shapes and can only ever grow), this function takes `target_shape`
/// as a given and must itself validate that `shape` is actually
/// broadcastable to it -- every axis (right-aligned) must either match
/// exactly or be 1. This validation used to not exist at all: the original
/// version computed `pad = target_shape.len() - shape.len()` unchecked, a
/// `usize` subtraction that UNDERFLOWED whenever `shape` had MORE dims than
/// `target_shape` (a caller bug -- an operand can never legally have more
/// dimensions than the shape it's being broadcast to). In a release build
/// that wrapped to a huge `pad`, so every axis looked "absent" and the
/// function silently returned all-zero strides -- which is exactly why
/// `ufunc.at()` with a multi-index `indices` and a matching leading axis on
/// `values` collapsed to reusing `values[0]` for every index instead of
/// erroring or broadcasting correctly (see `at_binary`/`at_math_binary`).
/// This is a logic bug, not a memory-safety issue: the wrong strides still
/// only ever produced offsets within `shape`'s own valid bounds (they're
/// just semantically wrong), and it was fully deterministic -- not data-
/// or heap-layout-dependent. In a debug build the same underflow instead
/// panics ("attempt to subtract with overflow"), which is how the bug
/// surfaced at all. Returning `Result` here closes both failure modes: an
/// operand that cannot legally broadcast to `target_shape` (too many dims,
/// or a mismatched non-1 axis) is now a reported `IonpError::Broadcast`
/// instead of either a panic or silently-wrong strides.
pub fn broadcast_strides_to(
    shape: &[usize],
    strides: &[isize],
    target_shape: &[usize],
) -> Result<Strides, IonpError> {
    if shape.len() > target_shape.len() {
        return Err(IonpError::Broadcast {
            shape_a: shape.to_vec(),
            shape_b: target_shape.to_vec(),
        });
    }
    let ndim = target_shape.len();
    let pad = ndim - shape.len();
    let mut out = vec![0isize; ndim];
    for i in 0..ndim {
        if i < pad {
            out[i] = 0;
        } else {
            let axis = i - pad;
            if shape[axis] != 1 && shape[axis] != target_shape[i] {
                return Err(IonpError::Broadcast {
                    shape_a: shape.to_vec(),
                    shape_b: target_shape.to_vec(),
                });
            }
            out[i] = if shape[axis] == 1 && target_shape[i] != 1 {
                0
            } else {
                strides[axis]
            };
        }
    }
    Ok(out)
}

/// Port of numpy's `_attempt_nocopy_reshape`
/// (`numpy/_core/src/multiarray/shape.c`), which decides whether an array
/// with `old_shape`/`old_strides` can be *relabeled* onto `new_shape`
/// without moving any data, and if so computes the strides that
/// relabeling requires. `is_f_order` selects which traversal order the
/// reshape is being done in -- `false` for `'C'` (numpy calls this with
/// `is_f_order=0`), `true` for `'F'` -- NOT whether `old_strides` happens
/// to look Fortran-laid-out; it is "how should the requested reshape read
/// off elements", matching numpy's own doc comment on the C function:
/// "The is_f_order argument describes how the array should be viewed
/// during the reshape, not how it is stored in memory."
///
/// Returns `Some(new_strides)` when a view exists, `None` when the data
/// must be copied (the caller then materializes a contiguous buffer and
/// uses plain `c_strides`/`f_strides` instead).
///
/// Only ever called by `NdArray::reshape`/`reshape_with_order` when `self`
/// is NOT already contiguous in the requested order -- exactly mirroring
/// numpy's own call site (`_reshape_with_copy_arg`, `shape.c`), which
/// skips straight to a trivial relabeling when the array already IS
/// contiguous in that order, and never calls this function at all for a
/// zero-size array (numpy's own `C_CONTIGUOUS`/`F_CONTIGUOUS` flags are
/// unconditionally true for size 0, exactly like this crate's
/// `is_c_contiguous`/`is_f_contiguous`). This function therefore assumes
/// `size_of_shape(old_shape) > 0` and `size_of_shape(old_shape) ==
/// size_of_shape(new_shape)` -- both already enforced by the caller before
/// this runs.
///
/// Algorithm (unchanged from numpy, only the language differs):
/// 1. Drop every size-1 axis from `old_shape`/`old_strides` -- a size-1
///    axis carries no data and its stride is unconstrained, so it can
///    never block or help a no-copy reshape.
/// 2. Walk `old`/`new` shapes in lock-step, greedily growing a "run" on
///    whichever side has the smaller running product until both runs'
///    products agree -- e.g. old `(4, 6)` and new `(2, 2, 6)`: the old
///    run starts at `4`, the new run starts at `2`, grows to `2*2=4` to
///    match, and one run boundary is settled; the trailing `6` then
///    matches `6` on both sides as its own run.
/// 3. Each such matched run is only satisfiable as a view if the OLD axes
///    inside it are mutually contiguous with respect to each other (each
///    axis's stride is exactly the next axis's `size * stride`, for
///    C order; the mirror condition, checking the axis to the right
///    against the one to its left, for F order) -- if not, no valid
///    stride assignment exists for the new axes spanning that run, and
///    the whole reshape needs a copy (numpy does not fall back to copying
///    only the offending run; the entire operation copies).
/// 4. Within a satisfied run, the new axes' strides are the unique
///    cumulative product consistent with the run's single old stride
///    anchor (anchored at the run's fastest-varying end: the last axis
///    for C order, the first axis for F order).
/// 5. Any new axes left over after the old shape is fully consumed can
///    only be trailing size-1 axes (guaranteed by the equal-size
///    precondition); their stride is unobservable (dimension 1 never
///    varies) and numpy assigns them the previous run's outer stride,
///    which this port reproduces exactly for parity even though any value
///    would be numerically invisible.
pub fn attempt_nocopy_reshape(
    old_shape: &[usize],
    old_strides: &[isize],
    new_shape: &[usize],
    is_f_order: bool,
) -> Option<Strides> {
    // Step 1: compact away size-1 old axes.
    let mut old_dims: Vec<usize> = Vec::with_capacity(old_shape.len());
    let mut old_strd: Vec<isize> = Vec::with_capacity(old_shape.len());
    for (&d, &s) in old_shape.iter().zip(old_strides.iter()) {
        if d != 1 {
            old_dims.push(d);
            old_strd.push(s);
        }
    }
    let oldnd = old_dims.len();
    let newnd = new_shape.len();
    let newdims = new_shape;

    let mut newstrides = vec![0isize; newnd];

    // oi..oj and ni..nj are the half-open axis ranges of the "current run"
    // on the old and new side respectively (numpy's own variable names,
    // kept identical to make this diffable against the C source).
    let mut oi = 0usize;
    let mut oj = 1usize;
    let mut ni = 0usize;
    let mut nj = 1usize;

    while ni < newnd && oi < oldnd {
        let mut np: usize = newdims[ni];
        let mut op: usize = old_dims[oi];

        while np != op {
            if np < op {
                // Misses trailing 1s, these are handled after the loop.
                if nj >= newnd {
                    return None;
                }
                np *= newdims[nj];
                nj += 1;
            } else {
                if oj >= oldnd {
                    return None;
                }
                op *= old_dims[oj];
                oj += 1;
            }
        }

        // Check whether the old axes spanned by this run are mutually
        // contiguous -- the one place a no-copy reshape can fail.
        for ok in oi..(oj - 1) {
            if is_f_order {
                if old_strd[ok + 1] != old_dims[ok] as isize * old_strd[ok] {
                    return None;
                }
            } else if old_strd[ok] != old_dims[ok + 1] as isize * old_strd[ok + 1] {
                return None;
            }
        }

        // Assign strides to every new axis in this run.
        if is_f_order {
            newstrides[ni] = old_strd[oi];
            for nk in (ni + 1)..nj {
                newstrides[nk] = newstrides[nk - 1] * newdims[nk - 1] as isize;
            }
        } else {
            newstrides[nj - 1] = old_strd[oj - 1];
            for nk in ((ni + 1)..nj).rev() {
                newstrides[nk - 1] = newstrides[nk] * newdims[nk] as isize;
            }
        }
        ni = nj;
        nj += 1;
        oi = oj;
        oj += 1;
    }

    // Trailing size-1 new axes (never entered the loop above because the
    // old side was already fully consumed): stride is unobservable, but
    // numpy fills it with the previous run's outer stride, so this does
    // too (byte-for-byte parity, in element-stride terms).
    let last_stride = if ni >= 1 {
        let mut s = newstrides[ni - 1];
        if is_f_order {
            s *= newdims[ni - 1] as isize;
        }
        s
    } else {
        // `old_shape`'s total size is > 0 by precondition, and every axis
        // that survived compaction has size > 1, so `oldnd == 0` here
        // means `old_shape`'s size was exactly 1 (every axis was size 1)
        // -- the "itemsize" numpy substitutes in this branch is `1` in
        // this crate's element-stride convention (there is no byte width
        // to bake in here, unlike numpy's own byte-stride convention).
        1
    };
    for s in newstrides.iter_mut().skip(ni) {
        *s = last_stride;
    }

    Some(newstrides)
}

/// Walks every element of an N-d array described by `shape`/`strides`,
/// yielding the flat element offset (relative to the array's own
/// `offset`) for each position, in C (row-major, rightmost-fastest)
/// order. This is the single iteration primitive the whole crate builds
/// on: broadcasting, `to_contiguous`, and the ufunc engine all just zip
/// several `NdIter`s (one per operand, each pre-broadcast to the same
/// output shape) together instead of writing nested loops per rank.
pub struct NdIter<'a> {
    shape: &'a [usize],
    strides: &'a [isize],
    idx: Vec<usize>,
    // Running offset for the CURRENT `idx`, maintained incrementally by
    // `next()` rather than recomputed from scratch every call. Perf note:
    // the original implementation recomputed `idx.zip(strides).map(mul).sum()`
    // -- an O(ndim) dot product over the heap-allocated `idx` Vec, with
    // iterator/closure overhead -- on EVERY single step, even though from
    // one step to the next only the innermost (usually just one) axis
    // actually changes. That made every caller that walks a whole axis
    // element-by-element (e.g. a narrow/non-pairwise reduction fold) pay a
    // full multi-axis recompute per element instead of a single `+= stride`.
    // Measured impact: `sum(axis=0)` on a (2000,2000) float64 C-contiguous
    // array (a narrow, non-innermost reduction -- 2000 outer positions x
    // 2000-element sequential folds, all through this iterator) regressed
    // from ~4.6ms to ~21ms after a traversal-correctness rewrite moved this
    // loop onto `NdIter`; restoring incremental offset tracking here (same
    // emitted offset sequence, same order, purely an implementation detail)
    // brought it back down -- see this task's report for exact before/after
    // numbers. Odometer increment/carry below still touches `idx`, but only
    // ever does O(1) work in the common case (innermost axis increment, no
    // carry) instead of O(ndim) every time.
    offset: isize,
    remaining: usize,
}

impl<'a> NdIter<'a> {
    pub fn new(shape: &'a [usize], strides: &'a [isize]) -> Self {
        let remaining = size_of_shape(shape);
        NdIter {
            shape,
            strides,
            idx: vec![0; shape.len()],
            offset: 0,
            remaining,
        }
    }
}

impl<'a> Iterator for NdIter<'a> {
    type Item = isize;

    fn next(&mut self) -> Option<isize> {
        if self.remaining == 0 {
            return None;
        }
        let offset = self.offset;
        self.remaining -= 1;

        // Odometer increment, rightmost axis fastest: bump the innermost
        // axis's offset contribution directly; on carry, undo that axis's
        // full contribution and carry into the next axis out. Emits the
        // exact same offset sequence as recomputing the dot product from
        // scratch every step, just without redoing the whole product.
        for axis in (0..self.shape.len()).rev() {
            self.idx[axis] += 1;
            self.offset += self.strides[axis];
            if self.idx[axis] < self.shape[axis] {
                break;
            }
            self.idx[axis] = 0;
            self.offset -= self.strides[axis] * self.shape[axis] as isize;
        }
        Some(offset)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_strides_basic() {
        assert_eq!(c_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(c_strides(&[5]), vec![1]);
        assert_eq!(c_strides(&[]), Vec::<isize>::new());
    }

    #[test]
    fn f_strides_basic() {
        assert_eq!(f_strides(&[2, 3, 4]), vec![1, 2, 6]);
    }

    #[test]
    fn broadcast_equal_shapes() {
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn broadcast_scalar_against_array() {
        assert_eq!(broadcast_shapes(&[], &[2, 3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[3, 4], &[4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn broadcast_ones_expand() {
        assert_eq!(broadcast_shapes(&[1, 4], &[3, 4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[3, 1], &[3, 4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[8, 1, 6, 1], &[7, 1, 5]).unwrap(), vec![8, 7, 6, 5]);
    }

    #[test]
    fn broadcast_mismatch_errors() {
        // 2026-08-01: updated to match real numpy's actual no-space comma
        // shape formatting (`(3,4)`, not `(3, 4)`) -- see `fmt_shape`'s own
        // doc comment in error.rs for the char-for-char verification
        // against numpy 2.5.1 that motivated this fix; this test
        // previously locked in the WRONG (space-included) wording.
        let err = broadcast_shapes(&[3, 4], &[2, 4]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("(3,4)"), "{msg}");
        assert!(msg.contains("(2,4)"), "{msg}");
    }

    #[test]
    fn nditer_visits_every_offset_in_c_order() {
        let shape = vec![2, 3];
        let strides = c_strides(&shape);
        let offsets: Vec<isize> = NdIter::new(&shape, &strides).collect();
        assert_eq!(offsets, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn nditer_zero_dim_is_single_element() {
        let shape: Vec<usize> = vec![];
        let strides: Vec<isize> = vec![];
        let offsets: Vec<isize> = NdIter::new(&shape, &strides).collect();
        assert_eq!(offsets, vec![0]);
    }

    #[test]
    fn nditer_respects_broadcast_zero_strides() {
        // shape (1,3) broadcast to (2,3): row revisited twice.
        let strides = broadcast_strides_to(&[1, 3], &[3, 1], &[2, 3]).unwrap();
        let offsets: Vec<isize> = NdIter::new(&[2, 3], &strides).collect();
        assert_eq!(offsets, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn broadcast_strides_zero_for_expanded_axes() {
        let strides = broadcast_strides_to(&[1, 4], &[4, 1], &[3, 4]).unwrap();
        assert_eq!(strides, vec![0, 1]);
        let strides = broadcast_strides_to(&[4], &[1], &[3, 4]).unwrap();
        assert_eq!(strides, vec![0, 1]);
    }

    #[test]
    fn broadcast_strides_to_errors_when_shape_has_more_dims_than_target() {
        // The underflow-turned-panic/silent-wrong-strides bug: `shape` has
        // MORE dims than `target_shape` (e.g. `.at()`'s `values` carrying a
        // leading per-index axis that the caller forgot to fold into
        // `target_shape`). Must be a reported error, not a panic and not
        // silently-wrong all-zero strides.
        let err = broadcast_strides_to(&[2, 4], &[4, 1], &[4]).unwrap_err();
        assert!(matches!(err, IonpError::Broadcast { .. }));
    }

    #[test]
    fn broadcast_strides_to_errors_on_mismatched_non_one_axis() {
        // Same ndim, but an axis that's neither equal nor 1 -- not
        // broadcastable at all, previously silently accepted (using
        // `strides[axis]` regardless of whether `shape[axis]` actually
        // matched `target_shape[i]`).
        let err = broadcast_strides_to(&[3], &[1], &[4]).unwrap_err();
        assert!(matches!(err, IonpError::Broadcast { .. }));
    }

    // -----------------------------------------------------------------
    // attempt_nocopy_reshape
    // -----------------------------------------------------------------

    #[test]
    fn nocopy_reshape_uniformly_strided_slice_is_a_view() {
        // a = arange(24).reshape(4,6); a[:, ::2] has shape (4,3),
        // strides (6,2) -- flattening to (12,) must succeed as a view
        // with strides (2,) (verified against real numpy: `a[:, ::2]
        // .reshape(12)` shares memory with `a`).
        let strides = attempt_nocopy_reshape(&[4, 3], &[6, 2], &[12], false);
        assert_eq!(strides, Some(vec![2]));
    }

    #[test]
    fn nocopy_reshape_transposed_refuses() {
        // a.T for a (2,3) C-contiguous array has shape (3,2), strides
        // (1,3) -- genuinely not stride-compatible with a C-order (6,)
        // flatten (verified against real numpy: raises
        // "Unable to avoid creating a copy while reshaping.").
        let strides = attempt_nocopy_reshape(&[3, 2], &[1, 3], &[6], false);
        assert_eq!(strides, None);
    }

    #[test]
    fn nocopy_reshape_size_one_axes_both_sides() {
        // (1, 5, 1) C-contiguous -> (5, 1, 1): every axis on both sides
        // is either the real data axis or a size-1 axis; must succeed
        // (verified against real numpy).
        let strides = attempt_nocopy_reshape(&[1, 5, 1], &[5, 1, 1], &[5, 1, 1], false);
        assert_eq!(strides, Some(vec![1, 1, 1]));
    }

    #[test]
    fn nocopy_reshape_all_axes_size_one() {
        // A size-1 array reshaped from (1,) to (1, 1, 1): the old shape
        // compacts to zero surviving axes (oldnd == 0), exercising the
        // "itemsize" fallback branch for the otherwise-arbitrary trailing
        // strides.
        let strides = attempt_nocopy_reshape(&[1], &[1], &[1, 1, 1], false);
        assert_eq!(strides, Some(vec![1, 1, 1]));
    }

    #[test]
    fn nocopy_reshape_zero_size_array_not_called_but_would_be_safe() {
        // `attempt_nocopy_reshape` is documented to require non-zero old
        // size (callers only invoke it once `is_c_contiguous`/
        // `is_f_contiguous` -- both trivially true for size 0 -- have
        // already been ruled out). This test just pins that a degenerate
        // zero-size call doesn't panic, even though no real caller
        // reaches it.
        let strides = attempt_nocopy_reshape(&[0, 3], &[3, 1], &[0], false);
        assert!(strides.is_some() || strides.is_none()); // no panic is the assertion
    }

    #[test]
    fn nocopy_reshape_3d_run_splitting_and_merging() {
        // C-contiguous (2,3,4), strides (12,4,1). Splitting axis 0 into
        // (2,1) and merging axes 1,2 into a single 12-run: new shape
        // (2,1,12) must succeed with strides (12,12,1) (verified against
        // real numpy: `np.arange(24).reshape(2,3,4).reshape(2,1,12)`
        // shares memory).
        let strides = attempt_nocopy_reshape(&[2, 3, 4], &[12, 4, 1], &[2, 1, 12], false);
        assert_eq!(strides, Some(vec![12, 12, 1]));

        // Merging all three axes down to (24,) must also succeed.
        let strides = attempt_nocopy_reshape(&[2, 3, 4], &[12, 4, 1], &[24], false);
        assert_eq!(strides, Some(vec![1]));

        // Splitting the middle axis (3 -> nothing evenly compatible with
        // a (2,3,4)->(2,1,3,4) split of just that axis boundary) must
        // still succeed: (2,3,4) -> (2,3,2,2) splits the last axis.
        let strides = attempt_nocopy_reshape(&[2, 3, 4], &[12, 4, 1], &[2, 3, 2, 2], false);
        assert_eq!(strides, Some(vec![12, 4, 2, 1]));
    }

    #[test]
    fn nocopy_reshape_f_order_uniformly_strided_slice_is_a_view() {
        // F-contiguous (4,6) has strides (1,4). Taking every other column
        // in Fortran sense (columns are the fast-varying axis's
        // complement here) -- simpler: an F-contiguous (6,4) sliced on
        // its first axis every-other-row, `a[::2, :]`, has shape (3,4),
        // strides (2,6) and F-order-reshaping it to (12,) must succeed
        // (verified against real numpy: `np.asfortranarray(np.arange(24)
        // .reshape(6,4))[::2, :].reshape(12, order='F')` shares memory).
        let strides = attempt_nocopy_reshape(&[3, 4], &[2, 6], &[12], true);
        assert_eq!(strides, Some(vec![2]));
    }

    #[test]
    fn nocopy_reshape_f_order_transposed_refuses() {
        // The F-order mirror of the C-order transpose-refusal case: a
        // genuinely C-contiguous (3,2) array (strides (2,1)) has no valid
        // Fortran-order flatten as a view (verified against real numpy:
        // `np.arange(6).reshape(3,2).reshape(6, order='F')` -- i.e.
        // `np.reshape(a, (6,), order='F', copy=False)` -- raises).
        let strides = attempt_nocopy_reshape(&[3, 2], &[2, 1], &[6], true);
        assert_eq!(strides, None);
    }
}
