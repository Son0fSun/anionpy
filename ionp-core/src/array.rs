//! `NdArray`: owned buffer + shape + strides + dtype + offset, the same
//! four-tuple numpy itself uses to describe both fresh arrays and views.
//! Slicing, transpose, and (when possible) reshape all return a new
//! `NdArray` that shares the *same* `Arc<Buffer>` — no data copy, matching
//! numpy's view semantics, including that writes through one view would
//! be visible through another (mutation is not implemented yet; today
//! every array is logically immutable after construction, see
//! KNOWN-DIFFERENCES.md).

use std::sync::Arc;

use crate::buffer::Buffer;
use crate::dtype::DType;
use crate::error::IonpError;
use crate::shape::{self, NdIter, Shape, Strides};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    C,
    F,
}

#[derive(Debug, Clone)]
pub struct NdArray {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) offset: isize,
}

impl NdArray {
    /// Build a fresh, owned, contiguous array from a `Buffer` whose length
    /// must equal the product of `shape`.
    pub fn from_buffer(buffer: Buffer, shape: Shape, order: Order) -> Result<Self, IonpError> {
        let expected = shape::size_of_shape(&shape);
        if buffer.len() != expected {
            return Err(IonpError::Value(format!(
                "buffer of length {} cannot be interpreted as shape {:?}",
                buffer.len(),
                shape
            )));
        }
        let mut strides = match order {
            Order::C => shape::c_strides(&shape),
            Order::F => shape::f_strides(&shape),
        };
        // A freshly allocated size-0 array gets all-zero strides in real
        // numpy (`np.zeros((0, 5)).strides == (0, 0)`), unlike a reshape/
        // transpose view of an existing array into a size-0 shape (which
        // keeps the `shape::c_strides`/`f_strides` formula's result --
        // see those functions' docs). This is the one place that
        // "freshly allocated" distinction is real, so it is the one place
        // it is applied.
        if expected == 0 {
            strides.iter_mut().for_each(|s| *s = 0);
        }
        Ok(NdArray {
            buffer: Arc::new(buffer),
            shape,
            strides,
            offset: 0,
        })
    }

    /// Build a fresh, owned array from a `Buffer` with EXPLICIT strides,
    /// skipping the C/F-only `Order` enum entirely -- for callers that have
    /// already computed a specific stride permutation themselves (today:
    /// the `zeros_like`/`ones_like`/`empty_like`/`full_like` family's
    /// `order='K'`/`'A'` support in `ionp-py/src/creation.rs`, matching a
    /// *different* array's memory layout via `axis_perm_for_order` below).
    pub fn from_buffer_with_strides(
        buffer: Buffer,
        shape: Shape,
        strides: Strides,
    ) -> Result<Self, IonpError> {
        let expected = shape::size_of_shape(&shape);
        if buffer.len() != expected {
            return Err(IonpError::Value(format!(
                "buffer of length {} cannot be interpreted as shape {:?}",
                buffer.len(),
                shape
            )));
        }
        if strides.len() != shape.len() {
            return Err(IonpError::Value(
                "strides length must match shape length".to_string(),
            ));
        }
        Ok(NdArray {
            buffer: Arc::new(buffer),
            shape,
            strides,
            offset: 0,
        })
    }

    /// Build an `NdArray` from an ALREADY-COMPUTED `(buffer, shape, strides,
    /// offset)` four-tuple, with none of `from_buffer`/
    /// `from_buffer_with_strides`'s "buffer length must equal
    /// `size_of_shape(shape)`" assumption. That assumption is exactly right
    /// for a fresh, densely-packed allocation, but wrong for a buffer that
    /// preserves a NON-contiguous source's true memory span (e.g. ingesting
    /// a strided numpy view: the buffer must cover every address between
    /// the view's own extreme reachable offsets, which is generally LARGER
    /// than the number of elements the view's shape actually visits -- the
    /// same "buffer bigger than what `shape` alone would imply" situation
    /// every internal view (`get_view`, transpose, ...) already relies on
    /// via `Arc`-shared buffers, just constructed directly here instead of
    /// derived from an existing `NdArray`. Only checks that `strides` and
    /// `shape` agree in rank; out-of-bounds `(shape, strides, offset)`
    /// combinations are the caller's responsibility, exactly as they are
    /// for every other direct field-setting constructor in this file.
    pub fn from_raw_layout(
        buffer: Buffer,
        shape: Shape,
        strides: Strides,
        offset: isize,
    ) -> Result<Self, IonpError> {
        if strides.len() != shape.len() {
            return Err(IonpError::Value(
                "strides length must match shape length".to_string(),
            ));
        }
        Ok(NdArray {
            buffer: Arc::new(buffer),
            shape,
            strides,
            offset,
        })
    }

    pub fn dtype(&self) -> DType {
        self.buffer.dtype()
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn size(&self) -> usize {
        shape::size_of_shape(&self.shape)
    }
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Returns a copy of this array with all strides forced to 0,
    /// keeping the same buffer, shape, and offset. Real numpy gives a
    /// freshly-allocated size-0 array all-zero strides regardless of
    /// requested order (verified directly against numpy 2.5.1). This is
    /// a narrow, explicit public escape hatch for `ionp-py` call sites
    /// (`strides`/`shape`/`offset`/`buffer` are `pub(crate)` to
    /// `ionp-core`, not visible cross-crate) that need to apply that
    /// exact rule to a result they've already computed some other way,
    /// without exposing field mutation more broadly or forcing them to
    /// route through a specific allocation primitive.
    pub fn zero_strides(&self) -> NdArray {
        NdArray {
            buffer: Arc::clone(&self.buffer),
            shape: self.shape.clone(),
            strides: vec![0isize; self.shape.len()],
            offset: self.offset,
        }
    }
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// The `Arc` itself, not the `Buffer` behind it. `indexing::scatter`
    /// needs it for two things `&Buffer` cannot express: `Arc::ptr_eq`, to
    /// decide whether an assignment's source and target are the same
    /// allocation (`a[1:] = a[:-1]` must buffer, per numpy), and the
    /// write-through mutation path that deliberately bypasses `Arc`'s
    /// copy-on-write so a write lands in every live alias.
    pub fn buffer_arc(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    /// Mutable, copy-on-write access to the underlying `Buffer`, for callers
    /// (e.g. `ionp-py`'s `shuffle`) that need true in-place mutation and
    /// live outside this crate, so cannot reach the `pub(crate) buffer`
    /// field or call `Arc::make_mut` on it directly themselves. Mirrors the
    /// exact `Arc::make_mut(&mut a.buffer)` pattern already used internally
    /// in `manip.rs`: if this `NdArray`'s buffer `Arc` is uniquely owned,
    /// mutates in place; if shared with another alias, clones first so the
    /// alias is left untouched (standard COW).
    pub fn buffer_mut(&mut self) -> &mut Buffer {
        Arc::make_mut(&mut self.buffer)
    }

    pub fn is_c_contiguous(&self) -> bool {
        // numpy treats every size-0 array as trivially both C- and
        // F-contiguous regardless of its strides (verified:
        // `np.zeros((0, 5)).flags['C_CONTIGUOUS'] is True`, and so is
        // `np.arange(0).reshape(0, 5).flags['C_CONTIGUOUS']` even though
        // that view's strides don't match the plain `c_strides` formula).
        //
        // BUG FOUND AND FIXED 2026-08-01: the previous check was a blunt
        // `self.strides == shape::c_strides(&self.shape)` -- an EXACT
        // stride-vector match. numpy's real definition ignores the stride
        // of any axis whose length is 1 (or 0), since a size-1 axis is
        // walked exactly once regardless of its stride, so that stride can
        // never actually be observed. Concretely: `np.arange(6).reshape(2,
        // 3)[::2]` has shape `(1, 3)` but a non-canonical row-stride
        // (inherited from the original 2-row buffer, not `3`) -- and real
        // numpy still reports `C_CONTIGUOUS=True` for it, while the old
        // exact-match check here reported `False`. That false negative
        // made `ionp.asarray(that_view, order='C', copy=False)` wrongly
        // raise `ValueError` where real numpy succeeds -- found via
        // independent byte-exact differential probing (not a scenario any
        // existing test constructed), not part of the originally assigned
        // gap list. Fixed by walking axes fastest-to-slowest and only
        // requiring the expected stride to match once TWO OR MORE
        // consecutive non-unit-length axes have been seen (the standard
        // "collapse size-1 axes" contiguity rule numpy's own
        // `PyArray_UpdateFlags` uses).
        // BUG FOUND AND FIXED 2026-08-02: the previous check additionally
        // required `self.offset == 0` -- but numpy's contiguity flags are a
        // PURE function of shape/strides/itemsize; the base offset into the
        // underlying allocation is irrelevant (verified against real numpy
        // 2.5.1: `np.arange(12).reshape(3,4).astype('f8')[1:3]` has shape
        // (2,4), strides (32,8) -- canonical C-strides -- and numpy reports
        // `C_CONTIGUOUS=True` despite starting at a nonzero byte offset
        // into the original buffer). Dropped the `self.offset == 0` gate;
        // `strides_match_ignoring_unit_axes` (shape+strides only) is
        // numpy's actual rule. See ionp/docs/contiguity-flags-fix.md.
        self.size() == 0 || strides_match_ignoring_unit_axes(&self.shape, &self.strides, true)
    }
    pub fn is_f_contiguous(&self) -> bool {
        self.size() == 0 || strides_match_ignoring_unit_axes(&self.shape, &self.strides, false)
    }

    /// Whether `self` and `other` are backed by the literal same
    /// allocation (`Arc::ptr_eq` on the shared buffer) -- i.e. whether one
    /// is a real VIEW of the other (same data, no copy), not merely
    /// value-equal. Used by `ionp-py`'s `copy=` tri-state handling
    /// (`reshape`/`asarray`'s numpy-2.x `copy=False` must raise when a
    /// requested transform can't avoid allocating a fresh buffer) -- the
    /// `buffer: Arc<Buffer>` field itself is `pub(crate)` (not visible
    /// outside `ionp-core`), so this is the narrow public accessor for
    /// that one bit of information rather than widening the field's own
    /// visibility.
    pub fn shares_buffer_with(&self, other: &NdArray) -> bool {
        Arc::ptr_eq(&self.buffer, &other.buffer)
    }

    /// Inclusive `[lo, hi]` range of element indices (element units, not
    /// bytes) that `self` can possibly touch inside its shared `Arc<Buffer>`
    /// allocation -- i.e. `offset` plus, per axis, whichever of `0` or
    /// `stride * (dim - 1)` is more negative/positive. `None` for a
    /// size-0 array (no elements are touched at all -- numpy's own
    /// `shares_memory`/`may_share_memory` always return `False` for an
    /// empty operand, verified against real numpy 2.5.1: `np.shares_memory(
    /// np.empty(0), np.empty(0))` is `False` even when both are literally
    /// the same object).
    ///
    /// This is a BOUNDS check, not an exact reachable-element check: two
    /// arrays whose bounds overlap are not guaranteed to actually share an
    /// element (e.g. `a[0::2]` and `a[1::2]` have overlapping bounds but
    /// touch disjoint offsets). `may_share_memory_with` below is exactly
    /// this bounds check (matching numpy's own conservative
    /// `may_share_memory`, which is documented to over-report). Callers
    /// needing exact `np.shares_memory` semantics must additionally verify
    /// the overlap is real for their specific access pattern before relying
    /// on this.
    pub fn memory_extent(&self) -> Option<(isize, isize)> {
        if self.size() == 0 {
            return None;
        }
        let mut lo = self.offset;
        let mut hi = self.offset;
        for (&dim, &stride) in self.shape.iter().zip(self.strides.iter()) {
            if dim <= 1 {
                continue;
            }
            let reach = stride * (dim as isize - 1);
            if reach >= 0 {
                hi += reach;
            } else {
                lo += reach;
            }
        }
        Some((lo, hi))
    }

    /// Conservative (bounds-based) memory-overlap test, matching numpy's
    /// `may_share_memory` semantics: `True` whenever the two arrays' element
    /// ranges within the SAME underlying allocation could possibly overlap.
    /// Different allocations (`Arc::ptr_eq` false) can never overlap in
    /// ionp, since ionp buffers are never constructed to alias a foreign
    /// allocation the way ctypes/mmap tricks can in real numpy.
    pub fn may_share_memory_with(&self, other: &NdArray) -> bool {
        if !self.shares_buffer_with(other) {
            return false;
        }
        match (self.memory_extent(), other.memory_extent()) {
            (Some((lo1, hi1)), Some((lo2, hi2))) => lo1 <= hi2 && lo2 <= hi1,
            _ => false,
        }
    }

    /// Iterator over this array's element offsets in logical (C, i.e.
    /// last-axis-fastest) order — the one place `NdIter` is driven
    /// directly by an `NdArray` rather than by the ufunc engine.
    pub fn iter_offsets(&self) -> NdIter<'_> {
        NdIter::new(&self.shape, &self.strides)
    }

    /// Force this array's buffer to be genuinely independent of any other
    /// `NdArray`/`Arc<Buffer>` clone that may currently share the exact
    /// same allocation, WITHOUT touching shape/strides/offset (unlike
    /// `to_contiguous`, which also relayouts to C-order). A cheap
    /// strong-count check (no-op, no allocation) when this array already
    /// is the sole owner -- the common case for any freshly-computed
    /// array -- and a real, eager buffer clone otherwise.
    ///
    /// Added 2026-08-03 for `ionp.array(x)` on an already-`ionp.ndarray`
    /// `x`: `array_impl`'s fast path for that source kind is a lazy
    /// `Arc::clone` (`arr.inner.clone()`, `lib.rs`), not a real copy --
    /// deliberately cheap, since most callers never write to the result.
    /// But `ionp.array(...)`'s documented contract (numpy-matching, unlike
    /// `asarray`) is an INDEPENDENT copy: `ionp.shares_memory(x,
    /// ionp.array(x))` must be `false`, immediately, not just "safe once
    /// written to" -- a caller inspecting identity/aliasing before ever
    /// writing anything must already see two unrelated arrays. This is also
    /// what makes `ufunc::write_out` safe to write through a shared buffer
    /// UNCONDITIONALLY again (see that function's own doc): by calling this
    /// eagerly, right here, `array()` (`ionp-py/src/lib.rs`) guarantees no
    /// `write_out` call ever again sees a `array()`-produced "copy"
    /// pretending to still share its source's buffer -- the ambiguity
    /// `write_out` used to have to (and couldn't correctly) resolve at
    /// write time is resolved here, once, at construction time instead.
    pub fn detach_buffer(&mut self) {
        Arc::make_mut(&mut self.buffer);
    }

    /// A new array holding the same data as `self`, laid out C-contiguous
    /// from offset 0. No-op copy avoidance: if already C-contiguous,
    /// still copies today (buffer sharing for the no-op case is a cheap
    /// follow-up, not attempted here to keep the invariant "to_contiguous
    /// always returns owned-fresh" simple to reason about).
    pub fn to_contiguous(&self) -> NdArray {
        macro_rules! gather {
            ($variant:ident, $v:expr) => {{
                let mut out = Vec::with_capacity(self.size());
                for off in self.iter_offsets() {
                    out.push($v[(self.offset + off) as usize].clone());
                }
                Buffer::$variant(out)
            }};
        }
        macro_rules! gather_str {
            ($variant:ident, $w:expr, $v:expr) => {{
                let mut out = Vec::with_capacity(self.size());
                for off in self.iter_offsets() {
                    out.push($v[(self.offset + off) as usize].clone());
                }
                Buffer::$variant(*$w, out)
            }};
        }
        let new_buffer = match &*self.buffer {
            Buffer::Bool(v) => gather!(Bool, v),
            Buffer::I8(v) => gather!(I8, v),
            Buffer::I16(v) => gather!(I16, v),
            Buffer::I32(v) => gather!(I32, v),
            Buffer::I64(v) => gather!(I64, v),
            Buffer::U8(v) => gather!(U8, v),
            Buffer::U16(v) => gather!(U16, v),
            Buffer::U32(v) => gather!(U32, v),
            Buffer::U64(v) => gather!(U64, v),
            Buffer::F16(v) => gather!(F16, v),
            Buffer::F32(v) => gather!(F32, v),
            Buffer::F64(v) => gather!(F64, v),
            Buffer::C64(v) => gather!(C64, v),
            Buffer::C128(v) => gather!(C128, v),
            Buffer::S(w, v) => gather_str!(S, w, v),
            Buffer::U(w, v) => gather_str!(U, w, v),
        };
        NdArray {
            strides: shape::c_strides(&self.shape),
            buffer: Arc::new(new_buffer),
            shape: self.shape.clone(),
            offset: 0,
        }
    }

    /// Elementwise cast to `dst`, materializing a new contiguous buffer
    /// (matching numpy's `astype`, which always copies).
    pub fn cast_to(&self, dst: DType) -> NdArray {
        if self.dtype() == dst && self.is_c_contiguous() {
            return self.clone();
        }
        let contig = if self.is_c_contiguous() { self.clone() } else { self.to_contiguous() };
        NdArray {
            buffer: Arc::new(contig.buffer.cast_to(dst)),
            shape: contig.shape,
            strides: contig.strides,
            // `contig.offset`, NOT 0. `Buffer::cast_to` maps elementwise
            // over the WHOLE backing vec and returns one of the same
            // length -- it does not slice down to this view's window. So
            // the offset is still live in the result and discarding it
            // re-reads the array from element 0.
            //
            // The `is_c_contiguous()` arm above is what makes this
            // reachable: a plain offset slice like `a[2:5]` IS contiguous
            // (shape (3,), strides (1,)) and merely carries `offset: 2`,
            // so it takes the cheap `self.clone()` path with its offset
            // intact and then had that offset thrown away here. Views
            // that are NOT contiguous were always correct, because
            // `to_contiguous()` gathers through the offset and genuinely
            // rebases to 0 -- which is exactly why this hid for so long:
            // reversed, transposed and strided views all behaved, and
            // only the *simplest* kind of view was wrong.
            //
            // Hardcoding 0 was silently wrong for every caller, not just
            // `copyto`: `a[2:5].astype('int8')`, `asarray(a[2:5],
            // dtype=...)` and `array(a[2:5], dtype=...)` all returned the
            // wrong ELEMENTS -- right dtype, right shape, right strides,
            // wrong data. Found 2026-08-03 by copyto's out-of-corpus
            // sweep; see the `castoff/` cases in
            // tests/differential/copyto_cases.py, which fail without this
            // line.
            offset: contig.offset,
        }
    }

    /// View with axes reversed — always possible without copying,
    /// regardless of contiguity, since transpose is nothing but relabeling
    /// which stride goes with which axis.
    pub fn transpose(&self) -> NdArray {
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.reverse();
        strides.reverse();
        NdArray {
            buffer: Arc::clone(&self.buffer),
            shape,
            strides,
            offset: self.offset,
        }
    }

    /// numpy `reshape`: returns a view when the data can be relabeled
    /// without moving it. `self` being C-contiguous is the cheap common
    /// case (fresh arrays, and slices/transposes that happen to still be
    /// C-contiguous) and is handled directly; otherwise this tries numpy's
    /// own no-copy-reshape search (`shape::attempt_nocopy_reshape`, a port
    /// of `_attempt_nocopy_reshape` from numpy's `shape.c`), which finds a
    /// valid view whenever the new shape is stride-compatible with the
    /// old one -- e.g. a uniformly-strided slice (`a[:, ::2]`), or
    /// regrouping/splitting a run of mutually-contiguous axes. Only when
    /// that search also fails (e.g. a genuine transpose) does this fall
    /// back to a copy, exactly like numpy's own `reshape` does for input
    /// it likewise cannot view.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<NdArray, IonpError> {
        self.reshape_impl(new_shape, true)
    }

    /// Identical to `reshape` except it never takes the §3a identity
    /// short-circuit. numpy's own identity check (`_reshape_with_copy_arg`,
    /// shape.c ~237-247) compares the RAW, unresolved `dimensions[i]`
    /// values the caller passed -- which, for a `-1` (or any negative)
    /// "infer this axis" marker, can never equal the corresponding old
    /// dimension, even though the *resolved* value numerically does. So
    /// `a.reshape(-1)` on an array whose flattened shape already equals its
    /// current shape does NOT take numpy's identity fast path: it falls
    /// through to a fresh view-construction-formula stride computation
    /// (verified against real numpy 2.5.1: `ma.masked_array([]).reshape(-1)`
    /// -- `.strides == (8,)`, not the source's `(0,)`). Callers that resolve
    /// `-1`/negative dims themselves (the PyO3 boundary in `lib.rs`'s
    /// `PyArray::reshape` and `creation.rs`'s `reshape`) must route through
    /// this entry point instead of `reshape` whenever any raw requested dim
    /// was negative. Every purely-internal call site in this crate always
    /// passes concrete, already-resolved literal shapes (never `-1`), so
    /// `reshape`'s identity short-circuit remains exactly correct for them.
    pub fn reshape_no_identity_shortcut(&self, new_shape: &[usize]) -> Result<NdArray, IonpError> {
        self.reshape_impl(new_shape, false)
    }

    fn reshape_impl(&self, new_shape: &[usize], identity_shortcut: bool) -> Result<NdArray, IonpError> {
        let new_shape = resolve_unknown_dim(new_shape, self.size())?;
        if shape::size_of_shape(&new_shape) != self.size() {
            return Err(IonpError::Reshape {
                size: self.size(),
                shape: new_shape,
            });
        }
        // numpy `_reshape_with_copy_arg` (shape.c ~237-247): if the
        // requested shape is elementwise identical to the current shape,
        // return an unchanged view -- `PyArray_View(array, NULL, NULL)` --
        // BEFORE any order-specific algorithm runs. The source array's own
        // strides are kept untouched, not recomputed. See
        // docs/NUMPY-STRIDE-ORDER-SPEC.md §3a. Skipped entirely when
        // `identity_shortcut` is false -- see `reshape_no_identity_shortcut`.
        if identity_shortcut && new_shape == self.shape {
            return Ok(NdArray {
                buffer: Arc::clone(&self.buffer),
                strides: self.strides.clone(),
                shape: new_shape,
                offset: self.offset,
            });
        }
        if self.is_c_contiguous() {
            return Ok(NdArray {
                buffer: Arc::clone(&self.buffer),
                strides: shape::c_strides(&new_shape),
                shape: new_shape,
                offset: self.offset,
            });
        }
        if let Some(strides) =
            shape::attempt_nocopy_reshape(&self.shape, &self.strides, &new_shape, false)
        {
            return Ok(NdArray {
                buffer: Arc::clone(&self.buffer),
                strides,
                shape: new_shape,
                offset: self.offset,
            });
        }
        let contig = self.to_contiguous();
        Ok(NdArray {
            strides: shape::c_strides(&new_shape),
            buffer: contig.buffer,
            shape: new_shape,
            offset: 0,
        })
    }

    /// numpy `reshape(..., order=...)`. `'C'` is the plain `reshape` above.
    /// `'F'` reads/writes elements in Fortran (first-axis-fastest) order:
    /// when `self` is already F-contiguous this is a cheap direct
    /// relabeling (mirroring `reshape`'s own C-contiguous fast path);
    /// otherwise it runs the SAME `shape::attempt_nocopy_reshape` search
    /// `reshape` uses, just with `is_f_order=true` -- numpy's own
    /// `_attempt_nocopy_reshape` is one function parameterized by
    /// traversal order, not two, so this mirrors that directly rather than
    /// composing it out of a transpose (a transpose-based composition
    /// would also be numerically correct here -- reversing both shape and
    /// strides turns the F-order contiguity check into the C-order one --
    /// but doubles the number of places the algorithm could be gotten
    /// subtly wrong, for no benefit once the direct port exists). Only
    /// once that search also fails does this gather into a fresh
    /// Fortran-ordered buffer (`to_contiguous_order("F")`, already proven
    /// correct) and relabel onto `new_shape` with plain F-strides -- valid
    /// because a buffer physically laid out in "read `self` in Fortran
    /// order" IS, by construction, the same flat sequence `new_shape`'s
    /// own Fortran traversal must read back. `'A'` means `'F'` when
    /// `self` is Fortran-contiguous (and not also C-contiguous, matching
    /// numpy's own tie-break — a 1-D or 0-D array is both, and either path
    /// gives the same answer there), `'C'` otherwise.
    pub fn reshape_with_order(&self, new_shape: &[usize], order: &str) -> Result<NdArray, IonpError> {
        self.reshape_with_order_impl(new_shape, order, true)
    }

    /// Identical to `reshape_with_order` except it never takes the §3a
    /// identity short-circuit, in any of the `"C"`/`"F"`/`"A"` arms -- see
    /// `reshape_no_identity_shortcut` for why this is needed at the
    /// PyO3-boundary call sites that resolve a raw `-1`/negative requested
    /// dim themselves.
    pub fn reshape_with_order_no_identity_shortcut(
        &self,
        new_shape: &[usize],
        order: &str,
    ) -> Result<NdArray, IonpError> {
        self.reshape_with_order_impl(new_shape, order, false)
    }

    fn reshape_with_order_impl(
        &self,
        new_shape: &[usize],
        order: &str,
        identity_shortcut: bool,
    ) -> Result<NdArray, IonpError> {
        match order {
            "C" => self.reshape_impl(new_shape, identity_shortcut),
            "F" => {
                let new_shape = resolve_unknown_dim(new_shape, self.size())?;
                if shape::size_of_shape(&new_shape) != self.size() {
                    return Err(IonpError::Reshape {
                        size: self.size(),
                        shape: new_shape,
                    });
                }
                // Same §3a identity short-circuit as the 'C' path above --
                // numpy resolves this before any order-specific dispatch,
                // regardless of which order was requested (the requested
                // order is never consulted when the shape didn't change).
                // Skipped when `identity_shortcut` is false.
                if identity_shortcut && new_shape == self.shape {
                    return Ok(NdArray {
                        buffer: Arc::clone(&self.buffer),
                        strides: self.strides.clone(),
                        shape: new_shape,
                        offset: self.offset,
                    });
                }
                if self.is_f_contiguous() {
                    return Ok(NdArray {
                        buffer: Arc::clone(&self.buffer),
                        strides: shape::f_strides(&new_shape),
                        shape: new_shape,
                        offset: self.offset,
                    });
                }
                if let Some(strides) =
                    shape::attempt_nocopy_reshape(&self.shape, &self.strides, &new_shape, true)
                {
                    return Ok(NdArray {
                        buffer: Arc::clone(&self.buffer),
                        strides,
                        shape: new_shape,
                        offset: self.offset,
                    });
                }
                let contig = self.to_contiguous_order("F")?;
                Ok(NdArray {
                    strides: shape::f_strides(&new_shape),
                    buffer: contig.buffer,
                    shape: new_shape,
                    offset: 0,
                })
            }
            "A" => {
                if self.is_f_contiguous() && !self.is_c_contiguous() {
                    self.reshape_with_order_impl(new_shape, "F", identity_shortcut)
                } else {
                    self.reshape_impl(new_shape, identity_shortcut)
                }
            }
            // numpy: 'K' is a recognized order letter but is explicitly
            // refused for reshaping specifically (verified against real
            // numpy 2.5.1: `np.reshape(a, shape, order='K')` raises
            // `ValueError: order 'K' is not permitted for reshaping`, while
            // `a.ravel(order='K')`/`a.copy(order='K')` etc. accept it fine
            // -- see `axis_perm_for_order`/`ravel_order`/`to_contiguous_order`
            // below, which is where 'K' actually gets its semantics for
            // those other call sites).
            "K" => Err(IonpError::Value(
                "order 'K' is not permitted for reshaping".to_string(),
            )),
            other => Err(IonpError::Value(format!(
                "order must be one of 'C', 'F', 'A', or 'K' (got '{other}')"
            ))),
        }
    }

    /// The axis order (slowest-to-fastest varying, i.e. axis 0 of the
    /// permutation is the outermost loop) that `order` selects, for the
    /// two primitives below (`ravel_order`, `to_contiguous_order`) that
    /// both need it: 'C' is axes in original order, 'F' is axes reversed,
    /// 'A' picks whichever of those two matches numpy's tie-break (F only
    /// when F-contiguous and not also C-contiguous), and 'K' sorts axes by
    /// descending stride magnitude -- "the order the elements already sit
    /// in memory" (verified against real numpy 2.5.1 via
    /// `PyArray_CreateSortedStridePerm`'s observable behavior: a stable
    /// descending sort on `abs(stride)`, ties (including broadcast/size-1
    /// axes, which carry stride 0) keeping the original relative axis
    /// order). Returns numpy's exact `ValueError` text for an unrecognized
    /// letter, shared with `reshape_with_order`'s own fallback arm.
    pub fn axis_perm_for_order(&self, order: &str) -> Result<Vec<usize>, IonpError> {
        let ndim = self.ndim();
        match order {
            "C" => Ok((0..ndim).collect()),
            "F" => Ok((0..ndim).rev().collect()),
            "A" => {
                if self.is_f_contiguous() && !self.is_c_contiguous() {
                    Ok((0..ndim).rev().collect())
                } else {
                    Ok((0..ndim).collect())
                }
            }
            "K" => {
                let mut perm: Vec<usize> = (0..ndim).collect();
                perm.sort_by(|&a, &b| {
                    self.strides[b].unsigned_abs().cmp(&self.strides[a].unsigned_abs())
                });
                Ok(perm)
            }
            other => Err(IonpError::Value(format!(
                "order must be one of 'C', 'F', 'A', or 'K' (got '{other}')"
            ))),
        }
    }

    /// Gather this array's elements into a fresh, densely packed buffer, in
    /// the traversal order given by `perm` (a permutation of `0..ndim`,
    /// slowest-to-fastest -- see `axis_perm_for_order`). This is the one
    /// element-walk shared by `ravel_order` (which just wraps the result in
    /// a 1-D shape) and `to_contiguous_order` (which relabels it back onto
    /// `self`'s own shape with order-appropriate strides); both need
    /// exactly this sequence of values and nothing else.
    fn gather_by_perm(&self, perm: &[usize]) -> Buffer {
        let permuted_shape: Vec<usize> = perm.iter().map(|&a| self.shape[a]).collect();
        let permuted_strides: Vec<isize> = perm.iter().map(|&a| self.strides[a]).collect();
        let it = NdIter::new(&permuted_shape, &permuted_strides);
        macro_rules! gather {
            ($variant:ident, $v:expr) => {{
                let mut out = Vec::with_capacity(self.size());
                for off in it {
                    out.push($v[(self.offset + off) as usize].clone());
                }
                Buffer::$variant(out)
            }};
        }
        macro_rules! gather_str {
            ($variant:ident, $w:expr, $v:expr) => {{
                let mut out = Vec::with_capacity(self.size());
                for off in it {
                    out.push($v[(self.offset + off) as usize].clone());
                }
                Buffer::$variant(*$w, out)
            }};
        }
        match &*self.buffer {
            Buffer::Bool(v) => gather!(Bool, v),
            Buffer::I8(v) => gather!(I8, v),
            Buffer::I16(v) => gather!(I16, v),
            Buffer::I32(v) => gather!(I32, v),
            Buffer::I64(v) => gather!(I64, v),
            Buffer::U8(v) => gather!(U8, v),
            Buffer::U16(v) => gather!(U16, v),
            Buffer::U32(v) => gather!(U32, v),
            Buffer::U64(v) => gather!(U64, v),
            Buffer::F16(v) => gather!(F16, v),
            Buffer::F32(v) => gather!(F32, v),
            Buffer::F64(v) => gather!(F64, v),
            Buffer::C64(v) => gather!(C64, v),
            Buffer::C128(v) => gather!(C128, v),
            Buffer::S(w, v) => gather_str!(S, w, v),
            Buffer::U(w, v) => gather_str!(U, w, v),
        }
    }

    /// numpy `ravel(order=...)`/`flatten(order=...)`'s actual order
    /// semantics, including 'K' (which `reshape_with_order` deliberately
    /// does not support -- numpy itself refuses 'K' for reshaping but
    /// accepts it here). Subsumes what `reshape_with_order(&[n], ...)`
    /// computes for 'C'/'F'/'A' too (same traversal, expressed via
    /// `axis_perm_for_order` instead of the transpose-composition trick),
    /// so ravel/flatten can go through this single path for all four
    /// letters.
    pub fn ravel_order(&self, order: &str) -> Result<NdArray, IonpError> {
        let perm = self.axis_perm_for_order(order)?;
        let buffer = self.gather_by_perm(&perm);
        let n = self.size();
        // `ravel_order` always allocates a fresh, densely-packed 1-D
        // buffer (its own doc above, and `ndarray_attrs.rs`'s `flatten`
        // relies on exactly that to satisfy numpy's "flatten never
        // returns a view" contract) -- the same "freshly allocated"
        // situation `NdArray::from_buffer` already special-cases for
        // size 0 (real numpy zeroes every stride of a fresh empty array,
        // e.g. `np.zeros((2,0,4)).flatten().strides == (0,)`, not the
        // `(8,)` `shape::c_strides(&[0])` would give here). Verified
        // directly against real numpy 2.5.1 for `.flatten()` on `(0,)`,
        // `(0,3)`, `(3,0)`, `(2,0,4)`, `(0,0)`.
        let strides = if n == 0 { vec![0isize] } else { shape::c_strides(&[n]) };
        Ok(NdArray {
            buffer: Arc::new(buffer),
            strides,
            shape: vec![n],
            offset: 0,
        })
    }

    /// numpy `copy(order=...)`/`astype(order=...)`/the `_like` family's
    /// `order=...`: a fresh, densely packed array with `self`'s own shape,
    /// laid out in memory according to `order` (default meaning for all of
    /// those is `'K'` -- match the input's existing layout as closely as
    /// possible). Strides are computed by taking C-contiguous strides for
    /// the `order`-permuted shape and relabeling them back onto `self`'s
    /// original axis positions (verified against real numpy 2.5.1 for
    /// C-contiguous, F-contiguous, transposed, and sliced-non-contiguous
    /// sources).
    pub fn to_contiguous_order(&self, order: &str) -> Result<NdArray, IonpError> {
        let perm = self.axis_perm_for_order(order)?;
        Ok(self.relayout_by_perm(&perm))
    }

    /// Shared tail of `to_contiguous_order` above: given an axis
    /// permutation ALREADY resolved by the caller (slowest-to-fastest,
    /// same convention `axis_perm_for_order` returns), gather `self`'s
    /// elements into a fresh densely-packed buffer in that traversal
    /// order and relabel C-contiguous-for-the-permuted-shape strides back
    /// onto `self`'s own axis positions. `to_contiguous_order` itself
    /// derives its perm from `self`'s OWN strides via `axis_perm_for_order`
    /// -- this is the piece factored out so a caller with a perm derived
    /// some OTHER way (ionp-py's `Ufunc.__call__` `order='K'`/`'A'`
    /// support: `multi_sorted_stride_perm` below, which decides the perm
    /// from MULTIPLE operands' shapes/strides, not `self`'s alone, since a
    /// binary ufunc's freshly-computed output has no memory layout of its
    /// own yet to sort by) can reuse the identical gather+relabel
    /// mechanics without duplicating them. Pure refactor of what was
    /// previously `to_contiguous_order`'s own body -- no behavior change.
    pub fn relayout_by_perm(&self, perm: &[usize]) -> NdArray {
        let buffer = self.gather_by_perm(perm);
        let permuted_shape: Vec<usize> = perm.iter().map(|&a| self.shape[a]).collect();
        let permuted_strides = shape::c_strides(&permuted_shape);
        let mut strides = vec![0isize; self.ndim()];
        for (i, &axis) in perm.iter().enumerate() {
            strides[axis] = permuted_strides[i];
        }
        if self.size() == 0 {
            // Same freshly-allocated-size-0 special case as `from_buffer`.
            strides.iter_mut().for_each(|s| *s = 0);
        }
        NdArray {
            buffer: Arc::new(buffer),
            shape: self.shape.clone(),
            strides,
            offset: 0,
        }
    }

    /// numpy REDUCTION output memory layout (`sum`/`mean`/`amin`/`amax`/
    /// `all`/`any`/`count_nonzero`/`nansum`/etc, and via composition
    /// `ptp`/`mean`/`nanmean` -- see `docs/reduction-output-layout.md` for
    /// the full empirical derivation, verified against real numpy 2.5.1
    /// with 138/138 matches across C/F/transposed/partial-swap/permuted/
    /// stepped-slice/negative-stride/size-1-axis inputs, every `axis=`
    /// shape (None/int/tuple), and `keepdims` True/False). This is a
    /// DIFFERENT rule from `axis_perm_for_order("K")`/
    /// `multi_sorted_stride_perm` above -- a reduction's output has FEWER
    /// (or, with `keepdims`, size-1-squashed) dimensions than its input, so
    /// there is no direct axis-to-axis correspondence to vote on. The rule:
    /// pretend every REDUCED axis has size 1 (but keep its REAL original
    /// stride for the comparison below -- this matters for `keepdims`),
    /// then run the exact same stable descending-`abs(stride)` sort
    /// `axis_perm_for_order("K")` uses over ALL `ndim` axes (reduced and
    /// retained together), lay out C-contiguous strides for that permuted
    /// pretend-shape, and un-permute back onto the original axis positions.
    /// Without `keepdims`, the reduced-axis entries are simply dropped from
    /// the (shape, strides) pair afterward -- numpy's own non-keepdims
    /// reduction output genuinely IS this array's squeeze, not a
    /// separately-computed layout (verified, not assumed: this single
    /// formula reproduced every non-keepdims case in the 138-case sweep
    /// too). A forced-size-1 axis contributes a multiplicative no-op to
    /// every OTHER axis's assigned stride (`shape.max(1)` in `c_strides`),
    /// so which axes are "reduced" vs "retained" never perturbs the
    /// strides assigned to the axes that stay real-sized.
    pub fn reduce_output_layout(
        orig_shape: &[usize],
        orig_strides: &[isize],
        reduced_axes: &[usize],
        keepdims: bool,
    ) -> (Vec<usize>, Vec<isize>) {
        let ndim = orig_shape.len();
        let mut pretend_shape = orig_shape.to_vec();
        for &ax in reduced_axes {
            pretend_shape[ax] = 1;
        }
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.sort_by(|&a, &b| orig_strides[b].unsigned_abs().cmp(&orig_strides[a].unsigned_abs()));
        let permuted_shape: Vec<usize> = perm.iter().map(|&a| pretend_shape[a]).collect();
        let permuted_strides = shape::c_strides(&permuted_shape);
        let mut strides = vec![0isize; ndim];
        for (i, &axis) in perm.iter().enumerate() {
            strides[axis] = permuted_strides[i];
        }
        let (out_shape, mut out_strides) = if keepdims {
            (pretend_shape, strides)
        } else {
            let retained: Vec<usize> = (0..ndim).filter(|a| !reduced_axes.contains(a)).collect();
            (
                retained.iter().map(|&a| orig_shape[a]).collect(),
                retained.iter().map(|&a| strides[a]).collect(),
            )
        };
        // Ticket #67: a reduction's output is a FRESH allocation (built by
        // `reduce_axis`'s per-dtype dispatch macro, never a view into the
        // input), so it is subject to the same "freshly allocated size-0
        // array gets all-zero strides" rule as `NdArray::from_buffer`/
        // `relayout_by_perm` -- see those functions' own doc comments. This
        // matters specifically for `keepdims=True`: when a NON-reduced axis
        // is itself size 0, the output stays size 0 even after the reduced
        // axes collapse to 1 (e.g. `np.zeros((0,3)).sum(axis=1,
        // keepdims=True)` has shape `(0,1)`), and the descending-stride-sort
        // formula above computes ordinary nonzero strides for that shape
        // exactly like `c_strides`/`f_strides` do for any other fresh-view
        // shape -- it has no size-0 special case of its own. Verified live
        // against real numpy 2.5.1: `np.zeros((0,3)).sum(axis=1,
        // keepdims=True).strides == (0, 0)`, `np.zeros((3,0)).sum(axis=0,
        // keepdims=True).strides == (0, 0)`, `.prod`/`.mean` identical. The
        // non-keepdims branch is already unreachable with size 0 here in
        // practice (its caller, `relayout_for_reduction`, zeros via
        // `relayout_by_perm` before ever reaching a keepdims lift), but the
        // check is applied uniformly rather than only inside the `if
        // keepdims` arm so this function stays correct on its own terms,
        // independent of how its caller happens to be sequenced today.
        if out_shape.iter().product::<usize>() == 0 {
            out_strides.iter_mut().for_each(|s| *s = 0);
        }
        (out_shape, out_strides)
    }

    /// Physically relays out `self` -- a freshly computed, C-contiguous,
    /// NON-keepdims reduction result (shape = the retained axes' sizes, in
    /// their original relative order; exactly what `reduce_axis`'s/
    /// `accumulate_axis`'s per-dtype dispatch macros build before this is
    /// called) -- to match real numpy's reduction output layout for the
    /// given original (pre-reduction) input shape/strides. See
    /// `reduce_output_layout`'s doc comment just above for the empirically-
    /// derived rule this implements. `keepdims` re-inserts size-1 axes at
    /// the reduced positions with numpy's own (non-arbitrary) computed
    /// stride for each, per that same rule.
    pub fn relayout_for_reduction(
        &self,
        orig_shape: &[usize],
        orig_strides: &[isize],
        reduced_axes: &[usize],
        keepdims: bool,
    ) -> NdArray {
        let ndim = orig_shape.len();
        let retained_axes: Vec<usize> = (0..ndim).filter(|a| !reduced_axes.contains(a)).collect();
        debug_assert_eq!(retained_axes.len(), self.ndim());

        // Only the RELATIVE order among retained axes determines the
        // traversal order `self`'s own (already retained-axes-only) data
        // must be gathered in -- a reduced/pretend-1 axis contributes a
        // multiplicative no-op to every neighboring stride computation (see
        // `reduce_output_layout`'s doc comment), so filtering the full
        // descending-stride sort down to retained axes and remapping to
        // LOCAL (0..retained_axes.len()) indices gives exactly that order.
        let mut full_order: Vec<usize> = (0..ndim).collect();
        full_order.sort_by(|&x, &y| orig_strides[y].unsigned_abs().cmp(&orig_strides[x].unsigned_abs()));
        let local_of: std::collections::HashMap<usize, usize> =
            retained_axes.iter().enumerate().map(|(i, &ax)| (ax, i)).collect();
        let local_perm: Vec<usize> = full_order.into_iter().filter_map(|ax| local_of.get(&ax).copied()).collect();
        let relaid = self.relayout_by_perm(&local_perm);

        if !keepdims {
            return relaid;
        }

        relaid.lift_reduction_keepdims(orig_shape, orig_strides, reduced_axes)
    }

    /// Re-inserts the reduced axes (dropped, i.e. `keepdims=false`) back
    /// into an already-correctly-laid-out retained-axes-only reduction
    /// result, at their original positions, with numpy's own computed
    /// stride for each (not an arbitrary/zero value) -- see
    /// `reduce_output_layout`'s doc comment. `self` must already carry the
    /// retained axes' correct (non-keepdims) layout -- this is purely
    /// relabeling `self`'s buffer onto the full-rank (shape, strides) pair,
    /// no data movement.
    ///
    /// This is split out from `relayout_for_reduction` so that COMPOSED
    /// reductions can work on NON-keepdims (retained-axes-only,
    /// always-plain-shape) intermediate operands and only lift to the full
    /// keepdims shape ONCE, on the final composed result, using the true
    /// original array's shape/strides. Voting on ALREADY-keepdims'd
    /// operands instead (each carrying a synthetic size-1-axis stride
    /// assigned by `reduce_output_layout`) is WRONG: that synthetic
    /// stride's numeric value can tie or misrank against a retained axis's
    /// real stride during a second vote, producing an incorrect permutation
    /// -- confirmed empirically (this was the exact bug that caused `ptp`
    /// keepdims=True mismatches when it instead composed from keepdims=True
    /// sub-reductions and voted on those directly via
    /// `k_order_relayout_composed`/`multi_sorted_stride_perm`, the mechanism
    /// that is still correct for `ptp` specifically -- see its doc comment).
    /// `mean`/`nanmean` do NOT use this two-operand-vote mechanism at all
    /// (see their own doc comments in `reductions.rs`): real numpy's
    /// `out=ret`-based divide means their result strides are simply, always,
    /// `sum`'s own -- so they compose at keepdims=false and call
    /// `relayout_for_reduction` ONCE directly on the divide's raw result,
    /// with no voting step and no separate `lift_reduction_keepdims` call.
    pub fn lift_reduction_keepdims(
        &self,
        orig_shape: &[usize],
        orig_strides: &[isize],
        reduced_axes: &[usize],
    ) -> NdArray {
        let (full_shape, full_strides) = NdArray::reduce_output_layout(orig_shape, orig_strides, reduced_axes, true);
        NdArray {
            buffer: self.buffer.clone(),
            shape: full_shape,
            strides: full_strides,
            offset: self.offset,
        }
    }

    /// numpy ufunc `order='K'`'s REAL algorithm when more than one input
    /// operand is in play (`PyArray_CreateMultiSortedStridePerm`'s
    /// observable behavior, reverse-engineered and verified live against
    /// numpy 2.5.1 -- see the `order=` implementation commit for the probe
    /// scripts). This is NOT simply "use whichever operand is F-contiguous"
    /// or "prefer the first/last operand" -- it's a stable insertion sort
    /// over axis PAIRS, starting from C (identity) order, where each
    /// operand "votes" on a pair only if its own shape is not 1 at BOTH
    /// axes (a broadcast/size-1 axis carries no stride information numpy
    /// can trust, so it never votes) and, when more than one operand votes
    /// on the same pair, a genuine disagreement between decisive votes
    /// makes that pair AMBIGUOUS -- which falls back to leaving the pair in
    /// its original (C/identity) relative order, it does NOT arbitrarily
    /// prefer either operand. Verified against three independent numpy
    /// probes: (1) two full-rank operands with opposite (F vs C) layout and
    /// no broadcasting produces plain C output (every pair is ambiguous);
    /// (2) one full-rank F operand plus one operand broadcast on axis 0
    /// produces a layout that is neither pure C nor pure F (the broadcast
    /// operand's shape-1 axis never votes, so only the non-broadcast
    /// operand's preference applies there, while OTHER axis pairs where
    /// both operands are decisive and disagree still fall back to C-order);
    /// (3) a single operand with a non-canonical stride on a size-1 axis
    /// (inherited from a slice) is correctly ignored -- that axis is never
    /// decisive against anything, matching numpy's own definition of
    /// contiguity (`is_c_contiguous`'s doc comment above already
    /// establishes this principle for the single-array case; this is the
    /// same principle applied across multiple operands).
    ///
    /// `operands` is `(shape, strides)` for each input, in the SAME order
    /// numpy's own C ufunc loop would see them (this project only ever
    /// calls this with <= 2 real operands -- ufuncs are unary or binary --
    /// so the "does a later operand's disagreement ever un-ambiguous a
    /// pair" question that would matter for 3+ operands never arises here).
    /// Each operand may have fewer dims than `ndim`; it is right-aligned
    /// (broadcasting's own rule) and contributes no vote for any axis to
    /// the left of that alignment, exactly as if it had implicit leading
    /// size-1 axes there.
    /// Composed-reduction K-order layout for `ptp` (`max - min`) SPECIFICALLY
    /// -- NOT `mean`/`nanmean` (`sum / count`), despite both being "two
    /// reduction operands combined by a further elementwise op": verified
    /// empirically (direct real-numpy comparison, no ionp involved) that
    /// `np.mean`'s divide and `np.ptp`'s subtract are NOT the same
    /// mechanism. `np.mean`'s real implementation
    /// (`numpy/_core/_methods.py::_mean`) computes
    /// `um.true_divide(ret, rcount, out=ret, casting='unsafe')` -- the
    /// explicit `out=ret` argument writes the quotient's VALUES directly
    /// into `ret`'s (the sum's) own pre-existing buffer/layout, bypassing
    /// any output-layout inference entirely. So `mean`'s result strides
    /// are simply, unconditionally, `sum`'s own strides -- confirmed with
    /// 15/15 matches across every layout (C/F/fulltranspose/partialswap/
    /// permuted) x every axis, plus masked (`where=`) C and F cases, all
    /// `keepdims=True`: `np.sum(...).strides == np.mean(...).strides`
    /// EXACTLY, every time (`/tmp/probe_div_mech2.py`). `np.ptp`, by
    /// contrast, is NOT `out=`-based (`max(axis) - min(axis)` is a genuine
    /// fresh subtract of two independently-allocated arrays), so it DOES go
    /// through real order='K' composition -- this function. Concretely, on
    /// the SAME partialswap array/axis where `sum`/`amax` both correctly
    /// give `(32,32,8)` bytes (keepdims=True), `np.ptp` gives `(64,32,8)`
    /// (this function's fresh-recompute answer) while `np.mean` gives
    /// `(32,32,8)` (`sum`'s own, unrecomputed, preserved answer) --
    /// directly disproving an earlier draft of this comment that assumed
    /// the same mechanism covered both. `mean`/`nanmean` do NOT call this
    /// function -- see their own doc comments in `reductions.rs` for the
    /// `relayout_for_reduction`-based fix that matches the real `out=ret`
    /// mechanism instead. This is DELIBERATELY separate from
    /// `multi_sorted_stride_perm` above rather than a fix to it, to avoid
    /// any risk of regressing that function's own already-verified
    /// (elementwise binary-op) behavior.
    ///
    /// Empirically discovered (`np.ptp`/manual `np.amax(kd=True) -
    /// np.amin(kd=True)` on real numpy 2.5.1, probed via `as_strided`
    /// synthetic operands to isolate the mechanism): when a `keepdims=True`
    /// reduction operand feeds into a further elementwise op, real numpy's
    /// ufunc order='K' machinery does NOT run its generic per-axis-pair
    /// vote (`PyArray_CreateMultiSortedStridePerm`) first -- it checks
    /// whether an operand is GENERALIZED C- or F-contiguous (the same
    /// size-<=1-axis-tolerant definition `is_c_contiguous`/
    /// `is_f_contiguous` already implement) and, if so, uses **freshly
    /// recomputed** `c_strides`/`f_strides` for the output shape --
    /// discarding whatever arbitrary stride value the input had on any
    /// size-1 axis, even though that same value would have been "voted"
    /// on (and wrongly treated as decisive-once-tie-broken) by the general
    /// per-axis-pair algorithm. Verified concretely: a `keepdims=True`
    /// `amax` result of shape `(1,2,4)` strides `(32,32,8)` bytes (real,
    /// non-arbitrary -- `reduce_output_layout`'s own correct answer for
    /// that single reduction) is `is_c_contiguous()` under the generalized
    /// definition, and real `ptp` over the same axis/array produces
    /// `(64,32,8)` -- fresh `c_strides((1,2,4))` -- NOT `(32,32,8)`
    /// (`multi_sorted_stride_perm`'s answer) and NOT the single-reduction
    /// answer either. A parallel F-contiguous case (`(1,3,4)` strides
    /// `(8,8,24)`) confirmed the same mechanism on the F side, producing
    /// fresh `f_strides((1,3,4)) == (8,8,24)` (coincidentally identical to
    /// the input here, but derived independently, not preserved).
    ///
    /// `raw` must already be `binary_op`'s always-C-contiguous composed
    /// result (same shape as every operand, post-broadcast). `operands` are
    /// checked IN ORDER, C before F for each, mirroring
    /// `multi_sorted_stride_perm`'s own "first decisive operand wins"
    /// precedence; the first operand that is C- or F-contiguous decides the
    /// whole output layout. If no operand is either, this falls back to the
    /// existing verified `multi_sorted_stride_perm` general algorithm
    /// unchanged (this covers every case the elementwise K-order task
    /// already proved correct -- none of those probes happened to compose
    /// a `keepdims=True` reduction operand, so they never exercised this
    /// fast path's absence).
    pub fn k_order_relayout_composed(raw: &NdArray, operands: &[&NdArray]) -> NdArray {
        let ndim = raw.ndim();
        // Check EACH operand, C then F, before moving to the next -- NOT
        // "C for every operand, then F for every operand". A later
        // operand's trivial (0-d/size-1, always-both-C-and-F) contiguity
        // must never pre-empt an EARLIER operand's genuine, decisive F
        // layout (found via the `mean`/`nanmean` no-`where=` case: the
        // scalar divisor `count_natural` is 0-d and so trivially
        // `is_c_contiguous() == true`, which was wrongly winning the C
        // fast path ahead of `sum_result`'s real F-contiguous layout when
        // both were checked in two separate full passes).
        for op in operands {
            if op.is_c_contiguous() {
                let identity: Vec<usize> = (0..ndim).collect();
                return raw.relayout_by_perm(&identity);
            }
            if op.is_f_contiguous() {
                let reversed: Vec<usize> = (0..ndim).rev().collect();
                return raw.relayout_by_perm(&reversed);
            }
        }
        let owned: Vec<(&[usize], &[isize])> = operands.iter().map(|o| (o.shape(), o.strides())).collect();
        let perm = NdArray::multi_sorted_stride_perm(ndim, &owned);
        raw.relayout_by_perm(&perm)
    }

    /// numpy `PyArray_CreateMultiSortedStridePerm` (`shape.c`), ported
    /// exactly, including two subtleties this port previously got wrong
    /// (ticket #7, 2026-08-08 -- the length-1-axis regression; see
    /// `docs/` for the dated writeup):
    ///
    /// 1. **Ambiguous comparisons must NOT stop the scan.** A comparison
    ///    between axes `ax_j0`/`ax_j1` is only "decided" by an operand
    ///    that has neither axis at size 1 (numpy's `SHAPE[ax_j0] != 1 &&
    ///    SHAPE[ax_j1] != 1` guard). If EVERY operand has a size-1 axis in
    ///    this pair (e.g. because the array itself has a length-1 axis),
    ///    the pair stays "ambiguous" (`ambig` never clears) and numpy's
    ///    `for (i1 = i0-1; i1 >= 0; --i1)` keeps walking further back
    ///    looking for a decisive comparison -- it does not stop at the
    ///    first ambiguous one. The prior Rust port `break`d on
    ///    `!decided`, which silently froze the whole permutation at
    ///    identity (C order) the moment a length-1 axis was involved,
    ///    even when a real, decisive comparison existed one position
    ///    further back. That is the exact, measured root cause of the
    ///    118-item defect (`np.abs` on `(4,1,2)`-shaped F-order input
    ///    losing F strides).
    ///
    /// 2. **Insertion is two-phase (scan-then-shift), not
    ///    scan-and-shift-as-you-go.** numpy only tracks the final
    ///    insertion index `ipos` while scanning (never mutating
    ///    `out_strideperm` mid-scan), then does a single bulk shift of
    ///    the range `(ipos, i0]` down by one slot, then drops `ax_j0`
    ///    into `ipos`. Combining that with point 1 (skipping over
    ///    ambiguous positions without writing) makes an incremental
    ///    "write as you decide to swap" version incorrect: an
    ///    ambiguous position in the middle of a decisive run must still
    ///    be shifted, even though no comparison at that position alone
    ///    justified a swap. Scan-then-bulk-shift handles this
    ///    automatically; the previous incremental-write version did not
    ///    (it would leave a stale duplicate/lost axis whenever an
    ///    ambiguous and a decisive comparison were interleaved in one
    ///    `i0` pass).
    pub fn multi_sorted_stride_perm(ndim: usize, operands: &[(&[usize], &[isize])]) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..ndim).collect();
        for i0 in 1..ndim {
            let ax_j0 = perm[i0];
            let mut ipos = i0;
            let mut i1 = i0;
            while i1 > 0 {
                i1 -= 1;
                let ax_j1 = perm[i1];
                let mut ambig = true;
                let mut shouldswap = false;
                for &(shape, strides) in operands {
                    let offset = ndim - shape.len();
                    if ax_j0 < offset || ax_j1 < offset {
                        // Axis outside this operand's (broadcast-padded)
                        // rank -- numpy pads missing leading axes to
                        // shape 1, and shape==1 axes never decide a
                        // comparison, so this operand contributes nothing.
                        continue;
                    }
                    let s0 = shape[ax_j0 - offset];
                    let s1 = shape[ax_j1 - offset];
                    if s0 == 1 || s1 == 1 {
                        continue;
                    }
                    let st0 = strides[ax_j0 - offset].unsigned_abs();
                    let st1 = strides[ax_j1 - offset].unsigned_abs();
                    if st0 <= st1 {
                        // Set (or keep) "don't swap" even if not already
                        // ambiguous -- on genuine conflict between
                        // operands, C order wins (matches numpy's comment
                        // verbatim).
                        shouldswap = false;
                    } else if ambig {
                        // Only set "swap" while still ambiguous; a later
                        // operand can never override an earlier decisive
                        // "don't swap".
                        shouldswap = true;
                    }
                    ambig = false;
                }
                if !ambig {
                    if shouldswap {
                        ipos = i1;
                    } else {
                        break;
                    }
                }
                // else: stays ambiguous at this i1 -- keep scanning
                // further back (i1 -= 1 on the next loop iteration)
                // instead of stopping here.
            }
            if ipos != i0 {
                let mut k = i0;
                while k > ipos {
                    perm[k] = perm[k - 1];
                    k -= 1;
                }
            }
            perm[ipos] = ax_j0;
        }
        perm
    }

    /// numpy `transpose(*axes)` with explicit axes (as opposed to the
    /// bare, always-full-reverse `transpose()`/`.T` above): permute this
    /// array's axes according to `axes`, a permutation of `0..ndim` given
    /// as (possibly negative, numpy-style) signed axis indices. Always a
    /// view — transpose, with or without explicit axes, is nothing but
    /// relabeling which stride goes with which axis.
    pub fn transpose_axes(&self, axes: &[isize]) -> Result<NdArray, IonpError> {
        let ndim = self.ndim();
        if axes.len() != ndim {
            return Err(IonpError::Value("axes don't match array".to_string()));
        }
        let ndim_i = ndim as isize;
        let mut seen = vec![false; ndim];
        let mut norm_axes = Vec::with_capacity(ndim);
        for &a in axes {
            let normalized = if a < 0 { a + ndim_i } else { a };
            if normalized < 0 || normalized >= ndim_i {
                return Err(IonpError::Index(format!(
                    "axis {a} is out of bounds for array of dimension {ndim}"
                )));
            }
            let idx = normalized as usize;
            if seen[idx] {
                return Err(IonpError::Value("repeated axis in transpose".to_string()));
            }
            seen[idx] = true;
            norm_axes.push(idx);
        }
        let shape: Vec<usize> = norm_axes.iter().map(|&i| self.shape[i]).collect();
        let strides: Vec<isize> = norm_axes.iter().map(|&i| self.strides[i]).collect();
        Ok(NdArray {
            buffer: Arc::clone(&self.buffer),
            shape,
            strides,
            offset: self.offset,
        })
    }

    /// Basic (integer/slice/ellipsis/newaxis) indexing along leading axes;
    /// any axes beyond the explicit index/slice items are kept whole
    /// (numpy's implicit trailing-`:'` rule). Always a view. Fancy/boolean
    /// indexing is not implemented (see KNOWN-DIFFERENCES.md).
    ///
    /// `Ellipsis` expands, in place, to as many full (`:`) slices as are
    /// needed to make the remaining explicit items (`Index`/`Slice`) cover
    /// every axis of `self` — e.g. for a 4-D array, `a[0, ..., 1]` expands
    /// to `a[0, :, :, 1]`. At most one `Ellipsis` is allowed, matching
    /// numpy's own `IndexError`. `NewAxis` (numpy's `np.newaxis`/bare
    /// `None`) inserts a length-1 axis at that position without consuming
    /// or moving through any axis of `self` — it never appears in the
    /// axis-counting used to size the `Ellipsis` expansion or the trailing-
    /// axis fill.
    pub fn get_view(&self, items: &[SliceItem]) -> Result<NdArray, IonpError> {
        let items = expand_ellipsis(items, self.ndim())?;

        let explicit_axes = items
            .iter()
            .filter(|i| matches!(i, SliceItem::Index(_) | SliceItem::Slice { .. }))
            .count();
        if explicit_axes > self.ndim() {
            return Err(IonpError::Index(format!(
                "too many indices for array: array is {}-dimensional, but {} were indexed",
                self.ndim(),
                explicit_axes
            )));
        }

        let mut shape = Vec::new();
        let mut strides = Vec::new();
        let mut offset = self.offset;
        let mut axis = 0usize;

        for item in &items {
            match item {
                SliceItem::NewAxis => {
                    shape.push(1);
                    strides.push(0);
                }
                SliceItem::Index(i) => {
                    let dim = self.shape[axis];
                    let stride = self.strides[axis];
                    let dim_i = dim as isize;
                    let idx = if *i < 0 { *i + dim_i } else { *i };
                    if idx < 0 || idx >= dim_i {
                        return Err(IonpError::Index(format!(
                            "index {i} is out of bounds for axis {axis} with size {dim}"
                        )));
                    }
                    offset += stride * idx;
                    axis += 1;
                }
                SliceItem::Slice { start, stop, step } => {
                    let dim = self.shape[axis];
                    let stride = self.strides[axis];
                    let (nstart, _nstop, nstep, out_len) =
                        normalize_slice(*start, *stop, *step, dim)?;
                    offset += stride * nstart;
                    shape.push(out_len);
                    strides.push(stride * nstep);
                    axis += 1;
                }
                SliceItem::Ellipsis => {
                    unreachable!("expand_ellipsis removes every Ellipsis before this loop")
                }
            }
        }
        // Trailing axes of `self` not covered by any explicit item are
        // kept in full.
        for axis in axis..self.ndim() {
            shape.push(self.shape[axis]);
            strides.push(self.strides[axis]);
        }

        Ok(NdArray {
            buffer: Arc::clone(&self.buffer),
            shape,
            strides,
            offset,
        })
    }
}

/// Replace a single `SliceItem::Ellipsis` (if present) with as many full
/// (`start=stop=step=None`) slices as are needed so that the explicit
/// (`Index`/`Slice`) items exactly cover `ndim` axes. More than one
/// `Ellipsis` is a numpy `IndexError` (`an index can only have a single
/// ellipsis ('...')`, verified against real numpy 2.5.1).
fn expand_ellipsis(items: &[SliceItem], ndim: usize) -> Result<Vec<SliceItem>, IonpError> {
    let ellipsis_positions: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, i)| matches!(i, SliceItem::Ellipsis))
        .map(|(pos, _)| pos)
        .collect();
    if ellipsis_positions.len() > 1 {
        return Err(IonpError::Index(
            "an index can only have a single ellipsis ('...')".to_string(),
        ));
    }
    let Some(&pos) = ellipsis_positions.first() else {
        return Ok(items.to_vec());
    };
    let explicit_axes = items
        .iter()
        .filter(|i| matches!(i, SliceItem::Index(_) | SliceItem::Slice { .. }))
        .count();
    let fill = ndim.saturating_sub(explicit_axes);
    let mut out = Vec::with_capacity(items.len() - 1 + fill);
    out.extend_from_slice(&items[..pos]);
    for _ in 0..fill {
        out.push(SliceItem::Slice { start: None, stop: None, step: None });
    }
    out.extend_from_slice(&items[pos + 1..]);
    Ok(out)
}

fn resolve_unknown_dim(new_shape: &[usize], _total: usize) -> Result<Vec<usize>, IonpError> {
    // `-1` inference is a signed-input concept; NdArray's public reshape
    // takes unsigned dims. The PyO3 layer resolves any `-1` before calling
    // in (see ionp-py). This function exists as the single seam where that
    // rule would extend if signed dims were threaded further down.
    Ok(new_shape.to_vec())
}

#[derive(Debug, Clone, Copy)]
pub enum SliceItem {
    Index(isize),
    Slice {
        start: Option<isize>,
        stop: Option<isize>,
        step: Option<isize>,
    },
    /// `np.newaxis` / a bare `None` in an index tuple: inserts a length-1
    /// axis without consuming any axis of the array being indexed.
    NewAxis,
    /// `...` in an index tuple: expands to as many full slices as needed
    /// to cover every remaining axis (see `expand_ellipsis`).
    Ellipsis,
}

/// CPython's `slice.indices()` algorithm: normalizes a possibly-negative,
/// possibly-open-ended `(start, stop, step)` against a dimension length
/// into a concrete `(start, stop, step, out_len)`, matching numpy's basic
/// slicing exactly (numpy delegates to the same semantics as Python
/// sequences for each axis).
/// Public re-export of `normalize_slice` for `indexing.rs`, which has to
/// resolve a slice item exactly the way `get_view` does (same clamping,
/// same negative-step handling, same `step == 0` error) while ALSO tracking
/// an advanced-index block that `get_view` knows nothing about. Two
/// independent slice normalisers would be two chances to disagree; this is
/// one implementation with one caller-visible name per crate boundary.
pub(crate) fn normalize_slice_pub(
    start: Option<isize>,
    stop: Option<isize>,
    step: Option<isize>,
    len: usize,
) -> Result<(isize, isize, isize, usize), IonpError> {
    normalize_slice(start, stop, step, len)
}

fn normalize_slice(
    start: Option<isize>,
    stop: Option<isize>,
    step: Option<isize>,
    len: usize,
) -> Result<(isize, isize, isize, usize), IonpError> {
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(IonpError::Value("slice step cannot be zero".to_string()));
    }
    let len_i = len as isize;
    let (lower, upper) = if step > 0 { (0, len_i) } else { (-1, len_i - 1) };

    let start = match start {
        None => if step < 0 { upper } else { lower },
        Some(s) => {
            let s = if s < 0 { (s + len_i).max(lower) } else { s.min(upper) };
            s
        }
    };
    let stop = match stop {
        None => if step < 0 { lower } else { upper },
        Some(s) => {
            let s = if s < 0 { (s + len_i).max(lower) } else { s.min(upper) };
            s
        }
    };

    let out_len = if step > 0 {
        ((stop - start + step - 1) / step).max(0) as usize
    } else {
        ((stop - start + step + 1) / step).max(0) as usize
    };
    Ok((start, stop, step, out_len))
}

/// numpy-correct contiguity test: matches `PyArray_UpdateFlags`'s own
/// "collapse size-1 axes" rule rather than a blunt exact-stride-vector
/// comparison. Any axis of length 0 or 1 contributes no observable
/// constraint on its own stride (it's visited at most once, so that
/// stride's value can never actually be exercised) -- only once a
/// length>=2 axis is encountered does its stride get checked against the
/// running expected value, and only length>=2 axes advance the running
/// product used to predict the next expected stride. `c_major`=true walks
/// axes fastest-to-slowest (last axis first, C order); false walks
/// slowest-to-fastest (first axis first, F order).
fn strides_match_ignoring_unit_axes(shape: &[usize], strides: &Strides, c_major: bool) -> bool {
    let mut expected: isize = 1;
    let indices: Vec<usize> = if c_major {
        (0..shape.len()).rev().collect()
    } else {
        (0..shape.len()).collect()
    };
    for i in indices {
        let dim = shape[i];
        if dim <= 1 {
            continue;
        }
        if strides[i] != expected {
            return false;
        }
        expected *= dim as isize;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;

    fn arange(n: i64) -> NdArray {
        let v: Vec<i64> = (0..n).collect();
        NdArray::from_buffer(Buffer::I64(v), vec![n as usize], Order::C).unwrap()
    }

    /// `cast_to` must not lose a contiguous view's offset.
    ///
    /// Regression guard, 2026-08-03. `cast_to` built its result with a
    /// hardcoded `offset: 0` even though `Buffer::cast_to` returns a
    /// buffer of the FULL original length. A plain slice like `a[2:5]` is
    /// contiguous and carries only `offset: 2`, so it took the cheap
    /// clone path and then had that offset discarded -- silently reading
    /// the array back from element 0.
    ///
    /// The blast radius was never limited to the `copyto` that exposed
    /// it: `astype`, `asarray(dtype=)` and `array(dtype=)` all route
    /// here, and all three returned an array of the right dtype, right
    /// shape and WRONG elements.
    ///
    /// The non-contiguous half of this test is not padding. Those views
    /// were always correct, because `to_contiguous()` gathers through the
    /// offset and genuinely rebases to 0 -- which is precisely why the
    /// bug hid: every exotic view worked and only the simplest one did
    /// not. Asserting both halves means a "fix" that just moves the
    /// breakage to the other path cannot pass.
    #[test]
    fn cast_to_preserves_view_offset() {
        // Values as i64 regardless of which integer buffer they landed in,
        // so the same assertion reads both the I64 and the I8 result.
        fn vals(a: &NdArray) -> Vec<i64> {
            let c = a.to_contiguous();
            match &*c.buffer {
                Buffer::I64(d) => d.clone(),
                Buffer::I8(d) => d.iter().map(|&x| x as i64).collect(),
                other => panic!("unexpected buffer {other:?}"),
            }
        }
        let a = arange(9);
        let view = |offset: isize, stride: isize| NdArray {
            buffer: Arc::clone(&a.buffer),
            shape: vec![3],
            strides: vec![stride],
            offset,
        };

        // Contiguous, offset 2 -- the case that was broken.
        let v = view(2, 1);
        assert!(v.is_c_contiguous(), "precondition: the broken path is the CONTIGUOUS one");
        assert_eq!(vals(&v), vec![2, 3, 4], "precondition: the view itself reads correctly");
        assert_eq!(
            vals(&v.cast_to(DType::I8)),
            vec![2, 3, 4],
            "cast_to discarded the offset and re-read from element 0"
        );

        // Same dtype: takes cast_to's early return, must also survive.
        assert_eq!(vals(&v.cast_to(DType::I64)), vec![2, 3, 4]);

        // Non-contiguous, offset-bearing -- always worked; must stay so.
        let r = view(6, -1);
        assert!(!r.is_c_contiguous());
        assert_eq!(vals(&r), vec![6, 5, 4]);
        assert_eq!(vals(&r.cast_to(DType::I8)), vec![6, 5, 4]);

        // Offset 0 -- the overwhelmingly common case, unchanged by the fix.
        assert_eq!(vals(&view(0, 1).cast_to(DType::I8)), vec![0, 1, 2]);
    }

    #[test]
    fn from_buffer_rejects_size_mismatch() {
        let err = NdArray::from_buffer(Buffer::I64(vec![1, 2, 3]), vec![2, 2], Order::C).unwrap_err();
        assert!(matches!(err, IonpError::Value(_)));
    }

    #[test]
    fn transpose_reverses_shape_and_strides() {
        let v: Vec<f64> = (0..6).map(|x| x as f64).collect();
        let a = NdArray::from_buffer(Buffer::F64(v), vec![2, 3], Order::C).unwrap();
        let t = a.transpose();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.strides(), &[1, 3]);
        // data unchanged, same Arc
        assert!(Arc::ptr_eq(&a.buffer, &t.buffer));
    }

    #[test]
    fn transpose_axes_explicit_permutation() {
        // np.arange(24).reshape(2,3,4).transpose(2,0,1).shape == (4,2,3)
        let v: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let a = NdArray::from_buffer(Buffer::F64(v), vec![2, 3, 4], Order::C).unwrap();
        let t = a.transpose_axes(&[2, 0, 1]).unwrap();
        assert_eq!(t.shape(), &[4, 2, 3]);
        // same permutation applied to strides
        assert_eq!(t.strides(), &[a.strides()[2], a.strides()[0], a.strides()[1]]);
        assert!(Arc::ptr_eq(&a.buffer, &t.buffer));
    }

    #[test]
    fn transpose_axes_negative_indices() {
        let v: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let a = NdArray::from_buffer(Buffer::F64(v), vec![2, 3, 4], Order::C).unwrap();
        // -1,-2,-3 == 2,1,0 == a full reverse, matching numpy
        let t = a.transpose_axes(&[-1, -2, -3]).unwrap();
        assert_eq!(t.shape(), &[4, 3, 2]);
        assert_eq!(t.shape(), a.transpose().shape());
    }

    #[test]
    fn transpose_axes_wrong_count_is_value_error() {
        let a = arange(6);
        let a = a.reshape(&[2, 3]).unwrap();
        let err = a.transpose_axes(&[0]).unwrap_err();
        assert!(matches!(err, IonpError::Value(msg) if msg == "axes don't match array"));
    }

    #[test]
    fn transpose_axes_repeated_is_value_error() {
        let a = arange(6);
        let a = a.reshape(&[2, 3]).unwrap();
        let err = a.transpose_axes(&[0, 0]).unwrap_err();
        assert!(matches!(err, IonpError::Value(msg) if msg == "repeated axis in transpose"));
    }

    #[test]
    fn transpose_axes_out_of_bounds_is_index_error() {
        let a = arange(6);
        let a = a.reshape(&[2, 3]).unwrap();
        let err = a.transpose_axes(&[0, 5]).unwrap_err();
        assert!(matches!(err, IonpError::Index(_)));
    }

    #[test]
    fn transpose_axes_zero_d_identity_only() {
        let a = NdArray::from_buffer(Buffer::F64(vec![5.0]), vec![], Order::C).unwrap();
        assert_eq!(a.transpose_axes(&[]).unwrap().shape(), &[] as &[usize]);
        assert!(a.transpose_axes(&[0]).is_err());
    }

    #[test]
    fn reshape_order_f_matches_transpose_reshape_transpose_identity() {
        // a.reshape(shape, order='F') == a.T.reshape(shape[::-1]).T,
        // verified against real numpy 2.5.1 (np.arange(12).reshape(3,4)
        // .reshape(2,6,order='F') == [[0,8,5,2,10,7],[4,1,9,6,3,11]]).
        let v: Vec<i64> = (0..12).collect();
        let a = NdArray::from_buffer(Buffer::I64(v), vec![3, 4], Order::C).unwrap();
        let r = a.reshape_with_order(&[2, 6], "F").unwrap();
        assert_eq!(r.shape(), &[2, 6]);
        let c = r.to_contiguous();
        match &*c.buffer {
            Buffer::I64(data) => assert_eq!(data, &vec![0, 8, 5, 2, 10, 7, 4, 1, 9, 6, 3, 11]),
            _ => panic!(),
        }
    }

    #[test]
    fn reshape_order_c_is_plain_reshape() {
        let a = arange(6).reshape(&[2, 3]).unwrap();
        let r = a.reshape_with_order(&[3, 2], "C").unwrap();
        assert_eq!(r.shape(), a.reshape(&[3, 2]).unwrap().shape());
    }

    #[test]
    fn reshape_order_invalid_is_value_error() {
        let a = arange(6);
        let err = a.reshape_with_order(&[6], "Q").unwrap_err();
        assert!(matches!(err, IonpError::Value(_)));
    }

    #[test]
    fn reshape_view_when_contiguous() {
        let v: Vec<f64> = (0..6).map(|x| x as f64).collect();
        let a = NdArray::from_buffer(Buffer::F64(v), vec![2, 3], Order::C).unwrap();
        let r = a.reshape(&[3, 2]).unwrap();
        assert!(Arc::ptr_eq(&a.buffer, &r.buffer));
        assert_eq!(r.shape(), &[3, 2]);
    }

    #[test]
    fn reshape_after_transpose_copies() {
        let v: Vec<f64> = (0..6).map(|x| x as f64).collect();
        let a = NdArray::from_buffer(Buffer::F64(v), vec![2, 3], Order::C).unwrap();
        let t = a.transpose(); // no longer C-contiguous
        assert!(!t.is_c_contiguous());
        let r = t.reshape(&[6]).unwrap();
        assert!(!Arc::ptr_eq(&t.buffer, &r.buffer));
        // reshape(6) of the transpose must equal numpy's a.T.reshape(6):
        // [0,3,1,4,2,5]
        match &*r.buffer {
            Buffer::F64(data) => assert_eq!(data, &vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]),
            _ => panic!(),
        }
    }

    #[test]
    fn reshape_size_mismatch_errors() {
        let a = arange(6);
        let err = a.reshape(&[4]).unwrap_err();
        assert!(matches!(err, IonpError::Reshape { .. }));
    }

    #[test]
    fn basic_slice_positive_step() {
        let a = arange(10);
        let v = a
            .get_view(&[SliceItem::Slice { start: Some(2), stop: Some(8), step: Some(2) }])
            .unwrap();
        assert_eq!(v.shape(), &[3]);
        let c = v.to_contiguous();
        match &*c.buffer {
            Buffer::I64(data) => assert_eq!(data, &vec![2, 4, 6]),
            _ => panic!(),
        }
    }

    #[test]
    fn negative_step_reverses() {
        let a = arange(5);
        let v = a
            .get_view(&[SliceItem::Slice { start: None, stop: None, step: Some(-1) }])
            .unwrap();
        let c = v.to_contiguous();
        match &*c.buffer {
            Buffer::I64(data) => assert_eq!(data, &vec![4, 3, 2, 1, 0]),
            _ => panic!(),
        }
    }

    #[test]
    fn negative_index_and_out_of_bounds() {
        let a = arange(5);
        let v = a.get_view(&[SliceItem::Index(-1)]).unwrap();
        assert_eq!(v.shape(), &Vec::<usize>::new());
        let c = v.to_contiguous();
        match &*c.buffer {
            Buffer::I64(data) => assert_eq!(data, &vec![4]),
            _ => panic!(),
        }
        let err = a.get_view(&[SliceItem::Index(5)]).unwrap_err();
        assert!(matches!(err, IonpError::Index(_)));
    }

    #[test]
    fn empty_slice_out_of_range() {
        let a = arange(5);
        let v = a
            .get_view(&[SliceItem::Slice { start: Some(3), stop: Some(1), step: Some(1) }])
            .unwrap();
        assert_eq!(v.shape(), &[0]);
    }

    #[test]
    fn cast_materializes_new_dtype() {
        let a = arange(3);
        let f = a.cast_to(DType::F64);
        assert_eq!(f.dtype(), DType::F64);
        match &*f.buffer {
            Buffer::F64(data) => assert_eq!(data, &vec![0.0, 1.0, 2.0]),
            _ => panic!(),
        }
    }

    /// Ticket #7 (2026-08-08): `apply_ufunc_order`'s `'K'` arm
    /// (`ionp-py/src/lib.rs`), reproduced HERE byte-for-byte, because
    /// `ionp-py` is a pyo3 crate with no Rust-level `#[test]`s of its own
    /// and the committed Python differential suite's `check_strides`
    /// mechanism (`tests/differential/harness.py::_strides_match`) is
    /// DOCUMENTED as deliberately masking stride comparisons at any axis
    /// position whose extent is <= 1 ("CLASS B" divergence, added
    /// 2026-08-04) -- which is EXACTLY where every one of this ticket's
    /// own repro cases diverges. That means the Python suite is
    /// structurally incapable of catching a regression here: measured
    /// directly (see docs/), reverting this fix and rerunning the full
    /// differential suite produces the identical 29-item baseline failure
    /// set, with zero new failures, even though the underlying defect is
    /// fully back. This test is the ONLY committed regression guard for
    /// this defect class; it reads `.strides()` off the object under
    /// test, not a re-derived/converted value, and every expected value
    /// below is the ticket's own measured real-numpy repro table.
    ///
    /// Two independent single-operand bugs, both proven to bite in
    /// isolation during this ticket's investigation:
    ///
    /// 1. `multi_sorted_stride_perm` used to `break` out of its
    ///    axis-pair scan on the first AMBIGUOUS comparison (an operand
    ///    where one of the two axes being compared has size 1) instead of
    ///    continuing to scan for a later, decisive comparison -- and its
    ///    insertion was incremental (mutating mid-scan) instead of
    ///    numpy's real two-phase scan-then-bulk-shift. Both are fixed by
    ///    porting numpy's `PyArray_CreateMultiSortedStridePerm`
    ///    (`shape.c`) exactly.
    /// 2. Even with (1) fixed, a SINGLE operand whose whole-array
    ///    contiguity is genuinely, wholly F (e.g. `zeros((4,1,2))
    ///    .copy('F')`, which real numpy reports as `F_CONTIGUOUS=True`
    ///    despite the interior size-1 axis) is not decidable by ANY
    ///    per-axis-pair comparison alone, because the size-1 axis is
    ///    ambiguous on every comparison that touches it. Real numpy
    ///    resolves this at a higher level first, via a whole-array
    ///    flags check -- ported here as `k_order_perm`'s "all operands
    ///    agree" fast path, which must require ALL operands to agree
    ///    (not "first operand wins", which was this ticket's own
    ///    previous, order-DEPENDENT and measurably-regressing attempt;
    ///    see docs/ for the 42-item regression it caused).
    fn k_order_perm(operands: &[&NdArray]) -> Vec<usize> {
        let ndim = operands[0].shape().len();
        if operands.iter().all(|a| a.is_c_contiguous()) {
            (0..ndim).collect()
        } else if operands.iter().all(|a| a.is_f_contiguous()) {
            (0..ndim).rev().collect()
        } else {
            let pairs: Vec<(&[usize], &[isize])> =
                operands.iter().map(|a| (a.shape(), a.strides())).collect();
            NdArray::multi_sorted_stride_perm(ndim, &pairs)
        }
    }

    fn f_contiguous(shape: &[usize]) -> NdArray {
        let n: usize = shape.iter().product();
        let v: Vec<f64> = (0..n as i64).map(|x| x as f64).collect();
        NdArray::from_buffer(Buffer::F64(v), shape.to_vec(), Order::F).unwrap()
    }

    /// `zeros((4,1,2)).copy('F')` in real numpy: strides (8,32,32) BYTES
    /// (itemsize 8) == (1,4,4) ELEMENTS -- `NdArray::strides()` is always
    /// in elements, never bytes (see `transpose_reverses_shape_and_strides`
    /// above for the same convention: a (2,3) C-order array's transpose is
    /// asserted as element-strides `[1,3]`, not byte-strides `[8,24]`).
    /// The ticket's headline repro case: a length-1 MIDDLE axis. Pre-fix
    /// anionpy gave byte-strides (16,16,8), i.e. element-strides (2,2,1)
    /// -- wrong.
    #[test]
    fn k_order_unary_f_contiguous_len1_middle_axis() {
        let a = f_contiguous(&[4, 1, 2]);
        assert_eq!(a.strides(), &[1, 4, 4], "precondition: base must genuinely be F-contiguous");
        assert!(a.is_f_contiguous());
        let perm = k_order_perm(&[&a]);
        let out = a.relayout_by_perm(&perm);
        assert_eq!(out.strides(), &[1, 4, 4]);
    }

    /// `zeros((1,4,2)).copy('F')`: strides (8,8,32) bytes == (1,1,4)
    /// elements. Length-1 axis 0.
    #[test]
    fn k_order_unary_f_contiguous_len1_first_axis() {
        let a = f_contiguous(&[1, 4, 2]);
        assert_eq!(a.strides(), &[1, 1, 4], "precondition");
        let perm = k_order_perm(&[&a]);
        let out = a.relayout_by_perm(&perm);
        assert_eq!(out.strides(), &[1, 1, 4]);
    }

    /// `zeros((4,2,1)).copy('F')`: strides (8,32,64) bytes == (1,4,8)
    /// elements. Length-1 axis 2.
    #[test]
    fn k_order_unary_f_contiguous_len1_last_axis() {
        let a = f_contiguous(&[4, 2, 1]);
        assert_eq!(a.strides(), &[1, 4, 8], "precondition");
        let perm = k_order_perm(&[&a]);
        let out = a.relayout_by_perm(&perm);
        assert_eq!(out.strides(), &[1, 4, 8]);
    }

    /// Three shapes without any size-1 axis must be unaffected (these
    /// already passed before this ticket's fix; guards against the fix
    /// itself introducing a regression on the plain case).
    #[test]
    fn k_order_unary_f_contiguous_no_len1_axis_unaffected() {
        for shape in [vec![4usize, 2], vec![3, 4], vec![2, 3, 4]] {
            let a = f_contiguous(&shape);
            assert!(a.is_f_contiguous());
            let expected = a.strides().to_vec();
            let perm = k_order_perm(&[&a]);
            let out = a.relayout_by_perm(&perm);
            assert_eq!(out.strides(), &expected[..], "shape {:?}", shape);
        }
    }

    /// Two genuinely conflicting C/F operands must resolve to C order
    /// regardless of argument order (verified live against real numpy:
    /// `np.add(c, f)` and `np.add(f, c)` both give the same C-order
    /// result) -- the previous "first operand wins" fast path was
    /// order-dependent and wrong.
    #[test]
    fn k_order_binary_c_f_conflict_resolves_to_c_order_independent() {
        let c = arange(6).reshape(&[2, 3]).unwrap();
        assert!(c.is_c_contiguous());
        let f = f_contiguous(&[2, 3]);
        assert!(f.is_f_contiguous());

        let perm_cf = k_order_perm(&[&c, &f]);
        let perm_fc = k_order_perm(&[&f, &c]);
        assert_eq!(perm_cf, perm_fc, "argument order must not change the decision");

        let out = c.relayout_by_perm(&perm_cf);
        assert_eq!(out.strides(), c.strides(), "conflict must resolve to C order");
    }
}
