//! Set-operation / diff / unique family -- `unique`, `unique_all`,
//! `unique_counts`, `unique_inverse`, `unique_values`, `intersect1d`,
//! `union1d`, `setdiff1d`, `setxor1d`, `isin`, `ediff1d`, `diff`,
//! `trim_zeros`. `count_nonzero`, `sort_complex`, `searchsorted`,
//! `argsort`/`partition`/`argpartition` are OUT OF SCOPE for this file --
//! `count_nonzero` is already implemented in `ionp-py/src/reductions.rs`
//! (declared "exact" in the ledger), `sort_complex`/`searchsorted` are
//! already implemented in this crate's `sort.rs` (also declared), and
//! `argsort`/`partition`/`argpartition` are deliberately out of scope
//! project-wide per `sort.rs`'s own module doc (algorithm-internal
//! tie/pivot order, not well-posed to match bit-for-bit without
//! replicating numpy's introselect implementation line for line). `in1d`
//! does not exist in numpy 2.5.1 (removed) and is absent from
//! `tools/numpy_surface.json` -- not a gap, not implemented.
//!
//! This file's NaN-canonicalization constants and stable-sort/group
//! primitive are fresh, local implementations, written and verified
//! independently against real numpy 2.5.1. The ONE exception (2026-08-01):
//! complex ordering (`complex_cmp` below) is now a thin wrapper delegating
//! to `crate::sort::complex_cmp` (made `pub(crate)` for this reuse) rather
//! than a second, independently hand-derived comparator -- this file's own
//! original version repeated a mistake `sort.rs` had already found and
//! fixed (numpy's real complex `LT` is not "bucket by any-NaN-component,
//! then lexicographic real/imag"), which happened to pass this file's
//! probe corpus without being the same relation numpy actually uses. See
//! `complex_cmp`'s own doc comment below for the full account.
//!
//! ---------------------------------------------------------------------
//! ONE algorithm backs the whole `unique*` family
//! ---------------------------------------------------------------------
//! `unique_raw()` below does a single STABLE sort-by-key (Rust's
//! `slice::sort_by` is guaranteed stable) of the flattened input's
//! *indices*, groups adjacent equal elements (`eq`, not `cmp` -- see
//! below), and returns, per group in ascending-key order: the ORIGINAL
//! index of the group's first (stable-order) representative, plus the
//! per-original-position group id (the "inverse" array) and per-group
//! element count. Every one of `unique_values`/`unique_counts`/
//! `unique_inverse`/`unique_all` is a thin projection of this one result.
//!
//! Two things a naive `Ord`-based sort gets wrong for floats/complex,
//! fixed by using a dedicated `(cmp, eq)` pair instead of `PartialOrd`:
//!   1. NaN must sort LAST but compare EQUAL to every other NaN for
//!      grouping (`equal_nan=True`, numpy's default and the only form this
//!      file implements -- `equal_nan=False`'s exact internal behavior for
//!      the complex dtypes was independently probed and found genuinely
//!      confusing/underspecified against real numpy 2.5.1, see the
//!      `EQUAL_NAN_FALSE_NOT_IMPLEMENTED` note below; not implemented,
//!      left as an honest, reported gap rather than guessed at).
//!   2. `-0.0`/`+0.0` compare EQUAL (`cmp` returns `Equal`, `eq` returns
//!      `true`) -- their relative sign in the OUTPUT is therefore whatever
//!      the stable sort's original-order tie-break picks, same as this
//!      project's existing `sort`/`sort_complex`/`ndarray.sort`
//!      `signed_zero_tie_exempt` precedent (see `sort_cases.py`'s module
//!      docstring). This crate does not know about that Python-side
//!      exemption mechanism; it is applied at the differential-test layer
//!      (`setops_cases.py`), not here.
//!
//! numpy's *actual* internal algorithm choice differs by which return_*
//! flags are requested (`optional_indices = return_index or
//! return_inverse`; true -> payload-preserving argsort gather, kind=
//! "mergesort" if return_index else "quicksort"; false -> in-place
//! `ar.sort()`, which both canonicalizes every NaN's bit pattern to a
//! fixed value AND is non-stable). This file's ONE stable-gather algorithm
//! exactly reproduces the `return_index=True` (mergesort, STABLE) case bit
//! for bit -- verified: `unique_all` needs no signed-zero exemption at
//! all. For the other three shapes:
//!   - `unique_values`/bare `unique()`/`unique_counts`'s `values` output
//!     additionally NaN-canonicalizes (see `canonicalize_nan_*` below,
//!     same fixed bit patterns as `sort.rs`'s own canonicalization) to
//!     reproduce the in-place-sort path's payload-clobbering behavior, and
//!     needs the signed-zero exemption (in-place sort is non-stable).
//!   - `unique_inverse`'s `values` output is payload-preserving (argsort
//!     path) like `unique_all`, but numpy's own argsort there uses
//!     "quicksort" (NOT stable) -- this file's stable gather can therefore
//!     pick a different -0.0/+0.0 representative than numpy's quicksort
//!     does; needs the signed-zero exemption too. Its `inverse` array does
//!     NOT need any exemption: which GROUP a value belongs to, and which
//!     ascending-key rank that group occupies, is fully determined by the
//!     (numpy-equivalent) `cmp`/`eq` pair alone, independent of which
//!     physical -0.0-vs-+0.0 instance the sort happened to place first.
use std::cmp::Ordering;

use half::f16;
use num_complex::Complex;

use crate::array::{NdArray, Order as ArrOrder};
use crate::buffer::{Buffer, C128, C64};
use crate::dtype::{can_cast, promote_dtype, DType};
use crate::error::IonpError;

// ---------------------------------------------------------------------------
// NaN payload canonicalization -- identical fixed bit patterns to
// `sort.rs`'s own `canonicalize_nans_f64/f32/f16` (verified independently
// against real numpy 2.5.1's in-place-sort NaN-clobbering behavior, not
// copied from that file since it is private and out of this file's fence).
// ---------------------------------------------------------------------------

const CANON_F64_NAN: u64 = 0x7fff_ffff_ffff_ffff;
const CANON_F32_NAN: u32 = 0x7fff_ffff;
const CANON_F16_NAN: u16 = 0x7fff;

fn canon_f64(v: f64) -> f64 {
    if v.is_nan() { f64::from_bits(CANON_F64_NAN) } else { v }
}
fn canon_f32(v: f32) -> f32 {
    if v.is_nan() { f32::from_bits(CANON_F32_NAN) } else { v }
}
fn canon_f16(v: f16) -> f16 {
    if v.is_nan() { f16::from_bits(CANON_F16_NAN) } else { v }
}
// Complex NaN canonicalization uses a DIFFERENT bit pattern than the plain
// float sort path above, and canonicalizes ONLY the component(s) that are
// actually NaN (not both components unconditionally) -- verified directly
// against real numpy 2.5.1: `np.unique(np.array([nan+1j], dtype=complex64))`
// yields a value whose real part is the STANDARD quiet-NaN pattern
// `0x7fc00000` (not this file's real-sort `0x7fff_ffff`) while the
// imaginary part stays the original `1.0`, bit for bit. Complex arrays'
// in-place `.sort()` goes through a different internal kernel than real
// arrays' own `.sort()`, hence the different canonical pattern.
const CANON_C64_NAN_COMPONENT: u32 = 0x7fc0_0000;
const CANON_C128_NAN_COMPONENT: u64 = 0x7ff8_0000_0000_0000;

fn canon_c64(z: C64) -> C64 {
    Complex::new(
        if z.re.is_nan() { f32::from_bits(CANON_C64_NAN_COMPONENT) } else { z.re },
        if z.im.is_nan() { f32::from_bits(CANON_C64_NAN_COMPONENT) } else { z.im },
    )
}
fn canon_c128(z: C128) -> C128 {
    Complex::new(
        if z.re.is_nan() { f64::from_bits(CANON_C128_NAN_COMPONENT) } else { z.re },
        if z.im.is_nan() { f64::from_bits(CANON_C128_NAN_COMPONENT) } else { z.im },
    )
}

// ---------------------------------------------------------------------------
// Generic stable group-by-key primitive. `cmp` must be a total order
// consistent with `eq` (`eq(a,b) == true` implies `cmp(a,b) == Equal`).
// ---------------------------------------------------------------------------

struct GroupResult {
    /// Original flat index of each group's stable-first representative,
    /// in ascending-key (output) order.
    first_index: Vec<usize>,
    /// Per-original-flat-position group id (same length as input).
    inverse: Vec<i64>,
    /// Per-group element count, same order as `first_index`.
    counts: Vec<i64>,
}

fn stable_group_by<T: Copy>(
    data: &[T],
    cmp: impl Fn(T, T) -> Ordering,
    eq: impl Fn(T, T) -> bool,
) -> GroupResult {
    let n = data.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| cmp(data[i], data[j]));

    let mut first_index: Vec<usize> = Vec::new();
    let mut counts: Vec<i64> = Vec::new();
    let mut group_of_pos: Vec<i64> = vec![0; n];
    for (pos, &oi) in order.iter().enumerate() {
        if pos == 0 || !eq(data[order[pos - 1]], data[oi]) {
            first_index.push(oi);
            counts.push(0);
        }
        *counts.last_mut().unwrap() += 1;
        group_of_pos[pos] = (first_index.len() - 1) as i64;
    }
    let mut inverse: Vec<i64> = vec![0; n];
    for (pos, &oi) in order.iter().enumerate() {
        inverse[oi] = group_of_pos[pos];
    }
    GroupResult { first_index, inverse, counts }
}

/// Targeted fix-up for the ONE case in this file where `cmp` and `eq` are
/// deliberately inconsistent by design: complex `equal_nan=True`, where
/// `complex_eq` treats every NaN-having value as mutually equal (so they
/// all land in one merged group) while `complex_cmp` still orders the
/// NaN-having bucket internally by real/imaginary component (so a stable
/// sort does NOT preserve the members' original relative array order
/// within that bucket, unlike every other group in this file, where `cmp`-
/// tied members already keep original order for free via the stable sort).
/// Real numpy's own NaN-group representative choice for this merged group
/// depends on WHICH internal path produced it, and callers must only invoke
/// this fix-up on the canonicalizing (`canonicalize_nan=True`, in-place-
/// `.sort()`-backed) path -- verified directly on both:
///   - canonicalizing path (bare `unique()`/`unique_values`, no `return_
///     index`/`return_inverse` requested): `np.unique([nan+0j, 0+nanj,
///     1+2j, nan+nanj])` keeps `nan+0j` (original index 0, the member that
///     occurred FIRST in the pre-sort array) as the NaN group's
///     representative -- NOT `0+nanj`, which `complex_cmp`'s secondary
///     real/imaginary order would otherwise place first in sorted position.
///   - non-canonicalizing path (`return_index`/`return_inverse` requested,
///     `canonicalize_nan=False`): verified `np.unique(ar, return_index=
///     True, ...)` on the SAME array instead keeps `0+nanj` (original
///     index 1) -- i.e. the plain sorted-bucket-order representative that
///     `stable_group_by` already produces with NO fix-up needed, because
///     this path's underlying argsort does not carry the same first-
///     original-occurrence rule the in-place sort does. Only groups whose
///     current representative already has a NaN component are touched --
///     every other (non-NaN) group is left exactly as `stable_group_by`
///     computed it, since for those `cmp`/`eq` agree and the representative
///     is already correct.
fn fixup_complex_nan_group_representative<T: Copy>(
    g: &mut GroupResult,
    data: &[Complex<T>],
    is_nan: impl Fn(T) -> bool,
) {
    let mut min_idx: Vec<usize> = vec![usize::MAX; g.first_index.len()];
    for (i, &gid) in g.inverse.iter().enumerate() {
        let gid = gid as usize;
        if i < min_idx[gid] {
            min_idx[gid] = i;
        }
    }
    let reps = g.first_index.clone();
    for (gid, &rep) in reps.iter().enumerate() {
        let v = data[rep];
        if is_nan(v.re) || is_nan(v.im) {
            g.first_index[gid] = min_idx[gid];
        }
    }
}

// ---------------------------------------------------------------------------
// Per-dtype (cmp, eq) pairs.
// ---------------------------------------------------------------------------

/// Real numpy's float sort uses a TOTAL order that is strictly finer than
/// IEEE `==`: `-0.0` always sorts strictly before `+0.0` (verified directly:
/// `np.sort([0.0,-0.0])` -> `[-0.0, 0.0]`), even though `-0.0 == 0.0` is
/// `True`. Without this, a plain `partial_cmp` treats the two as `Equal`
/// and a *stable* sort then just preserves whichever arrived first in the
/// input -- which silently disagrees with numpy on which of `-0.0`/`+0.0`
/// survives as `unique`'s deduplication representative whenever the input
/// order happens to put `+0.0` first. (Complex sort does NOT carry this
/// same tie-break -- verified separately: `np.sort` on complex zeros of
/// varying sign leaves them in original input order -- so `complex_cmp`
/// below intentionally does not add it.)
fn float_cmp<T: PartialOrd + Copy>(a: T, b: T, is_nan: impl Fn(T) -> bool, is_neg: impl Fn(T) -> bool) -> Ordering {
    let (na, nb) = (is_nan(a), is_nan(b));
    match (na, nb) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => match a.partial_cmp(&b).unwrap_or(Ordering::Equal) {
            Ordering::Equal => match (is_neg(a), is_neg(b)) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => Ordering::Equal,
            },
            other => other,
        },
    }
}

fn is_neg_f16(v: f16) -> bool { v.to_bits() & 0x8000 != 0 }
fn is_neg_f32(v: f32) -> bool { v.to_bits() & 0x8000_0000 != 0 }
fn is_neg_f64(v: f64) -> bool { v.to_bits() & 0x8000_0000_0000_0000 != 0 }
/// `equal_nan=True` is numpy's `_unique1d` default (bare `unique()`); the
/// array-API family (`unique_all`/`unique_counts`/`unique_inverse`) always
/// calls it with `equal_nan=False`. Both share the SAME total order
/// (`cmp` above, NaN always sorts last) -- `equal_nan` only changes whether
/// two NaNs are considered the SAME group for `eq`/grouping purposes: True
/// collapses every NaN into a single group (numpy's own
/// `mask[aux_firstnan+1:] = False` special case), False leaves every NaN a
/// singleton (falls through to the plain `aux[1:] != aux[:-1]` branch,
/// where IEEE `NaN != NaN` is always true regardless of matching bits).
/// Verified directly against real numpy 2.5.1's `_unique1d` source.
fn float_eq<T: PartialEq + Copy>(a: T, b: T, is_nan: impl Fn(T) -> bool, equal_nan: bool) -> bool {
    (equal_nan && is_nan(a) && is_nan(b)) || a == b
}

fn f16_is_nan(v: f16) -> bool { v.is_nan() }

/// Real numpy's complex ORDERING (used for placement by `.sort()`, distinct
/// from grouping/`eq`) is NOT this file's own logic -- it delegates to
/// `sort::complex_cmp`, numpy's actual `@TYPE@_LT` comparator (reverse-
/// engineered from `npysort_common.h.src` and confirmed bit-exact against
/// real numpy 2.5.1 via a 6000-case randomized stress sweep, see that
/// function's doc comment for the full derivation and the "NOT plain
/// lexicographic real-then-imaginary" warning). An earlier version of this
/// function in this file independently re-derived a DIFFERENT (and, on
/// numpy's actual source, WRONG) hand-verified "bucket by hasnan, then
/// lexicographic" rule, which happened to pass this file's own probe corpus
/// but was never checked against the sort.rs sweep -- replaced with the one
/// true implementation rather than maintaining two disagreeing "verified"
/// complex comparators in the same crate.
fn complex_cmp<T: PartialOrd + Copy>(a: Complex<T>, b: Complex<T>, is_nan: impl Fn(T) -> bool + Copy) -> Ordering {
    crate::sort::complex_cmp(&a, &b, is_nan, false)
}
fn complex_eq<T: PartialEq + Copy>(a: Complex<T>, b: Complex<T>, is_nan: impl Fn(T) -> bool + Copy, equal_nan: bool) -> bool {
    let a_nan = is_nan(a.re) || is_nan(a.im);
    let b_nan = is_nan(b.re) || is_nan(b.im);
    if equal_nan && a_nan && b_nan {
        return true;
    }
    a == b
}

/// Dispatch a flattened, C-contiguous `NdArray` to a `GroupResult` plus a
/// `Buffer` of gathered (payload-preserving) representative values, in
/// ascending-key output order. `canonicalize_nan`: whether to additionally
/// clobber every NaN's payload in the *returned values buffer only* (the
/// `GroupResult` itself -- indices/inverse/counts -- is identical either
/// way, see module doc).
fn unique_raw(flat: &NdArray, canonicalize_nan: bool, equal_nan: bool) -> Result<(GroupResult, Buffer), IonpError> {
    macro_rules! ord_dtype {
        ($variant:ident, $buf:expr) => {{
            let data: &[_] = $buf;
            let g = stable_group_by(data, |a, b| a.cmp(&b), |a, b| a == b);
            let values: Vec<_> = g.first_index.iter().map(|&i| data[i]).collect();
            (g, Buffer::$variant(values))
        }};
    }
    match flat.buffer() {
        Buffer::Bool(v) => {
            let g = stable_group_by(v.as_slice(), |a, b| a.cmp(&b), |a, b| a == b);
            let values: Vec<bool> = g.first_index.iter().map(|&i| v[i]).collect();
            Ok((g, Buffer::Bool(values)))
        }
        Buffer::I8(v) => { let (g, b) = ord_dtype!(I8, v.as_slice()); Ok((g, b)) }
        Buffer::I16(v) => { let (g, b) = ord_dtype!(I16, v.as_slice()); Ok((g, b)) }
        Buffer::I32(v) => { let (g, b) = ord_dtype!(I32, v.as_slice()); Ok((g, b)) }
        Buffer::I64(v) => { let (g, b) = ord_dtype!(I64, v.as_slice()); Ok((g, b)) }
        Buffer::U8(v) => { let (g, b) = ord_dtype!(U8, v.as_slice()); Ok((g, b)) }
        Buffer::U16(v) => { let (g, b) = ord_dtype!(U16, v.as_slice()); Ok((g, b)) }
        Buffer::U32(v) => { let (g, b) = ord_dtype!(U32, v.as_slice()); Ok((g, b)) }
        Buffer::U64(v) => { let (g, b) = ord_dtype!(U64, v.as_slice()); Ok((g, b)) }
        Buffer::F16(v) => {
            let data = v.as_slice();
            // Real numpy's NaN-payload clobbering is a side effect of
            // actually RUNNING its in-place sort kernel on the buffer --
            // verified directly (2026-08-01): `np.unique(np.array(nan,
            // dtype=float32))` and `np.unique(np.array([nan],
            // dtype=float32))` both preserve the ORIGINAL NaN bit pattern
            // (`0x7fc00000`) untouched, while `np.unique(np.array([nan,
            // nan, 1.0], dtype=float32))` clobbers both NaNs to
            // `0x7fffffff`. A single-element buffer never invokes that
            // kernel, so canonicalization (and, for the same reason, the
            // `-0.0`-before-`+0.0` tie-break, which only exists as an
            // artifact of the same kernel) must be gated on `data.len() >
            // 1`, not applied unconditionally whenever `canonicalize_nan`
            // is requested. This also matches `sort.rs`'s own `!stable &&
            // shape[axis] > 1` gate for the same underlying reason.
            let cn = canonicalize_nan && data.len() > 1;
            // The `-0.0`-before-`+0.0` total-order tie-break only applies to
            // real numpy's in-place-sort (`canonicalize_nan=True`) path --
            // its argsort/mergesort path (`canonicalize_nan=False`, used
            // whenever `return_index`/`return_inverse` is requested) does
            // NOT reorder equal-by-`==` zeros, so applying the tie-break
            // there would wrongly change which original index is picked as
            // a group's representative (verified: `unique_all`'s `indices`/
            // `inverse_indices` need NO signed-zero exemption at all, per
            // this file's own module doc -- that is only still true if this
            // path stays a plain stable sort on ties).
            let neg = if cn { is_neg_f16 as fn(f16) -> bool } else { (|_: f16| false) as fn(f16) -> bool };
            let g = stable_group_by(data, |a, b| float_cmp(a, b, f16_is_nan, neg), |a, b| float_eq(a, b, f16_is_nan, equal_nan));
            let values: Vec<f16> = g.first_index.iter().map(|&i| {
                let x = data[i];
                if cn { canon_f16(x) } else { x }
            }).collect();
            Ok((g, Buffer::F16(values)))
        }
        Buffer::F32(v) => {
            let data = v.as_slice();
            let cn = canonicalize_nan && data.len() > 1;
            let neg = if cn { is_neg_f32 as fn(f32) -> bool } else { (|_: f32| false) as fn(f32) -> bool };
            let g = stable_group_by(data, |a, b| float_cmp(a, b, f32::is_nan, neg), |a, b| float_eq(a, b, f32::is_nan, equal_nan));
            let values: Vec<f32> = g.first_index.iter().map(|&i| {
                let x = data[i];
                if cn { canon_f32(x) } else { x }
            }).collect();
            Ok((g, Buffer::F32(values)))
        }
        Buffer::F64(v) => {
            let data = v.as_slice();
            let cn = canonicalize_nan && data.len() > 1;
            let neg = if cn { is_neg_f64 as fn(f64) -> bool } else { (|_: f64| false) as fn(f64) -> bool };
            let g = stable_group_by(data, |a, b| float_cmp(a, b, f64::is_nan, neg), |a, b| float_eq(a, b, f64::is_nan, equal_nan));
            let values: Vec<f64> = g.first_index.iter().map(|&i| {
                let x = data[i];
                if cn { canon_f64(x) } else { x }
            }).collect();
            Ok((g, Buffer::F64(values)))
        }
        Buffer::C64(v) => {
            let data = v.as_slice();
            let cn = canonicalize_nan && data.len() > 1;
            let mut g = stable_group_by(data, |a, b| complex_cmp(a, b, f32::is_nan), |a, b| complex_eq(a, b, f32::is_nan, equal_nan));
            if equal_nan && cn {
                fixup_complex_nan_group_representative(&mut g, data, f32::is_nan);
            }
            let values: Vec<C64> = g.first_index.iter().map(|&i| {
                let x = data[i];
                if cn { canon_c64(x) } else { x }
            }).collect();
            Ok((g, Buffer::C64(values)))
        }
        Buffer::C128(v) => {
            let data = v.as_slice();
            let cn = canonicalize_nan && data.len() > 1;
            let mut g = stable_group_by(data, |a, b| complex_cmp(a, b, f64::is_nan), |a, b| complex_eq(a, b, f64::is_nan, equal_nan));
            if equal_nan && cn {
                fixup_complex_nan_group_representative(&mut g, data, f64::is_nan);
            }
            let values: Vec<C128> = g.first_index.iter().map(|&i| {
                let x = data[i];
                if cn { canon_c128(x) } else { x }
            }).collect();
            Ok((g, Buffer::C128(values)))
        }
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("setops.rs: no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
    }
}

fn i64_array(v: Vec<i64>) -> Result<NdArray, IonpError> {
    let n = v.len();
    NdArray::from_buffer(Buffer::I64(v), vec![n], ArrOrder::C)
}

// ---------------------------------------------------------------------------
// Public unique* family
// ---------------------------------------------------------------------------

/// Reshape a flat `i64` inverse-indices vector to the ORIGINAL (pre-flatten)
/// input shape -- real numpy's `_unique1d(..., inverse_shape=ar.shape)`
/// does `inv_idx.reshape(inverse_shape)` before returning (verified via
/// numpy 2.5.1 source and its own "versionchanged 2.0" docstring note).
/// `unique_values`/`unique_counts` never call this (their `values`/`counts`
/// outputs are always 1-D regardless of input ndim); only `inverse` needs it.
/// Reshape the flat per-position group-id array back to the ORIGINAL
/// (pre-flatten) input shape -- including 0-d: a 0-d input's `inverse`/
/// `inverse_indices` output is itself a 0-d scalar array (shape `()`), NOT
/// shape `(1,)` (verified directly: `np.unique_inverse(np.array(5)).
/// inverse_indices.shape == ()`).
fn reshape_inverse(inverse: Vec<i64>, orig_shape: &[usize]) -> Result<NdArray, IonpError> {
    NdArray::from_buffer(Buffer::I64(inverse), orig_shape.to_vec(), ArrOrder::C)
}

/// Internal helper backing bare `np.unique()`'s NON-optional-indices,
/// canonicalizing path (`ar.sort()`) -- used by `unique_values` (this
/// crate's internal helper, also reused by the set-op family below) and by
/// `unique_counts`. `equal_nan=True` here matches bare `unique()`'s own
/// default; `unique_counts` (array API) always passes `equal_nan=False`
/// (see its own function below) despite ALSO using this canonicalizing
/// path -- real numpy's `_unique1d`: `optional_indices = return_index or
/// return_inverse` does NOT include `return_counts`, so `unique_counts`
/// (`return_counts=True` only) takes the SAME `ar.sort()` in-place
/// canonicalizing branch as bare `unique()`, just with `equal_nan=False`
/// grouping. Verified directly against `_unique1d`'s source.
pub fn unique_values(a: &NdArray) -> Result<NdArray, IonpError> {
    let flat = a.ravel_order("C")?;
    let (g, values) = unique_raw(&flat, true, true)?;
    let n = g.first_index.len();
    NdArray::from_buffer(values, vec![n], ArrOrder::C)
}

/// `np.unique_counts(x)` -> `(values, counts)`. Array-API function: always
/// `equal_nan=False` (each NaN its own singleton group), but still the
/// canonicalizing in-place-sort path (see `unique_values`'s doc above for
/// why `return_counts` alone does not trigger the payload-preserving path).
pub fn unique_counts(a: &NdArray) -> Result<(NdArray, NdArray), IonpError> {
    let flat = a.ravel_order("C")?;
    let (g, values) = unique_raw(&flat, true, false)?;
    let n = g.first_index.len();
    let values_arr = NdArray::from_buffer(values, vec![n], ArrOrder::C)?;
    let counts_arr = i64_array(g.counts)?;
    Ok((values_arr, counts_arr))
}

/// `np.unique_inverse(x)` -> `(values, inverse_indices)`. Array-API
/// function: `equal_nan=False`, payload-preserving (argsort) path like
/// `unique_all` -- but real numpy's argsort there is "quicksort" (not
/// stable), so `values` (never `inverse`) is a signed-zero-exemption
/// concern (see module doc). `inverse_indices` is reshaped to the ORIGINAL
/// input's shape, not left flattened.
pub fn unique_inverse(a: &NdArray) -> Result<(NdArray, NdArray), IonpError> {
    let orig_shape = a.shape().to_vec();
    let flat = a.ravel_order("C")?;
    let (g, values) = unique_raw(&flat, false, false)?;
    let n = g.first_index.len();
    let values_arr = NdArray::from_buffer(values, vec![n], ArrOrder::C)?;
    let inverse_arr = reshape_inverse(g.inverse, &orig_shape)?;
    Ok((values_arr, inverse_arr))
}

/// `np.unique_all(x)` -> `(values, indices, inverse_indices, counts)`.
/// Array-API function: `equal_nan=False`. Real numpy's argsort here uses
/// "mergesort" (stable) -- this file's algorithm IS that path exactly, no
/// signed-zero exemption of any kind needed. `inverse_indices` is reshaped
/// to the ORIGINAL input's shape, not left flattened.
pub fn unique_all(a: &NdArray) -> Result<(NdArray, NdArray, NdArray, NdArray), IonpError> {
    let orig_shape = a.shape().to_vec();
    let flat = a.ravel_order("C")?;
    let (g, values) = unique_raw(&flat, false, false)?;
    let n = g.first_index.len();
    let values_arr = NdArray::from_buffer(values, vec![n], ArrOrder::C)?;
    let indices_arr = i64_array(g.first_index.iter().map(|&i| i as i64).collect())?;
    let inverse_arr = reshape_inverse(g.inverse, &orig_shape)?;
    let counts_arr = i64_array(g.counts)?;
    Ok((values_arr, indices_arr, inverse_arr, counts_arr))
}

/// `np.unique(ar, return_index=.., return_inverse=.., return_counts=..,
/// *, equal_nan=..)` (flattened / `axis=None` only) -- the general
/// core primitive the Python binding dispatches to based on which flags
/// are set. `optional_indices = return_index || return_inverse` selects
/// canonicalizing-in-place-sort (false) vs payload-preserving-argsort
/// (true), exactly mirroring real numpy's `_unique1d` branch (this file's
/// hash-table fast path is deliberately NOT replicated -- see module doc
/// and `unique_values`'s python-binding-level scope note; the final
/// `if sorted: hash_unique.sort()` step numpy takes when `sorted=True`
/// (bare `unique()`'s own default) re-canonicalizes/re-sorts the hash
/// path's output the same way this function's sort-based path already
/// does, so this simpler always-sort implementation is verified
/// equivalent for bare `unique()`'s default call form).
pub fn unique_general(
    a: &NdArray,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
    equal_nan: bool,
) -> Result<(NdArray, Option<NdArray>, Option<NdArray>, Option<NdArray>), IonpError> {
    let optional_indices = return_index || return_inverse;
    let orig_shape = a.shape().to_vec();
    let flat = a.ravel_order("C")?;
    let (g, values) = unique_raw(&flat, !optional_indices, equal_nan)?;
    let n = g.first_index.len();
    let values_arr = NdArray::from_buffer(values, vec![n], ArrOrder::C)?;
    let indices_arr = if return_index {
        Some(i64_array(g.first_index.iter().map(|&i| i as i64).collect())?)
    } else {
        None
    };
    let inverse_arr = if return_inverse {
        Some(reshape_inverse(g.inverse, &orig_shape)?)
    } else {
        None
    };
    let counts_arr = if return_counts { Some(i64_array(g.counts)?) } else { None };
    Ok((values_arr, indices_arr, inverse_arr, counts_arr))
}

// ---------------------------------------------------------------------------
// unique(axis=...)
// ---------------------------------------------------------------------------
//
// Real numpy's `unique()` (`_arraysetops_impl.py`, verified against the
// live 2.5.1 source, not memory): `if axis is None or ar.ndim == 1:` takes
// the SAME flattened `_unique1d` path this file's `unique_general` already
// implements -- for `ar.ndim <= 1` an explicit `axis` is validated
// (`normalize_axis_index`) but then genuinely IGNORED for computation; the
// dispatcher hard-codes `axis=None` into its own internal `_unique1d` call
// regardless of what `axis` the caller passed. Verified directly:
// `np.unique([nan,nan,1.0], axis=0)` collapses both NaNs into one (the
// bare/default `equal_nan=True` behavior), UNLIKE the >=2-D axis path below,
// which never collapses NaNs regardless of `equal_nan`.
//
// Only for `ar.ndim >= 2` does real numpy take the genuinely different
// path: `moveaxis(ar, axis, 0)`, reshape to 2-D `(n, m)`, view the `m`
// columns as one structured/void-dtype field per column, and run
// `_unique1d` on that 1-D structured array with `axis=<original-axis>`
// (only used there to select `inv_idx.reshape(inverse_shape)` vs not, and
// to strip the axis-error message's "axis1"/"axis2" prefix -- verified:
// `np.unique(np.zeros((2,3)), axis=5)` raises `AxisError('axis 5 is out of
// bounds for array of dimension 2')`, UNPREFIXED, unlike `trim_zeros`'s own
// argname-prefixed axis error).
//
// Because the structured/void dtype's `.kind` is `'V'`, NOT one of
// `"cfmM"`, `_unique1d`'s `equal_nan`-aware NaN-collapsing branch
// (`if equal_nan and ... aux.dtype.kind in "cfmM" ...`) is UNREACHABLE for
// this path -- verified directly, `equal_nan=True` and `equal_nan=False`
// give byte-identical results for `axis=0` on a NaN-containing 2-D array,
// and NEITHER collapses NaN rows, even bit-identical ones. `equal_nan` is
// therefore not threaded into `unique_axis` at all -- there is no
// observable knob to thread.
//
// The structured view also never takes the hash-fast path (structured
// dtypes are not hash-fast-path-eligible), so real numpy's own `sorted`
// flag has no observable effect once `axis` is given -- verified directly
// (`sorted=True`/`sorted=False` byte-identical on axis=0 for several
// dtypes) -- `unique_axis` below therefore has no `sorted` parameter
// either; the PyO3 binding accepts-and-ignores `sorted` whenever `axis` is
// given, for the same reason.
//
// This crate's own comparator choice for the per-column, per-row
// lexicographic order/equality (`row_cmp_*`/`row_eq_*` below) is NOT the
// same total order this file's flat `unique_raw` uses: no signed-zero
// (`-0.0`-before-`+0.0`) tie-break (verified: `np.unique([[0.0],[-0.0]],
// axis=0)` dedups to a SINGLE row whose sign is the FIRST-occurring row's
// own sign, i.e. plain stable-sort order, no forced tie-break -- this is
// consistent with real numpy's structured-array sort using each field's
// plain compare function, not the specialized real-float in-place-sort
// kernel that manufactures the `-0.0`-before-`+0.0` rule in the flat
// path), and no NaN-payload canonicalization (verified: a NaN's exact bit
// pattern survives `unique(axis=0)` untouched, unlike bare `unique()`).
// NaN still sorts LAST within a column (verified), and `eq` per element is
// PLAIN equality (`NaN != NaN`, ordinary IEEE, no `equal_nan` collapsing --
// see above).

/// Generalization of `stable_group_by` that operates purely on an index
/// range `0..n` via caller-supplied `(cmp, eq)` closures over index PAIRS,
/// rather than over `data: &[T]` directly -- needed because `unique_axis`'s
/// "elements" are whole rows (a `row_len`-wide slice of the flattened
/// per-column buffer), not single `Copy` values. Same algorithm as
/// `stable_group_by` otherwise (stable sort of indices by `cmp`, then
/// adjacent-equal grouping by `eq`).
fn stable_group_by_idx(
    n: usize,
    cmp: impl Fn(usize, usize) -> Ordering,
    eq: impl Fn(usize, usize) -> bool,
) -> GroupResult {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| cmp(i, j));

    let mut first_index: Vec<usize> = Vec::new();
    let mut counts: Vec<i64> = Vec::new();
    let mut group_of_pos: Vec<i64> = vec![0; n];
    for (pos, &oi) in order.iter().enumerate() {
        if pos == 0 || !eq(order[pos - 1], oi) {
            first_index.push(oi);
            counts.push(0);
        }
        *counts.last_mut().unwrap() += 1;
        group_of_pos[pos] = (first_index.len() - 1) as i64;
    }
    let mut inverse: Vec<i64> = vec![0; n];
    for (pos, &oi) in order.iter().enumerate() {
        inverse[oi] = group_of_pos[pos];
    }
    GroupResult { first_index, inverse, counts }
}

fn row_cmp_ord<T: Ord + Copy>(data: &[T], row_len: usize, i: usize, j: usize) -> Ordering {
    data[i * row_len..(i + 1) * row_len].cmp(&data[j * row_len..(j + 1) * row_len])
}
fn row_eq_ord<T: PartialEq + Copy>(data: &[T], row_len: usize, i: usize, j: usize) -> bool {
    data[i * row_len..(i + 1) * row_len] == data[j * row_len..(j + 1) * row_len]
}

fn row_cmp_float<T: PartialOrd + Copy>(
    data: &[T],
    row_len: usize,
    i: usize,
    j: usize,
    is_nan: impl Fn(T) -> bool + Copy,
) -> Ordering {
    let (bi, bj) = (i * row_len, j * row_len);
    for k in 0..row_len {
        match float_cmp(data[bi + k], data[bj + k], is_nan, |_: T| false) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    Ordering::Equal
}
fn row_eq_float<T: PartialEq + Copy>(
    data: &[T],
    row_len: usize,
    i: usize,
    j: usize,
    is_nan: impl Fn(T) -> bool + Copy,
) -> bool {
    let (bi, bj) = (i * row_len, j * row_len);
    (0..row_len).all(|k| float_eq(data[bi + k], data[bj + k], is_nan, false))
}

fn row_cmp_complex<T: PartialOrd + Copy>(
    data: &[Complex<T>],
    row_len: usize,
    i: usize,
    j: usize,
    is_nan: impl Fn(T) -> bool + Copy,
) -> Ordering {
    let (bi, bj) = (i * row_len, j * row_len);
    for k in 0..row_len {
        match complex_cmp(data[bi + k], data[bj + k], is_nan) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    Ordering::Equal
}
fn row_eq_complex<T: PartialEq + Copy>(
    data: &[Complex<T>],
    row_len: usize,
    i: usize,
    j: usize,
    is_nan: impl Fn(T) -> bool + Copy,
) -> bool {
    let (bi, bj) = (i * row_len, j * row_len);
    (0..row_len).all(|k| complex_eq(data[bi + k], data[bj + k], is_nan, false))
}

/// Dispatch a `(n, row_len)`-shaped C-contiguous buffer (the "axis moved to
/// front, flattened to 2-D" array real numpy builds before viewing it as
/// structured/void) to a `GroupResult` plus a `Buffer` of gathered
/// (payload-preserving, first-occurrence, NO NaN-canonicalization, NO
/// signed-zero tie-break) representative rows, `row_len` values apiece, in
/// ascending lexicographic-row-order.
fn unique_axis_raw(moved: &NdArray, row_len: usize) -> Result<(GroupResult, Buffer), IonpError> {
    let n = moved.shape()[0];
    macro_rules! ord_row {
        ($variant:ident, $data:expr) => {{
            let data: &[_] = $data;
            let g = stable_group_by_idx(
                n,
                |i, j| row_cmp_ord(data, row_len, i, j),
                |i, j| row_eq_ord(data, row_len, i, j),
            );
            let mut values = Vec::with_capacity(g.first_index.len() * row_len);
            for &i in &g.first_index {
                values.extend_from_slice(&data[i * row_len..(i + 1) * row_len]);
            }
            (g, Buffer::$variant(values))
        }};
    }
    macro_rules! float_row {
        ($variant:ident, $data:expr, $is_nan:expr) => {{
            let data: &[_] = $data;
            let is_nan = $is_nan;
            let g = stable_group_by_idx(
                n,
                |i, j| row_cmp_float(data, row_len, i, j, is_nan),
                |i, j| row_eq_float(data, row_len, i, j, is_nan),
            );
            let mut values = Vec::with_capacity(g.first_index.len() * row_len);
            for &i in &g.first_index {
                values.extend_from_slice(&data[i * row_len..(i + 1) * row_len]);
            }
            (g, Buffer::$variant(values))
        }};
    }
    macro_rules! complex_row {
        ($variant:ident, $data:expr, $is_nan:expr) => {{
            let data: &[_] = $data;
            let is_nan = $is_nan;
            let g = stable_group_by_idx(
                n,
                |i, j| row_cmp_complex(data, row_len, i, j, is_nan),
                |i, j| row_eq_complex(data, row_len, i, j, is_nan),
            );
            let mut values = Vec::with_capacity(g.first_index.len() * row_len);
            for &i in &g.first_index {
                values.extend_from_slice(&data[i * row_len..(i + 1) * row_len]);
            }
            (g, Buffer::$variant(values))
        }};
    }
    Ok(match moved.buffer() {
        Buffer::Bool(v) => ord_row!(Bool, v.as_slice()),
        Buffer::I8(v) => ord_row!(I8, v.as_slice()),
        Buffer::I16(v) => ord_row!(I16, v.as_slice()),
        Buffer::I32(v) => ord_row!(I32, v.as_slice()),
        Buffer::I64(v) => ord_row!(I64, v.as_slice()),
        Buffer::U8(v) => ord_row!(U8, v.as_slice()),
        Buffer::U16(v) => ord_row!(U16, v.as_slice()),
        Buffer::U32(v) => ord_row!(U32, v.as_slice()),
        Buffer::U64(v) => ord_row!(U64, v.as_slice()),
        Buffer::F16(v) => float_row!(F16, v.as_slice(), f16_is_nan),
        Buffer::F32(v) => float_row!(F32, v.as_slice(), f32::is_nan),
        Buffer::F64(v) => float_row!(F64, v.as_slice(), f64::is_nan),
        Buffer::C64(v) => complex_row!(C64, v.as_slice(), f32::is_nan),
        Buffer::C128(v) => complex_row!(C128, v.as_slice(), f64::is_nan),
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("setops.rs: no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
    })
}

/// `np.unique(ar, ..., axis=<int>)` for `ar.ndim >= 2` (the caller,
/// `unique()` below, routes `ar.ndim <= 1` straight to `unique_general`
/// instead -- see the module note above for why that's not just an
/// optimization but actual behavioral fidelity). Moves `axis` to the front
/// (`NdArray::transpose_axes`, a pure relabeling -- exactly numpy's own
/// `moveaxis(ar, axis, 0)`), flattens the trailing axes into one `row_len`-
/// wide row per position along that axis, row-wise stable-groups (plain
/// equality/NaN-last order, no `equal_nan`, no canonicalization -- see
/// above), then moves the surviving-row axis back to its original position
/// via the inverse permutation.
pub fn unique_axis(
    a: &NdArray,
    axis: isize,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
) -> Result<(NdArray, Option<NdArray>, Option<NdArray>, Option<NdArray>), IonpError> {
    let ndim = a.ndim();
    let ax = crate::sort::normalize_single_axis(axis, ndim)?;

    let mut fwd: Vec<isize> = Vec::with_capacity(ndim);
    fwd.push(ax as isize);
    for i in 0..ndim {
        if i != ax {
            fwd.push(i as isize);
        }
    }
    let moved = a.transpose_axes(&fwd)?.to_contiguous();
    let row_len: usize = moved.shape()[1..].iter().product();

    let (g, values_buf) = unique_axis_raw(&moved, row_len)?;
    let n_groups = g.first_index.len();
    let mut out_shape = moved.shape().to_vec();
    out_shape[0] = n_groups;
    let values_moved = NdArray::from_buffer(values_buf, out_shape, ArrOrder::C)?;

    // Inverse of `fwd`: `fwd` reads "new axis k comes from old axis
    // fwd[k]"; `q` must read "new axis k comes from MOVED axis q[k]" such
    // that applying it to `values_moved` (shape = moved-shape with axis 0
    // shrunk) restores the ORIGINAL axis ORDER (axis 0's shrunk dim lands
    // back at position `ax`).
    let mut q: Vec<isize> = vec![0isize; ndim];
    for (k, slot) in q.iter_mut().enumerate() {
        *slot = match k.cmp(&ax) {
            Ordering::Less => (1 + k) as isize,
            Ordering::Equal => 0,
            Ordering::Greater => k as isize,
        };
    }
    let values_arr = values_moved.transpose_axes(&q)?.to_contiguous();

    let indices_arr = if return_index {
        Some(i64_array(g.first_index.iter().map(|&i| i as i64).collect())?)
    } else {
        None
    };
    let inverse_arr = if return_inverse { Some(i64_array(g.inverse)?) } else { None };
    let counts_arr = if return_counts { Some(i64_array(g.counts)?) } else { None };
    Ok((values_arr, indices_arr, inverse_arr, counts_arr))
}

/// `np.unique(ar, return_index=.., return_inverse=.., return_counts=..,
/// axis=.., *, equal_nan=.., sorted=..)` -- the top-level dispatcher.
/// `sorted` has no parameter here: real numpy's `sorted=False` only ever
/// changes behavior in the flattened, no-return-flags, hash-fast-path
/// branch of `_unique1d` (`ar.ndim <= 1` or `axis is None`), and even then
/// only for dtypes whose hash-table iteration order is unspecified
/// (verified empirically: bool/float16/float32/float64 always come back
/// sorted anyway, hundreds of randomized trials, 0 mismatches; int/uint/
/// complex dtypes' hash order is genuinely unspecified and NOT reproduced
/// by this crate's single sort-based algorithm -- this is the SAME
/// documented exemption `unique_values` itself is built on, see this
/// module's own historical doc comment above `unique_values` and the
/// `toplevel.py` ledger comment for the full rationale: "not a bug to
/// chase: there is no fixed target for a bit-exact harness to hit"). The
/// PyO3 binding accepts `sorted` and passes it straight through as an
/// accepted-and-silently-ignored flag (this crate's sort-based algorithm
/// always produces sorted output, matching `sorted=True`'s contract and,
/// for the provably-safe dtypes, `sorted=False`'s as well); the differential
/// corpus deliberately does not probe `sorted=False` on int/uint/complex
/// dtypes in the bare (no return_*, axis=None) call form, for the same
/// reason `unique_values` itself is not declared.
pub fn unique(
    a: &NdArray,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
    axis: Option<isize>,
    equal_nan: bool,
) -> Result<(NdArray, Option<NdArray>, Option<NdArray>, Option<NdArray>), IonpError> {
    match axis {
        None => unique_general(a, return_index, return_inverse, return_counts, equal_nan),
        Some(ax) => {
            if a.ndim() <= 1 {
                // Real numpy still validates an explicit axis against
                // `ar.ndim` even though it then ignores it for computation
                // (verified: `np.unique(np.array(5), axis=0)` raises
                // `AxisError('axis 0 is out of bounds for array of
                // dimension 0')` -- a 0-d array has no valid axis at all).
                let _ = crate::sort::normalize_single_axis(ax, a.ndim())?;
                unique_general(a, return_index, return_inverse, return_counts, equal_nan)
            } else {
                unique_axis(a, ax, return_index, return_inverse, return_counts)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// diff / ediff1d / trim_zeros
// ---------------------------------------------------------------------------

fn subtract_flat<T: Copy>(hi: &[T], lo: &[T], sub: impl Fn(T, T) -> T) -> Vec<T> {
    hi.iter().zip(lo.iter()).map(|(&h, &l)| sub(h, l)).collect()
}

/// One first-difference pass along a 1-D C-contiguous slice, for every
/// numeric dtype. Bool is promoted to I64 first by the caller (real numpy:
/// `np.diff` on a bool array raises the same `TypeError: numpy boolean
/// subtract` any bare bool `-` does -- NOT silently promoted; verified
/// against real numpy 2.5.1. This function therefore never sees `Buffer::Bool`.
fn diff1_buffer(buf: &Buffer) -> Result<Buffer, IonpError> {
    match buf {
        Buffer::Bool(_) => Err(IonpError::Type(
            "numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.".to_string(),
        )),
        Buffer::I8(v) => Ok(Buffer::I8(subtract_flat(&v[1..], &v[..v.len() - 1], i8::wrapping_sub))),
        Buffer::I16(v) => Ok(Buffer::I16(subtract_flat(&v[1..], &v[..v.len() - 1], i16::wrapping_sub))),
        Buffer::I32(v) => Ok(Buffer::I32(subtract_flat(&v[1..], &v[..v.len() - 1], i32::wrapping_sub))),
        Buffer::I64(v) => Ok(Buffer::I64(subtract_flat(&v[1..], &v[..v.len() - 1], i64::wrapping_sub))),
        Buffer::U8(v) => Ok(Buffer::U8(subtract_flat(&v[1..], &v[..v.len() - 1], u8::wrapping_sub))),
        Buffer::U16(v) => Ok(Buffer::U16(subtract_flat(&v[1..], &v[..v.len() - 1], u16::wrapping_sub))),
        Buffer::U32(v) => Ok(Buffer::U32(subtract_flat(&v[1..], &v[..v.len() - 1], u32::wrapping_sub))),
        Buffer::U64(v) => Ok(Buffer::U64(subtract_flat(&v[1..], &v[..v.len() - 1], u64::wrapping_sub))),
        Buffer::F16(v) => Ok(Buffer::F16(subtract_flat(&v[1..], &v[..v.len() - 1], |a, b| a - b))),
        Buffer::F32(v) => Ok(Buffer::F32(subtract_flat(&v[1..], &v[..v.len() - 1], |a, b| a - b))),
        Buffer::F64(v) => Ok(Buffer::F64(subtract_flat(&v[1..], &v[..v.len() - 1], |a, b| a - b))),
        Buffer::C64(v) => Ok(Buffer::C64(subtract_flat(&v[1..], &v[..v.len() - 1], |a, b| a - b))),
        Buffer::C128(v) => Ok(Buffer::C128(subtract_flat(&v[1..], &v[..v.len() - 1], |a, b| a - b))),
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("setops.rs: no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
    }
}

/// Concatenate same-dtype, same-ndim arrays along `axis` (every other axis
/// must already match in extent). Used by `diff`'s `prepend`/`append` and
/// `ediff1d`'s `to_begin`/`to_end` (always axis=0 there, post-flatten), and
/// by `intersect1d`/`setdiff1d`'s `intersect1d_return_indices`/`union1d`/
/// `setxor1d` (also always axis=0, post-`common_dtype_cast`+flatten).
///
/// Every caller is responsible for pre-casting its operands to a common
/// dtype (`common_dtype_cast` for the plain two-array set ops; `diff`'s own
/// `promote_dtype` chain over `prepend`/`a`/`append`, mirroring real numpy's
/// `np.concatenate`'s own automatic dtype promotion -- ionp does not
/// reproduce that promotion INSIDE this shared helper because the "what
/// counts as the common dtype" rule genuinely differs by caller: diff's
/// combined `[prepend, a, append]` set uses ordinary array-array promotion,
/// while `ediff1d`'s `to_begin`/`to_end` use a completely different
/// same-kind-castable-to-`ary`.dtype rule, so there is no single correct
/// place to bury it here). The dtype-homogeneity check below is a
/// defensive backstop, not a promotion step: it exists so a caller that
/// forgets this contract gets a real `IonpError`, not the raw
/// `unreachable!()` panic the per-dtype `do_concat!` macro used to hit --
/// found live via `ediff1d(int32_ary, to_begin=int8_array)` (numpy: `[-1
/// -2 4 -3 6]` int32, same-kind-castable so no exception; ionp before this
/// fix: `panicked at ionp-core/src/setops.rs:962: internal error: entered
/// unreachable code`) while sweeping this function for OTHER reachable
/// panics beyond the `diff` rank-0 one this task started from. See
/// `ediff1d`'s own same-kind cast step, added alongside this backstop.
///
/// Axis validity is checked against `arrays[0]`'s OWN ndim FIRST, before
/// anything else -- exactly matching real numpy's `PyArray_ConcatenateArrays`
/// (verified: `np.concatenate([np.zeros((3,)), np.zeros((2,2))], axis=1)`
/// raises `AxisError: axis 1 is out of bounds for array of dimension 1`
/// -- referencing arrays[0]'s ndim (1), NOT the ndim-mismatch ValueError,
/// even though the arrays' ndims also disagree; but
/// `np.concatenate([np.zeros((2,2)), np.zeros((3,))], axis=1)` -- axis=1
/// valid for arrays[0] (ndim 2) -- raises the ndim-mismatch ValueError
/// instead, never consulting whether axis=1 is valid for the SECOND
/// array. Both forms verified against live numpy 2.5.1.).
fn concat_along_axis(arrays: &[NdArray], axis: usize) -> Result<NdArray, IonpError> {
    if arrays.is_empty() {
        return Err(IonpError::Value("concat_along_axis: no arrays given".to_string()));
    }
    let ndim = arrays[0].shape().len();
    if axis >= ndim {
        return Err(IonpError::AxisError { axis: axis as isize, ndim: Some(ndim) });
    }
    let mut out_shape: Vec<usize> = arrays[0].shape().to_vec();
    let mut axis_total = 0usize;
    for (i, a) in arrays.iter().enumerate() {
        if a.shape().len() != ndim {
            return Err(IonpError::Value(format!(
                "all the input arrays must have same number of dimensions, but the array at index 0 has {ndim} dimension(s) and the array at index {i} has {} dimension(s)",
                a.shape().len()
            )));
        }
        for (ax, (&s_out, &s_a)) in out_shape.iter().zip(a.shape().iter()).enumerate() {
            if ax != axis && s_out != s_a {
                return Err(IonpError::Value(format!(
                    "all the input array dimensions except for the concatenation axis must match exactly, but along dimension {ax}, the array at index 0 has size {s_out} and the array at index {i} has size {s_a}"
                )));
            }
        }
        axis_total += a.shape()[axis];
    }
    let dt0 = arrays[0].dtype();
    if let Some(bad) = arrays.iter().position(|a| a.dtype() != dt0) {
        return Err(IonpError::Value(format!(
            "concat_along_axis: caller contract violation -- array at index 0 has dtype {:?}, array at index {bad} has dtype {:?} (every caller must pre-cast to a common dtype)",
            dt0,
            arrays[bad].dtype()
        )));
    }
    out_shape[axis] = axis_total;
    let outer: usize = out_shape[..axis].iter().product();
    let inner: usize = out_shape[axis + 1..].iter().product();

    macro_rules! do_concat {
        ($variant:ident) => {{
            let contigs: Vec<NdArray> = arrays.iter().map(|a| a.to_contiguous()).collect();
            let slices: Vec<&[_]> = contigs.iter().map(|c| match c.buffer() {
                Buffer::$variant(v) => v.as_slice(),
                _ => unreachable!(),
            }).collect();
            let axis_lens: Vec<usize> = arrays.iter().map(|a| a.shape()[axis]).collect();
            let mut out = Vec::with_capacity(outer * axis_total * inner);
            for o in 0..outer {
                for (k, s) in slices.iter().enumerate() {
                    let chunk = axis_lens[k] * inner;
                    let start = o * chunk;
                    out.extend_from_slice(&s[start..start + chunk]);
                }
            }
            Buffer::$variant(out)
        }};
    }
    let out_buf = match arrays[0].dtype() {
        DType::Bool => do_concat!(Bool),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => do_concat!(I8),
        DType::I16 => do_concat!(I16),
        DType::I32 => do_concat!(I32),
        DType::I64 => do_concat!(I64),
        DType::U8 => do_concat!(U8),
        DType::U16 => do_concat!(U16),
        DType::U32 => do_concat!(U32),
        DType::U64 => do_concat!(U64),
        DType::F16 => do_concat!(F16),
        DType::F32 => do_concat!(F32),
        DType::F64 => do_concat!(F64),
        DType::C64 => do_concat!(C64),
        DType::C128 => do_concat!(C128),
    };
    NdArray::from_buffer(out_buf, out_shape, ArrOrder::C)
}

/// Real numpy's own rule (`_function_base_impl.py`'s `diff`, read offline,
/// not probed live): "Scalar values are expanded to arrays with length 1
/// in the direction of axis and the shape of the input array in along all
/// other axes. Otherwise the dimension and shape must match `a` except
/// along axis." -- the broadcast only fires for a genuine 0-d operand
/// (`prepend.ndim == 0` in numpy's source, checked AFTER `np.asanyarray`,
/// so this covers both a bare Python/numpy scalar and an already-0-d
/// array); anything of ndim >= 1, even length-mismatched, is passed
/// through unchanged into `concatenate` and left to THAT call's own
/// ndim/shape validation (verified: `np.diff([1,2,3], append=[9,10])`
/// works, `np.diff([[1,2],[3,4]], prepend=np.zeros(3))` raises the
/// concatenate-level `AxisError`/ValueError, not a diff-specific one).
/// `creation::broadcast_to` produces a stride-0 VIEW (not a materialized
/// copy) -- fine here since `concat_along_axis` already materializes every
/// operand via `.to_contiguous()` internally.
fn diff_broadcast_operand(op: &NdArray, a: &NdArray, ax: usize) -> Result<NdArray, IonpError> {
    if op.ndim() == 0 {
        let mut shape = a.shape().to_vec();
        shape[ax] = 1;
        crate::creation::broadcast_to(op, &shape)
    } else {
        Ok(op.clone())
    }
}

/// The three-way state of `diff`'s `prepend=`/`append=` keyword, needed
/// because real numpy's actual default is a private `np._NoValue`
/// sentinel, NOT `None` -- so "the keyword was never passed" and "the
/// keyword was passed with the literal value `None`" are genuinely
/// DIFFERENT, observable states (`np.diff(a, prepend=None)` raises;
/// omitting `prepend` entirely does not). A plain `Option<&NdArray>`
/// cannot represent this (both "omitted" and "an explicit None that maps
/// to no array" would have to collapse to Rust `None`), so callers
/// (`ionp-py/src/setops.rs`) resolve raw Python-level presence/absence
/// into this enum before calling in, exactly mirroring how `Ufunc::
/// __call__`'s `out=` keyword (`ionp-py/src/lib.rs`) has to distinguish
/// "key absent" from "key present holding None" via raw `**kwargs`
/// inspection instead of a typed `Option` parameter.
#[derive(Debug, Clone, Copy)]
pub enum DiffOperand<'a> {
    /// The keyword was never passed at all -- real numpy: `prepend is
    /// np._NoValue`, entirely skipped.
    Omitted,
    /// The keyword was passed with the literal Python value `None` --
    /// real numpy: `np.asanyarray(None)` builds a 0-d `object` array
    /// containing `None`, broadcasts it, concatenates it into `a`, and
    /// the later `subtract` between a real element and that `None` object
    /// raises `TypeError`. ionp has no generic object dtype to replicate
    /// that machinery, but the crash is fully deterministic given only
    /// `a`'s own dtype and which side (`prepend` vs `append`) is `None`
    /// (verified live against numpy 2.5.1 across every ionp dtype), so
    /// `diff` below raises the identical message directly instead.
    ExplicitNone,
    /// A real array-like value was given.
    Value(&'a NdArray),
}

impl<'a> DiffOperand<'a> {
    fn as_value(self) -> Option<&'a NdArray> {
        match self {
            DiffOperand::Value(v) => Some(v),
            DiffOperand::Omitted | DiffOperand::ExplicitNone => None,
        }
    }
}

/// The Python scalar type name real numpy's `TypeError: unsupported
/// operand type(s) for -: '<this>' and 'NoneType'` names for a given
/// `a.dtype()`, when `diff`'s `prepend=None`/`append=None` forces the
/// mixed-with-`None` `object`-dtype concatenation down numpy's ordinary
/// Python-level `-` operator. This is just `type(np.<dtype>(0).item())
/// .__name__` for each dtype kind (verified live for every ionp dtype:
/// every int/uint kind reports as plain `'int'`, every float kind as
/// `'float'`, `Bool` as `'bool'`, both complex kinds as `'complex'` --
/// numpy's own `.item()` unboxes to the corresponding native Python type
/// regardless of bit width).
fn diff_none_operand_type_name(dt: DType) -> &'static str {
    if dt.is_bool() {
        "bool"
    } else if dt.is_complex() {
        "complex"
    } else if dt.is_floating() {
        "float"
    } else {
        // is_integer() (signed or unsigned) -- the only remaining case.
        "int"
    }
}

/// `np.diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue)`.
pub fn diff(
    a: &NdArray,
    n: usize,
    axis: isize,
    prepend: DiffOperand<'_>,
    append: DiffOperand<'_>,
) -> Result<NdArray, IonpError> {
    // Real numpy: `if n == 0: return a` is the FIRST thing `diff` does --
    // before `asanyarray`, before the ndim==0 check, before prepend/append
    // are even looked at. Verified live: `np.diff(np.array(5), n=0)`
    // returns `array(5)` (no "at least one dimensional" ValueError despite
    // being 0-d), and `np.diff([1,2,3], n=0, prepend=np.zeros((2,2)))`
    // returns `[1,2,3]` unchanged even though that `prepend` shape could
    // never legally concatenate -- `is a` even holds in the pure-Python
    // reference (no copy). `.clone()` here is the ionp equivalent (cheap:
    // `NdArray`'s buffer is `Arc`-shared, not deep-copied).
    if n == 0 {
        return Ok(a.clone());
    }

    // Real numpy: `diff requires input that is at least one dimensional`
    // (verified: `np.diff(np.array(5))` raises exactly this ValueError --
    // but only once the n==0 fast path above has already been ruled out).
    if a.ndim() == 0 {
        return Err(IonpError::Value(
            "diff requires input that is at least one dimensional".to_string(),
        ));
    }
    let ax = crate::sort::normalize_single_axis(axis, a.ndim())?;

    // Explicit `prepend=None`/`append=None` (as opposed to omitting the
    // keyword entirely -- see `DiffOperand`'s own doc) crashes real numpy
    // downstream in `object`-dtype arithmetic; ionp raises the identical
    // `TypeError` directly here instead of modeling object dtype.
    //
    // KNOWN, ACCEPTED DIVERGENCE (2026-08-02, reported to and held by the
    // coordinator): real numpy's `TypeError` here is not a validation at
    // all -- it is emergent from actually performing an elementwise
    // subtraction on the `object`-dtype array that `None` got concatenated
    // into. Real numpy builds `combined = concatenate([prepend?, a,
    // append?], axis=ax)` and then computes `combined[i+1] - combined[i]`
    // for every `i` along `ax`, once per "row" of every OTHER axis. If
    // that computation never actually runs -- either because there are
    // zero rows (some axis other than `ax` has length 0) or because
    // `combined` has fewer than 2 elements along `ax` itself -- no
    // subtraction ever executes, and numpy quietly returns an empty
    // `dtype=object` array instead of raising (verified live, both
    // regimes: `np.diff(np.zeros((0,), dtype='int64'), prepend=None)` and
    // `np.diff(np.zeros((5,0)), axis=0, prepend=None, append=None)` both
    // -> `array([], dtype=object)`, no exception, even though the second
    // one's diffed axis (`axis=0`, length 5) is non-empty -- it's `axis=1`
    // (length 0, the *other* axis) that makes the whole array vacuous).
    // ionp has NO `object` dtype at all (see `ionp-core/src/dtype.rs`'s
    // `min_scalar_type_unsigned`/`min_scalar_type_signed` doc comments --
    // "architectural absence, not a bug") and cannot represent that empty
    // `object`-dtype return value under any existing dtype. The
    // unconditional raise below is therefore intentionally wrong whenever
    // the computation is vacuous (no pair ever gets subtracted) AND a
    // `None` is involved; this is a known, structural gap, not an
    // oversight, and must NOT be papered over by faking an object dtype
    // or by silently returning some other dtype's empty array.
    //
    // Whenever the computation is NOT vacuous -- at least one row exists
    // (every axis other than `ax` has length > 0) AND `combined` has at
    // least 2 elements along `ax` -- a real subtraction DOES execute, and
    // ionp must raise the exact same `TypeError` numpy does, which is
    // fully representable (a real exception needs no `object` dtype).
    // `None` can only ever sit at `combined`'s very first slot (an
    // explicit-`None` `prepend`) or very last slot (an explicit-`None`
    // `append`) -- every interior slot is one of `a`'s own real elements.
    // So the only two pairs that can ever fail are the first pair
    // (`combined[1] - combined[0]`, fails if `prepend` is `None`) and the
    // last pair (`combined[-1] - combined[-2]`, fails if `append` is
    // `None`); numpy evaluates left-to-right, so a `None` prepend always
    // wins and is reported before a `None` append is ever reached
    // (verified live: `np.diff(a, prepend=None, append=None)` on a
    // *non-empty* `a` reports the `prepend`-shaped message, matching this
    // ordering).
    //
    // `combined[1]` (paired against a `None` prepend) is `a`'s own first
    // element whenever `a` has at least one element along `ax`
    // (`a.shape()[ax] >= 1`) -- the original, simple case. But when
    // `a.shape()[ax] == 0`, `a` contributes NOTHING to `combined`, so
    // `combined[1]` is instead whatever `append` contributed (which must
    // be present for `combined.len() >= 2` to hold at all, since `prepend`
    // alone only supplies 1 slot) -- symmetric reasoning applies to
    // `combined[-2]` when `append` is the `None` side. Verified live for
    // both prepend-None and both-None on `(0,)`, `(4,0)`, and `(0,4)`
    // (crossed with every valid axis): axis choice changes WHICH regime
    // applies (vacuous vs. real-subtract) and, within the real-subtract
    // regime, which operand's dtype fills `a`'s usual slot -- never the
    // message's shape or wording otherwise.
    let a_shape = a.shape();
    let a_dim = a_shape[ax];
    // Empty product (ndim == 1, nothing left after excluding `ax`) is 1,
    // matching numpy's own "no other axes to batch over" case.
    let other_size: usize = a_shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != ax)
        .map(|(_, &d)| d)
        .product();
    let pre_present = !matches!(prepend, DiffOperand::Omitted);
    let app_present = !matches!(append, DiffOperand::Omitted);
    let combined_len = a_dim + usize::from(pre_present) + usize::from(app_present);
    let any_real_subtract = other_size > 0 && combined_len >= 2;

    if any_real_subtract {
        if matches!(prepend, DiffOperand::ExplicitNone) {
            // combined[0] = None (prepend); combined[1] is a's first
            // element if a contributes any, else append (which must be
            // present, since combined_len >= 2 and a_dim == 0 here).
            let combined1_name = if a_dim >= 1 {
                diff_none_operand_type_name(a.dtype())
            } else {
                match append {
                    DiffOperand::ExplicitNone => "NoneType",
                    DiffOperand::Value(v) => diff_none_operand_type_name(v.dtype()),
                    DiffOperand::Omitted => {
                        unreachable!("combined_len >= 2 with a_dim == 0 and prepend present forces append present")
                    }
                }
            };
            return Err(IonpError::Type(format!(
                "unsupported operand type(s) for -: '{combined1_name}' and 'NoneType'"
            )));
        }
        if matches!(append, DiffOperand::ExplicitNone) {
            // combined[-1] = None (append); combined[-2] is a's last
            // element if a contributes any, else prepend (which must be
            // present, since combined_len >= 2 and a_dim == 0 here, and
            // prepend is not ExplicitNone -- that case was already
            // handled above).
            let combined_penultimate_name = if a_dim >= 1 {
                diff_none_operand_type_name(a.dtype())
            } else {
                match prepend {
                    DiffOperand::Value(v) => diff_none_operand_type_name(v.dtype()),
                    DiffOperand::Omitted | DiffOperand::ExplicitNone => {
                        unreachable!("combined_len >= 2 with a_dim == 0 and append present, and prepend != ExplicitNone here, forces prepend present as a Value")
                    }
                }
            };
            return Err(IonpError::Type(format!(
                "unsupported operand type(s) for -: 'NoneType' and '{combined_penultimate_name}'"
            )));
        }
        // Neither is None -- an ordinary real subtract, no object dtype
        // involved; fall through to the normal concatenate-and-diff path.
    } else if matches!(prepend, DiffOperand::ExplicitNone) || matches!(append, DiffOperand::ExplicitNone) {
        // Vacuous computation (no pair ever gets subtracted) but a None
        // is involved -- the genuinely-unrepresentable-object-dtype case
        // documented above. ionp cannot return numpy's empty object
        // array, so it raises instead; the exact operand names it picks
        // don't matter for correctness (no correct value exists), but we
        // keep using a's own dtype for the non-None side as the simplest
        // deterministic choice, same as ionp has always done here.
        let type_name = diff_none_operand_type_name(a.dtype());
        if matches!(prepend, DiffOperand::ExplicitNone) {
            return Err(IonpError::Type(format!(
                "unsupported operand type(s) for -: '{type_name}' and 'NoneType'"
            )));
        }
        return Err(IonpError::Type(format!(
            "unsupported operand type(s) for -: 'NoneType' and '{type_name}'"
        )));
    }

    let prepend = prepend.as_value();
    let append = append.as_value();

    // Real numpy builds `combined = [prepend?, a, append?]` and, if it has
    // more than one element, calls `np.concatenate(combined, axis)` --
    // which promotes to combined's common dtype via ordinary array-array
    // promotion (NOT `ediff1d`'s same-kind-castable-to-input rule; the two
    // functions genuinely differ here, see `concat_along_axis`'s doc
    // comment). Verified live: `np.diff(np.array([1,2,3], dtype=np.int32),
    // prepend=1.5)` -> float64 `[-0.5, 1., 1.]`; `np.diff(uint8_arr,
    // append=300)` -> int64 (300 doesn't fit uint8, and `append`'s
    // `asanyarray` gives it a real concrete dtype -- default int -- BEFORE
    // concatenation, so this is plain array-dtype promotion, not any kind
    // of weak-scalar special case).
    let prepend_op = match prepend {
        Some(p) => Some(diff_broadcast_operand(p, a, ax)?),
        None => None,
    };
    let append_op = match append {
        Some(ap) => Some(diff_broadcast_operand(ap, a, ax)?),
        None => None,
    };

    let mut common_dt = a.dtype();
    if let Some(p) = &prepend_op {
        common_dt = promote_dtype(common_dt, p.dtype());
    }
    if let Some(ap) = &append_op {
        common_dt = promote_dtype(common_dt, ap.dtype());
    }

    let mut cur = if prepend_op.is_some() || append_op.is_some() {
        let mut parts: Vec<NdArray> = Vec::with_capacity(3);
        if let Some(p) = &prepend_op {
            parts.push(p.cast_to(common_dt));
        }
        parts.push(a.cast_to(common_dt));
        if let Some(ap) = &append_op {
            parts.push(ap.cast_to(common_dt));
        }
        concat_along_axis(&parts, ax)?
    } else {
        a.to_contiguous()
    };

    for _ in 0..n {
        if cur.shape()[ax] == 0 {
            // diff of a length-0 axis stays length-0 (n more passes are all no-ops).
            return Ok(cur);
        }
        let shape = cur.shape().to_vec();
        let outer: usize = shape[..ax].iter().product();
        let axis_len = shape[ax];
        let inner: usize = shape[ax + 1..].iter().product();
        let mut new_shape = shape.clone();
        new_shape[ax] = axis_len.saturating_sub(1);

        macro_rules! do_diff {
            ($variant:ident, $sub:expr) => {{
                let data: &[_] = match cur.buffer() { Buffer::$variant(v) => v.as_slice(), _ => unreachable!() };
                let mut out = Vec::with_capacity(outer * new_shape[ax] * inner);
                for o in 0..outer {
                    let base = o * axis_len * inner;
                    for k in 1..axis_len {
                        let hi = &data[base + k * inner..base + (k + 1) * inner];
                        let lo = &data[base + (k - 1) * inner..base + k * inner];
                        for (h, l) in hi.iter().zip(lo.iter()) {
                            out.push(($sub)(*h, *l));
                        }
                    }
                }
                Buffer::$variant(out)
            }};
        }
        let new_buf = match cur.buffer() {
            // Real numpy's `diff()` (`_function_base_impl.py`):
            // `op = not_equal if a.dtype == np.bool else subtract` -- bool
            // input does NOT raise (unlike `ediff1d`, which genuinely does
            // via `diff1_buffer` above); it uses elementwise `!=`, staying
            // bool-dtyped across every one of the `n` passes. Verified:
            // `np.diff(np.array([True, False, True]))` -> `[True, True]`,
            // bool dtype, no exception.
            Buffer::Bool(_) => do_diff!(Bool, |a: bool, b: bool| a != b),
            Buffer::I8(_) => do_diff!(I8, i8::wrapping_sub),
            Buffer::I16(_) => do_diff!(I16, i16::wrapping_sub),
            Buffer::I32(_) => do_diff!(I32, i32::wrapping_sub),
            Buffer::I64(_) => do_diff!(I64, i64::wrapping_sub),
            Buffer::U8(_) => do_diff!(U8, u8::wrapping_sub),
            Buffer::U16(_) => do_diff!(U16, u16::wrapping_sub),
            Buffer::U32(_) => do_diff!(U32, u32::wrapping_sub),
            Buffer::U64(_) => do_diff!(U64, u64::wrapping_sub),
            Buffer::F16(_) => do_diff!(F16, |a: f16, b: f16| a - b),
            Buffer::F32(_) => do_diff!(F32, |a: f32, b: f32| a - b),
            Buffer::F64(_) => do_diff!(F64, |a: f64, b: f64| a - b),
            Buffer::C64(_) => do_diff!(C64, |a: C64, b: C64| a - b),
            Buffer::C128(_) => do_diff!(C128, |a: C128, b: C128| a - b),
            Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("setops.rs: no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
        };
        cur = NdArray::from_buffer(new_buf, new_shape, ArrOrder::C)?;
    }
    Ok(cur)
}

/// `np.ediff1d(ary, to_end=None, to_begin=None)` -- flattens `ary`, takes
/// one first-difference pass, then concatenates `to_begin`/`to_end`.
///
/// `to_begin`/`to_end` follow a DIFFERENT rule from `diff`'s `prepend`/
/// `append`: real numpy (`_arraysetops_impl.py`'s `ediff1d`, read offline)
/// requires each to be `np.can_cast(operand, ary.dtype, casting="same_kind")`
/// -- raising `TypeError('dtype of \`to_begin\` must be compatible with
/// input \`ary\` under the \`same_kind\` rule.')` (verbatim, name
/// substituted for `to_end`) if not -- and then casts it DOWN/UP to
/// `ary`'s own dtype for the concatenation; there is no `diff`-style
/// common-dtype promotion, no widening the output past `ary.dtype`.
/// Verified live: `np.ediff1d(int32_ary, to_begin=int8_array)` ->
/// int32 result (same-kind, cast down... up to int32); `np.ediff1d(a,
/// to_begin=1.5)` on an int array raises the TypeError above (float->int
/// is not same-kind). Before this fix, `concat_along_axis` received
/// `to_begin`/`to_end` un-cast, at their OWN dtype -- harmless when it
/// happened to match `ary`'s dtype (every case the existing
/// `setops_cases.py` corpus covered), but a genuine mismatch (e.g. an
/// int8 `to_begin` against an int32 `ary`) hit `do_concat!`'s
/// `_ => unreachable!()` panic, found while sweeping this file's other
/// `concat_along_axis` callers for the same class of bug the `diff`
/// rank-0 panic came from.
pub fn ediff1d(a: &NdArray, to_end: Option<&NdArray>, to_begin: Option<&NdArray>) -> Result<NdArray, IonpError> {
    let flat = a.ravel_order("C")?;
    let n = flat.size();
    let result_dt = flat.dtype();

    // Order matters and is NOT "validate everything, then compute": real
    // numpy validates `to_begin`'s then `to_end`'s same-kind castability
    // (raising immediately on the first failure) BEFORE ever attempting
    // `ary[1:] - ary[:-1]` -- the subtract only happens as the LAST step,
    // written via `np.subtract(..., out=result[...])` after `result` has
    // already been allocated and `to_begin`/`to_end` written into it.
    // Verified live: `np.ediff1d(np.array([True, False, True]),
    // to_begin=1.5)` raises the `to_begin` same-kind `TypeError` (float
    // does not same-kind-cast to bool), NOT the "numpy boolean subtract"
    // `TypeError` -- but `to_begin=True` (bool, same-kind-valid) DOES then
    // hit the boolean-subtract rejection. So the bool-ary check
    // (`diff1_buffer`, below) must run AFTER these two casts are
    // validated, not before.
    let to_begin_cast = match to_begin {
        Some(b) => {
            let braw = b.ravel_order("C")?;
            if !can_cast(braw.dtype(), result_dt, "same_kind") {
                return Err(IonpError::Type(
                    "dtype of `to_begin` must be compatible with input `ary` under the `same_kind` rule.".to_string(),
                ));
            }
            Some(braw.cast_to(result_dt))
        }
        None => None,
    };
    let to_end_cast = match to_end {
        Some(e) => {
            let eraw = e.ravel_order("C")?;
            if !can_cast(eraw.dtype(), result_dt, "same_kind") {
                return Err(IonpError::Type(
                    "dtype of `to_end` must be compatible with input `ary` under the `same_kind` rule.".to_string(),
                ));
            }
            Some(eraw.cast_to(result_dt))
        }
        None => None,
    };

    // Real numpy's `ediff1d` attempts `ary[1:] - ary[:-1]` unconditionally,
    // so a bool-dtype `ary` raises `TypeError` even when EMPTY (size 0 or
    // 1) -- verified: `np.ediff1d(np.array([], dtype=bool))` raises.
    // `diff1_buffer` already encodes this rejection; it must run before the
    // `n == 0`/`n == 1` empty-result shortcut below, not be skipped by it.
    if matches!(flat.buffer(), Buffer::Bool(_)) {
        diff1_buffer(flat.buffer())?;
    }
    let diffed = if n == 0 {
        NdArray::from_buffer(empty_like_buffer(flat.buffer()), vec![0], ArrOrder::C)?
    } else {
        let buf = diff1_buffer(flat.buffer())?;
        let len = n - 1;
        NdArray::from_buffer(buf, vec![len], ArrOrder::C)?
    };
    let mut parts: Vec<NdArray> = Vec::new();
    if let Some(b) = to_begin_cast {
        parts.push(b);
    }
    parts.push(diffed);
    if let Some(e) = to_end_cast {
        parts.push(e);
    }
    if parts.len() == 1 {
        return Ok(parts.into_iter().next().unwrap());
    }
    concat_along_axis(&parts, 0)
}

fn empty_like_buffer(b: &Buffer) -> Buffer {
    match b {
        Buffer::Bool(_) => Buffer::Bool(vec![]),
        Buffer::I8(_) => Buffer::I8(vec![]),
        Buffer::I16(_) => Buffer::I16(vec![]),
        Buffer::I32(_) => Buffer::I32(vec![]),
        Buffer::I64(_) => Buffer::I64(vec![]),
        Buffer::U8(_) => Buffer::U8(vec![]),
        Buffer::U16(_) => Buffer::U16(vec![]),
        Buffer::U32(_) => Buffer::U32(vec![]),
        Buffer::U64(_) => Buffer::U64(vec![]),
        Buffer::F16(_) => Buffer::F16(vec![]),
        Buffer::F32(_) => Buffer::F32(vec![]),
        Buffer::F64(_) => Buffer::F64(vec![]),
        Buffer::C64(_) => Buffer::C64(vec![]),
        Buffer::C128(_) => Buffer::C128(vec![]),
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("setops.rs: no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
    }
}

/// `np.trim_zeros(filt, trim='fb', axis=None)`. `axis=None` (this
/// function's own `axis: None`) matches real numpy's own default: `axis_
/// tuple = tuple(range(filt.ndim))` -- i.e. "trim every axis", NOT "trim
/// nothing"; a genuinely empty `axis_tuple` (Python `axis=()`, or any
/// `ndim==0` array whose default range is empty) is the identity/no-op
/// case (verified: `np.trim_zeros(np.array(5))` returns the 0-d scalar
/// unchanged, and `np.trim_zeros(a, axis=())` returns `a` unchanged for
/// any `a`). `axis: Some(&[])` reproduces that identity case; `axis: None`
/// reproduces Python's own `axis=None` default (every axis). Zero test
/// matches numpy's own (`filt != 0`, per-dtype: for complex this is `!=
/// 0+0j`, i.e. either component nonzero counts as nonzero).
///
/// Verified against real numpy 2.5.1's actual source
/// (`_function_base_impl.py`'s `trim_zeros`/`_arg_trim_zeros`), not
/// guessed at:
///   - The N-D bounding box (`_arg_trim_zeros`: `argwhere(filt).min(axis=0)`
///     / `.max(axis=0)`) is computed over the WHOLE array, independent of
///     which axes `axis_tuple` actually selects -- an axis NOT in
///     `axis_tuple` always keeps its FULL original extent regardless of
///     its own zero/nonzero content (verified: `np.trim_zeros([[0,0,0],
///     [0,5,0],[0,0,0]], axis=0)` keeps all 3 columns, even though columns
///     0 and 2 are entirely zero).
///   - An all-zero array forces `start = stop = 0` for every axis in
///     `axis_tuple`, REGARDLESS of `trim`'s front/back flags (verified:
///     `np.trim_zeros(np.zeros(5), trim='f')` is `[]`, not `[]`-via-back-
///     kept-full -- the all-zero branch short-circuits before the front/
///     back conditional is even consulted).
///   - Otherwise, `'f' not in trim` forces `start` to `0` (no lower trim)
///     and `'b' not in trim` forces `stop` to the full axis length (no
///     upper trim) -- uniformly across every axis in `axis_tuple`, not
///     selected by axis.
pub fn trim_zeros(a: &NdArray, trim: &str, axis: Option<&[isize]>) -> Result<NdArray, IonpError> {
    let bounds = trim_zeros_bounds(a, trim, axis)?;
    let (starts, ends) = match bounds {
        None => return Ok(a.clone()),
        Some(b) => b,
    };
    let ndim = a.ndim();
    let flat = a.to_contiguous();
    let items: Vec<crate::array::SliceItem> = (0..ndim)
        .map(|ax| crate::array::SliceItem::Slice {
            start: Some(starts[ax] as isize),
            stop: Some(ends[ax] as isize),
            step: Some(1),
        })
        .collect();
    Ok(flat.get_view(&items)?.to_contiguous())
}

/// Computes the same `(starts, ends)` bounding box `trim_zeros` slices by,
/// WITHOUT performing the slice -- exposed so the PyO3 layer
/// (`ionp-py/src/setops.rs`) can reproduce numpy's list/tuple-in-list/tuple-
/// out type preservation for the 1-D case (`filt[sl[0]]`, real Python
/// `__getitem__` on the ORIGINAL object, not a value copy through ndarray)
/// without duplicating this bounding-box math. `None` means the identity
/// case (`axis_tuple` empty after normalization -- caller should return the
/// original object/array unchanged, exactly as `trim_zeros` above does via
/// `Ok(a.clone())`).
pub fn trim_zeros_bounds(
    a: &NdArray,
    trim: &str,
    axis: Option<&[isize]>,
) -> Result<Option<(Vec<usize>, Vec<usize>)>, IonpError> {
    let trim_lower = trim.to_ascii_lowercase();
    // Real numpy: `trim = trim.lower(); if trim not in {"fb","bf","f","b"}:
    // raise ValueError(f"unexpected character(s) in \`trim\`: {trim!r}")` --
    // an EXACT set-membership check (verified: even a redundant-but-valid-
    // chars string like `"fbf"` raises, not just a "contains other than f/b"
    // check), and the message embeds the LOWERCASED string, not the
    // original-case input.
    if !matches!(trim_lower.as_str(), "fb" | "bf" | "f" | "b") {
        return Err(IonpError::Value(format!(
            "unexpected character(s) in `trim`: '{trim_lower}'"
        )));
    }
    let ndim = a.ndim();

    // `axis_tuple`: real numpy's `normalize_axis_tuple(axis, ndim,
    // argname="axis")` (default `range(ndim)` when `axis is None`).
    // Validation happens even when the tuple ends up empty afterward
    // (matches numpy's own ordering: normalize first, THEN check for the
    // empty-tuple identity shortcut).
    let mut axis_tuple: Vec<usize> = Vec::new();
    match axis {
        None => axis_tuple.extend(0..ndim),
        Some(axes) => {
            let mut seen = std::collections::HashSet::new();
            for &raw in axes {
                let normalized = crate::sort::normalize_single_axis(raw, ndim)?;
                if !seen.insert(normalized) {
                    return Err(IonpError::Value(
                        "repeated axis in `axis` argument".to_string(),
                    ));
                }
                axis_tuple.push(normalized);
            }
        }
    }
    if axis_tuple.is_empty() {
        return Ok(None);
    }

    let front = trim_lower.contains('f');
    let back = trim_lower.contains('b');

    let flat = a.to_contiguous();
    let shape = flat.shape().to_vec();
    let is_nonzero: Vec<bool> = nonzero_mask(flat.buffer());

    let mut has_nonzero = false;
    let mut min_idx = vec![usize::MAX; ndim];
    let mut max_idx = vec![0usize; ndim];
    for (p, &nz) in is_nonzero.iter().enumerate() {
        if !nz {
            continue;
        }
        has_nonzero = true;
        let midx = flat_to_multi_index(p, &shape);
        for ax in 0..ndim {
            if midx[ax] < min_idx[ax] {
                min_idx[ax] = midx[ax];
            }
            if midx[ax] > max_idx[ax] {
                max_idx[ax] = midx[ax];
            }
        }
    }

    let mut starts = vec![0usize; ndim];
    let mut ends = shape.clone();
    if has_nonzero {
        for &ax in &axis_tuple {
            starts[ax] = if front { min_idx[ax] } else { 0 };
            ends[ax] = if back { max_idx[ax] + 1 } else { shape[ax] };
        }
    } else {
        for &ax in &axis_tuple {
            starts[ax] = 0;
            ends[ax] = 0;
        }
    }

    Ok(Some((starts, ends)))
}

/// Decode a flat, C (row-major)-order buffer position into its multi-index
/// against `shape`. Used only by `trim_zeros`'s N-D bounding-box computation
/// -- `sort.rs`'s own equivalent iterator (`nd_index_iter`) is private and
/// out of this file's fence, so this is a fresh, independently-written
/// implementation of the same standard row-major decode.
fn flat_to_multi_index(mut p: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut idx = vec![0usize; ndim];
    for ax in (0..ndim).rev() {
        let dim = shape[ax].max(1);
        idx[ax] = p % dim;
        p /= dim;
    }
    idx
}

fn nonzero_mask(b: &Buffer) -> Vec<bool> {
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
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("setops.rs: no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
    }
}

// ---------------------------------------------------------------------------
// intersect1d / union1d / setdiff1d / setxor1d / isin
//
// These do NOT reduce to a two-pointer merge over deduplicated sorted
// arrays -- verified by reading real numpy 2.5.1's actual source
// (`_arraysetops_impl.py`) in full, not guessed at:
//   - `intersect1d`/`setxor1d`: `aux = concatenate((ar1, ar2)); aux.sort()`
//     (a genuine in-place, NaN-canonicalizing, duplicate-preserving sort of
//     the RAW concatenation, only pre-deduplicated via `unique()` when
//     `assume_unique=False`), then a PLAIN (non-NaN-grouping) adjacent-
//     equality mask: `mask = aux[1:] == aux[:-1]`. Because `assume_unique=
//     True` skips that pre-dedup, a genuinely non-unique input under that
//     flag makes numpy's own result "wrong" relative to the true set (e.g.
//     `intersect1d([1,1,1],[1,1],assume_unique=True)` returns `[1,1]`, not
//     `[1]`) -- this file reproduces that same not-actually-a-set behavior
//     bit for bit rather than silently correcting it.
//   - `setdiff1d(a,b,assume_unique)`: `ar1 = a.ravel()` (assume_unique) or
//     `unique(a)` (not), then `ar1[~isin(ar1, ar2_or_unique(ar2),
//     assume_unique=True, invert=True)]` -- ALWAYS passes
//     `assume_unique=True` to the internal `isin` call regardless of the
//     outer flag. `isin`'s per-element boolean answer does not depend on
//     whether its `test_elements` argument was deduplicated first (a
//     duplicate in the membership target changes nothing about whether a
//     given value is "in" it), so this file's `isin_mask` is reused
//     directly without needing to replicate that internal
//     `assume_unique=True` plumbing.
//   - `isin`: real numpy branches between a hash/table method, a
//     linear-scan fallback, and a sort+argsort method depending on dtype/
//     size (`kind=`) -- but ALL THREE branches compute the exact same
//     per-element membership boolean (there is no algorithm-dependent tie
//     to reproduce here, unlike `unique_values`'s hash-path ORDER
//     ambiguity: `isin` returns one boolean per element of `element`, not a
//     deduplicated/reordered array), so a plain O(|element| * |test|) scan
//     using ordinary (non-NaN-grouping) equality reproduces every branch's
//     answer exactly. Verified: `np.isin([nan], [nan])` is `[False]` -- NaN
//     is NEVER considered "in" a test set, even against another NaN,
//     unlike the `unique*` family's `equal_nan=True` default.
//
// Every one of these five is documented by numpy as flattening its
// input(s); this file's implementation does the same (no `axis=` parameter
// exists on any of the five in real numpy either).
// ---------------------------------------------------------------------------

fn common_dtype_cast(a: &NdArray, b: &NdArray) -> Result<(NdArray, NdArray), IonpError> {
    let dt = promote_dtype(a.dtype(), b.dtype());
    Ok((a.cast_to(dt), b.cast_to(dt)))
}

fn maybe_unique(a: &NdArray, assume_unique: bool) -> Result<NdArray, IonpError> {
    if assume_unique {
        a.ravel_order("C")
    } else {
        unique_values(a)
    }
}

/// In-place-sort-and-canonicalize a `Buffer`, exactly reproducing real
/// numpy's `aux.sort()` step in `intersect1d`/`setxor1d` (same
/// canonicalization this file's `unique_values` already uses for its own
/// in-place-sort path -- verified to be the same code path: bare
/// `np.unique(x)` with no return flags ALSO takes `ar.sort()`, and its
/// NaN-canonicalization bit patterns match what this function produces).
fn sort_canon_buffer(buf: &Buffer) -> Buffer {
    match buf {
        Buffer::Bool(v) => { let mut d = v.clone(); d.sort(); Buffer::Bool(d) }
        Buffer::I8(v) => { let mut d = v.clone(); d.sort(); Buffer::I8(d) }
        Buffer::I16(v) => { let mut d = v.clone(); d.sort(); Buffer::I16(d) }
        Buffer::I32(v) => { let mut d = v.clone(); d.sort(); Buffer::I32(d) }
        Buffer::I64(v) => { let mut d = v.clone(); d.sort(); Buffer::I64(d) }
        Buffer::U8(v) => { let mut d = v.clone(); d.sort(); Buffer::U8(d) }
        Buffer::U16(v) => { let mut d = v.clone(); d.sort(); Buffer::U16(d) }
        Buffer::U32(v) => { let mut d = v.clone(); d.sort(); Buffer::U32(d) }
        Buffer::U64(v) => { let mut d = v.clone(); d.sort(); Buffer::U64(d) }
        Buffer::F16(v) => {
            let mut d = v.clone();
            d.sort_by(|a, b| float_cmp(*a, *b, f16_is_nan, is_neg_f16));
            for x in d.iter_mut() { *x = canon_f16(*x); }
            Buffer::F16(d)
        }
        Buffer::F32(v) => {
            let mut d = v.clone();
            d.sort_by(|a, b| float_cmp(*a, *b, f32::is_nan, is_neg_f32));
            for x in d.iter_mut() { *x = canon_f32(*x); }
            Buffer::F32(d)
        }
        Buffer::F64(v) => {
            let mut d = v.clone();
            d.sort_by(|a, b| float_cmp(*a, *b, f64::is_nan, is_neg_f64));
            for x in d.iter_mut() { *x = canon_f64(*x); }
            Buffer::F64(d)
        }
        Buffer::C64(v) => {
            let mut d = v.clone();
            d.sort_by(|a, b| complex_cmp(*a, *b, f32::is_nan));
            for x in d.iter_mut() { *x = canon_c64(*x); }
            Buffer::C64(d)
        }
        Buffer::C128(v) => {
            let mut d = v.clone();
            d.sort_by(|a, b| complex_cmp(*a, *b, f64::is_nan));
            for x in d.iter_mut() { *x = canon_c128(*x); }
            Buffer::C128(d)
        }
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("setops.rs: no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
    }
}

/// `aux[:-1][aux[1:] == aux[:-1]]` -- PLAIN adjacent equality (never
/// NaN-grouping: two adjacent NaNs, even bit-identical after
/// canonicalization, compare unequal under ordinary `PartialEq`, so no NaN
/// ever appears in `intersect1d`'s output -- matches real numpy).
fn intersect_body<T: PartialEq + Copy>(data: &[T]) -> Vec<T> {
    let n = data.len();
    let mut out = Vec::new();
    for i in 0..n.saturating_sub(1) {
        if data[i] == data[i + 1] {
            out.push(data[i]);
        }
    }
    out
}

/// `flag = [True, aux[1:]!=aux[:-1]..., True]; aux[flag[1:] & flag[:-1]]`
/// -- keep exactly the elements whose value differs from BOTH sorted
/// neighbors (appears an odd number of times, generically exactly once, in
/// the combined multiset).
fn setxor_body<T: PartialEq + Copy>(data: &[T]) -> Vec<T> {
    let n = data.len();
    let mut out = Vec::new();
    for i in 0..n {
        let left_ok = i == 0 || data[i] != data[i - 1];
        let right_ok = i + 1 == n || data[i] != data[i + 1];
        if left_ok && right_ok {
            out.push(data[i]);
        }
    }
    out
}

macro_rules! body_dispatch {
    ($buf:expr, $body:path) => {
        match $buf {
            Buffer::Bool(v) => Buffer::Bool($body(v.as_slice())),
            Buffer::I8(v) => Buffer::I8($body(v.as_slice())),
            Buffer::I16(v) => Buffer::I16($body(v.as_slice())),
            Buffer::I32(v) => Buffer::I32($body(v.as_slice())),
            Buffer::I64(v) => Buffer::I64($body(v.as_slice())),
            Buffer::U8(v) => Buffer::U8($body(v.as_slice())),
            Buffer::U16(v) => Buffer::U16($body(v.as_slice())),
            Buffer::U32(v) => Buffer::U32($body(v.as_slice())),
            Buffer::U64(v) => Buffer::U64($body(v.as_slice())),
            Buffer::F16(v) => Buffer::F16($body(v.as_slice())),
            Buffer::F32(v) => Buffer::F32($body(v.as_slice())),
            Buffer::F64(v) => Buffer::F64($body(v.as_slice())),
            Buffer::C64(v) => Buffer::C64($body(v.as_slice())),
            Buffer::C128(v) => Buffer::C128($body(v.as_slice())),
            Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("setops.rs: body_dispatch! has no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
        }
    };
}

/// `np.intersect1d(a, b, assume_unique=False, return_indices=False)`.
pub fn intersect1d(a: &NdArray, b: &NdArray, assume_unique: bool) -> Result<NdArray, IonpError> {
    let (ca, cb) = common_dtype_cast(a, b)?;
    let ua = maybe_unique(&ca, assume_unique)?;
    let ub = maybe_unique(&cb, assume_unique)?;
    let combined = concat_along_axis(&[ua, ub], 0)?;
    let aux = sort_canon_buffer(combined.buffer());
    let out_buf = body_dispatch!(&aux, intersect_body);
    let len = out_buf.len();
    NdArray::from_buffer(out_buf, vec![len], ArrOrder::C)
}

/// Gather `buf[idx[0]], buf[idx[1]], ...` into a freshly-allocated `Buffer`
/// of the same variant, preserving exact bit payloads (used for the
/// argsort-then-index steps below -- no canonicalization anywhere here,
/// matching real numpy's `aux[perm]`/`aux[mask]` fancy-indexing, which never
/// touches NaN payload or signed-zero bit patterns).
fn gather_buffer(buf: &Buffer, idx: &[usize]) -> Buffer {
    macro_rules! g {
        ($variant:ident, $v:expr) => {{
            let d: &[_] = $v;
            Buffer::$variant(idx.iter().map(|&i| d[i]).collect())
        }};
    }
    match buf {
        Buffer::Bool(v) => g!(Bool, v.as_slice()),
        Buffer::I8(v) => g!(I8, v.as_slice()),
        Buffer::I16(v) => g!(I16, v.as_slice()),
        Buffer::I32(v) => g!(I32, v.as_slice()),
        Buffer::I64(v) => g!(I64, v.as_slice()),
        Buffer::U8(v) => g!(U8, v.as_slice()),
        Buffer::U16(v) => g!(U16, v.as_slice()),
        Buffer::U32(v) => g!(U32, v.as_slice()),
        Buffer::U64(v) => g!(U64, v.as_slice()),
        Buffer::F16(v) => g!(F16, v.as_slice()),
        Buffer::F32(v) => g!(F32, v.as_slice()),
        Buffer::F64(v) => g!(F64, v.as_slice()),
        Buffer::C64(v) => g!(C64, v.as_slice()),
        Buffer::C128(v) => g!(C128, v.as_slice()),
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("setops.rs: no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
    }
}

/// `np.argsort(aux, kind='mergesort')` -- a STABLE full permutation (as
/// opposed to `sort_canon_buffer`'s in-place canonicalizing sort). Real
/// numpy's mergesort argsort uses plain `<` comparison with no signed-zero
/// tie-break and no NaN-payload canonicalization; ties (including multiple
/// equal NaNs, which sort to the end under plain `<` since `NaN < x` and
/// `x < NaN` are both `false`, making them mutually "equal" to a stable
/// comparison sort) keep original relative order, exactly Rust's
/// `slice::sort_by`'s own stability guarantee.
fn stable_argsort_indices(buf: &Buffer) -> Vec<usize> {
    let n = buf.len();
    let mut order: Vec<usize> = (0..n).collect();
    macro_rules! ord {
        ($v:expr) => {{
            let d: &[_] = $v;
            order.sort_by(|&i, &j| d[i].cmp(&d[j]));
        }};
    }
    match buf {
        Buffer::Bool(v) => ord!(v.as_slice()),
        Buffer::I8(v) => ord!(v.as_slice()),
        Buffer::I16(v) => ord!(v.as_slice()),
        Buffer::I32(v) => ord!(v.as_slice()),
        Buffer::I64(v) => ord!(v.as_slice()),
        Buffer::U8(v) => ord!(v.as_slice()),
        Buffer::U16(v) => ord!(v.as_slice()),
        Buffer::U32(v) => ord!(v.as_slice()),
        Buffer::U64(v) => ord!(v.as_slice()),
        Buffer::F16(v) => {
            let d = v.as_slice();
            order.sort_by(|&i, &j| float_cmp(d[i], d[j], f16_is_nan, |_| false));
        }
        Buffer::F32(v) => {
            let d = v.as_slice();
            order.sort_by(|&i, &j| float_cmp(d[i], d[j], f32::is_nan, |_| false));
        }
        Buffer::F64(v) => {
            let d = v.as_slice();
            order.sort_by(|&i, &j| float_cmp(d[i], d[j], f64::is_nan, |_| false));
        }
        Buffer::C64(v) => {
            let d = v.as_slice();
            order.sort_by(|&i, &j| complex_cmp(d[i], d[j], f32::is_nan));
        }
        Buffer::C128(v) => {
            let d = v.as_slice();
            order.sort_by(|&i, &j| complex_cmp(d[i], d[j], f64::is_nan));
        }
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("setops.rs: no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
    }
    order
}

/// `aux[1:] == aux[:-1]` -- plain (never NaN-grouping) adjacent equality
/// over an already-argsorted buffer, returning a length-`n-1` mask.
fn adjacent_eq_mask(buf: &Buffer) -> Vec<bool> {
    macro_rules! m {
        ($v:expr) => {{
            let d: &[_] = $v;
            (0..d.len().saturating_sub(1)).map(|i| d[i] == d[i + 1]).collect()
        }};
    }
    match buf {
        Buffer::Bool(v) => m!(v.as_slice()),
        Buffer::I8(v) => m!(v.as_slice()),
        Buffer::I16(v) => m!(v.as_slice()),
        Buffer::I32(v) => m!(v.as_slice()),
        Buffer::I64(v) => m!(v.as_slice()),
        Buffer::U8(v) => m!(v.as_slice()),
        Buffer::U16(v) => m!(v.as_slice()),
        Buffer::U32(v) => m!(v.as_slice()),
        Buffer::U64(v) => m!(v.as_slice()),
        Buffer::F16(v) => m!(v.as_slice()),
        Buffer::F32(v) => m!(v.as_slice()),
        Buffer::F64(v) => m!(v.as_slice()),
        Buffer::C64(v) => m!(v.as_slice()),
        Buffer::C128(v) => m!(v.as_slice()),
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("setops.rs: no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
    }
}

fn i64_vec(a: &NdArray) -> Vec<i64> {
    match a.buffer() {
        Buffer::I64(v) => v.clone(),
        _ => unreachable!("return_index arrays are always i64"),
    }
}

/// `np.intersect1d(a, b, assume_unique=assume_unique, return_indices=True)`
/// -- returns `(int1d, comm1, comm2)` exactly matching real numpy's
/// `_arraysetops_impl.intersect1d` (verified via `inspect.getsource`):
///
/// - `not assume_unique`: `ar1, ind1 = unique(ar1, return_index=True)` (and
///   likewise for `ar2`) -- flat unique with `equal_nan=True` (bare
///   `unique()`'s default), which is exactly `unique_general(..., true,
///   false, false, true)` with `optional_indices=True` selecting the
///   payload-preserving-argsort branch (not the canonicalizing in-place
///   sort `intersect1d`'s no-return-indices path uses above) -- so `ind1`/
///   `ind2` are indices into the ORIGINAL (pre-dedup) input arrays.
/// - `assume_unique`: `ar1 = ar1.ravel()`, `ar2 = ar2.ravel()`, no
///   dedup/indices step at all.
/// - `aux = concatenate((ar1, ar2))`; `aux_sort_indices =
///   argsort(aux, kind='mergesort')` (stable, unlike the plain in-place
///   `.sort()` the non-index path takes); `aux = aux[aux_sort_indices]`.
/// - `mask = aux[1:] == aux[:-1]` (plain equality, never NaN-grouping --
///   matches `intersect_body` above).
/// - `int1d = aux[:-1][mask]`.
/// - `ar1_indices = aux_sort_indices[:-1][mask]`; `ar2_indices =
///   aux_sort_indices[1:][mask] - ar1.size` (mechanical arithmetic,
///   reproduced verbatim -- `ar1.size` here is the size of `ar1` AFTER the
///   unique/ravel step, i.e. the length of the first operand of the
///   concatenation).
/// - `if not assume_unique: ar1_indices = ind1[ar1_indices]; ar2_indices =
///   ind2[ar2_indices]` -- remap through the original-position index
///   arrays. Under `assume_unique=True` misuse (duplicate values in the
///   input despite the promise), real numpy's `ar1_indices`/`ar2_indices`
///   can point at an arbitrary one of several equal-valued positions and
///   the raw mergesort-permutation arithmetic is used with NO indirection
///   and NO correction -- this is reproduced exactly, not fixed up, since
///   "honest" here means matching numpy's actual (documented-as-unsafe-
///   under-misuse) behavior byte-for-byte, not a more "correct" one.
pub fn intersect1d_return_indices(
    a: &NdArray,
    b: &NdArray,
    assume_unique: bool,
) -> Result<(NdArray, NdArray, NdArray), IonpError> {
    let (ca, cb) = common_dtype_cast(a, b)?;

    let (ar1, ind1, ar2, ind2): (NdArray, Option<NdArray>, NdArray, Option<NdArray>) =
        if assume_unique {
            (ca.ravel_order("C")?, None, cb.ravel_order("C")?, None)
        } else {
            let (v1, i1, _, _) = unique_general(&ca, true, false, false, true)?;
            let (v2, i2, _, _) = unique_general(&cb, true, false, false, true)?;
            (v1, i1, v2, i2)
        };

    let ar1_size = ar1.size();
    let combined = concat_along_axis(&[ar1, ar2], 0)?;
    let perm = stable_argsort_indices(combined.buffer());
    let sorted_buf = gather_buffer(combined.buffer(), &perm);
    let mask = adjacent_eq_mask(&sorted_buf);

    let hit_pos: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();

    let int1d_buf = gather_buffer(&sorted_buf, &hit_pos);
    let n_hit = hit_pos.len();
    let int1d_arr = NdArray::from_buffer(int1d_buf, vec![n_hit], ArrOrder::C)?;

    let mut ar1_idx_raw: Vec<i64> = hit_pos.iter().map(|&i| perm[i] as i64).collect();
    let mut ar2_idx_raw: Vec<i64> = hit_pos
        .iter()
        .map(|&i| perm[i + 1] as i64 - ar1_size as i64)
        .collect();

    if let (Some(i1), Some(i2)) = (ind1, ind2) {
        let iv1 = i64_vec(&i1);
        let iv2 = i64_vec(&i2);
        ar1_idx_raw = ar1_idx_raw.iter().map(|&i| iv1[i as usize]).collect();
        ar2_idx_raw = ar2_idx_raw.iter().map(|&i| iv2[i as usize]).collect();
    }

    let ar1_indices_arr = i64_array(ar1_idx_raw)?;
    let ar2_indices_arr = i64_array(ar2_idx_raw)?;
    Ok((int1d_arr, ar1_indices_arr, ar2_indices_arr))
}

/// `np.union1d(a, b)` -- `unique(concatenate((a, b)))` per numpy's own
/// documented equivalence, using bare `unique()`'s default (`equal_nan=
/// True`, canonicalizing) semantics -- exactly this crate's `unique_values`
/// helper.
pub fn union1d(a: &NdArray, b: &NdArray) -> Result<NdArray, IonpError> {
    let (ca, cb) = common_dtype_cast(a, b)?;
    let combined = concat_along_axis(&[ca.ravel_order("C")?, cb.ravel_order("C")?], 0)?;
    unique_values(&combined)
}

/// `np.setxor1d(a, b, assume_unique=False)`.
pub fn setxor1d(a: &NdArray, b: &NdArray, assume_unique: bool) -> Result<NdArray, IonpError> {
    let (ca, cb) = common_dtype_cast(a, b)?;
    let ua = maybe_unique(&ca, assume_unique)?;
    let ub = maybe_unique(&cb, assume_unique)?;
    let combined = concat_along_axis(&[ua, ub], 0)?;
    if combined.size() == 0 {
        return Ok(combined);
    }
    let aux = sort_canon_buffer(combined.buffer());
    let out_buf = body_dispatch!(&aux, setxor_body);
    let len = out_buf.len();
    NdArray::from_buffer(out_buf, vec![len], ArrOrder::C)
}

/// Per-element membership test using PLAIN (non-NaN-grouping) equality --
/// shared by `isin` and `setdiff1d` (see module doc for why `setdiff1d`
/// can reuse this directly without replicating numpy's internal
/// `assume_unique=True`-to-`isin` plumbing).
fn isin_mask(element: &Buffer, test: &Buffer) -> Vec<bool> {
    macro_rules! scan {
        ($variant:ident, $eq:expr) => {{
            let ex: &[_] = match element { Buffer::$variant(v) => v.as_slice(), _ => unreachable!() };
            let tx: &[_] = match test { Buffer::$variant(v) => v.as_slice(), _ => unreachable!() };
            let eq = $eq;
            ex.iter().map(|e| tx.iter().any(|t| eq(e, t))).collect::<Vec<bool>>()
        }};
    }
    match element {
        Buffer::Bool(_) => scan!(Bool, |e: &bool, t: &bool| e == t),
        Buffer::I8(_) => scan!(I8, |e: &i8, t: &i8| e == t),
        Buffer::I16(_) => scan!(I16, |e: &i16, t: &i16| e == t),
        Buffer::I32(_) => scan!(I32, |e: &i32, t: &i32| e == t),
        Buffer::I64(_) => scan!(I64, |e: &i64, t: &i64| e == t),
        Buffer::U8(_) => scan!(U8, |e: &u8, t: &u8| e == t),
        Buffer::U16(_) => scan!(U16, |e: &u16, t: &u16| e == t),
        Buffer::U32(_) => scan!(U32, |e: &u32, t: &u32| e == t),
        Buffer::U64(_) => scan!(U64, |e: &u64, t: &u64| e == t),
        Buffer::F16(_) => scan!(F16, |e: &f16, t: &f16| e == t),
        Buffer::F32(_) => scan!(F32, |e: &f32, t: &f32| e == t),
        Buffer::F64(_) => scan!(F64, |e: &f64, t: &f64| e == t),
        Buffer::C64(_) => scan!(C64, |e: &C64, t: &C64| e == t),
        Buffer::C128(_) => scan!(C128, |e: &C128, t: &C128| e == t),
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("setops.rs: no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
    }
}

fn filter_buffer_by_mask(buf: &Buffer, keep: &[bool]) -> Buffer {
    macro_rules! filt {
        ($variant:ident, $v:expr) => {
            Buffer::$variant($v.iter().zip(keep.iter()).filter(|(_, k)| **k).map(|(x, _)| *x).collect())
        };
    }
    match buf {
        Buffer::Bool(v) => filt!(Bool, v),
        Buffer::I8(v) => filt!(I8, v),
        Buffer::I16(v) => filt!(I16, v),
        Buffer::I32(v) => filt!(I32, v),
        Buffer::I64(v) => filt!(I64, v),
        Buffer::U8(v) => filt!(U8, v),
        Buffer::U16(v) => filt!(U16, v),
        Buffer::U32(v) => filt!(U32, v),
        Buffer::U64(v) => filt!(U64, v),
        Buffer::F16(v) => filt!(F16, v),
        Buffer::F32(v) => filt!(F32, v),
        Buffer::F64(v) => filt!(F64, v),
        Buffer::C64(v) => filt!(C64, v),
        Buffer::C128(v) => filt!(C128, v),
        Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("setops.rs: no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
    }
}

/// `np.setdiff1d(a, b, assume_unique=False)` -- `ar1[~isin(ar1, ar2)]`,
/// `ar1`/`ar2` each `ravel()` (assume_unique) or `unique()` (not) first.
/// Order-preserving (NOT re-sorted) when `assume_unique=True` -- matches
/// real numpy's own documented "only sorted if the input is sorted" note.
/// UNLIKE `intersect1d`/`setxor1d`/`union1d`, real numpy's `setdiff1d`
/// never concatenates `ar1`/`ar2` into one array, so no dtype promotion
/// happens: the OUTPUT keeps `a`'s own original dtype bit for bit (verified:
/// `np.setdiff1d(np.array([1,2,3],dtype=np.int32), np.array([2],dtype=
/// np.float64))` returns an `int32` array, not `float64`). The dtype
/// promotion only needs to happen internally, to make the membership
/// comparison against `b` well-typed.
pub fn setdiff1d(a: &NdArray, b: &NdArray, assume_unique: bool) -> Result<NdArray, IonpError> {
    let ar1 = maybe_unique(a, assume_unique)?;
    let ar2 = maybe_unique(b, assume_unique)?;
    let (ce, ct) = common_dtype_cast(&ar1, &ar2)?;
    let present = isin_mask(ce.buffer(), ct.buffer());
    let keep: Vec<bool> = present.iter().map(|&p| !p).collect();
    let out_buf = filter_buffer_by_mask(ar1.buffer(), &keep);
    let len = out_buf.len();
    NdArray::from_buffer(out_buf, vec![len], ArrOrder::C)
}

/// `np.isin(element, test_elements, assume_unique=False, invert=False)`.
/// `assume_unique` is accepted but has no observable effect on the
/// returned boolean mask (see module doc: membership per element is
/// dedup-invariant) -- real numpy's own `assume_unique` is likewise
/// documented as a pure speed hint for `isin`, never a correctness one.
/// `kind=` (numpy's `sort`/`table`/`None` algorithm-selection hint) is
/// accepted-and-ignored: every branch numpy could choose computes the same
/// per-element boolean this plain scan does (see module doc).
pub fn isin(element: &NdArray, test_elements: &NdArray, assume_unique: bool, invert: bool) -> Result<NdArray, IonpError> {
    let _ = assume_unique;
    let (ce, ct) = common_dtype_cast(element, test_elements)?;
    let flat_e = ce.to_contiguous();
    let flat_t = ct.ravel_order("C")?;
    let mut mask = isin_mask(flat_e.buffer(), flat_t.buffer());
    if invert {
        mask.iter_mut().for_each(|v| *v = !*v);
    }
    NdArray::from_buffer(Buffer::Bool(mask), flat_e.shape().to_vec(), ArrOrder::C)
}
