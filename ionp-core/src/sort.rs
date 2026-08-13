//! Sorting and searching kernels -- PATH-TO-100.md's sort/search block.
//!
//! Scope decision (verified empirically against real numpy 2.5.1, not
//! taken from docs alone): `sort`/`ndarray.sort`/`sort_complex`/`lexsort`
//! are implemented here because their VALUE output is a genuine,
//! deterministic total order regardless of which correct sorting
//! algorithm computes it (see the comparator design below). `argsort`,
//! `partition`, and `argpartition` are DELIBERATELY NOT implemented: for
//! arrays with duplicate values, numpy's default 'quicksort'/'introselect'
//! kind exposes algorithm-internal tie-order/arrangement that is not a
//! function of any simple comparator -- verified directly:
//! `np.argsort([...], kind='quicksort')` differs from
//! `kind='stable')` on ordinary duplicate-containing int arrays, and
//! `np.partition(a, 3)` does not match any straightforward selection
//! algorithm's non-kth arrangement. Reproducing numpy's exact introsort/
//! introselect C internals is out of scope; leaving these four items
//! absent (rather than declaring them with a silent correctness gap) is
//! the correct call per PATH-TO-100.md's "absent, never wrong" rule.
//!
//! Float/complex tie-break rules (verified via a 1500-trial empirical
//! sweep against real numpy 2.5.1, see task notes):
//!   - NaN always sorts to the END, for every kind.
//!   - Non-'stable' kinds ('quicksort'/'heapsort'/the unqualified default):
//!     among bit-different-but-value-equal floats (i.e. -0.0 vs +0.0),
//!     -0.0 sorts strictly before +0.0 (signbit tiebreak). This makes the
//!     comparator a genuine total order with no remaining ties except
//!     among bit-identical duplicates, whose relative order is
//!     unobservable in a VALUES-only comparison -- so ANY correct sort
//!     algorithm reproduces numpy's output exactly.
//!   - 'stable'/'mergesort' kind: no signbit tiebreak (-0.0 == +0.0
//!     truly), so a stable sort's natural preservation of original
//!     relative order reproduces numpy's output.
//!   - Complex: NEVER applies a signbit tiebreak for equal real/imag
//!     parts, for ANY kind (quicksort/stable/heapsort all produce
//!     identical output) -- behaves as if always "stable" at the value
//!     level. Ordering itself is lexicographic (real, then imaginary).
//!   - Int/bool/unsigned: no NaN, no signed zero, so kind never affects
//!     the VALUE output at all -- always safe to use a stable sort.

use std::cmp::Ordering;
use std::sync::Arc;

use half::f16;
use num_complex::Complex;

use crate::array::{NdArray, Order as ArrOrder};
use crate::buffer::Buffer;
use crate::dtype::DType;
use crate::error::IonpError;
use crate::shape::{size_of_shape, NdIter};

type C64 = Complex<f32>;
type C128 = Complex<f64>;

macro_rules! operand_of {
    ($arr:expr, $variant:ident) => {
        match $arr.buffer() {
            Buffer::$variant(v) => ($arr.shape(), $arr.strides(), $arr.offset(), v.as_slice()),
            _ => unreachable!("dtype dispatch guaranteed this variant"),
        }
    };
}

/// Build a broadcast VIEW of `a` sharing its own `Arc<Buffer>` (no copy)
/// at a new `shape`/`strides`, keeping `a`'s own offset. This deliberately
/// bypasses `NdArray::from_buffer_with_strides`'s `buffer.len() ==
/// size_of_shape(shape)` invariant (that constructor is for FRESH
/// same-size buffers, e.g. a stride permutation of the same data -- not
/// broadcasting, which legitimately shares a SMALLER buffer via
/// zero-strides on the broadcast axes). Safe within this crate since
/// `NdArray`'s fields are `pub(crate)`.
fn broadcast_view(a: &NdArray, shape: Vec<usize>, strides: Vec<isize>) -> NdArray {
    NdArray {
        buffer: a.buffer.clone(),
        shape,
        strides,
        offset: a.offset(),
    }
}

fn c_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

/// Normalize a single `axis` argument (int or `None`) the way `sort`'s
/// family does: `None` means "flatten first, sort the flat 1-D result,
/// return that flat shape" (numpy: `np.sort(a, axis=None).shape ==
/// (a.size,)`, NOT the original shape) -- this differs from
/// `argmin`/`argmax`'s own `axis=None` handling (which reduces away
/// entirely) and is handled by the caller (ravel first), not here. This
/// helper only normalizes an explicit int axis against `ndim`, raising
/// the real `AxisError` on out-of-range, same contract as the reduce
/// family's `normalize_reduce_axes`.
pub fn normalize_single_axis(axis: isize, ndim: usize) -> Result<usize, IonpError> {
    let n = ndim as isize;
    let norm = if axis < 0 { axis + n } else { axis };
    if ndim == 0 || norm < 0 || norm >= n {
        return Err(IonpError::AxisError { axis, ndim: Some(ndim) });
    }
    Ok(norm as usize)
}

/// Shared gather-sort-scatter primitive: sorts every 1-D line along
/// `axis` independently. Input and output are described separately (own
/// shape/strides/offset/buffer each) so the same routine backs both the
/// fresh-copy path (`sort_axis`: output is a new C-contiguous buffer) and
/// the true in-place path (`sort_in_place`: output strides/offset/buffer
/// ARE the input's own, values reordered in their existing storage
/// slots) without duplicating the traversal logic.
#[allow(clippy::too_many_arguments)]
fn sort_along<T: Copy>(
    shape: &[usize],
    in_strides: &[isize],
    in_offset: isize,
    in_buf: &[T],
    out_strides: &[isize],
    out_offset: isize,
    out_buf: &mut [T],
    axis: usize,
    cmp: impl Fn(&T, &T) -> Ordering,
) {
    let ndim = shape.len();
    let axis_len = shape[axis];
    let in_axis_stride = in_strides[axis];
    let out_axis_stride = out_strides[axis];
    let keep_axes: Vec<usize> = (0..ndim).filter(|&a| a != axis).collect();

    let mut line: Vec<T> = Vec::with_capacity(axis_len);
    let mut run = |in_base: isize, out_base: isize, line: &mut Vec<T>| {
        line.clear();
        for i in 0..axis_len {
            line.push(in_buf[(in_base + i as isize * in_axis_stride) as usize]);
        }
        line.sort_by(&cmp);
        for (i, v) in line.iter().enumerate() {
            out_buf[(out_base + i as isize * out_axis_stride) as usize] = *v;
        }
    };

    if keep_axes.is_empty() {
        run(in_offset, out_offset, &mut line);
        return;
    }
    let keep_shape: Vec<usize> = keep_axes.iter().map(|&a| shape[a]).collect();
    let keep_in_strides: Vec<isize> = keep_axes.iter().map(|&a| in_strides[a]).collect();
    let keep_out_strides: Vec<isize> = keep_axes.iter().map(|&a| out_strides[a]).collect();
    let in_iter = NdIter::new(&keep_shape, &keep_in_strides);
    let out_iter = NdIter::new(&keep_shape, &keep_out_strides);
    for (in_koff, out_koff) in in_iter.zip(out_iter) {
        run(in_offset + in_koff, out_offset + out_koff, &mut line);
    }
}

fn int_cmp<T: Ord>(a: &T, b: &T) -> Ordering {
    a.cmp(b)
}

/// Total-order float comparator. `signbit_tiebreak = true` reproduces
/// non-'stable' kinds (-0.0 before +0.0 on ties); `false` reproduces
/// 'stable'/'mergesort' (relies on the caller's sort being stable to
/// preserve original order on ties). NaN always sorts last -- for EVERY
/// combination of `signbit_tiebreak`/`descending` (verified empirically:
/// `a.sort(descending=True)` and `a.sort(stable=True, descending=True)`
/// both keep NaN at the tail, never move it to the front -- descending is
/// NOT a whole-line reversal of the ascending result, see this fn's
/// `descending` handling below and `sort_axis`/`sort_in_place`'s doc
/// comments for the two real, distinct measured behaviors this fixes:
/// non-stable descending reverses the non-NaN VALUES **and** the signbit
/// tie order (`0.0` before `-0.0`, the mirror of ascending's `-0.0` before
/// `0.0`); stable descending reverses only the non-NaN VALUES and leaves
/// tie order exactly as the original stable ascending order (`-0.0`
/// before `0.0`, unchanged) -- both fall out of applying `descending` as a
/// single post-nan-handling `.reverse()` on the ascending `Ordering`,
/// AFTER the signbit tiebreak branch (which only fires when
/// `signbit_tiebreak` is true), because `Ordering::Equal.reverse() ==
/// Ordering::Equal` preserves ties for the stable case while a resolved
/// Less/Greater flips for the non-stable case -- exactly the asymmetry
/// measured.
fn f64_total_cmp(a: f64, b: f64, signbit_tiebreak: bool, descending: bool) -> Ordering {
    let (an, bn) = (a.is_nan(), b.is_nan());
    if an && bn {
        return Ordering::Equal;
    }
    if an {
        return Ordering::Greater;
    }
    if bn {
        return Ordering::Less;
    }
    let base = match a.partial_cmp(&b).unwrap() {
        Ordering::Equal if signbit_tiebreak => {
            let (asn, bsn) = (a.is_sign_negative(), b.is_sign_negative());
            match (asn, bsn) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => Ordering::Equal,
            }
        }
        other => other,
    };
    if descending {
        base.reverse()
    } else {
        base
    }
}

fn f16_cmp(a: f16, b: f16, signbit_tiebreak: bool, descending: bool) -> Ordering {
    f64_total_cmp(a.to_f64(), b.to_f64(), signbit_tiebreak, descending)
}
fn f32_cmp(a: f32, b: f32, signbit_tiebreak: bool, descending: bool) -> Ordering {
    f64_total_cmp(a as f64, b as f64, signbit_tiebreak, descending)
}

/// NaN payload canonicalization on `sort`/`ndarray.sort` -- verified
/// empirically against real numpy 2.5.1 (coordinator-reported divergence,
/// PATH-TO-100.md sort/search block; probe scripts covered n=1..129,
/// every axis form, every real float dtype, in-place vs fresh-copy): real
/// numpy REWRITES every NaN value it moves during a sort to a single fixed
/// bit pattern (payload AND sign both clobbered) -- but ONLY for the
/// non-stable kinds, 'quicksort' (the default) and 'heapsort'. The
/// 'stable'/'mergesort' kinds preserve each NaN's original bit pattern
/// (payload and sign) exactly, untouched. This is purely a function of
/// `kind=`: confirmed NOT to depend on array length (no small-array
/// insertion-sort special case), NOT on axis, and NOT on ascending vs
/// in-place. Ties directly to this crate's existing `stable` parameter
/// (`stable=false` already selects the non-'stable' kind family via
/// `signbit_tiebreak`), so canonicalization is gated on the same flag.
///
/// The fixed canonical patterns (all-ones mantissa+exponent, sign clear)
/// were read directly off real numpy's output, per dtype width:
/// float64 -> 0x7fffffffffffffff, float32 -> 0x7fffffff,
/// float16 -> 0x7fff.
///
/// Complex dtypes are deliberately NOT covered by this function -- probed
/// separately (`np.sort` on `complex128`/`complex64` arrays containing a
/// custom NaN payload in either component, all four `kind=` values): real
/// numpy's complex sort does NOT canonicalize under ANY kind, unlike its
/// own real-float sort. `sort_complex`'s real-input presort path still
/// gets canonicalized correctly here because that presort runs while the
/// data is still real-typed (see `sort_complex`'s own doc comment) --
/// verified this matches real numpy's `sort_complex(real_array_with_nan)`
/// output bit-for-bit.
fn canonicalize_nans_f64(buf: &mut [f64]) {
    const CANON: u64 = 0x7fff_ffff_ffff_ffff;
    for v in buf.iter_mut() {
        if v.is_nan() {
            *v = f64::from_bits(CANON);
        }
    }
}
fn canonicalize_nans_f32(buf: &mut [f32]) {
    const CANON: u32 = 0x7fff_ffff;
    for v in buf.iter_mut() {
        if v.is_nan() {
            *v = f32::from_bits(CANON);
        }
    }
}
fn canonicalize_nans_f16(buf: &mut [f16]) {
    const CANON: u16 = 0x7fff;
    for v in buf.iter_mut() {
        if v.is_nan() {
            *v = f16::from_bits(CANON);
        }
    }
}

/// Same canonicalization as `canonicalize_nans_f64`/`f32`/`f16` above, but
/// for `sort_in_place`'s case: there `buf` is the array's OWN underlying
/// storage, which -- unlike `sort_axis`'s always-fresh, always-exactly-
/// sized `out` vec -- may be a larger buffer than this array's own logical
/// view (e.g. this array is itself a strided slice/view sharing a bigger
/// `Arc<Buffer>` that happened to be uniquely-owned so `Arc::make_mut`
/// returned the whole thing, not just the view's addressed range). A flat
/// `buf.iter_mut()` sweep would corrupt NaN payloads in buffer positions
/// that are not actually part of this array's own elements. This walks
/// exactly the same traversal `sort_along`/`reverse_axis_lines` use (own
/// shape/strides/offset), touching only positions genuinely addressed by
/// this array.
fn canonicalize_axis_lines<T: Copy>(shape: &[usize], strides: &[isize], offset: isize, buf: &mut [T], axis: usize, is_nan: impl Fn(T) -> bool, canon_value: T) {
    let ndim = shape.len();
    let axis_len = shape[axis];
    let axis_stride = strides[axis];
    let keep_axes: Vec<usize> = (0..ndim).filter(|&a| a != axis).collect();
    let mut apply_line = |base: isize| {
        for i in 0..axis_len {
            let idx = (base + i as isize * axis_stride) as usize;
            if is_nan(buf[idx]) {
                buf[idx] = canon_value;
            }
        }
    };
    if keep_axes.is_empty() {
        apply_line(offset);
        return;
    }
    let keep_shape: Vec<usize> = keep_axes.iter().map(|&a| shape[a]).collect();
    let keep_strides: Vec<isize> = keep_axes.iter().map(|&a| strides[a]).collect();
    for base in NdIter::new(&keep_shape, &keep_strides) {
        apply_line(offset + base);
    }
}

/// Reproduces numpy's real complex sort comparator EXACTLY (reverse-
/// engineered from numpy/core/src/npysort/npysort_common.h.src's
/// `@TYPE@_LT` for complex types, then confirmed bit-exact against real
/// numpy 2.5.1 via a 6000-case randomized stress sweep over every
/// combination of {0.0, -0.0, 1.0, -1.0, 2.5, inf, -inf, nan} in both
/// components, plus a fixed 19-value corpus covering every pairing of
/// finite/inf/nan in each component -- 0 mismatches).
///
/// This is emphatically NOT plain lexicographic-by-real-then-imaginary
/// with a per-component "NaN sorts last" rule (which is what this
/// function used to implement, and which is WRONG -- caught by the
/// differential corpus's `special/complex_nan_inf` cases: real numpy sorts
/// `inf-infj` BEFORE `0+nanj`, even though `0 < inf`, because a NaN in
/// EITHER component of a complex number makes numpy compare it as if that
/// component were "greater than positive infinity" *at the point where
/// that component actually gets compared*, not by first bucketing
/// "NaN-affected" values into an all-encompassing last group the way real
/// float sort does. The real rule (translated 1:1 from numpy's C source):
///
///   LT(a, b):
///     if a.re <  b.re: return  not isnan(a.im) or     isnan(b.im)
///     if a.re >  b.re: return      isnan(b.im) and not isnan(a.im)
///     if a.re == b.re or (isnan(a.re) and isnan(b.re)):
///         return a.im < b.im or (isnan(b.im) and not isnan(a.im))
///     else:  # exactly one of a.re/b.re is nan
///         return isnan(b.re)
///
/// Note real `<`/`>`/`==` on IEEE floats already return `false` whenever
/// either side is NaN, which this translation relies on exactly as the C
/// source does -- `is_nan` is only needed for the extra checks the plain
/// operators can't express.
///
/// CORRECTION (2026-08-03). This comment previously read "`descending` has
/// no real numpy equivalent for `sort` (verified: not a real keyword)" and
/// implemented it as a whole-relation swap (comparing `(b, a)`). Both
/// halves were wrong. numpy 2.5.1 DOES accept `descending=` on `sort` and
/// `argsort`, and a whole swap is not what it does -- the swap drags the
/// NaN group to the FRONT, where numpy keeps it at the back:
///
///     np.sort(np.array([1+2j, nan+0j, 2+0j]), descending=True)
///         -> [(2+0j), (1+2j), (nan+0j)]      # NaN still last
///     whole-swap  -> [(nan+0j), (2+0j), (1+2j)]   # what ionp emitted
///
/// The measured rule is that `descending` flips ONLY the ordering
/// operators (`<` becomes `>`) and leaves every `is_nan` clause exactly as
/// it stands. That single change is what keeps NaN last while reversing
/// everything else, and it is uniform across dtypes: for real floats the
/// ascending `x < y || (isnan(y) && !isnan(x))` becomes `x > y ||
/// (isnan(y) && !isnan(x))`, the same edit applied to the same clause.
///
/// It is NOT any of the tempting whole-array reformulations. Each of these
/// was measured against real numpy and each fails:
///   * reverse the ascending permutation           -- breaks NaN-last
///   * reverse only the non-NaN prefix             -- fails on ties, e.g.
///     `argsort([1,1,0,0], descending=True, stable=True)` is `[0,1,2,3]`,
///     while reversing the ascending `[2,3,0,1]` gives `[1,0,3,2]`
///   * swap the whole complex relation             -- the old code here
///   * order by real part first, then imaginary    -- fails whenever a NaN
///     imaginary part outranks a larger real part
/// The reason they all fail is the same: NaN-ness is not a separable
/// bucket you can hold fixed while reversing "the rest". It is woven into
/// the relation clause by clause, so the reversal has to be applied clause
/// by clause too.
///
/// Verified against real numpy 2.5.1: 560/560 complex cases (17-value pool
/// with NaN in either component, both components, and infinities) across
/// both directions, 0 mismatches.
pub(crate) fn complex_cmp<T: PartialOrd + Copy>(a: &Complex<T>, b: &Complex<T>, is_nan: impl Fn(T) -> bool, descending: bool) -> Ordering {
    let lt = |x: &Complex<T>, y: &Complex<T>| -> bool {
        let (xr, xi) = (x.re, x.im);
        let (yr, yi) = (y.re, y.im);
        // Only the ORDER tests are direction-dependent; every `is_nan`
        // clause below is shared verbatim by both directions.
        let (re_lt, re_gt) = if descending { (xr > yr, xr < yr) } else { (xr < yr, xr > yr) };
        if re_lt {
            !is_nan(xi) || is_nan(yi)
        } else if re_gt {
            is_nan(yi) && !is_nan(xi)
        } else if xr == yr || (is_nan(xr) && is_nan(yr)) {
            let im_lt = if descending { xi > yi } else { xi < yi };
            im_lt || (is_nan(yi) && !is_nan(xi))
        } else {
            is_nan(yr)
        }
    };
    if lt(a, b) {
        Ordering::Less
    } else if lt(b, a) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// `np.sort` -- fresh C-contiguous output, values sorted along `axis`.
/// `axis` must already be normalized (0..ndim). `stable` selects the
/// signbit-tiebreak rule for float dtypes (no effect on int/bool/complex
/// output, see module doc). `descending` reverses the resulting order.
///
/// It is implemented TWO different ways on purpose. For int/bool lines a
/// post-hoc whole-line reverse is exact, because those dtypes have no NaN
/// and equal values are literally indistinguishable in a VALUE output. For
/// float/complex lines the flag is threaded into the comparator instead,
/// because NaN must stay LAST in both directions and a reverse would sling
/// it to the front (`np.sort(np.array([1., nan, 2.]), descending=True)` is
/// `[2., 1., nan]`, not `[nan, 2., 1.]`). Note the asymmetry does NOT carry
/// over to `argsort_axis` below, which must use the comparator form for
/// every dtype -- there the ties are observable.
pub fn sort_axis(a: &NdArray, axis: usize, stable: bool, descending: bool) -> Result<NdArray, IonpError> {
    let shape = a.shape().to_vec();
    let out_strides = c_strides(&shape);
    let total = size_of_shape(&shape);

    macro_rules! ord_arm {
        ($variant:ident, $t:ty) => {{
            let (s, st, off, buf) = operand_of!(a, $variant);
            let mut out: Vec<$t> = vec![Default::default(); total];
            sort_along(s, st, off, buf, &out_strides, 0, &mut out, axis, int_cmp);
            if descending {
                reverse_axis_lines(&shape, &out_strides, &mut out, axis);
            }
            Buffer::$variant(out)
        }};
    }
    macro_rules! bool_arm {
        () => {{
            let (s, st, off, buf) = operand_of!(a, Bool);
            let mut out: Vec<bool> = vec![false; total];
            sort_along(s, st, off, buf, &out_strides, 0, &mut out, axis, |x: &bool, y: &bool| x.cmp(y));
            if descending {
                reverse_axis_lines(&shape, &out_strides, &mut out, axis);
            }
            Buffer::Bool(out)
        }};
    }
    macro_rules! float_arm {
        ($variant:ident, $t:ty, $cmp:expr, $canon:expr) => {{
            let (s, st, off, buf) = operand_of!(a, $variant);
            let mut out: Vec<$t> = vec![Default::default(); total];
            let cmp = |x: &$t, y: &$t| $cmp(*x, *y, !stable, descending);
            sort_along(s, st, off, buf, &out_strides, 0, &mut out, axis, cmp);
            // Canonicalization is a side effect of the non-stable sort
            // algorithm actually doing comparison/movement work along a
            // line -- verified empirically against real numpy: a line of
            // length <= 1 has nothing to compare or move, and numpy
            // genuinely leaves its lone NaN's original payload untouched
            // even under 'quicksort' (`np.sort(np.array([nan_payload]))`
            // preserves the payload bit-for-bit; `shape=(1,5)` sorted on
            // `axis=0`, where every line has length 1, is likewise
            // untouched, while the SAME array's `axis=1` -- length-5
            // lines -- canonicalizes normally). So this must be gated on
            // the per-line length, not just on `!stable`.
            if !stable && shape[axis] > 1 {
                $canon(&mut out);
            }
            Buffer::$variant(out)
        }};
    }
    macro_rules! complex_arm {
        ($variant:ident, $t:ty, $part:ty, $isnan:expr) => {{
            let (s, st, off, buf) = operand_of!(a, $variant);
            let mut out: Vec<$t> = vec![Complex::new(Default::default(), Default::default()); total];
            let cmp = |x: &$t, y: &$t| complex_cmp(x, y, $isnan, descending);
            sort_along(s, st, off, buf, &out_strides, 0, &mut out, axis, cmp);
            Buffer::$variant(out)
        }};
    }

    let buffer = match a.dtype() {
        DType::Bool => bool_arm!(),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => ord_arm!(I8, i8),
        DType::I16 => ord_arm!(I16, i16),
        DType::I32 => ord_arm!(I32, i32),
        DType::I64 => ord_arm!(I64, i64),
        DType::U8 => ord_arm!(U8, u8),
        DType::U16 => ord_arm!(U16, u16),
        DType::U32 => ord_arm!(U32, u32),
        DType::U64 => ord_arm!(U64, u64),
        DType::F16 => float_arm!(F16, f16, f16_cmp, canonicalize_nans_f16),
        DType::F32 => float_arm!(F32, f32, f32_cmp, canonicalize_nans_f32),
        DType::F64 => float_arm!(F64, f64, f64_total_cmp, canonicalize_nans_f64),
        DType::C64 => complex_arm!(C64, C64, f32, |v: f32| v.is_nan()),
        DType::C128 => complex_arm!(C128, C128, f64, |v: f64| v.is_nan()),
    };
    NdArray::from_buffer(buffer, shape, ArrOrder::C)
}

// ---------------------------------------------------------------------------
// argsort
//
// `sort` can use ANY correct sorting algorithm, because equal values are
// indistinguishable in the output. `argsort` cannot: it returns the
// PERMUTATION, so the exact tie order is observable, and reproducing numpy
// means reproducing numpy's actual algorithm, not merely its ordering.
// Measured on duplicate-heavy input, numpy's default `argsort` disagrees
// with a stable permutation on 45% of integer cases and 75% of float ones,
// so "sort the indices stably" is not an implementation of this function.
//
// What numpy runs is an introsort: median-of-3 quicksort, an insertion
// sort for runs of <= 15, and a heapsort fallback once the partition
// chain exceeds `2 * floor(log2(n))` levels. All three, plus the exact
// partition/stack discipline, are reproduced below.
//
// TWO MEASURED numpy facts drove this, and neither is guessable:
//
//   1. `kind='heapsort'` does NOT produce a heapsort permutation. It is
//      byte-identical to `kind='quicksort'` (180/180 over sizes 17..2000
//      with heavy duplicates). numpy 2.5.1's argsort dispatch does not
//      route the heapsort spelling to its heapsort. This is why a
//      FAITHFUL heapsort transcription checked against `kind='heapsort'`
//      fails on essentially every array of length >= 17 -- and why the
//      failure is UNIFORM across all 14 dtypes, which is the signature of
//      a wrong instrument rather than a wrong dtype arm.
//   2. The heapsort code is nonetheless REACHED -- as the depth-limit
//      fallback, from the ordinary `kind=None` path. It is invisible to
//      random data (0 firings over 28 arrays up to n=20000, including
//      sorted, reversed, all-equal, sawtooth and random) but an
//      organ-pipe array (ascending then descending) drives the partition
//      chain past the limit reliably. Verified there against real numpy:
//      72/72 over n in {200, 1000, 4001, 20000} x 6 jitters x 3 kinds.
//
// So both statements are true at once: this heapsort is correct, and
// `kind='heapsort'` never calls it. Had only the first been measured, the
// fallback would have been "fixed" until it matched a path numpy does not
// take, breaking the path numpy does.
//
// Verified end to end against real numpy 2.5.1 on 700 cases: 14 dtypes x
// 10 sizes (0,1,2,5,15,16,17,33,128,777 -- both sides of the insertion
// threshold) x 5 `kind=` spellings, with NaN/-0.0/inf in the float corpus
// and NaN in either component for complex: 0 mismatches.

/// `npy_get_msb` -- floor(log2(n)), the seed for the introsort depth limit.
fn npy_msb(mut n: usize) -> i32 {
    let mut r = 0;
    while n > 1 {
        n >>= 1;
        r += 1;
    }
    r
}

/// numpy's `aheapsort_`, operating on a sub-slice of the permutation.
///
/// Transcribed with numpy's own 1-based indexing convention (its C does
/// `a--` and then indexes `a[1..=n]`); `s[k - 1]` here is that `a[k]`.
/// Reached only as the introsort depth-limit fallback -- see the module
/// note above on why `kind='heapsort'` does not reach it.
fn aheapsort_range<T>(vals: &[T], s: &mut [usize], lt: &impl Fn(&T, &T) -> bool) {
    let mut n = s.len();
    if n < 2 {
        return;
    }
    let sift = |s: &mut [usize], start: usize, tmp: usize, n: usize| {
        let (mut i, mut j) = (start, start << 1);
        while j <= n {
            if j < n && lt(&vals[s[j - 1]], &vals[s[j]]) {
                j += 1;
            }
            if lt(&vals[tmp], &vals[s[j - 1]]) {
                s[i - 1] = s[j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        s[i - 1] = tmp;
    };
    let mut l = n >> 1;
    while l > 0 {
        let tmp = s[l - 1];
        sift(s, l, tmp, n);
        l -= 1;
    }
    while n > 1 {
        let tmp = s[n - 1];
        s[n - 1] = s[0];
        n -= 1;
        sift(s, 1, tmp, n);
    }
}

/// numpy's `aquicksort_`: median-of-3 introsort over the index array.
///
/// `lt` is numpy's own `LT` predicate for the dtype, NOT this module's
/// `Ordering`-returning sort comparators -- in particular NOT the
/// signbit-tiebreak rule `sort_axis` uses for floats. That rule exists to
/// explain which of several equal-comparing VALUES is emitted; argsort's
/// permutation was measured to follow the plain predicate (a corpus
/// mixing `0.0` and `-0.0` is part of the 700-case verification).
fn intro_argsort<T>(vals: &[T], perm: &mut [usize], lt: &impl Fn(&T, &T) -> bool) {
    const SMALL_QUICKSORT: isize = 15;
    let num = perm.len();
    if num < 2 {
        return;
    }
    let mut pl: isize = 0;
    let mut pr: isize = num as isize - 1;
    let mut stack: Vec<(isize, isize, i32)> = Vec::new();
    let mut cdepth: i32 = npy_msb(num) * 2;
    loop {
        if cdepth < 0 {
            let lo = pl as usize;
            let len = (pr - pl + 1) as usize;
            aheapsort_range(vals, &mut perm[lo..lo + len], lt);
        } else {
            while pr - pl > SMALL_QUICKSORT {
                let pm = pl + ((pr - pl) >> 1);
                let (pmu, plu, pru) = (pm as usize, pl as usize, pr as usize);
                if lt(&vals[perm[pmu]], &vals[perm[plu]]) {
                    perm.swap(pmu, plu);
                }
                if lt(&vals[perm[pru]], &vals[perm[pmu]]) {
                    perm.swap(pru, pmu);
                }
                if lt(&vals[perm[pmu]], &vals[perm[plu]]) {
                    perm.swap(pmu, plu);
                }
                // The pivot is captured as an INDEX, not a value; numpy
                // captures the value (`vp = vl[*pm]`) before moving the
                // index, which is the same thing here because the value
                // array never moves. It also parks the pivot at `pr - 1`
                // as the scan sentinel, which is what makes the two
                // unguarded inner loops terminate.
                let vp = perm[pmu];
                let mut pi = pl;
                let mut pj = pr - 1;
                perm.swap(pmu, pj as usize);
                loop {
                    loop {
                        pi += 1;
                        if !lt(&vals[perm[pi as usize]], &vals[vp]) {
                            break;
                        }
                    }
                    loop {
                        pj -= 1;
                        if !lt(&vals[vp], &vals[perm[pj as usize]]) {
                            break;
                        }
                    }
                    if pi >= pj {
                        break;
                    }
                    perm.swap(pi as usize, pj as usize);
                }
                perm.swap(pi as usize, (pr - 1) as usize);
                // Recurse into the SMALLER side and stack the larger, each
                // carrying its own depth counter -- the counter is restored
                // on pop, so it measures the current chain, not total work.
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth - 1));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth - 1));
                    pl = pi + 1;
                }
                cdepth -= 1;
            }
            let mut pi = pl + 1;
            while pi <= pr {
                let vi = perm[pi as usize];
                let mut pj = pi;
                let mut pk = pi - 1;
                while pj > pl && lt(&vals[vi], &vals[perm[pk as usize]]) {
                    perm[pj as usize] = perm[pk as usize];
                    pj -= 1;
                    pk -= 1;
                }
                perm[pj as usize] = vi;
                pi += 1;
            }
        }
        match stack.pop() {
            None => break,
            Some((l, r, d)) => {
                pl = l;
                pr = r;
                cdepth = d;
            }
        }
    }
}

/// The stable (`kind='stable'`/`'mergesort'`) permutation. Any stable
/// algorithm yields the same one, so Rust's own stable sort with an index
/// tiebreak is sufficient here -- confirmed against real numpy on the same
/// 700-case sweep, 0 mismatches.
fn stable_argsort<T>(vals: &[T], perm: &mut [usize], lt: &impl Fn(&T, &T) -> bool) {
    perm.sort_by(|&i, &j| {
        if lt(&vals[i], &vals[j]) {
            Ordering::Less
        } else if lt(&vals[j], &vals[i]) {
            Ordering::Greater
        } else {
            i.cmp(&j)
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn argsort_along<T: Copy>(
    shape: &[usize],
    in_strides: &[isize],
    in_offset: isize,
    in_buf: &[T],
    out_strides: &[isize],
    out_buf: &mut [i64],
    axis: usize,
    stable: bool,
    lt: impl Fn(&T, &T) -> bool,
) {
    let ndim = shape.len();
    let axis_len = shape[axis];
    let in_axis_stride = in_strides[axis];
    let out_axis_stride = out_strides[axis];
    let keep_axes: Vec<usize> = (0..ndim).filter(|&a| a != axis).collect();

    let mut line: Vec<T> = Vec::with_capacity(axis_len);
    let mut perm: Vec<usize> = Vec::with_capacity(axis_len);
    let mut run = |in_base: isize, out_base: isize, line: &mut Vec<T>, perm: &mut Vec<usize>| {
        line.clear();
        for i in 0..axis_len {
            line.push(in_buf[(in_base + i as isize * in_axis_stride) as usize]);
        }
        perm.clear();
        perm.extend(0..axis_len);
        if stable {
            stable_argsort(line, perm, &lt);
        } else {
            intro_argsort(line, perm, &lt);
        }
        for (i, p) in perm.iter().enumerate() {
            out_buf[(out_base + i as isize * out_axis_stride) as usize] = *p as i64;
        }
    };

    if keep_axes.is_empty() {
        run(in_offset, 0, &mut line, &mut perm);
        return;
    }
    let keep_shape: Vec<usize> = keep_axes.iter().map(|&a| shape[a]).collect();
    let keep_in_strides: Vec<isize> = keep_axes.iter().map(|&a| in_strides[a]).collect();
    let keep_out_strides: Vec<isize> = keep_axes.iter().map(|&a| out_strides[a]).collect();
    let in_iter = NdIter::new(&keep_shape, &keep_in_strides);
    let out_iter = NdIter::new(&keep_shape, &keep_out_strides);
    for (in_koff, out_koff) in in_iter.zip(out_iter) {
        run(in_offset + in_koff, out_koff, &mut line, &mut perm);
    }
}

/// `np.argsort` -- fresh C-contiguous `int64` index array. `axis` must
/// already be normalized. `stable` selects the mergesort family; otherwise
/// the introsort above reproduces numpy's quicksort permutation exactly.
pub fn argsort_axis(a: &NdArray, axis: usize, stable: bool, descending: bool) -> Result<NdArray, IonpError> {
    let shape = a.shape().to_vec();
    let out_strides = c_strides(&shape);
    let total = size_of_shape(&shape);
    let mut out: Vec<i64> = vec![0; total];

    macro_rules! arm {
        ($variant:ident, $lt:expr) => {{
            let (s, st, off, buf) = operand_of!(a, $variant);
            argsort_along(s, st, off, buf, &out_strides, &mut out, axis, stable, $lt);
        }};
    }
    // numpy's `LT` per dtype family: plain `<` for integers and bools;
    // `a < b || (b is NaN && a is not)` for floats, i.e. NaN sorts last
    // without the signbit tiebreak `sort_axis` applies to VALUES (that
    // tiebreak decides which of several equal-comparing values is emitted,
    // a question the permutation does not ask); and the same
    // reverse-engineered complex relation, reached through `complex_cmp`'s
    // `Ordering` rather than duplicated.
    //
    // `descending` is a comparator change, NOT an output reversal -- see
    // `complex_cmp`'s CORRECTION note for the measurements that rule the
    // reversal out. Here it means flipping `<` to `>` while leaving the
    // NaN clause alone, which keeps NaN last in both directions.
    let d = descending;
    fn float_lt<T: PartialOrd + Copy>(a: &T, b: &T, desc: bool, is_nan: impl Fn(T) -> bool) -> bool {
        let ord = if desc { a > b } else { a < b };
        ord || (is_nan(*b) && !is_nan(*a))
    }
    match a.dtype() {
        DType::Bool => arm!(Bool, |x: &bool, y: &bool| if d { *x && !*y } else { !*x && *y }),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => arm!(I8, |x: &i8, y: &i8| if d { x > y } else { x < y }),
        DType::I16 => arm!(I16, |x: &i16, y: &i16| if d { x > y } else { x < y }),
        DType::I32 => arm!(I32, |x: &i32, y: &i32| if d { x > y } else { x < y }),
        DType::I64 => arm!(I64, |x: &i64, y: &i64| if d { x > y } else { x < y }),
        DType::U8 => arm!(U8, |x: &u8, y: &u8| if d { x > y } else { x < y }),
        DType::U16 => arm!(U16, |x: &u16, y: &u16| if d { x > y } else { x < y }),
        DType::U32 => arm!(U32, |x: &u32, y: &u32| if d { x > y } else { x < y }),
        DType::U64 => arm!(U64, |x: &u64, y: &u64| if d { x > y } else { x < y }),
        DType::F16 => arm!(F16, |x: &f16, y: &f16| float_lt(x, y, d, |v: f16| v.is_nan())),
        DType::F32 => arm!(F32, |x: &f32, y: &f32| float_lt(x, y, d, |v: f32| v.is_nan())),
        DType::F64 => arm!(F64, |x: &f64, y: &f64| float_lt(x, y, d, |v: f64| v.is_nan())),
        DType::C64 => arm!(C64, |x: &C64, y: &C64| {
            complex_cmp(x, y, |v: f32| v.is_nan(), d) == Ordering::Less
        }),
        DType::C128 => arm!(C128, |x: &C128, y: &C128| {
            complex_cmp(x, y, |v: f64| v.is_nan(), d) == Ordering::Less
        }),
    }
    NdArray::from_buffer(Buffer::I64(out), shape, ArrOrder::C)
}

fn reverse_axis_lines<T: Copy>(shape: &[usize], strides: &[isize], buf: &mut [T], axis: usize) {
    let ndim = shape.len();
    let axis_len = shape[axis];
    let axis_stride = strides[axis];
    let keep_axes: Vec<usize> = (0..ndim).filter(|&a| a != axis).collect();
    if keep_axes.is_empty() {
        for i in 0..axis_len / 2 {
            let lo = (i as isize * axis_stride) as usize;
            let hi = ((axis_len - 1 - i) as isize * axis_stride) as usize;
            buf.swap(lo, hi);
        }
        return;
    }
    let keep_shape: Vec<usize> = keep_axes.iter().map(|&a| shape[a]).collect();
    let keep_strides: Vec<isize> = keep_axes.iter().map(|&a| strides[a]).collect();
    for base in NdIter::new(&keep_shape, &keep_strides) {
        for i in 0..axis_len / 2 {
            let lo = (base + i as isize * axis_stride) as usize;
            let hi = (base + (axis_len - 1 - i) as isize * axis_stride) as usize;
            buf.swap(lo, hi);
        }
    }
}

/// `ndarray.sort` -- true in-place mutation (copy-on-write via
/// `Arc::make_mut`, same established pattern as `ufunc::write_out`).
/// Values are reordered within the array's OWN existing strides/offset;
/// shape and memory layout are untouched, only which value lives at each
/// existing slot changes. Returns `()`: numpy's `ndarray.sort()` returns
/// `None`, the `ionp-py` wrapper maps that directly.
pub fn sort_in_place(a: &mut NdArray, axis: usize, stable: bool, descending: bool) -> Result<(), IonpError> {
    let shape = a.shape.to_vec();
    let strides = a.strides.to_vec();
    let offset = a.offset;

    macro_rules! ord_arm {
        ($variant:ident) => {{
            let buffer = Arc::make_mut(&mut a.buffer);
            if let Buffer::$variant(buf) = buffer {
                let snapshot = buf.clone();
                sort_along(&shape, &strides, offset, &snapshot, &strides, offset, buf, axis, int_cmp);
                if descending {
                    reverse_axis_lines(&shape, &strides, buf, axis);
                }
            }
        }};
    }
    macro_rules! bool_arm {
        () => {{
            let buffer = Arc::make_mut(&mut a.buffer);
            if let Buffer::Bool(buf) = buffer {
                let snapshot = buf.clone();
                sort_along(&shape, &strides, offset, &snapshot, &strides, offset, buf, axis, |x: &bool, y: &bool| x.cmp(y));
                if descending {
                    reverse_axis_lines(&shape, &strides, buf, axis);
                }
            }
        }};
    }
    macro_rules! float_arm {
        ($variant:ident, $t:ty, $cmp:expr, $isnan:expr, $canonval:expr) => {{
            let buffer = Arc::make_mut(&mut a.buffer);
            if let Buffer::$variant(buf) = buffer {
                let snapshot = buf.clone();
                let cmp = |x: &$t, y: &$t| $cmp(*x, *y, !stable, descending);
                sort_along(&shape, &strides, offset, &snapshot, &strides, offset, buf, axis, cmp);
                // Same per-line-length gate as `sort_axis`'s float_arm --
                // see its doc comment for the empirical justification
                // (a length<=1 line is never touched by real numpy's
                // canonicalization, verified directly).
                if !stable && shape[axis] > 1 {
                    canonicalize_axis_lines(&shape, &strides, offset, buf, axis, $isnan, $canonval);
                }
            }
        }};
    }
    macro_rules! complex_arm {
        ($variant:ident, $t:ty, $part:ty, $isnan:expr) => {{
            let buffer = Arc::make_mut(&mut a.buffer);
            if let Buffer::$variant(buf) = buffer {
                let snapshot = buf.clone();
                let cmp = |x: &$t, y: &$t| complex_cmp(x, y, $isnan, descending);
                sort_along(&shape, &strides, offset, &snapshot, &strides, offset, buf, axis, cmp);
            }
        }};
    }

    match a.dtype() {
        DType::Bool => bool_arm!(),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => ord_arm!(I8),
        DType::I16 => ord_arm!(I16),
        DType::I32 => ord_arm!(I32),
        DType::I64 => ord_arm!(I64),
        DType::U8 => ord_arm!(U8),
        DType::U16 => ord_arm!(U16),
        DType::U32 => ord_arm!(U32),
        DType::U64 => ord_arm!(U64),
        DType::F16 => float_arm!(F16, f16, f16_cmp, |v: f16| v.is_nan(), f16::from_bits(0x7fff)),
        DType::F32 => float_arm!(F32, f32, f32_cmp, |v: f32| v.is_nan(), f32::from_bits(0x7fff_ffff)),
        DType::F64 => float_arm!(F64, f64, f64_total_cmp, |v: f64| v.is_nan(), f64::from_bits(0x7fff_ffff_ffff_ffff)),
        DType::C64 => complex_arm!(C64, C64, f32, |v: f32| v.is_nan()),
        DType::C128 => complex_arm!(C128, C128, f64, |v: f64| v.is_nan()),
    }
    Ok(())
}

/// `np.sort_complex` -- always sorts along the last axis (numpy: no
/// `axis=` parameter at all) using the same lexicographic complex
/// comparator, and preserves the INPUT dtype if already complex
/// (verified against real numpy 2.5.1: `complex64` input stays
/// `complex64`, contradicting older docs claiming an unconditional
/// promotion to `complex128`); non-complex input is cast up to the
/// matching complex dtype first (real numpy: `float32`/`float16` ->
/// `complex64`, everything else real -> `complex128`), matching numpy's
/// own promotion table for this function.
pub fn sort_complex(a: &NdArray) -> Result<NdArray, IonpError> {
    // numpy's real rule (numpy/lib/_function_base_impl.py sort_complex,
    // verified directly against numpy 2.5.1 both by probing every dtype
    // AND by reading the actual source):
    //
    //   def sort_complex(a):
    //       b = array(a, copy=True)
    //       b.sort()
    //       if not issubclass(b.dtype.type, complexfloating):
    //           if b.dtype.char in 'bhBH':
    //               return b.astype('F')
    //           else:
    //               return b.astype('D')
    //       else:
    //           return b
    //
    // Two consequences that are easy to get wrong (this function used to
    // get BOTH wrong, caught by the differential corpus's
    // `special/finite_nan_inf` cases showing a `-0.0`/`0.0` tie-order
    // mismatch even after the dtype-promotion table itself was already
    // fixed):
    //
    //  1. ORDER OF OPERATIONS: numpy sorts `b` while it is STILL THE
    //     ORIGINAL (possibly real) dtype, and casts to complex only
    //     AFTERWARDS. Casting to complex first and then sorting (the
    //     previous ionp implementation) runs the wrong comparator
    //     entirely for non-complex input -- `complex_cmp`'s tie-breaking
    //     for a real value promoted to `x+0j` is not obligated to (and
    //     empirically does not) match plain real-typed `f64_total_cmp`'s
    //     signbit tiebreak, since complex ties are resolved with no
    //     signbit rule at all (see `complex_cmp`'s doc comment) while
    //     real float ties (under the default kind) resolve `-0.0` before
    //     `0.0`.
    //  2. KIND: `b.sort()` uses numpy's DEFAULT kind (quicksort), not
    //     `stable`/mergesort -- so the presort here must use
    //     `signbit_tiebreak = true` (`stable=false` in this crate's
    //     `sort_axis` signature), not `stable=true` as the previous
    //     implementation incorrectly hardcoded.
    let ndim = a.ndim();
    if ndim == 0 {
        return Err(IonpError::AxisError { axis: -1, ndim: Some(0) });
    }
    let sorted = sort_axis(a, ndim - 1, false, false)?;
    // Promotion table (unchanged from the earlier fix, still verified
    // correct on its own): 'bhBH' (int8/int16/uint8/uint16) -> complex64;
    // every other non-complex source -> complex128; already-complex input
    // keeps its own width. `cast_to` on an already-matching dtype is a
    // cheap no-op copy, not a behavior change.
    let target_dtype = match sorted.dtype() {
        DType::C64 | DType::C128 => sorted.dtype(),
        DType::I8 | DType::I16 | DType::U8 | DType::U16 => DType::C64,
        _ => DType::C128,
    };
    Ok(sorted.cast_to(target_dtype))
}

/// Decode a flat row-major index (`0..size_of_shape(shape)`) into its own
/// per-axis multi-index. Used by `lexsort` to walk every "line" (every
/// combination of the non-sorted axes) once, independently of any single
/// key's own strides -- each key supplies its OWN strides/offset for a
/// given multi-index via `element_cmp_fn_at`, so this only needs to agree
/// on iteration ORDER across keys, not on any one key's memory layout.
fn nd_index_iter(shape: &[usize]) -> impl Iterator<Item = Vec<usize>> + '_ {
    let total = size_of_shape(shape);
    let ndim = shape.len();
    (0..total).map(move |flat| {
        let mut rem = flat;
        let mut idx = vec![0usize; ndim];
        for d in (0..ndim).rev() {
            let dim = shape[d].max(1);
            idx[d] = rem % dim;
            rem /= dim;
        }
        idx
    })
}

/// `np.lexsort` -- ALWAYS stable, for every dtype (verified: single-key
/// `np.lexsort((a,))` exactly equals `np.argsort(a, kind='stable')`,
/// unlike plain `argsort`'s default kind). Sorts primarily by the LAST
/// key in `keys` (numpy: keys\[-1\] is primary), ties broken by
/// progressively earlier keys, final ties broken by original index
/// (stability). Returns an `I64` index array (numpy's `intp`) of the SAME
/// SHAPE as the keys (not flattened) -- numpy sorts independently along
/// `axis` (default -1) for every combination of the other axes, verified
/// directly: `np.lexsort((a, b))` for `(2, 3)`-shaped `a`/`b` returns a
/// `(2, 3)` result, each ROW sorted independently (not a single 1-D
/// permutation of all 6 elements). Every key must share the SAME shape
/// (numpy's own `lexsort` requires this too for anything beyond trivial
/// broadcasting; matching shapes is this crate's scope, same as the rest
/// of the sort/search family).
///
/// BUG FOUND + FIXED (2026-08-01, coordinator's harder probe): the
/// previous implementation only accepted 1-D keys at all, unconditionally
/// -- `np.lexsort((a, b))` for `a`/`b` of shape `(2, 3)`, `(1, 4)`,
/// `(4, 1)`, with either `axis=0` or `axis=-1`, all genuinely work in real
/// numpy and were previously rejected outright.
pub fn lexsort(keys: &[NdArray], axis: isize) -> Result<NdArray, IonpError> {
    if keys.is_empty() {
        return Err(IonpError::Value("need at least one array to lexsort".to_string()));
    }
    let shape = keys[0].shape().to_vec();
    let ndim = shape.len();
    for k in keys {
        if k.shape() != shape.as_slice() {
            return Err(IonpError::Value("lexsort keys must all have the same shape".to_string()));
        }
    }
    if ndim == 0 {
        // Mirrors `sort`'s own axis family: a 0-d array has no axis to
        // sort along at all (not the reduce family's courtesy no-op).
        return Err(IonpError::AxisError { axis, ndim: Some(0) });
    }
    let norm_axis = normalize_single_axis(axis, ndim)?;
    let axis_len = shape[norm_axis];
    let out_strides = c_strides(&shape);
    let keep_axes: Vec<usize> = (0..ndim).filter(|&a| a != norm_axis).collect();
    let keep_shape: Vec<usize> = keep_axes.iter().map(|&a| shape[a]).collect();

    let mut out: Vec<i64> = vec![0; size_of_shape(&shape)];
    for kidx in nd_index_iter(&keep_shape) {
        let out_base: isize = keep_axes.iter().zip(kidx.iter()).map(|(&a, &i)| i as isize * out_strides[a]).sum();
        let mut idx: Vec<i64> = (0..axis_len as i64).collect();
        // Stable sort by, in order, key[0], key[1], ... using Rust's
        // stable `sort_by` repeatedly (last sort = highest priority)
        // reproduces a full lexicographic-with-stability ordering keyed
        // primary-last, exactly matching numpy's documented "last key is
        // primary" contract -- equivalent to, and simpler than, a single
        // N-ary comparator.
        for k in keys.iter() {
            let key_base: isize = k.offset() + keep_axes.iter().zip(kidx.iter()).map(|(&a, &i)| i as isize * k.strides()[a]).sum::<isize>();
            let cmp_at = element_cmp_fn_at(k, k.strides()[norm_axis], key_base, true);
            idx.sort_by(|&a, &b| cmp_at(a as usize, b as usize));
        }
        for (i, v) in idx.iter().enumerate() {
            out[(out_base + i as isize * out_strides[norm_axis]) as usize] = *v;
        }
    }
    NdArray::from_buffer(Buffer::I64(out), shape, ArrOrder::C)
}

/// Builds an `(usize, usize) -> Ordering` comparator over positions ALONG
/// ONE AXIS of `a` (0-based logical index along that axis, not raw buffer
/// offset), dispatching per-dtype once and reusing the total-order rules
/// above (`stable` bool has the same meaning as `sort_axis`'s). `strides`
/// is `a`'s own stride for the axis being compared along; `base` is the
/// buffer offset of index-0 on that axis for the particular line being
/// compared (i.e. `a`'s own offset PLUS that line's own contribution from
/// every other, non-compared axis) -- letting `lexsort` reuse this once
/// per (key, line) pair for both the old 1-D-only shape (`base =
/// a.offset()`, `strides = a.strides()[0]`) and the current N-D-per-line
/// shape (`base` varies per line, `strides = a.strides()[axis]`) with no
/// separate code path.
fn element_cmp_fn_at(a: &NdArray, strides: isize, base: isize, stable: bool) -> Box<dyn Fn(usize, usize) -> Ordering + '_> {
    let offset = base;
    macro_rules! ord_box {
        ($variant:ident) => {{
            let (_, _, _, buf) = operand_of!(a, $variant);
            Box::new(move |i: usize, j: usize| {
                let oi = (offset + i as isize * strides) as usize;
                let oj = (offset + j as isize * strides) as usize;
                buf[oi].cmp(&buf[oj])
            })
        }};
    }
    match a.dtype() {
        DType::Bool => ord_box!(Bool),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => ord_box!(I8),
        DType::I16 => ord_box!(I16),
        DType::I32 => ord_box!(I32),
        DType::I64 => ord_box!(I64),
        DType::U8 => ord_box!(U8),
        DType::U16 => ord_box!(U16),
        DType::U32 => ord_box!(U32),
        DType::U64 => ord_box!(U64),
        DType::F16 => {
            let (_, _, _, buf) = operand_of!(a, F16);
            Box::new(move |i: usize, j: usize| {
                let oi = (offset + i as isize * strides) as usize;
                let oj = (offset + j as isize * strides) as usize;
                f16_cmp(buf[oi], buf[oj], !stable, false)
            })
        }
        DType::F32 => {
            let (_, _, _, buf) = operand_of!(a, F32);
            Box::new(move |i: usize, j: usize| {
                let oi = (offset + i as isize * strides) as usize;
                let oj = (offset + j as isize * strides) as usize;
                f32_cmp(buf[oi], buf[oj], !stable, false)
            })
        }
        DType::F64 => {
            let (_, _, _, buf) = operand_of!(a, F64);
            Box::new(move |i: usize, j: usize| {
                let oi = (offset + i as isize * strides) as usize;
                let oj = (offset + j as isize * strides) as usize;
                f64_total_cmp(buf[oi], buf[oj], !stable, false)
            })
        }
        DType::C64 => {
            let (_, _, _, buf) = operand_of!(a, C64);
            Box::new(move |i: usize, j: usize| {
                let oi = (offset + i as isize * strides) as usize;
                let oj = (offset + j as isize * strides) as usize;
                complex_cmp(&buf[oi], &buf[oj], |v: f32| v.is_nan(), false)
            })
        }
        DType::C128 => {
            let (_, _, _, buf) = operand_of!(a, C128);
            Box::new(move |i: usize, j: usize| {
                let oi = (offset + i as isize * strides) as usize;
                let oj = (offset + j as isize * strides) as usize;
                complex_cmp(&buf[oi], &buf[oj], |v: f64| v.is_nan(), false)
            })
        }
    }
}

// ---------------------------------------------------------------------------
// nonzero / flatnonzero / argwhere / extract / where
// ---------------------------------------------------------------------------

/// Truthiness test shared by `nonzero`/`where`/`extract`/`count_nonzero`'s
/// family: `!= 0` for numeric dtypes, `!= 0+0j` for complex (either
/// component nonzero), the value itself for bool. NaN is nonzero (`!= 0`
/// is true for NaN under IEEE comparison... but numpy's nonzero treats
/// NaN as truthy too, verified: `np.nonzero(np.array([np.nan]))` finds
/// it) -- direct `!= 0.0` on a NaN payload correctly returns `true` in
/// Rust (NaN != anything, including 0.0), so no special case is needed.
fn is_truthy_mask(a: &NdArray) -> Result<Vec<bool>, IonpError> {
    let flat = a.to_contiguous();
    macro_rules! arm {
        ($variant:ident, $zero:expr) => {{
            match flat.buffer() {
                Buffer::$variant(v) => v.iter().map(|&x| x != $zero).collect(),
                _ => unreachable!(),
            }
        }};
    }
    Ok(match flat.dtype() {
        DType::Bool => match flat.buffer() {
            Buffer::Bool(v) => v.clone(),
            _ => unreachable!(),
        },
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => arm!(I8, 0),
        DType::I16 => arm!(I16, 0),
        DType::I32 => arm!(I32, 0),
        DType::I64 => arm!(I64, 0),
        DType::U8 => arm!(U8, 0),
        DType::U16 => arm!(U16, 0),
        DType::U32 => arm!(U32, 0),
        DType::U64 => arm!(U64, 0),
        DType::F16 => match flat.buffer() {
            Buffer::F16(v) => v.iter().map(|&x| x.to_f64() != 0.0).collect(),
            _ => unreachable!(),
        },
        DType::F32 => arm!(F32, 0.0f32),
        DType::F64 => arm!(F64, 0.0f64),
        DType::C64 => match flat.buffer() {
            Buffer::C64(v) => v.iter().map(|&z| z.re != 0.0 || z.im != 0.0).collect(),
            _ => unreachable!(),
        },
        DType::C128 => match flat.buffer() {
            Buffer::C128(v) => v.iter().map(|&z| z.re != 0.0 || z.im != 0.0).collect(),
            _ => unreachable!(),
        },
    })
}

/// `np.nonzero` -- one 1-D `I64` index array per dimension, in C
/// (row-major) discovery order. 0-d input is rejected with `ValueError`
/// (verified against real numpy 2.5.1: "Calling nonzero on 0d arrays is
/// not allowed").
///
/// BUG FOUND + FIXED (2026-08-01, argmax/argmin/nonzero task): the 0-d
/// rejection message is NOT the same for every dtype -- probed directly
/// against real numpy 2.5.1: a 0-d BOOL array gets an extra trailing
/// sentence real numpy only appends for bool (`np.nonzero(np.array(True))`
/// -> 177 chars, ending "...instead. If the context of this error is of
/// the form `arr[nonzero(cond)]`, just use `arr[cond]`."), while every
/// other dtype (float/int/etc, e.g. `np.nonzero(np.array(3.0))`) gets the
/// short 89-char form with no such sentence. This makes sense given
/// `nonzero(cond)` is overwhelmingly used on bool masks in real code --
/// numpy's hint is bool-specific, not a generic nonzero hint. Both strings
/// verified byte-for-byte (`repr(str(e))`/`len(str(e))`) against live
/// numpy, not transcribed from memory.
pub fn nonzero(a: &NdArray) -> Result<Vec<NdArray>, IonpError> {
    if a.ndim() == 0 {
        let msg = if a.dtype() == DType::Bool {
            "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead. If the context of this error is of the form `arr[nonzero(cond)]`, just use `arr[cond]`."
        } else {
            "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead."
        };
        return Err(IonpError::Value(msg.to_string()));
    }
    let shape = a.shape().to_vec();
    let mask = is_truthy_mask(a)?;
    let ndim = shape.len();
    let mut cols: Vec<Vec<i64>> = vec![Vec::new(); ndim];
    let strides = c_strides(&shape);
    for (flat, &t) in mask.iter().enumerate() {
        if !t {
            continue;
        }
        let mut rem = flat as isize;
        for d in 0..ndim {
            let s = strides[d];
            let coord = if s == 0 { 0 } else { rem / s };
            cols[d].push(coord as i64);
            rem -= coord * s;
        }
    }
    cols.into_iter()
        .map(|c| {
            let n = c.len();
            NdArray::from_buffer(Buffer::I64(c), vec![n], ArrOrder::C)
        })
        .collect()
}

/// `np.flatnonzero` -- `nonzero` of the flattened (C-order raveled)
/// array, single 1-D `I64` result (`np.flatnonzero(a)` is documented as
/// exactly `np.nonzero(a.ravel())[0]`).
pub fn flatnonzero(a: &NdArray) -> Result<NdArray, IonpError> {
    let flat = a.ravel_order("C")?;
    let mut result = nonzero(&flat)?;
    Ok(result.remove(0))
}

/// `np.argwhere` -- transpose of `nonzero`'s stacked result: shape
/// `(count, ndim)`, row `i` is the i-th nonzero element's full
/// coordinate. 0-d input is explicitly ALLOWED here (unlike `nonzero`,
/// verified against real numpy 2.5.1: `np.argwhere(np.array(5))` returns
/// `array([], shape=(1, 0), dtype=int64)`, a single all-empty-coordinate
/// row since a 0-d array has exactly one truthy-or-not element and zero
/// axes to report a coordinate along).
pub fn argwhere(a: &NdArray) -> Result<NdArray, IonpError> {
    if a.ndim() == 0 {
        let truthy = is_truthy_mask(a)?[0];
        let rows = if truthy { 1 } else { 0 };
        return NdArray::from_buffer(Buffer::I64(vec![]), vec![rows, 0], ArrOrder::C);
    }
    let cols = nonzero(a)?;
    let ndim = cols.len();
    let count = cols.first().map(|c| c.shape()[0]).unwrap_or(0);
    let mut data = Vec::with_capacity(count * ndim);
    let col_bufs: Vec<&[i64]> = cols
        .iter()
        .map(|c| match c.buffer() {
            Buffer::I64(v) => v.as_slice(),
            _ => unreachable!(),
        })
        .collect();
    for row in 0..count {
        for col in col_bufs.iter() {
            data.push(col[row]);
        }
    }
    NdArray::from_buffer(Buffer::I64(data), vec![count, ndim], ArrOrder::C)
}

/// `np.extract` -- `arr.ravel()[condition.ravel().astype(bool)]`, i.e.
/// the flattened elements of `arr` (broadcast against `condition`'s
/// shape if they differ, matching real numpy's own broadcasting here)
/// selected where the correspondingly-flattened `condition` is truthy,
/// in C order. Returns a 1-D array of `arr`'s own dtype.
pub fn extract(condition: &NdArray, arr: &NdArray) -> Result<NdArray, IonpError> {
    let out_shape = crate::shape::broadcast_shapes(condition.shape(), arr.shape())?;
    let cond_strides = crate::shape::broadcast_strides_to(condition.shape(), condition.strides(), &out_shape)?;
    let arr_strides = crate::shape::broadcast_strides_to(arr.shape(), arr.strides(), &out_shape)?;
    let cond_view = broadcast_view(condition, out_shape.clone(), cond_strides);
    let mask = is_truthy_mask(&cond_view)?;
    let arr_view = broadcast_view(arr, out_shape.clone(), arr_strides);
    let arr_flat = arr_view.to_contiguous();

    macro_rules! sel {
        ($variant:ident) => {{
            match arr_flat.buffer() {
                Buffer::$variant(v) => {
                    let out: Vec<_> = v.iter().zip(mask.iter()).filter(|(_, &m)| m).map(|(x, _)| *x).collect();
                    let n = out.len();
                    Buffer::$variant(out).pipe(n)
                }
                _ => unreachable!(),
            }
        }};
    }
    trait Pipe {
        fn pipe(self, n: usize) -> (Buffer, usize);
    }
    impl Pipe for Buffer {
        fn pipe(self, n: usize) -> (Buffer, usize) {
            (self, n)
        }
    }

    let (buffer, n) = match arr_flat.dtype() {
        DType::Bool => sel!(Bool),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => sel!(I8),
        DType::I16 => sel!(I16),
        DType::I32 => sel!(I32),
        DType::I64 => sel!(I64),
        DType::U8 => sel!(U8),
        DType::U16 => sel!(U16),
        DType::U32 => sel!(U32),
        DType::U64 => sel!(U64),
        DType::F16 => sel!(F16),
        DType::F32 => sel!(F32),
        DType::F64 => sel!(F64),
        DType::C64 => sel!(C64),
        DType::C128 => sel!(C128),
    };
    NdArray::from_buffer(buffer, vec![n], ArrOrder::C)
}

// ---------------------------------------------------------------------------
// searchsorted
// ---------------------------------------------------------------------------

/// `np.searchsorted` -- binary search over a 1-D SORTED array
/// (ascending; behavior on an unsorted input is unspecified by numpy
/// itself and out of scope here). `right=false` is `side='left'`
/// (leftmost valid insertion index, numpy default), `right=true` is
/// `side='right'`. Comparator matches `sort`'s own 'stable' total order
/// (no signbit tiebreak -- searchsorted has no `kind=` parameter at all
/// in real numpy, and empirically its NaN placement matches treating NaN
/// as greater than everything, consistent with where `sort` puts NaN).
/// Output is an `I64` array (numpy's `intp`) shaped like `values`.
///
/// `sorter`, if given, is a 1-D integer index array such that
/// `sorted.take(sorter)` is itself ascending -- `sorted` need not already
/// be sorted in that case (numpy: "Array of integer indices that sort
/// array a into ascending order"). The returned index is into that
/// permutation's own order (numpy's documented contract:
/// `a[sorter[i-1]] <= v <= a[sorter[i]]`).
///
/// BUG FOUND + FIXED (2026-08-04): this used to MATERIALIZE the whole
/// permutation up front (`sorted[sorter[i]]` for every `i`, into a fresh
/// contiguous array) and bisect over that. Three observable divergences
/// came out of that one shortcut, all measured against numpy 2.5.1 with
/// `a = np.array([3, 1, 2])`:
///
///   * an out-of-range sorter entry was reported EAGERLY, and with the
///     wrong exception: ionp raised `IndexError: index 5 is out of bounds
///     for axis 0 with size 3` for every input, where numpy raises
///     `ValueError: Sorter index out of range.` -- and only if the
///     bisection actually DEREFERENCES that entry. numpy's check lives
///     inside the search loop, so it is genuinely lazy:
///         np.searchsorted(np.array([True, False, True]), 2,
///                         sorter=np.array([5, 0, 1]))          -> 3
///     (three elements, bisect touches indices 1 then 2, never 0, so the
///     bad entry at position 0 is never seen and no error is raised),
///     while the same sorter over `np.array([3, 1, 2])` DOES touch it and
///     raises. Eager validation cannot reproduce that, so the gather had
///     to go; the index is now resolved and bounds-checked inside
///     `bisect`, which also removes an O(n) copy from a search that is
///     supposed to be O(log n).
///   * a NEGATIVE sorter entry was silently accepted and wrapped
///     Python-style (`raw + n`). numpy does not wrap -- `sorter=
///     np.array([-1, 0, 1])` raises `Sorter index out of range.` too.
///   * the argument-shape errors were one home-grown message instead of
///     numpy's three distinct ones. Measured precedence, which is what
///     the ordering below encodes: ndim first, then dtype, then size --
///     a 2-D FLOAT sorter of the wrong length reports `could not parse
///     sorter argument`, not the dtype or size complaint.
///         sorter=np.array([[0, 1, 2]])   -> TypeError:  could not parse sorter argument
///         sorter=np.array([0., 1., 2.])  -> TypeError:  sorter must only contain integers
///         sorter=np.array([True, False, True])
///                                        -> TypeError:  sorter must only contain integers
///         sorter=np.array([0, 1])        -> ValueError: sorter.size must equal a.size
///     Unsigned integer sorters are accepted (`np.uint8`), booleans are
///     not -- numpy's `sorter must only contain integers` treats `bool`
///     as a non-integer here even though it is an integer everywhere
///     else in the promotion lattice.
pub fn searchsorted(sorted: &NdArray, values: &NdArray, right: bool, sorter: Option<&NdArray>) -> Result<NdArray, IonpError> {
    if sorted.ndim() != 1 {
        return Err(IonpError::Value("searchsorted requires a 1-D sorted array".to_string()));
    }
    let n = sorted.shape()[0];
    // Owned holder for the (possibly cast/copied) sorter buffer; the borrow
    // below points into it, so it has to outlive the search. Same shape as
    // the `permuted` holder this replaces.
    let sorter_i64;
    let sorter_buf: Option<&[i64]> = match sorter {
        None => None,
        Some(sorter_arr) => {
            // Order matters -- see this function's doc comment for the
            // measured precedence (ndim, then dtype, then size).
            if sorter_arr.ndim() != 1 {
                return Err(IonpError::Type("could not parse sorter argument".to_string()));
            }
            // `uint64` is the odd one out and it is NOT an oversight: numpy
            // coerces `sorter` to `intp` with SAFE casting, and `uint64 ->
            // int64` is not a safe cast, so it fails inside the coercion
            // rather than at the integer-kind check. That gives it numpy's
            // coercion-failure message under a DIFFERENT exception class
            // from the identically-worded ndim rejection above -- measured
            // 2026-08-04, `a = np.array([3, 1, 2])`:
            //     sorter=np.array([1,2,0], np.uint64)   -> ValueError: could not parse sorter argument
            //     sorter=np.array([[1,2,0]], np.uint64) -> TypeError:  could not parse sorter argument
            //     sorter=np.array([1,2],    np.uint64)  -> ValueError: could not parse sorter argument
            // i.e. ndim still wins, and the size check never runs. Every
            // other unsigned width (`uint8`/`uint16`/`uint32`) IS safely
            // castable and is accepted. This case was caught by the
            // differential corpus, not by the probe that preceded it -- the
            // probe only tried `np.uint8` and read "unsigned is fine".
            if sorter_arr.dtype() == DType::U64 {
                return Err(IonpError::Value("could not parse sorter argument".to_string()));
            }
            if !matches!(
                sorter_arr.dtype(),
                DType::I8 | DType::I16 | DType::I32 | DType::I64
                    | DType::U8 | DType::U16 | DType::U32
            ) {
                return Err(IonpError::Type("sorter must only contain integers".to_string()));
            }
            if sorter_arr.shape()[0] != n {
                return Err(IonpError::Value("sorter.size must equal a.size".to_string()));
            }
            sorter_i64 = sorter_arr.cast_to(DType::I64).to_contiguous();
            match sorter_i64.buffer() {
                Buffer::I64(v) => Some(v.as_slice()),
                _ => unreachable!(),
            }
        }
    };
    let bisect = |needle_cmp: &dyn Fn(usize) -> Ordering| -> Result<i64, IonpError> {
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid0 = (lo + hi) / 2;
            // Resolve through `sorter` HERE, not up front -- numpy's own
            // bounds check is inside its search loop, and an out-of-range
            // entry the bisection never touches is never reported.
            let mid = match sorter_buf {
                None => mid0,
                Some(sb) => {
                    let raw = sb[mid0];
                    if raw < 0 || raw >= n as i64 {
                        return Err(IonpError::Value("Sorter index out of range.".to_string()));
                    }
                    raw as usize
                }
            };
            // `ord` is `sorted[mid].cmp(&needle)`. bisect_left: shrink hi
            // when sorted[mid] >= needle (ord != Less). bisect_right: shrink
            // hi when sorted[mid] > needle (ord == Greater). The previous
            // version of this comparison was inverted on both branches
            // (verified against real numpy: `searchsorted([1,3,5],
            // [0,2,4,6])` returned `[3,3,0,0]` instead of `[0,1,2,3]`).
            let ord = needle_cmp(mid);
            let go_left = if right { ord == Ordering::Greater } else { ord != Ordering::Less };
            // NB: the loop bounds move by the UNPERMUTED position `mid0`;
            // only the comparison reads through the sorter. Using `mid`
            // here would search the sorter's values, not the array's.
            if go_left {
                hi = mid0;
            } else {
                lo = mid0 + 1;
            }
        }
        Ok(lo as i64)
    };

    let values_flat = values.to_contiguous();
    let out_shape = values.shape().to_vec();

    macro_rules! run {
        ($variant:ident, $needle_cmp:expr) => {{
            match values_flat.buffer() {
                Buffer::$variant(vs) => {
                    let out: Vec<i64> = vs
                        .iter()
                        .map(|nv| bisect(&|mid: usize| ($needle_cmp)(sorted, mid, *nv)))
                        .collect::<Result<Vec<i64>, IonpError>>()?;
                    out
                }
                _ => return Err(IonpError::Type("searchsorted: value dtype mismatch".to_string())),
            }
        }};
    }

    let out: Vec<i64> = match sorted.dtype() {
        DType::Bool => run!(Bool, |s: &NdArray, i: usize, v: bool| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() {
                Buffer::Bool(buf) => buf[off as usize].cmp(&v),
                _ => unreachable!(),
            }
        }),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => run!(I8, |s: &NdArray, i: usize, v: i8| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() { Buffer::I8(buf) => buf[off as usize].cmp(&v), _ => unreachable!() }
        }),
        DType::I16 => run!(I16, |s: &NdArray, i: usize, v: i16| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() { Buffer::I16(buf) => buf[off as usize].cmp(&v), _ => unreachable!() }
        }),
        DType::I32 => run!(I32, |s: &NdArray, i: usize, v: i32| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() { Buffer::I32(buf) => buf[off as usize].cmp(&v), _ => unreachable!() }
        }),
        DType::I64 => run!(I64, |s: &NdArray, i: usize, v: i64| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() { Buffer::I64(buf) => buf[off as usize].cmp(&v), _ => unreachable!() }
        }),
        DType::U8 => run!(U8, |s: &NdArray, i: usize, v: u8| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() { Buffer::U8(buf) => buf[off as usize].cmp(&v), _ => unreachable!() }
        }),
        DType::U16 => run!(U16, |s: &NdArray, i: usize, v: u16| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() { Buffer::U16(buf) => buf[off as usize].cmp(&v), _ => unreachable!() }
        }),
        DType::U32 => run!(U32, |s: &NdArray, i: usize, v: u32| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() { Buffer::U32(buf) => buf[off as usize].cmp(&v), _ => unreachable!() }
        }),
        DType::U64 => run!(U64, |s: &NdArray, i: usize, v: u64| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() { Buffer::U64(buf) => buf[off as usize].cmp(&v), _ => unreachable!() }
        }),
        DType::F16 => run!(F16, |s: &NdArray, i: usize, v: f16| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() { Buffer::F16(buf) => f16_cmp(buf[off as usize], v, false, false), _ => unreachable!() }
        }),
        DType::F32 => run!(F32, |s: &NdArray, i: usize, v: f32| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() { Buffer::F32(buf) => f32_cmp(buf[off as usize], v, false, false), _ => unreachable!() }
        }),
        DType::F64 => run!(F64, |s: &NdArray, i: usize, v: f64| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() { Buffer::F64(buf) => f64_total_cmp(buf[off as usize], v, false, false), _ => unreachable!() }
        }),
        // BUG FOUND + FIXED (2026-08-01, coordinator fuzzing): this used
        // to unconditionally reject complex input, but real numpy 2.5.1
        // handles it fine, using the exact same total order its `sort`
        // uses (verified empirically: `np.searchsorted` on a
        // `np.sort`-ed complex array agrees with `complex_cmp`'s
        // real-then-imaginary rule, including NaN sorting last) --
        // `complex_cmp` was already implemented, verified, and reused by
        // `sort`/`sort_complex` earlier this task; there was no reason
        // for `searchsorted` to special-case reject it, this was simply
        // never wired up.
        DType::C64 => run!(C64, |s: &NdArray, i: usize, v: C64| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() { Buffer::C64(buf) => complex_cmp(&buf[off as usize], &v, |x: f32| x.is_nan(), false), _ => unreachable!() }
        }),
        DType::C128 => run!(C128, |s: &NdArray, i: usize, v: C128| -> Ordering {
            let off = s.offset() + i as isize * s.strides()[0];
            match s.buffer() { Buffer::C128(buf) => complex_cmp(&buf[off as usize], &v, |x: f64| x.is_nan(), false), _ => unreachable!() }
        }),
    };
    NdArray::from_buffer(Buffer::I64(out), out_shape, ArrOrder::C)
}

// ---------------------------------------------------------------------------
// nanargmax / nanargmin
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanExtremeOp {
    Min,
    Max,
}

/// `np.nanargmax`/`np.nanargmin`.
///
/// BUG FOUND + FIXED (sort/search task, 2026-08-01): this used to literally
/// skip NaN entries (never considered as a candidate at all), which sounds
/// right but is NOT what real numpy does, and diverges from it whenever an
/// actual `-inf` (for max) / `+inf` (for min) is present alongside NaNs in
/// the same reduced group. Read numpy's actual source
/// (`numpy/lib/_nanfunctions_impl.py` `_nanargmax`/`_nanargmin`): it replaces
/// every NaN with `-inf` (max) / `+inf` (min) via `_replace_nan`, then calls
/// the ORDINARY `argmax`/`argmin` (whose tie-break is "first occurrence
/// wins"), and only afterwards checks whether an entire reduced group was
/// all-NaN (via a mask) to raise `ValueError`. The difference is observable:
/// verified against real numpy 2.5.1, `np.nanargmax([nan, nan, -inf])`
/// returns `0`, NOT `2` -- the filled-in `-inf` at index 0 ties with the
/// real `-inf` at index 2, and standard argmax's first-occurrence tie-break
/// picks index 0. A naive "skip NaN, keep best of what's left" (the
/// previous implementation here) wrongly returned `2`, since it never let
/// the NaN positions become candidates at all. Fixed by mirroring numpy's
/// actual algorithm: substitute an extreme sentinel for NaN as the
/// comparison value (never returned, only compared), keep the existing
/// first-occurrence tie-break unchanged, and separately track whether a
/// group was all-NaN (all substituted, zero genuine values seen) to raise
/// `ValueError: All-NaN slice encountered` (verified exception type/message
/// against real numpy 2.5.1). `axis=None` flattens first, same convention
/// as `argmin`/`argmax`.
pub fn nanargext(op: NanExtremeOp, a: &NdArray, axis: Option<usize>) -> Result<NdArray, IonpError> {
    let (target, axes): (NdArray, Vec<usize>) = match axis {
        None => (a.ravel_order("C")?, vec![0]),
        Some(ax) => (a.clone(), vec![ax]),
    };
    let shape = target.shape().to_vec();
    let ndim = shape.len();
    let keep_axes: Vec<usize> = (0..ndim).filter(|a| !axes.contains(a)).collect();
    let keep_shape: Vec<usize> = keep_axes.iter().map(|&a| shape[a]).collect();
    // BUG FOUND + FIXED (2026-08-01, coordinator fuzzing caught this --
    // same root cause and same fix shape as `reduce_axis_argext`'s sibling
    // fix in ufunc.rs: a 0-d array's `axis=0`/`axis=-1` is a numpy-granted
    // courtesy no-op (`np.nanargmin(np.array(3.0), axis=0) == 0`), so
    // `axes` can be `[0]` here even though `shape` (`target.shape()`) is
    // `[]` -- indexing `shape[0]` on that panics. There is no real axis to
    // look up in that case, just a single-element phantom group.
    let red_shape: Vec<usize> = if shape.is_empty() { axes.iter().map(|_| 1).collect() } else { axes.iter().map(|&a| shape[a]).collect() };

    // BUG FOUND + FIXED (2026-08-01, argmax/argmin task): a genuinely
    // empty reduced group (`red_shape` product == 0, e.g. `axis=0` on a
    // `(0, 3)`-shaped array, or the whole array when flattened) used to
    // fall through into the dtype dispatch below and hit the SAME
    // "did we see any real (non-NaN) value" check the true all-NaN case
    // uses, so it wrongly raised `ValueError: All-NaN slice encountered`
    // -- an empty group is not "all NaN", there is nothing there at all.
    // Verified against real numpy 2.5.1: `np.nanargmax(np.zeros((0, 3)),
    // axis=0)` raises `ValueError: attempt to get argmax of an empty
    // sequence` (op-name-specific, exactly like plain `np.argmax` on an
    // empty reduction), never the all-NaN message. Caught here, before the
    // dtype dispatch, so every dtype arm inherits the fix instead of
    // needing its own copy.
    if red_shape.iter().product::<usize>() == 0 {
        let name = match op {
            NanExtremeOp::Max => "argmax",
            NanExtremeOp::Min => "argmin",
        };
        return Err(IonpError::Value(format!("attempt to get {name} of an empty sequence")));
    }

    macro_rules! run_float {
        ($variant:ident, $isnan:expr, $neg_inf:expr, $pos_inf:expr) => {{
            let (_s, st, off, buf) = operand_of!(target, $variant);
            let keep_strides: Vec<isize> = keep_axes.iter().map(|&a| st[a]).collect();
            let red_strides: Vec<isize> = if st.is_empty() { axes.iter().map(|_| 0).collect() } else { axes.iter().map(|&a| st[a]).collect() };
            // NaN -> sentinel (mirrors numpy's `_replace_nan`: -inf for max,
            // +inf for min), so a NaN position can still tie with, and lose
            // the first-occurrence tie-break to, a genuine -inf/+inf value
            // -- see the doc comment above for why this is NOT the same as
            // skipping NaN entries outright.
            let sentinel = match op {
                NanExtremeOp::Max => $neg_inf,
                NanExtremeOp::Min => $pos_inf,
            };
            let seq = |base: isize| -> Result<i64, IonpError> {
                let mut best: Option<(usize, _)> = None;
                let mut saw_real = false;
                for (i, o) in NdIter::new(&red_shape, &red_strides).enumerate() {
                    let raw = buf[(base + o) as usize];
                    let is_nan = $isnan(raw);
                    if !is_nan {
                        saw_real = true;
                    }
                    let v = if is_nan { sentinel } else { raw };
                    best = Some(match best {
                        None => (i, v),
                        Some((bi, bv)) => {
                            let better = match op {
                                NanExtremeOp::Min => v < bv,
                                NanExtremeOp::Max => v > bv,
                            };
                            if better { (i, v) } else { (bi, bv) }
                        }
                    });
                }
                if !saw_real {
                    return Err(IonpError::Value("All-NaN slice encountered".to_string()));
                }
                Ok(best.map(|(i, _)| i as i64).unwrap_or(0))
            };
            let mut out = Vec::new();
            if keep_axes.is_empty() {
                out.push(seq(off)?);
            } else {
                for koff in NdIter::new(&keep_shape, &keep_strides) {
                    out.push(seq(off + koff)?);
                }
            }
            out
        }};
    }

    // Complex nanargmin/nanargmax -- verified empirically against real
    // numpy 2.5.1 (coordinator-reported gap, not previously wired up
    // despite `complex_cmp` already existing): a complex value counts as
    // "NaN" for this purpose if EITHER component is NaN (`re.is_nan() ||
    // im.is_nan()`), and numpy substitutes such entries with a sentinel
    // -- `-inf-infj` for nanargmax, `+inf+infj` for nanargmin -- before
    // running its ordinary complex extreme-finding, exactly mirroring the
    // real-float `run_float!` strategy just below (including the same
    // NaN-vs-genuine-extreme first-occurrence tie behavior: probed
    // directly, `np.nanargmax([nan+nanj, complex(-inf,-inf)])` returns
    // `0`, i.e. the NaN position wins the tie against a real all-time-
    // minimum value, same as the already-verified real-dtype case).
    // `-inf-infj`/`+inf+infj` are valid non-NaN total-order extremes under
    // `complex_cmp` (neither component is NaN, so the ordinary real-then-
    // imaginary comparison applies with no special-casing needed), so
    // reusing `complex_cmp` directly for the tie-break is correct.
    macro_rules! run_complex {
        ($variant:ident, $t:ty, $part:ty, $isnan_part:expr, $neg_inf:expr, $pos_inf:expr) => {{
            let (_s, st, off, buf) = operand_of!(target, $variant);
            let keep_strides: Vec<isize> = keep_axes.iter().map(|&a| st[a]).collect();
            let red_strides: Vec<isize> = if st.is_empty() { axes.iter().map(|_| 0).collect() } else { axes.iter().map(|&a| st[a]).collect() };
            let is_nan_c = |v: $t| -> bool { $isnan_part(v.re) || $isnan_part(v.im) };
            let sentinel: $t = match op {
                NanExtremeOp::Max => $neg_inf,
                NanExtremeOp::Min => $pos_inf,
            };
            let seq = |base: isize| -> Result<i64, IonpError> {
                let mut best: Option<(usize, $t)> = None;
                let mut saw_real = false;
                for (i, o) in NdIter::new(&red_shape, &red_strides).enumerate() {
                    let raw = buf[(base + o) as usize];
                    let is_nan = is_nan_c(raw);
                    if !is_nan {
                        saw_real = true;
                    }
                    let v = if is_nan { sentinel } else { raw };
                    best = Some(match best {
                        None => (i, v),
                        Some((bi, bv)) => {
                            let cmp = complex_cmp(&v, &bv, $isnan_part, false);
                            let better = match op {
                                NanExtremeOp::Min => cmp == Ordering::Less,
                                NanExtremeOp::Max => cmp == Ordering::Greater,
                            };
                            if better {
                                (i, v)
                            } else {
                                (bi, bv)
                            }
                        }
                    });
                }
                if !saw_real {
                    return Err(IonpError::Value("All-NaN slice encountered".to_string()));
                }
                Ok(best.map(|(i, _)| i as i64).unwrap_or(0))
            };
            let mut out = Vec::new();
            if keep_axes.is_empty() {
                out.push(seq(off)?);
            } else {
                for koff in NdIter::new(&keep_shape, &keep_strides) {
                    out.push(seq(off + koff)?);
                }
            }
            out
        }};
    }

    let out: Vec<i64> = match target.dtype() {
        DType::F16 => run_float!(F16, |v: f16| v.is_nan(), f16::NEG_INFINITY, f16::INFINITY),
        DType::F32 => run_float!(F32, |v: f32| v.is_nan(), f32::NEG_INFINITY, f32::INFINITY),
        DType::F64 => run_float!(F64, |v: f64| v.is_nan(), f64::NEG_INFINITY, f64::INFINITY),
        // Non-float dtypes never contain NaN -- behave exactly like plain
        // argmin/argmax (verified: `np.nanargmax` accepts int input and
        // matches `np.argmax` exactly since there's nothing to skip).
        DType::Bool => {
            let (s, st, off, buf) = operand_of!(target, Bool);
            let _ = s;
            nan_free_argext(op, st, off, buf, &axes, &keep_axes, &keep_shape, &red_shape)?
        }
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => { let (s, st, off, buf) = operand_of!(target, I8); let _=s; nan_free_argext(op, st, off, buf, &axes, &keep_axes, &keep_shape, &red_shape)? }
        DType::I16 => { let (s, st, off, buf) = operand_of!(target, I16); let _=s; nan_free_argext(op, st, off, buf, &axes, &keep_axes, &keep_shape, &red_shape)? }
        DType::I32 => { let (s, st, off, buf) = operand_of!(target, I32); let _=s; nan_free_argext(op, st, off, buf, &axes, &keep_axes, &keep_shape, &red_shape)? }
        DType::I64 => { let (s, st, off, buf) = operand_of!(target, I64); let _=s; nan_free_argext(op, st, off, buf, &axes, &keep_axes, &keep_shape, &red_shape)? }
        DType::U8 => { let (s, st, off, buf) = operand_of!(target, U8); let _=s; nan_free_argext(op, st, off, buf, &axes, &keep_axes, &keep_shape, &red_shape)? }
        DType::U16 => { let (s, st, off, buf) = operand_of!(target, U16); let _=s; nan_free_argext(op, st, off, buf, &axes, &keep_axes, &keep_shape, &red_shape)? }
        DType::U32 => { let (s, st, off, buf) = operand_of!(target, U32); let _=s; nan_free_argext(op, st, off, buf, &axes, &keep_axes, &keep_shape, &red_shape)? }
        DType::U64 => { let (s, st, off, buf) = operand_of!(target, U64); let _=s; nan_free_argext(op, st, off, buf, &axes, &keep_axes, &keep_shape, &red_shape)? }
        DType::C64 => run_complex!(C64, C64, f32, |v: f32| v.is_nan(), C64::new(f32::NEG_INFINITY, f32::NEG_INFINITY), C64::new(f32::INFINITY, f32::INFINITY)),
        DType::C128 => run_complex!(C128, C128, f64, |v: f64| v.is_nan(), C128::new(f64::NEG_INFINITY, f64::NEG_INFINITY), C128::new(f64::INFINITY, f64::INFINITY)),
    };
    NdArray::from_buffer(Buffer::I64(out), keep_shape, ArrOrder::C)
}

#[allow(clippy::too_many_arguments)]
fn nan_free_argext<T: Copy + Ord>(
    op: NanExtremeOp,
    strides: &[isize],
    offset: isize,
    buf: &[T],
    axes: &[usize],
    keep_axes: &[usize],
    keep_shape: &[usize],
    red_shape: &[usize],
) -> Result<Vec<i64>, IonpError> {
    // Same 0-d courtesy-axis guard as above (`run_float!`'s `red_strides`) --
    // `strides` is empty for a 0-d array, `axes` can still be `[0]`.
    let red_strides: Vec<isize> = if strides.is_empty() { axes.iter().map(|_| 0).collect() } else { axes.iter().map(|&a| strides[a]).collect() };
    let keep_strides: Vec<isize> = keep_axes.iter().map(|&a| strides[a]).collect();
    let seq = |base: isize| -> Result<i64, IonpError> {
        let mut iter = NdIter::new(red_shape, &red_strides);
        let o0 = match iter.next() {
            Some(o) => o,
            None => return Err(IonpError::Value("All-NaN slice encountered".to_string())),
        };
        let mut best = buf[(base + o0) as usize];
        let mut best_idx: i64 = 0;
        let mut i: i64 = 1;
        for o in iter {
            let v = buf[(base + o) as usize];
            let better = match op {
                NanExtremeOp::Min => v < best,
                NanExtremeOp::Max => v > best,
            };
            if better {
                best = v;
                best_idx = i;
            }
            i += 1;
        }
        Ok(best_idx)
    };
    if keep_axes.is_empty() {
        Ok(vec![seq(offset)?])
    } else {
        NdIter::new(keep_shape, &keep_strides).map(|koff| seq(offset + koff)).collect()
    }
}

// ---------------------------------------------------------------------------
// where (ternary select form)
// ---------------------------------------------------------------------------

/// `np.where(condition, x, y)` -- elementwise select, broadcasting all
/// three operands together, output dtype is the NEP-50 promotion of `x`
/// and `y`'s dtypes (condition's own dtype never participates in output
/// dtype, only its truthiness). `np.where(condition)` (no x/y) is handled
/// by the caller as an alias for `nonzero`.
pub fn where_select(condition: &NdArray, x: &NdArray, y: &NdArray) -> Result<NdArray, IonpError> {
    let out_dtype = crate::dtype::promote_dtype(x.dtype(), y.dtype());
    let x = x.cast_to(out_dtype);
    let y = y.cast_to(out_dtype);
    let shape1 = crate::shape::broadcast_shapes(condition.shape(), x.shape())?;
    let out_shape = crate::shape::broadcast_shapes(&shape1, y.shape())?;

    let cond_strides = crate::shape::broadcast_strides_to(condition.shape(), condition.strides(), &out_shape)?;
    let cond_view = broadcast_view(condition, out_shape.clone(), cond_strides);
    let mask = is_truthy_mask(&cond_view)?;

    let x_strides = crate::shape::broadcast_strides_to(x.shape(), x.strides(), &out_shape)?;
    let x_view = broadcast_view(&x, out_shape.clone(), x_strides).to_contiguous();
    let y_strides = crate::shape::broadcast_strides_to(y.shape(), y.strides(), &out_shape)?;
    let y_view = broadcast_view(&y, out_shape.clone(), y_strides).to_contiguous();

    macro_rules! sel {
        ($variant:ident) => {{
            match (x_view.buffer(), y_view.buffer()) {
                (Buffer::$variant(xv), Buffer::$variant(yv)) => {
                    let out: Vec<_> = mask.iter().enumerate().map(|(i, &m)| if m { xv[i] } else { yv[i] }).collect();
                    Buffer::$variant(out)
                }
                _ => unreachable!(),
            }
        }};
    }
    let buffer = match out_dtype {
        DType::Bool => sel!(Bool),
        DType::S(_) | DType::U(_) => unreachable!("ionp-core has no S/U Buffer storage yet -- DType::S/U cannot reach this numeric dispatch path (phase 2)"),
        DType::I8 => sel!(I8),
        DType::I16 => sel!(I16),
        DType::I32 => sel!(I32),
        DType::I64 => sel!(I64),
        DType::U8 => sel!(U8),
        DType::U16 => sel!(U16),
        DType::U32 => sel!(U32),
        DType::U64 => sel!(U64),
        DType::F16 => sel!(F16),
        DType::F32 => sel!(F32),
        DType::F64 => sel!(F64),
        DType::C64 => sel!(C64),
        DType::C128 => sel!(C128),
    };
    NdArray::from_buffer(buffer, out_shape, ArrOrder::C)
}
