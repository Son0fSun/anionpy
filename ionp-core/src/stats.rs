//! `median` / `percentile` / `quantile` / `nanmedian` / `nanpercentile` /
//! `nanquantile` -- PATH-TO-100.md item 3 continuation.
//!
//! Reimplements numpy's `_function_base_impl.py` quantile machinery
//! (`_QuantileMethods`, `_compute_virtual_index`, `_get_indexes`,
//! `_get_gamma`, `_lerp`, `_discrete_interpolation_to_boundaries`,
//! `_inverted_cdf`, `_closest_observation`) directly in Rust, read from
//! numpy 2.5.1 source rather than guessed. `_median`'s "average via mean()
//! to coerce dtype" behavior is replicated as a small dedicated
//! one-or-two-element combine (see `combine_median_pair` /
//! `promote_scalar_to_f64`) rather than routed through the generic
//! `ufunc::reduce_axis`/`mean` machinery, since at most two elements per
//! output position are ever combined here.
//!
//! DELIBERATE SCOPE CUTS (documented, not silently dropped):
//!   * `q` is always treated as float64 internally, matching real numpy's
//!     behavior for an ARRAY `q` (or any non-bare-Python-scalar `q`).
//!     Real numpy's NEP-50 "weak scalar" rule additionally preserves a
//!     narrower float dtype (float16/float32) for a genuinely bare Python
//!     `int`/`float` `q` combined with an INTERPOLATED method. NARROWED
//!     2026-08-04: the binding layer now carries the weak-scalar flag
//!     (`QuantileArgs.q_is_scalar` / `q_is_integral`) and the float64
//!     operand case IS reproduced. What remains undeclared is a
//!     float16/float32 OPERAND under that rule -- see KNOWN-DIFFERENCES.
//!   * The `method='linear'` (or `'averaged_inverted_cdf'`) + integer/bool
//!     `q` "supports_integers" dtype-preservation shortcut: reproduced as
//!     of 2026-08-04 via `QuantileArgs.q_is_integral`; `q` is still
//!     computed in float64, only the RESULT dtype follows numpy.
//!   * `axis`: CLOSED 2026-08-04. Tuple axes are supported (the binding
//!     layer normalizes them and this module reduces over the flattened
//!     product), as are the negative/duplicate/out-of-range forms numpy
//!     accepts or rejects. The nan-aware family additionally reproduces
//!     numpy's MOVED-AXIS result layout for a non-scalar `q` (numpy
//!     computes with the q axis last and moves it to the front, returning
//!     a view whose strides are F-ish); see the comment at the
//!     `moveaxis` call in the quantile builder for the measurements.
//!   * `weights=` is not supported at all (median/percentile/quantile's
//!     weighted `inverted_cdf` path is not implemented).
//!   * FIXED (was previously an undeclared gap): interpolated-method
//!     arithmetic reads every value from an f64-cast copy of the sorted
//!     input for convenience, but the `diff = next - previous` step used
//!     in `_lerp` is computed via `native_diff_f64` in the ORIGINAL
//!     dtype's own width first (matching real numpy's `_lerp`, which
//!     differences `next - previous` BEFORE any promotion happens) --
//!     for float16/float32 this means the diff is rounded to that
//!     dtype's own precision (`half::f16`'s `Sub` impl, which itself
//!     round-trips through f32 like every other f16 op in this crate);
//!     for the fixed-width integer dtypes (i8/i16/i32/i64/u8/u16/u32/u64)
//!     this means the diff WRAPS on overflow exactly like numpy's own
//!     native integer subtract does (e.g. `i8(70) - i8(-88)` is `158`,
//!     which does not fit in `i8` and wraps to `-98` -- verified against
//!     real numpy 2.5.1, not assumed). Only remaining gap: integer
//!     magnitude beyond 2**53 (float64's exact-integer range) after the
//!     wrap, where the final float64 cast itself can't represent the
//!     wrapped value exactly -- not separately reproduced, undeclared for
//!     that magnitude range (i64/u64 only; every other integer dtype's
//!     full range fits under 2**53 even after wrapping).

use crate::array::NdArray;
use crate::buffer::{Buffer, C128, C64};
use crate::creation::moveaxis;
use crate::dtype::DType;
use crate::error::IonpError;
use crate::sort::sort_axis;
use half::f16;

// ---------------------------------------------------------------------------
// Quantile method table
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuantileMethod {
    InvertedCdf,
    AveragedInvertedCdf,
    ClosestObservation,
    InterpolatedInvertedCdf,
    Hazen,
    Weibull,
    Linear,
    MedianUnbiased,
    NormalUnbiased,
    Lower,
    Higher,
    Midpoint,
    Nearest,
}

pub const QUANTILE_METHOD_NAMES: &[&str] = &[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
];

impl QuantileMethod {
    pub fn from_str(s: &str) -> Result<Self, IonpError> {
        use QuantileMethod::*;
        Ok(match s {
            "inverted_cdf" => InvertedCdf,
            "averaged_inverted_cdf" => AveragedInvertedCdf,
            "closest_observation" => ClosestObservation,
            "interpolated_inverted_cdf" => InterpolatedInvertedCdf,
            "hazen" => Hazen,
            "weibull" => Weibull,
            "linear" => Linear,
            "median_unbiased" => MedianUnbiased,
            "normal_unbiased" => NormalUnbiased,
            "lower" => Lower,
            "higher" => Higher,
            "midpoint" => Midpoint,
            "nearest" => Nearest,
            other => {
                return Err(IonpError::Value(format!(
                    "'{other}' is not a valid method. Use one of: dict_keys(['inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear', 'median_unbiased', 'normal_unbiased', 'lower', 'higher', 'midpoint', 'nearest'])"
                )))
            }
        })
    }

    fn is_direct(self) -> bool {
        use QuantileMethod::*;
        matches!(self, InvertedCdf | ClosestObservation | Lower | Higher | Nearest)
    }
}

fn round_half_even(x: f64) -> f64 {
    let f = x.floor();
    let diff = x - f;
    if diff < 0.5 {
        f
    } else if diff > 0.5 {
        f + 1.0
    } else if (f as i64).rem_euclid(2) == 0 {
        f
    } else {
        f + 1.0
    }
}

/// Mirrors `_discrete_interpolation_to_boundaries`: clip negative results
/// to 0 (upper bound never needs clipping for `q` in `[0, 1]`).
fn discrete_to_boundary(index: f64, take_previous: bool) -> i64 {
    let previous = index.floor();
    let next = previous + 1.0;
    let res = (if take_previous { previous } else { next }) as i64;
    res.max(0)
}

/// `previous_idx`, `next_idx`, `gamma` for interpolated methods, matching
/// `_get_indexes` + `_get_gamma` + each method's `fix_gamma`. `n` is the
/// (already fixed, per-row) sample size.
fn virtual_index_raw(method: QuantileMethod, n: usize, q: f64) -> f64 {
    let nf = n as f64;
    use QuantileMethod::*;
    match method {
        AveragedInvertedCdf => nf * q - 1.0,
        InterpolatedInvertedCdf => compute_virtual_index(nf, q, 0.0, 1.0),
        Hazen => compute_virtual_index(nf, q, 0.5, 0.5),
        Weibull => compute_virtual_index(nf, q, 0.0, 0.0),
        Linear => (nf - 1.0) * q,
        MedianUnbiased => compute_virtual_index(nf, q, 1.0 / 3.0, 1.0 / 3.0),
        NormalUnbiased => compute_virtual_index(nf, q, 3.0 / 8.0, 3.0 / 8.0),
        Midpoint => {
            let lo = ((nf - 1.0) * q).floor();
            let hi = ((nf - 1.0) * q).ceil();
            0.5 * (lo + hi)
        }
        _ => unreachable!("virtual_index_raw called for a direct-index method"),
    }
}

fn interp_index(method: QuantileMethod, n: usize, q: f64) -> (usize, usize, f64) {
    let nf = n as f64;
    let vi = virtual_index_raw(method, n, q);
    let (prev_f, next_f): (f64, f64) = if vi >= nf - 1.0 {
        (nf - 1.0, nf - 1.0)
    } else if vi < 0.0 {
        (0.0, 0.0)
    } else {
        (vi.floor(), vi.floor() + 1.0)
    };
    let raw_gamma = vi - prev_f;
    use QuantileMethod::*;
    let gamma = match method {
        AveragedInvertedCdf => {
            if raw_gamma == 0.0 {
                0.5
            } else {
                1.0
            }
        }
        Midpoint => {
            if vi.fract() == 0.0 {
                0.0
            } else {
                0.5
            }
        }
        _ => raw_gamma,
    };
    (prev_f as usize, next_f as usize, gamma)
}

/// The 0-based row-local sorted-order index for a single `q` under a
/// direct-index (`fix_gamma is None`) method, given a fixed sample size.
fn direct_index(method: QuantileMethod, n: usize, q: f64) -> usize {
    let nf = n as f64;
    use QuantileMethod::*;
    let idx = match method {
        InvertedCdf => {
            let idx = nf * q - 1.0;
            let gamma = idx - idx.floor();
            discrete_to_boundary(idx, gamma == 0.0)
        }
        ClosestObservation => {
            let idx = nf * q - 1.0 - 0.5;
            let gamma = idx - idx.floor();
            let floor_i = idx.floor() as i64;
            let take_prev = gamma == 0.0 && floor_i.rem_euclid(2) == 1;
            discrete_to_boundary(idx, take_prev)
        }
        Lower => (((nf - 1.0) * q).floor() as i64).max(0),
        Higher => (((nf - 1.0) * q).ceil() as i64).max(0),
        Nearest => (round_half_even((nf - 1.0) * q) as i64).max(0),
        // `linear` is NOT a direct method in general. It reaches this
        // function only through the INTEGRAL-`q` shortcut -- numpy's
        //     int_virtual_indices = np.issubdtype(virtual_indexes.dtype,
        //                                         np.integer)
        //     supports_integers = method == 'linear' and int_virtual_indices
        //     if supports_integers:  result = take(arr, virtual_indexes, ...)
        // -- where `virtual_indexes` for `linear` is `q * (n - 1)` computed
        // in `q`'s own dtype. When `q` is an integer/bool array that product
        // is an INTEGER array, so numpy skips interpolation entirely, takes
        // the element, and the result keeps the OPERAND's dtype instead of
        // promoting to float64. Since `q` is range-checked to [0, 1] an
        // integral `q` is exactly 0 or 1, so `(n-1)*q` is exact in f64 and
        // `round` merely discards the representation error of neither.
        Linear => ((((nf - 1.0) * q).round()) as i64).max(0),
        _ => unreachable!("direct_index called for an interpolated method"),
    };
    (idx as usize).min(n.saturating_sub(1))
}

fn compute_virtual_index(n: f64, q: f64, alpha: f64, beta: f64) -> f64 {
    n * q + (alpha + q * (1.0 - alpha - beta)) - 1.0
}

/// numpy's two-branch `_lerp` (precision-preserving near `t == 1`), given an
/// already-computed `diff = b - a`. Split out from `lerp` below so callers
/// that need the diff computed in NARROWER-than-f64 precision (see
/// `native_diff_f64`) can supply it directly.
fn lerp_with_diff(a: f64, b: f64, t: f64, diff: f64) -> f64 {
    if t < 0.5 {
        a + diff * t
    } else {
        b - diff * (1.0 - t)
    }
}

/// numpy's two-branch `_lerp` (precision-preserving near `t == 1`).
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    lerp_with_diff(a, b, t, b - a)
}

/// `_lerp` executed ENTIRELY in float32, for numpy's "weak `q`" path.
///
/// numpy's `_quantile` carries a `weak_q` flag (set by `percentile`/
/// `quantile`/`nanpercentile`/`nanquantile` as literally
/// `type(q) in (int, float)`) and, when it is set, does `gamma =
/// float(gamma)` -- demoting the float64 gamma array to a bare Python float
/// -- immediately before calling `_lerp`. Under NEP 50 a bare Python float
/// is a WEAK scalar, so every subsequent op in `_lerp` stays in the array's
/// own dtype and the result comes back float32 instead of float64.
///
/// `t` arrives here as f64 (it is computed from the index arithmetic, which
/// numpy also does in double) and is cast down at each use, exactly where
/// NEP 50 casts the weak Python scalar to the array dtype.
fn lerp_f32(a: f32, b: f32, t: f64) -> f32 {
    let diff = b - a;
    if t < 0.5 {
        a + diff * (t as f32)
    } else {
        b - diff * ((1.0 - t) as f32)
    }
}

/// `_lerp` executed entirely in float16 -- see `lerp_f32`.
///
/// Every intermediate is rounded back to f16 between operations, because
/// that is what a chain of numpy float16 ufunc calls does. `half::f16`
/// computes through f32 internally, matching the rest of this crate's f16
/// arithmetic (see `ufunc.rs`'s module docs).
fn lerp_f16(a: f16, b: f16, t: f64) -> f16 {
    let diff = f16::from_f32(b.to_f32() - a.to_f32());
    if t < 0.5 {
        let m = f16::from_f32(diff.to_f32() * f16::from_f64(t).to_f32());
        f16::from_f32(a.to_f32() + m.to_f32())
    } else {
        let m = f16::from_f32(diff.to_f32() * f16::from_f64(1.0 - t).to_f32());
        f16::from_f32(b.to_f32() - m.to_f32())
    }
}

/// Live-verified numpy quirk: for interpolated methods, real numpy's
/// `_lerp` computes `diff_b_a = subtract(b, a)` on the sorted array BEFORE
/// it is cast up to float64 -- when the input dtype is float16 or float32,
/// that subtraction happens (and rounds) in the NATIVE dtype's precision
/// (NEP 50: `float16 - float16 -> float16`, `float32 - float32 -> float32`),
/// and only the resulting (already-rounded) diff gets promoted to float64
/// when multiplied by `gamma` (a float64 array whenever `q` was passed as
/// an array rather than a bare Python scalar). ionp's own `build_regular`/
/// `build_nan_variant` used to cast the WHOLE sorted array to f64 up front
/// and compute `diff = b - a` at full double precision, which is a few ULP
/// off from numpy's narrower-then-widened diff for f16/f32 input -- this
/// helper reproduces numpy's actual rounding by computing `b - a` in the
/// ORIGINAL narrow dtype (via `half::f16`'s own `Sub` impl, which itself
/// round-trips through f32 exactly like every other f16 arithmetic op in
/// this crate -- see `ufunc.rs`'s module docs) and only then widening the
/// (already-rounded) result to f64. Not applicable to f64/int/bool input:
/// f64 has no narrower-precision step to lose, and int/bool are widened to
/// f64 losslessly (exact integers) before any subtraction happens on numpy's
/// side too (its own docstring: "the output data-type is float64" for
/// integer input).
fn native_diff_f64(dtype: DType, buf: &Buffer, ia: usize, ib: usize) -> f64 {
    match (dtype, buf) {
        (DType::F16, Buffer::F16(v)) => (v[ib] - v[ia]).to_f64(),
        (DType::F32, Buffer::F32(v)) => (v[ib] - v[ia]) as f64,
        // Integer dtypes: live-verified real numpy ALSO computes
        // `diff_b_a = subtract(b, a)` in the native fixed-width integer
        // dtype (not the float64 the final output gets promoted to), so
        // it WRAPS on overflow exactly like any other numpy integer
        // subtract -- e.g. `np.int8(70) - np.int8(-88)` is `158`, which
        // does not fit in `i8` (max 127) and wraps to `-98`. Since the
        // sorted-row invariant guarantees `ib`'s value >= `ia`'s value
        // (ascending sort, `ib` is the "next" index), the true
        // mathematical difference is always >= 0, but the WRAPPED
        // native-width result can still be negative -- matching that
        // wrap (via Rust's own `wrapping_sub`, same two's-complement
        // semantics as numpy's C integer subtract) is the whole point
        // here, not avoiding it.
        (DType::I8, Buffer::I8(v)) => v[ib].wrapping_sub(v[ia]) as f64,
        (DType::I16, Buffer::I16(v)) => v[ib].wrapping_sub(v[ia]) as f64,
        (DType::I32, Buffer::I32(v)) => v[ib].wrapping_sub(v[ia]) as f64,
        (DType::I64, Buffer::I64(v)) => v[ib].wrapping_sub(v[ia]) as f64,
        (DType::U8, Buffer::U8(v)) => v[ib].wrapping_sub(v[ia]) as f64,
        (DType::U16, Buffer::U16(v)) => v[ib].wrapping_sub(v[ia]) as f64,
        (DType::U32, Buffer::U32(v)) => v[ib].wrapping_sub(v[ia]) as f64,
        (DType::U64, Buffer::U64(v)) => v[ib].wrapping_sub(v[ia]) as f64,
        _ => unreachable!("native_diff_f64 does not apply to this dtype (bool is rejected earlier; F64/complex have no narrower-width diff to reproduce)"),
    }
}

// ---------------------------------------------------------------------------
// Shared "sort along axis, move axis to the end" plumbing
// ---------------------------------------------------------------------------

struct SortedRows {
    sorted: NdArray, // C-contiguous, axis moved to the last position
    other_shape: Vec<usize>,
    n: usize,
    rows: usize,
}

fn sort_rows(a: &NdArray, axis: usize) -> Result<SortedRows, IonpError> {
    let ndim = a.ndim();
    let moved = if ndim == 0 {
        a.clone()
    } else if axis == ndim - 1 {
        a.clone()
    } else {
        moveaxis(a, &[axis], &[ndim - 1])?
    };
    let sorted = if ndim == 0 {
        moved.to_contiguous()
    } else {
        sort_axis(&moved, ndim - 1, false, false)?.to_contiguous()
    };
    let shape = sorted.shape().to_vec();
    let n = if ndim == 0 { 1 } else { *shape.last().unwrap() };
    let other_shape = if ndim == 0 { vec![] } else { shape[..shape.len() - 1].to_vec() };
    let rows: usize = other_shape.iter().product();
    Ok(SortedRows { sorted, other_shape, n, rows })
}

macro_rules! gather_rows {
    ($buffer:expr, $total:expr, $idx:expr) => {
        match $buffer {
            Buffer::Bool(v) => Buffer::Bool((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::I8(v) => Buffer::I8((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::I16(v) => Buffer::I16((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::I32(v) => Buffer::I32((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::I64(v) => Buffer::I64((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::U8(v) => Buffer::U8((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::U16(v) => Buffer::U16((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::U32(v) => Buffer::U32((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::U64(v) => Buffer::U64((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::F16(v) => Buffer::F16((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::F32(v) => Buffer::F32((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::F64(v) => Buffer::F64((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::C64(v) => Buffer::C64((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::C128(v) => Buffer::C128((0..$total).map(|i| v[$idx(i)]).collect()),
            Buffer::S(_, _) | Buffer::U(_, _) => unreachable!("stats.rs: quantile/percentile has no semantics for string dtype here yet (phase 2 declines this operation on S/U)"),
        }
    };
}

fn nan_of_dtype(dtype: DType) -> Buffer1 {
    match dtype {
        DType::F16 => Buffer1::F16(f16::NAN),
        DType::F32 => Buffer1::F32(f32::NAN),
        DType::F64 => Buffer1::F64(f64::NAN),
        // `(NAN, NAN)` is correct HERE and only here. This is the EMPTY-slice
        // scalar: numpy reaches an empty complex median through `mean()` of
        // zero elements, i.e. `0j/0`, which is `nan+nanj` -- measured
        // 2026-08-04 on numpy 2.5.1, `np.median(np.zeros((0,), complex))` and
        // `np.nanmedian` of the same both give `nan+nanj`.
        //
        // The ALL-NAN (nonempty) slice is a DIFFERENT scalar, `nan+0j`, and is
        // built inline in `median_axis` -- see `nan_all_nan_slice`. Conflating
        // the two was a real bug caught by the 2026-08-04 sweep.
        DType::C64 => Buffer1::C64(C64::new(f32::NAN, f32::NAN)),
        DType::C128 => Buffer1::C128(C128::new(f64::NAN, f64::NAN)),
        _ => Buffer1::None,
    }
}

enum Buffer1 {
    F16(f16),
    F32(f32),
    F64(f64),
    C64(C64),
    C128(C128),
    None,
}

fn overwrite_with_nan(buf: &mut Buffer, positions: &[usize]) {
    if positions.is_empty() {
        return;
    }
    match buf {
        Buffer::F16(v) => {
            for &p in positions {
                v[p] = f16::NAN;
            }
        }
        Buffer::F32(v) => {
            for &p in positions {
                v[p] = f32::NAN;
            }
        }
        Buffer::F64(v) => {
            for &p in positions {
                v[p] = f64::NAN;
            }
        }
        Buffer::C64(v) => {
            for &p in positions {
                v[p] = C64::new(f32::NAN, f32::NAN);
            }
        }
        Buffer::C128(v) => {
            for &p in positions {
                v[p] = C128::new(f64::NAN, f64::NAN);
            }
        }
        _ => {} // non-float dtypes can never carry NaN
    }
}

/// Row-last-element-is-NaN test (only meaningful for float/complex dtypes;
/// always `false` otherwise, matching numpy's `supports_nans` gate).
fn row_has_nan(sorted: &NdArray, n: usize, rows: usize) -> Vec<bool> {
    if n == 0 {
        return vec![false; rows];
    }
    macro_rules! last_col {
        ($v:expr, $isnan:expr) => {
            (0..rows).map(|r| $isnan(&$v[r * n + n - 1])).collect()
        };
    }
    match sorted.buffer() {
        Buffer::F16(v) => last_col!(v, |x: &f16| x.is_nan()),
        Buffer::F32(v) => last_col!(v, |x: &f32| x.is_nan()),
        Buffer::F64(v) => last_col!(v, |x: &f64| x.is_nan()),
        Buffer::C64(v) => last_col!(v, |x: &C64| x.re.is_nan() || x.im.is_nan()),
        Buffer::C128(v) => last_col!(v, |x: &C128| x.re.is_nan() || x.im.is_nan()),
        _ => vec![false; rows],
    }
}

// ---------------------------------------------------------------------------
// median / nanmedian
// ---------------------------------------------------------------------------

fn median_out_dtype(dtype: DType) -> DType {
    if dtype == DType::Bool || dtype.is_integer() {
        DType::F64
    } else {
        dtype
    }
}

/// numpy's complex-divide-by-a-REAL-denominator, bit-for-bit.
///
/// `median` ALWAYS finishes through `mean()` -- `_median` returns
/// `mean(part[indexer], axis=axis)` for the odd (one-element slice) case
/// just as much as for the even (two-element) one -- and `mean` divides the
/// complex sum by a real `rcount`. That division is NOT componentwise. The
/// complex `true_divide` loop is Smith's algorithm, which for a denominator
/// `d + 0j` reduces to
///
///     rat = 0 / d = 0 ;  scl = 1 / (d + 0*rat) = 1/d
///     re' = (re + im*rat) * scl   ->   (re + im*0) / d
///     im' = (im - re*rat) * scl   ->   (im - re*0) / d
///
/// and `x * 0.0` is NaN, not 0.0, when `x` is infinite. So a perfectly
/// ordinary finite-median input containing an infinity comes back with a
/// NaN in the OTHER component: measured live against numpy 2.5.1 on
/// 2026-08-04, `np.median(np.array([complex(inf, 0)]))` is `inf+nanj`,
/// `np.median(np.array([complex(1, inf)]))` is `nan+infj`, and
/// `np.median(np.array([complex(inf, inf)]))` is `nan+nanj`. Returning the
/// element verbatim (which is what this file did until 2026-08-04, and what
/// every real-float dtype can safely do since `x / 1.0 == x` there) is
/// wrong for exactly those inputs.
///
/// `complex(nan, 1)` does NOT come back `nan+nanj` even though Smith says
/// it should -- but that is not this function being wrong, it is that a
/// NaN-bearing complex slice never reaches here at all: `_median_nancheck`
/// substitutes the raw element first. See `median_axis`.
///
/// Multiplying by the reciprocal (rather than dividing twice) is
/// deliberate -- it is what the C loop does, and `d` is only ever 1 or 2
/// here, both of which have exact reciprocals, so the two agree anyway.
fn cdiv_by_real(re: f64, im: f64, d: f64) -> (f64, f64) {
    let scl = 1.0 / d;
    ((re + im * 0.0) * scl, (im - re * 0.0) * scl)
}

/// `(lo + hi) / 2`, replicating `mean()`'s exact widen-divide-narrow
/// sequence for the two dtypes that widen (`F16`->via F32 sum->F64 divide,
/// `F32`/`C64` -> F64/C128 divide), and plain native-dtype arithmetic
/// otherwise. `dtype` is the OUTPUT dtype (post int/bool promotion).
fn combine_median_pair(lo_buf: &Buffer, lo_i: usize, hi_buf: &Buffer, hi_i: usize, dtype: DType) -> Buffer1 {
    match (lo_buf, hi_buf, dtype) {
        (Buffer::F16(l), Buffer::F16(h), DType::F16) => {
            let sum32 = l[lo_i].to_f32() + h[hi_i].to_f32();
            let div64 = (sum32 as f64) / 2.0;
            Buffer1::F16(f16::from_f32(div64 as f32))
        }
        (Buffer::F32(l), Buffer::F32(h), DType::F32) => {
            let sum32 = l[lo_i] + h[hi_i];
            let div64 = (sum32 as f64) / 2.0;
            Buffer1::F32(div64 as f32)
        }
        (Buffer::F64(l), Buffer::F64(h), DType::F64) => Buffer1::F64((l[lo_i] + h[hi_i]) / 2.0),
        (Buffer::C64(l), Buffer::C64(h), DType::C64) => {
            let sum = l[lo_i] + h[hi_i];
            let (re, im) = cdiv_by_real(sum.re as f64, sum.im as f64, 2.0);
            Buffer1::C64(C64::new(re as f32, im as f32))
        }
        (Buffer::C128(l), Buffer::C128(h), DType::C128) => {
            let sum = l[lo_i] + h[hi_i];
            let (re, im) = cdiv_by_real(sum.re, sum.im, 2.0);
            Buffer1::C128(C128::new(re, im))
        }
        _ => {
            // bool/int/uint promoted to f64: both elements cast to f64 first.
            let lo = scalar_to_f64(lo_buf, lo_i);
            let hi = scalar_to_f64(hi_buf, hi_i);
            Buffer1::F64((lo + hi) / 2.0)
        }
    }
}

fn scalar_to_f64(buf: &Buffer, i: usize) -> f64 {
    match buf {
        Buffer::Bool(v) => {
            if v[i] {
                1.0
            } else {
                0.0
            }
        }
        Buffer::I8(v) => v[i] as f64,
        Buffer::I16(v) => v[i] as f64,
        Buffer::I32(v) => v[i] as f64,
        Buffer::I64(v) => v[i] as f64,
        Buffer::U8(v) => v[i] as f64,
        Buffer::U16(v) => v[i] as f64,
        Buffer::U32(v) => v[i] as f64,
        Buffer::U64(v) => v[i] as f64,
        Buffer::F16(v) => v[i].to_f64(),
        Buffer::F32(v) => v[i] as f64,
        Buffer::F64(v) => v[i],
        _ => unreachable!("scalar_to_f64 called on a complex buffer"),
    }
}

fn single_element(buf: &Buffer, i: usize, out_dtype: DType) -> Buffer1 {
    match out_dtype {
        DType::F16 => Buffer1::F16(match buf {
            Buffer::F16(v) => v[i],
            _ => unreachable!(),
        }),
        DType::F32 => Buffer1::F32(match buf {
            Buffer::F32(v) => v[i],
            _ => unreachable!(),
        }),
        DType::F64 => {
            if let Buffer::F64(v) = buf {
                Buffer1::F64(v[i])
            } else {
                Buffer1::F64(scalar_to_f64(buf, i))
            }
        }
        DType::C64 => Buffer1::C64(match buf {
            Buffer::C64(v) => v[i],
            _ => unreachable!(),
        }),
        DType::C128 => Buffer1::C128(match buf {
            Buffer::C128(v) => v[i],
            _ => unreachable!(),
        }),
        _ => unreachable!("single_element: non-float out_dtype"),
    }
}

enum Buffer1Vec {
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    C64(Vec<C64>),
    C128(Vec<C128>),
}

impl Buffer1Vec {
    fn new(dtype: DType, cap: usize) -> Self {
        match dtype {
            DType::F16 => Buffer1Vec::F16(Vec::with_capacity(cap)),
            DType::F32 => Buffer1Vec::F32(Vec::with_capacity(cap)),
            DType::F64 => Buffer1Vec::F64(Vec::with_capacity(cap)),
            DType::C64 => Buffer1Vec::C64(Vec::with_capacity(cap)),
            DType::C128 => Buffer1Vec::C128(Vec::with_capacity(cap)),
            _ => unreachable!("Buffer1Vec::new: non-float dtype"),
        }
    }
    fn push(&mut self, v: Buffer1) {
        match (self, v) {
            (Buffer1Vec::F16(v1), Buffer1::F16(x)) => v1.push(x),
            (Buffer1Vec::F32(v1), Buffer1::F32(x)) => v1.push(x),
            (Buffer1Vec::F64(v1), Buffer1::F64(x)) => v1.push(x),
            (Buffer1Vec::C64(v1), Buffer1::C64(x)) => v1.push(x),
            (Buffer1Vec::C128(v1), Buffer1::C128(x)) => v1.push(x),
            _ => unreachable!("Buffer1Vec::push: dtype mismatch"),
        }
    }
    fn into_buffer(self) -> Buffer {
        match self {
            Buffer1Vec::F16(v) => Buffer::F16(v),
            Buffer1Vec::F32(v) => Buffer::F32(v),
            Buffer1Vec::F64(v) => Buffer::F64(v),
            Buffer1Vec::C64(v) => Buffer::C64(v),
            Buffer1Vec::C128(v) => Buffer::C128(v),
        }
    }
}

pub fn median_axis(a: &NdArray, axis: usize, keepdims: bool, skip_nan: bool) -> Result<NdArray, IonpError> {
    let out_dtype = median_out_dtype(a.dtype());
    let rows_info = sort_rows(a, axis)?;
    let n = rows_info.n;
    // NOTE: `rows_info.rows` is the true row count -- 0 when some OTHER
    // (non-reduced) axis has length 0, in which case the output is
    // legitimately empty and no buffer element is ever touched below.
    // Forcing this to `.max(1)` (an earlier version of this function did)
    // caused an out-of-bounds panic by pretending a row existed when the
    // backing buffer actually had zero elements.
    let rows = rows_info.rows;
    let buf = rows_info.sorted.buffer();

    let mut out = Buffer1Vec::new(out_dtype, rows);

    if !skip_nan {
        let has_nan = row_has_nan(&rows_info.sorted, n, rows_info.rows);
        for r in 0..rows {
            if n == 0 {
                out.push(nan_scalar(out_dtype));
                continue;
            }
            let mid = n / 2;
            let val = if !has_nan.get(r).copied().unwrap_or(false) {
                if n % 2 == 1 {
                    single_element_as_mean(buf, r * n + mid, out_dtype)
                } else {
                    combine_median_pair(buf, r * n + mid - 1, buf, r * n + mid, out_dtype)
                }
            } else if !matches!(out_dtype, DType::C64 | DType::C128) {
                // REAL float NaN slice: mint a fresh quiet NaN. See the long
                // note below for why the sorted element is NOT usable here.
                nan_all_nan_slice(out_dtype)
            } else {
                // numpy's NaN substitution here is NOT a freshly-minted NaN.
                // `numpy.lib._utils_impl._median_nancheck` computes
                // `potential_nans = data.take(-1, axis=axis)` -- the LAST
                // element of the partitioned/sorted data -- and returns (0-d
                // case) or `copyto`s (n-d case) THAT ELEMENT into the result
                // wherever `np.isnan(potential_nans)` holds. Since NaN sorts
                // to the end, that element is precisely the NaN-bearing one.
                //
                // For a REAL float dtype the distinction is invisible (the
                // substituted element is a NaN either way, and both carry the
                // default quiet payload for every input this library can
                // build). For COMPLEX it is load-bearing and was a real
                // divergence until 2026-08-04: `np.isnan` on a complex is true
                // when EITHER component is NaN, so `np.median([1+2j, nan+5j])`
                // is `nan+5j` -- the surviving `5j` imaginary part comes along
                // for the ride -- where this function previously produced
                // `nan+nanj`. Likewise `np.median([complex(1,nan),
                // complex(2,3)])` is `1+nanj`, keeping the REAL part.
                //
                // WHY THIS BRANCH IS COMPLEX-ONLY, and not the general
                // implementation of `_median_nancheck` it looks like it should
                // be. numpy's `_median` partitions; this function SORTS. Those
                // two differ in exactly one observable way, measured
                // 2026-08-04 against numpy 2.5.1 on this platform:
                //     a    = np.array([1.0, nan, 3.0])   -> ...00f87f...
                //     sort(a).tobytes()[-8:]             -> ffffffffffffff7f
                //     partition(a, 1).tobytes()[-8:]     -> 000000000000f87f
                // numpy's float SORT rewrites a NaN to the all-ones payload
                // 0x7fffffffffffffff; its PARTITION leaves the input's own
                // 0x7ff8... quiet NaN alone. So for a REAL float dtype
                // `data.take(-1)` off a sorted row is the WRONG BITS, and
                // minting a fresh `f64::NAN`/`f32::NAN`/`f16::NAN` (what the
                // `else` chain's own NaN path has always done) is what
                // actually reproduces numpy. Taking the sorted element there
                // regressed 225 sweep cases; this is measured, not assumed.
                //
                // Complex sort does NOT canonicalize (verified on the same
                // three payload cases: `nan+5j`, `1+nanj`, `nan+nanj` all
                // round-trip bit-identical through `ionp.sort`), so the
                // element-take is both correct and REQUIRED there -- it is
                // the only way to keep the surviving non-NaN component.
                //
                // `single_element_promoted` cannot promote on this branch:
                // it is gated on a complex `out_dtype`, and `median_out_dtype`
                // maps complex to itself.
                single_element_promoted(buf, r * n + n - 1, out_dtype)
            };
            out.push(val);
        }
    } else {
        // nanmedian: strip trailing NaNs per row (sorted puts them last),
        // then run the identical odd/even combine over the reduced count.
        for r in 0..rows {
            let valid = valid_count(&rows_info.sorted, r, n);
            if valid == 0 {
                // Two different NaNs live here; see `nan_all_nan_slice`. An
                // EMPTY row (`n == 0`) is numpy's `mean` of nothing (complex
                // `nan+nanj`); a NONEMPTY row that happens to be all-NaN is
                // numpy's real `np.nan` written into a complex slot
                // (`nan+0j`). Identical for every real dtype.
                out.push(if n == 0 { nan_scalar(out_dtype) } else { nan_all_nan_slice(out_dtype) });
                continue;
            }
            let mid = valid / 2;
            let val = if valid % 2 == 1 {
                single_element_as_mean(buf, r * n + mid, out_dtype)
            } else {
                let lo_i = r * n + mid - 1;
                let hi_i = r * n + mid;
                combine_median_pair(buf, lo_i, buf, hi_i, out_dtype)
            };
            out.push(val);
        }
    }

    let out_shape = rows_info.other_shape.clone();
    let natural = NdArray::from_buffer(out.into_buffer(), out_shape, crate::array::Order::C)?;
    // Real numpy's median/nanmedian ultimately reduce via a sort+index
    // (odd n) or sort+index+average (even n) step whose OWN output layout
    // (keepdims=False) exactly matches a plain `sum`/`mean` reduction over
    // the same single axis -- confirmed via direct real-numpy comparison
    // (`/tmp/probe_median_mech2.py`): `np.sum(...).strides ==
    // np.median(...).strides` for every one of C/F/fulltranspose/
    // partialswap/permuted layout x axis combination tested, keepdims=False.
    // Reuse `relayout_for_reduction` -- the exact function `sum`'s own
    // result is built with -- to get that layout, rather than the always-
    // plain-C layout `from_buffer` gives by construction.
    let natural = natural.relayout_for_reduction(a.shape(), a.strides(), &[axis], false);
    if !keepdims {
        return Ok(natural);
    }
    // UNLIKE `sum`/`mean` (which keep the array's REAL original stride at
    // the reduced axis's position when keepdims=True -- `reduce_output_
    // layout`'s "pretend size-1 axis" trick, via `lift_reduction_keepdims`),
    // real numpy's median keepdims=True result carries a genuine BROADCAST
    // stride of 0 at that axis instead -- confirmed empirically
    // (`/tmp/probe_median_mech2.py`): e.g. a C-contiguous (2,3,4) array
    // reduced on axis=0 gives `np.sum(..., keepdims=True).strides == (96,
    // 32, 8)` (real original axis-0 stride preserved) but
    // `np.median(..., keepdims=True).strides == (0, 32, 8)` (literal zero)
    // for the SAME input. `creation::insert_newaxis`'s stride-0 insert is
    // exactly this broadcast convention, so reuse it rather than
    // `lift_reduction_keepdims`'s pretend-real-stride one. NOT
    // `creation::expand_dims` -- since Task #newaxis-split (2026-08-08),
    // that function implements numpy's DIFFERENT `a.reshape(...)`-based
    // `expand_dims` rule, which would compute a genuine nonzero stride
    // here instead of the broadcast 0 this path is measured to need (the
    // "freshly allocated `natural` array" confound the split's own task
    // brief warned about does NOT apply here -- the two rules give
    // DIFFERENT answers on this exact C-contiguous `natural`, not
    // coincidentally the same one).
    Ok(crate::creation::insert_newaxis(&natural, &[axis])?)
}

/// The substitution for a NONEMPTY all-NaN slice (`nanmedian`).
///
/// Distinct from `nan_scalar` (the EMPTY-slice scalar) in the complex arms
/// only: numpy gets here by writing the REAL Python scalar `np.nan` into a
/// complex-dtype result, which leaves the imaginary part a plain zero --
/// `np.nanmedian(np.full(4, nan, dtype=complex))` is `nan+0j`, where
/// `np.nanmedian(np.zeros((0,), complex))` is `nan+nanj`. Measured
/// 2026-08-04 against numpy 2.5.1, not inferred from the source.
fn nan_all_nan_slice(dtype: DType) -> Buffer1 {
    match dtype {
        DType::C64 => Buffer1::C64(C64::new(f32::NAN, 0.0)),
        DType::C128 => Buffer1::C128(C128::new(f64::NAN, 0.0)),
        other => nan_scalar(other),
    }
}

fn nan_scalar(dtype: DType) -> Buffer1 {
    match nan_of_dtype(dtype) {
        Buffer1::None => Buffer1::F64(f64::NAN),
        other => other,
    }
}

/// Count of non-NaN elements in row `r` (NaNs are sorted to the end).
fn valid_count(sorted: &NdArray, r: usize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    macro_rules! count {
        ($v:expr, $isnan:expr) => {{
            let mut c = n;
            while c > 0 && $isnan(&$v[r * n + c - 1]) {
                c -= 1;
            }
            c
        }};
    }
    match sorted.buffer() {
        Buffer::F16(v) => count!(v, |x: &f16| x.is_nan()),
        Buffer::F32(v) => count!(v, |x: &f32| x.is_nan()),
        Buffer::F64(v) => count!(v, |x: &f64| x.is_nan()),
        Buffer::C64(v) => count!(v, |x: &C64| x.re.is_nan() || x.im.is_nan()),
        Buffer::C128(v) => count!(v, |x: &C128| x.re.is_nan() || x.im.is_nan()),
        _ => n, // non-float dtypes never carry NaN
    }
}

/// The ODD-length median: numpy takes a ONE-element slice and still runs it
/// through `mean`, i.e. divides by `rcount == 1`. For every real dtype that
/// division is the identity and this is just `single_element_promoted`; for
/// a complex dtype it is emphatically not -- see `cdiv_by_real`.
fn single_element_as_mean(buf: &Buffer, i: usize, out_dtype: DType) -> Buffer1 {
    match (buf, out_dtype) {
        (Buffer::C64(v), DType::C64) => {
            let (re, im) = cdiv_by_real(v[i].re as f64, v[i].im as f64, 1.0);
            Buffer1::C64(C64::new(re as f32, im as f32))
        }
        (Buffer::C128(v), DType::C128) => {
            let (re, im) = cdiv_by_real(v[i].re, v[i].im, 1.0);
            Buffer1::C128(C128::new(re, im))
        }
        _ => single_element_promoted(buf, i, out_dtype),
    }
}

fn single_element_promoted(buf: &Buffer, i: usize, out_dtype: DType) -> Buffer1 {
    if matches!(buf, Buffer::Bool(_) | Buffer::I8(_) | Buffer::I16(_) | Buffer::I32(_) | Buffer::I64(_) | Buffer::U8(_) | Buffer::U16(_) | Buffer::U32(_) | Buffer::U64(_))
    {
        Buffer1::F64(scalar_to_f64(buf, i))
    } else {
        single_element(buf, i, out_dtype)
    }
}

// ---------------------------------------------------------------------------
// percentile / quantile / nanpercentile / nanquantile
// ---------------------------------------------------------------------------

pub struct QuantileArgs<'a> {
    pub q: &'a [f64],
    pub q_is_scalar: bool,
    pub method: QuantileMethod,
    pub keepdims: bool,
    pub skip_nan: bool,
    /// numpy's own `weak_q`: literally `type(q) in (int, float)` at the
    /// Python boundary -- a bare `int`/`float`, NOT `bool` (excluded by
    /// `type(q) in`, even though `bool` subclasses `int`), NOT a numpy
    /// scalar, NOT a list/array. When set, an INTERPOLATED method keeps the
    /// operand's own float16/float32 dtype instead of promoting to float64.
    /// Direct (take-based) methods preserve dtype regardless and ignore it.
    pub weak_q: bool,
    /// Whether `np.asanyarray(q)` would have an INTEGER or BOOL dtype (a
    /// bare `int`, a `bool`, a numpy integer scalar, or a list/array of
    /// them -- but never a `float`, and never after `percentile`'s own
    /// `q / 100`). Combined with `method="linear"` this triggers numpy's
    /// take-the-element-exactly shortcut; see `direct_index`'s `Linear`
    /// arm.
    pub q_is_integral: bool,
    /// Whether the caller passed `out=None`. Load-bearing for exactly one
    /// numpy behaviour, in `_quantile`'s tail:
    ///     if np.any(slices_having_nans):
    ///         if result.ndim == 0 and out is None:
    ///             result = arr[-1]        # <- operand dtype, not float64
    ///         else:
    ///             np.copyto(result, arr[-1, ...], where=slices_having_nans)
    pub out_is_none: bool,
}

pub fn quantile_axis(a: &NdArray, axis: usize, args: &QuantileArgs) -> Result<NdArray, IonpError> {
    let dtype = a.dtype();
    if dtype == DType::C64 || dtype == DType::C128 {
        return Err(IonpError::Type("a must be an array of real numbers".to_string()));
    }
    // `linear` joins the direct (take-based) methods when `q` is integral;
    // see `direct_index`'s `Linear` arm for numpy's own gate. Everything
    // downstream that keys off `direct` -- the empty-axis message, the
    // bool-subtract TypeError, and both builders -- then follows numpy
    // automatically, because numpy's own control flow branches on exactly
    // this predicate.
    let direct = args.method.is_direct() || (args.q_is_integral && matches!(args.method, QuantileMethod::Linear));
    for &qi in args.q {
        if !(0.0..=1.0).contains(&qi) {
            return Err(IonpError::Value("Quantiles must be in the range [0, 1]".to_string()));
        }
    }

    let rows_info = sort_rows(a, axis)?;
    let n = rows_info.n;
    // See the matching comment in `median_axis`: `rows_info.rows` can be
    // legitimately 0 when some OTHER (non-reduced) axis has length 0 --
    // that produces an empty result, not an error, and must not be forced
    // up to 1 (which previously caused an out-of-bounds panic).
    let rows = rows_info.rows;
    let nq = args.q.len().max(1);

    // Live-verified against real numpy: for an EMPTY reduction axis, the
    // `take`/index lookup fails (IndexError) before the interpolated
    // methods' bool-subtract dtype check is ever reached -- e.g.
    // `np.quantile(np.array([],dtype=bool), 0.5, method="linear")` raises
    // IndexError, not the boolean-subtract TypeError. This ONLY applies to
    // the regular (non-`nan*`) path: `nanquantile`/`nanpercentile` of an
    // empty array silently return NaN for every dtype/method (live-
    // verified), matching `nanmedian`'s own empty-slice behavior.
    //
    // The exact message depends on dtype (int/bool vs float),
    // direct-vs-interpolated method, AND (for int/bool + interpolated
    // only) the SIGN of the first `q` value's virtual index -- per a live
    // sweep of all 13 methods x {bool,int8,uint8,float16,float32,float64}
    // x q in {0, 0.25, 0.5, 0.75, 1}:
    //   * float dtype (any method)                  -> "index -1 is out
    //     of bounds for axis 0 with size 0" (constant)
    //   * int/bool dtype + direct method             -> "cannot do a
    //     non-empty take from an empty axes." (constant)
    //   * int/bool dtype + interpolated method       -> real numpy's
    //     `_get_indexes` clip logic, applied to a size-0 array, computes
    //     `previous_indexes = -1` whenever the method's raw virtual index
    //     `vi` (using n=0) is `>= -1` (i.e. `vi >= n-1`), THEN
    //     unconditionally overwrites that to `0` whenever `vi < 0` (this
    //     second check runs after and always wins when both conditions
    //     hold, which is common near n=0) -- collapsing to: `vi < 0` ->
    //     "index 0 ...", else -> "index -1 ...". Multi-`q` arrays report
    //     using the FIRST `q` value only (numpy raises on the first
    //     offending element it touches).
    if !args.skip_nan && n == 0 {
        let is_float = matches!(dtype, DType::F16 | DType::F32 | DType::F64);
        let msg = if is_float {
            "index -1 is out of bounds for axis 0 with size 0"
        } else if direct {
            "cannot do a non-empty take from an empty axes."
        } else {
            let q0 = args.q.first().copied().unwrap_or(0.5);
            let vi = virtual_index_raw(args.method, 0, q0);
            if vi < 0.0 {
                "index 0 is out of bounds for axis 0 with size 0"
            } else {
                "index -1 is out of bounds for axis 0 with size 0"
            }
        };
        return Err(IonpError::Index(msg.to_string()));
    }
    // Live-verified genuine numpy quirk: `nanquantile`/`nanpercentile` where
    // there is either an EMPTY reduction axis (`n == 0`) OR simply NO ROWS
    // to reduce at all (`rows == 0`, e.g. shape `(2,0,3)` reduced over the
    // size-2 axis -- `n == 2` there, but the OTHER axes already multiply to
    // zero so there is nothing to iterate) short-circuits per output
    // position to a single NaN and DROPS THE `q` DIMENSION ENTIRELY,
    // regardless of how many `q` values were requested or whether `q` was
    // scalar or an array -- `np.nanquantile(np.empty((1,0)), [0.1,0.9],
    // axis=1)` returns shape `(1,)`, not `(2,1)`/`(1,2)`; likewise
    // `np.nanquantile(np.zeros((2,0,3),dtype=bool), [0.1,0.9], axis=-3)`
    // returns shape `(0,3)`. This is distinct from the "all values happen
    // to be NaN but the axis itself is nonempty AND there are rows" case,
    // which DOES respect `q`'s shape normally (verified separately). Not
    // reproduced for the *reason* numpy does it (this is presumably an
    // artifact of `_nanquantile_unchecked`'s empty-slice fast path
    // bypassing the normal q-indexed construction), but the OBSERVABLE
    // shape/value is matched exactly.
    if args.skip_nan && (n == 0 || rows == 0) {
        // Live-verified: this short-circuit preserves the native float
        // dtype for float16/32/64 input (nan of that same width), but
        // promotes bool/int/uint input to float64 -- exactly `nan_scalar`'s
        // rule (there is no way to represent a per-slot NaN in a bool/int
        // buffer, so numpy falls back to float64 there; e.g.
        // `np.nanquantile(np.empty((1,0),dtype=np.float32),0.5,axis=1)`
        // stays float32, but the int32 equivalent comes back float64).
        let buf = match nan_scalar(dtype) {
            Buffer1::F16(v) => Buffer::F16(vec![v; rows]),
            Buffer1::F32(v) => Buffer::F32(vec![v; rows]),
            Buffer1::F64(v) => Buffer::F64(vec![v; rows]),
            Buffer1::C64(v) => Buffer::C64(vec![v; rows]),
            Buffer1::C128(v) => Buffer::C128(vec![v; rows]),
            Buffer1::None => Buffer::F64(vec![f64::NAN; rows]),
        };
        let out_shape = rows_info.other_shape.clone();
        let natural = NdArray::from_buffer(buf, out_shape, crate::array::Order::C)?;
        if args.keepdims {
            // Broadcast stride-0 insert, same measured convention as
            // `median`'s keepdims path above (`insert_newaxis`, NOT
            // `expand_dims` -- see that function's doc comment).
            return crate::creation::insert_newaxis(&natural, &[axis]);
        }
        return Ok(natural);
    }
    // Applies to `nan*` variants too (once past the rows==0/n==0
    // short-circuit above) -- live-verified:
    // `np.nanquantile(np.array([True]),0.5,method="linear")` raises the
    // same boolean-subtract TypeError as plain `quantile` does, because a
    // normal nonempty nanquantile call still runs the same interpolated
    // (gamma/lerp) construction over bool data once NaNs are stripped.
    if !direct && dtype == DType::Bool && n > 0 {
        return Err(IonpError::Type(
            "numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead."
                .to_string(),
        ));
    }

    let mut result_buffer = if args.skip_nan {
        build_nan_variant(&rows_info, args, n, rows, direct)
    } else {
        build_regular(&rows_info, args, n, rows, direct)
    };

    // THE 0-D NaN SUBSTITUTION DTYPE FLIP. numpy's `_quantile` finishes with
    //     if np.any(slices_having_nans):
    //         if result.ndim == 0 and out is None:
    //             result = arr[-1]
    //         else:
    //             np.copyto(result, arr[-1, ...], where=slices_having_nans)
    // The n-d branch writes INTO the already-built (float64, for a
    // non-weak `q`) result and so leaves its dtype alone -- which is what
    // the builders above already produce. The 0-d branch REPLACES the
    // result object with a raw element of `arr`, so the returned scalar
    // carries the OPERAND's dtype: `np.percentile(np.array([1, nan],
    // dtype=np.float16), np.float64(50))` is a float16 NaN, not a float64
    // one, even though the same call on NaN-free data returns float64.
    // Measured live against numpy 2.5.1 on 2026-08-04.
    //
    // `direct` methods never promote in the first place, so the flip is
    // invisible there; only the interpolated builders can be holding a
    // widened buffer at this point.
    if !args.skip_nan
        && args.out_is_none
        && !args.keepdims
        && args.q_is_scalar
        && rows == 1
        && rows_info.other_shape.is_empty()
        && !direct
        && row_has_nan(&rows_info.sorted, n, rows).first().copied().unwrap_or(false)
    {
        result_buffer = match dtype {
            DType::F16 => Buffer::F16(vec![f16::NAN]),
            DType::F32 => Buffer::F32(vec![f32::NAN]),
            other => {
                debug_assert!(matches!(other, DType::F64), "only an inexact dtype can carry a NaN here");
                Buffer::F64(vec![f64::NAN])
            }
        };
    }

    let mut out_shape: Vec<usize> = Vec::new();
    if !args.q_is_scalar {
        out_shape.push(nq);
    }
    out_shape.extend(rows_info.other_shape.iter().copied());
    // The kept axis carries a BROADCAST stride of 0, not a real one -- the
    // same convention `median` already follows here (see the long comment on
    // `median`'s own keepdims path); building the shape with the 1 already in
    // it and letting `from_buffer` assign C strides gave a real stride
    // instead, which 24 sweep cases caught on 2026-08-04 (numpy
    // `((1,6),(0,4))` vs ionp `((1,6),(24,4))`). Build the natural shape,
    // then INSERT the axis, exactly as `median` does.
    let natural = NdArray::from_buffer(result_buffer, out_shape, crate::array::Order::C)?;
    // The nan-aware family does NOT hand back a C-contiguous q-major
    // result the way plain `quantile`/`percentile` does. numpy's
    // `_nanquantile_ureduce_func` computes with the q axis LAST (so the
    // packed buffer is `(other..., nq)` in C order) and then moves that
    // axis to the front, returning a VIEW whose strides are therefore
    // "F-ish". Measured live against numpy 2.5.1 on 2026-08-04, operand
    // `np.arange(24.).reshape(2, 3, 4)`, q=[0.1, 0.9]:
    //
    //     np.quantile(a, q, axis=0).strides    -> (96, 32, 8)   # C, q-major
    //     np.nanquantile(a, q, axis=0).strides -> ( 8, 64, 16)  # q moved
    //     np.nanquantile(a, q, axis=1).strides -> ( 8, 64, 16)
    //     np.nanquantile(a, q, axis=2).strides -> ( 8, 48, 16)
    //
    // and it is dtype-independent (bool/int64/float32 all F-ish), so this
    // is gated on the nan-awareness alone, not on inexactness. A scalar q
    // has no q axis to move, and `axis=None` leaves `other_shape` empty so
    // the move is the identity -- both fall through unchanged, which is
    // also what numpy does (`np.nanquantile(a, q, axis=None).strides` is
    // `(8,)`, same as plain `quantile`). 1686 corpus cases each on
    // `nanquantile`/`nanpercentile` caught this.
    let natural = if args.skip_nan && !args.q_is_scalar && natural.ndim() > 1 {
        let last = natural.ndim() - 1;
        let q_last = crate::creation::moveaxis(&natural, &[0], &[last])?;
        let packed = q_last.to_contiguous_order("C")?;
        crate::creation::moveaxis(&packed, &[last], &[0])?
    } else {
        natural
    };
    if args.keepdims {
        let q_ndim = if args.q_is_scalar { 0 } else { 1 };
        // Broadcast stride-0 insert (same convention as `median`'s
        // keepdims path -- `insert_newaxis`, NOT `expand_dims`).
        return crate::creation::insert_newaxis(&natural, &[q_ndim + axis]);
    }
    Ok(natural)
}

fn build_regular(rows_info: &SortedRows, args: &QuantileArgs, n: usize, rows: usize, direct: bool) -> Buffer {
    let nq = args.q.len().max(1);
    let has_nan = row_has_nan(&rows_info.sorted, n, rows_info.rows);
    let buf = rows_info.sorted.buffer();

    if direct {
        let idxs: Vec<usize> = (0..nq).map(|qi| direct_index(args.method, n, args.q.get(qi).copied().unwrap_or(0.5))).collect();
        let total = nq * rows;
        let mut out = gather_rows!(buf, total, |flat: usize| {
            let qi = flat / rows;
            let r = flat % rows;
            r * n + idxs[qi]
        });
        let nan_positions: Vec<usize> = (0..nq)
            .flat_map(|qi| (0..rows).filter(|&r| has_nan.get(r).copied().unwrap_or(false)).map(move |r| qi * rows + r))
            .collect();
        overwrite_with_nan(&mut out, &nan_positions);
        out
    } else {
        let dtype = rows_info.sorted.dtype();
        let narrow = matches!(
            dtype,
            DType::F16 | DType::F32 | DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::U8 | DType::U16 | DType::U32 | DType::U64
        );
        let f64_sorted = rows_info.sorted.cast_to(DType::F64);
        let fbuf = match f64_sorted.buffer() {
            Buffer::F64(v) => v,
            _ => unreachable!(),
        };
        let idxs: Vec<(usize, usize, f64)> = (0..nq).map(|qi| interp_index(args.method, n, args.q.get(qi).copied().unwrap_or(0.5))).collect();
        // WEAK-`q` NARROW-FLOAT PATH (see `QuantileArgs::weak_q` /
        // `lerp_f32`). Everything stays in the operand's own width, which is
        // both the dtype numpy returns AND the precision it computes in --
        // computing in f64 and casting down at the end is NOT equivalent.
        if args.weak_q && matches!(dtype, DType::F32) {
            let sv = match buf {
                Buffer::F32(v) => v,
                _ => unreachable!(),
            };
            let mut out: Vec<f32> = Vec::with_capacity(nq * rows);
            for &(pi, ni, gamma) in &idxs {
                for r in 0..rows {
                    out.push(if has_nan.get(r).copied().unwrap_or(false) {
                        f32::NAN
                    } else {
                        lerp_f32(sv[r * n + pi], sv[r * n + ni], gamma)
                    });
                }
            }
            return Buffer::F32(out);
        }
        if args.weak_q && matches!(dtype, DType::F16) {
            let sv = match buf {
                Buffer::F16(v) => v,
                _ => unreachable!(),
            };
            let mut out: Vec<f16> = Vec::with_capacity(nq * rows);
            for &(pi, ni, gamma) in &idxs {
                for r in 0..rows {
                    out.push(if has_nan.get(r).copied().unwrap_or(false) {
                        f16::NAN
                    } else {
                        lerp_f16(sv[r * n + pi], sv[r * n + ni], gamma)
                    });
                }
            }
            return Buffer::F16(out);
        }
        let mut out: Vec<f64> = Vec::with_capacity(nq * rows);
        for qi in 0..nq {
            let (pi, ni, gamma) = idxs[qi];
            for r in 0..rows {
                let a = fbuf[r * n + pi];
                let b = fbuf[r * n + ni];
                let v = if has_nan.get(r).copied().unwrap_or(false) {
                    f64::NAN
                } else if narrow {
                    lerp_with_diff(a, b, gamma, native_diff_f64(dtype, buf, r * n + pi, r * n + ni))
                } else {
                    lerp(a, b, gamma)
                };
                out.push(v);
            }
        }
        Buffer::F64(out)
    }
}

fn build_nan_variant(rows_info: &SortedRows, args: &QuantileArgs, n: usize, rows: usize, direct: bool) -> Buffer {
    let nq = args.q.len().max(1);
    let buf = rows_info.sorted.buffer();
    let valid: Vec<usize> = (0..rows).map(|r| valid_count(&rows_info.sorted, r, n)).collect();

    if direct {
        let total = nq * rows;
        let mut out = gather_rows!(buf, total, |flat: usize| {
            let qi = flat / rows;
            let r = flat % rows;
            let m = valid[r].max(1);
            let q = args.q.get(qi).copied().unwrap_or(0.5);
            r * n + direct_index(args.method, m, q).min(m.saturating_sub(1))
        });
        let nan_positions: Vec<usize> = (0..nq).flat_map(|qi| (0..rows).filter(|&r| valid[r] == 0).map(move |r| qi * rows + r)).collect();
        overwrite_with_nan(&mut out, &nan_positions);
        out
    } else {
        let dtype = rows_info.sorted.dtype();
        let narrow = matches!(
            dtype,
            DType::F16 | DType::F32 | DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::U8 | DType::U16 | DType::U32 | DType::U64
        );
        let f64_sorted = rows_info.sorted.cast_to(DType::F64);
        let fbuf = match f64_sorted.buffer() {
            Buffer::F64(v) => v,
            _ => unreachable!(),
        };
        // See the matching block in `build_regular` for why the narrow float
        // dtypes get their own native-precision loop under a weak `q`.
        if args.weak_q && matches!(dtype, DType::F32 | DType::F16) {
            let mut out32: Vec<f32> = Vec::with_capacity(nq * rows);
            let mut out16: Vec<f16> = Vec::with_capacity(nq * rows);
            let is32 = matches!(dtype, DType::F32);
            for qi in 0..nq {
                let q = args.q.get(qi).copied().unwrap_or(0.5);
                for r in 0..rows {
                    let m = valid[r];
                    if m == 0 {
                        if is32 {
                            out32.push(f32::NAN);
                        } else {
                            out16.push(f16::NAN);
                        }
                        continue;
                    }
                    let (pi, ni, gamma) = interp_index(args.method, m, q);
                    match buf {
                        Buffer::F32(sv) => out32.push(lerp_f32(sv[r * n + pi], sv[r * n + ni], gamma)),
                        Buffer::F16(sv) => out16.push(lerp_f16(sv[r * n + pi], sv[r * n + ni], gamma)),
                        _ => unreachable!(),
                    }
                }
            }
            return if is32 { Buffer::F32(out32) } else { Buffer::F16(out16) };
        }
        let mut out: Vec<f64> = Vec::with_capacity(nq * rows);
        for qi in 0..nq {
            let q = args.q.get(qi).copied().unwrap_or(0.5);
            for r in 0..rows {
                let m = valid[r];
                if m == 0 {
                    out.push(f64::NAN);
                    continue;
                }
                let (pi, ni, gamma) = interp_index(args.method, m, q);
                let a = fbuf[r * n + pi];
                let b = fbuf[r * n + ni];
                let v = if narrow { lerp_with_diff(a, b, gamma, native_diff_f64(dtype, buf, r * n + pi, r * n + ni)) } else { lerp(a, b, gamma) };
                out.push(v);
            }
        }
        // THE FIRST-SLICE DTYPE QUIRK. `nanquantile`/`nanpercentile` reduce
        // via `np.apply_along_axis(_nanquantile_1d, ...)`, and
        // `apply_along_axis` sizes its output buffer from the FIRST slice
        // alone: `res = asanyarray(func1d(inarr_view[ind0])); buff =
        // zeros(..., res.dtype)`. `_nanquantile_1d` returns
        // `np.full(q.shape, np.nan, dtype=a.dtype)` for an ALL-NaN slice but
        // delegates to the float64-producing `_quantile_unchecked` otherwise
        // -- so with a non-weak `q` the ENTIRE result is float32/float16 iff
        // slice 0 happens to be all-NaN, and float64 otherwise, with every
        // other slice cast into whichever the first one picked.
        //
        // Measured live 2026-08-04 on numpy 2.5.1, both directions:
        //   np.nanquantile(float32 [[nan,nan,nan],[1,2,3]], [.1,.9], axis=1)
        //     -> float32
        //   np.nanquantile(float32 [[1,2,3],[nan,nan,nan]], [.1,.9], axis=1)
        //     -> float64
        // This is a genuine numpy artifact rather than a documented rule; it
        // is reproduced, not repaired. The weak-`q` path above never reaches
        // here and is unaffected (it is float32/float16 unconditionally,
        // which is also what numpy does).
        //
        // The values stay float64-computed and are only cast at the end --
        // that is exactly what `apply_along_axis`'s own `buff[ind] = res`
        // assignment does to the later, float64 slices.
        if !args.weak_q && valid.first().copied() == Some(0) {
            match dtype {
                DType::F32 => return Buffer::F32(out.into_iter().map(|v| v as f32).collect()),
                DType::F16 => return Buffer::F16(out.into_iter().map(f16::from_f64).collect()),
                _ => {}
            }
        }
        Buffer::F64(out)
    }
}
