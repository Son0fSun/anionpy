//! Numeric kernels for numpy's LEGACY polynomial API
//! (`numpy.poly1d`/`numpy.polyval`/`numpy.polyadd`/... -- `numpy/lib/
//! _polynomial_impl.py`, NOT `numpy.polynomial.polynomial`, which
//! `poly.rs` already covers and which a different, concurrently-edited
//! part of this codebase owns).
//!
//! The legacy family uses the OPPOSITE coefficient convention from the
//! modern power-series module: `p[0]` is the HIGHEST-degree coefficient
//! and `p[-1]` is the constant term (`p[0]*x**(N-1) + ... + p[N-1]`),
//! vs. modern `c[0]` being the constant term. That reversal, plus
//! several genuinely different algorithms/edge-case rules (no trimming
//! in `polyadd`/`polysub`, an `allclose`-based remainder trim in
//! `polydiv` instead of exact-zero, a different companion-matrix layout
//! in `roots`), means the kernels below are NOT simply "call `poly.rs`
//! with a reversed array" wrappers -- ported directly from numpy
//! 2.5.1's own `_polynomial_impl.py` loops instead (paraphrased, not
//! copied verbatim; see `ionp/docs/TICKET-72-POLY1D-2026-08-08.md`).
//!
//! `PolyScalar` (from `poly.rs`) is reused as-is: the same real/complex
//! element-type seam applies here unchanged.

use crate::buffer::C128;
use crate::poly::{convolve_full, PolyScalar};

/// Local, minimal magnitude seam (kept separate from `poly.rs`'s own
/// `PolyScalar` trait deliberately -- `poly.rs` is concurrently edited
/// by another part of this task and this file must not need to touch
/// it). Only used by `polydiv_legacy`'s `allclose(r[0], 0, ...)` trim.
trait Magnitude {
    fn magnitude(self) -> f64;
}
impl Magnitude for f64 {
    fn magnitude(self) -> f64 {
        self.abs()
    }
}
impl Magnitude for C128 {
    fn magnitude(self) -> f64 {
        self.norm()
    }
}

/// `polyval`: Horner's method starting from a LITERAL zero accumulator
/// (`y = zeros_like(x)`, then `for pv in p: y = y*x + pv`), NOT from
/// `p[0]` directly -- verified to matter for `x` values where `0 * x !=
/// 0` exactly (`inf`, `nan`); starting the accumulator at `p[0]` instead
/// would silently skip that first `0 * x` term and diverge from real
/// numpy on exactly those inputs. Evaluated independently per point in
/// `xs`.
pub fn polyval_legacy<T: PolyScalar>(p: &[T], xs: &[T]) -> Vec<T> {
    xs.iter()
        .map(|&x| {
            let mut y = T::zero();
            for &pv in p {
                y = y.poly_mul(x) + pv;
            }
            y
        })
        .collect()
}

/// `polyval`'s pure-int64 fast path: legacy numpy does NOT force
/// int/bool coefficients through float (`np.polyval(np.array([1,2,3]),
/// 2)` stays `int64` -- measured directly, see
/// `ionp-py/src/poly_legacy.rs`'s `coerce` doc comment). Wrapping
/// arithmetic matches numpy's own C-level signed-integer overflow
/// (practically two's-complement wraparound on every platform this
/// codebase targets, same assumption already made elsewhere in this
/// crate for integer ufuncs).
pub fn polyval_legacy_i64(p: &[i64], xs: &[i64]) -> Vec<i64> {
    xs.iter()
        .map(|&x| {
            let mut y: i64 = 0;
            for &pv in p {
                y = y.wrapping_mul(x).wrapping_add(pv);
            }
            y
        })
        .collect()
}

/// `polyder`'s pure-int64 fast path: like `polyval_legacy_i64`, legacy
/// numpy keeps `polyder`'s coefficients int64 when the input is int64
/// (multiplication-only, no division, so no float promotion is ever
/// needed -- verified directly: `np.polyder(np.poly1d([1,1,1,1])).
/// coeffs.dtype` is `int64`). Wrapping multiply/subtract for the same
/// two's-complement-overflow reason as `polyval_legacy_i64`.
pub fn polyder_legacy_i64(p: &[i64], m: usize) -> Vec<i64> {
    let mut c = p.to_vec();
    for _ in 0..m {
        if c.is_empty() {
            continue;
        }
        let n = c.len() - 1;
        let mut y = vec![0i64; n];
        for i in 0..n {
            y[i] = c[i].wrapping_mul((n - i) as i64);
        }
        c = y;
    }
    c
}

/// `polyder`, single order. Mirrors numpy's own recursive `polyder(y, m
/// - 1)` structure, including the empty-input fixed point: once `p` is
/// empty it differentiates to itself forever, matching `p[:-1] *
/// arange(n, 0, -1)` on an empty `p` staying empty.
pub fn polyder_legacy_once<T: PolyScalar>(p: &[T]) -> Vec<T> {
    if p.is_empty() {
        return Vec::new();
    }
    let n = p.len() - 1;
    let mut y = vec![T::zero(); n];
    for i in 0..n {
        y[i] = p[i].poly_mul(T::from_usize(n - i));
    }
    y
}

/// `polyder`, `m`-fold: applies `polyder_legacy_once` `m` times, exactly
/// mirroring numpy's own recursive call structure (order of operations
/// matters here: each stage's *output* dtype/values feed the next
/// stage's input, not `m` independent single-order derivatives).
pub fn polyder_legacy<T: PolyScalar>(p: &[T], m: usize) -> Vec<T> {
    let mut c = p.to_vec();
    for _ in 0..m {
        c = polyder_legacy_once(&c);
    }
    c
}

/// `polyint`, single order: `y = concatenate((p / arange(len(p), 0,
/// -1), [k]))` -- divides each coefficient by its (1-indexed-from-the-
/// end) position, then appends the new constant term `k`.
pub fn polyint_legacy_once<T: PolyScalar>(p: &[T], k: T) -> Vec<T> {
    let n = p.len();
    let mut y = vec![T::zero(); n + 1];
    for j in 0..n {
        y[j] = p[j].poly_div(T::from_usize(n - j));
    }
    y[n] = k;
    y
}

/// `polyint`, `m`-fold: applies `polyint_legacy_once` once per entry of
/// `k` (`k.len()` IS `m` here -- the PyO3 binding is responsible for
/// building the zero-padded-per-numpy's-own-rule `k` array before
/// calling this, matching numpy's `k = list(k) + [0] * (cnt -
/// len(k))`/single-scalar-broadcast handling).
pub fn polyint_legacy<T: PolyScalar>(p: &[T], k: &[T]) -> Vec<T> {
    let mut c = p.to_vec();
    for &ki in k {
        c = polyint_legacy_once(&c, ki);
    }
    c
}

/// `polydiv`: numpy's own scale-and-subtract loop (NOT the synthetic
/// division `poly.rs::div` uses for the modern module -- a genuinely
/// different algorithm, ported separately), followed by numpy's
/// `while allclose(r[0], 0, rtol=1e-14) and r.shape[-1] > 1: r = r[1:]`
/// remainder-trim loop. `allclose(a, 0, rtol=1e-14)` with `atol` left at
/// its default (`1e-8`) reduces to `abs(a) <= 1e-8` (the `rtol * |b|`
/// term is `rtol * 0 == 0`) -- a real, deliberate TOLERANCE trim, not an
/// exact-zero one; reproduced here exactly, not approximated further.
/// `v[0] == 0` is not guarded here (mirrors numpy's own unguarded `1. /
/// v[0]`, which raises Python's own `ZeroDivisionError`/produces `inf`
/// exactly like a bare float division would -- the PyO3 binding lets
/// that division run rather than special-casing it, so `f64` produces
/// `inf`/`nan` and the binding surfaces whatever numpy itself would).
pub fn polydiv_legacy<T: PolyScalar + Magnitude>(u: &[T], v: &[T]) -> (Vec<T>, Vec<T>) {
    let m = u.len() - 1;
    let n = v.len() - 1;
    let scale = T::one().poly_div(v[0]);
    let qlen = if m >= n { m - n + 1 } else { 0 }.max(1);
    let mut q = vec![T::zero(); qlen];
    let mut r = u.to_vec();
    if m >= n {
        for k in 0..=(m - n) {
            let d = scale.poly_mul(r[k]);
            q[k] = d;
            for (idx, &vv) in v.iter().enumerate() {
                r[k + idx] = r[k + idx] - d.poly_mul(vv);
            }
        }
    }
    while is_close_to_zero(r[0]) && r.len() > 1 {
        r.remove(0);
    }
    (q, r)
}

fn is_close_to_zero<T: PolyScalar + Magnitude>(v: T) -> bool {
    // `abs(v) <= 1e-8`, expressed generically via `magnitude` below
    // (real: `|v|`; complex: `|v|` the modulus -- `NX.allclose` on a
    // complex scalar compares magnitudes the same way `np.abs` does).
    v.magnitude() <= 1e-8
}

/// `poly()`'s coefficient-from-roots construction: `a = [1]`, then `a =
/// convolve(a, [1, -zero])` per root, sequentially. Every convolution
/// step has a length-2 kernel, so each output tap is a sum of AT MOST 2
/// products -- no summation-order ambiguity is possible for a 2-term
/// add (ticket #45's open "convolve summation order" question concerns
/// reductions over 3+ overlapping terms, which this per-step shape
/// never produces), so reusing `poly.rs::convolve_full` here does NOT
/// inherit #45.
pub fn poly_from_roots_legacy<T: PolyScalar>(roots: &[T]) -> Vec<T> {
    let mut a = vec![T::one()];
    for &z in roots {
        a = convolve_full(&a, &[T::one(), -z]);
    }
    a
}

/// `roots()`'s companion matrix: `A = diag(ones(N-2), -1); A[0, :] =
/// -p[1:] / p[0]`. Deliberately a DIFFERENT layout from
/// `poly.rs::companion` (which builds the modern module's own
/// companion matrix with the correction column on the right, not the
/// top row) -- ported separately rather than reused because the two
/// really are different matrix constructions, not the same one
/// transposed. `p` must already be trimmed (`len(p) >= 2`, `p[0] !=
/// 0`) by the caller. Row-major `n x n`, `n = len(p) - 1`.
pub fn companion_legacy<T: PolyScalar>(p: &[T]) -> Vec<T> {
    let n = p.len() - 1;
    let mut mat = vec![T::zero(); n * n];
    for i in 0..n.saturating_sub(1) {
        mat[(i + 1) * n + i] = T::one();
    }
    let p0 = p[0];
    for j in 0..n {
        mat[j] = -(p[j + 1].poly_div(p0));
    }
    mat
}

// ---------------------------------------------------------------------
// N-D fallback paths for polyval / polyder / polyint (ticket #76)
// ---------------------------------------------------------------------
//
// Real numpy's own `polyval`/`polyder`/`polyint` (`_polynomial_impl.py`)
// place NO ndim restriction on `p` at all: `polyval` Horners over `p`'s
// FIRST axis (`for pv in p: y = y*x + pv`); `polyder`/`polyint` slice
// and broadcast-multiply/divide against `arange(...)` the same way
// regardless of `p.ndim`. ionp's own 1-D fast paths above
// (`polyval_legacy`/`polyder_legacy`/`polyint_legacy`) stay exactly as
// they are -- already independently verified bit-exact -- and remain
// the path used whenever `p.ndim() == 1` (see `ionp-py/src/
// poly_legacy.rs`'s dispatch). The three functions below are the
// fallback for every OTHER shape (0-d and ndim>=2), built by calling
// the same generic, already-shipped, already-broadcast-correct
// `NdArray`-level primitives (`ufunc::binary_op`, `manip::concatenate`,
// `NdArray::get_view`) real numpy's own pure-Python implementation
// itself composes -- not a hand-rolled broadcast/error-message
// reimplementation. `ufunc::binary_op`'s `IonpError::Broadcast` and
// `manip::concatenate`'s ndim-mismatch message are ALREADY verified
// byte-for-byte against real numpy elsewhere in this crate, so reusing
// them here reproduces numpy's own (quite peculiar, shape-dependent)
// success/failure pattern exactly, including the "computed but
// discarded" broadcast in `polyder`'s own `m == 0` base case -- see
// that function's doc comment for why the computation is not skipped
// even when its result is thrown away.
//
// The `p.ndim() == 0` (bare scalar / 0-d array) TypeError case is
// deliberately NOT handled here: numpy's `TypeError: len() of unsized
// object` comes from Python's own `len()` builtin, and anionpy's array
// wrapper already raises that exact message for a 0-d array (verified
// directly: `len(anionpy.array(5))` raises `TypeError: len() of
// unsized object`) -- so the PyO3 binding calls `len()` from the
// Python dispatch layer instead of duplicating the check here. That is
// dispatch/validation, not arithmetic, and needs no Rust counterpart.

use crate::array::{NdArray, Order, SliceItem};
use crate::buffer::Buffer;
use crate::error::IonpError;
use crate::manip;
use crate::ufunc::{self, BinaryOp};

/// `numpy.arange(n, 0, -1)`, as an int64 `NdArray` (numpy's own default
/// dtype for an all-Python-int `arange` call, matching this same file's
/// `polyder`/`polyint` doc comments on the 1-D fast path above). `n`
/// may be negative or zero, giving an empty range, matching `numpy.
/// arange`'s own convention for a start not "past" the stop in the
/// step's direction (`arange(0, 0, -1)` and `arange(-1, 0, -1)` are
/// both empty).
fn arange_desc_i64(n: isize) -> NdArray {
    let len = crate::creation::arange_len(n as f64, 0.0, -1.0);
    let vals: Vec<i64> = crate::creation::arange_values(n as f64, -1.0, len)
        .into_iter()
        .map(|v| v as i64)
        .collect();
    NdArray::from_buffer(Buffer::I64(vals), vec![len], Order::C)
        .expect("1-d int64 buffer/shape always agree")
}

/// `numpy.polyval`'s Horner loop, generalized to any `p.ndim()`. Caller
/// guarantees `p.ndim() >= 1` and `p.shape()[0] > 0` -- the `ndim() ==
/// 0` (iterating a 0-d array raises numpy's own `TypeError: iteration
/// over a 0-d array`) and `shape()[0] == 0` (numpy's loop just never
/// runs, leaving `y` as `zeros_like(x)`) cases are both handled by the
/// Python dispatch layer, which already had a correct, unchanged
/// `zeros_like`-based branch for the latter before this ticket. `y`
/// starts as `zeros_like(x)` (x's shape AND dtype, not p's -- matches
/// `NX.zeros_like(x)`), then each slice along `p`'s first axis is
/// folded in via `y = y*x + pv`, exactly mirroring numpy's own loop
/// body, with every actual multiply/add delegated to `ufunc::
/// binary_op` (full numpy broadcasting + dtype promotion).
pub fn polyval_legacy_horner_nd(p: &NdArray, x: &NdArray) -> Result<NdArray, IonpError> {
    let zero_buf = crate::creation::zeros_buffer(x.dtype(), x.size());
    let mut y = NdArray::from_buffer(zero_buf, x.shape().to_vec(), Order::C)?;
    let n0 = p.shape()[0];
    for i in 0..n0 {
        let pv = p.get_view(&[SliceItem::Index(i as isize)])?;
        y = ufunc::binary_op(BinaryOp::Multiply, &y, x)?;
        y = ufunc::binary_op(BinaryOp::Add, &y, &pv)?;
    }
    Ok(y)
}

/// `numpy.polyder`'s recursive body, generalized to any `p.ndim()`:
/// `n = len(p) - 1; y = p[:-1] * arange(n, 0, -1); return p if m == 0
/// else polyder(y, m - 1)`. `y` is computed UNCONDITIONALLY on every
/// call, including the `m == 0` base case whose result it then
/// discards -- reproduced faithfully here (not skipped as a "harmless"
/// optimization) because that stray computation is exactly what makes
/// some `m == 0`, ndim>=2 shapes raise where the same shape at `m == 1`
/// would not (measured: `np.polyder(np.array(5), 0)` still raises
/// `TypeError: len() of unsized object`, even though `m == 0` "should"
/// be a no-op). Caller guarantees `p.ndim() >= 1` (0-d is handled by
/// the Python dispatch layer's own `len()` call, per this module's doc
/// comment above).
pub fn polyder_legacy_nd(p: &NdArray, m: i64) -> Result<NdArray, IonpError> {
    let n = p.shape()[0] as isize - 1;
    let sliced = p.get_view(&[SliceItem::Slice { start: None, stop: Some(-1), step: None }])?;
    let ar = arange_desc_i64(n);
    let y = ufunc::binary_op(BinaryOp::Multiply, &sliced, &ar)?;
    if m == 0 {
        Ok(p.clone())
    } else {
        polyder_legacy_nd(&y, m - 1)
    }
}

/// `numpy.polyint`'s recursive body: `if m == 0: return p` (no
/// computation at all -- unlike `polyder` above, the short-circuit here
/// really does skip everything, so a 0-d `p` at `m == 0` succeeds,
/// returning `p` unchanged, matching `np.polyint(np.array(5), 0)`
/// measured directly against numpy 2.5.1); else `y = concatenate((p /
/// arange(len(p), 0, -1), [k[0]])); return polyint(y, m - 1, k[1:])`.
/// `k` is the already-length-`m`, already-numpy-`atleast_1d`-normalized
/// constant list built by the Python dispatch layer (see
/// `_polynomial_legacy.py`'s `polyint`), consumed one entry per
/// recursion level in the same order numpy's own slice `k[1:]` would.
/// Caller guarantees `p.ndim() >= 1` whenever `m > 0` reaches here (see
/// this module's doc comment on why `m == 0` needs no such guarantee).
pub fn polyint_legacy_nd(p: &NdArray, m: i64, k: &[f64]) -> Result<NdArray, IonpError> {
    if m == 0 {
        return Ok(p.clone());
    }
    let n = p.shape()[0] as isize;
    let ar = arange_desc_i64(n);
    let divided = ufunc::binary_op(BinaryOp::Divide, p, &ar)?;
    let k0 = NdArray::from_buffer(Buffer::F64(vec![k[0]]), vec![1], Order::C)?;
    let y = manip::concatenate(&[&divided, &k0], Some(0))?;
    polyint_legacy_nd(&y, m - 1, &k[1..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polyval_matches_numpy_example() {
        // np.polyval([3,0,1], 5) == 76
        assert_eq!(polyval_legacy(&[3.0, 0.0, 1.0], &[5.0]), vec![76.0]);
    }

    #[test]
    fn polyder_matches_numpy_example() {
        // np.polyder([1,1,1,1]) == [3,2,1]
        assert_eq!(polyder_legacy_once(&[1.0, 1.0, 1.0, 1.0]), vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn polyint_matches_numpy_example() {
        // np.polyint([1,2,3]) == [0.33333333, 1., 3., 0.]  (p=[1,2,3] deg2)
        let got = polyint_legacy_once(&[1.0, 2.0, 3.0], 0.0);
        assert_eq!(got, vec![1.0 / 3.0, 2.0 / 2.0, 3.0 / 1.0, 0.0]);
    }

    #[test]
    fn polydiv_matches_numpy_example() {
        // np.polydiv([3.,5.,2.],[2.,1.]) == ([1.5,1.75],[0.25])
        let (q, r) = polydiv_legacy(&[3.0, 5.0, 2.0], &[2.0, 1.0]);
        assert_eq!(q, vec![1.5, 1.75]);
        assert_eq!(r, vec![0.25]);
    }

    #[test]
    fn poly_from_roots_matches_numpy_example() {
        // np.poly((-1/2, 0, 1/2)) == [1, 0, -0.25, 0]
        let got = poly_from_roots_legacy(&[-0.5, 0.0, 0.5]);
        assert_eq!(got, vec![1.0, 0.0, -0.25, 0.0]);
    }

    #[test]
    fn companion_legacy_matches_numpy_example() {
        // np.roots([3.2,2,1])'s companion: A[0,:] = -[2,1]/3.2
        let m = companion_legacy(&[3.2, 2.0, 1.0]);
        assert_eq!(m, vec![-2.0 / 3.2, -1.0 / 3.2, 1.0, 0.0]);
    }
}
