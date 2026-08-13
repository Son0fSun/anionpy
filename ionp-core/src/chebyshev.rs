//! Chebyshev-series (first kind) polynomial kernels — the Rust core for
//! `numpy.polynomial.chebyshev`. UNLIKE `crate::hermite`/`crate::hermite_e`/
//! `crate::laguerre`/`crate::legendre`, which all share a three-term
//! recurrence throughout, Chebyshev's `chebmul`/`chebdiv`/`chebpow` are
//! built on a "z-series" (symmetric Laurent-series) representation via
//! private helpers ported from `numpy/polynomial/chebyshev.py`:
//! `_cseries_to_zseries`, `_zseries_to_cseries`, `_zseries_mul` (literally
//! `np.convolve`), `_zseries_div` (a two-pointer synthetic-division-style
//! loop). Every one of these is verified against the literal numpy source
//! (numpy 2.5.1, `.venv/lib/python3.14/site-packages/numpy/polynomial/
//! chebyshev.py`).
//!
//! GENUINE GAP in this task's own brief: numpy's source ALSO defines
//! `_zseries_der`/`_zseries_int`, which this file ports below
//! (`zseries_der`/`zseries_int`) for completeness per the brief's explicit
//! instruction — but they are DEAD CODE in real numpy. `chebder`/`chebint`
//! do **NOT** call them; both use their own distinct direct
//! array-recurrence algorithms with no z-series involvement whatsoever
//! (verified directly against `chebyshev.py`'s `chebder`/`chebint`
//! bodies). Only `chebmul`, `chebdiv`, and `chebpow` actually use the
//! z-series machinery. `zseries_der`/`zseries_int` below are therefore
//! unused by every public function in this module and carry
//! `#[allow(dead_code)]` — ported faithfully (including the `*2`/`/2`
//! Decimal-compatibility scaling trick documented in numpy's own
//! docstring) but never exercised by the differential corpus, since
//! nothing in numpy's own public API calls them either.
//!
//! Divergence traps from the sibling bases, each independently verified:
//! - `chebval`'s Clenshaw general-loop combine step is `c0 = c[-i] - c1`
//!   — NO multiplicative coefficient on `c1` at all (contrast every
//!   sibling basis, which all multiply `c1` by some `nd`-derived factor
//!   before subtracting). `x2 = 2*x` IS precomputed and used in the OTHER
//!   term: `c1 = tmp + c1*x2`.
//! - `chebmulx` is a DIRECT recurrence, NOT z-series based, despite
//!   `chebmul` (its neighbor in numpy's source) being z-series based.
//! - `chebder`'s `j`-loop counts DOWN from `n` to `3` inclusive
//!   (`range(n, 2, -1)`), mutating `c` in place
//!   (`c[j-2] += (j*c[j])/(j-2)`) — later (smaller-`j`) iterations read
//!   `c[j-2]` values already mutated by earlier (larger-`j`) iterations,
//!   so this is genuinely order-dependent accumulation.
//! - `chebcompanion`'s `scl` vector is `[1., sqrt(.5), sqrt(.5), ...]` —
//!   flat, NOT built via `multiply.accumulate` like every sibling basis's
//!   `scl`. The last-column correction is an array-valued
//!   `(c[:-1]/c[-1]) * (scl/scl[-1]) * .5` multiply, preserved as a
//!   literal array op (`scl[i]/scl[-1]` computed per-index) rather than
//!   algebraically simplified.

use crate::buffer::C128;
use crate::poly::{trim_trailing_zeros, PolyScalar};

/// Same seam as `crate::hermite_e::HermeScalar`: real `f64`-from-magnitude
/// construction plus `complex_div`/`complex_mul_fma`-routed
/// division/multiplication so the `C128` instantiation stays bit-exact
/// against numpy's `npy_cdivide`/FMA-based complex multiply rather than
/// `num_complex::Complex`'s native (measurably different) `Div`/`Mul`
/// impls. Declared independently — this file must never import anything
/// from `hermite.rs`/`hermite_e.rs`/`laguerre.rs`/`legendre.rs`.
///
/// Deliberately NOT using `crate::poly::convolve_full` for the z-series
/// multiply below even though `_zseries_mul` is literally `np.convolve`:
/// `convolve_full` is generic over `PolyScalar`'s bare `std::ops::Mul`,
/// which for `C128` is `num_complex::Complex`'s naive (non-FMA) multiply
/// — independently confirmed (see `ionp-core/src/ufunc.rs`'s
/// `complex_mul_fma` doc comment: ~44% mismatch rate vs real numpy) to be
/// the exact root cause `polynomial.polymul` was REVOKED for. This
/// module's own `zseries_mul` below routes every multiply through
/// `ChebScalar::cheb_mul` instead.
pub trait ChebScalar: PolyScalar {
    fn from_f64(v: f64) -> Self;
    fn cheb_div(self, other: Self) -> Self;
    fn cheb_mul(self, other: Self) -> Self;
}

impl ChebScalar for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn cheb_div(self, other: Self) -> Self {
        self / other
    }
    fn cheb_mul(self, other: Self) -> Self {
        self * other
    }
}

impl ChebScalar for C128 {
    fn from_f64(v: f64) -> Self {
        C128::new(v, 0.0)
    }
    fn cheb_div(self, other: Self) -> Self {
        crate::ufunc::complex_div(self, other)
    }
    fn cheb_mul(self, other: Self) -> Self {
        crate::ufunc::complex_mul_fma(self, other)
    }
}

/// Padded elementwise add, mirroring numpy's `polyutils._add` exactly —
/// same branch-for-branch shape as `crate::hermite_e::pad_add`'s doc
/// comment (independently declared here, not imported).
fn pad_add<T: ChebScalar>(a: &[T], b: &[T]) -> Vec<T> {
    if a.len() > b.len() {
        let mut out = a.to_vec();
        for (i, &v) in b.iter().enumerate() {
            out[i] = out[i] + v;
        }
        out
    } else {
        let mut out = b.to_vec();
        for (i, &v) in a.iter().enumerate() {
            out[i] = out[i] + v;
        }
        out
    }
}

/// Padded elementwise subtract, mirroring numpy's `polyutils._sub`.
fn pad_sub<T: ChebScalar>(a: &[T], b: &[T]) -> Vec<T> {
    if a.len() > b.len() {
        let mut out = a.to_vec();
        for (i, &v) in b.iter().enumerate() {
            out[i] = out[i] - v;
        }
        out
    } else {
        let mut out: Vec<T> = b.iter().map(|&v| -v).collect();
        for (i, &v) in a.iter().enumerate() {
            out[i] = out[i] + v;
        }
        out
    }
}

pub use crate::poly::add_trim as chebadd_trim;
pub use crate::poly::sub_trim as chebsub_trim;

// ───────────────────────── z-series machinery ─────────────────────────

/// `_cseries_to_zseries`: `zs = zeros(2n-1); zs[n-1:] = c/2; return zs +
/// zs[::-1]`. Ported literally, including the two-pass shape (build the
/// half-populated array, then combine with its own reverse) rather than
/// the mathematically-equal direct symmetric fill, to preserve numpy's
/// exact floating-point op sequence (`c/2` computed once, then ADDED to
/// its mirror — not multiplied out any other way).
pub fn cseries_to_zseries<T: ChebScalar>(c: &[T]) -> Vec<T> {
    let n = c.len();
    let mut zs = vec![T::zero(); 2 * n - 1];
    for (i, &v) in c.iter().enumerate() {
        zs[n - 1 + i] = v.cheb_div(T::from_f64(2.0));
    }
    let mut out = vec![T::zero(); 2 * n - 1];
    for i in 0..2 * n - 1 {
        out[i] = zs[i] + zs[2 * n - 2 - i];
    }
    out
}

/// `_zseries_to_cseries`: `n = (len(zs)+1)//2; c = zs[n-1:].copy(); c[1:n]
/// *= 2; return c`.
pub fn zseries_to_cseries<T: ChebScalar>(zs: &[T]) -> Vec<T> {
    let n = (zs.len() + 1) / 2;
    let mut c: Vec<T> = zs[n - 1..].to_vec();
    for item in c.iter_mut().take(n).skip(1) {
        *item = item.cheb_mul(T::from_f64(2.0));
    }
    c
}

/// `_zseries_mul`: literally `np.convolve(z1, z2)`. NOT `poly::
/// convolve_full` (see this module's doc comment) — routes every term
/// product through `ChebScalar::cheb_mul` so the `C128` instantiation is
/// FMA-consistent with real numpy.
pub fn zseries_mul<T: ChebScalar>(a: &[T], b: &[T]) -> Vec<T> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let n = a.len() + b.len() - 1;
    let mut out = vec![T::zero(); n];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            // numpy: `av * bv` inside `np.convolve` -- `av` (left operand
            // of the outer loop) first.
            out[i + j] = out[i + j] + av.cheb_mul(bv);
        }
    }
    out
}

/// `_zseries_div`: two-pointer synthetic-division-style loop, ported
/// literally from `chebyshev.py`'s body (see this module's doc comment
/// for the full transcription this follows). `lc2 == 1` and `lc1 < lc2`
/// short-circuit branches match numpy's own. The general branch's
/// `i - 1 + lc2` index (numpy: `i - 1`) is reordered to `i + lc2 - 1` to
/// avoid `usize` underflow at `i == 0` — same VALUE, associativity is not
/// load-bearing here since these are plain index arithmetic, not
/// floating-point ops.
pub fn zseries_div<T: ChebScalar>(z1_in: &[T], z2_in: &[T]) -> (Vec<T>, Vec<T>) {
    let mut z1 = z1_in.to_vec();
    let mut z2 = z2_in.to_vec();
    let lc1 = z1.len();
    let lc2 = z2.len();
    if lc2 == 1 {
        for v in z1.iter_mut() {
            *v = v.cheb_div(z2[0]);
        }
        let rem = vec![z1[0].cheb_mul(T::zero())];
        return (z1, rem);
    }
    if lc1 < lc2 {
        let quo = vec![z1[0].cheb_mul(T::zero())];
        return (quo, z1);
    }
    let dlen = lc1 - lc2;
    let scl = z2[0];
    for v in z2.iter_mut() {
        *v = v.cheb_div(scl);
    }
    let mut quo = vec![T::zero(); dlen + 1];
    let mut i = 0usize;
    let mut j = dlen;
    while i < j {
        let r = z1[i];
        quo[i] = r;
        quo[dlen - i] = r;
        let tmp: Vec<T> = z2.iter().map(|&v| r.cheb_mul(v)).collect();
        for (k, &t) in tmp.iter().enumerate() {
            z1[i + k] = z1[i + k] - t;
        }
        for (k, &t) in tmp.iter().enumerate() {
            z1[j + k] = z1[j + k] - t;
        }
        i += 1;
        j -= 1;
    }
    let r = z1[i];
    quo[i] = r;
    let tmp: Vec<T> = z2.iter().map(|&v| r.cheb_mul(v)).collect();
    for (k, &t) in tmp.iter().enumerate() {
        z1[i + k] = z1[i + k] - t;
    }
    for v in quo.iter_mut() {
        *v = v.cheb_div(scl);
    }
    let rem = z1[i + 1..i + lc2 - 1].to_vec();
    (quo, rem)
}

/// `_zseries_der` — DEAD CODE in real numpy (see this module's doc
/// comment): never called by the public `chebder`. Ported for
/// completeness per this task's brief, not exercised by the differential
/// corpus.
#[allow(dead_code)]
pub fn zseries_der<T: ChebScalar>(zs_in: &[T]) -> Vec<T> {
    let n = zs_in.len() / 2;
    let ns = [-T::one(), T::zero(), T::one()];
    let mut zs = zs_in.to_vec();
    for (k, v) in zs.iter_mut().enumerate() {
        let coeff = T::from_f64(((k as isize - n as isize) as f64) * 2.0);
        *v = v.cheb_mul(coeff);
    }
    let (d, _r) = zseries_div(&zs, &ns);
    d
}

/// `_zseries_int` — DEAD CODE in real numpy (see this module's doc
/// comment): never called by the public `chebint`. Ported for
/// completeness per this task's brief, not exercised by the differential
/// corpus.
#[allow(dead_code)]
pub fn zseries_int<T: ChebScalar>(zs_in: &[T]) -> Vec<T> {
    let n = 1 + zs_in.len() / 2;
    let ns = [-T::one(), T::zero(), T::one()];
    let mut zs = zseries_mul(zs_in, &ns);
    let div: Vec<f64> = (0..zs.len()).map(|k| ((k as isize - n as isize) as f64) * 2.0).collect();
    for k in 0..n {
        zs[k] = zs[k].cheb_div(T::from_f64(div[k]));
    }
    for k in (n + 1)..zs.len() {
        zs[k] = zs[k].cheb_div(T::from_f64(div[k]));
    }
    zs[n] = T::zero();
    zs
}

// ───────────────────────── chebmulx / mul / div / pow ─────────────────────────

/// `chebmulx`: DIRECT recurrence, NOT z-series based (see module doc
/// comment). `prd[0] = c[0]*0; prd[1] = c[0]`; if `len(c) > 1`:
/// `tmp = c[1:]/2; prd[2:] = tmp; prd[0:-2] += tmp`.
///
/// Ticket #84 (2026-08-08, Monday): trims `c` via `trim_trailing_zeros`
/// FIRST, matching numpy's own `[c] = pu.as_series([c])` -- same fix, same
/// root cause, same measured shape+sign defect as `legendre::legmulx`'s
/// identical fix; see that function's doc comment for the full trace.
pub fn chebmulx<T: ChebScalar>(c: &[T]) -> Vec<T> {
    let c = trim_trailing_zeros(c);
    let c = c.as_slice();
    if c.len() == 1 && c[0] == T::zero() {
        return c.to_vec();
    }
    let n = c.len();
    let mut prd = vec![T::zero(); n + 1];
    prd[0] = c[0].cheb_mul(T::zero());
    prd[1] = c[0];
    if n > 1 {
        let tmp: Vec<T> = c[1..].iter().map(|&v| v.cheb_div(T::from_f64(2.0))).collect();
        for (i, &t) in tmp.iter().enumerate() {
            prd[2 + i] = t;
        }
        for (i, &t) in tmp.iter().enumerate() {
            prd[i] = prd[i] + t;
        }
    }
    prd
}

/// `chebmul`: z-series based — `_cseries_to_zseries` both operands,
/// `_zseries_mul` (convolve), `_zseries_to_cseries`, then trim.
pub fn chebmul<T: ChebScalar>(c1_in: &[T], c2_in: &[T]) -> Vec<T> {
    let c1 = trim_trailing_zeros(c1_in);
    let c2 = trim_trailing_zeros(c2_in);
    let z1 = cseries_to_zseries(&c1);
    let z2 = cseries_to_zseries(&c2);
    let prd = zseries_mul(&z1, &z2);
    let ret = zseries_to_cseries(&prd);
    trim_trailing_zeros(&ret)
}

/// `chebdiv`: `lc1 < lc2` and `lc2 == 1` short-circuits match numpy's own
/// (NOT the z-series path for those two cases); the general branch is
/// z-series based, "more efficient than `pu._div(chebmul, c1, c2)`" per
/// numpy's own comment — genuinely NOT built on this module's own
/// `chebmul`, unlike every sibling basis's `*div`.
pub fn chebdiv<T: ChebScalar>(c1_in: &[T], c2_in: &[T]) -> Result<(Vec<T>, Vec<T>), ()> {
    let c1 = trim_trailing_zeros(c1_in);
    let c2 = trim_trailing_zeros(c2_in);
    if *c2.last().unwrap() == T::zero() {
        return Err(());
    }
    let lc1 = c1.len();
    let lc2 = c2.len();
    if lc1 < lc2 {
        return Ok((vec![c1[0].cheb_mul(T::zero())], c1));
    }
    if lc2 == 1 {
        let scl = c2[lc2 - 1];
        let q: Vec<T> = c1.iter().map(|&v| v.cheb_div(scl)).collect();
        return Ok((q, vec![c1[0].cheb_mul(T::zero())]));
    }
    let z1 = cseries_to_zseries(&c1);
    let z2 = cseries_to_zseries(&c2);
    let (quo_z, rem_z) = zseries_div(&z1, &z2);
    let quo = trim_trailing_zeros(&zseries_to_cseries(&quo_z));
    let rem = trim_trailing_zeros(&zseries_to_cseries(&rem_z));
    Ok((quo, rem))
}

/// `chebpow`: z-series based, repeated `np.convolve` — `power` is the
/// loop bound (degree-like, not data-size), so this whole function lives
/// in Rust rather than a Python loop calling a Rust `zseries_mul`
/// primitive per-iteration, matching numpy's own single-function shape
/// (`zs = to_z(c); prd = zs; for _ in 2..=power: prd = convolve(prd, zs);
/// return to_c(prd)`) exactly, with NO trailing trim (numpy's own
/// `chebpow` does not call `pu.trimseq` on its result either).
pub fn chebpow<T: ChebScalar>(c_in: &[T], power: usize) -> Vec<T> {
    let c = trim_trailing_zeros(c_in);
    if power == 0 {
        return vec![T::one()];
    }
    if power == 1 {
        return c;
    }
    let zs = cseries_to_zseries(&c);
    let mut prd = zs.clone();
    for _ in 2..=power {
        prd = zseries_mul(&prd, &zs);
    }
    zseries_to_cseries(&prd)
}

// ───────────────────────── der / int ─────────────────────────

/// `chebder`: DIRECT recurrence, NOT z-series based (see module doc
/// comment). `j`-loop counts DOWN from `n` to `3` inclusive, mutating `c`
/// in place — later iterations read values already mutated by earlier
/// ones (`c[j-2] += (j*c[j])/(j-2)`), a genuinely order-dependent
/// accumulation ported with the same left-to-right, high-to-low iteration
/// order as numpy's `range(n, 2, -1)`.
pub fn chebder<T: ChebScalar>(c_in: &[T], cnt: usize, scl: T) -> Vec<T> {
    let n0 = c_in.len();
    if cnt == 0 {
        return c_in.to_vec();
    }
    if cnt >= n0 {
        return vec![c_in[0].cheb_mul(T::zero())];
    }
    let mut c = c_in.to_vec();
    let mut n = n0;
    for _ in 0..cnt {
        n -= 1;
        for v in c.iter_mut() {
            *v = v.cheb_mul(scl);
        }
        let mut der = vec![T::zero(); n];
        // numpy: `for j in range(n, 2, -1): der[j-1] = (2*j)*c[j]; c[j-2]
        // += (j*c[j])/(j-2)` -- j from n down to 3 inclusive.
        let mut j = n;
        while j >= 3 {
            let jt = T::from_usize(j);
            der[j - 1] = T::from_f64(2.0).cheb_mul(jt).cheb_mul(c[j]);
            let incr = jt.cheb_mul(c[j]).cheb_div(T::from_usize(j - 2));
            c[j - 2] = c[j - 2] + incr;
            j -= 1;
        }
        if n > 1 {
            der[1] = T::from_f64(4.0).cheb_mul(c[2]);
        }
        der[0] = c[1];
        c = der;
    }
    c
}

/// `chebint`: DIRECT recurrence, NOT z-series based. `j`-loop is direct
/// assignment (`tmp[j+1] = c[j]/(2*(j+1))`, `tmp[j-1] -= c[j]/(2*(j-1))`)
/// for `j` in `2..n` (numpy: `range(2, n)`), with `tmp[2] = c[1]/4` set
/// before the loop when `n > 1`. The final `tmp[0] += k[i] -
/// chebval(lbnd, tmp)` fixup calls THIS module's own `cheb_eval`
/// (Clenshaw), matching numpy's own `chebval(lbnd, tmp)` call.
pub fn chebint<T: ChebScalar>(c_in: &[T], cnt: usize, k: &[T], lbnd: T, scl: T) -> Vec<T> {
    let mut c = c_in.to_vec();
    if cnt == 0 {
        return c;
    }
    for &ki in k.iter().take(cnt) {
        let n = c.len();
        for v in c.iter_mut() {
            *v = v.cheb_mul(scl);
        }
        if n == 1 && c[0] == T::zero() {
            c[0] = c[0] + ki;
        } else {
            let mut tmp = vec![T::zero(); n + 1];
            tmp[0] = c[0].cheb_mul(T::zero());
            tmp[1] = c[0];
            if n > 1 {
                tmp[2] = c[1].cheb_div(T::from_f64(4.0));
            }
            for j in 2..n {
                tmp[j + 1] = c[j].cheb_div(T::from_f64(2.0).cheb_mul(T::from_usize(j + 1)));
                let dec = c[j].cheb_div(T::from_f64(2.0).cheb_mul(T::from_usize(j - 1)));
                tmp[j - 1] = tmp[j - 1] - dec;
            }
            let at_lbnd = cheb_eval(&tmp, &[lbnd])[0];
            tmp[0] = tmp[0] + (ki - at_lbnd);
            c = tmp;
        }
    }
    c
}

// ───────────────────────── eval / vander / companion ─────────────────────────

/// `chebval`: Clenshaw recursion. `len(c) == 1`/`== 2` short-circuits
/// match every sibling basis's shape, but the general (`>= 3`-term) loop
/// combine step is `c0 = c[-i] - c1` — NO multiplicative coefficient on
/// `c1` at all (the sharpest divergence from every sibling basis's
/// Clenshaw loop). `x2 = 2*x` IS precomputed and used in the OTHER term:
/// `c1 = tmp + c1*x2`.
pub fn cheb_eval<T: ChebScalar>(c: &[T], xs: &[T]) -> Vec<T> {
    let n = c.len();
    xs.iter()
        .map(|&x| {
            if n == 1 {
                return c[0] + T::zero().cheb_mul(x);
            }
            if n == 2 {
                return c[0] + c[1].cheb_mul(x);
            }
            let x2 = T::from_f64(2.0).cheb_mul(x);
            let mut c0 = c[n - 2];
            let mut c1 = c[n - 1];
            for i in 3..=n {
                let tmp = c0;
                // numpy: `c0 = c[-i] - c1` -- NO coefficient on c1.
                c0 = c[n - i] - c1;
                // numpy: `c1 = tmp + c1*x2` -- c1 (left token) first.
                c1 = tmp + c1.cheb_mul(x2);
            }
            c0 + c1.cheb_mul(x)
        })
        .collect()
}

/// `chebvander(x, deg)`: `V[i,j] = T_j(x[i])`. `v[0] = x*0 + 1`; if
/// `deg > 0`: `v[1] = x` (plain, not `x2`), `v[i] = v[i-1]*x2 - v[i-2]`
/// (NO coefficient on the subtracted term, unlike every sibling basis's
/// `v[i-2]*(i-1)`-shaped subtraction).
pub fn chebvander<T: ChebScalar>(xs: &[T], deg: usize) -> Vec<T> {
    let width = deg + 1;
    let mut out = vec![T::zero(); xs.len() * width];
    for (row, &x) in xs.iter().enumerate() {
        let base = row * width;
        out[base] = x.cheb_mul(T::zero()) + T::one();
        if deg > 0 {
            let x2 = T::from_f64(2.0).cheb_mul(x);
            out[base + 1] = x;
            for i in 2..width {
                let a = out[base + i - 1].cheb_mul(x2);
                out[base + i] = a - out[base + i - 2];
            }
        }
    }
    out
}

/// `chebcompanion(c)`: SCALED. `scl = [1., sqrt(.5), sqrt(.5), ...]` — a
/// FLAT vector, NOT `multiply.accumulate`-built like every sibling
/// basis's `scl` (the sharpest structural divergence in this function).
/// `top[0] = sqrt(.5)`, `top[1:] = .5`, `bot[...] = top`; the last-column
/// correction `(c[:-1]/c[-1]) * (scl/scl[-1]) * .5` is preserved as a
/// literal per-index array op (`scl[i]/scl[-1]` computed explicitly),
/// NOT algebraically simplified to `sqrt(.5)`/`1` constants even though
/// they are mathematically equal.
pub fn chebcompanion<T: ChebScalar>(c: &[T]) -> Vec<T> {
    let len = c.len();
    if len == 2 {
        return vec![(-c[0]).cheb_div(c[1])];
    }
    let n = len - 1;
    let mut mat = vec![T::zero(); n * n];

    // scl = [1.] + [sqrt(.5)] * (n - 1) -- flat, not cumulative.
    let mut scl = vec![0.0f64; n];
    scl[0] = 1.0;
    for item in scl.iter_mut().take(n).skip(1) {
        *item = 0.5f64.sqrt();
    }
    let scl_last = scl[n - 1];

    // top[0] = sqrt(.5); top[1:] = .5; bot[...] = top -- super/sub
    // diagonal both get this SAME sequence of values.
    for k in 0..n.saturating_sub(1) {
        let v = if k == 0 { T::from_f64(0.5f64.sqrt()) } else { T::from_f64(0.5) };
        mat[k * n + (k + 1)] = v;
        mat[(k + 1) * n + k] = v;
    }

    // mat[:, -1] -= (c[:-1]/c[-1]) * (scl/scl[-1]) * .5
    let cn = c[len - 1];
    for i in 0..n {
        let ratio = c[i].cheb_div(cn);
        let scl_ratio = T::from_f64(scl[i] / scl_last);
        let corr = ratio.cheb_mul(scl_ratio).cheb_mul(T::from_f64(0.5));
        mat[i * n + (n - 1)] = mat[i * n + (n - 1)] - corr;
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cheb_eval_matches_numpy_example() {
        // C.chebval(1, [1,2,3]) -> 6.0 (measured against real numpy 2.5.1)
        let c = [1.0_f64, 2.0, 3.0];
        let out = cheb_eval(&c, &[1.0]);
        assert!((out[0] - 6.0).abs() < 1e-8, "{}", out[0]);
    }

    #[test]
    fn chebmulx_matches_numpy_example() {
        // C.chebmulx([1,2,3]) -> [1., 2.5, 1., 1.5] (measured against real numpy 2.5.1)
        let c = [1.0_f64, 2.0, 3.0];
        let out = chebmulx(&c);
        let expect = [1.0, 2.5, 1.0, 1.5];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn chebmul_matches_numpy_example() {
        // C.chebmul([1,2,3],[0,1,2]) -> measured against real numpy 2.5.1:
        // [4., 4.5, 3., 3.5, 3.]
        let c1 = [1.0_f64, 2.0, 3.0];
        let c2 = [0.0_f64, 1.0, 2.0];
        let out = chebmul(&c1, &c2);
        let expect = [4.0, 4.5, 3.0, 3.5, 3.0];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn chebdiv_matches_numpy_example() {
        // C.chebdiv((1,2,3),(3,2,1)) -> (array([3.]), array([-8., -4.]))
        let c1 = [1.0_f64, 2.0, 3.0];
        let c2 = [3.0_f64, 2.0, 1.0];
        let (q, r) = chebdiv(&c1, &c2).unwrap();
        assert!((q[0] - 3.0).abs() < 1e-8, "{:?}", q);
        assert!((r[0] - (-8.0)).abs() < 1e-8 && (r[1] - (-4.0)).abs() < 1e-8, "{:?}", r);
    }

    #[test]
    fn chebder_matches_numpy_example() {
        // C.chebder([1,2,3,4]) -> [14., 12., 24.] (measured against real numpy 2.5.1)
        let c = [1.0_f64, 2.0, 3.0, 4.0];
        let d1 = chebder(&c, 1, 1.0);
        let expect = [14.0, 12.0, 24.0];
        for (a, b) in d1.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn chebint_matches_numpy_example() {
        // C.chebint([1,2,3]) -> [0.5, -0.5, 0.5, 0.5] (measured against real numpy 2.5.1)
        let c = [1.0_f64, 2.0, 3.0];
        let out = chebint(&c, 1, &[0.0], 0.0, 1.0);
        let expect = [0.5, -0.5, 0.5, 0.5];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn chebvander_matches_numpy_example() {
        // C.chebvander([0,1,2], 3) row2 (x=2) = [1, 2, 7, 26] (verified live)
        let xs = [0.0_f64, 1.0, 2.0];
        let v = chebvander(&xs, 3);
        let expect_row2 = [1.0, 2.0, 7.0, 26.0];
        for (a, b) in v[8..12].iter().zip(expect_row2.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn chebcompanion_matches_numpy_example() {
        // C.chebcompanion([1,0,1]) -> measured against real numpy 2.5.1:
        // [[0., 1.1102230246251565e-16], [0.7071067811865476, 0.]] -- the
        // [0][1] entry is a rounding-noise artifact of the literal
        // sqrt(.5)-then-subtract sequence (see this module's doc comment
        // on chebcompanion NOT algebraically simplifying that step), not
        // an exact zero.
        let c = [1.0_f64, 0.0, 1.0];
        let m = chebcompanion(&c);
        let expect = [0.0, 1.1102230246251565e-16, 0.7071067811865476, 0.0];
        for (a, b) in m.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-15, "{a} vs {b}");
        }
    }

    #[test]
    fn chebpow_matches_numpy_example() {
        // C.chebpow([1,2,3,4], 2) -> [15.5, 22., 16., ..12.5, 12., 8.] (from
        // numpy's own docstring; degree-8 result, spot-check first three)
        let c = [1.0_f64, 2.0, 3.0, 4.0];
        let out = chebpow(&c, 2);
        assert!((out[0] - 15.5).abs() < 1e-8, "{:?}", out);
        assert!((out[1] - 22.0).abs() < 1e-8, "{:?}", out);
        assert!((out[2] - 16.0).abs() < 1e-8, "{:?}", out);
    }
}
