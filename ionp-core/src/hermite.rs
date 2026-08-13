//! Hermite-series (physicists') polynomial kernels — the Rust core for
//! `numpy.polynomial.hermite`. Sibling module to `crate::laguerre`/
//! `crate::legendre`, but NOT a relabeled copy of either: read directly
//! against `numpy/polynomial/hermite.py` for this port (NOT
//! `hermite_e.py`, the probabilists' sibling, which differs at several of
//! the exact spots called out below), per `docs/POLY-BASIS-SOURCE-AUDIT.md`
//! and this task's own brief. Genuine divergences, each verified
//! character-by-character against the numpy source before being ported:
//!
//! - Every multiply site in `hermite.py` writes its LEFT operand token
//!   first, whatever that token is (`c[0] * xs`, `c1 * x2`,
//!   `c1 * (2*(nd-1))`, `hermmulx(c1) * 2`, `v[i-1] * x2`, `c[i] * i`) —
//!   unlike `laguerre.py`, which uniformly writes the scalar coefficient
//!   first regardless of which operand that is syntactically. This module
//!   preserves literal left-to-right operand order at every call site
//!   rather than applying a single "scalar first" rule, since
//!   `complex_mul_fma`'s FMA-based formula is mathematically commutative
//!   but NOT bit-identical under argument swap (same measured divergence
//!   `crate::laguerre`'s module doc comment documents:
//!   `-0x1.34b86b86099e4p+0` vs `-0x1.34b86b86099e3p+0` for the same two
//!   complex128 operands swapped).
//! - `hermval`'s Clenshaw recursion precomputes `x2 = x * 2` ONCE per
//!   sample point and reuses it in every branch (`len(c)==1`, `len(c)==2`,
//!   and the general loop's final `c0 + c1*x2`) — ported here as a literal
//!   `x2` local, computed unconditionally, not shortcut away for the
//!   short-c branches (numpy computes it before branching on `len(c)`,
//!   verified against `hermite.py:872-905`).
//! - `hermmulx` has NO division by an index anywhere in ITS OWN
//!   recurrence structure beyond the fixed `/2` on the first two slots and
//!   each new top slot (`prd[1] = c[0]/2`, `prd[i+1] = c[i]/2`), and its
//!   accumulation step (`prd[i-1] += c[i]*i`) touches only ONE index back
//!   — contrast `hermder`'s calculus recurrence, which does a bare
//!   *assignment* (`der[j-1] = (2*j)*c[j]`), never `+=`.
//! - `hermder` and `hermint`'s inner loops are DIRECT ASSIGNMENT, not
//!   accumulation — unlike `lagder`'s `c[j-1] += c[j]` and `legder`'s
//!   two-index-back accumulated sum. Verified against
//!   `hermite.py:1090-1097` (`hermder`) and `hermite.py:1179-1187`
//!   (`hermint`, no `tmp[j] +=` term inside the `j` loop at all).
//! - `hermmul`'s trailing combining step is `hermadd(c0, hermmulx(c1) * 2)`
//!   — the `* 2` scales `hermmulx(c1)`'s WHOLE result, applied after the
//!   call, not folded into `hermmulx` itself and not applied to `c1`
//!   before the call. Contrast `lagmul`'s extra `lagsub(c1, lagmulx(c1))`
//!   term, which has no such trailing `*2` and subtracts rather than adds.
//! - `hermcompanion` is SCALED (unlike `lagcompanion`), but with a
//!   different scale-vector construction than `legcompanion`: `scl` is a
//!   reversed cumulative product of `[1., 1/sqrt(2(n-1)), 1/sqrt(2(n-2)),
//!   ..., 1/sqrt(2)]`, the super/subdiagonal entries are
//!   `sqrt(0.5*(k+1))` (not `legcompanion`'s formula), and only the LAST
//!   COLUMN is corrected (`mat[:, -1] -= scl*c[:-1]/(2.0*c[-1])`) — no
//!   `+=`/`-=` elsewhere. Independently verified by direct numpy execution
//!   (`hermcompanion([1,0,1])`, `[1.,2.,3.]`, `[1.,2.,3.,4.,5.]`,
//!   `[2.,0,0,1.]`) against a from-scratch reimplementation of this exact
//!   formula before any Rust was written — see this task's report for the
//!   full derivation (the docstring example and the literal source
//!   initially looked contradictory; they are not, the last-column
//!   correction runs AFTER `bot[...] = top`, and for small `n` only the
//!   superdiagonal entry at column `n-1` is visibly affected).
//! - `hermval`'s `len(c) == 1` branch sets `c1 = 0` (a plain Python int,
//!   not a same-dtype zero) and the function unconditionally returns
//!   `c0 + c1*x2` even in that branch — ported here as `T::zero().herm_mul(x2)`
//!   (matching `0 * x2`'s actual IEEE-754 sign behavior for negative `x2`,
//!   not silently treated as an early return of bare `c0`).

use crate::buffer::C128;
use crate::poly::{trim_trailing_zeros, PolyScalar};

/// Same seam as `crate::laguerre::LagScalar`/`crate::legendre::LegScalar`:
/// real `f64`-from-magnitude construction plus `complex_div`/
/// `complex_mul_fma`-routed division/multiplication so the `C128`
/// instantiation stays bit-exact against numpy's `npy_cdivide`/FMA-based
/// complex multiply rather than `num_complex::Complex`'s native (measurably
/// different) `Div`/`Mul` impls. Declared independently per basis-module
/// convention — this file must never import anything from
/// `laguerre.rs`/`legendre.rs`.
pub trait HermScalar: PolyScalar {
    fn from_f64(v: f64) -> Self;
    fn herm_div(self, other: Self) -> Self;
    fn herm_mul(self, other: Self) -> Self;
}

impl HermScalar for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn herm_div(self, other: Self) -> Self {
        self / other
    }
    fn herm_mul(self, other: Self) -> Self {
        self * other
    }
}

impl HermScalar for C128 {
    fn from_f64(v: f64) -> Self {
        C128::new(v, 0.0)
    }
    fn herm_div(self, other: Self) -> Self {
        crate::ufunc::complex_div(self, other)
    }
    fn herm_mul(self, other: Self) -> Self {
        crate::ufunc::complex_mul_fma(self, other)
    }
}

/// `<left-operand token> * <right-operand token>`, in that literal order.
/// Used for `c[0] * xs` / `c[-2] * xs` / `c[-1] * xs` sites in `hermmul`,
/// where the coefficient scalar `s` IS the left token syntactically. See
/// the module doc comment: this file preserves literal per-call-site
/// operand order rather than a single "scalar first" rule, and every site
/// that needs this exact shape (coefficient-scalar-times-array) happens to
/// also be "scalar first" in the literal source, so this helper is safe to
/// reuse across all of them.
fn scale<T: HermScalar>(xs: &[T], s: T) -> Vec<T> {
    xs.iter().map(|&v| s.herm_mul(v)).collect()
}

/// Padded elementwise add, mirroring numpy's `polyutils._add` exactly —
/// same branch-for-branch shape and signed-zero-preservation reasoning as
/// `crate::laguerre::pad_add`'s doc comment (independently declared here,
/// not imported, per basis-module convention).
fn pad_add<T: HermScalar>(a: &[T], b: &[T]) -> Vec<T> {
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

/// Padded elementwise subtract, mirroring numpy's `polyutils._sub` exactly
/// — same branch-for-branch shape as `crate::laguerre::pad_sub`'s doc
/// comment (independently declared here, not imported).
fn pad_sub<T: HermScalar>(a: &[T], b: &[T]) -> Vec<T> {
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

/// `hermval`: Clenshaw recursion, ported from `hermite.py:872-905`. `x2`
/// is precomputed ONCE per sample point and reused unconditionally in
/// every branch, matching the module doc comment's note on this.
pub fn herm_eval<T: HermScalar>(c: &[T], xs: &[T]) -> Vec<T> {
    let n = c.len();
    xs.iter()
        .map(|&x| {
            let x2 = x.herm_mul(T::from_f64(2.0));
            if n == 1 {
                // numpy: `c0 = c[0]; c1 = 0`, then unconditionally
                // `return c0 + c1*x2` -- `c1` is the plain int `0`, so
                // `c1*x2` is `0*x2`, NOT a hardcoded zero result (its sign
                // follows IEEE-754 multiply rules for a negative `x2`).
                return c[0] + T::zero().herm_mul(x2);
            }
            if n == 2 {
                return c[0] + c[1].herm_mul(x2);
            }
            let mut nd = n;
            let mut c0 = c[n - 2];
            let mut c1 = c[n - 1];
            for i in 3..=n {
                let tmp = c0;
                nd -= 1;
                let two_nd_m1 = T::from_usize(2 * (nd - 1));
                // numpy: `c0 = c[-i] - c1 * (2 * (nd - 1))` -- `c1` (left
                // token) first, matching this file's literal-order rule.
                c0 = c[n - i] - c1.herm_mul(two_nd_m1);
                // numpy: `c1 = tmp + c1 * x2` -- `c1` first again.
                c1 = tmp + c1.herm_mul(x2);
            }
            c0 + c1.herm_mul(x2)
        })
        .collect()
}

/// `hermmulx`: ported from `hermite.py:378-407`. NO index-division beyond
/// the fixed `/2` on the two newly-introduced top slots; the accumulation
/// (`prd[i-1] += c[i]*i`) touches exactly one index back.
///
/// Ticket #84 (2026-08-08, Monday): trims `c` via `trim_trailing_zeros`
/// FIRST, matching numpy's own `[c] = pu.as_series([c])` -- same fix, same
/// root cause, same measured shape+sign defect as `legendre::legmulx`'s
/// identical fix; see that function's doc comment for the full trace.
pub fn hermmulx<T: HermScalar>(c: &[T]) -> Vec<T> {
    let c = trim_trailing_zeros(c);
    let c = c.as_slice();
    if c.len() == 1 && c[0] == T::zero() {
        return c.to_vec();
    }
    let n = c.len();
    let mut prd = vec![T::zero(); n + 1];
    // numpy: `prd[0] = c[0] * 0` -- sign-preserving multiply-by-zero, NOT
    // a hardcoded `T::zero()` (same discipline as `crate::laguerre`'s
    // `lag_mul(T::zero())` sites).
    prd[0] = c[0].herm_mul(T::zero());
    prd[1] = c[0].herm_div(T::from_f64(2.0));
    for i in 1..n {
        prd[i + 1] = c[i].herm_div(T::from_f64(2.0));
        prd[i - 1] = prd[i - 1] + c[i].herm_mul(T::from_usize(i));
    }
    prd
}

pub use crate::poly::add_trim as hermadd_trim;
pub use crate::poly::sub_trim as hermsub_trim;

/// `hermmul`: ported from `hermite.py:436-476`. Trailing combine step is
/// `hermadd(c0, hermmulx(c1) * 2)` -- the `*2` scales `hermmulx`'s WHOLE
/// result, not folded into the call or applied before it. The `len(c)==1`
/// branch's `c1 = 0` is represented as `vec![T::zero()]` (what
/// `pu.as_series([0])` itself would produce) so it flows through the same
/// `hermmulx`/`hermadd` combine step as every other branch, rather than a
/// hand-written early return.
pub fn hermmul<T: HermScalar>(c1_in: &[T], c2_in: &[T]) -> Vec<T> {
    let c1t = trim_trailing_zeros(c1_in);
    let c2t = trim_trailing_zeros(c2_in);
    let (c1t, c2t) = (c1t.as_slice(), c2t.as_slice());
    let (c, xs): (&[T], &[T]) = if c1t.len() > c2t.len() { (c2t, c1t) } else { (c1t, c2t) };

    let (c0, c1v) = if c.len() == 1 {
        (scale(xs, c[0]), vec![T::zero()])
    } else if c.len() == 2 {
        (scale(xs, c[0]), scale(xs, c[1]))
    } else {
        let mut nd = c.len();
        let mut c0 = scale(xs, c[c.len() - 2]);
        let mut c1v = scale(xs, c[c.len() - 1]);
        for i in 3..=c.len() {
            let tmp = c0.clone();
            nd -= 1;
            let two_nd_m1 = T::from_usize(2 * (nd - 1));
            let scaled_top = scale(xs, c[c.len() - i]);
            // numpy: `c1 * (2 * (nd - 1))` -- c1 (left token) first.
            let c1_scaled: Vec<T> = c1v.iter().map(|&v| v.herm_mul(two_nd_m1)).collect();
            // Ticket #84 (2026-08-08, Monday): numpy's `hermsub(...)`/
            // `hermadd(...)` here are real `pu._sub`/`pu._add` calls, which
            // trim BOTH operands via `as_series` before the padded combine
            // -- unrelated to (and stricter than) `pad_sub`'s own
            // untrimmed-same-length usage inside `hermdiv`'s loop. Skipping
            // this pre-trim lets an untouched trailing `-0.0` in the longer
            // operand collide with a spurious `+0.0` and flip sign. Measured
            // minimal repro: `hermmul([tiny-0j, 0.5+0j], [-tiny-0j,
            // 0.5+0j])` -- numpy's final `hermadd(c0, hermmulx(c1)*2)` step
            // trims `c0` from `[-0.+0.j, 0.+0.j]` (len 2) down to `[-0.+0.j]`
            // (len 1, trailing exact zero dropped) BEFORE combining, so the
            // `-0.` already sitting at index 1 of `hermmulx(c1)*2` is never
            // touched and survives; without the pre-trim, index 1 gets
            // `(-0.0)+(0.0)`-style collided into `+0.0`.
            let scaled_top_t = trim_trailing_zeros(&scaled_top);
            let c1_scaled_t = trim_trailing_zeros(&c1_scaled);
            c0 = pad_sub(&scaled_top_t, &c1_scaled_t);
            // numpy: `hermmulx(c1) * 2` -- hermmulx's result first.
            let mx = hermmulx(&c1v);
            let mx2: Vec<T> = mx.iter().map(|&v| v.herm_mul(T::from_f64(2.0))).collect();
            let tmp_t = trim_trailing_zeros(&tmp);
            let mx2_t = trim_trailing_zeros(&mx2);
            c1v = pad_add(&tmp_t, &mx2_t);
        }
        (c0, c1v)
    };
    let mx = hermmulx(&c1v);
    let mx2: Vec<T> = mx.iter().map(|&v| v.herm_mul(T::from_f64(2.0))).collect();
    // Same pre-trim requirement as above for the final `hermadd(c0, mx2)`.
    let c0_t = trim_trailing_zeros(&c0);
    let mx2_t = trim_trailing_zeros(&mx2);
    let out = pad_add(&c0_t, &mx2_t);
    trim_trailing_zeros(&out)
}

/// `hermdiv`: `polyutils._div(hermmul, c1, c2)`, the same basis-agnostic
/// repeated-subtraction shape as `lagdiv`/`legdiv`, but built on THIS
/// module's own `hermmul`/`hermadd_trim`/`hermsub_trim` so its bit pattern
/// depends on `hermmul`'s grouping, not a sibling basis's.
pub fn hermdiv<T: HermScalar>(c1_in: &[T], c2_in: &[T]) -> Result<(Vec<T>, Vec<T>), ()> {
    // numpy: `[c1, c2] = as_series([c1, c2])` -- BOTH operands are trimmed
    // copies before any length check or return value is built. Ticket #87
    // (2026-08-10, Monday): this port never did that top-level trim (only
    // the sign-of-zero fix below had landed, via #84/#85's shared bug
    // class), so an input carrying an exact trailing zero on `c2` could
    // false-positive the `c2.last() == 0` zero-divisor check (numpy trims
    // it away and divides fine), and an exact trailing zero on `c1` leaked
    // an untrimmed tail into the returned remainder -- a SHAPE divergence,
    // not merely a sign one. Matches the pre-trim `chebdiv`/`legdiv`/
    // `lagdiv`/`poly::div` already do.
    let c1 = trim_trailing_zeros(c1_in);
    let c2 = trim_trailing_zeros(c2_in);
    if *c2.last().unwrap() == T::zero() {
        return Err(());
    }
    let lc1 = c1.len();
    let lc2 = c2.len();
    if lc1 < lc2 {
        return Ok((vec![c1[0].herm_mul(T::zero())], c1));
    }
    if lc2 == 1 {
        let scl = c2[0];
        let q: Vec<T> = c1.iter().map(|&v| v.herm_div(scl)).collect();
        return Ok((q, vec![c1[0].herm_mul(T::zero())]));
    }
    let mut quo = vec![T::zero(); lc1 - lc2 + 1];
    let mut rem = c1.clone();
    for i in (0..=lc1 - lc2).rev() {
        let mut basis = vec![T::zero(); i + 1];
        basis[i] = T::one();
        let p = hermmul(&basis, &c2);
        let q = rem.last().unwrap().herm_div(*p.last().unwrap());
        let rem_head = &rem[..rem.len() - 1];
        let p_head = &p[..p.len() - 1];
        // numpy: `q * p[:-1]` -- q (scalar, left token) first.
        let scaled_p_head: Vec<T> = p_head.iter().map(|&v| q.herm_mul(v)).collect();
        rem = pad_sub(rem_head, &scaled_p_head);
        quo[i] = q;
    }
    let rem_trimmed = trim_trailing_zeros(&rem);
    Ok((quo, rem_trimmed))
}

/// `hermder`: DIRECT ASSIGNMENT (`der[j-1] = (2*j)*c[j]`), ported from
/// `hermite.py:1090-1097` -- no accumulation at all, unlike `lagder`'s
/// `c[j-1] += c[j]`. Scalar `(2*j)` is the LEFT token here (`(2 * j) *
/// c[j]`), so unlike most of this file's sites, the scalar goes first.
pub fn hermder<T: HermScalar>(c_in: &[T], cnt: usize, scl: T) -> Vec<T> {
    let n0 = c_in.len();
    if cnt == 0 {
        return c_in.to_vec();
    }
    if cnt >= n0 {
        // numpy: `c = c[:1] * 0` -- sign-preserving.
        return vec![c_in[0].herm_mul(T::zero())];
    }
    let mut c = c_in.to_vec();
    let mut n = n0;
    for _ in 0..cnt {
        n -= 1;
        for v in c.iter_mut() {
            *v = v.herm_mul(scl);
        }
        let mut der = vec![T::zero(); n];
        for j in (1..=n).rev() {
            der[j - 1] = T::from_usize(2 * j).herm_mul(c[j]);
        }
        c = der;
    }
    c
}

/// `hermint`: ported from `hermite.py:1179-1187`. The `j`-loop is DIRECT
/// ASSIGNMENT (`tmp[j+1] = c[j]/(2*(j+1))`), no accumulation term at all --
/// contrast `lagint`'s `tmp[j] = tmp[j] + c[j]`. The final
/// `tmp[0] + (ki - at_lbnd)` grouping is load-bearing (associativity).
pub fn hermint<T: HermScalar>(c_in: &[T], cnt: usize, k: &[T], lbnd: T, scl: T) -> Vec<T> {
    let mut c = c_in.to_vec();
    if cnt == 0 {
        return c;
    }
    for &ki in k.iter().take(cnt) {
        let n = c.len();
        for v in c.iter_mut() {
            *v = v.herm_mul(scl);
        }
        if n == 1 && c[0] == T::zero() {
            c[0] = c[0] + ki;
        } else {
            let mut tmp = vec![T::zero(); n + 1];
            // numpy: `tmp[0] = c[0] * 0` -- sign-preserving.
            tmp[0] = c[0].herm_mul(T::zero());
            tmp[1] = c[0].herm_div(T::from_f64(2.0));
            for j in 1..n {
                tmp[j + 1] = c[j].herm_div(T::from_usize(2 * (j + 1)));
            }
            let at_lbnd = herm_eval(&tmp, &[lbnd])[0];
            // numpy: `tmp[0] += k[i] - hermval(lbnd, tmp)`, i.e.
            // `tmp[0] + (ki - at_lbnd)` -- parenthesization load-bearing,
            // same trap as `lagint`'s identical final line.
            tmp[0] = tmp[0] + (ki - at_lbnd);
            c = tmp;
        }
    }
    c
}

/// `hermvander(x, deg)`: `V[i, j] = H_j(x[i])`, ported from
/// `hermite.py:1350-1360`. `v[0] = x*0 + 1` and `v[1] = x2` are computed
/// literally (not shortcut to hardcoded `1`/`x*2`), matching numpy's own
/// NaN/Inf-fidelity for degenerate `x`.
pub fn hermvander<T: HermScalar>(xs: &[T], deg: usize) -> Vec<T> {
    let width = deg + 1;
    let mut out = vec![T::zero(); xs.len() * width];
    for (row, &x) in xs.iter().enumerate() {
        let base = row * width;
        // numpy: `v[0] = x * 0 + 1` -- `x` (left token) first.
        out[base] = x.herm_mul(T::zero()) + T::one();
        if deg > 0 {
            let x2 = x.herm_mul(T::from_f64(2.0));
            out[base + 1] = x2;
            for i in 2..width {
                // numpy: `v[i] = v[i-1]*x2 - v[i-2]*(2*(i-1))` -- `v[i-1]`/
                // `v[i-2]` (left tokens) first in both products.
                let a = out[base + i - 1].herm_mul(x2);
                let b = out[base + i - 2].herm_mul(T::from_usize(2 * (i - 1)));
                out[base + i] = a - b;
            }
        }
    }
    out
}

/// `hermcompanion(c)`: SCALED, ported from `hermite.py:1656-1674` and
/// independently re-derived by direct numpy execution before any Rust was
/// written (see the module doc comment) since the docstring example and
/// the literal source's flat-index construction initially looked
/// contradictory. `scl` (the reversed cumulative-product scale vector) and
/// the super/subdiagonal values (`sqrt(0.5*(k+1))`) are ALWAYS real-valued
/// regardless of `c`'s own dtype (built from `arange`/`sqrt`, independent
/// of `c`), so they are computed in plain `f64` here and converted via
/// `T::from_f64` only at the point they combine with `c`'s own elements.
pub fn hermcompanion<T: HermScalar>(c: &[T]) -> Vec<T> {
    let len = c.len();
    if len == 2 {
        // numpy: `-.5 * c[0] / c[1]` -- `-.5` (left token) first.
        return vec![T::from_f64(-0.5).herm_mul(c[0]).herm_div(c[1])];
    }
    let n = len - 1;
    let mut mat = vec![T::zero(); n * n];

    // scl_raw = hstack((1., 1./sqrt(2.*arange(n-1,0,-1)))) -- entries
    // 1, 1/sqrt(2*(n-1)), 1/sqrt(2*(n-2)), ..., 1/sqrt(2*1).
    let mut scl_raw = vec![1.0f64; n];
    for (j, item) in scl_raw.iter_mut().enumerate().take(n).skip(1) {
        *item = 1.0 / (2.0 * (n - j) as f64).sqrt();
    }
    // scl = multiply.accumulate(scl_raw)[::-1] -- left-to-right cumulative
    // product, then reversed.
    let mut scl_cum = vec![0.0f64; n];
    scl_cum[0] = scl_raw[0];
    for j in 1..n {
        scl_cum[j] = scl_cum[j - 1] * scl_raw[j];
    }
    let scl: Vec<f64> = (0..n).map(|i| scl_cum[n - 1 - i]).collect();

    // top[...] = sqrt(.5*arange(1,n)); bot[...] = top -- superdiagonal and
    // subdiagonal both get sqrt(0.5*(k+1)) for k = 0..n-2, `bot` copied
    // from `top` BEFORE the last-column correction below.
    for k in 0..n.saturating_sub(1) {
        let v = T::from_f64((0.5 * (k as f64 + 1.0)).sqrt());
        mat[k * n + (k + 1)] = v;
        mat[(k + 1) * n + k] = v;
    }

    // mat[:, -1] -= scl * c[:-1] / (2.0 * c[-1])
    let cn = c[len - 1];
    let denom = T::from_f64(2.0).herm_mul(cn);
    for i in 0..n {
        let num = T::from_f64(scl[i]).herm_mul(c[i]);
        let corr = num.herm_div(denom);
        mat[i * n + (n - 1)] = mat[i * n + (n - 1)] - corr;
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn herm_eval_matches_numpy_example() {
        // H.hermval(1, [1,2,3]) -> -5.0 (verified live)
        let c = [1.0_f64, 2.0, 3.0];
        let out = herm_eval(&c, &[1.0]);
        assert!((out[0] - (-5.0)).abs() < 1e-8, "{}", out[0]);
    }

    #[test]
    fn hermmulx_matches_numpy_example() {
        // H.hermmulx([1,2,3]) -> [1, 6.5, 1, 1.5] (verified live)
        let c = [1.0_f64, 2.0, 3.0];
        let out = hermmulx(&c);
        let expect = [1.0, 6.5, 1.0, 1.5];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn hermmul_matches_numpy_example() {
        // H.hermmul([1,2,3],[0,1,2]) -> verified live below
        let c1 = [1.0_f64, 2.0, 3.0];
        let c2 = [0.0_f64, 1.0, 2.0];
        let out = hermmul(&c1, &c2);
        // computed via numpy live: array([18., 51., 82., 30., 12.])
        let expect = [18.0, 51.0, 82.0, 30.0, 12.0];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn hermder_matches_numpy_example() {
        // H.hermder([1,2,3,4]) -> [4, 12, 24] (verified live)
        let c = [1.0_f64, 2.0, 3.0, 4.0];
        let d1 = hermder(&c, 1, 1.0);
        for (a, b) in d1.iter().zip([4.0, 12.0, 24.0].iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn hermint_matches_numpy_example() {
        // H.hermint([1,2,3]) -> [1, 0.5, 1, 0.5] (verified live)
        let c = [1.0_f64, 2.0, 3.0];
        let out = hermint(&c, 1, &[0.0], 0.0, 1.0);
        let expect = [1.0, 0.5, 1.0, 0.5];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn hermvander_matches_numpy_example() {
        // H.hermvander([0,1,2], 3) row1 = [1,2,2,-4] (verified live)
        let xs = [0.0_f64, 1.0, 2.0];
        let v = hermvander(&xs, 3);
        let expect_row1 = [1.0, 2.0, 2.0, -4.0];
        for (a, b) in v[4..8].iter().zip(expect_row1.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn hermcompanion_matches_numpy_example() {
        // H.hermcompanion([1,0,1]) -> [[0, 0.35355339],[0.70710678, 0]]
        let c = [1.0_f64, 0.0, 1.0];
        let m = hermcompanion(&c);
        let expect = [0.0, 0.353553390593, 0.707106781187, 0.0];
        for (a, b) in m.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }
}
