//! HermiteE-series (probabilists') polynomial kernels — the Rust core for
//! `numpy.polynomial.hermite_e`. Sibling module to `crate::hermite`
//! (physicists'), but NOT a relabeled copy with the 2s deleted: read
//! directly against `numpy/polynomial/hermite_e.py` for this port (NOT
//! `hermite.py`), per `docs/POLY-BASIS-SOURCE-AUDIT.md` and this task's own
//! brief, which flags this pair as the highest-risk port yet precisely
//! because it is the structurally CLOSEST pair of any two bases in this
//! project — same recurrence shape, differing essentially by factors of 2
//! (or their absence). Genuine divergences from `crate::hermite`, each
//! verified character-by-character against the literal numpy source before
//! being ported:
//!
//! - `hermeval`'s Clenshaw recursion has NO `x2` precompute at all — every
//!   site that `hermval` writes as `x2` this module writes as plain `x`.
//!   Verified against `hermite_e.py:862-886`.
//! - `hermeval`'s general-loop combine step is `c0 = c[-i] - c1*(nd-1)`,
//!   `c1 = tmp + c1*x` — NO leading `2 *` on the `(nd-1)` term (contrast
//!   `hermval`'s `c1 * (2*(nd-1))`).
//! - `hermemulx` has NO division anywhere in its recurrence (`prd[1] =
//!   c[0]`, `prd[i+1] = c[i]`) — contrast `hermmulx`'s `/2` on both newly
//!   introduced slots. The accumulation step (`prd[i-1] += c[i]*i`) is
//!   otherwise identical in shape (touches exactly one index back).
//!   Verified against `hermite_e.py:378-406`.
//! - `hermemul`'s trailing combining step is `hermeadd(c0, hermemulx(c1))`
//!   — NO `* 2` anywhere, on either the inner-loop `c1*(nd-1)` term or the
//!   final `hermemulx(c1)` result. Contrast `hermmul`'s `hermmulx(c1) * 2`.
//!   Verified against `hermite_e.py:436-475`.
//! - `hermeder` is direct assignment `der[j-1] = j*c[j]` — NO leading `2 *`
//!   (contrast `hermder`'s `(2*j)*c[j]`). Verified against
//!   `hermite_e.py:1090-1096`.
//! - `hermeint` is direct assignment `tmp[1] = c[0]` (no `/2`), `tmp[j+1] =
//!   c[j]/(j+1)` (no `2*` in the denominator) — contrast `hermint`'s
//!   `c[0]/2`, `c[j]/(2*(j+1))`. Verified against `hermite_e.py:1179-1187`.
//! - `hermevander`'s recurrence uses plain `x` throughout: `v[1] = x` (not
//!   `x2`), `v[i] = v[i-1]*x - v[i-2]*(i-1)` (no `2*` factor on the
//!   subtracted term). Verified against `hermite_e.py:1350-1360`.
//! - `hermecompanion`'s `len(c) == 2` branch is `[[-c[0] / c[1]]]` — NO
//!   `-0.5` factor at all (contrast `hermcompanion`'s `-.5 * c[0] / c[1]`).
//!   Note the numpy expression is `-c[0] / c[1]`, which Python parses as
//!   `(-c[0]) / c[1]` (unary minus binds tighter than `/`) — negate FIRST,
//!   then divide, ported here as `(-c[0]).herme_div(c[1])`, not
//!   `T::zero() - c[0].herme_div(c[1])` (a different operation shape even
//!   though mathematically equal). The general (`n > 2`) branch's `scl`
//!   construction has NO `2.` inside the `sqrt` (`1./sqrt(arange(n-1,0,-1))`,
//!   entries `1, 1/sqrt(n-1), ..., 1/sqrt(1)`), the super/subdiagonal is
//!   `sqrt(arange(1,n))` (no `.5` factor — plain `sqrt(k+1)` for
//!   `k=0..n-2`), and the last-column correction is `scl*c[:-1]/c[-1]` (no
//!   `2.0` in the denominator). Verified against `hermite_e.py:1415-1452`.
//! - `hermeroots`'s `len(c) == 2` branch (Python-orchestration layer, not
//!   this file) is `-c[0]/c[1]` — same no-`-0.5`-factor pattern, and unlike
//!   `hermroots`, `hermeroots` never calls any real-downcast helper at all
//!   (no `_to_real_if_imag_zero`-equivalent step in its literal source) —
//!   this file's `hermecompanion` is the only piece `hermeroots` needs from
//!   the Rust core; the roots themselves come from `eigvals`, already
//!   exposed elsewhere.
//! - Every multiply site in `hermite_e.py` writes its LEFT operand token
//!   first, same literal-order-preservation discipline as `crate::hermite`
//!   (see that module's doc comment for why this matters: `complex_mul_fma`
//!   is not bit-identical under argument swap).

use crate::buffer::C128;
use crate::poly::{trim_trailing_zeros, PolyScalar};

/// Same seam as `crate::hermite::HermScalar`/`crate::laguerre::LagScalar`/
/// `crate::legendre::LegScalar`: real `f64`-from-magnitude construction plus
/// `complex_div`/`complex_mul_fma`-routed division/multiplication so the
/// `C128` instantiation stays bit-exact against numpy's `npy_cdivide`/
/// FMA-based complex multiply rather than `num_complex::Complex`'s native
/// (measurably different) `Div`/`Mul` impls. Declared independently per
/// basis-module convention — this file must never import anything from
/// `hermite.rs`/`laguerre.rs`/`legendre.rs`.
pub trait HermeScalar: PolyScalar {
    fn from_f64(v: f64) -> Self;
    fn herme_div(self, other: Self) -> Self;
    fn herme_mul(self, other: Self) -> Self;
}

impl HermeScalar for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn herme_div(self, other: Self) -> Self {
        self / other
    }
    fn herme_mul(self, other: Self) -> Self {
        self * other
    }
}

impl HermeScalar for C128 {
    fn from_f64(v: f64) -> Self {
        C128::new(v, 0.0)
    }
    fn herme_div(self, other: Self) -> Self {
        crate::ufunc::complex_div(self, other)
    }
    fn herme_mul(self, other: Self) -> Self {
        crate::ufunc::complex_mul_fma(self, other)
    }
}

/// `<left-operand token> * <right-operand token>`, in that literal order.
/// Used for `c[0]*xs`/`c[-2]*xs`/`c[-1]*xs` sites in `hermemul`.
fn scale<T: HermeScalar>(xs: &[T], s: T) -> Vec<T> {
    xs.iter().map(|&v| s.herme_mul(v)).collect()
}

/// Padded elementwise add, mirroring numpy's `polyutils._add` exactly —
/// same branch-for-branch shape as `crate::hermite::pad_add`'s doc comment
/// (independently declared here, not imported).
fn pad_add<T: HermeScalar>(a: &[T], b: &[T]) -> Vec<T> {
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
/// — same branch-for-branch shape as `crate::hermite::pad_sub`'s doc
/// comment (independently declared here, not imported).
fn pad_sub<T: HermeScalar>(a: &[T], b: &[T]) -> Vec<T> {
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

/// `hermeval`: Clenshaw recursion, ported from `hermite_e.py:862-886`. NO
/// `x2` precompute anywhere — every site uses plain `x` (the module doc
/// comment's sharpest-flagged divergence from `herm_eval`).
pub fn herme_eval<T: HermeScalar>(c: &[T], xs: &[T]) -> Vec<T> {
    let n = c.len();
    xs.iter()
        .map(|&x| {
            if n == 1 {
                // numpy: `c0 = c[0]; c1 = 0`, unconditional `return c0 +
                // c1*x` -- `c1` is the plain int `0`, so `c1*x` is `0*x`,
                // not a hardcoded zero result.
                return c[0] + T::zero().herme_mul(x);
            }
            if n == 2 {
                return c[0] + c[1].herme_mul(x);
            }
            let mut nd = n;
            let mut c0 = c[n - 2];
            let mut c1 = c[n - 1];
            for i in 3..=n {
                let tmp = c0;
                nd -= 1;
                let nd_m1 = T::from_usize(nd - 1);
                // numpy: `c0 = c[-i] - c1 * (nd - 1)` -- `c1` (left token)
                // first, NO leading `2 *`.
                c0 = c[n - i] - c1.herme_mul(nd_m1);
                // numpy: `c1 = tmp + c1 * x` -- `c1` first, plain `x`.
                c1 = tmp + c1.herme_mul(x);
            }
            c0 + c1.herme_mul(x)
        })
        .collect()
}

/// `hermemulx`: ported from `hermite_e.py:378-406`. NO division anywhere —
/// the sharpest single divergence from `hermmulx` in this file.
///
/// Ticket #84 (2026-08-08, Monday): trims `c` via `trim_trailing_zeros`
/// FIRST, matching numpy's own `[c] = pu.as_series([c])` -- same fix, same
/// root cause, same measured shape+sign defect as `legendre::legmulx`'s
/// identical fix; see that function's doc comment for the full trace.
pub fn hermemulx<T: HermeScalar>(c: &[T]) -> Vec<T> {
    let c = trim_trailing_zeros(c);
    let c = c.as_slice();
    if c.len() == 1 && c[0] == T::zero() {
        return c.to_vec();
    }
    let n = c.len();
    let mut prd = vec![T::zero(); n + 1];
    // numpy: `prd[0] = c[0] * 0` -- sign-preserving multiply-by-zero.
    prd[0] = c[0].herme_mul(T::zero());
    prd[1] = c[0];
    for i in 1..n {
        prd[i + 1] = c[i];
        prd[i - 1] = prd[i - 1] + c[i].herme_mul(T::from_usize(i));
    }
    prd
}

pub use crate::poly::add_trim as hermeadd_trim;
pub use crate::poly::sub_trim as hermesub_trim;

/// `hermemul`: ported from `hermite_e.py:436-475`. Trailing combine step is
/// `hermeadd(c0, hermemulx(c1))` -- NO `*2` anywhere, contrast `hermmul`'s
/// `hermmulx(c1) * 2`. The `len(c)==1` branch's `c1 = 0` is represented as
/// `vec![T::zero()]` so it flows through the same combine step as every
/// other branch.
pub fn hermemul<T: HermeScalar>(c1_in: &[T], c2_in: &[T]) -> Vec<T> {
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
            let nd_m1 = T::from_usize(nd - 1);
            let scaled_top = scale(xs, c[c.len() - i]);
            // numpy: `c1 * (nd - 1)` -- c1 (left token) first, NO leading
            // `2 *`.
            // Ticket #84 (2026-08-08, Monday): `hermesub(...)`/
            // `hermeadd(...)` below are real `pu._sub`/`pu._add` calls,
            // which trim BOTH operands via `as_series` before the padded
            // combine -- same fix as `legendre.rs`/`laguerre.rs`/
            // `hermite.rs`, see those doc comments for the measured
            // signed-zero divergence this avoids. `pad_sub`'s OWN
            // untrimmed-same-length usage inside `hermediv`'s loop is
            // unaffected.
            let c1_scaled: Vec<T> = c1v.iter().map(|&v| v.herme_mul(nd_m1)).collect();
            let scaled_top_t = trim_trailing_zeros(&scaled_top);
            let c1_scaled_t = trim_trailing_zeros(&c1_scaled);
            c0 = pad_sub(&scaled_top_t, &c1_scaled_t);
            // numpy: `hermemulx(c1)` -- NO trailing `*2`.
            let mx = hermemulx(&c1v);
            let tmp_t = trim_trailing_zeros(&tmp);
            let mx_t = trim_trailing_zeros(&mx);
            c1v = pad_add(&tmp_t, &mx_t);
        }
        (c0, c1v)
    };
    // numpy: `hermeadd(c0, hermemulx(c1))` -- NO trailing `*2`.
    let mx = hermemulx(&c1v);
    let c0_t = trim_trailing_zeros(&c0);
    let mx_t = trim_trailing_zeros(&mx);
    let out = pad_add(&c0_t, &mx_t);
    trim_trailing_zeros(&out)
}

/// `hermediv`: `polyutils._div(hermemul, c1, c2)`, the same basis-agnostic
/// repeated-subtraction shape as `hermdiv`/`lagdiv`/`legdiv`, but built on
/// THIS module's own `hermemul` so its bit pattern depends on `hermemul`'s
/// (unscaled) grouping.
pub fn hermediv<T: HermeScalar>(c1_in: &[T], c2_in: &[T]) -> Result<(Vec<T>, Vec<T>), ()> {
    // numpy: `[c1, c2] = as_series([c1, c2])` -- BOTH operands are trimmed
    // copies before any length check or return value is built. Ticket #87
    // (2026-08-10, Monday): this port never did that top-level trim (only
    // the sign-of-zero fix below had landed, via #84/#85's shared bug
    // class), so an input carrying an exact trailing zero on `c2` could
    // false-positive the `c2.last() == 0` zero-divisor check (numpy trims
    // it away and divides fine), and an exact trailing zero on `c1` leaked
    // an untrimmed tail into the returned remainder -- a SHAPE divergence,
    // not merely a sign one. Matches the pre-trim `chebdiv`/`legdiv`/
    // `lagdiv`/`hermdiv`/`poly::div` already do.
    let c1 = trim_trailing_zeros(c1_in);
    let c2 = trim_trailing_zeros(c2_in);
    if *c2.last().unwrap() == T::zero() {
        return Err(());
    }
    let lc1 = c1.len();
    let lc2 = c2.len();
    if lc1 < lc2 {
        return Ok((vec![c1[0].herme_mul(T::zero())], c1));
    }
    if lc2 == 1 {
        let scl = c2[0];
        let q: Vec<T> = c1.iter().map(|&v| v.herme_div(scl)).collect();
        return Ok((q, vec![c1[0].herme_mul(T::zero())]));
    }
    let mut quo = vec![T::zero(); lc1 - lc2 + 1];
    let mut rem = c1.clone();
    for i in (0..=lc1 - lc2).rev() {
        let mut basis = vec![T::zero(); i + 1];
        basis[i] = T::one();
        let p = hermemul(&basis, &c2);
        let q = rem.last().unwrap().herme_div(*p.last().unwrap());
        let rem_head = &rem[..rem.len() - 1];
        let p_head = &p[..p.len() - 1];
        // numpy: `q * p[:-1]` -- q (scalar, left token) first.
        let scaled_p_head: Vec<T> = p_head.iter().map(|&v| q.herme_mul(v)).collect();
        rem = pad_sub(rem_head, &scaled_p_head);
        quo[i] = q;
    }
    let rem_trimmed = trim_trailing_zeros(&rem);
    Ok((quo, rem_trimmed))
}

/// `hermeder`: DIRECT ASSIGNMENT (`der[j-1] = j*c[j]`), ported from
/// `hermite_e.py:1090-1096` -- NO leading `2 *`, contrast `hermder`'s
/// `(2*j)*c[j]`.
pub fn hermeder<T: HermeScalar>(c_in: &[T], cnt: usize, scl: T) -> Vec<T> {
    let n0 = c_in.len();
    if cnt == 0 {
        return c_in.to_vec();
    }
    if cnt >= n0 {
        // numpy: `c = c[:1] * 0` -- sign-preserving.
        return vec![c_in[0].herme_mul(T::zero())];
    }
    let mut c = c_in.to_vec();
    let mut n = n0;
    for _ in 0..cnt {
        n -= 1;
        for v in c.iter_mut() {
            *v = v.herme_mul(scl);
        }
        let mut der = vec![T::zero(); n];
        for j in (1..=n).rev() {
            // numpy: `der[j - 1] = j * c[j]` -- `j` (left token) first, NO
            // leading `2 *`.
            der[j - 1] = T::from_usize(j).herme_mul(c[j]);
        }
        c = der;
    }
    c
}

/// `hermeint`: ported from `hermite_e.py:1179-1187`. `j`-loop is DIRECT
/// ASSIGNMENT (`tmp[j+1] = c[j]/(j+1)`) -- NO leading `2 *` in the
/// denominator, contrast `hermint`'s `c[j]/(2*(j+1))`. `tmp[1] = c[0]` --
/// NO `/2`, contrast `hermint`'s `c[0]/2`. The final `tmp[0] + (ki -
/// at_lbnd)` grouping is load-bearing (associativity), same trap as
/// `hermint`'s identical final line.
pub fn hermeint<T: HermeScalar>(c_in: &[T], cnt: usize, k: &[T], lbnd: T, scl: T) -> Vec<T> {
    let mut c = c_in.to_vec();
    if cnt == 0 {
        return c;
    }
    for &ki in k.iter().take(cnt) {
        let n = c.len();
        for v in c.iter_mut() {
            *v = v.herme_mul(scl);
        }
        if n == 1 && c[0] == T::zero() {
            c[0] = c[0] + ki;
        } else {
            let mut tmp = vec![T::zero(); n + 1];
            // numpy: `tmp[0] = c[0] * 0` -- sign-preserving.
            tmp[0] = c[0].herme_mul(T::zero());
            tmp[1] = c[0];
            for j in 1..n {
                tmp[j + 1] = c[j].herme_div(T::from_usize(j + 1));
            }
            let at_lbnd = herme_eval(&tmp, &[lbnd])[0];
            // numpy: `tmp[0] += k[i] - hermeval(lbnd, tmp)`, i.e.
            // `tmp[0] + (ki - at_lbnd)`.
            tmp[0] = tmp[0] + (ki - at_lbnd);
            c = tmp;
        }
    }
    c
}

/// `hermevander(x, deg)`: `V[i, j] = He_j(x[i])`, ported from
/// `hermite_e.py:1350-1360`. `v[1] = x` (plain, NOT `x2`) and
/// `v[i] = v[i-1]*x - v[i-2]*(i-1)` (no `2*` factor) -- the sharpest
/// single-bit divergence from `hermvander`'s `v[1] = x2`.
pub fn hermevander<T: HermeScalar>(xs: &[T], deg: usize) -> Vec<T> {
    let width = deg + 1;
    let mut out = vec![T::zero(); xs.len() * width];
    for (row, &x) in xs.iter().enumerate() {
        let base = row * width;
        // numpy: `v[0] = x * 0 + 1` -- `x` (left token) first.
        out[base] = x.herme_mul(T::zero()) + T::one();
        if deg > 0 {
            out[base + 1] = x;
            for i in 2..width {
                // numpy: `v[i] = v[i-1]*x - v[i-2]*(i-1)` -- `v[i-1]`/
                // `v[i-2]` (left tokens) first in both products, plain `x`,
                // NO leading `2 *` on the subtracted term.
                let a = out[base + i - 1].herme_mul(x);
                let b = out[base + i - 2].herme_mul(T::from_usize(i - 1));
                out[base + i] = a - b;
            }
        }
    }
    out
}

/// `hermecompanion(c)`: SCALED, ported from `hermite_e.py:1415-1452`. `scl`
/// (the reversed cumulative-product scale vector) and the super/subdiagonal
/// values (`sqrt(k+1)`, no `.5` factor) are ALWAYS real-valued regardless of
/// `c`'s own dtype, so they are computed in plain `f64` here and converted
/// via `T::from_f64` only at the point they combine with `c`'s own
/// elements.
pub fn hermecompanion<T: HermeScalar>(c: &[T]) -> Vec<T> {
    let len = c.len();
    if len == 2 {
        // numpy: `-c[0] / c[1]` -- Python parses this as `(-c[0]) / c[1]`
        // (unary minus binds tighter than `/`): negate FIRST, then divide.
        // NO `-0.5` factor at all (contrast `hermcompanion`'s
        // `-.5 * c[0] / c[1]`).
        return vec![(-c[0]).herme_div(c[1])];
    }
    let n = len - 1;
    let mut mat = vec![T::zero(); n * n];

    // scl_raw = hstack((1., 1./sqrt(arange(n-1,0,-1)))) -- entries
    // 1, 1/sqrt(n-1), 1/sqrt(n-2), ..., 1/sqrt(1). NO `2.` inside the
    // sqrt, contrast hermcompanion's `1/sqrt(2*(n-j))`.
    let mut scl_raw = vec![1.0f64; n];
    for (j, item) in scl_raw.iter_mut().enumerate().take(n).skip(1) {
        *item = 1.0 / ((n - j) as f64).sqrt();
    }
    // scl = multiply.accumulate(scl_raw)[::-1] -- left-to-right cumulative
    // product, then reversed.
    let mut scl_cum = vec![0.0f64; n];
    scl_cum[0] = scl_raw[0];
    for j in 1..n {
        scl_cum[j] = scl_cum[j - 1] * scl_raw[j];
    }
    let scl: Vec<f64> = (0..n).map(|i| scl_cum[n - 1 - i]).collect();

    // top[...] = sqrt(arange(1,n)); bot[...] = top -- superdiagonal and
    // subdiagonal both get sqrt(k+1) for k = 0..n-2 -- NO `.5` factor,
    // contrast hermcompanion's `sqrt(0.5*(k+1))`.
    for k in 0..n.saturating_sub(1) {
        let v = T::from_f64(((k as f64) + 1.0).sqrt());
        mat[k * n + (k + 1)] = v;
        mat[(k + 1) * n + k] = v;
    }

    // mat[:, -1] -= scl * c[:-1] / c[-1] -- NO `2.0` in the denominator,
    // contrast hermcompanion's `/(2.0*c[-1])`.
    let cn = c[len - 1];
    for i in 0..n {
        let num = T::from_f64(scl[i]).herme_mul(c[i]);
        let corr = num.herme_div(cn);
        mat[i * n + (n - 1)] = mat[i * n + (n - 1)] - corr;
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn herme_eval_matches_numpy_example() {
        // He.hermeval(1, [1,2,3]) -> 3.0 (verified live)
        let c = [1.0_f64, 2.0, 3.0];
        let out = herme_eval(&c, &[1.0]);
        assert!((out[0] - 3.0).abs() < 1e-8, "{}", out[0]);
    }

    #[test]
    fn hermemulx_matches_numpy_example() {
        // He.hermemulx([1,2,3]) -> [2, 7, 2, 3] (verified live)
        let c = [1.0_f64, 2.0, 3.0];
        let out = hermemulx(&c);
        let expect = [2.0, 7.0, 2.0, 3.0];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn hermemul_matches_numpy_example() {
        // He.hermemul([1,2,3],[0,1,2]) -> verified live: [14., 15., 28., 7., 6.]
        let c1 = [1.0_f64, 2.0, 3.0];
        let c2 = [0.0_f64, 1.0, 2.0];
        let out = hermemul(&c1, &c2);
        let expect = [14.0, 15.0, 28.0, 7.0, 6.0];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn hermeder_matches_numpy_example() {
        // He.hermeder([1,2,3,4]) -> [2, 6, 12] (verified live)
        let c = [1.0_f64, 2.0, 3.0, 4.0];
        let d1 = hermeder(&c, 1, 1.0);
        for (a, b) in d1.iter().zip([2.0, 6.0, 12.0].iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn hermeint_matches_numpy_example() {
        // He.hermeint([1,2,3]) -> [1, 1, 1, 1] (verified live)
        let c = [1.0_f64, 2.0, 3.0];
        let out = hermeint(&c, 1, &[0.0], 0.0, 1.0);
        let expect = [1.0, 1.0, 1.0, 1.0];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn hermevander_matches_numpy_example() {
        // He.hermevander([0,1,2], 3) row1 = [1,1,0,-2] (verified live)
        let xs = [0.0_f64, 1.0, 2.0];
        let v = hermevander(&xs, 3);
        let expect_row1 = [1.0, 1.0, 0.0, -2.0];
        for (a, b) in v[4..8].iter().zip(expect_row1.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn hermecompanion_matches_numpy_example() {
        // He.hermecompanion([1,0,1]) -> [[0, 0],[1, 0]] (verified live)
        let c = [1.0_f64, 0.0, 1.0];
        let m = hermecompanion(&c);
        let expect = [0.0, 0.0, 1.0, 0.0];
        for (a, b) in m.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }
}
