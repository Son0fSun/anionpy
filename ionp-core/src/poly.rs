//! Power-series polynomial kernels — the Rust core for
//! `numpy.polynomial.polynomial` (the "power basis" submodule: plain
//! `c[0] + c[1]*x + ... + c[n]*x^n`, as opposed to Chebyshev/Legendre/
//! Laguerre/Hermite/HermiteE, which are out of scope for this pass).
//!
//! Every numerical recurrence numpy's own `numpy/polynomial/polynomial.py`
//! runs as a Python `for` loop (Horner's method in `polyval`, synthetic
//! division in `polydiv`, the differentiation/integration loops in
//! `polyder`/`polyint`, the O(n*m) convolution inlined via `np.convolve`
//! in `polymul`/`polypow`) lives here instead, as real Rust loops over
//! real Rust slices. `anionpy/polynomial/polynomial.py` calls these
//! through the PyO3 boundary (`ionp-py/src/poly.rs`) and does nothing but
//! shape/dtype bookkeeping and argument assembly — no arithmetic on
//! coefficient or sample-point DATA happens in that file, mirroring every
//! other Rust/Python seam in this codebase.
//!
//! `PolyScalar` is the one generic seam: every kernel below is written
//! once and instantiated for both `f64` (the real path) and
//! `ionp_core::buffer::C128` (`num_complex::Complex<f64>`, the complex
//! path) — real numpy's own `numpy.polynomial.polyutils.as_series`
//! promotes bool/int/float input to `float64` and any complex input to
//! `complex128` via `np.common_type`, so those are the only two element
//! types this module (or numpy's) ever actually computes in.
//!
//! Every function below takes ALREADY-TRIMMED, already-promoted 1-D
//! coefficient slices (mirroring numpy's own `# c is a trimmed copy`
//! comments throughout `polynomial.py`) and leaves trimming policy
//! (whether the caller wants a trimmed result at all) to the caller —
//! `trim_trailing_zeros` is exposed separately for that reason, matching
//! numpy's own separate `trimseq`.

use crate::buffer::C128;

/// The element-type seam between the real (`f64`) and complex (`C128`)
/// instantiation of every kernel in this module. Deliberately minimal:
/// only the operations these algorithms actually use.
pub trait PolyScalar:
    Copy
    + PartialEq
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_usize(n: usize) -> Self;

    /// `self * other`, routed through `crate::ufunc::complex_mul_fma` on
    /// the `C128` instantiation instead of `num_complex::Complex`'s native
    /// `Mul` impl -- same "measure, don't assume" reasoning as every other
    /// basis module in this codebase (`chebyshev.rs`'s `cheb_mul`,
    /// `laguerre.rs`'s `lag_mul`, `legendre.rs`'s `leg_mul`): numpy's
    /// actual complex multiply ufunc matches the FMA formula exactly (0
    /// mismatches measured), not the naive two-rounding textbook formula
    /// `num_complex::Complex::mul` uses (~44% mismatches). Safe even for
    /// "one operand is a real-valued scalar lifted into `T`" sites
    /// (`T::from_usize(j) * c[j]` etc) because `complex_mul_fma`'s formula
    /// reduces to `fma(a, b, -0.0) == a * b` exactly whenever the other
    /// operand's imaginary part is `0.0` -- routing those sites uniformly
    /// through `poly_mul` cannot diverge from a bare `*`. On `f64` this is
    /// exactly `self * other`.
    fn poly_mul(self, other: Self) -> Self;

    /// `self / other`, routed through `crate::ufunc::complex_div` on the
    /// `C128` instantiation instead of `num_complex::Complex`'s native
    /// `Div` impl -- `num_complex`'s native division is NOT bit-exact
    /// against numpy's `npy_cdivide` (confirmed elsewhere in this codebase:
    /// `0.5j*3/5` gives numpy `0.30000000000000004j`, one ULP above
    /// `num_complex`'s `0.3j`). Every division site this trait method
    /// covers in this file (`polydiv`'s synthetic-division `/ c2[0]` and
    /// `/ scl`, `polyint`'s `/ (j+1)`) operates on a coefficient array
    /// whose dtype is fixed ONCE at the top of the numpy function (via
    /// `pu.as_series`/the caller's dtype promotion) and never changes
    /// mid-loop -- unlike `laguerre.rs`'s `lag_eval`/`legendre.rs`'s
    /// `leg_eval`, which run a scalar Clenshaw recursion whose accumulator
    /// variables can be numpy-scalar-typed `float64` for several loop
    /// iterations before first combining with a complex evaluation point
    /// and THEN becoming `complex128`-typed (a genuine "real until
    /// touched" transition that `complex_div`'s Smith's-algorithm
    /// reciprocal-multiply formula does NOT bit-exactly replicate for the
    /// still-real phase, since it double-rounds via `a * (1/c)` instead of
    /// `numpy`'s single-rounding real `a/c`). This file's functions never
    /// have that scalar-accumulator-plus-separately-typed-x shape --
    /// `horner_eval`'s single accumulator gets touched by `x` (multiply
    /// only, no division) on its very first loop iteration, and
    /// `div`/`int_` operate on whole fixed-dtype arrays throughout -- so
    /// routing every division here through `complex_div` is measured-safe,
    /// not merely assumed. On `f64` this is exactly `self / other`.
    fn poly_div(self, other: Self) -> Self;
}

impl PolyScalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_usize(n: usize) -> Self {
        n as f64
    }
    fn poly_mul(self, other: Self) -> Self {
        self * other
    }
    fn poly_div(self, other: Self) -> Self {
        self / other
    }
}

impl PolyScalar for C128 {
    fn zero() -> Self {
        C128::new(0.0, 0.0)
    }
    fn one() -> Self {
        C128::new(1.0, 0.0)
    }
    fn from_usize(n: usize) -> Self {
        C128::new(n as f64, 0.0)
    }
    fn poly_mul(self, other: Self) -> Self {
        crate::ufunc::complex_mul_fma(self, other)
    }
    fn poly_div(self, other: Self) -> Self {
        crate::ufunc::complex_div(self, other)
    }
}

/// numpy's `trimseq`: strip trailing zeros, never returning an empty
/// slice (an all-zero input collapses to a single `[0]`). Ported
/// mechanically from `polyutils.trimseq`'s `for i in range(len(seq)-1,
/// -1, -1): if seq[i] != 0: break` — the loop falls through to `i == 0`
/// on an all-zero (or empty-after-the-fast-path) input, exactly like the
/// Python version.
pub fn trim_trailing_zeros<T: PolyScalar>(c: &[T]) -> Vec<T> {
    if c.is_empty() {
        return Vec::new();
    }
    if *c.last().unwrap() != T::zero() {
        return c.to_vec();
    }
    let mut i = c.len() - 1;
    loop {
        if c[i] != T::zero() {
            break;
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    c[..=i].to_vec()
}

/// numpy's `polyutils.trimcoef`: like `trim_trailing_zeros` but the
/// trailing-zero test is `abs(c[i]) <= tol` instead of `== 0`. `tol` is
/// always a non-negative real magnitude bound (checked by the caller),
/// even on the complex path, so `abs_le` takes a real `f64` tolerance and
/// each scalar supplies its own magnitude.
pub fn trim_coef_f64(c: &[f64], tol: f64) -> Vec<f64> {
    let last_ok = c.iter().rposition(|&v| v.abs() > tol);
    match last_ok {
        None => vec![0.0],
        Some(i) => c[..=i].to_vec(),
    }
}

pub fn trim_coef_c128(c: &[C128], tol: f64) -> Vec<C128> {
    let last_ok = c.iter().rposition(|&v| v.norm() > tol);
    match last_ok {
        None => vec![C128::new(0.0, 0.0)],
        Some(i) => c[..=i].to_vec(),
    }
}

/// Horner's method, matching `polyval`'s recurrence exactly: `c0 = c[-1];
/// c0 = c[-i] + c0*x` for `i` in `2..=len(c)`. Evaluated independently
/// for every point in `xs` (the `tensor=True`/multi-dim broadcasting
/// numpy supports is handled by the Python caller reshaping `c`, not
/// here — this kernel only ever sees one coefficient series and a flat
/// list of evaluation points).
pub fn horner_eval<T: PolyScalar>(c: &[T], xs: &[T]) -> Vec<T> {
    xs.iter()
        .map(|&x| {
            let mut c0 = *c.last().unwrap();
            for i in (0..c.len() - 1).rev() {
                c0 = c[i] + c0.poly_mul(x);
            }
            c0
        })
        .collect()
}

/// `polyvalfromroots`: `p(x) = prod_n (x - r_n)`, evaluated independently
/// for every point in `xs`.
pub fn val_from_roots<T: PolyScalar>(xs: &[T], roots: &[T]) -> Vec<T> {
    xs.iter()
        .map(|&x| {
            let mut acc = T::one();
            for &r in roots {
                acc = acc.poly_mul(x - r);
            }
            acc
        })
        .collect()
}

/// Elementwise pad-and-add, THEN trim — `polyutils._add`: whichever
/// operand is shorter is zero-extended to the other's length before the
/// elementwise add; the result is passed through `trim_trailing_zeros`
/// exactly once, matching `_add`'s own single `trimseq(ret)` call.
///
/// COPIES the longer operand first, then folds only the overlapping
/// prefix of the shorter operand in — NOT a zero-init accumulator that
/// both operands get added into. The zero-init shape silently flips an
/// exact `-0.0` in either operand's untouched tail to `+0.0` (`0.0 + v`
/// rounds to `+0.0` for `v == -0.0` under IEEE 754 round-to-nearest even
/// though `v` itself was exactly `-0.0`), which is a real, measured
/// divergence source against numpy (which never materializes a zero
/// accumulator for the untouched tail). Matches the pattern
/// `chebyshev.rs`'s/`laguerre.rs`'s own `pad_add`/`pad_sub` already use.
///
/// Ticket #84 (2026-08-08, Monday): numpy's real `_add`/`_sub` FIRST call
/// `[c1, c2] = as_series([c1, c2])`, trimming trailing (exact) zeros off
/// EACH input independently, before the pad/combine step below -- this
/// function (and `sub_trim`) previously skipped that pre-trim and only
/// trimmed the final result. That's not equivalent: an untrimmed trailing
/// exact zero in the SHORTER operand still gets folded into the longer
/// operand's corresponding (otherwise-untouched) slot, and `+0.0 + (-0.0)`
/// (or the reverse) rounds to `+0.0` under IEEE-754 round-to-nearest even
/// when the longer operand held an exact `-0.0` there -- silently flipping
/// a sign real numpy's pre-trimmed `_add`/_sub` never touches. Every basis
/// module's `*add_trim`/`*sub_trim` (`legadd_trim`, `lagadd_trim`,
/// `hermadd_trim`, `hermeadd_trim`, `chebadd_trim`, and this module's own
/// `polyadd`/`polysub` bindings) alias straight through to this function,
/// so this single fix covers all of them. Minimal repro traced via
/// `hermmul`'s final `hermadd(c0, hermmulx(c1)*2)` combine (see
/// `hermite.rs::hermmul`'s doc comment for the exact bit pattern).
pub fn add_trim<T: PolyScalar>(a: &[T], b: &[T]) -> Vec<T> {
    let a = trim_trailing_zeros(a);
    let b = trim_trailing_zeros(b);
    let out = if a.len() >= b.len() {
        let mut out = a.clone();
        for (i, &v) in b.iter().enumerate() {
            out[i] = out[i] + v;
        }
        out
    } else {
        let mut out = b.clone();
        for (i, &v) in a.iter().enumerate() {
            out[i] = out[i] + v;
        }
        out
    };
    trim_trailing_zeros(&out)
}

/// `polyutils._sub`: `a - b`, same pad/trim shape as `add_trim` (and the
/// same copy-longer-first fix for the identical signed-zero reason, plus
/// the same ticket #84 pre-trim-both-inputs fix -- see `add_trim`'s doc
/// comment).
pub fn sub_trim<T: PolyScalar>(a: &[T], b: &[T]) -> Vec<T> {
    let a = trim_trailing_zeros(a);
    let b = trim_trailing_zeros(b);
    let out = if a.len() >= b.len() {
        let mut out = a.clone();
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
    };
    trim_trailing_zeros(&out)
}

/// Full (non-truncated) convolution, `len(a) + len(b) - 1` taps — the
/// same operation `numpy.polynomial.polynomial.polymul` gets from
/// `np.convolve(c1, c2)` (default `mode='full'`) before its own
/// `trimseq`. Also the multiplication primitive `polypow`'s repeated
/// squaring/multiplying is built from (see `ionp-py/src/poly.rs`, which
/// calls this in a loop rather than duplicating it).
pub fn convolve_full<T: PolyScalar>(a: &[T], b: &[T]) -> Vec<T> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let n = a.len() + b.len() - 1;
    let mut out = vec![T::zero(); n];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] = out[i + j] + av.poly_mul(bv);
        }
    }
    out
}

/// `polymulx`: multiply the (already-trimmed, non-zero-series) `c` by
/// `x` — prepend a zero coefficient, shifting every term up one degree.
/// The zero-series special case (`len(c) == 1 && c[0] == 0`) is handled
/// by the caller, which never calls this kernel for that input (mirrors
/// numpy's own early return before doing any array work).
pub fn mulx<T: PolyScalar>(c: &[T]) -> Vec<T> {
    let mut out = vec![T::zero(); c.len() + 1];
    // numpy: `prd[0] = c[0] * 0` -- a DERIVED zero (multiply the actual
    // first coefficient by zero), not a literal fill. For complex c[0]
    // with a negative real or imaginary part this yields a correctly
    // signed `-0.0` component that a bare `T::zero()` fill would lose
    // (measured: 1021/2000 signed-zero-only divergences before this fix).
    out[0] = c[0].poly_mul(T::zero());
    out[1..].copy_from_slice(c);
    out
}

/// `polydiv`: quotient-with-remainder via numpy's exact synthetic
/// division loop (`polynomial.py`'s `polydiv`, NOT the generic
/// repeated-subtraction `polyutils._div` — the module-level function
/// short-circuits to this dedicated loop, which is what this ports).
/// Returns `Err` for exact division by the zero series (`c2[-1] == 0`),
/// matching numpy's own bare `raise ZeroDivisionError`.
pub fn div<T: PolyScalar>(c1_in: &[T], c2_in: &[T]) -> Result<(Vec<T>, Vec<T>), ()> {
    // numpy: `[c1, c2] = pu.as_series([c1, c2])` -- BOTH operands are
    // trimmed copies before any length check, return value, or the
    // synthetic-division loop below. Ticket #85 (2026-08-08, Monday): this
    // port never did that top-level trim, so an input carrying an exact
    // trailing zero (e.g. `c1_in = [x, -0j]`) leaked the untrimmed tail
    // into the returned remainder/quotient shape and into the synthetic-
    // division loop's index bookkeeping (`lc1`/`dlen`/`j`) -- a SHAPE
    // divergence, not merely a sign one. Matches the pre-trim `chebdiv`
    // already does at its own top (`chebyshev.rs`).
    let c1_t = trim_trailing_zeros(c1_in);
    let c2 = trim_trailing_zeros(c2_in);
    if *c2.last().unwrap() == T::zero() {
        return Err(());
    }
    let lc1 = c1_t.len();
    let lc2 = c2.len();
    if lc1 < lc2 {
        // numpy: `c1[:1] * 0, c1` -- placeholder quotient is a DERIVED
        // zero (multiply the actual first coefficient by zero), not a
        // literal fill. A bare `T::zero()` silently loses the sign a
        // negative/negative-imaginary `c1[0]` produces (`(-5e-324) * 0 ==
        // -0.0` under IEEE-754) -- same bug class already fixed in
        // `mulx`'s `prd[0]` fill above and in `legendre.rs`'s/
        // `laguerre.rs`'s/`hermite.rs`'s/`hermite_e.rs`'s analogous
        // `*div` early-return branches. Remainder is the TRIMMED `c1`,
        // not the caller's original `c1_in`.
        return Ok((vec![c1_t[0].poly_mul(T::zero())], c1_t));
    }
    if lc2 == 1 {
        let q: Vec<T> = c1_t.iter().map(|&v| v.poly_div(c2[0])).collect();
        // numpy: `c1 / c2[-1], c1[:1] * 0` -- same derived-zero remainder
        // as the branch above, not a hardcoded zero.
        return Ok((q, vec![c1_t[0].poly_mul(T::zero())]));
    }
    let dlen = lc1 - lc2;
    let scl = c2[lc2 - 1];
    let c2n: Vec<T> = c2[..lc2 - 1].iter().map(|&v| v.poly_div(scl)).collect();
    let mut c1 = c1_t;
    let mut i: isize = dlen as isize;
    let mut j: isize = (lc1 - 1) as isize;
    while i >= 0 {
        let cj = c1[j as usize];
        let ii = i as usize;
        for (k, &c2v) in c2n.iter().enumerate() {
            c1[ii + k] = c1[ii + k] - c2v.poly_mul(cj);
        }
        i -= 1;
        j -= 1;
    }
    let quo_start = (j + 1) as usize;
    let quo: Vec<T> = c1[quo_start..].iter().map(|&v| v.poly_div(scl)).collect();
    let rem = trim_trailing_zeros(&c1[..quo_start]);
    Ok((quo, rem))
}

/// `polyder`: `cnt`-fold differentiation, multiplying by `scl` at each
/// stage before differentiating (so the end result carries `scl**cnt`
/// exactly like numpy's loop, not `scl` applied once at the end — those
/// differ in general for `scl` chosen adversarially, though not for the
/// scalar-multiplication case, which is why this still needs to be the
/// per-iteration loop and not `scl.powi(cnt)` applied once).
pub fn der<T: PolyScalar>(c_in: &[T], cnt: usize, scl: T) -> Vec<T> {
    let mut c = c_in.to_vec();
    if cnt == 0 {
        return c;
    }
    let mut n = c.len();
    if cnt >= n {
        // numpy: `c = c[:1] * 0` -- derived zero from the ORIGINAL
        // (pre-scaling) first coefficient, not a literal fill (same
        // signed-zero idiom as `mulx` above; see its comment).
        return vec![c[0].poly_mul(T::zero())];
    }
    for _ in 0..cnt {
        n -= 1;
        for v in c.iter_mut() {
            *v = v.poly_mul(scl);
        }
        let mut der = vec![T::zero(); n];
        for j in (1..=n).rev() {
            der[j - 1] = T::from_usize(j).poly_mul(c[j]);
        }
        c = der;
    }
    c
}

/// `polyint`: `cnt`-fold integration with per-stage constants `k` (length
/// `cnt`, zero-padded by the caller — mirrors numpy's own `k = list(k) +
/// [0] * (cnt - len(k))`) and lower bound `lbnd`. `tmp[0]` is set to `k[i]
/// - polyval(lbnd, tmp)` with `tmp[0]` still zero at evaluation time,
/// exactly like numpy's own build-then-patch order.
pub fn int_<T: PolyScalar>(c_in: &[T], cnt: usize, k: &[T], lbnd: T, scl: T) -> Vec<T> {
    let mut c = c_in.to_vec();
    if cnt == 0 {
        return c;
    }
    for &ki in k.iter().take(cnt) {
        let n = c.len();
        for v in c.iter_mut() {
            *v = v.poly_mul(scl);
        }
        if n == 1 && c[0] == T::zero() {
            c[0] = c[0] + ki;
        } else {
            let mut tmp = vec![T::zero(); n + 1];
            tmp[1] = c[0];
            for j in 1..n {
                tmp[j + 1] = c[j].poly_div(T::from_usize(j + 1));
            }
            let at_lbnd = horner_eval(&tmp, &[lbnd])[0];
            tmp[0] = ki - at_lbnd;
            c = tmp;
        }
    }
    c
}

/// `polyvander(x, deg)`: `V[i, j] = x[i]**j`, `0 <= j <= deg`, built by
/// repeated multiplication (matches numpy's own `v[i] = v[i-1] * x` loop,
/// not `x**arange(deg+1)`, so the rounding behaviour for non-exact powers
/// is identical). Row-major: `out[row*(deg+1) + col]`.
pub fn vander<T: PolyScalar>(xs: &[T], deg: usize) -> Vec<T> {
    let width = deg + 1;
    let mut out = vec![T::zero(); xs.len() * width];
    for (row, &x) in xs.iter().enumerate() {
        let base = row * width;
        out[base] = T::one();
        for i in 1..width {
            out[base + i] = out[base + i - 1].poly_mul(x);
        }
    }
    out
}

/// `polycompanion(c)`: the companion matrix of a trimmed series with
/// `len(c) >= 2`. Row-major `n x n`, `n = len(c) - 1`. Ported directly
/// from `polynomial.py`'s own construction: ones on the sub-diagonal,
/// `mat[i, -1] -= c[i] / c[-1]` down the last column.
pub fn companion<T: PolyScalar>(c: &[T]) -> Vec<T> {
    let n = c.len() - 1;
    if n == 0 {
        // Never reached by the Python dispatch (len(c) < 2 is handled
        // there, matching numpy's own ValueError), kept here only so the
        // kernel itself can't panic on an out-of-contract call.
        return Vec::new();
    }
    let mut mat = vec![T::zero(); n * n];
    for i in 0..n.saturating_sub(1) {
        mat[(i + 1) * n + i] = T::one();
    }
    let cn = c[n];
    for i in 0..n {
        mat[i * n + (n - 1)] = mat[i * n + (n - 1)] - c[i].poly_div(cn);
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horner_matches_hand_eval() {
        // p(x) = 1 + 2x + 3x^2 at x=2 -> 1+4+12=17
        let c = [1.0_f64, 2.0, 3.0];
        assert_eq!(horner_eval(&c, &[2.0]), vec![17.0]);
    }

    #[test]
    fn trim_all_zero_collapses_to_single_zero() {
        let c = [0.0_f64, 0.0, 0.0];
        assert_eq!(trim_trailing_zeros(&c), vec![0.0]);
    }

    #[test]
    fn add_pads_shorter_operand() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [3.0_f64, 2.0];
        assert_eq!(add_trim(&a, &b), vec![4.0, 4.0, 3.0]);
    }

    #[test]
    fn convolve_matches_hand_mul() {
        // (1+2x+3x^2)(3+2x+x^2) = 3 + 8x + 14x^2 + 8x^3 + 3x^4
        let a = [1.0_f64, 2.0, 3.0];
        let b = [3.0_f64, 2.0, 1.0];
        assert_eq!(convolve_full(&a, &b), vec![3.0, 8.0, 14.0, 8.0, 3.0]);
    }

    #[test]
    fn div_matches_numpy_example() {
        let c1 = [1.0_f64, 2.0, 3.0];
        let c2 = [3.0_f64, 2.0, 1.0];
        let (q, r) = div(&c1, &c2).unwrap();
        assert_eq!(q, vec![3.0]);
        assert_eq!(r, vec![-8.0, -4.0]);
    }

    #[test]
    fn der_matches_numpy_example() {
        let c = [1.0_f64, 2.0, 3.0, 4.0];
        assert_eq!(der(&c, 1, 1.0), vec![2.0, 6.0, 12.0]);
        assert_eq!(der(&c, 3, 1.0), vec![24.0]);
    }

    #[test]
    fn int_matches_numpy_example() {
        let c = [1.0_f64, 2.0, 3.0];
        assert_eq!(int_(&c, 1, &[0.0], 0.0, 1.0), vec![0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn vander_matches_numpy_example() {
        let xs = [-1.0_f64, 2.0, 3.0];
        let v = vander(&xs, 5);
        assert_eq!(
            v,
            vec![
                1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 1.0, 3.0, 9.0,
                27.0, 81.0, 243.0
            ]
        );
    }

    #[test]
    fn companion_matches_numpy_example() {
        let c = [1.0_f64, 2.0, 3.0];
        let m = companion(&c);
        // numpy: [[0, -1/3], [1, -2/3]]
        assert_eq!(m, vec![0.0, -1.0 / 3.0, 1.0, -2.0 / 3.0]);
    }
}
