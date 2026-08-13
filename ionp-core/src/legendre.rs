//! Legendre-series polynomial kernels — the Rust core for
//! `numpy.polynomial.legendre`. Sibling module to `crate::poly` (the
//! power-series basis), reusing that module's `PolyScalar` seam
//! (`f64`/`C128`) rather than redefining it — the two bases share the same
//! element-type story, only the recurrences differ.
//!
//! Every numerical loop numpy's own `numpy/polynomial/legendre.py` (and the
//! generic `polyutils._div`/`_pow`/`_fromroots` helpers it delegates to for
//! `legdiv`/`legpow`/`legfromroots`) runs as a Python `for` loop lives here
//! instead, ported index-for-index. Unlike the power-series basis, NONE of
//! Legendre's evaluation/multiplication/division machinery reduces to
//! Horner's method or plain convolution — every kernel below implements the
//! three-term Legendre recurrence
//! `x*P_i(x) = ((i+1)*P_{i+1}(x) + i*P_{i-1}(x)) / (2i+1)`
//! (or its inverse, used by `legder`/`legint`) directly, matching numpy's
//! own loops verified live against `numpy/polynomial/legendre.py` in this
//! project's venv (numpy 2.5.1), not reconstructed from memory.
//!
//! `legmul`/`legdiv` in particular are NOT the power-series basis's
//! `convolve_full`/synthetic-division: numpy's own `legdiv` is generic
//! repeated subtraction built from `legmul` (`polyutils._div(legmul, c1,
//! c2)`), and `legmul` itself is a dedicated "reprojection" recursion with
//! no power-series equivalent — both ported here as their own kernels.

use crate::buffer::C128;
use crate::poly::{trim_trailing_zeros, PolyScalar};

/// Extends `PolyScalar` with the one extra primitive the Legendre
/// recurrences need beyond power-series: constructing a scalar from a
/// plain Rust `f64` scaling factor (e.g. `1/(2i+1)`, or the `1/sqrt(2i+1)`
/// companion-matrix scale, which is always a REAL magnitude even on the
/// complex instantiation — `legcompanion` scales complex coefficient
/// entries by a real factor, it never needs a complex sqrt). Kept local to
/// this module (rather than added to `PolyScalar` itself in `poly.rs`) so
/// this task never has to touch the file another concurrent agent's power-
/// series work already shipped.
pub trait LegScalar: PolyScalar {
    fn from_f64(v: f64) -> Self;

    /// `self / other`, routed through `crate::ufunc::complex_div` on the
    /// `C128` instantiation instead of `num_complex::Complex`'s native
    /// `Div` impl. Every division in this module divides one Legendre
    /// coefficient (or coefficient-derived value) by another -- exactly
    /// the shape numpy's own `legendre.py` loops divide (`c[i]/scl`,
    /// `c[j]/(2j+1)`, `(a-b)/i`, etc., all ordinary ndarray-element /
    /// python-scalar or ndarray-element / ndarray-element divisions that
    /// go through numpy's complex divide ufunc when `c`'s dtype is
    /// complex). `num_complex::Complex`'s native `Div` is NOT bit-exact
    /// against numpy's `npy_cdivide` (confirmed via direct bit-pattern
    /// probe: `0.5j*3/5` gives numpy `0.30000000000000004j`, one ULP
    /// above `num_complex`'s `0.3j`) -- `complex_div` is this codebase's
    /// own extensively-verified (20000+ random draws, 0 mismatches)
    /// reimplementation of numpy's actual algorithm, so every division
    /// here must go through it rather than plain `/` to stay bit-exact
    /// on the complex instantiation. On `f64` this is exactly `self /
    /// other` -- real division needs no such routing.
    fn leg_div(self, other: Self) -> Self;

    /// `self * other`, routed through `crate::ufunc::complex_mul_fma` on
    /// the `C128` instantiation instead of `num_complex::Complex`'s native
    /// `Mul` impl -- the same "measure, don't assume" reasoning as
    /// `leg_div`'s doc comment (0 mismatches against real numpy's complex
    /// multiply ufunc vs. ~44% mismatches with the naive two-rounding
    /// textbook formula `num_complex::Complex::mul` uses; see
    /// `ufunc.rs::complex_mul_fma`'s own doc comment for the measurement).
    /// Every `T*T` product in this module -- including scalar-looking
    /// products like `c[i] * T::from_usize(j)` -- goes through this rather
    /// than bare `*`. This is provably safe even for the "one operand is a
    /// real-valued scalar lifted into `T` via `from_usize`/`from_f64`"
    /// case: `complex_mul_fma`'s formula reduces to `fma(a, b, -0.0)` on
    /// each component whenever the other operand's imaginary part is
    /// exactly `0.0`, and `fma(a, b, -0.0) == a * b` always holds in IEEE
    /// 754 (the FMA's addend contributes nothing when it's an exact zero
    /// input to an already-exact product-rounding step) -- so routing
    /// these sites uniformly through `leg_mul` cannot introduce any
    /// divergence a bare `*` wouldn't already have. On `f64` this is
    /// exactly `self * other`.
    fn leg_mul(self, other: Self) -> Self;
}

impl LegScalar for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }

    fn leg_div(self, other: Self) -> Self {
        self / other
    }

    fn leg_mul(self, other: Self) -> Self {
        self * other
    }
}

impl LegScalar for C128 {
    fn from_f64(v: f64) -> Self {
        C128::new(v, 0.0)
    }

    fn leg_div(self, other: Self) -> Self {
        crate::ufunc::complex_div(self, other)
    }

    fn leg_mul(self, other: Self) -> Self {
        crate::ufunc::complex_mul_fma(self, other)
    }
}

/// Elementwise scale: `s * c[i]` for every coefficient. The primitive
/// numpy's `c[0] * xs` / `c[i] * xs` (a scalar Legendre coefficient times a
/// whole trailing coefficient array) reduces to in `legmul` below.
///
/// Multiplies `s.leg_mul(v)` -- SCALAR FIRST, array element second -- not
/// `v.leg_mul(s)` (this function's shape until this fix). On `f64` the two
/// are identical (real multiplication is commutative bit-for-bit). On
/// `C128` they are NOT: `complex_mul_fma`'s formula is asymmetric under
/// argument swap (`im = fma(x.re, y.im, x.im*y.re)` uses a genuinely
/// different rounding than `fma(y.re, x.im, y.im*x.re)` -- the FMA'd pair
/// and the plain-multiplied addend trade places, and the addend is a
/// DIFFERENT real product in each ordering, not just a reordering of the
/// same one). numpy's own `legmul`/`polyutils._div` source (verified live,
/// numpy 2.5.1) always writes the scalar operand first at every call site
/// this function backs (`c[0] * xs`, `c[-i] * xs`, `q * p[:-1]`, etc.) --
/// every call site here passes `scale(array_arg, scalar_arg)`, so this
/// function must multiply scalar-first to match. Measured: the OLD
/// `v.leg_mul(s)` (array-first) order produced a genuine 1-ULP divergence
/// on the minimal `legmul` complex128 case (two length-1 series), confirmed
/// via `math.fma` bit-pattern replication of both orderings against real
/// numpy's actual product -- `fma(c[0], xs, ...)`-ordering matched numpy
/// bit-for-bit, `fma(xs, c[0], ...)`-ordering did not. See this task's
/// report (ticket #48) for the full trace.
fn scale<T: LegScalar>(c: &[T], s: T) -> Vec<T> {
    c.iter().map(|&v| s.leg_mul(v)).collect()
}

/// Pad-to-common-length elementwise subtract, WITHOUT the final
/// `trim_trailing_zeros` `add_trim`/`sub_trim` (in `crate::poly`) always
/// apply. `legdiv`'s inner loop (ported from `polyutils._div`) needs exact
/// positional `rem[:-1] - q*p[:-1]` on same-length slices at every step,
/// not a "polynomial subtract, then trim" — trimming there would silently
/// shorten `rem` mid-loop and desync the index bookkeeping the ported loop
/// depends on (real numpy's own version is a bare ndarray subtraction,
/// which raises on a length mismatch rather than padding; the lengths are
/// always exactly equal here by construction, so `pad_sub`'s padding path
/// is unreachable in practice and is kept only so the kernel can't panic on
/// a malformed call).
///
/// COPIES the longer operand first, then folds only the overlapping
/// prefix of the shorter operand in — NOT the "zero-init an accumulator,
/// add both operands into it" shape this function previously had. That
/// zero-init shape silently flips an exact `-0.0` in either operand's
/// untouched tail to `+0.0` (`0.0 + (-0.0) == 0.0`, `0.0 - (-0.0) == 0.0`
/// too under IEEE 754 round-to-nearest), which is a real, measured
/// divergence source against numpy (which never materializes a zero
/// accumulator for the untouched tail — it copies the longer array's
/// values through unchanged via `bool` masking in `pu._as_series`/`_sub`).
/// Matches the pattern `chebyshev.rs`'s and `laguerre.rs`'s own
/// `pad_add`/`pad_sub` already use.
/// Pad-to-common-length elementwise add, same copy-longer-first shape as
/// `pad_sub` below (and for the identical signed-zero reason: a zero-init
/// accumulator would silently flip an exact `-0.0` in either operand's
/// untouched tail to `+0.0`). Used by `legmul`'s `c1`-accumulator update
/// and its final `c0 + xs*c1` combine, both of which numpy computes as a
/// plain `ndarray + ndarray` add on padded/aligned coefficient arrays, not
/// a zero-accumulator fold.
fn pad_add<T: LegScalar>(a: &[T], b: &[T]) -> Vec<T> {
    if a.len() >= b.len() {
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

fn pad_sub<T: LegScalar>(a: &[T], b: &[T]) -> Vec<T> {
    if a.len() >= b.len() {
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

/// `legval`: Clenshaw recursion against the Legendre three-term recurrence,
/// ported from `legendre.py`'s `legval` loop:
/// ```text
/// c0, c1 = c[-2], c[-1]           # len(c) >= 3
/// for i in 3..=len(c):
///     nd -= 1
///     tmp = c0
///     c0 = c[-i] - c1*(nd-1)/nd
///     c1 = tmp + c1*x*(2nd-1)/nd
/// return c0 + c1*x
/// ```
/// evaluated independently for every point in `xs`. `c` is NOT Horner-
/// evaluable here (unlike the power basis): the coefficients index
/// Legendre polynomials, not powers of `x`.
pub fn leg_eval<T: LegScalar>(c: &[T], xs: &[T]) -> Vec<T> {
    let n = c.len();
    xs.iter()
        .map(|&x| {
            if n == 1 {
                return c[0];
            }
            if n == 2 {
                return c[0] + c[1].leg_mul(x);
            }
            let mut nd = n;
            let mut c0 = c[n - 2];
            let mut c1 = c[n - 1];
            for i in 3..=n {
                let tmp = c0;
                nd -= 1;
                let ratio_a = T::from_f64((nd - 1) as f64 / nd as f64);
                let ratio_b = T::from_f64((2 * nd - 1) as f64 / nd as f64);
                c0 = c[n - i] - c1.leg_mul(ratio_a);
                c1 = tmp + c1.leg_mul(x).leg_mul(ratio_b);
            }
            c0 + c1.leg_mul(x)
        })
        .collect()
}

/// `legmulx`: multiply a Legendre series by `x` via the three-term
/// recurrence `x*P_i = ((i+1)*P_{i+1} + i*P_{i-1}) / (2i+1)`, ported from
/// `legendre.py`'s `legmulx`. Handles the zero-series special case (`c ==
/// [0]`) internally (unlike `crate::poly::mulx`, which leaves that check to
/// its caller) because `legmul` below calls this kernel recursively on an
/// intermediate value that may itself be a bare zero series — keeping the
/// check here means every call site gets it for free.
///
/// Ticket #84 (2026-08-08, Monday): FIRST trims `c` via `trim_trailing_zeros`
/// -- numpy's own `legmulx` starts with `[c] = pu.as_series([c])`, which is
/// NOT cosmetic here: an untrimmed all-(signed-)zero `c` (e.g. `[-0j,-0j,
/// -0j]`, which arises legitimately mid-recursion when a `legmul` operand
/// underflows to zero) skips the `len(c)==1 && c[0]==0` fast-return path
/// entirely and falls into the general per-element loop below, whose
/// `prd[k] = prd[k] + ...` accumulation combines a `-0.0` with a `+0.0`
/// produced by an unrelated intermediate division -- `-0.0 + 0.0 == +0.0`
/// under IEEE 754, silently flipping the sign real numpy's trim-then-
/// early-return path preserves. Measured: `legmulx([-0j,-0j,-0j])` (complex128)
/// gave numpy `[-0.+0.j]` (length 1, sign preserved) vs. this kernel's
/// pre-fix `[0j,0j,0j,0j]` (length 4, WRONG SHAPE, sign flipped) -- a shape
/// mismatch, not merely a sign mismatch, so this was not even inside the
/// "array_equal is blind to it" class the ticket opened with. Root-caused
/// via a `#[cfg(test)]` bit-pattern trace (`dbg84_trace` in this file's test
/// module) isolating exactly which step introduced the flip. Same missing-
/// trim defect independently confirmed in `lagmulx`/`hermmulx`/
/// `hermemulx`/`chebmulx`/`poly::mulx` (fixed alongside this one) -- one
/// root cause across all six polynomial bases' `*mulx` kernel, not six
/// unrelated bugs.
pub fn legmulx<T: LegScalar>(c: &[T]) -> Vec<T> {
    let c = trim_trailing_zeros(c);
    let c = c.as_slice();
    if c.len() == 1 && c[0] == T::zero() {
        return c.to_vec();
    }
    let n = c.len();
    let mut prd = vec![T::zero(); n + 1];
    // numpy: `prd[0] = c[0] * 0` -- derived zero, not a literal fill (same
    // signed-zero idiom as `crate::poly::mulx`; see its comment).
    prd[0] = c[0].leg_mul(T::zero());
    prd[1] = c[0];
    for i in 1..n {
        let j = i + 1;
        let k = i - 1;
        let s = T::from_usize(i + j);
        prd[j] = c[i].leg_mul(T::from_usize(j)).leg_div(s);
        prd[k] = prd[k] + c[i].leg_mul(T::from_usize(i)).leg_div(s);
    }
    prd
}

/// `legadd`/`legsub` are exactly `crate::poly::add_trim`/`sub_trim` (numpy
/// itself: `legadd = pu._add`, `legsub = pu._sub` — the SAME generic
/// pad-and-add helper the power-series basis uses, shared across every
/// basis in `polyutils.py`). Re-exported here under the Legendre-specific
/// names this module's own kernels (`legmul`, `legdiv`) call them by, so
/// this file never needs `crate::poly::add_trim` sprinkled through it.
pub use crate::poly::add_trim as legadd_trim;
pub use crate::poly::sub_trim as legsub_trim;

/// `legmul`: the "reprojection" recursion ported from `legendre.py`'s
/// `legmul` — NOT a convolution (unlike the power basis's `polymul`, which
/// is exactly `np.convolve`). `xs` here is numpy's own local name for the
/// LONGER of the two trimmed input series (an unfortunate collision with
/// this project's usual "x = sample point" convention, kept because it
/// matches the ported algorithm's own variable, not sample points).
pub fn legmul<T: LegScalar>(c1: &[T], c2: &[T]) -> Vec<T> {
    // numpy's own `legmul` starts with `[c1, c2] = pu.as_series([c1, c2])`,
    // which trims trailing zeros off EACH input before anything else runs.
    // This is not cosmetic: `xs` (below) is the longer of the two operands,
    // scaled wholesale and fed through `legmulx`'s cross-index recurrence
    // every loop iteration, so an untrimmed trailing zero changes `xs`'s
    // LENGTH (and therefore every intermediate array's length/shape through
    // the loop) even though the extra tail is mathematically zero-valued at
    // every step -- measured to produce a real 2-ULP bit-pattern difference
    // against real numpy on the differential corpus's `legmul`
    // "trailing_zero_operand" case (`legmul([1,2,3,0,0], [3,2,1])`) despite
    // the printed decimal values matching, confirmed via a byte-level
    // `.tobytes()` comparison of a hand-simulated trimmed-vs-untrimmed run
    // against real numpy's actual output. Trimming here (not just at the
    // final `trim_trailing_zeros` this function already applies to its
    // return value) is what real numpy does and is required for bit-exact
    // agreement, not merely "equally valid but differently rounded".
    let c1 = trim_trailing_zeros(c1);
    let c2 = trim_trailing_zeros(c2);
    let (c1, c2) = (c1.as_slice(), c2.as_slice());
    let (c, xs): (&[T], &[T]) = if c1.len() > c2.len() { (c2, c1) } else { (c1, c2) };

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
            // Grouped to match numpy's own `(c1 * (nd - 1)) / nd` / `(legmulx(c1)
            // * (2 * nd - 1)) / nd` EXACTLY: multiply the whole array by the
            // INTEGER first, then divide the result by `nd` -- NOT (as this port
            // previously did) precompute the scalar ratio `(nd-1)/nd` first and
            // multiply by that. Both are mathematically identical in infinite
            // precision but regroup the floating-point rounding differently;
            // the precomputed-ratio form measured a 2-ULP mismatch against real
            // numpy on the differential corpus's `legmul` "trailing_zero_operand"
            // case (`legmul([1,2,3,0,0], [3,2,1])`). `leg_eval`'s OWN ratio
            // computation (`c1 * ((nd-1)/nd)`, precomputed-scalar-first) is a
            // SEPARATE, deliberately different grouping -- verified directly
            // against `legval`'s actual source, which groups it that way -- so
            // this is not "the same fix applied consistently", it is "the two
            // functions round differently in real numpy and must be ported
            // according to each one's own literal grouping".
            let nd_m1 = T::from_usize(nd - 1);
            let two_nd_m1 = T::from_usize(2 * nd - 1);
            let ndf = T::from_usize(nd);
            let scaled_top = scale(xs, c[c.len() - i]);
            let c1v_scaled: Vec<T> = c1v.iter().map(|&v| v.leg_mul(nd_m1).leg_div(ndf)).collect();
            // Ticket #84 (2026-08-08, Monday): `c0 = legsub(...)` and
            // `c1 = legadd(...)` below are numpy's ACTUAL `legsub`/`legadd`
            // calls, i.e. `pu._sub`/`pu._add`, which trim BOTH operands via
            // `as_series` before the padded combine -- not the raw
            // same-length subtract `legdiv`'s inner loop needs (that one
            // intentionally stays untrimmed, see `pad_sub`'s doc comment).
            // Skipping this pre-trim lets an untouched trailing `-0.0` in
            // the longer operand collide with a spurious `+0.0` at a
            // position that real numpy's trimmed-first `_add`/`_sub` never
            // touches, flipping its sign. Measured on `hermmul`'s analogous
            // combine step (`hermadd(c0, hermmulx(c1)*2)`), same defect
            // class, same fix required here.
            let scaled_top_t = trim_trailing_zeros(&scaled_top);
            let c1v_scaled_t = trim_trailing_zeros(&c1v_scaled);
            c0 = pad_sub(&scaled_top_t, &c1v_scaled_t);
            let mx = legmulx(&c1v);
            let mx_scaled: Vec<T> = mx.iter().map(|&v| v.leg_mul(two_nd_m1).leg_div(ndf)).collect();
            let tmp_t = trim_trailing_zeros(&tmp);
            let mx_scaled_t = trim_trailing_zeros(&mx_scaled);
            c1v = pad_add(&tmp_t, &mx_scaled_t);
        }
        (c0, c1v)
    };

    let mxc1 = legmulx(&c1v);
    // Same pre-trim requirement as above: numpy's final `legadd(c0, mxc1)`
    // is a real `pu._add` call, trimming both operands first.
    let c0_t = trim_trailing_zeros(&c0);
    let mxc1_t = trim_trailing_zeros(&mxc1);
    trim_trailing_zeros(&pad_add(&c0_t, &mxc1_t))
}

/// `legdiv`: `polyutils._div(legmul, c1, c2)` — generic quotient-with-
/// remainder via repeated subtraction of `c2` multiplied (in the Legendre
/// basis, via `legmul`) by the appropriate basis element, NOT the power
/// basis's synthetic-division shortcut (that shortcut is specific to
/// `polydiv`; every OTHER basis, including this one, goes through this
/// generic O(n) `legmul`-calls loop instead — see this task's report for
/// why that distinction matters).
pub fn legdiv<T: LegScalar>(c1_in: &[T], c2_in: &[T]) -> Result<(Vec<T>, Vec<T>), ()> {
    // numpy: `[c1, c2] = as_series([c1, c2])` -- BOTH operands are trimmed
    // copies before any length check or return value is built. Ticket #85
    // (2026-08-08, Monday): this port never did that top-level trim, so an
    // input carrying an exact trailing zero (e.g. `c1_in = [x, -0j]`, which
    // is a genuine `lc1 < lc2` scenario once trimmed to `[x]`) leaked the
    // untrimmed tail into the returned remainder -- a SHAPE divergence
    // (`[x]` vs numpy's `[x]`... i.e. `[x, -0j]`), not merely a sign one.
    // Matches the pre-trim `chebdiv` already does at its own top (see
    // `chebyshev.rs`); `legmul`/`legadd`/`legsub` etc. all re-trim
    // internally so passing the already-trimmed `c1`/`c2` through costs
    // nothing extra.
    let c1 = trim_trailing_zeros(c1_in);
    let c2 = trim_trailing_zeros(c2_in);
    if *c2.last().unwrap() == T::zero() {
        return Err(());
    }
    let lc1 = c1.len();
    let lc2 = c2.len();
    if lc1 < lc2 {
        // numpy: `c1[:1] * 0, c1` -- the placeholder quotient is a DERIVED
        // zero (multiply the actual first coefficient by zero), not a
        // literal fill. A bare `T::zero()` silently loses the sign a
        // negative/negative-imaginary `c1[0]` produces (`(-5e-324) * 0 ==
        // -0.0` under IEEE-754) -- same bug class already fixed in
        // `lagdiv`/`hermdiv`/`hermediv`'s analogous branches (see
        // `laguerre.rs`'s `lagdiv`) and in `legmulx`/`mulx`'s `prd[0]`
        // fill above. The remainder returned is the TRIMMED `c1`, not the
        // caller's original `c1_in`.
        return Ok((vec![c1[0].leg_mul(T::zero())], c1));
    }
    if lc2 == 1 {
        let scl = c2[0];
        let q: Vec<T> = c1.iter().map(|&v| v.leg_div(scl)).collect();
        // numpy: `c1 / c2[-1], c1[:1] * 0` -- same derived-zero remainder
        // as the branch above, not a hardcoded zero.
        return Ok((q, vec![c1[0].leg_mul(T::zero())]));
    }
    let mut quo = vec![T::zero(); lc1 - lc2 + 1];
    let mut rem = c1.clone();
    for i in (0..=lc1 - lc2).rev() {
        // basis = [0]*i + [1] -- the degree-i Legendre "unit" series.
        let mut basis = vec![T::zero(); i + 1];
        basis[i] = T::one();
        let p = legmul(&basis, &c2);
        let q = rem.last().unwrap().leg_div(*p.last().unwrap());
        let rem_head = &rem[..rem.len() - 1];
        let p_head = &p[..p.len() - 1];
        let scaled_p_head = scale(p_head, q);
        rem = pad_sub(rem_head, &scaled_p_head);
        quo[i] = q;
    }
    let rem_trimmed = trim_trailing_zeros(&rem);
    Ok((quo, rem_trimmed))
}

/// `legder`: `cnt`-fold differentiation, ported line-for-line from
/// `legendre.py`'s `legder` inner loop (`for j in range(n, 2, -1): der[j-1]
/// = (2j-1)*c[j]; c[j-2] += c[j]`, then the `n>1`/`der[1]`/`der[0]`
/// special-cased tail entries) — structurally different from
/// `crate::poly::der`'s single closed-form `T::from_usize(j) * c[j]` loop,
/// because differentiating a Legendre series does NOT resemble
/// differentiating a power series (see the module's own docstring).
pub fn legder<T: LegScalar>(c_in: &[T], cnt: usize, scl: T) -> Vec<T> {
    let mut c = c_in.to_vec();
    if cnt == 0 {
        return c;
    }
    let mut n = c.len();
    if cnt >= n {
        // numpy: `c = c[:1] * 0` -- derived zero from the ORIGINAL
        // (pre-scaling) first coefficient, not a literal fill (same
        // signed-zero idiom as `legmulx` above; see its comment).
        return vec![c[0].leg_mul(T::zero())];
    }
    for _ in 0..cnt {
        n -= 1;
        for v in c.iter_mut() {
            *v = v.leg_mul(scl);
        }
        let mut der = vec![T::zero(); n];
        for j in (3..=n).rev() {
            der[j - 1] = T::from_usize(2 * j - 1).leg_mul(c[j]);
            c[j - 2] = c[j - 2] + c[j];
        }
        if n > 1 {
            der[1] = T::from_usize(3).leg_mul(c[2]);
        }
        der[0] = c[1];
        c = der;
    }
    c
}

/// `legint`: `cnt`-fold integration, ported from `legendre.py`'s `legint`
/// inner loop. `tmp[2] = c[1]/3` and the `t = c[j]/(2j+1); tmp[j+1] = t;
/// tmp[j-1] -= t` fan-out are the Legendre-specific antiderivative
/// coefficients (the power basis's `int_` instead divides by `j+1`
/// uniformly and has no "subtract back into `j-1`" step — again, no shared
/// shape with the power basis beyond both being `cnt`-fold loops).
pub fn legint<T: LegScalar>(c_in: &[T], cnt: usize, k: &[T], lbnd: T, scl: T) -> Vec<T> {
    let mut c = c_in.to_vec();
    if cnt == 0 {
        return c;
    }
    for &ki in k.iter().take(cnt) {
        let n = c.len();
        for v in c.iter_mut() {
            *v = v.leg_mul(scl);
        }
        if n == 1 && c[0] == T::zero() {
            c[0] = c[0] + ki;
        } else {
            let mut tmp = vec![T::zero(); n + 1];
            tmp[1] = c[0];
            if n > 1 {
                tmp[2] = c[1].leg_div(T::from_usize(3));
            }
            for j in 2..n {
                let t = c[j].leg_div(T::from_usize(2 * j + 1));
                tmp[j + 1] = t;
                tmp[j - 1] = tmp[j - 1] - t;
            }
            let at_lbnd = leg_eval(&tmp, &[lbnd])[0];
            tmp[0] = tmp[0] + ki - at_lbnd;
            c = tmp;
        }
    }
    c
}

/// `legvander(x, deg)`: `V[i, j] = L_j(x[i])`, `0 <= j <= deg`, built by
/// the SAME forward three-term recurrence `legvander` in `legendre.py`
/// itself uses (`v[i] = (v[i-1]*x*(2i-1) - v[i-2]*(i-1)) / i` — numpy's own
/// docstring notes this forward recursion is less accurate than reverse
/// recursion but more efficient, and is what real numpy actually computes,
/// so it is what this ports). Row-major: `out[row*(deg+1) + col]`, same
/// layout convention as `crate::poly::vander`.
pub fn legvander<T: LegScalar>(xs: &[T], deg: usize) -> Vec<T> {
    let width = deg + 1;
    let mut out = vec![T::zero(); xs.len() * width];
    for (row, &x) in xs.iter().enumerate() {
        let base = row * width;
        out[base] = T::one();
        if deg > 0 {
            out[base + 1] = x;
            for i in 2..width {
                let a = out[base + i - 1].leg_mul(x).leg_mul(T::from_usize(2 * i - 1));
                let b = out[base + i - 2].leg_mul(T::from_usize(i - 1));
                out[base + i] = (a - b).leg_div(T::from_usize(i));
            }
        }
    }
    out
}

/// `legcompanion(c)`: the SCALED companion matrix (symmetric when `c` is a
/// pure Legendre basis polynomial), ported from `legendre.py`'s
/// `legcompanion`. `scl[i] = 1/sqrt(2i+1)` is always a REAL magnitude
/// (computed in plain `f64`, then lifted into `T` via `LegScalar::from_f64`
/// — see this trait's doc comment) even when `c` itself is complex; numpy's
/// own version does the identical real-scale-times-possibly-complex-array
/// multiply (`np.sqrt` on a real `arange`-derived array, applied to a
/// `dtype=c.dtype` matrix).
pub fn legcompanion<T: LegScalar>(c: &[T]) -> Vec<T> {
    let len = c.len();
    if len == 2 {
        return vec![(-c[0]).leg_div(c[1])];
    }
    let n = len - 1;
    let mut mat = vec![T::zero(); n * n];
    let scl: Vec<f64> = (0..n).map(|i| 1.0 / ((2 * i + 1) as f64).sqrt()).collect();
    // top[k] = mat[k, k+1] for k in 0..n-1 (superdiagonal); bot[k] = mat[k+1, k]
    // (subdiagonal) -- numpy builds both from the SAME flat `top`/`bot`
    // aliasing view (`bot[...] = top`), which this port makes explicit as
    // two separate but value-identical loops rather than replicate the
    // aliasing trick itself.
    for k in 0..n.saturating_sub(1) {
        let val = T::from_f64((k + 1) as f64 * scl[k] * scl[k + 1]);
        mat[k * n + (k + 1)] = val;
        mat[(k + 1) * n + k] = val;
    }
    let last_scl = scl[n - 1];
    let cn = c[len - 1];
    // Grouped to match numpy's own `(c[:-1]/c[-1]) * (scl/scl[-1]) *
    // (n/(2n-1))` associativity EXACTLY -- `((c_i/cn) * (scl_i/last_scl)) *
    // n_over_2n1`, left-to-right. Precomputing `(scl_i/last_scl) *
    // n_over_2n1` first (the previous form here) regroups the
    // floating-point multiplication and diverges from numpy by 1 ULP on
    // some inputs (measured via the differential corpus's `legcompanion`
    // "degree4" case, `c=[1.0, 0.0, -2.0, 0.0, 1.0]`) even though both
    // groupings are mathematically identical in infinite precision.
    let n_over_2n1 = (n as f64) / ((2 * n - 1) as f64);
    for i in 0..n {
        let a = c[i].leg_div(cn);
        let b = T::from_f64(scl[i] / last_scl);
        let corr = a.leg_mul(b).leg_mul(T::from_f64(n_over_2n1));
        mat[i * n + (n - 1)] = mat[i * n + (n - 1)] - corr;
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;

    // p(x) = 1*P0 + 2*P1 + 3*P2 = 1 + 2x + 3*(3x^2-1)/2 at x=2:
    // P0=1, P1=2, P2=(3*4-1)/2=5.5 -> 1 + 4 + 16.5 = 21.5
    #[test]
    fn leg_eval_matches_hand_eval() {
        let c = [1.0_f64, 2.0, 3.0];
        let out = leg_eval(&c, &[2.0]);
        assert!((out[0] - 21.5).abs() < 1e-12);
    }

    #[test]
    fn legmulx_matches_numpy_example() {
        // L.legmulx([1,2,3]) -> [0.66666667, 2.2, 1.33333333, 1.8]
        let c = [1.0_f64, 2.0, 3.0];
        let out = legmulx(&c);
        let expect = [2.0 / 3.0, 2.2, 4.0 / 3.0, 1.8];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn legmul_matches_numpy_example() {
        // L.legmul((1,2,3),(3,2)) -> [4.33333333, 10.4, 11.66666667, 3.6]
        let c1 = [1.0_f64, 2.0, 3.0];
        let c2 = [3.0_f64, 2.0];
        let out = legmul(&c1, &c2);
        let expect = [13.0 / 3.0, 10.4, 35.0 / 3.0, 3.6];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn legdiv_matches_numpy_example() {
        // L.legdiv((1,2,3),(3,2,1)) -> (array([3.]), array([-8., -4.]))
        let c1 = [1.0_f64, 2.0, 3.0];
        let c2 = [3.0_f64, 2.0, 1.0];
        let (q, r) = legdiv(&c1, &c2).unwrap();
        assert_eq!(q, vec![3.0]);
        assert!((r[0] - (-8.0)).abs() < 1e-8);
        assert!((r[1] - (-4.0)).abs() < 1e-8);
    }

    #[test]
    fn legder_matches_numpy_example() {
        // L.legder((1,2,3,4)) -> [6, 9, 20]; L.legder(c,3) -> [60.]
        let c = [1.0_f64, 2.0, 3.0, 4.0];
        let d1 = legder(&c, 1, 1.0);
        assert_eq!(d1, vec![6.0, 9.0, 20.0]);
        let d3 = legder(&c, 3, 1.0);
        assert_eq!(d3, vec![60.0]);
    }

    #[test]
    fn legint_matches_numpy_example() {
        // L.legint((1,2,3)) -> [0.33333333, 0.4, 0.66666667, 0.6]
        let c = [1.0_f64, 2.0, 3.0];
        let out = legint(&c, 1, &[0.0], 0.0, 1.0);
        let expect = [1.0 / 3.0, 0.4, 2.0 / 3.0, 0.6];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn legvander_matches_legval_identity() {
        // np.dot(legvander(x, n), c) == legval(x, c) up to roundoff.
        let xs = [-1.0_f64, 0.3, 2.5];
        let c = [1.0_f64, 2.0, 3.0, 4.0];
        let v = legvander(&xs, 3);
        let direct = leg_eval(&c, &xs);
        for (row, &want) in xs.iter().enumerate() {
            let _ = want;
            let mut acc = 0.0;
            for col in 0..4 {
                acc += v[row * 4 + col] * c[col];
            }
            assert!((acc - direct[row]).abs() < 1e-8);
        }
    }

    /// Ticket #84 regression: `legmulx` on an all-signed-negative-zero
    /// complex128 series must trim to length 1 and preserve the `-0.0`
    /// sign, matching real numpy's `[c] = pu.as_series([c])` + `len(c)==1
    /// && c[0]==0` early return -- see this fn's own doc comment for the
    /// full trace of how the pre-fix version both mis-shaped (length 4,
    /// not 1) AND flipped the sign (to `+0.0`) on this exact input.
    #[test]
    fn legmulx_underflow_all_zero_preserves_sign_and_shape() {
        let c = vec![C128::new(-0.0, 0.0); 3];
        let out = legmulx(&c);
        assert_eq!(out.len(), 1, "must trim to length 1 like numpy's as_series");
        assert_eq!(out[0].re.to_bits(), (-0.0f64).to_bits(), "real part must stay -0.0");
        assert_eq!(out[0].im.to_bits(), (0.0f64).to_bits(), "imag part must be +0.0");
    }

    /// Ticket #84 regression: the exact repro from the ticket (underflow-
    /// to-signed-zero feeding `legmul`'s multi-term recursion) must match
    /// numpy's `-0.+0.j` bit-for-bit, not just numerically.
    #[test]
    fn legmul_underflow_signed_zero_matches_numpy() {
        let tiny = 5e-324f64;
        let c1 = vec![C128::new(tiny, 0.0); 3];
        let c2 = vec![C128::new(-tiny, 0.0); 2];
        let real = legmul(&c1, &c2);
        assert_eq!(real.len(), 1);
        assert_eq!(real[0].re.to_bits(), (-0.0f64).to_bits(), "numpy gives -0.0 real part here");
        assert_eq!(real[0].im.to_bits(), (0.0f64).to_bits());
    }

    #[test]
    fn legcompanion_is_symmetric() {
        let c = [1.0_f64, 0.0, -2.0, 0.0, 1.0];
        let n = c.len() - 1;
        let m = legcompanion(&c);
        for i in 0..n {
            for j in 0..n {
                // off-diagonal-from-last-column symmetry (subdiagonal ==
                // superdiagonal); the last column carries the c-dependent
                // correction and is not expected to be symmetric with its
                // transpose row in general.
                if j != n - 1 && i != n - 1 {
                    assert!((m[i * n + j] - m[j * n + i]).abs() < 1e-12);
                }
            }
        }
    }
}
