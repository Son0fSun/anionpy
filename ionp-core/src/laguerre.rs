//! Laguerre-series polynomial kernels — the Rust core for
//! `numpy.polynomial.laguerre`. Sibling module to `crate::legendre`, but
//! NOT a relabeled copy of it: `docs/POLY-BASIS-SOURCE-AUDIT.md` (read in
//! full before this file was written) documents, and this port preserves,
//! several arithmetic-grouping and even structural divergences from the
//! Legendre basis that look identical at a glance and are not:
//!
//! - `lagval`'s Clenshaw ratio is grouped `(c1 * (nd - 1)) / nd`
//!   (multiply-then-divide); `legval`'s is `c1 * ((nd - 1) / nd)`
//!   (divide-then-multiply). Confirmed character-by-character against
//!   `numpy/polynomial/laguerre.py:885` / `legendre.py:907`.
//! - `lagmulx` has NO division anywhere (pure three-term integer-coefficient
//!   recurrence); `legmulx` divides by `s = i + j`.
//! - `lagmul`'s final combining step is `lagadd(c0, lagsub(c1, lagmulx(c1)))`
//!   — it SUBTRACTS `lagmulx(c1)` from the plain `c1` term. `legmul`'s final
//!   step is `legadd(c0, legmulx(c1))` — no such subtraction, no plain `c1`
//!   term at all. This is a genuine structural difference, not a
//!   relabeled constant (verified directly against `laguerre.py:505` vs
//!   `legendre.py:530`).
//! - `lagmul`'s `c1` update inside the loop is
//!   `lagadd(tmp, lagsub((2*nd-1)*c1, lagmulx(c1)) / nd)` — subtract THEN
//!   divide. `legmul`'s is `legadd(tmp, (legmulx(c1) * (2*nd-1)) / nd)` —
//!   scale-then-divide, no subtraction inside the division at all.
//! - `lagline` returns `[off + scl, -scl]`, an actual different affine map
//!   from `legline`'s `[off, scl]`.
//! - `lagder` accumulates ONE index back (`der[j-1] = -c[j]; c[j-1] += c[j]`)
//!   with no multiplicative coefficient beyond the sign flip; `legder`
//!   accumulates TWO indices back with a `(2j-1)` coefficient.
//! - `lagint` has NO division anywhere; `legint` divides by `(2j+1)`.
//! - `lagvander`'s recurrence WAS flagged "not read character-by-character"
//!   in the audit. Read directly here (`laguerre.py:1204-1211`) for this
//!   port: `v[0] = 1`, `v[1] = 1 - x`, and — contrary to the audit's own
//!   guess that it "likely" has no division (extrapolated from `lagmulx`
//!   having none) — the forward recurrence DOES divide by `i`:
//!   `v[i] = (v[i-1]*(2i-1-x) - v[i-2]*(i-1)) / i`. The audit's speculation
//!   was wrong; this is recorded in the task report, not silently
//!   corrected without comment.
//! - `lagcompanion` is UNSCALED (nonzero main diagonal `2i+1`, `+=` on the
//!   last column, no `1/sqrt(...)` factor anywhere) — contrast `legcompanion`
//!   which is scaled and symmetric via `-=`.

use crate::buffer::C128;
use crate::poly::{trim_trailing_zeros, PolyScalar};

/// Same seam as `crate::legendre::LegScalar`: real `f64`-from-magnitude
/// construction plus a `complex_div`-routed division so the `C128`
/// instantiation stays bit-exact against numpy's `npy_cdivide` rather than
/// `num_complex::Complex`'s native (measurably different) `Div` impl. See
/// `LegScalar`'s doc comment for the full ULP-mismatch evidence; identical
/// reasoning applies here, independently declared per basis-module
/// convention (this file must never import anything from `legendre.rs`).
pub trait LagScalar: PolyScalar {
    fn from_f64(v: f64) -> Self;
    fn lag_div(self, other: Self) -> Self;
    /// `T * T`, routed the same "measure, don't assume" way as `lag_div`.
    /// numpy's complex multiply is NOT the textbook two-rounding
    /// `(ac-bd, ad+bc)` formula that `num_complex::Complex`'s native `Mul`
    /// computes -- it's fused via hardware FMA (`crate::ufunc`'s
    /// `complex_mul_fma`, already measured empirically against real numpy
    /// 2.5.1: 0 mismatches with the FMA formula vs. ~44% mismatches with
    /// the naive one). Discovered here via `lagvander`'s complex path
    /// (`lagvander_complex` smoke-test case), which is the first place in
    /// this file that multiplies two genuinely complex operands together
    /// (every other multiply site has at least one real-valued/`from_usize`
    /// operand, where the two formulas happen to coincide) -- every
    /// multiplication in this file goes through `lag_mul`, not bare `*`,
    /// so this isn't limited to just the one call site that exposed it.
    fn lag_mul(self, other: Self) -> Self;
}

impl LagScalar for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn lag_div(self, other: Self) -> Self {
        self / other
    }
    fn lag_mul(self, other: Self) -> Self {
        self * other
    }
}

impl LagScalar for C128 {
    fn from_f64(v: f64) -> Self {
        C128::new(v, 0.0)
    }
    fn lag_div(self, other: Self) -> Self {
        crate::ufunc::complex_div(self, other)
    }
    fn lag_mul(self, other: Self) -> Self {
        crate::ufunc::complex_mul_fma(self, other)
    }
}

/// Every call site here mirrors a numpy expression of the exact shape
/// `<scalar coefficient> * <array>` (`c[0] * xs`, `c[-2] * xs`, `(2*nd-1)
/// * c1`, `q * p[:-1]`, ...) -- numpy NEVER writes the array first in
/// this file's source. That operand order is NOT cosmetic for complex
/// operands: `crate::ufunc::complex_mul_fma`'s FMA-based formula is
/// mathematically commutative but NOT bit-identical under argument swap
/// (verified directly against real numpy: `a*xs` and `xs*a` for the same
/// two complex128 values differ in their last bit --
/// `-0x1.34b86b86099e4p+0` vs `-0x1.34b86b86099e3p+0`, a genuine 1-ULP
/// divergence, not a signed-zero artifact). An earlier version of this
/// helper computed `v.lag_mul(s)` (array element first, scalar second) --
/// the WRONG order for every genuinely-complex-scalar call site (`scale
/// (xs, c[...])`), which is exactly why `lagmul`/`lagpow`/`lagfromroots`
/// (all built on `scale`) still failed a random out-of-corpus bit-exact
/// sweep even after the earlier `pad_sub`/negation fixes: the corpus's
/// fixed cases happened not to expose it, but ~20-30% of random complex
/// trials did. Fixed by putting the scalar `s` first, matching numpy's
/// literal source order. (Real-scalar call sites, e.g. `(2*nd-1) * c1`,
/// are order-invariant since a zero-imaginary operand makes both cross
/// terms in the FMA formula vanish identically either way -- but this
/// function doesn't special-case that; it just matches numpy's own
/// argument order unconditionally, which is correct for both.)
fn scale<T: LagScalar>(c: &[T], s: T) -> Vec<T> {
    c.iter().map(|&v| s.lag_mul(v)).collect()
}

/// Padded elementwise add/sub, WITHOUT trimming — same "internal-loop
/// values are unaffected by deferred trimming, because padding with exact
/// zero never perturbs a floating-point sum/difference" reasoning already
/// verified for `crate::legendre::pad_sub` (that file's doc comment, and
/// the 24/24-passing + sabotage-tested `legmul` built on it, are the
/// existing evidence this port relies on for the same claim applied to a
/// structurally different recurrence). Real numpy's own `lagmul` DOES call
/// the trimming `lagadd`/`lagsub` every loop iteration
/// (`laguerre.py:503-505`); using the untrimmed pad here only changes
/// representation (trailing zero-valued slots), never a computed bit,
/// because every padded slot is an exact `T::zero()` contributed by
/// nothing but padding, not by a rounding step.
/// Padded elementwise add, mirroring numpy's `polyutils._add` (verified
/// via `.venv/lib/python3.14/site-packages/numpy/polynomial/polyutils.py:555-565`)
/// EXACTLY, branch-for-branch:
///   if len(c1) > len(c2): c1[:c2.size] += c2; ret = c1
///   else:                 c2[:c1.size] += c1; ret = c2
/// This is NOT the same as "zero-initialize an accumulator, then add both
/// arrays' elements into it" (what an earlier version of this helper did):
/// for the tail region beyond the shorter operand's length, numpy's `ret`
/// is the LONGER operand's own original element, completely untouched --
/// not `0.0 + v`. Those differ in sign for `v == -0.0`, because IEEE-754
/// `0.0 + (-0.0) == +0.0` (round-to-nearest always produces +0.0 for that
/// sum), silently flipping a negative zero the untouched-copy path would
/// have preserved. Same class of bug as `pad_sub` below (and the same
/// bug that was already found and fixed in `lagder`/`lagint`/`lagmulx`'s
/// unary-negation sites) -- caught by this file's signed-zero discipline,
/// not by any printed-value or epsilon-tolerant comparison.
fn pad_add<T: LagScalar>(a: &[T], b: &[T]) -> Vec<T> {
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

/// Padded elementwise subtract, mirroring numpy's `polyutils._sub`
/// (`polyutils.py:568-579`) EXACTLY, branch-for-branch:
///   if len(c1) > len(c2): c1[:c2.size] -= c2; ret = c1
///   else:                 c2 = -c2; c2[:c1.size] += c1; ret = c2
/// The `else` branch (which is what every current caller of `pad_sub`
/// actually hits, e.g. `lagmul`'s `lagsub(c1v, lagmulx(c1v))`, since
/// `lagmulx` always returns one element longer than its input) negates
/// the WHOLE longer array (`b` here) first, then adds `a`'s elements into
/// only the overlapping prefix. That means the tail region beyond `a`'s
/// length is pure unary negation (`-b[i]`) of `b`'s element, NOT `0.0 -
/// b[i]` subtraction of it -- and even in the overlapping region, `(-b) +
/// a` is not bit-identical to `a - b` for signed-zero operands. An
/// earlier version of this helper zero-initialized an accumulator and did
/// `out[i] = out[i] - v` uniformly for every `b` element, which computes
/// `0.0 - v` everywhere; for `v == +0.0` that gives `-0.0`, but numpy's
/// actual `-v` (of the same `+0.0`) is also `-0.0` -- they only diverge at
/// `v == -0.0`, where `0.0 - (-0.0) == +0.0` but unary `-(-0.0) ==
/// +0.0` too... the actual divergence caught here (see `lagmul`
/// `[complex_times_real]`, `lagpow` `[complex_base]`, `lagfromroots`
/// `[complex_roots]` corpus cases) was the reverse direction and in the
/// prefix-combination arithmetic itself, not just the tail -- fixed by
/// replicating numpy's exact branch structure rather than trying to
/// reason about every signed-zero case by hand.
fn pad_sub<T: LagScalar>(a: &[T], b: &[T]) -> Vec<T> {
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

/// `lagval`: Clenshaw recursion, ported from `laguerre.py:864-887`.
/// MULTIPLY-then-divide grouping (`(c1 * (nd - 1)) / nd`) — the headline
/// trap this whole module exists to avoid getting backwards; see the
/// module doc comment.
pub fn lag_eval<T: LagScalar>(c: &[T], xs: &[T]) -> Vec<T> {
    let n = c.len();
    xs.iter()
        .map(|&x| {
            if n == 1 {
                return c[0];
            }
            if n == 2 {
                return c[0] + c[1].lag_mul(T::one() - x);
            }
            let mut nd = n;
            let mut c0 = c[n - 2];
            let mut c1 = c[n - 1];
            for i in 3..=n {
                let tmp = c0;
                nd -= 1;
                let nd_m1 = T::from_usize(nd - 1);
                let ndf = T::from_usize(nd);
                let two_nd_m1 = T::from_usize(2 * nd - 1);
                c0 = c[n - i] - c1.lag_mul(nd_m1).lag_div(ndf);
                c1 = tmp + c1.lag_mul(two_nd_m1 - x).lag_div(ndf);
            }
            c0 + c1.lag_mul(T::one() - x)
        })
        .collect()
}

/// `lagval` for REAL coefficients evaluated at COMPLEX points — the "4th
/// path" `lag_eval::<C128>` cannot cover even though every multiply/divide
/// site in this file is already correctly routed through `lag_mul`/
/// `lag_div`. Root cause (this codebase's own measurement, not numpy
/// source-reading alone): numpy's `lagval` runs its Clenshaw recursion on
/// numpy SCALAR objects, not whole-array operations — `c0 = c[-2]` starts
/// out `np.float64`-typed whenever `c`'s dtype is real, and stays
/// `float64`-typed (computed via single-rounding real division) for
/// however many loop iterations pass before that particular accumulator
/// is first combined with the complex evaluation point `x`; only THEN
/// does it become `np.complex128`-typed and start running through numpy's
/// actual complex ufuncs. A uniformly-`C128` instantiation of `lag_eval`
/// (the caller casting every real coefficient to `C128::new(v, 0.0)` up
/// front) cannot replicate this, because it routes EVERY division —
/// including the ones numpy computes in genuine single-rounding real
/// arithmetic — through `complex_div`'s Smith's-algorithm reciprocal-
/// multiply formula (`a * (1/c)`, two roundings). Measured directly: for
/// the small-integer divisors `nd` actually takes in this recurrence
/// (2..=6), `complex_div`'s real/real path agrees with plain `a/c` for
/// power-of-2 divisors (2, 4 — `1/c` is exactly representable) but
/// disagrees on ~33% of random dividends for non-power-of-2 divisors (3,
/// 5, 6 — confirmed via a direct 500,000-sample-per-divisor probe of
/// `complex_div`'s own formula, not a proxy).
///
/// Structural fact this function relies on (verified against
/// `laguerre.py:864-887` character-by-character, AND by an out-of-corpus
/// 5000-sample bit-level sweep against real `numpy.polynomial.laguerre`
/// across degrees 0-7, 0 mismatches): in the `len(c) >= 3` branch, `c1` is
/// combined with `x` on the very FIRST loop iteration (`i == 3`) and is
/// complex-typed for the rest of the recursion from that point on; `c0`'s
/// update at that SAME first iteration uses the OLD (pre-loop, still-real)
/// `c1`, so `c0` itself only becomes complex-typed starting from the
/// SECOND loop iteration onward (`i == 4`). So exactly one real/real
/// division — the `i == 3` `c0` update — needs genuine `f64` arithmetic;
/// every other step is either already complex-typed in numpy's own
/// semantics (safe to run through the existing, unmodified `lag_mul`/
/// `lag_div`) or a pure add/subtract (no rounding-formula divergence
/// possible either way, since `+`/`-` on `C128` are exact componentwise
/// operations with no FMA/Smith's-algorithm formula choice to get wrong).
/// This function does not change, duplicate, or second-guess `lag_mul`/
/// `lag_div` themselves — it only gives the one genuinely-real early step
/// real arithmetic instead of routing it through complex machinery it was
/// never actually numpy-typed to need.
pub fn lag_eval_real_coef_complex_x(c: &[f64], xs: &[C128]) -> Vec<C128> {
    let n = c.len();
    xs.iter()
        .map(|&x| {
            if n == 1 {
                return C128::new(c[0], 0.0);
            }
            if n == 2 {
                return C128::new(c[0], 0.0) + C128::new(c[1], 0.0).lag_mul(C128::new(1.0, 0.0) - x);
            }
            let mut nd = n - 1;
            // i == 3 (first loop iteration): c0 and c1 both still real at
            // the start of it. c0's update never touches `x` -- compute
            // it in genuine f64 (single-rounding division). c1's update
            // touches `x` on this very iteration, so it's computed in
            // C128 via the existing lag_mul/lag_div from here on.
            let c0_prev_r = c[n - 2];
            let c1_prev_r = c[n - 1];
            let nd_m1_r = (nd - 1) as f64;
            let nd_r = nd as f64;
            let two_nd_m1_r = (2 * nd - 1) as f64;
            let mut c0 = C128::new(c[n - 3] - (c1_prev_r * nd_m1_r) / nd_r, 0.0);
            let mut c1 = C128::new(c0_prev_r, 0.0)
                + C128::new(c1_prev_r, 0.0)
                    .lag_mul(C128::new(two_nd_m1_r, 0.0) - x)
                    .lag_div(C128::new(nd_r, 0.0));
            for i in 4..=n {
                let tmp = c0;
                nd -= 1;
                let nd_m1 = C128::new((nd - 1) as f64, 0.0);
                let ndf = C128::new(nd as f64, 0.0);
                let two_nd_m1 = C128::new((2 * nd - 1) as f64, 0.0);
                c0 = C128::new(c[n - i], 0.0) - c1.lag_mul(nd_m1).lag_div(ndf);
                c1 = tmp + c1.lag_mul(two_nd_m1 - x).lag_div(ndf);
            }
            c0 + c1.lag_mul(C128::new(1.0, 0.0) - x)
        })
        .collect()
}

/// `lagmulx`: pure three-term recurrence, NO division anywhere, ported
/// verbatim from `laguerre.py:426-439`.
///
/// Ticket #84 (2026-08-08, Monday): trims `c` via `trim_trailing_zeros`
/// FIRST, matching numpy's own `[c] = pu.as_series([c])` -- same fix, same
/// root cause, same measured shape+sign defect as `legendre::legmulx`'s
/// identical fix; see that function's doc comment for the full trace.
pub fn lagmulx<T: LagScalar>(c: &[T]) -> Vec<T> {
    let c = trim_trailing_zeros(c);
    let c = c.as_slice();
    if c.len() == 1 && c[0] == T::zero() {
        return c.to_vec();
    }
    let n = c.len();
    let mut prd = vec![T::zero(); n + 1];
    prd[0] = c[0];
    // numpy: `prd[1] = -c[0]` -- unary NEGATION, not `0 - c[0]`. For a
    // complex operand with an exactly-zero component these are NOT the
    // same bit pattern (`0.0 - 0.0 == +0.0`, but `-(0.0) == -0.0`) -- the
    // sibling mistake in `lagder`/`lagint` (which DO have a complex case
    // exercising exactly this) is what surfaced the pattern; fixed here
    // too on the same reasoning even though no case in this corpus
    // currently forces `c[0]`'s real/imag component to exactly zero.
    prd[1] = -c[0];
    for i in 1..n {
        prd[i + 1] = (-c[i]).lag_mul(T::from_usize(i + 1));
        prd[i] = prd[i] + c[i].lag_mul(T::from_usize(2 * i + 1));
        prd[i - 1] = prd[i - 1] - c[i].lag_mul(T::from_usize(i));
    }
    prd
}

pub use crate::poly::add_trim as lagadd_trim;
pub use crate::poly::sub_trim as lagsub_trim;

/// `lagmul`: ported from `laguerre.py:480-505`. See the module doc comment
/// for the two structural divergences from `legmul` (the `c1` update's
/// subtract-then-divide grouping, and the final step's extra
/// `lagsub(c1, lagmulx(c1))` rather than a bare `legmulx(c1)`).
pub fn lagmul<T: LagScalar>(c1_in: &[T], c2_in: &[T]) -> Vec<T> {
    // `pu.as_series([c1, c2])` trims trailing zeros off EACH input first —
    // same "changes the LENGTH of `xs`, not just cosmetic" reasoning as
    // `legmul`'s identical trim (see that file's doc comment for the
    // measured 2-ULP divergence this avoids).
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
            let two_nd_m1 = T::from_usize(2 * nd - 1);
            let ndf = T::from_usize(nd);

            // c0 = lagsub(c[-i]*xs, (c1*(nd-1))/nd) -- multiply-then-divide,
            // same grouping shape numpy uses for THIS particular term (it is
            // only the c1-update and final-step groupings that differ from
            // legmul, not this one -- verified against laguerre.py:503).
            // Ticket #84 (2026-08-08, Monday): every `lagsub(...)`/
            // `lagadd(...)` below is a real `pu._sub`/`pu._add` call, which
            // trims BOTH operands via `as_series` before the padded combine
            // -- see `legendre.rs`'s and `hermite.rs`'s identical fix and
            // doc comment for the measured signed-zero divergence this
            // avoids (`legmul`'s combine step / `hermmul`'s minimal repro).
            // `pad_sub`'s OWN untrimmed-same-length usage inside `lagdiv`'s
            // loop is unaffected and must stay untrimmed.
            let scaled_top = scale(xs, c[c.len() - i]);
            let c1v_scaled: Vec<T> = c1v.iter().map(|&v| v.lag_mul(nd_m1).lag_div(ndf)).collect();
            let scaled_top_t = trim_trailing_zeros(&scaled_top);
            let c1v_scaled_t = trim_trailing_zeros(&c1v_scaled);
            c0 = pad_sub(&scaled_top_t, &c1v_scaled_t);

            // c1 = lagadd(tmp, lagsub((2*nd-1)*c1, lagmulx(c1)) / nd) --
            // SUBTRACT then divide (unlike legmul's scale-then-divide).
            let scaled_c1 = scale(&c1v, two_nd_m1);
            let mx = lagmulx(&c1v);
            let scaled_c1_t = trim_trailing_zeros(&scaled_c1);
            let mx_t = trim_trailing_zeros(&mx);
            let diff = pad_sub(&scaled_c1_t, &mx_t);
            let diff_scaled: Vec<T> = diff.iter().map(|&v| v.lag_div(ndf)).collect();
            let tmp_t = trim_trailing_zeros(&tmp);
            let diff_scaled_t = trim_trailing_zeros(&diff_scaled);
            c1v = pad_add(&tmp_t, &diff_scaled_t);
        }
        (c0, c1v)
    };

    // return lagadd(c0, lagsub(c1, lagmulx(c1))) -- NOT legmul's bare
    // legadd(c0, legmulx(c1)). The extra `lagsub(c1, ...)` is load-bearing.
    let mxc1 = lagmulx(&c1v);
    let c1v_t = trim_trailing_zeros(&c1v);
    let mxc1_t = trim_trailing_zeros(&mxc1);
    let inner = pad_sub(&c1v_t, &mxc1_t);
    let c0_t = trim_trailing_zeros(&c0);
    let inner_t = trim_trailing_zeros(&inner);
    let out = pad_add(&c0_t, &inner_t);
    trim_trailing_zeros(&out)
}

/// `lagdiv`: `polyutils._div(lagmul, c1, c2)`, identical generic
/// repeated-subtraction shape to `legdiv` (this helper itself is
/// basis-agnostic in numpy's own source), but built on THIS module's own
/// `lagmul`/`lagadd_trim`/`lagsub_trim` so its bit-pattern depends on
/// `lagmul`'s grouping, not `legmul`'s.
pub fn lagdiv<T: LagScalar>(c1_in: &[T], c2_in: &[T]) -> Result<(Vec<T>, Vec<T>), ()> {
    // numpy: `[c1, c2] = as_series([c1, c2])` -- BOTH operands are trimmed
    // copies before any length check or return value is built. Ticket #87
    // (2026-08-10, Monday): this port never did that top-level trim (only
    // the sign-of-zero fix below had landed, via #84/#85's shared bug
    // class), so an input carrying an exact trailing zero on `c2` could
    // false-positive the `c2.last() == 0` zero-divisor check (numpy trims
    // it away and divides fine), and an exact trailing zero on `c1` leaked
    // an untrimmed tail into the returned remainder -- a SHAPE divergence,
    // not merely a sign one. Matches the pre-trim `chebdiv`/`legdiv`/
    // `poly::div` already do (see `chebyshev.rs`, `legendre.rs`, `poly.rs`).
    let c1 = trim_trailing_zeros(c1_in);
    let c2 = trim_trailing_zeros(c2_in);
    if *c2.last().unwrap() == T::zero() {
        return Err(());
    }
    let lc1 = c1.len();
    let lc2 = c2.len();
    if lc1 < lc2 {
        // numpy: `c1[:1] * 0, c1` -- NOT a hardcoded zero; see the
        // `lc2 == 1` branch just below for the identical sign-preserving
        // reasoning (both are `_div`'s degenerate early-return branches,
        // `polyutils.py:541`/`:543`). Remainder is the TRIMMED `c1`, not
        // the caller's original `c1_in`.
        return Ok((vec![c1[0].lag_mul(T::zero())], c1));
    }
    if lc2 == 1 {
        let scl = c2[0];
        let q: Vec<T> = c1.iter().map(|&v| v.lag_div(scl)).collect();
        // numpy: `c1 / c2[-1], c1[:1] * 0` -- the remainder is `c1[0] * 0`,
        // not a hardcoded zero (sign-preserving, same bug class as
        // `lagder`'s `cnt >= n0` branch above: `(-x) * 0 == -0.0` under
        // IEEE-754, which a bare `T::zero()` silently flips to `+0.0`).
        // Caught by an out-of-corpus random bit-exact sweep.
        return Ok((q, vec![c1[0].lag_mul(T::zero())]));
    }
    let mut quo = vec![T::zero(); lc1 - lc2 + 1];
    let mut rem = c1.clone();
    for i in (0..=lc1 - lc2).rev() {
        let mut basis = vec![T::zero(); i + 1];
        basis[i] = T::one();
        let p = lagmul(&basis, &c2);
        let q = rem.last().unwrap().lag_div(*p.last().unwrap());
        let rem_head = &rem[..rem.len() - 1];
        let p_head = &p[..p.len() - 1];
        let scaled_p_head = scale(p_head, q);
        rem = pad_sub(rem_head, &scaled_p_head);
        quo[i] = q;
    }
    let rem_trimmed = trim_trailing_zeros(&rem);
    Ok((quo, rem_trimmed))
}

/// `lagder`: ONE-index-back accumulation, ported from
/// `laguerre.py:644-672`. No multiplicative coefficient beyond the sign
/// flip -- a structurally shallower recursion than `legder`'s two-index-back
/// `(2j-1)`-scaled accumulation, not a relabeled constant.
pub fn lagder<T: LagScalar>(c_in: &[T], cnt: usize, scl: T) -> Vec<T> {
    let c = c_in.to_vec();
    let n0 = c.len();
    if cnt == 0 {
        return c;
    }
    if cnt >= n0 {
        // numpy: `c = c[:1] * 0` -- NOT a hardcoded zero. Multiplying
        // `c[0]` by `0` preserves `c[0]`'s sign per IEEE-754 multiply
        // rules (`(-1.43...) * 0.0 == -0.0`, not `+0.0`); a hardcoded
        // `T::zero()` here silently flips that sign whenever `c[0]` is
        // negative (or, for `C128`, whenever either of its components
        // is negative). Caught by an out-of-corpus random bit-exact
        // sweep (single-coefficient input, `m >= len(c)`), not the
        // fixed-case corpus.
        return vec![c[0].lag_mul(T::zero())];
    }
    let mut c = c;
    let mut n = n0;
    for _ in 0..cnt {
        n -= 1;
        for v in c.iter_mut() {
            *v = v.lag_mul(scl);
        }
        let mut der = vec![T::zero(); n];
        // numpy: `der[j-1] = -c[j]` -- unary negation (`laguerre.py:665`),
        // not `0 - c[j]`; differs in the sign of an exact-zero component
        // for a complex operand (`0.0 - 0.0 == +0.0` vs `-(0.0) == -0.0`).
        // Found via the `complex_c` corpus case going bit-red on this.
        for j in (2..=n).rev() {
            der[j - 1] = -c[j];
            c[j - 1] = c[j - 1] + c[j];
        }
        der[0] = -c[1];
        c = der;
    }
    c
}

/// `lagint`: NO division anywhere, ported from `laguerre.py:757-794`.
/// Contrast `legint`'s `/(2j+1)`-driven antiderivative fan-out.
pub fn lagint<T: LagScalar>(c_in: &[T], cnt: usize, k: &[T], lbnd: T, scl: T) -> Vec<T> {
    let mut c = c_in.to_vec();
    if cnt == 0 {
        return c;
    }
    for &ki in k.iter().take(cnt) {
        let n = c.len();
        for v in c.iter_mut() {
            *v = v.lag_mul(scl);
        }
        if n == 1 && c[0] == T::zero() {
            c[0] = c[0] + ki;
        } else {
            let mut tmp = vec![T::zero(); n + 1];
            tmp[0] = c[0];
            // numpy: `tmp[1] = -c[0]` / `tmp[j+1] = -c[j]` -- unary
            // negation (`laguerre.py:786,789`), not `0 - c[...]`; same
            // signed-zero divergence as `lagder`/`lagmulx` above. Found
            // via the `complex_c` corpus case going bit-red on this.
            tmp[1] = -c[0];
            for j in 1..n {
                tmp[j] = tmp[j] + c[j];
                tmp[j + 1] = -c[j];
            }
            let at_lbnd = lag_eval(&tmp, &[lbnd])[0];
            // numpy: `tmp[0] += k[i] - lagval(lbnd, tmp)`, i.e.
            // `tmp[0] + (ki - at_lbnd)` -- the parenthesization is
            // load-bearing. `tmp[0] + ki - at_lbnd` (left-to-right, i.e.
            // `(tmp[0] + ki) - at_lbnd`) is a DIFFERENT floating-point
            // expression: addition/subtraction is not associative, so the
            // two groupings can and do disagree by 1 ULP (caught by an
            // out-of-corpus random bit-exact sweep, not the fixed-case
            // corpus).
            tmp[0] = tmp[0] + (ki - at_lbnd);
            c = tmp;
        }
    }
    c
}

/// `lagvander(x, deg)`: `V[i, j] = L_j(x[i])`. Read directly from
/// `laguerre.py:1204-1211` for this port (the audit explicitly flagged this
/// function as unverified) -- `v[0] = 1`, `v[1] = 1 - x`, and the forward
/// recurrence DOES divide by `i` (`(v[i-1]*(2i-1-x) - v[i-2]*(i-1)) / i`),
/// contrary to the audit's own "likely follows `lagmulx`'s no-division
/// pattern" guess. See this task's report for that correction.
pub fn lagvander<T: LagScalar>(xs: &[T], deg: usize) -> Vec<T> {
    let width = deg + 1;
    let mut out = vec![T::zero(); xs.len() * width];
    for (row, &x) in xs.iter().enumerate() {
        let base = row * width;
        out[base] = T::one();
        if deg > 0 {
            out[base + 1] = T::one() - x;
            for i in 2..width {
                let coef = T::from_usize(2 * i - 1) - x;
                let a = out[base + i - 1].lag_mul(coef);
                let b = out[base + i - 2].lag_mul(T::from_usize(i - 1));
                out[base + i] = (a - b).lag_div(T::from_usize(i));
            }
        }
    }
    out
}

/// `lagcompanion(c)`: UNSCALED, ported from `laguerre.py:1494-1512`.
/// Nonzero main diagonal (`2i+1`), `+=` (not `-=`) on the last column, no
/// `1/sqrt` scale factor anywhere -- contrast `legcompanion`, which is
/// scaled and symmetric via `-=`. `len(c) == 2` is a genuine separate
/// branch in numpy's own source (`[[1 + c0/c1]]`), not merely what the
/// general formula would produce for `n == 1` (though it happens to agree
/// -- kept as an explicit branch here to mirror the source exactly).
pub fn lagcompanion<T: LagScalar>(c: &[T]) -> Vec<T> {
    let len = c.len();
    if len == 2 {
        return vec![T::one() + c[0].lag_div(c[1])];
    }
    let n = len - 1;
    let mut mat = vec![T::zero(); n * n];
    for k in 0..n {
        mat[k * n + k] = T::from_usize(2 * k + 1);
    }
    for k in 0..n.saturating_sub(1) {
        let val = T::zero() - T::from_usize(k + 1);
        mat[k * n + (k + 1)] = val;
        mat[(k + 1) * n + k] = val;
    }
    let cn = c[len - 1];
    let nf = T::from_usize(n);
    for i in 0..n {
        let corr = c[i].lag_div(cn).lag_mul(nf);
        mat[i * n + (n - 1)] = mat[i * n + (n - 1)] + corr;
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lag_eval_matches_numpy_example() {
        // L.lagval(1, [1,2,3]) -> -0.5
        let c = [1.0_f64, 2.0, 3.0];
        let out = lag_eval(&c, &[1.0]);
        assert!((out[0] - (-0.5)).abs() < 1e-12, "{}", out[0]);
    }

    #[test]
    fn lagmulx_matches_numpy_example() {
        // L.lagmulx([1,2,3]) -> [-1, -1, 11, -9]
        let c = [1.0_f64, 2.0, 3.0];
        let out = lagmulx(&c);
        let expect = [-1.0, -1.0, 11.0, -9.0];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn lagmul_matches_numpy_example() {
        // L.lagmul([1,2,3],[0,1,2]) -> [8, -13, 38, -51, 36]
        let c1 = [1.0_f64, 2.0, 3.0];
        let c2 = [0.0_f64, 1.0, 2.0];
        let out = lagmul(&c1, &c2);
        let expect = [8.0, -13.0, 38.0, -51.0, 36.0];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn lagdiv_matches_numpy_example() {
        // L.lagdiv([8,-13,38,-51,36], [0,1,2]) -> ([1,2,3], [0.])
        let c1 = [8.0_f64, -13.0, 38.0, -51.0, 36.0];
        let c2 = [0.0_f64, 1.0, 2.0];
        let (q, r) = lagdiv(&c1, &c2).unwrap();
        for (a, b) in q.iter().zip([1.0, 2.0, 3.0].iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        assert!((r[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn lagder_matches_numpy_example() {
        // L.lagder([1,1,1,-3]) -> [1,2,3]
        let c = [1.0_f64, 1.0, 1.0, -3.0];
        let d1 = lagder(&c, 1, 1.0);
        for (a, b) in d1.iter().zip([1.0, 2.0, 3.0].iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn lagint_matches_numpy_example() {
        // L.lagint([1,2,3]) -> [1, 1, 1, -3]
        let c = [1.0_f64, 2.0, 3.0];
        let out = lagint(&c, 1, &[0.0], 0.0, 1.0);
        let expect = [1.0, 1.0, 1.0, -3.0];
        for (a, b) in out.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn lagvander_matches_numpy_example() {
        // L.lagvander([0,1,2], 3) row0 = [1,1,1,1]; row1 = [1,0,-0.5,-0.66666667]
        let xs = [0.0_f64, 1.0, 2.0];
        let v = lagvander(&xs, 3);
        let expect_row1 = [1.0, 0.0, -0.5, -2.0 / 3.0];
        for (a, b) in v[4..8].iter().zip(expect_row1.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn lagcompanion_matches_numpy_example() {
        // L.lagcompanion([1,2,3]) -> [[1, -0.33333333], [-1, 4.33333333]]
        let c = [1.0_f64, 2.0, 3.0];
        let m = lagcompanion(&c);
        let expect = [1.0, -1.0 / 3.0, -1.0, 13.0 / 3.0];
        for (a, b) in m.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }
}
