"""anionpy.polynomial.hermite.* differential registry entries.

Same pattern as laguerre_cases.py/legendre_cases.py: builds a
`HERMITE_SPECS: dict[str, ItemSpec]` dict, merged into `registry.REGISTRY`
at the bottom of registry.py (collision-checked, same pattern as every
other `*_cases.py` merge).

Covers the same ~24 non-N-D items as laguerre_cases.py's scope
(`hermval2d/3d/nd`, `hermgrid2d/3d`, `hermvander2d/3d`, and the
`Hermite(ABCPolyBase)` class are NOT covered, not attempted in this pass).

This corpus is written BEFORE any Rust implementation exists (mandated
order), read directly against `numpy/polynomial/hermite.py` (physicists'
Hermite -- NOT `hermite_e.py`, the probabilists' variant, which differs in
several of the exact spots pinned below). Divergences from Laguerre/
Legendre that a naive copy-paste of either corpus would miss, each pinned
explicitly:

  - `herm2poly`'s `len(c) == 2` branch does `c[1] *= 2` -- a genuine WRONG
    VALUE bug if missed (not a rounding difference), and it does NOT exist
    in `herme2poly` (`hermite_e.py`'s sibling function). Pinned as
    `len2_pinned_doubling` below, not left to random degree sampling.
  - `hermval`'s Clenshaw grouping is `c0 = c[-i] - c1*(2*(nd-1))`,
    `c1 = tmp + c1*x2` with `x2 = x*2` PRECOMPUTED ONCE and reused
    (including in the `len(c) <= 2` branches' shared `return c0 + c1*x2`
    tail) -- verified directly against `hermite.py:872-905`.
  - `hermmul`'s trailing term is `hermadd(c0, hermmulx(c1) * 2)` -- the
    `* 2` is OUTSIDE `hermmulx`, applied to its whole result, not folded
    into `hermmulx` itself or applied before the call.
  - `hermmulx` has NO division anywhere (pure recurrence,
    `prd[i+1] = c[i]/2`, `prd[i-1] += c[i]*i`) -- contrast `hermint`'s
    calculus recurrence, which DOES divide (`tmp[j+1] = c[j]/(2*(j+1))`).
  - `hermder` does DIRECT ASSIGNMENT (`der[j-1] = (2*j)*c[j]`), NOT an
    accumulation (`+=`) -- unlike Laguerre's/Legendre's `*der`, which both
    accumulate into the target index. Pinned via a `four_term` case with
    every coefficient nonzero, where an accidental `+=` vs `=` bug would
    show up as a wrong (roughly doubled) value, not merely a rounding
    difference.
  - `hermint` also does direct assignment inside its `j` loop
    (`tmp[j+1] = c[j]/(2*(j+1))`), no accumulation -- same contrast as
    `hermder` above.
  - `hermcompanion`/`hermroots`/`hermmul`/`hermval` each have a dedicated
    `len(c) == 2` branch in numpy's own source -- pinned explicitly below
    as `len2_pinned*` cases (mirroring laguerre_cases.py's
    `degree1_pinned_len2branch` naming), not left to random sampling.
  - `hermgauss` DOES symmetrize both `x` and `w`
    (`w = (w + w[::-1]) / 2`, `x = (x - x[::-1]) / 2`) -- UNLIKE
    `laggauss`, which does neither -- and scales weights by
    `sqrt(pi) / w.sum()` (same scale constant as Laguerre's, but Hermite
    additionally symmetrizes first). Verified directly against
    `hermite.py`'s `hermgauss` body.
  - `hermweight(x) = exp(-x**2)`, domain `(-inf, inf)` -- unlike
    `lagweight`'s `exp(-x)` on `[0, inf)`. Verified live:
    `NH.hermweight(3.0)` returns `1.2340980408667956e-04`.
  - `hermline(off, scl)` returns `[off, scl/2]` (matching Legendre's
    `[off, scl]` SHAPE but with the `/2` scale factor Legendre lacks) --
    NOT Laguerre's genuinely different affine map `[off+scl, -scl]`.
"""
from __future__ import annotations

import numpy as np

import anionpy as _anionpy  # noqa: F401  (parity import with laguerre_cases.py; not directly used)

from registry import ItemSpec

# ---------------------------------------------------------------------------
# shared coefficient/point fixtures
# ---------------------------------------------------------------------------

INT_C = [1, 2, 3]
FLOAT_C = [1.0, 2.0, 3.0]
TRAILING_ZERO_C = [1.0, 2.0, 3.0, 0.0, 0.0]
ALL_ZERO_C = [0.0, 0.0, 0.0]
COMPLEX_C = [1 + 1j, 2 - 1j, 0.5j]
BOOL_C = [True, False, True]
SINGLE_C = [5.0]

C1 = [1.0, 2.0, 3.0]
C2 = [3.0, 2.0, 1.0]
C1_SHORT = [3.0, 2.0]
C_FOUR = [1.0, 2.0, 3.0, 4.0]

XS = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.5]
X_SCALAR = 2.0
X_COMPLEX = 1 + 2j

# Hermite's natural domain is (-inf, inf) -- no domain-restricted fixture
# analogous to Laguerre's XS_NONNEG is needed; XS already straddles zero.


def _std(pairs):
    return [(label, args, kwargs) for label, args, kwargs in pairs]


def _const_cases(attr):
    def gen():
        return [(f"{attr}_value", (), {})]
    return gen


# ---------------------------------------------------------------------------
# hermline / hermtrim
# ---------------------------------------------------------------------------

def hermline_cases():
    return _std([
        ("basic", (1.0, 2.0), {}),
        ("zero_scale", (3.0, 0.0), {}),
        ("negative", (-1.5, -2.5), {}),
        ("off_zero", (0.0, 4.0), {}),
        ("both_zero", (0.0, 0.0), {}),
    ])


def hermtrim_cases():
    return _std([
        ("trailing_zeros", (TRAILING_ZERO_C,), {}),
        ("all_zero", (ALL_ZERO_C,), {}),
        ("no_trim_needed", (FLOAT_C,), {}),
        ("with_tol", (list(TRAILING_ZERO_C) + [1e-10], 1e-8), {}),
        ("complex", (COMPLEX_C + [0j],), {}),
        ("int_input", (INT_C + [0],), {}),
        ("bool_input", (BOOL_C,), {}),  # as_series-family: rejects bool
    ])


# ---------------------------------------------------------------------------
# hermval
# ---------------------------------------------------------------------------

def hermval_cases():
    cases = []
    for c_label, c in [("int_c", INT_C), ("float_c", FLOAT_C), ("bool_c", BOOL_C),
                        ("complex_c", COMPLEX_C), ("trailing_zero_c", TRAILING_ZERO_C),
                        ("single_c", SINGLE_C)]:
        cases.append((f"scalar_x__{c_label}", (X_SCALAR, c), {}))
        cases.append((f"list_x__{c_label}", (XS, c), {}))
        cases.append((f"complex_x__{c_label}", (X_COMPLEX, c), {}))
    cases.append(("array_x_2d", (np.array([[0.0, 1.0], [2.0, 3.0]]), FLOAT_C), {}))
    # len(c) == 2 is a dedicated short-circuit branch (`c0=c[0]; c1=c[1]`)
    # sharing the module-level precomputed `x2 = x*2` with the general
    # Clenshaw loop -- pinned explicitly, mirroring lagval_cases'
    # len2_c_pinned.
    cases.append(("len2_c_pinned", (XS, C1_SHORT), {}))
    cases.append(("len1_c_pinned", (XS, SINGLE_C), {}))
    return cases


# ---------------------------------------------------------------------------
# add / sub / mulx / mul / div / pow
# ---------------------------------------------------------------------------

def hermadd_cases():
    return _std([
        ("same_len", (C1, C2), {}),
        ("c1_shorter", (C1_SHORT, C1), {}),
        ("c2_shorter", (C1, C1_SHORT), {}),
        ("int_plus_float", (INT_C, FLOAT_C), {}),
        ("bool_plus_int", (BOOL_C, INT_C), {}),
        ("complex_plus_real", (COMPLEX_C, FLOAT_C), {}),
        ("cancels_to_zero", ([1.0, 2.0], [-1.0, -2.0]), {}),
        ("trailing_zero_result", ([1.0, 2.0, 3.0], [0.0, 0.0, -3.0]), {}),
    ])


def hermsub_cases():
    return _std([
        ("same_len", (C1, C2), {}),
        ("c1_shorter", (C1_SHORT, C1), {}),
        ("identical_cancels", (FLOAT_C, FLOAT_C), {}),
        ("complex_minus_real", (COMPLEX_C, FLOAT_C), {}),
        ("int_minus_bool", (INT_C, BOOL_C), {}),
    ])


def hermmulx_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("int_c", (INT_C,), {}),
        ("zero_series", (ALL_ZERO_C[:1],), {}),  # numpy special-cases [0]
        ("complex_c", (COMPLEX_C,), {}),
        ("single_nonzero", (SINGLE_C,), {}),
        ("bool_c", (BOOL_C,), {}),  # as_series-family: rejects bool
        # ---- Added 2026-08-08 (Monday), ticket #84 ----
        # Same defect class and fix as legmulx's analogous cases in
        # legendre_cases.py (missing `pu.as_series` pre-trim before the
        # `len(c)==1 && c[0]==0` zero-series fast-path check, lost
        # sign on negative-first-coefficient input). Fixed in
        # `ionp-core/src/hermite.rs`. Permanent regression guard.
        ("negative_both_first_coef", ([-4.5 - 2.5j],), {}),
    ])


def hermmul_cases():
    return _std([
        ("basic", (C1, C2), {}),
        ("int_times_float", (INT_C, FLOAT_C), {}),
        ("complex_times_real", (COMPLEX_C, FLOAT_C), {}),
        ("with_zero_series", (FLOAT_C, ALL_ZERO_C[:1]), {}),
        ("trailing_zero_operand", (TRAILING_ZERO_C, C2), {}),
        ("single_by_single", ([2.0], [3.0]), {}),
        ("bool_operand", (BOOL_C, FLOAT_C), {}),  # as_series-family: rejects bool
        # len(c) == 1 and len(c) == 2 dedicated branches, plus the 4+-term
        # general loop that exercises `hermsub`/`hermadd(..., hermmulx(c1)*2)`.
        ("len1_operand", (SINGLE_C, C2), {}),
        ("len2_operand", (C1_SHORT, C2), {}),
        ("four_term", (C_FOUR, [1.0, -1.0, 2.0, -2.0]), {}),
        # ---- Added 2026-08-08 (Monday), ticket #84 ----
        # `hermfromroots` (one of the ticket's three originally-named
        # items) pairwise-reduces via `hermmul`, so this repro also
        # regression-guards `hermfromroots`. Root cause: `hermmul`'s
        # internal recursion calls `pad_add`/`pad_sub` at the same call
        # sites where real numpy calls `hermadd`/`hermsub` (i.e.
        # `pu._add`/`pu._sub`), which trim BOTH operands via
        # `as_series` before the padded elementwise combine --
        # `pad_add`/`pad_sub` never did this trim. Repro:
        # hermmul([5e-324-0j, 0.5+0j], [-5e-324-0j, 0.5+0j]) -- numpy
        # gives [0.5+0j, -0.+0j, 0.25+0j], pre-fix anionpy gave
        # [0.5+0j, 0.+0j, 0.25+0j] (index 1 real part sign lost). Fixed
        # in `ionp-core/src/hermite.rs` by trimming both operands via
        # `trim_trailing_zeros` immediately before every `pad_add`/
        # `pad_sub` call site. Permanent regression guard.
        ("underflow_signed_zero_ticket84",
         ([5e-324 - 0j, 0.5 + 0j], [-5e-324 - 0j, 0.5 + 0j]), {}),
    ])


def hermdiv_cases():
    return _std([
        ("basic", (C1, C2), {}),
        ("swap", (C2, C1), {}),
        ("c1_shorter_than_c2", (C1_SHORT, C1), {}),
        ("divide_by_scalar", (C1, [2.0]), {}),
        ("exact_division", ([2.0, 4.0, 6.0], [1.0, 2.0]), {}),
        ("complex", (COMPLEX_C, [1.0, 1.0]), {}),
        ("divide_by_zero_series", (C1, [0.0]), {}),
        ("bool_c1", (BOOL_C, [1.0, 2.0]), {}),  # as_series-family: rejects bool
        # ---- Ticket #87 (2026-08-10, Monday): `hermdiv` never did numpy
        # `_div`'s top-level `[c1, c2] = as_series([c1, c2])` pre-trim, so an
        # untrimmed trailing exact zero in `c1` leaked through into the
        # returned quotient/remainder as a SHAPE divergence. Same repro
        # shape as `laguerre.lagdiv`'s ticket #87 case (this generic
        # `_div`-family defect is basis-agnostic in numpy's own source).
        ("untrimmed_trailing_zero_c1_ticket87",
         ([-0.32019105, 1.45823375, -0.86985882, 0.0, 0.0], [1.67652212]), {}),
        # A SECOND case exercising the missing pre-trim on `c2`: a trailing
        # exact zero on the DIVISOR made anionpy's untrimmed
        # `c2.last() == 0` check false-positive a ZeroDivisionError numpy
        # never raises.
        ("untrimmed_trailing_zero_c2_ticket87",
         ([0.42691841, 0.212777, -0.02226693, 0.61897555, 0.0, 0.0],
          [-0.07402421, 0.23126075, 0.0]), {}),
    ])


def hermpow_cases():
    return _std([
        ("square", (C1, 2), {}),
        ("cube", (FLOAT_C, 3), {}),
        ("power_zero", (C1, 0), {}),
        ("power_zero_complex", (COMPLEX_C, 0), {}),
        ("power_one", (C1, 1), {}),
        ("complex_base", (COMPLEX_C, 2), {}),
        ("int_c", (INT_C, 2), {}),
    ])


# ---------------------------------------------------------------------------
# der / int
# ---------------------------------------------------------------------------

def hermder_cases():
    C = C_FOUR
    return _std([
        ("first_order", (C, 1), {}),
        ("third_order", (C, 3), {}),
        ("zero_order", (C, 0), {}),
        ("order_exceeds_degree", (C, 10), {}),
        ("with_scale", (C, 1, 2.0), {}),
        ("complex_c", (COMPLEX_C, 1), {}),
        ("int_c", (INT_C, 1), {}),
        ("bool_c", (BOOL_C, 1), {}),  # own-dtype path: bool promoted, not rejected
        # every coefficient nonzero -- an accidental accumulation (`+=`)
        # bug where numpy does a direct assignment would show up here as
        # a wrong (not merely rounded) value.
        ("all_nonzero_direct_assign_probe", ([1.0, 1.0, 1.0, 1.0, 1.0], 1), {}),
    ])


def hermint_cases():
    C = [1.0, 2.0, 3.0]
    return _std([
        ("first_order_default_k", (C, 1), {}),
        ("with_k", (C, 1, [1.0]), {}),
        ("second_order_with_k", (C, 2, [1.0, 2.0]), {}),
        ("with_lbnd", (C, 1, [0.0], 1.0), {}),
        ("with_scale", (C, 1, [0.0], 0.0, 2.0), {}),
        ("complex_c", (COMPLEX_C, 1, [0.0]), {}),
        ("int_c", (INT_C, 1, [0.0]), {}),
        ("bool_c", (BOOL_C, 1, [0.0]), {}),  # own-dtype path: bool promoted, not rejected
        ("all_nonzero_direct_assign_probe", ([1.0, 1.0, 1.0, 1.0], 1, [0.0]), {}),
    ])


# ---------------------------------------------------------------------------
# vander / companion
# ---------------------------------------------------------------------------

def hermvander_cases():
    return _std([
        ("basic", (XS, 5), {}),
        ("deg_zero", (XS, 0), {}),
        ("deg_one_pinned", (XS, 1), {}),  # v[1] = x*2, not x -- pinned explicitly
        ("scalar_x", (X_SCALAR, 3), {}),
        ("complex_x", ([1 + 1j, 2 - 1j, 0.5j], 4), {}),
        ("int_x", ([1, 2, 3], 3), {}),
        ("bool_x", ([True, False], 2), {}),  # own-dtype path: bool promoted, not rejected
    ])


def hermcompanion_cases():
    return _std([
        ("degree2", (C1,), {}),
        # len(c) == 2 special case: [[-.5 * c0/c1]] -- pinned explicitly.
        ("degree1_pinned_len2branch", (C1_SHORT,), {}),
        ("degree4", ([1.0, 0.0, -2.0, 0.0, 1.0],), {}),
        ("complex", (COMPLEX_C,), {}),
        ("int_c", (INT_C,), {}),
        ("bool_c", (BOOL_C,), {}),  # as_series-family: rejects bool
    ])


# ---------------------------------------------------------------------------
# fromroots / roots
# ---------------------------------------------------------------------------

def hermfromroots_cases():
    return _std([
        ("real_roots", ([-1.0, 0.0, 1.0],), {}),
        ("no_roots", ([],), {}),
        ("single_root", ([3.0],), {}),
        ("complex_roots", ([-1j, 0.0, 1j],), {}),
        ("repeated_root", ([1.0, 1.0, -1.0],), {}),
        ("five_roots", ([-2.0, -1.0, 0.0, 1.0, 2.0],), {}),
        ("bool_roots", ([True, False],), {}),  # as_series-family: rejects bool
    ])


def hermroots_cases():
    return _std([
        ("real_roots", (np.polynomial.hermite.hermfromroots([-1.0, 0.0, 1.0]),), {}),
        ("complex_roots", (np.polynomial.hermite.hermfromroots([-1j, 0.0, 1j]),), {}),
        # len(c) == 2 special case: -.5 * c0/c1 -- pinned explicitly, this
        # is the "degree exactly 2 [coefficient array]" pin the task brief
        # requires, mirroring lagroots_cases' degree1_pinned_len2branch.
        ("degree1_pinned_len2branch", (C1_SHORT,), {}),
        ("degree0_no_roots", (SINGLE_C,), {}),
        ("five_real_roots", (np.polynomial.hermite.hermfromroots(
            [-2.0, -1.0, 0.0, 1.0, 2.0]),), {}),
    ])


def _canonical_root_order(arr):
    """See laguerre_cases.py's `_canonical_root_order` -- identical
    stabilization strategy (round the sort KEY, not the compared values,
    to 6 decimals, three orders of magnitude coarser than
    HERMROOTS_COMPLEX_EPS below). `hermroots`'s output dtype also depends
    on `_to_real_if_imag_zero`-equivalent downcasting (see
    anionpy/polynomial/hermite.py's `hermroots`), so `arr` may be real or
    complex depending on the case; `.real`/`.imag` both work on a real
    float64 array (imag is all zero) so this helper is dtype-agnostic."""
    arr = np.asarray(arr)
    key = list(zip(np.round(arr.real, 6), np.round(np.imag(arr), 6)))
    order = sorted(range(len(arr)), key=lambda i: key[i])
    return arr[order]


def _hermroots_numpy_sorted(c):
    return _canonical_root_order(np.polynomial.hermite.hermroots(c))


def _hermroots_ionp_sorted(c):
    import anionpy as ap
    r = ap.polynomial.hermite.hermroots(ap.array(c))
    r_np = np.asarray(r.tolist() if hasattr(r, "tolist") else r)
    return _canonical_root_order(r_np)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def hermfit_cases():
    rng = np.random.default_rng(20260807)
    x = np.array(list(XS) + [3.0, 4.0, -3.0, -4.0, 0.25, -0.25, 1.5, -1.5,
                              2.25, -2.25, 0.75, -0.75, 1.75, -1.75, 0.1,
                              -0.1, 0.3, -0.3])
    y = np.polynomial.hermite.hermval(x, [0.0, 1.0, 0.0, -1.0])
    y_noisy = y + rng.normal(scale=0.01, size=x.shape)
    y2d = np.stack([y, x**2], axis=1)
    return _std([
        ("cubic_exact", (x, y, 3), {}),
        ("cubic_noisy", (x, y_noisy, 3), {}),
        ("linear_underfit", (x, y, 1), {}),
        ("y_2d", (x, y2d, 3), {}),
        ("small_n", ([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0], 2), {}),
    ])


# ---------------------------------------------------------------------------
# hermgauss / hermweight
# ---------------------------------------------------------------------------

def hermgauss_cases():
    return _std([
        ("deg1", (1,), {}),
        ("deg2", (2,), {}),
        ("deg5", (5,), {}),
        ("deg10", (10,), {}),
        ("deg25", (25,), {}),
    ])


def hermweight_cases():
    return _std([
        ("scalar", (0.5,), {}),
        ("list", (XS,), {}),
        ("array", (np.array(XS),), {}),
        ("negative", ([-3.0, -1.0, -0.1],), {}),
    ])


# ---------------------------------------------------------------------------
# herm2poly / poly2herm
# ---------------------------------------------------------------------------

def herm2poly_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("int_c", (INT_C,), {}),
        ("complex_c", (COMPLEX_C,), {}),
        ("trailing_zero_c", (TRAILING_ZERO_C,), {}),
        ("single_c", (SINGLE_C,), {}),
        # n == 2 branch does `c[1] *= 2` -- a genuine WRONG VALUE bug if
        # missed, absent from herme2poly's sibling function. Pinned
        # explicitly, NOT left to random degree sampling, per the task
        # brief's specific warning about this exact function.
        ("len2_pinned_doubling", (C1_SHORT,), {}),
        ("bool_c", (BOOL_C,), {}),  # as_series-family: rejects bool
    ])


def poly2herm_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("int_c", (INT_C,), {}),
        ("complex_c", (COMPLEX_C,), {}),
        ("trailing_zero_c", (TRAILING_ZERO_C,), {}),
        ("single_c", (SINGLE_C,), {}),
        ("two_term", (C1_SHORT,), {}),
        ("bool_c", (BOOL_C,), {}),  # as_series-family: rejects bool
    ])


# Hermite's OWN independently measured seeded-sweep constants, NOT
# inherited from LAGFIT_EPS/LAGROOTS_COMPLEX_EPS/LAGGAUSS_EPS in
# laguerre_cases.py or LEGFIT_EPS/... in legendre_cases.py. Measured by
# /private/tmp/herm_eps_sweep.py (hermroots complex/real, hermgauss) and
# /private/tmp/herm_eps_sweep2.py (hermfit's "rel" metric, floor=1.0,
# matching harness.max_rel_distance exactly), each an independent
# seed=9182736, N=20000 sweep against the real Rust implementation (run
# AFTER ionp-core/src/hermite.rs existed, not guessed in advance):
#   hermroots, complex128-coefficient input -> max abs root distance
#     3.3145124091591355e-13
#   hermroots, float64-coefficient input -> max abs root distance
#     9.094947017729282e-13
#   hermfit -> max rel (floor=1.0) coefficient distance
#     2.8248462118007495e-12
#   hermgauss -> max abs distance across BOTH outputs (x, w)
#     8.881784197001252e-16 (x) / 1.1102230246251565e-16 (w); the larger
#     of the two, 8.881784197001252e-16, is declared since epsilon_tolerance
#     applies one bound per dtype across a multi_output item.
HERMFIT_EPS = 2.8248462118007495e-12
HERMROOTS_COMPLEX_EPS = 3.3145124091591355e-13
HERMROOTS_REAL_EPS = 9.094947017729282e-13
HERMGAUSS_EPS = 8.881784197001252e-16


# ---------------------------------------------------------------------------
# Task #34 (2026-08-08): hermval2d/hermval3d/hermvalnd/hermgrid2d/
# hermgrid3d/hermvander2d/hermvander3d -- built on polyutils.py's shared
# _valnd/_gridnd/_vander_nd/_vander_nd_flat machinery (see that module's
# docstring for the ticket-premise correction: FOUR shared helpers back
# this family, not the two -- _vander_nd/_vander_nd_flat -- the ticket
# named). Corpus varies point count (1-4 per axis), degree (0-2 per axis),
# and dtype (float64, complex128, int -- promoted via _common_dtype).
# float16/float32/complex64 are DELIBERATELY EXCLUDED: the already-
# declared, already-shipped 1-D `hermval`/`hermvander` kernels this family
# is built on universally upcast those three dtypes to float64/complex128
# (values match real numpy bit-exact, but dtype does not) -- a
# pre-existing divergence in code this ticket did not write and has no
# permission to touch (lives in the shared Rust kernels, confirmed present
# for all six bases, not introduced here). See docs/TICKET-34-*.md.

def _herm_nd_base_cases():
    rng = np.random.default_rng(20260808)

    def mk(n, dx, dy, dz, dtype_label):
        if dtype_label == "int":
            xs = rng.integers(-4, 4, n).astype(np.int64)
            ys = rng.integers(-4, 4, n).astype(np.int64)
            zs = rng.integers(-4, 4, n).astype(np.int64)
            c2 = rng.integers(-4, 4, (dx + 1, dy + 1)).astype(np.int64)
            c3 = rng.integers(-4, 4, (dx + 1, dy + 1, dz + 1)).astype(np.int64)
        elif dtype_label == "complex128":
            xs = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            ys = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            zs = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            c2 = rng.standard_normal((dx + 1, dy + 1)) + 1j * rng.standard_normal((dx + 1, dy + 1))
            c3 = rng.standard_normal((dx + 1, dy + 1, dz + 1)) + 1j * rng.standard_normal((dx + 1, dy + 1, dz + 1))
        else:
            xs = rng.standard_normal(n)
            ys = rng.standard_normal(n)
            zs = rng.standard_normal(n)
            c2 = rng.standard_normal((dx + 1, dy + 1))
            c3 = rng.standard_normal((dx + 1, dy + 1, dz + 1))
        return xs.tolist(), ys.tolist(), zs.tolist(), c2.tolist(), c3.tolist(), [dx, dy, dz]

    cases = [
        ("single_point_degree_one", *mk(1, 1, 1, 1, "float64")),
        ("degree_zero_all_axes", *mk(2, 0, 0, 0, "float64")),
        ("complex_points_and_coefs", *mk(3, 1, 1, 1, "complex128")),
        ("int_points_and_coefs", *mk(3, 2, 1, 1, "int")),
    ]
    for dtype_label in ("float64", "complex128", "int"):
        for i in range(25):
            n = int(rng.integers(1, 5))
            dx, dy, dz = (int(rng.integers(0, 3)) for _ in range(3))
            cases.append((f"adversarial_{dtype_label}_{i}", *mk(n, dx, dy, dz, dtype_label)))
    return cases


def hermval2d_cases():
    return [(lbl, (xs, ys, c2), {}) for lbl, xs, ys, zs, c2, c3, deg in _herm_nd_base_cases()]


def hermval3d_cases():
    return [(lbl, (xs, ys, zs, c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _herm_nd_base_cases()]


def hermvalnd_cases():
    return [(lbl, ((xs, ys, zs), c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _herm_nd_base_cases()]


def hermgrid2d_cases():
    return [(lbl, (xs, ys, c2), {}) for lbl, xs, ys, zs, c2, c3, deg in _herm_nd_base_cases()]


def hermgrid3d_cases():
    return [(lbl, (xs, ys, zs, c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _herm_nd_base_cases()]


def hermvander2d_cases():
    return [(lbl, (xs, ys, deg[:2]), {}) for lbl, xs, ys, zs, c2, c3, deg in _herm_nd_base_cases()]


def hermvander3d_cases():
    return [(lbl, (xs, ys, zs, deg), {}) for lbl, xs, ys, zs, c2, c3, deg in _herm_nd_base_cases()]


# ---------------------------------------------------------------------------
# assembling the spec dict
# ---------------------------------------------------------------------------

def _build_hermite_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}

    specs["polynomial.hermite.hermdomain"] = ItemSpec(
        name="polynomial.hermite.hermdomain", kind="custom",
        custom_cases=_const_cases("hermdomain"),
        numpy_adapter=lambda: np.polynomial.hermite.hermdomain,
        ionp_adapter=lambda: __import__("anionpy").polynomial.hermite.hermdomain,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermzero"] = ItemSpec(
        name="polynomial.hermite.hermzero", kind="custom",
        custom_cases=_const_cases("hermzero"),
        numpy_adapter=lambda: np.polynomial.hermite.hermzero,
        ionp_adapter=lambda: __import__("anionpy").polynomial.hermite.hermzero,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermone"] = ItemSpec(
        name="polynomial.hermite.hermone", kind="custom",
        custom_cases=_const_cases("hermone"),
        numpy_adapter=lambda: np.polynomial.hermite.hermone,
        ionp_adapter=lambda: __import__("anionpy").polynomial.hermite.hermone,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermx"] = ItemSpec(
        name="polynomial.hermite.hermx", kind="custom",
        custom_cases=_const_cases("hermx"),
        numpy_adapter=lambda: np.polynomial.hermite.hermx,
        ionp_adapter=lambda: __import__("anionpy").polynomial.hermite.hermx,
        atol=0.0, rtol=0.0,
    )

    specs["polynomial.hermite.hermline"] = ItemSpec(
        name="polynomial.hermite.hermline", kind="custom",
        custom_cases=hermline_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermtrim"] = ItemSpec(
        name="polynomial.hermite.hermtrim", kind="custom",
        custom_cases=hermtrim_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermval"] = ItemSpec(
        name="polynomial.hermite.hermval", kind="custom",
        custom_cases=hermval_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermadd"] = ItemSpec(
        name="polynomial.hermite.hermadd", kind="custom",
        custom_cases=hermadd_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermsub"] = ItemSpec(
        name="polynomial.hermite.hermsub", kind="custom",
        custom_cases=hermsub_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermmulx"] = ItemSpec(
        name="polynomial.hermite.hermmulx", kind="custom",
        custom_cases=hermmulx_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermmul"] = ItemSpec(
        name="polynomial.hermite.hermmul", kind="custom",
        custom_cases=hermmul_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermdiv"] = ItemSpec(
        name="polynomial.hermite.hermdiv", kind="custom",
        custom_cases=hermdiv_cases, atol=0.0, rtol=0.0, multi_output=True,
    )
    specs["polynomial.hermite.hermpow"] = ItemSpec(
        name="polynomial.hermite.hermpow", kind="custom",
        custom_cases=hermpow_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermder"] = ItemSpec(
        name="polynomial.hermite.hermder", kind="custom",
        custom_cases=hermder_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermint"] = ItemSpec(
        name="polynomial.hermite.hermint", kind="custom",
        custom_cases=hermint_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermvander"] = ItemSpec(
        name="polynomial.hermite.hermvander", kind="custom",
        custom_cases=hermvander_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermcompanion"] = ItemSpec(
        name="polynomial.hermite.hermcompanion", kind="custom",
        custom_cases=hermcompanion_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermfromroots"] = ItemSpec(
        name="polynomial.hermite.hermfromroots", kind="custom",
        custom_cases=hermfromroots_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.hermroots"] = ItemSpec(
        name="polynomial.hermite.hermroots", kind="custom",
        custom_cases=hermroots_cases, atol=0.0, rtol=0.0,
        numpy_adapter=_hermroots_numpy_sorted,
        ionp_adapter=_hermroots_ionp_sorted,
        epsilon_tolerance={
            "complex128": ("abs", HERMROOTS_COMPLEX_EPS),
            "float64": ("abs", HERMROOTS_REAL_EPS),
        },
        epsilon_tolerance_justification=(
            "hermroots = hermcompanion (exact) + linalg.eigvals "
            "(independent eigensolver call path vs real numpy's LAPACK "
            "dgeev) + a _to_real_if_imag_zero-equivalent real-downcast "
            "step + sort. Genuinely non-bit-exact-achievable (two "
            "independent eigensolver implementations). Bound is the exact "
            "measured maximum from an independent seed=9182736, N=20000 "
            "sweep against the real Rust implementation -- see "
            "/private/tmp/herm_eps_sweep.py -- NOT a safe-margin guess: "
            "complex128 from complex-coefficient input (degree 3-6), "
            "float64 from real-coefficient input (degree 3-6)."
        ),
        epsilon_sweep={
            "complex128": (20000, HERMROOTS_COMPLEX_EPS),
            "float64": (20000, HERMROOTS_REAL_EPS),
        },
    )
    specs["polynomial.hermite.hermfit"] = ItemSpec(
        name="polynomial.hermite.hermfit", kind="custom",
        custom_cases=hermfit_cases,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"float64": ("rel", HERMFIT_EPS)},
        epsilon_tolerance_justification=(
            "Least-squares fit: normalized-Vandermonde build (exact, "
            "Rust) + linalg.lstsq (independent SVD call path vs real "
            "numpy). Genuinely non-bit-exact-achievable (SVD-based). "
            "Bound is the exact measured maximum relative (floor=1.0, "
            "matching harness.max_rel_distance) coefficient distance from "
            "an independent seed=9182736, N=20000 sweep (n in [5,20) "
            "samples, deg in [1,5)) against the real Rust implementation "
            "-- see /private/tmp/herm_eps_sweep2.py -- NOT a safe-margin "
            "guess."
        ),
        epsilon_sweep={"float64": (20000, HERMFIT_EPS)},
    )
    specs["polynomial.hermite.hermgauss"] = ItemSpec(
        name="polynomial.hermite.hermgauss", kind="custom",
        custom_cases=hermgauss_cases,
        atol=0.0, rtol=0.0, multi_output=True,
        epsilon_tolerance={"float64": ("abs", HERMGAUSS_EPS)},
        epsilon_tolerance_justification=(
            "hermgauss's nodes come from linalg.eigvalsh on the companion "
            "matrix (independent symmetric-eigensolver call path vs real "
            "numpy's LAPACK dsyevd) followed by one Newton polish step, a "
            "weight computation, and symmetrization -- all exact Rust "
            "arithmetic once the eigenvalues differ in their last bit or "
            "two. Genuinely non-bit-exact-achievable (eigensolver-based). "
            "Bound is the exact measured maximum absolute distance across "
            "BOTH outputs (nodes and weights; 8.88e-16 vs 1.11e-16, the "
            "larger declared) from an independent seed=9182736, N=20000 "
            "sweep (deg in [1,40)) against the real Rust implementation "
            "-- see /private/tmp/herm_eps_sweep.py -- NOT a safe-margin "
            "guess."
        ),
        epsilon_sweep={"float64": (20000, HERMGAUSS_EPS)},
    )
    specs["polynomial.hermite.hermweight"] = ItemSpec(
        name="polynomial.hermite.hermweight", kind="custom",
        custom_cases=hermweight_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.herm2poly"] = ItemSpec(
        name="polynomial.hermite.herm2poly", kind="custom",
        custom_cases=herm2poly_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite.poly2herm"] = ItemSpec(
        name="polynomial.hermite.poly2herm", kind="custom",
        custom_cases=poly2herm_cases, atol=0.0, rtol=0.0,
    )

    # -- Task #34 (2026-08-08): *val2d/*val3d/*valnd/*grid2d/*grid3d/
    # *vander2d/*vander3d, see this file's comment block above the
    # generators for corpus design and the float16/32/complex64 exclusion.
    for _fn in ("hermval2d", "hermval3d", "hermvalnd", "hermgrid2d",
                "hermgrid3d", "hermvander2d", "hermvander3d"):
        specs[f"polynomial.hermite.{_fn}"] = ItemSpec(
            name=f"polynomial.hermite.{_fn}", kind="custom",
            custom_cases=globals()[f"{_fn}_cases"], atol=0.0, rtol=0.0,
        )
    del _fn

    return specs


HERMITE_SPECS = _build_hermite_specs()
