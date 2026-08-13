"""anionpy.polynomial.hermite_e.* differential registry entries.

Same pattern as hermite_cases.py/laguerre_cases.py/legendre_cases.py:
builds a `HERMITE_E_SPECS: dict[str, ItemSpec]` dict, merged into
`registry.REGISTRY` at the bottom of registry.py (collision-checked, same
pattern as every other `*_cases.py` merge).

Covers the same ~24 non-N-D items as hermite_cases.py's scope
(`hermeval2d/3d/nd`, `hermegrid2d/3d`, `hermevander2d/3d`, and the
`HermiteE(ABCPolyBase)` class are NOT covered, not attempted in this pass).

This corpus is written BEFORE any Rust implementation exists (mandated
order), read directly against `numpy/polynomial/hermite_e.py`
(probabilists' HermiteE -- NOT `hermite.py`, the physicists' variant,
which differs in several of the exact spots pinned below, per
`docs/POLY-BASIS-SOURCE-AUDIT.md` and this task's brief). HermiteE is the
closest sibling to Hermite of any pair in this project -- same recurrence
SHAPE, differing essentially by factors of 2 (or their absence) -- which
makes it the highest-risk port, not the easiest. Divergences from Hermite
that a naive "delete the 2s" port would miss, each pinned explicitly:

  - `herme2poly`'s `len(c) == 2` branch does **NOT** double `c[1]` --
    `return c` unmodified (`hermite_e.py:187-188`). `herm2poly`'s sibling
    branch DOES double (`c[1] *= 2`). This is the single sharpest trap in
    the whole pair: an implementer who copies `herm2poly` and forgets to
    delete the doubling line produces a WRONG VALUE (not a rounding
    difference) at degree exactly 2, and nowhere else. Pinned as
    `len2_pinned_no_doubling` below, not left to random degree sampling.
  - `hermeval`'s Clenshaw grouping has NO `x2` precompute and NO leading
    `2 *`: `c0 = c[-i] - c1 * (nd - 1)`, `c1 = tmp + c1 * x` -- contrast
    `hermval`'s `c1 * (2 * (nd - 1))` / `c1 * x2`. Verified directly
    against `hermite_e.py:862-886`.
  - `hermemul`'s trailing combining step is `hermeadd(c0, hermemulx(c1))`
    -- NO `* 2` anywhere, contrast `hermmul`'s `hermmulx(c1) * 2`. Same
    for the inner-loop `c1 * (nd - 1)` term (no leading `2 *`).
  - `hermemulx` has NO division anywhere (`prd[1] = c[0]`, not
    `c[0] / 2`; `prd[i + 1] = c[i]`, not `c[i] / 2`) -- contrast
    `hermmulx`'s `/2` on both newly-introduced slots.
  - `hermeder` does direct assignment `der[j - 1] = j * c[j]` (no leading
    `2 *`) -- contrast `hermder`'s `(2 * j) * c[j]`.
  - `hermeint` does direct assignment `tmp[j + 1] = c[j] / (j + 1)` (no
    `2 *` in the denominator), and `tmp[1] = c[0]` (no `/2`) -- contrast
    `hermint`'s `tmp[1] = c[0] / 2`, `tmp[j + 1] = c[j] / (2 * (j + 1))`.
  - `hermevander`'s recurrence uses plain `x` throughout (`v[1] = x`,
    `v[i] = v[i-1]*x - v[i-2]*(i-1)`) -- NO `x2` precompute and no `2 *`
    factor on the subtracted term -- contrast `hermvander`'s `v[1] = x2`,
    `v[i] = v[i-1]*x2 - v[i-2]*(2*(i-1))`.
  - `hermecompanion`'s `len(c) == 2` branch is `[[-c[0] / c[1]]]` -- NO
    `-0.5` factor -- contrast `hermcompanion`'s `-.5 * c[0] / c[1]`. The
    general (`n > 2`) branch's `scl` construction has no `2.` inside the
    `sqrt` (`1. / sqrt(arange(n-1,0,-1))`, not `1/sqrt(2*arange(...))`),
    the super/subdiagonal is `sqrt(arange(1,n))` (no `.5` factor), and the
    last-column correction is `scl * c[:-1] / c[-1]` (no `2.0` in the
    denominator) -- contrast `hermcompanion`'s `.5`/`2.0` factors
    throughout. Verified directly against `hermite_e.py:1415-1452`.
  - `hermeroots`'s `len(c) == 2` branch is `-c[0] / c[1]` (no `-0.5`
    factor), and -- unlike `hermroots` -- `hermeroots` does **NOT** call
    `_to_real_if_imag_zero` at all (`hermite_e.py:1513-1516`: builds the
    rotated companion, calls `eigvals`, sorts, returns -- no real-downcast
    step). `numpy.linalg.eigvals` always upcasts its return to complex
    (`_linalg.py:1261`: `w.astype(_complexType(result_t), copy=False)`),
    so `hermeroots` on real input with degree >= 3 returns a COMPLEX
    array even when every root is real -- pinned explicitly via
    `degree_ge3_always_complex_dtype` below, since this is a genuine dtype
    divergence from `hermroots`'s behavior, not just an arithmetic one.
  - `hermegauss` DOES symmetrize both `x` and `w` (same as `hermgauss`),
    but its `_normed_hermite_e_n` helper's per-iteration scale is
    `sqrt(1./nd)` (not `hermgauss`'s `sqrt(2./nd)`) and, sharpest of all,
    its FINAL return line is `c0 + c1 * x` -- NO trailing `* sqrt(2)`
    factor at all, contrast `_normed_hermite_n`'s `c0 + c1 * x * sqrt(2)`.
    The overall weight scale is `sqrt(2*pi) / w.sum()` (not `hermgauss`'s
    bare `sqrt(pi) / w.sum()`). Verified directly against
    `hermite_e.py:1519-1621`.
  - `hermeweight(x) = exp(-0.5 * x**2)`, not `hermweight`'s `exp(-x**2)`.
  - `hermeline(off, scl)` returns `[off, scl]` (matching Legendre's shape
    exactly, no `/2`) -- NOT `hermline`'s `[off, scl/2]`.
  - `hermex = np.array([0, 1])` (int array, `[0, 1]`) -- NOT `hermx`'s
    `np.array([0, 0.5])` (forced-float by the `0.5`).
"""
from __future__ import annotations

import numpy as np

import anionpy as _anionpy  # noqa: F401  (parity import; not directly used)

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

# HermiteE's natural domain is (-inf, inf) -- no domain-restricted fixture
# analogous to Laguerre's XS_NONNEG is needed; XS already straddles zero.


def _std(pairs):
    return [(label, args, kwargs) for label, args, kwargs in pairs]


def _const_cases(attr):
    def gen():
        return [(f"{attr}_value", (), {})]
    return gen


# ---------------------------------------------------------------------------
# hermeline / hermetrim
# ---------------------------------------------------------------------------

def hermeline_cases():
    return _std([
        ("basic", (1.0, 2.0), {}),
        ("zero_scale", (3.0, 0.0), {}),
        ("negative", (-1.5, -2.5), {}),
        ("off_zero", (0.0, 4.0), {}),
        ("both_zero", (0.0, 0.0), {}),
    ])


def hermetrim_cases():
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
# hermeval
# ---------------------------------------------------------------------------

def hermeval_cases():
    cases = []
    for c_label, c in [("int_c", INT_C), ("float_c", FLOAT_C), ("bool_c", BOOL_C),
                        ("complex_c", COMPLEX_C), ("trailing_zero_c", TRAILING_ZERO_C),
                        ("single_c", SINGLE_C)]:
        cases.append((f"scalar_x__{c_label}", (X_SCALAR, c), {}))
        cases.append((f"list_x__{c_label}", (XS, c), {}))
        cases.append((f"complex_x__{c_label}", (X_COMPLEX, c), {}))
    cases.append(("array_x_2d", (np.array([[0.0, 1.0], [2.0, 3.0]]), FLOAT_C), {}))
    # len(c) == 2 is a dedicated short-circuit branch (`c0=c[0]; c1=c[1]`)
    # -- unlike hermval's, there is no shared x2 precompute here at all
    # (plain x throughout), pinned explicitly to catch an accidental x2
    # carried over from a Hermite-shaped port.
    cases.append(("len2_c_pinned", (XS, C1_SHORT), {}))
    cases.append(("len1_c_pinned", (XS, SINGLE_C), {}))
    return cases


# ---------------------------------------------------------------------------
# add / sub / mulx / mul / div / pow
# ---------------------------------------------------------------------------

def hermeadd_cases():
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


def hermesub_cases():
    return _std([
        ("same_len", (C1, C2), {}),
        ("c1_shorter", (C1_SHORT, C1), {}),
        ("identical_cancels", (FLOAT_C, FLOAT_C), {}),
        ("complex_minus_real", (COMPLEX_C, FLOAT_C), {}),
        ("int_minus_bool", (INT_C, BOOL_C), {}),
    ])


def hermemulx_cases():
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
        # `ionp-core/src/hermite_e.rs`. Permanent regression guard.
        ("negative_both_first_coef", ([-4.5 - 2.5j],), {}),
    ])


def hermemul_cases():
    return _std([
        ("basic", (C1, C2), {}),
        ("int_times_float", (INT_C, FLOAT_C), {}),
        ("complex_times_real", (COMPLEX_C, FLOAT_C), {}),
        ("with_zero_series", (FLOAT_C, ALL_ZERO_C[:1]), {}),
        ("trailing_zero_operand", (TRAILING_ZERO_C, C2), {}),
        ("single_by_single", ([2.0], [3.0]), {}),
        ("bool_operand", (BOOL_C, FLOAT_C), {}),  # as_series-family: rejects bool
        # len(c) == 1 and len(c) == 2 dedicated branches, plus the 4+-term
        # general loop that exercises `hermesub`/`hermeadd(..., hermemulx(c1))`
        # (no trailing *2, unlike hermmul's).
        ("len1_operand", (SINGLE_C, C2), {}),
        ("len2_operand", (C1_SHORT, C2), {}),
        ("four_term", (C_FOUR, [1.0, -1.0, 2.0, -2.0]), {}),
        # ---- Added 2026-08-08 (Monday), ticket #84 ----
        # Same defect class and fix as hermmul's analogous case in
        # hermite_cases.py (missing `as_series` pre-trim before
        # `pad_add`/`pad_sub` in `hermemul`'s internal recursion).
        # Fixed in `ionp-core/src/hermite_e.rs`. Permanent regression
        # guard.
        ("underflow_signed_zero_ticket84",
         ([5e-324 - 0j, 0.5 + 0j], [-5e-324 - 0j, 0.5 + 0j]), {}),
    ])


def hermediv_cases():
    return _std([
        ("basic", (C1, C2), {}),
        ("swap", (C2, C1), {}),
        ("c1_shorter_than_c2", (C1_SHORT, C1), {}),
        ("divide_by_scalar", (C1, [2.0]), {}),
        ("exact_division", ([2.0, 4.0, 6.0], [1.0, 2.0]), {}),
        ("complex", (COMPLEX_C, [1.0, 1.0]), {}),
        ("divide_by_zero_series", (C1, [0.0]), {}),
        ("bool_c1", (BOOL_C, [1.0, 2.0]), {}),  # as_series-family: rejects bool
        # ---- Ticket #87 (2026-08-10, Monday): `hermediv` never did numpy
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


def hermepow_cases():
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

def hermeder_cases():
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


def hermeint_cases():
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

def hermevander_cases():
    return _std([
        ("basic", (XS, 5), {}),
        ("deg_zero", (XS, 0), {}),
        # v[1] = x (plain, NOT x*2) -- pinned explicitly, the sharpest
        # single-bit divergence from hermvander's v[1] = x2.
        ("deg_one_pinned", (XS, 1), {}),
        ("scalar_x", (X_SCALAR, 3), {}),
        ("complex_x", ([1 + 1j, 2 - 1j, 0.5j], 4), {}),
        ("int_x", ([1, 2, 3], 3), {}),
        ("bool_x", ([True, False], 2), {}),  # own-dtype path: bool promoted, not rejected
    ])


def hermecompanion_cases():
    return _std([
        ("degree2", (C1,), {}),
        # len(c) == 2 special case: [[-c0/c1]] -- NO -0.5 factor, pinned
        # explicitly (the sharpest divergence from hermcompanion's
        # [[-.5*c0/c1]]).
        ("degree1_pinned_len2branch", (C1_SHORT,), {}),
        ("degree4", ([1.0, 0.0, -2.0, 0.0, 1.0],), {}),
        ("complex", (COMPLEX_C,), {}),
        ("int_c", (INT_C,), {}),
        ("bool_c", (BOOL_C,), {}),  # as_series-family: rejects bool
    ])


# ---------------------------------------------------------------------------
# fromroots / roots
# ---------------------------------------------------------------------------

def hermefromroots_cases():
    return _std([
        ("real_roots", ([-1.0, 0.0, 1.0],), {}),
        ("no_roots", ([],), {}),
        ("single_root", ([3.0],), {}),
        ("complex_roots", ([-1j, 0.0, 1j],), {}),
        ("repeated_root", ([1.0, 1.0, -1.0],), {}),
        ("five_roots", ([-2.0, -1.0, 0.0, 1.0, 2.0],), {}),
        ("bool_roots", ([True, False],), {}),  # as_series-family: rejects bool
    ])


def hermeroots_cases():
    return _std([
        ("real_roots", (np.polynomial.hermite_e.hermefromroots([-1.0, 0.0, 1.0]),), {}),
        ("complex_roots", (np.polynomial.hermite_e.hermefromroots([-1j, 0.0, 1j]),), {}),
        # len(c) == 2 special case: -c0/c1 -- pinned explicitly, this is
        # the "degree exactly 2 [coefficient array]" pin the task brief
        # requires.
        ("degree1_pinned_len2branch", (C1_SHORT,), {}),
        ("degree0_no_roots", (SINGLE_C,), {}),
        ("five_real_roots", (np.polynomial.hermite_e.hermefromroots(
            [-2.0, -1.0, 0.0, 1.0, 2.0]),), {}),
        # degree >= 3, all-real coefficients: hermeroots does NOT downcast
        # to real (unlike hermroots) -- numpy's own eigvals always
        # upcasts to complex128 and hermeroots never converts back. The
        # dtype-comparison harness catches this only if the comparator
        # checks dtype, not just value -- see the dedicated dtype probe
        # below for a belt-and-suspenders direct check.
        ("degree_ge3_always_complex_dtype", (np.polynomial.hermite_e.hermefromroots(
            [-1.0, 0.0, 1.0]),), {}),
    ])


def _canonical_root_order(arr):
    """See laguerre_cases.py's `_canonical_root_order` -- identical
    stabilization strategy (round the sort KEY, not the compared values,
    to 6 decimals, three orders of magnitude coarser than
    HERMEROOTS_COMPLEX_EPS below). `arr` may be real or complex depending
    on the case; `.real`/`.imag` both work on a real float64 array (imag
    is all zero) so this helper is dtype-agnostic."""
    arr = np.asarray(arr)
    key = list(zip(np.round(arr.real, 6), np.round(np.imag(arr), 6)))
    order = sorted(range(len(arr)), key=lambda i: key[i])
    return arr[order]


def _hermeroots_numpy_sorted(c):
    return _canonical_root_order(np.polynomial.hermite_e.hermeroots(c))


def _hermeroots_ionp_sorted(c):
    import anionpy as ap
    r = ap.polynomial.hermite_e.hermeroots(ap.array(c))
    r_np = np.asarray(r.tolist() if hasattr(r, "tolist") else r)
    return _canonical_root_order(r_np)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def hermefit_cases():
    rng = np.random.default_rng(20260807)
    x = np.array(list(XS) + [3.0, 4.0, -3.0, -4.0, 0.25, -0.25, 1.5, -1.5,
                              2.25, -2.25, 0.75, -0.75, 1.75, -1.75, 0.1,
                              -0.1, 0.3, -0.3])
    y = np.polynomial.hermite_e.hermeval(x, [0.0, 1.0, 0.0, -1.0])
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
# hermegauss / hermeweight
# ---------------------------------------------------------------------------

def hermegauss_cases():
    return _std([
        ("deg1", (1,), {}),
        ("deg2", (2,), {}),
        ("deg5", (5,), {}),
        ("deg10", (10,), {}),
        ("deg25", (25,), {}),
    ])


def hermeweight_cases():
    return _std([
        ("scalar", (0.5,), {}),
        ("list", (XS,), {}),
        ("array", (np.array(XS),), {}),
        ("negative", ([-3.0, -1.0, -0.1],), {}),
    ])


# ---------------------------------------------------------------------------
# herme2poly / poly2herme
# ---------------------------------------------------------------------------

def herme2poly_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("int_c", (INT_C,), {}),
        ("complex_c", (COMPLEX_C,), {}),
        ("trailing_zero_c", (TRAILING_ZERO_C,), {}),
        ("single_c", (SINGLE_C,), {}),
        # n == 2 branch is a BARE `return c` -- NO doubling. This is the
        # single sharpest, most implementer-hostile trap in this whole
        # pair: it is a WRONG VALUE bug (not a rounding difference) if an
        # implementer copies herm2poly's `c[1] *= 2; return c` here by
        # mistake, and it only fires at degree exactly 2. Pinned
        # explicitly, NOT left to random degree sampling, per the task
        # brief's specific warning about this exact function.
        ("len2_pinned_no_doubling", (C1_SHORT,), {}),
        ("bool_c", (BOOL_C,), {}),  # as_series-family: rejects bool
    ])


def poly2herme_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("int_c", (INT_C,), {}),
        ("complex_c", (COMPLEX_C,), {}),
        ("trailing_zero_c", (TRAILING_ZERO_C,), {}),
        ("single_c", (SINGLE_C,), {}),
        ("two_term", (C1_SHORT,), {}),
        ("bool_c", (BOOL_C,), {}),  # as_series-family: rejects bool
    ])


# HermiteE's OWN independently measured seeded-sweep constants, NOT
# inherited from HERMFIT_EPS/HERMROOTS_*_EPS/HERMGAUSS_EPS in
# hermite_cases.py or LAGFIT_EPS/... in laguerre_cases.py or LEGFIT_EPS/...
# in legendre_cases.py. Measured by /private/tmp/herme_eps_sweep.py
# (hermeroots complex/real, hermegauss) and
# /private/tmp/herme_eps_sweep2.py (hermefit's "rel" metric, floor=1.0,
# matching harness.max_rel_distance exactly), each an independent
# seed=9182736, N=20000 sweep against the real Rust implementation (run
# AFTER ionp-core/src/hermite_e.rs existed, not guessed in advance):
#   hermeroots, complex128-coefficient input -> max abs root distance
#     3.410605131648481e-13
#   hermeroots, float64-coefficient input -> max abs root distance
#     1.8189894035458565e-12
#   hermefit -> max rel (floor=1.0) coefficient distance
#     2.482499636227198e-12
#   hermegauss -> max abs distance across BOTH outputs (x, w)
#     1.7763568394002505e-15 (x) / 1.1102230246251565e-16 (w); the larger
#     of the two, 1.7763568394002505e-15, is declared since
#     epsilon_tolerance applies one bound per dtype across a multi_output
#     item.
HERMEFIT_EPS = 2.482499636227198e-12
HERMEROOTS_COMPLEX_EPS = 3.410605131648481e-13
HERMEROOTS_REAL_EPS = 1.8189894035458565e-12
HERMEGAUSS_EPS = 1.7763568394002505e-15


# ---------------------------------------------------------------------------
# Task #34 (2026-08-08): hermeval2d/hermeval3d/hermevalnd/hermegrid2d/
# hermegrid3d/hermevander2d/hermevander3d -- built on polyutils.py's shared
# _valnd/_gridnd/_vander_nd/_vander_nd_flat machinery (see that module's
# docstring for the ticket-premise correction: FOUR shared helpers back
# this family, not the two -- _vander_nd/_vander_nd_flat -- the ticket
# named). Corpus varies point count (1-4 per axis), degree (0-2 per axis),
# and dtype (float64, complex128, int -- promoted via _common_dtype).
# float16/float32/complex64 are DELIBERATELY EXCLUDED: the already-
# declared, already-shipped 1-D `hermeval`/`hermevander` kernels this
# family is built on universally upcast those three dtypes to
# float64/complex128 (values match real numpy bit-exact, but dtype does
# not) -- a pre-existing divergence in code this ticket did not write and
# has no permission to touch (lives in the shared Rust kernels, confirmed
# present for all six bases, not introduced here). See
# docs/TICKET-34-*.md.

def _herme_nd_base_cases():
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


def hermeval2d_cases():
    return [(lbl, (xs, ys, c2), {}) for lbl, xs, ys, zs, c2, c3, deg in _herme_nd_base_cases()]


def hermeval3d_cases():
    return [(lbl, (xs, ys, zs, c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _herme_nd_base_cases()]


def hermevalnd_cases():
    return [(lbl, ((xs, ys, zs), c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _herme_nd_base_cases()]


def hermegrid2d_cases():
    return [(lbl, (xs, ys, c2), {}) for lbl, xs, ys, zs, c2, c3, deg in _herme_nd_base_cases()]


def hermegrid3d_cases():
    return [(lbl, (xs, ys, zs, c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _herme_nd_base_cases()]


def hermevander2d_cases():
    return [(lbl, (xs, ys, deg[:2]), {}) for lbl, xs, ys, zs, c2, c3, deg in _herme_nd_base_cases()]


def hermevander3d_cases():
    return [(lbl, (xs, ys, zs, deg), {}) for lbl, xs, ys, zs, c2, c3, deg in _herme_nd_base_cases()]


# ---------------------------------------------------------------------------
# assembling the spec dict
# ---------------------------------------------------------------------------

def _build_hermite_e_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}

    specs["polynomial.hermite_e.hermedomain"] = ItemSpec(
        name="polynomial.hermite_e.hermedomain", kind="custom",
        custom_cases=_const_cases("hermedomain"),
        numpy_adapter=lambda: np.polynomial.hermite_e.hermedomain,
        ionp_adapter=lambda: __import__("anionpy").polynomial.hermite_e.hermedomain,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermezero"] = ItemSpec(
        name="polynomial.hermite_e.hermezero", kind="custom",
        custom_cases=_const_cases("hermezero"),
        numpy_adapter=lambda: np.polynomial.hermite_e.hermezero,
        ionp_adapter=lambda: __import__("anionpy").polynomial.hermite_e.hermezero,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermeone"] = ItemSpec(
        name="polynomial.hermite_e.hermeone", kind="custom",
        custom_cases=_const_cases("hermeone"),
        numpy_adapter=lambda: np.polynomial.hermite_e.hermeone,
        ionp_adapter=lambda: __import__("anionpy").polynomial.hermite_e.hermeone,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermex"] = ItemSpec(
        name="polynomial.hermite_e.hermex", kind="custom",
        custom_cases=_const_cases("hermex"),
        numpy_adapter=lambda: np.polynomial.hermite_e.hermex,
        ionp_adapter=lambda: __import__("anionpy").polynomial.hermite_e.hermex,
        atol=0.0, rtol=0.0,
    )

    specs["polynomial.hermite_e.hermeline"] = ItemSpec(
        name="polynomial.hermite_e.hermeline", kind="custom",
        custom_cases=hermeline_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermetrim"] = ItemSpec(
        name="polynomial.hermite_e.hermetrim", kind="custom",
        custom_cases=hermetrim_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermeval"] = ItemSpec(
        name="polynomial.hermite_e.hermeval", kind="custom",
        custom_cases=hermeval_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermeadd"] = ItemSpec(
        name="polynomial.hermite_e.hermeadd", kind="custom",
        custom_cases=hermeadd_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermesub"] = ItemSpec(
        name="polynomial.hermite_e.hermesub", kind="custom",
        custom_cases=hermesub_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermemulx"] = ItemSpec(
        name="polynomial.hermite_e.hermemulx", kind="custom",
        custom_cases=hermemulx_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermemul"] = ItemSpec(
        name="polynomial.hermite_e.hermemul", kind="custom",
        custom_cases=hermemul_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermediv"] = ItemSpec(
        name="polynomial.hermite_e.hermediv", kind="custom",
        custom_cases=hermediv_cases, atol=0.0, rtol=0.0, multi_output=True,
    )
    specs["polynomial.hermite_e.hermepow"] = ItemSpec(
        name="polynomial.hermite_e.hermepow", kind="custom",
        custom_cases=hermepow_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermeder"] = ItemSpec(
        name="polynomial.hermite_e.hermeder", kind="custom",
        custom_cases=hermeder_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermeint"] = ItemSpec(
        name="polynomial.hermite_e.hermeint", kind="custom",
        custom_cases=hermeint_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermevander"] = ItemSpec(
        name="polynomial.hermite_e.hermevander", kind="custom",
        custom_cases=hermevander_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermecompanion"] = ItemSpec(
        name="polynomial.hermite_e.hermecompanion", kind="custom",
        custom_cases=hermecompanion_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermefromroots"] = ItemSpec(
        name="polynomial.hermite_e.hermefromroots", kind="custom",
        custom_cases=hermefromroots_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.hermeroots"] = ItemSpec(
        name="polynomial.hermite_e.hermeroots", kind="custom",
        custom_cases=hermeroots_cases, atol=0.0, rtol=0.0,
        numpy_adapter=_hermeroots_numpy_sorted,
        ionp_adapter=_hermeroots_ionp_sorted,
        epsilon_tolerance={
            "complex128": ("abs", HERMEROOTS_COMPLEX_EPS),
            "float64": ("abs", HERMEROOTS_REAL_EPS),
        },
        epsilon_tolerance_justification=(
            "hermeroots = hermecompanion (exact) + linalg.eigvals "
            "(independent eigensolver call path vs real numpy's LAPACK "
            "dgeev) + sort -- NOTE hermeroots, unlike hermroots, never "
            "downcasts to real (no _to_real_if_imag_zero call in numpy's "
            "own source), so eigvals always returns complex128 here. "
            "Genuinely non-bit-exact-achievable (two independent "
            "eigensolver implementations). Bound is the exact measured "
            "maximum from an independent seed=9182736, N=20000 sweep "
            "against the real Rust implementation -- see "
            "/private/tmp/herme_eps_sweep.py -- NOT a safe-margin guess: "
            "complex128 from complex-coefficient input (degree 3-6), "
            "float64 from real-coefficient input (degree 3-6, len==2 "
            "special-case only, since degree>=3 real input still yields "
            "complex128 output as noted above)."
        ),
        epsilon_sweep={
            "complex128": (20000, HERMEROOTS_COMPLEX_EPS),
            "float64": (20000, HERMEROOTS_REAL_EPS),
        },
    )
    specs["polynomial.hermite_e.hermefit"] = ItemSpec(
        name="polynomial.hermite_e.hermefit", kind="custom",
        custom_cases=hermefit_cases,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"float64": ("rel", HERMEFIT_EPS)},
        epsilon_tolerance_justification=(
            "Least-squares fit: normalized-Vandermonde build (exact, "
            "Rust) + linalg.lstsq (independent SVD call path vs real "
            "numpy). Genuinely non-bit-exact-achievable (SVD-based). "
            "Bound is the exact measured maximum relative (floor=1.0, "
            "matching harness.max_rel_distance) coefficient distance from "
            "an independent seed=9182736, N=20000 sweep (n in [5,20) "
            "samples, deg in [1,5)) against the real Rust implementation "
            "-- see /private/tmp/herme_eps_sweep2.py -- NOT a safe-margin "
            "guess."
        ),
        epsilon_sweep={"float64": (20000, HERMEFIT_EPS)},
    )
    specs["polynomial.hermite_e.hermegauss"] = ItemSpec(
        name="polynomial.hermite_e.hermegauss", kind="custom",
        custom_cases=hermegauss_cases,
        atol=0.0, rtol=0.0, multi_output=True,
        epsilon_tolerance={"float64": ("abs", HERMEGAUSS_EPS)},
        epsilon_tolerance_justification=(
            "hermegauss's nodes come from linalg.eigvalsh on the "
            "companion matrix (independent symmetric-eigensolver call "
            "path vs real numpy's LAPACK dsyevd) followed by one Newton "
            "polish step, a weight computation, and symmetrization -- all "
            "exact Rust arithmetic once the eigenvalues differ in their "
            "last bit or two. Genuinely non-bit-exact-achievable "
            "(eigensolver-based). Bound is the exact measured maximum "
            "absolute distance across BOTH outputs (nodes and weights; "
            "the larger declared) from an independent seed=9182736, "
            "N=20000 sweep (deg in [1,40)) against the real Rust "
            "implementation -- see /private/tmp/herme_eps_sweep.py -- "
            "NOT a safe-margin guess."
        ),
        epsilon_sweep={"float64": (20000, HERMEGAUSS_EPS)},
    )
    specs["polynomial.hermite_e.hermeweight"] = ItemSpec(
        name="polynomial.hermite_e.hermeweight", kind="custom",
        custom_cases=hermeweight_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.herme2poly"] = ItemSpec(
        name="polynomial.hermite_e.herme2poly", kind="custom",
        custom_cases=herme2poly_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.hermite_e.poly2herme"] = ItemSpec(
        name="polynomial.hermite_e.poly2herme", kind="custom",
        custom_cases=poly2herme_cases, atol=0.0, rtol=0.0,
    )

    # -- Task #34 (2026-08-08): *val2d/*val3d/*valnd/*grid2d/*grid3d/
    # *vander2d/*vander3d, see this file's comment block above the
    # generators for corpus design and the float16/32/complex64 exclusion.
    for _fn in ("hermeval2d", "hermeval3d", "hermevalnd", "hermegrid2d",
                "hermegrid3d", "hermevander2d", "hermevander3d"):
        specs[f"polynomial.hermite_e.{_fn}"] = ItemSpec(
            name=f"polynomial.hermite_e.{_fn}", kind="custom",
            custom_cases=globals()[f"{_fn}_cases"], atol=0.0, rtol=0.0,
        )
    del _fn

    return specs


HERMITE_E_SPECS = _build_hermite_e_specs()
