"""anionpy.polynomial.chebyshev.* differential registry entries.

Same overall pattern as hermite_e_cases.py/legendre_cases.py/
laguerre_cases.py/hermite_cases.py: builds a
`CHEBYSHEV_SPECS: dict[str, ItemSpec]` dict, merged into
`registry.REGISTRY` at the bottom of registry.py (collision-checked, same
pattern as every other `*_cases.py` merge).

Covers 27 function-level items (`chebdomain`, `chebzero`, `chebone`,
`chebx`, `chebline`, `chebadd`, `chebsub`, `chebmulx`, `chebmul`,
`chebdiv`, `chebpow`, `chebder`, `chebint`, `chebval`, `chebvander`,
`chebfit`, `chebtrim`, `chebroots`, `chebcompanion`, `chebfromroots`,
`cheb2poly`, `poly2cheb`, `chebweight`, `chebgauss`, `chebpts1`,
`chebpts2`, `chebinterpolate`). Explicitly NOT covered, matching the
scope precedent of every landed basis: `chebval2d/3d/nd`,
`chebgrid2d/3d`, `chebvander2d/3d`, and the `Chebyshev(ABCPolyBase)`
class.

Chebyshev is structurally UNLIKE its four already-landed siblings
(hermite, hermite_e, laguerre, legendre): those four share a three-term
recurrence throughout. Chebyshev's `chebmul`/`chebdiv`/`chebpow` instead
go through a "z-series" (symmetric Laurent-series) representation via
private helpers `_cseries_to_zseries`/`_zseries_to_cseries`/
`_zseries_mul` (`np.convolve`)/`_zseries_div`, verified directly against
`numpy/polynomial/chebyshev.py`. `chebder`/`chebint`, by contrast, do
**NOT** go through the z-series machinery at all despite numpy also
defining `_zseries_der`/`_zseries_int` in the same file -- those two
helpers are dead code, never called by the public `chebder`/`chebint`,
which use their own distinct direct array-recurrence algorithms. This is
a genuine gap in this task's own brief (which implied
`chebmul/chebdiv/chebder/chebint` all route through z-series) --
independently confirmed by reading numpy's literal source, and reported
as such in the final report rather than silently working around it.

Divergence traps pinned explicitly below, each verified directly against
`numpy/polynomial/chebyshev.py`:

  - `chebval`'s Clenshaw general-loop combine step is `c0 = c[-i] - c1`
    (NO multiplicative coefficient on `c1` at all) -- genuinely distinct
    shape from every sibling basis's Clenshaw loop (which all multiply
    `c1` by some `nd`-derived factor). `x2 = 2*x` IS precomputed (unlike
    HermiteE) and used in the combine step's OTHER term:
    `c1 = tmp + c1*x2`.
  - `chebmulx` is a DIRECT recurrence (`prd[1] = c[0]`; for `i >= 1`:
    `tmp = c[i]/2`; `prd[i+2] = tmp`; `prd[i] += tmp`) -- NOT z-series
    based, despite `chebmul` (its neighbor in the source) being z-series
    based. An implementer who assumes "Chebyshev multiply-by-x must also
    be z-series, since chebmul is" would be wrong.
  - `chebmul`/`chebdiv`/`chebpow` ARE z-series based (`_cseries_to_zseries`
    then `np.convolve`/two-pointer-divide/repeated-convolve, then
    `_zseries_to_cseries`).
  - `chebder`/`chebint` use their OWN direct recurrences, NOT z-series
    (see the dead-code note above). `chebder`'s `j`-loop counts DOWN from
    `n` to `3` inclusive (`range(n, 2, -1)`), mutating `c` in place
    (`c[j-2] += (j*c[j])/(j-2)`) as it goes -- later (smaller-`j`)
    iterations read `c[j-2]` values already mutated by earlier
    (larger-`j`) iterations, so this is a genuinely order-dependent
    accumulation, not independent per-index work.
  - `chebint`'s `j`-loop is direct assignment: `tmp[j+1] = c[j]/(2*(j+1))`
    and `tmp[j-1] -= c[j]/(2*(j-1))`, with a `tmp[2] = c[1]/4` special
    first term (`j` starts at 2) -- distinct constants from every sibling
    basis's `*int` recurrence.
  - `chebcompanion`'s scale vector `scl = [1., sqrt(.5), sqrt(.5), ...]`
    is NOT cumulative-product-built (contrast every sibling's
    `multiply.accumulate`-based `scl`) -- it is `1.` in slot 0 and a flat
    `sqrt(.5)` in every other slot. `top[0] = sqrt(.5)`, `top[1:] = .5`
    (super/subdiagonal), and the last-column correction is
    `(c[:-1]/c[-1]) * (scl/scl[-1]) * .5` -- an array-valued
    `scl/scl[-1]` multiply (`[sqrt(2), 1, 1, ..., 1]`), NOT a scalar
    constant, preserved as a literal array op rather than algebraically
    simplified.
  - `chebgauss` is CLOSED-FORM (`x = cos(pi*arange(1,2n,2)/(2n))`,
    `w = ones(n)*(pi/n)`) -- NO eigensolver at all, unlike every sibling
    basis's `*gauss` (all of which use `eigvalsh` + Newton refinement).
    Independently confirmed by reading numpy's literal source; this
    resolves the brief's own flagged open question ("chebgauss may be a
    special case worth checking"). Declared bit-exact below, pending the
    build's own measurement (not assumed).
  - `chebpts1(npts)`: `x = 0.5*pi/npts * arange(-npts+1, npts+1, 2);
    return sin(x)` -- implemented via `sin`, NOT `cos`.
  - `chebpts2(npts)`: `x = linspace(-pi, 0, npts); return cos(x)`.
  - `chebweight(x) = 1./(sqrt(1.+x)*sqrt(1.-x))` -- TWO separate `sqrt`
    calls multiplied together, NOT one combined `sqrt((1-x)*(1+x))` (a
    mathematically-equal but bit-different grouping).
  - `chebroots` uses the ROTATED companion + `eigvals` + `sort`, with NO
    `_to_real_if_imag_zero`-equivalent downcast step (same shape as
    `hermeroots`, not `hermroots`) -- verified directly against
    `chebyshev.py`.
  - `chebfromroots` = `pu._fromroots(chebline, chebmul, roots)` -- same
    balanced-pairing-tree shape as every sibling basis.
"""
from __future__ import annotations

import numpy as np

import anionpy as _anionpy  # noqa: F401  (parity import; not directly used)
import anionpy.polynomial as _anionpy_poly

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

# Chebyshev's natural domain is [-1, 1], but chebval/chebmul/etc accept
# out-of-domain x same as every sibling basis -- XS straddles +/-1 on
# purpose (in-domain AND out-of-domain points in the same sweep).
XS = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.5]
XS_INDOMAIN = [-1.0, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0]
X_SCALAR = 0.5
X_COMPLEX = 1 + 2j


def _std(pairs):
    return [(label, args, kwargs) for label, args, kwargs in pairs]


def _const_cases(attr):
    def gen():
        return [(f"{attr}_value", (), {})]
    return gen


# ---------------------------------------------------------------------------
# chebline / chebtrim
# ---------------------------------------------------------------------------

def chebline_cases():
    return _std([
        ("basic", (1.0, 2.0), {}),
        ("zero_scale", (3.0, 0.0), {}),
        ("negative", (-1.5, -2.5), {}),
        ("off_zero", (0.0, 4.0), {}),
        ("both_zero", (0.0, 0.0), {}),
    ])


def chebtrim_cases():
    return _std([
        ("trailing_zeros", (TRAILING_ZERO_C,), {}),
        ("all_zero", (ALL_ZERO_C,), {}),
        ("no_trim_needed", (FLOAT_C,), {}),
        ("with_tol", (list(TRAILING_ZERO_C) + [1e-10], 1e-8), {}),
        ("complex", (COMPLEX_C + [0j],), {}),
        ("int_input", (INT_C + [0],), {}),
        ("bool_input", (BOOL_C,), {}),
    ])


# ---------------------------------------------------------------------------
# chebval
# ---------------------------------------------------------------------------

def chebval_cases():
    cases = []
    for c_label, c in [("int_c", INT_C), ("float_c", FLOAT_C), ("bool_c", BOOL_C),
                        ("complex_c", COMPLEX_C), ("trailing_zero_c", TRAILING_ZERO_C),
                        ("single_c", SINGLE_C)]:
        cases.append((f"scalar_x__{c_label}", (X_SCALAR, c), {}))
        cases.append((f"list_x__{c_label}", (XS, c), {}))
        cases.append((f"complex_x__{c_label}", (X_COMPLEX, c), {}))
    cases.append(("array_x_2d", (np.array([[0.0, 1.0], [2.0, 3.0]]), FLOAT_C), {}))
    # len(c) == 1/2 dedicated short-circuit branches, plus the general
    # (>= 3 term) Clenshaw loop whose combine step has NO multiplicative
    # coefficient on c1 -- pinned explicitly since a naive port that
    # copies a sibling basis's `c1 * nd_factor` shape would be a WRONG
    # VALUE bug here, not a rounding difference.
    cases.append(("len2_c_pinned", (XS, C1_SHORT), {}))
    cases.append(("len1_c_pinned", (XS, SINGLE_C), {}))
    cases.append(("indomain_x", (XS_INDOMAIN, C_FOUR), {}))
    return cases


# ---------------------------------------------------------------------------
# add / sub / mulx / mul / div / pow
# ---------------------------------------------------------------------------

def chebadd_cases():
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


def chebsub_cases():
    return _std([
        ("same_len", (C1, C2), {}),
        ("c1_shorter", (C1_SHORT, C1), {}),
        ("identical_cancels", (FLOAT_C, FLOAT_C), {}),
        ("complex_minus_real", (COMPLEX_C, FLOAT_C), {}),
        ("int_minus_bool", (INT_C, BOOL_C), {}),
    ])


def chebmulx_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("int_c", (INT_C,), {}),
        ("zero_series", (ALL_ZERO_C[:1],), {}),
        ("complex_c", (COMPLEX_C,), {}),
        ("single_nonzero", (SINGLE_C,), {}),
        ("bool_c", (BOOL_C,), {}),
        ("two_term", (C1_SHORT,), {}),
        # ---- Added 2026-08-08 (Monday), ticket #84 ----
        # Same defect class and fix as legmulx's analogous cases in
        # legendre_cases.py (missing `pu.as_series` pre-trim before the
        # `len(c)==1 && c[0]==0` zero-series fast-path check, lost
        # sign on negative-first-coefficient input). Fixed in
        # `ionp-core/src/chebyshev.rs`. Permanent regression guard.
        ("negative_both_first_coef", ([-4.5 - 2.5j],), {}),
    ])


def chebmul_cases():
    return _std([
        ("basic", (C1, C2), {}),
        ("int_times_float", (INT_C, FLOAT_C), {}),
        ("complex_times_real", (COMPLEX_C, FLOAT_C), {}),
        ("complex_times_complex", (COMPLEX_C, [1 - 1j, 0.5, -2j]), {}),
        ("with_zero_series", (FLOAT_C, ALL_ZERO_C[:1]), {}),
        ("trailing_zero_operand", (TRAILING_ZERO_C, C2), {}),
        ("single_by_single", ([2.0], [3.0]), {}),
        ("bool_operand", (BOOL_C, FLOAT_C), {}),
        ("len1_operand", (SINGLE_C, C2), {}),
        ("len2_operand", (C1_SHORT, C2), {}),
        ("four_term", (C_FOUR, [1.0, -1.0, 2.0, -2.0]), {}),
    ])


def chebdiv_cases():
    return _std([
        ("basic", (C1, C2), {}),
        ("swap", (C2, C1), {}),
        ("c1_shorter_than_c2", (C1_SHORT, C1), {}),
        ("divide_by_scalar", (C1, [2.0]), {}),
        ("exact_division", ([2.0, 4.0, 6.0], [1.0, 2.0]), {}),
        ("complex", (COMPLEX_C, [1.0, 1.0]), {}),
        ("complex_by_complex", (COMPLEX_C, [1 - 1j, 0.5, -2j]), {}),
        ("divide_by_zero_series", (C1, [0.0]), {}),
        ("bool_c1", (BOOL_C, [1.0, 2.0]), {}),
    ])


def chebpow_cases():
    return _std([
        ("square", (C1, 2), {}),
        ("cube", (FLOAT_C, 3), {}),
        ("power_zero", (C1, 0), {}),
        ("power_zero_complex", (COMPLEX_C, 0), {}),
        ("power_one", (C1, 1), {}),
        ("complex_base", (COMPLEX_C, 2), {}),
        ("complex_base_cube", (COMPLEX_C, 3), {}),
        ("int_c", (INT_C, 2), {}),
    ])


# ---------------------------------------------------------------------------
# der / int
# ---------------------------------------------------------------------------

def chebder_cases():
    C = C_FOUR
    return _std([
        ("first_order", (C, 1), {}),
        ("third_order", (C, 3), {}),
        ("zero_order", (C, 0), {}),
        ("order_exceeds_degree", (C, 10), {}),
        ("with_scale", (C, 1, 2.0), {}),
        ("complex_c", (COMPLEX_C, 1), {}),
        ("int_c", (INT_C, 1), {}),
        ("bool_c", (BOOL_C, 1), {}),
        # every coefficient nonzero, degree >= 4 -- the in-place
        # accumulation `c[j-2] += (j*c[j])/(j-2)` reads values mutated by
        # earlier (larger-j) loop iterations, so this probes the
        # order-dependence explicitly, not just per-index correctness.
        ("all_nonzero_accum_probe", ([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1), {}),
        ("five_term_second_order", ([1.0, 2.0, 3.0, 4.0, 5.0], 2), {}),
    ])


def chebint_cases():
    C = [1.0, 2.0, 3.0]
    return _std([
        ("first_order_default_k", (C, 1), {}),
        ("with_k", (C, 1, [1.0]), {}),
        ("second_order_with_k", (C, 2, [1.0, 2.0]), {}),
        ("with_lbnd", (C, 1, [0.0], 1.0), {}),
        ("with_scale", (C, 1, [0.0], 0.0, 2.0), {}),
        ("complex_c", (COMPLEX_C, 1, [0.0]), {}),
        ("int_c", (INT_C, 1, [0.0]), {}),
        ("bool_c", (BOOL_C, 1, [0.0]), {}),
        ("all_nonzero_probe", ([1.0, 1.0, 1.0, 1.0, 1.0], 1, [0.0]), {}),
    ])


# ---------------------------------------------------------------------------
# vander / companion
# ---------------------------------------------------------------------------

def chebvander_cases():
    return _std([
        ("basic", (XS, 5), {}),
        ("deg_zero", (XS, 0), {}),
        ("deg_one_pinned", (XS, 1), {}),
        ("scalar_x", (X_SCALAR, 3), {}),
        ("complex_x", ([1 + 1j, 2 - 1j, 0.5j], 4), {}),
        ("int_x", ([1, 2, 3], 3), {}),
        ("bool_x", ([True, False], 2), {}),
        ("indomain_x", (XS_INDOMAIN, 6), {}),
    ])


def chebcompanion_cases():
    return _std([
        ("degree2", (C1,), {}),
        ("degree1_pinned_len2branch", (C1_SHORT,), {}),
        ("degree4", ([1.0, 0.0, -2.0, 0.0, 1.0],), {}),
        ("degree5", ([1.0, 0.5, -2.0, 0.0, 1.0, 0.25],), {}),
        ("complex", (COMPLEX_C,), {}),
        ("int_c", (INT_C,), {}),
        ("bool_c", (BOOL_C,), {}),
    ])


# ---------------------------------------------------------------------------
# fromroots / roots
# ---------------------------------------------------------------------------

def chebfromroots_cases():
    return _std([
        ("real_roots", ([-1.0, 0.0, 1.0],), {}),
        ("no_roots", ([],), {}),
        ("single_root", ([3.0],), {}),
        ("complex_roots", ([-1j, 0.0, 1j],), {}),
        ("repeated_root", ([1.0, 1.0, -1.0],), {}),
        ("five_roots", ([-2.0, -1.0, 0.0, 1.0, 2.0],), {}),
        ("bool_roots", ([True, False],), {}),
        # ---- Added 2026-08-07 (Monday), and these FAIL by design. ----
        # Every case above uses values that are exact in binary or that happen
        # to round the same way, so this corpus was green while the item was
        # not bit-exact on ~46% of a random sweep (seed 0xA11CE, N=2000).
        # `chebfromroots` is `pu._fromroots(chebline, chebmul, roots)` and
        # `chebmul` routes through `np.convolve`, whose summation order we
        # cannot reproduce -- the same unidentified mechanism that caused
        # `polynomial.polynomial.polymul`/`polypow`/`polyfromroots` to be
        # REVOKED. The declaration here has been revoked for the same reason.
        # Divergence scales with root count because it accumulates per
        # multiply: 1 root 0/400, 2 roots 51/400, 3 roots 174/400,
        # 4 roots 292/400, 5 roots 342/400. One root does no multiply at all.
        # DO NOT "fix" these by loosening the tolerance or deleting the cases.
        # They exist so the corpus detects the defect instead of passing by
        # luck. They come off the failing list when chebmul is reproduced
        # bit-for-bit, not before.
        ("nonexact_real_roots_DIVERGES", ([0.7, -1.3, 2.9],), {}),
        ("nonexact_complex_roots_DIVERGES", ([0.1 + 0.2j, -0.3j, 1.7],), {}),
    ])


def chebroots_cases():
    return _std([
        ("real_roots", (np.polynomial.chebyshev.chebfromroots([-1.0, 0.0, 1.0]),), {}),
        ("complex_roots", (np.polynomial.chebyshev.chebfromroots([-1j, 0.0, 1j]),), {}),
        ("degree1_pinned_len2branch", (C1_SHORT,), {}),
        ("degree0_no_roots", (SINGLE_C,), {}),
        ("five_real_roots", (np.polynomial.chebyshev.chebfromroots(
            [-2.0, -1.0, 0.0, 1.0, 2.0]),), {}),
        ("degree_ge3_always_complex_dtype", (np.polynomial.chebyshev.chebfromroots(
            [-1.0, 0.0, 1.0]),), {}),
    ])


def _canonical_root_order(arr):
    arr = np.asarray(arr)
    key = list(zip(np.round(arr.real, 6), np.round(np.imag(arr), 6)))
    order = sorted(range(len(arr)), key=lambda i: key[i])
    return arr[order]


def _chebroots_numpy_sorted(c):
    return _canonical_root_order(np.polynomial.chebyshev.chebroots(c))


def _chebroots_ionp_sorted(c):
    import anionpy as ap
    r = ap.polynomial.chebyshev.chebroots(ap.array(c))
    r_np = np.asarray(r.tolist() if hasattr(r, "tolist") else r)
    return _canonical_root_order(r_np)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def chebfit_cases():
    rng = np.random.default_rng(20260807)
    x = np.array(list(XS) + [3.0, 4.0, -3.0, -4.0, 0.25, -0.25, 1.5, -1.5,
                              2.25, -2.25, 0.75, -0.75, 1.75, -1.75, 0.1,
                              -0.1, 0.3, -0.3])
    y = np.polynomial.chebyshev.chebval(x, [0.0, 1.0, 0.0, -1.0])
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
# chebgauss / chebweight / chebpts1 / chebpts2
# ---------------------------------------------------------------------------

def chebgauss_cases():
    return _std([
        ("deg1", (1,), {}),
        ("deg2", (2,), {}),
        ("deg5", (5,), {}),
        ("deg10", (10,), {}),
        ("deg25", (25,), {}),
    ])


def chebweight_cases():
    return _std([
        ("scalar", (0.5,), {}),
        ("list", (XS_INDOMAIN,), {}),
        ("array", (np.array(XS_INDOMAIN),), {}),
        ("negative", ([-0.9, -0.5, -0.1],), {}),
    ])


def chebpts1_cases():
    return _std([
        ("small", (1,), {}),
        ("five", (5,), {}),
        ("ten", (10,), {}),
        ("twenty_five", (25,), {}),
    ])


def chebpts2_cases():
    return _std([
        ("small", (2,), {}),
        ("five", (5,), {}),
        ("ten", (10,), {}),
        ("twenty_five", (25,), {}),
    ])


# ---------------------------------------------------------------------------
# chebinterpolate
# ---------------------------------------------------------------------------

def _interp_func_linear(x):
    return 2.0 * x - 1.0


def _interp_func_cubic(x):
    return x**3 - 2.0 * x


def _interp_func_trig(x):
    return np.sin(3.0 * x) + np.cos(x)


def chebinterpolate_cases():
    return _std([
        ("linear_deg3", (_interp_func_linear, 3), {}),
        ("cubic_deg3", (_interp_func_cubic, 3), {}),
        ("cubic_deg5", (_interp_func_cubic, 5), {}),
        ("trig_deg10", (_interp_func_trig, 10), {}),
        ("deg0", (_interp_func_linear, 0), {}),
        ("deg1", (_interp_func_linear, 1), {}),
    ])


# ---------------------------------------------------------------------------
# Chebyshev.interpolate (the classmethod -- distinct item from the
# module-level chebinterpolate function above; ticket #34 dunder cluster).
# Direct port of real numpy's `Chebyshev.interpolate`: remaps `func` through
# `mapdomain` onto `cls.window`, then delegates to `chebinterpolate` --
# already exact per the entry above, so this classmethod's only NEW
# behavior to verify is the domain remap plus `cls(coef, domain=domain)`
# construction. Non-default `domain` is exercised deliberately (the remap
# is a no-op at `domain == window == [-1, 1]`, so a non-default domain is
# the only case that actually exercises `mapdomain`).
# ---------------------------------------------------------------------------

def cheb_class_interpolate_cases():
    return _std([
        ("linear_deg3_default_domain", (_interp_func_linear, 3), {}),
        ("cubic_deg5_custom_domain", (_interp_func_cubic, 5), {"domain": [-3.0, 3.0]}),
        ("trig_deg10_custom_domain", (_interp_func_trig, 10), {"domain": [0.0, 5.0]}),
        ("deg0_custom_domain", (_interp_func_linear, 0), {"domain": [-2.0, 2.0]}),
    ])


# ---------------------------------------------------------------------------
# herme2poly / poly2herme
# ---------------------------------------------------------------------------

def cheb2poly_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("int_c", (INT_C,), {}),
        ("complex_c", (COMPLEX_C,), {}),
        ("trailing_zero_c", (TRAILING_ZERO_C,), {}),
        ("single_c", (SINGLE_C,), {}),
        ("two_term", (C1_SHORT,), {}),
        ("four_term", (C_FOUR,), {}),
        ("bool_c", (BOOL_C,), {}),
    ])


def poly2cheb_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("int_c", (INT_C,), {}),
        ("complex_c", (COMPLEX_C,), {}),
        ("trailing_zero_c", (TRAILING_ZERO_C,), {}),
        ("single_c", (SINGLE_C,), {}),
        ("two_term", (C1_SHORT,), {}),
        ("bool_c", (BOOL_C,), {}),
    ])


# Chebyshev's OWN independently measured seeded-sweep constants, NOT
# inherited from any sibling basis's *_EPS constants. Measured AFTER the
# Rust implementation was built, by a dedicated seed=9182736, N=20000
# sweep (deg in [1,7), x/coefficients real-normal and complex-normal, m
# (sample count for chebfit) in [deg+2, deg+8)) against the real Rust
# implementation, using `harness.max_rel_distance`'s exact formula
# (`max(|numpy_out - ionp_out| / max(|numpy_out|, 1.0))`, NOT a
# `max(|a|,|b|,1.0)` variant). chebinterpolate was measured bit-exact (0
# mismatches, max_rel_distance == 0.0) over a SEPARATE N=20000 sweep across
# four different trig/poly/exp test functions and deg in [0,12) -- its
# constant below is the true measured maximum (zero), not a placeholder;
# `epsilon_tolerance` is NOT declared for it (see chebinterpolate_cases()
# and this file's spec-assembly section) since real bit-exactness needs no
# tolerance.
CHEBFIT_EPS = 3.357865793305266e-10
CHEBROOTS_COMPLEX_EPS = 6.3825755133549975e-15
CHEBROOTS_REAL_EPS = 1.6812769892473484e-14
CHEBINTERPOLATE_EPS = 0.0


# ---------------------------------------------------------------------------
# Task #34 (2026-08-08): chebval2d/chebval3d/chebvalnd/chebgrid2d/
# chebgrid3d/chebvander2d/chebvander3d -- built on polyutils.py's shared
# _valnd/_gridnd/_vander_nd/_vander_nd_flat machinery (see that module's
# docstring for the ticket-premise correction: FOUR shared helpers back
# this family, not the two -- _vander_nd/_vander_nd_flat -- the ticket
# named). Corpus varies point count (1-4 per axis), degree (0-2 per axis),
# and dtype (float64, complex128, int -- promoted via _common_dtype).
# float16/float32/complex64 are DELIBERATELY EXCLUDED: the already-
# declared, already-shipped 1-D `chebval`/`chebvander` kernels this family
# is built on universally upcast those three dtypes to float64/complex128
# (values match real numpy bit-exact, but dtype does not) -- a
# pre-existing divergence in code this ticket did not write and has no
# permission to touch (lives in the shared Rust kernels, confirmed present
# for all six bases, not introduced here). See docs/TICKET-34-*.md.

def _cheb_nd_base_cases():
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


def chebval2d_cases():
    return [(lbl, (xs, ys, c2), {}) for lbl, xs, ys, zs, c2, c3, deg in _cheb_nd_base_cases()]


def chebval3d_cases():
    return [(lbl, (xs, ys, zs, c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _cheb_nd_base_cases()]


def chebvalnd_cases():
    return [(lbl, ((xs, ys, zs), c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _cheb_nd_base_cases()]


def chebgrid2d_cases():
    return [(lbl, (xs, ys, c2), {}) for lbl, xs, ys, zs, c2, c3, deg in _cheb_nd_base_cases()]


def chebgrid3d_cases():
    return [(lbl, (xs, ys, zs, c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _cheb_nd_base_cases()]


def chebvander2d_cases():
    return [(lbl, (xs, ys, deg[:2]), {}) for lbl, xs, ys, zs, c2, c3, deg in _cheb_nd_base_cases()]


def chebvander3d_cases():
    return [(lbl, (xs, ys, zs, deg), {}) for lbl, xs, ys, zs, c2, c3, deg in _cheb_nd_base_cases()]


# ---------------------------------------------------------------------------
# assembling the spec dict
# ---------------------------------------------------------------------------

def _build_chebyshev_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}

    specs["polynomial.chebyshev.chebdomain"] = ItemSpec(
        name="polynomial.chebyshev.chebdomain", kind="custom",
        custom_cases=_const_cases("chebdomain"),
        numpy_adapter=lambda: np.polynomial.chebyshev.chebdomain,
        ionp_adapter=lambda: __import__("anionpy").polynomial.chebyshev.chebdomain,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebzero"] = ItemSpec(
        name="polynomial.chebyshev.chebzero", kind="custom",
        custom_cases=_const_cases("chebzero"),
        numpy_adapter=lambda: np.polynomial.chebyshev.chebzero,
        ionp_adapter=lambda: __import__("anionpy").polynomial.chebyshev.chebzero,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebone"] = ItemSpec(
        name="polynomial.chebyshev.chebone", kind="custom",
        custom_cases=_const_cases("chebone"),
        numpy_adapter=lambda: np.polynomial.chebyshev.chebone,
        ionp_adapter=lambda: __import__("anionpy").polynomial.chebyshev.chebone,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebx"] = ItemSpec(
        name="polynomial.chebyshev.chebx", kind="custom",
        custom_cases=_const_cases("chebx"),
        numpy_adapter=lambda: np.polynomial.chebyshev.chebx,
        ionp_adapter=lambda: __import__("anionpy").polynomial.chebyshev.chebx,
        atol=0.0, rtol=0.0,
    )

    specs["polynomial.chebyshev.chebline"] = ItemSpec(
        name="polynomial.chebyshev.chebline", kind="custom",
        custom_cases=chebline_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebtrim"] = ItemSpec(
        name="polynomial.chebyshev.chebtrim", kind="custom",
        custom_cases=chebtrim_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebval"] = ItemSpec(
        name="polynomial.chebyshev.chebval", kind="custom",
        custom_cases=chebval_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebadd"] = ItemSpec(
        name="polynomial.chebyshev.chebadd", kind="custom",
        custom_cases=chebadd_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebsub"] = ItemSpec(
        name="polynomial.chebyshev.chebsub", kind="custom",
        custom_cases=chebsub_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebmulx"] = ItemSpec(
        name="polynomial.chebyshev.chebmulx", kind="custom",
        custom_cases=chebmulx_cases, atol=0.0, rtol=0.0,
    )
    # chebmul is z-series based: numpy's private `_zseries_mul` calls
    # `np.convolve` directly. Measured (seed=9182736): a plain
    # `np.convolve(a, b)` on small arbitrary real arrays disagrees with a
    # naive double-loop-accumulate equivalent in ~58% of trials (1165/2000).
    # Tried and ruled out as reproduction strategies: i-outer/j-inner,
    # j-outer/i-inner, k-indexed forward accumulation, k-indexed reverse
    # accumulation, FMA-based (multiple orders), and pairwise/tree summation
    # -- none reproduce numpy's exact bits. This is the same root cause as
    # this codebase's own `polynomial.polynomial.polymul` REVOKED entry (see
    # `anionpy/_state/polynomial.py`): numpy's C-level convolve/correlate
    # uses SIMD-vectorized/blocked accumulation that no simple sequential
    # algorithm reproduces. The curated corpus below happens to pass
    # bit-exact (small hand-picked values rarely land on a divergent
    # accumulation order) -- caught only by out-of-corpus sweeping, same as
    # polymul. NOT declared "exact" in `anionpy/_state/polynomial.py`; see
    # that file's chebyshev section for the REVOKED-style comment.
    specs["polynomial.chebyshev.chebmul"] = ItemSpec(
        name="polynomial.chebyshev.chebmul", kind="custom",
        custom_cases=chebmul_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebdiv"] = ItemSpec(
        name="polynomial.chebyshev.chebdiv", kind="custom",
        custom_cases=chebdiv_cases, atol=0.0, rtol=0.0, multi_output=True,
    )
    # chebpow is a repeated z-series convolve loop (same `zseries_mul` /
    # `np.convolve` root cause as chebmul directly above -- see that item's
    # comment block). NOT declared "exact" in `anionpy/_state/polynomial.py`.
    specs["polynomial.chebyshev.chebpow"] = ItemSpec(
        name="polynomial.chebyshev.chebpow", kind="custom",
        custom_cases=chebpow_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebder"] = ItemSpec(
        name="polynomial.chebyshev.chebder", kind="custom",
        custom_cases=chebder_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebint"] = ItemSpec(
        name="polynomial.chebyshev.chebint", kind="custom",
        custom_cases=chebint_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebvander"] = ItemSpec(
        name="polynomial.chebyshev.chebvander", kind="custom",
        custom_cases=chebvander_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebcompanion"] = ItemSpec(
        name="polynomial.chebyshev.chebcompanion", kind="custom",
        custom_cases=chebcompanion_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebfromroots"] = ItemSpec(
        name="polynomial.chebyshev.chebfromroots", kind="custom",
        custom_cases=chebfromroots_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebroots"] = ItemSpec(
        name="polynomial.chebyshev.chebroots", kind="custom",
        custom_cases=chebroots_cases, atol=0.0, rtol=0.0,
        numpy_adapter=_chebroots_numpy_sorted,
        ionp_adapter=_chebroots_ionp_sorted,
        epsilon_tolerance={
            "complex128": ("abs", CHEBROOTS_COMPLEX_EPS),
            "float64": ("abs", CHEBROOTS_REAL_EPS),
        },
        epsilon_tolerance_justification=(
            "chebroots = chebcompanion (exact) + linalg.eigvals "
            "(independent eigensolver call path vs real numpy's LAPACK "
            "dgeev) + sort -- like hermeroots (not hermroots), chebroots "
            "never downcasts to real (no _to_real_if_imag_zero call in "
            "numpy's own source), so eigvals always returns complex128 "
            "here. Genuinely non-bit-exact-achievable (two independent "
            "eigensolver implementations). Bound is the exact measured "
            "maximum from an independent seed=9182736, N=20000 sweep "
            "against the real Rust implementation -- see the task's final "
            "report for the sweep script -- NOT a safe-margin guess."
        ),
        epsilon_sweep={
            "complex128": (20000, CHEBROOTS_COMPLEX_EPS),
            "float64": (20000, CHEBROOTS_REAL_EPS),
        },
    )
    specs["polynomial.chebyshev.chebfit"] = ItemSpec(
        name="polynomial.chebyshev.chebfit", kind="custom",
        custom_cases=chebfit_cases,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"float64": ("rel", CHEBFIT_EPS)},
        epsilon_tolerance_justification=(
            "Least-squares fit: normalized-Vandermonde build (exact, "
            "Rust) + linalg.lstsq (independent SVD call path vs real "
            "numpy). Genuinely non-bit-exact-achievable (SVD-based). "
            "Bound is the exact measured maximum relative (floor=1.0, "
            "matching harness.max_rel_distance) coefficient distance from "
            "an independent seed=9182736, N=20000 sweep (n in [5,20) "
            "samples, deg in [1,5)) against the real Rust implementation "
            "-- NOT a safe-margin guess."
        ),
        epsilon_sweep={"float64": (20000, CHEBFIT_EPS)},
    )
    specs["polynomial.chebyshev.chebweight"] = ItemSpec(
        name="polynomial.chebyshev.chebweight", kind="custom",
        custom_cases=chebweight_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebgauss"] = ItemSpec(
        name="polynomial.chebyshev.chebgauss", kind="custom",
        custom_cases=chebgauss_cases, atol=0.0, rtol=0.0, multi_output=True,
    )
    specs["polynomial.chebyshev.chebpts1"] = ItemSpec(
        name="polynomial.chebyshev.chebpts1", kind="custom",
        custom_cases=chebpts1_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebpts2"] = ItemSpec(
        name="polynomial.chebyshev.chebpts2", kind="custom",
        custom_cases=chebpts2_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.chebinterpolate"] = ItemSpec(
        name="polynomial.chebyshev.chebinterpolate", kind="custom",
        custom_cases=chebinterpolate_cases,
        atol=0.0, rtol=0.0,
        # Measured genuinely bit-exact: an independent seed=9182736, N=20000
        # sweep across four trig/poly/exp test functions and deg in [0,12)
        # found 0 mismatches (max_rel_distance == 0.0, see CHEBINTERPOLATE_EPS
        # above). chebinterpolate builds on chebpts1 (exact, trig-based) and
        # chebvander/matmul (exact, Rust); the feared last-bit libm
        # divergence in the sample-function evaluations did not materialize
        # in measurement, so no epsilon_tolerance is declared here -- a real
        # zero-tolerance bit-exact item needs no tolerance mechanism.
    )
    specs["polynomial.Chebyshev.interpolate"] = ItemSpec(
        name="polynomial.Chebyshev.interpolate", kind="custom",
        custom_cases=cheb_class_interpolate_cases,
        numpy_adapter=lambda func, deg, **kw: (
            lambda p: (p.coef.tolist(), p.domain.tolist())
        )(np.polynomial.Chebyshev.interpolate(func, deg, **kw)),
        ionp_adapter=lambda func, deg, **kw: (
            lambda p: (list(p.coef.tolist()), list(p.domain.tolist()))
        )(_anionpy_poly.Chebyshev.interpolate(func, deg, **kw)),
        scalar_like=True,
        # Out-of-corpus verified (/private/tmp/probe34_dunders.py, this
        # ticket): 24/24 checks (2 test functions x 4 degrees x 3 domains
        # incl. default), coefficients AND domain both bit-exact. This
        # classmethod is a thin wrapper -- `mapdomain` remap (already-exact,
        # `_polybase.py`'s own port) into the already-exact module-level
        # `chebinterpolate` above -- so no new numeric kernel is introduced,
        # only composition of two already-verified pieces.
    )
    specs["polynomial.chebyshev.cheb2poly"] = ItemSpec(
        name="polynomial.chebyshev.cheb2poly", kind="custom",
        custom_cases=cheb2poly_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.chebyshev.poly2cheb"] = ItemSpec(
        name="polynomial.chebyshev.poly2cheb", kind="custom",
        custom_cases=poly2cheb_cases, atol=0.0, rtol=0.0,
    )

    # -- Task #34 (2026-08-08): *val2d/*val3d/*valnd/*grid2d/*grid3d/
    # *vander2d/*vander3d, see this file's comment block above the
    # generators for corpus design and the float16/32/complex64 exclusion.
    for _fn in ("chebval2d", "chebval3d", "chebvalnd", "chebgrid2d",
                "chebgrid3d", "chebvander2d", "chebvander3d"):
        specs[f"polynomial.chebyshev.{_fn}"] = ItemSpec(
            name=f"polynomial.chebyshev.{_fn}", kind="custom",
            custom_cases=globals()[f"{_fn}_cases"], atol=0.0, rtol=0.0,
        )
    del _fn

    return specs


CHEBYSHEV_SPECS = _build_chebyshev_specs()
