"""anionpy.polynomial.polynomial.* differential registry entries.

NEW FILE (permitted, same as linalg_cases.py/ufunc_registry.py): builds a
`POLYNOMIAL_SPECS: dict[str, ItemSpec]` dict, merged into `registry.REGISTRY`
at the bottom of registry.py (collision-checked, same pattern as every
other `*_cases.py` merge in that file).

Covers the 21 `numpy.polynomial.polynomial.*` items implemented in this
vertical slice (see anionpy/polynomial/polynomial.py's module docstring
for the exact, deliberate scope boundary -- the 7 `*2d`/`*3d`/`*nd`
composition items are NOT covered here, not attempted in this pass).

Every `custom_cases` generator below deliberately exercises: int input
(dtype promotion to float64), float input, complex input, trailing-zero
coefficients (trimming behavior), and -- for the handful of items where it
is meaningful -- return-type/shape identity. Tolerances are declared only
where actually measured (see the bottom of this file for the `polyfit`/
`polyroots` epsilon declarations and their sweep evidence).
"""
from __future__ import annotations

import cmath

import numpy as np

import anionpy as _anionpy

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

XS = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.5]
X_SCALAR = 2.0
X_COMPLEX = 1 + 2j


def _std(pairs):
    return [(label, args, kwargs) for label, args, kwargs in pairs]


# ---------------------------------------------------------------------------
# constants (kind="custom" with zero-arg cases: just fetch the attribute)
# ---------------------------------------------------------------------------

def _const_cases(attr):
    def gen():
        return [(f"{attr}_value", (), {})]
    return gen


# ---------------------------------------------------------------------------
# polyline
# ---------------------------------------------------------------------------

def polyline_cases():
    return _std([
        ("basic", (1.0, 2.0), {}),
        ("zero_scale", (3.0, 0.0), {}),
        ("negative", (-1.5, -2.5), {}),
        ("off_zero", (0.0, 4.0), {}),
        ("both_zero", (0.0, 0.0), {}),
    ])


# ---------------------------------------------------------------------------
# polytrim
# ---------------------------------------------------------------------------

def polytrim_cases():
    return _std([
        ("trailing_zeros", (TRAILING_ZERO_C,), {}),
        ("all_zero", (ALL_ZERO_C,), {}),
        ("no_trim_needed", (FLOAT_C,), {}),
        ("with_tol", (list(TRAILING_ZERO_C) + [1e-10], 1e-8), {}),
        ("complex", (COMPLEX_C + [0j],), {}),
        ("int_input", (INT_C + [0],), {}),
        # as_series rejects bool outright (np.common_type has no bool
        # support) -- verified: P.polytrim([True, False]) raises
        # ValueError("Coefficient arrays have no common type").
        ("bool_input", (BOOL_C,), {}),
    ])


# ---------------------------------------------------------------------------
# polyval / polyvalfromroots
# ---------------------------------------------------------------------------

def polyval_cases():
    cases = []
    for c_label, c in [("int_c", INT_C), ("float_c", FLOAT_C), ("bool_c", BOOL_C),
                        ("complex_c", COMPLEX_C), ("trailing_zero_c", TRAILING_ZERO_C),
                        ("single_c", SINGLE_C)]:
        cases.append((f"scalar_x__{c_label}", (X_SCALAR, c), {}))
        cases.append((f"list_x__{c_label}", (XS, c), {}))
        cases.append((f"complex_x__{c_label}", (X_COMPLEX, c), {}))
    cases.append(("array_x_2d", (np.array([[0.0, 1.0], [2.0, 3.0]]), FLOAT_C), {}))
    return cases


def polyvalfromroots_cases():
    roots_real = [-1.0, 0.0, 1.0]
    roots_complex = [-1j, 0.0, 1j]
    cases = []
    for r_label, r in [("real_roots", roots_real), ("complex_roots", roots_complex),
                        ("empty_roots", [])]:
        cases.append((f"scalar_x__{r_label}", (X_SCALAR, r), {}))
        cases.append((f"list_x__{r_label}", (XS, r), {}))
    return cases


# ---------------------------------------------------------------------------
# add / sub / mulx / mul / div / pow
# ---------------------------------------------------------------------------

def polyadd_cases():
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


def polysub_cases():
    return _std([
        ("same_len", (C1, C2), {}),
        ("c1_shorter", (C1_SHORT, C1), {}),
        ("identical_cancels", (FLOAT_C, FLOAT_C), {}),
        ("complex_minus_real", (COMPLEX_C, FLOAT_C), {}),
        ("int_minus_bool", (INT_C, BOOL_C), {}),
    ])


def polymulx_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("int_c", (INT_C,), {}),
        ("zero_series", (ALL_ZERO_C[:1], ), {}),  # numpy special-cases [0]
        ("complex_c", (COMPLEX_C,), {}),
        ("single_nonzero", (SINGLE_C,), {}),
        ("bool_c", (BOOL_C,), {}),  # as_series-family: rejects bool
        # 2026-08-07 (Monday): numpy's `polymulx` derives its padding zero
        # via `prd[0] = c[0] * 0`, NOT a literal `0.0`/`0j` fill -- for a
        # first coefficient with a negative real or imaginary part, `x*0`
        # is IEEE754 `-0.0`, not `+0.0`. Every case above has a
        # non-negative first coefficient (COMPLEX_C[0] == 1+1j), so none
        # of them can distinguish a correct derived-zero fill from a wrong
        # literal-zero fill -- comparing VALUES can never catch this
        # (0.0 == -0.0), only bit/signbit comparison can, which is exactly
        # what harness.py's default "exact" comparison already does. Out-
        # of-corpus sweep (seed 0xDEADBE, N=2000, complex coefficients)
        # found 1021/2000 signed-zero-only divergences before the
        # `ionp-core/src/poly.rs` `mulx` fix (literal `T::zero()` fill ->
        # `c[0].poly_mul(T::zero())` derived fill, matching numpy exactly).
        ("negative_real_first_coef", ([-1.02777029 + 0.06451574j, 2.0 - 1.0j],), {}),
        ("negative_imag_first_coef", ([0.5 - 3.25j, 1.0 + 1.0j],), {}),
        ("negative_both_first_coef", ([-4.5 - 2.5j],), {}),
        ("negative_real_float", ([-2.5, 1.0, 3.0],), {}),
    ])


# 2026-08-07 (Monday): the fixed-value cases above use small integral
# coefficients ([1,2,3]-class values), which are exactly representable in
# float64 and whose products/sums stay exact regardless of accumulation
# order -- they cannot distinguish anionpy's `convolve_full` (own
# outer-product-accumulation order) from real numpy's `np.convolve`
# (BLAS-backed `correlate`, different accumulation order/instruction
# sequence) because there is no rounding for either side to disagree
# about. Out-of-corpus sweeping (see KNOWN-DIFFERENCES.md's "(b) polymul,
# polypow and polyfromroots are declared exact but are not bit-exact"
# entry) found 853/1500 (56.9%) non-bit-exact against real numpy for
# generic float64 coefficients -- a corpus that cannot catch that is not a
# corpus. These extra cases fix that: non-representable (`rng.uniform`)
# float64 coefficients across degrees 1-7, generated with a fixed seed
# for reproducibility. Comparison is via this item's declared
# `atol=0.0, rtol=0.0` (exact-value, i.e. bit-exact for finite floats --
# see `ItemSpec.atol`/`rtol`'s docstring and `harness.compare_values`).
def _polymul_adversarial_cases():
    rng = np.random.default_rng(20260807)
    cases = []
    for i in range(60):
        n1 = int(rng.integers(2, 9))
        n2 = int(rng.integers(2, 9))
        c1 = rng.uniform(-9.0, 9.0, size=n1).tolist()
        c2 = rng.uniform(-9.0, 9.0, size=n2).tolist()
        cases.append((f"adversarial_float_{i}", (c1, c2), {}))
    return cases


def polymul_cases():
    return _std([
        ("basic", (C1, C2), {}),
        ("int_times_float", (INT_C, FLOAT_C), {}),
        ("complex_times_real", (COMPLEX_C, FLOAT_C), {}),
        ("with_zero_series", (FLOAT_C, ALL_ZERO_C[:1]), {}),
        ("trailing_zero_operand", (TRAILING_ZERO_C, C2), {}),
        ("single_by_single", ([2.0], [3.0]), {}),
        ("bool_operand", (BOOL_C, FLOAT_C), {}),  # as_series-family: rejects bool
    ]) + _polymul_adversarial_cases()


def polydiv_cases():
    return _std([
        ("basic", (C1, C2), {}),
        ("swap", (C2, C1), {}),
        ("c1_shorter_than_c2", (C1_SHORT, C1), {}),
        ("divide_by_scalar", (C1, [2.0]), {}),
        ("exact_division", ([2.0, 4.0, 6.0], [1.0, 2.0]), {}),
        ("complex", (COMPLEX_C, [1.0, 1.0]), {}),
        ("divide_by_zero_series", (C1, [0.0]), {}),
        ("bool_c1", (BOOL_C, [1.0, 2.0]), {}),  # as_series-family: rejects bool
        # ---- Ticket #85 (2026-08-08, Monday): `lc2 == 1` early-return
        # branch's placeholder remainder was a literal `T::zero()` fill,
        # not numpy's `c1[:1] * 0` DERIVED zero -- loses the sign a
        # negative denormal `c1[0]` produces (`(-5e-324) * 0 == -0.0`
        # under IEEE-754). Measured 1069/4000 (26.7%) signed-zero-only
        # divergences before the `ionp-core/src/poly.rs` `div` fix
        # (`vec![T::zero()]` -> `vec![c1_t[0].poly_mul(T::zero())]`, same
        # class as `mulx`'s `prd[0]` fill). Exact repro from the ticket's
        # own measurement.
        ("underflow_signed_zero_remainder_ticket85",
         ([complex(-5e-324, -5e-324), complex(5e-324, 5e-324),
           complex(5e-324, -5e-324), complex(5e-324, -5e-324),
           complex(5e-324, 5e-324)],
          [complex(1.8510053383590988, 0.8565034784435615)]), {}),
        # A SECOND, independently-discovered defect in the same two
        # early-return branches PLUS the synthetic-division loop below:
        # this port never did numpy `polydiv`'s top-level
        # `[c1, c2] = pu.as_series([c1, c2])` pre-trim, so an untrimmed
        # trailing exact zero in `c1` leaked into the returned
        # remainder/quotient SHAPE and the loop's length bookkeeping
        # (`lc1`/`dlen`/`j`). Fixed by trimming both operands via
        # `trim_trailing_zeros` at the top of `div`, matching `chebdiv`'s
        # already-correct pattern. Found by an out-of-corpus diversity
        # sweep (seed 0xC0FFEE), NOT the ticket's own probe.
        ("untrimmed_trailing_zero_c1_ticket85",
         ([complex(-1.328486243839e-311, 0.0), complex(-0.0, 0.0)],
          [complex(4.336557665821166, 3.255238001378803),
           complex(-2.102326022659824, 4.765853139103664),
           complex(-4.312630257938191, -2.4080670455371225),
           complex(-0.9814047316697447, 2.699753210019039),
           complex(-3.9544228256592495, -1.6298063095319257),
           complex(0.07396333253658405, 4.430937833861501)]), {}),
    ])


# 2026-08-07 (Monday): same "small-integer corpus can't see rounding
# divergence" gap as `_polymul_adversarial_cases` above -- `polypow` is
# `polymul` applied repeatedly (see `anionpy/polynomial/polynomial.py`'s
# `polypow`), so it inherits the exact same defect. Out-of-corpus sweeping
# found 809/1500 (53.9%) non-bit-exact. Non-representable float64
# coefficients, degrees 1-7, powers 0-4 (0/1 are the identity/no-op
# shortcuts real numpy's own `_pow` takes, kept in the mix deliberately so
# this corpus doesn't only ever exercise the multiplying branch).
def _polypow_adversarial_cases():
    rng = np.random.default_rng(20260807 + 1)
    cases = []
    for i in range(60):
        n = int(rng.integers(2, 9))
        c = rng.uniform(-9.0, 9.0, size=n).tolist()
        p = int(rng.integers(0, 5))
        cases.append((f"adversarial_float_{i}", (c, p), {}))
    return cases


def polypow_cases():
    return _std([
        ("square", (C1, 2), {}),
        ("cube", (FLOAT_C, 3), {}),
        ("power_zero", (C1, 0), {}),
        ("power_one", (C1, 1), {}),
        ("complex_base", (COMPLEX_C, 2), {}),
        ("int_c", (INT_C, 2), {}),
    ]) + _polypow_adversarial_cases()


# ---------------------------------------------------------------------------
# der / int
# ---------------------------------------------------------------------------

def polyder_cases():
    C = [1.0, 2.0, 3.0, 4.0]
    return _std([
        ("first_order", (C, 1), {}),
        ("third_order", (C, 3), {}),
        ("zero_order", (C, 0), {}),
        ("order_exceeds_degree", (C, 10), {}),
        ("with_scale", (C, 1, 2.0), {}),
        ("complex_c", (COMPLEX_C, 1), {}),
        ("int_c", (INT_C, 1), {}),
        # polyder does its OWN dtype handling (`if c.dtype.char in
        # '?bBhHiIlLqQpP': c = c + 0.0`), NOT `as_series` -- bool is
        # promoted to float64, never rejected. Verified:
        # P.polyder([True, False, True]) -> array([0., 2.]), no error.
        ("bool_c", (BOOL_C, 1), {}),
        # 2026-08-07 (Monday): the `cnt >= n` (order-exceeds-degree)
        # branch derives its result via `c = c[:1] * 0` in real numpy, not
        # a literal zero fill -- for a negative first coefficient this is
        # IEEE754 `-0.0`. `order_exceeds_degree` above uses an
        # all-positive `C`, which cannot distinguish a correct derived
        # zero from a wrong literal one (harness.py's default "exact"
        # comparison is already bit/signbit-exact, so this is purely a
        # corpus coverage gap). Out-of-corpus sweep found 161/2000
        # signed-zero-only divergences before the `ionp-core/src/poly.rs`
        # `der` fix (literal `vec![T::zero()]` -> `vec![c[0].poly_mul(T::zero())]`).
        ("order_exceeds_degree_negative_real", ([-3.0, 2.0, 1.0], 10), {}),
        ("order_exceeds_degree_negative_complex", ([-1.5 - 2.5j, 1.0 + 1.0j], 5), {}),
        ("order_exceeds_degree_negative_imag", ([2.5 - 1.5j, 1.0], 5), {}),
    ])


def polyint_cases():
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
    ])


def polyint_k_too_long_cases():
    return _std([
        ("k_longer_than_m", ([1.0, 2.0], 1, [1.0, 2.0]), {}),
    ])


# ---------------------------------------------------------------------------
# vander / companion
# ---------------------------------------------------------------------------

def polyvander_cases():
    return _std([
        ("basic", (XS, 5), {}),
        ("deg_zero", (XS, 0), {}),
        ("scalar_x", (X_SCALAR, 3), {}),
        ("complex_x", ([1 + 1j, 2 - 1j, 0.5j], 4), {}),
        ("int_x", ([1, 2, 3], 3), {}),
        ("bool_x", ([True, False], 2), {}),  # own-dtype path: bool promoted, not rejected
    ])


def polycompanion_cases():
    return _std([
        ("degree2", (C1,), {}),
        ("degree1", (C1_SHORT,), {}),
        ("degree4", ([1.0, 0.0, -2.0, 0.0, 1.0],), {}),
        ("complex", (COMPLEX_C,), {}),
        ("int_c", (INT_C,), {}),
        ("single_coef_toosmall", (SINGLE_C,), {}),
        ("bool_c", (BOOL_C,), {}),  # as_series-family: rejects bool
    ])


# ---------------------------------------------------------------------------
# fromroots / roots
# ---------------------------------------------------------------------------

# 2026-08-07 (Monday): same "small-integer corpus can't see rounding
# divergence" gap as `_polymul_adversarial_cases` above -- `polyfromroots`
# is a balanced-pairing tree of `polymul` calls of linear `[-r, 1]`
# factors (see `anionpy/polynomial/polynomial.py`'s `polyfromroots`), so
# it inherits the same defect. Out-of-corpus sweeping found 419/1500
# (27.9%) non-bit-exact. Non-representable float64 roots, 1-7 roots per
# case.
def _polyfromroots_adversarial_cases():
    rng = np.random.default_rng(20260807 + 2)
    cases = []
    for i in range(60):
        n = int(rng.integers(1, 8))
        roots = rng.uniform(-9.0, 9.0, size=n).tolist()
        cases.append((f"adversarial_float_{i}", (roots,), {}))
    return cases


def polyfromroots_cases():
    return _std([
        ("real_roots", ([-1.0, 0.0, 1.0],), {}),
        ("no_roots", ([],), {}),
        ("single_root", ([3.0],), {}),
        ("complex_roots", ([-1j, 0.0, 1j],), {}),
        ("repeated_root", ([1.0, 1.0, -1.0],), {}),
        ("five_roots", ([-2.0, -1.0, 0.0, 1.0, 2.0],), {}),
        ("bool_roots", ([True, False],), {}),  # as_series-family: rejects bool
    ]) + _polyfromroots_adversarial_cases()


def polyroots_cases():
    return _std([
        ("real_roots", (np.polynomial.polynomial.polyfromroots([-1.0, 0.0, 1.0]),), {}),
        ("complex_roots", (np.polynomial.polynomial.polyfromroots([-1j, 0.0, 1j]),), {}),
        ("degree1", (C1_SHORT,), {}),
        ("degree0_no_roots", (SINGLE_C,), {}),
        ("five_real_roots", (np.polynomial.polynomial.polyfromroots(
            [-2.0, -1.0, 0.0, 1.0, 2.0]),), {}),
    ])


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def polyfit_cases():
    rng = np.random.default_rng(20260807)
    x = np.linspace(-1.0, 1.0, 25)
    y = x**3 - x
    y_noisy = y + rng.normal(scale=0.01, size=x.shape)
    y2d = np.stack([y, x**2], axis=1)
    return _std([
        ("cubic_exact", (x, y, 3), {}),
        ("cubic_noisy", (x, y_noisy, 3), {}),
        ("linear_underfit", (x, y, 1), {}),
        ("y_2d", (x, y2d, 3), {}),
        ("small_n", ([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0], 2), {}),
    ])


# Measured 2026-08-07 via /private/tmp/poly_eps_sweep.py, run OUT of this
# corpus (independent seeded sweeps, 20,000 samples each -- the
# MIN_ULP_SWEEP_N floor `ItemSpec.__post_init__` enforces on every
# epsilon_tolerance below). Real numpy's own `polyfit`/`_fit` goes through
# a scaled-lstsq path; `polyroots` goes through `linalg.eigvals` on the
# companion matrix -- neither is bit-identical to anionpy's independent
# call path by construction (different LAPACK/Rust SVD/eigensolver
# routines), so these are measured epsilon tolerances, not bugs:
#   polyfit float64:            max rel = 2.1612483045051664e-11 (N=20000)
#   polyroots float64 (real):   max abs = 8.310059886440513e-05  (N=20000)
#   polyroots complex128:       max abs = 1.834393877421201e-09  (N=20000)
# The polyroots float64 bound is visibly larger than polyfit's or
# polyroots' own complex128 bound: real-coefficient polynomials built from
# up to 7 independently-drawn real roots in [-5, 5] are, for a fraction of
# draws, genuinely ill-conditioned (Wilkinson-polynomial-style root
# sensitivity) -- an eigenvalue perturbation on the order of 1e-13 in the
# companion matrix entries can move a root by orders of magnitude more
# than that. This is a property of root-finding-via-companion-eigenvalues
# itself (numpy's own algorithm), not evidence anionpy's eigvals is worse;
# it is why a single, generously-measured absolute bound is the honest
# choice here rather than a tight bound that would flake.
POLYFIT_EPS = 2.1612483045051664e-11
POLYROOTS_REAL_EPS = 8.310059886440513e-05
POLYROOTS_COMPLEX_EPS = 1.834393877421201e-09
# `Polynomial.__truediv__`'s complex-scalar path (`polydiv` under the hood):
# a genuine, pre-existing 1-ULP-class divergence in the already-shipped,
# already-exact-declared module-level `polydiv`'s complex division, first
# surfaced by THIS item's corpus (the original 21-function corpus never
# exercised complex-scalar division specifically). Verified NOT an
# `_as_series1`/dunder-layer artifact: reproduces identically calling
# `anionpy.polynomial.polynomial.polydiv` directly with a 1-element complex
# array, bit-for-bit the same last-ULP disagreement as through the dunder.
# Bound is the max relative coefficient error over a 20,000-sample seeded
# sweep (degree 1-5 complex-coefficient polynomials / complex scalars,
# independent of poly_truediv_cases() above) -- see
# /private/tmp/poly_eps_sweep.py.
POLYDIV_COMPLEX_EPS = 5.801959274522398e-16
# `Polynomial.__mul__`/`__rmul__` (module-level `polymul`), `__pow__`
# (`polypow`, repeated `polymul`), `fromroots` (`polyfromroots`, repeated
# multiplication of linear factors), and `convert`/`cast` (`_compose_affine`
# in `_polybase.py`, a Horner-loop substitution that is itself
# multiplication-heavy) all measurably disagree with real numpy at the
# float64 last-few-ULP level for RANDOM (non-corpus) inputs -- summation-
# order divergence between the Rust convolution kernel and numpy's own,
# same underlying story as `POLYDIV_COMPLEX_EPS` above but for real
# multiplication instead of complex division. This is a PRE-EXISTING
# property of the already-shipped, already-"exact"-declared module-level
# `polymul` (verified identically reproducing calling
# `anionpy.polynomial.polynomial.polymul` directly, unrelated to anything
# added this session) -- the original 21-function corpus's small,
# hand-picked coefficient values never happened to expose it. Not
# fixable here (would require matching the Rust kernel's summation order
# bit-for-bit against numpy's own C loop; out of this file's scope and
# out of the "no Rust edits" constraint). Each bound below is the max
# relative coefficient error over its own 20,000-sample seeded sweep,
# independent of this file's corpus generators -- see
# /private/tmp/poly_eps_sweep.py.
POLYMUL_EPS = 9.143279708839359e-12
POLYPOW_EPS = 1.6190647066373651e-13
POLYFROMROOTS_EPS = 1.75779875021322e-11
POLYCONVERT_EPS = 1.1465959568111e-12


# ---------------------------------------------------------------------------
# Polynomial / ABCPolyBase (`polynomial.Polynomial.*`)
# ---------------------------------------------------------------------------
# `Polynomial` (anionpy/polynomial/polynomial.py) is assembly over the 21
# functions above via `ABCPolyBase` (anionpy/polynomial/_polybase.py) --
# see both files' module docstrings. Every adapter below constructs a
# REAL `numpy.polynomial.Polynomial` on the numpy side (this is a test
# file: calling real numpy for the reference value is the entire point of
# a differential harness, not the "never call real numpy" rule, which
# governs anionpy's own shipped code) and an `anionpy.polynomial.Polynomial`
# on the ionp side, calls the SAME method/dunder on each, and compares.
#
# Most of these methods/dunders return a Polynomial INSTANCE, not an
# array or plain scalar -- `_state()` below extracts a plain-Python,
# `==`-comparable, non-ambiguous-truth-value tuple (coef list, domain
# list, window list, symbol str) from an instance, and every such item is
# declared `scalar_like=True` (see registry.py's ItemSpec docstring: this
# is exactly the "not an ndarray" case that flag exists for, and a tuple
# of plain floats/strs is one of the safe scalar_like shapes the summary
# for this task flagged -- unlike a tuple containing a raw ndarray, `==`
# on a tuple of plain Python floats/complex/str never raises).

POLY_D1 = [0.0, 2.0]
POLY_D2 = [-2.0, 3.0]


def _state(p):
    """Plain-Python, exactly-`==`-comparable snapshot of a Polynomial
    instance: (coef list, domain list, window list, symbol str)."""
    coef = p.coef
    coef = coef.tolist() if hasattr(coef, "tolist") else list(coef)
    domain = p.domain
    domain = domain.tolist() if hasattr(domain, "tolist") else list(domain)
    window = p.window
    window = window.tolist() if hasattr(window, "tolist") else list(window)
    return (coef, domain, window, p.symbol)


def _np_poly(coef, domain=None, window=None, symbol="x"):
    kw = {"symbol": symbol}
    if domain is not None:
        kw["domain"] = domain
    if window is not None:
        kw["window"] = window
    return np.polynomial.Polynomial(coef, **kw)


def _ionp_poly(coef, domain=None, window=None, symbol="x"):
    kw = {"symbol": symbol}
    if domain is not None:
        kw["domain"] = domain
    if window is not None:
        kw["window"] = window
    return _anionpy.polynomial.Polynomial(coef, **kw)


# -- __init__ / construction --

def poly_init_cases():
    return _std([
        ("basic_float", (FLOAT_C,), {}),
        ("int_promote", (INT_C,), {}),
        ("complex", (COMPLEX_C,), {}),
        ("trailing_zero_not_trimmed", (TRAILING_ZERO_C,), {}),
        ("single", (SINGLE_C,), {}),
        ("custom_domain_window", (FLOAT_C,), {"domain": POLY_D1, "window": [-1.0, 1.0]}),
        ("custom_symbol", (FLOAT_C,), {"symbol": "z"}),
    ])


def _np_init(coef, **kw):
    return _state(np.polynomial.Polynomial(coef, **kw))


def _ionp_init(coef, **kw):
    return _state(_anionpy.polynomial.Polynomial(coef, **kw))


# -- domain / window / basis_name / symbol / maxpower (plain attributes) --

def poly_domain_cases():
    return _std([
        ("default", (FLOAT_C,), {}),
        ("custom", (FLOAT_C,), {"domain": POLY_D1}),
    ])


def _np_domain(coef, **kw):
    return np.polynomial.Polynomial(coef, **kw).domain


def _ionp_domain(coef, **kw):
    return _anionpy.polynomial.Polynomial(coef, **kw).domain


def poly_window_cases():
    return _std([
        ("default", (FLOAT_C,), {}),
        ("custom", (FLOAT_C,), {"window": [-2.0, 2.0]}),
    ])


def _np_window(coef, **kw):
    return np.polynomial.Polynomial(coef, **kw).window


def _ionp_window(coef, **kw):
    return _anionpy.polynomial.Polynomial(coef, **kw).window


def poly_basis_name_cases():
    return _std([("value", (FLOAT_C,), {})])


def _np_basis_name(coef):
    return np.polynomial.Polynomial(coef).basis_name


def _ionp_basis_name(coef):
    return _anionpy.polynomial.Polynomial(coef).basis_name


def poly_symbol_cases():
    return _std([
        ("default", (FLOAT_C,), {}),
        ("custom", (FLOAT_C,), {"symbol": "z"}),
    ])


def _np_symbol(coef, **kw):
    return np.polynomial.Polynomial(coef, **kw).symbol


def _ionp_symbol(coef, **kw):
    return _anionpy.polynomial.Polynomial(coef, **kw).symbol


def poly_maxpower_cases():
    return _std([("value", (), {})])


def _np_maxpower():
    return np.polynomial.Polynomial.maxpower


def _ionp_maxpower():
    return _anionpy.polynomial.Polynomial.maxpower


# -- __call__ --

def poly_call_cases():
    cases = []
    for c_label, c in [("float_c", FLOAT_C), ("int_c", INT_C), ("complex_c", COMPLEX_C)]:
        cases.append((f"scalar__{c_label}", (c, X_SCALAR), {}))
        cases.append((f"list__{c_label}", (c, XS), {}))
        cases.append((f"complex_x__{c_label}", (c, X_COMPLEX), {}))
    cases.append(("custom_domain", (FLOAT_C, XS), {"domain": POLY_D1}))
    return cases


def _np_call(coef, x, **kw):
    return np.polynomial.Polynomial(coef, **kw)(x)


def _ionp_call(coef, x, **kw):
    return _anionpy.polynomial.Polynomial(coef, **kw)(x)


# -- __iter__ / __len__ --

def poly_iter_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("complex_c", (COMPLEX_C,), {}),
        ("single", (SINGLE_C,), {}),
    ])


def _np_iter(coef):
    return np.array(list(iter(np.polynomial.Polynomial(coef))))


def _ionp_iter(coef):
    return _anionpy.array(list(iter(_anionpy.polynomial.Polynomial(coef))))


def poly_len_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("trailing_zero", (TRAILING_ZERO_C,), {}),
        ("single", (SINGLE_C,), {}),
    ])


def _np_len(coef):
    return len(np.polynomial.Polynomial(coef))


def _ionp_len(coef):
    return len(_anionpy.polynomial.Polynomial(coef))


# -- __hash__ (unhashable -- exercised purely for the raised exception) --

def poly_hash_cases():
    return _std([("value", (FLOAT_C,), {})])


def _np_hash(coef):
    return hash(np.polynomial.Polynomial(coef))


def _ionp_hash(coef):
    return hash(_anionpy.polynomial.Polynomial(coef))


# -- __array_ufunc__ (blocked -- exercised via the TypeError np.add raises) --

def poly_array_ufunc_cases():
    return _std([("value", (FLOAT_C, C2), {})])


def _np_array_ufunc(c1, c2):
    return np.add(np.polynomial.Polynomial(c1), np.polynomial.Polynomial(c2))


def _ionp_array_ufunc(c1, c2):
    return np.add(_anionpy.polynomial.Polynomial(c1), _anionpy.polynomial.Polynomial(c2))


# -- __eq__ / __ne__ --

def poly_eq_cases():
    return _std([
        ("equal", (FLOAT_C, FLOAT_C), {}),
        ("different_coef", (FLOAT_C, C2), {}),
        ("different_len", (FLOAT_C, C1_SHORT), {}),
        ("different_domain", (FLOAT_C, FLOAT_C), {"domain2": POLY_D1}),
        ("different_symbol", (FLOAT_C, FLOAT_C), {"symbol2": "z"}),
        ("not_a_polynomial", (FLOAT_C, None), {}),
    ])


def _np_eq(c1, c2, domain2=None, symbol2="x"):
    p = np.polynomial.Polynomial(c1)
    q = c2 if c2 is None else np.polynomial.Polynomial(c2, domain=domain2, symbol=symbol2)
    return p == q


def _ionp_eq(c1, c2, domain2=None, symbol2="x"):
    p = _anionpy.polynomial.Polynomial(c1)
    q = c2 if c2 is None else _anionpy.polynomial.Polynomial(c2, domain=domain2, symbol=symbol2)
    return p == q


def _np_ne(c1, c2, domain2=None, symbol2="x"):
    p = np.polynomial.Polynomial(c1)
    q = c2 if c2 is None else np.polynomial.Polynomial(c2, domain=domain2, symbol=symbol2)
    return p != q


def _ionp_ne(c1, c2, domain2=None, symbol2="x"):
    p = _anionpy.polynomial.Polynomial(c1)
    q = c2 if c2 is None else _anionpy.polynomial.Polynomial(c2, domain=domain2, symbol=symbol2)
    return p != q


# -- arithmetic dunders: __add__/__radd__/__sub__/__rsub__/__mul__/__rmul__/
#    __truediv__/__rtruediv__/__floordiv__/__rfloordiv__/__mod__/__rmod__/
#    __divmod__/__rdivmod__/__pow__/__neg__/__pos__ --

def poly_binop_cases():
    return _std([
        ("same_len", (C1, C2), {}),
        ("c1_shorter", (C1_SHORT, C1), {}),
        ("int_and_float", (INT_C, FLOAT_C), {}),
        ("complex_and_real", (COMPLEX_C, FLOAT_C), {}),
        ("scalar_other", (C1, 2.0), {}),
    ])


def poly_truediv_cases():
    # `__truediv__` (unlike every other binop here) is restricted by real
    # numpy to `numbers.Number` rhs ONLY -- a Polynomial rhs raises
    # `TypeError` on both sides, but that message embeds `type(self)`/
    # `type(other)`, which differ in TEXT purely because of the
    # `numpy.polynomial.polynomial.Polynomial` vs
    # `anionpy.polynomial.polynomial.Polynomial` module-path difference --
    # an inherent, unfixable-by-us divergence (matching it would mean
    # impersonating numpy's `__module__`), not a logic bug. So this item's
    # corpus is deliberately narrower than `poly_binop_cases` (used by the
    # other binops): scalar rhs only, so it actually exercises the real
    # division path instead of just re-proving the module-path text
    # divergence on every case.
    return _std([
        ("float_scalar", (C1, 2.0), {}),
        ("int_scalar", (C1_SHORT, 3), {}),
        ("complex_scalar", (FLOAT_C, 1.0 + 2.0j), {}),
    ])


def _np_truediv(c1, c2):
    return (np.polynomial.Polynomial(c1) / c2).coef


def _ionp_truediv(c1, c2):
    return (_anionpy.polynomial.Polynomial(c1) / c2).coef


def _mk_binop(name, ionp=False):
    ctor = _anionpy.polynomial.Polynomial if ionp else np.polynomial.Polynomial

    def adapter(c1, c2):
        p = ctor(c1)
        other = c2 if isinstance(c2, (int, float, complex)) else ctor(c2)
        result = getattr(p, name)(other)
        if isinstance(result, tuple):
            return tuple(_state(r) for r in result)
        return _state(result)
    return adapter


def poly_rbinop_cases():
    # `2.0 + p` / `2.0 * p` etc.: exercises the reflected dunders with a
    # plain Python scalar on the left, which is the only way Python's
    # operator protocol ever actually reaches `__radd__`/`__rmul__`/etc.
    # (a Polynomial-vs-Polynomial `p + q` always resolves via `__add__`).
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("int_c", (INT_C,), {}),
        ("complex_c", (COMPLEX_C,), {}),
    ])


def _mk_rbinop(name, ionp=False):
    ctor = _anionpy.polynomial.Polynomial if ionp else np.polynomial.Polynomial

    def adapter(c):
        p = ctor(c)
        result = getattr(p, name)(2.0)
        if isinstance(result, tuple):
            return tuple(_state(r) for r in result)
        return _state(result)
    return adapter


def poly_pow_cases():
    return _std([
        ("square", (C1, 2), {}),
        ("cube", (FLOAT_C, 3), {}),
        ("power_zero", (C1, 0), {}),
        ("power_one", (C1, 1), {}),
    ])


# `.coef`-only, epsilon-tolerant: see `POLYPOW_EPS`'s comment block above
# for why `__pow__` cannot be graded bit-exact like most other dunders
# here (measurably NOT bit-exact against real numpy for random,
# out-of-corpus inputs -- caught by out-of-corpus sweeping, not by this
# item's own small fixed-value corpus, which happens to pass exactly).
# `domain`/`window`/`symbol` construction on this path is the same exact,
# already-covered-elsewhere code regardless of the power's rounding, same
# "scope to the one genuinely inexact piece" reasoning as `fit`/
# `__truediv__` above.
def _np_pow(c, n):
    return (np.polynomial.Polynomial(c) ** n).coef


def _ionp_pow(c, n):
    return (_anionpy.polynomial.Polynomial(c) ** n).coef


def poly_unary_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("complex_c", (COMPLEX_C,), {}),
    ])


def _np_neg(c):
    return _state(-np.polynomial.Polynomial(c))


def _ionp_neg(c):
    return _state(-_anionpy.polynomial.Polynomial(c))


def _np_pos(c):
    return _state(+np.polynomial.Polynomial(c))


def _ionp_pos(c):
    return _state(+_anionpy.polynomial.Polynomial(c))


# -- __getstate__ / __setstate__ --

def poly_getstate_cases():
    return _std([
        ("basic", (FLOAT_C,), {}),
        ("custom_domain", (FLOAT_C,), {"domain": POLY_D1, "symbol": "z"}),
    ])


def _state_dict(d):
    return {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in sorted(d.items())
    }


def _np_getstate(coef, **kw):
    return _state_dict(np.polynomial.Polynomial(coef, **kw).__getstate__())


def _ionp_getstate(coef, **kw):
    return _state_dict(_anionpy.polynomial.Polynomial(coef, **kw).__getstate__())


def poly_setstate_cases():
    return _std([
        ("basic", (FLOAT_C,), {}),
        ("custom_domain", (FLOAT_C,), {"domain": POLY_D1, "symbol": "z"}),
    ])


def _np_setstate(coef, **kw):
    src = np.polynomial.Polynomial(coef, **kw)
    dst = np.polynomial.Polynomial.__new__(np.polynomial.Polynomial)
    dst.__setstate__(src.__getstate__())
    return _state(dst)


def _ionp_setstate(coef, **kw):
    src = _anionpy.polynomial.Polynomial(coef, **kw)
    dst = _anionpy.polynomial.Polynomial.__new__(_anionpy.polynomial.Polynomial)
    dst.__setstate__(src.__getstate__())
    return _state(dst)


# -- copy / degree / cutdeg / trim / truncate / mapparms --

def poly_copy_cases():
    return _std([("basic", (FLOAT_C,), {"domain": POLY_D1, "symbol": "z"})])


def _np_copy(coef, **kw):
    return _state(np.polynomial.Polynomial(coef, **kw).copy())


def _ionp_copy(coef, **kw):
    return _state(_anionpy.polynomial.Polynomial(coef, **kw).copy())


def poly_degree_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("trailing_zero", (TRAILING_ZERO_C,), {}),
        ("single", (SINGLE_C,), {}),
    ])


def _np_degree(coef):
    return np.polynomial.Polynomial(coef).degree()


def _ionp_degree(coef):
    return _anionpy.polynomial.Polynomial(coef).degree()


def poly_cutdeg_cases():
    C = [1.0, 2.0, 3.0, 4.0, 5.0]
    return _std([
        ("reduce", (C, 2), {}),
        ("no_change", (C, 10), {}),
        ("to_zero", (C, 0), {}),
    ])


def _np_cutdeg(coef, deg):
    return _state(np.polynomial.Polynomial(coef).cutdeg(deg))


def _ionp_cutdeg(coef, deg):
    return _state(_anionpy.polynomial.Polynomial(coef).cutdeg(deg))


def poly_trim_cases():
    return _std([
        ("trailing_zero", (TRAILING_ZERO_C,), {}),
        ("no_trim_needed", (FLOAT_C,), {}),
        ("with_tol", (list(TRAILING_ZERO_C) + [1e-10], 1e-8), {}),
    ])


def _np_trim(coef, tol=0):
    return _state(np.polynomial.Polynomial(coef).trim(tol))


def _ionp_trim(coef, tol=0):
    return _state(_anionpy.polynomial.Polynomial(coef).trim(tol))


def poly_truncate_cases():
    C = [1.0, 2.0, 3.0, 4.0, 5.0]
    return _std([
        ("shrink", (C, 3), {}),
        ("no_change", (C, 10), {}),
        ("to_one", (C, 1), {}),
    ])


def _np_truncate(coef, size):
    return _state(np.polynomial.Polynomial(coef).truncate(size))


def _ionp_truncate(coef, size):
    return _state(_anionpy.polynomial.Polynomial(coef).truncate(size))


def poly_mapparms_cases():
    return _std([
        ("default", (FLOAT_C,), {}),
        ("custom_domain", (FLOAT_C,), {"domain": POLY_D1}),
    ])


def _np_mapparms(coef, **kw):
    off, scl = np.polynomial.Polynomial(coef, **kw).mapparms()
    return (float(off), float(scl))


def _ionp_mapparms(coef, **kw):
    off, scl = _anionpy.polynomial.Polynomial(coef, **kw).mapparms()
    return (float(off), float(scl))


# -- convert / cast --

def poly_convert_cases():
    return _std([
        ("default_to_default", (FLOAT_C,), {}),
        ("to_custom_domain", (FLOAT_C,), {"target_domain": POLY_D1}),
        ("from_custom_to_default", (FLOAT_C,), {"src_domain": POLY_D1}),
        ("from_custom_to_custom", (FLOAT_C,), {"src_domain": POLY_D1, "target_domain": POLY_D2}),
    ])


# `.coef`-only, epsilon-tolerant: `convert`'s underlying `_compose_affine`
# (`_polybase.py`) is a Horner-loop substitution -- multiplication-heavy,
# same POLYMUL-class "not proven bit-identical to numpy's own" story, and
# measurably diverges at the float64 last-few-ULP level for random
# (non-corpus) domain remaps. See `POLYCONVERT_EPS`'s comment block above.
# `domain`/`window`/`symbol` construction on this path is the same exact,
# already-covered-elsewhere code (`mapparms`, `has_samedomain`, etc.)
# regardless of the coefficient rounding.
def _np_convert(coef, src_domain=None, target_domain=None):
    p = np.polynomial.Polynomial(coef, domain=src_domain)
    kw = {} if target_domain is None else {"domain": target_domain}
    return p.convert(**kw).coef


def _ionp_convert(coef, src_domain=None, target_domain=None):
    p = _anionpy.polynomial.Polynomial(coef, domain=src_domain)
    kw = {} if target_domain is None else {"domain": target_domain}
    return p.convert(**kw).coef


def poly_cast_cases():
    return _std([
        ("default_to_default", (FLOAT_C,), {}),
        ("from_custom", (FLOAT_C,), {"src_domain": POLY_D1}),
    ])


# `.coef`-only, epsilon-tolerant: `cast` = `convert` to `cls`'s own
# default domain/window (see `ABCPolyBase.cast`), same
# `_compose_affine`-driven divergence as `convert` above -- same
# `POLYCONVERT_EPS` bound reused, not a fresh sweep, since it is the
# identical affine-substitution code path.
def _np_cast(coef, src_domain=None):
    p = np.polynomial.Polynomial(coef, domain=src_domain)
    return np.polynomial.Polynomial.cast(p).coef


def _ionp_cast(coef, src_domain=None):
    p = _anionpy.polynomial.Polynomial(coef, domain=src_domain)
    return _anionpy.polynomial.Polynomial.cast(p).coef


# -- has_samecoef / has_samedomain / has_samewindow / has_sametype --

def poly_has_same_cases():
    return _std([
        ("identical", (FLOAT_C, FLOAT_C, None, None), {}),
        ("different_coef", (FLOAT_C, C2, None, None), {}),
        ("different_domain", (FLOAT_C, FLOAT_C, POLY_D1, None), {}),
        ("different_window", (FLOAT_C, FLOAT_C, None, [-2.0, 2.0]), {}),
    ])


def _mk_has_same(name, ionp=False):
    ctor = _anionpy.polynomial.Polynomial if ionp else np.polynomial.Polynomial

    def adapter(c1, c2, domain2, window2):
        p = ctor(c1)
        q = ctor(c2, domain=domain2, window=window2)
        return getattr(p, name)(q)
    return adapter


# -- integ / deriv --

def poly_integ_cases():
    C = [1.0, 2.0, 3.0]
    return _std([
        ("default", (C,), {}),
        ("with_k", (C,), {"m": 1, "k": [1.0]}),
        ("custom_domain", (C,), {"domain": POLY_D1}),
        ("complex_c", (COMPLEX_C,), {}),
    ])


def _np_integ(coef, m=1, k=[], **kw):
    return _state(np.polynomial.Polynomial(coef, **kw).integ(m, k))


def _ionp_integ(coef, m=1, k=[], **kw):
    return _state(_anionpy.polynomial.Polynomial(coef, **kw).integ(m, k))


def poly_deriv_cases():
    C = [1.0, 2.0, 3.0, 4.0]
    return _std([
        ("default", (C,), {}),
        ("second_order", (C,), {"m": 2}),
        ("custom_domain", (C,), {"domain": POLY_D1}),
        ("complex_c", (COMPLEX_C,), {}),
    ])


def _np_deriv(coef, m=1, **kw):
    return _state(np.polynomial.Polynomial(coef, **kw).deriv(m))


def _ionp_deriv(coef, m=1, **kw):
    return _state(_anionpy.polynomial.Polynomial(coef, **kw).deriv(m))


# -- roots / fromroots / identity / basis / linspace / fit --

def poly_roots_cases():
    return _std([
        ("real_roots", (np.polynomial.polynomial.polyfromroots([-1.0, 0.0, 1.0]),), {}),
        ("complex_roots", (np.polynomial.polynomial.polyfromroots([-1j, 0.0, 1j]),), {}),
        ("degree1", (C1_SHORT,), {}),
        ("custom_domain", ([1.0, -1.0, -1.0, 1.0],), {"domain": POLY_D1}),
    ])


def _np_roots(coef, **kw):
    return np.polynomial.Polynomial(coef, **kw).roots()


def _ionp_roots(coef, **kw):
    return _anionpy.polynomial.Polynomial(coef, **kw).roots()


def poly_fromroots_cases():
    return _std([
        ("real_roots", ([-1.0, 0.0, 1.0],), {}),
        ("no_roots", ([],), {}),
        ("complex_roots", ([-1j, 0.0, 1j],), {}),
        ("custom_domain", ([-1.0, 0.0, 1.0],), {"domain": POLY_D1}),
    ])


# `.coef`-only, epsilon-tolerant: `fromroots` = `polyfromroots` (repeated
# multiplication of linear `(x - root)` factors) + domain/window wiring.
# See `POLYFROMROOTS_EPS`'s comment block above -- same pre-existing
# `polymul`-class divergence, not something new here. `domain`/`window`/
# `symbol` construction is the same exact, already-covered-elsewhere code
# (`identity`, `basis` above are both bit-exact) regardless of the
# coefficient rounding.
def _np_fromroots(roots, **kw):
    return np.polynomial.Polynomial.fromroots(roots, **kw).coef


def _ionp_fromroots(roots, **kw):
    return _anionpy.polynomial.Polynomial.fromroots(roots, **kw).coef


def poly_identity_cases():
    return _std([
        ("default", (), {}),
        ("custom_domain", (), {"domain": POLY_D1}),
        ("custom_symbol", (), {"symbol": "z"}),
    ])


def _np_identity(**kw):
    return _state(np.polynomial.Polynomial.identity(**kw))


def _ionp_identity(**kw):
    return _state(_anionpy.polynomial.Polynomial.identity(**kw))


def poly_basis_cases():
    return _std([
        ("deg0", (0,), {}),
        ("deg3", (3,), {}),
        ("custom_domain", (2,), {"domain": POLY_D1}),
    ])


def _np_basis(deg, **kw):
    return _state(np.polynomial.Polynomial.basis(deg, **kw))


def _ionp_basis(deg, **kw):
    return _state(_anionpy.polynomial.Polynomial.basis(deg, **kw))


def poly_linspace_cases():
    return _std([
        ("default", (FLOAT_C,), {}),
        ("small_n", (FLOAT_C,), {"n": 5}),
        ("custom_domain_arg", (FLOAT_C,), {"lin_domain": POLY_D1}),
    ])


def _np_linspace(coef, n=100, lin_domain=None):
    x, y = np.polynomial.Polynomial(coef).linspace(n, domain=lin_domain)
    return (x, y)


def _ionp_linspace(coef, n=100, lin_domain=None):
    x, y = _anionpy.polynomial.Polynomial(coef).linspace(n, domain=lin_domain)
    return (x, y)


def poly_fit_cases():
    rng = np.random.default_rng(20260807)
    x = np.linspace(-1.0, 1.0, 25)
    y = x**3 - x
    y_noisy = y + rng.normal(scale=0.01, size=x.shape)
    return _std([
        ("cubic_exact", (x, y, 3), {}),
        ("cubic_noisy", (x, y_noisy, 3), {}),
        ("linear_underfit", (x, y, 1), {}),
        ("explicit_domain", (x, y, 3), {"domain": POLY_D1}),
    ])


# Only `.coef` is graded (with the same measured epsilon as top-level
# `polyfit` -- `.fit()` IS `polyfit` plus deterministic, exact-arithmetic
# domain bookkeeping around it, see `ABCPolyBase.fit` in _polybase.py: the
# `domain is None` default path is plain float min/max, not a LAPACK
# call). `.domain`/`.window`/`.symbol` assignment on the classmethod path
# is the same exact code every other classmethod here (`identity`/`basis`/
# `fromroots`) already covers bit-exact against real numpy, so scoping
# this item to the one genuinely inexact piece (the fitted coefficients)
# avoids forcing an epsilon-tolerant array comparison to also silently
# swallow a real domain-wiring bug it isn't designed to catch.
def _np_fit(x, y, deg, **kw):
    return np.polynomial.Polynomial.fit(x, y, deg, **kw).coef


def _ionp_fit(x, y, deg, **kw):
    return _anionpy.polynomial.Polynomial.fit(x, y, deg, **kw).coef


POLY_EPS = 1e-9  # generous shared bound for the assembly-layer items
# below whose float path runs through an already-epsilon-tolerant
# building block (polyfit's lstsq or polyroots' eigvals) PLUS extra
# domain/window mapping arithmetic on top -- see the per-item
# epsilon_tolerance_justification at each spec for which block and why.


# ---------------------------------------------------------------------------
# Task #34 (2026-08-08): polyval2d/polyval3d/polyvalnd/polygrid2d/
# polygrid3d/polyvander2d/polyvander3d -- built on polyutils.py's shared
# _valnd/_gridnd/_vander_nd/_vander_nd_flat machinery (see that module's
# docstring for the ticket-premise correction: FOUR shared helpers back
# this family, not the two -- _vander_nd/_vander_nd_flat -- the ticket
# named). Corpus varies point count (1-4 per axis), degree (0-2 per axis),
# and dtype (float64, complex128, int -- promoted via _common_dtype).
# float16/float32/complex64 are DELIBERATELY EXCLUDED: the already-
# declared, already-shipped 1-D `polyval`/`polyvander` kernels this family
# is built on universally upcast those three dtypes to float64/complex128
# (values match real numpy bit-exact, but dtype does not) -- a
# pre-existing divergence in code this ticket did not write and has no
# permission to touch (lives in the shared Rust kernels, confirmed present
# for all six bases, not introduced here). See docs/TICKET-34-*.md.

def _poly_nd_base_cases():
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


def polyval2d_cases():
    return [(lbl, (xs, ys, c2), {}) for lbl, xs, ys, zs, c2, c3, deg in _poly_nd_base_cases()]


def polyval3d_cases():
    return [(lbl, (xs, ys, zs, c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _poly_nd_base_cases()]


def polyvalnd_cases():
    return [(lbl, ((xs, ys, zs), c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _poly_nd_base_cases()]


def polygrid2d_cases():
    return [(lbl, (xs, ys, c2), {}) for lbl, xs, ys, zs, c2, c3, deg in _poly_nd_base_cases()]


def polygrid3d_cases():
    return [(lbl, (xs, ys, zs, c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _poly_nd_base_cases()]


def polyvander2d_cases():
    return [(lbl, (xs, ys, deg[:2]), {}) for lbl, xs, ys, zs, c2, c3, deg in _poly_nd_base_cases()]


def polyvander3d_cases():
    return [(lbl, (xs, ys, zs, deg), {}) for lbl, xs, ys, zs, c2, c3, deg in _poly_nd_base_cases()]


# ---------------------------------------------------------------------------
# assembling the spec dict
# ---------------------------------------------------------------------------

def _build_polynomial_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}

    specs["polynomial.polynomial.polydomain"] = ItemSpec(
        name="polynomial.polynomial.polydomain", kind="custom",
        custom_cases=_const_cases("polydomain"),
        numpy_adapter=lambda: np.polynomial.polynomial.polydomain,
        ionp_adapter=lambda: __import__("anionpy").polynomial.polynomial.polydomain,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polyzero"] = ItemSpec(
        name="polynomial.polynomial.polyzero", kind="custom",
        custom_cases=_const_cases("polyzero"),
        numpy_adapter=lambda: np.polynomial.polynomial.polyzero,
        ionp_adapter=lambda: __import__("anionpy").polynomial.polynomial.polyzero,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polyone"] = ItemSpec(
        name="polynomial.polynomial.polyone", kind="custom",
        custom_cases=_const_cases("polyone"),
        numpy_adapter=lambda: np.polynomial.polynomial.polyone,
        ionp_adapter=lambda: __import__("anionpy").polynomial.polynomial.polyone,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polyx"] = ItemSpec(
        name="polynomial.polynomial.polyx", kind="custom",
        custom_cases=_const_cases("polyx"),
        numpy_adapter=lambda: np.polynomial.polynomial.polyx,
        ionp_adapter=lambda: __import__("anionpy").polynomial.polynomial.polyx,
        atol=0.0, rtol=0.0,
    )

    specs["polynomial.polynomial.polyline"] = ItemSpec(
        name="polynomial.polynomial.polyline", kind="custom",
        custom_cases=polyline_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polytrim"] = ItemSpec(
        name="polynomial.polynomial.polytrim", kind="custom",
        custom_cases=polytrim_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polyval"] = ItemSpec(
        name="polynomial.polynomial.polyval", kind="custom",
        custom_cases=polyval_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polyvalfromroots"] = ItemSpec(
        name="polynomial.polynomial.polyvalfromroots", kind="custom",
        custom_cases=polyvalfromroots_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polyadd"] = ItemSpec(
        name="polynomial.polynomial.polyadd", kind="custom",
        custom_cases=polyadd_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polysub"] = ItemSpec(
        name="polynomial.polynomial.polysub", kind="custom",
        custom_cases=polysub_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polymulx"] = ItemSpec(
        name="polynomial.polynomial.polymulx", kind="custom",
        custom_cases=polymulx_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polymul"] = ItemSpec(
        name="polynomial.polynomial.polymul", kind="custom",
        custom_cases=polymul_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polydiv"] = ItemSpec(
        name="polynomial.polynomial.polydiv", kind="custom",
        custom_cases=polydiv_cases, atol=0.0, rtol=0.0, multi_output=True,
    )
    specs["polynomial.polynomial.polypow"] = ItemSpec(
        name="polynomial.polynomial.polypow", kind="custom",
        custom_cases=polypow_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polyder"] = ItemSpec(
        name="polynomial.polynomial.polyder", kind="custom",
        custom_cases=polyder_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polyint"] = ItemSpec(
        name="polynomial.polynomial.polyint", kind="custom",
        custom_cases=polyint_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polyvander"] = ItemSpec(
        name="polynomial.polynomial.polyvander", kind="custom",
        custom_cases=polyvander_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polycompanion"] = ItemSpec(
        name="polynomial.polynomial.polycompanion", kind="custom",
        custom_cases=polycompanion_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polyfromroots"] = ItemSpec(
        name="polynomial.polynomial.polyfromroots", kind="custom",
        custom_cases=polyfromroots_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.polynomial.polyroots"] = ItemSpec(
        name="polynomial.polynomial.polyroots", kind="custom",
        custom_cases=polyroots_cases, atol=0.0, rtol=0.0,
        # eigenvalues of the companion matrix: same "independent LAPACK/
        # Rust eigensolver call path, not bit-identical" story as
        # linalg.eigvals itself. See KNOWN-DIFFERENCES.md for the measured
        # bound this epsilon is set from.
        epsilon_tolerance={
            "float64": ("abs", POLYROOTS_REAL_EPS),
            "complex128": ("abs", POLYROOTS_COMPLEX_EPS),
        },
        epsilon_tolerance_justification=(
            "polyroots = polycompanion (exact) + linalg.eigvals (independent "
            "eigensolver call path vs real numpy's LAPACK dgeev/zgeev, "
            "non-bit-identical by construction) + sort + real-cast. Bound is "
            "the max abs coordinate error over a 20,000-sample seeded sweep "
            "(degree 1-7 polynomials built from independently-drawn roots), "
            "independent of polyroots_cases() above -- see "
            "/private/tmp/poly_eps_sweep.py and KNOWN-DIFFERENCES.md. The "
            "float64 bound is large relative to complex128's because "
            "real-rooted polynomials at these degrees are occasionally "
            "genuinely ill-conditioned (Wilkinson-polynomial-style root "
            "sensitivity), not because the eigensolver disagrees more."
        ),
        epsilon_sweep={
            "float64": (20000, POLYROOTS_REAL_EPS),
            "complex128": (20000, POLYROOTS_COMPLEX_EPS),
        },
    )
    specs["polynomial.polynomial.polyfit"] = ItemSpec(
        name="polynomial.polynomial.polyfit", kind="custom",
        custom_cases=polyfit_cases,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"float64": ("rel", POLYFIT_EPS)},
        epsilon_tolerance_justification=(
            "Least-squares fit: normalized-Vandermonde build (exact, Rust) + "
            "linalg.lstsq (independent SVD call path vs real numpy, not "
            "bit-identical) + un-normalize. Bound is the max relative "
            "coefficient disagreement over a 20,000-fit seeded sweep, "
            "independent of polyfit_cases() -- see "
            "/private/tmp/poly_eps_sweep.py and KNOWN-DIFFERENCES.md."
        ),
        epsilon_sweep={"float64": (20000, POLYFIT_EPS)},
    )

    # -- Polynomial / ABCPolyBase --

    specs["polynomial.Polynomial.__init__"] = ItemSpec(
        name="polynomial.Polynomial.__init__", kind="custom",
        custom_cases=poly_init_cases,
        numpy_adapter=_np_init, ionp_adapter=_ionp_init, scalar_like=True,
    )
    specs["polynomial.Polynomial.domain"] = ItemSpec(
        name="polynomial.Polynomial.domain", kind="custom",
        custom_cases=poly_domain_cases,
        numpy_adapter=_np_domain, ionp_adapter=_ionp_domain,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.Polynomial.window"] = ItemSpec(
        name="polynomial.Polynomial.window", kind="custom",
        custom_cases=poly_window_cases,
        numpy_adapter=_np_window, ionp_adapter=_ionp_window,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.Polynomial.basis_name"] = ItemSpec(
        name="polynomial.Polynomial.basis_name", kind="custom",
        custom_cases=poly_basis_name_cases,
        numpy_adapter=_np_basis_name, ionp_adapter=_ionp_basis_name,
        scalar_like=True,
    )
    specs["polynomial.Polynomial.symbol"] = ItemSpec(
        name="polynomial.Polynomial.symbol", kind="custom",
        custom_cases=poly_symbol_cases,
        numpy_adapter=_np_symbol, ionp_adapter=_ionp_symbol, scalar_like=True,
    )
    specs["polynomial.Polynomial.maxpower"] = ItemSpec(
        name="polynomial.Polynomial.maxpower", kind="custom",
        custom_cases=poly_maxpower_cases,
        numpy_adapter=_np_maxpower, ionp_adapter=_ionp_maxpower, scalar_like=True,
    )
    specs["polynomial.Polynomial.__call__"] = ItemSpec(
        name="polynomial.Polynomial.__call__", kind="custom",
        custom_cases=poly_call_cases,
        numpy_adapter=_np_call, ionp_adapter=_ionp_call, atol=0.0, rtol=0.0,
    )
    specs["polynomial.Polynomial.__iter__"] = ItemSpec(
        name="polynomial.Polynomial.__iter__", kind="custom",
        custom_cases=poly_iter_cases,
        numpy_adapter=_np_iter, ionp_adapter=_ionp_iter, atol=0.0, rtol=0.0,
    )
    specs["polynomial.Polynomial.__len__"] = ItemSpec(
        name="polynomial.Polynomial.__len__", kind="custom",
        custom_cases=poly_len_cases,
        numpy_adapter=_np_len, ionp_adapter=_ionp_len, scalar_like=True,
    )
    specs["polynomial.Polynomial.__hash__"] = ItemSpec(
        name="polynomial.Polynomial.__hash__", kind="custom",
        custom_cases=poly_hash_cases,
        numpy_adapter=_np_hash, ionp_adapter=_ionp_hash, scalar_like=True,
    )
    specs["polynomial.Polynomial.__array_ufunc__"] = ItemSpec(
        name="polynomial.Polynomial.__array_ufunc__", kind="custom",
        custom_cases=poly_array_ufunc_cases,
        numpy_adapter=_np_array_ufunc, ionp_adapter=_ionp_array_ufunc,
        scalar_like=True,
    )
    specs["polynomial.Polynomial.__eq__"] = ItemSpec(
        name="polynomial.Polynomial.__eq__", kind="custom",
        custom_cases=poly_eq_cases,
        numpy_adapter=_np_eq, ionp_adapter=_ionp_eq, scalar_like=True,
    )
    specs["polynomial.Polynomial.__ne__"] = ItemSpec(
        name="polynomial.Polynomial.__ne__", kind="custom",
        custom_cases=poly_eq_cases,
        numpy_adapter=_np_ne, ionp_adapter=_ionp_ne, scalar_like=True,
    )

    for _op in ("__add__", "__sub__",
                "__floordiv__", "__mod__", "__divmod__"):
        specs[f"polynomial.Polynomial.{_op}"] = ItemSpec(
            name=f"polynomial.Polynomial.{_op}", kind="custom",
            custom_cases=poly_binop_cases,
            numpy_adapter=_mk_binop(_op, ionp=False),
            ionp_adapter=_mk_binop(_op, ionp=True),
            scalar_like=True,
        )
    # `__mul__`/`__rmul__` are NOT in the exact loops above: `polymul`
    # (the underlying kernel) measurably disagrees with real numpy at the
    # float64 last-few-ULP level for random inputs -- see `POLYMUL_EPS`'s
    # comment block above. This item's own small fixed-value corpus
    # happens to pass bit-exact (caught only by out-of-corpus sweeping),
    # so it is graded here with a measured epsilon instead, `.coef`-only
    # (same "scope to the one genuinely inexact piece" reasoning used
    # throughout this file).
    def _np_mul_coef(c1, c2):
        p = np.polynomial.Polynomial(c1)
        other = c2 if isinstance(c2, (int, float, complex)) else np.polynomial.Polynomial(c2)
        return p.__mul__(other).coef

    def _ionp_mul_coef(c1, c2):
        p = _anionpy.polynomial.Polynomial(c1)
        other = c2 if isinstance(c2, (int, float, complex)) else _anionpy.polynomial.Polynomial(c2)
        return p.__mul__(other).coef

    specs["polynomial.Polynomial.__mul__"] = ItemSpec(
        name="polynomial.Polynomial.__mul__", kind="custom",
        custom_cases=poly_binop_cases,
        numpy_adapter=_np_mul_coef, ionp_adapter=_ionp_mul_coef,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"float64": ("rel", POLYMUL_EPS), "complex128": ("rel", POLYMUL_EPS)},
        epsilon_tolerance_justification=(
            "polymul (Rust kernel) vs real numpy's own convolution: not "
            "proven bit-identical, measurably diverges at the last few "
            "ULPs for random (non-corpus) inputs. See POLYMUL_EPS's "
            "comment block above for the measured bound and sweep. "
            "complex128 was independently swept too (20,000 complex-"
            "coefficient samples, max relative error "
            "1.173223634621268e-14) -- well inside POLYMUL_EPS, so the "
            "same bound is reused rather than declaring a separate, "
            "tighter one that would only invite a future flake."
        ),
        epsilon_sweep={"float64": (20000, POLYMUL_EPS), "complex128": (20000, POLYMUL_EPS)},
    )
    # `__truediv__` uses its own narrower, scalar-only corpus -- see
    # `poly_truediv_cases`'s docstring for why (Polynomial-vs-Polynomial
    # rhs correctly raises TypeError on both sides, but with an inherent,
    # unfixable module-path text divergence in the message).
    specs["polynomial.Polynomial.__truediv__"] = ItemSpec(
        name="polynomial.Polynomial.__truediv__", kind="custom",
        custom_cases=poly_truediv_cases,
        numpy_adapter=_np_truediv, ionp_adapter=_ionp_truediv,
        atol=0.0, rtol=0.0,
        # NOT scalar_like: this item's complex-scalar path measurably needs
        # epsilon tolerance (see POLYDIV_COMPLEX_EPS above), and
        # `scalar_like=True` items never reach the epsilon-tolerance code
        # path in harness.py (checked and short-circuits first). Only
        # `.coef` is graded, same "scope to the one genuinely inexact
        # piece" reasoning as `fit`/`roots` above -- `domain`/`window`/
        # `symbol` construction on this path is the same exact,
        # already-covered-elsewhere code regardless of the divide's
        # rounding.
        epsilon_tolerance={"complex128": ("rel", POLYDIV_COMPLEX_EPS)},
        epsilon_tolerance_justification=(
            "polydiv's complex division (Rust kernel) vs real numpy's own "
            "complex division: not proven bit-identical algorithms, and "
            "this item's complex_scalar case measurably disagrees in the "
            "last ULP. See POLYDIV_COMPLEX_EPS's comment above for the "
            "measured bound and sweep. float64/int cases in this item's "
            "corpus are unaffected (real division has no such divergence) "
            "and remain bit-exact against that bound in practice."
        ),
        epsilon_sweep={"complex128": (20000, POLYDIV_COMPLEX_EPS)},
    )
    # `__radd__`/`__rsub__`/`__rmul__`/`__rtruediv__`/`__rfloordiv__`/
    # `__rmod__`/`__rdivmod__` deliberately use `poly_rbinop_cases`
    # (scalar-only LHS), NOT `poly_binop_cases`: as that generator's own
    # docstring says, Python's operator protocol only ever actually
    # invokes a reflected dunder with a bare non-Polynomial LHS, since
    # `Polynomial.__add__`/etc. always handle a Polynomial RHS first.
    # Calling e.g. `p.__radd__(other_polynomial)` DIRECTLY (bypassing that
    # protocol, which `poly_binop_cases` + `_mk_binop` used to do for
    # these three before this comment) exercises a path real code never
    # takes, and measurably reproduces a genuine real-numpy quirk: `other`
    # is passed RAW into `self._add(other, self.coef)` with no coefficient
    # extraction, so a bare Polynomial `other` becomes a 1-element
    # `dtype=object` array (numpy's own `as_series` wraps an un-iterated
    # Polynomial as a 0-d object scalar, same as verified here), which
    # numpy's pure-Python `polyadd` then broadcast-pads into a mixed
    # Polynomial/float object array -- not a coefficient array anionpy's
    # Rust-backed kernels can or should accept. Matching that output
    # byte-for-byte would mean deliberately reproducing a numpy
    # implementation accident nothing sane relies on, for an operand shape
    # normal code can never actually produce.
    for _op in ("__radd__", "__rsub__", "__rtruediv__",
                "__rfloordiv__", "__rmod__", "__rdivmod__"):
        specs[f"polynomial.Polynomial.{_op}"] = ItemSpec(
            name=f"polynomial.Polynomial.{_op}", kind="custom",
            custom_cases=poly_rbinop_cases,
            numpy_adapter=_mk_rbinop(_op, ionp=False),
            ionp_adapter=_mk_rbinop(_op, ionp=True),
            scalar_like=True,
        )
    del _op

    # `__rmul__`: same `POLYMUL_EPS` story as `__mul__` above, on
    # `poly_rbinop_cases`'s scalar-LHS corpus (see that generator's
    # docstring for why reflected dunders are scalar-LHS-only).
    def _np_rmul_coef(c):
        return (2.0 * np.polynomial.Polynomial(c)).coef

    def _ionp_rmul_coef(c):
        return (2.0 * _anionpy.polynomial.Polynomial(c)).coef

    specs["polynomial.Polynomial.__rmul__"] = ItemSpec(
        name="polynomial.Polynomial.__rmul__", kind="custom",
        custom_cases=poly_rbinop_cases,
        numpy_adapter=_np_rmul_coef, ionp_adapter=_ionp_rmul_coef,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"float64": ("rel", POLYMUL_EPS), "complex128": ("rel", POLYMUL_EPS)},
        epsilon_tolerance_justification=(
            "Same POLYMUL_EPS story as __mul__ above -- __rmul__ = "
            "self._mul(_as_series1(other), self.coef), the identical "
            "polymul kernel (same measured bound covers both dtypes, "
            "see __mul__'s spec above for the complex128 sweep detail)."
        ),
        epsilon_sweep={"float64": (20000, POLYMUL_EPS), "complex128": (20000, POLYMUL_EPS)},
    )

    specs["polynomial.Polynomial.__pow__"] = ItemSpec(
        name="polynomial.Polynomial.__pow__", kind="custom",
        custom_cases=poly_pow_cases,
        numpy_adapter=_np_pow, ionp_adapter=_ionp_pow,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"float64": ("rel", POLYPOW_EPS)},
        epsilon_tolerance_justification=(
            "Polynomial.__pow__ = repeated polymul under the hood "
            "(ABCPolyBase.__pow__ -> self._pow -> polypow -> repeated "
            "polymul). See POLYMUL_EPS/POLYPOW_EPS's comment block above "
            "for the measured bound and why this is a pre-existing "
            "property of the already-shipped polymul kernel, not "
            "something new here."
        ),
        epsilon_sweep={"float64": (20000, POLYPOW_EPS)},
    )
    specs["polynomial.Polynomial.__neg__"] = ItemSpec(
        name="polynomial.Polynomial.__neg__", kind="custom",
        custom_cases=poly_unary_cases,
        numpy_adapter=_np_neg, ionp_adapter=_ionp_neg, scalar_like=True,
    )
    specs["polynomial.Polynomial.__pos__"] = ItemSpec(
        name="polynomial.Polynomial.__pos__", kind="custom",
        custom_cases=poly_unary_cases,
        numpy_adapter=_np_pos, ionp_adapter=_ionp_pos, scalar_like=True,
    )
    specs["polynomial.Polynomial.__getstate__"] = ItemSpec(
        name="polynomial.Polynomial.__getstate__", kind="custom",
        custom_cases=poly_getstate_cases,
        numpy_adapter=_np_getstate, ionp_adapter=_ionp_getstate,
        scalar_like=True,
    )
    specs["polynomial.Polynomial.__setstate__"] = ItemSpec(
        name="polynomial.Polynomial.__setstate__", kind="custom",
        custom_cases=poly_setstate_cases,
        numpy_adapter=_np_setstate, ionp_adapter=_ionp_setstate,
        scalar_like=True,
    )
    specs["polynomial.Polynomial.copy"] = ItemSpec(
        name="polynomial.Polynomial.copy", kind="custom",
        custom_cases=poly_copy_cases,
        numpy_adapter=_np_copy, ionp_adapter=_ionp_copy, scalar_like=True,
    )
    specs["polynomial.Polynomial.degree"] = ItemSpec(
        name="polynomial.Polynomial.degree", kind="custom",
        custom_cases=poly_degree_cases,
        numpy_adapter=_np_degree, ionp_adapter=_ionp_degree, scalar_like=True,
    )
    specs["polynomial.Polynomial.cutdeg"] = ItemSpec(
        name="polynomial.Polynomial.cutdeg", kind="custom",
        custom_cases=poly_cutdeg_cases,
        numpy_adapter=_np_cutdeg, ionp_adapter=_ionp_cutdeg, scalar_like=True,
    )
    specs["polynomial.Polynomial.trim"] = ItemSpec(
        name="polynomial.Polynomial.trim", kind="custom",
        custom_cases=poly_trim_cases,
        numpy_adapter=_np_trim, ionp_adapter=_ionp_trim, scalar_like=True,
    )
    specs["polynomial.Polynomial.truncate"] = ItemSpec(
        name="polynomial.Polynomial.truncate", kind="custom",
        custom_cases=poly_truncate_cases,
        numpy_adapter=_np_truncate, ionp_adapter=_ionp_truncate, scalar_like=True,
    )
    specs["polynomial.Polynomial.mapparms"] = ItemSpec(
        name="polynomial.Polynomial.mapparms", kind="custom",
        custom_cases=poly_mapparms_cases,
        numpy_adapter=_np_mapparms, ionp_adapter=_ionp_mapparms, scalar_like=True,
    )
    specs["polynomial.Polynomial.convert"] = ItemSpec(
        name="polynomial.Polynomial.convert", kind="custom",
        custom_cases=poly_convert_cases,
        numpy_adapter=_np_convert, ionp_adapter=_ionp_convert,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"float64": ("rel", POLYCONVERT_EPS)},
        epsilon_tolerance_justification=(
            "convert = _compose_affine (Horner-loop substitution, "
            "multiplication-heavy) -- see POLYCONVERT_EPS's comment block "
            "above for the measured bound and why this is not provably "
            "bit-identical to numpy's own affine substitution."
        ),
        epsilon_sweep={"float64": (20000, POLYCONVERT_EPS)},
    )
    specs["polynomial.Polynomial.cast"] = ItemSpec(
        name="polynomial.Polynomial.cast", kind="custom",
        custom_cases=poly_cast_cases,
        numpy_adapter=_np_cast, ionp_adapter=_ionp_cast,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"float64": ("rel", POLYCONVERT_EPS)},
        epsilon_tolerance_justification=(
            "cast = convert to cls's own default domain/window -- same "
            "_compose_affine divergence, POLYCONVERT_EPS bound reused "
            "(see convert's spec above)."
        ),
        epsilon_sweep={"float64": (20000, POLYCONVERT_EPS)},
    )
    for _name in ("has_samecoef", "has_samedomain", "has_samewindow", "has_sametype"):
        specs[f"polynomial.Polynomial.{_name}"] = ItemSpec(
            name=f"polynomial.Polynomial.{_name}", kind="custom",
            custom_cases=poly_has_same_cases,
            numpy_adapter=_mk_has_same(_name, ionp=False),
            ionp_adapter=_mk_has_same(_name, ionp=True),
            scalar_like=True,
        )
    del _name

    specs["polynomial.Polynomial.integ"] = ItemSpec(
        name="polynomial.Polynomial.integ", kind="custom",
        custom_cases=poly_integ_cases,
        numpy_adapter=_np_integ, ionp_adapter=_ionp_integ, scalar_like=True,
    )
    specs["polynomial.Polynomial.deriv"] = ItemSpec(
        name="polynomial.Polynomial.deriv", kind="custom",
        custom_cases=poly_deriv_cases,
        numpy_adapter=_np_deriv, ionp_adapter=_ionp_deriv, scalar_like=True,
    )
    specs["polynomial.Polynomial.roots"] = ItemSpec(
        name="polynomial.Polynomial.roots", kind="custom",
        custom_cases=poly_roots_cases,
        numpy_adapter=_np_roots, ionp_adapter=_ionp_roots,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "float64": ("abs", POLYROOTS_REAL_EPS),
            "complex128": ("abs", POLYROOTS_COMPLEX_EPS),
        },
        epsilon_tolerance_justification=(
            "Polynomial.roots() = polycompanion (exact) + linalg.eigvals "
            "(independent eigensolver call path, same story as top-level "
            "polyroots) + an exact affine domain/window remap on top. "
            "Reuses polyroots' own measured bound (POLYROOTS_REAL_EPS / "
            "POLYROOTS_COMPLEX_EPS, see that spec above) rather than a "
            "fresh sweep: the remap is `off + scl * roots` with `scl` "
            "derived from the SAME domain/window this corpus's cases use "
            "(magnitude <= a few units), so it cannot materially widen an "
            "already-generously-measured absolute bound."
        ),
        epsilon_sweep={
            "float64": (20000, POLYROOTS_REAL_EPS),
            "complex128": (20000, POLYROOTS_COMPLEX_EPS),
        },
    )
    specs["polynomial.Polynomial.fromroots"] = ItemSpec(
        name="polynomial.Polynomial.fromroots", kind="custom",
        custom_cases=poly_fromroots_cases,
        numpy_adapter=_np_fromroots, ionp_adapter=_ionp_fromroots,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={
            "float64": ("rel", POLYFROMROOTS_EPS),
            "complex128": ("rel", POLYFROMROOTS_EPS),
        },
        epsilon_tolerance_justification=(
            "fromroots = polyfromroots (repeated polymul of linear "
            "factors) -- see POLYFROMROOTS_EPS's comment block above for "
            "the measured bound; same pre-existing polymul-kernel "
            "divergence as __mul__/__pow__ above."
        ),
        epsilon_sweep={
            "float64": (20000, POLYFROMROOTS_EPS),
            # complex128: measured 20,000-sample seeded sweep of random
            # complex roots (deg 1-9) via direct polyfromroots calls,
            # maxrel = 6.36564362046448e-14, well inside POLYFROMROOTS_EPS.
            "complex128": (20000, POLYFROMROOTS_EPS),
        },
    )
    specs["polynomial.Polynomial.identity"] = ItemSpec(
        name="polynomial.Polynomial.identity", kind="custom",
        custom_cases=poly_identity_cases,
        numpy_adapter=_np_identity, ionp_adapter=_ionp_identity,
        scalar_like=True,
    )
    specs["polynomial.Polynomial.basis"] = ItemSpec(
        name="polynomial.Polynomial.basis", kind="custom",
        custom_cases=poly_basis_cases,
        numpy_adapter=_np_basis, ionp_adapter=_ionp_basis, scalar_like=True,
    )
    specs["polynomial.Polynomial.linspace"] = ItemSpec(
        name="polynomial.Polynomial.linspace", kind="custom",
        custom_cases=poly_linspace_cases,
        numpy_adapter=_np_linspace, ionp_adapter=_ionp_linspace,
        atol=0.0, rtol=0.0, multi_output=True,
    )
    specs["polynomial.Polynomial.fit"] = ItemSpec(
        name="polynomial.Polynomial.fit", kind="custom",
        custom_cases=poly_fit_cases,
        numpy_adapter=_np_fit, ionp_adapter=_ionp_fit,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"float64": ("rel", POLYFIT_EPS)},
        epsilon_tolerance_justification=(
            "Polynomial.fit(...).coef IS polyfit's own coefficient output "
            "(see ABCPolyBase.fit in _polybase.py: `cls._fit(xnew, y, deg, "
            "...)` calls polyfit directly) reused unchanged after an exact, "
            "deterministic domain remap of x -- same measured bound as "
            "top-level polyfit above (POLYFIT_EPS), not a fresh sweep, "
            "since it is the identical lstsq/SVD call producing the "
            "identical coefficients."
        ),
        epsilon_sweep={"float64": (20000, POLYFIT_EPS)},
    )

    # -- Task #34 (2026-08-08): *val2d/*val3d/*valnd/*grid2d/*grid3d/
    # *vander2d/*vander3d, see this file's comment block above the
    # generators for corpus design and the float16/32/complex64 exclusion.
    for _fn in ("polyval2d", "polyval3d", "polyvalnd", "polygrid2d",
                "polygrid3d", "polyvander2d", "polyvander3d"):
        specs[f"polynomial.polynomial.{_fn}"] = ItemSpec(
            name=f"polynomial.polynomial.{_fn}", kind="custom",
            custom_cases=globals()[f"{_fn}_cases"], atol=0.0, rtol=0.0,
        )
    del _fn

    return specs


POLYNOMIAL_SPECS = _build_polynomial_specs()
