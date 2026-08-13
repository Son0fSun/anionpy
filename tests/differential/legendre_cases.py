"""anionpy.polynomial.legendre.* differential registry entries.

NEW FILE (permitted, same pattern as polynomial_cases.py/linalg_cases.py):
builds a `LEGENDRE_SPECS: dict[str, ItemSpec]` dict, merged into
`registry.REGISTRY` at the bottom of registry.py (collision-checked, same
pattern as every other `*_cases.py` merge in that file).

Covers the 24 `numpy.polynomial.legendre.*` items implemented in this
vertical slice (see anionpy/polynomial/legendre.py's module docstring for
the exact, deliberate scope boundary -- `Legendre(ABCPolyBase)` and the
`*2d`/`*3d`/`*nd` composition items are NOT covered here, not attempted in
this pass).

Bool-input dtype split (verified directly against
`numpy/polynomial/legendre.py`, see legendre.rs's module docstring for the
same finding on the Rust side): `legadd`/`legsub`/`legmulx`/`legmul`/
`legdiv`/`legpow`/`legtrim`/`legcompanion`/`legfromroots` route through
`pu.as_series` and REJECT bool input; `legder`/`legint`/`legval`/
`legvander` do their own `if c.dtype.char in '?bBhH...': c = c.astype(np.double)`
dtype handling and PROMOTE bool input. Every `custom_cases` generator below
exercises int/float/complex input plus, where meaningful, this bool split
and trailing-zero/trimming behavior. Tolerances are declared only where
actually measured (see the bottom of this file for the `legfit`/`legroots`/
`leggauss` epsilon declarations and their sweep evidence).
"""
from __future__ import annotations

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


def _const_cases(attr):
    def gen():
        return [(f"{attr}_value", (), {})]
    return gen


# ---------------------------------------------------------------------------
# legline / legtrim
# ---------------------------------------------------------------------------

def legline_cases():
    return _std([
        ("basic", (1.0, 2.0), {}),
        ("zero_scale", (3.0, 0.0), {}),
        ("negative", (-1.5, -2.5), {}),
        ("off_zero", (0.0, 4.0), {}),
        ("both_zero", (0.0, 0.0), {}),
    ])


def legtrim_cases():
    return _std([
        ("trailing_zeros", (TRAILING_ZERO_C,), {}),
        ("all_zero", (ALL_ZERO_C,), {}),
        ("no_trim_needed", (FLOAT_C,), {}),
        ("with_tol", (list(TRAILING_ZERO_C) + [1e-10], 1e-8), {}),
        ("complex", (COMPLEX_C + [0j],), {}),
        ("int_input", (INT_C + [0],), {}),
        # as_series rejects bool outright -- verified: L.legtrim([True,
        # False]) raises ValueError("Coefficient arrays have no common type").
        ("bool_input", (BOOL_C,), {}),
    ])


# ---------------------------------------------------------------------------
# legval
# ---------------------------------------------------------------------------

def legval_cases():
    cases = []
    for c_label, c in [("int_c", INT_C), ("float_c", FLOAT_C), ("bool_c", BOOL_C),
                        ("complex_c", COMPLEX_C), ("trailing_zero_c", TRAILING_ZERO_C),
                        ("single_c", SINGLE_C)]:
        cases.append((f"scalar_x__{c_label}", (X_SCALAR, c), {}))
        cases.append((f"list_x__{c_label}", (XS, c), {}))
        cases.append((f"complex_x__{c_label}", (X_COMPLEX, c), {}))
    cases.append(("array_x_2d", (np.array([[0.0, 1.0], [2.0, 3.0]]), FLOAT_C), {}))
    # ---- Added 2026-08-07 (Monday): same real-coefficient/complex-x,
    # multiple-degree regression guard as `laguerre_cases.py`'s
    # `lagval_cases` addition -- see that comment for the full mechanism
    # (`legval`'s ratio is precomputed via `T::from_f64` before the
    # multiply so it never actually needed the `lag_eval`-style hybrid
    # kernel, but the divisor-3/5/6 combination is still worth pinning
    # here directly rather than trusting that asymmetry to hold forever).
    _real_x_complex = 0.9505 - 1.3820j
    for _deg, _c in [
        (4, [0.5993, -1.21, 2.07, 0.44]),
        (5, [0.5993, -1.21, 2.07, 0.44, -0.83]),
        (6, [0.5993, -1.21, 2.07, 0.44, -0.83, 1.6]),
        (7, [0.5993, -1.21, 2.07, 0.44, -0.83, 1.6, -2.05]),
    ]:
        cases.append((f"real_c_len{_deg}_complex_x_DEGREE_REGRESSION",
                       (_real_x_complex, _c), {}))
    return cases


# ---------------------------------------------------------------------------
# add / sub / mulx / mul / div / pow
# ---------------------------------------------------------------------------

def legadd_cases():
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


def legsub_cases():
    return _std([
        ("same_len", (C1, C2), {}),
        ("c1_shorter", (C1_SHORT, C1), {}),
        ("identical_cancels", (FLOAT_C, FLOAT_C), {}),
        ("complex_minus_real", (COMPLEX_C, FLOAT_C), {}),
        ("int_minus_bool", (INT_C, BOOL_C), {}),
    ])


def legmulx_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("int_c", (INT_C,), {}),
        ("zero_series", (ALL_ZERO_C[:1],), {}),  # numpy special-cases [0]
        ("complex_c", (COMPLEX_C,), {}),
        ("single_nonzero", (SINGLE_C,), {}),
        ("bool_c", (BOOL_C,), {}),  # as_series-family: rejects bool
        # 2026-08-07 (Monday): same signed-zero derived-vs-literal gap as
        # `polynomial_cases.py`'s `polymulx_cases` -- numpy's `legmulx`
        # sets `prd[0] = c[0] * 0` (derived zero), not a literal fill.
        # Every case above has a non-negative first coefficient, so none
        # can distinguish correct from wrong. Out-of-corpus sweep found
        # 177/2000 signed-zero-only divergences before the
        # `ionp-core/src/legendre.rs` `legmulx` fix (literal `T::zero()`
        # -> `c[0].leg_mul(T::zero())`).
        ("negative_real_first_coef", ([-1.02777029 + 0.06451574j, 2.0 - 1.0j],), {}),
        ("negative_imag_first_coef", ([0.5 - 3.25j, 1.0 + 1.0j],), {}),
        ("negative_both_first_coef", ([-4.5 - 2.5j],), {}),
        ("negative_real_float", ([-2.5, 1.0, 3.0],), {}),
    ])


def legmul_cases():
    return _std([
        ("basic", (C1, C2), {}),
        ("int_times_float", (INT_C, FLOAT_C), {}),
        ("complex_times_real", (COMPLEX_C, FLOAT_C), {}),
        ("with_zero_series", (FLOAT_C, ALL_ZERO_C[:1]), {}),
        ("trailing_zero_operand", (TRAILING_ZERO_C, C2), {}),
        ("single_by_single", ([2.0], [3.0]), {}),
        ("bool_operand", (BOOL_C, FLOAT_C), {}),  # as_series-family: rejects bool
        # ---- Added 2026-08-07 (Monday) as a FAILS-by-design regression
        # guard; FIXED 2026-08-08 (Monday, ticket #48), kept (renamed) as a
        # permanent regression case rather than deleted. ----
        # Every OTHER complex case above uses values ("nice" real/imag
        # parts) that happen to round the same way as real numpy, so this
        # corpus was green while `legmul` was NOT bit-exact on ~85% of a
        # random complex128 sweep (seed 0x5EEDC1A55, N=25000). The root
        # cause (ticket #48): `legendre.rs`'s `scale()` helper multiplied
        # array-element-first (`v.leg_mul(s)`) instead of numpy's own
        # scalar-first order (`c[0]*xs`, `c[-i]*xs`, etc.) -- invisible on
        # `f64` (commutative), a real divergence on `C128` because
        # `complex_mul_fma` is asymmetric under argument swap. Fixed by
        # swapping `scale` to `s.leg_mul(v)`. This case (and its
        # `legpow_cases`/`legfromroots_cases` siblings below) stays in the
        # corpus under its original values so a regression trips it again.
        ("complex_reprojection",
         ([0.7 + 1.3j, -2.9 + 0.4j, 1.1 - 0.3j], [0.3 - 0.8j, 2.2 + 1.7j]), {}),
        # ---- Added 2026-08-08 (Monday), ticket #84 ----
        # Ticket #84's own repro: `legmul(c1, c2)` with `c1=[5e-324]*3`,
        # `c2=[-5e-324]*2` (subnormal, all-underflow inputs). Pre-fix,
        # numpy gave a result with `-0.+0.j` at an index where anionpy
        # gave `+0.+0.j` -- `np.array_equal` is BLIND to this, only a
        # `.view(np.uint64)` bit compare (which harness.py's atol=0.0/
        # rtol=0.0 exact-comparison path performs) catches it. Root
        # cause: `legmul`'s internal recursion calls `pad_add`/
        # `pad_sub` at the same call sites where real numpy calls
        # `legadd`/`legsub` (i.e. `pu._add`/`pu._sub`), which trim BOTH
        # operands via `as_series` before the padded elementwise
        # combine; `pad_add`/`pad_sub` never did this trim, so an
        # untrimmed trailing zero in the shorter operand could collide
        # with (and flip the sign of) an untouched -0.0 in the longer
        # operand at that index. Fixed in `ionp-core/src/legendre.rs`
        # by trimming both operands via `trim_trailing_zeros`
        # immediately before every `pad_add`/`pad_sub` call site in
        # `legmul` (same fix applied to lagmul/hermmul/hermemul, and to
        # the shared `poly::add_trim`/`sub_trim` covering every basis's
        # public `*add`/`*sub` bindings). This case stays in the corpus
        # permanently as a regression guard.
        ("underflow_signed_zero_ticket84",
         ([5e-324, 5e-324, 5e-324], [-5e-324, -5e-324]), {}),
    ])


def legdiv_cases():
    return _std([
        ("basic", (C1, C2), {}),
        ("swap", (C2, C1), {}),
        ("c1_shorter_than_c2", (C1_SHORT, C1), {}),
        ("divide_by_scalar", (C1, [2.0]), {}),
        ("exact_division", ([2.0, 4.0, 6.0], [1.0, 2.0]), {}),
        ("complex", (COMPLEX_C, [1.0, 1.0]), {}),
        ("divide_by_zero_series", (C1, [0.0]), {}),
        ("bool_c1", (BOOL_C, [1.0, 2.0]), {}),  # as_series-family: rejects bool
        # ---- Ticket #85 (2026-08-08, Monday): the `lc1 < lc2` early-return
        # branch's placeholder quotient was a literal `T::zero()` fill, not
        # numpy's `c1[:1] * 0` DERIVED zero -- loses the sign a negative
        # denormal `c1[0]` produces (`(-5e-324) * 0 == -0.0` under
        # IEEE-754). Measured 1028/4000 (25.7%) signed-zero-only
        # divergences before the `ionp-core/src/legendre.rs` `legdiv` fix
        # (`vec![T::zero()]` -> `vec![c1[0].leg_mul(T::zero())]`, same
        # class as `legmulx`/`mulx`'s `prd[0]` fill). Exact repro from the
        # ticket's own measurement.
        ("underflow_signed_zero_quotient_ticket85",
         ([complex(-5e-324, -5e-324)],
          [complex(-1.224523546275244, 0.6987276967558644),
           complex(-0.8346534520393827, -1.1215839526744682),
           complex(0.778393747052665, 1.108389599473554)]), {}),
        # A SECOND, independently-discovered defect in the same two
        # early-return branches: this port never did numpy `_div`'s
        # top-level `[c1, c2] = as_series([c1, c2])` pre-trim, so an
        # untrimmed trailing exact zero in `c1` leaked through into the
        # returned remainder as a SHAPE divergence (numpy's remainder is
        # the length-1 trimmed `c1`; pre-fix anionpy returned the full
        # untrimmed length-2 `c1_in`). Fixed by trimming both operands via
        # `trim_trailing_zeros` at the top of `legdiv`, matching `chebdiv`'s
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


def legpow_cases():
    return _std([
        ("square", (C1, 2), {}),
        ("cube", (FLOAT_C, 3), {}),
        # power_zero is dtype-PRESERVING for legpow (unlike polypow, which
        # always forces float64 [1.0]) -- verified live:
        # L.legpow([1j, 2j], 0) -> array([1.+0.j]) complex128.
        ("power_zero", (C1, 0), {}),
        ("power_zero_complex", (COMPLEX_C, 0), {}),
        ("power_one", (C1, 1), {}),
        ("complex_base", (COMPLEX_C, 2), {}),
        ("int_c", (INT_C, 2), {}),
        # ---- Added 2026-08-07 (Monday) as a FAILS-by-design regression
        # guard; FIXED 2026-08-08 (Monday, ticket #48) via the `legmul`
        # `scale()` operand-order fix (`legpow` is repeated `legmul`) --
        # see `legmul_cases`' `complex_reprojection` case comment above for
        # the mechanism. Kept (renamed) as a permanent regression case.
        ("complex_reprojection",
         ([0.1 + 0.2j, -0.3j, 1.7 - 0.9j], 3), {}),
    ])


# ---------------------------------------------------------------------------
# der / int
# ---------------------------------------------------------------------------

def legder_cases():
    C = [1.0, 2.0, 3.0, 4.0]
    return _std([
        ("first_order", (C, 1), {}),
        ("third_order", (C, 3), {}),
        ("zero_order", (C, 0), {}),
        ("order_exceeds_degree", (C, 10), {}),
        ("with_scale", (C, 1, 2.0), {}),
        ("complex_c", (COMPLEX_C, 1), {}),
        ("int_c", (INT_C, 1), {}),
        # legder does its OWN dtype handling, NOT `as_series` -- bool is
        # promoted to float64, never rejected. Verified:
        # L.legder([True, False, True], 1) -> array([0., 6.]), no error.
        ("bool_c", (BOOL_C, 1), {}),
        # 2026-08-07 (Monday): same signed-zero derived-vs-literal gap as
        # `polynomial_cases.py`'s `polyder_cases` -- numpy's `legder`'s
        # `cnt >= n` branch sets `c = c[:1] * 0` (derived zero from the
        # ORIGINAL first coefficient), not a literal fill.
        # `order_exceeds_degree` above uses an all-positive `C`, which
        # cannot distinguish correct from wrong. Out-of-corpus sweep found
        # 163/2000 signed-zero-only divergences before the
        # `ionp-core/src/legendre.rs` `legder` fix (literal
        # `vec![T::zero()]` -> `vec![c[0].leg_mul(T::zero())]`).
        ("order_exceeds_degree_negative_real", ([-3.0, 2.0, 1.0], 10), {}),
        ("order_exceeds_degree_negative_complex", ([-1.5 - 2.5j, 1.0 + 1.0j], 5), {}),
        ("order_exceeds_degree_negative_imag", ([2.5 - 1.5j, 1.0], 5), {}),
    ])


def legint_cases():
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


# ---------------------------------------------------------------------------
# vander / companion
# ---------------------------------------------------------------------------

def legvander_cases():
    return _std([
        ("basic", (XS, 5), {}),
        ("deg_zero", (XS, 0), {}),
        ("scalar_x", (X_SCALAR, 3), {}),
        ("complex_x", ([1 + 1j, 2 - 1j, 0.5j], 4), {}),
        ("int_x", ([1, 2, 3], 3), {}),
        ("bool_x", ([True, False], 2), {}),  # own-dtype path: bool promoted, not rejected
    ])


def legcompanion_cases():
    return _std([
        ("degree2", (C1,), {}),
        ("degree1", (C1_SHORT,), {}),
        ("degree4", ([1.0, 0.0, -2.0, 0.0, 1.0],), {}),
        ("complex", (COMPLEX_C,), {}),
        ("int_c", (INT_C,), {}),
        ("bool_c", (BOOL_C,), {}),  # as_series-family: rejects bool
    ])


# ---------------------------------------------------------------------------
# fromroots / roots
# ---------------------------------------------------------------------------

def legfromroots_cases():
    return _std([
        ("real_roots", ([-1.0, 0.0, 1.0],), {}),
        ("no_roots", ([],), {}),
        ("single_root", ([3.0],), {}),
        ("complex_roots", ([-1j, 0.0, 1j],), {}),
        ("repeated_root", ([1.0, 1.0, -1.0],), {}),
        ("five_roots", ([-2.0, -1.0, 0.0, 1.0, 2.0],), {}),
        ("bool_roots", ([True, False],), {}),  # as_series-family: rejects bool
        # ---- Added 2026-08-07 (Monday) as a FAILS-by-design regression
        # guard; FIXED 2026-08-08 (Monday, ticket #48) via the `legmul`
        # `scale()` operand-order fix (`legfromroots` is a balanced-pairing
        # tree of `legmul` calls of linear factors) -- see `legmul_cases`'
        # `complex_reprojection` case comment above for the mechanism.
        # Kept (renamed) as a permanent regression case.
        ("complex_reprojection",
         ([0.1 + 0.2j, -0.3j, 1.7 - 0.9j, 2.2 + 0.6j],), {}),
    ])


def legroots_cases():
    return _std([
        ("real_roots", (np.polynomial.legendre.legfromroots([-1.0, 0.0, 1.0]),), {}),
        ("complex_roots", (np.polynomial.legendre.legfromroots([-1j, 0.0, 1j]),), {}),
        ("degree1", (C1_SHORT,), {}),
        ("degree0_no_roots", (SINGLE_C,), {}),
        ("five_real_roots", (np.polynomial.legendre.legfromroots(
            [-2.0, -1.0, 0.0, 1.0, 2.0]),), {}),
    ])


def _canonical_root_order(arr):
    """Stabilizes root order for cross-implementation comparison without
    knowledge of the other side's output (each `numpy_adapter`/
    `ionp_adapter` call only sees its own side). A naive `np.sort`
    (lexicographic on raw real part, then imaginary) is UNSTABLE exactly
    when a root's true real (or imaginary) part is 0 but each independent
    eigensolver lands on a different-SIGNED near-zero float -- confirmed
    on the "complex_roots" case (roots {0, i, -i}): numpy's near-zero root
    has real part `+1.08e-16`, anionpy's has `-1.11e-16`, so plain
    `np.sort` puts them in genuinely different array positions even though
    every coordinate is well within `LEGROOTS_COMPLEX_EPS` (2.2e-06) of
    its counterpart. Rounding the sort KEY (not the compared values
    themselves) to 6 decimal places -- three orders of magnitude coarser
    than the declared epsilon bound -- collapses ties among values already
    considered equal for grading purposes, so it cannot hide any mismatch
    beyond what epsilon_tolerance already tolerates; two roots that are
    genuinely more than 1e-6 apart still sort correctly on their
    unrounded coordinates once the rounded leading key differs."""
    key = list(zip(np.round(arr.real, 6), np.round(arr.imag, 6)))
    order = sorted(range(len(arr)), key=lambda i: key[i])
    return arr[order]


def _legroots_numpy_sorted(c):
    """`numpy_adapter` for `legroots`: canonicalizes the raw root array's
    order before comparison (see `_canonical_root_order`'s docstring).
    numpy's own `legroots` docstring/source make NO ordering guarantee --
    the roots are exactly whatever order `linalg.eigvals` hands back from
    LAPACK's `dgeev`/`zgeev` for the companion matrix, an implementation
    detail of the specific eigensolver, not a documented contract.
    anionpy's `legroots` goes through its own, independent eigensolver
    (not LAPACK) on the SAME (bit-exact, post the `legcompanion` fix
    above) companion matrix, so it is entitled to return the same SET of
    roots in a different order -- comparing unsorted arrays
    position-by-position spuriously fails on nothing more than
    eigenvalue-ordering divergence between two different algorithms, not
    a genuine value defect."""
    return _canonical_root_order(np.polynomial.legendre.legroots(c))


def _legroots_ionp_sorted(c):
    import anionpy as ap
    r = ap.polynomial.legendre.legroots(ap.array(c))
    r_np = np.asarray(r.tolist() if hasattr(r, "tolist") else r)
    return _canonical_root_order(r_np)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def legfit_cases():
    rng = np.random.default_rng(20260807)
    x = np.linspace(-1.0, 1.0, 25)
    y = np.polynomial.legendre.legval(x, [0.0, 1.0, 0.0, -1.0])
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
# leggauss / legweight
# ---------------------------------------------------------------------------

def leggauss_cases():
    return _std([
        ("deg1", (1,), {}),
        ("deg2", (2,), {}),
        ("deg5", (5,), {}),
        ("deg10", (10,), {}),
        ("deg25", (25,), {}),
    ])


def legweight_cases():
    return _std([
        ("scalar", (0.5,), {}),
        ("list", (XS,), {}),
        ("array", (np.array(XS),), {}),
    ])


# ---------------------------------------------------------------------------
# leg2poly / poly2leg
# ---------------------------------------------------------------------------

def leg2poly_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("int_c", (INT_C,), {}),
        ("complex_c", (COMPLEX_C,), {}),
        ("trailing_zero_c", (TRAILING_ZERO_C,), {}),
        ("single_c", (SINGLE_C,), {}),
        ("two_term", (C1_SHORT,), {}),
        ("bool_c", (BOOL_C,), {}),  # as_series-family: rejects bool
    ])


def poly2leg_cases():
    return _std([
        ("float_c", (FLOAT_C,), {}),
        ("int_c", (INT_C,), {}),
        ("complex_c", (COMPLEX_C,), {}),
        ("trailing_zero_c", (TRAILING_ZERO_C,), {}),
        ("single_c", (SINGLE_C,), {}),
        ("bool_c", (BOOL_C,), {}),  # as_series-family: rejects bool
    ])


# Measured 2026-08-07 via /private/tmp/leg_eps_sweep.py, run OUT of this
# corpus (independent seeded sweeps, 20,000 samples each -- the
# MIN_ULP_SWEEP_N floor `ItemSpec.__post_init__` enforces on every
# epsilon_tolerance below). Real numpy's own `legfit` goes through a
# scaled-lstsq path (same shape as `polyfit`); `legroots` goes through
# `linalg.eigvals` on the companion matrix; `leggauss` goes through
# `linalg.eigvalsh` on the companion matrix -- none of these are
# bit-identical to anionpy's independent call path by construction
# (different LAPACK/Rust SVD/eigensolver routines), so these are measured
# epsilon tolerances, not bugs:
#   legfit float64:              max rel = 7.539497872611918e-10  (N=20000)
#   legroots complex128:         max abs = 2.2034570981155355e-06 (N=20000)
#   leggauss x/w float64:        max abs = 1.222980050563649e-15  (N=20000)
#
# legroots' OUTPUT dtype is unconditionally complex128: unlike `polyroots`,
# real numpy's `legroots` does NOT call `_to_real_if_imag_zero` (verified
# directly against numpy/polynomial/legendre.py -- no such call exists in
# its body, confirmed live: `NL.legroots((1,2,3,4))` returns complex128
# with `+0.j` entries even though every root is real). anionpy's
# `legroots` matches this exactly (no real-cast). But the differential
# harness's epsilon_tolerance/epsilon_sweep dtype key is looked up from
# the INPUT operand's dtype, not the output's -- so a real float64
# coefficient array (e.g. "five_real_roots") still needs a "float64" key
# even though what's actually being bounded is complex128 output; the
# spec below declares BOTH keys with the same measured value. (This
# sweep's own float64 accumulator never moved off its 0.0 initial value
# across 20,000 real-rooted draws -- consistent with the real part of the
# root error also being tiny -- but the "float64"-KEYED epsilon_tolerance
# entry still needs to exist and cover the actual complex128-valued
# comparison, hence reusing LEGROOTS_COMPLEX_EPS under both keys rather
# than introducing a separate, unmeasured "true" float64 bound.)
LEGFIT_EPS = 7.539497872611918e-10
LEGROOTS_COMPLEX_EPS = 2.2034570981155355e-06
LEGGAUSS_EPS = 1.222980050563649e-15


# ---------------------------------------------------------------------------
# Task #34 (2026-08-08): legval2d/legval3d/legvalnd/leggrid2d/leggrid3d/
# legvander2d/legvander3d -- built on polyutils.py's shared _valnd/_gridnd/
# _vander_nd/_vander_nd_flat machinery (see that module's docstring for the
# ticket-premise correction: FOUR shared helpers back this family, not the
# two -- _vander_nd/_vander_nd_flat -- the ticket named). Corpus varies
# point count (1-4 per axis), degree (0-2 per axis), and dtype (float64,
# complex128, int -- promoted via _common_dtype). float16/float32/
# complex64 are DELIBERATELY EXCLUDED: the already-declared, already-
# shipped 1-D `legval`/`legvander` kernels this family is built on
# universally upcast those three dtypes to float64/complex128 (values
# match real numpy bit-exact, but dtype does not) -- a pre-existing
# divergence in code this ticket did not write and has no permission to
# touch (lives in the shared Rust kernels, confirmed present for all six
# bases, not introduced here). See docs/TICKET-34-*.md.

def _leg_nd_base_cases():
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


def legval2d_cases():
    return [(lbl, (xs, ys, c2), {}) for lbl, xs, ys, zs, c2, c3, deg in _leg_nd_base_cases()]


def legval3d_cases():
    return [(lbl, (xs, ys, zs, c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _leg_nd_base_cases()]


def legvalnd_cases():
    return [(lbl, ((xs, ys, zs), c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _leg_nd_base_cases()]


def leggrid2d_cases():
    return [(lbl, (xs, ys, c2), {}) for lbl, xs, ys, zs, c2, c3, deg in _leg_nd_base_cases()]


def leggrid3d_cases():
    return [(lbl, (xs, ys, zs, c3), {}) for lbl, xs, ys, zs, c2, c3, deg in _leg_nd_base_cases()]


def legvander2d_cases():
    return [(lbl, (xs, ys, deg[:2]), {}) for lbl, xs, ys, zs, c2, c3, deg in _leg_nd_base_cases()]


def legvander3d_cases():
    return [(lbl, (xs, ys, zs, deg), {}) for lbl, xs, ys, zs, c2, c3, deg in _leg_nd_base_cases()]


# ---------------------------------------------------------------------------
# assembling the spec dict
# ---------------------------------------------------------------------------

def _build_legendre_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}

    specs["polynomial.legendre.legdomain"] = ItemSpec(
        name="polynomial.legendre.legdomain", kind="custom",
        custom_cases=_const_cases("legdomain"),
        numpy_adapter=lambda: np.polynomial.legendre.legdomain,
        ionp_adapter=lambda: __import__("anionpy").polynomial.legendre.legdomain,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legzero"] = ItemSpec(
        name="polynomial.legendre.legzero", kind="custom",
        custom_cases=_const_cases("legzero"),
        numpy_adapter=lambda: np.polynomial.legendre.legzero,
        ionp_adapter=lambda: __import__("anionpy").polynomial.legendre.legzero,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legone"] = ItemSpec(
        name="polynomial.legendre.legone", kind="custom",
        custom_cases=_const_cases("legone"),
        numpy_adapter=lambda: np.polynomial.legendre.legone,
        ionp_adapter=lambda: __import__("anionpy").polynomial.legendre.legone,
        atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legx"] = ItemSpec(
        name="polynomial.legendre.legx", kind="custom",
        custom_cases=_const_cases("legx"),
        numpy_adapter=lambda: np.polynomial.legendre.legx,
        ionp_adapter=lambda: __import__("anionpy").polynomial.legendre.legx,
        atol=0.0, rtol=0.0,
    )

    specs["polynomial.legendre.legline"] = ItemSpec(
        name="polynomial.legendre.legline", kind="custom",
        custom_cases=legline_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legtrim"] = ItemSpec(
        name="polynomial.legendre.legtrim", kind="custom",
        custom_cases=legtrim_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legval"] = ItemSpec(
        name="polynomial.legendre.legval", kind="custom",
        custom_cases=legval_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legadd"] = ItemSpec(
        name="polynomial.legendre.legadd", kind="custom",
        custom_cases=legadd_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legsub"] = ItemSpec(
        name="polynomial.legendre.legsub", kind="custom",
        custom_cases=legsub_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legmulx"] = ItemSpec(
        name="polynomial.legendre.legmulx", kind="custom",
        custom_cases=legmulx_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legmul"] = ItemSpec(
        name="polynomial.legendre.legmul", kind="custom",
        custom_cases=legmul_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legdiv"] = ItemSpec(
        name="polynomial.legendre.legdiv", kind="custom",
        custom_cases=legdiv_cases, atol=0.0, rtol=0.0, multi_output=True,
    )
    specs["polynomial.legendre.legpow"] = ItemSpec(
        name="polynomial.legendre.legpow", kind="custom",
        custom_cases=legpow_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legder"] = ItemSpec(
        name="polynomial.legendre.legder", kind="custom",
        custom_cases=legder_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legint"] = ItemSpec(
        name="polynomial.legendre.legint", kind="custom",
        custom_cases=legint_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legvander"] = ItemSpec(
        name="polynomial.legendre.legvander", kind="custom",
        custom_cases=legvander_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legcompanion"] = ItemSpec(
        name="polynomial.legendre.legcompanion", kind="custom",
        custom_cases=legcompanion_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legfromroots"] = ItemSpec(
        name="polynomial.legendre.legfromroots", kind="custom",
        custom_cases=legfromroots_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.legroots"] = ItemSpec(
        name="polynomial.legendre.legroots", kind="custom",
        custom_cases=legroots_cases, atol=0.0, rtol=0.0,
        # eigenvalues of the companion matrix: same "independent LAPACK/
        # Rust eigensolver call path, not bit-identical" story as
        # linalg.eigvals itself / polyroots. See the sweep block above this
        # function for the measured bound. `numpy_adapter`/`ionp_adapter`
        # sort the raw root array before comparison (see
        # `_legroots_numpy_sorted`'s docstring) -- legroots makes no
        # ordering guarantee, and comparing two independent eigensolvers'
        # raw (unsorted) output positionally is not a meaningful
        # comparison at all, let alone one this epsilon bound was ever
        # meant to cover.
        numpy_adapter=_legroots_numpy_sorted,
        ionp_adapter=_legroots_ionp_sorted,
        # The `epsilon_tolerance`/`epsilon_sweep` dtype key is looked up
        # from the INPUT operand's dtype (`_dtype_key_candidates`), not
        # the output's -- and `legroots`'s output is unconditionally
        # complex128 (see below) even when the INPUT coefficient array is
        # real float64 (e.g. "five_real_roots"). Both keys carry the SAME
        # measured bound: the quantity being bounded (max abs coordinate
        # distance between two independent eigensolvers' complex128 root
        # sets, post-sort) is identical regardless of which dtype the
        # input coefficients happened to be.
        epsilon_tolerance={
            "complex128": ("abs", LEGROOTS_COMPLEX_EPS),
            "float64": ("abs", LEGROOTS_COMPLEX_EPS),
        },
        epsilon_tolerance_justification=(
            "legroots = legcompanion (exact) + linalg.eigvals (independent "
            "eigensolver call path vs real numpy's LAPACK dgeev/zgeev, "
            "non-bit-identical by construction) + sort (see numpy_adapter/ "
            "ionp_adapter above -- legroots makes no ordering guarantee, so "
            "the two independent eigensolvers' outputs are canonicalized by "
            "np.sort before comparison, identically on both sides). Unlike "
            "polyroots, legroots never casts to real (verified against "
            "numpy source), so its output dtype is unconditionally "
            "complex128 regardless of the input coefficient array's dtype "
            "-- both the 'complex128' and 'float64' keys carry the same "
            "bound because the dtype-key lookup is keyed off the INPUT "
            "operand (real float64 coefficients for e.g. "
            "'five_real_roots'), not the always-complex128 output. Bound "
            "is the max abs coordinate error over a 20,000-sample seeded "
            "sweep (degree 1-7 real-rooted polynomials, sorted before "
            "comparison, exactly matching the numpy_adapter/ionp_adapter "
            "methodology above), independent of legroots_cases() above -- "
            "see /private/tmp/leg_eps_sweep.py."
        ),
        epsilon_sweep={
            "complex128": (20000, LEGROOTS_COMPLEX_EPS),
            "float64": (20000, LEGROOTS_COMPLEX_EPS),
        },
    )
    specs["polynomial.legendre.legfit"] = ItemSpec(
        name="polynomial.legendre.legfit", kind="custom",
        custom_cases=legfit_cases,
        atol=0.0, rtol=0.0,
        epsilon_tolerance={"float64": ("rel", LEGFIT_EPS)},
        epsilon_tolerance_justification=(
            "Least-squares fit: normalized-Vandermonde build (exact, Rust) + "
            "linalg.lstsq (independent SVD call path vs real numpy, not "
            "bit-identical) + un-normalize. Bound is the max relative "
            "coefficient disagreement over a 20,000-fit seeded sweep, "
            "independent of legfit_cases() -- see "
            "/private/tmp/leg_eps_sweep.py."
        ),
        epsilon_sweep={"float64": (20000, LEGFIT_EPS)},
    )
    specs["polynomial.legendre.leggauss"] = ItemSpec(
        name="polynomial.legendre.leggauss", kind="custom",
        custom_cases=leggauss_cases,
        atol=0.0, rtol=0.0, multi_output=True,
        epsilon_tolerance={"float64": ("abs", LEGGAUSS_EPS)},
        epsilon_tolerance_justification=(
            "leggauss's nodes come from linalg.eigvalsh on the companion "
            "matrix (independent symmetric-eigensolver call path vs real "
            "numpy's LAPACK dsyevd, not bit-identical) followed by one "
            "Newton polish step and a weight computation built from legval/ "
            "legder -- all exact Rust arithmetic once the eigenvalues "
            "differ in their last bit or two. Bound is the max abs "
            "coordinate error (over both nodes x and weights w) across a "
            "20,000-sample seeded sweep of degrees 1-59, independent of "
            "leggauss_cases() above -- see /private/tmp/leg_eps_sweep.py. "
            "Tiny (~1e-15): this is float64 ULP-level eigenvalue noise, not "
            "a structural discrepancy."
        ),
        epsilon_sweep={"float64": (20000, LEGGAUSS_EPS)},
    )
    specs["polynomial.legendre.legweight"] = ItemSpec(
        name="polynomial.legendre.legweight", kind="custom",
        custom_cases=legweight_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.leg2poly"] = ItemSpec(
        name="polynomial.legendre.leg2poly", kind="custom",
        custom_cases=leg2poly_cases, atol=0.0, rtol=0.0,
    )
    specs["polynomial.legendre.poly2leg"] = ItemSpec(
        name="polynomial.legendre.poly2leg", kind="custom",
        custom_cases=poly2leg_cases, atol=0.0, rtol=0.0,
    )

    # -- Task #34 (2026-08-08): *val2d/*val3d/*valnd/*grid2d/*grid3d/
    # *vander2d/*vander3d, see this file's comment block above the
    # generators for corpus design and the float16/32/complex64 exclusion.
    for _fn in ("legval2d", "legval3d", "legvalnd", "leggrid2d",
                "leggrid3d", "legvander2d", "legvander3d"):
        specs[f"polynomial.legendre.{_fn}"] = ItemSpec(
            name=f"polynomial.legendre.{_fn}", kind="custom",
            custom_cases=globals()[f"{_fn}_cases"], atol=0.0, rtol=0.0,
        )
    del _fn

    return specs


LEGENDRE_SPECS = _build_legendre_specs()
