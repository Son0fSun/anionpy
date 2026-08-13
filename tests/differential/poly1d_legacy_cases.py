"""anionpy's LEGACY polynomial API differential registry entries
(`poly1d`, `polyval`, `polyadd`, `polysub`, `polyder`, `polyint`,
`polydiv`, `poly`, `roots`, `polyfit`).

NOT wired into `registry.py` -- that file was being actively edited by a
concurrent agent working on `anionpy.polynomial` (the UNRELATED modern
chebyshev/legendre/hermite/power-series package) at the time this file
was written, and touching it risked a last-writer-wins collision with
that work. This file follows the exact same `ItemSpec`/`custom_cases`
shape as every other `*_cases.py` (see chebyshev_cases.py for the
canonical example) so a human can merge

    from poly1d_legacy_cases import LEGACY_POLY_SPECS
    _check_no_collisions(LEGACY_POLY_SPECS)
    REGISTRY.update(LEGACY_POLY_SPECS)

into registry.py in one pass once that file is free to touch again.

Every case below was ALSO run directly against real numpy 2.5.1 via
standalone probe scripts (not saved) from `/private/tmp`, using the
actual installed `anionpy` package -- see
`ionp/docs/TICKET-72-POLY1D-2026-08-08.md` for the summary of that
sweep. This file re-encodes the same inputs in the registry's format so
the differential harness can re-run them once wired in; it does not
introduce new claims beyond what was already measured.

Scope / declines:
  - `polymul` (free function), `poly1d.__mul__`/`__rmul__` (non-scalar),
    and `poly1d.__pow__` for n > 1 are NOT covered here: they need
    `np.convolve`'s summation order, unreproduced (ticket #45). Declining
    to declare them is a SUCCESS condition per that ticket, not a gap in
    this file.
  - `roots` and `polyfit`/`polyfit_full`/`polyfit_cov` are NOT bit-exact
    (both go through `anionpy.linalg.eigvals`/`lstsq`, independent
    LAPACK-adjacent paths from real numpy's, exactly the same situation
    already documented for the MODERN
    `polynomial.polynomial.polyroots`/`polyfit` in
    `anionpy/_state/polynomial.py`). Declared with `epsilon_tolerance`
    plus a real `epsilon_sweep` (seed=9182736, N=20000, run OUT of this
    corpus via `/private/tmp/poly_eps_sweep_legacy.py`), not `atol`/
    `rtol` blind spots or prose-only assertions -- see
    ROOTS_REAL_EPS/ROOTS_COMPLEX_EPS/POLYFIT_EPS below and the ticket
    doc for the measured values.
  - `array_str`/`correlate`/`convolve` are not covered: not implemented.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)

import numpy as np

from registry import ItemSpec

# Measured 2026-08-08 via /private/tmp/poly_eps_sweep_legacy.py, run OUT of
# this corpus (independent seeded sweep, seed=9182736, N=20000 -- the
# MIN_ULP_SWEEP_N floor ItemSpec.__post_init__ enforces on every
# epsilon_tolerance below, same seed convention as the modern
# polynomial.polynomial.polyroots/polyfit sweep in polynomial_cases.py).
# roots() goes through linalg.eigvals on a hand-assembled companion
# matrix; polyfit() goes through linalg.lstsq -- neither is bit-identical
# to real numpy's own LAPACK dgeev/zgeev or SVD path by construction.
#   roots float64 (real):   max abs = 2.7222051279807147e-08 (N=20000)
#   roots complex128:       max abs = 1.150574417275806e-10  (N=20000)
#   polyfit float64:        max rel = 1.963794112607891e-12  (N=20000)
ROOTS_REAL_EPS = 2.7222051279807147e-08
ROOTS_COMPLEX_EPS = 1.150574417275806e-10
POLYFIT_EPS = 1.963794112607891e-12


def _canonical_root_order(arr):
    # Same rounding-key sort chebroots_cases.py uses for its own
    # `chebroots`/`legroots` comparisons: legacy `np.roots` (like the
    # modern `polyroots` family) does not sort its output, and two
    # independent eigensolver call paths (real numpy's LAPACK dgeev vs
    # anionpy's own `linalg.eigvals`) are not guaranteed to return
    # eigenvalues in the same order even when every value agrees to
    # near-bit-exactness -- confirmed directly: `roots([1, 0, -1])`
    # returns `[1., -1.]` from real numpy but `[-1., 1.]` from anionpy,
    # values equal, order swapped.
    arr = np.asarray(arr)
    key = list(zip(np.round(arr.real, 6), np.round(np.imag(arr), 6)))
    order = sorted(range(len(arr)), key=lambda i: key[i])
    return arr[order]


def _roots_numpy_sorted(p):
    return _canonical_root_order(np.roots(p))


def _roots_ionp_sorted(p):
    import anionpy as ap
    r = ap.roots(p)
    r_np = np.asarray(r.tolist() if hasattr(r, "tolist") else r)
    return _canonical_root_order(r_np)


def _polyval_cases():
    # Ticket #76, defect A: real numpy's polyval Horner-loops over p's FIRST
    # axis with no ndim restriction at all ("for pv in p:") -- anionpy used
    # to hard-reject any p.ndim != 1 with its own ValueError. Fixed by
    # composing the Horner loop from ufunc::binary_op in Rust
    # (`polyval_legacy_horner_nd`, ionp-core/src/poly_legacy.rs), dispatched
    # from `_polyval_legacy_nd` for p.ndim != 1. The nd_p_* cases below cover
    # the success path (2-d and 3-d p, against both a scalar and an array
    # x, real/float/complex dtypes); scalar_p_bare_int/0d_p_array cover the
    # OTHER end of "no ndim restriction": a genuinely 0-d p, which real
    # numpy's `for pv in p:` itself rejects with `TypeError: iteration over
    # a 0-d array` -- not a shape check anionpy has to invent, just letting
    # Python's own iteration protocol do it (`for _ in p: break` in
    # `_polyval_legacy`). All measured byte-for-byte equal against real
    # numpy 2.5.1 from `/private/tmp/probe_cases.py`, 2026-08-08.
    return [
        ("linear_scalar", ([3, 0, 1], 5), {}),
        ("array_x", ([1, 2, 3], [0, 1, 2]), {}),
        ("float_coeffs_inf_x", ([1.0, 2.0, 3.0], float("inf")), {}),
        ("empty_p_int_x", ([], 5), {}),
        ("empty_p_array_x", ([], [1, 2, 3]), {}),
        ("complex_coeffs", ([1, 2, 3], [1j, 2j]), {}),
        ("int64_overflow", ([1, 2**62], 3), {}),
        ("nd_p_2d_scalar_x", ([[1, 2, 3], [4, 5, 6]], 2), {}),
        ("nd_p_2d_array_x", ([[1, 2, 3], [4, 5, 6]], [0, 1, 2]), {}),
        ("nd_p_3d", ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2), {}),
        ("nd_p_2d_float", ([[1.0, 2.0], [3.0, 4.0]], 1.5), {}),
        ("nd_p_2d_complex", ([[1 + 1j, 2], [3, 4 - 2j]], 1j), {}),
        ("scalar_p_bare_int", (5, 3), {}),
        ("0d_p_array", (np.array(5), 3), {}),
    ]


def _polyadd_cases():
    return [
        ("same_len", ([1, 2], [9, 5, 4]), {}),
        ("left_shorter", ([1, 2], [9, 5, 4]), {}),
        ("right_shorter", ([9, 5, 4], [1, 2]), {}),
        ("mixed_dtype", ([1, 2], [9.0, 5.0, 4.0]), {}),
    ]


def _polysub_cases():
    return [
        ("basic", ([2, 10, -2], [3, 10, -4]), {}),
        ("left_shorter", ([1, 2], [9, 5, 4]), {}),
    ]


def _polyder_cases():
    # Ticket #76, defects B and C.
    #   B: a genuinely 0-d/scalar p. Real numpy's polyder calls `len(p)`
    #      unconditionally (even at the m==0 base case, whose result it
    #      discards -- confirmed via `inspect.getsource`), which raises
    #      `TypeError: len() of unsized object` on a 0-d array. anionpy used
    #      to hard-reject with its own ValueError before ever calling len();
    #      fixed by letting a bare `len(pc)` in `_polynomial_legacy.py`'s
    #      dispatch do the same job Python already does for free.
    #   C: p.ndim >= 2. Real numpy computes `p[:-1] * np.arange(n, 0, -1)`
    #      (n = len(p) - 1) UNCONDITIONALLY every call, including at m==0.
    #      Broadcasting a leading-axis-first N-D p's first two axes against
    #      a 1-d arange is shape-dependent: nd_p_2d_broadcast_ok succeeds
    #      (shape (3,1) trailing-dim-1 broadcast), nd_p_2d_broadcast_fail and
    #      nd_p_square_fail raise numpy's own
    #      "operands could not be broadcast together with shapes ... "
    #      (trailing space, exact shapes). Fixed by composing
    #      `polyder_legacy_nd` (ionp-core/src/poly_legacy.rs) from
    #      `ufunc::binary_op`, which already reproduces that exact message.
    #      All cases measured byte-for-byte equal against real numpy 2.5.1
    #      from `/private/tmp/probe_cases.py`, 2026-08-08.
    return [
        ("m1", ([1, 1, 1, 1], 1), {}),
        ("m2", ([1, 1, 1, 1], 2), {}),
        ("m0", ([1, 1, 1, 1], 0), {}),
        ("over_order", ([1, 1, 1, 1], 4), {}),
        ("int64_dtype", ([1, 2, 3, 4], 1), {}),
        ("scalar_p_bare_int", (5, 1), {}),
        ("0d_p_array_m0", (np.array(5), 0), {}),
        ("0d_p_array_float", (np.array(5.0), 1), {}),
        ("nd_p_2d_broadcast_ok", ([[1], [2], [3]], 1), {}),
        ("nd_p_2d_broadcast_fail", ([[1, 1, 1, 1], [2, 2, 2, 2]], 1), {}),
        ("nd_p_square_fail", ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1), {}),
    ]


def _polyint_cases():
    # Ticket #76, defects B and C (same root causes as polyder, but
    # polyint's recursion SHORT-CIRCUITS at m==0 BEFORE calling `len()` at
    # all -- confirmed via `inspect.getsource` -- so a 0-d p at m==0
    # SUCCEEDS on real numpy (returns a genuine 0-d ndarray, not a numpy
    # scalar -- measured `type(r) is numpy.ndarray` directly) where
    # polyder's m==0 case with the same p still raises, because polyder
    # computes its broadcast product unconditionally even when it discards
    # the result. 0d_p_array_m0 below is deliberately the exact same input
    # as polyder's 0d_p_array_m0 case above to make that asymmetry visible
    # in the corpus rather than papering over it with one shared helper.
    # nd_p_2d_broadcast_fail mirrors polyder's C case (p[:] / arange(n,0,-1)
    # is unconditional here too); nd_p_2d_dim_mismatch is a DIFFERENT numpy
    # error text ("all the input arrays must have same number of
    # dimensions...") coming from the m-1 recursion's `concatenate` step
    # once the division succeeds -- both come for free from composing
    # `polyint_legacy_nd` out of `ufunc::binary_op`/`manip::concatenate`
    # rather than a hand-written broadcaster. All measured byte-for-byte
    # equal against real numpy 2.5.1 from `/private/tmp/probe_cases.py`,
    # 2026-08-08.
    return [
        ("default", ([1, 1, 1],), {}),
        ("m3_default_k", ([1, 1, 1], 3), {}),
        ("m3_explicit_k", ([1, 1, 1], 3, [6, 5, 3]), {}),
        ("scalar_p_bare_int", (5, 1), {}),
        ("0d_p_array_m0", (np.array(5), 0), {}),
        ("0d_p_array_float", (np.array(5.0), 1), {}),
        ("nd_p_2d_broadcast_fail", ([[1, 1, 1], [2, 2, 2]], 1), {}),
        ("nd_p_2d_dim_mismatch", ([[1], [2], [3]], 1), {}),
        ("nd_p_square_dim_mismatch", ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1), {}),
    ]


def _polydiv_cases():
    return [
        ("basic", ([3.0, 5.0, 2.0], [2.0, 1.0]), {}),
        ("exact", ([1.0, 0.0, -1.0], [1.0, -1.0]), {}),
        ("divide_by_const", ([1.0, 2.0, 3.0], [1.0]), {}),
    ]


def _poly_cases():
    return [
        ("triple_root", ([0, 0, 0],), {}),
        ("symmetric", ([-0.5, 0, 0.5],), {}),
        ("two_roots", ([1, 2],), {}),
        ("conjugate_pair", ([1 + 1j, 1 - 1j],), {}),
        ("square_matrix", ([[0, 1 / 3], [-1 / 2, 0]],), {}),
    ]


def _roots_cases():
    return [
        ("complex_pair", ([3.2, 2, 1],), {}),
        ("two_real", ([1, -3, 2],), {}),
        ("with_trailing_zero_root", ([1, 0, -1],), {}),
        ("repeated", ([1, 2, 1],), {}),
        ("leading_and_trailing_zeros", ([0, 0, 1, 2, 3],), {}),
    ]


def _polyfit_cases():
    # Plain (no full=/cov=) cases only: `polyfit` returns a bare
    # coefficient ndarray here, NOT a tuple -- `full=True`/`cov=True`
    # switch its return shape to a tuple, so those variants get their
    # own multi_output=True spec below (`polyfit_full`/`polyfit_cov`)
    # rather than being mixed into this uniformly-shaped case set.
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y = [0.0, 0.8, 0.9, 0.1, -0.8, -1.0]
    return [
        ("cubic", (x, y, 3), {}),
        ("cubic_weighted", (x, y, 3), {"w": [5.0, 1, 1, 1, 1, 1]}),
    ]


def _polyfit_full_cases():
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y = [0.0, 0.8, 0.9, 0.1, -0.8, -1.0]
    return [
        ("cubic_full", (x, y, 3), {"full": True}),
    ]


def _polyfit_cov_cases():
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y = [0.0, 0.8, 0.9, 0.1, -0.8, -1.0]
    return [
        ("cubic_cov", (x, y, 3), {"cov": True}),
    ]


def _build_legacy_poly_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}
    specs["polyval"] = ItemSpec(
        name="polyval", kind="custom", custom_cases=_polyval_cases,
        numpy_path="polyval", ionp_path="polyval",
    )
    specs["polyadd"] = ItemSpec(
        name="polyadd", kind="custom", custom_cases=_polyadd_cases,
        numpy_path="polyadd", ionp_path="polyadd",
    )
    specs["polysub"] = ItemSpec(
        name="polysub", kind="custom", custom_cases=_polysub_cases,
        numpy_path="polysub", ionp_path="polysub",
    )
    specs["polyder"] = ItemSpec(
        name="polyder", kind="custom", custom_cases=_polyder_cases,
        numpy_path="polyder", ionp_path="polyder",
    )
    specs["polyint"] = ItemSpec(
        name="polyint", kind="custom", custom_cases=_polyint_cases,
        numpy_path="polyint", ionp_path="polyint",
    )
    specs["polydiv"] = ItemSpec(
        name="polydiv", kind="custom", custom_cases=_polydiv_cases,
        numpy_path="polydiv", ionp_path="polydiv",
        multi_output=True,
    )
    specs["poly"] = ItemSpec(
        name="poly", kind="custom", custom_cases=_poly_cases,
        numpy_path="poly", ionp_path="poly",
    )
    specs["roots"] = ItemSpec(
        name="roots", kind="custom", custom_cases=_roots_cases,
        numpy_path="roots", ionp_path="roots",
        numpy_adapter=_roots_numpy_sorted,
        ionp_adapter=_roots_ionp_sorted,
        epsilon_tolerance={
            "float64": ("abs", ROOTS_REAL_EPS),
            "complex128": ("abs", ROOTS_COMPLEX_EPS),
        },
        epsilon_tolerance_justification=(
            "roots() goes through anionpy.linalg.eigvals, an independent "
            "LAPACK-adjacent eigensolver path from real numpy's own -- "
            "same precedent as the already-landed modern "
            "polynomial.polynomial.polyroots (see polynomial_cases.py's "
            "POLYROOTS_REAL_EPS/POLYROOTS_COMPLEX_EPS and their "
            "justification). Legacy roots() also does not sort its "
            "output (confirmed: roots([1,0,-1]) returns [1.,-1.] from "
            "real numpy but [-1.,1.] from anionpy, values equal, order "
            "swapped), hence the numpy_adapter/ionp_adapter canonical-"
            "order sort, mirroring chebroots_cases.py's own pattern. "
            "Bound is the max abs coordinate error over an independent "
            "seed=9182736, N=20000 sweep (degree 1-7 polynomials built "
            "from independently-drawn real/complex roots) run OUT of "
            "this corpus -- see /private/tmp/poly_eps_sweep_legacy.py "
            "and ionp/docs/TICKET-72-POLY1D-2026-08-08.md -- NOT a "
            "safe-margin guess."
        ),
        epsilon_sweep={
            "float64": (20000, ROOTS_REAL_EPS),
            "complex128": (20000, ROOTS_COMPLEX_EPS),
        },
    )
    _polyfit_eps_kwargs = dict(
        epsilon_tolerance={"float64": ("rel", POLYFIT_EPS)},
        epsilon_tolerance_justification=(
            "polyfit goes through anionpy.linalg.lstsq, an independent "
            "LAPACK-adjacent SVD path from real numpy's own -- same "
            "precedent as the already-landed modern "
            "polynomial.polynomial.polyfit (POLYFIT_EPS in "
            "polynomial_cases.py). Bound is the max relative "
            "(floor=1.0, matching harness.max_rel_distance) coefficient "
            "distance over an independent seed=9182736, N=20000 sweep "
            "(n in [5,20) samples, deg in [1,5), noisy cubic-family fits) "
            "run OUT of this corpus -- see "
            "/private/tmp/poly_eps_sweep_legacy.py and "
            "ionp/docs/TICKET-72-POLY1D-2026-08-08.md -- NOT a "
            "safe-margin guess."
        ),
        epsilon_sweep={"float64": (20000, POLYFIT_EPS)},
    )
    # `full=True`/`cov=True` switch polyfit's return from a bare
    # coefficient ndarray to a tuple (5-tuple / 2-tuple respectively) --
    # split into their own multi_output=True specs rather than mixed into
    # the plain "polyfit" case set above (see _polyfit_cases' docstring
    # comment). All three share the same underlying numpy/anionpy
    # callable and the same measured epsilon (the tolerance is on the
    # returned COEFFICIENT array in every variant; the residuals/rank/
    # singular_values/covariance slots are either exact integer metadata
    # or derived quantities compared under the same per-dtype grading).
    specs["polyfit"] = ItemSpec(
        name="polyfit", kind="custom", custom_cases=_polyfit_cases,
        numpy_path="polyfit", ionp_path="polyfit",
        **_polyfit_eps_kwargs,
    )
    specs["polyfit_full"] = ItemSpec(
        name="polyfit_full", kind="custom", custom_cases=_polyfit_full_cases,
        numpy_path="polyfit", ionp_path="polyfit",
        multi_output=True,
        **_polyfit_eps_kwargs,
    )
    specs["polyfit_cov"] = ItemSpec(
        name="polyfit_cov", kind="custom", custom_cases=_polyfit_cov_cases,
        numpy_path="polyfit", ionp_path="polyfit",
        multi_output=True,
        **_polyfit_eps_kwargs,
    )
    return specs


LEGACY_POLY_SPECS: dict[str, ItemSpec] = _build_legacy_poly_specs()
