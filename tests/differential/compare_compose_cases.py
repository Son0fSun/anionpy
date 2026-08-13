"""Differential coverage for `anionpy/_compare_compose.py`: `isclose`.

Same collision-checked-merge pattern as manip_compose_cases.py -- builds a
dict of ItemSpecs, merged into registry.REGISTRY from the bottom of
registry.py. `isclose` is a Python-only wrapper over already-existing/
already-verified anionpy primitives (`asarray`, `multiply`, `subtract`,
`absolute`, `add`, `less_equal`, `equal`, `isfinite`, `isnan`,
`logical_and`, `logical_or`) -- no Rust changes. See `_compare_compose.py`'s
module docstring for the numpy source this was transcribed from and the one
deliberate substitution (multiply-by-1.0 standing in for the known-broken
result_type/promote_types/can_cast trio).

kind="custom" (not "binary"): `isclose` takes rtol/atol/equal_nan keyword
arguments the generic binary corpus doesn't vary, and needs 0-d/scalar
inputs (numpy's own documented "if both a and b are scalars, returns a
single boolean value" path) which corpus.binary_corpus() does not
specifically target.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401  (sys.path wiring, see that module's docstring)
from registry import ItemSpec


def isclose_cases():
    out = []
    rng = np.random.default_rng(20260803)

    # --- special-value grid per real dtype family, both directions ------
    special = {
        "int8": [-128, -1, 0, 1, 127],
        "int32": [-2**31, -1, 0, 1, 2**31 - 1],
        "int64": [-2**63, 0, 2**63 - 1],
        "uint8": [0, 1, 255],
        "uint64": [0, 1, 2**64 - 1],
        "float16": [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 5e-8],
        "float32": [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 1e-9, 2e-9],
        "float64": [np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 1e-9, 2e-9, 5e-324],
        "complex64": [complex(np.nan, 0), complex(np.inf, -np.inf), 0 + 0j, 1 + 1j, 1e-9 + 1e-9j],
        "complex128": [complex(np.nan, 0), complex(np.inf, -np.inf), 0 + 0j, 1 + 1j, 1e-9 + 1e-9j],
    }
    for dt, vals in special.items():
        a = np.array(vals, dtype=dt)
        b = np.array(list(reversed(vals)), dtype=dt)
        for equal_nan in (False, True):
            out.append((f"special/{dt}/equal_nan_{equal_nan}", (a, b), {"equal_nan": equal_nan}))
        out.append((f"special/{dt}/rtol0_atol0", (a, b), {"rtol": 0.0, "atol": 0.0}))
        out.append((f"special/{dt}/loose_tol", (a, b), {"rtol": 1e-3, "atol": 1e-3}))

    # --- dtype sweep, near-equal random data ------------------------------
    for dt in ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
               "uint32", "uint64", "float16", "float32", "float64",
               "complex64", "complex128"]:
        if dt == "bool":
            a = rng.integers(0, 2, size=20).astype(bool)
            b = rng.integers(0, 2, size=20).astype(bool)
        elif dt.startswith("uint"):
            a = rng.integers(0, 200, size=20).astype(dt)
            b = rng.integers(0, 200, size=20).astype(dt)
        elif dt.startswith("int"):
            a = rng.integers(-100, 100, size=20).astype(dt)
            b = rng.integers(-100, 100, size=20).astype(dt)
        elif dt.startswith("complex"):
            a = (rng.standard_normal(20) + 1j * rng.standard_normal(20)).astype(dt)
            b = (rng.standard_normal(20) + 1j * rng.standard_normal(20)).astype(dt)
        else:
            a = rng.standard_normal(20).astype(dt)
            b = (a + rng.standard_normal(20).astype(dt) * 1e-6).astype(dt)
        out.append((f"sweep/{dt}", (a, b), {}))

    # --- 0-d / true python-scalar inputs (numpy's "returns a single
    #     boolean value" documented path) ---------------------------------
    out.append(("0d_int_equal", (np.array(5), np.array(5)), {}))
    out.append(("0d_float_close", (np.array(5.0), np.array(5.0000001)), {}))
    out.append(("0d_float_far", (np.array(1e-9), np.array(0.0)), {}))
    out.append(("python_scalar_pair", (3.0, 3.0000001), {}))
    out.append(("python_scalar_int", (3, 3), {}))

    # --- shapes / broadcasting / contiguity --------------------------------
    a2 = rng.standard_normal((3, 4)).astype(np.float64)
    b2 = rng.standard_normal((1, 4)).astype(np.float64)
    out.append(("broadcast_3x4_1x4", (a2, b2), {}))
    out.append(("transposed_operands", (np.ascontiguousarray(a2.T), np.ascontiguousarray(b2.T)), {}))
    out.append(("fortran_operand", (a2.T, b2.T), {}))

    # --- empty --------------------------------------------------------------
    out.append(("empty", (np.array([], dtype=np.float64), np.array([], dtype=np.float64)), {}))

    # --- array-valued rtol/atol (numpy explicitly documents these as
    #     array_like, not just scalars) -------------------------------------
    a3 = np.array([1.0, 2.0, 3.0])
    b3 = np.array([1.01, 2.02, 3.5])
    out.append(("array_rtol", (a3, b3), {"rtol": np.array([0.02, 0.02, 0.02]), "atol": 0.0}))

    # --- non-finite atol/rtol: real numpy warns via geterr() but still
    #     returns a real boolean result (see _compare_compose.py's
    #     docstring -- the warning channel itself is not replicated, only
    #     verified separately to still agree on VALUE) ----------------------
    out.append(("nonfinite_atol", (np.array(1.0), np.array(2.0)), {"atol": np.nan}))
    out.append(("nonfinite_rtol", (np.array([1.0, 2.0]), np.array([1.0, 2.5])), {"rtol": np.nan}))

    # --- REGRESSION: complex non-finite self-compare (2026-08-03) ----------
    # Pins the exact defect that got the first "isclose" declaration REVOKED
    # the same day it was declared (f82c2f4 -> 12bfa48): the promotion step
    # substituted `anionpy.multiply(y, 1.0)` (ARITHMETIC) for numpy's
    # `asanyarray(y, dtype=result_type(y, 1.))` (a CAST). Complex multiply
    # computes `imag = a.real*b.imag + a.imag*b.real`, so
    # `multiply(inf+0j, 1.0)` evaluated to `inf+nanj` -- the corrupted
    # imaginary part then failed the `x == y` branch numpy's isclose routes
    # all non-finite input through. Live-verified 2026-08-03, all four MISMATCHED
    # under the old multiply-based composition (np.isclose -> True in every
    # case, old anionpy.isclose -> False in every case):
    #   inf+0j, -inf+0j, infj, inf+infj  (each compared against itself)
    # Fixed by replacing the multiply-based promotion with a literal NEP-50
    # promotion-dtype table + a genuine `anionpy.asarray(y, dtype=...)` cast --
    # see `_ISCLOSE_Y_PROMOTE_DTYPE` in `anionpy/_compare_compose.py`. If that
    # cast regresses back to arithmetic, this case's mismatch will bite: the
    # "run.py --selftest"-style guard-bite proof deliberately restored the
    # multiply substitution and confirmed this exact case flips to "fail" --
    # see the isclose declaration's regenerated notes in
    # `anionpy/_state/toplevel.py`.
    #
    # The independent-real/imag matrix below covers every combination of
    # {finite, -0.0, +inf, -inf, nan} across the real and imaginary parts of
    # BOTH operands, self-compared and cross-compared, for complex64 and
    # complex128, equal_nan both ways -- the exact axis that caught the bug
    # (per-value special-case grids that only vary one part at a time, like
    # the `special/complex64` case above, do NOT exercise independent
    # real/imag combinations and would not have caught this).
    nonfinite_parts = [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan]
    for cdt in ("complex64", "complex128"):
        a_vals = [complex(re, im) for re in nonfinite_parts for im in nonfinite_parts]
        b_vals = list(reversed(a_vals))
        a = np.array(a_vals, dtype=cdt)
        b = np.array(b_vals, dtype=cdt)
        a_self = np.array(a_vals, dtype=cdt)
        for equal_nan in (False, True):
            out.append((f"complex_nonfinite_matrix/{cdt}/vs_reversed/equal_nan_{equal_nan}",
                        (a, b), {"equal_nan": equal_nan}))
            out.append((f"complex_nonfinite_matrix/{cdt}/self/equal_nan_{equal_nan}",
                        (a, a_self), {"equal_nan": equal_nan}))
        # the four originally-mismatching values, self-compared, isolated in
        # their own 1-element-per-case group so a regression here can't hide
        # behind an otherwise-passing larger array
        for label, val in [
            ("inf_plus_0j", complex(np.inf, 0.0)),
            ("neg_inf_plus_0j", complex(-np.inf, 0.0)),
            ("infj", complex(0.0, np.inf)),
            ("inf_plus_infj", complex(np.inf, np.inf)),
        ]:
            va = np.array(val, dtype=cdt)
            vb = np.array(val, dtype=cdt)
            out.append((f"complex_nonfinite_selfcompare/{cdt}/{label}", (va, vb), {}))

    return out


def asarray_chkfinite_cases():
    out = []

    # --- dtype sweep, finite data (never raises) ----------------------------
    for dt in ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
               "uint32", "uint64", "float16", "float32", "float64",
               "complex64", "complex128"]:
        if dt == "bool":
            data = [True, False, True]
        elif dt.startswith("complex"):
            data = [1 + 2j, 3 - 4j, 0j]
        else:
            data = [0, 1, 2]
        out.append((f"finite_sweep/{dt}", (np.array(data, dtype=dt),), {}))

    # --- non-finite data, per inexact dtype, both element positions --------
    for dt in ["float16", "float32", "float64"]:
        out.append((f"raises_inf/{dt}", (np.array([1.0, np.inf, 3.0], dtype=dt),), {}))
        out.append((f"raises_neg_inf/{dt}", (np.array([-np.inf, 1.0, 2.0], dtype=dt),), {}))
        out.append((f"raises_nan/{dt}", (np.array([1.0, 2.0, np.nan], dtype=dt),), {}))
    for dt in ["complex64", "complex128"]:
        out.append((f"raises_inf_real/{dt}", (np.array([1 + 1j, complex(np.inf, 0)], dtype=dt),), {}))
        out.append((f"raises_inf_imag/{dt}", (np.array([1 + 1j, complex(0, np.inf)], dtype=dt),), {}))
        out.append((f"raises_nan/{dt}", (np.array([1 + 1j, complex(np.nan, 0)], dtype=dt),), {}))

    # --- denormals: finite, must NOT raise -----------------------------------
    out.append(("denormal_float64", (np.array([5e-324, -5e-324], dtype=np.float64),), {}))
    out.append(("denormal_float32", (np.array([np.float32(1e-40)], dtype=np.float32),), {}))

    # --- dtype=/order= conversion (valid order values only -- 'Z'/invalid
    #     order is a separate, already-disclosed, pre-existing gap in
    #     anionpy.asarray itself, see asarray_chkfinite's own docstring) --------
    out.append(("dtype_cast_int_to_float64", (np.array([1, 2, 3], dtype=np.int32),), {"dtype": "float64"}))
    out.append(("dtype_cast_int_to_float32", (np.array([1, 2, 3], dtype=np.int64),), {"dtype": "float32"}))
    a2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    for order in (None, "C", "F", "A", "K"):
        out.append((f"order_{order}", (a2,), {"order": order}))
    out.append(("order_F_non_contig_input", (np.ascontiguousarray(a2.T),), {"order": "F"}))

    # --- 0-d / scalar / empty ------------------------------------------------
    out.append(("0d_finite", (np.array(5.0),), {}))
    out.append(("0d_nonfinite", (np.array(np.inf),), {}))
    out.append(("python_scalar_finite", (5.0,), {}))
    out.append(("python_scalar_int", (5,), {}))
    out.append(("empty_float64", (np.array([], dtype=np.float64),), {}))
    out.append(("empty_complex128", (np.array([], dtype=np.complex128),), {}))

    # --- already-an-array, no conversion needed (shares_memory check lives
    #     in the differential harness's aliasing checks, not this label) ----
    out.append(("already_array_passthrough", (np.array([1.0, 2.0, 3.0]),), {}))

    # --- list/tuple/nested input (not pre-materialized numpy arrays) -------
    out.append(("plain_list", ([1.0, 2.0, np.inf],), {}))
    out.append(("nested_list", ([[1.0, 2.0], [3.0, np.nan]],), {}))

    return out


def _build_compare_compose_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}
    specs["isclose"] = ItemSpec(name="isclose", kind="custom", custom_cases=isclose_cases)
    specs["asarray_chkfinite"] = ItemSpec(
        name="asarray_chkfinite", kind="custom", custom_cases=asarray_chkfinite_cases
    )
    return specs


COMPARE_COMPOSE_SPECS = _build_compare_compose_specs()
