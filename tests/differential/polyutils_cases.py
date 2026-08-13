"""anionpy.polynomial.polyutils.* differential registry entries.

NEW FILE (permitted, same as linalg_cases.py/manip_cases.py): builds a
`POLYUTILS_SPECS: dict[str, ItemSpec]` dict, merged into `registry.REGISTRY`
at the bottom of registry.py (collision-checked, same pattern as every
other `*_cases.py` merge in that file).

Covers the six `numpy.polynomial.polyutils.*` items implemented in
`anionpy/polynomial/polyutils.py`: `trimseq`, `as_series`, `trimcoef`,
`getdomain`, `mapparms`, `mapdomain`. `format_float` is deliberately not
implemented (see that module's docstring) and has no entry here.

`as_series` returns a variable-length Python `list` under both real numpy
and anionpy -- not a `tuple` -- so, exactly like `manip_cases.py`'s
`split`/`array_split` family, it is wrapped through `tuple(...)` by a
dedicated `numpy_adapter`/`ionp_adapter` pair and declared
`multi_output=True` (`compare_multi_output` requires `isinstance(x, tuple)`
on both sides; a bare `list` fails that check even when every element
matches).

Every generator below deliberately varies: dtype (float16/32/64, int,
uint, complex64/128, bool-rejection), degree, empty and length-1
coefficient arrays, leading/trailing zeros, and NaN/inf coefficients --
per the task's required corpus axes. Every numeric case is declared
`atol=0.0, rtol=0.0` (bit-exact) -- these are pure Python-level
promotion/slicing/arithmetic functions with no floating-point
accumulation of their own (the one exception, `mapparms`/`mapdomain`'s
scalar arithmetic, was independently confirmed to reproduce the exact
`ZeroDivisionError` real numpy raises for a degenerate domain -- see the
`mapparms_cases`/`mapdomain_cases` degenerate-domain cases below), so no
epsilon/ULP tolerance was ever needed or measured for this file.
"""
from __future__ import annotations

import numpy as np

from registry import ItemSpec

# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------

INT_C = [1, 2, 3]
UINT_ARR = np.array([1, 2, 3], dtype=np.uint32)
FLOAT16_ARR = np.array([1.0, 2.0, 3.0], dtype=np.float16)
FLOAT32_ARR = np.array([1.0, 2.0, 3.0], dtype=np.float32)
FLOAT_C = [1.0, 2.0, 3.0]
TRAILING_ZERO_C = [1.0, 2.0, 3.0, 0.0, 0.0]
LEADING_ZERO_C = [0.0, 0.0, 1.0, 2.0]
ALL_ZERO_C = [0.0, 0.0, 0.0]
COMPLEX64_ARR = np.array([1 + 1j, 2 - 1j], dtype=np.complex64)
COMPLEX_C = [1 + 1j, 2 - 1j, 0.5j]
BOOL_C = [True, False, True]
SINGLE_C = [5.0]
NAN_C = [1.0, float("nan"), 3.0]
INF_C = [1.0, float("inf"), -float("inf")]


def _std(pairs):
    return [(label, args, kwargs) for label, args, kwargs in pairs]


# ---------------------------------------------------------------------------
# trimseq
# ---------------------------------------------------------------------------

def trimseq_cases():
    return _std([
        ("trailing_zeros", (np.array(TRAILING_ZERO_C),), {}),
        ("leading_zeros_no_trim", (np.array(LEADING_ZERO_C),), {}),
        ("all_zero", (np.array(ALL_ZERO_C),), {}),
        ("no_trim_needed", (np.array(FLOAT_C),), {}),
        ("empty", (np.array([], dtype=np.float64),), {}),
        ("single_nonzero", (np.array([5.0]),), {}),
        ("single_zero", (np.array([0.0]),), {}),
        ("complex_trailing_zero", (np.array(COMPLEX_C + [0j]),), {}),
        ("int_trailing_zero", (np.array(INT_C + [0]),), {}),
        ("trailing_nan_not_trimmed", (np.array([1.0, 0.0, float("nan")]),), {}),
        ("trailing_inf_not_trimmed", (np.array([1.0, 0.0, float("inf")]),), {}),
        ("float16", (FLOAT16_ARR,), {}),
    ])


# ---------------------------------------------------------------------------
# as_series (wrapped through tuple(...), multi_output=True)
# ---------------------------------------------------------------------------

def as_series_cases():
    return _std([
        ("single_float", ([FLOAT_C],), {}),
        ("single_int", ([INT_C],), {}),
        ("int_and_float", ([INT_C, FLOAT_C],), {}),
        ("float16_and_int", ([FLOAT16_ARR, INT_C],), {}),
        ("float32_and_float16", ([FLOAT32_ARR, FLOAT16_ARR],), {}),
        ("uint_and_float", ([UINT_ARR, FLOAT_C],), {}),
        ("int_and_complex", ([INT_C, COMPLEX_C],), {}),
        ("float32_and_complex64", ([FLOAT32_ARR, COMPLEX64_ARR],), {}),
        ("float16_and_complex64", ([FLOAT16_ARR, COMPLEX64_ARR],), {}),
        ("all_complex128", ([COMPLEX_C, COMPLEX_C],), {}),
        ("trailing_zeros_trim_true", ([TRAILING_ZERO_C],), {}),
        ("trailing_zeros_trim_false", ([TRAILING_ZERO_C],), {"trim": False}),
        ("single_element", ([SINGLE_C],), {}),
        ("three_arrays_mixed", ([INT_C, FLOAT_C, TRAILING_ZERO_C],), {}),
        ("nan_coeff", ([NAN_C],), {}),
        ("inf_coeff", ([INF_C],), {}),
        ("bool_raises", ([BOOL_C],), {}),
        ("empty_raises", ([[]],), {}),
        ("one_empty_one_ok_raises", ([[], FLOAT_C],), {}),
        ("two_d_raises", ([np.array([[1.0, 2.0], [3.0, 4.0]])],), {}),
    ])


# ---------------------------------------------------------------------------
# trimcoef
# ---------------------------------------------------------------------------

def trimcoef_cases():
    return _std([
        ("trailing_zeros", (TRAILING_ZERO_C,), {}),
        ("all_zero", (ALL_ZERO_C,), {}),
        ("no_trim_needed", (FLOAT_C,), {}),
        ("with_tol", (list(TRAILING_ZERO_C) + [1e-10], 1e-8), {}),
        ("tol_kwarg", (list(TRAILING_ZERO_C) + [1e-10],), {"tol": 1e-8}),
        ("complex", (COMPLEX_C + [0j],), {}),
        ("int_input", (INT_C + [0],), {}),
        ("single_element", (SINGLE_C,), {}),
        ("negative_tol_raises", (FLOAT_C, -1.0), {}),
        ("bool_raises", (BOOL_C,), {}),
        ("empty_raises", ([],), {}),
    ])


# ---------------------------------------------------------------------------
# getdomain
# ---------------------------------------------------------------------------

def getdomain_cases():
    return _std([
        ("real_values", ([1.0, 5.0, 3.0, -2.0],), {}),
        ("single_value", ([3.0],), {}),
        ("int_values", ([1, 5, 3, -2],), {}),
        ("complex_values", ([1 + 1j, -2 + 3j, 0 - 1j],), {}),
        ("nan_value", ([1.0, float("nan"), 3.0],), {}),
        ("inf_value", ([1.0, float("inf"), -float("inf")],), {}),
        ("all_equal", ([2.0, 2.0, 2.0],), {}),
        ("float16", (FLOAT16_ARR,), {}),
        ("empty_raises", ([],), {}),
    ])


# ---------------------------------------------------------------------------
# mapparms
# ---------------------------------------------------------------------------

def mapparms_cases():
    return _std([
        ("basic", ([-1.0, 1.0], [0.0, 10.0]), {}),
        ("identity", ([-1.0, 1.0], [-1.0, 1.0]), {}),
        ("reversed_new", ([-1.0, 1.0], [10.0, 0.0]), {}),
        ("negative_old", ([-5.0, -2.0], [0.0, 1.0]), {}),
        ("integers", ([0, 10], [0, 1]), {}),
        ("degenerate_old_raises", ([1.0, 1.0], [0.0, 10.0]), {}),
        ("as_ndarray_domains", (np.array([-1.0, 1.0]), np.array([0.0, 10.0])), {}),
    ])


# ---------------------------------------------------------------------------
# mapdomain
# ---------------------------------------------------------------------------

def mapdomain_cases():
    return _std([
        ("list_x", ([-1.0, 0.0, 1.0], [-1.0, 1.0], [0.0, 10.0]), {}),
        ("scalar_x", (0.5, [-1.0, 1.0], [0.0, 10.0]), {}),
        ("int_scalar_x", (0, [-1, 1], [0, 10]), {}),
        ("complex_x", (0.5 + 0.5j, [-1.0, 1.0], [0.0, 10.0]), {}),
        ("array_x", (np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), [-1.0, 1.0], [0.0, 10.0]), {}),
        ("as_ndarray_domains", (np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 1.0]),
                                 np.array([0.0, 10.0])), {}),
        ("identity_domains", ([1.0, 2.0, 3.0], [-1.0, 1.0], [-1.0, 1.0]), {}),
        ("degenerate_old_raises", ([1.0], [2.0, 2.0], [0.0, 10.0]), {}),
    ])


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def _tuple_adapters(name: str):
    def numpy_adapter(*args, **kwargs):
        import numpy.polynomial.polyutils as npu
        return tuple(getattr(npu, name)(*args, **kwargs))

    def ionp_adapter(*args, **kwargs):
        from anionpy.polynomial import polyutils as pu
        return tuple(getattr(pu, name)(*args, **kwargs))

    return numpy_adapter, ionp_adapter


def _build_polyutils_specs() -> dict[str, ItemSpec]:
    specs: dict[str, ItemSpec] = {}

    specs["polynomial.polyutils.trimseq"] = ItemSpec(
        name="polynomial.polyutils.trimseq", kind="custom",
        custom_cases=trimseq_cases, atol=0.0, rtol=0.0,
        numpy_adapter=lambda *a, **k: __import__(
            "numpy.polynomial.polyutils", fromlist=["trimseq"]
        ).trimseq(*a, **k),
        ionp_adapter=lambda *a, **k: __import__(
            "anionpy.polynomial.polyutils", fromlist=["trimseq"]
        ).trimseq(*a, **k),
    )

    _as_series_np, _as_series_ionp = _tuple_adapters("as_series")
    specs["polynomial.polyutils.as_series"] = ItemSpec(
        name="polynomial.polyutils.as_series", kind="custom",
        custom_cases=as_series_cases,
        numpy_adapter=_as_series_np, ionp_adapter=_as_series_ionp,
        atol=0.0, rtol=0.0, multi_output=True,
    )

    specs["polynomial.polyutils.trimcoef"] = ItemSpec(
        name="polynomial.polyutils.trimcoef", kind="custom",
        custom_cases=trimcoef_cases, atol=0.0, rtol=0.0,
        numpy_adapter=lambda *a, **k: __import__(
            "numpy.polynomial.polyutils", fromlist=["trimcoef"]
        ).trimcoef(*a, **k),
        ionp_adapter=lambda *a, **k: __import__(
            "anionpy.polynomial.polyutils", fromlist=["trimcoef"]
        ).trimcoef(*a, **k),
    )

    specs["polynomial.polyutils.getdomain"] = ItemSpec(
        name="polynomial.polyutils.getdomain", kind="custom",
        custom_cases=getdomain_cases, atol=0.0, rtol=0.0,
        numpy_adapter=lambda *a, **k: __import__(
            "numpy.polynomial.polyutils", fromlist=["getdomain"]
        ).getdomain(*a, **k),
        ionp_adapter=lambda *a, **k: __import__(
            "anionpy.polynomial.polyutils", fromlist=["getdomain"]
        ).getdomain(*a, **k),
    )

    _mapparms_np, _mapparms_ionp = _tuple_adapters("mapparms")
    specs["polynomial.polyutils.mapparms"] = ItemSpec(
        name="polynomial.polyutils.mapparms", kind="custom",
        custom_cases=mapparms_cases,
        numpy_adapter=_mapparms_np, ionp_adapter=_mapparms_ionp,
        atol=0.0, rtol=0.0, multi_output=True,
    )

    specs["polynomial.polyutils.mapdomain"] = ItemSpec(
        name="polynomial.polyutils.mapdomain", kind="custom",
        custom_cases=mapdomain_cases, atol=0.0, rtol=0.0,
        numpy_adapter=lambda *a, **k: __import__(
            "numpy.polynomial.polyutils", fromlist=["mapdomain"]
        ).mapdomain(*a, **k),
        ionp_adapter=lambda *a, **k: __import__(
            "anionpy.polynomial.polyutils", fromlist=["mapdomain"]
        ).mapdomain(*a, **k),
    )

    return specs


POLYUTILS_SPECS: dict[str, ItemSpec] = _build_polyutils_specs()
