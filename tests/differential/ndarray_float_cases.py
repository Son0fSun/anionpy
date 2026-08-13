"""NEW test-cases file: `ndarray.__float__` (`float(arr)`), 2026-08-03
(Monday, ndarray-attrs/dunders coverage-audit task).

WHY THIS NEEDED A NEW FILE
---------------------------------------------------------------------------
`ndarray.__float__` had ZERO differential coverage before this task --
absent from `registry.py`, `ndarray_attrs_cases.py`, and every other
`*_cases.py` file (confirmed by grep across the whole `tests/differential/`
tree). It doesn't fit `kind="unary"`/`kind="binary_op"`/`kind="method"`
cleanly: numpy's `float(arr)` (a) only succeeds for a 0-d (or, in modern
numpy, ALSO no size>1 1-d) array, (b) must raise `TypeError` for size>1 and
for complex dtypes, and (c) returns a plain Python `float`, not an
ndarray -- so this is a `kind="custom"` item with its own `custom_cases()`
callable and `scalar_like=True` (per `ItemSpec`'s own docstring in
registry.py), following the same shape as `ndarray.item`/`ndarray.tolist`
in `ndarray_attrs_cases.py`.

MEASUREMENT (out-of-corpus probe, /tmp/nda_float_probe.py, 33/33 matching)
---------------------------------------------------------------------------
Every real numpy scalar dtype (int8/16/32/64, uint8/16/32/64, float16/32/64,
bool) converts to the expected Python float, both 0-d. `nan`/`inf`/`-inf`
round-trip exactly (`nan != nan` handled by the probe's own nan-safe
compare, not by tolerance -- both sides independently produce IEEE `nan`
and the probe treats that pairing as a match rather than papering over a
real mismatch). A size-1 1-d array (`np.array([7.0])`) raises
`TypeError: only 0-dimensional arrays can be converted to Python scalars`
on BOTH sides -- modern numpy (2.x) removed the old "deprecated but still
works" size-1 exemption entirely, and anionpy already matches this, not the
older behavior. A size>1 1-d array and an empty array both raise the same
`TypeError` with the identical message on both sides. A complex128 0-d
array raises `TypeError: float() argument must be a string or a real
number, not 'complex'` on both sides byte-for-byte (exception message
compared exactly, not just exception type).

NOT verified: multi-dimensional 0-d-shaped edge cases beyond plain
`np.array(x)` (there are none -- 0-d is 0-d), and no F-order/non-contiguous
axis applies (a 0-d array has no strides to vary). Non-contiguity /
negative-stride / F-order axes are structurally inapplicable to this item,
not skipped.
"""
from __future__ import annotations

import math

import numpy as np

from registry import ItemSpec


def _nan_safe(v):
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return v


def _make_case(label, np_array_factory):
    """Returns (label, args, kwargs) where args=(np_array_factory,) --
    the actual numpy vs anionpy dispatch happens in the adapters below, which
    take the SAME zero-arg factory and build a fresh array on whichever
    side is calling, so numpy and anionpy never share (and can't accidentally
    mutate) the same underlying object.
    """
    return (label, (np_array_factory,), {})


def _float_numpy_adapter(factory):
    arr = factory()
    try:
        return ("ok", _nan_safe(float(arr)))
    except BaseException as e:  # noqa: BLE001 -- must match anionpy's own PyO3 panics too
        return ("err", type(e).__name__, str(e))


def _float_ionp_adapter(factory):
    import anionpy
    arr = factory()
    ia = anionpy.array(arr)
    try:
        return ("ok", _nan_safe(float(ia)))
    except BaseException as e:  # noqa: BLE001
        return ("err", type(e).__name__, str(e))


def _float_custom_cases():
    cases = []
    dtypes = [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64, np.bool_,
    ]
    for dt in dtypes:
        cases.append(_make_case(f"scalar_pos/{dt.__name__}", lambda dt=dt: np.array(3, dtype=dt)))
        cases.append(_make_case(f"scalar_zero/{dt.__name__}", lambda dt=dt: np.array(0, dtype=dt)))
    cases.append(_make_case("float_frac", lambda: np.array(1.5, dtype=np.float64)))
    cases.append(_make_case("float_neg_frac32", lambda: np.array(-2.25, dtype=np.float32)))
    cases.append(_make_case("nan", lambda: np.array(np.nan)))
    cases.append(_make_case("posinf", lambda: np.array(np.inf)))
    cases.append(_make_case("neginf", lambda: np.array(-np.inf)))
    # Signed zero (#86): this item's `scalar_like=True` result is a plain
    # Python `float`, which routes through `harness._compare_scalar_like`
    # -- the exact comparison function this ticket found blind to sign of
    # zero (its `==`-based fallback graded `-0.0` and `0.0` equal). Neither
    # this file nor any other case source ever fed `ndarray.__float__` a
    # `-0.0` input before, so the gap was doubly blind here: the generator
    # never produced the property, and the comparator couldn't have seen it
    # even if it had.
    cases.append(_make_case("neg_zero", lambda: np.array(-0.0)))
    cases.append(_make_case("pos_zero", lambda: np.array(0.0)))
    cases.append(_make_case("size1_1d_must_raise", lambda: np.array([7.0])))
    cases.append(_make_case("size_gt1_must_raise", lambda: np.array([1.0, 2.0])))
    cases.append(_make_case("empty_must_raise", lambda: np.array([], dtype=np.float64)))
    cases.append(_make_case("complex_must_raise", lambda: np.array(1 + 2j)))
    return cases


NDARRAY_FLOAT_SPECS = {
    "ndarray.__float__": ItemSpec(
        name="ndarray.__float__", kind="custom",
        scalar_like=True,
        custom_cases=_float_custom_cases,
        numpy_adapter=_float_numpy_adapter,
        ionp_adapter=_float_ionp_adapter,
    ),
}
