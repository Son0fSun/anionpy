"""NEW test-cases file: the mandatory large-magnitude x fractional-divisor
"crossing" corpus for `floor_divide`/`divmod`, permanently closing the exact
gap that got `floor_divide` REVOKED in `anionpy/_state/toplevel.py` (see that
file's still-present revocation-history comment for the full original
measurement: a 576-case grid found 19 mismatches, e.g.
`np.floor_divide(2.0**53, -1.5) == -6004799503160662.0` vs anionpy's
`-6004799503160661.0`, 1 ULP low on the quotient for a large-magnitude float
dividend paired with a fractional divisor).

WHY THIS IS A SEPARATE FILE/NAMESPACE, NOT MORE ENTRIES ON THE REAL
"floor_divide"/"divmod" ITEMS IN ufunc_registry.py
---------------------------------------------------------------------------
Same registry-key-collision reason `ufunc_order_cases.py`/`ufunc_dtype_cases.py`
already documented for their own namespaces: "floor_divide"/"divmod" are
already declared by `ufunc_registry.py` (auto-derived from every numpy
ufunc name), and the tail-merge pattern this file uses refuses to redeclare
an existing name. This file follows that exact precedent, one level up:
fresh `"crossing/ufunc/<name>"` keys, `kind="custom"` ItemSpecs, merged into
`registry.REGISTRY` the same way -- imported by `run.py` for its side
effect (see the import line added there), not by `registry.py` itself
(registry.py is the highest-blast-radius file in this suite and is not
touched by this task).

THE ACTUAL BUG, so this corpus targets it precisely (not just "big numbers")
---------------------------------------------------------------------------
numpy's real `npy_divmod` computes the raw remainder first (`mod =
fmod(a, b)`), derives the quotient from that RAW (not sign-adjusted)
remainder (`div = (a - mod) / b`), and only THEN separately corrects both
`div` and `mod` for sign as a second, independent step
(`if mod: if (b<0) != (mod<0): mod += b; div -= 1`). The defect this
corpus guards against was computing an already-sign-adjusted remainder
and dividing by THAT instead -- mathematically equivalent, but NOT
bit-identical, because `(x - adjusted_mod) / y` loses the low bit `(x -
raw_mod) / y` (computed BEFORE the later `- 1.0` correction) preserves.
This bit only becomes visible when `x` is large enough that `x`'s ULP is
coarser than 1 (so the choice of which subtraction happens first changes
the rounded result) AND `y` is fractional (an integer `y` never exposes
this because `raw_mod` and `adjusted_mod` differ by an exact integer
multiple of `y` that both paths absorb identically when `y`'s own ULP
structure is coarse enough at this magnitude). Hence: LARGE dividend x
FRACTIONAL divisor is the exact crossing that must stay in the corpus,
not a generic big-number sweep that could pass by accident with integer
divisors.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401
import registry
from registry import ItemSpec

# ---------------------------------------------------------------------------
# The crossing corpus itself.
# ---------------------------------------------------------------------------

_LARGE_DIVIDENDS_F64 = [
    2.0**53, 2.0**53 + 2.0, 2.0**54, 2.0**52 + 0.5,
    -(2.0**53), -(2.0**53 + 2.0), -(2.0**54), -(2.0**52 + 0.5),
    8388609.0, 16777217.0, -8388609.0, -16777217.0,
    0.49999999999999994, -0.49999999999999994,
    1e300, -1e300, 1.7976931348623157e308, -1.7976931348623157e308,
]
_FRACTIONAL_DIVISORS_F64 = [
    1.5, -1.5, 0.5, -0.5, 2.5, -2.5, 0.1, -0.1, 3.7, -3.7,
    1e-300, -1e-300, 0.9999999999999999, -0.9999999999999999,
]

_LARGE_DIVIDENDS_F32 = [
    np.float32(2.0**24), np.float32(2.0**24 + 2.0), np.float32(2.0**25),
    np.float32(-(2.0**24)), np.float32(-(2.0**25)),
    np.float32(8388609.0), np.float32(-8388609.0),
    np.float32(3.4028235e38), np.float32(-3.4028235e38),
]
_FRACTIONAL_DIVISORS_F32 = [
    np.float32(1.5), np.float32(-1.5), np.float32(0.5), np.float32(-0.5),
    np.float32(2.5), np.float32(-2.5), np.float32(0.1), np.float32(-0.1),
]


def _crossing_pairs_f64():
    out = []
    for a in _LARGE_DIVIDENDS_F64:
        for b in _FRACTIONAL_DIVISORS_F64:
            out.append((f"crossing/f64/{a!r}_/_{b!r}",
                        (np.float64(a), np.float64(b)), {}))
    return out


def _crossing_pairs_f32():
    out = []
    for a in _LARGE_DIVIDENDS_F32:
        for b in _FRACTIONAL_DIVISORS_F32:
            out.append((f"crossing/f32/{a!r}_/_{b!r}",
                        (a, b), {}))
    return out


def _crossing_pairs_array_form():
    # Same crossing, but as real arrays (not 0-d scalar pairs), to exercise
    # the actual SIMD/loop path floor_divide/divmod dispatch through for a
    # contiguous buffer, not just the scalar fast path.
    a64 = np.array(_LARGE_DIVIDENDS_F64, dtype=np.float64)
    b64 = np.array(_FRACTIONAL_DIVISORS_F64[:len(_LARGE_DIVIDENDS_F64)]
                   + _FRACTIONAL_DIVISORS_F64[:max(0, len(_LARGE_DIVIDENDS_F64)
                                                     - len(_FRACTIONAL_DIVISORS_F64))],
                   dtype=np.float64)
    a32 = np.array(_LARGE_DIVIDENDS_F32, dtype=np.float32)
    b32 = np.array(_FRACTIONAL_DIVISORS_F32[:len(_LARGE_DIVIDENDS_F32)]
                   + _FRACTIONAL_DIVISORS_F32[:max(0, len(_LARGE_DIVIDENDS_F32)
                                                     - len(_FRACTIONAL_DIVISORS_F32))],
                   dtype=np.float32)
    return [
        ("crossing/f64/array_form", (a64, b64), {}),
        ("crossing/f32/array_form", (a32, b32), {}),
        ("crossing/f64/array_form/broadcast_scalar_divisor", (a64, np.float64(-1.5)), {}),
        ("crossing/f32/array_form/broadcast_scalar_divisor", (a32, np.float32(-1.5)), {}),
    ]


def _floor_divide_crossing_cases():
    return _crossing_pairs_f64() + _crossing_pairs_f32() + _crossing_pairs_array_form()


def _divmod_crossing_cases():
    # Identical corpus; divmod's quotient half is exactly the same
    # `npy_divmod` computation floor_divide reuses (see this module's
    # docstring) -- the multi-output=True path additionally exercises the
    # remainder half and the x == q*y + r invariant's own reconstruction.
    return _floor_divide_crossing_cases()


FLOORDIV_CROSSING_SPECS: dict[str, ItemSpec] = {
    "crossing/ufunc/floor_divide": ItemSpec(
        name="crossing/ufunc/floor_divide",
        kind="custom",
        numpy_path="floor_divide",
        ionp_path="floor_divide",
        atol=0.0,
        rtol=0.0,
        custom_cases=_floor_divide_crossing_cases,
    ),
    "crossing/ufunc/divmod": ItemSpec(
        name="crossing/ufunc/divmod",
        kind="custom",
        numpy_path="divmod",
        ionp_path="divmod",
        atol=0.0,
        rtol=0.0,
        multi_output=True,
        custom_cases=_divmod_crossing_cases,
    ),
}

_crossing_collisions = set(FLOORDIV_CROSSING_SPECS) & set(registry.REGISTRY)
if _crossing_collisions:
    raise AssertionError(
        f"floordiv_crossing_cases.py: {sorted(_crossing_collisions)} already "
        f"present in registry.REGISTRY -- refusing to silently overwrite an "
        f"existing item"
    )
registry.REGISTRY.update(FLOORDIV_CROSSING_SPECS)
