"""Differential ItemSpecs for the window-function / near-free toplevel math
block this task implements (`bartlett`, `blackman`, `hamming`, `hanning`,
`kaiser`, `i0`, `angle`, `sinc`, `unwrap`), backed by pure Python
composition of existing `_anionpy` primitives -- see
`anionpy/_window_math.py`'s module docstring, and `anionpy/_state/toplevel.py`
for the per-item declaration evidence.

All nine are `kind="custom"`: none is a plain unary/binary ufunc-shaped
call (`bartlett`/`blackman`/`hamming`/`hanning` take a bare `M`, `kaiser`
takes `(M, beta)`, `unwrap` takes an array plus `axis=`/`period=`/
`discont=` keywords), so none fits `unary`/`binary`/`binary_op`/`method`.

atol/rtol = 0.0 (bit-exact) for all nine -- every one of these is a
closed-form / finite-coefficient composition with no approximation of
its own beyond what real numpy's own reference implementation already
performs (verified directly: see each function's cases below, which
deliberately include out-of-corpus M/beta/axis/dtype combinations, not
just the values used to derive the formulas).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from registry import ItemSpec

WINDOW_SPECS: dict[str, ItemSpec] = {}


def _reg(name: str, cases_fn) -> None:
    WINDOW_SPECS[name] = ItemSpec(
        name=name, kind="custom", custom_cases=cases_fn,
        numpy_path=name, ionp_path=name, atol=0.0, rtol=0.0,
    )


# ---------------------------------------------------------------------------
# bartlett / hanning / hamming / blackman -- all take a bare M, all share
# the same edge-case shape (M<1 empty, M==1 single 1.0, non-integer M,
# negative M).
# ---------------------------------------------------------------------------

_WINDOW_M_CASES = [
    ("neg_5", -5), ("neg_1", -1), ("zero", 0), ("one", 1), ("two", 2),
    ("three", 3), ("four", 4), ("five", 5), ("seven", 7), ("twelve", 12),
    ("fiftyone", 51), ("hundred", 100), ("noninteger_2_5", 2.5),
    ("noninteger_3_9", 3.9), ("noninteger_neg_2_5", -2.5),
    ("noninteger_0_5", 0.5), ("bool_true", True), ("bool_false", False),
]


def _window_m_cases():
    return [(label, (m,), {}) for label, m in _WINDOW_M_CASES]


_reg("bartlett", _window_m_cases)
_reg("hanning", _window_m_cases)
_reg("hamming", _window_m_cases)
_reg("blackman", _window_m_cases)


# ---------------------------------------------------------------------------
# kaiser -- (M, beta) cross product, including beta=0.0/negative/huge and
# non-integer M.
# ---------------------------------------------------------------------------

_KAISER_BETAS = [0.0, 1.0, 5.0, 6.0, 8.6, 14.0, 20.0, 50.0, -5.0, 100.0, 1000.0]


def _kaiser_cases():
    cases = []
    for mlabel, m in _WINDOW_M_CASES:
        for beta in _KAISER_BETAS:
            cases.append((f"{mlabel}/beta_{beta}", (m, beta), {}))
    return cases


_reg("kaiser", _kaiser_cases)


# ---------------------------------------------------------------------------
# i0 -- unary, but not registered as kind="unary" since it rejects complex
# input with a TypeError (a real, deliberate divergence point from the
# generic unary_corpus() path) and needs the |x|==8.0 Chebyshev-domain-
# boundary explicitly exercised, which the generic corpus does not target.
# ---------------------------------------------------------------------------

_I0_XS = [
    0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.9, 8.0, 8.1, 10.0, 20.0, 50.0, 100.0,
    -3.0, -8.0, -8.1, -0.0, float("nan"), float("inf"), -float("inf"),
]


def _i0_cases():
    cases = [("sweep_f64", (np.array(_I0_XS),), {})]
    cases.append(("sweep_f32", (np.array(_I0_XS, dtype=np.float32),), {}))
    cases.append(("int_input", (np.array([0, 1, 2, 3, 8, 20]),), {}))
    cases.append(("scalar_0", (np.array(0.0),), {}))
    cases.append(("empty", (np.array([], dtype=np.float64),), {}))
    return cases


_reg("i0", _i0_cases)


# ---------------------------------------------------------------------------
# angle -- complex/real input, deg=True/False.
# ---------------------------------------------------------------------------

_ANGLE_COMPLEX = [1 + 2j, 3 - 4j, 0 + 0j, -1 - 1j, 5 + 0j, 0 + 5j, -5 + 0j, 0 - 5j]
_ANGLE_REAL = [1.0, -1.0, 0.0, 5.0, -5.0, float("nan"), float("inf"), -float("inf")]


def _angle_cases():
    cases = [
        ("complex_rad", (np.array(_ANGLE_COMPLEX),), {}),
        ("complex_deg", (np.array(_ANGLE_COMPLEX),), {"deg": True}),
        ("real_rad", (np.array(_ANGLE_REAL),), {}),
        ("real_deg", (np.array(_ANGLE_REAL),), {"deg": True}),
        ("complex64_rad", (np.array(_ANGLE_COMPLEX, dtype=np.complex64),), {}),
        ("scalar_complex", (np.array(1 + 1j),), {}),
        ("empty", (np.array([], dtype=np.complex128),), {}),
    ]
    # DTYPE CROSSING (added 2026-08-04). The list above is all float64 or
    # complex128 -- a control that agreed the whole time. `angle`'s
    # declaration was REVOKED 2026-08-03 for exactly the row this omitted:
    # bool_ input gave numpy float64 and anionpy float16, VALUES equal, dtype
    # wrong. Root cause was `zimag = _zeros(z.shape, dtype=z.dtype)` (a
    # STRONG zero) where numpy's own source writes the bare literal
    # `zimag = 0` (a NEP 50 WEAK Python int). Every narrow dtype is crossed
    # here, not just bool_, because the same substitution changes the
    # arctan2 promotion tier for int8/uint8 (-> float16), int16/uint16
    # (-> float32) and complex64 (-> float32) too.
    for name in ("bool", "int8", "uint8", "int16", "uint16", "int32",
                 "int64", "uint64", "float16", "float32", "complex64"):
        signed = not (name.startswith("u") or name == "bool")
        vals = [1, 0, 2] if not signed else [-1, 0, 3]
        arr = np.array(vals, dtype=name)
        cases.append((f"dtype/{name}/rad", (arr,), {}))
        cases.append((f"dtype/{name}/deg", (arr,), {"deg": True}))
    return cases


_reg("angle", _angle_cases)


# ---------------------------------------------------------------------------
# sinc -- x==0 special case, NaN/Inf, int/float32/complex input.
# ---------------------------------------------------------------------------

def _sinc_cases():
    xs = np.linspace(-4, 4, 41).tolist() + [
        0.0, -0.0, 1e-300, -1e-300, float("nan"), float("inf"), -float("inf"),
    ]
    cases = [
        ("sweep_f64", (np.array(xs),), {}),
        ("int_input", (np.array([0, 1, 2, -2]),), {}),
        ("float32_input", (np.array([0, 1, 2, 2.5], dtype=np.float32),), {}),
        ("complex_input", (np.array([0 + 0j, 1 + 1j, 2 - 1j]),), {}),
        ("empty", (np.array([], dtype=np.float64),), {}),
    ]
    # DTYPE CROSSING (added 2026-08-04). `sinc`'s declaration was REVOKED
    # 2026-08-03 because complex64 input came back complex128 -- a row the
    # list above misses (its "complex_input" is complex128, i.e. a control
    # that could not fail). The root cause was NOT in `sinc` at all: it was
    # `anionpy.where(cond, <bare Python float>, arr)` giving the scalar a
    # STRONG float64/complex128 dtype instead of the NEP 50 weak one, so
    # `where` itself was the phantom (see `_where_weak_scalar_cases` in
    # sort_cases.py). Crossed here anyway: a composition can be wrong when
    # every primitive in it is right, and the reverse -- a fix one level
    # down -- deserves a guard at BOTH levels.
    for name in ("bool", "int8", "uint8", "int16", "int32", "int64",
                 "uint64", "float16", "float32", "complex64"):
        signed = not (name.startswith("u") or name == "bool")
        vals = [0, 1, 2] if not signed else [0, 1, -2]
        cases.append((f"dtype/{name}", (np.array(vals, dtype=name),), {}))
    return cases


_reg("sinc", _sinc_cases)


# ---------------------------------------------------------------------------
# unwrap -- default axis/period/discont, explicit axis on 2-D input
# (including negative axes), integer period, custom discont, length-1
# axis, and plain-list (non-ndarray) input.
# ---------------------------------------------------------------------------

def _unwrap_cases():
    phase = np.linspace(0, np.pi, num=5)
    phase[3:] += np.pi
    rng = np.random.default_rng(0)
    p2 = rng.uniform(-10, 10, size=(4, 7))
    cases = [
        ("basic_1d", (phase,), {}),
        ("with_discont", (phase,), {"discont": 1.0}),
        ("int_period4", (np.array([0, 1, 2, -1, 0]),), {"period": 4}),
        ("deg_period360", (np.mod(np.linspace(0, 720, 19), 360) - 180,), {"period": 360}),
        ("2d_axis0", (p2,), {"axis": 0}),
        ("2d_axis1", (p2,), {"axis": 1}),
        ("2d_axis_neg1", (p2,), {"axis": -1}),
        ("2d_axis_neg2", (p2,), {"axis": -2}),
        ("len1_axis0", (np.array([[1.0, 2.0, 3.0]]),), {"axis": 0}),
        ("single_element", (np.array([5.0]),), {}),
        ("plain_list", ([0.0, 6.0, 0.5],), {}),
    ]
    # DTYPE CROSSING (added 2026-08-04). Every case above is float64 or a
    # plain int list. `unwrap`'s declaration was REVOKED 2026-08-03 with
    # two claims: (a) float16 returned float64 where numpy returns float16,
    # and (b) non-float64 output was "SHIFTED BY ONE ELEMENT". (a)
    # reproduced and is fixed (same weak-scalar `where` root cause as
    # `sinc` -- `interval_high` is a bare Python float). (b) did NOT
    # reproduce on 2026-08-04 for bool_/uint64/int8/float16/float32 on the
    # recorded probe: values matched numpy exactly in every one. A claim I
    # could not reproduce is not the same as a claim I disproved, so the
    # grid below is deliberately wider than the original probe -- multiple
    # lengths, 2-D with an explicit axis, and a real wrapping phase (not
    # just the monotone [0,3,6,9,12] the revocation used) -- so that if the
    # shift is real under some shape it fails HERE rather than silently.
    for name in ("bool", "int8", "uint8", "int16", "uint16", "int32",
                 "int64", "uint64", "float16", "float32"):
        signed = not (name.startswith("u") or name == "bool")
        ramp = np.array([0, 3, 6, 9, 12], dtype=name)
        cases.append((f"dtype/{name}/ramp", (ramp,), {}))
        cases.append((f"dtype/{name}/short", (np.array([0, 3], dtype=name),), {}))
        cases.append((f"dtype/{name}/single", (np.array([2], dtype=name),), {}))
        wrapped = np.array([0, 3, 1, 4, 2, 5], dtype=name)
        if signed:
            wrapped = np.array([0, 3, -3, 4, -2, 5], dtype=name)
        cases.append((f"dtype/{name}/wrapping", (wrapped,), {}))
        cases.append((f"dtype/{name}/period4", (ramp,), {"period": 4}))
        twod = np.array([[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]], dtype=name)
        cases.append((f"dtype/{name}/2d_axis0", (twod,), {"axis": 0}))
        cases.append((f"dtype/{name}/2d_axis1", (twod,), {"axis": 1}))
    return cases


_reg("unwrap", _unwrap_cases)
