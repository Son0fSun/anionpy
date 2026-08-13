"""NEW test-cases file: permanent coverage for `complex_div`
(`ionp-core/src/ufunc.rs`)'s two branches, split by `abs(denom.imag) <=
abs(denom.real)` ("if" branch, Smith's algorithm with `rat = d/c`) vs
`abs(denom.imag) > abs(denom.real)` ("else" branch, `rat = c/d`).

WHY THIS IS A SEPARATE FILE/NAMESPACE, NOT MORE ENTRIES ON THE REAL
"divide"/"true_divide" ITEMS
---------------------------------------------------------------------------
Same reason `floordiv_crossing_cases.py` documents for its own namespace:
"divide"/"true_divide"/"ndarray.__truediv__" are already declared by the
auto-derived ufunc registry, and the tail-merge pattern used here refuses to
redeclare an existing name. Fresh `"crossing/ufunc/<name>"` keys,
`kind="custom"` ItemSpecs, merged into `registry.REGISTRY` the same way --
imported by `run.py` for its side effect, not by `registry.py` itself.

THE BUG THIS GUARDS AGAINST (if-branch, `a != a` i.e. NaN real part of the
numerator, clean -- non-NaN -- denominator)
---------------------------------------------------------------------------
`complex_div`'s if-branch originally computed the imaginary part as
`(-a).mul_add_ext(rat, b) * scl`. On this toolchain/target
(aarch64-apple-darwin, cargo release+LTO), when `a` is NaN, ANY fused
multiply-add form that uses `a` (negated or not) as a MULTIPLIED operand of
the hardware FMA canonicalizes the result to a fixed-sign NaN, diverging
from numpy's `npy_cdivide` (which does not use FMA on this path and
preserves the propagated NaN's actual sign bit). This is a hardware/codegen
quirk, not something visible from source-level algebraic reasoning --
`(-a).mul_add_ext(rat, b)` and `a.mul_add_ext(-rat, b)` measured BIT-IDENTICAL
(both wrong) despite being algebraically equivalent rewrites, and an
isolated `rustc -O` single-file test of the same formulas did NOT reproduce
the bug (misleading in isolation). The fix conditionally falls back to a
PLAIN (non-fused) `b - a * rat` computation only when `a` is NaN --
confirmed via classified fuzzing to fix the NaN-sign case with zero
precision regression on finite draws (the unconditional plain formula alone
DOES regress precision by 1 ULP on some finite inputs, e.g. complex64
`(0.00019026575+6.535693e-26j) / (5.666147e-12-5.058966e-26j)`).

THE ELSE BRANCH (`abs(denom.imag) > abs(denom.real)`) HAS NO KNOWN BUG
---------------------------------------------------------------------------
Measured clean before any fix was applied: a 12168-case structured sweep
(every numerator component in a NaN/inf/signed-zero/finite grid, crossed
against every denominator pair landing in the else branch after dtype cast,
both complex64 and complex128) plus a 120000-case randomized fuzz sweep,
both 0/N mismatches against numpy bit-for-bit (`tobytes()`). The else
branch's `im = b.mul_add_ext(rat, -a) * scl` puts `a` NEGATED as the
ADDEND (not a multiplicand) of the FMA, which does not trigger the
NaN-canonicalization quirk above -- consistent with the observed 0
mismatches. This corpus locks that measurement in as a permanent
regression guard; it deliberately makes NO source change to the else
branch.

REMAINS OUT OF SCOPE / KNOWN DIVERGENT (NOT covered or fixed by this file)
---------------------------------------------------------------------------
`inf_inf_denom`: when both denominator components are +-inf, `rat = d/c`
(if-branch) or `rat = c/d` (else-branch) is `inf/inf = NaN`, which then
propagates into further mismatches against numpy's dedicated inf-handling
paths in `npy_cdivide`. This is a SEPARATE, pre-existing bug (confirmed via
byte-identical mismatch records present in the very first baseline probe,
before any change in this session) and is intentionally NOT covered here.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401
import registry
from registry import ItemSpec

nan = float("nan")
inf = float("inf")

# ---------------------------------------------------------------------------
# Denominator pairs, hand-classified by which branch they land in AFTER
# casting to the target dtype (branch selection happens post-cast in the
# Rust code, on the casted magnitudes -- NOT on these Python-double literals
# -- so every pair below is re-classified per-dtype at case-generation time,
# same methodology as the probe scripts that measured this).
# ---------------------------------------------------------------------------
_DENOM_PAIRS_RAW = [
    (1.0, 2.0), (-1.0, 2.0), (1.0, -2.0), (-1.0, -2.0),
    (0.0, 1.0), (-0.0, 1.0), (0.0, -1.0), (-0.0, -1.0),
    (1.0, 1e30), (-1.0, 1e30), (1.0, -1e30), (-1.0, -1e30),
    (1e30, 2e30), (-1e30, 2e30),
    (1e-30, 2e-30), (-1e-30, 2e-30),
    (1.0, inf), (-1.0, inf), (1.0, -inf), (-1.0, -inf),
    (0.0, inf), (-0.0, inf), (0.0, -inf),
    (nan, 1.0), (nan, -1.0), (nan, inf), (nan, nan),
    (1.0, nan), (-1.0, nan), (0.0, nan), (inf, nan),
    (5.0, 1e30), (-5.0, -1e30),
    (1e-38, 2e-38),
    (1e-44, 2e-44),
    (1e300, 2e300),
    (1e-300, 2e-300),
    (2.0, 1.0), (2.0, -1.0), (-2.0, 1.0), (-2.0, -1.0),
    (1e30, 1.0), (-1e30, 1.0), (1e30, -1.0), (-1e30, -1.0),
]

_NUMERATOR_COMPONENTS = [
    0.0, -0.0, 1.0, -1.0, 2.5, -3.5, inf, -inf, nan,
    1e300, -1e300, 1e-300, -1e-300,
]


def _branch_for(c_raw, d_raw, comp_dt):
    c = float(comp_dt(c_raw))
    d = float(comp_dt(d_raw))
    if c == 0.0 and d == 0.0:
        return None, c, d
    c_is_nan, d_is_nan = c != c, d != d
    if c_is_nan or d_is_nan:
        # Branch selection itself is well-defined (abs comparisons with NaN
        # are always False), but this corpus only classifies clean-denom
        # cases into "if"/"else"; NaN-containing denominators land wherever
        # the `<=` comparison sends them and are exercised too, just not
        # asserted against a specific branch label below.
        return "nan_denom", c, d
    return ("if" if abs(d) <= abs(c) else "else"), c, d


def _structured_cases(dtype_name, np_dt, comp_dt):
    out = []
    for (c_raw, d_raw) in _DENOM_PAIRS_RAW:
        branch, c, d = _branch_for(c_raw, d_raw, comp_dt)
        if branch is None:
            continue
        if abs(c) == inf and abs(d) == inf:
            # Both denominator components infinite (including cases that
            # only become (inf, inf) AFTER casting to the narrower dtype,
            # e.g. (1e300, 2e300) -> complex64 overflow) is the separate,
            # pre-existing `inf_inf_denom` bug documented in this module's
            # docstring: `rat = d/c` or `c/d` becomes `inf/inf = NaN`
            # there. Explicitly out of scope for this corpus/fix -- do not
            # let it sneak in via dtype-cast overflow and silently fail
            # this permanent regression guard.
            continue
        for a_raw in _NUMERATOR_COMPONENTS:
            for b_raw in _NUMERATOR_COMPONENTS:
                a = float(comp_dt(a_raw))
                b = float(comp_dt(b_raw))
                label = (f"complex_div/{dtype_name}/{branch}/"
                         f"a={a!r}_b={b!r}_c={c!r}_d={d!r}")
                out.append((label,
                            (np_dt(complex(a, b)), np_dt(complex(c, d))),
                            {}))
    return out


def _complex_div_structured_cases():
    out = []
    out += _structured_cases("complex64", np.complex64, np.float32)
    out += _structured_cases("complex128", np.complex128, np.float64)
    return out


COMPLEX_DIV_SPECS: dict[str, ItemSpec] = {
    "crossing/ufunc/complex_divide_branch_coverage": ItemSpec(
        name="crossing/ufunc/complex_divide_branch_coverage",
        kind="custom",
        numpy_path="divide",
        ionp_path="divide",
        atol=0.0,
        rtol=0.0,
        custom_cases=_complex_div_structured_cases,
    ),
}

_complex_div_collisions = set(COMPLEX_DIV_SPECS) & set(registry.REGISTRY)
if _complex_div_collisions:
    raise AssertionError(
        f"complex_div_cases.py: {sorted(_complex_div_collisions)} already "
        f"present in registry.REGISTRY -- refusing to silently overwrite an "
        f"existing item"
    )
registry.REGISTRY.update(COMPLEX_DIV_SPECS)
