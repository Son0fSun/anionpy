"""Rigorous differential coverage for the ufunc `dtype=` kwarg's LOOP
SELECTION + INPUT PRECASTING behaviour, across every plain-call-eligible
ufunc x {bool, int8, int64, uint8, float16, float32, float64, complex64,
complex128} input x the same nine dtypes as the explicit `dtype=` target.

WHY THIS EXISTS (bug this file is evidence for/against)
---------------------------------------------------------------------------
Real numpy resolves an explicit `ufunc(..., dtype=X)` call by finding the
loop in the ufunc's OWN declared type table whose OUTPUT matches `X`, then
casting the INPUT operand(s) to THAT loop's declared input dtype(s) before
computing -- not by computing on the operand's native dtype and casting the
OUTPUT afterward. anionpy used to do the latter, which silently diverges from
numpy on two axes: bool inputs (e.g. `np.negative(bool_arr, dtype=np.int64)`
succeeds in real numpy, raised in anionpy) and complex<->real narrowing in
either direction (e.g. `np.absolute(complex128_arr, dtype=np.float32)`
raises `UFuncTypeError` in real numpy, silently succeeded with the wrong
value in anionpy). A 56-op x 9-in x 9-target sweep (4536 cases) measured 63
divergent cases concentrated in `sign`/`abs`/`absolute`/`negative`/
`subtract`, which is why those five were withdrawn from "exact" in
`anionpy/_state/toplevel.py` (see that file's "<- do not restore without input
precasting" comments) rather than left declared over a known-false claim.

This file is the corpus that re-proves (or disproves) the fix: it is
INTENTIONALLY broader than just those five ops -- every ufunc this module
can build a typed sample for, not just the five the original measurement
happened to name -- because a previous investigation of this same defect
under-reported its size by only characterizing cases it happened to hit.
Declaration in `toplevel.py` is NOT this file's job (explicitly out of
scope per this task's brief); this file only supplies honest measurement.

WHY THIS IS A STANDALONE PYTEST MODULE, NOT A `kind="custom"` REGISTRY ITEM
GOING THROUGH `harness.run_case`
---------------------------------------------------------------------------
`harness._call` catches only `except Exception`, and `harness.run_case`
compares caught exceptions by `type(ionp_exc) not in allowed` (an object-
identity/membership check against a small declared set), not by exact class
NAME. Neither is sufficient here:

  1. A PyO3 Rust panic surfaces to Python as `pyo3_runtime.PanicException`,
     which derives from `BaseException`, NOT `Exception` -- `except
     Exception` silently lets a panic escape the comparison entirely
     instead of recording it as a mismatch. This file's own call wrapper
     therefore catches `BaseException`.
  2. numpy's `UFuncTypeError` (`numpy._core._exceptions.UFuncTypeError`,
     covering both `_UFuncNoLoopError` and `_UFuncInputCastingError` --
     both private classes whose `__name__` is overridden to the shared
     public-looking name `"UFuncTypeError"`) IS-A `TypeError`. A harness
     that treats "both raised *some* `TypeError`" as a pass cannot detect
     the exact defect this file exists to catch: anionpy raising the WRONG
     shape of `TypeError` (plain `TypeError` where numpy raises the richer
     `UFuncTypeError`-named one, or vice versa) while still nominally
     "matching" under an `isinstance`/type-membership check. This file
     therefore compares `type(exc).__name__` (a string) on both sides, not
     `type(exc)` object identity/membership.

Two known, OUT-OF-SCOPE, already-diagnosed exceptions to "0 mismatches"
this file will surface and that a reader should NOT mistake for a new bug
this task caused:

  - `signbit` on complex64/complex128 input requesting `bool`: real numpy
    raises plain `TypeError` (`_UFuncNoLoopError`-shaped, `"No loop
    matching..."`), anionpy raises `UFuncTypeError` (`InputCast`-shaped,
    `"Cannot cast ufunc 'signbit' input..."`). This is a DIFFERENT,
    unrelated defect (an `order=` gap already documented in
    `toplevel.py`'s `signbit` comment as the sole declaration blocker,
    predating and unrelated to the precasting fix this file evidences) --
    left unfixed here deliberately; not a regression from this task.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import _bootstrap  # noqa: F401
import ufunc_introspect as ui
import ufunc_registry  # noqa: F401 -- populates ufunc_registry.UFUNC_SPECS

import anionpy

DTYPES: tuple[str, ...] = (
    "bool", "int8", "int64", "uint8",
    "float16", "float32", "float64",
    "complex64", "complex128",
)

_SCALARS: dict[str, object] = {
    "bool": True, "int8": 1, "int64": 1, "uint8": 1,
    "float16": 1.0, "float32": 1.0, "float64": 1.0,
    "complex64": 1 + 1j, "complex128": 1 + 1j,
}

# Cases where real numpy itself is known, live-verified, to be non-honest
# about the "shape" of exception it raises for the identical call replayed
# twice in a row on the SAME ufunc object (see `no_ufunc_loop_err_dtypes`'s
# doc in ionp-py/src/lib.rs for the underlying order-dependence trap) are
# irrelevant here: this file always builds a FRESH ufunc-name resolution per
# case and never reuses a mutated ufunc object across dtype targets in a way
# that could trip that trap, so no allowance for it is needed.
KNOWN_UNRELATED_MISMATCHES: frozenset[tuple[str, str, str]] = frozenset({
    ("signbit", "complex64", "bool"),
    ("signbit", "complex128", "bool"),
})


def _sample(dtype_name: str) -> np.ndarray:
    return np.array([_SCALARS[dtype_name]], dtype=np.dtype(dtype_name))


_UFUNC_T = type(anionpy.negative)


def _resolve_ionp_ufunc(name: str):
    """Returns the resolved `anionpy.<name>` object iff it exists AND is
    actually a ufunc instance (not e.g. a plain function or missing
    attribute) -- `ufunc_registry.UFUNC_SPECS` contains entries for ops
    registered for OTHER kinds of coverage (e.g. via `ionp_adapter`/
    `numpy_adapter` bridges) that are not necessarily exposed as a real
    `anionpy.<name>` ufunc object at all, so membership there alone is not
    sufficient (verified live: `bitwise_count`/`gcd`/`lcm`/`ldexp`/
    `logaddexp`/`logaddexp2`/`nextafter`/`spacing`/`heaviside`/`isnat`/
    `fmin`/`fmax`/`float_power` are all registered but raise
    `AttributeError` resolving straight off the `anionpy` module)."""
    obj = anionpy
    for part in name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if isinstance(obj, _UFUNC_T) else None


def _eligible_ufunc_names() -> list[str]:
    """Every ufunc this module can build a plain-call typed sample for AND
    that genuinely resolves to a real `anionpy.<name>` ufunc object: already
    registered in `ufunc_registry.UFUNC_SPECS` (mirroring the precedent
    `ufunc_order_cases.py` already set for a new, non-colliding registry
    namespace), single-output, non-gufunc (no core-dimension `signature`),
    and not a string ufunc (whose domain isn't one of DTYPES at all)."""
    names = []
    for name in ui.UFUNC_NAMES:
        if name not in ufunc_registry.UFUNC_SPECS:
            continue
        if _resolve_ionp_ufunc(name) is None:
            continue
        info = ui.describe(name)
        if info.is_string or info.nout != 1 or info.signature is not None:
            continue
        if info.nin not in (1, 2):
            continue
        names.append(name)
    return names


ELIGIBLE_UFUNC_NAMES: list[str] = _eligible_ufunc_names()


def _outcome(fn, args: tuple, target_dtype: str) -> tuple[str, str]:
    """Call `fn(*args, dtype=target_dtype)`, catching `BaseException` (NOT
    `Exception` -- see module docstring: a PyO3 `PanicException` derives
    only from `BaseException`). Returns `("ok", "<result dtype str>")` or
    `("exc", "<type(exc).__name__>")` -- the exact class NAME, not the type
    object, so that two different classes sharing an `isinstance` chain
    (e.g. `TypeError` and numpy's `UFuncTypeError`, both real `TypeError`
    subclasses) are NOT conflated."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fn(*args, dtype=np.dtype(target_dtype))
        return ("ok", str(out.dtype))
    except BaseException as exc:  # noqa: BLE001 -- see docstring: deliberate.
        return ("exc", type(exc).__name__)


@pytest.mark.parametrize("name", ELIGIBLE_UFUNC_NAMES)
def test_dtype_grid(name: str) -> None:
    info = ui.describe(name)
    numpy_fn = ui.resolve_numpy_ufunc(name)
    ionp_fn = _resolve_ionp_ufunc(name)
    assert ionp_fn is not None

    mismatches: list[str] = []
    for in_dtype in DTYPES:
        a = _sample(in_dtype)
        if info.nin == 1:
            np_args = (a,)
            ionp_args = (a,)
        else:
            np_args = (a, a)
            ionp_args = (a, a)

        for target_dtype in DTYPES:
            if (name, in_dtype, target_dtype) in KNOWN_UNRELATED_MISMATCHES:
                continue
            np_kind, np_detail = _outcome(numpy_fn, np_args, target_dtype)
            ionp_kind, ionp_detail = _outcome(ionp_fn, ionp_args, target_dtype)
            if (np_kind, np_detail) != (ionp_kind, ionp_detail):
                mismatches.append(
                    f"{name}({in_dtype} input, dtype={target_dtype}): "
                    f"numpy={np_kind}:{np_detail!r} anionpy={ionp_kind}:{ionp_detail!r}"
                )

    assert not mismatches, f"{len(mismatches)} dtype-grid mismatch(es):\n" + "\n".join(mismatches)
