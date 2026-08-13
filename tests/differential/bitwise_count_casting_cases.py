"""NEW test-cases file: `bitwise_count`'s `casting=` axis with `dtype=`
ABSENT -- the exact blind spot named in `anionpy/_state/toplevel.py`'s
"RE-WITHDRAWN 2026-08-06" comment for `bitwise_count`.

WHY THIS EXISTS
---------------------------------------------------------------------------
`bitwise_count` was declared "exact" twice and withdrawn twice for the SAME
underlying reason, both times because the corpus that "proved" it never
actually varied the axis that broke it:

  1. WITHDRAWN 2026-08-02: a 3,640-case grid (52 ufuncs x 14 input dtypes x
     5 `casting=` values) with `dtype=` ABSENT found the live counterexample
     `anionpy.bitwise_count(anionpy.array([True, False]), casting='no')` silently
     PERFORMS the bool->int8 cast real numpy's `_UFuncInputCastingError`
     refuses (`anionpy` returns `array([1, 0], dtype=uint8)`; real numpy
     raises `UFuncTypeError`).
  2. The very same day, `bitwise_count` was RE-declared while fixing an
     unrelated 0-d/`out=` defect. That re-declaration's own verification
     list (see `toplevel.py`, ~line 6626) swept input dtypes, 0-d, the
     1-element boundary, and `out=` object identity -- and never varied
     `casting=` at all. It also never touched the corpus, because
     `ee70a63` (`ufunc_dtype_cases.py`, "cross casting= with dtype=") always
     passes an explicit `dtype=` target alongside `casting=`. That crossing
     cannot reach this bug: the bug is specifically in the codepath that
     fires when `dtype=` is NOT given (see `ionp-py/src/lib.rs`'s
     `casting_check_active` gate, `dtype.is_none()`), so a corpus that
     always supplies `dtype=` structurally cannot exercise it, no matter
     how many `casting=` values it crosses.

An omitted kwarg is a VALUE of that axis, not the absence of the axis --
crossing `casting=` with `dtype=` PRESENT (as `ufunc_dtype_cases.py` does)
and crossing `casting=` with `dtype=` ABSENT (this file) are two genuinely
different corpora, not variants of the same one. This file is the second
one, and it did not exist anywhere in this suite before now.

WHY THIS IS A SEPARATE FILE/NAMESPACE, NOT MORE ENTRIES ON THE REAL
"bitwise_count" ITEM IN ufunc_registry.py
---------------------------------------------------------------------------
Same registry-key-collision precedent `ufunc_order_cases.py`/
`ufunc_dtype_cases.py`/`floordiv_crossing_cases.py` already document:
`ufunc_registry.py` auto-derives a "bitwise_count" entry for every ufunc
numpy exposes regardless of `toplevel.py`'s declaration state, so this file
uses a fresh `"casting/ufunc/bitwise_count"` key, `kind="custom"` ItemSpec,
merged into `registry.REGISTRY` the tail-merge way, imported by `run.py`
for its side effect.

SCOPE
---------------------------------------------------------------------------
Every one of the 5 real `casting=` values numpy accepts ('no', 'equiv',
'safe', 'same_kind', 'unsafe') PLUS the no-`casting=`-kwarg-at-all default,
crossed with 5 input dtypes bitwise_count actually accepts: bool, int8,
uint8, int64, uint64 -- the task's named minimum, covering both the bool
boundary (the dtype with NO direct loop of its own, promoted to int8) and
representative signed/unsigned/narrow/wide integer dtypes that DO have a
direct loop (so `casting='no'`/`'equiv'` must NOT raise for them -- see the
"int8 with casting='no' matches numpy exactly" control case in
`toplevel.py`'s withdrawal comment, reproduced here as
`in_int8/casting_no`). 30 cases total.

NUMPY IS ORACLE ONLY. Every reference value/exception in this file's cases
comes from a real, live call to `np.bitwise_count(...)` inside
`harness.run_case` -- this module never calls numpy to compute anionpy's
answer or borrows a numpy-derived string for anionpy's expected exception
text; it only supplies (label, args, kwargs) tuples for the harness to run
both sides itself.
"""
from __future__ import annotations

import numpy as np

import _bootstrap  # noqa: F401
import anionpy
import registry
from registry import ItemSpec

CASTINGS: tuple[str | None, ...] = (None, "no", "equiv", "safe", "same_kind", "unsafe")

_VALUES: dict[str, list] = {
    "bool": [True, False],
    "int8": [3, -3],
    "uint8": [3, 5],
    "int64": [11, -11],
    "uint64": [7, 9],
}


def _sample(dtype_name: str) -> np.ndarray:
    return np.array(_VALUES[dtype_name], dtype=np.dtype(dtype_name))


def _bitwise_count_casting_cases() -> list[tuple]:
    cases: list[tuple] = []
    for in_dtype in _VALUES:
        a = _sample(in_dtype)
        for casting in CASTINGS:
            if casting is None:
                label = f"in_{in_dtype}"
                kwargs: dict = {}
            else:
                label = f"in_{in_dtype}/casting_{casting}"
                kwargs = {"casting": casting}
            cases.append((label, (a,), kwargs))
    return cases


# `registry.py`'s own `_DEFAULT_UFUNC_EXC_EQUIV` only covers numpy's
# `_UFuncNoLoopError`/`_UFuncOutputCastingError` private subclasses (probed
# via `sign` on bool / `add` with a bad `out=`). The bug this file targets
# raises numpy's THIRD private `UFuncTypeError` subclass,
# `_UFuncInputCastingError` (raised when the resolved loop's own input
# dtype cannot receive the operand under the active `casting=` rule) --
# `ufunc_dtype_cases.py` already probed and documented this exact gap for
# its own (dtype=-present) corpus; this file needs the identical
# equivalence for the same reason, reprobed independently here (not
# imported from that file) so this corpus does not depend on that file's
# import order.
_UFUNC_INPUT_CASTING_ERROR = registry._numpy_exc_type(
    np.negative, np.array([1], dtype=np.int8), dtype=np.uint8,
)
_EXC_EQUIV = {
    _UFUNC_INPUT_CASTING_ERROR: {anionpy._anionpy.UFuncTypeError},
}

BITWISE_COUNT_CASTING_SPECS: dict[str, ItemSpec] = {
    "casting/ufunc/bitwise_count": ItemSpec(
        name="casting/ufunc/bitwise_count",
        kind="custom",
        numpy_path="bitwise_count",
        ionp_path="bitwise_count",
        atol=0.0,
        rtol=0.0,
        exception_equivalences=_EXC_EQUIV,
        custom_cases=_bitwise_count_casting_cases,
    ),
}

_casting_collisions = set(BITWISE_COUNT_CASTING_SPECS) & set(registry.REGISTRY)
if _casting_collisions:
    raise AssertionError(
        f"bitwise_count_casting_cases.py: {sorted(_casting_collisions)} "
        f"already present in registry.REGISTRY -- refusing to silently "
        f"overwrite an existing item"
    )
registry.REGISTRY.update(BITWISE_COUNT_CASTING_SPECS)
