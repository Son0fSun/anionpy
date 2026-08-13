"""Small array-API-alignment wrappers over already-existing anionpy primitives:
`asanyarray`, module-level `astype`, and `unstack`.

Python-only, no Rust changes -- every function here is a thin composition of
existing `anionpy`-exposed operations (`asarray`, `.astype`, `moveaxis`,
`isinstance`/attribute checks). No arithmetic happens in this file; the only
loop is `unstack`'s implicit iteration over anionpy's own `__iter__`/indexing
(one whole-array slice per iteration, not a per-element loop -- see that
function's docstring).

Each function's declared scope is deliberately narrower than numpy's full
signature where the missing part hinges on the open "is output memory
layout/identity part of anionpy's contract?" question flagged by the task brief
(anionpy.ndarray has no `.base`/`.flags`, and `ndarray.astype(..., copy=False)`
does not implement the real-array passthrough numpy documents) -- see each
function's docstring for exactly what is excluded and why, and
`anionpy/_state/toplevel.py`'s declarations for the differential evidence.
"""
from __future__ import annotations

import anionpy


def asanyarray(a, dtype=None, order=None, *, device=None, copy=None, like=None):
    """numpy-2.5.1-signature-matching `asanyarray(a, dtype=None, order=None,
    *, device=None, copy=None, like=None)`.

    Scope actually declared/tested (see manip_compose_cases.py): `order` in
    {None, "K"} only -- both mean "don't force a layout change", which this
    function honors without needing to inspect `a`'s actual layout at all.
    `order` in {"C", "F", "A"} is accepted and forwarded (falls through to
    the `anionpy.asarray` copy path below, so values are still correct) but
    deliberately NOT part of the declared/tested scope: distinguishing
    "already conforms, return the same object" from "conforming to this
    stricter layout needs a real copy" requires the C/F-contiguity flags
    anionpy.ndarray does not expose (`.flags` raises AttributeError) -- that is
    exactly the open layout-contract question this task must not resolve
    unilaterally, so it is left unverified rather than guessed.

    `like` is not supported (raises `NotImplementedError` -- anionpy has no
    `__array_function__` dispatch protocol to honor it against).
    """
    if like is not None:
        raise NotImplementedError("anionpy.asanyarray: like= is not supported")
    if device is not None and device != "cpu":
        raise ValueError(
            f'Device not understood. Only "cpu" is allowed, but received: {device}'
        )
    if copy is True:
        return anionpy.asarray(a, dtype=dtype, order=order)
    already_ok = (
        isinstance(a, anionpy.ndarray)
        and (dtype is None or a.dtype == dtype)
        and order in (None, "K")
    )
    if already_ok:
        return a
    if copy is False:
        raise ValueError(
            "Unable to avoid copy while creating an array as requested.\n"
            "If using `np.array(obj, copy=False)` replace it with "
            "`np.asarray(obj)` to allow a copy when needed (no behavior "
            "change in NumPy 1.x).\n"
            "For more details, see https://numpy.org/devdocs/"
            "numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword."
        )
    return anionpy.asarray(a, dtype=dtype, order=order)


def astype(x, dtype, *, copy=True, device=None):
    """numpy-2.5.1-signature-matching module-level `astype(x, dtype, /, *,
    copy=True, device=None)` (the array-API alias of `ndarray.astype`,
    distinct from -- and delegating to -- that method).

    Scope actually declared/tested: `x` must be an `anionpy.ndarray` (matching
    real numpy's own dedicated `TypeError` for anything else, replicated
    below verbatim) and `copy` is left at its default `True`. `copy=False`
    is accepted and forwarded to `ndarray.astype` (so it still runs, no
    crash) but deliberately excluded from the declared/tested scope: real
    numpy's `copy=False` returns the SAME object when `dtype` already
    matches (a genuine identity-passthrough contract), and
    `anionpy.ndarray.astype(..., copy=False)` does not implement that -- it
    always copies (verified live: `a.astype(a.dtype, copy=False) is a` is
    `False` in anionpy, `True` in real numpy) -- the same open
    identity/layout-contract question `asanyarray` above defers.
    """
    if not isinstance(x, anionpy.ndarray):
        raise TypeError(f"Input should be a NumPy array or scalar. It is a {type(x)} instead.")
    if device is not None and device != "cpu":
        raise ValueError(
            f'Device not understood. Only "cpu" is allowed, but received: {device}'
        )
    return x.astype(dtype, copy=copy)


def unstack(x, *, axis=0):
    """numpy-2.5.1-signature-matching `unstack(x, /, *, axis=0)`.

    Implemented exactly the way numpy's own source documents it ("equivalent
    to `tuple(np.moveaxis(x, axis, 0))`, since iterating on an array iterates
    along the first axis") -- `anionpy.moveaxis` already exists (undeclared
    upstream only for its own view-vs-copy metadata gap, not a value bug;
    see toplevel.py's Class C note), and `tuple(...)` over an anionpy.ndarray
    dispatches to anionpy's own `__getitem__`/iteration, materializing one
    whole-array slice per element of the outer axis -- no per-element Python
    loop.

    Deliberately does NOT coerce non-ndarray `x` via `asarray` first: real
    numpy's own `unstack` doesn't either (its body reads `x.ndim` directly,
    with no `np.asarray` call anywhere), so a plain list raises
    `AttributeError: 'list' object has no attribute 'ndim'` -- verified live
    -- rather than silently succeeding. Matching that means leaving this
    function's `x` access just as unguarded.
    """
    if x.ndim == 0:
        raise ValueError("Input array must be at least 1-d.")
    return tuple(anionpy.moveaxis(x, axis, 0))
