"""`ndarray` METHOD forms composed as thin pass-throughs over already-declared-
exact anionpy TOP-LEVEL functions of the same name (`anionpy.mean`, `anionpy.std`,
`anionpy.var`, `anionpy.take`, `anionpy.compress` -- see `anionpy/_state/toplevel.py` for
the differential evidence behind each one's "exact" declaration). No Rust
changes, no arithmetic of its own: each function here only forwards `self`
plus its keyword arguments into the already-verified top-level call and
returns its result unchanged.

Real numpy's `ndarray.mean`/`.std`/`.var`/`.take`/`.compress` methods share
the exact same parameter list (minus the leading `a`) as their `numpy.mean`/
`numpy.std`/`numpy.var`/`numpy.take`/`numpy.compress` module-function
counterparts (confirmed directly via `inspect.signature` against numpy
2.5.1: `numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>,
*, where=<no value>)` vs `ndarray.mean(self, /, axis=None, dtype=None,
out=None, *, keepdims=<no value>, where=<no value>)`, and likewise for
std/var/take/compress) -- real numpy's own method implementations
(`_core/_methods.py`) are themselves thin wrappers that just insert `self`
as the array argument and call the same reduction machinery the module
function uses. This file does the same thing on the anionpy side: each
function's signature mirrors `anionpy`'s own top-level function's signature
verbatim (same parameter names, same defaults, e.g. `keepdims=None` /
`where=None`, not numpy's `<no value>` sentinel spelling, since anionpy's own
top-level functions already use `None` as their "not given" default and
this file must not invent a new default that top-level function doesn't
already accept identically), with only the leading array argument dropped
in favor of `self` and (for `compress`, whose module form takes
`condition` before `a`) the argument order adjusted to match numpy's own
`ndarray.compress(condition, axis=None, out=None)` method signature.

Attached onto `anionpy.ndarray` via `setattr` in `anionpy/__init__.py`, directly
after the class is imported from the Rust extension module -- confirmed
live that `anionpy.ndarray` (a pyclass) accepts arbitrary new Python-level
attributes via plain `setattr`/class-body assignment (no `frozen` flag set
on the pyclass), so this is genuine attribute attachment, not a second/
shadow dispatch mechanism.

NONE of these five are declared "exact" in `anionpy/_state/ndarray.py` (see
that file's "REVOKED 2026-08-03" notes for `ndarray.mean`/`.var`/`.std`,
and its notes on `ndarray.take`/`.compress` staying absent) -- real,
disclosed Rust-level defects were found while verifying each, all
out-of-scope for this Python-only composition task to fix:

  - mean/var/std: `a.mean(axis=0)` (any explicit numeric axis) on a
    0-DIMENSIONAL array PANICS (`ionp-core/src/ufunc.rs:6564`, "index out
    of bounds") instead of raising numpy's `AxisError` -- reproduces on
    the underlying already-"exact"-declared top-level `anionpy.mean`/
    `anionpy.var`/`anionpy.std` these methods forward into unchanged, so it is
    inherited, not introduced, by this file. `var`/`std` ALSO forward
    `correction=`/`mean=` unconditionally, which is MORE permissive than
    real numpy's own `ndarray.var`/`ndarray.std` METHODS (confirmed live
    against numpy 2.5.1: `np.array([1.,2.,3.]).var(correction=1)` itself
    raises `TypeError: _var() got an unexpected keyword argument
    'correction'` -- only the separate `numpy.var` MODULE FUNCTION accepts
    `correction=`) -- so this file's `var`/`std` signatures are wider than
    real numpy's method surface, a second, independent reason (beyond the
    panic) these two are not declared.
  - take/compress: real Rust-level defect found by crossing the existing
    narrow top-level `take_cases()`/`compress_cases()` edge cases against
    the full varied differential corpus (which the narrow originals never
    did): when the indexed/compressed axis has length 0, both silently
    skip out-of-bounds validation and return an empty array instead of
    raising `IndexError`.

Kept in the tree anyway (not deleted) because the composition itself is
correct and harmless -- it changes nothing about when these bugs are
reachable, it just makes the existing top-level bugs reachable through one
more call spelling. Whoever fixes the underlying Rust defects can revisit
declaring these METHOD items at that point.
"""
from __future__ import annotations

import anionpy


def mean(self, axis=None, dtype=None, out=None, keepdims=None, *, where=None):
    """`ndarray.mean(axis=None, dtype=None, out=None, *, keepdims=<no value>,
    where=<no value>)` -- thin pass-through to the already-exact
    `anionpy.mean(a, axis=None, dtype=None, out=None, keepdims=None,
    where=None)`.
    """
    return anionpy.mean(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)


def std(
    self,
    axis=None,
    dtype=None,
    out=None,
    ddof=0.0,
    keepdims=None,
    *,
    where=None,
    mean=None,
    correction=None,
):
    """`ndarray.std(axis=None, dtype=None, out=None, ddof=0, *,
    keepdims=<no value>, where=<no value>, mean=<no value>,
    correction=<no value>)` -- thin pass-through to the already-exact
    `anionpy.std(a, axis=None, dtype=None, out=None, ddof=0.0, keepdims=None,
    where=None, mean=None, correction=None)`.
    """
    return anionpy.std(
        self,
        axis=axis,
        dtype=dtype,
        out=out,
        ddof=ddof,
        keepdims=keepdims,
        where=where,
        mean=mean,
        correction=correction,
    )


def var(
    self,
    axis=None,
    dtype=None,
    out=None,
    ddof=0.0,
    keepdims=None,
    *,
    where=None,
    mean=None,
    correction=None,
):
    """`ndarray.var(axis=None, dtype=None, out=None, ddof=0, *,
    keepdims=<no value>, where=<no value>, mean=<no value>,
    correction=<no value>)` -- thin pass-through to the already-exact
    `anionpy.var(a, axis=None, dtype=None, out=None, ddof=0.0, keepdims=None,
    where=None, mean=None, correction=None)`.
    """
    return anionpy.var(
        self,
        axis=axis,
        dtype=dtype,
        out=out,
        ddof=ddof,
        keepdims=keepdims,
        where=where,
        mean=mean,
        correction=correction,
    )


def take(self, indices, axis=None, out=None, mode="raise"):
    """`ndarray.take(indices, axis=None, out=None, mode='raise')` -- thin
    pass-through to `anionpy.take(a, indices, axis=None, out=None,
    mode=Ellipsis)`. That top-level function's own default for `mode` is a
    PyO3-internal sentinel that prints as `Ellipsis` in its signature/
    help() text but is NOT actually accepted as an explicit argument value
    (`anionpy.take(a, i, mode=Ellipsis)` raises `TypeError: 'ellipsis' object
    is not an instance of 'str'`, confirmed live) -- passing the literal
    string `'raise'` instead (real numpy's own documented default for this
    exact parameter) produces identical behavior to omitting `mode`
    entirely, confirmed live for both the in-bounds and out-of-bounds
    cases, so this method's default is spelled as the real string, not the
    unusable sentinel.
    """
    return anionpy.take(self, indices, axis=axis, out=out, mode=mode)


def compress(self, condition, axis=None, out=None):
    """`ndarray.compress(condition, axis=None, out=None)` -- thin
    pass-through to the already-exact `anionpy.compress(condition, a,
    axis=None, out=None)`. Note the argument-order difference between the
    method form (`condition` first, `self` implicit) and the top-level
    function form (`condition` then `a`) is real numpy's own API, not
    introduced here -- both forms take `condition` before the array.
    """
    return anionpy.compress(condition, self, axis=axis, out=out)
