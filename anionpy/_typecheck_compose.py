"""`real_if_close`: a whole-array-call-only transcription of real numpy's
`numpy/lib/_type_check_impl.py`.

This is the first entry in anionpy's `_type_check_impl` family. numpy implements
every function in that file in PYTHON -- there is no C driver and no ufunc
behind `real_if_close` -- so composing it in Python here is not a shortcut
around Rust, it is the same shape numpy itself has. No per-element loop
exists in this file; every numeric step is one whole-array `anionpy` call.

THE SOURCE, transcribed
-----------------------
Verbatim from `inspect.getsource(np.real_if_close)` against numpy 2.5.1::

    a = asanyarray(a)
    type_ = a.dtype.type
    if not issubclass(type_, _nx.complexfloating):
        return a
    if tol > 1:
        f = getlimits.finfo(type_)
        tol = f.eps * tol
    if _nx.all(_nx.absolute(a.imag) < tol):
        a = a.real
    return a

Four things in those seven lines are load-bearing and were each measured
before anything was written (`/tmp/mg_ric*.py`, numpy 2.5.1, aarch64):

1. **`tol > 1` is PYTHON's `>`, not a numpy call.** Whatever the caller
   passes is compared as-is, so the error surface for a bad `tol` is
   Python's own and comes along for free::

       real_if_close(c, "x")   -> TypeError: '>' not supported between
                                  instances of 'str' and 'int'
       real_if_close(c, None)  -> ... 'NoneType' and 'int'
       real_if_close(c, 1j)    -> ... 'complex' and 'int'
       real_if_close(c, [10])  -> ... 'list' and 'int'   (a LIST, note, not
                                  an array -- a list is never broadcast here)

   but an ARRAY `tol` goes through the array comparison and then through
   `__bool__`, which is why this item was blocked until `ndarray.__bool__`
   existed (declared 2026-08-03)::

       real_if_close(c, np.array([10, 20])) -> ValueError: The truth value of
                                               an array with more than one
                                               element is ambiguous. ...
       real_if_close(c, np.array([]))       -> ValueError: The truth value of
                                               an empty array is ambiguous. ...
       real_if_close(c, np.array([1000]))   -> works, size-1 array
       real_if_close(c, np.array(1000))     -> works, 0-d

2. **The `tol > 1` gate is strict, so `tol=1` takes the ABSOLUTE-tolerance
   branch** -- `tol` stays the literal 1 and is compared against
   `|imag|` directly. Same for `True` (`True > 1` is False, so the
   threshold is 1), for `False`/`0` (threshold 0 -- nothing but an exactly
   zero imaginary part converts) and for negatives. Measured, all four.

3. **`f.eps` is a numpy scalar of the type's REAL counterpart, and its
   dtype decides the comparison's dtype.** `finfo(complex64).eps` is
   `float32`; `finfo(complex128).eps` is `float64`. That matters because
   `f.eps * tol` follows NEP 50: a weak Python `int`/`float`/`bool` keeps
   float32 for complex64, but a STRONG `np.int64(100)`/`np.float64(100)`
   promotes the threshold to float64 -- a genuinely different number
   (`float32(2**-23)*100` = 1.1920929e-05, `float64(2**-23)*100` =
   1.1920928955078125e-05) and therefore a different verdict for an
   imaginary part sitting between the two. Verified live that anionpy's
   0-d arrays reproduce all four promotions exactly.

   **This is why `anionpy.finfo` is NOT used below.** `anionpy.finfo(...).eps`
   currently returns a plain Python `float` (a known, already-recorded gap
   -- "finfo typed scalars"), which is WEAK under NEP 50 and would silently
   compute the complex64 threshold in float64. The eps values here are
   instead built as 0-d anionpy arrays of the exact right dtype from the IEEE
   definitions -- `2.0**-23` for binary32, `2.0**-52` for binary64 -- both
   of which are exactly representable, so no rounding happens on the way in.
   The constants are asserted against the corresponding `anionpy.finfo` values
   at import time, so if `finfo` is ever fixed to return typed scalars the
   two cannot silently drift apart.

4. **`a.real` is numpy's VIEW; anionpy's is a copy.** The VALUES and dtype
   match exactly (verified across the whole corpus); the view/base
   relationship does not. That is the same declared-on-value divergence
   already recorded for `asarray`/`linspace`/`asanyarray` and is disclosed
   in `anionpy/_state/toplevel.py` rather than papered over. Likewise the
   non-complex early return is `a` ITSELF (`asanyarray` is
   identity-preserving on both sides -- verified `is`-identical), so a real
   input is passed straight back untouched on both sides.

DELIBERATELY OUT OF SCOPE
-------------------------
* `clongdouble` input. numpy's `issubclass(type_, complexfloating)` covers
  it; anionpy has no longdouble/clongdouble at all. UNREACHABLE, not skipped.
* object- and string-dtype input. `real_if_close(None)` gives numpy an
  object array; `anionpy.asanyarray(None)` raises TypeError. That is the
  pre-existing, already-disclosed ingestion gap (`anionpy.array(['x'])` also
  raises), not a `real_if_close` defect, and those rows are excluded from
  the corpus rather than silently tolerated.
* `nan_to_num` and the rest of `_type_check_impl`. `nan_to_num` is built on
  `_nx.copyto(d, val, where=idx)` mutating `x.real`/`x.imag` as VIEWS; anionpy
  has neither `copyto(where=)` nor views nor `__setitem__`, and substituting
  `where()` DIVERGES (copyto same_kind-CASTS the fill value into the
  destination dtype, `where` PROMOTES). It stays absent until a real
  `copyto` exists.
"""

from __future__ import annotations

import anionpy

__all__ = ["real_if_close"]

# numpy's `issubclass(a.dtype.type, complexfloating)`, spelled against the
# dtype names anionpy can actually produce. Kept as a frozenset so the check is
# a membership test, not a chain of comparisons that could drift.
_COMPLEX_NAMES = frozenset({"complex64", "complex128"})

# `finfo(complex64).eps` / `finfo(complex128).eps`, as 0-d anionpy arrays of the
# REAL counterpart dtype -- see point 3 of the module docstring for why the
# dtype (and therefore not `anionpy.finfo`) is load-bearing here.
_EPS = {
    "complex64": anionpy.asarray(2.0**-23).astype("float32"),
    "complex128": anionpy.asarray(2.0**-52).astype("float64"),
}

# Guard against silent drift if `anionpy.finfo` is ever fixed to return typed
# scalars: the VALUES must agree even though the TYPES deliberately do not.
for _cname, _rname in (("complex64", "float32"), ("complex128", "float64")):
    _want = float(anionpy.finfo(_rname).eps)
    _got = float(_EPS[_cname])
    if _want != _got:
        raise AssertionError(
            f"eps constant for {_cname} disagrees with anionpy.finfo({_rname}).eps: "
            f"{_got!r} != {_want!r}"
        )
del _cname, _rname, _want, _got


def real_if_close(a, tol=100):
    """If `a` is complex with every imaginary part below `tol` machine
    epsilons, return its real part; otherwise return `a` unchanged."""
    a = anionpy.asanyarray(a)
    if a.dtype.name not in _COMPLEX_NAMES:
        return a
    if tol > 1:
        tol = anionpy.multiply(_EPS[a.dtype.name], tol)
    if anionpy.all(anionpy.less(anionpy.absolute(a.imag), tol)):
        a = a.real
    return a
