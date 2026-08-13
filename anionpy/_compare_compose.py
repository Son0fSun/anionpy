"""`isclose`: a thin Python composition over anionpy primitives (`asarray`,
`subtract`, `absolute`, `add`, `multiply`, `less_equal`, `equal`,
`isfinite`, `isnan`, `logical_and`, `logical_or`).

IMPORTANT CORRECTION (2026-08-03): most of the above ARE declared "exact" in
`anionpy/_state/toplevel.py` -- but `equal` and `less_equal` are NOT. A
2026-08-02 main-session correction (see toplevel.py's "CORRECTION 2026-08-02"
block, right above the six comparison-ufunc notes) found a real, reproducible
divergence in all six comparison ufuncs (`equal`/`not_equal`/`greater`/
`greater_equal`/`less`/`less_equal`) on non-numeric operands
(`object()`/`None`/string dtype): numpy falls back to Python's own comparison
protocol or raises a specific promotion error; anionpy raises a generic
operand-type TypeError instead. That gap is real and those six items stay
undeclared until the object/string dtype family exists.

This composition still uses `equal`/`less_equal` anyway, because the gap
above is UNREACHABLE from `isclose`'s own call path: `x = anionpy.asarray(a)` /
`y = anionpy.asarray(b)` already raises TypeError on object/string input before
`equal`/`less_equal` are ever called (anionpy.array() only supports
bool/int8-64/uint8-64/float16-64/complex64-128 -- verified live,
`/tmp/check_isclose_nonnumeric.py`: object-int, object-None, and string-array
inputs all fail at the `asarray` step, with the same TypeError family real
numpy raises for the same inputs -- `numpy.exceptions.DTypePromotionError`
included, which is itself a `TypeError` subclass). So `equal`/`less_equal`
are only ever invoked here on already-numeric anionpy arrays, which is exactly
the domain the 2026-08-02 correction found clean (its 18/1272 mismatches
were ALL on the non-numeric-operand axis, 0 on the 14x14 numeric-dtype
cross). `isclose`'s own dedicated 66-case differential corpus
(`tests/differential/compare_compose_cases.py`) independently verifies this
end-to-end against real numpy with 0 mismatches -- the declaration below
rests on THAT evidence, not on borrowed credit from equal/less_equal's
(currently withheld) top-level declaration.

No Rust changes, no per-element Python loop -- every step below is one
whole-array anionpy call. Transcribed from real numpy 2.5.1's own
`np.isclose` body (`inspect.getsource(np.isclose)`, `numpy/_core/numeric.py`)
and reduced to the parts anionpy can already execute exactly:

    x, y = asanyarray(a), asanyarray(b)
    dt = result_type(y, 1.)          # promote y to an inexact dtype so the
    y = asanyarray(y, dtype=dt)      # later `abs(x - y)` can't wrap around
                                      # on unsigned/narrow-int inputs
    result = (less_equal(abs(x - y), atol + rtol * abs(y))
              & isfinite(y)
              | (x == y))
    if equal_nan:
        result |= isnan(x) & isnan(y)
    return result[()]

The one substitution: real numpy promotes `y` via
`multiarray.result_type(y, 1.)`, but `anionpy.result_type`/`anionpy.promote_types`/
`anionpy.can_cast` are themselves NOT declared exact in
`anionpy/_state/toplevel.py` (real, live-verified divergences on exotic
non-array inputs -- see that file). Depending on a known-broken primitive to
build a new one would just inherit the bug under a different name.

FIRST ATTEMPT (REVOKED 2026-08-03, same day as the first declaration -- see
the "isclose": REVOKED block in toplevel.py): substituted
`anionpy.multiply(y, 1.0)` for the promotion. That reaches the same promoted
DTYPE as `result_type(y, 1.)`, but it is ARITHMETIC, not a CAST, and
arithmetic can perturb a value where a cast cannot. Concretely, complex
multiply computes `imag = a.real*b.imag + a.imag*b.real`, so
`multiply(inf+0j, 1.0)` evaluates `imag = inf*0.0 + 0.0*1.0 = nan`, turning
`inf+0j` into `inf+nanj`. numpy's own `isclose` routes all non-finite input
through the `x == y` branch, so that corrupted imaginary part flips the
result. Verified live (2026-08-03): `anionpy.multiply(inf+0j, 1.0) ==
(inf+nanj)`; `np.isclose(inf+0j, inf+0j)` is `True`, the multiply-based
composition returned `False`. Every primitive in the composition
(`isfinite`, `equal`, `subtract`, `absolute`, `multiply`, `logical_and`,
`logical_or`) agreed with numpy in isolation -- the defect was in the
composition alone.

CURRENT FIX (2026-08-03): a literal NEP-50 weak-float promotion table,
`_ISCLOSE_Y_PROMOTE_DTYPE` below, keyed on `y.dtype.name`, followed by a
genuine CAST via `anionpy.asarray(y, dtype=...)`. No numpy call, no arithmetic,
no dependency on `result_type`/`promote_types`/`can_cast`. The table itself
was verified live against real `np.result_type(<dtype>, 1.)` for all 14
numeric dtypes on 2026-08-03 (see `_ISCLOSE_Y_PROMOTE_DTYPE`'s own comment
for the transcript); bool and every integer width promote to float64,
float16/float32/float64 and complex64/complex128 are unchanged. Separately
verified that `anionpy.asarray(y, dtype=<same dtype>)` is a byte-for-byte
identity cast on the complex non-finite corpus below (`inf+0j`, `-inf+0j`,
`0+infj`, `inf+infj`, `nan+1j`, `1+nanj`, ...) -- it does not touch the
imaginary lane the way `multiply` did -- and that an int/bool -> float64
cast through `anionpy.asarray` reproduces numpy's `asarray(y,
dtype=result_type(y, 1.))` byte-for-byte on the same integer corpus. See
`tests/differential/compare_compose_cases.py` for the corpus, including a
dedicated regression case (`complex_nonfinite_selfcompare`) that pins the
four previously-mismatching values above plus their real/imag-swapped and
`nan`-mixed relatives.

`result[()]` mirrors numpy's own final line: on a 0-d result this unwraps to
a real `numpy.bool` scalar (anionpy's `ndarray.__getitem__(())` on a 0-d array
already does this correctly -- verified live), matching numpy's documented
"returns a single boolean value" behavior for scalar `a`/`b`; on a non-0-d
result `arr[()]` is a no-op identity read.

Deliberately NOT reproduced: the `RuntimeWarning`/`FloatingPointError`
`geterr()`-routed side channel numpy emits when `atol`/`rtol` themselves
contain non-finite values (`errstate`/`geterr` are not something this task
touches) -- the returned boolean VALUES still match numpy's in every such
case tested (verified: a non-finite atol/rtol still produces the same
per-element boolean as real numpy, only the warning is not replicated), so
this is a warning-channel gap, not a value gap.
"""
from __future__ import annotations

import anionpy

# NEP-50 weak-float promotion table for isclose's `y = asanyarray(y,
# dtype=result_type(y, 1.))` step, keyed on `y.dtype.name`.
#
# VERIFIED LIVE 2026-08-03 against real `np.result_type(np.dtype(<key>),
# 1.)` for every dtype anionpy's isclose can receive (probe:
# /private/tmp/claude-501/-Users-rabite-Monday/adfd2105-b58b-49da-a6ea-a3a25c1e615b/scratchpad/probe_promo.py,
# run against real numpy 2.5.1 via .venv/bin/python):
#
#   bool    -> float64   int8   -> float64   int16  -> float64
#   int32   -> float64   int64  -> float64   uint8  -> float64
#   uint16  -> float64   uint32 -> float64   uint64 -> float64
#   float16 -> float16   float32 -> float32  float64 -> float64
#   complex64 -> complex64   complex128 -> complex128
#
# i.e. bool/every integer width widens to float64 (NEP-50 weak-python-float
# promotion); every already-inexact dtype (float16/32/64, complex64/128) is
# left unchanged. This is a lookup table, not a computation -- it cannot
# diverge from numpy the way calling `anionpy.result_type` itself could.
_ISCLOSE_Y_PROMOTE_DTYPE = {
    "bool": "float64",
    "int8": "float64",
    "int16": "float64",
    "int32": "float64",
    "int64": "float64",
    "uint8": "float64",
    "uint16": "float64",
    "uint32": "float64",
    "uint64": "float64",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "complex64": "complex64",
    "complex128": "complex128",
}


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """numpy-2.5.1-signature-matching `isclose(a, b, rtol=1e-05,
    atol=1e-08, equal_nan=False)`."""
    x = anionpy.asarray(a)
    y = anionpy.asarray(b)
    y = anionpy.asarray(y, dtype=_ISCLOSE_Y_PROMOTE_DTYPE[y.dtype.name])
    diff = anionpy.absolute(anionpy.subtract(x, y))
    tol = anionpy.add(atol, anionpy.multiply(rtol, anionpy.absolute(y)))
    result = anionpy.logical_or(
        anionpy.logical_and(anionpy.less_equal(diff, tol), anionpy.isfinite(y)),
        anionpy.equal(x, y),
    )
    if equal_nan:
        result = anionpy.logical_or(
            result, anionpy.logical_and(anionpy.isnan(x), anionpy.isnan(y))
        )
    return result[()]


# Names of the dtypes numpy's `typecodes['AllFloat']` covers (`efdgFDG`) that
# anionpy actually supports (no float128/'g', no complex256/'G' -- anionpy's array
# construction only spans bool/int8-64/uint8-64/float16-64/complex64-128, see
# `isclose`'s own notes above). Used by `asarray_chkfinite` below in place of
# `dtype.kind` -- VERIFIED LIVE 2026-08-03: `anionpy.dtype` has no `.kind`
# attribute at all (`AttributeError: 'anionpy.dtype' object has no attribute
# 'kind'`), only `.name`/`.itemsize`/`.dtype`, so the `.kind in ('f', 'c')`
# check real numpy's `asarray_chkfinite` uses has to be done by dtype NAME
# instead. This is the same inexact-dtype set as `_ISCLOSE_Y_PROMOTE_DTYPE`'s
# unchanged half, so it is kept as a plain set literal rather than importing
# that dict's keys, to avoid coupling two independently-reasoned-about
# tables.
_INEXACT_DTYPE_NAMES = {
    "float16", "float32", "float64", "complex64", "complex128",
}


def asarray_chkfinite(a, dtype=None, order=None):
    """numpy-2.5.1-signature-matching `asarray_chkfinite(a, dtype=None,
    order=None)`.

    Transcribed from real numpy 2.5.1's own body
    (`inspect.getsource(np.asarray_chkfinite)`):

        a = asarray(a, dtype=dtype, order=order)
        if a.dtype.char in typecodes['AllFloat'] and not np.isfinite(a).all():
            raise ValueError("array must not contain infs or NaNs")
        return a

    `typecodes['AllFloat']` (`'efdgFDG'`, i.e. every float/complex dtype
    char) is replaced with `_INEXACT_DTYPE_NAMES` above since `anionpy.dtype`
    has no `.kind`/`.char` at all -- verified live. `asarray`/`isfinite`/
    `all` are each already declared "exact" in `anionpy/_state/toplevel.py`;
    this composes them with no new arithmetic and no per-element Python
    loop, one whole-array `anionpy.isfinite(a).all()` reduction.

    Verified end-to-end against real numpy 2.5.1 (see
    `tests/differential/compare_compose_cases.py::asarray_chkfinite_cases`):
    finite/non-finite data across bool/every dtype width incl. denormals,
    `dtype=`/`order=` conversion (incl. an `order='F'` strides+flags check,
    not just `.tobytes()`, which is layout-insensitive), 0-d/scalar/empty
    inputs, the ValueError message text, and the already-ionp-array
    `shares_memory`-when-no-conversion-needed case (consistent with
    `anionpy.asarray`'s own already-declared aliasing note: numpy's asarray
    returns the identical object when no conversion is needed, anionpy's
    returns a distinct object that still shares the same buffer --
    `shares_memory` is True either way, `is` is not, which mirrors
    `asarray`'s own accepted, already-declared behavior rather than
    introducing a new one here).

    KNOWN, OUT-OF-CORPUS, PRE-EXISTING GAP DISCOVERED WHILE BUILDING THIS
    (2026-08-03, not introduced by this function): `anionpy.asarray(a,
    order=<invalid string>)` (e.g. `order='Z'`) silently accepts the bogus
    value instead of raising -- real numpy's own `asarray` raises
    `ValueError: order must be one of 'C', 'F', 'A', or 'K' (got 'Z')`.
    Live-verified 2026-08-03: `np.asarray([[1.,2.],[3.,4.]], order='Z')`
    raises that ValueError; `anionpy.asarray(.... order='Z')` returns a
    normal array instead. This is a gap in the ALREADY-DECLARED-exact
    `asarray` primitive itself, not something `asarray_chkfinite`
    introduces -- `asarray_chkfinite` simply inherits it by delegation, the
    same way it inherits `asarray`'s already-accepted `is`-vs-
    `shares_memory` aliasing difference noted above. Flagged here rather
    than silently retested-and-ignored; not fixed or revoked by this
    Python-only, no-Rust lane -- see this module's git history / the
    2026-08-03 isclose fix notes in `anionpy/_state/toplevel.py` for the report
    this was disclosed in. `asarray_chkfinite`'s own corpus only exercises
    valid `order` values (`None`/`'C'`/`'F'`/`'A'`/`'K'`), matching the
    domain `asarray`'s own declaration was already graded over.
    """
    a = anionpy.asarray(a, dtype=dtype, order=order)
    if a.dtype.name in _INEXACT_DTYPE_NAMES and not bool(anionpy.isfinite(a).all()):
        raise ValueError("array must not contain infs or NaNs")
    return a
