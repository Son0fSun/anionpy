"""anionpy.polynomial.polyutils -- port of `numpy.polynomial.polyutils`'s
public, basis-agnostic helper surface.

Task #34 context: `polyutils` backs every one of the six polynomial bases
(`polynomial`, `chebyshev`, `legendre`, `laguerre`, `hermite`, `hermite_e`)
-- it is the ONE genuinely shared implementation this project's whole
polynomial surface is built on, not six independent ones. Private,
single-array-only ports of most of these functions (`_trimseq`,
`_as_series1`, `_trimcoef`, `_getdomain`, `_mapparms`, `_mapdomain`)
already exist in `anionpy/polynomial/_polybase.py` and are exercised
indirectly by every one of the ~424 already-declared `polynomial.*` items
(every `ABCPolyBase` method that touches `coef`/`domain`/`window` routes
through them). This module is a SEPARATE, independent implementation --
it does not import from `_polybase.py` and `_polybase.py` does not import
from here -- so a bug in one cannot silently hide behind the other's
already-passing corpus, and so this new module carries zero regression
risk to the 424 already-declared items regardless of what it does.

Six of `numpy.polynomial.polyutils`'s seven public names are implemented
below: `trimseq`, `as_series`, `trimcoef`, `getdomain`, `mapparms`,
`mapdomain`. All are 1:1 transcriptions of numpy 2.5.1's own source
(`polyutils.py`), not reinterpretations -- see each function's docstring
for the specific line(s) transcribed.

`format_float` is DELIBERATELY NOT implemented and NOT declared, for the
same reason `_polybase.py`'s module docstring already gives for the same
gap: it is numpy's own `dragon4_positional`/`dragon4_scientific`-backed
float formatter, not achievable without calling into real numpy's own
internals -- which this project's "never call real numpy to produce a
value" rule forbids -- and reimplementing Dragon4 float formatting from
scratch is a distinct, multi-day project, not a couple of functions. This
is a genuine scope decision, not a divergence blocked by ticket #45 or
#48 -- it is named here so it isn't silently re-attempted as "just a
string formatter."

`as_series`'s dtype promotion (the one genuinely NEW piece of logic here
-- `_as_series1` only ever handles a single array, never a LIST of
differently-typed arrays) replicates `np.common_type`'s algorithm
directly rather than calling it (anionpy has no `common_type` of its
own): integer/uint dtypes promote to float64 (never to a narrower float);
float16/32/64 promote to the widest float64/float32/float16 present;
complex64/128 promote the same way but force the whole result complex;
`bool` and any dtype outside `{float16,float32,float64,complex64,
complex128,ints,uint}` are rejected the same way real numpy's
`as_series` rejects them for a single-array input (`ValueError:
Coefficient arrays have no common type") -- MEASURED directly against
real numpy 2.5.1 (see `tests/differential/polyutils_cases.py`'s corpus
comments for the exact probe commands and results), not assumed from
reading `common_type`'s docstring. Real numpy's `as_series` additionally
has an object-dtype fallback path for genuinely mixed non-numeric input
(`np.dtypes.ObjectDType`); anionpy has no object dtype support anywhere
else in this project, so that branch is unreachable here and always
raises instead -- a real, intentional divergence for object-dtype input
specifically (not a hidden bug), left out of the declared corpus's scope
because no other `anionpy.polynomial.*` item accepts object dtype either.

`trimseq` here only accepts an actual ndarray (uses `.shape`/`.nonzero()`
internally). Real numpy's `trimseq` is written generically against the
plain sequence protocol (`len()`, `__getitem__`) and will happily accept
and return a bare Python `list` unchanged in shape/type. That generic
list-in/list-out path is NOT covered here or in the differential corpus
-- every other `polyutils` function only ever calls `trimseq` on an
already-materialized array (via `as_series`), which is the only call
shape this project's own code needs; a bare-list call is left absent by
scope, not measured and declined.
"""
from __future__ import annotations

import functools
import operator

import anionpy as _ap

__all__ = ["as_series", "trimseq", "trimcoef", "getdomain", "mapparms", "mapdomain"]

_INT_LIKE_DTYPES = (
    "bool", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
)


# Precision ladder, mirroring numpy's `lib._type_check_impl.array_precision`
# / `array_type` tables (see this module's docstring) -- restricted to the
# dtypes anionpy itself supports (no longdouble/clongdouble anywhere in this
# project, so those two precision-3 rungs of numpy's own table are simply
# unreachable input here, not modeled).
_REAL_PRECISION = {"float16": 0, "float32": 1, "float64": 2}
_COMPLEX_PRECISION = {"complex64": 1, "complex128": 2}
_REAL_BY_PRECISION = ("float16", "float32", "float64")
_COMPLEX_BY_PRECISION = (None, "complex64", "complex128")


def _common_dtype(arrays):
    """Port of `np.common_type`'s algorithm (see module docstring), taken
    over a list of already-constructed anionpy arrays rather than resolved
    via a generic `common_type()` entry point (anionpy has none).
    """
    is_complex = False
    precision = 0
    for a in arrays:
        dt = str(a.dtype)
        if dt in _COMPLEX_PRECISION:
            is_complex = True
            p = _COMPLEX_PRECISION[dt]
        elif dt in _REAL_PRECISION:
            p = _REAL_PRECISION[dt]
        elif dt.startswith("int") or dt.startswith("uint"):
            p = 2  # numpy: every integer kind promotes to (at least) float64
        else:
            raise ValueError("Coefficient arrays have no common type")
        precision = max(precision, p)
    if is_complex:
        result = _COMPLEX_BY_PRECISION[precision]
        if result is None:  # unreachable given the table above; guarded anyway
            raise ValueError("Coefficient arrays have no common type")
        return result
    return _REAL_BY_PRECISION[precision]


def trimseq(seq):
    """Port of `numpy.polynomial.polyutils.trimseq` (ndarray input only --
    see module docstring). Vectorized (`nonzero`, not a per-element Python
    loop), matching the existing `_trimseq` private port's character.
    """
    if seq.shape[0] == 0 or seq[-1] != 0:
        return seq
    nz = _ap.nonzero(seq)[0]
    if nz.shape[0] == 0:
        return seq[:1]
    last = int(nz[-1])
    return seq[: last + 1]


def as_series(alist, trim=True):
    """Port of `numpy.polynomial.polyutils.as_series`. Accepts a list of
    array_likes (each converted independently, `ndmin=1`), rejects any
    empty or >1-d result exactly like real numpy, optionally trims trailing
    zeros per-array BEFORE the common-dtype promotion (matching real
    numpy's own order of operations -- trimming happens before, not after,
    dtype resolution), then promotes every array to one shared dtype via
    `_common_dtype` (see module docstring).
    """
    arrays = [_ap.array(a, ndmin=1) for a in alist]
    for a in arrays:
        if a.size == 0:
            raise ValueError("Coefficient array is empty")
        if a.ndim != 1:
            raise ValueError("Coefficient array is not 1-d")
    if trim:
        arrays = [trimseq(a) for a in arrays]
    dtype = _common_dtype(arrays)
    return [a.astype(dtype) for a in arrays]


def trimcoef(c, tol=0):
    """Port of `numpy.polynomial.polyutils.trimcoef`."""
    if tol < 0:
        raise ValueError("tol must be non-negative")
    [carr] = as_series([c])
    idx = _ap.nonzero(_ap.abs(carr) > tol)[0]
    if idx.shape[0] == 0:
        return carr[:1] * 0
    last = int(idx[-1])
    return carr[: last + 1].astype(carr.dtype)


def getdomain(x):
    """Port of `numpy.polynomial.polyutils.getdomain`. Real numpy builds the
    2-element result via `np.array((x.min(), x.max()))`, i.e. it lets
    `np.array` infer the result dtype from two already-`x.dtype`-typed
    scalars -- so a float16 input yields a float16 domain, not float64.
    anionpy's `array()` constructor only accepts plain Python
    bool/int/float/complex scalars (or another anionpy/numpy ndarray), not
    a bare numpy scalar object, so the equivalent here goes through Python
    float/complex first and then casts back to `xarr.dtype` explicitly --
    every float16/32/64/complex64/128 value round-trips exactly through a
    float/complex64/128-or-wider intermediate (there is no narrower dtype
    to lose precision to), so this is bit-exact, not an approximation.
    """
    [xarr] = as_series([x], trim=False)
    if _ap.iscomplexobj(xarr):
        r = _ap.real(xarr)
        i = _ap.imag(xarr)
        result = _ap.array([
            complex(float(r.min()), float(i.min())),
            complex(float(r.max()), float(i.max())),
        ])
    else:
        result = _ap.array([float(xarr.min()), float(xarr.max())])
    return result.astype(xarr.dtype)


def mapparms(old, new):
    """Port of `numpy.polynomial.polyutils.mapparms` -- plain scalar
    arithmetic on the two 2-element domain/window sequences (no array work
    at all, matching real numpy's own implementation exactly, including
    its un-guarded `ZeroDivisionError` when `old[0] == old[1]`).
    """
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    off = (old[1] * new[0] - old[0] * new[1]) / oldlen
    scl = newlen / oldlen
    return off, scl


def mapdomain(x, old, new):
    """Port of `numpy.polynomial.polyutils.mapdomain`."""
    if not isinstance(x, (int, float, complex)):
        x = _ap.asanyarray(x)
    off, scl = mapparms(old, new)
    return off + scl * x


# ---------------------------------------------------------------------------
# Task #34 Part 3: the `_vander_nd` / `_vander_nd_flat` / `_valnd` / `_gridnd`
# shared machinery backing every `*val2d`/`*val3d`/`*valnd`/`*grid2d`/
# `*grid3d`/`*vander2d`/`*vander3d` item across all six bases (42 items --
# see anionpy/polynomial/{polynomial,chebyshev,legendre,laguerre,hermite,
# hermite_e}.py's own module docstrings for the per-basis wiring). Private
# (leading underscore), same as real numpy's own `polyutils` module -- these
# four are numpy-internal helpers, not part of `polyutils`'s public surface
# (`__all__` above is intentionally unchanged).
#
# IMPORTANT CORRECTION vs this ticket's premise: real numpy 2.5.1 does NOT
# route `*val2d`/`*val3d`/`*valnd`/`*grid2d`/`*grid3d` through `_vander_nd`
# at all -- they go through two SEPARATE, smaller helpers, `_valnd` and
# `_gridnd` (see numpy/polynomial/polyutils.py; only `*vander2d`/`*vander3d`
# call `_vander_nd_flat`). All four helpers are implemented here since all
# four are genuinely shared, basis-agnostic machinery, but "one function
# unlocks all 42" underclaims the real shape by one function, not one basis:
# it is 4 shared helpers (not 1) driving 42 items across 6 bases, still a
# dramatic reduction from 42 independent implementations.
#
# A second, load-bearing correction: `_valnd`/`_gridnd` call the PLAIN 1-D
# `<basis>val` function (e.g. `polyval`) with multi-dimensional `c` and a
# `tensor=` kwarg. anionpy's own `polyval`/`chebval`/`legval`/`lagval`/
# `hermval`/`hermeval` (already declared, already tested) are explicitly
# 1-D-`c`-only by design (see each one's own docstring) -- passing a 2-D/3-D
# `c` into any of them raises `ValueError("Coefficient array is not
# 1-d")` (measured directly, not assumed). Re-scoping those five already-
# shipped, already-tested public functions to accept multi-D `c` was
# rejected as too high a regression-risk-to-benefit ratio for this pass
# (touching an already-declared, already-passing item's semantics for a
# DIFFERENT ticket's items). Instead, each basis module below defines its
# own PRIVATE `_<basis>val_nd` companion -- a direct, line-for-line
# transcription of that basis's real numpy Horner/Clenshaw recursion
# (verified algorithm identity against numpy 2.5.1 source, cited in each
# one's own docstring), generalized to accept the multi-D `c` + `tensor`
# call form `_valnd`/`_gridnd` need, entirely in Python/anionpy-array
# arithmetic (indexing, reshape, multiply, add/subtract -- every one of
# which executes in Rust; no numeric Python loop over array DATA, only
# over `c`'s leading axis, which is a handful of coefficients, not a data
# loop). `_valnd`/`_gridnd` themselves take that per-basis nd-eval
# function as their `val_f` argument, exactly like real numpy's do.
def _nth_slice(i, ndim):
    """Port of `numpy.polynomial.polyutils._nth_slice`: an ndim-length
    index tuple that is `slice(None)` at position `i` and `np.newaxis`
    everywhere else -- used to place each dimension's 1-D Vandermonde
    matrix into its own independent trailing axis before the outer-product
    reduction in `_vander_nd`.
    """
    sl = [_ap.newaxis] * ndim
    sl[i] = slice(None)
    return tuple(sl)


def _prep_points(points):
    """Promote a tuple of point arrays to one shared shape/dtype, the same
    "convert to the same shape and type" step `_vander_nd`'s numpy source
    performs via `np.asarray(tuple(points)) + 0.0` (a literal stack of the
    whole tuple into one array, which both (a) requires every point to
    share one exact shape -- numpy's stack raises an "inhomogeneous shape"
    `ValueError` the instant two points differ, MEASURED directly against
    real numpy 2.5.1, not assumed -- and (b) promotes every point to a
    shared dtype via the same rules `np.array`'s own dtype-inference uses).
    anionpy's `array()` constructor has no array-of-arrays stacking form
    (measured: `TypeError`), so this is reimplemented per-point (convert,
    check shapes equal, promote via `_common_dtype` -- the SAME promotion
    table already used above for `as_series`) instead of an actual stack;
    the reachable *valid*-input behavior is identical, the *invalid*-input
    (mismatched-shape) error message is NOT chased byte-for-byte (a plain
    `ValueError` here, not numpy's internal "inhomogeneous shape" wording)
    since that message is a numpy `asarray`-internals accident, not a
    documented part of this API's contract.
    """
    arrs = [a if isinstance(a, _ap.ndarray) else _ap.asanyarray(a) for a in points]
    shape0 = arrs[0].shape
    if not all(a.shape == shape0 for a in arrs[1:]):
        raise ValueError(
            f"Expected {len(arrs)} dimensions of sample points, all of the "
            "same shape"
        )
    dtype = _common_dtype(arrs)
    return tuple(a.astype(dtype) for a in arrs)


def _vander_nd(vander_fs, points, degrees):
    """Port of `numpy.polynomial.polyutils._vander_nd`. `vander_fs` is a
    sequence of the already-implemented, already-declared plain 1-D
    `<basis>vander` functions (one per axis -- may repeat the same
    function, e.g. `(polyvander, polyvander)` for `polyvander2d`). Builds
    the N-D pseudo-Vandermonde array as the outer product (via broadcast
    multiply, `functools.reduce`+`operator.mul`, matching numpy's own
    implementation exactly) of each axis's independent 1-D Vandermonde
    matrix, each placed into its own trailing axis via `_nth_slice`.
    """
    n_dims = len(vander_fs)
    if n_dims != len(points):
        raise ValueError(
            f"Expected {n_dims} dimensions of sample points, got {len(points)}"
        )
    if n_dims != len(degrees):
        raise ValueError(
            f"Expected {n_dims} dimensions of degrees, got {len(degrees)}"
        )
    if n_dims == 0:
        raise ValueError("Unable to guess a dtype or shape when no points are given")

    points = _prep_points(points)

    vander_arrays = (
        vander_fs[i](points[i], degrees[i])[(...,) + _nth_slice(i, n_dims)]
        for i in range(n_dims)
    )
    return functools.reduce(operator.mul, vander_arrays)


def _vander_nd_flat(vander_fs, points, degrees):
    """Port of `numpy.polynomial.polyutils._vander_nd_flat`: `_vander_nd`,
    then flatten the last `len(degrees)` axes into one -- backs the public
    `<basis>vander2d`/`<basis>vander3d` functions.
    """
    v = _vander_nd(vander_fs, points, degrees)
    return v.reshape(v.shape[: v.ndim - len(degrees)] + (-1,))


def _valnd(val_f, c, *args):
    """Port of `numpy.polynomial.polyutils._valnd`. `val_f` here is each
    basis's PRIVATE nd-capable evaluator (`_<basis>val_nd`, see module
    docstring above), not the public 1-D-only `<basis>val` real numpy's
    own `_valnd` uses -- same call shape (`val_f(x0, c)` then
    `val_f(xi, c, tensor=False)` for every subsequent argument), same
    error messages for mismatched-shape points.
    """
    args = [a if isinstance(a, _ap.ndarray) else _ap.asanyarray(a) for a in args]
    shape0 = args[0].shape
    if not all(a.shape == shape0 for a in args[1:]):
        if len(args) == 3:
            raise ValueError("x, y, z are incompatible")
        elif len(args) == 2:
            raise ValueError("x, y are incompatible")
        else:
            raise ValueError("ordinates are incompatible")
    it = iter(args)
    x0 = next(it)
    c = val_f(x0, c)
    for xi in it:
        c = val_f(xi, c, tensor=False)
    return c


def _gridnd(val_f, c, *args):
    """Port of `numpy.polynomial.polyutils._gridnd`: like `_valnd`, but
    evaluates on the full Cartesian product (tensor=True, the default,
    every call) rather than pointwise -- backs the public
    `<basis>grid2d`/`<basis>grid3d` functions.
    """
    for xi in args:
        c = val_f(xi, c)
    return c
