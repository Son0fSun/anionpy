"""`round` / `around` / `ndarray.round`: a whole-array-call-only transcription
of real numpy's `PyArray_Round` C driver
(`numpy/_core/src/multiarray/calculation.c`).

`np.round` is NOT a ufunc. `numpy.round` is literally
``return _wrapfunc(a, 'round', decimals=decimals, out=out)``
(confirmed live via `inspect.getsource(np.round)` against numpy 2.5.1), and
the method behind it is a small C driver that CALLS the `multiply`,
`true_divide` and `rint` ufuncs. That is why this lives in Python: there is
no arithmetic here to move into Rust. Every numeric step below is one
whole-array `anionpy` call; this file only decides WHICH calls to make, in
which order, on which dtype branch -- exactly the job the C driver does.
The three things the C driver does that no ufunc call can express
(`PyArray_CopyInto`, `arr.real = ...`, `arr.imag = ...`) and numpy's own
`power_of_ten` scale-factor table are all in Rust, in
`ionp-core/src/round.rs` (bound as `_copy_into`/`_set_real`/`_set_imag`/
`_power_of_ten`). No Python-level loop over elements exists in this file,
and no scale factor is computed in Python.

THE DRIVER, transcribed
-----------------------
In C, with the branch order preserved exactly (the order is observable --
see "BRANCH ORDER IS OBSERVABLE" below)::

    if (out && PyArray_SIZE(out) != PyArray_SIZE(a))   -> ValueError "invalid output shape"
    if (PyArray_ISCOMPLEX(a)) {
        arr = out ? out : PyArray_Copy(a);
        arr.real = PyArray_Round(a.real, decimals, NULL);
        arr.imag = PyArray_Round(a.imag, decimals, NULL);
        return arr;
    }
    if (PyArray_ISINTEGER(a) && decimals >= 0)         -> copy of `a` (or copy into out)
    if (decimals == 0)                                 -> rint(a)  (or rint(a, out))
    op1, op2 = (multiply, true_divide) if decimals >= 0 else (true_divide, multiply)
    f = power_of_ten(abs(decimals))
    if out is None:
        out = empty(a.shape, float64 if a is integer else a.dtype, order=ISFORTRAN(a))
        ret_int = a is integer
    op1(a, f, out); rint(out, out); op2(out, f, out)
    if ret_int: out = out.astype(a.dtype)
    return out

The whole model above was checked against real numpy BEFORE any of it was
implemented, on a 13-dtype x 11-`decimals` cross product
(`/tmp/mg_rnd1.py`): **133/143 exact, dtype and values and signed zeros**.
The 10 misses were all one cell -- `bool` input with `decimals != 0` -- and
they were misses of the MODEL, not of numpy: bool is not `PyArray_ISINTEGER`
and not complex, so it falls through to the `multiply`/`divide` branch,
where the freshly-allocated `out` has bool dtype and the ufunc refuses::

    np.round(np.array([True,False]),  1) -> UFuncTypeError: Cannot cast ufunc 'multiply'
                                            output from dtype('float64') to dtype('bool')
                                            with casting rule 'same_kind'
    np.round(np.array([True,False]), -1) -> ... ufunc 'divide' ...

That is not a special case in the code below; it falls out of transcribing
the branch order faithfully, and `anionpy.multiply`/`anionpy.divide` already emit
those two messages byte-for-byte (verified, `/tmp/mg_rnd3.py`). Which is
the point of composing over anionpy's real ufuncs rather than reimplementing
the rounding in Rust: the error surface comes along for free and cannot
drift away from the ufuncs' own.

BRANCH ORDER IS OBSERVABLE
--------------------------
Two orderings that look like implementation detail are not:

  * the integer branch is tested BEFORE the `decimals == 0` branch, so
    `np.round(int_array, 0)` is a COPY, not `rint`. Same values, but a
    different `out=` error surface: `np.round(int64([1,2,3]), 0,
    out=empty(3, 'int64'))` succeeds (a copy), while the same call on a
    float input raises `Cannot cast ufunc 'rint' output ...`.
  * the size check is tested BEFORE the complex branch, so a complex
    input with a wrong-SIZED `out` raises `ValueError: invalid output
    shape` and never reaches the "Cannot set imaginary part" TypeError.

Also observable, and easy to get backwards: `out=` with the RIGHT size but
the WRONG shape does NOT raise `invalid output shape`. It gets past the
size gate and then fails inside whichever primitive runs next, with that
primitive's own wording -- and the two wordings differ::

    np.round(arange(6.).reshape(2,3), 0, empty(6))      # ufunc path
        ValueError: operands could not be broadcast together with shapes (2,3) (6,)
    np.round(arange(6).reshape(2,3),  2, empty(6,'i8')) # copy path
        ValueError: could not broadcast input array from shape (2,3) into shape (6,)

RETURN SHAPE: 0-d BECOMES A SCALAR, BUT ONLY WITHOUT `out=`
----------------------------------------------------------
Measured, all four branches (`/tmp/mg_rnd8.py`)::

    np.array(2.5).round()        -> np.float64(2.0)     # rint branch
    np.array(7).round()          -> np.int64(7)         # integer-copy branch
    np.array(17).round(-1)       -> np.int64(20)        # scale branch
    np.array(1.5+2.5j).round()   -> np.complex128(2+2j) # complex branch
    np.array(2.5).round(0, o)    -> array(2.)           # `o` itself, NOT a scalar

The integer-copy branch also proves the copy is a real copy and not an
incref of the input: ``x = np.array([1,2,3]); x.round() is x`` is
**False**.

DELIBERATELY NOT IMPLEMENTED HERE: `fix`
----------------------------------------
`np.fix` was measured in the same sitting and is NOT part of this change.
Two reasons, both measured rather than assumed:

  1. it emits ``DeprecationWarning: numpy.fix is deprecated. Use
     numpy.trunc instead, which is faster and follows the Array API
     standard.`` on every call, and anionpy has no deprecation-warning
     mechanism at all -- an implementation that silently omits the warning
     is a divergence on the very first call, not an edge case; and
  2. its dtype behaviour is NOT `round`'s. `np.fix` on a bool array
     returns bool, while `np.round` on the same input returns float16; and
     `np.fix` on complex128 RAISES (``ufunc 'trunc' not supported for the
     input types``) where `np.round` succeeds. So `fix` is not a sibling of
     this file's driver and sharing code with it would be wrong.

`fix` stays an absent item. Absence is a known state; a near-miss is not.
"""

import operator

import anionpy
from anionpy._anionpy import _copy_into, _power_of_ten, _set_imag, _set_real

__all__ = ["round", "around"]

# dtype-kind classification by name. `anionpy.dtype` exposes only `.name`,
# `.itemsize` and `.dtype` (no `.kind`), so the C-level `PyArray_ISINTEGER`
# / `PyArray_ISCOMPLEX` predicates are spelled as explicit name sets rather
# than invented as a new dtype attribute. `bool` is deliberately in
# NEITHER set: that is precisely what makes `np.round(bool_array, 1)` raise
# (see the module doc).
_INTEGER_NAMES = frozenset(
    {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
)
_COMPLEX_NAMES = frozenset({"complex64", "complex128"})

# `PyArg_ParseTupleAndKeywords(..., "|iO&:round", ...)` converts `decimals`
# through a C `long` first and then into a C `int`, and the two failures
# have DIFFERENT messages (measured, `/tmp/mg_rnd6.py` and `/tmp/mg_rnd8.py`):
#     np.round(x, 10**20)   -> OverflowError: Python int too large to convert to C long
#     np.round(x, 2**31)    -> OverflowError: signed integer is greater than maximum
#     np.round(x, -2**31-1) -> OverflowError: signed integer is less than minimum
# `2**31 - 1` and `-2**31` themselves are accepted and produce `nan`
# (the scale factor overflows to inf, then inf/inf).
_C_LONG_MAX = 2**63 - 1
_C_LONG_MIN = -(2**63)
_C_INT_MAX = 2**31 - 1
_C_INT_MIN = -(2**31)


def _as_c_int(decimals):
    """`decimals` through numpy's own `"i"` argument converter."""
    # `operator.index` reproduces the exact TypeError wording numpy shows
    # for a non-integer `decimals` -- "'float' object cannot be interpreted
    # as an integer", "'NoneType' ...", "'str' ..." -- because numpy's
    # converter uses the same `__index__` protocol. `True` is accepted and
    # means 1, in both libraries.
    value = operator.index(decimals)
    if value > _C_LONG_MAX or value < _C_LONG_MIN:
        raise OverflowError("Python int too large to convert to C long")
    if value > _C_INT_MAX:
        raise OverflowError("signed integer is greater than maximum")
    if value < _C_INT_MIN:
        raise OverflowError("signed integer is less than minimum")
    return value


def _empty_like_order(a):
    """`PyArray_ISFORTRAN(a)` -> the `order=` for the scratch output.

    numpy's macro is `F_CONTIGUOUS && !C_CONTIGUOUS`, so a 1-d array (which
    is both) allocates C-order, not F-order. Spelled out rather than
    shortened to `flags['F_CONTIGUOUS']`, which would be wrong for every
    1-d and 0-d input.
    """
    flags = a.flags
    return "F" if flags["F_CONTIGUOUS"] and not flags["C_CONTIGUOUS"] else "C"


def _round_array(a, decimals, out):
    """`PyArray_Round(a, decimals, out)`. `a` is already an `anionpy.ndarray`,
    `decimals` is already a validated C int, `out` is `None` or an
    `anionpy.ndarray`. Always returns an array (never a scalar); the 0-d ->
    scalar conversion belongs to the caller, because it does NOT happen
    when `out=` is given."""
    if out is not None and out.size != a.size:
        raise ValueError("invalid output shape")

    name = a.dtype.name

    if name in _COMPLEX_NAMES:
        arr = out if out is not None else anionpy.copy(a)
        # Both assignments happen in this order even if the first one is
        # the last thing that succeeds: `np.round(complex_a, d,
        # out=float_o)` writes the real part into `out` and THEN raises
        # from the imaginary assignment. Partial mutation is numpy's
        # observable behaviour here, not an accident of this transcription.
        _set_real(arr, _round_array(a.real, decimals, None))
        _set_imag(arr, _round_array(a.imag, decimals, None))
        return arr

    if name in _INTEGER_NAMES and decimals >= 0:
        if out is None:
            return anionpy.copy(a)
        _copy_into(out, a)
        return out

    if decimals == 0:
        if out is None:
            return anionpy.rint(a)
        return anionpy.rint(a, out)

    op1, op2 = (
        (anionpy.multiply, anionpy.divide) if decimals >= 0 else (anionpy.divide, anionpy.multiply)
    )
    scale = _power_of_ten(abs(decimals))

    ret_int = False
    if out is None:
        if name in _INTEGER_NAMES:
            # numpy computes the scaled round in double and casts back at
            # the end, so `np.round(int8([15,25,-15]), -1)` stays int8 and
            # gives [20, 20, -20] -- half-to-even applied at the scaled
            # magnitude, with no range guard on the way back.
            ret_int = True
            work_dtype = "float64"
        else:
            work_dtype = a.dtype
        out = anionpy.empty(a.shape, dtype=work_dtype, order=_empty_like_order(a))

    op1(a, scale, out)
    anionpy.rint(out, out)
    op2(out, scale, out)
    if ret_int:
        out = anionpy.astype(out, a.dtype)
    return out


def _round_method(self, decimals=0, out=None):
    """`ndarray.round(decimals=0, out=None)`.

    numpy's method signature is `(self, /, decimals=0, out=None)` --
    `self` positional-only, both others accepting positional OR keyword
    (verified: `x.round(1)`, `x.round(decimals=1)`, `x.round(0, o)` and
    `x.round(out=o)` all work).
    """
    decimals = _as_c_int(decimals)
    if out is not None and not isinstance(out, anionpy.ndarray):
        # numpy's `PyArray_OutputConverter` wording, and it fires AFTER
        # the `decimals` conversion because `PyArg_ParseTupleAndKeywords`
        # walks the format string left to right.
        raise TypeError("output must be an array")
    result = _round_array(self, decimals, out)
    # 0-d -> scalar, but only when the caller did not supply storage.
    if out is None and result.ndim == 0:
        return result[()]
    return result


def round(a, decimals=0, out=None):  # noqa: A001 - shadows the builtin exactly as numpy does
    """`numpy.round`: `_wrapfunc(a, 'round', decimals=decimals, out=out)`."""
    return _round_method(anionpy.asarray(a), decimals, out)


def around(a, decimals=0, out=None):
    """`numpy.around`.

    NOT an alias object: `np.around is np.round` is **False** and
    `np.around.__name__` is `'around'` (measured). It is a separately
    defined function with an identical body, so it is spelled that way
    here too rather than as `around = round`.
    """
    return _round_method(anionpy.asarray(a), decimals, out)
