"""`binary_repr` and `base_repr`: integer-to-string formatting.

Both are transcriptions of numpy's OWN Python bodies (`binary_repr` from
`numpy/_core/numeric.py`, `base_repr` from the same module). numpy does not
implement either in C, and there is nothing here for a Rust core to do that
would not be a downgrade: `base_repr(2**200)` and `binary_repr(-2**200)` are
correct in numpy because they run on CPython's arbitrary-precision `int`,
which no fixed-width Rust integer can represent. Every operation below is on
a Python `int` or a Python `str`; not one of them is array arithmetic, so
the "no numerical loop in a .py file" rule is not in tension with the
`while num:` division loop in `base_repr` -- that loop is numpy's own, it
runs on a bignum, and moving it to Rust would mean reimplementing bignum
division to get the same answer.

The reason these two were blocked until now is `ndarray.__index__` /
`ndarray.__int__` (both declared 2026-08-03): `binary_repr` opens with
`operator.index(num)` and `base_repr` with `int(number)`, so before those
existed neither function could accept an anionpy 0-d array at all, and
declaring them would have meant declaring a function that worked only on
Python ints.

LOAD-BEARING DETAILS, each measured live against numpy 2.5.1 rather than
inferred from the source:

1. `binary_repr(0, width=w)` returns `'0' * (w or 1)` and NEVER calls
   `err_if_insufficient`. So `binary_repr(0, width=0)` is `'0'` (the
   `or 1` makes width 0 behave like width None) and no width is ever
   "insufficient" for zero. That asymmetry is real: `binary_repr(1, 0)`
   DOES raise.

2. The gh-8679 correction. For a negative `num` with a width, the
   two's-complement width is computed from `len(f'{-num:b}')` and then
   decremented when `-num` is an exact power of two, because
   `-2**(k-1)` needs k bits, not k+1. Dropping that line makes
   `binary_repr(-4, 3)` return the 4-character `'1100'` instead of
   `'100'`.

3. `err_if_insufficient` is called AFTER the string is built and compares
   against `binwidth` (the built string's length), not against the input
   magnitude -- and in the negative branch `binwidth` is the
   two's-complement length, which is one more than the positive one. The
   check and the `max()` that computes `outwidth` therefore see different
   quantities and cannot be collapsed into one comparison.

4. `base_repr` uses `abs(int(number))` for the magnitude but tests the
   ORIGINAL `number < 0` for the sign. On a fractional negative input the
   two disagree: `base_repr(-0.5)` truncates to magnitude 0, so the digit
   list is empty, but the sign test still fires -- giving `'-'`, a string
   with a sign and no digits. `res or '0'` then does NOT substitute the
   zero, because `['-']` is a non-empty list. This is numpy's behaviour,
   verified, and it is transcribed rather than "fixed".

5. `base_repr` validates `base` with `>` and `<` against Python ints, so a
   non-integer base survives validation and fails later inside
   `digits[num % base]`. That inherited error surface is part of the
   contract and is tested, not smoothed over.

6. `builtins.max` in numpy's source is only there because numpy's own
   namespace shadows `max`. Here `max` is already the builtin, so the
   qualification is dropped -- a no-op, recorded so the difference from
   the transcription source is not mistaken for a divergence.

DELIBERATELY OUT OF SCOPE: `np.binary_repr` accepts anything with
`__index__` and `np.base_repr` anything `int()` accepts, including
arbitrary user classes. Only the types anionpy can actually produce or ingest
are claimed -- Python ints/bools/floats, numpy scalars, and anionpy 0-d
arrays.
"""
from __future__ import annotations

import operator

# NOTE: no `import anionpy` here, unlike every other compose-layer module.
# Neither function touches an array primitive -- they operate entirely on
# Python `int` and `str`, and the only anionpy contact is through the
# `__index__`/`__int__`/`__bool__` protocols an anionpy array brings with it
# when it is passed in as an argument. An unused import would be the
# opposite of load-bearing.

_DIGITS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def binary_repr(num, width=None):
    def err_if_insufficient(width, binwidth):
        if width is not None and width < binwidth:
            raise ValueError(
                f"Insufficient bit {width=} provided for {binwidth=}"
            )

    # Ensure that num is a Python integer to avoid overflow or unwanted
    # casts to floating point.
    num = operator.index(num)

    if num == 0:
        return '0' * (width or 1)

    elif num > 0:
        binary = f'{num:b}'
        binwidth = len(binary)
        outwidth = (binwidth if width is None
                    else max(binwidth, width))
        err_if_insufficient(width, binwidth)
        return binary.zfill(outwidth)

    elif width is None:
        return f'-{-num:b}'

    else:
        poswidth = len(f'{-num:b}')

        # See gh-8679: remove extra digit
        # for numbers at boundaries.
        if 2**(poswidth - 1) == -num:
            poswidth -= 1

        twocomp = 2**(poswidth + 1) + num
        binary = f'{twocomp:b}'
        binwidth = len(binary)

        outwidth = max(binwidth, width)
        err_if_insufficient(width, binwidth)
        return '1' * (outwidth - binwidth) + binary


def base_repr(number, base=2, padding=0):
    digits = _DIGITS
    if base > len(digits):
        raise ValueError("Bases greater than 36 not handled in base_repr.")
    elif base < 2:
        raise ValueError("Bases less than 2 not handled in base_repr.")

    num = abs(int(number))
    res = []
    while num:
        res.append(digits[num % base])
        num //= base
    if padding:
        res.append('0' * padding)
    if number < 0:
        res.append('-')
    return ''.join(reversed(res or '0'))
