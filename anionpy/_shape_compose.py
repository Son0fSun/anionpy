"""`kron`: a whole-array-call-only transcription of real numpy's
`numpy/lib/_shape_base_impl.py`.

Like `real_if_close`, this is numpy's OWN Python implementation, not a
Python stand-in for something numpy does in C. `np.kron` contains no loop,
no accumulation and no ufunc of its own -- it is a broadcast `multiply`
between two arrays that have had alternating length-1 axes inserted, then a
single `reshape`. That is why the Kronecker product comes out BIT-EXACT for
free: every output element is exactly one `a[i] * b[j]` product, computed by
the same `multiply` ufunc anionpy already declares exact, with no summation
anywhere to make the ordering observable.

THE SOURCE, transcribed
-----------------------
Verbatim from `inspect.getsource(np.kron)` against numpy 2.5.1::

    b = asanyarray(b)
    a = array(a, copy=None, subok=True, ndmin=b.ndim)
    is_any_mat = isinstance(a, matrix) or isinstance(b, matrix)
    ndb, nda = b.ndim, a.ndim
    nd = max(ndb, nda)
    if (nda == 0 or ndb == 0):
        return _nx.multiply(a, b)
    as_ = a.shape
    bs = b.shape
    if not a.flags.contiguous:
        a = reshape(a, as_)
    if not b.flags.contiguous:
        b = reshape(b, bs)
    as_ = (1,) * max(0, ndb - nda) + as_
    bs = (1,) * max(0, nda - ndb) + bs
    a_arr = expand_dims(a, axis=tuple(range(ndb - nda)))
    b_arr = expand_dims(b, axis=tuple(range(nda - ndb)))
    a_arr = expand_dims(a_arr, axis=tuple(range(1, nd * 2, 2)))
    b_arr = expand_dims(b_arr, axis=tuple(range(0, nd * 2, 2)))
    result = _nx.multiply(a_arr, b_arr, subok=(not is_any_mat))
    result = result.reshape(_nx.multiply(as_, bs))
    return result if not is_any_mat else matrix(result, copy=False)

Four notes, each measured rather than assumed:

1. **`a = array(a, copy=None, ...)` is `copy=None`, not `copy=False`.**
   Under numpy 2.x those are different: `copy=False` means "never copy,
   raise if you would have to". `copy=None` is the old "copy only if
   needed". Transcribing the wrong one would turn a working call into a
   ValueError on any list input. anionpy's `array` takes the same keyword with
   the same three-valued meaning (verified live).

2. **`ndmin=b.ndim` is asymmetric on purpose.** `a` is promoted to at least
   `b`'s rank, but `b` is never promoted to `a`'s. That asymmetry is
   OBSERVABLE -- it is why `kron(scalar, matrix)` and `kron(matrix, scalar)`
   take different branches -- and the corpus tests both orders of every
   rank pair rather than assuming symmetry.

3. **`is_any_mat` is always False here.** `np.matrix` has no anionpy
   counterpart and anionpy has no ndarray subclass machinery, so both the
   `subok=` argument to `multiply` and the `matrix(result)` return are
   dead code in this transcription. They are written out above and dropped
   below deliberately, not overlooked.

4. **The `flags.contiguous` reshape is a no-op in numpy too.**
   `reshape(a, a.shape)` on a non-contiguous array returns a contiguous
   copy; it changes the array's LAYOUT and nothing about its values. It is
   kept because dropping it would make the output's memory layout depend
   on the input's, and layout is the one thing this file should not quietly
   change. (Whether output layout is part of anionpy's contract at all is an
   open question for Mother; keeping the line costs nothing either way.)

The only deliberate departure: the final `reshape` shape. numpy computes it
as `_nx.multiply(as_, bs)`, a two-element-per-axis int64 ARRAY. anionpy's
`multiply` accepts the same two tuples and its `reshape` accepts the
resulting array (both verified live), so the line is transcribed as-is
rather than replaced with a Python `zip`-and-multiply -- shape metadata is
still computed by the same ufunc numpy uses.
"""

from __future__ import annotations

import anionpy

__all__ = ["kron"]


def kron(a, b):
    """Kronecker product of two arrays."""
    b = anionpy.asanyarray(b)
    a = anionpy.array(a, copy=None, subok=True, ndmin=b.ndim)
    ndb, nda = b.ndim, a.ndim
    nd = max(ndb, nda)

    if nda == 0 or ndb == 0:
        return anionpy.multiply(a, b)

    as_ = a.shape
    bs = b.shape
    if not a.flags.contiguous:
        a = anionpy.reshape(a, as_)
    if not b.flags.contiguous:
        b = anionpy.reshape(b, bs)

    # Equalise the shapes by prepending smaller one with 1s
    as_ = (1,) * max(0, ndb - nda) + as_
    bs = (1,) * max(0, nda - ndb) + bs

    # Insert empty dimensions
    a_arr = anionpy.expand_dims(a, axis=tuple(range(ndb - nda)))
    b_arr = anionpy.expand_dims(b, axis=tuple(range(nda - ndb)))

    # Compute the product
    a_arr = anionpy.expand_dims(a_arr, axis=tuple(range(1, nd * 2, 2)))
    b_arr = anionpy.expand_dims(b_arr, axis=tuple(range(0, nd * 2, 2)))
    result = anionpy.multiply(a_arr, b_arr)

    # Reshape back
    return anionpy.reshape(result, anionpy.multiply(as_, bs))
